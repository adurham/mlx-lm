#!/usr/bin/env python3
"""pU9 -- cost of the draft head's per-token collectives, and their necessity.

The census (pU8) showed the draft head makes, per draft block of width g:
    3 all_sums (one per stage, the sharded draft MoE)
  + g all_sums (one per markov step, the exact cross-rank argmax)
The g-per-token ones sit INSIDE the left-to-right markov loop; the question is
whether they can be removed or are inherent.

This measures, at world 2 on one node, layers 20,37,38,39:
  A  draft as built (sharded MoE + sharded markov + combine_argmax per step)
  B  draft with the markov loop de-collectivised: same per-rank logits, tokens
     taken by rank-local argmax (INEXACT -- only to price the collective; a
     different token stream is the expected and reported outcome)
  C  draft with the markov head replicated (no vocab shard) and rank-local
     argmax: exact again in the sense that both ranks hold the same logits, and
     the collective count drops to the 3 stage MoE sums
  D  repeat of A for determinism

Reports per-arm ms and the all_sum count, plus token equality where it is
expected (A vs C: must be equal; A vs B: must not be trusted).

Env: PU9_RANK, PU9_HOSTFILE, PU9_LAYERS, PU9_GAMMA, PU9_PKG.
NOTE: ``mx.eval`` is MLX's graph evaluation, not python's builtin.
"""
import json
import os
import statistics as S
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
PKG = os.environ.get("PU9_PKG", HOME + "/dsv41-ws2/U")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "bench"))

import mlx.core as mx  # noqa: E402

import pU_cap  # noqa: E402

_b, _h = os.environ.get("PU9_BODY_EXPERTS"), os.environ.get("PU9_HEAD_EXPERTS")
pU_cap.apply_caps(int(_b) if _b else None, int(_h) if _h else None)

HF = os.environ.get("PU9_HOSTFILE", "")
group = mx.distributed.init(strict=True, backend="ring") if HF else None
RANK = group.rank() if group else 0
WORLD = group.size() if group else 1
log = lambda *a: print(f"[pU9 r{RANK}]", *a, flush=True)

N = [0]
_real = mx.distributed.all_sum


def counted(x, group=None, **kw):
    N[0] += 1
    return _real(x, group=group, **kw)


mx.distributed.all_sum = counted

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import mtp as MT  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PU9_LAYERS", "20,37,38,39").split(",")]
GAMMA = int(os.environ.get("PU9_GAMMA", "3"))

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=RANK,
                          world=WORLD, group=group)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ck = Exl3Checkpoint(MODEL)
head = eb.build_mtp(ck, model.args, rank=RANK, world=WORLD, group=group)
mx.eval(mx.ones(1))
log(f"built layers={LAYERS} world={WORLD} sharded={getattr(head, 'vocab_sharded', False)} "
    f"active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")

prompt = json.load(open(HOME + "/p30_prompt_ids.json"))[:24]
TAPS = list(model.args.dspark_target_layer_ids)
cache = model.make_cache(1, max_seq_len=len(prompt) + 128)
am, taps = model(mx.array([prompt]), cache, last_logit_only=True,
                 return_taps=True, argmax=True)
mx.eval(am)
dsc = head.make_cache(1)
head.append_ctx(mx.concatenate([taps[L] for L in TAPS], axis=-1), dsc)
mx.eval([c.win_kv for c in dsc])
anchor = am[:, -1].astype(mx.int32)
mx.eval(anchor)


def time_it(label, fn, n=12):
    o = None
    for _ in range(3):
        o = fn()
        mx.eval(*[x for x in (o if isinstance(o, (list, tuple)) else [o])
                  if isinstance(x, mx.array)])
    ts, cnts = [], []
    for _ in range(n):
        c0 = N[0]
        s = time.perf_counter()
        o = fn()
        mx.eval(*[x for x in (o if isinstance(o, (list, tuple)) else [o])
                  if isinstance(x, mx.array)])
        ts.append(time.perf_counter() - s)
        cnts.append(N[0] - c0)
    r = [np.array(x).tolist() if isinstance(x, mx.array) else x
         for x in (o if isinstance(o, (list, tuple)) else [o])]
    log(f"{label:44s} {S.median(ts) * 1e3:7.2f} ms  all_sum={int(S.median(cnts))}")
    return r


A = time_it("A draft as built", lambda: head.draft(anchor, model.embed,
                                                  model.head, dsc, width=GAMMA))


# B: same rank-local logits, but the per-markov-step cross-rank merge replaced
# by a rank-local argmax (WRONG tokens by construction -- it prices the
# collective, nothing else).
class _LocalArgmaxHead:
    """Stands in for the body head inside draft(): local slice, no collective."""

    def __init__(self, body):
        self._body = body

    def local(self, h):
        return self._body.local(h)

    def combine_argmax(self, y):
        return mx.argmax(y, axis=-1).astype(mx.int32)


head.vocab_sharded = True          # keep the local-slice path
B = time_it("B draft, no per-step cross-rank argmax", lambda: head.draft(
    anchor, model.embed, _LocalArgmaxHead(model.head), dsc, width=GAMMA))

# C: replicate the markov head so no vocab shard is needed
mh = head.markov_head
full_w = mh.weight
if WORLD > 1:
    # gather the full markov weight from both ranks (host-side, once)
    w = np.array(full_w)
    pad = np.zeros((model.args.vocab_size, w.shape[1]), dtype=np.float32)
    lo = RANK * w.shape[0]
    pad[lo:lo + w.shape[0]] = w
    import mlx.core as mx2
    t = mx2.array(pad)
    tot = mx2.distributed.all_sum(t, group=group)
    mx2.eval(tot)
    head.markov_head = MT.nn.Linear(w.shape[1], model.args.vocab_size, bias=False)
    head.markov_head.weight = tot[:model.args.vocab_size]
    mx2.eval(head.markov_head.weight)
    sh_mh = getattr(head, "vocab_sharded", False)
    head.vocab_sharded = False
    C = time_it("C draft, replicated markov (no per-step merge)", lambda: head.draft(
        anchor, model.embed, model.head, dsc, width=GAMMA))
    head.markov_head = mh
    head.vocab_sharded = sh_mh
    sh_mh2 = getattr(head, "vocab_sharded", False)
    log(f"token check: A==B {A[0] == B[0]}  A==C {A[0] == C[0]}  "
        f"A={A[0]}  C={C[0]}")
else:
    log("C skipped (world=1: nothing is sharded)")

A2 = time_it("D draft as built (repeat)", lambda: head.draft(
    anchor, model.embed, model.head, dsc, width=GAMMA))
log(f"A vs D identical: {A[0] == A2[0]}")

log(f"peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU9_DONE")
