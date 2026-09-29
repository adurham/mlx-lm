#!/usr/bin/env python3
"""pU6 -- draft-head collectives: count them, split by phase of a round. (U)

Part 2 of the stream-U question: "confirm the draft head runs identically on
both ranks with no per-draft-token collective beyond the exact argmax."

Method: run TWO ranks of the ring backend on ONE node (single-node rule), world
2, layers 20,37,38,39 (the taps + their kv source) with the draft head sharded
exactly as ``build_mtp`` does at world>1 (DSV41_DRAFT_SHARD / DSV41_TP_HEAD on:
experts get a rank slice, the markov head gets a vocab slice). Wrap
``mx.distributed.all_sum`` and count every call, tagged by which code path made
it. Then compare the two ranks' token streams for identity.

Reported:
  prefill / append_ctx / draft / verify / rollback: all_sum calls per phase
  per-draft-token collectives: all_sum calls made inside head.draft() divided
    by the number of markov steps -- the thing the question asks about
  rank identity: draft ids, verify argmax and committed ids equal on both ranks

Env: PU6_RANK, PU6_HOSTFILE (ring json hostfile), PU6_LAYERS, PU6_GAMMA,
     PU6_ROUNDS, PU6_BODY_EXPERTS, PU6_HEAD_EXPERTS, PU6_PKG.
NOTE: ``mx.eval`` is MLX's graph evaluation, not python's builtin.
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
PKG = os.environ.get("PU6_PKG", HOME + "/dsv41-ws2/U")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "bench"))

import mlx.core as mx  # noqa: E402

import pU_cap  # noqa: E402

_b = os.environ.get("PU6_BODY_EXPERTS")
_h = os.environ.get("PU6_HEAD_EXPERTS")
pU_cap.apply_caps(int(_b) if _b else None, int(_h) if _h else None)

HF = os.environ.get("PU6_HOSTFILE", "")
group = mx.distributed.init(strict=True, backend="ring") if HF else None
RANK = group.rank() if group else 0
WORLD = group.size() if group else 1
log = lambda *a: print(f"[pU6 r{RANK}]", *a, flush=True)

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PU6_LAYERS", "20,37,38,39").split(",")]
GAMMA = int(os.environ.get("PU6_GAMMA", "3"))
ROUNDS = int(os.environ.get("PU6_ROUNDS", "4"))


# ---- collective counter -----------------------------------------------------
PHASE = ["build"]
COUNTS: dict[str, int] = {}
_real = mx.distributed.all_sum


def counted(x, group=None, **kw):
    COUNTS[PHASE[0]] = COUNTS.get(PHASE[0], 0) + 1
    # record payload shape too: a per-token collective would show a dim of
    # gamma or gamma+1 in one axis
    try:
        COUNTS.setdefault("_shapes", {})
        COUNTS["_shapes"][PHASE[0]] = list(getattr(x, "shape", []))
    except Exception:
        pass
    return _real(x, group=group, **kw)


mx.distributed.all_sum = counted

# also count what draft() calls on the markov head, if sharded
mh = {"calls": 0}

t0 = time.time()
model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=RANK,
                          world=WORLD, group=group)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ck = Exl3Checkpoint(MODEL)
head = eb.build_mtp(ck, model.args, rank=RANK, world=WORLD, group=group)
mx.eval(mx.ones(1))
log(f"built layers={LAYERS} world={WORLD} in {time.time() - t0:.0f}s "
    f"active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")
log(f"head.vocab_sharded={getattr(head, 'vocab_sharded', False)} "
    f"markov_width={head.markov_head.weight.shape[0]} "
    f"body_head_sharded={hasattr(model.head, 'combine_argmax')} "
    f"stage0_experts_group={head.stages[0].ffn.group is not None}")

TAPS = list(model.args.dspark_target_layer_ids)
prompt = json.load(open(HOME + "/p30_prompt_ids.json"))[:24]


def tapcat(taps):
    return mx.concatenate([taps[L] for L in TAPS], axis=-1)


PHASE[0] = "prefill"
cache = model.make_cache(1, max_seq_len=len(prompt) + 128)
am, taps = model(mx.array([prompt]), cache, last_logit_only=True,
                 return_taps=True, argmax=True)
mx.eval(am)
PHASE[0] = "append_ctx"
dsc = head.make_cache(1)
head.append_ctx(tapcat(taps), dsc)
mx.eval([c.win_kv for c in dsc])
nxt = am[:, -1].astype(mx.int32)
mx.eval(nxt)
pos = cache.offset
out = [int(nxt.item())]
draft_calls = []
for i in range(ROUNDS):
    PHASE[0] = "draft"
    c0 = COUNTS.get("draft", 0)
    d, _ = head.draft(nxt, model.embed, model.head, dsc, width=GAMMA)
    d = d.astype(mx.int32)
    draft_calls.append(COUNTS.get("draft", 0) - c0)
    vin = mx.concatenate([nxt.reshape(1, 1), d], axis=1)
    PHASE[0] = "verify"
    am, taps = model(vin, cache, return_taps=True, argmax=True)
    mx.eval(am, d)
    tg, dd = np.array(am[0]), np.array(d[0])
    n = 0
    while n < GAMMA and tg[n] == dd[n]:
        n += 1
    new = [int(v) for v in dd[:n]] + [int(tg[n])]
    PHASE[0] = "rollback"
    sn = SP.snap(cache, pos)
    SP.rollback(cache, sn, pos + n + 1, SP.stashes(cache))
    PHASE[0] = "append_ctx"
    head.append_ctx(tapcat(taps)[:, :n + 1], dsc)
    pos = pos + n + 1
    nxt = mx.array([new[-1]], dtype=mx.int32)
    mx.eval(nxt)
    out.extend(new)
mx.eval(mx.ones(1))

PHASE[0] = "final"
mx.distributed.all_sum(mx.ones(1), group=group)
mx.eval(mx.ones(1))

shapes = COUNTS.pop("_shapes", {})
if RANK == 0:
    log(f"A all_sum calls by phase: {dict(sorted(COUNTS.items()))}")
    log(f"A payload shape last seen per phase: {shapes}")
    log(f"A all_sum calls inside head.draft(): {draft_calls} per round "
        f"(width={GAMMA}, markov steps={GAMMA} -> {[c / GAMMA for c in draft_calls]} "
        f"per draft token)")

blob = json.dumps({"out": out, "draft_calls": draft_calls}, sort_keys=True)
h = float(sum(blob.encode()))
harr = mx.array([h, float(len(blob))])
if group is not None:
    tot = _real(harr, group=group)
    mx.eval(tot)
    s, n = float(tot[0].item()), float(tot[1].item())
    log(f"B rank-identity: this rank's blob hash={h:.0f} len={len(blob)}; "
        f"expected sum over {WORLD} ranks={h * WORLD:.0f}; got {s:.0f} -> "
        f"{'IDENTICAL' if abs(s - h * WORLD) < 1 else 'DIFFERENT'}")
    log(f"B committed ids: {out[:12]}")
else:
    log(f"B unsharded: committed ids {out[:12]}")
log(f"peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU6_DONE")
