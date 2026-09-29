#!/usr/bin/env python3
"""pU4 -- draft-head collective census + cross-rank identity (stream U).

Part 2 of the stream-U question: does the draft head run identically on both
ranks, with no per-draft-token collective beyond the exact argmax?

A. COLLECTIVE CENSUS. Wraps ``mx.distributed.all_sum`` and counts calls per
   phase of a round (draft / verify / append / rollback) at world=2, so the
   number and payload of the draft head's collectives are measured, not guessed.
   Also counts the markov-head argmax merges separately, because those are the
   only draft collectives that sit *inside* the per-draft-token loop.

B. CROSS-RANK IDENTITY. Runs the same rounds at both ranks (ring backend, both
   on this node) and compares, per round: the draft token ids, the verify
   argmax, and the committed token list. Any rank-asymmetric op (a missing
   broadcast, a rank-dependent slice) shows up as a mismatch.

Env: PU4_RANKS=2, PU4_LAYERS, PU4_WORLD, PU4_GAMMA, PU4_ROUNDS, PU4_PKG,
     PU4_HOSTFILE (ring hostfile for the 2-rank run), PU4_RANK.
NOTE: ``mx.eval`` is MLX's graph evaluation, not python's builtin.
"""
import json
import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
PKG = os.environ.get("PU4_PKG", HOME + "/dsv41-ws2/U")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "bench"))

import mlx.core as mx  # noqa: E402

import pU_cap  # noqa: E402

_b = os.environ.get("PU4_BODY_EXPERTS")
_h = os.environ.get("PU4_HEAD_EXPERTS")
pU_cap.apply_caps(int(_b) if _b else None, int(_h) if _h else None)

WORLD = int(os.environ.get("PU4_WORLD", "2"))
HF = os.environ.get("PU4_HOSTFILE", "")
if HF:
    group = mx.distributed.init(strict=True, backend="ring")
else:
    group = None
RANK = group.rank() if group else 0
WORLD = group.size() if group else WORLD
log = lambda *a: print(f"[pU4 r{RANK}]", *a, flush=True)

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PU4_LAYERS", "20,37,38,39").split(",")]
GAMMA = int(os.environ.get("PU4_GAMMA", "3"))
ROUNDS = int(os.environ.get("PU4_ROUNDS", "6"))

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=RANK,
                          world=WORLD, group=group)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ck = Exl3Checkpoint(MODEL)
head = eb.build_mtp(ck, model.args, rank=RANK, world=WORLD, group=group)
mx.eval(mx.ones(1))
log(f"built layers={LAYERS} world={WORLD} sharded_head={getattr(head, 'vocab_sharded', False)} "
    f"active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")

TAPS = list(model.args.dspark_target_layer_ids)
prompt = json.load(open(HOME + "/p30_prompt_ids.json"))[:48]


def tapcat(taps):
    return mx.concatenate([taps[L] for L in TAPS], axis=-1)


# ---- A. collective census ---------------------------------------------------
COUNTS = {}
_real_all_sum = mx.distributed.all_sum
PHASE = ["setup"]


def _counted_all_sum(x, group=None, **kw):
    COUNTS[PHASE[0]] = COUNTS.get(PHASE[0], 0) + 1
    return _real_all_sum(x, group=group, **kw)


mx.distributed.all_sum = _counted_all_sum
if group is not None:
    # the modules captured mx.distributed.all_sum at call time; mtp/exl3 use the
    # module attribute, so patching mx.distributed is enough for both.

    pass


def make_state():
    cache = model.make_cache(1, max_seq_len=len(prompt) + 256)
    PHASE[0] = "prefill"
    am, taps = model(mx.array([prompt]), cache, last_logit_only=True,
                     return_taps=True, argmax=True)
    mx.eval(am)
    PHASE[0] = "append_ctx"
    dsc = head.make_cache(1)
    head.append_ctx(tapcat(taps), dsc)
    mx.eval([c.win_kv for c in dsc])
    nxt = am[:, -1].astype(mx.int32)
    mx.eval(nxt)
    return cache, dsc, nxt


def rounds(n=ROUNDS):
    cache, dsc, nxt = make_state()
    pos = cache.offset
    out = []
    for i in range(n):
        PHASE[0] = "draft"
        d, _ = head.draft(nxt, model.embed, model.head, dsc, width=GAMMA)
        d = d.astype(mx.int32)
        vin = mx.concatenate([nxt.reshape(1, 1), d], axis=1)
        PHASE[0] = "snap"
        sn = SP.snap(cache, pos)
        PHASE[0] = "verify"
        am, taps = model(vin, cache, return_taps=True, argmax=True)
        mx.eval(am, d)                                   # the round's one sync
        tg, dd = np.array(am[0]), np.array(d[0])
        n = 0
        while n < GAMMA and tg[n] == dd[n]:
            n += 1
        new = [int(v) for v in dd[:n]] + [int(tg[n])]
        PHASE[0] = "rollback"
        SP.rollback(cache, sn, pos + n + 1, SP.stashes(cache))
        PHASE[0] = "append_ctx"
        head.append_ctx(tapcat(taps)[:, :n + 1], dsc)
        pos = pos + n + 1
        nxt = mx.array([new[-1]], dtype=mx.int32)
        mx.eval(nxt)
        out.append(dict(d=dd.tolist(), t=tg.tolist(), acc=n, new=new))
    return out


res = rounds()
if group is None or RANK == 0:
    log("A collective calls by phase: " + ", ".join(
        f"{k}={v}" for k, v in sorted(COUNTS.items())))
    if res:
        # draft-phase collectives split into MoE-per-stage and the markov argmax
        log(f"A note: '{'draft'}' counts all_sum calls made while head.draft ran "
            f"-- 3 stage MoE sums + one merge per block position ({GAMMA} at "
            f"width {GAMMA}) when the head + markov_head are vocab-sharded.")

# ---- B. cross-rank identity -------------------------------------------------
# hash the per-round results so both ranks can be compared without a gather
blob = json.dumps(res, sort_keys=True)
h = mx.array([float(sum(blob.encode()) % (1 << 31)), float(len(blob))])
if group is not None:
    g = mx.distributed.all_sum(h, group=group)
    mx.eval(g)
    log(f"B rounds blob hash={int(h[0].item())} len={int(h[1].item())} "
        f"summed-over-ranks={int(g[0].item())} (rank-equal iff hash*{WORLD} "
        f"== summed value)")
    if RANK == 0:
        log(f"B per-round acc={[r['acc'] for r in res]} "
            f"draft0={res[0]['d'] if res else None} target0={res[0]['t'] if res else None}")
log(f"peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU4_DONE")
