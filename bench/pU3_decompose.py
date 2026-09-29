#!/usr/bin/env python3
"""pU3 -- decompose a DSv4.1 DSpark spec round; account for the ~11 ms. (U)

Geometry: full 40 layers, single node, rank 0 of world 2 (half-width experts,
no collectives -- the geometry p48_bench/p56 use for single-node timing),
optionally expert-capped to fit the node budget. Absolute ms are NOT the
two-node production numbers; the ROUND STRUCTURE (which terms exist, which work
is forced at which point, how the terms add up) is.

Buckets per round (wall ms, host clocks):
  draft_host   head.draft() graph construction (host, + async dispatch)
  draft_sync   extra force of the draft tokens (iso=1 arms only)
  body_host    concat/snap/model() construction (host, + async dispatch, + any
               implicit sync inside -- e.g. the engram hasher materialises ids
               with np.array)
  wait         mx.eval(am, d) -- the round's own sync
  readback     np.array of the token outputs
  post         SP.rollback + head.append_ctx construction
  loop_gap     end of one round to the start of the next
The sum is compared against the measured loop wall time, so the residue is
printed as `unattributed`.

Arms: A async ON (production), B async OFF, C quiesced (GPU drained first),
D determinism at each eval placement, E gamma sweep, F verify row cost.

Env: PU3_LAYERS, PU3_WORLD, PU3_GAMMA, PU3_STEPS, PU3_BODY_EXPERTS,
     PU3_HEAD_EXPERTS, PU3_PKG.
NOTE: ``mx.eval`` is MLX's graph evaluation, not python's builtin.
"""
import json
import os
import statistics as S
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
PKG = os.environ.get("PU3_PKG", HOME + "/dsv41-ws2/U")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "bench"))

import mlx.core as mx  # noqa: E402

import pU_cap  # noqa: E402

_b = os.environ.get("PU3_BODY_EXPERTS")
_h = os.environ.get("PU3_HEAD_EXPERTS")
pU_cap.apply_caps(int(_b) if _b else None, int(_h) if _h else None)

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP  # noqa: E402
import mlx_lm.models.deepseek_v41.model as M  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get(
    "PU3_LAYERS", ",".join(str(i) for i in range(40))).split(",")]
WORLD = int(os.environ.get("PU3_WORLD", "2"))
GAMMA = int(os.environ.get("PU3_GAMMA", "3"))
STEPS = int(os.environ.get("PU3_STEPS", "16"))

mx.random.seed(0)
log = lambda *a: print("[pU3]", *a, flush=True)

t0 = time.time()
model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0,
                          world=WORLD)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ck = Exl3Checkpoint(MODEL)
head = eb.build_mtp(ck, model.args, rank=0, world=WORLD)
mx.eval(mx.ones(1))
log(f"built layers={len(LAYERS)} world={WORLD} body_cap={_b} head_cap={_h} "
    f"in {time.time() - t0:.0f}s active={mx.get_active_memory() / 1e9:.2f}GB "
    f"peak={mx.get_peak_memory() / 1e9:.2f}GB")

TAPS = list(model.args.dspark_target_layer_ids)
prompt = json.load(open(HOME + "/p30_prompt_ids.json"))[:48]


def tapcat(taps):
    return mx.concatenate([taps[L] for L in TAPS], axis=-1)


def make_state():
    cache = model.make_cache(1, max_seq_len=len(prompt) + STEPS * 8 + 64)
    am, taps = model(mx.array([prompt]), cache, last_logit_only=True,
                     return_taps=True, argmax=True)
    mx.eval(am)
    dsc = head.make_cache(1)
    head.append_ctx(tapcat(taps), dsc)
    mx.eval([c.win_kv for c in dsc])
    return cache, dsc, am[:, -1].astype(mx.int32)


def spec_rounds(gamma=GAMMA, n=STEPS, iso=0, quiesce=False):
    """The production round body, instrumented.

    iso=1 forces the draft tokens before the body graph is built, so the draft's
    GPU time is separable from body construction. quiesce additionally drains
    the GPU before the draft, so draft_host carries no previous-round tail.
    """
    cache, dsc, nxt = make_state()
    mx.eval(nxt)
    pos = cache.offset
    rows, toks = [], []
    t_prev = time.perf_counter()
    for _ in range(n):
        ta = time.perf_counter()
        gap = (ta - t_prev) * 1e3
        if quiesce:
            mx.synchronize()
        d, _ = head.draft(nxt, model.embed, model.head, dsc, width=gamma)
        d = d.astype(mx.int32)
        tb = time.perf_counter()
        if iso:
            mx.eval(d)
        tc = time.perf_counter()
        vin = mx.concatenate([nxt.reshape(1, 1), d], axis=1)
        sn = SP.snap(cache, pos)
        am, taps = model(vin, cache, return_taps=True, argmax=True)
        td = time.perf_counter()
        mx.eval(am, d)                                   # the round's one sync
        t1 = time.perf_counter()
        tg, dd = np.array(am[0]), np.array(d[0])
        t2 = time.perf_counter()
        nacc = 0
        while nacc < gamma and tg[nacc] == dd[nacc]:
            nacc += 1
        new = [int(v) for v in dd[:nacc]] + [int(tg[nacc])]
        SP.rollback(cache, sn, pos + nacc + 1, SP.stashes(cache))
        head.append_ctx(tapcat(taps)[:, :nacc + 1], dsc)
        pos = pos + nacc + 1
        nxt = mx.array([new[-1]], dtype=mx.int32)
        te = time.perf_counter()
        toks.extend(new)
        rows.append(dict(g=gamma, draft_host=(tb - ta) * 1e3,
                         draft_sync=(tc - tb) * 1e3,
                         body_host=(td - tc) * 1e3, wait=(t1 - td) * 1e3,
                         readback=(t2 - t1) * 1e3, post=(te - t2) * 1e3,
                         loop_gap=gap, acc=nacc, total=(te - ta) * 1e3))
        t_prev = te
    return rows, toks


TERMS = ["draft_host", "draft_sync", "body_host", "wait", "readback", "post"]


def show(tag, rows, toks):
    if not rows:
        log(f"{tag}: no rounds")
        return
    tot = lambda k: sum(r[k] for r in rows) / len(rows)
    dt = sum(r["total"] + r["loop_gap"] for r in rows) * 1e-3
    wall = dt * 1e3 / len(rows)
    log(f"--- {tag}: {len(rows)} rounds gamma={rows[0]['g']}  "
        f"{wall:.2f} ms/round  acc {np.mean([r['acc'] for r in rows]):.2f}  "
        f"tok/s {len(toks) / dt:.2f}")
    s = 0.0
    for k in TERMS + ["loop_gap"]:
        v = tot(k)
        s += v
        if abs(v) > 0.005:
            log(f"    {k:12s} mean {v:8.2f} ms/round")
    log(f"    {'sum':12s} mean {s:8.2f} ms/round  => unattributed "
        f"{s - wall:+.2f} ms/round")


rows, toks = spec_rounds(GAMMA, STEPS, iso=0)
show("A async ON  (production loop)", rows, toks)

M._ASYNC_EVAL = False
rows_b, toks_b = spec_rounds(GAMMA, STEPS, iso=0)
show("B async OFF", rows_b, toks_b)
M._ASYNC_EVAL = True

rows_c, toks_c = spec_rounds(GAMMA, STEPS, iso=1, quiesce=True)
show("C quiesced, draft forced", rows_c, toks_c)
log(f"tokens: A==B {toks == toks_b}   A==C {toks == toks_c}")

r1, k1 = spec_rounds(GAMMA, STEPS, iso=0)
r2, k2 = spec_rounds(GAMMA, STEPS, iso=0)
log(f"D iso=0 deterministic over 2 runs: {k1 == k2}")
r3, k3 = spec_rounds(GAMMA, STEPS, iso=1)
r4, k4 = spec_rounds(GAMMA, STEPS, iso=1)
log(f"D iso=1 deterministic over 2 runs: {k3 == k4}")
log(f"D iso=0 vs iso=1 streams equal: {k1 == k3}")

for g in (1, 2, 3, 4):
    rows_g, toks_g = spec_rounds(g, STEPS, iso=0)
    show(f"E gamma={g}", rows_g, toks_g)

cache, dsc, nxt = make_state()
for R in (1, 2, 3, 4, 5, 6):
    ts = []
    for _ in range(8):
        pos = cache.offset
        sn = SP.snap(cache, pos)
        s0 = time.perf_counter()
        lg = model(mx.full((1, R), 100, dtype=mx.int32), cache, argmax=True)
        mx.eval(lg)
        ts.append(time.perf_counter() - s0)
        SP.rollback(cache, sn, pos + 1, SP.stashes(cache))
    log(f"F verify R={R}: {S.median(ts) * 1e3:6.2f} ms")
log("F a gamma=3 round charges R=4 rows, not R=3")
log(f"peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU3_DONE")
