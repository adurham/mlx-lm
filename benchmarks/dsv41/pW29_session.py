#!/usr/bin/env python3
"""pW29 -- does the fix hold on a session-continuation delta prefill?

Round-2's real workload is a long first prompt followed by SHORT delta prefills
(one Hermes turn = a few hundred new tokens at a large offset). This probe
measures, with the driver default (prime on), at 30 layers:

  1. prefill 16K                       -> decode steps timed
  2. delta prefill of 128 new tokens   -> decode steps timed
  3. delta prefill of 512 new tokens   -> decode steps timed
and reports first-step vs steady for each, plus each prefill's own time
(the boundary prime's cost is billed there).

Env: PW_LAYERS (0..29), PW_CTX (16384), PW_DELTAS (128,512), PW_STEPS (4)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(30))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
DELTAS = [int(x) for x in os.environ.get("PW_DELTAS", "128,512").split(",")]
STEPS = int(os.environ.get("PW_STEPS", "4"))

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW29] layers={LAYERS} n={len(model.layers)} active={mx.get_active_memory()/1e9:.2f}GB",
      flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
pool = (base * 8)[:2048]


def run_steps(cache, nxt, label, off):
    out = []
    for i in range(STEPS):
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        mx.eval(lg)
        dt = time.perf_counter() - t0
        out.append((tb * 1e3, (dt - tb) * 1e3))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    steady = float(np.median([t[0] + t[1] for t in out[2:]]))
    first = out[0][0] + out[0][1]
    print(f"[pW29] {label} (offset {off}): " + " ".join(
        f"s{i} {b_:7.2f}+{e:5.2f}" for i, (b_, e) in enumerate(out))
        + f" | first {first:7.2f}ms steady {steady:6.2f}ms ratio {first/steady:.2f}x",
        flush=True)
    return nxt


cache = model.make_cache(1, max_seq_len=CTX + 4096)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None
mx.reset_peak_memory()
ids = (base * (CTX // len(base) + 1))[:CTX]
t0 = time.perf_counter()
am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True)
mx.eval(am)
print(f"[pW29] prefill {CTX} in {time.perf_counter()-t0:.1f}s "
      f"peak {mx.get_peak_memory()/1e9:.2f}GB (includes the boundary prime)", flush=True)
nxt = am.reshape(-1)[-1:].astype(mx.int32)
nxt = run_steps(cache, nxt, "turn-1 decode", int(cache.offset))

for D in DELTAS:
    piece = pool[:D]
    off0 = int(cache.offset)
    t0 = time.perf_counter()
    am = PF.prefill(model, piece, cache, last_logit_only=True, argmax=True)
    mx.eval(am)
    tdelta = time.perf_counter() - t0
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    print(f"[pW29] delta prefill {D} tok at offset {off0}: {tdelta*1e3:.1f}ms "
          f"({D/tdelta:.0f} tok/s, includes boundary prime)", flush=True)
    nxt = run_steps(cache, nxt, f"delta{D} decode", int(cache.offset))
print(f"[pW29] peak {mx.get_peak_memory()/1e9:.2f}GB PW29_DONE", flush=True)
