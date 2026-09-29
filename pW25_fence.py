#!/usr/bin/env python3
"""pW25 -- is the per-layer mx.eval fence in sparse_attn the carrier of the
first-step premium? (30 layers / 16K)

pW23: sparse_attn holds 530/549 ms of step-0 host time, same call count as
steady (38 ms). sparse_attn ends each call with mx.eval(out) (_FENCE='qtile'),
i.e. 30 blocking host syncs per decode step -- fences that exist to bound
PREFILL transients and buy nothing for a single-row decode step.

Arms:
  A  default            (DSV41_SPARSE_FENCE=qtile)
  B  no fence           (DSV41_SPARSE_FENCE=0)
  C  fence='ktile'      (per key tile; strictly more syncs)
Reported per arm: steps 0..3 build/eval ms. A vs B answers whether the fence
carries the premium; parity follows from the module docstring (fenced and
unfenced runs are bit-identical: same ops, same dtypes, only sync points move).

Env: PW_LAYERS (0..29), PW_CTX (16384), PW_STEPS (4)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
import mlx.core as mx  # noqa: E402

# FENCE must be set before sparse_attention is imported
FENCES = os.environ.get("PW_FENCE", "qtile,0,ktile").split(",")
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(30))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "4"))
MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
os.environ["DSV41_SPARSE_FENCE"] = FENCES[0]
from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402
from mlx_lm.models.deepseek_v41 import sparse_attention as SA  # noqa: E402

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW25] layers={LAYERS} n={len(model.layers)} active={mx.get_active_memory()/1e9:.2f}GB",
      flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]


def arm(fence):
    SA._FENCE = fence
    cache = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
    mx.eval(am)
    tp = time.perf_counter() - t0
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
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
    print(f"[pW25] fence={fence:6s}: prefill {tp:.1f}s | " + " ".join(
        f"s{i} {b_:7.2f}+{e:5.2f}" for i, (b_, e) in enumerate(out))
        + f" | first {out[0][0]+out[0][1]:7.2f}ms steady {steady:6.2f}ms "
          f"ratio {(out[0][0]+out[0][1])/steady:.2f}x", flush=True)
    del cache
    mx.clear_cache()


for f in FENCES:
    arm(f)
print(f"[pW25] peak {mx.get_peak_memory()/1e9:.2f}GB PW25_DONE", flush=True)
