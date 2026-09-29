#!/usr/bin/env python3
"""pW23 -- cProfile of the slow first decode step at 30 layers / 16K.

Everything else is excluded so far (pW18-pW22): GPU/eval flat, compile keys
warm, allocation ~6 ms, priming neutral-to-worse, GC 0 collections during the
step and gc.disable() no help. Host build 480-650 ms on step 0, 50 ms steady,
and a repeat of the SAME step after rollback is 50 ms (pW21 E) -- so it is a
one-time host event, not per-shape work.

This run profiles step 0 and step 2 exactly (cProfile, tottime) and prints
both, plus explicit timers for the model's non-block pre/post sections.

Env: PW_LAYERS (0..29), PW_CTX (16384)
"""
import cProfile
import io
import json
import os
import pstats
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

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW23] layers={LAYERS} n={len(model.layers)} active={mx.get_active_memory()/1e9:.2f}GB",
      flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]
cache = model.make_cache(1, max_seq_len=CTX + 512)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None
mx.reset_peak_memory()
t0 = time.perf_counter()
am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
mx.eval(am)
print(f"[pW23] prefill {CTX} in {time.perf_counter()-t0:.2f}s peak {mx.get_peak_memory()/1e9:.2f}GB",
      flush=True)
nxt = am.reshape(-1)[-1:].astype(mx.int32)


def prof(label):
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
    tb = time.perf_counter() - t0
    pr.disable()
    t1 = time.perf_counter()
    mx.eval(lg)
    te = time.perf_counter() - t1
    sio = io.StringIO()
    pstats.Stats(pr, stream=sio).sort_stats("tottime").print_stats(26)
    print(f"[pW23] === {label}: build {tb*1e3:.1f}ms eval {te*1e3:.2f}ms", flush=True)
    print(sio.getvalue()[:6000], flush=True)
    return lg


lg = prof("step0")
nxt = lg.reshape(-1)[-1:].astype(mx.int32)
for i in range(1, 3):
    t0 = time.perf_counter()
    lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
    mx.eval(lg)
    print(f"[pW23] step{i} {((time.perf_counter()-t0)*1e3):.2f}ms", flush=True)
    nxt = lg.reshape(-1)[-1:].astype(mx.int32)
prof("steady-step")
print(f"[pW23] peak {mx.get_peak_memory()/1e9:.2f}GB PW23_DONE", flush=True)
