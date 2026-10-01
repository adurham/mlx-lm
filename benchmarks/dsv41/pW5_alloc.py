#!/usr/bin/env python3
"""pW5 -- is the first post-prefill decode step paying ALLOCATOR cost, not compile?

Instrumentation: MLX_LOG_NEW_BUFFER_PATH (the fork's allocator logs every
device_->newBuffer call). We record the log-file offset before/after each
decode step -> newBuffer CALLS and BYTES per step, next to build/eval ms.

Scenarios (same 2-layer build, 8K prefill each time, fresh cache):
  A default driver (clear_cache_every=4)
  B driver with clear_cache_every=0
  C default driver, then mx.set_cache_limit(0) before decoding  (pool starved)
  D default driver, then a decode-shape POOL PRIME, then decode

Env: PW_LAYERS (2,20), PW_CTX (8192), PW_STEPS (6)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW5_bufs.log")
import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS", "2,20").split(",")]
CTX = int(os.environ.get("PW_CTX", "8192"))
STEPS = int(os.environ.get("PW_STEPS", "6"))
BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW5] layers={LAYERS} active={mx.get_active_memory()/1e9:.2f}GB", flush=True)
t0 = time.perf_counter()
PF.warmup(model)
print(f"[pW5] warmup {time.perf_counter()-t0:.1f}s", flush=True)

base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]


def buf_off():
    try:
        return os.path.getsize(BUFL)
    except OSError:
        return 0


def buf_delta(o0):
    with open(BUFL, "rb") as f:
        f.seek(o0)
        lines = f.read().split(b"\n")
    vals = [int(v) for v in lines if v.strip()]
    return len(vals), sum(vals)


def scenario(name, clear_every=4, starve=False, prime=False):
    cache = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True,
                    clear_cache_every=clear_every)
    mx.eval(am)
    tp = time.perf_counter() - t0
    peak_pf = mx.get_peak_memory()
    if starve:
        mx.set_cache_limit(0)
    if prime:
        # allocate & free one decode step's worth of the biggest transient
        # shapes so the pool holds decode-shaped buffers (NOT a model run)
        primes = []
        for _ in range(1):
            primes.append(mx.zeros((1, 1, 5120), dtype=mx.bfloat16))
            primes.append(mx.zeros((1, 1, 4, 5120), dtype=mx.bfloat16))
            primes.append(mx.zeros((1, 1, 64, 512), dtype=mx.bfloat16))
            primes.append(mx.zeros((1, 1, 640, 512), dtype=mx.bfloat16))
            primes.append(mx.zeros((1, 1, 64, 640), dtype=mx.float32))
        mx.eval(primes)
        del primes
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    rows = []
    for i in range(STEPS):
        o0 = buf_off()
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        mx.eval(lg)
        dt = time.perf_counter() - t0
        nb_cnt, nb_bytes = buf_delta(o0)
        rows.append((tb * 1e3, (dt - tb) * 1e3, nb_cnt, nb_bytes / 1e6))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    print(f"[pW5] {name}: prefill {tp:.2f}s peak {peak_pf/1e9:.2f}GB", flush=True)
    for i, (b, e, c, mb) in enumerate(rows):
        print(f"[pW5]   step{i} build {b:8.2f}ms eval {e:8.2f}ms  newBuffer {c:5d} calls {mb:7.2f} MB",
              flush=True)
    print(f"[pW5]   peak {mx.get_peak_memory()/1e9:.2f}GB active {mx.get_active_memory()/1e9:.2f}GB "
          f"cache {mx.metal.get_cache_memory()/1e9:.2f}GB", flush=True)
    del cache
    mx.clear_cache()


scenario("A default(clear=4)")
scenario("B clear=0", clear_every=0)
scenario("C starved(cache_limit=0)", starve=True)
scenario("D prime", prime=True)
print("PW5_DONE", flush=True)
