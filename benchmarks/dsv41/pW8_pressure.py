#!/usr/bin/env python3
"""pW8 -- first-decode-step cost after long prefill: allocation, not compile.

Reproduces the full-scale state that the 2-layer subset cannot reach:
  * real single-node 8-layer prefill of 16K (pool left as prefill leaves it);
  * optional memory pressure (memory_limit = active + 1 GB) so every malloc
    hits MLX's gc branch (allocator.cpp: mem_required >= gc_limit_ ->
    buffer_cache_.release_cached_buffers over the whole pool);
  * per decode step: build ms / eval ms / GPU ms / device newBuffer calls +
    bytes (MLX_LOG_NEW_BUFFER_PATH) / pool size.

Arms (all after the same prefill; fresh cache per arm):
  F default            (no pressure)
  G pressure           (limit = active+1GB)
  H clear+pressure     (prefill(final_clear=True) then pressure)  <- candidate fix
  I seed+pressure      (clear, then a decode-shaped alloc/free sweep, then pressure)

Env: PW_LAYERS (0,1,2,3,20,21,24,25), PW_CTX (16384), PW_STEPS (6)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW8_bufs.log")
import mlx.core as mx  # noqa: E402

BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]
from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS", "0,1,2,3,20,21,24,25").split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "6"))
PRESSURE_GB = float(os.environ.get("PW_PRESSURE_GB", "1"))

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW8] layers={LAYERS} n={len(LAYERS)} active={mx.get_active_memory()/1e9:.2f}GB "
      f"memlimit={mx.get_memory_limit()/1e9:.1f}GB", flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]
MEMLIMIT = mx.get_memory_limit()


def off():
    try:
        return os.path.getsize(BUFL)
    except OSError:
        return 0


def delta(o0):
    with open(BUFL, "rb") as f:
        f.seek(o0)
        vals = [int(x) for x in f.read().split(b"\n") if x.strip()]
    return len(vals), sum(vals)


def seed_decode_shapes():
    """Allocate+free the decode-step transient shapes so the pool holds
    decode-sized buffers. ~30 MB, ~2 ms -- NOT a model forward."""
    d = 5120
    out = []
    for i in range(4):
        out.append(mx.zeros((1, 1, d), dtype=mx.bfloat16))
        out.append(mx.zeros((1, 1, 4, d), dtype=mx.bfloat16))
        out.append(mx.zeros((1, 1, 64, 512), dtype=mx.bfloat16))
        out.append(mx.zeros((1, 64), dtype=mx.float32))
    mx.eval(out)
    out2 = [mx.zeros((1, 1, 640, 512), dtype=mx.bfloat16) for _ in range(2)]
    mx.eval(out2)


def arm(name, *, pressure=False, final_clear=False, seed=False):
    cache = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True,
                    final_clear=final_clear)
    mx.eval(am)
    tp = time.perf_counter() - t0
    pool_pf = mx.get_cache_memory()
    pf_peak = mx.get_peak_memory()
    if seed:
        seed_decode_shapes()
    pool_seed = mx.get_cache_memory()
    if pressure:
        mx.set_memory_limit(mx.get_active_memory() + int(PRESSURE_GB * 1e9))
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    print(f"[pW8] {name}: prefill {tp:.2f}s peak {pf_peak/1e9:.2f}GB  "
          f"pool_after_prefill {pool_pf/1e9:.2f}GB  pool_after_seed {pool_seed/1e9:.2f}GB  "
          f"active {mx.get_active_memory()/1e9:.2f}GB  gc_limit~{mx.get_memory_limit()/1e9:.2f}GB",
          flush=True)
    rows = []
    for i in range(STEPS):
        o0 = off()
        p0 = mx.get_cache_memory()
        mx.metal.reset_gpu_time()
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        d0 = mx.metal.dispatch_count()
        mx.eval(lg)
        dt = time.perf_counter() - t0
        c, b = delta(o0)
        rows.append((tb * 1e3, (dt - tb) * 1e3, mx.metal.gpu_time_ns() / 1e6,
                     mx.metal.dispatch_count() - d0, c, b / 1e6,
                     p0 / 1e9, mx.get_cache_memory() / 1e9))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    for i, (b_, e, g, d, c, mb, p0, p1) in enumerate(rows):
        print(f"[pW8]   step{i} build {b_:8.2f} eval {e:8.2f} gpu {g:7.2f} disp {d:5d} "
              f"newBuf {c:5d}/{mb:7.2f}MB pool {p0:.2f}->{p1:.2f}GB", flush=True)
    mx.set_memory_limit(MEMLIMIT)
    del cache
    mx.clear_cache()


arm("F default")
arm("G pressure", pressure=True)
arm("H clear+pressure", pressure=True, final_clear=True)
arm("I seed+pressure", pressure=True, final_clear=True, seed=True)
print(f"[pW8] peak {mx.get_peak_memory()/1e9:.2f}GB PW8_DONE", flush=True)
