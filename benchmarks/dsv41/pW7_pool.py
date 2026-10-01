#!/usr/bin/env python3
"""pW7 -- the post-prefill decode cost is ALLOCATOR CACHE pressure, not compile.

Mechanism under test (from reading mlx/backend/metal/allocator.cpp):
  * the Metal pool (buffer_cache_) is filled by every freed transient; prefill
    frees hundreds of MB per 512-row chunk into it;
  * malloc() calls buffer_cache_.release_cached_buffers(...) whenever
    active + cache + size >= gc_limit_ (gc_limit_ = 0.95 * recommended working
    set ~ 114 GB at full scale). At 105 GB active + a prefill-fattened pool,
    EVERY decode-step allocation triggers a release sweep over the pool =
    thousands of MTL::Buffer releases -> the 1.3-1.9 s first steps;
  * it recurs after every prefill because each prefill refills the pool, and it
    decays over ~3 steps as the sweep eats the backlog.

Arms (2-layer subset, 8K prefill each):
  A  driver default                          (no pressure)
  B  A + trailing mx.clear_cache() after prefill   [the candidate fix]
  C  pressure emulated: set_memory_limit(active+2GB) after prefill
  D  C + trailing clear (fix under pressure)
  E  pool capped DURING prefill (set_cache_limit 256MB) + trailing clear

Env: PW_LAYERS (2,20), PW_CTX (8192), PW_STEPS (6)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW7_bufs.log")
import mlx.core as mx  # noqa: E402

BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]
from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS", "2,20").split(",")]
CTX = int(os.environ.get("PW_CTX", "8192"))
STEPS = int(os.environ.get("PW_STEPS", "6"))

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW7] layers={LAYERS} active={mx.get_active_memory()/1e9:.2f}GB", flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]

DEFAULT_MEMLIMIT = mx.get_memory_limit()
DEFAULT_CACHELIMIT = mx.get_memory_limit()
print(f"[pW7] default memory_limit={DEFAULT_MEMLIMIT/1e9:.1f}GB "
      f"pool={mx.get_cache_memory()/1e9:.2f}GB", flush=True)


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


def arm(name, *, pressure=False, trailing_clear=False, cap_during=False):
    cache = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None
    mx.reset_peak_memory()
    if cap_during:
        mx.set_cache_limit(256 << 20)
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True,
                    clear_cache_every=4, final_clear=trailing_clear)
    mx.eval(am)
    tp = time.perf_counter() - t0
    pf_cache = mx.get_cache_memory()
    if cap_during:
        mx.set_cache_limit(DEFAULT_MEMLIMIT)   # restore
    if trailing_clear and not cap_during:
        pass                      # prefill() did it
    if pressure:
        mx.set_memory_limit(mx.get_active_memory() + (2 << 30))
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    rows = []
    for i in range(STEPS):
        o0 = off()
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        mx.eval(lg)
        dt = time.perf_counter() - t0
        c, b = delta(o0)
        rows.append((tb * 1e3, (dt - tb) * 1e3, c, b / 1e6))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    if pressure:
        mx.set_memory_limit(DEFAULT_MEMLIMIT)
    print(f"[pW7] {name}: prefill {tp:.2f}s pool_after_prefill={pf_cache/1e9:.2f}GB "
          f"peak={mx.get_peak_memory()/1e9:.2f}GB", flush=True)
    for i, (b, e, c, mb) in enumerate(rows):
        print(f"[pW7]   step{i} build {b:8.2f}ms eval {e:8.2f}ms total {b+e:8.2f}ms "
              f"newBuffer {c:5d}/{mb:7.2f}MB  pool_now={mx.get_cache_memory()/1e6:8.1f}MB "
              f"active={mx.get_active_memory()/1e9:.2f}GB", flush=True)
    del cache
    mx.clear_cache()


arm("A default")
arm("B trailing-clear", trailing_clear=True)
arm("C pressure", pressure=True)
arm("D pressure+trailing-clear", pressure=True, trailing_clear=True)
arm("E cap-during-prefill+trailing-clear", cap_during=True, trailing_clear=True)
print("PW7_DONE", flush=True)
