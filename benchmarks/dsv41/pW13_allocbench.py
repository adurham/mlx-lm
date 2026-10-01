#!/usr/bin/env python3
"""pW13 -- price the two MLX Metal-allocator operations that can cost ~1s.

From mlx/backend/metal/allocator.cpp + common/buffer_cache.h:
  * malloc(): reuse_from_cache (pool hit) else, when
    active+cache+size >= gc_limit_, buffer_cache_.release_cached_buffers(..)
    which either clear()s the WHOLE pool or walks it from the tail;
  * free(): recycle_to_cache when cache < max_pool_size_, else release to OS.

This probe measures, with NO model loaded, and for pool sizes 0.25..8 GB:
  1. ms to FILL a pool of P bytes (assorted sizes);
  2. ms for ONE malloc that triggers a full sweep (cache_limit set to 1 byte);
  3. ms/op for fresh device allocations (cache_limit=0) at various active sizes
     -- i.e. is device newBuffer itself expensive?
  4. ms for a malloc that only misses against a big pool (no sweep).

Memory: peak stays under ~10 GB. Prints ms numbers per pool size.
Env: PW13_MAX_GB (8), PW13_ACTIVE_GB (0)
"""
import os
import sys
import time

import mlx.core as mx

print("mlx", getattr(mx, "__version__", "?"), flush=True)
dev = mx.device_info()
print(f"[pW13] working_set={dev['max_recommended_working_set_size']/1e9:.1f}GB "
      f"ram={dev['memory_size']/1e9:.1f}GB memlimit={mx.get_memory_limit()/1e9:.1f}GB", flush=True)
DEFAULT_LIMIT = mx.get_memory_limit()


def sizes_for(total_bytes):
    """A mix of sizes that sums to ~total_bytes (0.25-4 MB each)."""
    out, tot, s = [], 0, 262144
    while tot < total_bytes:
        n = min(s, total_bytes - tot)
        if n >= 256:
            out.append(n)
            tot += n
        s = 262144 if s >= (4 << 20) else s * 2
    return out


def fill_pool(total):
    arrs = [mx.zeros((n,), dtype=mx.uint8) for n in sizes_for(total)]
    mx.eval(arrs)
    del arrs                # frees -> pool
    mx.eval(mx.zeros((8,)))
    return mx.get_cache_memory()


def timed(fn, reps=1):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


ACTIVE_GB = float(os.environ.get("PW13_ACTIVE_GB", "0"))
keep = None
if ACTIVE_GB > 0:
    t0 = time.perf_counter()
    keep = [mx.zeros((int(ACTIVE_GB * 1e9 / 2),), dtype=mx.uint8)]
    mx.eval(keep)
    print(f"[pW13] active ballast {ACTIVE_GB}GB in {(time.perf_counter()-t0)*1e3:.0f}ms "
          f"active={mx.get_active_memory()/1e9:.2f}GB", flush=True)

for gb in (0.25, 0.5, 1, 2, 4, 8):
    if gb > float(os.environ.get("PW13_MAX_GB", "8")):
        break
    mx.clear_cache()
    p = fill_pool(int(gb * 1e9))
    # 1) miss against a full pool, no sweep (normal case, plenty of headroom)
    def miss_only():
        a = mx.zeros((1 << 16,), dtype=mx.float16)   # 128 KB, likely in pool
        mx.eval(a)
    # 2) a malloc that forces the gc branch: limit = 1 byte below active+cache
    mx.set_cache_limit(1)
    def sweep():
        a = mx.zeros((1 << 20,), dtype=mx.uint8)     # 1 MB
        mx.eval(a)
    tsweep = timed(sweep)
    pool_after = mx.get_cache_memory()
    mx.set_cache_limit(DEFAULT_LIMIT)
    tmiss = timed(miss_only, reps=3)
    # 3) device allocation: cache_limit 0 -> every alloc is a fresh newBuffer
    mx.set_cache_limit(0)
    def fresh():
        a = mx.zeros((1 << 20,), dtype=mx.uint8)
        mx.eval(a)
        del a
    tfresh = timed(fresh, reps=3)
    mx.set_cache_limit(DEFAULT_LIMIT)
    print(f"[pW13] pool={gb:5.2f}GB filled={p/1e9:5.2f}GB:  sweep-1MB={tsweep:8.2f}ms "
          f"(pool left {pool_after/1e9:.2f}GB)  pool-miss={tmiss:6.2f}ms  "
          f"fresh-device-alloc={tfresh:6.2f}ms  active={mx.get_active_memory()/1e9:.2f}GB",
          flush=True)

mx.set_cache_limit(DEFAULT_LIMIT)
mx.clear_cache()
print(f"[pW13] peak {mx.get_peak_memory()/1e9:.2f}GB PW13_DONE", flush=True)
