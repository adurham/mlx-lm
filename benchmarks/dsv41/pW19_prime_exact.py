#!/usr/bin/env python3
"""pW19 -- exact-shape prime (page-exact) vs the boundary, at 30 layers/16K.

Root cause chain (measured):
  pW9/pW18: after a long prefill the first decode step makes ~200-340 fresh
  MTLBuffer allocations (device_->newBuffer) where steady steps make 0, and the
  per-allocation cost grows sharply with process footprint: ~0.1 ms at 23 GB,
  ~0.5 ms at 55 GB, ~2.5 ms at 80 GB. Host build ms: 15 (8 layers) -> 78
  (20) -> 485 (30). GPU work is FLAT (eval 3 ms, gpu_time 0) and every
  mx.compile key is warm, so this is pure allocation, not compilation.

  pW16/pW18 prime attempt 1 FAILED (269 misses, still 509 ms) because the
  primed row count was one page short of what decode asks for: MLX rounds a
  request up to vm_page_size before the pool lookup, and reuse requires
  size <= cached < size+32KB, so a one-page-short buffer is never reused.

  Exact decode shapes (n=1, start=offset, end=offset+1):
    kv_all rows = min(offset, window) + 1 + (offset+1)//ratio
    idxs   cols = min(window, kv_all_rows) + index_topk
This probe primes EXACTLY those (page-exact by construction) and checks the
miss count goes to ~0 with first-step ms == steady ms.

Env: PW_LAYERS (0..29), PW_CTX (16384), PW_STEPS (5)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW19_bufs.log")
import mlx.core as mx  # noqa: E402

BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]
from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(30))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "5"))


def off():
    try:
        return os.path.getsize(BUFL)
    except OSError:
        return 0


def misses(o0):
    with open(BUFL, "rb") as f:
        f.seek(o0)
        return [int(x) for x in f.read().split(b"\n") if x.strip()]


model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW19] layers={LAYERS} n={len(model.layers)} active={mx.get_active_memory()/1e9:.2f}GB",
      flush=True)
t0 = time.perf_counter()
PF.warmup(model)
print(f"[pW19] warmup {time.perf_counter()-t0:.1f}s", flush=True)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]
args = model.args


def prime_exact(cache, clear=True):
    """Page-exact prime of the decode working set for cache.offset."""
    t0 = time.perf_counter()
    offset = int(cache.offset)
    win, hd = int(args.window_size), int(args.head_dim)
    topk = int(getattr(args, "index_topk", 512))
    if clear:
        mx.clear_cache()
    bufs = []
    for layer in model.layers:
        ratio = int(getattr(layer.attn, "ratio", 0) or 0)
        n_kv = min(offset, win) + 1 + ((offset + 1) // ratio if ratio else 0)
        n_idx = min(win, n_kv) + (topk if ratio else 0)
        bufs.append(mx.zeros((1, n_kv, hd), dtype=mx.bfloat16))
        if n_idx:
            bufs.append(mx.zeros((1, n_idx), dtype=mx.int32))
        # the per-layer window gather [1, win, hd] is reused across steps
        bufs.append(mx.zeros((1, min(offset, win), hd), dtype=mx.bfloat16))
    mx.eval(bufs)
    del bufs
    mx.eval(mx.zeros((8,), dtype=mx.uint8))
    return time.perf_counter() - t0


def arm(name, prime=False):
    cache = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
    mx.eval(am)
    tp = time.perf_counter() - t0
    pf_peak = mx.get_peak_memory()
    tp2 = prime_exact(cache) if prime else 0.0
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    ts = []
    for i in range(STEPS):
        o0 = off()
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        mx.eval(lg)
        dt = time.perf_counter() - t0
        sz = misses(o0)
        ts.append((tb * 1e3, (dt - tb) * 1e3, len(sz), sum(sz) / 1e6))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    steady = float(np.median([t[0] + t[1] for t in ts[3:]]))
    print(f"[pW19] {name}: prefill {tp:.1f}s peak {pf_peak/1e9:.2f}GB prime {tp2*1e3:.1f}ms",
          flush=True)
    for i, (b_, e, nb, mb) in enumerate(ts):
        print(f"[pW19]   step{i} build {b_:8.2f} eval {e:6.2f} newBuf {nb:4d}/{mb:7.2f}MB "
              f"pool {mx.get_cache_memory()/1e9:.2f}", flush=True)
    print(f"[pW19] {name}: first {ts[0][0]+ts[0][1]:.2f}ms steady {steady:.2f}ms "
          f"ratio {(ts[0][0]+ts[0][1])/steady:.2f}x", flush=True)
    del cache
    mx.clear_cache()


arm("A default")
arm("B prime-exact", prime=True)
arm("C default-again")
arm("D prime-exact-again", prime=True)
print(f"[pW19] peak {mx.get_peak_memory()/1e9:.2f}GB PW19_DONE", flush=True)
