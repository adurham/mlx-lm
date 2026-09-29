#!/usr/bin/env python3
"""pW15 -- price what the decode boundary actually does: many fresh MTLBuffer
allocations at context-sized shapes, in one lazy graph, with the pool holding
PREFILL-shaped buffers only.

pW13 priced the *allocator* ops (reuse/sweep/release) and found them cheap;
pW10/pW11 measured 133 fresh device allocations on step 0 costing ~45 ms of
host build with eval flat. This probe closes the loop with NO model:

  1. build a "prefill-shaped" pool (2 MB tiles, ~3.7 GB) -- what the driver
     leaves behind;
  2. then allocate the "decode-shaped" working set of 20 layers, i.e. 20 x
     {kv_all 16514x512, idxs 1x640} + 20 x {kv_all 8321x512} etc, into one
     graph, and eval;
  3. repeat with the boundary warmed (same shapes allocated+eval'd+deleted
     once before), with and without a trailing clear_cache();

Prints host ms, eval ms, and device newBuffer counts per phase. Peak < 12 GB.
Env: PW15_LAYERS (20), PW15_CTX (16384)
"""
import os
import time

import mlx.core as mx

LAYERS = int(os.environ.get("PW15_LAYERS", "20"))
CTX = int(os.environ.get("PW15_CTX", "16384"))
D = 512


def fill_pool(total=3.7e9):
    """Prefill-shaped pool: 512-row x 512-d bf16 tiles (~0.5 MB) + 2 MB chunks."""
    arrs = []
    s, tot = 1 << 19, 0
    while tot < total:
        arrs.append(mx.zeros((512, 512), dtype=mx.bfloat16))
        tot += 512 * 512 * 2
        if len(arrs) % 8 == 0:
            s = min(s * 2, 1 << 21)
    mx.eval(arrs)
    del arrs
    mx.eval(mx.zeros((8,)))
    return mx.get_cache_memory()


def boundary_graph(nb1, nb2, layers):
    """One decode step's context-sized transients: per layer a kv_all concat
    (nb x 512 bf16) and an idxs concat (1 x 640 int32)."""
    outs = []
    for _ in range(layers):
        k1 = mx.zeros((1, nb1, D), dtype=mx.bfloat16)
        i1 = mx.zeros((1, 640), dtype=mx.int32)
        k2 = mx.zeros((1, nb2, D), dtype=mx.bfloat16)
        outs += [mx.concatenate([k1, mx.zeros((1, 1, D), dtype=mx.bfloat16)], axis=1),
                 mx.concatenate([i1, mx.zeros((1, 2), dtype=mx.int32)], axis=1),
                 mx.concatenate([k2, mx.zeros((1, 1, D), dtype=mx.bfloat16)], axis=1)]
    return outs


def nbuf_off():
    try:
        return os.path.getsize(LOG)
    except OSError:
        return 0


def nbuf(o0):
    with open(LOG, "rb") as f:
        f.seek(o0)
        return [int(x) for x in f.read().split(b"\n") if x.strip()]


LOG = "/tmp/pW15_bufs.log"
os.environ["MLX_LOG_NEW_BUFFER_PATH"] = LOG
try:
    os.remove(LOG)
except OSError:
    pass
# NOTE: MLX reads the env var at allocator construction; re-exec if needed.
if mx.get_active_memory() == 0 and not os.path.exists(LOG):
    pass

# 8K context: layer 2 sees nb=4096+128, layer 20 sees nb=8192+128 (ratio 1/2)
print(f"mlx {getattr(mx, '__version__', '?')} layers={LAYERS} ctx={CTX} log={LOG}", flush=True)

for phase, clear in (("pool-prefill-shaped", False), ("pool-empty", True)):
    mx.clear_cache()
    p = fill_pool()
    if clear:
        mx.clear_cache()
        p = mx.get_cache_memory()
    # cold boundary (like step 0)
    o0 = nbuf_off()
    t0 = time.perf_counter()
    outs = boundary_graph((CTX // 2) + 128, CTX + 128, LAYERS)
    tb = time.perf_counter() - t0
    t1 = time.perf_counter()
    mx.eval(outs)
    te = time.perf_counter() - t1
    n0 = nbuf(o0)
    print(f"[pW15] {phase} pool={p/1e9:.2f}GB COLD  build {tb*1e3:7.2f}ms eval {te*1e3:7.2f}ms "
          f"newBuf {len(n0)} {sum(n0)/1e6:.1f}MB", flush=True)
    del outs
    mx.eval(mx.zeros((8,)))
    # second boundary (pool now holds decode-shaped buffers)
    o0 = nbuf_off()
    t0 = time.perf_counter()
    outs = boundary_graph((CTX // 2) + 128, CTX + 128, LAYERS)
    tb = time.perf_counter() - t0
    t1 = time.perf_counter()
    mx.eval(outs)
    te = time.perf_counter() - t1
    n1 = nbuf(o0)
    print(f"[pW15] {phase} WARM  build {tb*1e3:7.2f}ms eval {te*1e3:7.2f}ms "
          f"newBuf {len(n1)} {sum(n1)/1e6:.1f}MB  pool {mx.get_cache_memory()/1e9:.2f}GB",
          flush=True)
    del outs
    mx.eval(mx.zeros((8,)))
    # third: warm again, to show steady reuse
    o0 = nbuf_off()
    t0 = time.perf_counter()
    outs = boundary_graph((CTX // 2) + 128, CTX + 128, LAYERS)
    tb = time.perf_counter() - t0
    t1 = time.perf_counter()
    mx.eval(outs)
    te = time.perf_counter() - t1
    n2 = nbuf(o0)
    print(f"[pW15] {phase} STEADY build {tb*1e3:7.2f}ms eval {te*1e3:7.2f}ms "
          f"newBuf {len(n2)} {sum(n2)/1e6:.1f}MB", flush=True)
    del outs
    mx.eval(mx.zeros((8,)))

print(f"[pW15] peak {mx.get_peak_memory()/1e9:.2f}GB PW15_DONE", flush=True)
