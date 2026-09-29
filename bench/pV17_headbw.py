#!/usr/bin/env python3
"""pV17 -- is the LM head bandwidth-bound? (the R>=2 verify floor)

pV16 measured the head at 2.253 ms (R=1) -> ~3.6 ms (R>=2) on one node. The
head trellis is 129280 x 5120 x ~2.56 effective bits ~= 1.32 GB, so a full read
at M4 Max sustained bandwidth is ~3.3-3.7 ms. This probe tests that directly:
decompress-equivalent reads of a same-size buffer, and the head's own cost with
the trellis read count varied by split count. If the head tracks raw read
bandwidth, the R>=2 floor is not removable by better kernels -- only by
shrinking the trellis (fewer bits) or not reading all of it.

Run (production down, 1 layer's worth of memory, ~3 GB):
  PV17_PKG=~/dsv41-ws2/V EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV17_headbw.py
"""
import os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV17_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx

from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_linear
from mlx_lm.models.deepseek_v41 import exl3_build as eb

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
REPS = int(os.environ.get("PV17_REPS", "21"))


def log(*a):
    print("[pV17]", *a, flush=True)


def timed(fn, reps=REPS, warm=4):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


ck = Exl3Checkpoint(CK)
head = eb.Exl3Proj(load_dense_linear(ck, "head"))
mx.eval(head.parameters())
rt = head._lin._rt
tbytes = int(np.prod(rt.trellis.shape)) * 2
log(f"head trellis: {rt.trellis.shape} = {tbytes/1e9:.3f} GB, "
    f"out={head._lin.out_features} in={head._lin.in_features}")

# --- raw read bandwidth reference: sum a same-size uint16 buffer -------------
for gb in (0.25, 1.32):
    n = int(gb * 1e9 // 2)
    buf = mx.zeros((n,), dtype=mx.uint16)
    mx.eval(buf)
    t = timed(lambda buf=buf: mx.sum(buf.astype(mx.uint32)))
    log(f"  raw read {gb:.2f} GB: {t:7.3f} ms -> {gb/(t/1e3):6.0f} GB/s")
    del buf

rng = np.random.RandomState(71)
log("--- head cost vs R, and what fraction of the trellis it reads ---")
for R in (1, 2, 3, 4, 6, 8):
    h = mx.array(rng.randn(1, R, 5120).astype(np.float32))
    mx.eval(h)
    t = timed(lambda h=h: head(h))
    log(f"  R={R}: {t:7.3f} ms  -> implied read rate {tbytes/(t/1e3)/1e9:.2f} GB/s "
        f"if it reads the whole trellis")
log("--- the argmax/head split in the real path (local logits only, no all_sum) ---")
from mlx_lm.models.deepseek_v41.exl3_build import ShardedHead
from mlx_lm.models.exl3 import EXL3Linear
log(f"  single-node build has head=Exl3Proj (not ShardedHead); the two-node "
    f"model shards vocab across ranks, so per-node head bytes are ~half: "
    f"{tbytes/2/1e9:.3f} GB")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV17_DONE")
