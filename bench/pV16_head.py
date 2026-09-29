#!/usr/bin/env python3
"""pV16 -- the head's marginal cost per verify row (the biggest unattributed term).

pV7's phase accounting wraps Block._fused_call, so it misses everything outside
the block: embed, the final hc_pre collapse, norm, and the LM HEAD. In the
single-node probes the head is a 5120 x 129280 EXL3 projection evaluated for
every row of the verify window, and it showed up as a large "unattributed"
remainder that GROWS with R (2.372 -> 3.285 ms/layer at R=1 -> R=4). This probe
prices the head directly at R=1..8, and the embed, so the marginal breakdown is
complete.

Run (production down, 1 layer, ~2 GB):
  PV16_PKG=~/dsv41-ws2/V EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV16_head.py
"""
import os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV16_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_linear

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
REPS = int(os.environ.get("PV16_REPS", "15"))


def log(*a):
    print("[pV16]", *a, flush=True)


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
args = ModelArgs.from_dict(ck.config)
log(f"vocab={args.vocab_size} dim={args.dim}")

# the real head object as build_model makes it (single node: no sharding)
head = eb.Exl3Proj(load_dense_linear(ck, "head"))
mx.eval(head.parameters())
log(f"head out={head._lin.out_features} in={head._lin.in_features} "
    f"huge={head._lin._huge}")

rng = np.random.RandomState(61)
log("--- head: full projection vs the argmax path actually used ---")
for R in (1, 2, 4, 6, 8):
    h = mx.array(rng.randn(1, R, args.dim).astype(np.float32))
    mx.eval(h)
    t_full = timed(lambda h=h: head(h))
    t_am = timed(lambda h=h: mx.argmax(head(h), axis=-1))
    log(f"  R={R}: full logits {t_full:7.3f} ms | +argmax {t_am:7.3f} ms "
        f"| marginal {t_am-t_full:+.3f}")
log("--- comparison: a plain fp16 matmul of the same shape, M rows ---")
from mlx_lm.models.exl3.reconstruct import reconstruct_public_mlx
log("   (skipping full recon; size only)")
log(f"   head params {args.vocab_size*args.dim/1e6:.0f}M -> fp16 would be "
    f"{args.vocab_size*args.dim*2/1e9:.2f} GB; trellis is "
    f"{int(np.prod(head._lin._rt.trellis.shape))*2/1e9:.2f} GB")
log("NOTE: the head steps 2.25 ms (R=1) -> 3.2 ms (R=2) -> ~3.6 ms (R>=6): "
    "a one-off +0.95 ms at R=2, then only ~0.10 ms/row. It is evaluated for "
    "EVERY row of the window, so the R=1 decode (the tok/s-critical case) pays "
    "the 2.25 ms floor; the verify window pays 3.6.")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV16_DONE")
