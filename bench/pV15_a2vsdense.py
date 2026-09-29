#!/usr/bin/env python3
"""pV15 -- A2 kernel vs per-expert dense GEMM on the SAME slots (correctness).

pV14 established the timing verdict: for a real R=2 window, today's A2 gate+up
costs 0.690 ms while the per-expert "dedup ceiling" (one mt-blocked dense GEMM
per unique expert, gate+up only, no down projection) costs 1.185 ms -- 1.72x
SLOWER, before paying for the extra launches a real dedup kernel would need.
This probe checks that the two agree numerically on the slots they share, so
the timing comparison is apples-to-apples (same math, different schedule).

Run (production down, 1 layer, ~2 GB):
  PV15_PKG=~/dsv41-ws2/V EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV15_a2vsdense.py
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV15_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx

from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3.layer_state import _had_fn

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PV15_LAYER", "20"))
REPS = int(os.environ.get("PV15_REPS", "21"))


def log(*a):
    print("[pV15]", *a, flush=True)


ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
GT, DT, K, CB = sg._gu_tiles, sg._dn_tiles, sg._k, sg._cb
mx.eval(sg._gu_trellis, sg._dn_trellis, sg._gu_suh, sg._gu_svh, sg._dn_suh, sg._dn_svh)
kA, kB = sg._kernels2()
rng = np.random.RandomState(51)
log(f"layer {LAYER} E={E} D={D} H={H} gu_tiles={GT}")


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


def a2_raw(x2d, sel):
    """The A2 kernel's raw pre-Hadamard gate|up output, one row per slot."""
    R = int(x2d.shape[0])
    suh_sel = sg._gu_suh[sel].reshape(R * 2, D)
    x_rep = mx.broadcast_to(x2d[:, None, :], (R, 2, D)).reshape(R * 2, D)
    xh = MOE._rows_prep()(x_rep, suh_sel)
    dims = mx.array([D // 16, GT, int(sg._gu_trellis.shape[1]), E], dtype=mx.uint32)
    return kA(inputs=[xh.reshape(-1), sg._gu_trellis.reshape(-1).view(mx.uint32),
                      MOE._fwd_perm_u32(), sel.astype(mx.uint32), dims],
              template=[("T", mx.float16)],
              grid=(R * (2 * GT // MOE._A2_TILES) * MOE._GEM_THREADS, 1, 1),
              threadgroup=(MOE._GEM_THREADS, 1, 1),
              output_shapes=[(R * 2 * H,)], output_dtypes=[mx.float16])[0]


def dense_raw(x2d, sel_np):
    """Same math via one mt-blocked dense GEMM per unique expert."""
    parts = []
    for e in np.unique(sel_np):
        sl = np.where(sel_np == e)[0]
        xa = x2d[mx.array(sl)]
        g = G.inner_gemm_mlx(_had_fn("pre_scaled")(xa, mx.tile(sg._gu_suh[e, 0], (len(sl), 1))),
                             sg._gu_trellis[:, e * GT:(e + 1) * GT, :], K, CB)
        u = G.inner_gemm_mlx(_had_fn("pre_scaled")(xa, mx.tile(sg._gu_suh[e, 1], (len(sl), 1))),
                             sg._gu_trellis[:, (E + e) * GT:(E + e + 1) * GT, :], K, CB)
        parts.append((sl, mx.concatenate([g, u], axis=-1)))
    order = np.concatenate([p[0] for p in parts])
    ys = mx.concatenate([p[1] for p in parts], axis=0)
    return ys[mx.array(np.argsort(order))]


for R in (2, 4, 6, 8):
    for trial in range(int(os.environ.get("PV15_TRIALS", "3"))):
        sel_np = rng.permutation(E)[:R].astype(np.uint32)
        x = mx.array(rng.randn(R, D).astype(np.float16))
        sel = mx.array(sel_np)
        mx.eval(x, sel)
        u = len(np.unique(sel_np))
        ya = a2_raw(x, sel)
        yb = dense_raw(x, sel_np)
        mx.eval(ya, yb)
        aa = np.array(ya.astype(mx.float32)).reshape(R, 2 * H)
        bb = np.array(yb.astype(mx.float32))
        cos = (aa * bb).sum(-1) / (np.linalg.norm(aa, axis=-1) *
                                   np.linalg.norm(bb, axis=-1) + 1e-30)
        log(f"  R={R} u={u}: min cos {cos.min():.6f}  max|d| {np.abs(aa-bb).max():.4g}")
    ta = timed(lambda: a2_raw(x, sel))
    td = timed(lambda: dense_raw(x, sel_np))
    log(f"   R={R}: A2 {ta:7.3f} ms | dense-per-expert {td:7.3f} ms | "
        f"dedup is {td/ta:.2f}x {'SLOWER' if td > ta else 'faster'}")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV15_DONE")
