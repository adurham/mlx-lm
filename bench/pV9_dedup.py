#!/usr/bin/env python3
"""pV9 -- is the MoE verify path worth batching rows per expert?

The dense small-batch GEMM already decodes a trellis tile ONCE per MT=8 rows
(mt-blocked). The MoE A2/B2 kernels are one-row-per-threadgroup. This probe
prices both on the SAME geometry (one real layer-20 expert, k=2, packed=32):

  t_dense(R): inner_gemm_mlx on a single expert's gate+up trellis, R rows
  t_a2(R)   : the MoE A2 kernel over R slots (R distinct experts, 1 row each)

If t_dense(4) is far below t_a2(4), batching rows per unique expert is the win.

Run:
  PV9_PKG=~/dsv41-ws2/V EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV9_dedup.py
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV9_PKG", HOME + "/dsv41-ws2/V"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3.layer_state import _had_fn

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PV9_LAYER", "20"))
REPS = int(os.environ.get("PV9_REPS", "15"))


def log(*a):
    print("[pV9]", *a, flush=True)


def timeit(fn, reps=REPS, warm=4):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
GT, DT, K, CB = sg._gu_tiles, sg._dn_tiles, sg._k, sg._cb
mx.eval(sg._gu_trellis, sg._dn_trellis, sg._gu_suh, sg._gu_svh, sg._dn_suh, sg._dn_svh)
log(f"layer {LAYER} E={E} D={D} H={H} k={K} gu_tiles={GT} dn_tiles={DT} "
    f"gu_trellis={sg._gu_trellis.shape} dn_trellis={sg._dn_trellis.shape}")
rng = np.random.RandomState(9)
EXP = int(os.environ.get("PV9_EXPERT", "7"))

# --- one expert's gate+up trellis, contiguous (in, 2*GT, P) ----------------
gu_e = mx.concatenate([
    sg._gu_trellis[:, EXP * GT:(EXP + 1) * GT, :],
    sg._gu_trellis[:, (E + EXP) * GT:(E + EXP + 1) * GT, :],
], axis=1)
dn_e = sg._dn_trellis[:, EXP * DT:(EXP + 1) * DT, :]
mx.eval(gu_e, dn_e)
log(f"single-expert gu {gu_e.shape} dn {dn_e.shape}")
suh_g = sg._gu_suh[EXP, 0].astype(mx.float16)
suh_u = sg._gu_suh[EXP, 1].astype(mx.float16)
suh_d = sg._dn_suh[EXP].astype(mx.float16)
mx.eval(suh_g, suh_u, suh_d)


def dense_gu(x):
    xh_g = _had_fn("pre_scaled")(x, mx.tile(suh_g, (x.shape[0], 1)))
    xh_u = _had_fn("pre_scaled")(x, mx.tile(suh_u, (x.shape[0], 1)))
    xh = mx.concatenate([xh_g, xh_u], axis=0)
    return G.inner_gemm_mlx(xh, gu_e, K, CB)


def dense_dn(h):
    xh = _had_fn("pre_scaled")(h, mx.tile(suh_d, (h.shape[0], 1)))
    return G.inner_gemm_mlx(xh, dn_e, K, CB)


# --- the MoE A2/B2 kernels, one row per slot -------------------------------
kA, kB = sg._kernels2()


def a2(x2d, sel):
    R = int(x2d.shape[0])
    E_sel = R
    sel_u = sel.astype(mx.uint32)
    suh_sel = sg._gu_suh[sel].reshape(E_sel * 2, D)
    x_rep = mx.broadcast_to(x2d[:, None, :], (R, 2, D)).reshape(E_sel * 2, D)
    xh = MOE._rows_prep()(x_rep, suh_sel)
    dims = mx.array([D // 16, GT, int(sg._gu_trellis.shape[1]), E], dtype=mx.uint32)
    return kA(inputs=[xh.reshape(-1), sg._gu_trellis.reshape(-1).view(mx.uint32),
                      MOE._fwd_perm_u32(), sel_u, dims],
              template=[("T", mx.float16)],
              grid=(E_sel * (2 * GT // MOE._A2_TILES) * MOE._GEM_THREADS, 1, 1),
              threadgroup=(MOE._GEM_THREADS, 1, 1),
              output_shapes=[(E_sel * 2 * H,)], output_dtypes=[mx.float16])[0]


def b2(h, sel):
    E_sel = int(sel.shape[0])
    sel_u = sel.astype(mx.uint32)
    dims_b = mx.array([H // 16, DT, int(sg._dn_trellis.shape[1]), E], dtype=mx.uint32)
    return kB(inputs=[h, sg._dn_trellis.reshape(-1).view(mx.uint32),
                      MOE._fwd_perm_u32(), sel_u, sg._gu_svh.reshape(-1),
                      sg._dn_suh.reshape(-1), sg._dn_svh.reshape(-1), dims_b],
              template=[("T", mx.float16)],
              grid=(E_sel * (DT // 8) * MOE._GEM_THREADS, 1, 1),
              threadgroup=(MOE._GEM_THREADS, 1, 1),
              output_shapes=[(E_sel * D,)], output_dtypes=[mx.float16])[0]


log("--- one expert gate+up: dense MT-blocked vs MoE A2 (ms) ---")
log(f"{'R':>3} {'dense_gu':>10} {'a2':>10} {'ratio':>7}   {'dense_dn':>10} {'b2':>10} {'ratio':>7}")
for R in (1, 2, 4, 8):
    x = mx.array(rng.randn(R, D).astype(np.float16))
    mx.eval(x)
    td = timeit(lambda: dense_gu(x)) * 1e3
    sel = mx.array(rng.permutation(E)[:R].astype(np.uint32))
    mx.eval(sel)
    ta = timeit(lambda: a2(x, sel)) * 1e3
    hh = mx.array(rng.randn(R, 2 * H).astype(np.float16))
    mx.eval(hh)
    td2 = timeit(lambda: dense_dn(xh_dummy := hh[:, :H])) * 1e3
    tb = timeit(lambda: b2(hh, sel)) * 1e3
    log(f"{R:3d} {td:10.3f} {ta:10.3f} {ta/td:7.2f}   {td2:10.3f} {tb:10.3f} {tb/td2:7.2f}")

log("--- the dedup workload shape: u unique experts, R rows each ---")
log("   (approximate: dense path for u*R rows through u DIFFERENT experts")
log("    cannot be expressed with one trellis; use the a2 per-slot slope)")
x1 = mx.array(rng.randn(1, D).astype(np.float16))
mx.eval(x1)
sel1 = mx.array([EXP], dtype=mx.uint32)
mx.eval(sel1)
t1 = timeit(lambda: a2(x1, sel1)) * 1e3
log(f"   a2 1 slot  : {t1:.3f} ms")
for slots in (7, 14, 28, 42):
    sel = mx.array(rng.permutation(E)[:slots].astype(np.uint32))
    x = mx.array(rng.randn(slots, D).astype(np.float16))
    mx.eval(sel, x)
    t = timeit(lambda: a2(x, sel)) * 1e3
    log(f"   a2 {slots:3d} slots: {t:7.3f} ms  ({t/slots:.4f} ms/slot, "
        f"{t/slots/2/GT/160:.4f} relative)")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV9_DONE")
