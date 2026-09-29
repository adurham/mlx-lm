#!/usr/bin/env python3
"""pV14 -- the dedup ceiling vs today's A2, measured on real routed windows.

Correcting pV11/pV12: A2 reads one tile per (slot, tile) pair, so it does NOT
dedup -- its cost is flat in unique-expert count because its WORK is flat, one
decode per slot. That is precisely the waste PLAN 2b(a) targets.

Here the ceiling is measured the honest way: take a real R-row window, group
slots by expert, and run ONE mt-blocked dense GEMM per unique expert over the
rows routed to it (each expert's tiles decoded exactly once for all its rows).
Compare against _decode_fused2 and against the existing _prefill segmented path.

Run (production down, 1 layer, ~2 GB):
  PV14_PKG=~/dsv41-ws2/V EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV14_ceiling2.py
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV14_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx

from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3.layer_state import _had_fn

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PV14_LAYER", "20"))
REPS = int(os.environ.get("PV14_REPS", "21"))
SWEEPS = int(os.environ.get("PV14_SWEEPS", "3"))
TRACE = os.environ.get("PV14_TRACE", "1") == "1"


def log(*a):
    print("[pV14]", *a, flush=True)


ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
GT, DT, K, CB = sg._gu_tiles, sg._dn_tiles, sg._k, sg._cb
mx.eval(sg._gu_trellis, sg._dn_trellis, sg._gu_suh, sg._gu_svh, sg._dn_suh, sg._dn_svh)
log(f"layer {LAYER} E={E} D={D} H={H} k={K} gu_tiles={GT} dn_tiles={DT}")
rng = np.random.RandomState(41)


def timed(fn, reps=REPS, warm=4):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def window(R, kk=7):
    """One R-row verify window; routed experts from the real p30 trace."""
    if TRACE and os.path.exists(f"{HOME}/p30-records-clamp/L{LAYER:02d}.json"):
        d = json.load(open(f"{HOME}/p30-records-clamp/L{LAYER:02d}.json"))
        I = np.array([e[1] for e in d])
        s = rng.randint(0, len(I) - R)
        routed = I[s:s + R]
    else:
        routed = np.stack([rng.permutation(E - 1)[:kk - 1] for _ in range(R)])
    idx = np.concatenate([routed, np.full((R, 1), E - 1)], axis=1)
    return mx.array(idx[None].astype(np.uint32)), idx


def ceiling(idx_np, x):
    """One mt-blocked dense GEMM per unique expert over its rows.

    Emits raw pre-Hadamard y, same convention as the A2 kernel, laid out
    (n_slots_seen_so_far, 2H) in slot order (gate first, then up)."""
    R = idx_np.shape[0]
    flat = idx_np.reshape(-1)
    rows = np.repeat(np.arange(R), idx_np.shape[1])
    parts = []
    for i in range(len(flat)):
        pass
    for e in np.unique(flat):
        if e == E - 1:
            continue
        sel = np.where(flat == e)[0]
        xa = x[0][mx.array(rows[sel])]
        gu_g = sg._gu_trellis[:, e * GT:(e + 1) * GT, :]
        gu_u = sg._gu_trellis[:, (E + e) * GT:(E + e + 1) * GT, :]
        xh_g = _had_fn("pre_scaled")(xa, mx.tile(sg._gu_suh[e, 0], (len(sel), 1)))
        xh_u = _had_fn("pre_scaled")(xa, mx.tile(sg._gu_suh[e, 1], (len(sel), 1)))
        g = G.inner_gemm_mlx(xh_g, gu_g, K, CB)
        u = G.inner_gemm_mlx(xh_u, gu_u, K, CB)
        parts.append((sel, mx.concatenate([g, u], axis=-1)))
    order = np.concatenate([p[0] for p in parts])
    ys = mx.concatenate([p[1] for p in parts], axis=0)
    inv = np.argsort(order)
    return ys[mx.array(inv)], order, inv


RS = [int(v) for v in os.environ.get("PV14_ROWS", "2,4,6").split(",")]
for R in RS:
    idx, idx_np = window(R)
    x = mx.array(rng.randn(1, R, D).astype(np.float16))
    mx.eval(idx, x)
    n_uniq = len(np.unique(idx_np))
    slots = R * 7

    def a2():
        return sg._decode_fused2(x.reshape(R, D), idx.reshape(R, 7))
    a2_med, a2_min = timed(a2)

    def ceil():
        return ceiling(idx_np, x)[0]
    c_med, c_min = timed(ceil)

    def pf():
        return sg._prefill(x, idx)
    try:
        pf_med, pf_min = timed(pf)
    except Exception as ex:
        pf_med = pf_min = float("nan")
        log(f"  _prefill failed: {type(ex).__name__}: {ex}")

    log(f"  R={R} ({slots} slots, {n_uniq} distinct experts incl shared):")
    log(f"     _decode_fused2 (today)  {a2_med:7.3f} ms (min {a2_min:7.3f})")
    log(f"     dedup ceiling (dense/u) {c_med:7.3f} ms (min {c_min:7.3f})  "
        f"ratio {c_med/a2_med:.2f}x")
    log(f"     _prefill (segmented)    {pf_med:7.3f} ms  ratio {pf_med/a2_med:.2f}x")

    # correctness: does the ceiling agree with A2?  (fp16 dense path vs the
    # kernel: expect near-identity, NOT bit-identity)
    ya = sg._decode_fused2(x.reshape(R, D), idx.reshape(R, 7))
    yb, order, inv = ceiling(idx_np, x)
    mx.eval(ya, yb)
    a = np.array(ya.astype(mx.float32)).reshape(R * 7, D)
    b = np.array(yb.astype(mx.float32))
    # A2 slot s covers (row s//7, routed slot s%7) for the non-shared experts;
    # the shared slot (s%7==6) is not in the ceiling.
    keep = [s for s in range(R * 7) if idx_np.reshape(-1)[s] != E - 1]
    bmap = {int(o): i for i, o in enumerate(order)}
    idxs = [bmap[s] for s in keep]
    aa = a[np.array(keep)]
    bb = b[np.array(idxs)]
    cos = (aa * bb).sum(-1) / (np.linalg.norm(aa, axis=-1) *
                               np.linalg.norm(bb, axis=-1) + 1e-30)
    log(f"     ceiling vs A2 on the shared slots ({len(keep)} of {R*7}): "
        f"min cos {cos.min():.6f}, max|d| {np.abs(aa-bb).max():.4g}")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV14_DONE")
