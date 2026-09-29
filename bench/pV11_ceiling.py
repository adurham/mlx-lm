#!/usr/bin/env python3
"""pV11 -- the dedup CEILING, measured properly.

Question: at FIXED slot count, does the MoE verify cost depend on how many
DISTINCT experts those slots point at? If yes, "read each unique expert once"
is addressable. If no, expert traffic is not the bottleneck and the premise in
PLAN 2b(a) cannot pay off.

Method (avoids the pV5 noise: interleaved arms, 31 reps, median AND min,
several repetitions of the whole sweep):
  * R=4 window, 24 routed slots (+shared), rows assigned round-robin.
  * u = number of distinct experts the 24 slots are drawn from
    (u=24 = all distinct = worst case for traffic; u=6 = maximum sharing).
  * measure sg._decode_fused2 end to end.
  * ALSO measure a dedup PROXY: group slots by expert and run the mt-blocked
    dense kernel once per (expert, its rows) -- the best case any dedup kernel
    could reach, since it decodes each expert's tiles exactly once.

Run (production down, 1 layer, ~2 GB):
  PV11_PKG=~/dsv41-ws2/V EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV11_ceiling.py
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV11_PKG", HOME + "/dsv41-ws2/V"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3.layer_state import _had_fn

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PV11_LAYER", "20"))
REPS = int(os.environ.get("PV11_REPS", "31"))
SWEEPS = int(os.environ.get("PV11_SWEEPS", "3"))
R = int(os.environ.get("PV11_ROWS", "4"))
KK = int(os.environ.get("PV11_KK", "7"))


def log(*a):
    print("[pV11]", *a, flush=True)


ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
GT, DT, K, CB = sg._gu_tiles, sg._dn_tiles, sg._k, sg._cb
mx.eval(sg._gu_trellis, sg._dn_trellis, sg._gu_suh, sg._gu_svh, sg._dn_suh, sg._dn_svh)
log(f"layer {LAYER} E={E} D={D} H={H} k={K} gu_tiles={GT} dn_tiles={DT}")
log(f"per-expert gu bytes = {320*144*32*2/1e6:.2f} MB, dn = {72*320*32*2/1e6:.2f} MB")
rng = np.random.RandomState(23)

# ---- fixed window: R rows x KK slots, u distinct experts, round-robin -----
routed = KK - 1          # MoE stacks the shared expert last


def make_idx(u):
    """(1,R,KK) uint32: routed slots drawn from exactly u distinct experts,
    rows assigned round-robin so a row never repeats an expert within itself."""
    pool = rng.permutation(E - 1)[:u]
    rows = []
    for r in range(R):
        e = [int(pool[(r * routed + j) % u]) for j in range(routed)]
        rows.append(e + [E - 1])
    return mx.array(np.stack(rows)[None].astype(np.uint32))


x = mx.array(rng.randn(1, R, D).astype(np.float16))
mx.eval(x)
log(f"window: R={R} rows, KK={KK} slots ({R*KK} total), routed={routed}/row")

# ---- the dedup proxy: per unique expert, one mt-blocked dense pass ----------
def dedup_proxy(idx_np):
    """Best case: each expert's tiles decoded exactly once for all its rows."""
    tot = 0.0
    flat = idx_np.reshape(-1)
    rows = np.repeat(np.arange(R), KK)
    for e in np.unique(flat):
        if e == E - 1:
            continue
        rs = rows[flat == e]                 # the rows routed to this expert
        xa = x[0][mx.array(rs)]              # (n_rows, D)
        gu_e = mx.concatenate([
            sg._gu_trellis[:, e * GT:(e + 1) * GT, :],
            sg._gu_trellis[:, (E + e) * GT:(E + e + 1) * GT, :],
        ], axis=1)
        suh_g = sg._gu_suh[e, 0]
        suh_u = sg._gu_suh[e, 1]
        xh_g = _had_fn("pre_scaled")(xa, mx.tile(suh_g, (xa.shape[0], 1)))
        xh_u = _had_fn("pre_scaled")(xa, mx.tile(suh_u, (xa.shape[0], 1)))
        y = G.inner_gemm_mlx(mx.concatenate([xh_g, xh_u], axis=0), gu_e, K, CB)
        mx.eval(y)
        tot += 1
    return tot


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


US = [int(v) for v in os.environ.get("PV11_US", "6,8,12,16,20,24").split(",")]
# keep index tensors fixed across sweeps so each u is measured repeatedly
IDX = {u: make_idx(u) for u in US}
for u, t in IDX.items():
    mx.eval(t)
    n_uniq = len(np.unique(np.array(t)))
    log(f"  u={u:3d}: generated window has {n_uniq} distinct slots "
        f"(unique/all = {n_uniq/(R*KK):.2f})")

log("--- end-to-end _decode_fused2 vs unique-expert count (slots FIXED) ---")
accum = {u: {"e2e": [], "px": []} for u in US}
for sw in range(SWEEPS):
    for u in US:
        idx = IDX[u]
        med, mn = timed(lambda idx=idx: sg._decode_fused2(x.reshape(R, D), idx.reshape(R, KK)))
        accum[u]["e2e"].append(med)
    for u in US:
        idx = IDX[u]
        med, mn = timed(lambda idx=idx: dedup_proxy(np.array(idx)))
        accum[u]["px"].append(med)
    log(f"  sweep {sw+1}/{SWEEPS} done")
log(f"{'u':>4} {'distinct':>9} {'e2e ms':>9} {'e2e min':>9} {'proxy ms':>9} {'proxy/e2e':>10}")
base = None
for u in US:
    e = float(np.median(accum[u]["e2e"]))
    p = float(np.median(accum[u]["px"]))
    n_uniq = len(np.unique(np.array(IDX[u])))
    if base is None:
        base = (u, e, p)
    log(f"{u:4d} {n_uniq:9d} {e:9.3f} {min(accum[u]['e2e']):9.3f} "
        f"{p:9.3f} {p/e:10.2f}")
log(f"NOTE: A2 reads one trellis tile per (slot, tile) pair, so the BYTES read "
    f"are the SAME for every u (S*2*gu_tiles tiles = {R*KK*2*GT} tiles, "
    f"{R*KK*2*144*32*8/1e6:.0f} MB). Flat cost across u therefore shows A2 does "
    f"NOT dedup -- it is not evidence that expert bytes are free. Compare with "
    f"the per-expert dense GEMM in pV9 for the dedup ceiling.")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV11_DONE")
