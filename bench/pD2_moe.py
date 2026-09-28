#!/usr/bin/env python3
"""pD2 -- EXL3SwitchGLU._prefill benchmark at DSv4.1 rank-0 MoE geometry.

Real build: load_experts(ck, layer, rank=0, world=2) -> 384 experts, half
intermediate width (hidden 1152), D=5120, top-6.  R=512 rows => N=3072
(token,slot) pairs over 384 experts => ~8 rows/expert (tiny segments).

Measures the v19c segmented path vs the gather_mm fallback, block-table
occupancy (what fraction of each 64-row block is live), and the decode-vs-mma
split via EXL3_MM_NODECODE (ablation must be run separately since the kernel is
built at import from the env var).

Env: PD_PKG, PD_ROWS (512), PD_KK (6), PD_REPS (20), PD_EXPERTS (384 or fewer),
     PD_MOE_MM (1/0), PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
PKG = os.environ.get("PD_PKG", HOME + "/dsv41-ws/D")
sys.path.insert(0, PKG)
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
ROWS = int(os.environ.get("PD_ROWS", "512"))
KK = int(os.environ.get("PD_KK", "6"))
REPS = int(os.environ.get("PD_REPS", "20"))
NE = os.environ.get("PD_EXPERTS")
NE = int(NE) if NE else None
OUT = os.environ.get("PD_JSON", HOME + "/dsv41-ws/D/pD2_moe.json")


def log(*a):
    print("[pD2]", *a, flush=True)


def timeit(fn, reps=REPS, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, n_experts=NE, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
log(f"layer={LAYER} E={E} D={D} H={H} k={sg._k} act={sg._activation} "
    f"gu_tiles={sg._gu_tiles} dn_tiles={sg._dn_tiles} "
    f"gu_trellis={tuple(sg._gu_trellis.shape)} dn_trellis={tuple(sg._dn_trellis.shape)}")
log(f"trellis bytes: gu={sg._gu_trellis.size*2/1e6:.0f}MB dn={sg._dn_trellis.size*2/1e6:.0f}MB")

rng = np.random.RandomState(7)
x = mx.array(rng.randn(1, ROWS, D).astype(np.float16))
idx = mx.array(np.stack([rng.permutation(E)[:KK] for _ in range(ROWS)])[None].astype(np.uint32))
mx.eval(x, idx)

res = {"layer": LAYER, "rows": ROWS, "kk": KK, "E": E, "D": D, "H": H,
       "n_pairs": ROWS * KK, "n_experts_loaded": E}

# --- block-table occupancy: what the v19c kernel actually covers ------------
flat = idx.reshape(-1).astype(mx.uint32)
sidx = mx.sort(flat)
counts = np.bincount(np.array(sidx), minlength=E)
live_blocks = int(np.sum((counts + 63) // 64))
bm_rows = live_blocks * 64
res["seg"] = {
    "pairs": int(flat.size),
    "avg_rows_per_expert": float(flat.size / E),
    "max_rows_per_expert": int(counts.max()),
    "experts_hit": int((counts > 0).sum()),
    "live_blocks_bm64": live_blocks,
    "block_rows_bm64": bm_rows,
    "row_occupancy_bm64": float(flat.size / bm_rows),
    "mma_waste_factor": float(bm_rows / flat.size),
}
# what a bm=16 table would give
lb16 = int(np.sum((counts + 15) // 16))
res["seg"]["mma_waste_factor_bm16"] = float(lb16 * 16 / flat.size)
lb8 = int(np.sum((counts + 7) // 8))
res["seg"]["mma_waste_factor_bm8"] = float(lb8 * 8 / flat.size)
log("segment stats:", json.dumps(res["seg"], indent=1))

# --- parity vs the gather_mm fallback (which is the reference semantics) -----
MOE._MOE_MM = True
mx.reset_peak_memory()
y_mm = sg._prefill(x, idx)
mx.eval(y_mm)
res["peak_mm_MB"] = mx.get_peak_memory() / 1e6
res["ms_mm"] = timeit(lambda: sg._prefill(x, idx)) * 1e3

MOE._MOE_MM = False
try:
    mx.reset_peak_memory()
    y_g = sg._prefill(x, idx)
    mx.eval(y_g)
    res["peak_gather_MB"] = mx.get_peak_memory() / 1e6
    res["ms_gather"] = timeit(lambda: sg._prefill(x, idx)) * 1e3
    res["bit_parity_mm_vs_gather"] = bool(mx.array_equal(y_mm, y_g))
    if not res["bit_parity_mm_vs_gather"]:
        a = np.array(y_mm.astype(mx.float32)); b = np.array(y_g.astype(mx.float32))
        res["maxabs_mm_vs_gather"] = float(np.abs(a - b).max())
        res["cos_mm_vs_gather"] = float(np.dot(a.ravel(), b.ravel()) /
            (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-30))
finally:
    MOE._MOE_MM = True

# --- whole __call__ (what the model actually invokes) ------------------------
res["ms_call_512"] = timeit(lambda: sg(x, idx)) * 1e3
log(f"N={ROWS*KK} pairs: v19c={res['ms_mm']:.2f}ms gather={res.get('ms_gather', float('nan')):.2f}ms "
    f"call={res['ms_call_512']:.2f}ms  peak(mm)={res['peak_mm_MB']:.0f}MB "
    f"peak(gather)={res.get('peak_gather_MB', float('nan')):.0f}MB "
    f"parity={res.get('bit_parity_mm_vs_gather')}")
log(f"per-layer MLP floor at 500GB/s if trellis read once: "
    f"{(sg._gu_trellis.size+sg._dn_trellis.size)*2/5e11*1e3:.2f}ms")

json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
