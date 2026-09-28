#!/usr/bin/env python3
"""pD4 -- _prefill timing + the gather-fallback overflow, at DSv4.1 geometry."""
import json, os, sys, time, traceback
HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
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
OUT = os.environ.get("PD_JSON", HOME + "/dsv41-ws/D/pD4_moe.json")


def log(*a):
    print("[pD4]", *a, flush=True)


def timeit(fn, reps=REPS, warm=3):
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
rng = np.random.RandomState(7)
x = mx.array(rng.randn(1, ROWS, D).astype(np.float16))
idx = mx.array(np.stack([rng.permutation(E)[:KK] for _ in range(ROWS)])[None].astype(np.uint32))
mx.eval(x, idx)
res = {"E": E, "D": D, "H": H, "rows": ROWS, "kk": KK}

# 1) v19c path timing (kernel forced on)
MOE._MOE_MM = True
mx.reset_peak_memory()
y_mm = sg._prefill(x, idx); mx.eval(y_mm)
res["v19c_peak_MB"] = mx.get_peak_memory() / 1e6
res["v19c_ms"] = timeit(lambda: sg._prefill(x, idx)) * 1e3
log(f"v19c N={ROWS*KK}: {res['v19c_ms']:.2f} ms  peak={res['v19c_peak_MB']:.0f} MB")

# 2) gather fallback: prove it overflows rather than measuring it
try:
    wd = G.decode_full_eg_mlx(sg._dn_trellis, sg._k, sg._cb, tiles_per_e=sg._dn_tiles)
    mx.eval(wd)
    res["gather_ok"] = True
    res["gather_dn_shape"] = list(wd.shape)
except Exception as e:
    res["gather_ok"] = False
    res["gather_error"] = f"{type(e).__name__}: {e}"
    res["gather_dn_bytes_would_be"] = sg.num_experts * sg.input_dims * sg.hidden_dims * 2
    log(f"gather dn decode FAILED: {res['gather_error']}")
    log(f"  (that output would be {res['gather_dn_bytes_would_be']/1e9:.2f} GB fp16)")

# 3) NODECODE ablation: how much of v19c is the trellis decode?
log(f"mma_waste_factor at bm64 = {384*64/(ROWS*KK):.1f}x  (384 blocks x 64 rows for {ROWS*KK} pairs)")
log(f"per-layer trellis bytes = {(sg._gu_trellis.size+sg._dn_trellis.size)*2/1e6:.0f} MB; "
    f"ideal read-once time @500GB/s = {(sg._gu_trellis.size+sg._dn_trellis.size)*2/5e11*1e3:.2f} ms")
res["trellis_MB"] = (sg._gu_trellis.size + sg._dn_trellis.size) * 2 / 1e6
res["ideal_read_once_ms"] = (sg._gu_trellis.size + sg._dn_trellis.size) * 2 / 5e11 * 1e3

# 4) whole __call__
res["call_ms"] = timeit(lambda: sg(x, idx)) * 1e3
log(f"__call__ = {res['call_ms']:.2f} ms")
json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
