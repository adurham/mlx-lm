#!/usr/bin/env python3
"""pDA -- pin the dense fused-vs-fullW crossover in the 16..64 row band.

FUSED_GEMM_ROW_LIMIT currently sends rows 17..64 to inner_gemm_mlx (which
re-decodes the trellis once per 8-row group), but pD8 measured fullW ahead from
~R=32 at DSv4.1 rank-0 shapes. Sweep the band finely and report the per-shape
crossover, the cost of the current threshold, and the numerical distance
between the two paths (cos + max|d|) so the switch can be judged.

Env: PD_LAYER, PD_REPS, PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import EXL3Linear
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
REPS = int(os.environ.get("PD_REPS", "25"))
ROWS = [int(x) for x in os.environ.get(
    "PD_ROWS_LIST", "16,17,20,24,28,32,40,48,56,64,80,96").split(",")]
OUT = os.environ.get("PD_JSON", HOME + "/dsv41-ws/D/pDA_cross_band.json")


def log(*a):
    print("[pDA]", *a, flush=True)


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


def sliced(name, axis):
    lay = load_dense_layer(ck, name)
    t = lay.trellis
    n = t.shape[1] if axis == "out" else t.shape[0]
    a, b = 0, n // 2
    kw = dict(key=name, k=lay.k, mul1=lay.mul1)
    if axis == "out":
        return type(lay)(in_features=lay.in_features, out_features=(b - a) * 16,
                         trellis=np.ascontiguousarray(t[:, a:b]),
                         suh=lay.suh, svh=lay.svh[a * 16:b * 16], **kw)
    return type(lay)(in_features=(b - a) * 16, out_features=lay.out_features,
                     trellis=np.ascontiguousarray(t[a:b]),
                     suh=lay.suh[a * 16:b * 16], svh=lay.svh, **kw)


SPECS = []
for nm, axis in (("attn.wq_a", None), ("attn.wkv", None), ("attn.wq_b", "out"),
                 ("attn.wo_b", "in"),
                 ("ffn.shared_experts.w1", "out"), ("ffn.shared_experts.w2", "in")):
    full = f"layers.{LAYER}.{nm}"
    if ck.has(full + ".trellis"):
        SPECS.append((nm, EXL3Linear(sliced(full, axis) if axis else load_dense_layer(ck, full))))

res = {"layer": LAYER, "shapes": {}, "rows": ROWS}
for nm, lin in SPECS:
    rt = lin._rt
    ent = {"in": lin.in_features, "out": lin.out_features, "rows": {}}
    for R in ROWS:
        x2d = mx.array(np.random.RandomState(R + 1).randn(R, lin.in_features).astype(np.float16))
        mx.eval(x2d)
        def f_fused():
            return rt.finish_y(G.inner_gemm_mlx(rt.prepare_xh(x2d), rt.trellis, rt.k, rt.cb).astype(mx.float16))
        def f_full():
            w = G.decode_full_mlx(rt.trellis, rt.k, rt.cb)
            return rt.finish_y((rt.prepare_xh(x2d) @ w).astype(mx.float16))
        row = {"fused_ms": timeit(f_fused) * 1e3, "fullW_ms": timeit(f_full) * 1e3}
        yf = f_fused(); yw = f_full(); mx.eval(yf, yw)
        a = np.array(yf.astype(mx.float32)); b = np.array(yw.astype(mx.float32))
        row["maxabs"] = float(np.abs(a - b).max())
        row["cos"] = float(np.dot(a.ravel(), b.ravel()) /
                           (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-30))
        row["fused_over_fullW"] = row["fused_ms"] / row["fullW_ms"]
        ent["rows"][R] = row
        log(f"{nm:26s} R={R:3d} fused={row['fused_ms']:7.3f} fullW={row['fullW_ms']:7.3f} "
            f"ratio={row['fused_over_fullW']:5.2f}x cos={row['cos']:.6f} max|d|={row['maxabs']:.4f}")
    # crossover: first R where fullW wins by >3%
    xo = None
    for R in ROWS:
        if ent["rows"][R]["fused_over_fullW"] > 1.03:
            xo = R
            break
    ent["crossover_R"] = xo
    res["shapes"][nm] = ent

# cost of the current threshold 64 vs a 16 threshold over the 17..64 band
tot = {}
for nm, ent in res["shapes"].items():
    band = [r for r in ROWS if 17 <= r <= 64]
    cur = sum(ent["rows"][r]["fused_ms"] for r in band)
    new = sum(ent["rows"][r]["fullW_ms"] for r in band)
    tot[nm] = {"rows_17_64_fused_ms": cur, "rows_17_64_fullW_ms": new,
               "ratio": cur / new, "crossover_R": ent["crossover_R"]}
res["band_cost"] = tot
log("--- cost of keeping the 64-row threshold on the 17..64 band ---")
for nm, v in tot.items():
    log(f"{nm:26s} fused={v['rows_17_64_fused_ms']:7.3f} fullW={v['rows_17_64_fullW_ms']:7.3f} "
        f"ratio={v['ratio']:5.2f}x crossover_R={v['crossover_R']}")
json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
