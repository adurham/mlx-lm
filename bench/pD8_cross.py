#!/usr/bin/env python3
"""pD8 -- dense EXL3Linear path crossovers at DSv4.1 rank-0 shapes.

For each real projection, sweep R in {8,16,32,64,96,128,192,256,384,512,768}
and time:
  fullW   decode_full_mlx + native matmul (transient fp16 W)
  striped stripe_weight_mlx 512-col chunks + fp32 matmul
  fused   inner_gemm_mlx (trellis-direct)
and report the crossover rows for fullW vs striped vs fused, plus the transient
peak. This is what decides FUSED_GEMM_ROW_LIMIT / DECODE_FULL_MAX_BYTES for
DSv4.1's real shapes (5120x512, 5120x1280, 1280x16384, 4096x5120, 5120x1152...).

Env: PD_LAYER, PD_REPS, PD_ROWS_LIST, PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import EXL3Linear, exl3_moe as MOE
from mlx_lm.models.exl3.layer_state import stripe_weight_mlx
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
REPS = int(os.environ.get("PD_REPS", "15"))
ROWS_LIST = [int(x) for x in os.environ.get(
    "PD_ROWS_LIST", "8,16,32,64,96,128,192,256,384,512,768").split(",")]
OUT = os.environ.get("PD_JSON", HOME + "/dsv41-ws/D/pD8_cross.json")


def log(*a):
    print("[pD8]", *a, flush=True)


def timeit(fn, reps=REPS, warm=2):
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
# rank-0 slices, exactly as build_block builds them (world=2)
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
    if not ck.has(full + ".trellis"):
        log("skip", full)
        continue
    SPECS.append((nm, EXL3Linear(sliced(full, axis) if axis else load_dense_layer(ck, full))))

res = {"layer": LAYER, "shapes": {}}
for nm, lin in SPECS:
    ent = {"in": lin.in_features, "out": lin.out_features, "rows": {}}
    rt = lin._rt
    for R in ROWS_LIST:
        x2d = mx.array(np.random.RandomState(R).randn(R, lin.in_features).astype(np.float16))
        mx.eval(x2d)
        row = {}
        def f_fullw():
            w = G.decode_full_mlx(rt.trellis, rt.k, rt.cb)
            return rt.finish_y((rt.prepare_xh(x2d) @ w).astype(mx.float16))
        def f_stripe():
            xh32 = rt.prepare_xh(x2d).astype(mx.float32)
            cs = []
            for n0 in range(0, lin.out_features, 512):
                n = min(512, lin.out_features - n0)
                w = stripe_weight_mlx(lin._exl3, n0, n, use_cache=True)
                cs.append((xh32 @ w.astype(mx.float32)).astype(mx.float16))
            return rt.finish_y(mx.concatenate(cs, axis=1))
        def f_fused():
            return rt.finish_y(G.inner_gemm_mlx(rt.prepare_xh(x2d), rt.trellis, rt.k, rt.cb).astype(mx.float16))
        for label, fn in (("fullW", f_fullw), ("stripe", f_stripe), ("fused", f_fused)):
            try:
                if label == "stripe" and R in ROWS_LIST[:1]:
                    pass
                mx.reset_peak_memory()
                fn(); mx.eval(fn())
                row[f"{label}_peak_MB"] = mx.get_peak_memory() / 1e6
                row[f"{label}_ms"] = timeit(fn) * 1e3
            except Exception as e:
                row[f"{label}_ms"] = None
                row[f"{label}_err"] = f"{type(e).__name__}"
        # parity of stripe+fused vs fullW at this R
        ya = f_fullw(); mx.eval(ya)
        yb = f_stripe(); mx.eval(yb)
        yc = f_fused(); mx.eval(yc)
        row["p_stripe"] = bool(mx.array_equal(ya, yb))
        row["p_fused"] = bool(mx.array_equal(ya, yc))
        ent["rows"][R] = row
        log(f"{nm:28s} {lin.in_features:5d}x{lin.out_features:6d} R={R:4d} "
            f"fullW={row['fullW_ms']:7.3f} stripe={row['stripe_ms']:7.3f} "
            f"fused={row['fused_ms']:7.3f}  pk(fW)={row.get('fullW_peak_MB',0):6.0f} "
            f"pk(st)={row.get('stripe_peak_MB',0):6.0f}MB "
            f"p(st/fu)={row['p_stripe']}/{row['p_fused']}")
    res["shapes"][nm] = ent
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
