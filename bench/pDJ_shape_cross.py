#!/usr/bin/env python3
"""pDJ -- is FUSED_GEMM_ROW_LIMIT=16 safe for NON-DSv4.1 shapes in this library?

The old value (64) was measured on M5 Max 27B-scale layers; DSv4.1 measured 16.
The crossover depends on the projection GEOMETRY, not the weight values, so we
can synthesize any shape by slicing a real DSv4.1 trellis: same kernel path,
27B-like (in_tiles, out_tiles). Sweep the shape grid at R=17..64 and find where
fullW stops beating the fused GEMM.

Shapes probed (in x out), covering the 27B/35B projection families:
  5120x17408 (27B mlp up), 5120x5120, 5120x13312, 4096x14336, 2048x2048 (tiny)
Env: PD_REPS, PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import EXL3Linear
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer
from mlx_lm.models.exl3.ref.layer import EXL3Layer

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
REPS = int(os.environ.get("PD_REPS", "20"))
ROWS = [int(x) for x in os.environ.get("PD_ROWS_LIST", "16,17,24,32,40,48,64").split(",")]
OUT = os.environ.get("PD_JSON")


def log(*a):
    print("[pDJ]", *a, flush=True)


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
# a big source trellis to slice arbitrary geometries from (k=2, 48 packed)
from mlx_lm.models.exl3.loader import load_experts
sg = load_experts(ck, 20, rank=0, world=2)   # (320, 55296) gu and (72, 122880) dn
# the dn trellis is (72 in-tiles, 122880 out-tiles): 1152 in x 5120 out per
# expert at k=2. Slice a 320-in-tile source out of the gu trellis instead:
# gu is (320, 55296) = 5120 in x 1152 out per expert.
# gu layout is (320 in-tiles, 2*E*gu_tiles out-tiles, 32 packed): out-tiles
# group per expert, so out-tiles 0..71 are expert 0's gate. Take a prefix.
T = sg._gu_trellis
src_k = sg._k
log(f"source trellis {tuple(T.shape)} k={src_k}")

SHAPES = [(5120, 17408), (5120, 5120), (5120, 13312), (4096, 14336),
          (5120, 1280), (2048, 2048)]
res = {"source": {"trellis": list(T.shape), "k": src_k}, "shapes": {}}
for (IN, OUTD) in SHAPES:
    it, ot = IN // 16, OUTD // 16
    if it > T.shape[0] or ot > T.shape[1]:
        log(f"skip {IN}x{OUTD}: needs ({it},{ot}) > source {T.shape[:2]}")
        continue
    lay = EXL3Layer(key=f"synth{IN}x{OUTD}", in_features=IN, out_features=OUTD,
                    k=src_k, trellis=np.ascontiguousarray(T[:it, :ot]),
                    suh=np.ones(IN, np.float16), svh=np.ones(OUTD, np.float16), mul1=True)
    lin = EXL3Linear(lay)
    rt = lin._rt
    ent = {"in": IN, "out": OUTD, "w_MB": IN * OUTD * 2 / 1e6, "rows": {}}
    for R in ROWS:
        x = mx.array(np.random.RandomState(R).randn(R, IN).astype(np.float16))
        mx.eval(x)
        def f_fused():
            return rt.finish_y(G.inner_gemm_mlx(
                rt.prepare_xh(x), rt.trellis, rt.k, rt.cb).astype(mx.float16))
        def f_full():
            w = G.decode_full_mlx(rt.trellis, rt.k, rt.cb)
            return rt.finish_y((rt.prepare_xh(x) @ w).astype(mx.float16))
        ent["rows"][R] = {"fused_ms": timeit(f_fused) * 1e3,
                          "fullW_ms": timeit(f_full) * 1e3}
        ent["rows"][R]["ratio"] = ent["rows"][R]["fused_ms"] / ent["rows"][R]["fullW_ms"]
    xo = next((R for R in ROWS if ent["rows"][R]["ratio"] > 1.03), None)
    ent["crossover_R"] = xo
    res["shapes"][f"{IN}x{OUTD}"] = ent
    log(f"{IN:5d}x{OUTD:6d} W={ent['w_MB']:6.0f}MB crossover_R={xo}  " +
        " ".join(f"R{R}:{ent['rows'][R]['ratio']:.2f}" for R in ROWS))
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
