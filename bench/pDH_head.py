#!/usr/bin/env python3
"""pDH -- the lm_head at R=512: fullW (1.3 GB transient) vs striped.

DECODE_FULL_MAX_BYTES=1536 MB is the only dense threshold DSv4.1 actually
brushes: head is 5120x129280 -> 1323 MB fp16 W, just under the cap, so R=512
materializes it per chunk. Measure both paths at the real prefill chunk size,
plus a few stripe widths, to decide whether the cap should move.

Env: PD_REPS, PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import EXL3Linear
from mlx_lm.models.exl3.layer_state import stripe_weight_mlx
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
REPS = int(os.environ.get("PD_REPS", "10"))
ROWS = [int(x) for x in os.environ.get("PD_ROWS_LIST", "128,512").split(",")]
OUT = os.environ.get("PD_JSON")


def log(*a):
    print("[pDH]", *a, flush=True)


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
lay = load_dense_layer(ck, "head")
lin = EXL3Linear(lay)
rt = lin._rt
log(f"head {lin.in_features}x{lin.out_features} W={lin.in_features*lin.out_features*2/1e6:.0f}MB "
    f"trellis={lay.trellis.shape} k={lay.k}")

res = {"in": lin.in_features, "out": lin.out_features,
       "w_MB": lin.in_features * lin.out_features * 2 / 1e6, "rows": {}}
for R in ROWS:
    x2d = mx.array(np.random.RandomState(R).randn(R, lin.in_features).astype(np.float16))
    mx.eval(x2d)
    ent = {}
    # warm the stripe cache first so the timing is the steady state (it is cached
    # in the real stripe path too)
    for cols in (512, 1024, 2048, 4096):
        for n0 in range(0, lin.out_features, cols):
            stripe_weight_mlx(lay, n0, min(cols, lin.out_features - n0), use_cache=True)
    w_full = G.decode_full_mlx(rt.trellis, rt.k, rt.cb)
    mx.eval(w_full)
    def f_full():
        return rt.finish_y((rt.prepare_xh(x2d) @ w_full).astype(mx.float16))
    mx.reset_peak_memory()
    f_full(); mx.eval(f_full())
    ent["fullW_peak_MB"] = mx.get_peak_memory() / 1e6
    ent["fullW_ms"] = timeit(f_full) * 1e3
    for cols in (512, 1024, 2048, 4096):
        def f_st(cols=cols):
            xh32 = rt.prepare_xh(x2d).astype(mx.float32)
            cs = []
            for n0 in range(0, lin.out_features, cols):
                w = stripe_weight_mlx(lay, n0, min(cols, lin.out_features - n0),
                                      use_cache=True)
                cs.append((xh32 @ w.astype(mx.float32)).astype(mx.float16))
            return rt.finish_y(mx.concatenate(cs, axis=1))
        mx.reset_peak_memory()
        f_st(); mx.eval(f_st())
        ent[f"stripe{cols}_peak_MB"] = mx.get_peak_memory() / 1e6
        ent[f"stripe{cols}_ms"] = timeit(f_st) * 1e3
    # parity: stripe must equal fullW exactly (both fp16 rotations, fp32 gemm)
    ya = f_full(); mx.eval(ya)
    for cols in (512, 1024, 2048, 4096):
        def f_st2(cols=cols):
            xh32 = rt.prepare_xh(x2d).astype(mx.float32)
            cs = []
            for n0 in range(0, lin.out_features, cols):
                w = stripe_weight_mlx(lay, n0, min(cols, lin.out_features - n0),
                                      use_cache=True)
                cs.append((xh32 @ w.astype(mx.float32)).astype(mx.float16))
            return rt.finish_y(mx.concatenate(cs, axis=1))
        yb = f_st2(); mx.eval(yb)
        ent[f"parity_stripe{cols}"] = bool(mx.array_equal(ya, yb))
    res["rows"][R] = ent
    log(f"R={R:4d} fullW={ent['fullW_ms']:8.3f}ms pk={ent['fullW_peak_MB']:7.0f}MB | " +
        " ".join(f"s{c}={ent[f'stripe{c}_ms']:8.3f}ms/pk{ent[f'stripe{c}_peak_MB']:6.0f}MB" for c in (512, 1024, 2048, 4096)) +
        f" | parity={[ent[f'parity_stripe{c}'] for c in (512, 1024, 2048, 4096)]}")
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
