#!/usr/bin/env python3
"""pD1 -- EXL3Linear prefill-path benchmark at DSv4.1 rank-0 projection shapes.

For every dense projection of a DSv4.1 layer (rank 0 of world 2, the real
prefill geometry) time the three candidate paths at R rows:

  A full-W   : decode_full_mlx (transient fp16 W) + native matmul   [current default]
  B striped  : stripe_weight_mlx per 512-col chunk (cached) + fp32 matmul
  C fused    : inner_gemm_mlx (trellis-direct, re-reads trellis per 8 rows)

and the fused-group launch (Exl3FusedGroup.run_stacked) where members share
an input.  Reports ms/projection, ms/layer-total, peak transient bytes, and a
bit-exact parity check of B and C against A.

Env: PD_PKG (package root, default ~/dsv41-ws/D), PD_ROWS (default 512),
     PD_LAYER (default 20), PD_REPS (default 30), PD_JSON (results file).
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
from mlx_lm.models.exl3.stripe import DEFAULT_STRIPE_COLS

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
ROWS = int(os.environ.get("PD_ROWS", "512"))
REPS = int(os.environ.get("PD_REPS", "30"))
OUT = os.environ.get("PD_JSON", HOME + "/dsv41-ws/D/pD1_dense.json")


def log(*a):
    print("[pD1]", *a, flush=True)


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


# ---- path implementations (mirroring EXL3Linear.__call__) -------------------
def path_fullw(lin, x2d):
    rt = lin._rt
    w = G.decode_full_mlx(rt.trellis, rt.k, rt.cb)
    return rt.finish_y((rt.prepare_xh(x2d) @ w).astype(mx.float16))


def path_stripe(lin, x2d, cols=DEFAULT_STRIPE_COLS):
    rt = lin._rt
    xh32 = rt.prepare_xh(x2d).astype(mx.float32)
    chunks = []
    for n0 in range(0, lin.out_features, cols):
        n = min(cols, lin.out_features - n0)
        w = stripe_weight_mlx(lin._exl3, n0, n, use_cache=True)
        chunks.append((xh32 @ w.astype(mx.float32)).astype(mx.float16))
    return rt.finish_y(mx.concatenate(chunks, axis=1))


def path_fused(lin, x2d):
    rt = lin._rt
    return rt.finish_y(G.inner_gemm_mlx(rt.prepare_xh(x2d), rt.trellis, rt.k, rt.cb).astype(mx.float16))


ck = Exl3Checkpoint(CK)
projs = {}
for g in sorted({k[: k.rfind(".")] for k in ck.index
                 if k.startswith(f"layers.{LAYER}.") and ".ffn.experts." not in k
                 and k.endswith((".trellis",))}):
    projs[g] = None
# rank-0 slices exactly as build_block does (world=2)
def sliced(name, axis):
    lay = load_dense_layer(ck, name)
    t = lay.trellis
    n = t.shape[1] if axis == "out" else t.shape[0]
    a, b = 0, n // 2
    if axis == "out":
        return type(lay)(key=name, in_features=lay.in_features,
                         out_features=(b - a) * 16, k=lay.k,
                         trellis=np.ascontiguousarray(t[:, a:b]),
                         suh=lay.suh, svh=lay.svh[a * 16:b * 16], mul1=lay.mul1)
    return type(lay)(key=name, in_features=(b - a) * 16,
                     out_features=lay.out_features, k=lay.k,
                     trellis=np.ascontiguousarray(t[a:b]),
                     suh=lay.suh[a * 16:b * 16], svh=lay.svh, mul1=lay.mul1)


SHAPES = []
for nm, axis in (("wq_a", None), ("wkv", None), ("wq_b", "out"), ("wo_b", "in"),
                 ("wo_a.slice.0", None),
                 ("shared_experts.w1", "out"), ("shared_experts.w2", "in"),
                 ("shared_experts.w3", "out")):
    full = f"layers.{LAYER}.attn.{nm}" if nm.startswith(("wq", "wkv", "wo")) \
        else f"layers.{LAYER}.ffn.{nm}"
    if not ck.has(full + ".trellis"):
        log("skip missing", full)
        continue
    lay = sliced(full, axis) if axis else load_dense_layer(ck, full)
    SHAPES.append((nm, EXL3Linear(lay)))

log(f"layer {LAYER} rows={ROWS} reps={REPS} shapes=" +
    ", ".join(f"{n}({l.in_features}x{l.out_features})" for n, l in SHAPES))

x = mx.array(np.random.RandomState(0).randn(ROWS, SHAPES[0][1].in_features).astype(np.float16))
res = {"layer": LAYER, "rows": ROWS, "shapes": {}}
mx.reset_peak_memory()
base_peak = mx.get_peak_memory()

for name, lin in SHAPES:
    x2d = mx.array(np.random.RandomState(1).randn(ROWS, lin.in_features).astype(np.float16))
    x2d = mx.array(x2d)
    mx.eval(x2d)
    ent = {"in": lin.in_features, "out": lin.out_features,
           "w_bytes": lin.in_features * lin.out_features * 2}
    # parity: B and C vs A, bit-exact
    mx.reset_peak_memory()
    ya = path_fullw(lin, x2d)
    mx.eval(ya)
    ent["peak_fullw_MB"] = mx.get_peak_memory() / 1e6
    mx.reset_peak_memory()
    yb = path_stripe(lin, x2d)
    mx.eval(yb)
    ent["peak_stripe_MB"] = mx.get_peak_memory() / 1e6
    ent["stripe_cache_MB"] = lin.in_features * lin.out_features * 2 / 1e6
    mx.reset_peak_memory()
    yc = path_fused(lin, x2d)
    mx.eval(yc)
    ent["peak_fused_MB"] = mx.get_peak_memory() / 1e6
    ent["bit_parity_A_B"] = bool(mx.array_equal(ya, yb))
    ent["bit_parity_A_C"] = bool(mx.array_equal(ya, yc))
    if not ent["bit_parity_A_B"]:
        d = np.abs(np.array(ya.astype(mx.float32)) - np.array(yb.astype(mx.float32)))
        ent["maxabs_A_B"] = float(d.max())
    if not ent["bit_parity_A_C"]:
        d = np.abs(np.array(ya.astype(mx.float32)) - np.array(yc.astype(mx.float32)))
        ent["maxabs_A_C"] = float(d.max())
        ent["cos_A_C"] = float(np.dot(np.array(ya).ravel().astype(np.float64),
                                      np.array(yc).ravel().astype(np.float64)) /
                               (np.linalg.norm(np.array(ya).ravel().astype(np.float64)) *
                                np.linalg.norm(np.array(yc).ravel().astype(np.float64))))
    ent["ms_fullw"] = timeit(lambda: path_fullw(lin, x2d)) * 1e3
    ent["ms_stripe"] = timeit(lambda: path_stripe(lin, x2d)) * 1e3
    ent["ms_fused"] = timeit(lambda: path_fused(lin, x2d)) * 1e3
    res["shapes"][name] = ent
    log(f"{name:22s} {lin.in_features:5d}x{lin.out_features:6d} "
        f"fullW={ent['ms_fullw']:7.3f} stripe={ent['ms_stripe']:7.3f} "
        f"fused={ent['ms_fused']:7.3f} ms  peak(fullW)={ent['peak_fullw_MB']:7.1f}MB "
        f"parity A=B:{ent['bit_parity_A_B']} A=C:{ent['bit_parity_A_C']}")

tot = {k: sum(res["shapes"][n][k] for n in res["shapes"]) for k in ("ms_fullw", "ms_stripe", "ms_fused")}
res["totals_ms"] = tot
log("TOTAL per layer (8 rank-0 dense projs): " +
    " ".join(f"{k}={v:.2f}" for k, v in tot.items()))
json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
