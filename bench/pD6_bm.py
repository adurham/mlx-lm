#!/usr/bin/env python3
"""pD6 -- isolate WHAT the tiny-segment cost is: bm sweep + v19d guard.

The v19c grid is (out_col_blocks, nb_blocks) for ONE launch. With 384 experts
holding ~8 rows each and bm=64, each out-col-block re-stages 64 rows of x of
which only ~8 are live -> 8x the x traffic. Sweep bm to prove the model and
time the row-fragment guard (v19d) at each bm. Parity v19c==v19d must be exact.

Env: PD_BMS (comma list, default "64,32,16,8"), PD_REPS, PD_LAYER, PD_ROWS, PD_KK.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
sys.path.insert(0, HOME + "/dsv41-ws/D/bench")
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
import v19d

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
ROWS = int(os.environ.get("PD_ROWS", "512"))
KK = int(os.environ.get("PD_KK", "6"))
REPS = int(os.environ.get("PD_REPS", "20"))
BMS = [int(x) for x in os.environ.get("PD_BMS", "64,32,16,8").split(",")]
OUT = os.environ.get("PD_JSON", HOME + "/dsv41-ws/D/pD6_bm.json")


def log(*a):
    print("[pD6]", *a, flush=True)


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
N = ROWS * KK
res = {"E": E, "D": D, "H": H, "rows": ROWS, "kk": KK, "N": N, "bm": {}}
K, CB = sg._k, sg._cb
GU_T = sg._gu_trellis
DN_T = sg._dn_trellis
GT, DT = sg._gu_tiles, sg._dn_tiles

flat = idx.reshape(-1)
order = mx.argsort(flat)
sidx = flat[order]
tok = (mx.arange(N, dtype=mx.uint32) // KK)[order]
x_base = x.reshape(ROWS, D)


def run(v, xin, trellis, tn_base, tiles_per_e, out_e, tab, nbr, n_rows=N):
    return v19d.seg_mm(xin, trellis, K, CB, tab, nbr, n_rows=n_rows,
                       tn_base=tn_base, tiles_per_e=tiles_per_e, out_e=out_e,
                       version=v)


for bm in BMS:
    tok_x = mx.concatenate([tok, mx.zeros((bm,), dtype=tok.dtype)])
    sidx_x = mx.concatenate([sidx, mx.zeros((bm,), dtype=sidx.dtype)])
    xp = x_base[tok_x]
    xg = MOE._rows_prep()(xp, sg._gu_suh[sidx_x, 0])
    xu = MOE._rows_prep()(xp, sg._gu_suh[sidx_x, 1])
    nb_max = (N + bm - 1) // bm + E + 1
    tab, nbr = MOE._seg_table_fn(E, nb_max, bm)(sidx)
    mx.eval(xg, xu, tab, nbr)
    live = int(np.array(nbr)[0])
    ent = {"bm": bm, "nb_max": nb_max, "live_blocks": live,
           "block_rows": live * bm, "waste": live * bm / N}
    for v in ("v19c", "v19d"):
        ent[f"gu_{v}_ms"] = timeit(lambda: (
            run(v, xg, GU_T, 0, GT, H, tab, nbr),
            run(v, xu, GU_T, E * GT, GT, H, tab, nbr))[0]) * 1e3
    res["bm"][bm] = ent
    log(f"bm={bm:3d} live={live:4d} blk_rows={live*bm:6d} waste={live*bm/N:5.2f}x "
        f"gu v19c={ent['gu_v19c_ms']:7.2f} v19d={ent['gu_v19d_ms']:7.2f} ms "
        f"gain={ent['gu_v19c_ms']/ent['gu_v19d_ms']:.2f}x")

# parity at each bm: v19d must reproduce v19c bit-exactly
for bm in BMS:
    tok_x = mx.concatenate([tok, mx.zeros((bm,), dtype=tok.dtype)])
    sidx_x = mx.concatenate([sidx, mx.zeros((bm,), dtype=sidx.dtype)])
    xp = x_base[tok_x]
    xg = MOE._rows_prep()(xp, sg._gu_suh[sidx_x, 0])
    nb_max = (N + bm - 1) // bm + E + 1
    tab, nbr = MOE._seg_table_fn(E, nb_max, bm)(sidx)
    mx.eval(xg, tab, nbr)
    ya = run("v19c", xg, GU_T, 0, GT, H, tab, nbr)
    yb = run("v19d", xg, GU_T, 0, GT, H, tab, nbr)
    mx.eval(ya, yb)
    eq = bool(mx.array_equal(ya, yb))
    res["bm"][bm]["parity_v19c_v19d"] = eq
    if not eq:
        a = np.array(ya.astype(mx.float32)); b = np.array(yb.astype(mx.float32))
        res["bm"][bm]["maxabs"] = float(np.abs(a - b).max())
    log(f"bm={bm:3d} parity v19c==v19d: {eq}")

json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
