#!/usr/bin/env python3
"""pD9 -- MoE prefill kernel variants at DSv4.1 geometry (R=512, top-6, 384E).

v19c  : stock  (2 in-tiles decoded per barrier pair)
v19d  : + row-fragment guard (pure work elimination; bit-exact vs v19c)
v19e  : v19d + all 4 A-fragment loads issued before any mma
v19f  : v19d + 4 in-tiles per stage (16 tiles per barrier pair, 2x fewer
        barriers, 4x A-loads amortized per stage)
v19g  : v19f + hoisted A loads

Parity: every variant must equal v19c bit-for-bit on the live output rows.
Timing: gu (gate) + up in one measurement (what _prefill issues), then dn.

Env: PD_LAYER, PD_ROWS, PD_KK, PD_REPS, PD_VARIANTS, PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
sys.path.insert(0, HOME + "/dsv41-ws/D/bench")
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
import v19f

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
ROWS = int(os.environ.get("PD_ROWS", "512"))
KK = int(os.environ.get("PD_KK", "6"))
REPS = int(os.environ.get("PD_REPS", "20"))
VARS = os.environ.get("PD_VARIANTS", "v19c,v19d,v19e,v19f,v19g").split(",")
OUT = os.environ.get("PD_JSON", HOME + "/dsv41-ws/D/pD9_variants.json")


def log(*a):
    print("[pD9]", *a, flush=True)


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
K, CB = sg._k, sg._cb
GT, DT = sg._gu_tiles, sg._dn_tiles
rng = np.random.RandomState(7)
x = mx.array(rng.randn(1, ROWS, D).astype(np.float16))
idx = mx.array(np.stack([rng.permutation(E)[:KK] for _ in range(ROWS)])[None].astype(np.uint32))
mx.eval(x, idx)
N = ROWS * KK
res = {"E": E, "D": D, "H": H, "N": N, "rows": ROWS, "kk": KK, "variants": {}}

flat = idx.reshape(-1)
order = mx.argsort(flat)
sidx = flat[order]
tok = (mx.arange(N, dtype=mx.uint32) // KK)[order]
x_base = x.reshape(ROWS, D)
bm = 64
# pad rows: the kernel loads A device-direct; pad by one BLOCK (bm)
tok_x = mx.concatenate([tok, mx.zeros((bm,), dtype=tok.dtype)])
sidx_x = mx.concatenate([sidx, mx.zeros((bm,), dtype=sidx.dtype)])
xp = x_base[tok_x]
xg = MOE._rows_prep()(xp, sg._gu_suh[sidx_x, 0])
xu = MOE._rows_prep()(xp, sg._gu_suh[sidx_x, 1])
nb_max = (N + bm - 1) // bm + E + 1
tab, nbr = MOE._seg_table_fn(E, nb_max, bm)(sidx)
mx.eval(xg, xu, tab, nbr)
log(f"E={E} D={D} H={H} N={N} live_blocks={int(np.array(nbr)[0])} nb_max={nb_max}")
mx.reset_peak_memory()

ygu = None
for v in VARS:
    try:
        def gu_pair():
            a = v19f.seg_mm(xg, sg._gu_trellis, K, CB, tab, nbr, n_rows=N, tn_base=0,
                            tiles_per_e=GT, out_e=H, version=v)
            b = v19f.seg_mm(xu, sg._gu_trellis, K, CB, tab, nbr, n_rows=N,
                            tn_base=E * GT, tiles_per_e=GT, out_e=H, version=v)
            return a, b
        g, u = gu_pair()
        mx.eval(g, u)
        ent = {}
        ent["gu_ms"] = timeit(gu_pair) * 1e3
        # down projection on the real h (post-activation dims)
        hh = g if ygu is None else ygu
        ygu = g
        h_pad = mx.concatenate([hh, mx.zeros((bm, H), dtype=hh.dtype)])
        xd = MOE._rows_prep()(h_pad, sg._dn_suh[sidx_x])
        mx.eval(xd)
        def dn():
            return v19f.seg_mm(xd, sg._dn_trellis, K, CB, tab, nbr, n_rows=N,
                               tn_base=0, tiles_per_e=DT, out_e=D, version=v)
        ent["dn_ms"] = timeit(dn) * 1e3
        res["variants"][v] = ent
        log(f"{v:6s} gu(2 launches)={ent['gu_ms']:7.2f} ms  dn={ent['dn_ms']:7.2f} ms "
            f"total={ent['gu_ms']+ent['dn_ms']:7.2f} ms")
    except Exception as e:
        res["variants"][v] = {"error": f"{type(e).__name__}: {e}"}
        log(f"{v:6s} FAILED {type(e).__name__}: {e}")

# parity vs v19c on both projections
ref_g = None
ref_d = None
if "v19c" in res["variants"]:
    ref_g = v19f.seg_mm(xg, sg._gu_trellis, K, CB, tab, nbr, n_rows=N, tn_base=0,
                        tiles_per_e=GT, out_e=H, version="v19c")
    hh = ref_g
    h_pad = mx.concatenate([hh, mx.zeros((bm, H), dtype=hh.dtype)])
    xd = MOE._rows_prep()(h_pad, sg._dn_suh[sidx_x])
    ref_d = v19f.seg_mm(xd, sg._dn_trellis, K, CB, tab, nbr, n_rows=N, tn_base=0,
                        tiles_per_e=DT, out_e=D, version="v19c")
    mx.eval(ref_g, ref_d)
    for v in VARS:
        if "error" in res["variants"].get(v, {}):
            continue
        g = v19f.seg_mm(xg, sg._gu_trellis, K, CB, tab, nbr, n_rows=N, tn_base=0,
                        tiles_per_e=GT, out_e=H, version=v)
        d = v19f.seg_mm(xd, sg._dn_trellis, K, CB, tab, nbr, n_rows=N, tn_base=0,
                        tiles_per_e=DT, out_e=D, version=v)
        mx.eval(g, d)
        eqg = bool(mx.array_equal(ref_g, g))
        eqd = bool(mx.array_equal(ref_d, d))
        res["variants"][v]["parity_gu"] = eqg
        res["variants"][v]["parity_dn"] = eqd
        if not eqg or not eqd:
            for nm, a, b in (("gu", ref_g, g), ("dn", ref_d, d)):
                aa = np.array(a.astype(mx.float32)); bb = np.array(b.astype(mx.float32))
                res["variants"][v][f"maxabs_{nm}"] = float(np.abs(aa - bb).max())
        log(f"{v:6s} parity gu/dn vs v19c: {eqg}/{eqd}")

# speedups
base = res["variants"].get("v19c", {})
bg, bd = base.get("gu_ms"), base.get("dn_ms")
for v, e in res["variants"].items():
    if bg and bd and e.get("gu_ms"):
        e["speedup_gu"] = bg / e["gu_ms"]
        e["speedup_dn"] = bd / e["dn_ms"]
        e["speedup_total"] = (bg + bd) / (e["gu_ms"] + e["dn_ms"])
        log(f"{v:6s} speedup gu={e['speedup_gu']:.2f}x dn={e['speedup_dn']:.2f}x "
            f"total={e['speedup_total']:.2f}x")
res["peak_MB"] = mx.get_peak_memory() / 1e6
json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
