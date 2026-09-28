#!/usr/bin/env python3
"""pD7 -- price the v19c/v19d decode: NODECODE and LUT ablations.

Per launch the segmented kernel decodes exactly the weight set of one
projection (each (expert, weight) once), so decode COUNT is already minimal;
what is unknown is the decode THROUGHPUT. Variants:

  v19d       : guarded, stock decode_3inst (mul1 swar)
  v19d_nodc  : guarded, decode replaced by `float(cw & 1u)`  -> prices mma+stg
  v19d_lut   : guarded, 16-bit codeword indexed into the 128KB fp16 decode LUT

Env: PD_LAYER/PD_ROWS/PD_KK/PD_REPS, PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
sys.path.insert(0, HOME + "/dsv41-ws/D/bench")
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
import v19d

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
ROWS = int(os.environ.get("PD_ROWS", "512"))
KK = int(os.environ.get("PD_KK", "6"))
REPS = int(os.environ.get("PD_REPS", "20"))
OUT = os.environ.get("PD_JSON", HOME + "/dsv41-ws/D/pD7_decode.json")


def log(*a):
    print("[pD7]", *a, flush=True)


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


# ------------------------------------------------------------------ variants
def src_nodecode(k, cb):
    return v19d.guarded_source(k, cb).replace(
        "            float dq_val = float(dq_h * dq_k_inv + dq_k_bias);",
        "            float dq_val = float(cw & 1u);", 1)


def src_lut(k, cb):
    s = v19d.guarded_source(k, cb)
    # replace the whole inline decode with one fp16 table read (the codeword
    # is the full 16-bit sliding window, exactly what the LUT is indexed by)
    out = []
    skip = 0
    for line in s.splitlines():
        st = line.strip()
        if st.startswith("uint dq_cw") or st.startswith("uint dq_t") or \
           st.startswith("uint dq_sum") or st.startswith("half dq_h") or \
           st.startswith("half dq_k_inv") or st.startswith("half dq_k_bias") or \
           st.startswith("float dq_val = float(dq_h"):
            continue
        out.append(line)
    s = "\n".join(out).replace(
        "                    uint cw = uint(merged >> sh_[j]) & 0xFFFFu;",
        "                    uint cw = uint(merged >> sh_[j]) & 0xFFFFu;\n"
        "                    float dq_val = float(lut[cw]);", 1)
    assert "lut[cw]" in s
    return s


KERNEL_CACHE: dict = {}


def kern(k, cb, variant):
    key = (k, int(cb), variant)
    kk = KERNEL_CACHE.get(key)
    if kk is not None:
        return kk
    if variant == "v19d":
        src, names = v19d.guarded_source(k, cb), ["xh", "trellis", "inv_perm", "blk_tab", "nbr", "dims"]
    elif variant == "v19d_nodc":
        src, names = src_nodecode(k, cb), ["xh", "trellis", "inv_perm", "blk_tab", "nbr", "dims"]
    elif variant == "v19d_lut":
        src = src_lut(k, cb)
        names = ["xh", "trellis", "inv_perm", "blk_tab", "nbr", "lut", "dims"]
    else:
        raise ValueError(variant)
    kk = mx.fast.metal_kernel(
        name=f"exl3_mm_seg_k{k}_cb{int(cb)}_{variant}",
        input_names=names, output_names=["out"], source=src,
        header="#include <metal_simdgroup_matrix>\n#include <metal_stdlib>\nusing namespace metal;\n")
    KERNEL_CACHE[key] = kk
    return kk


def run(variant, xh, trellis, k, cb, tab, nbr, *, n_rows, tn_base, tiles_per_e,
        out_e, lut=None):
    in_tiles, src_tiles, _ = trellis.shape
    nb_max = int(tab.shape[1])
    dims = mx.array([in_tiles, tiles_per_e, src_tiles, tn_base, out_e, nb_max],
                    dtype=mx.uint32)
    ins = [xh.reshape(-1), trellis.reshape(-1).view(mx.uint32),
           G._inv_perm_u32(), tab.reshape(-1), nbr]
    if variant == "v19d_lut":
        ins.append(lut)
    ins.append(dims)
    return kern(k, cb, variant)(
        inputs=ins, template=[("T", mx.float16)],
        grid=((out_e // 64) * G._GEM_THREADS, nb_max, 1),
        threadgroup=(G._GEM_THREADS, 1, 1),
        output_shapes=[(n_rows, out_e)], output_dtypes=[mx.float16])[0]


ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
K, CB = sg._k, sg._cb
rng = np.random.RandomState(7)
x = mx.array(rng.randn(1, ROWS, D).astype(np.float16))
idx = mx.array(np.stack([rng.permutation(E)[:KK] for _ in range(ROWS)])[None].astype(np.uint32))
mx.eval(x, idx)
N = ROWS * KK
lut = G._decode_lut(CB)
res = {"E": E, "D": D, "H": H, "N": N, "rows": ROWS, "kk": KK}

flat = idx.reshape(-1)
order = mx.argsort(flat)
sidx = flat[order]
tok = (mx.arange(N, dtype=mx.uint32) // KK)[order]
x_base = x.reshape(ROWS, D)
bm = 64
tok_x = mx.concatenate([tok, mx.zeros((bm,), dtype=tok.dtype)])
sidx_x = mx.concatenate([sidx, mx.zeros((bm,), dtype=sidx.dtype)])
xp = x_base[tok_x]
xg = MOE._rows_prep()(xp, sg._gu_suh[sidx_x, 0])
xu = MOE._rows_prep()(xp, sg._gu_suh[sidx_x, 1])
nb_max = (N + bm - 1) // bm + E + 1
tab, nbr = MOE._seg_table_fn(E, nb_max, bm)(sidx)
mx.eval(xg, xu, tab, nbr)
log(f"E={E} D={D} H={H} N={N} bm={bm} live={int(np.array(nbr)[0])}")

GU_T = sg._gu_trellis
GT = sg._gu_tiles
VAR = ("v19d", "v19d_nodc", "v19d_lut")

for v in VAR:
    # warm the lut arg
    if v == "v19d_lut":
        mx.eval(lut)
    try:
        t = timeit(lambda: (
            run(v, xg, GU_T, K, CB, tab, nbr, n_rows=N, tn_base=0,
                tiles_per_e=GT, out_e=H, lut=lut),
            run(v, xu, GU_T, K, CB, tab, nbr, n_rows=N, tn_base=E * GT,
                tiles_per_e=GT, out_e=H, lut=lut))[0]) * 1e3
        mx.reset_peak_memory()
        g = run(v, xg, GU_T, K, CB, tab, nbr, n_rows=N, tn_base=0,
                tiles_per_e=GT, out_e=H, lut=lut)
        mx.eval(g)
        res[f"{v}_gu_ms"] = t
        res[f"{v}_peak_MB"] = mx.get_peak_memory() / 1e6
        log(f"{v:10s} gu(2 launches)={t:7.2f} ms  peak={mx.get_peak_memory()/1e6:6.0f}MB")
    except Exception as e:
        res[f"{v}_gu_ms"] = None
        log(f"{v:10s} FAILED {type(e).__name__}: {e}")

# parity: nodc/lut are NOT expected to match (they change math) except lut must
ya = run("v19d", xg, GU_T, K, CB, tab, nbr, n_rows=N, tn_base=0, tiles_per_e=GT, out_e=H)
yl = run("v19d_lut", xg, GU_T, K, CB, tab, nbr, n_rows=N, tn_base=0, tiles_per_e=GT,
         out_e=H, lut=lut)
mx.eval(ya, yl)
res["parity_v19d_lut"] = bool(mx.array_equal(ya, yl))
log(f"parity v19d == v19d_lut: {res['parity_v19d_lut']}")

if res.get("v19d_gu_ms") and res.get("v19d_nodc_gu_ms"):
    res["decode_share"] = 1 - res["v19d_nodc_gu_ms"] / res["v19d_gu_ms"]
    log(f"decode share of guarded time: {res['decode_share']*100:.0f}%")
json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
