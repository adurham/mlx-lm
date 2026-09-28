#!/usr/bin/env python3
"""pD5 -- size the MoE prefill cost: decode vs mma, and prototype the
blk_len row-frag guard (skips mma whose results the store loop discards).

Runs three kernel variants over the same real geometry (layer 20, rank 0):
  v19c      : stock
  v19c-nodc : EXL3_MM_NODECODE (decode replaced by cw&1) -> mma+staging cost
  v19d      : row-frag guard (rmax from blk_len) + same guards on the bf loads
Bit-parity of v19d vs v19c is asserted (guards must not change live values).
"""
import json, os, sys, time

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
OUT = os.environ.get("PD_JSON", HOME + "/dsv41-ws/D/pD5_guard.json")


def log(*a):
    print("[pD5]", *a, flush=True)


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


# ---------------------------------------------------------------- kernel variants
def guarded_source(k, cb, tag):
    """v19d: skip token-row fragments (and their B-fragment loads) beyond blk_len.

    The store loop already discards rows >= blk_len; with 8 rows/expert and
    bm=64, sgr=1's rows are ALL dead on 384 of 433 blocks. Skipping them is a
    pure win (no live value changes)."""
    src = G._mm_seg_source(k, cb)
    assert "    uint rbase = sgr * 32u;\n" in src
    # rmax must be computed at the top of the kernel, before the mma section;
    # blk_len is already loaded there. Use a distinct name: the store loop
    # defines `rbase` itself.
    src = src.replace(
        "    threadgroup half wblk[32u * 72u];   // 2 in-tiles x 4 out-tiles",
        "    uint rlive = 0u;\n"
        "    if (blk_len > sgr * 32u) {\n"
        "        uint rrem = blk_len - sgr * 32u;\n"
        "        rlive = (rrem > 32u) ? 4u : ((rrem + 7u) >> 3u);\n"
        "    }\n"
        "    threadgroup half wblk[32u * 72u];   // 2 in-tiles x 4 out-tiles", 1)
    old = """        for (uint k2 = 0u; k2 < 4u; k2++) {
            simdgroup_half8x8 bf0;
            simdgroup_half8x8 bf1;
            simdgroup_half8x8 bf2;
            simdgroup_half8x8 bf3;
            const threadgroup half* wrow =
                &wblk[k2 * 8u * 72u + sgc * 32u];
            simdgroup_load(bf0, wrow, 72u);
            simdgroup_load(bf1, wrow + 8u, 72u);
            simdgroup_load(bf2, wrow + 16u, 72u);
            simdgroup_load(bf3, wrow + 24u, 72u);
            for (uint r = 0u; r < 4u; r++) {"""
    new = """        if (rlive == 0u) {
            continue;
        }
        for (uint k2 = 0u; k2 < 4u; k2++) {
            simdgroup_half8x8 bf0;
            simdgroup_half8x8 bf1;
            simdgroup_half8x8 bf2;
            simdgroup_half8x8 bf3;
            const threadgroup half* wrow =
                &wblk[k2 * 8u * 72u + sgc * 32u];
            simdgroup_load(bf0, wrow, 72u);
            simdgroup_load(bf1, wrow + 8u, 72u);
            simdgroup_load(bf2, wrow + 16u, 72u);
            simdgroup_load(bf3, wrow + 24u, 72u);
            for (uint r = 0u; r < rlive; r++) {"""
    assert old in src, "mma block pattern not found"
    return src.replace(old, new, 1)


def launch(variant, xh, trellis, k, cb, tab, nbr, *, n_rows, tn_base,
           tiles_per_e, out_e):
    in_tiles, src_tiles, _ = trellis.shape
    nb_max = int(tab.shape[1])
    key = (k, int(cb), variant)
    kernel = MOE._MOE_KERNELS.get(key)
    if kernel is None:
        if variant == "v19c":
            source = G._mm_seg_source(k, cb)
        elif variant == "v19c-nodc":
            source = G._mm_seg_source(k, cb)     # built under EXL3_MM_NODECODE
        else:
            source = guarded_source(k, cb, variant)
        kernel = mx.fast.metal_kernel(
            name=f"exl3_mm_seg_k{k}_cb{int(cb)}_{variant}",
            input_names=["xh", "trellis", "inv_perm", "blk_tab", "nbr", "dims"],
            output_names=["out"],
            source=source,
            header="#include <metal_simdgroup_matrix>\n#include <metal_stdlib>\nusing namespace metal;\n",
        )
        MOE._MOE_KERNELS[key] = kernel
    dims = mx.array([in_tiles, tiles_per_e, src_tiles, tn_base, out_e, nb_max],
                    dtype=mx.uint32)
    return kernel(
        inputs=[xh.reshape(-1), trellis.reshape(-1).view(mx.uint32),
                G._inv_perm_u32(), tab.reshape(-1), nbr, dims],
        template=[("T", mx.float16)],
        grid=((out_e // 64) * G._GEM_THREADS, nb_max, 1),
        threadgroup=(G._GEM_THREADS, 1, 1),
        output_shapes=[(n_rows, out_e)],
        output_dtypes=[mx.float16],
    )[0]


ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
rng = np.random.RandomState(7)
x = mx.array(rng.randn(1, ROWS, D).astype(np.float16))
idx = mx.array(np.stack([rng.permutation(E)[:KK] for _ in range(ROWS)])[None].astype(np.uint32))
mx.eval(x, idx)

N = ROWS * KK
B, S, kk = 1, ROWS, KK
flat = idx.reshape(-1)
order = mx.argsort(flat)
inv = mx.argsort(order)
sidx = flat[order]
tok = (mx.arange(N, dtype=mx.uint32) // kk)[order]
tok_x = mx.concatenate([tok, mx.zeros((64,), dtype=tok.dtype)])
sidx_x = mx.concatenate([sidx, mx.zeros((64,), dtype=sidx.dtype)])
x_pairs = x.reshape(ROWS, D)[tok_x]
xg = MOE._rows_prep()(x_pairs, sg._gu_suh[sidx_x, 0])
xu = MOE._rows_prep()(x_pairs, sg._gu_suh[sidx_x, 1])
nb_max = N // 64 + E + 1
tab, nbr = MOE._seg_table_fn(E, nb_max, 64)(sidx)
mx.eval(xg, xu, tab, nbr)

# down-projection input needs h; use the real gu output
y_gu = MOE.inner_mm_seg_mlx(xg, sg._gu_trellis, sg._k, sg._cb, tab, nbr,
                            n_rows=N, tn_base=0, tiles_per_e=sg._gu_tiles, out_e=H)
h = mx.zeros_like(y_gu)  # content irrelevant for timing
h_pad = mx.concatenate([h, mx.zeros((64, H), dtype=h.dtype)])
xd = MOE._rows_prep()(h_pad, sg._dn_suh[sidx_x])
mx.eval(xd)

res = {"E": E, "D": D, "H": H, "rows": ROWS, "kk": KK, "N": N,
       "live_blocks": int(np.array(nbr)[0]), "nb_max": nb_max}
log(f"E={E} D={D} H={H} N={N} live_blocks={res['live_blocks']} nb_max={nb_max}")

GU = dict(trellis=sg._gu_trellis, k=sg._k, cb=sg._cb, tab=tab, nbr=nbr,
          n_rows=N, tn_base=0, tiles_per_e=sg._gu_tiles, out_e=H)
DN = dict(trellis=sg._dn_trellis, k=sg._k, cb=sg._cb, tab=tab, nbr=nbr,
          n_rows=N, tn_base=0, tiles_per_e=sg._dn_tiles, out_e=D)

for variant in ("v19c", "v19d"):
    for nm, spec, xin in (("gu", GU, xg), ("dn", DN, xd)):
        y = launch(variant, xin, **spec)
        mx.eval(y)
        res[f"{nm}_{variant}_ms"] = timeit(lambda: launch(variant, xin, **spec)) * 1e3
        mx.reset_peak_memory()
        launch(variant, xin, **spec)
        mx.eval(launch(variant, xin, **spec))
        res[f"{nm}_{variant}_peak_MB"] = mx.get_peak_memory() / 1e6
    log(f"{variant}: gu={res[f'gu_{variant}_ms']:.2f}ms dn={res[f'dn_{variant}_ms']:.2f}ms "
        f"peaks gu={res[f'gu_{variant}_peak_MB']:.0f} dn={res[f'dn_{variant}_peak_MB']:.0f}MB")

# parity between stock and guarded
for nm, spec, xin in (("gu", GU, xg), ("dn", DN, xd)):
    ya = launch("v19c", xin, **spec); mx.eval(ya)
    yb = launch("v19d", xin, **spec); mx.eval(yb)
    eq = bool(mx.array_equal(ya, yb))
    res[f"parity_{nm}_v19c_v19d"] = eq
    if not eq:
        a = np.array(ya.astype(mx.float32)); b = np.array(yb.astype(mx.float32))
        res[f"maxabs_{nm}"] = float(np.abs(a - b).max())
    log(f"parity {nm} v19c==v19d: {eq}")

res["speedup_gu"] = res["gu_v19c_ms"] / res["gu_v19d_ms"]
res["speedup_dn"] = res["dn_v19c_ms"] / res["dn_v19d_ms"]
res["speedup_total"] = ((res["gu_v19c_ms"] + res["dn_v19c_ms"]) /
                        (res["gu_v19d_ms"] + res["dn_v19d_ms"]))
log(f"v19d speedup: gu={res['speedup_gu']:.2f}x dn={res['speedup_dn']:.2f}x "
    f"total={res['speedup_total']:.2f}x")
json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
