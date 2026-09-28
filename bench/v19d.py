#!/usr/bin/env python3
"""v19d prototype: row-fragment guard + configurable block size (bm).

The guard is a pure work-elimination transform: the v19c store loop already
discards rows >= blk_len, so skipping their A/B fragment loads and mmas cannot
change any live output value (asserted bit-exact against v19c by callers).

bm is the token-rows-per-block knob of the segmented grid. It trades:
  x traffic  : nb_blocks * (bm rows * in_features * 2B) per out-col-block
  mma waste  : ceil(blk_len/bm)*bm rows carried vs blk_len live
With 384 experts and ~8 rows each, bm=64 reads 8x more x than it needs.
"""
import mlx.core as mx

from mlx_lm.models.exl3 import gemv_metal as G

GUARD_ANCHOR = "    threadgroup half wblk[32u * 72u];   // 2 in-tiles x 4 out-tiles"
MMA_OLD = """        for (uint k2 = 0u; k2 < 4u; k2++) {
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
MMA_NEW = """        if (rlive == 0u) {
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


def guarded_source(k, cb):
    src = G._mm_seg_source(k, cb)
    if "rlive" in src:
        return src
    assert GUARD_ANCHOR in src, "wblk anchor not found"
    src = src.replace(
        GUARD_ANCHOR,
        "    uint rlive = 0u;\n"
        "    if (blk_len > sgr * 32u) {\n"
        "        uint rrem = blk_len - sgr * 32u;\n"
        "        rlive = (rrem > 32u) ? 4u : ((rrem + 7u) >> 3u);\n"
        "    }\n" + GUARD_ANCHOR, 1)
    assert MMA_OLD in src, "mma block not found"
    return src.replace(MMA_OLD, MMA_NEW, 1)


_kernels: dict = {}


def kernel(k, cb, version="v19d"):
    key = (k, int(cb), version)
    kk = _kernels.get(key)
    if kk is None:
        if version == "v19c":
            src = G._mm_seg_source(k, cb)
        else:
            src = guarded_source(k, cb)
        kk = _kernels[key] = mx.fast.metal_kernel(
            name=f"exl3_mm_seg_k{k}_cb{int(cb)}_{version}",
            input_names=["xh", "trellis", "inv_perm", "blk_tab", "nbr", "dims"],
            output_names=["out"],
            source=src,
            header="#include <metal_simdgroup_matrix>\n#include <metal_stdlib>\nusing namespace metal;\n",
        )
    return kk


def seg_mm(xh_sorted, trellis_u16, k, cb, tab, nbr, *, n_rows, tn_base,
           tiles_per_e, out_e, version="v19d"):
    """inner_mm_seg_mlx with a pluggable kernel version (same launch geometry)."""
    in_tiles, src_tiles, _ = trellis_u16.shape
    nb_max = int(tab.shape[1])
    if out_e % 64:
        raise ValueError(f"out_e {out_e} not a multiple of 64")
    dims = mx.array([in_tiles, tiles_per_e, src_tiles, tn_base, out_e, nb_max],
                    dtype=mx.uint32)
    return kernel(k, cb, version)(
        inputs=[xh_sorted.reshape(-1), trellis_u16.reshape(-1).view(mx.uint32),
                G._inv_perm_u32(), tab.reshape(-1), nbr, dims],
        template=[("T", mx.float16)],
        grid=((out_e // 64) * G._GEM_THREADS, nb_max, 1),
        threadgroup=(G._GEM_THREADS, 1, 1),
        output_shapes=[(n_rows, out_e)],
        output_dtypes=[mx.float16],
    )[0]
