#!/usr/bin/env python3
"""v19e: guarded row fragments + A-fragment loads hoisted out of the k2 loop.

v19d removes the dead row-fragments (pure work elimination, bit-exact).  What
remains is a dependent chain per k2 iteration: device A-fragment load (long
latency) -> 4 mmas, repeated 4x per stage under two threadgroup barriers.  With
rlive=1 the threadgroup has ONE row fragment, so those 4 A loads are the only
memory-level parallelism available inside a stage; issuing them together instead
of serially is free (they are independent) and lifts the stall per stage.

Bit-exactness: only the ORDER of independent loads changes, plus the same
work-elimination as v19d, so live values are unchanged (asserted by callers).
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
            for (uint r = 0u; r < 4u; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(
                    a, xrow + (ulong)(r * 8u) * in_features + ks * 32u + k2 * 8u,
                    in_features);
                simdgroup_multiply_accumulate(C[r][0], a, bf0, C[r][0]);
                simdgroup_multiply_accumulate(C[r][1], a, bf1, C[r][1]);
                simdgroup_multiply_accumulate(C[r][2], a, bf2, C[r][2]);
                simdgroup_multiply_accumulate(C[r][3], a, bf3, C[r][3]);
            }
        }"""

MMA_NEW = """        for (uint r = 0u; r < rlive; r++) {
            // issue all 4 A-fragment loads before any mma: they are
            // independent, so this exposes 4x the memory-level parallelism in
            // the (rlive=1) tiny-segment case where the k2 loop is short.
            simdgroup_half8x8 af[4];
            for (uint k2 = 0u; k2 < 4u; k2++) {
                simdgroup_load(
                    af[k2],
                    xrow + (ulong)(r * 8u) * in_features + ks * 32u + k2 * 8u,
                    in_features);
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
                simdgroup_multiply_accumulate(C[r][0], af[k2], bf0, C[r][0]);
                simdgroup_multiply_accumulate(C[r][1], af[k2], bf1, C[r][1]);
                simdgroup_multiply_accumulate(C[r][2], af[k2], bf2, C[r][2]);
                simdgroup_multiply_accumulate(C[r][3], af[k2], bf3, C[r][3]);
            }
        }"""


def _guard(src):
    assert GUARD_ANCHOR in src
    return src.replace(
        GUARD_ANCHOR,
        "    uint rlive = 0u;\n"
        "    if (blk_len > sgr * 32u) {\n"
        "        uint rrem = blk_len - sgr * 32u;\n"
        "        rlive = (rrem > 32u) ? 4u : ((rrem + 7u) >> 3u);\n"
        "    }\n" + GUARD_ANCHOR, 1)


def guarded_source(k, cb):
    """v19d source (guard only)."""
    src = G._mm_seg_source(k, cb)
    if "uint rlive" in src:
        return src
    return _guard(src)


def hoisted_source(k, cb):
    """v19e source (guard + hoisted A loads)."""
    src = guarded_source(k, cb)
    if "simdgroup_half8x8 af[4]" in src:
        return src
    assert MMA_OLD in src, "mma block not found"
    return src.replace(MMA_OLD, MMA_NEW, 1)


SOURCES = {"v19c": lambda k, cb: G._mm_seg_source(k, cb),
           "v19d": guarded_source,
           "v19e": hoisted_source}

_kernels: dict = {}


def kernel(k, cb, version):
    key = (k, int(cb), version)
    kk = _kernels.get(key)
    if kk is None:
        kk = _kernels[key] = mx.fast.metal_kernel(
            name=f"exl3_mm_seg_k{k}_cb{int(cb)}_{version}",
            input_names=["xh", "trellis", "inv_perm", "blk_tab", "nbr", "dims"],
            output_names=["out"], source=SOURCES[version](k, cb),
            header="#include <metal_simdgroup_matrix>\n#include <metal_stdlib>\nusing namespace metal;\n")
    return kk


def seg_mm(xh_sorted, trellis_u16, k, cb, tab, nbr, *, n_rows, tn_base,
           tiles_per_e, out_e, version):
    in_tiles, src_tiles, _ = trellis_u16.shape
    nb_max = int(tab.shape[1])
    dims = mx.array([in_tiles, tiles_per_e, src_tiles, tn_base, out_e, nb_max],
                    dtype=mx.uint32)
    return kernel(k, cb, version)(
        inputs=[xh_sorted.reshape(-1), trellis_u16.reshape(-1).view(mx.uint32),
                G._inv_perm_u32(), tab.reshape(-1), nbr, dims],
        template=[("T", mx.float16)],
        grid=((out_e // 64) * G._GEM_THREADS, nb_max, 1),
        threadgroup=(G._GEM_THREADS, 1, 1),
        output_shapes=[(n_rows, out_e)], output_dtypes=[mx.float16])[0]
