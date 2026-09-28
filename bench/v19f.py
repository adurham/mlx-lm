#!/usr/bin/env python3
"""Segmented-GEMM kernel variants for the DSv4.1 MoE prefill shape.

Premise (measured, pD2/pD6/pD7): at R=512 x top-6 over 384 half-width experts,
the segmented grid gets ~8 live rows per expert with bm=64, so 8x of the mma
work and 4x of the A-fragment traffic is dead.  Decode is only ~3% of the time
(pD7), so the wins must come from work elimination and from the dependent-chain
structure of the inner loop, not from cheaper decode.

v19c : stock PonyExl3 v19c (baseline, bit-reference)
v19d : + row-fragment guard.  rlive = 4 clamped to ceil((blk_len-sgr*32)/8);
       dead row fragments skip their A/B loads and all 16 mmas.  The stock
       store loop already discards rows >= blk_len, so no live value changes.
v19e : v19d + all 4 A-fragment loads issued before the mma block (they are
       independent; with rlive=1 the k2 chain is the only MLP in a stage).
v19f : v19d + 4 in-tiles per barrier stage (2x fewer barriers, 2x the
       A-traffic amortization per stage).
v19g : v19f + A-load hoisting.

All variants are bit-exact vs v19c (asserted by the caller on real outputs).
"""
import mlx.core as mx

from mlx_lm.models.exl3 import gemv_metal as G

_WS = G._MM_WS           # 72: wblk row stride in halfs
_BN = G._MM_BN           # 64: out cols per threadgroup
_KERNELS: dict = {}


def source(k: int, cb, version: str) -> str:
    """Delegate to the library's own generators so the prototype can never
    drift from mlx_lm/models/exl3/gemv_metal.py."""
    base = G._mm_seg_source(k, cb)
    if version == "v19c":
        return base
    if version == "v19d":
        return G._mm_seg_guarded(base)
    if version == "v19e":
        return G._mm_seg_hoisted(base)
    raise ValueError(version)


_kernels: dict = {}


def kernel(k: int, cb, version: str):
    key = (k, int(cb), version)
    kk = _kernels.get(key)
    if kk is None:
        kk = _kernels[key] = mx.fast.metal_kernel(
            name=f"exl3_mm_seg_k{k}_cb{int(cb)}_{version}",
            input_names=["xh", "trellis", "inv_perm", "blk_tab", "nbr", "dims"],
            output_names=["out"], source=source(k, cb, version),
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
        grid=((out_e // _BN) * G._GEM_THREADS, nb_max, 1),
        threadgroup=(G._GEM_THREADS, 1, 1),
        output_shapes=[(n_rows, out_e)], output_dtypes=[mx.float16])[0]
