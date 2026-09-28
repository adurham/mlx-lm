# Vendored from PonyExl3 (https://github.com/beamivalice/PonyExl3; Apache-2.0,
# see LICENSE and NOTICE alongside this file).
# Local change vs upstream: absolute `ponyexl3.*` imports rewritten to relative
# imports. Numerical code is unchanged unless noted below.
# Numerical changes vs upstream (see README.md): v3 chunked-prologue retune; v4 ALU
# optimizations defaulted ON (EXL3_DECODE_SWAR, EXL3_XDIRECT); additive "silu_clamp"
# MoE activation for DeepSeek V4.1's clamped SwiGLU (EXL3_MOE_CLAMP, default 10.0).
"""EXL3 MoE experts — SwitchGLU replacement for Qwen3.5/3.6 A-series.

The inversion of the dense problem: 256 tiny experts x 3 projections per
layer, only 8 active per token. Per layer the expert trellises are stacked
along out_tiles into ONE device buffer (gate+up interleaved per expert, down
separate), so decode runs ONE mapped GEMV per projection group covering all
selected experts (``tile_map`` indirection into the stacked trellis,
``tile_sub`` selecting each (expert, proj)'s rotated activation row).

Prefill decodes the whole stacked trellis to transient fp16 (v13t kernel) and
routes flattened (token, slot) pairs through ``mx.gather_mm`` — the same
shape mlx_lm's SwitchGLU uses, with EXL3 rotations applied per pair. Pairs
are sorted by expert first (gather_sort pattern) so the GEMM streams each
expert's weights once per chunk, not once per pair.

Interface matches mlx_lm's SwitchGLU: ``__call__(x, indices) -> (..., k, D)``;
routing, shared expert, and the weighted sum stay in Qwen3NextSparseMoeBlock.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .gemv_metal import (
    _GEM_THREADS,  # pyright: ignore[reportPrivateUsage]
    decode_full_eg_mlx,
    _fwd_perm_u32,  # pyright: ignore[reportPrivateUsage]
    _gemv_simd_kernel,  # pyright: ignore[reportPrivateUsage]
    inner_mm_seg_mlx,
)
from .ref.codebook import CodebookMode

_HAD_SCALE = 1.0 / 128.0**0.5
import os
_MOE_UNFUSED = os.environ.get("EXL3_MOE_FUSED", "1") == "0"
# v2 decode kernels: A2 one-tile-per-threadgroup gate+up (16x grid
# parallelism) + B2 down with the gate/up finish in its prologue
_MOE_V2 = os.environ.get("EXL3_MOE_V2", "1") == "1"
# v19 segmented simdgroup GEMM for prefill (EXL3_MOE_MM=0 falls back to
# decode-all + sorted gather_mm)
_MOE_MM = os.environ.get("EXL3_MOE_MM", "1") == "1"
_MOE_KERNELS: dict[tuple[int, int] | tuple[int, int, str], tuple[Any, Any]] = {}
_SEG_BM = 64
# Hybrid crossover: above this many sorted (token, slot) rows the steel
# gather_mm over expert-grouped decoded W out-runs the v19c/v19e kernel —
# steel's mma pipeline beats ours at huge M, while the segmented kernel wins
# below by skipping the fixed decode-all and W materialization (1.25-1.8x at
# S=512; measured Phase 19). The crossover is only reachable when the fallback
# can actually allocate its rhs — see _mm_fallback_viable.
_MM_MAX_ROWS = int(os.environ.get("EXL3_MM_MAX_ROWS", str(9 * 1024)))


def _mm_fallback_bytes(E: int, D: int, H: int) -> int:
    """Bytes the gather_mm fallback would materialize for one MoE layer."""
    return max(E * H * D, 2 * E * D * H) * 2


# MLX builds tensor shapes/offsets in int32; a buffer over this cannot allocate.
_INT32_MAX = 2**31 - 1


def _mm_fallback_viable(E: int, D: int, H: int) -> bool:
    """Whether the decode-all + gather_mm path can allocate for this geometry.

    The DSv4.1 MoE (E=384, D=5120, H=1152) cannot: the rhs is decode_full_eg's
    fp16 W, 4.53 GB (dn) + 9.06 GB (gu), both past MLX's int32 buffer limit. The
    failure is an uncaught nanobind OverflowError that kills the process, so the
    cap must be decided by geometry, not by the mma crossover."""
    return _mm_fallback_bytes(E, D, H) <= _INT32_MAX
# A2 out-tiles per threadgroup (1 = one 16-wide out-tile, the v3 default).
# >1 reuses each loaded trellis word across several out-tiles; must divide
# gu_tiles (hidden/16) so a group stays inside one projection.
_A2_TILES = int(os.environ.get("EXL3_MOE_A2_TILES", "1") or 1)

# Gate activation for stacked experts: Qwen MoE uses SwiGLU (silu*up);
# Gemma4 routed experts use GeGLU (gelu_approx(gate)*up), matching exllamav3.
# "silu_clamp" is DeepSeek V4.1 clamped SwiGLU: up two-sided, gate
# upper-only, both at EXL3_MOE_CLAMP (default 10.0). Additive mode --
# the stock modes are unchanged and remain the default.
_MOE_ACTIVATIONS = frozenset({"silu", "gelu", "silu_clamp"})

_CLAMP_LIMIT = float(os.environ.get("EXL3_MOE_CLAMP", "10.0") or 10.0)


def _moe_glu_metal_expr(act: str) -> str:
    """Metal expression: gate ``g``, value ``u`` -> activated hidden."""
    if act == "gelu":
        return (
            "float tanh_arg = 0.797884560803f * (g + 0.044715f * g * g * g);\n"
            "        float h = 0.5f * g * (1.0f + tanh(tanh_arg)) * u;"
        )
    if act == "silu_clamp":
        return (
            "g = fmin(g, %.1ff);\n" % _CLAMP_LIMIT
            + "        u = fmin(fmax(u, -%.1ff), %.1ff);\n" % (_CLAMP_LIMIT, _CLAMP_LIMIT)
            + "        float h = (g / (1.0f + exp(-g))) * u;"
        )
    return "float h = (g / (1.0f + exp(-g))) * u;"


def _moe_gate_activation(g: mx.array, u: mx.array, act: str) -> mx.array:
    if act == "gelu":
        return nn.gelu_approx(g) * u
    if act == "silu_clamp":
        g = mx.minimum(g, _CLAMP_LIMIT)
        u = mx.clip(u, -_CLAMP_LIMIT, _CLAMP_LIMIT)
        return nn.silu(g) * u
    return nn.silu(g) * u


@lru_cache(maxsize=None)
def _seg_table_fn(E: int, nb_max: int, bm: int) -> Callable[[mx.array], tuple[mx.array, mx.array]]:
    """Compiled (token,slot)->block table for the segmented GEMM: per-expert
    counts -> contiguous row blocks of <=bm sorted rows. Static shapes per
    (E, nb_max); nb_real stays a device scalar (no host sync)."""

    @mx.compile
    def _fn(sidx: mx.array):
        counts = mx.zeros((E,), dtype=mx.uint32).at[sidx].add(1)
        seg_start = mx.cumsum(counts) - counts
        nblk = (counts + (bm - 1)) // bm
        blk_off = mx.cumsum(nblk)  # inclusive: end block of each expert
        nb_real = blk_off[E - 1 : E]
        marks = mx.zeros((nb_max + 1,), dtype=mx.uint32).at[blk_off].add(1)
        blk_e = mx.minimum(mx.cumsum(marks)[:nb_max], E - 1)
        local = mx.arange(nb_max, dtype=mx.uint32) - (blk_off - nblk)[blk_e]
        row0 = seg_start[blk_e] + local * bm
        ln = mx.minimum(counts[blk_e] - local * bm, bm)
        tab = mx.stack([blk_e, row0, ln]).astype(mx.uint32)
        return tab, nb_real.astype(mx.uint32)

    return _fn

_compiled_rows_prep: Callable[[mx.array, mx.array], mx.array] | None = None
_compiled_rows_finish: Callable[[mx.array, mx.array], mx.array] | None = None


def _rows_prep() -> Callable[[mx.array, mx.array], mx.array]:
    """had(x_rows * suh_rows) with PER-ROW sign vectors, fp16, one dispatch."""
    global _compiled_rows_prep
    if _compiled_rows_prep is None:

        @mx.compile
        def _fn(x: mx.array, suh: mx.array) -> mx.array:
            _rows, n = x.shape[-2], x.shape[-1]
            xs = (x.astype(mx.float16) * suh.astype(mx.float16)).reshape(
                -1, n // 128, 128
            )
            return mx.hadamard_transform(xs, scale=_HAD_SCALE).reshape(x.shape)

        _compiled_rows_prep = _fn
    return _compiled_rows_prep


def _rows_finish() -> Callable[[mx.array, mx.array], mx.array]:
    """had(y_rows) * svh_rows with PER-ROW sign vectors, fp16."""
    global _compiled_rows_finish
    if _compiled_rows_finish is None:

        @mx.compile
        def _fn(y: mx.array, svh: mx.array) -> mx.array:
            n = y.shape[-1]
            yh = mx.hadamard_transform(
                y.astype(mx.float16).reshape(-1, n // 128, 128), scale=_HAD_SCALE
            ).reshape(y.shape)
            return yh * svh.astype(mx.float16)

        _compiled_rows_finish = _fn
    return _compiled_rows_finish


def _mapped_gemv(
    xh: mx.array,  # (n_sub, in) fp16, rotated per (expert-slot, proj)
    trellis_u16: mx.array,  # stacked (in_tiles, total_out_tiles, P)
    k: int,
    cb: CodebookMode,
    tile_map: mx.array,  # (local_out_tiles,) -> source tile index
    tile_sub: mx.array,  # (local_out_tiles,) -> xh row
) -> mx.array:
    """One GEMV launch over the selected experts' tiles (M=1)."""
    in_tiles, src_tiles, _ = trellis_u16.shape
    local_tiles = int(tile_map.shape[0])
    n_sub = int(xh.shape[0])
    out_features = local_tiles * 16
    dims = mx.array(
        [in_tiles, local_tiles, 1, 1, n_sub, src_tiles], dtype=mx.uint32
    )
    kernel = _gemv_simd_kernel(k, cb, 1)
    out = kernel(
        inputs=[
            xh.reshape(-1).astype(mx.float16),
            trellis_u16.reshape(-1).view(mx.uint32),
            _fwd_perm_u32(),
            tile_sub,
            tile_map,
            dims,
        ],
        template=[("T", mx.float32)],
        grid=(local_tiles * _GEM_THREADS, 1, 1),
        threadgroup=(_GEM_THREADS, 1, 1),
        output_shapes=[(out_features,)],
        output_dtypes=[mx.float32],
    )[0]
    return out


_BUTTERFLY = """
    for (uint s_ = 0u; s_ < 7u; s_++) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint bit = 1u << s_;
        for (uint p = tid; p < {n} / 2u; p += 128u) {{
            uint b_ = p >> 6u;
            uint w_ = p & 63u;
            uint i_ = b_ * 128u + (((w_ & ~(bit - 1u)) << 1u) | (w_ & (bit - 1u)));
            uint jj_ = i_ + bit;
            float a_ = {buf}[i_];
            float c_ = {buf}[jj_];
            {buf}[i_] = a_ + c_;
            {buf}[jj_] = a_ - c_;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_HS = "0.08838834764831845f"  # 1/sqrt(128)


def _tile_loop(decode: str, tile_expr: str, xrow_expr: str, dest: str, tiles: int = 8) -> str:
    """v18-style: ``tiles`` sequential out-tiles, v12 inner loop each, reduce
    into ``dest[ot*16+col]`` (fp16-cast to match the unfused pipeline)."""
    return f"""
    for (uint ot = 0u; ot < {tiles}u; ot++) {{
        uint tn_src = {tile_expr};
        float acc[8] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
        for (uint tk = sg; tk < in_tiles; tk += 4u) {{
            const device uint* words = trellis + (tk * src_tiles + tn_src) * PACKED_U32;
            float x_lane = {xrow_expr};
            for (uint g = 0u; g < 2u; g++) {{
                ulong merged = ((ulong)words[ig0_[g]] << 32) | (ulong)words[ig1_[g]];
                uint s = s3_[g];
                uint cws[4] = {{uint(merged >> (s + 3u * K_BITS)) & 0xFFFFu,
                               uint(merged >> (s + 2u * K_BITS)) & 0xFFFFu,
                               uint(merged >> (s + K_BITS)) & 0xFFFFu,
                               uint(merged >> s) & 0xFFFFu}};
                for (uint q = 0u; q < 4u; q++) {{
                    uint jw = g * 4u + q;
                    uint cw = cws[q];
{decode}
                    acc[jw] = fma(simd_shuffle(x_lane, ushort(row_[jw])), dq_val, acc[jw]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < 8u; j++) {{
            tg_w[sg][pos_[j]] = acc[j];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 16u) {{
            float s = 0.0f;
            for (uint r = 0u; r < 16u; r++) {{
                uint p = r * 16u + tid;
                s += tg_w[0][p] + tg_w[1][p] + tg_w[2][p] + tg_w[3][p];
            }}
            {dest}[ot * 16u + tid] = float(half(s));
        }}
    }}
"""


def _lane_setup() -> str:
    return """
    uint tid = thread_position_in_threadgroup.x;     // 128 threads
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint pos_[8];
    uint row_[8];
    for (uint j = 0u; j < 4u; j++) {
        uint t = lane * 4u + j;
        pos_[j * 2u] = perm[t * 2u];
        pos_[j * 2u + 1u] = perm[t * 2u + 1u];
        row_[j * 2u] = pos_[j * 2u] >> 4u;
        row_[j * 2u + 1u] = pos_[j * 2u + 1u] >> 4u;
    }
    uint ig0_[2];
    uint ig1_[2];
    uint s3_[2];
    for (uint g = 0u; g < 2u; g++) {
        uint c0 = lane * 8u + g * 4u;
        int e_last = int(c0 + 4u) * K_BITS + 256 * K_BITS;
        int i_end = (e_last - 1) / 32;
        ig1_[g] = uint(i_end) % PACKED_U32;
        ig0_[g] = uint(i_end - 1) % PACKED_U32;
        s3_[g] = uint((i_end + 1) * 32 - e_last);
    }
    threadgroup float tg_w[4][256];
"""


def _moe_gateup_source(k: int, cb: CodebookMode, act: str = "silu") -> str:
    """Kernel A: gate+up GEMV pair + dual-block post-Hadamard + svh + GLU,
    one dispatch for ALL selected experts. Threadgroup owns a (gate-block,
    up-block) pair: 128 hidden cols of one expert."""
    glu = _moe_glu_metal_expr(act)
    decode = __import__(f"{__package__}.gemv_metal", fromlist=["_decode_expr"])._decode_expr(cb, cw_in="cw")
    return f"""
#define PACKED_U32 {k * 256 // 32}
#define K_BITS {k}
    uint in_tiles = dims[0];
    uint gu_tiles = dims[1];          // tiles per gate (== per up)
    uint src_tiles = dims[2];         // stride of the stacked trellis
    uint in_features = in_tiles * 16u;
    uint hidden = gu_tiles * 16u;
    uint E_total = dims[3];
    uint bpe = gu_tiles / 8u;         // 128-col blocks per projection
    uint blkg = threadgroup_position_in_grid.x;
    uint slot = blkg / bpe;
    uint block = blkg % bpe;
    uint e = sel[slot];
{_lane_setup()}
    threadgroup float tg_g[128];
    threadgroup float tg_u[128];
    uint gbase = e * gu_tiles + block * 8u;                 // [all gates | all ups]
    uint ubase = (E_total + e) * gu_tiles + block * 8u;
    uint xg = (slot * 2u) * in_features;
    uint xu = (slot * 2u + 1u) * in_features;
{_tile_loop(decode, "gbase + ot", "float(xh[xg + tk * 16u + (lane & 15u)])", "tg_g")}
{_tile_loop(decode, "ubase + ot", "float(xh[xu + tk * 16u + (lane & 15u)])", "tg_u")}
{_BUTTERFLY.format(n="128u", buf="tg_g")}
{_BUTTERFLY.format(n="128u", buf="tg_u")}
    if (tid < 128u) {{
        uint col = block * 128u + tid;
        float g = tg_g[tid] * {_HS} * float(gu_svh[e * 2u * hidden + col]);
        float u = tg_u[tid] * {_HS} * float(gu_svh[e * 2u * hidden + hidden + col]);
        {glu}
        out[slot * hidden + col] = half(h);
    }}
"""


def _moe_down_source(k: int, cb: CodebookMode) -> str:
    """Kernel B: down projection with INLINE pre-rotation (the hidden is only
    ~512 wide — 2 KB threadgroup butterfly, no occupancy penalty) + post-had
    epilogue + svh. One dispatch for all selected experts."""
    decode = __import__(f"{__package__}.gemv_metal", fromlist=["_decode_expr"])._decode_expr(cb, cw_in="cw")
    return f"""
#define PACKED_U32 {k * 256 // 32}
#define K_BITS {k}
    uint in_tiles = dims[0];          // hidden tiles
    uint out_tiles_e = dims[1];       // out tiles per expert
    uint src_tiles = dims[2];
    uint in_features = in_tiles * 16u;
    uint out_features = out_tiles_e * 16u;
    uint bpe = out_tiles_e / 8u;
    uint blkg = threadgroup_position_in_grid.x;
    uint slot = blkg / bpe;
    uint block = blkg % bpe;
    uint e = sel[slot];
{_lane_setup()}
    threadgroup float tg_xh[1024];
    threadgroup float tg_y[128];
    // inline pre-rotate: h[slot] * dn_suh[e] -> butterfly
    for (uint idx = tid; idx < in_features; idx += 128u) {{
        tg_xh[idx] = float(hbuf[slot * in_features + idx])
                   * float(dn_suh[e * in_features + idx]) * {_HS};
    }}
{_BUTTERFLY.format(n="in_features", buf="tg_xh")}
    uint dbase = e * out_tiles_e + block * 8u;
{_tile_loop(decode, "dbase + ot", "tg_xh[tk * 16u + (lane & 15u)]", "tg_y")}
{_BUTTERFLY.format(n="128u", buf="tg_y")}
    if (tid < 128u) {{
        uint col = block * 128u + tid;
        out[slot * out_features + col] =
            half(tg_y[tid] * {_HS} * float(dn_svh[e * out_features + col]));
    }}
"""


def _moe_gateup2_source(k: int, cb: CodebookMode, tiles: int = 1) -> str:
    """Kernel A2: gate+up mapped GEMV at ONE out-tile per threadgroup (v12
    grid shape — 2*gu_tiles*E_sel threadgroups instead of A's E_sel*bpe=36,
    which left the GPU ~3x under its ALU floor at decode). Emits RAW
    pre-Hadamard y; the butterfly/svh/SwiGLU moved into kernel B2's
    prologue, keeping the layer at 3 dispatches."""
    decode = __import__(f"{__package__}.gemv_metal", fromlist=["_decode_expr"])._decode_expr(cb, cw_in="cw")
    return f"""
#define PACKED_U32 {k * 256 // 32}
#define K_BITS {k}
    uint in_tiles = dims[0];
    uint gu_tiles = dims[1];          // tiles per gate (== per up)
    uint src_tiles = dims[2];         // stride of the stacked trellis
    uint in_features = in_tiles * 16u;
    uint E_total = dims[3];
    // v3: A2T adjacent out-tiles per threadgroup (load reuse). A2T must
    // divide gu_tiles, so `t` never straddles the gate/up boundary.
    uint groups = 2u * (gu_tiles / {tiles}u);
    uint blkg = threadgroup_position_in_grid.x;
    uint slot = blkg / groups;
    uint t = (blkg % groups) * {tiles}u;
    uint is_up = (t >= gu_tiles) ? 1u : 0u;
    uint tile0 = t - is_up * gu_tiles;
    uint e = sel[slot];
{_lane_setup()}
    uint tnsrc0 = (is_up ? (E_total + e) : e) * gu_tiles + tile0;
    uint xoff = (slot * 2u + is_up) * in_features;
    uint obase = slot * (2u * gu_tiles * 16u) + t * 16u;
{_tile_loop(decode, "tnsrc0 + ot", "float(xh[xoff + tk * 16u + (lane & 15u)])", "(out + obase)", tiles=tiles)}
"""


def _moe_down2_source(k: int, cb: CodebookMode, act: str = "silu",
                      hidden: int = 0) -> str:
    """Kernel B2: down projection whose prologue FINISHES gate+up (dual
    butterfly + svh + GLU) from kernel A2's raw output, then pre-rotates
    and runs the down tile loop.

    v3 chunked prologue: walks ``hidden`` in whole-Hadamard-block chunks
    (CH = 256, or 128 when hidden is only a multiple of 128) instead of staging
    the entire 2*hidden gate|up buffer. The butterfly is 128-block-local by
    construction (its ``i_``/``jj_`` indices are ``b_*128 + ...`` and
    ``i_+bit`` with bit<=64, so both operands always sit in one 128-block), so
    grouping whole blocks differently is bit-identical -- chunking only changes
    which blocks are live at once.

    ``hidden`` is interpolated into the source (``tg_xh``'s extent) AND into the
    kernel name/cache key: ``tg_xh`` must be exactly ``hidden`` floats because
    the down tile loop indexes it by hidden-tile, and two different hidden sizes
    must never share a compiled Metal kernel.
    """
    if hidden <= 0:
        raise ValueError("_moe_down2_source requires an explicit hidden > 0")
    # Chunk size: the largest whole number of 128-blocks that divides hidden
    # exactly, bounded by EXL3_MOE_CHUNK if set (for tile-size sweeps).
    # EXL3_MOE_CHUNK=0 means "no chunking" -- one pass over 2*hidden, which is
    # the original kernel's semantics and needs 2*hidden of threadgroup memory.
    import os as _os
    _env = int(_os.environ.get("EXL3_MOE_CHUNK", "0") or 0)
    if _env:
        if hidden % _env:
            raise ValueError(f"EXL3_MOE_CHUNK={_env} does not divide hidden={hidden}")
        ch = _env
    elif hidden % 256 == 0:
        ch = 256
    elif hidden % 128 == 0:
        ch = 128
    else:
        raise ValueError(
            f"hidden={hidden} must be a multiple of 128 (Hadamard block size)"
        )
    if ch == 2 * hidden:
        ch = 0   # signal: single-pass mode (emit the original whole-buffer form)
    glu = _moe_glu_metal_expr(act)
    decode = __import__(f"{__package__}.gemv_metal", fromlist=["_decode_expr"])._decode_expr(cb, cw_in="cw")
    return f"""
#define PACKED_U32 {k * 256 // 32}
#define K_BITS {k}
    uint in_tiles = dims[0];          // hidden tiles
    uint out_tiles_e = dims[1];       // out tiles per expert
    uint src_tiles = dims[2];
    uint E_total = dims[3];
    uint in_features = in_tiles * 16u;   // hidden
    uint out_features = out_tiles_e * 16u;
    uint bpe = out_tiles_e / 8u;
    uint blkg = threadgroup_position_in_grid.x;
    uint slot = blkg / bpe;
    uint block = blkg % bpe;
    uint e = sel[slot];
{_lane_setup()}
    uint hidden = in_features;
    const uint CH = {ch}u;                // whole Hadamard blocks
    threadgroup float tg_gc[{ch if ch else 2 * hidden}];   // gate chunk (v3)
    threadgroup float tg_uc[{ch if ch else 2 * hidden}];   // up chunk (v3)
    threadgroup float tg_xh[{hidden}];    // full hidden; the tile loop indexes it
    threadgroup float tg_y[128];
    // prologue 1+2 fused, chunked: finish gate+up (butterfly + svh + GLU)
    // -> pre-rotate hidden. Every chunk is a whole number of 128-blocks, so
    // this is bit-identical to the former whole-2*hidden butterfly.
    for (uint c = 0u; c < hidden; c += CH) {{
        for (uint i = tid; i < CH; i += 128u) {{
            tg_gc[i] = float(ygu[slot * 2u * hidden + c + i]) * {_HS};
            tg_uc[i] = float(ygu[slot * 2u * hidden + hidden + c + i]) * {_HS};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
{_BUTTERFLY.format(n="CH", buf="tg_gc")}
{_BUTTERFLY.format(n="CH", buf="tg_uc")}
        for (uint i = tid; i < CH; i += 128u) {{
            float g = tg_gc[i] * float(gu_svh[e * 2u * hidden + c + i]);
            float u = tg_uc[i] * float(gu_svh[e * 2u * hidden + hidden + c + i]);
            {glu}
            h = float(half(h));   // match the unfused pipeline's fp16 handoff
            tg_xh[c + i] = h * float(dn_suh[e * in_features + c + i]) * {_HS};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
{_BUTTERFLY.format(n="in_features", buf="tg_xh")}
    uint dbase = e * out_tiles_e + block * 8u;
{_tile_loop(decode, "dbase + ot", "tg_xh[tk * 16u + (lane & 15u)]", "tg_y")}
{_BUTTERFLY.format(n="128u", buf="tg_y")}
    if (tid < 128u) {{
        uint col = block * 128u + tid;
        out[slot * out_features + col] =
            half(tg_y[tid] * {_HS} * float(dn_svh[e * out_features + col]));
    }}
"""


# --- phase 9 spike: env-gated, bit-identical cross-lane removal
# (EXL3_XDIRECT, default on). ``simd_shuffle(x_lane, ushort(row_[jw]))``
# broadcasts lane row_[jw]'s x_lane, which holds x[..., (row_[jw] & 15)] --
# so a direct indexed read computes the same fp16 value with no cross-lane
# op. md5-identical over R=1/4/6/8; measured R=1 1.031 -> 0.823 ms (2.65x ->
# 2.11x vs production), R=4-8 -> ~2.25-2.33x (see phase9 doc).
_XDIRECT = __import__("os").environ.get("EXL3_XDIRECT", "1") == "1"


def _xdirect_src(src: str) -> str:
    """Rewrite shuffle-broadcast sites to direct indexed reads."""
    import re as _re
    if not _XDIRECT:
        return src
    exprs = _re.findall(r"float x_lane = (.+);", src)
    if not exprs:
        return src
    it = iter(exprs)

    def _sub(_m):
        return next(it).replace("(lane & 15u)", "(row_[jw] & 15u)")
    out, n = _re.subn(
        _re.escape("simd_shuffle(x_lane, ushort(row_[jw]))"), _sub, src)
    if n != len(exprs):
        raise ValueError(
            f"xdirect: {n} shuffle sites vs {len(exprs)} x_lane defs")
    return out


if _XDIRECT:
    def _mk_xdirect(_fn):
        def _wrapped(*a, **kw):
            return _xdirect_src(_fn(*a, **kw))
        return _wrapped
    for _n in ("_moe_gateup_source", "_moe_down_source",
               "_moe_gateup2_source", "_moe_down2_source"):
        globals()[_n] = _mk_xdirect(globals()[_n])


class EXL3SwitchGLU(nn.Module):
    """Stacked-expert EXL3 SwitchGLU (gate+up fused per expert, then down)."""

    def __init__(
        self,
        *,
        gu_trellis: mx.array,  # (in_tiles, E*2*gu_tiles, P) [gate|up] per expert
        gu_suh: mx.array,  # (E, 2, in)
        gu_svh: mx.array,  # (E, 2*hidden)
        dn_trellis: mx.array,  # (hid_tiles, E*out_tiles, P)
        dn_suh: mx.array,  # (E, hidden)
        dn_svh: mx.array,  # (E, in)
        k: int,
        cb: CodebookMode,
        activation: str = "silu",
    ):
        super().__init__()
        if activation not in _MOE_ACTIVATIONS:
            raise ValueError(f"unsupported MoE activation {activation!r}")
        self._gu_trellis = gu_trellis
        self._gu_suh = gu_suh
        self._gu_svh = gu_svh
        self._dn_trellis = dn_trellis
        self._dn_suh = dn_suh
        self._dn_svh = dn_svh
        self._k = k
        self._cb = cb
        self._activation = activation
        self.num_experts = int(gu_suh.shape[0])
        self.input_dims = int(gu_suh.shape[2])
        self.hidden_dims = int(dn_suh.shape[1])
        self._gu_tiles = self.hidden_dims // 16  # tiles per gate (== per up)
        self._dn_tiles = self.input_dims // 16
        # v3 threadgroup-budget guard: the chunked B2 prologue holds
        # tg_xh[hidden_dims] plus two ch-sized chunks and tg_y[128], all fp32.
        # Apple GPUs expose 32 KiB of threadgroup memory; verify we are inside
        # it and fail loudly at construction rather than at kernel dispatch.
        import os as _os
        _env = int(_os.environ.get("EXL3_MOE_CHUNK", "0") or 0)
        _ch = _env if _env else (256 if self.hidden_dims % 256 == 0 else 128)
        if _ch == 2 * self.hidden_dims:
            _ch = 0
        _tg_bytes = 4 * (self.hidden_dims + (2 * _ch if _ch else 2 * self.hidden_dims) + 128)
        _TG_LIMIT = 32768
        if _tg_bytes > _TG_LIMIT:
            raise ValueError(
                f"B2 prologue needs {_tg_bytes} B of threadgroup memory for "
                f"hidden_dims={self.hidden_dims} (limit {_TG_LIMIT} B); the "
                f"chunk size must shrink further for this model"
            )

    def _kernels(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        key = (self._k, int(self._cb), self._activation, "v1")
        pair = _MOE_KERNELS.get(key)
        if pair is None:
            act = self._activation
            kA = mx.fast.metal_kernel(
                name=f"exl3_moe_gateup_k{self._k}_cb{int(self._cb)}_{act}_v1",
                input_names=["xh", "trellis", "perm", "sel", "gu_svh", "dims"],
                output_names=["out"],
                source=_moe_gateup_source(self._k, self._cb, act),
            )
            kB = mx.fast.metal_kernel(
                name=f"exl3_moe_down_k{self._k}_cb{int(self._cb)}_{act}_v1",
                input_names=["hbuf", "trellis", "perm", "sel", "dn_suh", "dn_svh", "dims"],
                output_names=["out"],
                source=_moe_down_source(self._k, self._cb),
            )
            pair = _MOE_KERNELS[key] = (kA, kB)
        return pair

    def _gateup_fused(self, x2d: mx.array, sel: mx.array) -> mx.array:
        """Kernel A: gate+up+SwiGLU for selected experts/slots."""
        kA, _ = self._kernels()
        E_sel = int(sel.shape[0])
        D, H = self.input_dims, self.hidden_dims
        sel_u = sel.astype(mx.uint32)
        suh_sel = self._gu_suh[sel].reshape(E_sel * 2, D)
        xh = _rows_prep()(mx.broadcast_to(x2d, (E_sel * 2, D)), suh_sel)
        dims = mx.array(
            [D // 16, self._gu_tiles, int(self._gu_trellis.shape[1]), self.num_experts],
            dtype=mx.uint32,
        )
        return kA(
            inputs=[
                xh.reshape(-1),
                self._gu_trellis.reshape(-1).view(mx.uint32),
                _fwd_perm_u32(),
                sel_u,
                self._gu_svh.reshape(-1),
                dims,
            ],
            template=[("T", mx.float16)],
            grid=(E_sel * (self._gu_tiles // 8) * _GEM_THREADS, 1, 1),
            threadgroup=(_GEM_THREADS, 1, 1),
            output_shapes=[(E_sel * H,)],
            output_dtypes=[mx.float16],
        )[0]

    def _down_mapped(self, h: mx.array, sel: mx.array) -> mx.array:
        """Mapped GEMV down path (safe for large hidden; MoE expert scale)."""
        E_sel = int(sel.shape[0])
        H, D = self.hidden_dims, self.input_dims
        h2d = h.reshape(E_sel, H)
        ar_dn = mx.arange(self._dn_tiles, dtype=mx.uint32)
        tile_map = (
            sel[:, None].astype(mx.uint32) * self._dn_tiles + ar_dn
        ).reshape(-1)
        tile_sub = mx.repeat(mx.arange(E_sel, dtype=mx.uint32), self._dn_tiles)
        xh = _rows_prep()(h2d, self._dn_suh[sel])
        y = _mapped_gemv(
            xh, self._dn_trellis, self._k, self._cb, tile_map, tile_sub
        ).reshape(E_sel, D)
        return _rows_finish()(y.astype(mx.float16), self._dn_svh[sel])

    def _down_fused(self, h: mx.array, sel: mx.array) -> mx.array:
        """Kernel B: inline pre-rotate down (hidden must fit tg_xh[1024])."""
        _, kB = self._kernels()
        E_sel = int(sel.shape[0])
        D, H = self.input_dims, self.hidden_dims
        sel_u = sel.astype(mx.uint32)
        dims = mx.array(
            [H // 16, self._dn_tiles, int(self._dn_trellis.shape[1]), 0],
            dtype=mx.uint32,
        )
        y = kB(
            inputs=[
                h,
                self._dn_trellis.reshape(-1).view(mx.uint32),
                _fwd_perm_u32(),
                sel_u,
                self._dn_suh.reshape(-1),
                self._dn_svh.reshape(-1),
                dims,
            ],
            template=[("T", mx.float16)],
            grid=(E_sel * (self._dn_tiles // 8) * _GEM_THREADS, 1, 1),
            threadgroup=(_GEM_THREADS, 1, 1),
            output_shapes=[(E_sel * D,)],
            output_dtypes=[mx.float16],
        )[0]
        return y.reshape(E_sel, D)

    def _kernels2(self):
        if self._gu_tiles % _A2_TILES:
            raise ValueError(
                f"EXL3_MOE_A2_TILES={_A2_TILES} must divide gu_tiles="
                f"{self._gu_tiles} (hidden/16)"
            )
        key = (self._k, int(self._cb), self._activation, "v3",
               int(self.hidden_dims), _A2_TILES)
        pair = _MOE_KERNELS.get(key)
        if pair is None:
            act = self._activation
            kA = mx.fast.metal_kernel(
                name=(
                    f"exl3_moe_gateup2_k{self._k}_cb{int(self._cb)}_{act}"
                    f"_v2_h{int(self.hidden_dims)}_t{_A2_TILES}"
                ),
                input_names=["xh", "trellis", "perm", "sel", "dims"],
                output_names=["out"],
                source=_moe_gateup2_source(self._k, self._cb, tiles=_A2_TILES),
            )
            kB = mx.fast.metal_kernel(
                name=f"exl3_moe_down2_k{self._k}_cb{int(self._cb)}_{act}_v3_h{int(self.hidden_dims)}",
                input_names=[
                    "ygu", "trellis", "perm", "sel",
                    "gu_svh", "dn_suh", "dn_svh", "dims",
                ],
                output_names=["out"],
                source=_moe_down2_source(self._k, self._cb, act,
                                          hidden=int(self.hidden_dims)),
            )
            pair = _MOE_KERNELS[key] = (kA, kB)
        return pair

    def _decode_fused2(self, x2d: mx.array, indices: mx.array) -> mx.array:
        """A2 (one tile/threadgroup, raw output) + B2 (finish in prologue):
        same 3 dispatches as v1, ~16x the gate+up grid parallelism.

        Shape-generic over rows: ``x2d`` is (R, D), ``indices`` (R, kk) —
        slots are (row, expert) pairs, so an 8-row spec verify is 72 slots
        (4.6k A2 threadgroups), decode-class cost instead of the _prefill
        sort/table machinery."""
        kA, kB = self._kernels2()
        R, kk = int(indices.shape[0]), int(indices.shape[1])
        E_sel = R * kk
        D, H = self.input_dims, self.hidden_dims
        sel = indices.reshape(-1)
        sel_u = sel.astype(mx.uint32)
        suh_sel = self._gu_suh[sel].reshape(E_sel * 2, D)
        x_rep = mx.broadcast_to(
            x2d[:, None, :], (R, kk * 2, D)
        ).reshape(E_sel * 2, D)
        xh = _rows_prep()(x_rep, suh_sel)
        dims = mx.array(
            [D // 16, self._gu_tiles, int(self._gu_trellis.shape[1]), self.num_experts],
            dtype=mx.uint32,
        )
        ygu = kA(
            inputs=[
                xh.reshape(-1),
                self._gu_trellis.reshape(-1).view(mx.uint32),
                _fwd_perm_u32(),
                sel_u,
                dims,
            ],
            template=[("T", mx.float16)],
            grid=(E_sel * (2 * self._gu_tiles // _A2_TILES) * _GEM_THREADS, 1, 1),
            threadgroup=(_GEM_THREADS, 1, 1),
            output_shapes=[(E_sel * 2 * H,)],
            output_dtypes=[mx.float16],
        )[0]
        dims_b = mx.array(
            [H // 16, self._dn_tiles, int(self._dn_trellis.shape[1]), self.num_experts],
            dtype=mx.uint32,
        )
        y = kB(
            inputs=[
                ygu,
                self._dn_trellis.reshape(-1).view(mx.uint32),
                _fwd_perm_u32(),
                sel_u,
                self._gu_svh.reshape(-1),
                self._dn_suh.reshape(-1),
                self._dn_svh.reshape(-1),
                dims_b,
            ],
            template=[("T", mx.float16)],
            grid=(E_sel * (self._dn_tiles // 8) * _GEM_THREADS, 1, 1),
            threadgroup=(_GEM_THREADS, 1, 1),
            output_shapes=[(E_sel * D,)],
            output_dtypes=[mx.float16],
        )[0]
        return y.reshape(R, kk, D)

    def _v2_ok(self) -> bool:
        # v3: the chunked B2 prologue removed the old hidden<=512 cap (that cap
        # came from tg_g[1024] being sized 2*hidden). What remains is purely
        # structural: hidden must be a whole number of Hadamard blocks (128),
        # because the prologue works in whole blocks and tg_xh is exactly
        # `hidden` floats. The pre-existing %128 requirement is therefore kept
        # and the 512 ceiling is dropped -- so hidden=128/384 (which already
        # worked) still work, and hidden=2304 now works too.
        return _MOE_V2 and self.hidden_dims % 128 == 0

    # ---- decode: one token, k selected experts, THREE dispatches --------
    def _decode_fused(self, x2d: mx.array, sel: mx.array) -> mx.array:
        if self._v2_ok():
            return self._decode_fused2(x2d, sel[None, :])[0]
        h = self._gateup_fused(x2d, sel)
        if self.hidden_dims > 1024:
            return self._down_mapped(h, sel)
        return self._down_fused(h, sel)

    # ---- decode (unfused fallback) --------------------------------------
    def _decode(self, x2d: mx.array, sel: mx.array) -> mx.array:
        E_sel = int(sel.shape[0])
        gt = self._gu_tiles
        ar = mx.arange(gt, dtype=mx.uint32)
        sel_u = sel[:, None].astype(mx.uint32)
        tile_map = mx.concatenate(
            [sel_u * gt + ar, (self.num_experts + sel_u) * gt + ar], axis=1
        ).reshape(-1)
        ar_gu = mx.arange(2 * gt, dtype=mx.uint32)
        proj = (ar_gu >= gt).astype(mx.uint32)
        tile_sub = (
            mx.arange(E_sel, dtype=mx.uint32)[:, None] * 2 + proj
        ).reshape(-1)

        suh_sel = self._gu_suh[sel].reshape(E_sel * 2, self.input_dims)
        xh = _rows_prep()(
            mx.broadcast_to(x2d, (E_sel * 2, self.input_dims)), suh_sel
        )
        y = _mapped_gemv(
            xh, self._gu_trellis, self._k, self._cb, tile_map, tile_sub
        ).reshape(E_sel, 2 * self.hidden_dims)
        y = _rows_finish()(y.astype(mx.float16), self._gu_svh[sel])
        g, u = mx.split(y, 2, axis=-1)
        h = _moe_gate_activation(g, u, self._activation)

        ar_dn = mx.arange(self._dn_tiles, dtype=mx.uint32)
        tile_map = (
            sel[:, None].astype(mx.uint32) * self._dn_tiles + ar_dn
        ).reshape(-1)
        tile_sub = mx.repeat(mx.arange(E_sel, dtype=mx.uint32), self._dn_tiles)
        xh = _rows_prep()(h, self._dn_suh[sel])
        y = _mapped_gemv(
            xh, self._dn_trellis, self._k, self._cb, tile_map, tile_sub
        ).reshape(E_sel, self.input_dims)
        return _rows_finish()(y.astype(mx.float16), self._dn_svh[sel])

    # ---- prefill: decode-all + sorted gather_mm --------------------------
    def _prefill(self, x: mx.array, indices: mx.array) -> mx.array:
        E, D, H = self.num_experts, self.input_dims, self.hidden_dims
        B, S, kk = indices.shape
        N = B * S * kk
        flat = indices.reshape(-1)

        # Sort (token, slot) pairs by expert (mlx_lm SwitchGLU's gather_sort)
        # so gather_mm streams each expert's weights once over a contiguous
        # run instead of once per pair — 10.6 -> 2.4 ms per projection at
        # S=512 on 35B-A3B dims (Phase 18 probe).
        order = mx.argsort(flat)
        inv = mx.argsort(order)
        sidx = flat[order]
        idx = sidx.reshape(N, 1).astype(mx.uint32)
        tok = (mx.arange(N, dtype=mx.uint32) // kk)[order]

        # Path selection. The segmented kernel needs H,D % 64; the gather_mm
        # fallback needs the decode-all rhs to fit in an int32-shaped buffer
        # (it does NOT at DSv4.1 dims -- see _mm_fallback_viable). Where the
        # fallback is impossible the cap must not send work to it: the failure
        # mode is an uncaught OverflowError, not a slow path.
        seg_ok = _MOE_MM and H % 64 == 0 and D % 64 == 0
        fb_viable = _mm_fallback_viable(E, D, H)
        if not seg_ok and not fb_viable:
            raise ValueError(
                f"EXL3SwitchGLU: no viable prefill path (seg needs H,D %64: "
                f"H={H} D={D}; gather fallback needs "
                f"{_mm_fallback_bytes(E, D, H)/1e9:.2f} GB rhs <= int32)"
            )
        # Stock rule: segmented below the crossover, gather_mm above it. When the
        # rhs cannot allocate, the crossover is unreachable -> stay segmented.
        use_mm = seg_ok and (N <= _MM_MAX_ROWS or not fb_viable)
        tab: mx.array = mx.array([])
        nbr: mx.array = mx.array([])
        if use_mm:
            # v19c loads x rows device-direct: pad the prep by one block so
            # ragged tail blocks stay in bounds (pad rows feed discarded C).
            tok_x = mx.concatenate([tok, mx.zeros((_SEG_BM,), dtype=tok.dtype)])
            sidx_x = mx.concatenate(
                [sidx, mx.zeros((_SEG_BM,), dtype=sidx.dtype)]
            )
        else:
            tok_x, sidx_x = tok, sidx
        x_pairs = x.reshape(B * S, D)[tok_x]

        xg = _rows_prep()(x_pairs, self._gu_suh[sidx_x, 0])
        xu = _rows_prep()(x_pairs, self._gu_suh[sidx_x, 1])

        if use_mm:
            # v19c segmented simdgroup GEMM: decode each weight tile once into
            # threadgroup memory, multiply against the sorted token blocks —
            # no transient fp16 W, no gather_mm.
            nb_max = N // _SEG_BM + E + 1
            tab, nbr = _seg_table_fn(E, nb_max, _SEG_BM)(sidx)
            gt = self._gu_tiles
            g = inner_mm_seg_mlx(
                xg, self._gu_trellis, self._k, self._cb, tab, nbr,
                n_rows=N, tn_base=0, tiles_per_e=gt, out_e=H,
            )
            u = inner_mm_seg_mlx(
                xu, self._gu_trellis, self._k, self._cb, tab, nbr,
                n_rows=N, tn_base=E * gt, tiles_per_e=gt, out_e=H,
            )
        else:
            # gate and up are contiguous in the stacked trellis: one decode,
            # expert-grouped (2E, D, H) so the gather rhs is contiguous
            w_gu = decode_full_eg_mlx(
                self._gu_trellis, self._k, self._cb, tiles_per_e=self._gu_tiles
            )
            g = mx.gather_mm(
                xg.reshape(N, 1, 1, D), w_gu[:E],
                rhs_indices=idx, sorted_indices=True,
            )
            u = mx.gather_mm(
                xu.reshape(N, 1, 1, D), w_gu[E:],
                rhs_indices=idx, sorted_indices=True,
            )
        g = _rows_finish()(g.reshape(N, H).astype(mx.float16), self._gu_svh[sidx, :H])
        u = _rows_finish()(u.reshape(N, H).astype(mx.float16), self._gu_svh[sidx, H:])
        h = _moe_gate_activation(g, u, self._activation)

        if use_mm:
            h_pad = mx.concatenate([h, mx.zeros((_SEG_BM, H), dtype=h.dtype)])
            xd = _rows_prep()(h_pad, self._dn_suh[sidx_x])
            y = inner_mm_seg_mlx(
                xd, self._dn_trellis, self._k, self._cb, tab, nbr,
                n_rows=N, tn_base=0, tiles_per_e=self._dn_tiles, out_e=D,
            )
        else:
            xd = _rows_prep()(h, self._dn_suh[sidx])
            wd = decode_full_eg_mlx(
                self._dn_trellis, self._k, self._cb, tiles_per_e=self._dn_tiles
            )
            y = mx.gather_mm(
                xd.reshape(N, 1, 1, H), wd,
                rhs_indices=idx, sorted_indices=True,
            )
        y = _rows_finish()(y.reshape(N, D).astype(mx.float16), self._dn_svh[sidx])
        return y[inv].reshape(B, S, kk, D)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        B, S, kk = indices.shape
        R = B * S
        if R == 1:
            fn = self._decode if _MOE_UNFUSED else self._decode_fused
            y = fn(
                x.reshape(1, self.input_dims), indices.reshape(-1).astype(mx.int32)
            )
            return y.reshape(B, S, kk, self.input_dims)
        if R <= 8 and not _MOE_UNFUSED and self._v2_ok():
            # spec verify / small-batch serving: decode-class A2/B2 over
            # R*kk slots instead of the _prefill sort/table machinery
            y = self._decode_fused2(
                x.reshape(R, self.input_dims),
                indices.reshape(R, kk).astype(mx.int32),
            )
            return y.reshape(B, S, kk, self.input_dims)
        return self._prefill(x, indices)


class EXL3MoEBlock(nn.Module):
    """Replaces Qwen3NextSparseMoeBlock: the shared expert is stacked as the
    LAST expert (its gate value ``sigmoid(shared_expert_gate(x))`` is just
    another routing weight), so the whole block is router math + the two
    mapped expert GEMV chains + one weighted sum — instead of a separate
    3-projection shared-expert chain and ~6 extra dispatches per layer."""

    def __init__(self, gate: nn.Linear, shared_gate: nn.Linear, switch: EXL3SwitchGLU,
                 top_k: int, norm_topk_prob: bool):
        super().__init__()
        self.gate = gate
        self.shared_expert_gate = shared_gate
        self.switch_mlp = switch
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self._shared_idx = switch.num_experts - 1

    def _router(self):
        if not hasattr(self, "_router_fn"):
            k = self.top_k
            norm = self.norm_topk_prob
            shared_idx = self._shared_idx

            @mx.compile
            def _fn(
                x: mx.array, gate_w: mx.array, sg_w: mx.array
            ) -> tuple[mx.array, mx.array]:
                gates = mx.softmax(x @ gate_w.T, axis=-1, precise=True)
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                if norm:
                    scores = scores / scores.sum(axis=-1, keepdims=True)
                shared_w = mx.sigmoid(x @ sg_w.T)
                inds = mx.concatenate(
                    [inds, mx.full(inds.shape[:-1] + (1,), shared_idx, dtype=inds.dtype)],
                    axis=-1,
                )
                scores = mx.concatenate([scores.astype(x.dtype), shared_w], axis=-1)
                return inds, scores

            self._router_fn = _fn
        return self._router_fn

    def __call__(self, x: mx.array) -> mx.array:
        # one compiled graph for the whole routing decision (was ~8 dispatches)
        inds, scores = self._router()(
            x, self.gate.weight, self.shared_expert_gate.weight
        )
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


class EXL3Gemma4MoEBlock(nn.Module):
    """Replaces Gemma4 ``Router`` + ``Experts`` with one compiled routed path.

  Mounted on ``layer.router``; ``_Gemma4ExpertsShim`` forwards the expert
  call so ``DecoderLayer`` is unchanged and router weights are not duplicated
  in the parameter tree.
    """

    def __init__(
        self,
        proj: nn.Linear,
        scale: mx.array,
        per_expert_scale: mx.array,
        switch: EXL3SwitchGLU,
        top_k: int,
        eps: float,
        hidden_size: int,
    ):
        super().__init__()
        self.proj = proj
        self.scale = scale
        self.per_expert_scale = per_expert_scale
        self.switch_mlp = switch
        self.top_k = top_k
        self.eps = eps
        self._root_scale = hidden_size**-0.5

    def _route(self, h: mx.array) -> tuple[mx.array, mx.array]:
        normed = mx.fast.rms_norm(h, self.scale * self._root_scale, self.eps)
        expert_scores = self.proj(normed)
        inds = mx.argpartition(expert_scores, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        weights = mx.take_along_axis(expert_scores, inds, axis=-1)
        weights = mx.softmax(weights, axis=-1)
        weights = weights * self.per_expert_scale[inds]
        return inds, weights

    def _router(self):
        if not hasattr(self, "_router_fn"):

            @mx.compile
            def _fn(
                h: mx.array,
                scale: mx.array,
                root_scale: float,
                eps: float,
                proj_w: mx.array,
                per_expert_scale: mx.array,
                k: int,
            ) -> tuple[mx.array, mx.array]:
                normed = mx.fast.rms_norm(h, scale * root_scale, eps)
                expert_scores = normed @ proj_w.T
                inds = mx.argpartition(expert_scores, kth=-k, axis=-1)[..., -k:]
                weights = mx.take_along_axis(expert_scores, inds, axis=-1)
                weights = mx.softmax(weights, axis=-1)
                weights = weights * per_expert_scale[inds]
                return inds, weights

            self._router_fn = _fn
        return self._router_fn

    def __call__(
        self,
        x: mx.array,
        top_k_indices: mx.array | None = None,
        top_k_weights: mx.array | None = None,
    ) -> mx.array | tuple[mx.array, mx.array]:
        if top_k_indices is None:
            use_compile = os.environ.get("EXL3_GEMMA4_ROUTER_COMPILE", "0") == "1"
            if use_compile:
                return self._router()(
                    x,
                    self.scale,
                    self._root_scale,
                    self.eps,
                    self.proj.weight,
                    self.per_expert_scale,
                    self.top_k,
                )
            return self._route(x)
        y = self.switch_mlp(x, top_k_indices)
        return (top_k_weights[..., None] * y).sum(axis=-2)


class _Gemma4ExpertsShim(nn.Module):
    """DecoderLayer ``experts`` slot without duplicating router params in the tree."""

    def __init__(self, block: EXL3Gemma4MoEBlock):
        super().__init__()
        self._block = block

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array:
        return self._block(x, top_k_indices, top_k_weights)
