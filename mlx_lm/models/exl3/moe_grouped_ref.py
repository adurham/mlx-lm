# NumPy reference for the EXL3 expert-grouped multi-row MoE decode (W4).
#
# Pure-NumPy reimplementation of BOTH slot->output paths of
# ``EXL3SwitchGLU._decode_fused2`` (the shipped R<=8 spec-verify path) and
# ``EXL3SwitchGLU._decode_grouped`` (the DSV41_MOE_GROUPED=1 expert-grouped
# variant), against the SAME stacked-trellis layout the Metal kernels read:
#
#   gu_trellis : (in_tiles, E*2*gu_tiles, packed) uint16  [gate | up] stacked
#   dn_trellis : (hid_tiles, E*dn_tiles, packed) uint16
#   gu_suh     : (E, 2, D) fp16      per-expert, per-proj input signs
#   gu_svh     : (E, 2*hidden) fp16  per-expert output signs [gate | up]
#   dn_suh     : (E, hidden) fp16    down input signs
#   dn_svh     : (E, D) fp16         down output signs
#
# DESIGN (deliberate): the trellis codeword EXTRACTION (which 16 bits of a
# packed tile form the codeword that feeds decode_3inst, and which weight
# slot it lands in) is INJECTED as the ``tile_weights`` callable. The
# vendored CPU build has no Metal backend, so the only runnable ground truth
# for that extraction is the Mac. The grouping/scatter proof below is a
# statement about INDEX ARITHMETIC — every slot's outputs are computed from
# the same per-expert weight values and land at the same addresses — so it
# holds for ANY fixed ``tile_weights``. Two implementations ship:
#
# * ``metal_sim_tile_weights``: a mechanical NumPy transliteration of the
#   Metal extraction loops (_tile_loop + _lane_setup in exl3_moe.py: same
#   integer ops, same window order, then the vendored kernel_order_to_row_major
#   perm). It is a transliteration of the kernel SOURCE, not verified against
#   Metal output on this box; the Mac microbench asserts the real kernels
#   instead (benchmarks/w4k_grouped_bench.py).
# * ``hash_tile_weights``: deterministic pseudo-random 16x16 fixed by the
#   tile bytes — proves the equality is extraction-agnostic.
#
# The non-decode math is anchored against real MLX ops where the CPU build
# can run them: ``_rows_prep`` here is bit-exact against the compiled
# ``mlx_lm.models.exl3.exl3_moe._rows_prep`` (mx.hadamard_transform path) on
# the CPU build — had128 reproduces its stage order.
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .ref.codebook import CodebookMode, decode_3inst
from .ref.perm import kernel_order_to_row_major, tensor_core_perm

_HS = np.float32(1.0 / np.sqrt(128.0))  # 0.08838834764831845, the Metal _HS


# --- trellis extraction (injected) --------------------------------------------


def metal_sim_tile_weights(
    words_u32: np.ndarray, k: int, cb: CodebookMode
) -> np.ndarray:
    """Mechanical transliteration of the Metal A2 tile loop's tile decode.

    Reproduces, integer op for integer op, what ONE threadgroup computes for
    ONE tile in ``_tile_loop`` + ``_lane_setup`` (exl3_moe.py): for each
    (lane, g): c0 = lane*8 + g*4; e_last = (c0+4)*K + 256*K; i_end =
    (e_last-1)/32; merged = words[(i_end-1)%PU]<<32 | words[i_end%PU];
    s = (i_end+1)*32-e_last; cws[q] = (merged >> (s + (3-q)*K)) & 0xFFFF;
    the decoded value lands at tile position pos_[jw] for jw = g*4+q, where
    pos_ comes from the vendored tensor-core perm exactly like _lane_setup's
    ``pos_[j] = perm[t*2 (+1)]`` walk. Finally kernel_order_to_row_major maps
    the tile's kernel-order values to row-major (16 in, 16 out) fp32, the
    promotion ``float(dq_val)`` the kernel applies.
    """
    pu = k * 256 // 32
    assert words_u32.shape == (pu,), (words_u32.shape, pu)
    perm = tensor_core_perm()
    # pos_[8] per lane, exactly the _lane_setup walk
    pos_ = np.zeros((32, 8), np.int64)
    for lane in range(32):
        for j in range(4):
            t = lane * 4 + j
            pos_[lane, j * 2] = perm[t * 2]
            pos_[lane, j * 2 + 1] = perm[t * 2 + 1]
    tile_vals = np.zeros(256, np.float32)
    for lane in range(32):
        for g in range(2):
            c0 = lane * 8 + g * 4
            e_last = (c0 + 4) * k + 256 * k
            i_end = (e_last - 1) // 32
            a = np.uint64(int(words_u32[(i_end - 1) % pu]))
            b = np.uint64(int(words_u32[i_end % pu]))
            merged = (a << np.uint64(32)) | b
            s = (i_end + 1) * 32 - e_last
            for q in range(4):
                jw = g * 4 + q
                cw = int((merged >> np.uint64(s + (3 - q) * k)) & 0xFFFF)
                tile_vals[pos_[lane, jw]] = np.float32(decode_3inst(cw, cb))
    return kernel_order_to_row_major(tile_vals).reshape(16, 16).astype(np.float32)


def hash_tile_weights(
    words_u32: np.ndarray, k: int, cb: CodebookMode
) -> np.ndarray:
    """Deterministic pseudo-random 16x16 fixed by the tile bytes.

    Proves the fused2-vs-grouped equality is independent of the codeword
    extraction: same bytes -> same weights, whatever the layout.
    """
    seed = int(np.frombuffer(words_u32.tobytes(), np.uint32).sum(dtype=np.uint64))
    rng = np.random.default_rng(seed)
    return rng.standard_normal((16, 16)).astype(np.float32)


# --- hadamard + rows -----------------------------------------------------------


def butterfly(tg: np.ndarray) -> None:
    """In-place Hadamard butterfly per 128-block, in the Metal stage order.

    Mirrors ``_BUTTERFLY`` in exl3_moe.py: stages s=0..6 (bit = 1<<s), pairs
    ``(i, i+bit)`` with ``i = b_*128 + (((w & ~(bit-1)) << 1) | (w & (bit-1)))``
    — fp32 add/sub, the same rounding point per stage.
    """
    n = tg.shape[0]
    assert n % 128 == 0, f"butterfly needs a multiple of 128, got {n}"
    for s_ in range(7):
        bit = 1 << s_
        for blk in range(n // 128):
            b = tg[blk * 128 : (blk + 1) * 128]
            for w in range(64):
                i = ((w & ~(bit - 1)) << 1) | (w & (bit - 1))
                a = b[i]
                c = b[i + bit]
                b[i] = a + c
                b[i + bit] = a - c


def had128(x: np.ndarray) -> np.ndarray:
    """mx.hadamard_transform over the last 128-axis (fp16 in, fp16 out).

    Matches the CPU op exactly (verified bit-exact in the tests): fp16
    multiply upstream, fp32 arithmetic WITHIN each butterfly stage, but a
    fp16 round-trip BETWEEN stages (that between-stage rounding is what
    separates it from kernel B2's all-fp32 ``_BUTTERFLY``), scale by _HS in
    fp32, fp16 store. This models the ``_rows_prep`` output the kernels
    consume — it feeds BOTH the A2 and A2G paths identically.
    """
    x = np.asarray(x)
    n = x.shape[-1]
    assert n % 128 == 0, n
    blocks = x.reshape(-1, 128).astype(np.float16)
    out = np.empty_like(blocks)
    for i in range(blocks.shape[0]):
        v = blocks[i].astype(np.float32).copy()
        for s_ in range(7):
            bit = 1 << s_
            nv = v.copy()
            for w in range(64):
                j = ((w & ~(bit - 1)) << 1) | (w & (bit - 1))
                a = v[j]
                c = v[j + bit]
                nv[j] = a + c
                nv[j + bit] = a - c
            v = nv.astype(np.float16).astype(np.float32)
        out[i] = (v * _HS).astype(np.float16)
    return out.reshape(x.shape)


def _rows_prep(x2d: np.ndarray, sel: np.ndarray, gu_suh: np.ndarray):
    """rows_prep on the ORIGINAL slot order: (E_sel*2, D) fp16, rows
    (slot*2 + proj). Identical in both paths.

    Slot s = row-major (r, j) of the (R, kk) index grid, so its x row is
    r = s // kk (exactly the broadcast x_rep/reshape build in
    _decode_fused2 / _decode_grouped).
    """
    R = x2d.shape[0]
    D = x2d.shape[1]
    E_sel = sel.shape[0]
    kk = E_sel // R
    xh = np.zeros((E_sel * 2, D), np.float16)
    for s in range(E_sel):
        e = int(sel[s])
        row = s // kk
        for proj in (0, 1):
            v = (x2d[row].astype(np.float16) * gu_suh[e, proj]).astype(np.float16)
            xh[s * 2 + proj] = had128(v)
    return xh


# --- gate/up + down ------------------------------------------------------------


def _tile_run(
    trellis_u16: np.ndarray,
    tn: int,
    k: int,
    cb: CodebookMode,
    tile_weights: Callable[[np.ndarray, int, CodebookMode], np.ndarray],
) -> np.ndarray:
    """Decode one tile-run's in-tiles -> (in_tiles, 16, 16) fp32."""
    in_tiles = trellis_u16.shape[0]
    src_tiles = trellis_u16.shape[1]
    pu = k * 256 // 32
    flat = trellis_u16.reshape(-1).view(np.uint32)
    w = np.zeros((in_tiles, 16, 16), np.float32)
    for tk in range(in_tiles):
        base = (tk * src_tiles + tn) * pu
        w[tk] = tile_weights(flat[base : base + pu], k, cb)
    return w


def _apply_run(w: np.ndarray, xrow: np.ndarray) -> np.ndarray:
    """Apply a decoded tile-run to one slot's xh row -> (16,) fp16.

    Same order as the Metal _tile_loop: per in-tile fp32 fma over the
    16-wide slice, summed across in-tiles; the fp16 handoff at the store.
    """
    in_tiles = w.shape[0]
    acc = np.zeros((16,), np.float32)
    for tk in range(in_tiles):
        acc += w[tk].T @ xrow[tk * 16 : tk * 16 + 16].astype(np.float32)
    return acc.astype(np.float16)


def _b2_slot(
    ygu_row: np.ndarray,
    e: int,
    gu_svh: np.ndarray,
    dn_trellis: np.ndarray,
    dn_suh: np.ndarray,
    dn_svh: np.ndarray,
    k: int,
    cb: CodebookMode,
    act: str,
    clamp: float,
    hidden: int,
    ch: int,
    D: int,
    tile_weights: Callable[[np.ndarray, int, CodebookMode], np.ndarray],
) -> np.ndarray:
    """Kernel B2 for ONE slot -> (D,) fp16 (chunked prologue + down loop)."""
    hid_tiles = hidden // 16
    dn_tiles = D // 16
    src_tiles = dn_trellis.shape[1]
    pu = k * 256 // 32
    flat = dn_trellis.reshape(-1).view(np.uint32)
    tg_xh = np.zeros((hidden,), np.float32)
    for c in range(0, hidden, ch):
        tg_gc = ygu_row[c : c + ch].astype(np.float32) * _HS
        tg_uc = ygu_row[hidden + c : hidden + c + ch].astype(np.float32) * _HS
        butterfly(tg_gc)
        butterfly(tg_uc)
        g = tg_gc * gu_svh[e, c : c + ch].astype(np.float32)
        u = tg_uc * gu_svh[e, hidden + c : hidden + c + ch].astype(np.float32)
        if act == "silu_clamp":
            g = np.minimum(g, np.float32(clamp))
            u = np.clip(u, np.float32(-clamp), np.float32(clamp))
            h = (g / (np.float32(1.0) + np.exp(-g))) * u
        elif act == "gelu":
            t = np.float32(0.797884560803) * (
                g + np.float32(0.044715) * g * g * g
            )
            h = np.float32(0.5) * g * (np.float32(1.0) + np.tanh(t)) * u
        else:
            h = (g / (np.float32(1.0) + np.exp(-g))) * u
        # the kernel stores h through a fp16 register (float(half(h)))
        h = h.astype(np.float16).astype(np.float32)
        tg_xh[c : c + ch] = (
            h * dn_suh[e, c : c + ch].astype(np.float32) * _HS
        )
    butterfly(tg_xh)
    y = np.zeros((D,), np.float16)
    for block in range(D // 128):
        tg_y = np.zeros((128,), np.float32)
        for ot in range(8):
            tn = e * dn_tiles + block * 8 + ot
            acc = np.zeros((16,), np.float32)
            for tk in range(hid_tiles):
                base = (tk * src_tiles + tn) * pu
                w = tile_weights(flat[base : base + pu], k, cb)
                acc += w.T @ tg_xh[tk * 16 : tk * 16 + 16]
            tg_y[ot * 16 : ot * 16 + 16] = acc
        butterfly(tg_y)
        for col in range(128):
            c = block * 128 + col
            y[c] = np.float16(
                tg_y[col] * _HS * np.float32(dn_svh[e, c])
            )
    return y


def _b2_all(
    ygu, sel, gu_svh, dn_trellis, dn_suh, dn_svh, k, cb, act, clamp, ch,
    H, D, tile_weights,
):
    """B2 over all slots (identical between the two paths).

    The kernel's B2 reads the gate|up pair of slot s as the contiguous rows
    ``ygu[2s]`` (gate) and ``ygu[2s+1]`` (up) — i.e. slot s's pair is the
    length-2H vector ``ygu[2s] ++ ygu[2s+1]``.
    """
    E_sel = sel.shape[0]
    H = ygu.shape[1]
    y = np.zeros((E_sel, D), np.float16)
    for s in range(E_sel):
        y[s] = _b2_slot(
            ygu[2 * s : 2 * s + 2].reshape(2 * H),
            int(sel[s]), gu_svh, dn_trellis, dn_suh, dn_svh,
            k, cb, act, clamp, H, ch, D, tile_weights,
        )
    return y


# --- grouped tables (host mirror of _grouped_srt + _grouped_tables_fn) --------


def grouped_tables(sel: np.ndarray, E: int, sm: int):
    """Host mirror of ``EXL3SwitchGLU._grouped_srt`` + ``_grouped_tables_fn``.

    Sort key = expert * (2*n) + slot (unique keys -> deterministic order:
    grouped by expert, ascending original slot id); pad keys sort at expert
    id E, after every real key. Returns (srt, slots, tabs, slist): expert id
    and ORIGINAL slot id per sorted position, the (2, E) uint32 [expert id,
    real count] table, and the (E, sm) uint32 per-expert ORIGINAL-slot lists,
    ascending.
    """
    sel = np.asarray(sel, np.int64).reshape(-1)
    n = sel.shape[0]
    stride = 2 * n
    key = sel.astype(np.uint64) * stride + np.arange(n, dtype=np.uint64)
    padk = np.full(n, E * stride, dtype=np.uint64)
    sk = np.sort(np.concatenate([key, padk]))
    srt = (sk // stride).astype(np.uint32)
    slots = (sk % stride).astype(np.uint32)
    tabs = np.zeros((2, E), np.uint32)
    slist = np.zeros((E, sm), np.uint32)
    for p in range(srt.shape[0]):
        e = int(srt[p])
        if e >= E:
            continue
        if tabs[1, e] < sm:
            slist[e, tabs[1, e]] = slots[p]
            tabs[1, e] += 1
    tabs[0] = np.arange(E)
    return srt, slots, tabs, slist


# --- the two slot->output paths ------------------------------------------------


def decode_fused2_ref(
    x2d, indices, gu_trellis, gu_suh, gu_svh, dn_trellis, dn_suh, dn_svh,
    k, cb, act="silu", clamp=10.0, ch=128,
    tile_weights=metal_sim_tile_weights,
):
    """Reference ``_decode_fused2`` (slot-major A2, then B2 per slot).

    Returns (y, ygu): (E_sel, D) fp16 outputs and the (E_sel*2, H) fp16 raw
    gate|up buffer. ygu is the intermediate the grouped path must reproduce
    bit-exactly.
    """
    R, kk = indices.shape
    E_sel = R * kk
    D = x2d.shape[1]
    H = gu_svh.shape[1] // 2
    E = gu_suh.shape[0]
    gu_tiles = H // 16
    sel = indices.reshape(-1).astype(np.int64)
    xh = _rows_prep(x2d, sel, gu_suh)
    ygu = np.zeros((E_sel * 2, H), np.float16)
    for s in range(E_sel):
        e = sel[s]
        for proj in (0, 1):
            tn0 = (E + e) * gu_tiles if proj else e * gu_tiles
            for t0 in range(gu_tiles):
                w = _tile_run(gu_trellis, tn0 + t0, k, cb, tile_weights)
                ygu[s * 2 + proj, t0 * 16 : t0 * 16 + 16] = _apply_run(
                    w, xh[s * 2 + proj]
                )
    return _b2_all(ygu, sel, gu_svh, dn_trellis, dn_suh, dn_svh, k, cb,
                   act, clamp, ch, H, D, tile_weights), ygu


def decode_grouped_ref(
    x2d, indices, gu_trellis, gu_suh, gu_svh, dn_trellis, dn_suh, dn_svh,
    k, cb, act="silu", clamp=10.0, ch=128, sm=8,
    tile_weights=metal_sim_tile_weights,
):
    """Reference ``_decode_grouped`` (expert-grouped A2G, then B2 per slot).

    Same xh (ORIGINAL slot order), same B2. The A2G walk: for each expert
    with cnt > 0, each tile-run is decoded ONCE and applied to every slot in
    its list (in ascending ORIGINAL order — the same per-slot arithmetic).
    Returns (y, ygu, srt, tabs, slist).
    """
    R, kk = indices.shape
    E_sel = R * kk
    D = x2d.shape[1]
    H = gu_svh.shape[1] // 2
    E = gu_suh.shape[0]
    gu_tiles = H // 16
    sel = indices.reshape(-1).astype(np.int64)
    xh = _rows_prep(x2d, sel, gu_suh)
    srt, srt_slots, tabs, slist = grouped_tables(sel, E, sm)
    ygu = np.zeros((E_sel * 2, H), np.float16)
    for e in range(E):
        cnt = int(tabs[1, e])
        if cnt == 0:
            continue
        for proj in (0, 1):
            tn0 = (E + e) * gu_tiles if proj else e * gu_tiles
            for t0 in range(gu_tiles):
                w = _tile_run(gu_trellis, tn0 + t0, k, cb, tile_weights)
                for j in range(cnt):
                    s = int(slist[e, j])
                    ygu[s * 2 + proj, t0 * 16 : t0 * 16 + 16] = _apply_run(
                        w, xh[s * 2 + proj]
                    )
    return (
        _b2_all(ygu, sel, gu_svh, dn_trellis, dn_suh, dn_svh, k, cb,
                act, clamp, ch, H, D, tile_weights),
        ygu,
        srt,
        tabs,
        slist,
    )