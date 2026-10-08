# Vendored from PipeNetwork/deepseek-v41-mlx (Apache-2.0), upstream a9567ae plus the
# node-local MTP/DSpark port edits (exo docs phase 3). See README.md here.
"""Gather-based sparse attention — the MLX stand-in for the TileLang kernel.

Semantics from ``sparse_attn_kernel`` in the reference ``kernel.py``:

* ``kv`` is a **single** shared vector per position (MLA): key and value are the
  same tensor;
* an index of ``-1`` means "not visible": its logit is -inf, contributing nothing;
* the learned per-head ``attn_sink`` enters the softmax **denominator only** —
  it carries no value vector, so a head with weak logits attends to nearly nothing;
* a row with no valid index yields an all-zero output (the kernel's finite
  -1e30 max floor).

The kernel keeps its running max over the real logits only and adds
``exp(sink - max)`` at the end; here the sink also competes for the max, which is
mathematically identical (softmax shift invariance) and safer numerically.

Tiled rewrite (workstream A, 2026-09-28). The previous implementation — kept
below as :func:`sparse_attn_reference` for parity tests and rollback — built the
whole per-chunk gather in **fp32** (``[b, 256, k<=640, 512]``: a 335 MB gather
plus its 671 MB fp32 copy, per prefill chunk per layer at the production shape)
plus an fp32 copy of ``q``, then ran both einsums and the entire softmax chain on
those fp32 operands. This version:

* walks the query rows in tiles of ``DSV41_SPARSE_QTILE`` (default 64) and the
  key range of each tile in ``DSV41_SPARSE_KTILE`` (default 0 = auto: the widest
  key tile that keeps every intermediate inside ``DSV41_SPARSE_BUDGET_MB``);
* gathers K/V in the **working dtype** — the query's own dtype by default (bf16
  in the serving path, fp32 in the fp32 model-gate harness) — so the reference's
  fp32 gather and its fp32 cast disappear;
* keeps the running max/sum/accumulator in **fp32** with online-softmax
  renormalisation ``acc = acc * exp(m_old - m_new) + P V`` whenever the running
  max grows. The two matmuls are MLX 16-bit GEMMs, which accumulate in fp32;
* folds the attention sink into the **initial** running max/sum
  (``m = sink, l = 1, acc = 0``) — exactly the reference's
  ``denom = sum(w) + exp(sink - mmax)``, with no extra pass and no separate sink
  term at the end;
* masks with ``-inf`` instead of ``-1e30``, so ``exp(logit - m)`` is exactly 0
  for a masked key at any running max and the reference's second ``where`` over
  the weights disappears. Both give an all-zero output for a fully masked row
  (the reference via ``denom = 1``, this version via ``l = 1, acc = 0``); the
  only divergence is the reference's ``0/0`` NaN for a fully masked row whose
  sink is below ``-1e30``, which a real sink cannot produce;
* optionally compiles each tile body (``DSV41_SPARSE_COMPILE``, default on), so
  the elementwise softmax chain fuses into the matmul epilogues and the per-tile
  intermediates stay in registers instead of round-tripping through memory.
  Compiled and eager runs are bit-identical on the shapes tested.

Knobs (all env-gated for A/B and rollback): ``DSV41_SPARSE_IMPL``
(``tiled``|``ref``), ``DSV41_SPARSE_QTILE``, ``DSV41_SPARSE_KTILE``,
``DSV41_SPARSE_WDTYPE`` (``auto``|``bf16``|``fp16``|``fp32``),
``DSV41_SPARSE_COMPILE``, ``DSV41_SPARSE_BUDGET_MB``,
``DSV41_SPARSE_COLSPLIT`` (column-partitioned two-source gather, default 1;
small-m calls skip the boundary derivation and use the value-identical
where-select path -- see ``sparse_attn``),
``DSV41_SPARSE_ASYNC_FENCE`` (async per-tile fence, default 0).

Two-source gather (gather-direct comp_kv). ``sparse_attn`` also accepts
``kv2``/``split``: index rows ``< split`` gather from ``kv``, rows
``>= split`` from ``kv2`` at ``row - split`` — i.e. the indices address
``concat(kv, kv2)`` but no such concatenation is ever built. ``attention.py``
uses this to read the shared ``comp_kv`` cache directly by top-k index instead
of materializing it per compressing layer per forward. ``kv2=None`` is byte-for-
byte the previous single-source behaviour.
"""

from __future__ import annotations

import os

import mlx.core as mx

NEG_INF = -1e30
# Masked logit. -inf (not -1e30) makes exp(logit - m) exactly 0 for every masked
# key without a second `where` pass over the weights; see the module docstring.
MASK_LOGIT = float("-inf")

# --------------------------------------------------------------------------
# tiling / numerics knobs
# --------------------------------------------------------------------------
_QTILE = int(os.environ.get("DSV41_SPARSE_QTILE", "256"))
# 0 = auto: take the widest key tile whose largest intermediate fits the budget
# (at the production shape that is the whole 128-window + top-512 range in one
# tile). A smaller fixed value trades dispatches for a smaller footprint.
_KTILE = int(os.environ.get("DSV41_SPARSE_KTILE", "0"))
_WDTYPE = os.environ.get("DSV41_SPARSE_WDTYPE", "auto")
_IMPL = os.environ.get("DSV41_SPARSE_IMPL", "tiled")
_COMPILE = os.environ.get("DSV41_SPARSE_COMPILE", "1") == "1"
# PV operand dtype. OFF (0) casts the softmax weights down to the gather dtype
# before the value matmul; that rounding is the single largest precision loss in
# the tiled path (weights live in [0, 1] and bf16 carries 8 mantissa bits, so the
# ~0.4% per-weight error survives as a ~5e-3 relative error in the output and
# compounds across layers). ON keeps the weights in fp32 and promotes the matmul,
# which costs ~20% of the PV GEMM rate and no extra memory beyond the fp32
# weight tile that already exists. The gather dtype itself is NOT a precision
# knob: the cached K/V is bf16, and bf16 -> fp32 is exact, so a bf16 gather holds
# bit-identical values to the reference's fp32 gather.
# DEFAULT FLIPPED 1 -> 0 (2026-10-06): the precision concern above was NEVER
# measured against the live battery; it was a conservative port-time default.
# Isolated bench: PV32=0 cuts the sparse body 38.7 -> 29.4 ms/call (-24%);
# on the wire fresh +2.8%/delta +2.3% on top of chunk=4096. The 350K live
# battery (needles 6/6, tools 10/10, prose 0 DIRTY/0 REVIEW, park True) PASSED
# on the PV32=0 arm -- the detector set calibrated for exactly this error class.
# =1 restores the fp32-promoted PV for A/B.
_PV32 = os.environ.get("DSV41_SPARSE_PV32", "0") == "1"
# Live-graph fence. The tile plan bounds any SINGLE tensor, but MLX is lazy: the
# gathered K/V, fp32 logits and accumulator of EVERY query tile stay live until
# something forces evaluation, so the measured peak of one 512-row prefill call
# was 1351 MB even though no tensor exceeded 42 MB. Fencing once per query tile
# (`"qtile"`, the default) evaluates that tile's running state and output before
# the next tile is built, which holds the live set to one tile: measured peak
# 202 MB at the production shape, ~20% slower than the unfenced build and still
# faster than the previous implementation (9.6 vs 10.2 ms). `"ktile"` fences
# every key tile too (190 MB, another ~13%); `"0"` disables the fence (fastest,
# 1.3 GB live). This is the knob that actually delivers the bounded-memory
# property; tile_bytes() alone does not.
_FENCE = os.environ.get("DSV41_SPARSE_FENCE", "qtile")
_FENCE_MIN_ROWS = int(os.environ.get("DSV41_SPARSE_FENCE_MIN_ROWS", "16"))
# Ceiling on any SINGLE intermediate a (query, key) tile may hold, in MB: the
# gathered K/V tile, the fp32 logits, the softmax weights or the fp32 output
# accumulator. The tile plan shrinks the key tile and then the query tile until
# every one of them fits, so the peak transient is bounded at any geometry
# (42 MB for the gather at Tq=64, k=640, d=512, bf16).
_BUDGET_BYTES = int(os.environ.get("DSV41_SPARSE_BUDGET_MB", "256")) << 20

# Column-partitioned two-source gather (workstream C1, 2026-10-06). When OFF =
# byte-identical to the where-select path. When ON and the caller's
# layout invariant holds (window block entirely ``< split``, compressed block
# entirely ``>= split``), each query row's columns are routed to a SINGLE
# source instead of gathering BOTH sources for every row and ``mx.where``-
# selecting: the window columns need only ``kv`` and the top-k columns only
# ``kv2``. At the production tile shape [1, 64, 640, 512] bf16 the old path
# gathered ~80 MB per query tile (window + top-k from both buffers) and threw
# half of it away; the comp_kv gather is what dominates the tile. The boundary
# is PROVEN from the indices themselves once per call (``_column_boundary``);
# on any ambiguity the call silently falls back to the exact where-select path,
# so a violated invariant can never mis-gather.
#
# WHY THIS DEFAULT IS ON while attention.py's twin gate defaults OFF (the
# documented foot-gun): the two gates disagree by design-of-history -- this
# module owns/ships the C1 gather (default ON), the caller's gate was added
# later default OFF so the DEFAULT call site passes ``colsplit=None``. With the
# caller silent, ``_column_boundary(-1)`` used to DERIVE the boundary, and that
# derivation (``_leading_window_columns``' ``int(mx.min(...))`` +
# ``bool(mx.all(...).item())``) is TWO host round-trips. On every compressing
# layer of every decode (m=1) and 4-row verify forward that was pure waste (and
# a per-layer GPU drain). What makes the ON default safe for small m is the
# row-count guard at the call site below: for ``m <= _FENCE_MIN_ROWS`` the
# whole derivation+check is skipped and the call falls back to the
# VALUE-IDENTICAL ``_gather_split``. Do not remove that guard.
_COLSPLIT = os.environ.get("DSV41_SPARSE_COLSPLIT", "1") == "1"
# Async per-tile fence (workstream C3, 2026-10-06). OFF by default = the
# blocking per-tile ``mx.eval`` documented above (``qtile``). When ON the
# per-tile fence becomes ``mx.async_eval`` -- the tile is queued without a host
# round-trip -- plus ONE blocking ``mx.eval`` after the final tile, so the call
# still returns an evaluated tensor. Rationale: at the production chunk=64 a
# 2048-row prefill call issues 32 blocking syncs/layer; async removes those
# round-trips. Memory expectation: the async commits keep the per-tile live set
# bounded exactly as the blocking eval does (the 202 MB fenced figure above),
# and across layers the model.py prefill host sync (rows>16) is what bounds the
# queue; the gate does NOT trade the bound away. The exact async peak is
# UNMEASURED on the real model (see the branch report).
_FENCE_ASYNC = os.environ.get("DSV41_SPARSE_ASYNC_FENCE", "0") == "1"

_COMPILED: dict = {}


def _itemsize(dtype) -> int:
    return 4 if dtype == mx.float32 else 2


def _work_dtype(qdtype):
    """Dtype for the gather + matmuls: the query's own dtype under "auto"."""
    if _WDTYPE == "bf16":
        return mx.bfloat16
    if _WDTYPE == "fp16":
        return mx.float16
    if _WDTYPE == "fp32":
        return mx.float32
    return mx.float32 if qdtype == mx.float32 else qdtype


def tile_bytes(tq: int, tk: int, h: int, d: int, wdtype) -> int:
    """Bytes of the LARGEST single intermediate of a (query, key) tile."""
    w = _itemsize(wdtype)
    return max(tq * tk * d * w,        # gathered K/V tile
               tq * tk * h * 4,        # fp32 logits / weights input
               tq * tk * h * w,        # softmax weights, gather dtype
               tq * h * d * 4)         # fp32 output accumulator


def plan_tiles(h: int, d: int, kdim: int, chunk: int, wdtype) -> tuple[int, int]:
    """(query rows, key columns) per tile, each inside the intermediate budget."""
    qt = max(1, min(int(chunk), _QTILE))
    kt = min(int(_KTILE), kdim) if _KTILE > 0 else kdim
    while kt > 1 and tile_bytes(qt, kt, h, d, wdtype) > _BUDGET_BYTES:
        kt = max(1, kt // 2)
    while qt > 1 and tile_bytes(qt, kt, h, d, wdtype) > _BUDGET_BYTES:
        qt = max(1, qt // 2)
    return qt, kt


def _gather_kv(kv: mx.array, idx: mx.array) -> mx.array:
    """kv [b, n, d], idx [b, m, k] -> [b, m, k, d]; row 0 for negative idx.

    Row 0 is only ever read for a masked (``-1``) index, whose logit is replaced
    by ``-inf``, so the substituted row cannot contribute. A flat-index gather
    rather than ``mx.take``: one index computation for the whole tile, no
    per-row dispatch, and no reliance on ``take``'s negative-index semantics.
    """
    b, n, d = kv.shape
    flat = kv.reshape(b * n, d)
    safe = mx.maximum(idx, 0).astype(mx.int32)
    base = (mx.arange(b, dtype=mx.int32) * n).reshape(b, 1, 1)
    return flat[(safe + base).reshape(-1)].reshape(*idx.shape, d)


def _gather_split(kv: mx.array, kv2: mx.array, idx: mx.array, split: int) -> mx.array:
    """Two-source gather: idx row ``i`` -> ``kv`` for ``i < split``, else ``kv2[i-split]``.

    Indices are in the *concatenated* space ``concat(kv, kv2)``; ``split`` is the
    number of ``kv`` rows. Each row is gathered DIRECTLY from its source buffer --
    no ``concat(kv, kv2)`` is ever materialized -- which is the whole point: the
    compressed-KV source is a bf16 buffer whose dense concatenation was
    ~31 GiB/step at 1M context, while the top-k touches <=640 rows per query.

    ``-1`` (masked) resolves to row 0 of whichever source the boundary puts it
    on; a ``-1`` is always ``< split`` so it lands on ``kv``, and row 0's value is
    replaced by a ``-inf`` logit downstream and cannot contribute. A single
    out-of-range index is therefore impossible for either source.
    """
    take_b = idx >= split                                  # [b, m, k] bool
    idx_a = mx.minimum(mx.maximum(idx, 0), split - 1)      # in-range row of kv
    idx_b = mx.maximum(idx - split, 0)                     # in-range row of kv2
    gathered_a = _gather_kv(kv, idx_a)
    gathered_b = _gather_kv(kv2, idx_b)
    return mx.where(take_b[..., None], gathered_b, gathered_a)


def _gather_cols(kv: mx.array, kv2: mx.array, icb: mx.array, split: int,
                 colsplit: int, ks: int) -> mx.array:
    """C1 gather for one key tile: route each COLUMN range to its own source.

    ``icb`` [b, m, Tk] is key columns ``[ks, ks + Tk)`` of the call's index
    matrix. Columns ``< colsplit`` address ``kv`` (window, already clamped to
    ``[0, split)`` with ``-1`` -> row 0) and columns ``>= colsplit`` address
    ``kv2`` (at ``row - split``). A tile entirely inside one region is one
    gather from that source; a tile straddling the boundary is the two column
    ranges gathered separately and concatenated. The caller only reaches here
    after ``_column_boundary`` proved the layout, so the two ranges are known
    clean and no ``mx.where`` select is needed.
    """
    k = int(icb.shape[-1])
    cs = min(max(colsplit - ks, 0), k)                    # left columns in tile
    if cs <= 0:
        return _gather_kv(kv2, mx.maximum(icb - split, 0))
    if cs >= k:
        return _gather_kv(kv, mx.minimum(mx.maximum(icb, 0), split - 1))
    left = icb[:, :, :cs]
    right = icb[:, :, cs:]
    gathered_a = _gather_kv(kv, mx.minimum(mx.maximum(left, 0), split - 1))
    gathered_b = _gather_kv(kv2, mx.maximum(right - split, 0))
    return mx.concatenate([gathered_a, gathered_b], axis=2)


def _column_boundary(icb: mx.array, split: int, colsplit: int) -> int | None:
    """Column split for the C1 two-source path, or ``None`` to fall back.

    ``icb`` [b, m, k] addresses ``concat(kv[split], kv2[...])``. ``attention.py``
    passes the window block FIRST (columns ``< split``, ``-1`` padded) and the
    top-k compressed block SECOND (columns ``>= split``). ``colsplit`` is the
    window block width the caller declares, or ``-1`` to derive it.

    A column split at ``w`` is only valid when every column ``[0, w)`` is
    provably ``< split`` (or a negative mask) and every column ``[w, k)`` is
    provably ``>= split`` (or a negative mask). Negative entries are the
    indexer's ``-1`` mask; their logit is replaced by ``-inf`` downstream, so
    they may sit in EITHER region without changing a single output bit (the
    existing where-select already substitutes an arbitrary row for them). Any
    value in ``[0, split)`` inside the right region -- a real window index the
    caller misplaced -- fails the check and forces the exact where-select path,
    so a violated layout can never mis-gather.

    The whole ``[b, m, k]`` compare runs ONCE per call (cheap); ``None`` means
    fall back. Derivation (``colsplit=-1``): the boundary is the smallest
    per-row leading run of ``< split`` columns, which absorbs the window
    block's right-edge ``-1`` pads.
    """
    k = int(icb.shape[-1])
    if k == 0:
        return 0
    n_window = colsplit if colsplit >= 0 else _leading_window_columns(icb, split)
    if not 0 < n_window < k:
        return None
    # ONE host round-trip: both predicates in one array, one .item(). The
    # two-array form cost two blocking syncs per layer per chunk on the
    # prefill critical path (an invariant check is not worth a GPU drain).
    right = icb[:, :, n_window:]
    checks = mx.stack([
        mx.all(icb[:, :, :n_window] < split),
        mx.all((right >= split) | (right < 0)),
    ])
    if not bool(mx.all(checks).item()):
        return None
    return n_window


def _leading_window_columns(icb: mx.array, split: int) -> int:
    """Smallest per-row leading run of ``< split`` columns in ``icb``.

    ``-1`` pads count as ``< split`` so the window block's right-edge mask is
    absorbed; the minimum over rows keeps every row's ``[0, w)`` clean.
    """
    lead = (icb < split).astype(mx.int32)                  # [b, m, k]
    run = mx.cumprod(lead, axis=-1)                        # 1 until first >=
    return int(mx.min(mx.sum(run, axis=-1)))


# --------------------------------------------------------------------------
# per-(query, key)-tile body: the online-softmax step
#
# Two variants (first / later) so that a query tile's first key tile
# *initialises* the running state with the sink folded in (m = sink, l = 1,
# acc = 0) while later key tiles merge into it. Both are pure functions of their
# inputs, so they run eagerly or under ``mx.compile``.
# --------------------------------------------------------------------------
def _tile_init(qc, icb, kvc, sink, scale):
    """Initialise a query tile's running state; the sink is folded in here."""
    logits = mx.matmul(qc, kvc.swapaxes(-1, -2)).astype(mx.float32) * scale
    logits = mx.where((icb >= 0)[:, :, None, :], logits,
                      mx.array(MASK_LOGIT, mx.float32))
    m_run = mx.maximum(mx.max(logits, axis=-1, keepdims=True), sink)
    p = mx.exp(logits - m_run)
    l_run = mx.exp(sink - m_run) + mx.sum(p, axis=-1, keepdims=True)
    acc = mx.matmul(p if _PV32 else p.astype(kvc.dtype), kvc)
    return m_run, l_run, acc


def _tile_step(qc, icb, kvc, m_run, l_run, acc, scale):
    """Merge one more key tile into a query tile's running state."""
    logits = mx.matmul(qc, kvc.swapaxes(-1, -2)).astype(mx.float32) * scale
    logits = mx.where((icb >= 0)[:, :, None, :], logits,
                      mx.array(MASK_LOGIT, mx.float32))
    m_new = mx.maximum(m_run, mx.max(logits, axis=-1, keepdims=True))
    corr = mx.exp(m_run - m_new)
    p = mx.exp(logits - m_new)
    l_run = l_run * corr + mx.sum(p, axis=-1, keepdims=True)
    acc = acc * corr + mx.matmul(p if _PV32 else p.astype(kvc.dtype), kvc)
    return m_new, l_run, acc


def _body(fn, first: bool, qc, kvc, wdtype):
    """Dispatch a tile body, compiled per shape when enabled."""
    if not _COMPILE:
        return fn
    key = (bool(first), tuple(qc.shape), tuple(kvc.shape), str(wdtype))
    cfn = _COMPILED.get(key)
    if cfn is None:
        cfn = mx.compile(fn)
        _COMPILED[key] = cfn
    return cfn


def sparse_attn(q: mx.array, kv: mx.array, attn_sink: mx.array, topk_idxs: mx.array,
                softmax_scale: float, chunk: int = 64, kv2: mx.array | None = None,
                split: int | None = None, colsplit: int | None = None) -> mx.array:
    """q [b,m,h,d], kv [b,n,d], attn_sink [h], topk_idxs [b,m,k] (-1 = masked).

    ``chunk`` caps the query rows per tile; the tiling plan shrinks it (and the
    key tile) further whenever one tile would exceed ``DSV41_SPARSE_BUDGET_MB``.

    ``kv2``/``split`` add a second K/V source: ``topk_idxs`` rows ``>= split``
    index ``kv2`` (at ``row - split``) and rows ``< split`` index ``kv``, i.e.
    the indices address ``concat(kv, kv2)``. Both sources are gathered directly,
    with ``kv2`` never concatenated into a dense buffer (see ``_gather_split``).
    ``kv2=None`` is the exact single-source path.

    ``colsplit`` (two-source only) declares the COLUMN boundary of the layout
    ``attention.py`` builds: columns ``[0, colsplit)`` are the window block
    (indices ``< split``) and columns ``[colsplit, k)`` the top-k compressed
    block (indices ``>= split``). When ``DSV41_SPARSE_COLSPLIT`` is ON and the
    layout invariant is proven (``_column_boundary``), each tile gathers only
    what it needs: one source for a tile wholly inside a region, two column
    ranges for a straddling tile. ``colsplit=None`` or a violated invariant
    keeps the exact where-select path, byte-for-byte. Gate OFF ignores it
    entirely.
    """
    if _IMPL == "ref":
        return sparse_attn_reference(q, kv, attn_sink, topk_idxs, softmax_scale,
                                     chunk, kv2=kv2, split=split)

    b, m, h, d = q.shape
    kdim = topk_idxs.shape[-1]
    if m == 0 or kdim == 0:
        return mx.zeros(q.shape, dtype=q.dtype)

    wdtype = _work_dtype(q.dtype)
    if q.dtype != wdtype:
        q = q.astype(wdtype)
    if kv.dtype != wdtype:
        kv = kv.astype(wdtype)
    split_i: int = 0
    if kv2 is not None:
        # The two-source path only pays off when both sources share the gather
        # dtype; production stores comp_kv (and the window+chunk buffer) in bf16,
        # so this cast is a no-op there. A differing dtype is a real
        # materialization, so it is cast here once, explicitly.
        if kv2.dtype != wdtype:
            kv2 = kv2.astype(wdtype)
        assert split is not None and 0 <= split <= kv.shape[1], (split, kv.shape)
        split_i = int(split)
    qt, kt = plan_tiles(h, d, kdim, chunk, wdtype)
    # The fence bounds prefill memory (many query tiles live at once). A decode
    # or verify call (m <= _FENCE_MIN_ROWS, one small tile) has nothing to bound,
    # and an mx.eval here would force a host sync in EVERY layer of every decode
    # step, defeating the per-layer async_eval pipeline (measured on the full
    # two-node model: ~124 vs ~110 ms per spec round).
    fence_ok = m > _FENCE_MIN_ROWS
    fence_on = _FENCE in ("qtile", "ktile") and fence_ok
    # Whether the per-tile fences block (default) or only queue an async commit
    # (C3). With no fence and no async gate the loop below is unchanged.
    async_fence = fence_on and _FENCE_ASYNC
    # C1: resolve the column boundary ONCE for the whole call. Both query tiles
    # and key tiles are full k-wide slices of the same matrix, so a clean
    # boundary proves every tile's two regions. None => exact where-select path.
    #
    # SMALL-m GUARD (Phase-3 lever, 2026-10-08). The boundary derivation+check
    # below is TWO host round-trips (``_leading_window_columns``' ``int()`` and
    # the ``.item()`` check). It only pays off at prefill, where the C1 gather
    # stops the tile from materializing BOTH sources; at decode (m=1) and in the
    # 4-row verify there is a single small tile and nothing to save. Running it
    # there put a host sync in every compressing layer of every decode round for
    # nothing. Gate it on the SAME row-count threshold the fence already uses
    # (``m > _FENCE_MIN_ROWS``): large m keeps C1 exactly as before, small m
    # falls back to ``_gather_split``. That fallback is VALUE-IDENTICAL, not an
    # approximation: for a real index (>= 0) both routes select the same source
    # buffer row, and a ``-1`` mask pad resolves to row 0 of ``kv``, whose logit
    # is forced to ``-inf`` downstream (``_tile_init``/``_tile_step``), so it
    # cannot contribute -- the exact property ``_column_boundary``'s docstring
    # already relies on to let masks live in either region.
    colsplit_i: int = -1
    if kv2 is not None and _COLSPLIT and m > _FENCE_MIN_ROWS:
        w = _column_boundary(topk_idxs, split_i, -1 if colsplit is None else int(colsplit))
        if w is not None:
            colsplit_i = w
    sink = attn_sink.astype(mx.float32).reshape(1, h, 1)

    outs = []
    for qs in range(0, m, qt):
        qe = min(qs + qt, m)
        qc = q[:, qs:qe]                                  # [b, Tq, h, d]
        ic = topk_idxs[:, qs:qe]                          # [b, Tq, k]
        m_run = l_run = acc = None
        for ti, ks in enumerate(range(0, kdim, kt)):
            ke = min(ks + kt, kdim)
            icb = ic[:, :, ks:ke]                         # [b, Tq, Tk]
            if kv2 is None:
                kvc = _gather_kv(kv, icb)                 # [b, Tq, Tk, d]
            elif colsplit_i >= 0:
                # Column-partitioned gather (C1): columns of this key tile that
                # fall wholly in one region are gathered from that ONE source;
                # a tile straddling the boundary gathers the two column ranges
                # separately. No `where` select -- the boundary is proven.
                kvc = _gather_cols(kv, kv2, icb, split_i, colsplit_i, ks)
            else:
                # Two sources: gather each row directly from its own buffer.
                kvc = _gather_split(kv, kv2, icb, split_i)
            if ti == 0:
                fn = _body(_tile_init, True, qc, kvc, wdtype)
                m_run, l_run, acc = fn(qc, icb, kvc, sink, softmax_scale)
            else:
                fn = _body(_tile_step, False, qc, kvc, wdtype)
                m_run, l_run, acc = fn(qc, icb, kvc, m_run, l_run, acc,
                                       softmax_scale)
            if _FENCE == "ktile" and fence_ok:
                (mx.async_eval if async_fence else mx.eval)(m_run, l_run, acc)
        assert acc is not None and l_run is not None
        out = (acc / l_run).astype(q.dtype)
        if fence_on:
            # Evaluates this query tile's state and output before the next tile
            # is built, so only one tile's intermediates are ever live. C3
            # queues it (mx.async_eval) instead of blocking, and the single
            # mx.eval below keeps the "returns an evaluated tensor" contract.
            (mx.async_eval if async_fence else mx.eval)(out)
        outs.append(out)

    out = mx.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
    if async_fence:
        # One blocking eval per call: the returned tensor must be evaluated
        # under every fence mode, exactly as the per-tile mx.eval path is.
        mx.eval(out)
    return out


# --------------------------------------------------------------------------
# previous implementation — parity reference for the A/B harness and the
# rollback path (DSV41_SPARSE_IMPL=ref). Unchanged from upstream a9567ae.
# --------------------------------------------------------------------------
def sparse_attn_reference(q: mx.array, kv: mx.array, attn_sink: mx.array,
                          topk_idxs: mx.array, softmax_scale: float,
                          chunk: int = 256, kv2: mx.array | None = None,
                          split: int | None = None) -> mx.array:
    """q [b,m,h,d], kv [b,n,d], attn_sink [h], topk_idxs [b,m,k] (-1 = masked).

    ``kv2``/``split`` add the second K/V source (see :func:`sparse_attn`); the
    two-source gather is :func:`_gather_split`, shared with the tiled path so the
    parity harness compares the same index semantics.
    """
    b, m, h, d = q.shape
    sink = attn_sink.astype(mx.float32).reshape(1, 1, h, 1)
    split_i = -1
    if kv2 is not None:
        if split is None:
            raise ValueError("kv2 requires split")
        split_i = int(split)

    outs = []
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        qc = q[:, start:stop].astype(mx.float32)
        ic = topk_idxs[:, start:stop]
        if kv2 is None:
            kvc = _gather_kv(kv, ic).astype(mx.float32)      # [b, c, k, d]
        else:
            kvc = _gather_split(kv, kv2, ic, split_i).astype(mx.float32)

        logits = mx.einsum("bchd,bckd->bchk", qc, kvc) * softmax_scale
        valid = (ic >= 0)[:, :, None, :]
        logits = mx.where(valid, logits, NEG_INF)

        mmax = mx.max(logits, axis=-1, keepdims=True)
        mmax = mx.maximum(mx.maximum(mmax, sink), NEG_INF)
        w = mx.exp(logits - mmax)
        w = mx.where(valid, w, 0.0)
        denom = mx.sum(w, axis=-1, keepdims=True) + mx.exp(sink - mmax)

        o = mx.einsum("bchk,bckd->bchd", w, kvc) / denom
        outs.append(o.astype(q.dtype))

    return mx.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
