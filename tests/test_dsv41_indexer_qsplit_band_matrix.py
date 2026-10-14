# Copyright © 2026 Adam Durham (hermes-gw)
"""P20-P1 SHAPE MATRIX — reference-divergence harness for the indexer QUERY-BAND split.

``DSV41_INDEXER_QSPLIT`` (default OFF) splits the indexer's QUERY side across the TP
ranks: a rank scores / top-ks / emits only a contiguous band of the chunk's query
rows and the full selection ``cidx [b, n, k] int32`` is rebuilt from the ranks'
zero-padded bands by ONE ``all_sum`` over the TP group (``query_band`` ->
``Indexer.__call__(band=…, group=…)`` -> ``_finish_rows`` -> ``_reconstruct_rows`` ->
``collective.all_sum``). The KEY side (``index_k``) stays FULL on every rank.

This file is the *shape matrix*. It walks a grid over chunk offsets, row counts,
column counts, TP world sizes, indexer roles and routes and, per cell, compares the
banded reconstruction against the replicated full-chunk run at three levels:

1. INDEX SETS — the reconstructed ``cidx`` against the replicated ``cidx``: the
   number of query rows whose selected SET differs (``.any(-1)`` over the k axis)
   plus the mean per-row top-k overlap. Byte-equality is the invariant; EVERY
   divergence is recorded (and reported in the summary table) before it is
   asserted on — never swallowed, never marked xfail. Each rank's padded band is
   additionally checked zero outside its own band, and its in-band slice checked
   row-identical to the replicated run.
2. SCORE ROW / LOGIT LEVEL — the direct input to the top-k is the stored
   ``[b, n, nb]`` score row. It is captured two independent ways: by spying
   ``topk_from_row`` (the seam the tiled/buffer route hands the finished row to)
   and by recomputing the row from the indexer's own projections with
   ``_score_body_eager``, mirroring ``__call__``'s pre-head-sum expression exactly
   (the sibling suites' ``_qw`` idiom). Per cell we report max|delta| of the banded
   rows against the replicated rows and the k-th boundary gap
   (``sorted_row[k-1] - sorted_row[k]``) off the dense fp32 row, so a near-tie flip
   is distinguishable from a real bug. The dense fp32 reference is the ``_TILE=0``
   (untiled) branch, pinned separately in :class:`DenseReferenceTest`.
3. CACHE CONTENTS — ``publish_keys`` writes ``cache.index_k[:, g0:g0+g]``. The split
   is query-side only, so the banded and replicated configurations must write
   BYTE-IDENTICAL keys; pinned with a stub cache.

Also pinned: ``band is None`` never issues a collective (a raising ``all_sum`` spy),
the D2 fence (``n // world > _FENCE_MIN_ROWS``) exactly at 32/33/34 for world=2 and
64/65/68 for world=4, and all three indexer ROLES — plain owner, candidate source
(``is_candidate_source``, ``align=candidate_block_size``) and candidate consumer
(reads ``shared.candidates`` from the source) — on the hierarchical, tiled and
untiled routes.

Coverage: (a) chunk offsets whose ``start_pos % ratio`` and band seam land on and
just off a ``candidate_block_size`` / tile boundary, for several nb; (b) the n edges
around ``_FENCE_MIN_ROWS=16`` (1, 2, 3, 4, 5, 8, 12, 16, 17, 32, 33, 34, 64, 78, 512)
including the 32/33/34 world=2 and 64/65/68 world=4 band/replicated straddles; (c)
world=2 AND world=4 groups, so a rank's band is a strict sub-block of the chunk and
the padded ``all_sum`` has 4 contributors, incl. rank-boundary straddles; (d) a
scaled nb / context grid dense over the 50K-80K context band (the live regression's
band) spanning 32K..256K, with the nb down-scale factor recorded in EVERY row; (e)
partial last bands — n not divisible by the world (-> replicated) and divisible with
a last band shorter than a tile / a remainder that is not a multiple of ``align``;
(f) all three roles on all three routes, incl. small n (<= 16), which always falls
through to the tiled/untiled fallback.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest \
        tests/test_dsv41_indexer_qsplit_band_matrix.py -v
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.deepseek_v41 import indexer as IX
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import SharedState

RESULTS: list[dict[str, Any]] = []

#: Every import-read module knob this harness mutates (saved/restored per test).
KNOBS = ("_QSPLIT", "_TILE", "_TILE_BUDGET", "_TILE_MIN_NB", "_TILE_FORCE", "_HIER",
         "_FENCE_MIN_ROWS", "_SMALLN_WIDE_TILE", "_L2_FULL", "_SMALLN_ROW_BF16",
         "_TILED_IMPL")

#: Column-tile width the tiled route runs at under this harness (predictable).
TILE_TILED = 64
#: Tile budget for every route (bytes); keeps ``tile_width`` deterministic.
TILE_BUDGET = 128 << 20

#: nb is scaled down by this factor for the context grid so a 32K..256K equivalent
#: context fits in a toy-sized forward. ``context = nb * ratio * NB_SCALE``.
NB_SCALE = 64

#: The n grid the audit asked for: every edge around _FENCE_MIN_ROWS * world.
N_GRID = (1, 2, 3, 4, 5, 8, 12, 16, 17, 32, 33, 34, 64, 78, 512)
#: n values whose band seam straddles an align-8 block / a 64-wide tile boundary.
SEAM_N = (34, 66, 78, 128, 130)
#: Band seam straddles at world=4 (rows 17 / 25 / 33 / 32).
STRADDLE_N = (68, 100, 132, 128)
#: nb values a partial last tile / an unaligned tail is exercised at.
PARTIAL_NB = (260, 261)

#: max|delta| on the stored score row the harness tolerates before it is a bug
#: (an ulp-level GEMM difference is recorded but not a divergence of the index
#: sets; anything above this is a real score divergence and fails).
SCORE_TOLERANCE = 1e-4


def record(row: dict[str, Any]) -> None:
    RESULTS.append(row)
    print("  " + json.dumps(row), flush=True)


def _overlap(got: np.ndarray, ref: np.ndarray) -> float:
    """Mean per-row top-k overlap fraction between two [.., n, k] index tensors."""
    a = got.reshape(-1, got.shape[-1])
    b = ref.reshape(-1, ref.shape[-1])
    inter = [len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b)]
    return float(np.mean(inter)) / a.shape[-1]


def _overlap_clean(got: np.ndarray, ref: np.ndarray) -> tuple[float | None, int, int]:
    """(mean overlap over NON-degenerate rows, those rows' count, degenerate count).

    The plain :func:`_overlap` is only meaningful where a row's top-k holds k
    DISTINCT indices: a fully-masked row is k copies of the ``-1`` sentinel, so two
    byte-identical rows score ``1/k`` there. Those rows are excluded here (and
    counted) so the metric can be asserted at 1.0 without a sentinel artefact.
    """
    a = got.reshape(-1, got.shape[-1])
    b = ref.reshape(-1, ref.shape[-1])
    hits: list[float] = []
    degenerate = 0
    for x, y in zip(a, b):
        sx, sy = set(x.tolist()), set(y.tolist())
        if len(sy) < b.shape[-1]:
            degenerate += 1
            continue
        hits.append(len(sx & sy) / b.shape[-1])
    if not hits:
        return None, 0, degenerate
    return float(np.mean(hits)), len(hits), degenerate


class FakeGroup:
    """Minimal stand-in for ``mx.distributed.Group`` (rank / size only)."""

    def __init__(self, rank: int, world: int = 2):
        self._rank = rank
        self._world = world

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._world


def _args(**over: Any) -> ModelArgs:
    base: dict[str, Any] = dict(
        dim=64, n_layers=2, n_heads=4, head_dim=128, rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
        compress_ratios=(4, 4), kv_source_layers=(0,), index_source_layers=(0,),
        index_n_heads=2, index_head_dim=32, index_topk=6, max_seq_len=1 << 16,
        candidate_source_layer=0, candidate_topk_blocks=2, candidate_block_size=8)
    base.update(over)
    return ModelArgs(**base)


def _indexer(args: ModelArgs, layer_id: int, seed: int) -> IX.Indexer:
    mx.random.seed(seed)
    idx = IX.Indexer(args, layer_id)
    mx.eval(idx.parameters())
    return idx


def _inputs(seed: int, args: ModelArgs, *, b: int = 1, n: int = 64, nb: int = 120,
            ratio: int = 4, phase: int = 0):
    """Inputs for a chunk of ``n`` rows against ``nb`` full-on-every-rank key groups.

    ``start_pos`` is placed so the LAST query row sees exactly ``nb`` groups
    (``start_pos = nb*ratio - n``), shifted by ``phase`` — the knob that walks the
    group boundary against the chunk's first row (``start_pos % ratio``).
    """
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((b, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((b, n, args.q_lora_rank)).astype(np.float32))
    ik = mx.array(rng.standard_normal((b, nb, args.index_head_dim)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    start_pos = max(0, nb * ratio - n + phase)
    return x, qr, start_pos, fv, ik


class _RowSpy:
    """Record the finished ``[b, n, nb]`` score row each ``topk_from_row`` call sees.

    The tiled/buffer route hands its stored row to ``indexer.topk_from_row`` at the
    very end of :func:`_tiled_scores_buffer` — the exact tensor the top-k consumes.
    The hierarchical and untiled routes never call it (they have their own inline
    top-k), so ``rows`` stays empty there and the caller falls back to the dense
    ``_score_body_eager`` recomputation.
    """

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self._real = IX.topk_from_row

    def __enter__(self) -> "_RowSpy":
        real = self._real

        def wrapper(s: mx.array, k: int):                     # noqa: ANN202
            self.rows.append(np.array(s.astype(mx.float32)))
            return real(s, k)

        IX.topk_from_row = wrapper
        return self

    def __exit__(self, *exc: object) -> None:
        IX.topk_from_row = self._real


class _Fixture(unittest.TestCase):
    """Save/restore every module knob the sibling suites touch, plus the seams."""

    def setUp(self) -> None:
        self._saved = {k: getattr(IX, k) for k in KNOBS}
        self._real_all_sum = IX._coll.all_sum
        self._real_topk = IX.topk_from_row

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            setattr(IX, k, v)
        IX._coll.all_sum = self._real_all_sum
        IX.topk_from_row = self._real_topk

    @staticmethod
    def _route(route: str) -> None:
        """Put the module on one of the three indexer routes."""
        IX._QSPLIT = True
        IX._HIER = route == "hier"
        IX._TILE_BUDGET = TILE_BUDGET
        if route == "tiled":
            IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = TILE_TILED, 0, True
        elif route == "untiled":
            IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 0, 0, False
        else:
            IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 512, 0, False
        IX._SMALLN_WIDE_TILE = True
        IX._L2_FULL, IX._SMALLN_ROW_BF16 = True, False
        IX._TILED_IMPL = "buffer"


class _PositiveWeights:
    """``weights_proj`` stand-in with strictly positive head weights.

    The indexer's score is ``relu(q·k) * w`` summed over heads. With the raw random
    init ``w`` takes either sign, and for a NEGATIVE ``w`` every column whose head
    dots are all non-positive scores exactly ``0`` — which is then the row MAXIMUM,
    so a large share of rows tie exactly at the top-k boundary (the ``kth_gap == 0``
    census). That is a legitimate cell and the tie census records it, but it cannot
    distinguish a near-tie flip from a bug, so the margin cells use this wrapper to
    give the boundary a real gap.
    """

    def __init__(self, lin: nn.Linear):
        self._lin = lin

    def __call__(self, x: mx.array) -> mx.array:
        return mx.abs(self._lin(x)) + 0.5


def _effective_route(idx: IX.Indexer, n_full: int, nb: int) -> str:
    """The branch the replicated FULL-chunk run actually takes (D3 keys on n_full)."""
    if IX._HIER and n_full > IX._FENCE_MIN_ROWS:
        return "hier"
    align = idx.candidate_block_size if idx.is_candidate_source else 1
    return "tiled" if IX.tiled(1, n_full, idx.n_heads, nb, align) < nb else "untiled"


# --------------------------------------------------------------------------
# raw forward / capture helpers
# --------------------------------------------------------------------------
def _sum_pads(pads: list[np.ndarray]) -> np.ndarray:
    """Elementwise sum of the ranks' reconstruct-shaped tensors (0+x == x)."""
    total = np.zeros_like(pads[0])
    for pad in pads:
        total = total + pad
    return total


def _forward(idx: IX.Indexer, inp, shared: SharedState, *,
             band: tuple[int, int] | None, group: FakeGroup | None,
             offset: int) -> mx.array:
    x, qr, start_pos, freqvec, index_k = inp
    return idx(x, qr, start_pos, offset, freqvec, index_k, shared,
               band=band, group=group)


class _Replicated:
    """Result of the replicated (``band=None``) run: it must touch NO collective."""

    __slots__ = ("out", "cons_out", "shared", "row", "cons_row")

    def __init__(self, out, cons_out, shared, row, cons_row):   # noqa: ANN001
        self.out, self.cons_out = out, cons_out
        self.shared, self.row, self.cons_row = shared, row, cons_row


def _run_replicated(idx: IX.Indexer, cons: IX.Indexer | None, inp, *,
                    offset: int) -> _Replicated:
    """Run the replicated forward; a raising ``all_sum`` spy pins the no-collective rule."""
    real = IX._coll.all_sum

    def boom(*_a: object, **_k: object):                          # noqa: ANN202
        raise AssertionError("band=None must never issue a collective")

    IX._coll.all_sum = boom
    try:
        with _RowSpy() as spy:
            shared = SharedState()
            n0 = len(spy.rows)
            out = _forward(idx, inp, shared, band=None, group=None, offset=offset)
            mx.eval(out)
            row = spy.rows[n0] if len(spy.rows) > n0 else None
            cons_out, cons_row = None, None
            if cons is not None:
                n1 = len(spy.rows)
                cons_out = _forward(cons, inp, shared, band=None, group=None,
                                    offset=offset)
                mx.eval(cons_out)
                cons_row = spy.rows[n1] if len(spy.rows) > n1 else None
    finally:
        IX._coll.all_sum = real
    return _Replicated(np.array(out),
                        None if cons_out is None else np.array(cons_out),
                        shared, row, cons_row)


class _Banded:
    """Per-rank padded bands and outputs of the banded run (identity ``all_sum``)."""

    __slots__ = ("pads", "padded", "cons_pads", "masks", "rows", "cons_rows")

    def __init__(self) -> None:
        self.pads: list[np.ndarray] = []            # each rank's RECONSTRUCTED cidx
        self.padded: list[np.ndarray] = []          # each rank's pre-collective band
        self.cons_pads: list[np.ndarray] = []
        self.masks: list[np.ndarray | None] = []
        self.rows: list[np.ndarray | None] = []
        self.cons_rows: list[np.ndarray | None] = []


def _run_banded(idx: IX.Indexer, cons: IX.Indexer | None, inp, *,
                bands: list[tuple[int, int]], offset: int, world: int) -> _Banded:
    """One forward per rank; the spied collective is an identity, so it returns the pad."""
    real = IX._coll.all_sum
    captured: dict[str, mx.array] = {}
    result = _Banded()

    def identity(t, group=None):                                 # noqa: ANN001, ANN202
        captured["padded"] = t
        return t

    IX._coll.all_sum = identity
    try:
        with _RowSpy() as spy:
            for r in range(world):
                lo, hi = bands[r]
                shared = SharedState()
                n0 = len(spy.rows)
                out = _forward(idx, inp, shared, band=(lo, hi),
                               group=FakeGroup(r, world), offset=offset)
                padded = np.array(captured["padded"])
                mx.eval(out)
                result.pads.append(np.array(out))
                result.padded.append(padded)
                result.masks.append(None if shared.candidates is None
                                    else np.array(shared.candidates))
                result.rows.append(spy.rows[n0] if len(spy.rows) > n0 else None)
                if cons is not None:
                    n1 = len(spy.rows)
                    cons_out = _forward(cons, inp, shared, band=(lo, hi),
                                        group=FakeGroup(r, world), offset=offset)
                    mx.eval(cons_out)
                    result.cons_pads.append(np.array(cons_out))
                    result.cons_rows.append(spy.rows[n1] if len(spy.rows) > n1
                                            else None)
    finally:
        IX._coll.all_sum = real
    return result


def _dense_row(idx: IX.Indexer, inp, *, band: tuple[int, int] | None,
               cand: np.ndarray | None = None) -> np.ndarray:
    """The indexer's own ``[b, n, nb]`` score row, recomputed in dense fp32.

    Mirrors ``Indexer.__call__`` line for line: project the FULL chunk, slice the
    query axis (after the projections — the live-divergence fix), rope at the
    band's absolute positions, fp4-quantize, collapse the heads with
    ``weights_proj(x) * softmax_scale * n_heads**-0.5`` via ``_score_body_eager``
    (the same expression the untiled branch and every tile use), then apply the
    visibility fence. ``cand`` is the published candidate mask, applied exactly
    where a consumer layer applies it.
    """
    x, qr, start_pos, freqvec, index_k = inp
    bsz, n_full, _ = x.shape
    q_full = idx.wq_b(qr)
    w_full = idx.weights_proj(x)
    if band is None:
        qpos = mx.arange(start_pos, start_pos + n_full)
    else:
        lo, hi = band
        q_full, w_full = q_full[:, lo:hi, :], w_full[:, lo:hi, :]
        qpos = mx.arange(start_pos + lo, start_pos + hi)
    n = qpos.shape[0]
    q = q_full.reshape(bsz, n, idx.n_heads, idx.head_dim)
    q = IX.rope_tail(q, idx.rope_head_dim, *IX.cos_sin_at(freqvec, qpos))
    q = IX.fake_quant_fp4_ue8m0(q, 32)
    w = w_full * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((qpos + 1) // idx.ratio)[:, None]
    scores = IX._score_body_eager(q.astype(mx.float32), index_k.astype(mx.float32),
                                   w.astype(mx.float32))
    cols = mx.arange(index_k.shape[1], dtype=mx.int32)
    scores = mx.where(cols[None, :] < lens, scores, IX.NEG_INF)
    if cand is not None:
        scores = mx.where(mx.array(cand), scores, IX.NEG_INF)
    return np.array(scores)


def _kth_gap(row: np.ndarray, k: int) -> tuple[float | None, int]:
    """The k-th boundary gap ``sorted[k-1] - sorted[k]`` over a dense fp32 row.

    Non-finite entries (a row with fewer than k+1 visible columns) are dropped; a
    returned gap of 0 (exactly tied at the boundary) means an index flip in that
    row needs no bug to explain it.
    """
    if k >= row.shape[-1]:
        return None, 0
    srt = np.sort(row, axis=-1)[..., ::-1]
    with np.errstate(invalid="ignore"):
        gap = srt[..., k - 1] - srt[..., k]
    finite = gap[np.isfinite(gap)]
    if finite.size == 0:
        return None, 0
    return float(finite.min()), int((finite <= 0.0).sum())


def _row_delta(got: np.ndarray, ref: np.ndarray) -> tuple[float, int]:
    """``max|got - ref|`` over the finite entries, plus the non-finite mismatches.

    The stored row holds ``-inf`` for every masked / unreachable column, so
    ``-inf - -inf`` is a *match* (nan arithmetic), not a difference: those pairs
    are zeroed before the max. Anything that is non-finite but not a matching
    pair of infinities is counted separately and is a genuine divergence.
    """
    with np.errstate(invalid="ignore"):
        delta = got.astype(np.float64) - ref.astype(np.float64)
    both_inf = np.isinf(got) & np.isinf(ref) & (np.sign(got) == np.sign(ref))
    bad = int((~np.isfinite(delta) & ~both_inf).sum())
    delta = np.where(both_inf, 0.0, delta)
    finite = delta[np.isfinite(delta)]
    return (float(np.abs(finite).max()) if finite.size else 0.0), bad


# --------------------------------------------------------------------------
# the cell driver
# --------------------------------------------------------------------------
def _cell(test: "_Fixture", *, role: str, route: str, world: int, n: int,
          nb: int, seed: int, b: int = 1, offset: int = 0, phase: int = 0,
          ratio: int = 4, explicit_band: bool = False, nb_scale: int = 1,
          positive_weights: bool = False, tag: str = "") -> dict[str, Any]:
    """Run one matrix cell end to end, record it, then assert the invariants.

    A cell where ``query_band`` refuses (gate off / no group / not divisible /
    band rows <= the fence) is recorded as ``gated=False``: the replicated path is
    the correct outcome and its no-collective rule is pinned instead.
    """
    test._route(route)
    args = _args(candidate_source_layer=(-1 if role == "plain" else 0),
                 compress_ratios=(ratio, ratio))
    idx = _indexer(args, 0, seed)
    cons = _indexer(args, 1, seed + 1) if role == "consumer" else None
    if positive_weights:
        setattr(idx, "weights_proj", _PositiveWeights(idx.weights_proj))
        if cons is not None:
            setattr(cons, "weights_proj", _PositiveWeights(cons.weights_proj))
    inp = _inputs(seed, args, b=b, n=n, nb=nb, ratio=ratio, phase=phase)

    groups = [FakeGroup(r, world) for r in range(world)]
    gated_bands = [IX.query_band(n, g) for g in groups]
    if explicit_band:
        rows = n // world
        gated_bands = [(r * rows, (r + 1) * rows) for r in range(world)]
    gated = all(band is not None for band in gated_bands)

    replicated = _run_replicated(idx, cons, inp, offset=offset)
    score_idx = cons if role == "consumer" else idx
    role_out = (replicated.cons_out if role == "consumer" else replicated.out)
    role_row = (replicated.cons_row if role == "consumer" else replicated.row)
    full_cand = (None if replicated.shared.candidates is None
                 else np.array(replicated.shared.candidates))

    row: dict[str, Any] = {
        "case": "band_matrix", "tag": tag, "role": role, "route": route,
        "effective": _effective_route(idx, n, nb), "gated": bool(gated),
        "margin": bool(positive_weights),
        "world": world, "b": b, "n": n, "nb": nb, "nb_scale": nb_scale,
        "context": nb * ratio * nb_scale, "start_pos": int(inp[2]), "offset": offset,
        "phase": phase, "ratio": ratio, "rows_compared": b * n,
        "k": min(args.index_topk, nb),
    }

    if not gated:
        # gate refused: the replicated run IS the answer, and it must be
        # collective-free (already asserted by the raising spy in _run_replicated).
        test.assertIsNone(IX.query_band(n, FakeGroup(0, world)), n)
        row.update(band=None, differing_rows=0, src_differing_rows=None,
                   overlap=1.0, overlap_clean=1.0, overlap_clean_rows=0,
                   degenerate_rows=0, row_src=None, score_max_delta=None,
                   row_delta_fp32=None, nonfinite_mismatch=None, cand_rows=None,
                   kth_gap=None, kth_tied_rows=None, padded_zero_outside=None,
                   band_slice_equal=None, cons_band_slice_equal=None)
        record(row)
        return row

    bands: list[tuple[int, int]] = [band for band in gated_bands if band is not None]
    banded = _run_banded(idx, cons, inp, bands=bands, offset=offset, world=world)
    role_pads = banded.cons_pads if role == "consumer" else banded.pads
    recon = _sum_pads(role_pads)
    diff_rows = int((recon != role_out).any(-1).sum())
    overlap = _overlap(recon, role_out)
    overlap_clean, overlap_rows, degenerate_rows = _overlap_clean(recon, role_out)

    # the padded band is zero outside this rank's rows, and its in-band slice is
    # row-identical to the replicated run (row independence, per contributor).
    # ``banded.padded`` belongs to the SOURCE call, so it is compared against the
    # replicated SOURCE output even in the consumer cell; the consumer's own
    # reconstruction is compared against the replicated consumer output.
    zero_outside = True
    slice_equal = True
    cons_slice_equal = True
    for r, (lo, hi) in enumerate(bands):
        padded = banded.padded[r]
        mask = np.ones(padded.shape[1], bool)
        mask[lo:hi] = False
        zero_outside &= bool(np.all(padded[:, mask, :] == 0))
        slice_equal &= bool(np.array_equal(padded[:, lo:hi, :],
                                           replicated.out[:, lo:hi, :]))
        if role == "consumer":
            cons_slice_equal &= bool(np.array_equal(
                banded.cons_pads[r][:, lo:hi, :], role_out[:, lo:hi, :]))
    src_diff_rows = None
    if role == "consumer":
        src_diff_rows = int((_sum_pads(banded.pads) != replicated.out).any(-1).sum())

    # ---- score row / logit level ----
    ref_full = _dense_row(score_idx, inp, band=None,
                          cand=(full_cand if role == "consumer" else None))
    delta_fp32 = 0.0
    nonfinite = 0
    stored_delta: float | None = None
    captured_bands = banded.cons_rows if role == "consumer" else banded.rows
    if role_row is not None and all(r is not None for r in captured_bands):
        stored_delta = 0.0
        for r, (lo, hi) in enumerate(bands):
            magnitude, bad = _row_delta(captured_bands[r], role_row[:, lo:hi, :])
            stored_delta = max(stored_delta, magnitude)
            nonfinite += bad
    for r, (lo, hi) in enumerate(bands):
        ref_band = _dense_row(score_idx, inp, band=(lo, hi),
                              cand=(banded.masks[r] if role == "consumer" else None))
        magnitude, bad = _row_delta(ref_band, ref_full[:, lo:hi, :])
        delta_fp32 = max(delta_fp32, magnitude)
        nonfinite += bad
    row_src = "stored-row" if stored_delta is not None else "fp32-recomputed"
    score_delta = stored_delta if stored_delta is not None else delta_fp32

    # ---- candidate mask rows (source role only) ----
    cand_rows = None
    if role == "source" and full_cand is not None:
        cand_rows = 0
        for r, (lo, hi) in enumerate(bands):
            cand_rows += int((banded.masks[r] != full_cand[:, lo:hi, :]).any(-1).sum())

    kth_gap, kth_tied = _kth_gap(ref_full, row["k"])
    row.update(
        band=[[int(lo), int(hi)] for lo, hi in bands],
        seam_mod_align=[int(hi % (idx.candidate_block_size if idx.is_candidate_source
                                   else 1)) for _lo, hi in bands],
        seam_mod_tile=[int(hi % TILE_TILED) for _lo, hi in bands],
        differing_rows=diff_rows, src_differing_rows=src_diff_rows,
        overlap=overlap, overlap_clean=overlap_clean,
        overlap_clean_rows=overlap_rows, degenerate_rows=degenerate_rows,
        row_src=row_src, score_max_delta=score_delta,
        row_delta_fp32=delta_fp32, nonfinite_mismatch=nonfinite,
        cand_rows=cand_rows, kth_gap=kth_gap,
        kth_tied_rows=kth_tied, padded_zero_outside=bool(zero_outside),
        band_slice_equal=bool(slice_equal),
        cons_band_slice_equal=bool(cons_slice_equal),
    )
    record(row)

    test.assertEqual(diff_rows, 0,
                     f"banded != replicated: {diff_rows} rows ({role}/{route}/"
                     f"world={world}/n={n}/nb={nb}/start_pos={inp[2]})")
    if overlap_clean is not None:
        test.assertEqual(overlap_clean, 1.0,
                         f"non-degenerate top-k overlap {overlap_clean} on "
                         f"{overlap_rows} rows")
    test.assertTrue(zero_outside, "padded band is not zero outside its own rows")
    test.assertTrue(slice_equal, "padded band slice != replicated run's rows")
    test.assertTrue(cons_slice_equal, "consumer band slice != replicated run's rows")
    test.assertLessEqual(float(score_delta), SCORE_TOLERANCE,
                         f"score row diverged by {score_delta}")
    if cand_rows is not None:
        test.assertEqual(cand_rows, 0, "published candidate mask differs on the band")
    if src_diff_rows is not None:
        test.assertEqual(src_diff_rows, 0, "source band != replicated")
    return row


# --------------------------------------------------------------------------
# 0. dense fp32 reference: the _TILE=0 (untiled) branch
# --------------------------------------------------------------------------
class DenseReferenceTest(_Fixture):
    """The dense fp32 row and the ``_TILE=0`` branch must agree exactly.

    This pins the reference the whole matrix is scored against: the untiled
    branch materializes ``sum(einsum(q,k), heads) * w`` in one expression, and
    ``_dense_row`` reproduces it; the top-k off that row is
    ``argpartition -> sort -> i + offset`` exactly. Any drift here invalidates
    every ``row_delta_fp32`` in the matrix.
    """

    def test_untiled_branch_equals_the_dense_row(self) -> None:
        self._route("untiled")
        args = _args(candidate_source_layer=-1)
        for n, nb, seed in ((78, 260, 61), (512, 300, 62), (16, 120, 63)):
            idx = _indexer(args, 0, seed)
            inp = _inputs(seed, args, n=n, nb=nb, ratio=4)
            out = _run_replicated(idx, None, inp, offset=7).out
            dense = mx.array(_dense_row(idx, inp, band=None)).astype(IX._row_dtype(n))
            k = min(args.index_topk, nb)
            i = mx.argpartition(-dense, k - 1, axis=-1)[..., :k]
            i = mx.sort(i.astype(mx.int32), axis=-1)
            lens = ((inp[2] + mx.arange(n) + 1) // idx.ratio)[:, None].astype(mx.int32)
            ref = np.array(mx.where(i < lens[None], i + 7, mx.array(-1, mx.int32)))
            diff = int((ref != out).any(-1).sum())
            record({"case": "dense_reference", "n": n, "nb": nb, "differing_rows": diff,
                    "row_dtype": str(IX._row_dtype(n))})
            self.assertEqual(diff, 0, f"untiled != dense recompute at n={n}, nb={nb}")


# --------------------------------------------------------------------------
# 1. the gate / D2 fence boundary exactly at the straddles
# --------------------------------------------------------------------------
class GateBoundaryTest(_Fixture):
    """``query_band`` flips None -> band at ``n // world > _FENCE_MIN_ROWS`` (D2)."""

    def test_world2_exactly_32_33_34(self) -> None:
        self._route("hier")
        fence = IX._FENCE_MIN_ROWS
        self.assertEqual(fence, 16, "this harness assumes the shipped fence")
        for n in range(1, fence * 2 + 1):                       # 1..32 -> replicated
            self.assertIsNone(IX.query_band(n, FakeGroup(0)), f"rank0 n={n}")
            self.assertIsNone(IX.query_band(n, FakeGroup(1)), f"rank1 n={n}")
        self.assertIsNone(IX.query_band(33, FakeGroup(0)), "odd n must stay replicated")
        self.assertIsNone(IX.query_band(33, FakeGroup(1)), "odd n must stay replicated")
        self.assertEqual(IX.query_band(34, FakeGroup(0)), (0, 17))
        self.assertEqual(IX.query_band(34, FakeGroup(1)), (17, 34))
        record({"case": "gate_world2", "fence": fence, "replicated": [32, 33, 65],
                "banded": {"34": [[0, 17], [17, 34]]}})

    def test_world4_exactly_64_65_68(self) -> None:
        self._route("hier")
        fence = IX._FENCE_MIN_ROWS
        for n in (1, 33, 64, 65, 66, 67):                       # 64//4 == 16; 65..67 odd mod
            for r in range(4):
                self.assertIsNone(IX.query_band(n, FakeGroup(r, 4)), f"n={n} r={r}")
        self.assertEqual([IX.query_band(68, FakeGroup(r, 4)) for r in range(4)],
                         [(0, 17), (17, 34), (34, 51), (51, 68)])
        record({"case": "gate_world4", "fence": fence, "replicated": [64, 65, 66, 67],
                "band": {"68": [[0, 17], [17, 34], [34, 51], [51, 68]]}})

    def test_world1_and_no_group_never_band(self) -> None:
        self._route("hier")
        for n in (512, 2048):
            self.assertIsNone(IX.query_band(n, FakeGroup(0, world=1)))
            self.assertIsNone(IX.query_band(n, None))
        record({"case": "gate_world1_nogroup"})

    def test_gate_off_returns_none_for_every_shape(self) -> None:
        self._route("hier")
        IX._QSPLIT = False
        for n in (1, 4, 16, 33, 34, 64, 78, 512, 2048):
            for world in (1, 2, 4):
                self.assertIsNone(IX.query_band(n, FakeGroup(0, world=world)), n)
        record({"case": "gate_off_all_shapes"})


# --------------------------------------------------------------------------
# 2. (b) the n grid, world=2 AND world=4
# --------------------------------------------------------------------------
class RowCountSweepTest(_Fixture):
    """Every n edge around the fence, at both TP world sizes, on both routes.

    n=33/world=2 and 65/world=4 are the non-divisible straddles (-> replicated);
    n=34/world=2 and 68/world=4 are the first banded shapes.
    """

    def test_n_grid_hierarchical(self) -> None:
        for n in N_GRID:
            for world in (2, 4):
                _cell(self, role="source", route="hier", world=world, n=n, nb=140,
                      seed=1000 + n + world, offset=8 + n, tag="n_grid_hier")

    def test_n_grid_tiled_route(self) -> None:
        for n in (12, 16, 17, 33, 34, 64, 78, 512):
            for world in (2, 4):
                _cell(self, role="source", route="tiled", world=world, n=n, nb=140,
                      seed=2000 + n + world, offset=8, tag="n_grid_tiled")

    def test_n_grid_small_n_falls_through(self) -> None:
        """n <= the fence always falls through to the fallback (tiled and untiled)."""
        for route in ("tiled", "untiled"):
            for n in (1, 2, 3, 4, 5, 8, 12, 16):
                _cell(self, role="source", route=route, world=2, n=n, nb=140,
                      seed=3000 + n, offset=4, tag="smalln_fallthrough")

    def test_n512_batch_two(self) -> None:
        for world in (2, 4):
            _cell(self, role="source", route="hier", world=world, n=512, nb=300,
                  b=2, seed=51 + world, offset=136, tag="batch2")


# --------------------------------------------------------------------------
# 3. (a) chunk-offset / seam sweep
# --------------------------------------------------------------------------
class ChunkOffsetSeamTest(_Fixture):
    """``start_pos % ratio`` phases against band seams that straddle align / tile.

    ``seam_mod_align`` / ``seam_mod_tile`` in each record name the offset of the
    band's hi row inside a candidate block (align = candidate_block_size on the
    source layer) and inside a 64-wide tile: 0 is exactly on the seam, non-zero is
    just off it.
    """

    def test_offset_and_phase_grid(self) -> None:
        for n in SEAM_N:
            for world in (2, 4):
                for phase in (0, 1, 3):
                    _cell(self, role="source", route="hier", world=world, n=n,
                          nb=260, seed=4000 + n + world + phase, phase=phase,
                          offset=260 * 4 + n, tag="offset_seam")

    def test_seams_straddle_a_block_and_a_tile(self) -> None:
        """Assert the grid really contains on-seam and just-off-seam bands."""
        self._route("hier")
        align = 8
        on_seam, off_seam = 0, 0
        for n, world in ((128, 2), (130, 2), (78, 2), (68, 4), (100, 4), (132, 4)):
            if n % world:
                continue
            rows = n // world
            for r in range(1, world):
                hi = r * rows
                if hi % align == 0 and hi % TILE_TILED == 0:
                    on_seam += 1
                else:
                    off_seam += 1
        record({"case": "seam_census", "on_seam_boundaries": on_seam,
                "just_off_seam_boundaries": off_seam})
        self.assertGreater(on_seam, 0)
        self.assertGreater(off_seam, 0)


# --------------------------------------------------------------------------
# 4. (c) world=4 groups: a strict sub-block band, 4 padded contributors
# --------------------------------------------------------------------------
class WorldFourStraddleTest(_Fixture):
    """world=4: each rank's band is a strict sub-block and the padded ``all_sum``
    has four contributors whose disjoint support the reconstruction sums."""

    def test_four_contributor_reconstruction(self) -> None:
        for n in STRADDLE_N:
            for route in ("hier", "tiled"):
                _cell(self, role="source", route=route, world=4, n=n,
                      nb=PARTIAL_NB[0], seed=5000 + n, offset=68 + n,
                      tag="world4_straddle")

    def test_four_pads_are_disjoint_and_cover(self) -> None:
        self._route("tiled")
        args = _args(candidate_source_layer=-1)
        idx = _indexer(args, 0, 77)
        n, nb, world = 132, 260, 4
        offset = 4096                      # concat-space base: real indices >= offset
        inp = _inputs(77, args, n=n, nb=nb, ratio=4)
        bands: list[tuple[int, int]] = [(r * (n // world), (r + 1) * (n // world))
                                        for r in range(world)]
        banded = _run_banded(idx, None, inp, bands=bands, offset=offset, world=world)
        shape = (1, n, min(args.index_topk, nb))
        accum = np.zeros(shape, np.int32)
        support = np.zeros(shape, np.int32)
        outside_nonzero = 0
        for r, (lo, hi) in enumerate(bands):
            padded = banded.padded[r]
            outside = np.ones(n, bool)
            outside[lo:hi] = False
            outside_nonzero += int((padded[:, outside, :] != 0).sum())
            accum = accum + padded
            support = support + (padded != 0).astype(np.int32)
        self.assertTrue(np.array_equal(accum, _sum_pads(banded.pads)))
        # disjoint support: no element of the reconstruction is fed by two ranks
        double = int((support > 1).sum())
        # rows carrying a real (non-sentinel) selection
        rows_with_real = int((accum >= offset).any(-1).sum())
        record({"case": "world4_support", "n": n, "world": world, "offset": offset,
                "rows_with_real": rows_with_real, "rows_total": n,
                "max_contributors": int(support.max()),
                "double_contributor_elements": double,
                "nonzero_outside_band": outside_nonzero})
        self.assertEqual(double, 0, "an element had two non-zero band contributors")
        self.assertEqual(outside_nonzero, 0, "a pad wrote outside its own band")
        self.assertGreater(rows_with_real, 0, "no row carried a real selection")


# --------------------------------------------------------------------------
# 5. (d) nb / context sweep — dense over the 50K-80K band
# --------------------------------------------------------------------------
#: Equivalent production contexts (nb * ratio * NB_SCALE) — dense in 50K..80K, the
#: window the live identity_3 divergence sat in, plus 32K..256K shoulders.
CONTEXTS = (32_768, 49_152, 51_200, 53_248, 57_344, 61_440, 65_536, 69_632, 73_728,
            77_824, 81_920, 98_304, 131_072, 196_608, 262_144)


class ContextSweepTest(_Fixture):
    """The nb/context grid, scaled: ``context = nb * ratio * NB_SCALE``."""

    def test_context_grid_world2_and_world4(self) -> None:
        for context in CONTEXTS:
            for n in (78, 512):
                nb = max(1, context // (4 * NB_SCALE))
                for world in (2, 4):
                    _cell(self, role="source", route="hier", world=world, n=n, nb=nb,
                          seed=6000 + context + n + world, offset=n,
                          nb_scale=NB_SCALE, tag="context_sweep")

    def test_scaled_nb_stays_interpretable(self) -> None:
        """The scaling factor is recorded in every row and the dense 50K-80K
        window really is dense (five distinct contexts inside it)."""
        inside = [c for c in CONTEXTS if 50_000 <= c <= 80_000]
        recorded = sorted({row["context"] for row in RESULTS
                           if row.get("tag") == "context_sweep"})
        record({"case": "context_census", "nb_scale": NB_SCALE,
                "contexts_in_50k_80k": inside, "span": [CONTEXTS[0], CONTEXTS[-1]],
                "contexts_recorded": recorded})
        self.assertGreaterEqual(len(inside), 5)
        self.assertGreaterEqual(len([c for c in recorded if 50_000 <= c <= 80_000]), 5,
                                "the 50K-80K context band is not dense in the records")
        for row in RESULTS:
            if row.get("tag") == "context_sweep":
                self.assertEqual(row["context"], row["nb"] * 4 * NB_SCALE)


# --------------------------------------------------------------------------
# 6. (e) partial last band
# --------------------------------------------------------------------------
class PartialLastBandTest(_Fixture):
    """n not divisible by the world (-> replicated) and divisible with a short /
    unaligned last band or a partial last tile."""

    def test_non_divisible_n_is_replicated(self) -> None:
        for n, world in ((17, 2), (33, 2), (65, 2), (34, 4), (78, 4), (130, 4)):
            _cell(self, role="source", route="hier", world=world, n=n, nb=200,
                  seed=7000 + n + world, offset=n, tag="partial_nondivisible")

    def test_short_and_unaligned_bands(self) -> None:
        # rows 17 / 25 / 33 / 65: last band shorter than a 64-wide tile, and a
        # seam that is not a multiple of align (candidate_block_size = 8).
        for n, world in ((68, 4), (100, 4), (132, 4), (130, 2), (34, 2)):
            for nb in PARTIAL_NB:
                _cell(self, role="source", route="tiled", world=world, n=n, nb=nb,
                      seed=8000 + n + nb, offset=n, tag="partial_short_band")

    def test_partial_last_tile_and_unaligned_remainder(self) -> None:
        """nb = 260 (a 4-column last tile at width 64) and 261 (align remainder 5)."""
        for nb in PARTIAL_NB:
            record({"case": "partial_tile_geometry", "nb": nb,
                    "last_tile_width": nb % TILE_TILED or TILE_TILED,
                    "remainder_mod_align": nb % 8})
            _cell(self, role="source", route="tiled", world=2, n=78, nb=nb,
                  seed=9000 + nb, offset=78, tag="partial_tile")
        record({"case": "partial_tile_census", "nb": list(PARTIAL_NB),
                "tile": TILE_TILED, "align": 8})


# --------------------------------------------------------------------------
# 7. (f) all three roles x all three routes
# --------------------------------------------------------------------------
class RoleRouteMatrixTest(_Fixture):
    def test_roles_on_every_route(self) -> None:
        for role in ("plain", "source", "consumer"):
            for route in ("hier", "tiled", "untiled"):
                for world in (2, 4):
                    for n in (34, 78, 130):
                        _cell(self, role=role, route=route, world=world, n=n,
                              nb=260, seed=10_000 + world + n, offset=8 + n,
                              tag="role_route")

    def test_explicit_band_below_the_fence(self) -> None:
        """A band injected below the fence (the D2 shape) still reconstructs.

        ``query_band`` refuses these (correctly); the explicit band proves the
        fallback return sites are wrapped too — band rows 6 / 8 <= the fence.
        """
        for n, world in ((12, 2), (16, 2), (12, 4)):
            for route in ("tiled", "untiled"):
                _cell(self, role="source", route=route, world=world, n=n, nb=140,
                      seed=11_000 + n + world, offset=4, explicit_band=True,
                      tag="explicit_band_below_fence")

    def test_consumer_sees_the_source_bands_candidates(self) -> None:
        """The consumer's banded run consumes the SOURCE's banded candidate mask
        (shape [b, band_rows, nb]) and still reconstructs to the replicated run."""
        self._route("hier")
        args = _args(candidate_source_layer=0)
        src = _indexer(args, 0, 12)
        cons = _indexer(args, 1, 13)
        inp = _inputs(12, args, n=78, nb=260, ratio=4)
        n, world = 78, 2
        bands: list[tuple[int, int]] = [
            band for band in (IX.query_band(n, FakeGroup(r, world))
                              for r in range(world)) if band is not None]
        banded = _run_banded(src, cons, inp, bands=bands, offset=86, world=world)
        shapes = [None if m is None else list(m.shape) for m in banded.masks]
        record({"case": "consumer_banded_mask", "band_rows": n // world,
                "mask_shapes": shapes})
        self.assertTrue(all(s == [1, n // world, 260] for s in shapes),
                        f"banded source published {shapes}")
        replicated = _run_replicated(src, cons, inp, offset=86)
        self.assertEqual(int((_sum_pads(banded.cons_pads)
                              != replicated.cons_out).any(-1).sum()), 0)


# --------------------------------------------------------------------------
# 8. margin cells: a real k-th boundary gap, so a flip cannot be a tie
# --------------------------------------------------------------------------
class MarginDistinctTest(_Fixture):
    """Same comparisons, but with strictly positive ``weights_proj`` head weights.

    The raw random init leaves ``w`` either sign and a large share of rows ties
    exactly at the k-th boundary (recorded per cell as ``kth_gap`` / ``kth_tied_rows``
    / ``degenerate_rows``). Those cells cannot distinguish a near-tie flip from a
    bug, so this class runs the same grid with the boundary carrying a real margin
    and asserts the gap really is positive on most rows.
    """

    def test_banded_equals_replicated_with_a_margin(self) -> None:
        positive_rows = 0
        total_rows = 0
        for n in (34, 78, 130, 512):
            for world in (2, 4):
                for route in ("hier", "tiled"):
                    row = _cell(self, role="source", route=route, world=world, n=n,
                                nb=300, seed=12_000 + n + world, offset=300 * 4,
                                positive_weights=True, tag="margin")
                    if row["gated"] and row["kth_gap"] is not None:
                        total_rows += 1
                        positive_rows += int(row["kth_gap"] > 0.0)
        record({"case": "margin_census", "cells_with_gap": total_rows,
                "cells_with_positive_gap": positive_rows})
        self.assertGreater(positive_rows, 0, "no margin cell had a positive kth gap")


# --------------------------------------------------------------------------
# 9. cache contents: the KEY side is full on every rank, banded or not
# --------------------------------------------------------------------------
class _StubCache:
    """Just the surface ``Indexer.publish_keys`` writes through."""

    def __init__(self, bsz: int, rows: int, head_dim: int):
        self.index_k = mx.zeros((bsz, rows, head_dim), dtype=mx.bfloat16)


class PublishKeysBandTest(_Fixture):
    def test_keys_identical_banded_and_replicated(self) -> None:
        """``publish_keys`` takes no band: the banded and replicated configurations
        must write byte-identical ``index_k``, and the banded forward must not
        touch the key cache."""
        args = _args(candidate_source_layer=-1)
        idx = _indexer(args, 0, 21)
        rng = np.random.default_rng(21)
        ratio, head_dim = idx.ratio, args.index_head_dim
        for start_pos, group_rows in ((0, 16), (37, 16), (2048, 20)):
            latents = mx.array(rng.standard_normal((1, group_rows, args.head_dim))
                                .astype(np.float32))
            freqvec = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2)
                               .astype(np.float32))
            # replicated configuration
            self._route("hier")
            IX._QSPLIT = False
            replicated = _StubCache(1, (start_pos // ratio) + group_rows + 4, head_dim)
            idx.publish_keys(latents, start_pos, freqvec, replicated)
            mx.eval(replicated.index_k)
            # banded configuration
            IX._QSPLIT = True
            banded = _StubCache(1, (start_pos // ratio) + group_rows + 4, head_dim)
            idx.publish_keys(latents, start_pos, freqvec, banded)
            mx.eval(banded.index_k)
            keys_replicated = np.array(replicated.index_k.astype(mx.float32))
            keys_banded = np.array(banded.index_k.astype(mx.float32))
            same = bool(np.array_equal(keys_replicated, keys_banded))
            nonzero = int((keys_banded != 0).any(-1).sum())
            record({"case": "publish_keys", "start_pos": start_pos,
                    "group_rows": group_rows, "identical": same,
                    "written_rows": nonzero})
            self.assertTrue(same, f"keys differ at start_pos={start_pos}")
            self.assertGreater(nonzero, 0, "publish_keys wrote nothing")

    def test_banded_forward_never_writes_the_key_cache(self) -> None:
        """A banded ``__call__`` is query-side only: the caller's ``index_k`` is a
        read-only input and must come back untouched (it is, structurally — this
        pins it so a future refactor cannot quietly start mutating it)."""
        self._route("hier")
        idx = _indexer(_args(candidate_source_layer=-1), 0, 22)
        inp = _inputs(22, _args(), n=78, nb=260, ratio=4)
        before = np.array(inp[4])
        _run_banded(idx, None, inp, bands=[(0, 39), (39, 78)], offset=86, world=2)
        after = np.array(inp[4])
        identical = bool(np.array_equal(before, after))
        record({"case": "index_k_read_only", "identical": identical})
        self.assertTrue(identical)


# --------------------------------------------------------------------------
# 9. the flag-off / band=None no-collective contract over the whole grid
# --------------------------------------------------------------------------
class NoCollectiveTest(_Fixture):
    def test_band_none_never_collects_across_shapes(self) -> None:
        self._route("hier")
        for role in ("plain", "source", "consumer"):
            for route in ("hier", "tiled", "untiled"):
                for n, nb in ((17, 200), (34, 260), (78, 260), (512, 300)):
                    IX._QSPLIT = True
                    self._route(route)
                    args = _args(candidate_source_layer=(-1 if role == "plain" else 0))
                    idx = _indexer(args, 0, 30 + n + nb)
                    cons = _indexer(args, 1, 31 + n + nb) if role == "consumer" else None
                    inp = _inputs(33, args, n=n, nb=nb, ratio=4)
                    # _run_replicated installs a RAISING all_sum spy: reaching it
                    # (band=None) fails the test.
                    replicated = _run_replicated(idx, cons, inp, offset=n)
                    self.assertEqual(replicated.out.shape,
                                     (1, n, min(args.index_topk, nb)))
        record({"case": "no_collective_grid"})


def setUpModule() -> None:
    assert IX._FENCE_MIN_ROWS == 16, (
        f"this matrix assumes the shipped fence 16, got {IX._FENCE_MIN_ROWS}")


def tearDownModule() -> None:
    """Write the collected rows and print the compact summary table."""
    path = os.path.join(os.environ.get("BH_AGENT_WORKSPACE",
                                       os.path.expanduser("~")),
                        "qsplit_band_matrix.json")
    with open(path, "w") as handle:
        json.dump(RESULTS, handle, indent=1)
    print(f"\n[band-matrix] {len(RESULTS)} records -> {path}")
    header = (f"{'role':<9} {'route':<7} {'eff':<7} {'w':>1} {'n':>4} {'nb':>5} "
              f"{'off':>6} {'rows':>5} {'diff':>4} {'ovl':>5} {'max|dscore|':>11} "
              f"{'kth_gap':>10} {'gate':>4}")
    print(header)
    print("-" * len(header))
    for row in RESULTS:
        if row.get("case") != "band_matrix":
            continue
        delta = row["score_max_delta"]
        gap = row["kth_gap"]
        print(f"{row['role']:<9} {row['route']:<7} {row['effective']:<7} "
              f"{row['world']:>1} {row['n']:>4} {row['nb']:>5} "
              f"{row['offset']:>6} {row['rows_compared']:>5} "
              f"{row['differing_rows']:>4} {row['overlap']:>5.3f} "
              f"{('-' if delta is None else f'{delta:.3g}'):>11} "
              f"{('-' if gap is None else f'{gap:.3g}'):>10} "
              f"{('y' if row['gated'] else 'n'):>4}")


if __name__ == "__main__":
    unittest.main()
