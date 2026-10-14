# Copyright © 2026 Adam Durham (hermes-gw)
"""QSPLIT-PATH reference-divergence harness — the ``DSV41_INDEXER_QSPLIT=1`` surface.

Why this exists.  The 2026-10-14 audit (``docs/dsv41-indexer-refdiv-audit-2026-10-14.md``,
harness ``audit/dsv41-indexer-refdiv-2026-10-14`` @ ``3255ba4``) reduced the whole
6-commit mlx-lm change surface at ``DSV41_INDEXER_QSPLIT=0`` to ONE hunk (the small-n
wide tile) and measured it selection-neutral over 97 cells — but it explicitly left
the ``QSPLIT=1`` path UNVERIFIED ("the qsplit path's own correctness").  That is the
path that was ENGAGED during the 2026-10-13 degeneration event, its first live A/B
ABORTED with a "token-identity divergence on a banded non-first prefill chunk", and
its post-fix re-verification passed only a 5-probe gate.  This harness replaces that
weak gate with a shape matrix.

The path.  ``DSV41_INDEXER_QSPLIT`` (default OFF) splits the indexer's QUERY side
across the TP ranks: each rank scores / top-ks / emits a contiguous band of the
chunk's query rows, and the full ``cidx [b, n, index_topk]`` int32 selection is
rebuilt from the ranks' zero-padded bands by ONE ``all_sum`` on the TP group
(``query_band`` -> ``Indexer.__call__(band=..., group=...)`` -> ``_finish_rows`` ->
``_reconstruct_rows`` -> ``collective.all_sum``).  The KEY side stays FULL on every
rank.  The load-bearing invariant is **"band row slice == replicated run's row
slice, byte-exact"**.

The five qsplit commits, and what each is tested for here:

| commit | what it fixed | probe |
|---|---|---|
| ``dc0e79c`` | the band itself (query-side split) | A, B, C |
| ``f728bb6`` | typed ``_DistributedGroup`` shim | (typing only) |
| ``e034062`` | D1 call signature + D2 band fence (band rows, not full rows, decide the regime) | C (the D2 hazard, measured) |
| ``de6caa2`` | **project the FULL chunk, then band** — the live divergence class | F, H |
| ``6a5c78f`` | D3: key the tiled/untiled ROUTE on the FULL chunk's rows | B, H |

Sections (each records JSON rows, and the driver asserts its invariants per cell):

* **A** band geometry census: ``query_band`` over the (n, world) grid incl. rank-boundary
  straddles and partial last bands; and the **band-vs-full asymmetry census** — every
  shape-derived quantity the banded and replicated runs resolve differently.
* **B** the prefill-chunk matrix: non-first prefill chunks, band edges, nb corresponding
  to 32K..262K production contexts with a dense 50K..80K inset, TP world 2 and 4, both
  indexer roles, HIER and tiled routes, random-fp4 and tie-lattice keys.
* **C** decode / small-n shapes at 32K..262K contexts (the gate must refuse: no band, no
  collective), plus the **D2 hazard** — an explicitly injected below-fence band, which is
  the one shape where the band's regime differs from the chunk's.
* **D** ``index_topk=512`` edges: nb ~ k, duplicate-index census in the reconstructed
  selection, band disjointness, and the ``_TILED_IMPL=merge`` argpartition merge.
* **E** bf16 small-n rows (``DSV41_INDEXER_L2_FULL=0`` / ``SMALLN_ROW_BF16=1``).
* **F** the ORIGINAL divergence class, made testable: a row-count-dependent projection
  stub (mimicking ``EXL3Linear``'s <=16 / <=64 / else kernel crossover and MLX's
  M-dependent bf16 matmul).  The shipped code must be immune at EVERY perturbation
  size; a faithful model of the pre-fix expression (project the band's own rows) must
  diverge, and the perturbation threshold is measured.
* **G** the residual-class probe: the HIER exact pass derives its strip width from the
  BAND's row count (``indexer.py:796-799`` -> ``hier_strip_for_budget(bsz, n, ...)``),
  while the replicated run derives it from the full chunk.  The merge length is
  ``k + bp*block``, so a strip-width difference is a tie-break difference.
* **H** the D3 read-out: the row count each projection is actually CALLED with, spied at
  the call site (the behavioural form of "the projection happens before the slice").
* **I** non-vacuity controls: a margin-sized perturbation must flip a selection, and the
  tie lattice must put exact ties on rank k.

Production geometry: ``index_n_heads=32``, ``index_head_dim=128``, ``index_topk=512``,
``DSV41_INDEXER_TILE=512``, ``TILE_MB=128``, ``TILE_MIN_NB=0``, ``_FENCE_MIN_ROWS=16``,
``_L2_FULL=1`` (fp32 small-n row), ``_HIER=1``, ``_HIER_BLOCK=8``.  ``dim``/``q_lora_rank``
are shrunk: the score GEMM is ``(M = bsz*n, K = index_head_dim, N = tile)`` and neither
enters it, so the row dtype, row width, top-k and mask grid stay production-exact.

Usage (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python bench/dsv41_indexer_qsplit_refdiv.py \\
        [--out <json>] [--quick]

Exit status 1 if any cell shows a SELECTION divergence (index set, candidate mask bit,
or a non-finite mismatch).  A row-level ulp difference is reported, not a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as IX
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0
from mlx_lm.models.deepseek_v41.model import SharedState

HEADS = 32
HEAD_DIM = 128
INDEX_TOPK = 512
TILE = 512
BUDGET_MB = 128
BLOCK = 8
#: ``_FENCE_MIN_ROWS`` / ``DSV41_INDEXER_HIER_BLOCK`` (must coincide on the source layer).
FENCE = 16
#: nb is scaled: ``context = nb * ratio * NB_SCALE`` for a toy-sized forward.
NB_SCALE = 16

#: max|delta| on the stored score row tolerated before it is a bug rather than an ulp.
SCORE_TOL = 1e-4

RESULTS: list[dict[str, Any]] = []
#: knob values saved/restored around every section (import-read module globals).
KNOBS = ("_QSPLIT", "_TILE", "_TILE_BUDGET", "_TILE_MIN_NB", "_TILE_FORCE", "_HIER",
         "_FENCE_MIN_ROWS", "_SMALLN_WIDE_TILE", "_L2_FULL", "_SMALLN_ROW_BF16",
         "_TILED_IMPL", "_HIER_EXACT_STRIP", "_HIER_EXACT_MB", "_HIER_STRIP",
         "_HIER_OVERFETCH", "_HIER_BLOCK", "_HIER_CONSUMER_SKIP", "_ROW_DTYPE")


def record(row: dict[str, Any]) -> dict[str, Any]:
    RESULTS.append(row)
    print("  " + json.dumps(row, default=str), flush=True)
    return row


def knobs(*, hier: bool = True, qsplit: bool = True, tiled_impl: str = "buffer",
          l2_full: bool = True, smalln_bf16: bool = False, tile: int = TILE,
          budget_mb: int = BUDGET_MB, tiled: bool = False, untiled: bool = False,
          estrip: int = 0) -> None:
    """The production knob set, with one route forced per arm."""
    IX._TILE = 0 if untiled else tile
    IX._TILE_BUDGET = budget_mb << 20
    IX._TILE_MIN_NB = 0
    IX._TILE_FORCE = bool(tiled)
    IX._L2_FULL, IX._SMALLN_ROW_BF16 = l2_full, smalln_bf16
    IX._HIER, IX._QSPLIT, IX._TILED_IMPL = hier, qsplit, tiled_impl
    IX._SMALLN_WIDE_TILE = True
    IX._FENCE_MIN_ROWS = FENCE
    IX._HIER_EXACT_STRIP = estrip


def save_knobs() -> dict[str, Any]:
    return {k: getattr(IX, k) for k in KNOBS}


def restore_knobs(saved: dict[str, Any]) -> None:
    for k, v in saved.items():
        setattr(IX, k, v)


def args_of(role: str, *, ratio: int = 4, topk: int = INDEX_TOPK, block: int = BLOCK,
            topk_blocks: int = 32) -> ModelArgs:
    return ModelArgs(
        dim=256, n_layers=3, n_heads=4, head_dim=128, rope_head_dim=16,
        q_lora_rank=128, o_lora_rank=32, o_groups=4, window_size=8,
        compress_ratios=(ratio, ratio, ratio), kv_source_layers=(0,),
        index_source_layers=(0,), index_n_heads=HEADS, index_head_dim=HEAD_DIM,
        index_topk=topk, max_seq_len=1 << 21,
        candidate_source_layer=(0 if role != "plain" else -1),
        candidate_topk_blocks=topk_blocks, candidate_block_size=block,
    )


def indexer_of(args: ModelArgs, layer_id: int, seed: int) -> IX.Indexer:
    mx.random.seed(seed)
    idx = IX.Indexer(args, layer_id)
    mx.eval(idx.parameters())
    return idx


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def keys_of(seed: int, nb: int, mode: str) -> mx.array:
    """``index_k`` as the cache holds it: fp4_ue8m0 fake-quantized (``publish_keys``).

    ``mode="ties"`` builds the columns from only 97 distinct key rows, so ~nb/97
    columns of every query share an EXACT score — that is what puts exact ties on
    rank k without needing a contrived key.
    """
    rng = np.random.default_rng(seed)
    if mode == "ties":
        p = 97
        base = rng.standard_normal((p, HEAD_DIM)).astype(np.float32)
        raw = mx.array(base[np.arange(nb) % p][None])
    elif mode == "same":
        # EVERY column shares one key: every column of a row then scores identically,
        # so the top-k is an arbitrary k-subset of the visible columns -- the maximal
        # exact tie, and the only structure that can expose a layout-dependent tie pick.
        base = rng.standard_normal((1, HEAD_DIM)).astype(np.float32)
        raw = mx.array(np.repeat(base, nb, axis=0)[None])
    elif mode == "onehot":
        # A CONTROLLED wide exact tie that survives fp4 quantization: the keys are
        # +e_0 at 900 scattered columns and -e_0 everywhere else, so every block has the
        # same max magnitude (one scale) and each query's score is ``relu(+-q_0)*w`` --
        # 900 columns score exactly the same positive value and every other column
        # relu-clips to exactly 0, strictly below.  The k=512 boundary then sits INSIDE a
        # 900-way exact tie: the regime that separates a one-pass argpartition from a
        # multi-round merge (see :func:`argpartition_tie_probe`).
        keys = np.zeros((nb, HEAD_DIM), dtype=np.float32)
        keys[:, 0] = -1.0
        keys[np.linspace(0, nb - 1, min(900, nb)).astype(int), 0] = 1.0
        raw = mx.array(keys[None])
    else:
        raw = mx.array(rng.standard_normal((1, nb, HEAD_DIM)).astype(np.float32))
    return fake_quant_fp4_ue8m0(raw, 32)


def inputs_for(seed: int, role: str, n: int, nb: int, mode: str = "lattice",
                *, ratio: int = 4, phase: int = 0, b: int = 1, topk: int = INDEX_TOPK,
                block: int = BLOCK):
    """(idx, cons, x, qr, start_pos, freqvec, index_k) for a NON-FIRST prefill chunk.

    ``start_pos = nb*ratio - n + phase``: the chunk that ENDS at context ``nb*ratio``,
    i.e. a mid-prefill chunk (never chunk 0 unless nb==n//ratio).  ``phase`` walks
    ``start_pos % ratio`` so the chunk's first row lands on and just off a group
    boundary; a non-zero phase also moves the band seam against the block grid.
    """
    args = args_of(role, ratio=ratio, topk=topk, block=block)
    idx = indexer_of(args, 0, seed)
    cons = indexer_of(args, 1, seed + 1) if role == "consumer" else None
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((b, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((b, n, args.q_lora_rank)).astype(np.float32))
    ik = keys_of(seed, nb, mode)
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    start_pos = max(0, nb * ratio - n + phase)
    return idx, cons, x, qr, start_pos, fv, ik


class Group:
    """Minimal stand-in for ``mx.distributed.Group`` (rank / size only)."""

    def __init__(self, rank: int, world: int = 2):
        self._rank, self._world = rank, world

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._world


# ---------------------------------------------------------------------------
# the two runs
# ---------------------------------------------------------------------------
class _RowSpy:
    """Record the ``[b, n, nb]`` stored row each ``topk_from_row`` call sees.

    The tiled/buffer route hands its finished row to ``indexer.topk_from_row``; the
    hierarchical and untiled routes have their own inline top-k and never call it,
    so the caller falls back to the dense ``_score_body_eager`` recomputation.
    """

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self._real = IX.topk_from_row

    def __enter__(self) -> "_RowSpy":
        real = self._real

        def wrapper(s: mx.array, k: int):                      # noqa: ANN202
            self.rows.append(np.array(s.astype(mx.float32)))
            return real(s, k)

        IX.topk_from_row = wrapper
        return self

    def __exit__(self, *exc: object) -> None:
        IX.topk_from_row = self._real


def _forward(idx, inp, shared, *, band, group, offset: int):
    _, _, x, qr, start_pos, fv, ik = inp
    return idx(x, qr, start_pos, offset, fv, ik, shared, band=band, group=group)


def run_replicated(idx, cons, inp, *, offset: int, forbid_collect: bool = True) -> dict[str, Any]:
    """The replicated (``band=None``) run.  Must touch NO collective."""
    real = IX._coll.all_sum
    if forbid_collect:
        def boom(*_a: object, **_k: object):                    # noqa: ANN202
            raise AssertionError("band=None must never issue a collective")
        IX._coll.all_sum = boom
    try:
        with _RowSpy() as spy:
            shared = SharedState()
            n0 = len(spy.rows)
            out = _forward(idx, inp, shared, band=None, group=None, offset=offset)
            mx.eval(out)
            cons_out, cons_row = None, None
            if cons is not None:
                n1 = len(spy.rows)
                cons_out = _forward(cons, inp, shared, band=None, group=None, offset=offset)
                mx.eval(cons_out)
                cons_row = spy.rows[n1] if len(spy.rows) > n1 else None
            row = spy.rows[n0] if len(spy.rows) > n0 else None
    finally:
        IX._coll.all_sum = real
    return {"out": np.array(out),
            "cons_out": None if cons_out is None else np.array(cons_out),
            "shared": shared, "row": row, "cons_row": cons_row}


def run_banded(idx, cons, inp, *, bands, offset: int, world: int) -> dict[str, Any]:
    """One forward per rank; the spied collective is an identity, so the pad returns."""
    real = IX._coll.all_sum
    captured: dict[str, mx.array] = {}
    pads: list[np.ndarray] = []
    padded: list[np.ndarray] = []
    cons_pads: list[np.ndarray] = []
    masks: list[np.ndarray | None] = []
    rows: list[np.ndarray | None] = []
    cons_rows: list[np.ndarray | None] = []

    def identity(t, group=None):                               # noqa: ANN001, ANN202
        captured["padded"] = t
        return t

    IX._coll.all_sum = identity
    try:
        with _RowSpy() as spy:
            for r, (lo, hi) in enumerate(bands):
                shared = SharedState()
                n0 = len(spy.rows)
                out = _forward(idx, inp, shared, band=(lo, hi),
                               group=Group(r, world), offset=offset)
                pad = np.array(captured["padded"])
                mx.eval(out)
                pads.append(np.array(out))
                padded.append(pad)
                masks.append(None if shared.candidates is None else np.array(shared.candidates))
                rows.append(spy.rows[n0] if len(spy.rows) > n0 else None)
                if cons is not None:
                    n1 = len(spy.rows)
                    cons_out = _forward(cons, inp, shared, band=(lo, hi),
                                        group=Group(r, world), offset=offset)
                    mx.eval(cons_out)
                    cons_pads.append(np.array(cons_out))
                    cons_rows.append(spy.rows[n1] if len(spy.rows) > n1 else None)
    finally:
        IX._coll.all_sum = real
    return {"pads": pads, "padded": padded, "cons_pads": cons_pads, "masks": masks,
            "rows": rows, "cons_rows": cons_rows}


def sum_pads(pads: list[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(pads[0])
    for pad in pads:
        total = total + pad
    return total


# ---------------------------------------------------------------------------
# dense references / metrics
# ---------------------------------------------------------------------------
def dense_row(idx: IX.Indexer, inp, *, band: tuple[int, int] | None,
              cand: np.ndarray | None = None) -> np.ndarray:
    """The indexer's own ``[b, n_band, nb]`` score row, recomputed in dense fp32.

    Mirrors ``Indexer.__call__`` line for line through the SHIPPED score body: project
    the FULL chunk, slice the band AFTER the projections, rope at the band's absolute
    positions, fp4-quantize, collapse the heads with the shipped expression, then apply
    the visibility fence.  Any row-count dependence of the projections or the rope/quant
    shows up as a band-vs-full slice difference here, which is exactly what the
    ``de6caa2`` invariant claims is impossible.
    """
    _, _, x, qr, start_pos, fv, ik = inp
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
    q = IX.rope_tail(q, idx.rope_head_dim, *IX.cos_sin_at(fv, qpos))
    q = IX.fake_quant_fp4_ue8m0(q, 32)
    w = w_full * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((qpos + 1) // idx.ratio)[:, None]
    scores = IX._score_body_eager(q.astype(mx.float32), ik.astype(mx.float32),
                                   w.astype(mx.float32))
    cols = mx.arange(ik.shape[1], dtype=mx.int32)
    scores = mx.where(cols[None, :] < lens, scores, IX.NEG_INF)
    if cand is not None:
        scores = mx.where(mx.array(cand), scores, IX.NEG_INF)
    return np.array(scores)


def prefix_banded_cidx(idx: IX.Indexer, inp, band: tuple[int, int], offset: int) -> np.ndarray:
    """A faithful model of the PRE-``de6caa2`` expression: project the BAND's own rows.

    Transcribed from the commit's own diff — ``xr/qrr = x[:, lo:hi], qr[:, lo:hi]`` then
    ``wq_b(qrr)`` / ``weights_proj(xr)`` — then the same rope, fp4 quant, untiled score
    and untiled top-k the replicated run uses.  With a row-count-INVARIANT projection
    this is identical to the replicated run (which is why the pre-fix bug was invisible
    offline); with a row-count-DEPENDENT one it is not.
    """
    _, _, x, qr, start_pos, fv, ik = inp
    bsz, _n_full, _ = x.shape
    lo, hi = band
    xr, qrr = x[:, lo:hi, :], qr[:, lo:hi, :]
    qpos = mx.arange(start_pos + lo, start_pos + hi)
    n = qpos.shape[0]
    q = idx.wq_b(qrr).reshape(bsz, n, idx.n_heads, idx.head_dim)
    q = IX.rope_tail(q, idx.rope_head_dim, *IX.cos_sin_at(fv, qpos))
    q = IX.fake_quant_fp4_ue8m0(q, 32)
    w = idx.weights_proj(xr) * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((qpos + 1) // idx.ratio)[:, None]
    scores = IX._score_body_eager(q.astype(mx.float32), ik.astype(mx.float32),
                                   w.astype(mx.float32))
    # the shipped untiled branch's own cast, applied on the BAND's row count
    scores = scores.astype(IX._row_dtype(n))
    cols = mx.arange(ik.shape[1], dtype=mx.int32)
    scores = mx.where(cols[None, :] < lens, scores, IX.NEG_INF)
    k = min(idx.index_topk, ik.shape[1])
    i = mx.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(mx.int32)
    i = mx.sort(i, axis=-1)
    vis = i < lens.astype(mx.int32)[None]
    return np.array(mx.where(vis, i + offset, mx.array(-1, mx.int32)))


class RowKernelLinear:
    """A projection whose OUTPUT depends on the number of rows it is CALLED with.

    Models the two real mechanisms the ``de6caa2`` fix exists for: ``EXL3Linear``'s
    row-count kernel crossover (<=16 fused simd GEMM, <=64 fused trellis GEMM, else
    decode_full + native matmul) and MLX's M-dependent native bf16 matmul.  The
    perturbation is a relative scale per bucket — i.e. a stand-in for "a different
    kernel computed this row", which is exactly what differed live (2027 q elements
    at n=78 -> band 39, across the 64-row crossover).
    """

    def __init__(self, lin, eps: float = 1e-4, boundaries: tuple[int, ...] = (16, 64)):
        self._lin, self._eps, self._b = lin, float(eps), boundaries
        self.calls: list[int] = []

    def _bucket(self, rows: int) -> int:
        bucket = 0
        for t in self._b:
            if rows <= t:
                return bucket
            bucket += 1
        return bucket

    def __call__(self, x: mx.array) -> mx.array:
        y = self._lin(x)
        rows = int(np.prod(x.shape[:-1]))
        self.calls.append(rows)
        bucket = self._bucket(rows)
        return y if bucket == 0 else y * (1.0 + bucket * self._eps)


class RowCountSpy:
    """Wrap a projection so it records the ROW COUNT it is actually called with."""

    def __init__(self, lin):
        self._lin = lin
        self.rows: list[int] = []

    def __call__(self, x: mx.array) -> mx.array:
        self.rows.append(int(np.prod(x.shape[:-1])))
        return self._lin(x)


def overlap_clean(got: np.ndarray, ref: np.ndarray) -> tuple[float | None, int, int]:
    """Mean per-row top-k overlap over rows holding k DISTINCT indices (+ their count).

    A fully-masked row is k copies of the ``-1`` sentinel, so two identical rows would
    score 1/k there; those rows are excluded and counted instead (the sibling suites'
    ``_overlap_clean`` idiom).
    """
    a, b = got.reshape(-1, got.shape[-1]), ref.reshape(-1, ref.shape[-1])
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


def row_delta(got: np.ndarray, ref: np.ndarray) -> tuple[float, int]:
    """max|got - ref| over finite entries, plus non-finite mismatches.

    ``-inf - -inf`` is a MATCH (masked columns agree), so those pairs are zeroed first;
    anything non-finite that is not a matching pair of infinities is a real divergence
    and is counted separately.
    """
    with np.errstate(invalid="ignore"):
        delta = got.astype(np.float64) - ref.astype(np.float64)
    both_inf = np.isinf(got) & np.isinf(ref) & (np.sign(got) == np.sign(ref))
    bad = int((~np.isfinite(delta) & ~both_inf).sum())
    delta = np.where(both_inf, 0.0, delta)
    finite = delta[np.isfinite(delta)]
    return (float(np.abs(finite).max()) if finite.size else 0.0), bad


def kth_gap(row: np.ndarray, k: int) -> tuple[float | None, int]:
    """k-th boundary gap ``sorted[k-1] - sorted[k]`` off a dense fp32 row (+ tie rows)."""
    if k >= row.shape[-1]:
        return None, 0
    srt = np.sort(row, axis=-1)[..., ::-1]
    with np.errstate(invalid="ignore"):
        gap = srt[..., k - 1] - srt[..., k]
    finite = gap[np.isfinite(gap)]
    if finite.size == 0:
        return None, 0
    return float(finite.min()), int((finite <= 0.0).sum())


def dup_real(idx_arr: np.ndarray, offset: int) -> int:
    """Rows carrying a DUPLICATED real index (>= offset); ``-1`` sentinels excluded."""
    real = idx_arr >= offset
    counted = np.where(real, idx_arr, -1)
    dup = 0
    flat = counted.reshape(-1, counted.shape[-1])
    for row in flat:
        vals = row[row >= 0]
        dup += int(vals.size - np.unique(vals).size)
    return dup


# ---------------------------------------------------------------------------
# A. band geometry + the band-vs-full asymmetry census
# ---------------------------------------------------------------------------
def estrip_of(n: int, nb: int) -> tuple[int, int, int]:
    """(estrip, blocks-per-strip, strips) at the production exact-pass budget."""
    est = IX.hier_strip_for_budget(1, n, HEAD_DIM, IX._TILE_BUDGET, block=IX._HIER_BLOCK)
    bp = max(1, est // IX._HIER_BLOCK)
    nb_blocks = -(-nb // IX._HIER_BLOCK)
    kb = min(nb_blocks, INDEX_TOPK + IX._HIER_OVERFETCH)
    return est, bp, -(-kb // bp)


def section_a() -> dict[str, Any]:
    print("== A. band geometry + band-vs-full asymmetry census ==", flush=True)
    saved = save_knobs()
    try:
        knobs()
        fenced = 0
        banded = 0
        bad_gate = 0
        straddle_samples: list[dict[str, Any]] = []
        for world in (2, 4):
            for n in range(1, 40):
                for rank in range(world):
                    b = IX.query_band(n, Group(rank, world))
                    want_band = (n % world == 0) and (n // world > FENCE)
                    banded += int(b is not None)
                    fenced += int(b is None)
                    bad_gate += int((b is not None) != want_band)
                    if n % world == 0 and 14 <= n // world <= 18:
                        straddle_samples.append(dict(world=world, n=n, rank=rank,
                                                     band=None if b is None else list(b)))
        # the D2 fence straddles: the first banded n is world*(FENCE+1)
        first_banded = {w_: next(n for n in range(1, 200)
                                 if IX.query_band(n, Group(0, w_)) is not None)
                        for w_ in (2, 4)}
        first_banded_odd = {world_f: [n for n in (world_f * FENCE + 1, world_f * FENCE + 2)
                                     if IX.query_band(n, Group(0, world_f)) is not None]
                            for world_f in (2, 4)}
        # the band-vs-full asymmetry census: every shape-derived quantity each arm
        # resolves.  dtype and the HIER predicate MUST agree (that is what the D2 fence
        # buys); the ROUTE must agree because D3 passes n_full (the pre-D3 rule, which
        # passed the band's own rows, is transcribed here and counted); estrip/strip
        # count are NOT covered by either fix -- section G measures what that costs.
        asym = []
        for world in (2, 4):
            for n in (34, 66, 78, 130, 132, 512, 2048):
                if n % world:
                    continue
                rows = n // world
                if rows <= FENCE:
                    continue
                for nb in (260, 1024, 3000, 8192):
                    align = BLOCK
                    route_d3 = IX.tiled(1, n, HEADS, nb, align) < nb        # shipped (D3)
                    route_pre_d3 = IX.tiled(1, rows, HEADS, nb, align) < nb  # pre-6a5c78f
                    d_band = IX._row_dtype(rows)
                    d_full = IX._row_dtype(n)
                    e_band, bp_band, s_band = estrip_of(rows, nb)
                    e_full, bp_full, s_full = estrip_of(n, nb)
                    asym.append(dict(world=world, n=n, band=[0, rows], nb=nb,
                                     estrip_band=e_band, estrip_full=e_full,
                                     strips_band=s_band, strips_full=s_full,
                                     dtype_band=str(d_band), dtype_full=str(d_full),
                                     dtype_agree=bool(d_band == d_full),
                                     hier_band=rows > FENCE, hier_full=n > FENCE,
                                     route_d3=route_d3, route_pre_d3=route_pre_d3))
        n_asym_dtype = sum(1 for a in asym if not a["dtype_agree"])
        n_asym_route_d3 = sum(1 for a in asym if a["route_d3"] != a["route_pre_d3"])
        n_asym_estrip = sum(1 for a in asym if a["estrip_band"] != a["estrip_full"])
        n_asym_strips = sum(1 for a in asym if a["strips_band"] != a["strips_full"])
        examples = [dict(n=a["n"], world=a["world"], nb=a["nb"],
                         estrip_band=a["estrip_band"], estrip_full=a["estrip_full"],
                         strips_band=a["strips_band"], strips_full=a["strips_full"])
                    for a in asym if a["strips_band"] != a["strips_full"]][:6]
    finally:
        restore_knobs(saved)
    out = record(dict(
        case="band_geometry", fence=FENCE, banded_shapes=banded, fenced_shapes=fenced,
        gate_mismatch=bad_gate, first_banded_n=first_banded,
        odd_n_banded=first_banded_odd,
        straddles=straddle_samples[:8],
        asym_cells=len(asym), asym_dtype_mismatch=n_asym_dtype,
        asym_pre_d3_route_mismatch=n_asym_route_d3,
        asym_estrip_mismatch=n_asym_estrip,
        asym_strip_count_mismatch=n_asym_strips, asym_examples=examples))
    assert bad_gate == 0, "query_band engages exactly iff n%world==0 and n//world>16"
    assert banded > 0, "no shape in the grid bands at all"
    assert n_asym_dtype == 0, "D2: the band must resolve the SAME row dtype as the chunk"
    assert n_asym_estrip == len(asym), (
        "expected the band-derived estrip to differ from the chunk-derived one everywhere")
    return out


# ---------------------------------------------------------------------------
# the cell driver
# ---------------------------------------------------------------------------
class CellResult(dict):
    pass


def cell(*, role: str, route: str, world: int, n: int, nb: int, seed: int,
         offset: int | None = None, phase: int = 0, ratio: int = 4,
         mode: str = "lattice", b: int = 1, tag: str = "", topk: int = INDEX_TOPK,
         tiled_impl: str = "buffer", l2_full: bool = True, smalln_bf16: bool = False,
         explicit_band: bool = False, check: bool = True,
         expect_d2_hazard: bool = False, record_row: bool = True,
         row_kernel_eps: float | None = None) -> dict[str, Any]:
    """One matrix cell: run both arms, record every level, then assert.

    ``route``: "hier" | "tiled" | "untiled".  A cell the gate refuses (``query_band``
    -> None) is recorded with ``gated=False`` and pinned to the replicated no-collective
    path instead.

    ``row_kernel_eps`` installs :class:`RowKernelLinear` on both indexers' query
    projections — a row-count-DEPENDENT kernel, i.e. the class that actually broke live.
    This is the arm a plain ``nn.Linear`` harness is blind to: it must change nothing
    while the shipped (``de6caa2``) call order holds, and it is the arm that goes red if
    a future change ever runs a projection on the band's own rows.
    """
    if route == "hier":
        knobs(hier=True, tiled_impl=tiled_impl, l2_full=l2_full,
              smalln_bf16=smalln_bf16)
    elif route == "tiled":
        knobs(hier=False, tiled=True, tiled_impl=tiled_impl, l2_full=l2_full,
              smalln_bf16=smalln_bf16)
    else:
        knobs(hier=False, untiled=True, tiled_impl=tiled_impl, l2_full=l2_full,
              smalln_bf16=smalln_bf16)
    idx, cons, x, qr, start_pos, fv, ik = inputs_for(
        seed, role, n, nb, mode, ratio=ratio, phase=phase, b=b, topk=topk)
    if row_kernel_eps is not None:
        for module in (idx, cons):
            if module is None:
                continue
            setattr(module, "wq_b", RowKernelLinear(module.wq_b, eps=row_kernel_eps))
            setattr(module, "weights_proj",
                    RowKernelLinear(module.weights_proj, eps=row_kernel_eps))
    inp = (idx, cons, x, qr, start_pos, fv, ik)
    if offset is None:
        offset = start_pos + n

    raw_bands: list[tuple[int, int] | None] = [IX.query_band(n, Group(r, world))
                                              for r in range(world)]
    if explicit_band:
        rows = n // world
        raw_bands = [(r * rows, (r + 1) * rows) for r in range(world)]
    if raw_bands and all(bd is not None for bd in raw_bands):
        bands: list[tuple[int, int]] | None = [bd for bd in raw_bands if bd is not None]
    else:
        bands = None
    gated = bands is not None

    replicated = run_replicated(idx, cons, inp, offset=offset)
    score_idx = cons if role == "consumer" else idx
    role_out = replicated["cons_out"] if role == "consumer" else replicated["out"]
    role_row = replicated["cons_row"] if role == "consumer" else replicated["row"]
    full_cand = (None if replicated["shared"].candidates is None
                 else np.array(replicated["shared"].candidates))
    est_b, _, str_b = estrip_of(n // world, nb)
    est_f, _, str_f = estrip_of(n, nb)

    row: dict[str, Any] = dict(
        case="qsplit_cell", tag=tag, role=role, route=route, world=world, b=b, n=n,
        nb=nb, ratio=ratio, phase=phase, start_pos=int(start_pos), offset=int(offset),
        offset_pos=int(start_pos) + n, keys=mode, topk=topk, k=min(topk, nb),
        tiled_impl=tiled_impl, l2_full=l2_full, smalln_bf16=smalln_bf16,
        context=nb * ratio * NB_SCALE, nb_scale=NB_SCALE, gated=bool(gated),
        band_rows_compared=b * (n // world if n % world == 0 else 0),
        rows_compared=b * n,
        dtype_band=str(IX._row_dtype(n // world)), dtype_full=str(IX._row_dtype(n)),
        estrip_band=est_b, estrip_full=est_f, strips_band=str_b, strips_full=str_f,
    )

    if not gated:
        row.update(band=None, differing_rows=0, overlap_clean=1.0, score_max_delta=None,
                   nonfinite_mismatch=0, cand_rows=None, kth_gap=None, kth_tied_rows=None,
                   padded_zero_outside=None, band_slice_equal=None,
                   dup_real_indices=None, max_contributors=None, d2_hazard_rows=None)
        if record_row:
            record(row)
        assert not explicit_band, "an explicit band must engage"
        return row

    banded = run_banded(idx, cons, inp, bands=bands, offset=offset, world=world)
    role_pads = banded["cons_pads"] if role == "consumer" else banded["pads"]
    recon = sum_pads(role_pads)
    diff_rows = int((recon != role_out).any(-1).sum())
    ovl, ovl_rows, degenerate = overlap_clean(recon, role_out)

    # per-rank: the pad is zero outside its own rows, its in-band slice is row-identical
    # to the replicated run, and no element has two contributors.
    zero_outside = True
    slice_equal = True
    cons_slice_equal = True
    support = np.zeros(recon.shape, np.int32)
    for r, (lo, hi) in enumerate(bands):
        padded = banded["padded"][r]
        mask = np.ones(padded.shape[1], bool)
        mask[lo:hi] = False
        zero_outside &= bool(np.all(padded[:, mask, :] == 0))
        slice_equal &= bool(np.array_equal(padded[:, lo:hi, :], replicated["out"][:, lo:hi, :]))
        support = support + (padded != 0).astype(np.int32)
        if role == "consumer":
            cons_slice_equal &= bool(np.array_equal(
                banded["cons_pads"][r][:, lo:hi, :], role_out[:, lo:hi, :]))

    # score-row level: band-slice vs full-slice of the SHIPPED score body (the
    # de6caa2 "band row == replicated row" invariant), plus the shipped stored row
    # where the tiled/buffer route exposes it.
    delta = 0.0
    nonfinite = 0
    ref_full = dense_row(score_idx, inp, band=None,
                         cand=(full_cand if role == "consumer" else None))
    for lo, hi in bands:
        ref_band = dense_row(score_idx, inp, band=(lo, hi),
                             cand=(banded["masks"][bands.index((lo, hi))]
                                   if role == "consumer" else None))
        mag, bad = row_delta(ref_band, ref_full[:, lo:hi, :])
        delta, nonfinite = max(delta, mag), nonfinite + bad
    stored = None
    captured = banded["cons_rows"] if role == "consumer" else banded["rows"]
    if role_row is not None and all(r is not None for r in captured):
        stored = 0.0
        for r, (lo, hi) in enumerate(bands):
            mag, bad = row_delta(captured[r], role_row[:, lo:hi, :])
            stored, nonfinite = max(stored, mag), nonfinite + bad

    cand_rows = None
    if role == "source" and full_cand is not None:
        cand_rows = sum(int((banded["masks"][r] != full_cand[:, lo:hi, :]).any(-1).sum())
                        for r, (lo, hi) in enumerate(bands))

    gap, tied = kth_gap(ref_full, row["k"])
    d2 = None
    if n // world <= FENCE and n > FENCE:
        # the D2 class: the band's own rows are below the fence while the chunk's are
        # above it, so band and chunk resolve DIFFERENT regimes.  Measured, not asserted.
        d2 = diff_rows
    row.update(
        band=[[int(lo), int(hi)] for lo, hi in bands],
        differing_rows=diff_rows, overlap_clean=ovl, overlap_clean_rows=ovl_rows,
        degenerate_rows=degenerate, row_src=("stored" if stored is not None else "fp32"),
        score_max_delta=(stored if stored is not None else delta),
        score_delta_fp32=delta, nonfinite_mismatch=nonfinite, cand_rows=cand_rows,
        kth_gap=gap, kth_tied_rows=tied, padded_zero_outside=bool(zero_outside),
        band_slice_equal=bool(slice_equal), cons_band_slice_equal=bool(cons_slice_equal),
        dup_real_indices=dup_real(recon, int(offset)),
        max_contributors=int(support.max()), d2_hazard_rows=d2,
    )
    if record_row:
        record(row)
    if check:
        if expect_d2_hazard:
            assert bands is not None
        assert int(support.max()) == 1, "an element had two band contributors"
        assert zero_outside, "a pad wrote outside its own band"
        assert slice_equal, "the band slice != the replicated run's rows"
        assert cons_slice_equal, "the consumer band slice != the replicated run's rows"
        assert (diff_rows == 0) or expect_d2_hazard, (
            f"banded != replicated: {diff_rows} rows "
            f"({role}/{route}/world={world}/n={n}/nb={nb}/mode={mode})")
        if ovl is not None:
            assert ovl == 1.0, f"non-degenerate top-k overlap {ovl} on {ovl_rows} rows"
        assert row["dup_real_indices"] == 0, "the reconstructed selection has duplicate indices"
        assert float(row["score_max_delta"]) <= SCORE_TOL, (
            f"score row diverged by {row['score_max_delta']}")
        if cand_rows is not None:
            assert cand_rows == 0, "the published candidate mask differs on the band"
    return row


#: contexts a cell's nb is derived from (dense across the live 50K..80K band).
CONTEXTS = (32_768, 42_000, 49_152, 51_200, 53_248, 55_296, 57_344, 61_440, 65_536,
            69_632, 73_728, 77_824, 81_920, 98_304, 131_072, 196_608, 262_144)


def nb_for(context: int, ratio: int = 4) -> int:
    return max(1, context // (ratio * NB_SCALE))


# ---------------------------------------------------------------------------
# B. the prefill-chunk matrix
# ---------------------------------------------------------------------------
def section_b(quick: bool) -> list[dict[str, Any]]:
    print("== B. non-first prefill chunks / band edges (HIER + tiled) ==", flush=True)
    saved = save_knobs()
    out: list[dict[str, Any]] = []
    try:
        cells: list[dict[str, Any]] = []
        n_grid = (34, 66, 78, 130, 512) if quick else (34, 66, 78, 130, 132, 512)
        ctx_grid = (32_768, 57_344, 77_824, 131_072, 262_144) if quick else CONTEXTS
        for n in n_grid:
            for world in (2, 4):
                for ctx in ctx_grid:
                    cells.append(dict(role="source", route="hier", world=world, n=n,
                                      nb=nb_for(ctx), phase=0, mode="lattice"))
                for phase in (1, 3):
                    cells.append(dict(role="source", route="hier", world=world, n=n,
                                      nb=nb_for(57_344), phase=phase, mode="lattice"))
        cells.append(dict(role="consumer", route="hier", world=2, n=78, nb=nb_for(57_344),
                          phase=0, mode="lattice"))
        cells.append(dict(role="consumer", route="hier", world=4, n=132, nb=nb_for(77_824),
                          phase=0, mode="ties"))
        cells.append(dict(role="source", route="hier", world=2, n=34, nb=nb_for(51_200),
                          phase=1, mode="ties"))
        cells.append(dict(role="plain", route="hier", world=2, n=512, nb=520, phase=0,
                          mode="ties"))
        for n in (520, 580, 1024) if quick else (520, 580, 1024, 2048):
            for world in (2, 4):
                cells.append(dict(role="source", route="tiled", world=world, n=n,
                                  nb=max(n + 8, 600), phase=0, mode="lattice"))
        for i, spec in enumerate(cells):
            out.append(cell(seed=10_000 + i, tag="prefill_matrix", **spec))
    finally:
        restore_knobs(saved)
    return out


# ---------------------------------------------------------------------------
# C. decode / small-n shapes + the D2 hazard
# ---------------------------------------------------------------------------
def section_c() -> list[dict[str, Any]]:
    print("== C. decode / verify shapes at 32K..262K + the D2 hazard ==", flush=True)
    saved = save_knobs()
    out: list[dict[str, Any]] = []
    try:
        cells: list[dict[str, Any]] = []
        small = (1, 2, 3, 4, 5, 8, 12, 16, 17, 32, 33)
        ctx_grid = (32_768, 49_152, 57_344, 65_536, 77_824, 98_304, 196_608, 262_144)
        for n in small:
            for ctx in ctx_grid:
                for route in ("untiled", "tiled"):
                    cells.append(dict(role="source", route=route, world=2, n=n,
                                      nb=nb_for(ctx), phase=0, mode="lattice",
                                      expect_gate_refusal=True))
        for i, spec in enumerate(cells):
            spec = dict(spec)
            spec.pop("expect_gate_refusal", None)
            out.append(cell(seed=20_000 + i, tag="decode_smalln", **spec))
        # the D2 hazard: a band injected BELOW the fence on a chunk ABOVE it.  The gate
        # refuses this shape, so the cell is run explicitly and the divergence is
        # MEASURED (``d2_hazard_rows``) rather than asserted away -- it is the evidence
        # for why the fence keys on the band's rows.
        for i, n in enumerate((20, 26, 34)):
            for route in ("hier", "tiled"):
                out.append(cell(seed=21_000 + i, tag="d2_hazard", role="source",
                                route=route, world=2, n=n, nb=nb_for(131_072),
                                phase=0, mode="ties", explicit_band=True,
                                check=False, expect_d2_hazard=True))
    finally:
        restore_knobs(saved)
    return out


# ---------------------------------------------------------------------------
# D. top-k = 512 edges, duplicates, the merge implementation
# ---------------------------------------------------------------------------
def section_d() -> list[dict[str, Any]]:
    print("== D. top-k edge, duplicate indices, the merge impl ==", flush=True)
    saved = save_knobs()
    out: list[dict[str, Any]] = []
    try:
        # nb ~ k: the whole row is the selection (k = min(512, nb)); nb > k: the k-th
        # boundary is real.  A 512-wide argpartition over 512..520 columns is the
        # degenerate edge the audit named (all-visible rows).
        for nb in (512, 513, 515, 520, 1024):
            for world in (2, 4):
                for mode in ("lattice", "ties"):
                    out.append(cell(seed=30_000 + nb + world, tag="topk_edge",
                                    role="source", route="tiled", world=world, n=1024,
                                    nb=nb, mode=mode, phase=0))
        for impl in ("buffer", "merge"):
            for mode in ("lattice", "ties"):
                for world in (2, 4):
                    out.append(cell(seed=31_000 + world, tag="merge_impl",
                                    role="source", route="tiled", world=world, n=1024,
                                    nb=2048, mode=mode, tiled_impl=impl, phase=0))
        for mode in ("ties", "lattice"):
            out.append(cell(seed=32_000, tag="merge_impl", role="consumer", route="tiled",
                            world=2, n=1024, nb=1200, mode=mode,
                            tiled_impl="merge", phase=0))
    finally:
        restore_knobs(saved)
    return out


# ---------------------------------------------------------------------------
# E. bf16 small-n rows
# ---------------------------------------------------------------------------
def section_e() -> list[dict[str, Any]]:
    print("== E. bf16 small-n rows (L2_FULL=0 / SMALLN_ROW_BF16=1) ==", flush=True)
    saved = save_knobs()
    out: list[dict[str, Any]] = []
    try:
        for l2_full, sn_bf16 in ((True, True), (False, False), (False, True)):
            for n, world in ((78, 2), (132, 4), (512, 2)):
                for route in ("hier", "tiled"):
                    out.append(cell(seed=40_000 + n + world, tag="bf16_smalln",
                                    role="source", route=route, world=world, n=n,
                                    nb=nb_for(131_072), mode="ties", phase=0,
                                    l2_full=l2_full, smalln_bf16=sn_bf16))
    finally:
        restore_knobs(saved)
    return out


# ---------------------------------------------------------------------------
# F. the ORIGINAL divergence class: a row-count-dependent projection
# ---------------------------------------------------------------------------
def section_f() -> dict[str, Any]:
    print("== F. row-count-dependent projection (the live divergence class) ==", flush=True)
    saved = save_knobs()
    try:
        knobs(hier=False, untiled=True)
        n, world, nb = 78, 2, 1024
        idx, _, x, qr, start_pos, fv, ik = inputs_for(50_001, "plain", n, nb, "lattice")
        inp = (idx, None, x, qr, start_pos, fv, ik)
        offset = start_pos + n
        bands_f: list[tuple[int, int]] = [bd for bd in
                                         (IX.query_band(n, Group(r, world))
                                          for r in range(world)) if bd is not None]
        assert len(bands_f) == world
        replicated = run_replicated(idx, None, inp, offset=offset)["out"]

        # (1) non-vacuity of the STUB itself: the same row, projected with 39 rows vs
        #     78, must differ (the EXL3 crossover in vivo).
        plain_wq, plain_wp = idx.wq_b, idx.weights_proj
        stub_wq = RowKernelLinear(plain_wq, eps=1e-4)
        setattr(idx, "wq_b", stub_wq)
        q_band_rows = np.array(idx.wq_b(qr[:, 0:39, :]))
        q_full_rows = np.array(idx.wq_b(qr))[:, 0:39, :]
        stub_q_differing = int((q_band_rows != q_full_rows).sum())
        stub_q_total = int(q_band_rows.size)

        def pre_fix_model(eps: float | None) -> np.ndarray:
            """The pre-``de6caa2`` banded expression, padded back into the chunk's rows.

            ``eps=None`` installs the PLAIN (row-count-invariant) projections — the
            control that shows why the live bug was invisible to every offline harness
            that used a plain ``nn.Linear``.
            """
            if eps is None:
                setattr(idx, "wq_b", plain_wq)
                setattr(idx, "weights_proj", plain_wp)
            else:
                setattr(idx, "wq_b", RowKernelLinear(plain_wq, eps=eps))
                setattr(idx, "weights_proj", RowKernelLinear(plain_wp, eps=eps))
            full = np.full_like(replicated, -1)
            for lo, hi in bands_f:
                full[:, lo:hi, :] = prefix_banded_cidx(idx, inp, (lo, hi), offset)
            return full

        # (2) IMMUNITY: with the stub installed, the SHIPPED banded path must still be
        #     byte-identical to the replicated run -- for EVERY perturbation size,
        #     because the two arms now call the projections on identical inputs with an
        #     identical row count.  (eps is a stand-in for "a different kernel computed
        #     this row"; it is swept up to a magnitude-sized 1e-2 to show the immunity is
        #     structural, not numerical.)
        immunity = []
        for eps in (1e-7, 1e-5, 1e-4, 1e-3, 1e-2):
            setattr(idx, "wq_b", RowKernelLinear(plain_wq, eps=eps))
            setattr(idx, "weights_proj", RowKernelLinear(plain_wp, eps=eps))
            # the reference must be computed under the SAME stub: both arms then call
            # the projection with the same rows and the same row count, so a
            # row-count-dependent kernel is the one thing that cannot separate them.
            ref_stub = np.array(run_replicated(idx, None, inp, offset=offset)["out"])
            banded = run_banded(idx, None, inp, bands=bands_f, offset=offset, world=world)
            recon = sum_pads(banded["pads"])
            # (3) SABOTAGE (the pre-fix expression, modelled from de6caa2's own diff):
            #     project the BAND's own rows.  Same stub, same shape.
            pre = pre_fix_model(eps)
            pre_rows = int((pre != ref_stub).any(-1).sum())
            # the control wave-1 could not run: with a ROW-COUNT-INVARIANT projection
            # (the plain nn.Linear the sibling harnesses use) the pre-fix expression is
            # exact -- which is precisely why the live bug was invisible offline.
            pre_plain = pre_fix_model(None)
            ref_plain = np.array(run_replicated(idx, None, inp, offset=offset)["out"])
            pre_plain_rows = int((pre_plain != ref_plain).any(-1).sum())
            immunity.append(dict(
                eps=eps,
                shipped_differing_rows=int((recon != ref_stub).any(-1).sum()),
                shipped_bytes_equal=bool((recon == ref_stub).all()),
                pre_fix_differing_rows=pre_rows,
                pre_fix_plain_differing_rows=pre_plain_rows,
                pre_fix_plain_reference_equal=bool((pre_plain == ref_plain).all())))
        setattr(idx, "wq_b", plain_wq)
        setattr(idx, "weights_proj", plain_wp)
        # (4) the threshold: the smallest projection perturbation the harness can turn
        #     into a selection difference through the pre-fix expression.
        threshold = None
        threshold_rows = None
        for eps in (2 ** -30, 2 ** -24, 2 ** -20, 2 ** -16, 2 ** -12, 2 ** -8, 2 ** -4):
            setattr(idx, "wq_b", RowKernelLinear(plain_wq, eps=eps))
            setattr(idx, "weights_proj", RowKernelLinear(plain_wp, eps=eps))
            ref_stub = np.array(run_replicated(idx, None, inp, offset=offset)["out"])
            pre = pre_fix_model(eps)
            rows = int((pre != ref_stub).any(-1).sum())
            if rows > 0:
                threshold, threshold_rows = eps, rows
                break
        setattr(idx, "wq_b", plain_wq)
        setattr(idx, "weights_proj", plain_wp)
        out = record(dict(
            case="projection_kernel", n=n, world=world, nb=nb, offset=offset,
            stub_row_dependence_differing=stub_q_differing,
            stub_row_dependence_total=stub_q_total,
            immunity=immunity, pre_fix_threshold_eps=threshold,
            pre_fix_threshold_rows=threshold_rows))
        assert stub_q_differing > 0, "the row-count stub is not row-count dependent"
        assert all(e["shipped_bytes_equal"] for e in immunity), (
            "the shipped banded path moved a selection under a row-count-dependent projection")
        assert any(e["pre_fix_differing_rows"] > 0 for e in immunity), (
            "the pre-fix model did not diverge at ANY perturbation size -- vacuous probe")
        assert all(e["pre_fix_plain_reference_equal"] for e in immunity), (
            "the pre-fix model diverged from a row-count-INVARIANT projection")
        assert threshold is not None, "no perturbation size flipped the pre-fix selection"
    finally:
        restore_knobs(saved)
    return out


def argpartition_tie_probe() -> dict[str, Any]:
    """Does the shipped top-k's tie pick depend on the array's LENGTH / layout?

    The hierarchical exact pass and the ``merge`` implementation both select with
    ``argpartition(-v, k-1)[..., :k]`` over ``concat([best_v, strip_v])`` — a length of
    ``k + bp*block`` that changes when the strip width changes.  This probe feeds the
    SHIPPED ``topk_from_row`` (one global pass) and ``merge_topk`` (per-strip running
    merge) the SAME columns, with a large block of EXACTLY tied top values spread across
    the strips, and compares the selected index SETS against the index-ordered ``ref``.
    """
    k, first, rest = INDEX_TOPK, 2712, 1512
    rng = np.random.default_rng(4242)
    total = first + rest
    tied = np.zeros(total, bool)
    tied[np.linspace(0, total - 1, 900).astype(int)] = True
    vals = np.where(tied, np.float32(1.0), rng.random(total).astype(np.float32) * 0.5)
    s = mx.array(vals[None, None, :])
    neg = mx.full((1, 1, k), IX.NEG_INF, dtype=mx.float32)
    idx = mx.arange(total, dtype=mx.int32)[None, None, :]

    # one global pass (the shipped buffer path / untiled path / hier's single strip)
    v_a = mx.concatenate([neg, s], axis=-1)
    _, i_a = IX.topk_from_row(v_a, k)
    set_a = set(int(t) for t in np.array(i_a).reshape(-1))
    # two merge rounds (the shipped per-strip running merge, widths first then rest)
    bv, bi = mx.full((1, 1, k), IX.NEG_INF, dtype=mx.float32), mx.zeros((1, 1, k), mx.int32)
    bv, bi = IX.merge_topk(bv, bi, s[..., :first], idx[..., :first], k)
    bv, bi = IX.merge_topk(bv, bi, s[..., first:], idx[..., first:], k)
    set_b = set(int(t) for t in np.array(bi).reshape(-1))
    # three rounds (a third strip width) -- the same query, a different merge arity
    bv2, bi2 = mx.full((1, 1, k), IX.NEG_INF, dtype=mx.float32), mx.zeros((1, 1, k), mx.int32)
    cuts = [0, 900, 2712, total]
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        bv2, bi2 = IX.merge_topk(bv2, bi2, s[..., c0:c1], idx[..., c0:c1], k)
    set_c = set(int(t) for t in np.array(bi2).reshape(-1))
    ref = set(int(t) for t in np.flatnonzero(tied)[:k])
    return dict(k=k, columns=total, tied_columns=int(tied.sum()),
                one_pass_equals_two_round=bool(set_a == set_b),
                one_pass_equals_three_round=bool(set_a == set_c),
                all_equal_index_ordered=bool(set_a == set_b == set_c == ref),
                overlap_one_vs_two=len(set_a & set_b) / k,
                overlap_one_vs_index_ordered=len(set_a & ref) / k)


def section_g() -> dict[str, Any]:
    print("== G. exact-pass strip width: band-derived vs chunk-derived ==", flush=True)
    saved = save_knobs()
    out: dict[str, Any] = {}
    try:
        # The strip width is derived from the ROW COUNT (indexer.py:796-799), and the
        # banded run passes its own band rows while the replicated run passes the full
        # chunk.  The merge it feeds is an argpartition over concat([best, strip]) whose
        # LENGTH is k + bp*block, so a strip-count difference is a tie-break difference.
        # (a) the asymmetry is present at real shapes:
        asym = []
        for host_n, world, nb in ((78, 2, 3000), (64, 2, 3000), (132, 4, 3000),
                                  (78, 2, 4096), (2048, 2, 8192)):
            if host_n % world:
                continue
            b_est, _, b_str = estrip_of(host_n // world, nb)
            f_est, _, f_str = estrip_of(host_n, nb)
            asym.append(dict(host_n=host_n, world=world, nb=nb, band_rows=host_n // world,
                             estrip_band=b_est, estrip_full=f_est,
                             strips_band=b_str, strips_full=f_str,
                             differs=b_str != f_str))
        # (b) DIRECT strip-width sensitivity, no band involved: the SAME replicated
        #     (band=None) forward, twice, with _HIER_EXACT_STRIP forced to the band's and
        #     to the chunk's derived value.  This isolates the one variable.
        direct = []
        for host_n, nb, mode in ((78, 3000, "ties"), (78, 3000, "lattice"),
                                 (132, 3000, "ties"), (512, 4096, "ties")):
            outs = {}
            for arm, est in (("band_estrip", estrip_of(host_n // 2, nb)[0]),
                             ("full_estrip", estrip_of(host_n, nb)[0])):
                knobs(hier=True, estrip=est)
                idx, _, x, qr, start_pos, fv, ik = inputs_for(62_000 + host_n, "plain",
                                                              host_n, nb, mode)
                inp = (idx, None, x, qr, start_pos, fv, ik)
                outs[arm] = (np.array(run_replicated(idx, None, inp,
                                                     offset=start_pos + host_n)["out"]), est)
            a, ea = outs["band_estrip"]
            b_, eb = outs["full_estrip"]
            direct.append(dict(n=host_n, nb=nb, mode=mode, estrip_a=ea, estrip_b=eb,
                               differing_rows=int((a != b_).any(-1).sum()),
                               bytes_equal=bool((a == b_).all())))
        # (c) band vs chunk on the DEFAULT (band-derived) estrip.
        band_vs_full = []
        for host_n, world, nb, mode in ((78, 2, 3000, "ties"), (64, 2, 4096, "ties"),
                                        (132, 4, 3000, "ties")):
            r = cell(seed=61_000 + host_n, tag="strip_band_vs_full", role="source",
                     route="hier", world=world, n=host_n, nb=nb, mode=mode, phase=0,
                     check=False)
            band_vs_full.append(dict(host_n=host_n, world=world, nb=nb, mode=mode,
                                     differing_rows=r["differing_rows"],
                                     strips_band=r["strips_band"],
                                     strips_full=r["strips_full"],
                                     kth_tied_rows=r["kth_tied_rows"]))
        # (d) the tie-break MECHANISM: is `argpartition(-v, k-1)`'s pick among exactly
        #     tied values invariant to the array's length and to how many merge rounds
        #     produced it?  If it is (index-ordered), the merge LENGTH cannot move a
        #     selection and the estrip asymmetry is benign by construction.
        # (e) REACHABILITY: the maximal exact tie ("same" keys -> every column of a row
        #     scores identically).  If the band arm and the chunk arm still agree here,
        #     no tie structure at all can separate them at this shape.
        wide = []
        for host_n, world, nb, mode in ((78, 2, 3000, "onehot"), (78, 2, 1024, "onehot"),
                                        (132, 4, 3000, "onehot"), (512, 2, 4096, "onehot"),
                                        (34, 2, 3000, "onehot"), (78, 2, 3000, "same")):
            r = cell(seed=63_000 + host_n + nb, tag="strip_wide_tie", role="source",
                     route="hier", world=world, n=host_n, nb=nb, mode=mode, phase=0,
                     check=False)
            wide.append(dict(n=host_n, world=world, nb=nb, mode=mode,
                             differing_rows=r["differing_rows"],
                             overlap_clean=r["overlap_clean"],
                             kth_tied_rows=r["kth_tied_rows"],
                             strips_band=r["strips_band"], strips_full=r["strips_full"],
                             band_slice_equal=r["band_slice_equal"]))
        tie = argpartition_tie_probe()
        out = record(dict(case="strip_width", asymmetry=asym, direct_estrip=direct,
                          band_vs_full=band_vs_full, wide_tie=wide, tie_break=tie,
                          selection_sensitive=bool(any(d["differing_rows"] for d in direct)
                                                  or any(b_["differing_rows"]
                                                         for b_ in band_vs_full)
                                                  or any(w["differing_rows"] for w in wide))))
    finally:
        restore_knobs(saved)
    return out


# ---------------------------------------------------------------------------
# H. the D3 read-out: the row count each projection is CALLED with
# ---------------------------------------------------------------------------
def section_h() -> dict[str, Any]:
    print("== H. projection call row counts at the banded call site ==", flush=True)
    saved = save_knobs()
    try:
        knobs(hier=True)
        rows_seen: dict[str, Any] = {}
        for route, n, world, nb in (("hier", 78, 2, 1024), ("tiled", 1024, 2, 2048),
                                    ("untiled", 78, 4, 520), ("hier", 2048, 2, 8192)):
            knobs(hier=route == "hier", tiled=route == "tiled", untiled=route == "untiled")
            idx, _, x, qr, start_pos, fv, ik = inputs_for(70_000 + n, "plain", n, nb,
                                                          "lattice")
            inp = (idx, None, x, qr, start_pos, fv, ik)
            wq_spy, wp_spy = RowCountSpy(idx.wq_b), RowCountSpy(idx.weights_proj)
            setattr(idx, "wq_b", wq_spy)
            setattr(idx, "weights_proj", wp_spy)
            bands_h: list[tuple[int, int]] = [
                bd for bd in (IX.query_band(n, Group(r, world)) for r in range(world))
                if bd is not None]
            run_banded(idx, None, inp, bands=bands_h, offset=start_pos + n, world=world)
            replicated = run_replicated(idx, None, inp, offset=start_pos + n)
            del replicated
            rows_seen[f"{route}_n{n}_w{world}"] = dict(
                n_full=n, band_rows=n // world,
                wq_rows=sorted(set(wq_spy.rows)), weights_proj_rows=sorted(set(wp_spy.rows)),
                per_rank_calls=wq_spy.rows)
        out = record(dict(case="d3_projection_rows", observed=rows_seen,
                          all_calls_full=all(v["wq_rows"] == [v["n_full"]] and
                                             v["weights_proj_rows"] == [v["n_full"]]
                                             for v in rows_seen.values())))
        assert out["all_calls_full"], (
            "a projection was CALLED with band rows -- the de6caa2 fix is incomplete")
    finally:
        restore_knobs(saved)
    return out


# ---------------------------------------------------------------------------
# I. non-vacuity controls
# ---------------------------------------------------------------------------
def section_i() -> dict[str, Any]:
    print("== I. non-vacuity: margin flips + the tie lattice ==", flush=True)
    saved = save_knobs()
    try:
        knobs(hier=False, untiled=True)
        n, nb = 78, 1024
        idx, _, x, qr, start_pos, fv, ik = inputs_for(80_001, "plain", n, nb, "lattice")
        inp = (idx, None, x, qr, start_pos, fv, ik)
        row = dense_row(idx, inp, band=None)
        k = min(INDEX_TOPK, nb)
        rows2 = row.reshape(-1, nb)
        flips = 0
        margins: list[float] = []
        for i in range(rows2.shape[0]):
            r = rows2[i]
            order = np.argsort(-r, kind="stable")
            j_in, j_out = int(order[k - 1]), int(order[k])
            margin = float(r[j_in] - r[j_out])
            if margin > 0.0:
                margins.append(margin)
            bumped = r.copy()
            bumped[j_out] = np.float32(r[j_in]) + np.float32(max(margin, 0.0) * 2.0)
            got = set(int(t) for t in np.argsort(-bumped, kind="stable")[:k])
            flips += int(got != set(int(t) for t in order[:k]))
        ik_t = keys_of(80_002, nb, "ties")
        inp_t = (idx, None, x, qr, start_pos, fv, ik_t)
        row_t = dense_row(idx, inp_t, band=None)
        srt = np.sort(row_t, axis=-1)[..., ::-1]
        tied = int((srt[..., k - 1] == srt[..., k]).sum())
        out = record(dict(case="non_vacuity", rows=int(rows2.shape[0]), margin_flips=flips,
                          min_margin=(min(margins) if margins else 0.0),
                          tie_rows_at_k=tied, k=k))
        assert flips > 0, "a margin-sized perturbation did not flip any selection"
        assert tied > 0, "the tie lattice put no exact tie on rank k"
    finally:
        restore_knobs(saved)
    return out


# ---------------------------------------------------------------------------
def gate() -> int:
    """The SELECTION gate: exit 1 if any recorded cell shows a selection divergence."""
    bad = []
    for r in RESULTS:
        if r.get("case") == "qsplit_cell":
            if r.get("tag") == "d2_hazard":
                continue                      # a measured, documented hazard
            if r.get("differing_rows"):
                bad.append((r.get("tag"), r.get("role"), r.get("route"), r.get("world"),
                            r.get("n"), r.get("nb"), r.get("keys"), r["differing_rows"]))
            elif r.get("cand_rows"):
                bad.append((r.get("tag"), "cand_mask", r.get("n"), r.get("nb"),
                            r["cand_rows"]))
            elif r.get("nonfinite_mismatch"):
                bad.append((r.get("tag"), "nonfinite", r.get("n"), r.get("nb"),
                            r["nonfinite_mismatch"]))
    cells = [r for r in RESULTS if r.get("case") == "qsplit_cell"]
    print(f"\n[qsplit-refdiv] cells={len(cells)} divergences={len(bad)}", flush=True)
    for b in bad[:40]:
        print("   DIVERGENCE " + repr(b), flush=True)
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# J. the matrix re-run with a ROW-COUNT-DEPENDENT projection installed
# ---------------------------------------------------------------------------
def section_j() -> list[dict[str, Any]]:
    """The whole point of the ``de6caa2`` invariant, at matrix scale.

    With a row-count-dependent projection installed, a banded call that projected its own
    rows would land in a different kernel bucket than the replicated run and the selection
    would separate.  Nothing may change while the shipped call order holds — and this is
    the arm that goes RED if the fix is ever reverted (a plain ``nn.Linear`` harness is
    structurally blind to it, which is why the live divergence survived its predecessor
    suites).
    """
    print("== J. matrix with a row-count-dependent projection ==", flush=True)
    saved = save_knobs()
    out: list[dict[str, Any]] = []
    try:
        for i, (eps, n, world, nb, route) in enumerate((
                (1e-4, 78, 2, 1024, "hier"), (1e-2, 78, 2, 1024, "hier"),
                (1e-4, 132, 4, 3000, "hier"), (1e-2, 78, 2, 1024, "tiled"),
                (1e-4, 512, 2, 4096, "hier"), (1e-2, 130, 2, 3000, "hier"))):
            out.append(cell(seed=100_000 + i, tag="row_kernel_matrix", role="source",
                            route=route, world=world, n=n, nb=nb, mode="lattice",
                            phase=0, row_kernel_eps=eps))
    finally:
        restore_knobs(saved)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("BH_AGENT_WORKSPACE", os.path.expanduser("~")),
        "dsv41_indexer_qsplit_refdiv.json"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    base = save_knobs()
    try:
        section_a()
        section_b(args.quick)
        section_c()
        section_d()
        section_e()
        section_f()
        section_g()
        section_h()
        section_i()
        section_j()
    finally:
        restore_knobs(base)
    with open(args.out, "w") as handle:
        json.dump(RESULTS, handle, indent=1, default=str)
    print(f"[qsplit-refdiv] {len(RESULTS)} records -> {args.out}", flush=True)
    return gate()


if __name__ == "__main__":
    sys.exit(main())
