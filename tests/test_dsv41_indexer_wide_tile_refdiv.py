# Copyright © 2026 Adam Durham (hermes-gw)
"""Reference-divergence pins for the dsv41 indexer's small-n WIDE TILE.

The audit this file belongs to (2026-10-14).  Between the clean baseline pin
``689e4ea`` and the live pin ``a0573e3`` the complete mlx-lm change surface is 6
commits.  With ``DSV41_INDEXER_QSPLIT=0`` (the second live degeneration event's
config) every qsplit hunk is a no-op and the ONLY serving-math change is the
``tile_width()`` small-n wide-tile branch (``DSV41_INDEXER_SMALLN_WIDE_TILE``,
default ON).  A token-level "it does not loop any more" proves nothing -- greedy
decode is chaotic -- so this file pins the change at the level where it can be
proved: the ROUTE it selects, the stored score ROW (the top-k's own input), and the
SELECTION (``cidx`` + the candidate mask) that comes out.

What is pinned:

1. ROUTE (pure algebra, production constants).  The tiled/untiled decision is
   invariant under the width rule over a dense context grid (0 mismatches in 10,374
   cells in the sibling harness; asserted here on a smaller grid), and the two
   transcribed widths reproduce the shipped ``tile_width`` exactly for every cell.
2. ENGAGEMENT (the context-inversion question).  The width rule engages only for
   ``n <= _FENCE_MIN_ROWS`` AND a TILED route, i.e. ``nb > cap(n)``.  At production
   geometry that is ``context > cap(n) * ratio``: n=4 -> 254,200 (ratio 1), n=8 ->
   127,096, n=12 -> 84,728, n=16 -> 63,504.  So with a gamma<=3 verify shape
   (n<=4) the rule cannot engage anywhere in the 50-80K band of the two loop
   events; ONLY n=16 does.  Pinned, because the whole regression hypothesis turns
   on it.
3. ROW / SELECTION NEUTRALITY at the engagement shapes.  The width change DOES
   move the stored row at the ulp level at some shapes -- exactly the documented
   phenomenon, and always in the shape where the new arm's TAIL tile is one column
   wide (``nb = cap + 1``), never otherwise in this grid.  Every differing element
   sits at rank >= ~1e4 of the row, i.e. far outside the top-``index_topk`` (512)
   selection, and no index set, candidate mask or real ``Indexer.__call__`` output
   differs in any cell.  Both facts are asserted, not assumed: a differing element
   inside the top-k, or any selection difference, fails the test.
4. SENSITIVITY (the control that makes (3) non-vacuous).  Moving the first
   out-of-set column above the k-th boundary by its own margin + 1 ulp flips that
   row's SET for every row; and tie-injected keys put exact ties ON the boundary.
   So the comparison in (3) is able to see a difference of the size the arms could
   produce.
5. KILL SWITCH.  ``DSV41_INDEXER_SMALLN_WIDE_TILE`` is read once at import; unset
   and ``=1`` are the new rule, ``=0`` reproduces the old formula byte-for-byte at
   production constants.

The wider measurement (97 cells, 3 roles, 2 key lattices, ``nb`` up to 1M, the
``merge`` implementation, the shape census) lives in
``bench/dsv41_indexer_refdiv.py``; this file is the bounded regression pin.

Run (mlx-lm worktree root, off-repo copy so the outer ``tests/`` package cannot win
the import race):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_indexer_wide_tile_refdiv.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from typing import Any

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as IX
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0
from mlx_lm.models.deepseek_v41.model import SharedState

RESULTS: list[dict[str, Any]] = []

HEADS = 32                                  # production index_n_heads
HEAD_DIM = 128                              # production index_head_dim
INDEX_TOPK = 512                            # production index_topk
TILE = 512                                  # DSV41_INDEXER_TILE default
BUDGET = 128 << 20                          # DSV41_INDEXER_TILE_MB default
BLOCK = 8                                   # candidate_block_size in the toy args
FMR = IX._FENCE_MIN_ROWS                    # 16 unless the env overrode it

_KNOBS = ("_SMALLN_WIDE_TILE", "_TILE", "_TILE_BUDGET", "_TILE_MIN_NB", "_TILE_FORCE",
          "_HIER", "_QSPLIT", "_L2_FULL", "_SMALLN_ROW_BF16", "_TILED_IMPL",
          "_FENCE_MIN_ROWS", "_INDEXER_COMPILE")


def record(row: dict[str, Any]) -> dict[str, Any]:
    RESULTS.append(row)
    print("  " + json.dumps(row, default=str), flush=True)
    return row


def production_knobs() -> None:
    """The live knob values (env-unset defaults) at production geometry."""
    IX._TILE, IX._TILE_BUDGET = TILE, BUDGET
    IX._TILE_MIN_NB, IX._TILE_FORCE = 0, False
    IX._L2_FULL, IX._SMALLN_ROW_BF16 = True, False
    IX._HIER, IX._QSPLIT, IX._TILED_IMPL = True, False, "buffer"
    IX._SMALLN_WIDE_TILE = True
    IX._FENCE_MIN_ROWS = FMR


def old_width(bsz: int, n: int, nb: int, align: int) -> int:
    """The pre-a0573e3 rule, transcribed verbatim from 6a5c78f4."""
    cap = max(align, BUDGET // max(1, bsz * n * (HEADS * 4 + IX._row_dtype(n).size)))
    return min(nb, max(align, (min(TILE, cap) // align) * align))


def new_width(bsz: int, n: int, nb: int, align: int) -> int:
    """The a0573e3 rule: the wide branch is small-n only."""
    cap = max(align, BUDGET // max(1, bsz * n * (HEADS * 4 + IX._row_dtype(n).size)))
    if n <= IX._FENCE_MIN_ROWS and not IX._TILE_FORCE:
        return min(nb, max(align, (cap // align) * align))
    return min(nb, max(align, (min(TILE, cap) // align) * align))


def args_of(role: str, *, block: int = BLOCK, topk_blocks: int = 32) -> ModelArgs:
    """Toy dims, production score geometry (heads / head_dim / k / block grid)."""
    return ModelArgs(
        dim=256, n_layers=3, n_heads=4, head_dim=128, rope_head_dim=16,
        q_lora_rank=128, o_lora_rank=32, o_groups=4, window_size=8,
        compress_ratios=(1, 1, 1), kv_source_layers=(0,), index_source_layers=(0,),
        index_n_heads=HEADS, index_head_dim=HEAD_DIM, index_topk=INDEX_TOPK,
        max_seq_len=1 << 21,
        candidate_source_layer=(0 if role != "plain" else -1),
        candidate_topk_blocks=topk_blocks, candidate_block_size=block)


def keys_of(seed: int, nb: int, mode: str) -> mx.array:
    """``index_k`` as the owner's cache holds it: fp4_ue8m0 fake-quantized."""
    rng = np.random.default_rng(seed)
    if mode == "ties":
        p = 97
        base = rng.standard_normal((p, HEAD_DIM)).astype(np.float32)
        raw = mx.array(base[np.arange(nb) % p][None])
    else:
        raw = mx.array(rng.standard_normal((1, nb, HEAD_DIM)).astype(np.float32))
    return fake_quant_fp4_ue8m0(raw, 32)


def inputs_of(seed: int, role: str, n: int, nb: int, mode: str):
    args = args_of(role)
    mx.random.seed(seed)
    idx = IX.Indexer(args, 1)
    mx.eval(idx.parameters())
    cons = IX.Indexer(args, 2) if role == "consumer" else None
    if cons is not None:
        mx.eval(cons.parameters())
    rng = np.random.default_rng(seed + 1)
    x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    return args, idx, cons, x, qr, fv, keys_of(seed + 2, nb, mode)


def qw_of(idx: IX.Indexer, x, qr, sp, fv):
    """The indexer's own q / w / lens for a replicated forward (mirrors __call__)."""
    n = x.shape[1]
    q = idx.wq_b(qr).reshape(1, n, idx.n_heads, idx.head_dim)
    qpos = mx.arange(sp, sp + n)
    q = IX.rope_tail(q, idx.rope_head_dim, *IX.cos_sin_at(fv, qpos))
    q = fake_quant_fp4_ue8m0(q, 32)
    w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((qpos + 1) // idx.ratio)[:, None]
    return q.astype(mx.float32), w.astype(mx.float32), lens


def row_at_width(n: int, q32, ik, w32, lens, nb: int, width: int) -> np.ndarray:
    """The stored ``[1, n, nb]`` row exactly as the shipped path builds it.

    ``width=0`` is the untiled branch; otherwise the tiled loop in ``width`` columns
    through the shipped (per-width compiled) score body, fenced per tile.
    """
    row_dtype = IX._row_dtype(n)
    if width <= 0:
        scores = mx.einsum("bshd,btd->bsht", q32, ik.astype(mx.float32))
        scores = mx.maximum(scores, 0.0) * w32[..., None]
        scores = mx.sum(scores, axis=2).astype(row_dtype)
        vis = mx.arange(nb)[None, :] < lens
        scores = mx.where(vis[None], scores, IX.NEG_INF)
        mx.eval(scores)
        return np.array(scores)
    body = IX._score_body(1, n, HEADS, HEAD_DIM, width)
    row = mx.zeros((1, n, nb), dtype=row_dtype)
    for c0 in range(0, nb, width):
        c1 = min(c0 + width, nb)
        s = body(q32, ik[:, c0:c1].astype(mx.float32), w32)
        cols = mx.arange(c0, c1, dtype=mx.int32)
        s = mx.where(cols[None, :] < lens, s, IX.NEG_INF)
        mx.eval(s)
        row[:, :, c0:c1] = s
    mx.eval(row)
    return np.array(row)


def topk_sets(row: np.ndarray, k: int) -> list[set[int]]:
    return [set(int(i) for i in np.argsort(-r, kind="stable")[:k])
            for r in row.reshape(-1, row.shape[-1])]


def arm_row(arm: str, n: int, q32, ik, w32, lens, nb: int, align: int,
            impl: str = "buffer") -> tuple[np.ndarray, int, bool]:
    """The row the SHIPPED code produces for one arm, through its own ROUTE.

    Returns ``(row, effective_width, tiled)``.  At ``nb`` close to ``cap`` an arm is
    UNTILED (the budget clause), and there the width rule is irrelevant -- forcing a
    tiled row would over-state the change.
    """
    production_knobs()
    IX._TILED_IMPL = impl
    if arm == "UNT":
        IX._TILE = 0
    else:
        IX._SMALLN_WIDE_TILE = arm == "NEW"
    width = IX.tiled(1, n, HEADS, nb, align)
    tiled = width < nb
    production_knobs()
    IX._TILED_IMPL = impl
    return row_at_width(n, q32, ik, w32, lens, nb, width if tiled else 0), width, tiled


def _cell(role: str, n: int, nb: int, mode: str, seed: int,
          impl: str = "buffer") -> dict[str, Any]:
    """Row + index set + candidate mask across the three arms, production geometry."""
    production_knobs()
    IX._TILED_IMPL = impl
    args, idx, _, x, qr, fv, ik = inputs_of(seed, role, n, nb, mode)
    align = args.candidate_block_size if role == "source" else 1
    sp = nb - n                                   # ratio 1: the last row sees nb groups
    q32, w32, lens = qw_of(idx, x, qr, sp, fv)
    assert IX._row_dtype(n) is mx.float32, "small-n row must be fp32 under L2-full"
    wo, wn = old_width(1, n, nb, align), new_width(1, n, nb, align)
    rows, widths, routed = {}, {}, {}
    for arm in ("OLD", "NEW", "UNT"):
        rows[arm], widths[arm], routed[arm] = arm_row(arm, n, q32, ik, w32, lens, nb,
                                                       align, impl)
    sets = {k: topk_sets(r, INDEX_TOPK) for k, r in rows.items()}
    old, new = rows["OLD"], rows["NEW"]
    both = np.isfinite(new) & np.isfinite(old)
    dmask = both & (new != old)
    delta = np.abs(np.where(dmask, new, 0.0) - np.where(dmask, old, 0.0))
    flat = new.reshape(-1, nb)
    ranks = [int((flat[r0] > flat[r0, c0]).sum()) for r0, c0 in np.argwhere(dmask.reshape(-1, nb))]
    mask_diff = None
    if role == "source":
        m_old = np.array(IX.select_candidate_blocks(mx.array(old), mx.array(lens),
                                                     args.candidate_topk_blocks, BLOCK))
        m_new = np.array(IX.select_candidate_blocks(mx.array(new), mx.array(lens),
                                                     args.candidate_topk_blocks, BLOCK))
        mask_diff = int((m_old != m_new).sum())
    rec = record(dict(
        case="cell", impl=impl, role=role, n=int(n), nb=int(nb), keys=mode,
        width_old=int(wo), width_new=int(wn),
        effective_old=int(widths["OLD"]), effective_new=int(widths["NEW"]),
        tiled_old=bool(routed["OLD"]), tiled_new=bool(routed["NEW"]),
        tail_old=int(nb % wo) or int(wo), tail_new=int(nb % wn) or int(wn),
        tiles_old=int(-(-nb // wo)), tiles_new=int(-(-nb // wn)),
        row_elems_differing=int(dmask.sum()), max_abs_row_delta=float(delta.max()),
        differing_rank=ranks,
        sets_differing_new_vs_old=sum(1 for a, b in zip(sets["NEW"], sets["OLD"]) if a != b),
        sets_differing_new_vs_unt=sum(1 for a, b in zip(sets["NEW"], sets["UNT"]) if a != b),
        mask_diff_bits=mask_diff))
    return rec


class _Fixture(unittest.TestCase):
    """Save/restore the import-read module constants the way the sibling suites do."""

    def setUp(self) -> None:
        self._saved = {k: getattr(IX, k) for k in _KNOBS}
        self._real_all_sum = IX._coll.all_sum

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            setattr(IX, k, v)
        IX._coll.all_sum = self._real_all_sum
        IX._COMPILED_BODIES.clear()


# ---------------------------------------------------------------------------
# 1 + 2. route invariance and the engagement set
# ---------------------------------------------------------------------------
class RouteEngagementTest(_Fixture):
    def setUp(self) -> None:
        super().setUp()
        production_knobs()

    def test_width_formulas_reproduce_the_shipped_function(self) -> None:
        """Both transcriptions equal ``tile_width`` for every cell of a wide grid,
        and the old rule is exactly what the kill switch reproduces."""
        cells = 0
        for bsz in (1, 2):
            for n in (1, 2, 3, 4, 5, 8, 12, 16, 17, 32, 78, 512):
                for nb in (1, 511, 512, 513, 63_503, 63_504, 63_505, 84_527, 84_528,
                           127_095, 127_096, 254_199, 254_200, 254_201, 1_016_800,
                           1_016_801):
                    for align in (1, BLOCK):
                        IX._SMALLN_WIDE_TILE = False
                        self.assertEqual(IX.tile_width(bsz, n, HEADS, nb, align),
                                         old_width(bsz, n, nb, align),
                                         ("old", bsz, n, nb, align))
                        IX._SMALLN_WIDE_TILE = True
                        self.assertEqual(IX.tile_width(bsz, n, HEADS, nb, align),
                                         new_width(bsz, n, nb, align),
                                         ("new", bsz, n, nb, align))
                        cells += 1
        # the production widths the wide-tile buildnote derived (align 8)
        IX._SMALLN_WIDE_TILE = True
        want = {1: 1_016_800, 4: 254_200, 8: 127_096, 12: 84_728, 16: 63_544}
        for n, w in want.items():
            self.assertEqual(IX.tile_width(1, n, HEADS, 2_000_000, 8), w,
                             ("production width", n))
        production_knobs()
        record(dict(case="width_formulas", cells=cells, production_widths=want))

    def test_route_is_width_rule_invariant(self) -> None:
        """The tiled/untiled ROUTE never moves with the width rule: ``nb <= cap``
        implies the budget clause, so both arms return ``nb`` (untiled) together."""
        grid = list(range(32_000, 1_120_001, 16_000)) + list(range(48_000, 82_001, 1_024))
        cells = mismatch = 0
        for ratio in (1, 2):
            for n in (1, 4, 5, 8, 12, 16, 17, 32, 78, 512):
                for align in (1, BLOCK):
                    for ctx in grid:
                        nb = max(1, ctx // ratio)
                        IX._SMALLN_WIDE_TILE = False
                        r_old = IX.tiled(1, n, HEADS, nb, align) < nb
                        IX._SMALLN_WIDE_TILE = True
                        r_new = IX.tiled(1, n, HEADS, nb, align) < nb
                        cells += 1
                        mismatch += int(r_old != r_new)
        production_knobs()
        record(dict(case="route_invariance", cells=cells, mismatch=mismatch))
        self.assertEqual(mismatch, 0)

    def test_engagement_bands_and_the_context_inversion(self) -> None:
        """Where the width can change anything, expressed as a context threshold: the
        rule needs a TILED route, i.e. ``context > cap(n) * ratio``.  With a gamma<=3
        verify shape (n<=4) that is 254,200 (ratio 1) and above -- outside BOTH the
        clean baseline's 110-241K and the loop events' 50-80K band.  Only n=16 engages
        inside 48-82K, so the wide tile cannot explain a loop there unless the live
        small-n forward was 16 rows wide."""
        changed_in_band: set[int] = set()
        thresholds: dict[str, int] = {}
        for n in (1, 2, 3, 4, 5, 8, 12, 16):
            cap = new_width(1, n, 10 ** 9, 1)
            thresholds[f"n{n}"] = cap
            for ctx in range(48_000, 82_001, 512):
                # only a TILED arm exposes the width at all
                IX._SMALLN_WIDE_TILE = False
                eff_old = IX.tiled(1, n, HEADS, ctx, 1)
                IX._SMALLN_WIDE_TILE = True
                eff_new = IX.tiled(1, n, HEADS, ctx, 1)
                if eff_new != eff_old:
                    changed_in_band.add(n)
        production_knobs()
        record(dict(case="engagement", n_threshold_context_ratio1=thresholds,
                    n_changed_in_48_82K=sorted(changed_in_band)))
        self.assertEqual(sorted(changed_in_band), [16] if FMR >= 16 else sorted(changed_in_band))
        self.assertEqual(old_width(1, 4, 254_201, 1), 512)      # old: 512-wide tiles
        self.assertEqual(new_width(1, 4, 254_201, 1), 254_200)  # new: one wide tile
        # The observable change needs a TILED route AND a width difference.  Below the
        # 254.2K crossing (ratio 1) every n <= 4 cell is UNTILED -- the width rule
        # cannot be observed at all.
        for ctx in (48_000, 56_000, 63_500, 70_000, 78_000, 110_000, 241_000):
            for n in (1, 2, 3, 4):
                for align in (1, BLOCK):
                    IX._SMALLN_WIDE_TILE = False
                    eff_old = IX.tiled(1, n, HEADS, ctx, align)
                    IX._SMALLN_WIDE_TILE = True
                    eff_new = IX.tiled(1, n, HEADS, ctx, align)
                    self.assertEqual(eff_old, eff_new, ("inversion", n, ctx))
        production_knobs()


# ---------------------------------------------------------------------------
# 3. row / selection neutrality at the engagement shapes
# ---------------------------------------------------------------------------
class RowNeutralityTest(_Fixture):
    def setUp(self) -> None:
        super().setUp()
        production_knobs()
        IX._INDEXER_COMPILE = True
        IX._COMPILED_BODIES.clear()

    def _assert_no_selection_moves(self, rec: dict[str, Any]) -> None:
        self.assertEqual(rec["sets_differing_new_vs_old"], 0, rec)
        self.assertEqual(rec["sets_differing_new_vs_unt"], 0, rec)
        if rec["mask_diff_bits"] is not None:
            self.assertEqual(rec["mask_diff_bits"], 0, rec)

    def test_tail_tile_ulp_difference_never_reaches_the_top_k(self) -> None:
        """A ONE-COLUMN tail tile (``nb = cap + 1``) is the shape class where the arms'
        rows differ at all: the tail GEMM's N collapses to 1 while the old arm's tail
        is dozens-to-hundreds of columns wide, so that one column round-trips
        differently at the ulp level (|d| <= 4.5e-08 -- the documented phenomenon,
        reproduced here at the rule's OWN engagement shapes).  Every differing element
        sits far outside the top-``index_topk`` selection, so the index set, the
        candidate mask and the real ``__call__`` are untouched."""
        saw_ulp = 0
        for n, nb in ((4, 254_201), (2, 508_401), (16, 63_551), (12, 84_734), (8, 127_101)):
            rec = _cell("plain", n, nb, "lattice", 100 + n)
            self.assertTrue(rec["tiled_new"], rec)
            self.assertEqual(rec["tail_new"], 1, rec)
            self.assertLessEqual(rec["row_elems_differing"], 2, rec)
            self.assertLessEqual(rec["max_abs_row_delta"], 1e-7, rec)
            for rank in rec["differing_rank"]:
                self.assertGreater(rank, INDEX_TOPK,
                                   ("a differing element inside the top-k", rec))
            saw_ulp += int(bool(rec["row_elems_differing"]))
            self._assert_no_selection_moves(rec)
        self.assertGreater(saw_ulp, 0, "the ulp shape class was not exercised")

    def test_other_tail_widths_are_bitwise_identical(self) -> None:
        """A tail of 2+ columns, or the UNTILED route (where the width is not used at
        all), reproduces the old row bitwise."""
        for n, nb in ((4, 254_202), (4, 254_709), (4, 2 * 254_200 + 7), (16, 63_552),
                      (8, 127_097), (4, 240_000)):
            rec = _cell("plain", n, nb, "lattice", 200 + n)
            self.assertNotEqual(rec["tail_new"], 1, rec)
            self._assert_no_selection_moves(rec)
            self.assertEqual(rec["row_elems_differing"], 0, rec)

    def test_deep_and_tie_injected_cells(self) -> None:
        """The documented deep cells (nb up to 1M, incl. 512-vs-untiled) and the
        exact-tie lattice: no selection moves, even where EVERY row has an exact tie
        on the k-th boundary (so the tie-break is what decides)."""
        for n, nb, mode in ((4, 1 << 20, "lattice"), (4, 1 << 20, "ties"),
                            (16, 1 << 20, "ties"), (4, 1 << 19, "lattice")):
            rec = _cell("plain", n, nb, mode, 300 + n)
            self._assert_no_selection_moves(rec)

    def test_candidate_source_mask_and_the_merge_implementation(self) -> None:
        """The layer-20 mask is ONE global pass over the row and every later index
        layer consumes it, so a single flipped block-max would amplify: pinned
        bitwise.  The non-default running ``merge`` implementation is the one place
        a width change can move a tie pick (its per-tile argpartition length is
        ``k + width``), so it is pinned too."""
        for impl in ("buffer", "merge"):
            rec = _cell("source", 4, 254_300, "ties", 400, impl=impl)
            self._assert_no_selection_moves(rec)
            rec8 = _cell("source", 8, 127_097, "lattice", 401, impl=impl)
            self._assert_no_selection_moves(rec8)

    def test_real_indexer_call_arms_are_byte_identical(self) -> None:
        """End-to-end through the real ``Indexer.__call__`` (source + consumer, so the
        published candidate mask is consumed too): the NEW / OLD / UNT arms return
        byte-identical ``cidx`` and byte-identical candidate masks."""
        for role, n, nb, mode in (("plain", 4, 254_300, "lattice"),
                                  ("source", 4, 254_201, "ties"),
                                  ("consumer", 16, 63_700, "lattice")):
            production_knobs()
            args, idx, cons, x, qr, fv, ik = inputs_of(500 + n, role, n, nb, mode)
            sp = nb - n
            cidx: dict[str, np.ndarray] = {}
            masks: dict[str, np.ndarray | None] = {}
            for arm in ("NEW", "OLD", "UNT"):
                production_knobs()
                IX._SMALLN_WIDE_TILE = arm != "OLD"
                if arm == "UNT":
                    IX._TILE = 0
                shared = SharedState()
                first = idx(x, qr, sp, 0, fv, ik, shared)
                out = cons(x, qr, sp, 0, fv, ik, shared) if cons is not None else first
                cidx[arm] = np.array(out)
                masks[arm] = (None if shared.candidates is None
                              else np.array(shared.candidates))
            production_knobs()
            rec = record(dict(
                case="real_call", role=role, n=int(n), nb=int(nb), keys=mode,
                bytes_equal_new_vs_old=bool((cidx["NEW"] == cidx["OLD"]).all()),
                bytes_equal_new_vs_unt=bool((cidx["NEW"] == cidx["UNT"]).all()),
                mask_bits_new_vs_old=(None if masks["NEW"] is None else
                                      int((masks["NEW"] != masks["OLD"]).sum())),
                mask_bits_new_vs_unt=(None if masks["NEW"] is None else
                                      int((masks["NEW"] != masks["UNT"]).sum()))))
            self.assertTrue(rec["bytes_equal_new_vs_old"], rec)
            self.assertTrue(rec["bytes_equal_new_vs_unt"], rec)
            if rec["mask_bits_new_vs_old"] is not None:
                self.assertEqual(rec["mask_bits_new_vs_old"], 0, rec)
                self.assertEqual(rec["mask_bits_new_vs_unt"], 0, rec)


# ---------------------------------------------------------------------------
# 4. sensitivity: the neutrality pins above are not vacuous
# ---------------------------------------------------------------------------
class SensitivityTest(_Fixture):
    def setUp(self) -> None:
        super().setUp()
        production_knobs()
        IX._INDEXER_COMPILE = True
        IX._COMPILED_BODIES.clear()

    def test_a_margin_sized_perturbation_flips_a_set(self) -> None:
        """Raise the first out-of-set column above the k-th boundary by that row's own
        margin + 1 ulp: every row's SET must change.  The set comparison used in
        :class:`RowNeutralityTest` can therefore see a difference of the size the
        arms could produce (they produce <= 4.5e-08 here)."""
        n, nb = 4, 254_201
        args, idx, _, x, qr, fv, ik = inputs_of(601, "plain", n, nb, "lattice")
        q32, w32, lens = qw_of(idx, x, qr, nb - n, fv)
        row = row_at_width(n, q32, ik, w32, lens, nb, new_width(1, n, nb, 1))
        rows2 = row.reshape(-1, nb)
        base = topk_sets(row, INDEX_TOPK)
        order = np.argsort(-rows2, axis=-1)
        flips, margins = 0, []
        for i in range(rows2.shape[0]):
            cur = rows2[i].copy()
            j_in, j_out = int(order[i, INDEX_TOPK - 1]), int(order[i, INDEX_TOPK])
            margin = float(cur[j_in] - cur[j_out])
            if margin > 0.0:
                margins.append(margin)
            cur[j_out] = np.nextafter(np.float32(cur[j_in] + np.float32(margin)),
                                      np.float32(np.inf))
            flipped = set(int(t) for t in np.argsort(-cur, kind="stable")[:INDEX_TOPK])
            flips += int(flipped != base[i])
        record(dict(case="sensitivity", rows=int(rows2.shape[0]), flips=int(flips),
                    min_margin=(min(margins) if margins else 0.0)))
        self.assertEqual(flips, rows2.shape[0])

    def test_exact_ties_sit_on_the_boundary(self) -> None:
        """The tie lattice really does put exact ties ON rank k (so a tie-break, not a
        margin, is what decides there)."""
        n, nb = 4, 254_300
        args, idx, _, x, qr, fv, ik = inputs_of(602, "plain", n, nb, "ties")
        q32, w32, lens = qw_of(idx, x, qr, nb - n, fv)
        row = row_at_width(n, q32, ik, w32, lens, nb, new_width(1, n, nb, 1))
        srt = np.sort(row, axis=-1)[..., ::-1]
        tied = int((srt[..., INDEX_TOPK - 1] == srt[..., INDEX_TOPK]).sum())
        record(dict(case="tie_census", rows=int(row.shape[0]), tied_at_k=tied))
        self.assertGreater(tied, 0)


# ---------------------------------------------------------------------------
# 5. kill switch
# ---------------------------------------------------------------------------
class KillSwitchTest(_Fixture):
    def _flag(self, value: str | None) -> bool:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {k: v for k, v in os.environ.items()
               if k != "DSV41_INDEXER_SMALLN_WIDE_TILE"}
        if value is not None:
            env["DSV41_INDEXER_SMALLN_WIDE_TILE"] = value
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run(
            [sys.executable, "-c",
             "from mlx_lm.models.deepseek_v41 import indexer as I;"
             "print('FLAG', int(I._SMALLN_WIDE_TILE))"],
            env=env, capture_output=True, text=True, check=True, cwd=root)
        line = [s for s in out.stdout.splitlines() if s.startswith("FLAG")][-1]
        return line.split()[1] == "1"

    def test_default_on_and_zero_off_then_old_widths(self) -> None:
        """unset -> on, ``=1`` -> on, ``=0`` -> off; and OFF reproduces the old formula
        byte-for-byte at production constants (the relaunch's kill-switch arm)."""
        self.assertTrue(self._flag(None))
        self.assertTrue(self._flag("1"))
        self.assertFalse(self._flag("0"))
        production_knobs()
        IX._SMALLN_WIDE_TILE = False
        for n in (1, 2, 4, 8, 12, 16):
            for nb in (300_000, 254_201, 63_551):
                self.assertEqual(IX.tile_width(1, n, HEADS, nb, 1),
                                 old_width(1, n, nb, 1), (n, nb))
        IX._SMALLN_WIDE_TILE = True
        record(dict(case="kill_switch", default=True, zero=False))


if __name__ == "__main__":
    unittest.main()
