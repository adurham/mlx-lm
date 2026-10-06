# Copyright © 2026 Adam Durham (hermes-gw)
"""Recall-focused contract for the bf16 indexer score row in
``mlx_lm/models/deepseek_v41/indexer.py``.

Background. ``_tiled_scores_buffer`` buffers a ``[b, n, nb]`` score row that the
top-k selection and (layer 20) the candidate-block selection read. At deep
context that row dominates a prefill transient: at n=2048, nb=1M it is 8.6 GB
fp32 per index layer. The row is now **bf16** by default
(``DSV41_INDEXER_ROW_BF16=0`` restores fp32), which halves it (4.3 GB) and
routes the score GEMM through 16-bit paths. The per-tile
``[b, n, heads, tile]`` head-sum transient stays fp32 — only the *stored* row is
bf16. **This is not bit-exact**: two scores that round to the same bf16 value
can swap which lands in the top-k. Recall, not bitwise equality, is the gate.

What is pinned (all on CPU):

1. RECALL: synthetic q/k/w with a known, wide-margin ground-truth top-k; the
   bf16-row selection must overlap the fp32-row selection (asserted >= 99%).
   A near-tie variant reports the flip rate without asserting it. Selected
   values are asserted finite / sane (no NaN, no -inf leak).
2. SHAPE/DTYPE: the produced row is ``(bsz, n, nb)`` bf16 (fp32 under the env
   gate); selected indices stay int32; the candidate-block path returns a sane
   block-aligned bool mask.
3. TILED PARITY: the tiled bf16 row and the untiled reference path agree on the
   top-k for wide-margin data (same-module cross-check, not bit-exactness).
4. The memory arithmetic (8.6 GB -> 4.3 GB at n=2048, nb=1M) is asserted.

Run (in an mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_indexer_bf16_row.py -v
    PYTHONPATH=$PWD <venv>/bin/python tests/test_dsv41_indexer_bf16_row.py
"""

import importlib
import json
import os
import sys
import unittest
from dataclasses import dataclass

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as IX
from mlx_lm.models.deepseek_v41.config import ModelArgs

# hot/cold margin: hot columns' key AND query share a direction, so the head-0
# gap ~ margin**2. WIDE gives a gap far above bf16's resolution; TIE gives a
# sub-ulp gap so flips are possible and expected.
WIDE_MARGIN = 4.0
TIE_MARGIN = 0.05
# selection floor on wide-margin data: a flip there is a bug, not precision
OVERLAP_MIN = 0.99

_RESULTS = []


@dataclass
class RecallCase:
    """One measured (fp32 row vs bf16 row) selection-quality sample."""

    label: str
    margin: float
    boundary_gap: float
    bf16_ulp: float
    gap_over_ulp: float
    fp32_overlap_truth: float
    bf16_overlap_truth: float
    fp32_vs_bf16_overlap: float
    all_finite: bool
    nan_free: bool
    neg_inf_leak: int

    def as_dict(self) -> dict[str, object]:
        return {
            "case": self.label, "margin": self.margin,
            "boundary_gap": round(self.boundary_gap, 5),
            "bf16_ulp": round(self.bf16_ulp, 6),
            "gap_over_ulp": round(self.gap_over_ulp, 1),
            "fp32_overlap_truth": round(self.fp32_overlap_truth, 4),
            "bf16_overlap_truth": round(self.bf16_overlap_truth, 4),
            "fp32_vs_bf16_overlap": round(self.fp32_vs_bf16_overlap, 4),
            "all_finite": self.all_finite, "nan_free": self.nan_free,
            "neg_inf_leak": self.neg_inf_leak,
        }


def _f32(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32))


def _row_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-row fraction of shared indices between two [..., k] selections."""
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    inter = [len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b)]
    return float(np.mean(inter)) / a.shape[-1]


def _reload_row_dtype(bf16: bool):
    """Reload indexer with the env gate set; return its _ROW_DTYPE.

    The gate is read once at import (like every other DSV41_* knob), so a
    process restart is required to flip it — modelled here by a module reload.
    Leaves the module back in its default (bf16) state.
    """
    old = os.environ.get("DSV41_INDEXER_ROW_BF16")
    os.environ["DSV41_INDEXER_ROW_BF16"] = "1" if bf16 else "0"
    try:
        dt = importlib.reload(IX)._ROW_DTYPE
    finally:
        if old is None:
            os.environ.pop("DSV41_INDEXER_ROW_BF16", None)
        else:
            os.environ["DSV41_INDEXER_ROW_BF16"] = old
        importlib.reload(IX)                       # back to default (bf16)
    return dt


def _make_case(rng, *, n, nb, h, d, n_hot, margin):
    """q/k/w with a known top-k: ``n_hot`` columns made hot by ``margin``.

    All query rows are identical so the hot set is the same for every row.
    ``w`` is kept positive so the head-sum is a positive combination. The hot
    columns' keys and query-0's head 0 are both pushed along one shared random
    direction ``u``, so the resulting head-0 gap is ~``margin**2`` with no
    cross-column interference (every hot column shares ``u``). ``margin`` large
    -> a wide boundary; tiny -> a deliberate near-tie.
    """
    b = 1
    q = rng.standard_normal((b, 1, h, d)) * (1.0 / np.sqrt(d))
    q = np.broadcast_to(q, (b, n, h, d)).copy()
    k = rng.standard_normal((b, nb, d)) * 0.1
    w = rng.random((b, n, h)) + 0.5                       # positive weights
    u = rng.standard_normal(d)
    u = u / (np.linalg.norm(u) + 1e-9)
    hot = rng.choice(nb, size=n_hot, replace=False)
    for j in hot:
        k[0, j, :] += margin * u
    q[0, :, 0, :] += margin * u
    lens = np.full((n, 1), nb, np.int32)
    return (mx.array(q.astype(np.float32)), mx.array(k.astype(np.float32)),
            mx.array(w.astype(np.float32)), mx.array(lens))


def _untiled_topk(q, k, w, lens, nb, kk, *, row_dtype):
    """The untiled branch of Indexer.__call__ up to top-k, at a chosen row dtype."""
    IX._ROW_DTYPE = row_dtype
    s = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), k.astype(mx.float32))
    s = mx.maximum(s, 0.0) * w[..., None].astype(mx.float32)
    s = mx.sum(s, axis=2).astype(IX._ROW_DTYPE)
    vis = mx.arange(nb)[None, :] < lens
    s = mx.where(vis[None], s, float("-inf"))
    idx = mx.argpartition(-s, kk - 1, axis=-1)[..., :kk].astype(mx.int32)
    idx = mx.sort(idx, axis=-1)
    mx.eval(idx)
    return np.array(idx)


def _fp32_matrix(q, k, w, lens):
    """The fp32 untiled score matrix, visibility-masked."""
    s = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), k.astype(mx.float32))
    s = mx.maximum(s, 0.0) * w[..., None].astype(mx.float32)
    s = mx.sum(s, axis=2)
    vis = mx.arange(k.shape[1])[None, :] < lens
    return _f32(mx.where(vis[None], s, float("-inf")))


def _boundary(scores: np.ndarray, kk: int) -> tuple[float, float]:
    """(min k-th boundary gap, ulp near that boundary) over all query rows."""
    srt = np.sort(scores.reshape(-1, scores.shape[-1]), axis=-1)[:, ::-1]
    sk, sk1 = srt[:, kk - 1], srt[:, kk]
    gaps = sk - sk1
    gaps = gaps[np.isfinite(gaps)]
    worst = float(gaps.min()) if gaps.size else float("inf")
    mag = float(np.abs(srt[:, kk - 1]).max()) if srt.size else 1.0
    return worst, 2.0 ** (np.floor(np.log2(mag)) - 7)          # bf16 ulp at |s_k|


def _fp32_truth(q, k, w, lens, nb, kk):
    """Exact top-k from the fp32 untiled score matrix (independent oracle)."""
    s = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), k.astype(mx.float32))
    s = mx.maximum(s, 0.0) * w[..., None].astype(mx.float32)
    s = mx.sum(s, axis=2)
    vis = mx.arange(nb)[None, :] < lens
    s = mx.where(vis[None], s, float("-inf"))
    idx = mx.argpartition(-s, kk - 1, axis=-1)[..., :kk].astype(mx.int32)
    mx.eval(idx)
    return np.array(idx)


class RowDtypeTest(unittest.TestCase):
    """The env gate and the produced row's shape/dtype."""

    _saved: object = None

    def setUp(self):
        self._saved = IX._ROW_DTYPE

    def tearDown(self):
        IX._ROW_DTYPE = self._saved

    def test_default_is_bf16(self):
        self.assertEqual(_reload_row_dtype(True), mx.bfloat16)

    def test_env_gate_restores_fp32(self):
        self.assertEqual(_reload_row_dtype(False), mx.float32)
        self.assertEqual(_reload_row_dtype(True), mx.bfloat16)   # and back

    def _run_capture_row(self, row_dtype):
        rng = np.random.default_rng(1)
        b, n, nb, h, d = 1, 6, 20, 4, 8
        q = mx.array(rng.standard_normal((b, n, h, d)).astype(np.float32))
        k = mx.array(rng.standard_normal((b, nb, d)).astype(np.float32))
        w = mx.array(rng.standard_normal((b, n, h)).astype(np.float32))
        lens = mx.array(np.array([[min(i + 1, nb)] for i in range(n)], np.int32))
        seen = {}
        real = IX.topk_from_row

        def spy(row, kk):
            seen["dtype"], seen["shape"] = row.dtype, row.shape
            return real(row, kk)

        IX.topk_from_row = spy
        try:
            IX._ROW_DTYPE = row_dtype
            v, i, _ = IX._tiled_scores_buffer(q, k, w, lens, nb, 5, 4)
            mx.eval(v, i)
        finally:
            IX.topk_from_row = real
        return seen, i

    def test_row_shape_and_dtype_bf16(self):
        seen, i = self._run_capture_row(mx.bfloat16)
        self.assertEqual(seen["dtype"], mx.bfloat16, "row is not bf16")
        self.assertEqual(tuple(seen["shape"]), (1, 6, 20))
        self.assertEqual(i.dtype, mx.int32, "selected indices must stay int32")
        row = {"case": "row_shape_dtype", "row_dtype": str(seen["dtype"]),
               "row_shape": list(seen["shape"]), "idx_dtype": str(i.dtype)}
        _RESULTS.append(row)
        print("  " + json.dumps(row))

    def test_row_dtype_fp32_under_gate(self):
        seen, _ = self._run_capture_row(mx.float32)
        self.assertEqual(seen["dtype"], mx.float32)

    def test_candidate_mask_is_sane(self):
        rng = np.random.default_rng(2)
        b, n, nb, h, d = 1, 8, 32, 4, 8
        q = mx.array(rng.standard_normal((b, n, h, d)).astype(np.float32))
        k = mx.array(rng.standard_normal((b, nb, d)).astype(np.float32))
        w = mx.array(rng.standard_normal((b, n, h)).astype(np.float32))
        lens = mx.array(np.array([[min(i + 1, nb)] for i in range(n)], np.int32))
        topk_blocks, block_size = 3, 4
        _, _, mask = IX._tiled_scores_buffer(
            q, k, w, lens, nb, 5, 4, cand_src=(topk_blocks, block_size))
        mx.eval(mask)
        m = np.array(mask)
        self.assertEqual(m.dtype, np.bool_)
        self.assertEqual(m.shape, (b, n, nb))
        for r in range(b):
            for row in range(n):
                runs = np.split(m[r, row],
                                np.where(np.diff(m[r, row].astype(int)))[0] + 1)
                for run in runs:
                    self.assertEqual(len(run) % block_size, 0, "mask not block-aligned")
        self.assertGreater(int(m.sum()), 0, "candidate mask selected nothing")
        row = {"case": "candidate_mask", "shape": list(m.shape),
               "kept_cols": int(m.sum()), "block_aligned": True}
        _RESULTS.append(row)
        print("  " + json.dumps(row))


class RecallTest(unittest.TestCase):
    """Selection quality: bf16 row vs fp32 row, wide-margin and near-tie."""

    _saved: object = None

    def setUp(self):
        self._saved = IX._ROW_DTYPE
        IX._ROW_DTYPE = mx.bfloat16

    def tearDown(self):
        IX._ROW_DTYPE = self._saved

    def _measure(self, label, *, margin, n=6, nb=64, kk=8, n_hot=12, seed=0) -> RecallCase:
        rng = np.random.default_rng(seed)
        q, k, w, lens = _make_case(rng, n=n, nb=nb, h=4, d=8,
                                   n_hot=n_hot, margin=margin)
        truth = _fp32_truth(q, k, w, lens, nb, kk)
        sm = _fp32_matrix(q, k, w, lens)
        gap, ulp = _boundary(sm, kk)

        IX._ROW_DTYPE = mx.float32
        v32, i32, _ = IX._tiled_scores_buffer(q, k, w, lens, nb, kk, 8)
        mx.eval(v32, i32)
        IX._ROW_DTYPE = mx.bfloat16
        vb, ib, _ = IX._tiled_scores_buffer(q, k, w, lens, nb, kk, 8)
        mx.eval(vb, ib)

        a, bsel = np.array(i32), np.array(ib)
        v = _f32(vb)
        f32v = _f32(v32)
        # a bf16 slot that is -inf where the fp32 selection was finite would be
        # an -inf leak (padding/masking bug), not a precision flip
        case = RecallCase(
            label=label, margin=margin, boundary_gap=gap, bf16_ulp=ulp,
            gap_over_ulp=(gap / ulp) if ulp else 0.0,
            fp32_overlap_truth=_row_overlap(a, truth),
            bf16_overlap_truth=_row_overlap(bsel, truth),
            fp32_vs_bf16_overlap=_row_overlap(a, bsel),
            all_finite=bool(np.isfinite(v).all()),
            nan_free=not bool(np.isnan(v).any()),
            neg_inf_leak=int(np.sum(~np.isfinite(v) & np.isfinite(f32v))),
        )
        _RESULTS.append(case.as_dict())
        print("  " + json.dumps(case.as_dict()))
        return case

    def test_wide_margin_recall(self):
        # exactly k hot columns: the top-k boundary separates hot from cold, so
        # bf16 rounding cannot reorder *within* the selected set
        r = self._measure("wide", margin=WIDE_MARGIN, seed=11, kk=8, n_hot=8)
        self.assertGreater(r.gap_over_ulp, 20.0,
                           f"wide case not wide: gap/ulp={r.gap_over_ulp}")
        self.assertEqual(r.fp32_overlap_truth, 1.0,
                         "fp32 row itself missed the wide-margin truth")
        self.assertGreaterEqual(r.bf16_overlap_truth, OVERLAP_MIN,
                                "bf16 recall below floor")
        self.assertGreaterEqual(r.fp32_vs_bf16_overlap, OVERLAP_MIN,
                                "bf16 vs fp32 overlap below floor")
        self.assertTrue(r.all_finite, "non-finite value in bf16 selection")
        self.assertTrue(r.nan_free, "NaN in bf16 selected values")
        self.assertEqual(r.neg_inf_leak, 0, "-inf leaked into a finite slot")

    def test_wide_margin_recall_sweep(self):
        worst = 1.0
        for seed in (3, 7, 19, 42, 101):
            r = self._measure(f"wide/s{seed}", margin=WIDE_MARGIN, seed=seed,
                              nb=96, n_hot=12, kk=12)
            self.assertTrue(r.all_finite)
            self.assertTrue(r.nan_free)
            self.assertEqual(r.neg_inf_leak, 0)
            self.assertGreater(r.gap_over_ulp, 20.0,
                               f"seed {seed} not wide: {r.gap_over_ulp}")
            worst = min(worst, r.bf16_overlap_truth, r.fp32_vs_bf16_overlap)
        _RESULTS.append({"case": "wide_sweep_worst", "worst_overlap": round(worst, 4)})
        print("  " + json.dumps({"case": "wide_sweep_worst",
                                 "worst_overlap": round(worst, 4)}))
        self.assertGreaterEqual(worst, OVERLAP_MIN, f"worst sweep overlap {worst}")

    def test_near_tie_flip_rate_reported(self):
        """Near-ties: report the bf16 flip rate; deliberately do NOT assert it."""
        rates, total_flips, total_slots = [], 0, 0
        for seed in (5, 13, 29, 64, 128):
            r = self._measure(f"tie/s{seed}", margin=TIE_MARGIN, seed=seed,
                              nb=64, n_hot=16, kk=8)
            self.assertTrue(r.nan_free, "NaN on near-tie case")
            self.assertEqual(r.neg_inf_leak, 0, "-inf leak on near-tie case")
            rows, kk = 6, 8
            total_flips += int(round((1.0 - r.fp32_vs_bf16_overlap) * rows * kk))
            total_slots += rows * kk
            rates.append(r.fp32_vs_bf16_overlap)
        rate = total_flips / max(total_slots, 1)
        msg = {"case": "near_tie_flip_rate", "observed_flip_rate": round(rate, 5),
               "per_seed_overlap": [round(x, 4) for x in rates], "asserted": False}
        _RESULTS.append(msg)
        print("  " + json.dumps(msg))
        # report-only: near-tie flips are the accepted cost of a bf16 row.


class TiledUntiledParityTest(unittest.TestCase):
    """Tiled bf16 row vs the untiled reference path on wide-margin data."""

    _saved: object = None

    def setUp(self):
        self._saved = IX._ROW_DTYPE
        IX._ROW_DTYPE = mx.bfloat16

    def tearDown(self):
        IX._ROW_DTYPE = self._saved

    def test_tiled_vs_untiled_wide_margin(self):
        worst = 1.0
        for tile in (4, 8, 16, 32):
            rng = np.random.default_rng(tile)
            # exactly k=8 hot columns -> the boundary is hot|cold, unambiguous
            q, k, w, lens = _make_case(rng, n=6, nb=48, h=4, d=8,
                                       n_hot=8, margin=WIDE_MARGIN)
            _, it, _ = IX._tiled_scores_buffer(q, k, w, lens, 48, 8, tile)
            mx.eval(it)
            # the untiled *fp32* reference (the exact pre-change pipeline): the
            # tiled bf16 row must not miss its top-k on wide-margin data
            iu = _untiled_topk(q, k, w, lens, 48, 8, row_dtype=mx.float32)
            truth = _fp32_truth(q, k, w, lens, 48, 8)
            ov = _row_overlap(np.array(it), iu)
            ot = _row_overlap(np.array(it), truth)
            row = {"case": f"tiled{tile}_vs_untiled_fp32", "overlap": round(ov, 4),
                   "tiled_vs_truth": round(ot, 4)}
            _RESULTS.append(row)
            print("  " + json.dumps(row))
            worst = min(worst, ov, ot)
        self.assertGreaterEqual(worst, OVERLAP_MIN,
                                f"tiled/untiled wide-margin overlap {worst} low")


class MemoryFormulaTest(unittest.TestCase):
    """The memory claim the change rests on, pinned as arithmetic."""

    def test_row_sizes(self):
        # n=2048, nb=1<<20 at full context (1M tokens)
        n, nb = 2048, 1 << 20
        fp32, bf16 = n * nb * 4, n * nb * 2
        gib = float(1 << 30)
        self.assertAlmostEqual(fp32 / gib, 8.0, places=1)     # 8.0 GiB ~ 8.6 GB
        self.assertAlmostEqual(bf16 / gib, 4.0, places=1)     # 4.0 GiB ~ 4.3 GB
        self.assertEqual(fp32 // bf16, 2, "bf16 must halve the row")
        # a 2048 MB transient budget: viable n at 1M context doubles under bf16
        budget = 2048 * (1 << 20)
        chunk_fp32 = budget // (nb * 4)
        chunk_bf16 = budget // (nb * 2)
        self.assertEqual(chunk_bf16, 2 * chunk_fp32, "bf16 must double the chunk")
        row = {"case": "memory_formula", "n": n, "nb": nb,
               "fp32_GiB": round(fp32 / gib, 2), "bf16_GiB": round(bf16 / gib, 2),
               "chunk_fp32_tokens": chunk_fp32, "chunk_bf16_tokens": chunk_bf16}
        _RESULTS.append(row)
        print("  " + json.dumps(row))

    def test_runtime_dtype_size_matches_formula(self):
        self.assertEqual(IX._ROW_DTYPE.size, 2)
        self.assertEqual(mx.zeros((2,), dtype=IX._ROW_DTYPE).dtype, mx.bfloat16)


_TINY_ARGS = ModelArgs(
    dim=64, n_layers=1, n_heads=4, head_dim=128, rope_head_dim=16,
    q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
    compress_ratios=(2,), kv_source_layers=(0,), index_source_layers=(0,),
    index_n_heads=2, index_head_dim=32, index_topk=6, max_seq_len=4096,
    candidate_source_layer=0, candidate_topk_blocks=2, candidate_block_size=4)

_TINY_INDEXER: list[IX.Indexer] = []


def _tiny_indexer() -> IX.Indexer:
    """One deterministic tiny Indexer, built once (same weights every call)."""
    if not _TINY_INDEXER:
        mx.random.seed(0)
        idx = IX.Indexer(_TINY_ARGS, 0)
        mx.eval(idx.parameters())
        _TINY_INDEXER.append(idx)
    return _TINY_INDEXER[0]


class IndexerCallIntegrationTest(unittest.TestCase):
    """End-to-end ``Indexer.__call__`` on the tiled path (real rope + fakequant).

    Proves the bf16-row path is reachable and correct through the same entry
    the model uses: the tiled path (forced via DSV41_INDEXER_TILE_FORCE) must
    return the same top-k *selection quality* as the untiled path for
    wide-margin data, on the real query/key construction.
    """

    _saved: dict[str, object] = {}

    def setUp(self):
        self._saved = {"tile": IX._TILE, "min_nb": IX._TILE_MIN_NB,
                       "force": IX._TILE_FORCE, "row": IX._ROW_DTYPE,
                       # These tests exercise the TILED/UNTILED production paths
                       # directly; with HIER default-ON (2026-10-06) they must
                       # pin it off so the non-HIER paths are what runs.
                       "hier": IX._HIER}
        IX._HIER = False

    def tearDown(self):
        IX._TILE = self._saved["tile"]
        IX._TILE_MIN_NB = self._saved["min_nb"]
        IX._TILE_FORCE = self._saved["force"]
        IX._ROW_DTYPE = self._saved["row"]
        IX._HIER = self._saved["hier"]

    def _run(self, *, tiled: bool, row_dtype=None, want_mask=False):
        from mlx_lm.models.deepseek_v41.model import SharedState
        idx = _tiny_indexer()
        rng = np.random.default_rng(5)
        b, n, nb = 1, 8, 120
        # start_pos chosen so lens = (sp + i + 1)//2 >= nb for every row: all
        # columns are visible, so no -inf padding ties dominate the top-k
        sp = 2 * nb
        x = mx.array(rng.standard_normal((b, n, _TINY_ARGS.dim)).astype(np.float32))
        qr = mx.array(rng.standard_normal((b, n, _TINY_ARGS.q_lora_rank)).astype(np.float32))
        index_k = mx.array(rng.standard_normal((b, nb, _TINY_ARGS.index_head_dim)).astype(np.float32))
        freqvec = mx.array(np.linspace(1.0, 0.01,
                                       _TINY_ARGS.rope_head_dim // 2).astype(np.float32))
        IX._TILE = 8
        IX._TILE_MIN_NB = 0
        IX._TILE_FORCE = tiled
        IX._ROW_DTYPE = row_dtype if row_dtype is not None else mx.bfloat16
        shared = SharedState()
        out = idx(x, qr, sp, 0, freqvec, index_k, shared)
        mx.eval(out, shared.candidates)
        return (np.array(out), shared.candidates) if want_mask else np.array(out)

    def test_tiled_call_returns_sane_topk(self):
        out, cand = self._run(tiled=True, want_mask=True)
        self.assertEqual(out.shape, (1, 8, 6))
        self.assertEqual(out.dtype, np.int32)
        # -1 marks masked slots; every other index must address a valid column
        valid = out[out >= 0]
        self.assertTrue(((valid >= 0) & (valid < 120)).all(),
                        "indexer emitted an out-of-range index")
        self.assertEqual(int((out >= 0).sum()), out.size,
                         "all columns were visible but some slots came back -1")
        self.assertIsNotNone(cand, "candidate source did not publish a mask")
        row = {"case": "indexer_call_tiled", "out_shape": list(out.shape),
               "valid_slots": int((out >= 0).sum()), "has_mask": cand is not None}
        _RESULTS.append(row)
        print("  " + json.dumps(row))

    def test_tiled_call_matches_untiled(self):
        for dt, name in ((mx.bfloat16, "bf16"), (mx.float32, "fp32")):
            t = self._run(tiled=True, row_dtype=dt)
            u = self._run(tiled=False, row_dtype=dt)
            ov = _row_overlap(t, u)
            row = {"case": f"indexer_call_tiled_vs_untiled_{name}",
                   "overlap": round(ov, 4)}
            _RESULTS.append(row)
            print("  " + json.dumps(row))
            # same selection modulo tie pick; with every column visible this is
            # not tie-dominated, so the two paths must select the same columns
            self.assertGreaterEqual(ov, OVERLAP_MIN,
                                    f"tiled/untiled {name} selection diverged: {ov}")

    def test_untiled_bf16_matches_fp32(self):
        """Pure bf16-row effect (no tiling): fp32 vs bf16 row selection."""
        a = self._run(tiled=False, row_dtype=mx.float32)
        b = self._run(tiled=False, row_dtype=mx.bfloat16)
        ov = _row_overlap(a, b)
        row = {"case": "untiled_fp32_vs_bf16_row", "overlap": round(ov, 4)}
        _RESULTS.append(row)
        print("  " + json.dumps(row))
        self.assertGreaterEqual(ov, OVERLAP_MIN,
                                f"bf16 row changed selection on non-tie data: {ov}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
    bad = [r for r in _RESULTS
           if not r.get("nan_free", True) or r.get("neg_inf_leak", 0) != 0]
    print(f"DSV41_INDEXER_BF16_ROW {json.dumps({'cases': len(_RESULTS)})}")
    sys.exit(1 if bad else 0)
