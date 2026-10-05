# Copyright © 2026 Adam Durham (hermes-gw)
"""M2 integration tests — the hierarchical / streamed exact pass wired into the
PRODUCTION indexer (``mlx_lm/models/deepseek_v41/indexer.py``) behind
``DSV41_INDEXER_HIER`` (default OFF).

The M1 prototype (``indexer_hierarchical.py`` + its own tests) proved the
mechanism on synthetic tensors. These tests pin the *integration*:

1. ENV OFF == PRODUCTION, byte-identical. With the gate off the new branch is
   never entered (spy), and ``Indexer.__call__`` returns exactly what the
   production impl (``_tiled_scores_buffer`` / the untiled reference) returns
   on the same inputs — selected indices AND scores. Flipping the feature on
   must not perturb the off path.
2. ENV ON == the exact path within the recall gate (>= 99 % rank-weighted
   top-k overlap) on realistic distributions, reusing the M1 harness.
3. CANDIDATE-MASK PARITY for the layer-20-equivalent role: the fused mask
   derived from the coarse maxima buffer equals ``select_candidate_blocks`` fed
   the *same* coarse row (exact, not approximate), and the coarse block maxima
   are bit-equal to the reshape-max of that row.
4. COMPOSITION inside an existing ``shared.candidates`` mask (consumer role):
   every returned index lies inside the mask and the selection matches the
   production consumer path.
5. STRIP SIZING from the KEY budget: ``hier_strip_for_budget`` honours
   ``c <= M / (b*n*(6d+4))`` (the M1-review ``6d+4`` correction) — at
   n=2048, d=128, M=128 MiB that is ~10-11 blocks, NOT the 15 a score-only
   ``4+4d`` bound would give.
6. NO ``[b, n, nb]`` ALLOCATION when ON (alloc spy, like M1's), with a positive
   control.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_hier_integration.py -v
    PYTHONPATH=$PWD <venv>/bin/python tests/test_dsv41_hier_integration.py
"""

from __future__ import annotations

import json
import math
import unittest
from typing import Any

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as IX
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as H
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0
from mlx_lm.models.deepseek_v41.layers import cos_sin_at, rope_tail
from mlx_lm.models.deepseek_v41.model import SharedState

RESULTS: list[dict[str, Any]] = []
GATE = 0.99


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def one_hot_keys(target: np.ndarray[Any, Any]):
    """(q, index_k, w) whose score row equals ``target`` [n, nb] (see M1 tests)."""
    n, nb = target.shape
    q = np.zeros((1, n, 1, n), np.float32)
    q[0, :, 0, :] = np.eye(n, dtype=np.float32)
    kk = np.zeros((1, nb, n), np.float32)
    kk[0, :, :] = target.T
    w = np.ones((1, n, 1), np.float32)
    return mx.array(q), mx.array(kk).astype(mx.bfloat16), mx.array(w)


def ref_row_one_hot(index_k: mx.array, k: int):
    """Production exact top-k over the materialized row from the bf16 keys."""
    row = mx.swapaxes(index_k.astype(mx.float32), 1, 2)
    v, i = IX.topk_from_row(row, k)
    mx.eval(v, i)
    return v, np.array(i)


def rank_weighted_recall(got_row, ref_row) -> float:
    """Recall of ``ref_row``, weighting rank r by 1/log2(r+2)."""
    g = {int(x) for x in np.asarray(got_row).tolist()}
    num = den = 0.0
    for r, idx in enumerate(int(x) for x in np.asarray(ref_row).tolist()):
        wt = 1.0 / math.log2(r + 2)
        den += wt
        if idx in g:
            num += wt
    return num / den if den else 1.0


def mean_rw_recall(got_i: np.ndarray[Any, Any], ref_i: np.ndarray[Any, Any]) -> float:
    return float(np.mean([rank_weighted_recall(got_i[r], ref_i[r])
                          for r in range(ref_i.shape[0])]))


def index_overlap(a, b) -> float:
    a = np.asarray(a).reshape(-1, np.asarray(a).shape[-1])
    b = np.asarray(b).reshape(-1, np.asarray(b).shape[-1])
    inter = [len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b)]
    return float(np.mean(inter)) / a.shape[-1]


def broad_margin_case(rng, *, n, nb, h, d, n_hot, margin=4.0):
    """q/k/w with a wide-margin hot set (same construction as the bf16 tests)."""
    q = rng.standard_normal((1, n, h, d)) * (1.0 / np.sqrt(d))
    k = rng.standard_normal((1, nb, d)) * 0.1
    w = rng.random((1, n, h)) + 0.5
    u = rng.standard_normal(d)
    u = u / (np.linalg.norm(u) + 1e-9)
    hot = rng.choice(nb, size=n_hot, replace=False)
    for j in hot:
        k[0, j, :] += margin * u
    q[0, :, 0, :] += margin * u
    return (mx.array(q.astype(np.float32)), mx.array(k.astype(np.float32)),
            mx.array(w.astype(np.float32)))


def production_row(q, index_k, w, lens, nb, row_dtype=mx.float32):
    """The production score row (visibility-masked) at a chosen row dtype."""
    s = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), index_k.astype(mx.float32))
    s = mx.maximum(s, 0.0) * w[..., None].astype(mx.float32)
    s = mx.sum(s, axis=2).astype(row_dtype)
    vis = mx.arange(nb)[None, :] < lens
    return mx.where(vis[None], s, IX.NEG_INF)


def record(row: dict[str, Any]):
    RESULTS.append(row)
    print("  " + json.dumps(row))


def _run_under_alloc_spy(fn):
    """Record shapes passed to mx.zeros / mx.full and produced by concatenate."""
    allocs, concats = [], []
    real_z, real_f, real_c = mx.zeros, mx.full, mx.concatenate

    def z(shape, *a, **k):
        allocs.append(("zeros", tuple(shape)))
        return real_z(shape, *a, **k)

    def f(shape, *a, **k):
        allocs.append(("full", tuple(shape)))
        return real_f(shape, *a, **k)

    def c(arrays, *a, **k):
        out = real_c(arrays, *a, **k)
        concats.append(tuple(out.shape))
        return out

    mx.zeros, mx.full, mx.concatenate = z, f, c
    try:
        fn()
    finally:
        mx.zeros, mx.full, mx.concatenate = real_z, real_f, real_c
    return allocs, concats


# --------------------------------------------------------------------------
# tiny Indexer fixtures
# --------------------------------------------------------------------------
def _hier_args(**over):
    """Tiny args with candidate_block_size == HIER_BLOCK (8) for positive tests."""
    base: dict[str, Any] = dict(
        dim=64, n_layers=1, n_heads=4, head_dim=128, rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
        compress_ratios=(2, 2), kv_source_layers=(0,), index_source_layers=(0,),
        index_n_heads=2, index_head_dim=32, index_topk=6, max_seq_len=4096,
        candidate_source_layer=0, candidate_topk_blocks=2, candidate_block_size=8)
    base.update(over)
    return ModelArgs(**base)


_INDEXERS: dict[Any, IX.Indexer] = {}


def _indexer(args: ModelArgs, layer_id: int) -> IX.Indexer:
    """Deterministic tiny Indexer (same weights every call).

    Not cached on ``id(args)`` — Python reuses object ids after GC, which made
    a stale indexer leak across tests. The model is a couple of small linears,
    so rebuilding is cheap; the seed fixes the weights.
    """
    mx.random.seed(0)
    idx = IX.Indexer(args, layer_id)
    mx.eval(idx.parameters())
    return idx


def _call_inputs(rng, args, *, b=1, n=8, nb=120):
    sp = 2 * nb                                   # all columns visible
    x = mx.array(rng.standard_normal((b, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((b, n, args.q_lora_rank)).astype(np.float32))
    ik = mx.array(rng.standard_normal((b, nb, args.index_head_dim)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    return x, qr, sp, fv, ik


class _GateFixture(unittest.TestCase):
    """Save/restore the env-read module constants the way the bf16 tests do."""

    def setUp(self):
        self._saved = {"tile": IX._TILE, "min_nb": IX._TILE_MIN_NB,
                       "force": IX._TILE_FORCE, "row": IX._ROW_DTYPE,
                       "hier": IX._HIER, "block": IX._HIER_BLOCK,
                       "strip": IX._HIER_STRIP, "of": IX._HIER_OVERFETCH,
                       "estrip": IX._HIER_EXACT_STRIP}
        IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 8, 0, True

    def tearDown(self):
        IX._TILE = self._saved["tile"]
        IX._TILE_MIN_NB = self._saved["min_nb"]
        IX._TILE_FORCE = self._saved["force"]
        IX._ROW_DTYPE = self._saved["row"]
        IX._HIER = self._saved["hier"]
        IX._HIER_BLOCK = self._saved["block"]
        IX._HIER_STRIP = self._saved["strip"]
        IX._HIER_OVERFETCH = self._saved["of"]
        IX._HIER_EXACT_STRIP = self._saved["estrip"]


# --------------------------------------------------------------------------
# 1. env OFF == production, byte-identical
# --------------------------------------------------------------------------
class EnvOffByteIdentityTest(_GateFixture):
    def test_default_gate_is_off(self):
        self.assertFalse(IX._HIER, "DSV41_INDEXER_HIER must default to OFF")
        record({"case": "gate_default_off", "hier": IX._HIER})

    def test_off_never_enters_hier_branch(self):
        """With the gate off, hierarchical_topk_prod is never called."""
        seen = {"called": 0}
        real = H.hierarchical_topk_prod

        def spy(*a, **k):
            seen["called"] += 1
            return real(*a, **k)

        H.hierarchical_topk_prod = spy
        try:
            IX._HIER = False
            args = _hier_args()
            idx = _indexer(args, 0)
            x, qr, sp, fv, ik = _call_inputs(np.random.default_rng(1), args)
            out = idx(x, qr, sp, 0, fv, ik, SharedState())
            mx.eval(out)
        finally:
            H.hierarchical_topk_prod = real
        self.assertEqual(seen["called"], 0, "OFF path entered the hier branch")
        record({"case": "off_never_calls_hier", "calls": seen["called"]})

    def test_off_output_is_byte_identical_to_production_impl(self):
        """OFF returns exactly what the production impl returns, same inputs.

        The production impl is captured via a spy (so the *real* q/k/w/lens the
        Indexer built are used), then re-run to derive the reference. The off
        output must match it in indices, and the captured impl's selected
        *scores* must be bit-equal to the untiled reference row's values at
        those indices (the documented tiled/untiled exactness).
        """
        IX._HIER = False
        args = _hier_args()
        idx = _indexer(args, 0)
        x, qr, sp, fv, ik = _call_inputs(np.random.default_rng(2), args)
        offset = 0

        captured: dict[str, Any] = {}
        real = IX._tiled_scores_buffer

        def spy(q32, index_k, w32, lens, nb, k, tile, *, cand_mask=None, cand_src=None):
            out = real(q32, index_k, w32, lens, nb, k, tile,
                       cand_mask=cand_mask, cand_src=cand_src)
            captured["args"] = (q32, index_k, w32, lens, nb, k, tile, cand_mask, cand_src)
            return out

        IX._tiled_scores_buffer = spy
        try:
            shared = SharedState()
            out = idx(x, qr, sp, offset, fv, ik, shared)
            mx.eval(out, shared.candidates)
        finally:
            IX._tiled_scores_buffer = real

        self.assertIn("args", captured, "production impl was not used when OFF")
        q32, index_k, w32, lens, nb, k, tile, cmask, csrc = captured["args"]
        v, i, blk = real(q32, index_k, w32, lens, nb, k, tile,
                         cand_mask=cmask, cand_src=csrc)
        valid = mx.isfinite(v) & (i < lens.astype(mx.int32)[None])
        ref = mx.where(valid, i + offset, mx.array(-1, mx.int32))
        mx.eval(ref, v, i)

        got, refa = np.array(out), np.array(ref)
        self.assertTrue(np.array_equal(got, refa),
                        "OFF indices diverge from production impl")

        # selected scores: the impl's finite values vs the untiled reference row
        # gathered at the selected indices -> bit-equal (tiled/untiled exactness)
        row = production_row(q32, index_k, w32, lens, nb, row_dtype=IX._ROW_DTYPE)
        vs = mx.take_along_axis(row, i, axis=-1)
        mx.eval(vs)
        # cast off bf16 before numpy (np.array on a bf16 mlx array yields raw
        # 2-byte records, not numbers)
        v_np = np.array(v.astype(mx.float32))
        vs_np = np.array(vs.astype(mx.float32))
        fin = np.isfinite(v_np)
        self.assertTrue(np.array_equal(v_np[fin], vs_np[fin]),
                        "OFF selected scores differ from the untiled reference")
        record({"case": "off_byte_identical", "indices_equal": True,
                "scores_equal": True, "slots": int(got.size),
                "valid": int((got >= 0).sum()), "finite_scores": int(fin.sum())})

    def test_off_tiled_matches_off_untiled(self):
        """Both production branches (tiled/untiled) agree with the gate OFF."""
        IX._HIER = False
        rng = np.random.default_rng(4)
        n, nb, h, d = 8, 400, 4, 32
        # hot columns aligned to query head 0's direction -> an unconditional
        # top-6 boundary (no near-ties), so the two production paths must agree
        q = (rng.standard_normal((1, n, h, d)) * 0.1).astype(np.float32)
        u = q[0, 0, 0, :].copy()
        u = u / (np.linalg.norm(u) + 1e-9)
        ik = (rng.standard_normal((1, nb, d)) * 0.01).astype(np.float32)
        for c in range(6):
            ik[0, c, :] = u * 50.0
        w = (rng.random((1, n, h)) + 0.5).astype(np.float32)
        lens = mx.full((n, 1), nb, mx.int32)
        qm, km, wm = mx.array(q), mx.array(ik), mx.array(w)
        vt, it, _ = IX._tiled_scores_buffer(qm, km, wm, lens, nb, 6, 8)
        mx.eval(vt, it)
        row = production_row(qm, km, wm, lens, nb, row_dtype=IX._ROW_DTYPE)
        _, iu = IX.topk_from_row(row, 6)
        mx.eval(iu)
        ov = index_overlap(np.array(it), np.array(iu))
        # the unconditional hot set is exactly {0..5} for every row
        self.assertEqual(set(np.array(it)[0, 0].tolist()), set(range(6)))
        record({"case": "off_tiled_vs_untiled", "overlap": round(ov, 4)})
        self.assertGreaterEqual(ov, GATE)


# --------------------------------------------------------------------------
# 2. env ON == exact path within the recall gate
# --------------------------------------------------------------------------
class EnvOnRecallTest(_GateFixture):
    K = 512

    def test_on_matches_exact_realistic_one_hot(self):
        rng = np.random.default_rng(0)
        n, nb, k = 48, 16384, self.K
        gens = {
            "sparse_peaks": lambda: (rng.random((n, nb)) * 0.5
                                     + (rng.random((n, nb)) < 0.05)
                                     * (5 + rng.random((n, nb)) * 10)),
            "exp_tail": lambda: -np.log(rng.random((n, nb)) + 1e-12),
            "pareto_1p5": lambda: (rng.random((n, nb)) + 1e-12) ** (-1 / 1.5),
            "gauss_relu": lambda: np.maximum(rng.standard_normal((n, nb)), 0.0),
            "top_heavy": lambda: (rng.random((n, nb)) * 10
                                  + (rng.random((n, nb)) < 0.001) * 100),
        }
        lens = mx.full((n, 1), nb, mx.int32)
        worst = 1.0
        for name, gen in gens.items():
            target = gen().astype(np.float32)
            q, kk, w = one_hot_keys(target)
            rv, ri = ref_row_one_hot(kk, k)
            hv, hi, _ = H.hierarchical_topk_prod(
                q, kk, w, lens, k, 8, 1024, 80, 16)
            mx.eval(hv, hi)
            rw = mean_rw_recall(np.array(hi)[0], ri[0])
            worst = min(worst, rw)
            record({"case": f"on_recall/{name}", "rw_recall": round(rw, 6),
                    "gate": GATE})
            self.assertGreaterEqual(rw, GATE, f"{name}: rw recall {rw:.4f}")
        record({"case": "on_recall/worst", "rw_recall": round(worst, 6)})

    def test_on_matches_exact_real_keys_multihead(self):
        rng = np.random.default_rng(1)
        for tag, (n, nb, h, d, k) in {
            "n48x16384xh8xd64": (48, 16384, 8, 64, 512),
            "n24x32768xh4xd32": (24, 32768, 4, 32, 512),
        }.items():
            q, kk, w = broad_margin_case(rng, n=n, nb=nb, h=h, d=d, n_hot=k)
            lens = mx.full((n, 1), nb, mx.int32)
            row = production_row(q, kk, w, lens, nb, row_dtype=mx.bfloat16)
            rv, ri = IX.topk_from_row(row, k)
            mx.eval(rv, ri)
            hv, hi, _ = H.hierarchical_topk_prod(q, kk, w, lens, k, 8, 4096, 80, 16)
            mx.eval(hv, hi)
            rw = mean_rw_recall(np.array(hi)[0], np.array(ri)[0])
            record({"case": f"on_recall/real/{tag}", "rw_recall": round(rw, 6)})
            self.assertGreaterEqual(rw, GATE, f"{tag}: rw recall {rw:.4f}")

    def test_on_end_to_end_indexer_recall(self):
        """Full Indexer.__call__: ON vs OFF selection overlap on wide-margin."""
        args = _hier_args(candidate_source_layer=-1)
        idx = _indexer(args, 0)
        rng = np.random.default_rng(6)
        b, n, nb = 1, 8, 400
        sp = 2 * nb
        x = mx.array(rng.standard_normal((b, n, args.dim)).astype(np.float32))
        qr = mx.array(rng.standard_normal((b, n, args.q_lora_rank)).astype(np.float32))
        # hot columns = the real (roped, fake-quantized) query head-0 direction,
        # scaled hugely -> an unconditional top-6, so ON/OFF must agree exactly
        qd = idx.wq_b(qr).reshape(b, n, args.index_n_heads, args.index_head_dim)
        fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
        qd = rope_tail(qd, args.rope_head_dim, *cos_sin_at(fv, mx.arange(sp, sp + n)))
        qd = fake_quant_fp4_ue8m0(qd, 32)
        mx.eval(qd)
        base = (rng.standard_normal((b, nb, args.index_head_dim)) * 0.01).astype(np.float32)
        for c in range(6):
            base[0, c, :] = np.array(qd)[0, 0, 0, :] * 50.0
        ik = mx.array(base)
        IX._HIER = False
        o_off = np.array(idx(x, qr, sp, 0, fv, ik, SharedState()))
        IX._HIER = True
        IX._HIER_BLOCK = 8
        IX._HIER_STRIP = 4096
        o_on = np.array(idx(x, qr, sp, 0, fv, ik, SharedState()))
        ov = index_overlap(o_on, o_off)
        record({"case": "on_end_to_end_overlap", "overlap": round(ov, 4),
                "gate": GATE})
        # ON must reproduce the OFF (production) selection through the real
        # entry point; the model's head weights may sign-flip, so this is a
        # parity check, not a ground-truth one (ground truth is pinned at the
        # impl level by test_on_matches_exact_*).
        self.assertGreaterEqual(ov, GATE, f"end-to-end overlap {ov:.4f}")
        # every ON index addresses a valid column (or is the -1 mask), and the
        # selection is non-degenerate (not all -1)
        valid = o_on[o_on >= 0]
        self.assertTrue(((valid >= 0) & (valid < nb)).all(), "out-of-range index")
        self.assertGreater(int((o_on >= 0).sum()), 0, "everything was masked")


# --------------------------------------------------------------------------
# 3. candidate-mask parity (layer-20-equivalent role)
# --------------------------------------------------------------------------
class CandidateMaskParityTest(_GateFixture):
    def test_fused_mask_equals_select_candidate_blocks(self):
        """The fused mask == select_candidate_blocks fed the SAME coarse row."""
        rng = np.random.default_rng(7)
        n, nb, h, d, TKB, BS = 16, 2048, 4, 64, 64, 8
        q, kk, w = broad_margin_case(rng, n=n, nb=nb, h=h, d=d, n_hot=200)
        lens = mx.full((n, 1), nb, mx.int32)

        # the coarse row exactly as H scores it (bf16 inputs + accumulate)
        coarse = H._score_shared_columns(q.astype(mx.bfloat16), kk.astype(mx.bfloat16),
                                         w.astype(mx.bfloat16))
        vis = mx.arange(nb)[None, :] < lens
        coarse = mx.where(vis[None], coarse, mx.array(H.NEG_INF, dtype=mx.bfloat16))
        coarse_f = coarse.astype(mx.float32)
        ref = IX.select_candidate_blocks(coarse_f, lens, TKB, BS)
        mx.eval(ref)

        bm = H.coarse_block_scores(q, kk, w, lens, block=BS, strip=512)
        fused = H.candidate_mask_from_block_maxima(bm, lens, TKB, BS)
        mx.eval(fused)
        a, b = np.array(ref), np.array(fused)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.array_equal(a, b), "fused candidate mask != production mask")
        # block maxima bit-equal to the reshape-max of that coarse row
        refbm = coarse_f.reshape(1, n, nb // BS, BS).max(axis=-1)
        bmax = float(np.abs(np.array(bm) - np.array(refbm)).max())
        record({"case": "candidate_mask_parity", "exact_equal": True,
                "kept": int(b.sum()), "blockmax_max_abs_diff": bmax})
        self.assertEqual(bmax, 0.0, "coarse block maxima differ from row reshape-max")

    def test_fused_mask_block_aligned_and_sane(self):
        rng = np.random.default_rng(8)
        n, nb, h, d = 8, 1024, 4, 32
        q, kk, w = broad_margin_case(rng, n=n, nb=nb, h=h, d=d, n_hot=120)
        lens = mx.full((n, 1), nb, mx.int32)
        _, _, mask = H.hierarchical_topk_prod(q, kk, w, lens, 512, 8, 512, 80, 16,
                                              cand_src=(8, 8))
        mx.eval(mask)
        m = np.array(mask)
        self.assertEqual(m.dtype, np.bool_)
        self.assertEqual(m.shape, (1, n, nb))
        for row in range(n):
            runs = np.split(m[0, row], np.where(np.diff(m[0, row].astype(int)))[0] + 1)
            for run in runs:
                self.assertEqual(len(run) % 8, 0, "mask not block-aligned")
        self.assertGreater(int(m.sum()), 0)
        record({"case": "fused_mask_block_aligned", "kept": int(m.sum())})

    def test_block_size_mismatch_is_rejected(self):
        """candidate_block_size != HIER_BLOCK on the source layer must assert."""
        IX._HIER = True
        IX._HIER_BLOCK = 8
        args = _hier_args(candidate_source_layer=0, candidate_block_size=4)  # 4 != 8
        idx = _indexer(args, 0)
        x, qr, sp, fv, ik = _call_inputs(np.random.default_rng(9), args)
        with self.assertRaises(AssertionError):
            idx(x, qr, sp, 0, fv, ik, SharedState())
        record({"case": "block_size_mismatch_asserts", "block": 8, "cand_block": 4})

    def test_source_publishes_nb_wide_mask_even_uneven(self):
        """Published mask width == nb exactly, incl. nb not divisible by block."""
        IX._HIER = True
        IX._HIER_BLOCK = 8
        IX._HIER_STRIP = 1024
        IX._HIER_EXACT_STRIP = 80
        args = _hier_args(candidate_source_layer=0, candidate_block_size=8)
        idx = _indexer(args, 0)
        for nb in (120, 123, 117):                 # 15, 15.4, 14.6 blocks
            x, qr, sp, fv, ik = _call_inputs(np.random.default_rng(nb), args, nb=nb)
            shared = SharedState()
            out = idx(x, qr, sp, 0, fv, ik, shared)
            mx.eval(out, shared.candidates)
            self.assertEqual(shared.candidates.shape, (1, 8, nb),
                             f"published mask is not nb-wide at nb={nb}")
            valid = np.array(out)[np.array(out) >= 0]
            self.assertTrue((valid < nb).all(), f"index out of range at nb={nb}")
            record({"case": f"source_mask_width/nb{nb}",
                    "shape": list(shared.candidates.shape),
                    "kept": int(np.array(shared.candidates).sum())})

    def test_source_publishes_mask_from_call(self):
        """Indexer.__call__ with ON publishes shared.candidates (layer-20 role)."""
        IX._HIER = True
        IX._HIER_BLOCK = 8
        IX._HIER_STRIP = 1024
        IX._HIER_EXACT_STRIP = 80
        args = _hier_args(candidate_source_layer=0, candidate_block_size=8)
        idx = _indexer(args, 0)
        x, qr, sp, fv, ik = _call_inputs(np.random.default_rng(10), args)
        shared = SharedState()
        out = idx(x, qr, sp, 0, fv, ik, shared)
        mx.eval(out, shared.candidates)
        cand = shared.candidates
        self.assertIsNotNone(cand, "source did not publish candidates")
        assert cand is not None
        self.assertEqual(cand.shape[-1], 120)
        record({"case": "source_publishes_mask", "shape": list(cand.shape),
                "kept": int(np.array(cand).sum())})


# --------------------------------------------------------------------------
# 4. composition inside an existing candidates mask (consumer role)
# --------------------------------------------------------------------------
class CompositionTest(_GateFixture):
    def _consumer_case(self, rng):
        n, nb, h, d = 16, 2048, 4, 64
        q, kk, w = broad_margin_case(rng, n=n, nb=nb, h=h, d=d, n_hot=200)
        lens = mx.full((n, 1), nb, mx.int32)
        # a candidate mask published by the production block selection
        row = production_row(q, kk, w, lens, nb, row_dtype=mx.float32)
        cmask = IX.select_candidate_blocks(row, lens, 64, 8)
        mx.eval(cmask)
        return q, kk, w, lens, cmask

    def test_consumer_all_indices_within_mask(self):
        rng = np.random.default_rng(11)
        q, kk, w, lens, cmask = self._consumer_case(rng)
        hv, hi, _ = H.hierarchical_topk_prod(q, kk, w, lens, 512, 8, 1024, 80, 16,
                                             cand_mask=cmask)
        mx.eval(hv, hi)
        idx = np.array(hi)[0]
        inside = np.take_along_axis(np.array(cmask)[0], idx, axis=1)
        # a padded (-inf) slot may carry an unmasked column; check finite slots
        fin = np.isfinite(np.array(hv)[0])
        leaked = (~inside) & fin
        self.assertFalse(bool(leaked.any()), "consumer selected a non-candidate column")
        record({"case": "consumer_within_mask", "finite_slots": int(fin.sum()),
                "out_of_mask_finite": int(leaked.sum())})

    def test_consumer_matches_production_consumer_path(self):
        rng = np.random.default_rng(12)
        q, kk, w, lens, cmask = self._consumer_case(rng)
        # production consumer: mask then exact top-k over the row
        vp, ip, _ = IX._tiled_scores_buffer(q, kk, w, lens, 2048, 512, 512,
                                            cand_mask=cmask)
        mx.eval(vp, ip)
        hv, hi, _ = H.hierarchical_topk_prod(q, kk, w, lens, 512, 8, 1024, 80, 16,
                                             cand_mask=cmask)
        mx.eval(hv, hi)
        rw = mean_rw_recall(np.array(hi)[0], np.array(ip)[0])
        record({"case": "consumer_vs_production", "rw_recall": round(rw, 6),
                "overlap": round(index_overlap(np.array(hi), np.array(ip)), 4)})
        self.assertGreaterEqual(rw, GATE, f"consumer recall {rw:.4f}")

    def test_consumer_composes_through_indexer_call(self):
        """Source then consumer layer through the real __call__ entry.

        Layer 0 publishes the candidate mask; layer 1 (a consumer) runs with it.
        The consumer's ON output must equal the OFF (production) output on the
        same mask — a real parity check, not just a range check.
        """
        IX._HIER = True
        IX._HIER_BLOCK = 8
        IX._HIER_STRIP = 1024
        IX._HIER_EXACT_STRIP = 80
        args = _hier_args(candidate_source_layer=0, candidate_block_size=8)
        src = _indexer(args, 0)
        con = _indexer(args, 1)                 # 1 > 0 => uses_candidates
        self.assertTrue(con.uses_candidates)
        self.assertTrue(con.ratio, "fixture: consumer ratio must be nonzero")
        rng = np.random.default_rng(13)
        x, qr, sp, fv, ik = _call_inputs(rng, args)
        shared = SharedState()
        out_src = src(x, qr, sp, 0, fv, ik, shared)
        mx.eval(out_src, shared.candidates)
        cmask = shared.candidates
        self.assertIsNotNone(cmask)
        # production consumer with the SAME mask
        shared_off = SharedState()
        shared_off.candidates = cmask
        IX._HIER = False
        out_off = con(x, qr, sp, 0, fv, ik, shared_off)
        mx.eval(out_off)
        # hierarchical consumer, fresh shared state seeded with the same mask
        shared_on = SharedState()
        shared_on.candidates = cmask
        IX._HIER = True
        out_on = con(x, qr, sp, 0, fv, ik, shared_on)
        mx.eval(out_on)
        # the consumer must not have overwritten the candidate mask
        self.assertTrue(np.array_equal(np.array(cmask), np.array(shared_on.candidates)),
                        "consumer clobbered shared.candidates")
        a, b = np.array(out_on), np.array(out_off)
        self.assertGreater(int((b >= 0).sum()), 0, "fixture: production consumer all-masked")
        ov = index_overlap(a, b)
        record({"case": "consumer_through_call", "overlap": round(ov, 4),
                "src_valid": int((np.array(out_src) >= 0).sum()),
                "con_valid": int((a >= 0).sum())})
        self.assertGreaterEqual(ov, GATE, f"consumer ON vs OFF overlap {ov:.4f}")


# --------------------------------------------------------------------------
# 5. strip sizing from the KEY budget ((6d+4))
# --------------------------------------------------------------------------
class StripFormulaTest(unittest.TestCase):
    def test_budget_formula_and_block_rounding(self):
        M = 128 << 20
        for (b, n, d) in [(1, 2048, 128), (1, 512, 128), (2, 2048, 128), (1, 2048, 64)]:
            c = IX.hier_strip_for_budget(b, n, d, M, block=8)
            self.assertEqual(c % 8, 0, "strip not a whole number of blocks")
            self.assertLessEqual(c * b * n * (6 * d + 4), M,
                                 "strip exceeds the (6d+4) key budget")
            # and the +one-block step would break the budget (budget is tight)
            self.assertGreater((c + 8) * b * n * (6 * d + 4), M - 1,
                               "strip is not the tight budget multiple")
        record({"case": "strip_budget_formula", "budget_MiB": M >> 20,
                "checks": [(b, n, d, IX.hier_strip_for_budget(b, n, d, M))
                           for (b, n, d) in [(1, 2048, 128), (1, 512, 128)]]})

    def test_production_point_is_10_to_11_blocks_not_15(self):
        """The M1-review correction: (6d+4) gives ~10-11 blocks, NOT 15."""
        M = 128 << 20
        n, d, b = 2048, 128, 1
        cols = IX.hier_strip_for_budget(b, n, d, M, block=8)
        blocks = cols // 8
        old_cols = M // (b * n * (4 + 4 * d))     # the superseded 4+4d bound
        old_blocks = old_cols // 8
        record({"case": "strip_point", "n": n, "d": d, "M_MiB": M >> 20,
                "cols": cols, "blocks": blocks,
                "old_cols": old_cols, "old_blocks": old_blocks})
        self.assertTrue(10 <= blocks <= 11, f"expected 10-11 blocks, got {blocks}")
        self.assertEqual(old_blocks, 15, "sanity: the old 4+4d bound gives 15")
        self.assertLess(blocks, old_blocks,
                        "the (6d+4) key budget must be tighter than score-only")

    def test_env_override_and_defaults(self):
        self.assertGreater(IX._HIER_STRIP, 0)
        self.assertGreaterEqual(IX._HIER_OVERFETCH, 16)
        self.assertEqual(IX._HIER_BLOCK, H.HIER_BLOCK)
        record({"case": "hier_knobs", "block": IX._HIER_BLOCK, "strip": IX._HIER_STRIP,
                "overfetch": IX._HIER_OVERFETCH, "exact_mb": IX._HIER_EXACT_MB})


# --------------------------------------------------------------------------
# 6. no [b, n, nb] materialization when ON
# --------------------------------------------------------------------------
class NoMaterializationTest(unittest.TestCase):
    def _spy_prod(self, *, cand_src=None, cand_mask=None):
        rng = np.random.default_rng(14)
        n, nb, h, d, k, block, strip = 32, 65536, 4, 64, 512, 8, 1024
        q, kk, w = broad_margin_case(rng, n=n, nb=nb, h=h, d=d, n_hot=512)
        lens = mx.full((n, 1), nb, mx.int32)
        allocs, concats = _run_under_alloc_spy(
            lambda: mx.eval(H.hierarchical_topk_prod(
                q, kk, w, lens, k, block, strip, 80, 16,
                cand_mask=cand_mask, cand_src=cand_src)))
        return nb, n, strip, k, allocs, concats

    def test_no_full_row_allocation(self):
        nb, n, strip, k, allocs, concats = self._spy_prod()
        record({"case": "no_mat/owner", "allocs": allocs[:8], "n_allocs": len(allocs)})
        bad = [a for a in allocs if len(a[1]) >= 2 and a[1][-1] >= nb and a[1][-2] == n]
        self.assertEqual(bad, [], f"[.., n, nb] allocated: {bad}")
        self.assertTrue(all(a[1][-1] < nb for a in allocs if len(a[1]) >= 1),
                        "an allocation is nb-wide or wider")
        wide = [s for s in concats if s and s[-1] > strip + k]
        self.assertEqual(wide, [], f"a concatenate exceeded strip+k: {wide}")

    def test_no_full_row_allocation_source_and_consumer(self):
        rng = np.random.default_rng(15)
        n, nb, h, d = 16, 32768, 4, 64
        q, kk, w = broad_margin_case(rng, n=n, nb=nb, h=h, d=d, n_hot=512)
        lens = mx.full((n, 1), nb, mx.int32)
        # a candidate mask that is a real [b, n, nb] bool (allowed input, not an
        # allocation we make); the point is the *selection* never materializes a row
        cmask = mx.zeros((1, n, nb), mx.bool_)
        cmask[:, :, ::4] = True
        mx.eval(cmask)
        for tag, kw in {"source": {"cand_src": (64, 8)},
                        "consumer": {"cand_mask": cmask}}.items():
            allocs, _ = _run_under_alloc_spy(
                lambda kw=kw: mx.eval(H.hierarchical_topk_prod(
                    q, kk, w, lens, 512, 8, 1024, 80, 16, **kw)))
            bad = [a for a in allocs if len(a[1]) >= 2 and a[1][-1] >= nb
                   and a[1][-2] == n]
            self.assertEqual(bad, [], f"{tag}: [.., n, nb] allocated: {bad}")
            record({"case": f"no_mat/{tag}", "n_allocs": len(allocs),
                    "max_last": max(a[1][-1] for a in allocs if a[1])})

    def test_alloc_spy_positive_control(self):
        n, nb = 8, 4096
        allocs, _ = _run_under_alloc_spy(lambda: mx.zeros((1, n, nb), mx.float32))
        bad = [a for a in allocs if len(a[1]) >= 2 and a[1][-1] >= nb and a[1][-2] == n]
        self.assertGreaterEqual(len(bad), 1, "spy failed to catch a full-row alloc")
        record({"case": "no_mat/positive_control", "catches": len(bad)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
    print(f"DSV41_HIER_INTEGRATION {json.dumps({'cases': len(RESULTS)})}")
