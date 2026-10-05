# Copyright © 2026 Adam Durham (hermes-gw)
"""Milestone-1 tests for the hierarchical / streamed indexer exact pass
(``mlx_lm/models/deepseek_v41/indexer_hierarchical.py``).

The module under test is a prototype (not wired into production). These tests
pin, on synthetic tensors only:

1. RECALL vs EXACT — rank-weighted top-k recall against a *materialized* ``[n, nb]``
   reference row, on realistic (peaked/heavy-tailed) distributions (assert
   ``>= 0.99``) and on adversarial distributions (assert ``>= 0.95``, reported).
   The reference is scored from the SAME bf16 key buffer the algorithm reads
   (production upcasts the bf16 cache), so the comparison isolates the
   hierarchical prune, not key quantization.
2. NO-MATERIALIZATION — an allocation spy proves the streaming path never
   allocates an ``[.., n, nb]``-shaped tensor.
3. INDEX-SPACE — indices equal the production :func:`indexer.topk_from_row`
   output (ascending, raw compressed-column space, clamped to ``[0, nb)``).
4. OVER-FETCH — an engineered coarse (bf16) near-tie case where the pure
   ``k``-block prune drops below full recall and ``k+overfetch`` recovers it.
5. EDGES — all-masked rows, ``nb < block``, uneven ``nb``, strips not a block
   multiple, ``top_blocks`` clamping.

Score construction: for the fully-controllable cases, queries are one-hot and
keys are stacked along the query axis so the produced score row *is* the target
matrix (``index_k[b, col, s] = target[s, col]``); for the "real-key" cases,
multi-head random keys are used and the reference row is built with the exact
production score expression.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_indexer_hierarchical.py -v
    PYTHONPATH=$PWD <venv>/bin/python tests/test_dsv41_indexer_hierarchical.py
"""

from __future__ import annotations

import json
import math
import unittest
from typing import Any

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer_hierarchical as H
from mlx_lm.models.deepseek_v41.indexer import topk_from_row

RESULTS: list[dict[str, Any]] = []


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def one_hot_keys(target: np.ndarray[Any, Any]):
    """Build (q, index_k, w) whose score row equals ``target`` [n, nb].

    ``score[s, c] = sum_h relu(q[s,h] . k[c]) * w[s,h]``. With ``h = 1`` and
    ``q[s] = e_s`` (dim ``d = n``), ``k[c] = target[:, c]`` and ``w = 1``, the
    score is exactly ``target[s, c]`` (up to bf16 rounding of the stored keys).
    """
    n, nb = target.shape
    q = np.zeros((1, n, 1, n), np.float32)
    q[0, :, 0, :] = np.eye(n, dtype=np.float32)
    kk = np.zeros((1, nb, n), np.float32)
    kk[0, :, :] = target.T                          # k[col, s] = target[s, col]
    w = np.ones((1, n, 1), np.float32)
    return mx.array(q), mx.array(kk).astype(mx.bfloat16), mx.array(w)


def ref_one_hot(index_k: mx.array, k: int):
    """Exact reference top-k over the materialized row from the bf16 keys."""
    row = mx.swapaxes(index_k.astype(mx.float32), 1, 2)          # [b, n, nb]
    v, i = topk_from_row(row, k)
    mx.eval(v, i)
    return v, np.array(i)


def ref_real_keys(q: mx.array, index_k: mx.array, w: mx.array, k: int):
    """Exact reference top-k with the production score expression, bf16 keys."""
    s = mx.einsum("bshd,btd->bsht", q.astype(mx.float32),
                  index_k.astype(mx.float32))
    s = mx.maximum(s, 0.0) * w.astype(mx.float32)[..., None]
    row = mx.sum(s, axis=2)                                      # [b, n, nb]
    v, i = topk_from_row(row, k)
    mx.eval(v, i)
    return v, np.array(i)


def rank_weighted_recall(got_row, ref_row) -> float:
    """Recall of ``ref_row``, weighting rank r by 1/log2(r+2) (rank-heavy)."""
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


def value_recall(got_v, ref_v) -> float:
    """Fraction of reference top-k *values* reproduced (values are exact)."""
    a = np.sort(np.asarray(got_v).reshape(-1))[::-1]
    b = np.sort(np.asarray(ref_v).reshape(-1))[::-1]
    m = min(a.shape[0], b.shape[0])
    fin = np.isfinite(b[:m])
    if not fin.any():
        return 1.0
    return float(np.mean(np.isclose(a[:m][fin], b[:m][fin], rtol=0, atol=1e-4)))


def record(row: dict[str, Any]):
    RESULTS.append(row)
    print("  " + json.dumps(row))


def gaussian_keys(rng, n, nb, h, d, scale=0.3):
    q = (rng.standard_normal((1, n, h, d)) * scale).astype(np.float32)
    kk = mx.array((rng.standard_normal((1, nb, d)) * scale).astype(np.float32))
    kk = kk.astype(mx.bfloat16)
    w = np.abs(rng.standard_normal((1, n, h))).astype(np.float32)
    return mx.array(q), kk, mx.array(w)


def lens_full(n, nb):
    return mx.full((n, 1), nb, mx.int32)


# --------------------------------------------------------------------------
# 1. recall vs exact
# --------------------------------------------------------------------------
class RecallTest(unittest.TestCase):
    K = 512
    NB = 16384
    N = 64

    def test_realistic_peaked_one_hot(self):
        """Peaked/heavy-tailed targets (typical of attention logits): >= 0.99."""
        rng = np.random.default_rng(0)
        n, nb, k = self.N, self.NB, self.K
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
        for name, gen in gens.items():
            target = gen().astype(np.float32)
            q, kk, w = one_hot_keys(target)
            rv, ri = ref_one_hot(kk, k)
            hv, hi = H.hierarchical_topk(q, kk, w, lens_full(n, nb), k=k,
                                         block=8, strip=1024, overfetch=16)
            mx.eval(hv, hi)
            hi = np.array(hi)[0]
            rw = mean_rw_recall(hi, ri[0])
            vr = value_recall(np.array(hv)[0], rv[0])
            record({"case": f"recall/{name}", "nb": nb, "rw_recall": round(rw, 6),
                    "value_recall": round(vr, 6), "gate": 0.99})
            self.assertGreaterEqual(rw, 0.99, f"{name}: rw recall {rw:.4f}")

    def test_realistic_real_keys_multihead(self):
        """Real multi-head random keys at production k: >= 0.99."""
        rng = np.random.default_rng(0)
        for tag, (n, nb, h, d, k) in {
            "n64x16384xh8xd128": (64, 16384, 8, 128, 512),
            "n32x32768xh4xd64": (32, 32768, 4, 64, 512),
        }.items():
            q, kk, w = gaussian_keys(rng, n, nb, h, d)
            rv, ri = ref_real_keys(q, kk, w, k)
            hv, hi = H.hierarchical_topk(q, kk, w, lens_full(n, nb), k=k,
                                         block=8, strip=1024, overfetch=16)
            mx.eval(hv, hi)
            rw = mean_rw_recall(np.array(hi)[0], ri[0])
            record({"case": f"recall/real/{tag}", "rw_recall": round(rw, 6),
                    "gate": 0.99})
            self.assertGreaterEqual(rw, 0.99, f"{tag}: rw recall {rw:.4f}")

    def test_big_nb_emulation(self):
        """Deep-context-emulating nb (524288) with small n: >= 0.99."""
        rng = np.random.default_rng(5)
        n, nb, h, d, k = 16, 524288, 2, 32, 512
        q, kk, w = gaussian_keys(rng, n, nb, h, d)
        rv, ri = ref_real_keys(q, kk, w, k)
        hv, hi = H.hierarchical_topk(q, kk, w, lens_full(n, nb), k=k,
                                     block=8, strip=4096, overfetch=16)
        mx.eval(hv, hi)
        rw = mean_rw_recall(np.array(hi)[0], ri[0])
        record({"case": "recall/big_nb_emulation", "nb": nb, "n": n,
                "rw_recall": round(rw, 6), "gate": 0.99})
        self.assertGreaterEqual(rw, 0.99, f"nb={nb}: rw recall {rw:.4f}")

    def test_adversarial_reported(self):
        """Adversarial distributions: assert >= 0.95, report the observed value."""
        rng = np.random.default_rng(1)
        n, nb, k = self.N, self.NB, self.K
        block = 8

        def uniform():
            return (rng.random((n, nb)) * 10).astype(np.float32)

        def clustered_near_ties():
            t = (rng.random((n, nb)) * 0.5).astype(np.float32)
            blocks = rng.choice(nb // block, size=600, replace=False)
            for i, blk in enumerate(blocks):
                col = blk * block + int(rng.integers(0, block))
                t[:, col] = 10.0 + np.arange(n) * 1e-4 + (i % 5) * 1e-5
            return t

        def few_dominant_peaks():
            t = (rng.random((n, nb)) * 0.2).astype(np.float32)
            for _ in range(40):
                s = int(rng.integers(0, n))
                c = int(rng.integers(0, nb))
                t[s, c] = 100.0
            return t

        def boundary_straddling():
            t = (rng.random((n, nb)) * 0.5).astype(np.float32)
            for i in range(600):
                blk = int(rng.integers(0, nb // block - 1))
                # deliberately sit ON block edges (last col / first col)
                for off in (block - 1, 0):
                    t[:, blk * block + off] = 10.0 + np.arange(n) * 1e-4 + (i % 3) * 1e-5
            return t

        cases = {
            "uniform_random": uniform(),
            "clustered_near_ties": clustered_near_ties(),
            "top_heavy_few_peaks": few_dominant_peaks(),
            "peaks_on_block_boundaries": boundary_straddling(),
        }
        for name, target in cases.items():
            q, kk, w = one_hot_keys(target)
            rv, ri = ref_one_hot(kk, k)
            best = 0.0
            for of in (16, 64, 256):
                hv, hi = H.hierarchical_topk(q, kk, w, lens_full(n, nb), k=k,
                                             block=block, strip=1024, overfetch=of)
                mx.eval(hv, hi)
                r = mean_rw_recall(np.array(hi)[0], ri[0])
                best = max(best, r)
                record({"case": f"adversarial/{name}", "overfetch": of,
                        "rw_recall": round(r, 6), "gate": 0.95})
            self.assertGreaterEqual(best, 0.95, f"{name}: best rw recall {best:.4f}")


# --------------------------------------------------------------------------
# 2. over-fetch effect
# --------------------------------------------------------------------------
class OverfetchTest(unittest.TestCase):
    N, NB, K, BLOCK = 64, 8192, 512, 8

    def _alias_target(self, nblk_tied, gap=0.004, seed=0, scale=100.0):
        """Per-block single hot column whose fp32 magnitudes are distinct but
        collapse to the SAME bf16 value (ULP at 100 is ~0.5) -> the coarse pass
        sees exact block-max ties around the k-th cut."""
        rng = np.random.default_rng(seed)
        n, nb, block = self.N, self.NB, self.BLOCK
        q = np.zeros((1, n, 2, 8), np.float32)
        q[0, :, 0, 0] = 1.0
        q[0, :, 1, 1] = 1.0 + 0.3 * np.arange(n) / n          # per-row noise gain
        kk = np.zeros((1, nb, 8), np.float32)
        cols = np.arange(nblk_tied) * block + rng.integers(0, block, size=nblk_tied)
        kk[0, cols, 0] = scale - np.arange(nblk_tied) * gap   # fp32-distinct, bf16-aliased
        kk[0, :, 1] = rng.random(nb) * 0.1                    # small per-column noise
        w = np.ones((1, n, 2), np.float32)
        return mx.array(q), mx.array(kk).astype(mx.bfloat16), mx.array(w)

    def test_overfetch_sweep_engineered(self):
        q, kk, w = self._alias_target(528)                    # > k blocks at the cut
        rv, ri = ref_real_keys(q, kk, w, self.K)
        lens = lens_full(self.N, self.NB)
        got = {}
        for of in (0, 4, 16, 64):
            hv, hi = H.hierarchical_topk(q, kk, w, lens, k=self.K, block=self.BLOCK,
                                         strip=1024, overfetch=of)
            mx.eval(hv, hi)
            r = mean_rw_recall(np.array(hi)[0], ri[0])
            vr = value_recall(np.array(hv)[0], rv[0])
            got[of] = r
            record({"case": "overfetch_sweep/engineered", "overfetch": of,
                    "rw_recall": round(r, 6), "value_recall": round(vr, 6)})
        self.assertLess(got[0], got[16] - 1e-6,
                        f"overfetch=0 ({got[0]:.4f}) did not drop vs 16 ({got[16]:.4f})")
        self.assertGreaterEqual(got[16], 0.99, f"overfetch=16 only {got[16]:.4f}")

    def test_overfetch_sweep_real_keys(self):
        """Reported for the record: real keys, sweep 0/4/16."""
        rng = np.random.default_rng(3)
        n, nb, h, d, k = 64, 32768, 4, 64, 512
        q, kk, w = gaussian_keys(rng, n, nb, h, d, scale=3.0)
        rv, ri = ref_real_keys(q, kk, w, k)
        lens = lens_full(n, nb)
        for of in (0, 4, 16):
            hv, hi = H.hierarchical_topk(q, kk, w, lens, k=k, block=8,
                                         strip=1024, overfetch=of)
            mx.eval(hv, hi)
            record({"case": "overfetch_sweep/real_keys", "overfetch": of,
                    "rw_recall": round(mean_rw_recall(np.array(hi)[0], ri[0]), 6)})


# --------------------------------------------------------------------------
# 3. index-space correctness
# --------------------------------------------------------------------------
class IndexSpaceTest(unittest.TestCase):
    def test_indices_match_production_semantics(self):
        rng = np.random.default_rng(2)
        n, nb, k, block = 16, 4096, 64, 8
        target = (rng.random((n, nb)) * 0.001).astype(np.float32)
        forced = [7, 100, 511, 4095]                       # unique, bf16-distinct, top
        target[:, forced] = 50.0
        target[:, 2048] = 40.0                             # straddles a block edge
        q, kk, w = one_hot_keys(target)
        rv, ri = ref_one_hot(kk, k)
        hv, hi = H.hierarchical_topk(q, kk, w, lens_full(n, nb), k=k, block=block,
                                     strip=512, overfetch=16)
        mx.eval(hv, hi)
        hi = np.array(hi)[0]
        ri = ri[0]
        # production contract: ascending, in [0, nb)
        self.assertTrue(bool(np.all(np.diff(hi, axis=-1) >= 0)), "not ascending")
        self.assertTrue(hi.min() >= 0 and hi.max() < nb, "index out of range")
        self.assertTrue(bool(np.array_equal(hi, ri)),
                        "hierarchical indices differ from production topk_from_row")
        for c in forced:
            self.assertIn(c, hi[0].tolist())
        record({"case": "index_space", "exact_match_production": True,
                "ascending": True, "in_range": True, "nb": nb, "k": k})

    def test_value_parity(self):
        """Values are the exact global top-k (indices may tie-break)."""
        rng = np.random.default_rng(4)
        n, nb, k = 32, 16384, 512
        target = (rng.random((n, nb)) ** 2 * 10).astype(np.float32)
        q, kk, w = one_hot_keys(target)
        rv, ri = ref_one_hot(kk, k)
        hv, hi = H.hierarchical_topk(q, kk, w, lens_full(n, nb), k=k, block=8,
                                     strip=1024, overfetch=16)
        mx.eval(hv, hi)
        vr = value_recall(np.array(hv)[0], rv[0])
        record({"case": "value_parity", "value_recall": round(vr, 6)})
        self.assertGreaterEqual(vr, 0.999, f"value recall {vr:.4f}")


# --------------------------------------------------------------------------
# 4. no-materialization
# --------------------------------------------------------------------------
def _run_under_alloc_spy(fn):
    """Record the shapes passed to mx.zeros / mx.full and produced by mx.concatenate."""
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


class NoMaterializationTest(unittest.TestCase):
    def test_streaming_never_allocates_full_row(self):
        rng = np.random.default_rng(6)
        n, nb, h, d, k, block, strip = 32, 65536, 4, 64, 512, 8, 1024
        q, kk, w = gaussian_keys(rng, n, nb, h, d)
        lens = lens_full(n, nb)

        allocs, concats = _run_under_alloc_spy(
            lambda: mx.eval(H.hierarchical_topk(q, kk, w, lens, k=k, block=block,
                                                strip=strip, overfetch=16)))
        record({"case": "no_materialization", "allocs": allocs, "concats": concats})

        # no allocation whose trailing (n, nb) signature matches the full row;
        # the widest allocation is the block-maxima buffer nb/block < nb.
        bad = [a for a in allocs
               if len(a[1]) >= 2 and a[1][-1] >= nb and a[1][-2] == n]
        self.assertEqual(bad, [], f"an [.., n, nb]-shaped tensor was allocated: {bad}")
        self.assertTrue(all(a[1][-1] < nb for a in allocs if len(a[1]) >= 1),
                        "an allocation is nb-wide or wider")
        # every concatenate produces a strip-scale (<= strip + k) transient
        wide = [s for s in concats if s and s[-1] > strip + k]
        self.assertEqual(wide, [], f"a concatenate produced a >strip transient: {wide}")
        self.assertTrue(concats, "expected strip concatenations (structure check)")

    def test_alloc_spy_positive_control(self):
        """The spy must catch a REAL full-row materialization."""
        n, nb = 8, 4096
        allocs, _ = _run_under_alloc_spy(lambda: mx.zeros((1, n, nb), mx.float32))
        bad = [a for a in allocs
               if len(a[1]) >= 2 and a[1][-1] >= nb and a[1][-2] == n]
        record({"case": "no_materialization/positive_control",
                "materializing_allocs": len(bad)})
        self.assertGreaterEqual(len(bad), 1, "spy failed to catch a full-row alloc")

    def test_largest_transient_bounded_algebraically(self):
        """Largest score transient is [b, n, strip]; key transient [b, n, strip, d]."""
        rng = np.random.default_rng(6)
        n, nb, h, d, k, block, strip = 32, 65536, 4, 64, 512, 8, 1024
        q, kk, w = gaussian_keys(rng, n, nb, h, d)
        bp = strip // block
        allocs, concats = _run_under_alloc_spy(
            lambda: mx.eval(H.hierarchical_topk(q, kk, w, lens_full(n, nb), k=k,
                                                block=block, strip=strip, overfetch=16)))
        max_width = max([a[1][-1] for a in allocs if a[1]] + [c[-1] for c in concats if c])
        # widest legitimate transient is the coarse block-maxima buffer (nb/block);
        # the strip merge concatenates [k] + [strip]. Never anything nb-wide.
        expected = max(nb // block, 2 * k + strip)
        record({"case": "transient_bound", "max_last_dim": max_width,
                "expected_max": expected, "row_width": nb, "bp_times_block": bp * block})
        self.assertLessEqual(max_width, expected, "a transient exceeded the expected bound")
        self.assertLess(max_width, nb, "a transient reached the full row width")


# --------------------------------------------------------------------------
# 5. edges
# --------------------------------------------------------------------------
class EdgeTest(unittest.TestCase):
    def test_all_masked_row_no_leak(self):
        """lens = 0 -> every value -inf and no visible index survives."""
        n, nb, block, k = 8, 4096, 8, 64
        target = np.ones((n, nb), np.float32)
        q, kk, w = one_hot_keys(target)
        hv, hi = H.hierarchical_topk(q, kk, w, mx.zeros((n, 1), mx.int32),
                                     k=k, block=block, strip=512, overfetch=16)
        mx.eval(hv, hi)
        self.assertEqual(int(mx.isfinite(hv).sum()), 0, "finite value in masked row")
        self.assertTrue(int(hi.min()) >= 0 and int(hi.max()) < nb,
                        "returned index out of [0, nb)")
        record({"case": "edge/all_masked", "finite_values": 0,
                "idx_min": int(hi.min()), "idx_max": int(hi.max()), "nb": nb})

    def test_partial_visibility(self):
        """lens < nb: only visible columns may appear."""
        n, nb, block, k = 8, 4096, 8, 64
        rng = np.random.default_rng(7)
        q, kk, w = gaussian_keys(rng, n, nb, 2, 16, scale=1.0)
        for L in (1, 7, 1023, 4096):
            lens = mx.full((n, 1), L, mx.int32)
            hv, hi = H.hierarchical_topk(q, kk, w, lens, k=k, block=block,
                                         strip=512, overfetch=16)
            mx.eval(hv, hi)
            visible = hi < L
            self.assertTrue(bool(mx.all(visible | ~mx.isfinite(hv)).item()),
                            f"non-visible index with a finite value at lens={L}")
            record({"case": "edge/partial_visibility", "lens": L,
                    "max_index": int(hi.max())})

    def test_nb_not_divisible_and_tiny(self):
        n, h, d = 8, 2, 16
        rng = np.random.default_rng(8)
        for nb2 in (4090, 4093, 1, 7, 3):
            q = mx.array((rng.standard_normal((1, n, h, d)) * 0.3).astype(np.float32))
            kk = mx.array((rng.standard_normal((1, nb2, d)) * 0.3).astype(np.float32))
            kk = kk.astype(mx.bfloat16)
            w = mx.array(np.abs(rng.standard_normal((1, n, h))).astype(np.float32))
            for strip in (512, 11, 8, 4096):
                hv, hi = H.hierarchical_topk(q, kk, w, lens_full(n, nb2), k=16,
                                             block=8, strip=strip, overfetch=4)
                mx.eval(hv, hi)
                self.assertTrue(int(hi.min()) >= 0 and int(hi.max()) < max(nb2, 1),
                                f"nb={nb2} strip={strip}: index out of range")
            record({"case": "edge/nb_uneven", "nb": nb2, "ok": True})

    def test_strip_smaller_than_block(self):
        """strip < block must still process every block at least once."""
        n, nb, block = 8, 512, 8
        rng = np.random.default_rng(9)
        target = (rng.random((n, nb)) * 5).astype(np.float32)
        q, kk, w = one_hot_keys(target)
        rv, ri = ref_one_hot(kk, 200)
        hv, hi = H.hierarchical_topk(q, kk, w, lens_full(n, nb), k=200, block=block,
                                     strip=1, overfetch=16)
        mx.eval(hv, hi)
        rw = mean_rw_recall(np.array(hi)[0], ri[0])
        record({"case": "edge/strip_below_block", "strip": 1, "block": block,
                "rw_recall": round(rw, 6)})
        self.assertGreaterEqual(rw, 0.99, f"strip<block rw recall {rw:.4f}")

    def test_top_blocks_clamps_and_all_neg_inf(self):
        bm = mx.full((2, 3, 4), -np.inf, mx.float32)
        tb = H.top_blocks(bm, 50, block_size=8)               # k > nb_blocks
        mx.eval(tb)
        self.assertEqual(tb.shape, (2, 3, 4))
        tb0 = H.top_blocks(bm, 0, block_size=8)
        mx.eval(tb0)
        self.assertEqual(tb0.shape, (2, 3, 0))
        record({"case": "edge/top_blocks_clamp", "shape_k50": list(tb.shape),
                "shape_k0": list(tb0.shape)})

    def test_coarse_matches_blockmax_of_materialized_row(self):
        """coarse_block_scores == reshape-max of the (bf16-scored) full row."""
        rng = np.random.default_rng(10)
        n, nb, h, d, block, strip = 16, 8192, 4, 64, 8, 512
        q, kk, w = gaussian_keys(rng, n, nb, h, d)
        lens = lens_full(n, nb)
        bm = H.coarse_block_scores(q, kk, w, lens, block=block, strip=strip)
        s = mx.einsum("bshd,btd->bsht", q.astype(mx.bfloat16),
                      kk.astype(mx.bfloat16))
        s = mx.maximum(s, 0.0) * w.astype(mx.bfloat16)[..., None]
        row = mx.sum(s, axis=2)
        ref_bm = row.reshape(1, n, nb // block, block).max(axis=-1)  # [1, n, nb/block]
        mx.eval(bm, ref_bm)
        a = np.array(bm.astype(mx.float32))[0].reshape(-1)
        b = np.array(ref_bm.astype(mx.float32)).reshape(-1)
        rel = np.abs(a - b) / (np.abs(b) + 1e-6)
        record({"case": "coarse_vs_materialized", "max_rel_diff": round(float(rel.max()), 6),
                "mean_rel_diff": round(float(rel.mean()), 6)})
        self.assertLess(float(rel.max()), 0.05, "coarse block maxima diverge from row")


if __name__ == "__main__":
    unittest.main(verbosity=2)
    n = len(RESULTS)
    print(f"DSV41_HIERARCHICAL {json.dumps({'cases': n})}")
