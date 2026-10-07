# Copyright © 2026 Adam Durham (hermes-gw)
"""Consumer-layer coarse-pass restriction — BIT-EXACTNESS against the old path.

Consumer index layers (24/28/32/36: ``uses_candidates``, mask from layer 20)
used to run the coarse pass over all ``nb`` columns and mask non-candidates to
``-inf`` *after* scoring. :func:`coarse_block_scores_candidates` scores only the
blocks holding a candidate column. These tests pin that the change is a pure
work reduction: block maxima AND the final ``(top_v, top_i)`` are
``np.array_equal`` to the full-width path (``consumer_skip=False``), including:

* zero-candidate rows (all-``-inf`` maxima; top_blocks still picks k+of blocks);
* candidate blocks partially past the visible window (``lens`` < nb, causal);
* ``nb`` not a multiple of ``block`` (partial tail block);
* non-block-aligned masks (the function is exact for ANY mask);
* the production geometry (h=32, d=128, gather strip 1360);
* the real ``Indexer.__call__`` path, and the consumer block-size assert.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_consumer_skip.py -q
"""

from __future__ import annotations

import unittest

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as IX
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as H


def _inputs(seed, *, b=1, n=64, nb=2048, h=8, d=64):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((b, n, h, d)).astype(np.float32))
    k = mx.array(rng.standard_normal((b, nb, d)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.standard_normal((b, n, h)).astype(np.float32))
    return q, k, w


def _block_mask(rng, b, n, nb, block, n_keep, zero_rows=()):
    """Block-constant [b, n, nb] mask keeping n_keep random blocks per row."""
    NB = -(-nb // block)
    keep = np.zeros((b, n, NB), bool)
    for bi in range(b):
        for r in range(n):
            keep[bi, r, rng.choice(NB, size=min(n_keep, NB), replace=False)] = True
    for r in zero_rows:
        keep[:, r] = False
    return mx.array(np.repeat(keep, block, axis=-1)[..., :nb])


def _both(q, k, w, lens, cmask, *, kk=64, block=8, cstrip=512, estrip=80, of=16):
    out = {}
    for skip in (False, True):
        v, i, m = H.hierarchical_topk_prod(q, k, w, lens, kk, block, cstrip, estrip, of,
                                           cand_mask=cmask, consumer_skip=skip)
        mx.eval(v, i)
        assert m is None
        out[skip] = (np.array(v), np.array(i))
    return out[False], out[True]


class ConsumerSkipBitExact(unittest.TestCase):
    def assertSame(self, old, new, tag):
        (ov, oi), (nv, ni) = old, new
        self.assertTrue(np.array_equal(ov, nv), f"{tag}: top_v differs "
                        f"({int((ov != nv).sum())} slots)")
        self.assertTrue(np.array_equal(oi, ni), f"{tag}: top_i differs "
                        f"({int((oi != ni).sum())} slots)")

    def _maxima_same(self, q, k, w, lens, cmask, block=8, cstrip=512):
        old = H.coarse_block_scores(q, k, w, lens, block=block, strip=cstrip,
                                    col_mask=cmask)
        new = H.coarse_block_scores_candidates(
            q, k, w, lens, cmask, block=block,
            strip=H.coarse_gather_strip(cstrip, q.shape[2], q.shape[3], block))
        mx.eval(old, new)
        o, nw = np.array(old), np.array(new)
        self.assertEqual(o.shape, nw.shape)
        self.assertTrue(np.array_equal(o, nw),
                        f"block maxima differ in {int((o != nw).sum())} cells")
        return o

    # -- the mandated probe: b=1, n=64, nb=2048, d=64, h=8, lens full --------
    def test_probe_full_lens_subset_blocks(self):
        b, n, nb, h, d = 1, 64, 2048, 8, 64
        q, k, w = _inputs(0, b=b, n=n, nb=nb, h=h, d=d)
        lens = mx.full((n, 1), nb, mx.int32)
        cmask = _block_mask(np.random.default_rng(1), b, n, nb, 8, 40)
        bm = self._maxima_same(q, k, w, lens, cmask)
        self.assertTrue(np.isfinite(bm).sum(-1).max() == 40)   # restriction is real
        self.assertSame(*_both(q, k, w, lens, cmask), "probe")

    def test_zero_candidate_rows(self):
        b, n, nb = 1, 64, 2048
        q, k, w = _inputs(2, n=n, nb=nb)
        lens = mx.full((n, 1), nb, mx.int32)
        cmask = _block_mask(np.random.default_rng(3), b, n, nb, 8, 30,
                            zero_rows=(0, 7, 63))
        bm = self._maxima_same(q, k, w, lens, cmask)
        self.assertTrue(np.all(np.isneginf(bm[0, [0, 7, 63]])))
        self.assertSame(*_both(q, k, w, lens, cmask), "zero-rows")

    def test_all_rows_zero_candidates(self):
        n, nb = 16, 512
        q, k, w = _inputs(4, n=n, nb=nb)
        lens = mx.full((n, 1), nb, mx.int32)
        cmask = mx.zeros((1, n, nb), mx.bool_)
        self._maxima_same(q, k, w, lens, cmask)
        self.assertSame(*_both(q, k, w, lens, cmask), "all-zero")

    def test_causal_lens_partial_visibility(self):
        # rows see 37..(37+n-1)*... columns: candidate blocks straddle lens
        n, nb = 64, 2048
        q, k, w = _inputs(5, n=n, nb=nb)
        lens = mx.array((np.arange(n) * 29 + 37).clip(max=nb)[:, None].astype(np.int32))
        cmask = _block_mask(np.random.default_rng(6), 1, n, nb, 8, 64)
        self._maxima_same(q, k, w, lens, cmask)
        self.assertSame(*_both(q, k, w, lens, cmask), "causal-lens")

    def test_tail_block_partial(self):
        n, nb = 32, 2045                     # 2045 = 255*8 + 5
        q, k, w = _inputs(7, n=n, nb=nb)
        lens = mx.full((n, 1), nb, mx.int32)
        cmask = _block_mask(np.random.default_rng(8), 1, n, nb, 8, 50)
        cmask[:, :, -5:] = True             # the partial tail block is a candidate
        self._maxima_same(q, k, w, lens, cmask)
        self.assertSame(*_both(q, k, w, lens, cmask), "tail-block")

    def test_unaligned_mask_still_exact(self):
        # not the production shape, but the function claims exactness for ANY mask
        n, nb = 32, 1000
        q, k, w = _inputs(9, n=n, nb=nb)
        lens = mx.array(np.linspace(100, nb, n).astype(np.int32)[:, None])
        rng = np.random.default_rng(10)
        cmask = mx.array(rng.random((1, n, nb)) < 0.03)
        self._maxima_same(q, k, w, lens, cmask)
        self.assertSame(*_both(q, k, w, lens, cmask), "unaligned")

    def test_batch2(self):
        b, n, nb = 2, 16, 1024
        q, k, w = _inputs(11, b=b, n=n, nb=nb)
        lens = mx.full((n, 1), nb, mx.int32)
        cmask = _block_mask(np.random.default_rng(12), b, n, nb, 8, 20, zero_rows=(3,))
        self._maxima_same(q, k, w, lens, cmask)
        self.assertSame(*_both(q, k, w, lens, cmask), "batch2")

    def test_production_geometry(self):
        # h=32, d=128, coarse strip 4096 -> gather strip 1360 (multi-strip),
        # candidate count beyond one strip, k=512, overfetch 16.
        n, nb, h, d = 64, 16384, 32, 128
        q, k, w = _inputs(13, n=n, nb=nb, h=h, d=d)
        lens = mx.full((n, 1), nb, mx.int32)
        cmask = _block_mask(np.random.default_rng(14), 1, n, nb, 8, 600)
        self.assertEqual(H.coarse_gather_strip(4096, h, d, 8), 1360)
        self._maxima_same(q, k, w, lens, cmask, cstrip=4096)
        self.assertSame(*_both(q, k, w, lens, cmask, kk=512, cstrip=4096, estrip=88),
                        "production-geometry")

    def test_sabotage_negative_control(self):
        # Dropping the mask from the restricted scoring must be DETECTED.
        n, nb = 64, 2048
        q, k, w = _inputs(15, n=n, nb=nb)
        lens = mx.full((n, 1), nb, mx.int32)
        cmask = _block_mask(np.random.default_rng(16), 1, n, nb, 8, 40)
        old = H.coarse_block_scores(q, k, w, lens, block=8, strip=512, col_mask=cmask)
        allm = mx.ones((1, n, nb), mx.bool_)
        bad = H.coarse_block_scores_candidates(q, k, w, lens, allm, block=8, strip=512)
        mx.eval(old, bad)
        self.assertFalse(np.array_equal(np.array(old), np.array(bad)))


class ConsumerSkipThroughIndexer(unittest.TestCase):
    """Real Indexer.__call__: consumer layer ON-skip == ON-full-width, bitwise."""

    def setUp(self):
        self._saved = (IX._HIER, IX._HIER_BLOCK, IX._HIER_STRIP,
                       IX._HIER_EXACT_STRIP, IX._HIER_CONSUMER_SKIP,
                       IX._HIER_OVERFETCH)

    def tearDown(self):
        (IX._HIER, IX._HIER_BLOCK, IX._HIER_STRIP,
         IX._HIER_EXACT_STRIP, IX._HIER_CONSUMER_SKIP,
         IX._HIER_OVERFETCH) = self._saved

    def _setup(self, block=8):
        from tests.test_dsv41_hier_integration import _call_inputs, _hier_args, _indexer
        from mlx_lm.models.deepseek_v41.model import SharedState
        IX._HIER, IX._HIER_BLOCK, IX._HIER_STRIP, IX._HIER_EXACT_STRIP = True, 8, 1024, 80
        # overfetch 0 + nb=480 (60 blocks) + k=6: top_blocks keeps only 6 of 60
        # blocks, so a wrong coarse ranking WOULD change the output (with the
        # stock 15-block fixture k+of >= NB and every block is kept anyway).
        IX._HIER_OVERFETCH = 0
        args = _hier_args(candidate_source_layer=0, candidate_block_size=block,
                          candidate_topk_blocks=4)
        src = _indexer(args, 0)
        # _indexer() reseeds 0, which would give the consumer the SOURCE's
        # weights: its top coarse blocks would then BE the candidates, and a
        # broken consumer coarse pass could hide behind the exact-pass mask.
        mx.random.seed(1)
        con = IX.Indexer(args, 1)
        mx.eval(con.parameters())
        x, qr, sp, fv, ik = _call_inputs(np.random.default_rng(21), args, nb=480)
        return SharedState, src, con, (x, qr, sp, 0, fv, ik)

    def test_consumer_call_bit_exact(self):
        SharedState, src, con, a = self._setup()
        self.assertTrue(con.uses_candidates)
        sh = SharedState()
        mx.eval(src(*a, sh), sh.candidates)
        kept = np.array(sh.candidates).sum(-1)
        self.assertTrue(0 < kept.max() < 480, "fixture: mask must restrict")
        outs = {}
        for skip in (False, True):
            IX._HIER_CONSUMER_SKIP = skip
            s2 = SharedState()
            s2.candidates = sh.candidates
            o = con(*a, s2)
            mx.eval(o)
            outs[skip] = np.array(o)
        self.assertGreater(int((outs[False] >= 0).sum()), 0)
        self.assertTrue(np.array_equal(outs[False], outs[True]))

    def test_consumer_block_size_mismatch_asserts(self):
        SharedState, src, con, a = self._setup(block=8)
        sh = SharedState()
        mx.eval(src(*a, sh), sh.candidates)
        con.candidate_block_size = 16        # consumer grid != coarse grid
        s2 = SharedState()
        s2.candidates = sh.candidates
        with self.assertRaises(AssertionError):
            con(*a, s2)


if __name__ == "__main__":
    unittest.main()
