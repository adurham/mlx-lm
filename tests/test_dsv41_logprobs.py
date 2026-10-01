"""Sharded top-k log-probs are exact: two vocab halves combined == full row."""
import unittest

import mlx.core as mx

from mlx_lm.models.deepseek_v41 import logprobs as lp


def _two_rank(y, k):
    half = y.shape[-1] // 2
    bufs = [lp.local_buffer(y[:, :half], k, 0), lp.local_buffer(y[:, half:], k, half)]
    return lp.combine(mx.stack(bufs), k)


class TestShardedLogprobs(unittest.TestCase):
    def test_matches_full_row(self):
        mx.random.seed(0)
        y = mx.random.normal((7, 1000)) * 4
        ids, sel, tid, tlp = _two_rank(y, 5)
        full = y - mx.logsumexp(y, axis=-1, keepdims=True)
        self.assertTrue(mx.array_equal(ids, mx.argmax(y, axis=-1).astype(mx.int32)))
        ref_sel = mx.take_along_axis(full, ids[:, None].astype(mx.int32), axis=-1)[:, 0]
        self.assertTrue(mx.allclose(sel, ref_sel, atol=1e-5))
        ref_ids = mx.argsort(-y, axis=-1)[:, :5].astype(mx.int32)
        self.assertTrue(mx.array_equal(tid, ref_ids))
        ref_lp = mx.take_along_axis(full, ref_ids, axis=-1)
        self.assertTrue(mx.allclose(tlp, ref_lp, atol=1e-5))

    def test_tie_takes_lowest_id_like_argmax(self):
        y = mx.zeros((1, 10))
        y[0, 3] = 2.0
        y[0, 8] = 2.0
        ids, _, _, _ = _two_rank(y, 3)
        self.assertEqual(int(ids[0]), 3)

    def test_single_row_helper(self):
        y = mx.array([[0.0, 1.0, 3.0, 2.0]])
        ids, sel, tid, tlp = lp.from_logits(y, 2)
        self.assertEqual(int(ids[0]), 2)
        self.assertEqual(tid.tolist(), [[2, 3]])
        self.assertAlmostEqual(float(sel[0]), float(tlp[0, 0]), places=6)


if __name__ == "__main__":
    unittest.main()
