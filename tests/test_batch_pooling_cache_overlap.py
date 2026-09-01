# Copyright © 2026 adurham. Regression guard for the BatchPoolingCache
# overlap-carry batch-width bug.

import unittest

import mlx.core as mx
from mlx_lm.models.cache import BatchPoolingCache


class TestBatchPoolingCacheOverlapCarryResize(unittest.TestCase):
    """Overlap-carry structures must track the decode batch width across
    mid-decode `extend()` (stream join) and `filter()` (stream leave).

    Regression guard for the production crash
    ``ValueError: [reshape] Cannot reshape array of size 1 into shape (2,1,1,1)``
    at cache.py fetch_overlap_carry: extend()/filter() resized the per-stream
    bookkeeping lists (remainder, _pool_lengths, ...) but NOT the four overlap
    structures, leaving _overlap_carry_valid stale at the construction width.
    """

    RATIO = 4
    HALF_DIM = 8
    DTYPE = mx.float16

    def _fresh(self, width):
        return BatchPoolingCache(self.RATIO, [0] * width)

    def _store_stream0_carry(self, cache, value):
        """Force a persisted per-stream overlap carry: stream 0 stores
        `value` in every channel, stream 1 produces no window this call."""
        cache._overlap_windows_this_call = [2, 0]
        b = len(cache.remainder)
        last_kv = mx.full(
            (b, 1, self.RATIO, self.HALF_DIM), value, dtype=self.DTYPE
        )
        last_gate = mx.full(
            (b, 1, self.RATIO, self.HALF_DIM), -20.0, dtype=self.DTYPE
        )
        cache.store_overlap_carry(last_kv, last_gate)  # valid[0] -> True
        return last_kv

    def test_extend_widens_carry_and_preserves_surviving_stream(self):
        cache = self._fresh(2)
        stored = self._store_stream0_carry(cache, 2.0)
        # 2 real streams in flight + 1 new stream joins mid-decode.
        other = self._fresh(1)
        cache.extend(other)

        self.assertEqual(
            len(cache._overlap_carry_valid),
            3,
            "extend must grow _overlap_carry_valid to the new batch width",
        )
        self.assertEqual(
            len(cache._overlap_windows_this_call),
            3,
            "extend must grow _overlap_windows_this_call to the new batch width",
        )
        # Surviving stream 0's carry byte-for-byte preserved in row 0.
        self.assertEqual(cache._overlap_carry_valid, [True, False, False])
        self.assertIsNotNone(cache._overlap_kv_carry)
        self.assertEqual(cache._overlap_kv_carry.shape[0], 3)

        kv_carry, gate_carry = cache.fetch_overlap_carry(
            3, self.RATIO, self.HALF_DIM, self.DTYPE
        )
        # fetch_overlap_carry at the widened width must NOT raise, and row 0
        # must still be the persisted carry (not zeroed).
        self.assertTrue(
            mx.array_equal(kv_carry[0], stored[0].astype(self.DTYPE)),
            "surviving stream 0's overlap carry was corrupted across extend",
        )
        # Streams 1 (in flight, no carry) and 2 (newly joined) get the
        # sequence-start pad.
        self.assertTrue(mx.all(kv_carry[1] == 0).item())
        self.assertTrue(mx.all(kv_carry[2] == 0).item())

    def test_filter_narrows_carry_and_preserves_surviving_stream(self):
        cache = self._fresh(2)
        stored = self._store_stream0_carry(cache, 3.0)
        other = self._fresh(1)
        cache.extend(other)  # now width 3
        # Streams 1 and 2 leave; keep ONLY stream 0 -> resulting batch width
        # (1) differs from the original construction width (2), which is what
        # trips the stale-list reshape on the unfixed code.
        cache.filter([0])

        self.assertEqual(
            len(cache._overlap_carry_valid),
            1,
            "filter must reindex _overlap_carry_valid to the surviving batch",
        )
        self.assertEqual(
            len(cache._overlap_windows_this_call),
            1,
            "filter must reindex _overlap_windows_this_call to the surviving batch",
        )
        self.assertEqual(cache._overlap_carry_valid, [True])
        self.assertEqual(cache._overlap_kv_carry.shape[0], 1)

        kv_carry, _ = cache.fetch_overlap_carry(
            1, self.RATIO, self.HALF_DIM, self.DTYPE
        )
        self.assertTrue(
            mx.array_equal(kv_carry[0], stored[0].astype(self.DTYPE)),
            "surviving stream 0's overlap carry was corrupted across filter",
        )

    def test_filter_respects_surviving_index_order(self):
        """filter keeps the caller's index order: row i == stream[i]."""
        cache = self._fresh(3)
        # Stream 1 stores a carry; streams 0 and 2 produce none.
        cache._overlap_windows_this_call = [0, 2, 0]
        b = len(cache.remainder)
        last_kv = mx.full(
            (b, 1, self.RATIO, self.HALF_DIM), 5.0, dtype=self.DTYPE
        )
        last_gate = mx.full(
            (b, 1, self.RATIO, self.HALF_DIM), -20.0, dtype=self.DTYPE
        )
        cache.store_overlap_carry(last_kv, last_gate)  # valid[1] -> True

        # Reorder: surviving stream set is {1 (carry), 2 (none)}, ordered as
        # [2, 1]. Row 0 must be stream 2 (no carry), row 1 stream 1 (carry).
        cache.filter([2, 1])
        self.assertEqual(cache._overlap_carry_valid, [False, True])

        kv_carry, _ = cache.fetch_overlap_carry(
            2, self.RATIO, self.HALF_DIM, self.DTYPE
        )
        self.assertTrue(mx.all(kv_carry[0] == 0).item())
        self.assertTrue(
            mx.all(kv_carry[1] == 5.0).item(),
            "carry must follow the stream, not its old row position",
        )

    def test_fetch_overlap_carry_no_carry_untouched_by_extend(self):
        """extend on a cache that never stored a carry keeps tensors None and
        fetch_overlap_carry at the widened width returns the placeholder."""
        cache = self._fresh(1)
        self.assertIsNone(cache._overlap_kv_carry)
        other = self._fresh(1)
        cache.extend(other)
        self.assertIsNone(cache._overlap_kv_carry)
        self.assertEqual(len(cache._overlap_carry_valid), 2)
        kv_carry, gate_carry = cache.fetch_overlap_carry(
            2, self.RATIO, self.HALF_DIM, self.DTYPE
        )
        self.assertTrue(mx.all(kv_carry == 0).item())
        self.assertTrue(mx.all(gate_carry == -mx.inf).item())


if __name__ == "__main__":
    unittest.main()
