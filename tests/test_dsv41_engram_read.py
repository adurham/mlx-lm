# Copyright © 2026 Adam Durham (hermes-gw)
"""Byte-identity + task-shape contract for ``LazyEngramTable._read``.

The Stage-B candidate replaces ``_read``'s per-row pool fan-out (one task per
unique row for the weights AND one for the scales -- 2*U tiny submit/result
round trips per lookup, 80 816 tasks for a 2048-row chunk at U = 40 408) with
coarse contiguous-index slices, each read by ONE task looping its rows. This
test pins both halves of the contract on a synthetic safetensors shard:

1. **byte-identity**: the coarse path returns EXACTLY the arrays the legacy
   path returns (same bytes, same shape, same dtype, same writeable flag) for
   random, duplicate, boundary (0, R-1) and 40 408-row inputs -- and both equal
   the shard's own rows;
2. **task shape**: a 40 408-row read submits <= 256 tasks under "coarse" (it is
   128 with the default 64 slices) versus the legacy 80 816. The count
   assertion fails on a revert: with the original code the "coarse" arm IS the
   per-row fan-out, and the module-level ``DSV41_ENGRAM_READ`` selector this
   test asserts does not exist at all.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_engram_read.py -v
"""

import json
import os
import struct
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np

from mlx_lm.models.deepseek_v41 import exl3_build as eb

ROWS = 60000
WW, WS = 256, 8
W_NAME = "layers.1.engram.embed.weight"
S_NAME = "layers.1.engram.embed.scale"
N_40K = 40408  # the same-layout chunk scale of the Stage-B A/B


class _ShardStub:
    """Minimal stand-in for exl3.loader._Shard: header + base + fd."""

    def __init__(self, path, header, base):
        self.path = path
        self.header = header
        self.base = base
        self._fd = os.open(path, os.O_RDONLY)

    def _open(self):
        return self._fd


class _CkStub:
    """Minimal stand-in for Exl3Checkpoint: header() and _shard()."""

    def __init__(self, shard):
        self._shard_obj = shard

    def header(self, name):
        return self._shard_obj.header[name]

    def _shard(self, name):
        return self._shard_obj


class _CountingPool(ThreadPoolExecutor):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.submitted = 0

    def submit(self, fn, *a, **k):
        self.submitted += 1
        return super().submit(fn, *a, **k)


class EngramReadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dsv41_engram_read_")
        cls.path = os.path.join(cls.tmp, "native_engram_layer1.safetensors")
        rng = np.random.default_rng(11)
        cls.W = rng.integers(0, 255, (ROWS, WW), np.uint8)
        cls.S = rng.integers(120, 135, (ROWS, WS), np.uint8)
        payload_w, payload_s = cls.W.tobytes(), cls.S.tobytes()
        header = {
            W_NAME: {"dtype": "U8", "shape": [ROWS, WW],
                     "data_offsets": [0, len(payload_w)]},
            S_NAME: {"dtype": "U8", "shape": [ROWS, WS],
                     "data_offsets": [len(payload_w), len(payload_w) + len(payload_s)]},
        }
        hdr = json.dumps(header).encode()
        with open(cls.path, "wb") as f:
            f.write(struct.pack("<Q", len(hdr)))
            f.write(hdr)
            f.write(payload_w)
            f.write(payload_s)
        base = 8 + len(hdr)
        cls.tab = eb.LazyEngramTable(_CkStub(_ShardStub(cls.path, header, base)), 1)

    def _read_with(self, mode, uniq):
        """Run the REAL LazyEngramTable._read in ``mode`` with a counting pool."""
        self.assertTrue(
            hasattr(eb, "_ENGRAM_READ"),
            "DSV41_ENGRAM_READ selector missing -- patch reverted?")
        pool = _CountingPool(max_workers=32)
        prev = eb._ENGRAM_READ
        eb._ENGRAM_READ = mode
        try:
            with mock.patch.object(eb, "_engram_pool", lambda: pool):
                w, sc = self.tab._read(np.asarray(uniq, dtype=np.int64))
        finally:
            eb._ENGRAM_READ = prev
            pool.shutdown()
        return w, sc, pool.submitted

    def _assert_identical(self, label, uniq):
        u = np.asarray(uniq, dtype=np.int64)
        a_w, a_s, _ = self._read_with("legacy", u)
        b_w, b_s, n_tasks = self._read_with("coarse", u)
        for name, a, b in (("weight", a_w, b_w), ("scale", a_s, b_s)):
            self.assertEqual(a.shape, b.shape, f"{label}: {name} shape")
            self.assertEqual(a.dtype, b.dtype, f"{label}: {name} dtype")
            self.assertEqual(a.flags.writeable, b.flags.writeable,
                             f"{label}: {name} writeable flag")
            self.assertTrue(np.array_equal(a, b),
                            f"{label}: {name} legacy != coarse")
        self.assertTrue(np.array_equal(a_w, self.W[u]), f"{label}: weight != W[u]")
        self.assertTrue(np.array_equal(a_s, self.S[u]), f"{label}: scale != S[u]")
        return n_tasks

    def test_module_flag_contract(self):
        self.assertEqual(eb._ENGRAM_READ,
                         os.environ.get("DSV41_ENGRAM_READ", "coarse"))
        self.assertIn(eb._ENGRAM_READ, ("legacy", "coarse"))

    def test_byte_identity_empty_and_single(self):
        self.assertEqual(self._assert_identical("empty", []), 0)
        n = self._assert_identical("single", [ROWS - 1])
        self.assertEqual(n, 2)

    def test_byte_identity_boundary_duplicates(self):
        self._assert_identical("boundary+dupes", [0, ROWS - 1, 0, 1, 1, ROWS - 1])

    def test_byte_identity_random_with_duplicates(self):
        rng = np.random.default_rng(5)
        u = rng.choice(ROWS, 251, replace=True).astype(np.int64)
        u = np.concatenate([u, [0, ROWS - 1]])
        self._assert_identical("random+dupes", u)

    def test_byte_identity_40k_and_task_shape(self):
        rng = np.random.default_rng(33)
        u = np.unique(rng.choice(ROWS, N_40K, replace=False)).astype(np.int64)
        self.assertEqual(len(u), N_40K)
        n_coarse = self._assert_identical("40k", u)
        # root-cause contract: coarse submits 2*min(slices, n) tasks (128 with
        # the default of 64 slices) -- NOT the 2*n per-row fan-out. On a revert
        # this arm runs the per-row code and submits 80 816.
        self.assertLessEqual(n_coarse, 256,
                             f"coarse read submitted {n_coarse} tasks")
        _, _, n_legacy = self._read_with("legacy", u)
        self.assertEqual(n_legacy, 2 * N_40K,
                         "legacy arm no longer submits one task per row per tensor")
        self.assertGreater(n_legacy, 100 * n_coarse)


if __name__ == "__main__":
    unittest.main()
