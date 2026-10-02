# Copyright © 2026 Adam Durham (hermes-gw)
"""Byte-identity of the engram READ under the W5b exposure levers.

``DSV41_ENGRAM_SERIAL`` (order the two tables' prefetches) and
``DSV41_ENGRAM_NOCACHE`` (F_NOCACHE/F_RDAHEAD=0 on macOS, POSIX_FADV_RANDOM
elsewhere) must return exactly the bytes of the default path, for both the
legacy and coarse read shapes, through ``prefetch`` + ``__call__`` (the
production entry points) and through the no-prefetch fallback. A revert
removes the module flags and every test here fails on the contract assert.

Run (mlx-lm worktree root):

    PYTHONPATH=<exo>/src:$PWD <venv>/bin/python -m pytest tests/test_dsv41_engram_read_w5b.py -v
"""

import json
import os
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import exl3_build as eb

ROWS, WW, WS = 50000, 256, 8


class _Shard:
    def __init__(self, path, header, base):
        self.header, self.base = header, base
        self._fd = os.open(path, os.O_RDONLY)

    def _open(self):
        return self._fd


class _Ck:
    def __init__(self, sh):
        self.sh = sh

    def header(self, name):
        return self.sh.header[name]

    def _shard(self, name):
        return self.sh


def _shard(tmp, layer, seed):
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 0x7E, (ROWS, WW), dtype=np.uint8)
    s = rng.integers(120, 135, (ROWS, WS), dtype=np.uint8)
    wn, sn = f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"
    pw, ps = w.tobytes(), s.tobytes()
    header = {wn: {"dtype": "U8", "shape": [ROWS, WW], "data_offsets": [0, len(pw)]},
              sn: {"dtype": "U8", "shape": [ROWS, WS], "data_offsets": [len(pw), len(pw) + len(ps)]}}
    hb = json.dumps(header).encode()
    path = os.path.join(tmp, f"l{layer}.safetensors")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(pw)
        f.write(ps)
    return _Ck(_Shard(path, header, 8 + len(hb))), w, s


class EngramReadW5bTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.mkdtemp(prefix="dsv41_engram_w5b_")
        ck1, cls.W1, cls.S1 = _shard(tmp, 1, 1)
        ck14, cls.W14, cls.S14 = _shard(tmp, 14, 2)
        cls.t1 = eb.LazyEngramTable(ck1, 1)
        cls.t14 = eb.LazyEngramTable(ck14, 14)

    def _forward_pair(self, read, serial, nocache, prefetch=True):
        """Both tables, production order: prefetch(1), prefetch(14), call(1), call(14)."""
        self.assertTrue(hasattr(eb, "_ENGRAM_SERIAL") and hasattr(eb, "_ENGRAM_NOCACHE"),
                        "W5b engram flags missing -- reverted?")
        rng = np.random.default_rng(99)
        i1 = rng.integers(0, ROWS, (1, 2048, 24)).astype(np.int64)
        i14 = rng.integers(0, ROWS, (1, 2048, 24)).astype(np.int64)
        reads = []
        real = eb.LazyEngramTable._read

        def spy(this, uniq):
            w, s = real(this, uniq)
            reads.append((this._w, w.tobytes(), s.tobytes(), w.flags.writeable))
            return w, s

        with mock.patch.object(eb, "_ENGRAM_READ", read), \
             mock.patch.object(eb, "_ENGRAM_SERIAL", serial), \
             mock.patch.object(eb, "_ENGRAM_NOCACHE", nocache), \
             mock.patch.object(eb.LazyEngramTable, "_read", spy):
            eb._ENGRAM_LAST[0] = None
            if prefetch:
                self.t1.prefetch(i1)
                self.t14.prefetch(i14)
            o1 = self.t1(i1)
            o14 = self.t14(i14)
            mx.eval(o1, o14)
        return np.array(o1), np.array(o14), sorted(reads)

    def test_flags_default_off(self):
        self.assertEqual(eb._ENGRAM_SERIAL, os.environ.get("DSV41_ENGRAM_SERIAL", "0") == "1")
        self.assertEqual(eb._ENGRAM_NOCACHE, os.environ.get("DSV41_ENGRAM_NOCACHE", "0") == "1")

    def test_identity_all_arms(self):
        for read in ("legacy", "coarse"):
            ref = self._forward_pair(read, False, False)
            for serial, nocache in ((True, False), (False, True), (True, True)):
                got = self._forward_pair(read, serial, nocache)
                for a, b, name in ((ref[0], got[0], "layer1"), (ref[1], got[1], "layer14")):
                    self.assertTrue(np.array_equal(a.view(np.uint8), b.view(np.uint8)),
                                    f"{read} serial={serial} nocache={nocache}: {name} differs")
                self.assertEqual(ref[2], got[2],
                                 f"{read} serial={serial} nocache={nocache}: read bytes differ")
        # legacy == coarse across the arms too (the coarse-arm property)
        self.assertEqual(self._forward_pair("legacy", True, True)[2],
                         self._forward_pair("coarse", True, True)[2])

    def test_identity_no_prefetch(self):
        ref = self._forward_pair("coarse", False, False, prefetch=False)
        got = self._forward_pair("coarse", True, True, prefetch=False)
        self.assertTrue(np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1]))

    def test_serial_orders_reads(self):
        """Layer 14's read must not start before layer 1's read has finished."""
        log = []
        real = eb.LazyEngramTable._read

        def spy(this, uniq):
            log.append(("start", this._w))
            out = real(this, uniq)
            log.append(("end", this._w))
            return out

        i = np.random.default_rng(5).integers(0, ROWS, (1, 1024, 24)).astype(np.int64)
        with mock.patch.object(eb, "_ENGRAM_READ", "coarse"), \
             mock.patch.object(eb, "_ENGRAM_SERIAL", True), \
             mock.patch.object(eb.LazyEngramTable, "_read", spy):
            eb._ENGRAM_LAST[0] = None
            self.t1.prefetch(i)
            self.t14.prefetch(i)
            mx.eval(self.t1(i), self.t14(i))
        w1, w14 = self.t1._w, self.t14._w
        self.assertEqual(log, [("start", w1), ("end", w1), ("start", w14), ("end", w14)])

    def test_nocache_applies(self):
        fd = self.t1._loc(self.t1._w)[0]
        eb._NOCACHE_DONE.discard(fd)
        what = eb._engram_nocache(fd)
        self.assertIn(what, ("F_NOCACHE+F_RDAHEAD=0", "POSIX_FADV_RANDOM", "none"))
        self.assertEqual(eb._engram_nocache(fd), "done")


if __name__ == "__main__":
    unittest.main()
