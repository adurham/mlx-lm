# Copyright © 2026 Adam Durham (hermes-gw)
"""Contract for ``DSV41_ENGRAM_TILE_ROWS`` (W5b): bound the Engram prefill
transient WITHOUT changing a single output byte or the table read.

The multi-row Engram step (``engram.py`` ``Engram.__call__``) builds ~8
full-size fp32 copies of the hc stream ``[B, L, hc, dim]`` (h, square(h), key,
square(key), h*w, h*w*key, gate*value, h+...) in ONE lazy graph; at the served
shape (L=4096, hc=4, dim=5120) each is 335 MB. The flag evaluates that chain in
fenced row tiles. This test pins:

1. **byte-identity**: the tiled forward returns exactly the untiled bytes
   (fp32 and bf16 streams, odd row counts, tail merge, with and without the
   projection also tiled), through the REAL ``LazyEngramTable`` on a synthetic
   shard -- so the read path is exercised end to end;
2. **read unchanged**: the table is read ONCE per call with the same unique
   ids in both modes (the prefetch still hits; ``_read`` bytes untouched);
3. **bound**: on the MLX allocator, the tiled peak is well under the untiled
   peak at a multi-row shape. On a revert (no tiling code) the peak assertion
   and the ``row_tiles``/flag contract both fail;
4. **tile shape**: ``row_tiles`` covers [0, n) exactly, never emits a tile
   below ``_TILE_MIN_ROWS`` (so the projection and the last-axis reductions
   keep their multi-row kernels), and is a no-op when the flag is 0.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_engram_tile.py -v
"""

import json
import os
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.deepseek_v41 import engram as eg
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41.config import ModelArgs

TABLE_ROWS = 20000
HD = 32           # engram_head_dim (real: 256); 32 keeps one fp8 scale block per row
HEADS, NGRAM = 8, 4
COLS = (NGRAM - 1) * HEADS   # n_hash_cols = 24, as served
W_NAME = "layers.1.engram.embed.weight"
S_NAME = "layers.1.engram.embed.scale"


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


def _make_shard(tmp):
    rng = np.random.default_rng(7)
    w = rng.integers(0, 0x7E, (TABLE_ROWS, HD), dtype=np.uint8)
    w |= rng.integers(0, 2, (TABLE_ROWS, HD), dtype=np.uint8) << 7  # sign, no NaN codes
    s = rng.integers(120, 134, (TABLE_ROWS, HD // 32), dtype=np.uint8)
    pw, ps = w.tobytes(), s.tobytes()
    header = {W_NAME: {"dtype": "U8", "shape": [TABLE_ROWS, HD], "data_offsets": [0, len(pw)]},
              S_NAME: {"dtype": "U8", "shape": [TABLE_ROWS, HD // 32],
                       "data_offsets": [len(pw), len(pw) + len(ps)]}}
    hb = json.dumps(header).encode()
    path = os.path.join(tmp, "engram.safetensors")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(pw)
        f.write(ps)
    return _Ck(_Shard(path, header, 8 + len(hb)))


def _module(ck, dim, hc, seed=0):
    args = ModelArgs(dim=dim, hc_mult=hc, engram_layer_ids=(1, 14),
                     engram_max_ngram_size=NGRAM, engram_n_heads=HEADS,
                     engram_head_dim=HD, engram_num_embeddings=(TABLE_ROWS, TABLE_ROWS),
                     norm_eps=1e-20)
    m = eg.Engram(args, 0)
    m.embed = eb.LazyEngramTable(ck, 1)
    k = mx.random.key(seed)
    k1, k2, k3 = mx.random.split(k, 3)
    lin = nn.Linear(COLS * HD, dim * (hc + 1), bias=False)
    lin.weight = (mx.random.normal(lin.weight.shape, key=k1) * 0.05).astype(mx.float16)
    m.wkv = lin
    m.q_weight = mx.random.uniform(0.5, 1.5, (hc, dim), key=k2)
    m.k_weight = mx.random.uniform(0.5, 1.5, (hc, dim), key=k3)
    mx.eval(m.parameters())
    return m


class _Flags:
    """Flip the module-level flags (read at import) for one call."""

    def __init__(self, rows, proj=False):
        self.p = [mock.patch.object(eg, "_TILE_ROWS", rows),
                  mock.patch.object(eg, "_TILE_PROJ", proj)]

    def __enter__(self):
        for p in self.p:
            p.start()

    def __exit__(self, *a):
        for p in self.p:
            p.stop()


def _run(m, x, idx, rows, proj=False, prefetch=True):
    with _Flags(rows, proj):
        if prefetch:
            m.embed.prefetch(idx)
        out = m(x, mx.array(idx))
        mx.eval(out)
    return np.array(out.astype(mx.float32)) if out.dtype == mx.bfloat16 else np.array(out)


class EngramTileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dsv41_engram_tile_")
        cls.ck = _make_shard(cls.tmp)

    # ---- 4. tile shape -------------------------------------------------
    def test_flag_contract(self):
        self.assertTrue(hasattr(eg, "_TILE_ROWS"), "DSV41_ENGRAM_TILE_ROWS missing -- reverted?")
        self.assertEqual(eg._TILE_ROWS, int(os.environ.get("DSV41_ENGRAM_TILE_ROWS", "0")))
        self.assertGreaterEqual(eg._TILE_MIN_ROWS, 65)   # > EXL3_FUSED_ROW_LIMIT, > 32

    def test_row_tiles(self):
        rt, mn = eg.row_tiles, eg._TILE_MIN_ROWS
        self.assertEqual(rt(4096, 0), [(0, 4096)])
        self.assertEqual(rt(1, 512), [(0, 1)])          # decode: never tiled
        self.assertEqual(rt(512, 512), [(0, 512)])
        self.assertEqual(rt(4096, 512), [(i, i + 512) for i in range(0, 4096, 512)])
        for n in (129, 300, 1000, 2047, 2048, 4095, 4096, 4097, 16384):
            for t in (1, 64, 100, 128, 256, 512, 1000):
                tiles = rt(n, t)
                self.assertEqual(tiles[0][0], 0)
                self.assertEqual(tiles[-1][1], n)
                for (a, b), (c, _) in zip(tiles, tiles[1:]):
                    self.assertEqual(b, c)
                if len(tiles) > 1:
                    self.assertTrue(all(b - a >= mn for a, b in tiles), (n, t, tiles))

    # ---- 1 + 2. byte identity and an unchanged read -----------------------
    def _identity(self, n, dim, hc, dtype, rows, proj):
        m = _module(self.ck, dim, hc, seed=n)
        rng = np.random.default_rng(n)
        idx = rng.integers(0, TABLE_ROWS, (1, n, COLS)).astype(np.int64)
        x = mx.random.normal((1, n, hc, dim), key=mx.random.key(n + 1)).astype(dtype)
        mx.eval(x)
        calls = []
        real = eb.LazyEngramTable._read

        def spy(this, uniq):
            calls.append(np.asarray(uniq).copy())
            return real(this, uniq)

        with mock.patch.object(eb.LazyEngramTable, "_read", spy):
            ref = _run(m, x, idx, 0)
            got = _run(m, x, idx, rows, proj)
        self.assertGreater(len(eg.row_tiles(n, rows)), 1, "shape did not tile")
        self.assertEqual(ref.dtype, got.dtype)
        self.assertEqual(ref.shape, got.shape)
        self.assertTrue(np.array_equal(ref.view(np.uint8), got.view(np.uint8)),
                        f"n={n} rows={rows} proj={proj}: tiled bytes != untiled "
                        f"(max|d|={np.abs(ref - got).max()})")
        # one read per call, same unique ids, in both modes
        self.assertEqual(len(calls), 2)
        self.assertTrue(np.array_equal(calls[0], calls[1]))

    def test_identity_bf16_stream(self):
        for n, rows in ((1000, 128), (1000, 300), (777, 256)):
            self._identity(n, 256, 4, mx.bfloat16, rows, proj=False)

    def test_identity_fp32_stream(self):
        self._identity(600, 256, 4, mx.float32, 128, proj=False)

    def test_identity_tiled_projection(self):
        self._identity(1000, 256, 4, mx.bfloat16, 256, proj=True)

    def test_no_prefetch_path(self):
        m = _module(self.ck, 128, 4)
        idx = np.random.default_rng(3).integers(0, TABLE_ROWS, (1, 400, COLS)).astype(np.int64)
        x = mx.random.normal((1, 400, 4, 128), key=mx.random.key(9)).astype(mx.bfloat16)
        a = _run(m, x, idx, 0, prefetch=False)
        b = _run(m, x, idx, 128, prefetch=False)
        self.assertTrue(np.array_equal(a.view(np.uint8), b.view(np.uint8)))

    def test_single_row_untouched(self):
        m = _module(self.ck, 128, 4)
        idx = np.random.default_rng(4).integers(0, TABLE_ROWS, (1, 1, COLS)).astype(np.int64)
        x = mx.random.normal((1, 1, 4, 128), key=mx.random.key(5)).astype(mx.bfloat16)
        a = _run(m, x, idx, 0)
        b = _run(m, x, idx, 128)
        self.assertTrue(np.array_equal(a.view(np.uint8), b.view(np.uint8)))

    # ---- 3. the bound ------------------------------------------------------
    def _peak(self, m, x, idx, rows):
        with _Flags(rows):
            m.embed.prefetch(idx)
            m.embed._pending[2].result()        # I/O done: measure compute only
            mx.clear_cache()
            mx.reset_peak_memory()
            base = mx.get_active_memory()
            out = m(x, mx.array(idx))
            mx.eval(out)
            return mx.get_peak_memory() - base

    def test_peak_bound(self):
        n, dim, hc = 2048, 1024, 4
        m = _module(self.ck, dim, hc)
        idx = np.random.default_rng(1).integers(0, TABLE_ROWS, (1, n, COLS)).astype(np.int64)
        x = mx.random.normal((1, n, hc, dim), key=mx.random.key(2)).astype(mx.bfloat16)
        mx.eval(x)
        s_full = n * hc * dim * 4                    # one fp32 hc-stream copy
        base = self._peak(m, x, idx, 0)
        tiled = self._peak(m, x, idx, 256)
        print(f"\n[engram tile bound] n={n} dim={dim} S={s_full/1e6:.1f} MB "
              f"untiled peak={base/1e6:.1f} MB ({base/s_full:.2f} S) "
              f"tiled(256) peak={tiled/1e6:.1f} MB ({tiled/s_full:.2f} S)")
        self.assertGreater(base, 3 * s_full, "untiled transient smaller than modelled")
        self.assertLess(tiled, 0.6 * base, "tiling no longer bounds the transient")


if __name__ == "__main__":
    unittest.main()
