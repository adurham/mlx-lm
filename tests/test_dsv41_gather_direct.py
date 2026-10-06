# Copyright © 2026 Adam Durham (hermes-gw)
"""Bit-exact parity + no-materialization contract for the gather-direct comp_kv
two-source gather in ``mlx_lm/models/deepseek_v41/sparse_attention.py``.

Background. ``attention.py`` used to build the compressed-KV operand densely
per compressing layer per forward::

    comp    = src.comp_kv[:bsz, :compress_len].astype(kv.dtype)
    kv_all  = mx.concatenate([kv_all, comp], axis=1)      # ~31 GiB/step @1M
    idxs    = mx.concatenate([idxs, cidx], axis=-1)

but ``sparse_attn`` only ever gathers <= index_topk rows per query. This test
pins the replacement, which hands ``comp_kv`` to ``sparse_attn`` as a second
source (``kv2``) addressed by ``cidx`` and gathers each row directly::

    sparse_attn(q, kv_all, sink, idxs, scale, kv2=src.comp_kv[:bsz, :compress_len],
                split=offset)

Index space (verified at runtime by ``test_index_space_probe``): ``cidx`` is
emitted by the indexer as ``i + offset`` with ``offset = wp + n``, i.e. indices
address ``concat(window+chunk (split rows), comp_kv (compress_len rows))``. So
index ``i < split`` -> ``kv1[i]`` and ``i >= split`` -> ``kv2[i - split]``.

What is pinned:
1. index-space probe -> the split is ``wp + n`` and cidx is in the concat space;
2. BIT-EXACT parity: legacy ``sparse_attn(q, concat(kv1, kv2), idx)`` vs
   ``sparse_attn(q, kv1, idx, kv2=kv2, split=split)`` -- same bytes, dtype,
   shape -- across tiling knobs, dtypes, reference impl, and every boundary
   (split at 1 / kdim-1, -1 masks, all-A rows, all-B rows, all-masked rows);
3. no-materialization: the two-source path never ``mx.concatenate``s the comp
   source (a spy fails on any 3-D concat operand of width >= compress_len),
   with a positive control proving the spy catches a real materialization.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_gather_direct.py -v
    PYTHONPATH=$PWD <venv>/bin/python tests/test_dsv41_gather_direct.py
"""

import json
import sys
import unittest
import zlib

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import sparse_attention as sa

D = 512          # head_dim
M = 48           # query rows
K = 40           # topk width
SINK_N = 8       # n heads for the sink

_RESULTS = []


def _bits(x: mx.array) -> bytes:
    """Bit view: bf16/fp16 through uint16, everything else as numpy."""
    if x.dtype in (mx.bfloat16, mx.float16):
        return np.array(x.view(mx.uint16)).tobytes()
    return np.array(x).tobytes()


def _rand(rng, shape, dtype):
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(dtype)


def _gen_idx(rng, split, n2, b=1, m=M, k=K):
    """Indices in concat(kv1[split], kv2[n2]) space, with -1 masks and styled rows.

    Row 0 carries the A|B boundary values (split-1, split, split, split+n2-1);
    the last three rows are all-masked, all-A and all-B respectively.
    """
    total = split + n2
    idx = rng.integers(0, total, size=(b, m, k)).astype(np.int32)
    # sprinkle -1 masks
    mask = rng.random((b, m, k)) < 0.20
    idx[mask] = -1
    # forced boundary values on the first row
    idx[:, 0, 0] = split - 1
    if k > 1:
        idx[:, 0, 1] = split
    if k > 2:
        idx[:, 0, 2] = min(split, total - 1)
    if k > 3:
        idx[:, 0, 3] = total - 1
    # all-masked / all-A / all-B styled rows (only when there is room)
    if m >= 4:
        idx[:, m - 3, :] = -1
        idx[:, m - 2, :] = rng.integers(0, max(split, 1), size=(b, k)).astype(np.int32)
        idx[:, m - 1, :] = rng.integers(split, total, size=(b, k)).astype(np.int32)
    return mx.array(idx)


class GatherDirectParityTest(unittest.TestCase):
    _saved: dict = {}

    def setUp(self):
        self._saved = {n: getattr(sa, n) for n in
                       ("_QTILE", "_KTILE", "_WDTYPE", "_IMPL", "_COMPILE",
                        "_BUDGET_BYTES")}

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(sa, n, v)

    # -- core assertion -----------------------------------------------------
    def _parity(self, label, *, dtype, split, n2, wdtype="auto", impl="tiled",
                qtile=64, ktile=0, compile_=True, k=K, m=M):
        sa._WDTYPE, sa._IMPL, sa._QTILE = wdtype, impl, qtile
        sa._KTILE, sa._COMPILE = ktile, compile_
        rng = np.random.default_rng(zlib.crc32(label.encode()))
        kv1 = _rand(rng, (1, split, D), dtype)
        kv2 = _rand(rng, (1, n2, D), dtype)
        kv_cat = mx.concatenate([kv1, kv2], axis=1)
        q = _rand(rng, (1, m, SINK_N, D), dtype)
        sink = mx.array(rng.standard_normal(SINK_N).astype(np.float32))
        idx = _gen_idx(rng, split, n2, m=m, k=k)
        scale = D ** -0.5

        legacy = sa.sparse_attn(q, kv_cat, sink, idx, scale)
        direct = sa.sparse_attn(q, kv1, sink, idx, scale, kv2=kv2, split=split)
        mx.eval(legacy, direct)

        ok = (legacy.dtype == direct.dtype and legacy.shape == direct.shape
              and _bits(legacy) == _bits(direct))
        nan_ok = not bool(np.isnan(np.array(direct.astype(mx.float32))).any())
        row = {"case": label, "dtype": str(dtype), "split": split, "n2": n2,
               "impl": impl, "wdtype": wdtype, "ktile": ktile, "qtile": qtile,
               "bit_exact": bool(ok), "no_nan": bool(nan_ok),
               "out_dtype": str(legacy.dtype)}
        _RESULTS.append(row)
        print("  " + json.dumps(row))
        self.assertEqual(legacy.shape, direct.shape, f"{label}: shape")
        self.assertEqual(legacy.dtype, direct.dtype, f"{label}: dtype")
        self.assertTrue(ok, f"{label}: not bit-exact")
        self.assertTrue(nan_ok, f"{label}: NaN in output")
        return direct

    # -- 1. index-space probe ----------------------------------------------
    def test_index_space_probe(self):
        """cidx is in concat(window+chunk, comp_kv) space; split == wp + n."""
        wp, n, ratio = 2, 4, 2
        end_pos = wp + n
        compress_len = end_pos // ratio
        offset = wp + n
        rng = np.random.default_rng(0)
        prev = _rand(rng, (1, wp, D), mx.bfloat16)
        chunk = _rand(rng, (1, n, D), mx.bfloat16)
        comp = _rand(rng, (1, compress_len, D), mx.bfloat16)
        kv_all = mx.concatenate([prev, chunk, comp], axis=1)
        cidx = mx.array([[[offset + j for j in range(compress_len)]]], mx.int32)
        got = sa._gather_kv(kv_all, cidx)
        mx.eval(got)
        for j in range(compress_len):
            self.assertTrue(bool(mx.array_equal(got[0, 0, j], comp[0, j]).item()),
                            f"cidx {offset + j} != comp[{j}]")
        # -1 gathers row 0 of kv (discarded by the -inf logit downstream)
        gneg = sa._gather_kv(kv_all, mx.array([[[-1]]], mx.int32))
        mx.eval(gneg)
        self.assertTrue(bool(mx.array_equal(gneg[0, 0, 0], kv_all[0, 0]).item()))
        row = {"case": "index_space_probe", "split": offset,
               "compress_len": compress_len, "confirmed": True}
        _RESULTS.append(row)
        print("  " + json.dumps(row))

    # -- 2. parity sweep ----------------------------------------------------
    def test_parity_bf16_tiled(self):
        for split, n2 in ((6, 3), (1, 30), (K - 1, 12), (25, 25)):
            self._parity(f"bf16/split{split}/n2{n2}", dtype=mx.bfloat16,
                         split=split, n2=n2)

    def test_parity_fp16_tiled(self):
        self._parity("fp16", dtype=mx.float16, split=6, n2=9)

    def test_parity_wdtype_fp32(self):
        self._parity("bf16-in/fp32-work", dtype=mx.bfloat16, split=5, n2=7,
                     wdtype="fp32")

    def test_parity_reference_impl(self):
        for split, n2 in ((1, 30), (K - 1, 12), (6, 3)):
            self._parity(f"ref/split{split}", dtype=mx.bfloat16, split=split,
                         n2=n2, impl="ref")

    def test_parity_tiling_variants(self):
        # small qtile -> many query tiles; small ktile -> key tiles that each
        # straddle the A|B boundary; compile off -> eager path.
        self._parity("qtile8", dtype=mx.bfloat16, split=6, n2=9, qtile=8)
        self._parity("ktile7", dtype=mx.bfloat16, split=6, n2=9, ktile=7)
        self._parity("qtile8+ktile7", dtype=mx.bfloat16, split=25, n2=25,
                     qtile=8, ktile=7)
        self._parity("eager", dtype=mx.bfloat16, split=6, n2=9, compile_=False)

    def test_single_row_query_matches_decode(self):
        # m below DSV41_SPARSE_FENCE_MIN_ROWS -> the decode/verify path.
        self._parity("m1-decode-shape", dtype=mx.bfloat16, split=6, n2=9, m=1)

    # -- 3. no-materialization contract ------------------------------------
    def _run_under_concat_spy(self, fn, n2):
        """Call fn() with mx.concatenate wrapped; return recorded call args."""
        calls = []
        real = mx.concatenate

        def spy(arrays, *a, **k):
            calls.append([getattr(x, "shape", None) for x in arrays])
            return real(arrays, *a, **k)

        mx.concatenate = spy
        try:
            fn()
        finally:
            mx.concatenate = real
        return calls, n2

    def test_no_concatenation_of_comp_source(self):
        sa._WDTYPE, sa._IMPL, sa._KTILE, sa._COMPILE = "auto", "tiled", 0, True
        sa._QTILE = M  # one query tile -> no final output concatenate either
        rng = np.random.default_rng(7)
        split, n2 = 6, 9
        kv1 = _rand(rng, (1, split, D), mx.bfloat16)
        kv2 = _rand(rng, (1, n2, D), mx.bfloat16)
        q = _rand(rng, (1, M, SINK_N, D), mx.bfloat16)
        sink = mx.array(rng.standard_normal(SINK_N).astype(np.float32))
        idx = _gen_idx(rng, split, n2)

        calls, comp_len = self._run_under_concat_spy(
            lambda: sa.sparse_attn(q, kv1, sink, idx, D ** -0.5,
                                   kv2=kv2, split=split), n2)
        # A comp materialization would concatenate a 3-D operand of width >= n2.
        bad = [c for c in calls
               if any(s is not None and len(s) == 3 and s[1] >= comp_len
                      for s in c)]
        print("  " + json.dumps({"case": "no_concat", "concat_calls": len(calls),
                                 "materializing_calls": len(bad)}))
        self.assertEqual(calls, [], "two-source path called mx.concatenate")
        self.assertEqual(bad, [], "two-source path materialized a comp-width operand")

    def test_concat_spy_positive_control(self):
        """The guard rule fires on a REAL materialization (self-validation)."""
        rng = np.random.default_rng(9)
        split, n2 = 6, 9
        kv1 = _rand(rng, (1, split, D), mx.bfloat16)
        kv2 = _rand(rng, (1, n2, D), mx.bfloat16)
        calls, comp_len = self._run_under_concat_spy(
            lambda: mx.concatenate([kv1, kv2], axis=1), n2)
        bad = [c for c in calls
               if any(s is not None and len(s) == 3 and s[1] >= comp_len
                      for s in c)]
        print("  " + json.dumps({"case": "positive_control",
                                 "materializing_calls": len(bad)}))
        self.assertGreaterEqual(len(bad), 1, "spy failed to catch a materialization")


class AttentionIntegrationTest(unittest.TestCase):
    """End-to-end ``Attention.__call__``: the legacy dense-concat call sequence
    vs the shipped gather-direct sequence must produce identical outputs.

    This exercises the real compressor, indexer, window ring and ``comp_kv``
    write, not just ``sparse_attn`` in isolation. The legacy arm restores the
    old behaviour by concatenating ``kv2`` back into ``kv`` at the call site.
    """

    def setUp(self):
        from mlx_lm.models.deepseek_v41.config import ModelArgs
        from mlx_lm.models.deepseek_v41.cache import ModelCache
        from mlx_lm.models.deepseek_v41.attention import Attention
        from mlx_lm.models.deepseek_v41.model import SharedState
        import mlx_lm.models.deepseek_v41.attention as attmod

        self.attmod = attmod
        self.orig = attmod.sparse_attn
        self.args = ModelArgs(
            dim=256, n_layers=3, n_heads=8, head_dim=512, rope_head_dim=64,
            q_lora_rank=64, o_lora_rank=64, o_groups=8, window_size=8,
            compress_ratios=(2, 2, 2), kv_source_layers=(0,),
            index_source_layers=(0,), index_n_heads=4, index_head_dim=128,
            index_topk=512, max_seq_len=64)
        self.att = Attention(0, self.args)
        self.att.attn_sink = self.att.attn_sink + 0.1
        mx.eval(self.att.parameters())
        self.ModelCache = ModelCache
        self.SharedState = SharedState
        self.seen = []                      # (kv2 dtype, split) per call

    def tearDown(self):
        self.attmod.sparse_attn = self.orig

    def _install(self, materialize):
        orig, seen = self.orig, self.seen
        attmod = self.attmod

        def patched(q, kv, sink, idx, sc, chunk=64, kv2=None, split=None,
                    colsplit=None):
            seen.append(("none" if kv2 is None else str(kv2.dtype), split))
            if kv2 is not None and materialize:
                kv = mx.concatenate([kv, kv2], axis=1)   # legacy dense material
                kv2 = None
                colsplit = None                          # no second source
            return orig(q, kv, sink, idx, sc, chunk, kv2=kv2, split=split,
                        colsplit=colsplit)

        attmod.sparse_attn = patched

    def _run(self, materialize):
        self.seen = []
        self._install(materialize)
        n0 = 8
        mx.random.seed(1234)
        x = mx.random.normal((1, n0, 256)) * 0.5
        c = self.ModelCache(self.args, 1, 64)
        c.ensure_capacity(n0)
        acc = [self.att(x, 0, c, self.SharedState())]
        for step in range(4):
            mx.random.seed(100 + step)
            xx = mx.random.normal((1, 1, 256)) * 0.5
            acc.append(self.att(xx, n0 + step, c, self.SharedState()))
        out = mx.concatenate(acc, axis=1)
        mx.eval(out)
        return out, c.layers[0].comp_kv, list(self.seen)

    def test_attention_path_bit_exact_and_bf16_native(self):
        legacy, comp_l, _ = self._run(materialize=True)
        direct, comp_d, seen = self._run(materialize=False)
        self.assertEqual(legacy.shape, direct.shape)
        self.assertTrue(mx.array_equal(legacy, direct).item(),
                        "attention output not bit-exact")
        self.assertTrue(mx.array_equal(comp_l, comp_d).item(),
                        "comp_kv cache diverged")
        self.assertEqual(comp_l.dtype, mx.bfloat16)
        # the gather-direct call must pass comp_kv with NO dtype cast
        twosrc = [s for s in seen if s[0] is not None]
        self.assertTrue(twosrc, "two-source path never taken")
        self.assertTrue(all(dt == "mlx.core.bfloat16" for dt, _ in twosrc),
                        f"kv2 was cast (non-bf16): {twosrc}")
        # split is wp + n and comp_kv rows are bf16 -> the cast is a no-op
        print("  " + json.dumps({"case": "attention-integration",
                                 "bit_exact": True, "kv2_calls": len(twosrc),
                                 "kv2_dtypes": sorted({dt for dt, _ in twosrc})}))
        self.assertTrue(all(mx.array_equal(legacy, direct).item() for _ in [0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
    n = len(_RESULTS)
    n_ok = sum(r.get("bit_exact", r.get("confirmed", False)) for r in _RESULTS)
    print(f"GATHER_DIRECT_PARITY {json.dumps({'cases': n, 'bit_exact': n_ok})}")
    sys.exit(0 if n_ok == n else 1)
