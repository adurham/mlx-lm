# Copyright © 2026 Adam Durham (hermes-gw)
"""TP-sharded affine dense mode: slice geometry + quantization equivalence.

``DENSE_MODE = affineN`` used to disable TP sharding of the dense groups (the
``attn_tp``/``shared_tp`` gates were ``... and DENSE_MODE == "exl3"``), so each
rank reconstructed and quantized the FULL dense group -- the priced 2.4x was
lost. The fix makes affine shard exactly like exl3: reconstruct the full dense
weight, slice the fp16 ``[out, in]`` tensor on the SAME 128-wide block
boundaries ``_slice_dense`` uses, then affine-quantize the slice
(``AffineProj.from_weight``).

This test pins, on a synthetic EXL3 group (no Metal quant checkpoint needed):

1. **geometry**: ``_slice_weight`` returns rank r's 128-boundary slice and the
   rank-0/rank-1 slices concatenate back to the full in/out features -- and the
   weight slice equals ``reconstruct_public_mlx(_slice_dense(...))`` (the
   trellis slice), so the two geometries agree;
2. **quantization equivalence**: every per-rank affine slice equals the
   corresponding block of the full-reconstruct-then-``mx.quantize`` path at
   ``group_size=64`` (bits 8/6/5) -- the affine groups (64) and the Hadamard
   blocks (128) both divide the slice boundary, so the quantization is exact;
3. **orientation guard**: ``from_weight`` refuses a transposed weight;
4. **dispatcher**: ``_dense_slice`` in ``affineN`` mode returns an
   ``AffineProj`` whose weight matches the same full-quantize blocks.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_dense_affine_shard.py -x -q
"""

import unittest
from unittest import mock

import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3.ref.layer import EXL3Layer
from mlx_lm.models.exl3.reconstruct import reconstruct_public_mlx
from mlx_lm.models.deepseek_v41 import exl3_build as eb

IN_TILES, OUT_TILES, K = 16, 32, 2                      # in 256, out 512
IN_F, OUT_F = IN_TILES * 16, OUT_TILES * 16
WORLD = 2
GROUP = 64


def _synth(key: str = "synth", seed: int = 0) -> EXL3Layer:
    rng = np.random.default_rng(seed)
    packed = 256 * K // 16
    return EXL3Layer(
        key=key,
        in_features=IN_F,
        out_features=OUT_F,
        k=K,
        trellis=rng.integers(0, 2 ** 16, size=(IN_TILES, OUT_TILES, packed), dtype=np.uint16),
        suh=rng.choice(np.array([-1.0, 1.0], np.float16), size=IN_F),
        svh=rng.choice(np.array([-1.0, 1.0], np.float16), size=OUT_F),
        mul1=True,
    )


def _full_wt(layer: EXL3Layer) -> mx.array:
    w = reconstruct_public_mlx(layer)                   # [in, out] fp16
    mx.eval(w)
    return mx.contiguous(w.T)                           # [out, in] fp16


def _tuples_equal(a, b) -> bool:
    mx.eval(*a, *b)
    return all(mx.array_equal(x, y) for x, y in zip(a, b))


class GeometryTest(unittest.TestCase):
    """Rank slices tile the full weight exactly, on 128-wide boundaries."""

    def setUp(self):
        self.layer = _synth()
        self.full = _full_wt(self.layer)

    def test_out_axis_slices_concatenate_to_full(self):
        parts = [eb._slice_weight(self.layer, axis="out", rank=r, world=WORLD)
                 for r in range(WORLD)]
        mx.eval(*parts)
        # each rank owns OUT_F / WORLD = 256 output features (2 x 128-blocks)
        self.assertEqual(parts[0].shape, (OUT_F // WORLD, IN_F))
        self.assertEqual(parts[1].shape, (OUT_F // WORLD, IN_F))
        for r, p in enumerate(parts):
            lo = r * OUT_F // WORLD
            self.assertTrue(mx.array_equal(p, self.full[lo:lo + OUT_F // WORLD]),
                            f"rank {r} out-slice != full rows")
        self.assertTrue(mx.array_equal(mx.concatenate(parts, axis=0), self.full),
                        "out-axis ranks do not reconstruct the full weight")

    def test_in_axis_slices_concatenate_to_full(self):
        parts = [eb._slice_weight(self.layer, axis="in", rank=r, world=WORLD)
                 for r in range(WORLD)]
        mx.eval(*parts)
        self.assertEqual(parts[0].shape, (OUT_F, IN_F // WORLD))
        for r, p in enumerate(parts):
            lo = r * IN_F // WORLD
            self.assertTrue(mx.array_equal(p, self.full[:, lo:lo + IN_F // WORLD]),
                            f"rank {r} in-slice != full cols")
        self.assertTrue(mx.array_equal(mx.concatenate(parts, axis=1), self.full),
                        "in-axis ranks do not reconstruct the full weight")

    def test_weight_slice_equals_trellis_slice(self):
        # the fp16 weight slice must equal reconstructing the _slice_dense trellis
        for axis in ("out", "in"):
            for r in range(WORLD):
                trellis_slice = reconstruct_public_mlx(
                    eb._slice_dense(self.layer, axis=axis, rank=r, world=WORLD))
                mx.eval(trellis_slice)
                w_slice = eb._slice_weight(self.layer, axis=axis, rank=r, world=WORLD)
                mx.eval(w_slice)
                self.assertTrue(mx.array_equal(w_slice, mx.contiguous(trellis_slice.T)),
                                f"{axis} rank {r}: weight slice != trellis slice")

    def test_block_bounds_check(self):
        # 16 tiles / world=2 -> 8 tiles per rank; a non-divisible count raises
        self.assertEqual(eb._block_bounds(OUT_TILES, 0, WORLD, "out_tiles", "k"), (0, 16))
        with self.assertRaises(ValueError):
            eb._block_bounds(OUT_TILES, 0, 3, "out_tiles", "k")


class QuantEquivalenceTest(unittest.TestCase):
    """Per-rank affine slices equal the full-quantize path's matching blocks."""

    def setUp(self):
        self.layer = _synth(seed=1)
        self.full = _full_wt(self.layer)

    def _check(self, axis: str, bits: int):
        ref = mx.quantize(self.full, group_size=GROUP, bits=bits)     # full path
        mx.eval(*ref)
        for r in range(WORLD):
            w_slice = eb._slice_weight(self.layer, axis=axis, rank=r, world=WORLD)
            proj = eb.AffineProj.from_weight(w_slice, bits, GROUP)
            if axis == "out":
                lo = r * OUT_F // WORLD
                want = (ref[0][lo:lo + OUT_F // WORLD],
                        ref[1][lo:lo + OUT_F // WORLD],
                        ref[2][lo:lo + OUT_F // WORLD])
            else:
                lo = (r * IN_F // WORLD) // GROUP
                hi = ((r + 1) * IN_F // WORLD) // GROUP
                want = (ref[0][:, r * (IN_F // WORLD) * bits // 32:
                               (r + 1) * (IN_F // WORLD) * bits // 32],
                        ref[1][:, lo:hi],
                        ref[2][:, lo:hi])
            self.assertTrue(_tuples_equal(proj._q, want),
                            f"{axis} rank {r} bits {bits}: from_weight != full quantize")

    def test_out_axis_bits8_6_5(self):
        for bits in (8, 6, 5):
            self._check("out", bits)

    def test_in_axis_bits8_6_5(self):
        for bits in (8, 6, 5):
            self._check("in", bits)

    def test_functional_matmul_out(self):
        # the per-rank out-slice matmuls concatenate to the full matmul
        bits = 8
        x = mx.random.normal((3, IN_F)).astype(mx.float16)
        ref = mx.quantize(self.full, group_size=GROUP, bits=bits)
        y_full = mx.quantized_matmul(x, *ref, transpose=True, group_size=GROUP, bits=bits)
        mx.eval(y_full)
        ys = []
        for r in range(WORLD):
            proj = eb.AffineProj.from_weight(
                eb._slice_weight(self.layer, axis="out", rank=r, world=WORLD), bits, GROUP)
            ys.append(proj(x))
        mx.eval(*ys)
        self.assertTrue(mx.array_equal(mx.concatenate(ys, axis=-1), y_full),
                        "out-slice matmuls do not concatenate to the full matmul")

    def test_functional_matmul_in(self):
        # the per-rank in-slice partial matmuls sum to the full matmul. Unlike
        # the out-axis (row) slice, this is NOT bit-exact: summing two fp16
        # partials reassociates the accumulation order, so it holds to fp16
        # precision (the exact claim is the quantization-block equivalence above).
        # Seed the input so this tolerance check is deterministic (it flaked
        # ~1/12 unseeded -- the tolerance is a property of the arithmetic, not
        # of a lucky draw).
        mx.random.seed(20261009)
        bits = 8
        x = mx.random.normal((3, IN_F)).astype(mx.float16)
        ref = mx.quantize(self.full, group_size=GROUP, bits=bits)
        y_full = mx.quantized_matmul(x, *ref, transpose=True, group_size=GROUP, bits=bits)
        mx.eval(y_full)
        acc = None
        for r in range(WORLD):
            lo = r * IN_F // WORLD
            proj = eb.AffineProj.from_weight(
                eb._slice_weight(self.layer, axis="in", rank=r, world=WORLD), bits, GROUP)
            part = proj(x[:, lo:lo + IN_F // WORLD])
            acc = part if acc is None else acc + part
        mx.eval(acc)
        self.assertTrue(mx.allclose(acc, y_full, rtol=1e-2, atol=1e-2),
                        "in-slice partials do not sum to the full matmul")


class OrientationGuardTest(unittest.TestCase):
    def test_transposed_weight_rejected(self):
        layer = _synth()
        full = _full_wt(layer)                        # [out, in]
        # a still-[in, out] weight with the [out, in] expectation must raise
        with self.assertRaises(ValueError):
            eb.AffineProj.from_weight(mx.contiguous(full.T), 8, GROUP,
                                      expect=(OUT_F, IN_F))
        # correct orientation passes
        eb.AffineProj.from_weight(full, 8, GROUP, expect=(OUT_F, IN_F))

    def test_non_2d_rejected(self):
        with self.assertRaises(ValueError):
            eb.AffineProj.from_weight(mx.zeros((4, 8, 16)), 8, GROUP)


class DispatcherTest(unittest.TestCase):
    """_dense_slice in affineN mode builds an AffineProj on the sliced weight."""

    def test_affine_mode_returns_affine_proj(self):
        layer = _synth(seed=2)
        full = _full_wt(layer)
        loader = "mlx_lm.models.exl3.loader.load_dense_layer"
        for mode, bits in (("affine8", 8), ("affine6", 6), ("affine5", 5)):
            with mock.patch.object(eb, "DENSE_MODE", mode), mock.patch(loader,
                                                                       lambda ck, n: layer):
                proj = eb._dense_slice(None, "layers.0.attn.wq_b", axis="out",
                                       rank=0, world=WORLD)
            self.assertIsInstance(proj, eb.AffineProj)
            self.assertEqual(proj._bits, bits)
            ref = mx.quantize(full, group_size=GROUP, bits=bits)
            mx.eval(*ref)
            want = (ref[0][0:OUT_F // WORLD], ref[1][0:OUT_F // WORLD],
                    ref[2][0:OUT_F // WORLD])
            self.assertTrue(_tuples_equal(proj._q, want),
                            f"{mode}: dispatcher slice != full quantize block")

    def test_exl3_mode_returns_exl3_proj(self):
        layer = _synth(seed=3)
        loader = "mlx_lm.models.exl3.loader.load_dense_layer"
        with mock.patch.object(eb, "DENSE_MODE", "exl3"), mock.patch(loader,
                                                                     lambda ck, n: layer):
            proj = eb._dense_slice(None, "layers.0.attn.wq_b", axis="out",
                                   rank=0, world=WORLD)
        self.assertIsInstance(proj, eb.Exl3Proj)


if __name__ == "__main__":
    unittest.main()
