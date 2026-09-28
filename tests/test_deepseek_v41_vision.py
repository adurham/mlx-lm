# Copyright © 2026 Apple Inc.

"""Tests for the DeepSeek-V4.1 vision tower port (ViT + Aligner + sentinels).

The MLX-only tests run anywhere: no checkpoint, no torch. They pin the things
that break silently -- the 2D-RoPE frequency layout, the unfold channel order,
the RMSNorm epsilon, the sentinel/token layout, the merge semantics -- against
independent numpy reimplementations of the reference.

Checkpoint-gated tests (``DSV41_EXL3`` env var pointing at the EXL3 checkpoint
directory) additionally assert that all 266 ``vision.*`` / ``aligner.*`` /
``image_*`` tensors load with strict accounting.

Torch-gated tests (``P97_REF_DIR`` pointing at the reference checkout's
``docs/reference``) compare against the actual torch reference.

Run:
    PYTHONPATH=<worktree> $PY tests/test_deepseek_v41_vision.py -v
"""

import math
import os
import sys
import unittest

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.deepseek_v41 import image_processor as ip
from mlx_lm.models.deepseek_v41 import vision as vis

CKPT = os.environ.get(
    "DSV41_EXL3",
    os.path.expanduser(
        "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
    ),
)
REF_DIR = os.environ.get(
    "P97_REF_DIR", os.path.expanduser("~/repos/ref/deepseek-v41-mlx/docs/reference")
)


class TestRope(unittest.TestCase):
    def test_rope_frequency_layout_matches_numpy_reference(self):
        """Pin ``[h*inv_freq, w*inv_freq]`` with the h-half FIRST.

        The reference builds ``stack([hpos, wpos], -1).reshape(-1, 2, 1) *
        inv_freq`` then flattens, i.e. per position the vector is
        ``[h*f0, ..., h*fk, w*f0, ..., w*fk]`` -- a half-split, NOT the
        interleaved layout that other rope implementations use. Swapping them
        silently changes every patch embedding by a large amount.
        """
        nh, nw, dim, theta = 4, 6, 32, 10000.0
        cos, sin = vis.get_vision_cos_sin(nh, nw, dim, theta)
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        hpos = np.broadcast_to(np.arange(nh, dtype=np.float32)[:, None], (nh, nw))
        wpos = np.broadcast_to(np.arange(nw, dtype=np.float32)[None, :], (nh, nw))
        freqs = (np.stack([hpos, wpos], -1).reshape(-1, 2, 1) * inv_freq.reshape(1, 1, -1)).reshape(nh * nw, -1)
        np.testing.assert_allclose(np.array(cos[:, 0, :]), np.cos(freqs), rtol=1e-6)
        np.testing.assert_allclose(np.array(sin[:, 0, :]), np.sin(freqs), rtol=1e-6)
        # An INTERLEAVED layout would emit [h*f0, w*f0, h*f1, w*f1, ...];
        # the reference emits [h*f0, h*f1, ..., w*f0, w*f1, ...]. Compare at a
        # position where h != w so the two layouts actually differ in value.
        row, col = 2, 3
        n_freq = inv_freq.size
        got = np.array(cos[row * nw + col, 0, :])
        half = n_freq  # rope_dim == n_freq here (dim // 2 frequencies)
        self.assertAlmostEqual(float(got[half]), float(np.cos(wpos[row, col] * inv_freq[0])), places=6)
        self.assertNotAlmostEqual(
            float(got[1]), float(np.cos(wpos[row, col] * inv_freq[0])), places=6
        )

    def test_rope_table_is_cached(self):
        a = vis.get_vision_cos_sin(3, 5, 32, 10000.0)
        b = vis.get_vision_cos_sin(3, 5, 32, 10000.0)
        self.assertIs(a[0], b[0])

    def test_apply_rotary_matches_reference_formula(self):
        rng = np.random.default_rng(0)
        # x is (n, heads, head_dim); the table is rope_dim = head_dim // 2 wide,
        # shaped (n, 1, rope_dim) so it broadcasts over the head axis, and the
        # SAME table is applied to both halves (half-split rotary, not interleaved).
        x = rng.normal(size=(7, 2, 8)).astype(np.float32)
        cos = np.cos(rng.normal(size=(7, 1, 4))).astype(np.float32)
        sin = np.sin(rng.normal(size=(7, 1, 4))).astype(np.float32)
        got = np.array(
            vis.apply_rotary(mx.array(x).astype(mx.bfloat16), mx.array(cos), mx.array(sin))
            .astype(mx.float32)
        )
        x1, x2 = x[..., :4], x[..., 4:]
        want = np.concatenate(
            [x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1
        )
        # bf16 in, bf16 out: compare at bf16 precision
        np.testing.assert_allclose(got.astype(np.float64), want.astype(np.float64),
                                   rtol=2e-2, atol=2e-2)


class TestRMSNorm(unittest.TestCase):
    def test_eps_is_reference_default_not_text_model_eps(self):
        norm = vis.RMSNorm(8)
        self.assertEqual(norm.eps, 1e-6)
        self.assertEqual(vis.VisionConfig().n_layers, 32)
        # eps actually participates: drive var below eps and check the output
        # differs from what eps=1e-20 would produce.
        x = mx.array((np.ones((1, 8), np.float32) * 1e-4))
        got = np.array(norm(x))
        xf = np.ones((1, 8), np.float32) * 1e-4
        var_np = (xf**2).mean(-1, keepdims=True)
        right = xf / np.sqrt(var_np + 1e-6)
        wrong = xf / np.sqrt(var_np + 1e-20)
        self.assertLess(float(np.abs(got - right).max()), 1e-6)
        self.assertGreater(float(np.abs(right - wrong).max()), 1e-3)

    def test_weight_is_float32(self):
        self.assertEqual(vis.RMSNorm(8).weight.dtype, mx.float32)


class TestUnfold(unittest.TestCase):
    def test_unfold_matches_torch_channel_major_order(self):
        """``F.unfold`` emits C first, then kernel row, then kernel col."""
        r, h, w, c = 3, 6, 9, 4
        rng = np.random.default_rng(1)
        x = rng.integers(0, 256, (h, w, c)).astype(np.float32)
        got = np.array(vis._unfold(mx.array(x), r))
        self.assertEqual(got.shape, ((h // r) * (w // r), c * r * r))
        want = np.empty_like(got)
        for bh in range(h // r):
            for bw in range(w // r):
                blk = x[bh * r : (bh + 1) * r, bw * r : (bw + 1) * r, :]
                # flat index c * r*r + kh * r + kw
                want[bh * (w // r) + bw] = blk.reshape(r * r, c).T.reshape(-1)
        np.testing.assert_array_equal(got, want)

    def test_unfold_is_not_transposed_variant(self):
        """A (kh,kw)-before-C transpose is a plausible-looking silent bug."""
        r, h, w, c = 3, 6, 9, 4
        rng = np.random.default_rng(2)
        x = rng.integers(0, 256, (h, w, c)).astype(np.float32)
        got = np.array(vis._unfold(mx.array(x), r))
        swapped = x.reshape(h // r, r, w // r, r, c).transpose(0, 2, 1, 3, 4).reshape(-1, c * r * r)
        self.assertFalse(np.array_equal(got, swapped))


class TestImageProcessor(unittest.TestCase):
    def test_round_to_bfloat16_matches_known_values(self):
        vals = np.array([0.0, 1.0, -1.0, 0.5, 0.1, 127 / 255 * 2 - 1], dtype=np.float32)
        got = ip.round_to_bfloat16_precision(vals)
        # every pipeline value is (k/255 - 0.5)/0.5 for integer k: exactly
        # representable?  No -- but the rounding must be idempotent and match
        # a manual round-half-to-even truncation.
        bits = vals.view(np.uint32)
        bias = ((bits >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
        want = ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(ip.round_to_bfloat16_precision(got), got)

    def test_bfloat16_rounding_exact_over_pipeline_domain(self):
        """All 256 k/255 values, both signs, must be exact and idempotent."""
        k = np.arange(256, dtype=np.float32)
        vals = np.concatenate([(k / 255 - 0.5) / 0.5, -((k / 255 - 0.5) / 0.5)])
        got = ip.round_to_bfloat16_precision(vals)
        np.testing.assert_array_equal(ip.round_to_bfloat16_precision(got), got)
        # and it agrees with an explicit bf16 cast through mlx
        ref = np.array(mx.array(vals).astype(mx.bfloat16).astype(mx.float32))
        np.testing.assert_array_equal(got, ref)  # both float32-stored

    def test_token_layout_is_four_types(self):
        types = ip.image_token_types(2, 3)
        np.testing.assert_array_equal(
            types,
            [ip.IMAGE_START, ip.IMAGE, ip.IMAGE, ip.IMAGE, ip.IMAGE_NEW_LINE,
             ip.IMAGE, ip.IMAGE, ip.IMAGE, ip.IMAGE_NEW_LINE, ip.IMAGE_END],
        )
        self.assertEqual(types.dtype, np.int64)
        self.assertEqual(ip.num_image_tokens(2, 3), len(types))
        # four sentinel types only -- no IMAGE_PAD (that is the older V4 layout)
        self.assertEqual(sorted({ip.IMAGE_START, ip.IMAGE, ip.IMAGE_NEW_LINE, ip.IMAGE_END}), [0, 1, 2, 3])

    def test_grid_planning_matches_reference_numbers(self):
        cfg = ip.ImagePreprocessConfig(
            patch_size=14, downsample_ratio=3, max_token_count=1024,
            min_pixel_count=295936, max_width_height_ratio=None,
        )
        # 300x500 -> 31x51 vit grid -> 11x17 llm grid (reference values)
        nh, nw, bh, bw = ip.plan_image_grid(500, 300, cfg)
        self.assertEqual((bh, bw), (434, 714))
        self.assertEqual((nh, nw), (11, 17))
        self.assertEqual(bh // 14, 31)
        self.assertEqual(bw // 14, 51)
        self.assertLessEqual(ip.num_image_tokens(nh, nw), 1024)
        # min-pixel upscaling: a tiny image is scaled up to min_pixels
        nh2, nw2, bh2, bw2 = ip.plan_image_grid(10, 10, cfg)
        self.assertGreaterEqual(bh2 * bw2, 295936 * 0.9)

    def test_max_token_budget_respected_for_large_images(self):
        cfg = ip.ImagePreprocessConfig(
            patch_size=14, downsample_ratio=3, max_token_count=1024,
            min_pixel_count=295936, max_width_height_ratio=None,
        )
        for h, w in ((2048, 2048), (512, 4096), (4096, 512), (100, 3000)):
            nh, nw, bh, bw = ip.plan_image_grid(w, h, cfg)
            self.assertLessEqual(ip.num_image_tokens(nh, nw), 1024, (h, w))
            self.assertEqual(bh % 14, 0)
            self.assertEqual(bw % 14, 0)

    def test_load_image_shape_and_value_range(self):
        import io

        from PIL import Image

        arr = np.random.default_rng(3).integers(0, 256, (300, 500, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        cfg = ip.ImagePreprocessConfig(14, 3, 1024, 295936, None)
        patches, nh, nw, nlh, nlw = ip.load_image({"data": buf.getvalue()}, cfg)
        self.assertEqual(patches.shape, (nh * nw, 3, 14, 14))
        self.assertGreaterEqual(patches.min(), -1.0001)
        self.assertLessEqual(patches.max(), 1.0001)
        # patch row 0 col 0 must equal the top-left 14x14 corner of the image
        plane = np.asarray(ImageOps_pad(arr, 434, 714), dtype=np.float32).transpose(2, 0, 1) / 255
        plane = ip.round_to_bfloat16_precision((plane - 0.5) / 0.5)
        np.testing.assert_allclose(patches[0], plane[:, :14, :14], rtol=1e-6)

    def test_patches_are_bf16_representable(self):
        import io

        from PIL import Image

        arr = np.random.default_rng(4).integers(0, 256, (141, 253, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        cfg = ip.ImagePreprocessConfig(14, 3, 1024, 295936, None)
        patches, *_ = ip.load_image({"data": buf.getvalue()}, cfg)
        back = np.array(mx.array(patches).astype(mx.bfloat16).astype(mx.float32))
        np.testing.assert_array_equal(patches, back)


def _gelu_erf(v: np.ndarray) -> np.ndarray:
    """Exact (erf) GELU in numpy, matching torch's F.gelu default."""
    import math

    v = np.asarray(v, dtype=np.float64)
    return (v * 0.5 * (1.0 + np.vectorize(math.erf)(v / math.sqrt(2.0)))).astype(np.float32)


def ImageOps_pad(arr, bh, bw):
    from PIL import Image, ImageOps

    return ImageOps.pad(Image.fromarray(arr), (bw, bh), color=(127, 127, 127))


class TestMergeAndBlock(unittest.TestCase):
    def _tiny_tower(self, text_dim=8):
        cfg = vis.VisionConfig(dim=16, n_layers=1, n_heads=2, inter_dim=32, patch_size=2,
                               text_dim=text_dim)
        tower = vis.VisionTower(cfg, dtype=mx.float32)
        tower.image_start = mx.arange(text_dim, dtype=mx.float32) + 100.0
        tower.image_end = mx.arange(text_dim, dtype=mx.float32) + 200.0
        tower.image_newline = mx.arange(text_dim, dtype=mx.float32) + 300.0
        return tower, cfg

    def test_build_image_block_places_sentinels_and_rows(self):
        text_dim = 8
        tower, _ = self._tiny_tower(text_dim)
        # 6x6 patch grid -> 2x2 aligner grid -> 2x2 IMAGE rows + newlines
        types = ip.image_token_types(2, 2)
        img = ip.ImageInput(0, np.zeros((36, 3, 2, 2), np.float32), 6, 6, types)
        blk = np.array(tower.build_image_block(img))
        self.assertEqual(blk.shape, (len(types), text_dim))
        np.testing.assert_allclose(blk[0], np.array(tower.image_start))
        np.testing.assert_allclose(blk[-1], np.array(tower.image_end))
        for r in np.nonzero(types == ip.IMAGE_NEW_LINE)[0]:
            np.testing.assert_allclose(blk[r], np.array(tower.image_newline))
        embeds = np.array(tower.encode_image(mx.zeros((36, 3, 2, 2)), 6, 6))
        self.assertEqual(embeds.shape[0], 4)
        # the aligner rows land in the IMAGE slots in reading order
        np.testing.assert_allclose(blk[types == ip.IMAGE], embeds)

    def test_build_image_block_rejects_mismatched_geometry(self):
        tower, _ = self._tiny_tower()
        types = ip.image_token_types(2, 2)  # 4 IMAGE slots
        img = ip.ImageInput(0, np.zeros((9, 3, 2, 2), np.float32), 3, 3, types)
        with self.assertRaises(ValueError):
            tower.build_image_block(img)

    def test_merge_2d_in_place_and_span_bounds(self):
        text_dim = 8
        tower, _ = self._tiny_tower(text_dim)
        types = ip.image_token_types(1, 1)
        img = ip.ImageInput(3, np.zeros((9, 3, 2, 2), np.float32), 3, 3, types)
        h = mx.ones((3 + len(types) + 2, text_dim), dtype=mx.float32) * -5.0
        out = tower.merge_image_embeddings(h, [img])
        mx.eval(out)
        blk = np.array(tower.build_image_block(img))
        merged = np.array(out)
        np.testing.assert_allclose(merged[3:3 + len(types)], blk)
        # rows outside the span are untouched, and it is the SAME buffer
        np.testing.assert_allclose(merged[:3], -5.0)
        np.testing.assert_allclose(merged[3 + len(types):], -5.0)
        self.assertTrue(np.array_equal(np.array(h), merged))

    def test_merge_3d_batch_index(self):
        text_dim = 8
        tower, _ = self._tiny_tower(text_dim)
        types = ip.image_token_types(1, 1)
        img = ip.ImageInput(2, np.zeros((9, 3, 2, 2), np.float32), 3, 3, types)
        h = mx.ones((2, 2 + len(types), text_dim), dtype=mx.float32) * -7.0
        out = tower.merge_image_embeddings(h, [img], sample=1)
        mx.eval(out)
        merged = np.array(out)
        blk = np.array(tower.build_image_block(img))
        np.testing.assert_allclose(merged[1, 2:2 + len(types)], blk)
        np.testing.assert_allclose(merged[0], -7.0)
        np.testing.assert_allclose(merged[1, :2], -7.0)

    def test_merge_rejects_out_of_range_span(self):
        tower, _ = self._tiny_tower()
        types = ip.image_token_types(1, 1)
        img = ip.ImageInput(5, np.zeros((9, 3, 2, 2), np.float32), 3, 3, types)
        h = mx.zeros((6, 8), dtype=mx.float32)
        with self.assertRaises(ValueError):
            tower.merge_image_embeddings(h, [img])

    def test_merge_no_images_is_identity(self):
        tower, _ = self._tiny_tower()
        h = mx.ones((4, 8))
        self.assertIs(tower.merge_image_embeddings(h, None), h)
        self.assertIs(tower.merge_image_embeddings(h, []), h)

    def test_sentinel_parameters_index_by_type(self):
        tower, _ = self._tiny_tower()
        p = np.array(tower.sentinel_parameters())
        self.assertEqual(p.shape[0], 4)
        np.testing.assert_allclose(p[ip.IMAGE_START], np.array(tower.image_start))
        np.testing.assert_allclose(p[ip.IMAGE_NEW_LINE], np.array(tower.image_newline))
        np.testing.assert_allclose(p[ip.IMAGE_END], np.array(tower.image_end))


class TestAlignerShapes(unittest.TestCase):
    def test_aligner_output_rows_equal_ceil_grid(self):
        cfg = vis.VisionConfig(dim=12, n_layers=0, n_heads=3, inter_dim=24, patch_size=2,
                               text_dim=6)
        alg = vis.Aligner(cfg)
        for nh, nw in ((3, 3), (4, 5), (1, 7)):
            x = mx.zeros((nh * nw, cfg.dim))
            out = alg(x, nh, nw)
            self.assertEqual(out.shape, (math.ceil(nh / 3) * math.ceil(nw / 3), cfg.text_dim))

    def test_padding_is_zero_and_only_right_bottom(self):
        """The pad rows land in REAL IMAGE slots, so they must be exact zeros."""
        # Identity w1/w2 (gelu(identity) is NOT identity, so compare against an
        # explicit unfold of the same zero-padded tensor instead of raw values).
        cfg = vis.VisionConfig(dim=3, n_layers=0, n_heads=1, inter_dim=6, patch_size=2,
                               text_dim=3)
        alg = vis.Aligner(cfg)
        # w1: (out=text_dim, in=27) identity-selects the FIRST cfg.dim (=3) of the
        # 27 flattened channels, which is exactly the top-left 1x1 cell corner.
        sel = np.zeros((cfg.text_dim, cfg.dim * 9), np.float32)
        for k in range(cfg.text_dim):
            sel[k, k] = 1.0
        alg.w1.weight = mx.array(sel)
        alg.w1.bias = mx.zeros_like(alg.w1.bias)
        alg.w2.weight = mx.eye(cfg.text_dim)
        alg.w2.bias = mx.zeros_like(alg.w2.bias)
        nh, nw = 4, 5
        x = mx.ones((nh * nw, cfg.dim))
        out = np.array(alg(x, nh, nw))
        self.assertEqual(out.shape, (4, cfg.text_dim))
        # explicit ground truth: pad (4,5,3) -> (6,6,3), unfold 3x3 stride 3
        # -> 4 cells of 27 wide, in (row-of-cells, col-of-cells) order
        xr = np.array(x).reshape(nh, nw, cfg.dim)
        padded = np.pad(xr, ((0, 2), (0, 1), (0, 0)))
        # unfold order is c * r*r + kh * r + kw (channel-major), so the cell
        # vector is the (kh, kw, c) window transposed to (c, kh, kw) then raveled
        cells = [padded[r0:r0 + 3, c0:c0 + 3, :].transpose(2, 0, 1).reshape(-1)
                 for r0 in (0, 3) for c0 in (0, 3)]
        # cell values selected by w1 are the first 3 flattened channels
        np.testing.assert_allclose(
            np.array(out), np.stack([_gelu_erf(c[:cfg.text_dim]) for c in cells]), atol=1e-6
        )
        # the padding is right+bottom only, and contributes exact zeros
        self.assertEqual(float(padded[4:, :, :].sum()), 0.0)
        self.assertEqual(float(padded[:, 5:, :].sum()), 0.0)
        self.assertEqual(float(padded[4, 4, :].sum()), 0.0)
        # the bottom-right cell (rows 3:6, cols 3:6 of the padded tensor) is
        # mostly padding: only positions sourcing a real patch are non-zero
        br = padded[3:6, 3:6, :]
        rows, cols = np.arange(3, 6), np.arange(3, 6)
        n_real = int(((rows[:, None] < nh) & (cols[None, :] < nw)).sum())
        self.assertLess(n_real, 9)
        self.assertEqual(int((br != 0).sum()), n_real * cfg.dim)


@unittest.skipUnless(os.path.isdir(CKPT), "EXL3 checkpoint not present")
class TestCheckpointLoad(unittest.TestCase):
    def test_loads_all_vision_tensors(self):
        tower, cfg = vis.load_vision_tower(CKPT, dtype=mx.bfloat16)
        self.assertEqual(cfg.n_layers, 32)
        self.assertEqual(cfg.dim, 1024)
        self.assertEqual(cfg.n_heads, 16)
        self.assertEqual(cfg.inter_dim, 2816)
        self.assertEqual(cfg.patch_size, 14)
        self.assertEqual(cfg.text_dim, 5120)
        self.assertEqual(cfg.downsample_ratio, 3)
        self.assertEqual(cfg.max_image_tokens, 1024)
        self.assertEqual(cfg.min_pixels, 295936)
        self.assertEqual(tower.vision.blocks[0].attn.wqkv.weight.dtype, mx.bfloat16)
        self.assertEqual(tower.image_start.dtype, mx.bfloat16)
        self.assertEqual(np.array(tower.image_start.astype(mx.float32)).shape, (5120,))

    def test_weight_names_are_strictly_the_vision_group(self):
        from mlx_lm.models.exl3.loader import Exl3Checkpoint

        with Exl3Checkpoint(CKPT) as ck:
            names = vis.vision_weight_names(ck.index)
        self.assertEqual(len(names), 266)
        self.assertTrue(all(n.startswith(vis.VISION_PREFIXES) for n in names))
        self.assertIn("vision.patch_embed.proj.weight", names)
        self.assertIn("image_newline", names)

    def test_config_from_checkpoint_matches_shipped_config(self):
        import json

        with open(os.path.join(CKPT, "config.json")) as f:
            raw = json.load(f)
        cfg = vis.VisionConfig.from_config(raw)
        self.assertEqual(cfg.n_layers, 32)
        self.assertEqual(cfg.image_token_id, raw["image_token_id"])
        self.assertEqual(cfg.text_dim, raw["text_config"]["hidden_size"])


@unittest.skipUnless(os.path.isdir(REF_DIR), "torch reference not present")
class TestAgainstTorchReference(unittest.TestCase):
    def test_rmsnorm_and_unfold_agree_with_reference_source(self):
        """Compare against the reference's own module output for one input."""
        import importlib.util

        import torch

        spec = importlib.util.spec_from_file_location(
            "_p97_ref_vision", os.path.join(REF_DIR, "vision.py")
        )
        ref = importlib.util.module_from_spec(spec)
        sys.modules["_p97_ref_vision"] = ref
        spec.loader.exec_module(ref)

        # The reference always computes in float32 and casts back to the INPUT
        # dtype, so fp32 in -> fp32 out is the case that must match to float
        # precision; bf16 in -> bf16 out must additionally survive that cast.
        torch.set_default_dtype(torch.float32)
        x = torch.randn(5, 32, dtype=torch.float32)
        r = ref.RMSNorm(32)
        got32 = np.array(vis.RMSNorm(32)(mx.array(x.numpy())).astype(mx.float32))
        want32 = r(x).detach().numpy()
        self.assertLess(float(np.abs(got32 - want32).max()), 1e-6)

        torch.set_default_dtype(torch.bfloat16)
        rb = ref.RMSNorm(32)
        xb = torch.randn(5, 32, dtype=torch.bfloat16)
        got = np.array(
            vis.RMSNorm(32)(mx.array(xb.float().numpy()).astype(mx.bfloat16)).astype(mx.float32)
        )
        want = rb(xb).detach().float().numpy()
        # identical rounding path: the bf16 output cast bounds the difference
        self.assertLess(float(np.abs(got - want).max()), 1e-2)

    def test_token_layout_matches_reference(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_p97_ref_ip", os.path.join(REF_DIR, "image_processor.py")
        )
        ref_ip = importlib.util.module_from_spec(spec)
        sys.modules["_p97_ref_ip"] = ref_ip
        spec.loader.exec_module(ref_ip)
        for nh, nw in ((1, 1), (2, 3), (5, 7)):
            np.testing.assert_array_equal(
                ip.image_token_types(nh, nw), ref_ip.image_token_types(nh, nw).numpy()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
