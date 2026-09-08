# Copyright © 2026 Apple Inc.

"""Numerical-parity + quantization-exclusion tests for the DeepSeek-V4-Flash
vision tower (ViT + Aligner) ported from PyTorch to MLX.

Run with the SUBMODULE on PYTHONPATH so these exercise the working tree rather
than an installed copy:

    PYTHONPATH=/Users/adam.durham/repos/exo/mlx-lm \\
      /Users/adam.durham/repos/exo/.venv/bin/pytest \\
      /Users/adam.durham/repos/exo/mlx-lm/tests/test_deepseek_v4_vision.py -s

`test_running_against_submodule_copy` asserts that actually happened -- a green
run against a stale site-packages copy would be worthless.
"""

import math
import os
import sys
import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812  (universal PyTorch convention)
from mlx_lm.models import deepseek_v4 as dsv4

# The PyTorch ground truth. Kept out of the repo (it is upstream's file, not
# ours); tests that need it skip cleanly when it is absent.
_REF_DIR = os.path.expanduser("~/dsv4_vision_ref/inference")


def _load_torch_reference():
    """Import the reference vision.py by path, without polluting sys.path."""
    import importlib.util

    path = os.path.join(_REF_DIR, "vision.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("_dsv4_ref_vision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REF = _load_torch_reference()


class _RefArgs:
    """The reference modules read flat `vision_*` attributes off an args object.

    Values are the real ones from the HF config.json (top-level keys, there is
    no nested vision_config).
    """

    vision_n_layers = 32
    vision_dim = 1024
    vision_n_heads = 16
    vision_inter_dim = 2816
    vision_patch_size = 14
    vision_rope_theta = 10000.0
    vision_downsample_ratio = 3
    vision_max_n_token = 384
    vision_min_pixels = 147456
    vision_max_wh_ratio = 8
    dim = 4096  # text hidden_size == the Aligner's output width


def _mlx_config(n_layers=_RefArgs.vision_n_layers):
    return dsv4.ModelArgs(
        model_type="deepseek_v4",
        hidden_size=_RefArgs.dim,
        vision_n_layers=n_layers,
        vision_dim=_RefArgs.vision_dim,
        vision_n_heads=_RefArgs.vision_n_heads,
        vision_inter_dim=_RefArgs.vision_inter_dim,
        vision_patch_size=_RefArgs.vision_patch_size,
        vision_rope_theta=_RefArgs.vision_rope_theta,
        vision_downsample_ratio=_RefArgs.vision_downsample_ratio,
    )


def _copy_linear(torch_linear, mlx_linear):
    """torch.nn.Linear and mlx.nn.Linear use the SAME (out, in) weight layout.

    Both compute x @ W.T + b, so the weight transfers verbatim -- no transpose.
    (Verified end-to-end by the parity numbers; a wrong layout here would blow
    the diffs up by orders of magnitude, not hide.)
    """
    mlx_linear.weight = mx.array(
        torch_linear.weight.detach().to(torch.float32).numpy()
    )
    if "bias" in mlx_linear:
        mlx_linear.bias = mx.array(torch_linear.bias.detach().to(torch.float32).numpy())


def _copy_vit(torch_vit, mlx_vit):
    _copy_linear(torch_vit.patch_embed.proj, mlx_vit.patch_embed.proj)
    for tb, mb in zip(torch_vit.blocks, mlx_vit.blocks, strict=True):
        mb.norm1.weight = mx.array(tb.norm1.weight.detach().float().numpy())
        mb.norm2.weight = mx.array(tb.norm2.weight.detach().float().numpy())
        _copy_linear(tb.attn.wqkv, mb.attn.wqkv)
        _copy_linear(tb.attn.wo, mb.attn.wo)
        _copy_linear(tb.mlp.w1, mb.mlp.w1)
        _copy_linear(tb.mlp.w2, mb.mlp.w2)
    mlx_vit.norm.weight = mx.array(torch_vit.norm.weight.detach().float().numpy())


def _copy_aligner(torch_aligner, mlx_aligner):
    _copy_linear(torch_aligner.w1, mlx_aligner.w1)
    _copy_linear(torch_aligner.w2, mlx_aligner.w2)


def _diffs(torch_out, mlx_out):
    a = torch_out.detach().to(torch.float32).numpy()
    b = np.array(mlx_out.astype(mx.float32))
    assert a.shape == b.shape, f"shape mismatch: torch {a.shape} vs mlx {b.shape}"
    d = np.abs(a - b)
    return float(d.max()), float(d.mean())


# Grid shapes. (12, 9) is the aligned case; (10, 7) has BOTH n_h % 3 != 0 and
# n_w % 3 != 0, exercising the aligner's right/bottom zero-pad path.
GRID_SHAPES = [(12, 9), (10, 7)]

# fp32 isolates algorithmic error: at this tolerance any real algorithmic
# divergence (wrong rotary style, transposed unfold, wrong eps) shows up
# immediately, while pure float-reassociation noise does not.
FP32_MAX_TOL = 1e-4

# NOTE on bf16: a fixed absolute tolerance is the WRONG criterion here.
# torch-bf16 and mlx-bf16 round a long chain of identical math at different
# points, so they differ by a few bf16 ULPs of the output scale (~4.7e-2 on a
# tensor whose max |value| is ~4.6, i.e. ~3 ULP) even though neither is
# "wrong". The meaningful question is whether MLX's bf16 result is as close to
# the exact fp32 answer as torch's own bf16 result is -- see
# test_bf16_no_worse_than_torch_bf16. BF16_MAX_TOL is only a loose sanity
# ceiling to catch gross breakage.
#
# Measured max_abs_diff across both GRID_SHAPES: ViT 3.9e-2 to 4.7e-2,
# Aligner 5.9e-3 to 7.8e-3 (see test_bf16_parity output). ViT is the larger
# of the two and drives the ceiling. 8e-2 gives ~1.7x headroom above the
# highest observed ViT value (4.7e-2) -- comfortable margin for legitimate
# bf16-rounding variance across mlx versions/hardware, while still catching
# an order-of-magnitude regression (e.g. a mis-wired eps produces errors in
# the 1e-3-to-3.0 range at the tiny-magnitude scale used elsewhere in this
# file, and an interleaved-vs-half-split rotary mixup diverges by ~6.7 --
# both would blow straight through 8e-2). The old 1e-1 ceiling sat 2.1x-13x
# above reality and would not have caught either.
BF16_MAX_TOL = 8e-2

# MLX's bf16 error may exceed torch's by at most this factor before we treat it
# as a real regression rather than rounding. Measured ratio is ~0.79-0.95
# (MLX is actually slightly MORE accurate than torch here).
BF16_ERROR_RATIO_TOL = 1.5


class TestRunsAgainstSubmodule(unittest.TestCase):
    def test_running_against_submodule_copy(self):
        path = dsv4.__file__
        print(f"\n[provenance] mlx_lm.models.deepseek_v4.__file__ = {path}")
        print(f"[provenance] mlx_lm package             = {sys.modules['mlx_lm'].__file__}")
        self.assertIn(
            "/repos/exo/mlx-lm/",
            path,
            "tests are importing a DIFFERENT mlx_lm copy (site-packages?); "
            "re-run with PYTHONPATH=/Users/adam.durham/repos/exo/mlx-lm",
        )


class TestUnfold(unittest.TestCase):
    """The unfold reimplementation must be BITWISE identical to F.unfold.

    A transposed channel ordering here is silent: shapes match, magnitudes look
    sane, and only the final embeddings are wrong.
    """

    def test_unfold_matches_torch(self):
        rng = np.random.default_rng(0)
        r = 3
        cases = [(6, 9, 4), (9, 6, 5), (3, 3, 2), (12, 15, 7), (30, 21, 1024)]
        for h, w, c in cases:
            with self.subTest(h=h, w=w, c=c):
                x = rng.standard_normal((h, w, c), dtype=np.float32)
                # Reference: (C,H,W) -> unfold -> (L, C*r*r)
                t_chw = torch.from_numpy(x).permute(2, 0, 1).contiguous()
                ref = (
                    F.unfold(t_chw.unsqueeze(0), r, stride=r)
                    .squeeze(0)
                    .transpose(0, 1)
                    .numpy()
                )
                got = np.array(dsv4._vision_unfold(mx.array(x), r))
                self.assertEqual(ref.shape, got.shape)
                n_bad = int((ref != got).sum())
                print(
                    f"[unfold] ({h},{w},{c}) shape={ref.shape} "
                    f"mismatched_elements={n_bad}/{ref.size} "
                    f"bitwise_equal={n_bad == 0}"
                )
                self.assertTrue(
                    np.array_equal(ref, got),
                    f"_vision_unfold differs from F.unfold for ({h},{w},{c})",
                )


@unittest.skipIf(_REF is None, f"torch reference not found under {_REF_DIR}")
class TestVisionParity(unittest.TestCase):
    """End-to-end ViT + Aligner parity against the PyTorch reference."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        # 4 layers rather than the shipped 32: parity is per-block and 4 layers
        # already compounds the residual stream 4x, while keeping a CPU-only
        # torch reference run fast. The full 32-layer depth is covered by
        # test_full_depth_vit_parity below.
        cls.args = _RefArgs()

    def _build(self, n_layers, dtype):
        class A(_RefArgs):
            pass

        A.vision_n_layers = n_layers
        torch.manual_seed(1234)
        t_vit = _REF.ViT(A).to(torch.float32).eval()
        t_align = _REF.Aligner(A).to(torch.float32).eval()
        # RMSNorm weights init to ones; randomize so a dropped norm weight
        # cannot pass by accident.
        for m in list(t_vit.modules()) + list(t_align.modules()):
            if isinstance(m, _REF.RMSNorm):
                with torch.no_grad():
                    m.weight.copy_(torch.randn_like(m.weight) * 0.1 + 1.0)

        cfg = _mlx_config(n_layers)
        m_vit = dsv4.VisionTransformer(cfg)
        m_align = dsv4.VisionAligner(cfg)
        _copy_vit(t_vit, m_vit)
        _copy_aligner(t_align, m_align)

        if dtype == "bf16":
            t_vit = t_vit.to(torch.bfloat16)
            t_align = t_align.to(torch.bfloat16)
            # RMSNorm weight is float32 in the reference by construction; .to()
            # would demote it, so restore float32 to match the checkpoint.
            for m in list(t_vit.modules()) + list(t_align.modules()):
                if isinstance(m, _REF.RMSNorm):
                    m.weight.data = m.weight.data.to(torch.float32)
            m_vit.apply(lambda p: p.astype(mx.bfloat16) if p.ndim > 0 else p)
            m_align.apply(lambda p: p.astype(mx.bfloat16))
            for mod in m_vit.modules():
                if isinstance(mod, dsv4.VisionRMSNorm):
                    mod.weight = mod.weight.astype(mx.float32)
        return t_vit, t_align, m_vit, m_align

    def _run_shape(self, n_h, n_w, n_layers, dtype):
        t_vit, t_align, m_vit, m_align = self._build(n_layers, dtype)
        p = _RefArgs.vision_patch_size
        rng = np.random.default_rng(n_h * 1000 + n_w)
        patches = rng.standard_normal((n_h * n_w, 3, p, p), dtype=np.float32)

        tdt = torch.bfloat16 if dtype == "bf16" else torch.float32
        t_in = torch.from_numpy(patches).to(tdt)
        m_in = mx.array(patches)
        if dtype == "bf16":
            m_in = m_in.astype(mx.bfloat16)

        with torch.inference_mode():
            t_vit_out = t_vit(t_in, n_h, n_w)
            t_align_out = t_align(t_vit_out, n_h, n_w)
        m_vit_out = m_vit(m_in, n_h, n_w)
        m_align_out = m_align(m_vit_out, n_h, n_w)
        mx.eval(m_vit_out, m_align_out)

        vit_max, vit_mean = _diffs(t_vit_out, m_vit_out)
        al_max, al_mean = _diffs(t_align_out, m_align_out)
        pads = f"pad_h={-n_h % 3} pad_w={-n_w % 3}"
        print(
            f"\n[{dtype}] grid=({n_h},{n_w}) layers={n_layers} {pads} "
            f"n_patches={n_h * n_w} -> aligner_tokens={t_align_out.shape[0]}"
        )
        print(f"  ViT     max_abs_diff={vit_max:.6e}  mean_abs_diff={vit_mean:.6e}")
        print(f"  Aligner max_abs_diff={al_max:.6e}  mean_abs_diff={al_mean:.6e}")
        return vit_max, vit_mean, al_max, al_mean

    def test_fp32_parity(self):
        for n_h, n_w in GRID_SHAPES:
            with self.subTest(grid=(n_h, n_w)):
                vit_max, _, al_max, _ = self._run_shape(n_h, n_w, 4, "fp32")
                self.assertLess(vit_max, FP32_MAX_TOL)
                self.assertLess(al_max, FP32_MAX_TOL)

    def test_bf16_parity(self):
        for n_h, n_w in GRID_SHAPES:
            with self.subTest(grid=(n_h, n_w)):
                vit_max, _, al_max, _ = self._run_shape(n_h, n_w, 4, "bf16")
                self.assertLess(vit_max, BF16_MAX_TOL)
                self.assertLess(al_max, BF16_MAX_TOL)

    def test_full_depth_vit_parity(self):
        """All 32 shipped layers, fp32, on the unaligned grid."""
        vit_max, _, al_max, _ = self._run_shape(10, 7, 32, "fp32")
        self.assertLess(vit_max, FP32_MAX_TOL)
        self.assertLess(al_max, FP32_MAX_TOL)

    def test_bf16_no_worse_than_torch_bf16(self):
        """The REAL bf16 criterion: measure both backends against fp32 truth.

        A raw mlx-bf16-vs-torch-bf16 diff conflates "MLX is wrong" with "the two
        backends round a long dependency chain at different points". Comparing
        each against the exact fp32 result separates them: if MLX's error is
        <= torch's own error, MLX is not the inaccurate one and the residual
        gap is pure bf16 rounding.
        """
        for n_h, n_w in GRID_SHAPES:
            with self.subTest(grid=(n_h, n_w)):
                t_vit32, t_al32, m_vit32, m_al32 = self._build(4, "fp32")
                p = _RefArgs.vision_patch_size
                rng = np.random.default_rng(n_h * 1000 + n_w)
                patches = rng.standard_normal((n_h * n_w, 3, p, p), dtype=np.float32)

                # Exact-ish ground truth.
                with torch.inference_mode():
                    g_vit = t_vit32(torch.from_numpy(patches), n_h, n_w)
                    g_al = t_al32(g_vit, n_h, n_w)

                t_vit_b, t_al_b, m_vit_b, m_al_b = self._build(4, "bf16")
                with torch.inference_mode():
                    tb_vit = t_vit_b(torch.from_numpy(patches).bfloat16(), n_h, n_w)
                    tb_al = t_al_b(tb_vit, n_h, n_w)
                mb_vit = m_vit_b(mx.array(patches).astype(mx.bfloat16), n_h, n_w)
                mb_al = m_al_b(mb_vit, n_h, n_w)
                mx.eval(mb_vit, mb_al)

                for name, truth, t_bf, m_bf in (
                    ("ViT", g_vit, tb_vit, mb_vit),
                    ("Aligner", g_al, tb_al, mb_al),
                ):
                    torch_err, torch_err_mean = _diffs(truth, mx.array(
                        t_bf.detach().to(torch.float32).numpy()
                    ))
                    mlx_err, mlx_err_mean = _diffs(truth, m_bf)
                    cross_max, cross_mean = _diffs(t_bf, m_bf)
                    scale = float(
                        np.abs(truth.detach().to(torch.float32).numpy()).max()
                    )
                    ratio = mlx_err / torch_err if torch_err else 0.0
                    # Mean-of-errors ratio alongside the existing max-of-max
                    # ratio: max-of-max is a single noisy order statistic, so
                    # report the mean too rather than only the flattering
                    # extremum. Measured mean_ratio is ~1.004-1.041 here (MLX
                    # is very slightly WORSE than torch on mean error, even
                    # though it is better on max) -- this assertion does NOT
                    # gate on mean_ratio, it only makes the printed number
                    # honest.
                    mean_ratio = (
                        mlx_err_mean / torch_err_mean if torch_err_mean else 0.0
                    )
                    print(
                        f"\n[bf16-truth] grid=({n_h},{n_w}) {name} "
                        f"output_scale={scale:.4f}"
                    )
                    print(
                        f"  torch_bf16 vs fp32 truth: max={torch_err:.6e} "
                        f"mean={torch_err_mean:.6e}"
                    )
                    print(
                        f"  mlx_bf16   vs fp32 truth: max={mlx_err:.6e} "
                        f"mean={mlx_err_mean:.6e}"
                    )
                    print(
                        f"  mlx_bf16   vs torch_bf16: max={cross_max:.6e} "
                        f"mean={cross_mean:.6e}"
                    )
                    print(
                        f"  -> mlx_err / torch_err: max_ratio={ratio:.3f} "
                        f"({'MLX more accurate' if ratio <= 1 else 'MLX less accurate'}) "
                        f"mean_ratio={mean_ratio:.3f} "
                        f"({'MLX more accurate' if mean_ratio <= 1 else 'MLX less accurate'})"
                    )
                    self.assertLess(
                        ratio,
                        BF16_ERROR_RATIO_TOL,
                        f"{name}: MLX bf16 error ({mlx_err:.3e}) is "
                        f"{ratio:.2f}x torch's own bf16 error ({torch_err:.3e}) "
                        "-- that is an algorithmic gap, not rounding",
                    )

    def test_rmsnorm_eps_matches_reference_default(self):
        """Pin eps=1e-6 with an input small enough that eps actually matters.

        At the ~N(0,1) magnitudes the parity tests use, mean(x^2) ~ 1 and eps
        is 6 orders of magnitude below it -- so 1e-6 vs the text model's 1e-20
        is invisible and a mis-wired eps passes every other test in this file
        (verified by sabotage). Driving the input down to ~1e-4 puts mean(x^2)
        ~1e-8, BELOW eps, where the two values differ by ~2x in the output.
        """
        dim = 128
        rng = np.random.default_rng(11)
        w = rng.standard_normal(dim).astype(np.float32) * 0.1 + 1.0

        t_norm = _REF.RMSNorm(dim)
        with torch.no_grad():
            t_norm.weight.copy_(torch.from_numpy(w))
        self.assertEqual(
            t_norm.eps, 1e-6, "reference RMSNorm default eps is not 1e-6"
        )

        m_norm = dsv4.VisionRMSNorm(dim)
        m_norm.weight = mx.array(w)
        self.assertEqual(
            m_norm.eps,
            1e-6,
            "VisionRMSNorm must use the reference default eps=1e-6, NOT the "
            "text model's rms_norm_eps=1e-20",
        )

        for scale, label in ((1.0, "unit"), (1e-4, "tiny (eps-sensitive)")):
            with self.subTest(scale=label):
                x = (rng.standard_normal((16, dim)).astype(np.float32)) * scale
                ref = t_norm(torch.from_numpy(x))
                got = m_norm(mx.array(x))
                max_d, mean_d = _diffs(ref, got)

                # What a 1e-20 eps would produce, for contrast.
                xf = x.astype(np.float64)
                wrong = (
                    w
                    * xf
                    / np.sqrt((xf**2).mean(-1, keepdims=True) + 1e-20)
                )
                wrong_delta = float(
                    np.abs(wrong - np.array(got.astype(mx.float32))).max()
                )
                print(
                    f"[rmsnorm] scale={label}: max={max_d:.6e} mean={mean_d:.6e} "
                    f"| delta vs eps=1e-20 variant: {wrong_delta:.6e}"
                )
                self.assertLess(max_d, 1e-5)
                if scale < 1e-3:
                    self.assertGreater(
                        wrong_delta,
                        1e-2,
                        "this input is not eps-sensitive, so the test cannot "
                        "distinguish 1e-6 from 1e-20",
                    )

    def test_full_model_parity_at_eps_sensitive_magnitude(self):
        """End-to-end parity through a CONSTRUCTED VisionTransformer at input
        magnitude ~1e-4, where the first block's norm1 operates on
        mean(x^2) ~ 1e-8 -- well below eps=1e-6.

        test_rmsnorm_eps_matches_reference_default above pins eps only on a
        FRESH STANDALONE VisionRMSNorm; it says nothing about the eps of the
        norm instances actually living inside a constructed VisionTransformer.
        A mis-wired eps=1e-20 at any of the three VisionBlock/
        VisionTransformer call sites (leaving the constructor default alone)
        diverges from the torch reference's eps=1e-6 output at THIS input
        magnitude -- unlike the N(0,1)-scale inputs used by test_fp32_parity,
        where mean(x^2) ~ 1 swamps eps and the divergence is invisible (see
        the BF16_MAX_TOL / FP32_MAX_TOL comments above). This is a numerical
        trip-wire on the constructed model, not merely a metadata check.
        """
        for n_h, n_w in GRID_SHAPES:
            with self.subTest(grid=(n_h, n_w)):
                t_vit, t_align, m_vit, m_align = self._build(4, "fp32")
                p = _RefArgs.vision_patch_size
                rng = np.random.default_rng(n_h * 1000 + n_w)
                patches = (
                    rng.standard_normal((n_h * n_w, 3, p, p)).astype(np.float32)
                    * 1e-4
                )

                t_in = torch.from_numpy(patches)
                m_in = mx.array(patches)
                with torch.inference_mode():
                    t_vit_out = t_vit(t_in, n_h, n_w)
                    t_align_out = t_align(t_vit_out, n_h, n_w)
                m_vit_out = m_vit(m_in, n_h, n_w)
                m_align_out = m_align(m_vit_out, n_h, n_w)
                mx.eval(m_vit_out, m_align_out)

                vit_max, vit_mean = _diffs(t_vit_out, m_vit_out)
                al_max, al_mean = _diffs(t_align_out, m_align_out)
                print(
                    f"\n[tiny-magnitude parity] grid=({n_h},{n_w}) "
                    f"ViT max={vit_max:.6e} mean={vit_mean:.6e}  "
                    f"Aligner max={al_max:.6e} mean={al_mean:.6e}"
                )
                self.assertLess(vit_max, FP32_MAX_TOL)
                self.assertLess(al_max, FP32_MAX_TOL)

    def test_all_norms_in_constructed_tower_use_reference_eps(self):
        """Walk a CONSTRUCTED VisionTransformer and assert eps==1e-6 on EVERY
        VisionRMSNorm reachable from it -- not just the constructor default.

        test_rmsnorm_eps_matches_reference_default only pins the DEFAULT on a
        fresh standalone VisionRMSNorm(dim); it says nothing about what eps
        the norm instances actually wired into a real model got. This test
        walks the module tree programmatically (named_modules(), not three
        hardcoded attribute paths) so norm1/norm2 on every block plus the
        final top-level norm are all covered, and any future added norm is
        covered automatically too.
        """
        cfg = _mlx_config(n_layers=6)
        vit = dsv4.VisionTransformer(cfg)

        norms = {
            name: mod
            for name, mod in vit.named_modules()
            if isinstance(mod, dsv4.VisionRMSNorm)
        }
        # Sanity: this must actually enumerate something, and specifically
        # norm1/norm2 for every block plus the top-level norm, or the walk
        # itself is broken and the assertion below would vacuously pass.
        expected_names = {"norm"} | {
            f"blocks.{i}.{which}"
            for i in range(cfg.vision_n_layers)
            for which in ("norm1", "norm2")
        }
        self.assertEqual(
            set(norms.keys()),
            expected_names,
            "module walk did not find exactly the expected VisionRMSNorm "
            "instances -- the walk itself may be broken",
        )
        print(f"\n[eps-wiring] {len(norms)} VisionRMSNorm instances found in tower")
        for name, mod in sorted(norms.items()):
            with self.subTest(norm=name):
                self.assertEqual(
                    mod.eps,
                    1e-6,
                    f"{name}.eps = {mod.eps!r}, must be the reference default "
                    "1e-6, NOT the text model's rms_norm_eps=1e-20",
                )
        print(
            "[eps-wiring] all eps == 1e-6: "
            f"{sorted(name for name in norms)}"
        )

    def test_rope_tables_match_reference(self):
        dim = _RefArgs.vision_dim // _RefArgs.vision_n_heads // 2
        for n_h, n_w in GRID_SHAPES:
            t_cos, t_sin = _REF.get_vision_cos_sin(
                n_h, n_w, dim, _RefArgs.vision_rope_theta
            )
            m_cos, m_sin = dsv4._vision_cos_sin(
                n_h, n_w, dim, _RefArgs.vision_rope_theta
            )
            self.assertEqual(tuple(t_cos.shape), tuple(m_cos.shape))
            cmax, cmean = _diffs(t_cos, m_cos)
            smax, smean = _diffs(t_sin, m_sin)
            print(
                f"[rope] grid=({n_h},{n_w}) shape={tuple(t_cos.shape)} "
                f"cos max={cmax:.3e} mean={cmean:.3e} | "
                f"sin max={smax:.3e} mean={smean:.3e}"
            )
            self.assertLess(cmax, 1e-6)
            self.assertLess(smax, 1e-6)

    def test_apply_rotary_is_half_split_not_interleaved(self):
        """Negative control: interleaved (even/odd) rotary must NOT match.

        Guards the single easiest-to-get-silently-wrong detail in the port.
        """
        rng = np.random.default_rng(7)
        n, heads, head_dim = 21, _RefArgs.vision_n_heads, 64
        x = rng.standard_normal((n, heads, head_dim), dtype=np.float32)
        cos, sin = dsv4._vision_cos_sin(
            3, 7, head_dim // 2, _RefArgs.vision_rope_theta
        )
        ref = _REF.apply_rotary(
            torch.from_numpy(x),
            torch.from_numpy(np.array(cos)),
            torch.from_numpy(np.array(sin)),
        )
        got = dsv4._vision_apply_rotary(mx.array(x), cos, sin)
        max_d, mean_d = _diffs(ref, got)
        print(f"[rotary] half-split max={max_d:.3e} mean={mean_d:.3e}")
        self.assertLess(max_d, 1e-6)

        # Interleaved variant of the SAME input -- must differ materially.
        xi = x.reshape(n, heads, head_dim // 2, 2)
        c = np.array(cos)[:, :, :, None].squeeze(-1)
        x1, x2 = xi[..., 0], xi[..., 1]
        inter = np.stack(
            [x1 * c - x2 * np.array(sin), x2 * c + x1 * np.array(sin)], axis=-1
        ).reshape(n, heads, head_dim)
        inter_delta = float(np.abs(inter - np.array(got)).max())
        print(f"[rotary] interleaved-vs-half-split max delta={inter_delta:.3e}")
        self.assertGreater(
            inter_delta,
            1e-3,
            "interleaved and half-split rotary are indistinguishable here -- "
            "this negative control has no teeth",
        )


class TestQuantizationExclusion(unittest.TestCase):
    """make_quantization_config() must never quantize the bf16 vision tower,
    and must not perturb any pre-existing text-side assignment."""

    # Captured from the PRE-change function (git show HEAD~:...) on the same
    # tiny text-only config. This is the regression guard: if any of these
    # drift, the vision exclusion changed text-side behaviour.
    EXPECTED_TEXT_SIDE = {
        "mxfp4_suffixes": ("gate_proj", "down_proj", "up_proj"),
        "top_level": {"group_size": 64, "bits": 8, "mode": "affine"},
    }

    @staticmethod
    def _tiny(vision_n_layers=0):
        return dsv4.ModelArgs(
            num_hidden_layers=4,
            n_routed_experts=4,
            num_attention_heads=8,
            hidden_size=256,
            vocab_size=512,
            moe_intermediate_size=64,
            intermediate_size=128,
            q_lora_rank=64,
            o_lora_rank=64,
            o_groups=2,
            head_dim=64,
            qk_rope_head_dim=16,
            index_n_heads=8,
            index_head_dim=16,
            num_hash_layers=1,
            compress_ratios=[0, 4, 128, 0],
            vision_n_layers=vision_n_layers,
            vision_dim=64,
            vision_n_heads=4,
            vision_inter_dim=128,
            vision_patch_size=4,
            vision_downsample_ratio=3,
        )

    def _text_side_view(self, cfg):
        """Everything the config says about NON-vision keys."""
        return {
            k: v
            for k, v in cfg.items()
            if k in ("group_size", "bits", "mode")
            or not dsv4._is_unquantized_key(k)
        }

    def test_vision_and_aligner_are_never_quantized(self):
        model = dsv4.Model(self._tiny(vision_n_layers=3))
        cfg = dsv4.make_quantization_config(model)

        import mlx.nn as nn
        from mlx.utils import tree_flatten

        all_keys = [
            k
            for k, _ in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
        ]
        vision_keys = [k for k in all_keys if dsv4._is_unquantized_key(k)]
        self.assertTrue(vision_keys, "test built no vision.*/aligner.* modules")
        print(f"\n[quant] {len(vision_keys)} vision/aligner leaf modules built")

        # (a) The specific names the old substring test would have caught.
        swept_by_old_rule = [k for k in vision_keys if ".attn.w" in k]
        self.assertTrue(
            swept_by_old_rule,
            "no vision .attn.w* keys present -- the regression this guards "
            "against is not actually reproduced by this fixture",
        )
        print(
            f"[quant] {len(swept_by_old_rule)} of them match the old "
            f'".attn.w" rule, e.g. {swept_by_old_rule[0]}'
        )

        # (b) Every vision/aligner key is present-and-False (explicit opt-out).
        for k in vision_keys:
            with self.subTest(key=k):
                self.assertIn(
                    k, cfg, f"{k} would fall through to the affine catch-all"
                )
                self.assertIs(cfg[k], False, f"{k} got a quant override: {cfg[k]!r}")

        # (c) No quantizing override leaked in under any vision/aligner name.
        leaked = {
            k: v for k, v in cfg.items() if dsv4._is_unquantized_key(k) and v is not False
        }
        self.assertEqual(leaked, {}, f"vision keys received overrides: {leaked}")

        # (d) Sentinels are excluded by predicate (they are bare parameters, so
        # nn.quantize cannot reach them, but the predicate must still say no).
        for name in ("image_start", "image_end", "image_newline", "image_pad"):
            self.assertTrue(hasattr(model, name), f"{name} not constructed")
            self.assertTrue(dsv4._is_unquantized_key(name))
            self.assertNotIn(
                name,
                [k for k, v in cfg.items() if v is not False],
                f"{name} received a quantization override",
            )

    def test_nn_quantize_actually_skips_vision(self):
        """The real path: feed the config through nn.quantize's predicate.

        Asserting on the dict alone would not prove `False` is honoured.
        """
        import mlx.nn as nn

        model = dsv4.Model(self._tiny(vision_n_layers=3))
        cfg = dsv4.make_quantization_config(model)

        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            if p in cfg:
                return cfg[p]
            return True  # catch-all default, as mlx_lm.utils does

        nn.quantize(
            model,
            group_size=cfg["group_size"],
            bits=cfg["bits"],
            mode=cfg["mode"],
            class_predicate=class_predicate,
        )

        quantized, kept = [], []
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
                quantized.append(name)
            elif isinstance(mod, (nn.Linear, nn.Embedding)):
                kept.append(name)

        bad = [n for n in quantized if dsv4._is_unquantized_key(n)]
        vision_kept = [n for n in kept if dsv4._is_unquantized_key(n)]
        print(
            f"[quant] after nn.quantize: {len(quantized)} quantized, "
            f"{len(vision_kept)} vision/aligner left unquantized, "
            f"{len(bad)} vision/aligner wrongly quantized"
        )
        self.assertEqual(bad, [], f"vision modules were quantized: {bad}")
        self.assertTrue(vision_kept, "no vision modules survived as nn.Linear")

        # dtype proof: an untouched Linear still holds float weights.
        self.assertEqual(model.vision.blocks[0].attn.wqkv.weight.dtype, mx.float32)
        self.assertIsInstance(model.vision.blocks[0].attn.wqkv, nn.Linear)
        self.assertNotIsInstance(
            model.vision.blocks[0].attn.wqkv, nn.QuantizedLinear
        )

    def test_text_side_assignments_unchanged(self):
        """REGRESSION GUARD: text-only model must produce the exact same config
        it produced before the vision exclusion existed."""
        model = dsv4.Model(self._tiny(vision_n_layers=0))
        cfg = dsv4.make_quantization_config(model)

        # No vision built -> no False entries at all; byte-identical shape to
        # the pre-change output.
        self.assertEqual(
            [k for k, v in cfg.items() if v is False],
            [],
            "text-only model somehow produced unquantized-opt-out entries",
        )
        self.assertEqual(
            {k: cfg[k] for k in ("group_size", "bits", "mode")},
            self.EXPECTED_TEXT_SIDE["top_level"],
        )

        experts = {k: v for k, v in cfg.items() if ".ffn.switch_mlp." in k}
        shared = {k: v for k, v in cfg.items() if ".ffn.shared_experts." in k}
        attn = {
            k: v
            for k, v in cfg.items()
            if isinstance(k, str)
            and (".attn.w" in k or ".attn.indexer.wq" in k)
        }
        self.assertTrue(experts and shared and attn)
        for k, v in experts.items():
            self.assertEqual(v, {"group_size": 32, "bits": 4, "mode": "mxfp4"}, k)
            self.assertTrue(k.endswith(self.EXPECTED_TEXT_SIDE["mxfp4_suffixes"]))
        for k, v in shared.items():
            self.assertEqual(v, {"group_size": 32, "bits": 8, "mode": "mxfp8"}, k)
        for k, v in attn.items():
            self.assertEqual(v, {"group_size": 32, "bits": 8, "mode": "mxfp8"}, k)
        print(
            f"[quant] text-only: {len(experts)} mxfp4 experts, "
            f"{len(shared)} mxfp8 shared, {len(attn)} mxfp8 attn, "
            f"top-level={self.EXPECTED_TEXT_SIDE['top_level']}"
        )

    def test_text_side_identical_with_and_without_vision(self):
        """Adding the vision tower must not alter ANY text-side entry."""
        cfg_text = dsv4.make_quantization_config(dsv4.Model(self._tiny(0)))
        cfg_vis = dsv4.make_quantization_config(dsv4.Model(self._tiny(3)))
        a, b = self._text_side_view(cfg_text), self._text_side_view(cfg_vis)
        self.assertEqual(
            a,
            b,
            "text-side quantization assignments changed when the vision "
            "tower was present",
        )
        print(f"[quant] text-side entries identical with/without vision: {len(a)} keys")


class TestVisionModelWiring(unittest.TestCase):
    def test_text_only_config_builds_no_vision_tower(self):
        model = dsv4.Model(TestQuantizationExclusion._tiny(vision_n_layers=0))
        self.assertFalse(hasattr(model, "vision"))
        self.assertFalse(hasattr(model, "aligner"))

    def test_encode_image_end_to_end_shapes(self):
        cfg = TestQuantizationExclusion._tiny(vision_n_layers=2)
        model = dsv4.Model(cfg)
        n_h, n_w = 5, 4  # both indivisible by 3
        p = cfg.vision_patch_size
        patches = mx.random.normal((n_h * n_w, 3, p, p))
        out = model.encode_image(patches, n_h, n_w)
        mx.eval(out)
        exp_tokens = math.ceil(n_h / 3) * math.ceil(n_w / 3)
        print(
            f"\n[wiring] encode_image({n_h}x{n_w}) -> {out.shape} "
            f"(expected ({exp_tokens}, {cfg.hidden_size}))"
        )
        self.assertEqual(out.shape, (exp_tokens, cfg.hidden_size))


if __name__ == "__main__":
    unittest.main()
