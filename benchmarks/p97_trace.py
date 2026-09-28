"""Stage-by-stage localization of MLX-vs-torch divergence in the DSv4.1 ViT.

Three runs of the same pipeline, all on checkpoint weights:

  fp32   -- torch reference in fp32, MLX tower in fp32  (algorithmic check)
  bf16   -- torch reference in bf16, MLX tower in bf16  (shipped numerics)
  cross  -- torch bf16 vs MLX fp32                      (whose rounding differs)

Reports cosine and max-abs per stage (patch_embed, selected blocks, final norm,
aligner w1/gelu/out) so the first diverging stage is visible, plus the
"MLX is no worse than torch" criterion: ||mlx_bf16 - fp32|| / ||torch_bf16 - fp32||.

Env: P97_WHICH=0..2 selects the fixture; P97_TRACE_MLX_F32=1 runs the tower fp32.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys

import numpy as np

REF_DIR = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/docs/reference")
PKG = os.path.expanduser("~/dsv41-ws/H")
CKPT = os.path.expanduser("~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
sys.path.insert(0, PKG)

import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm.models.deepseek_v41 import image_processor as mip  # noqa: E402
from mlx_lm.models.deepseek_v41 import vision as mv  # noqa: E402


def load_ref_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def fixtures():
    rng = np.random.default_rng(1234)
    flat = np.full((300, 500, 3), 96, np.uint8)
    grad = np.clip(
        (np.linspace(0, 255, 500, dtype=np.float32)[None, :, None]
         + np.linspace(0, 255, 300, dtype=np.float32)[:, None, None]) / 2,
        0, 255,
    ).astype(np.uint8).repeat(3, axis=2)
    odd = rng.integers(0, 256, (141, 253, 3), dtype=np.uint8)
    return [flat, grad, odd]


def run_torch(ref_vis, ref_ip, rec, args, wnp, torch_dtype):
    """Reference pipeline in one dtype; returns per-stage numpy arrays."""
    torch.set_default_dtype(torch_dtype)
    tv, ta = ref_vis.ViT(args), ref_vis.Aligner(args)
    tv.load_state_dict(
        {k[len("vision."):]: torch.from_numpy(v).to(torch_dtype)
         for k, v in wnp.items() if k.startswith("vision.")},
        strict=True,
    )
    ta.load_state_dict(
        {k[len("aligner."):]: torch.from_numpy(v).to(torch_dtype)
         for k, v in wnp.items() if k.startswith("aligner.")},
        strict=True,
    )
    tv.eval(), ta.eval()
    patches, nh, nw, nlh, nlw = ref_ip.load_image(rec, args)
    # the reference's load_image always emits bf16-precision patches; a fp32 run
    # therefore widens exactly the values bf16 would have rounded to.
    with torch.inference_mode():
        x = tv.patch_embed(patches.to(torch_dtype))
        c, s = ref_vis.get_vision_cos_sin(nh, nw, tv.rope_dim, tv.rope_theta)
        stages = {"patch_embed": x.float().numpy()}
        for idx, blk in enumerate(tv.blocks):
            x = x + blk.attn(blk.norm1(x), c, s)
            x = x + blk.mlp(blk.norm2(x))
            if idx in (0, 1, 2, 7, 15, 31):
                stages[f"block{idx}"] = x.float().numpy()
        xn = tv.norm(x)
        stages["vit_norm"] = xn.float().numpy()
        r = ta.downsample_ratio
        flat = TF.pad(xn.view(nh, nw, -1).permute(2, 0, 1), (0, -nw % r, 0, -nh % r))
        flat = TF.unfold(flat.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        a1 = ta.w1(flat)
        stages["aligner_w1"] = a1.float().numpy()
        ag = TF.gelu(a1)
        stages["aligner_gelu"] = ag.float().numpy()
        stages["aligner_out"] = ta.w2(ag).float().numpy()
    return stages, (nh, nw, nlh, nlw)


def run_mlx(tower, cfg, rec, mlx_dtype):
    patches, nh, nw, nlh, nlw = mip.load_image(rec, cfg.preprocess)
    x = tower.vision.patch_embed(mip.patches_to_mlx(patches).astype(mlx_dtype))
    c, s = mv.get_vision_cos_sin(nh, nw, tower.vision.rope_dim, tower.vision.rope_theta)
    if mlx_dtype == mx.float32:
        c, s = c.astype(mx.float32), s.astype(mx.float32)
    mx.eval(x)
    stages = {"patch_embed": np.array(x.astype(mx.float32))}
    for idx, blk in enumerate(tower.vision.blocks):
        x = x + blk.attn(blk.norm1(x), c, s)
        x = x + blk.mlp(blk.norm2(x))
        if idx in (0, 1, 2, 7, 15, 31):
            mx.eval(x)
            stages[f"block{idx}"] = np.array(x.astype(mx.float32))
    xn = tower.vision.norm(x)
    stages["vit_norm"] = np.array(xn.astype(mx.float32))
    r = tower.aligner.downsample_ratio
    flat = xn.reshape(nh, nw, -1)
    ph, pw = -nh % r, -nw % r
    if ph or pw:
        flat = mx.pad(flat, [(0, ph), (0, pw), (0, 0)])
    flat = mv._unfold(flat, r)
    a1 = tower.aligner.w1(flat)
    ag = nn.gelu(a1)
    out = tower.aligner.w2(ag)
    mx.eval(out)
    stages["aligner_w1"] = np.array(a1.astype(mx.float32))
    stages["aligner_gelu"] = np.array(ag.astype(mx.float32))
    stages["aligner_out"] = np.array(out.astype(mx.float32))
    return stages, (nh, nw, nlh, nlw)


STAGE_ORDER = ["patch_embed", "block0", "block1", "block2", "block7", "block15", "block31",
               "vit_norm", "aligner_w1", "aligner_gelu", "aligner_out"]


def main() -> int:
    which = int(os.environ.get("P97_WHICH", "0"))
    mlx_f32 = os.environ.get("P97_TRACE_MLX_F32") == "1"
    torch_f32 = os.environ.get("P97_TRACE_F32") == "1"

    ref_vis = load_ref_module("vision", os.path.join(REF_DIR, "vision.py"))
    ref_ip = load_ref_module("image_processor", os.path.join(REF_DIR, "image_processor.py"))

    args = type("RA", (), {})()
    for k, v in dict(vision_patch_size=14, vision_dim=1024, vision_n_heads=16,
                     vision_inter_dim=2816, vision_n_layers=32, vision_rope_theta=10000.0,
                     vision_downsample_ratio=3, vision_max_n_token=1024,
                     vision_min_pixels=295936, vision_max_wh_ratio=None, dim=5120).items():
        setattr(args, k, v)

    from mlx_lm.models.exl3.loader import Exl3Checkpoint

    with Exl3Checkpoint(CKPT) as ck:
        names = mv.vision_weight_names(ck.index)
        wnp = {n: ck.np(n) for n in names}
    tower, cfg = mv.load_vision_tower(
        CKPT, dtype=mx.float32 if mlx_f32 else mx.bfloat16
    )

    rec_image = fixtures()[which]
    buf = io.BytesIO()
    Image.fromarray(rec_image).save(buf, format="PNG")
    rec = {"data": buf.getvalue()}

    t_st, geo = run_torch(ref_vis, ref_ip, rec, args, wnp,
                          torch.float32 if torch_f32 else torch.bfloat16)
    m_st, m_geo = run_mlx(tower, cfg, rec, mx.float32 if mlx_f32 else mx.bfloat16)
    assert geo == m_geo, f"geometry mismatch: torch {geo} vs mlx {m_geo}"

    label = f"torch={'fp32' if torch_f32 else 'bf16'} mlx={'fp32' if mlx_f32 else 'bf16'}"
    print(f"image {which} grid {geo[0]}x{geo[1]}  [{label}]")
    print(f"{'stage':>14} {'cos':>15} {'maxabs':>11} {'rel':>11} {'ref_max':>11}")
    for k in STAGE_ORDER:
        a, b = m_st[k], t_st[k]
        d = np.abs(a - b)
        rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
        print(f"{k:>14} {cos(a, b):>15.10f} {d.max():>11.4e} {rel:>11.4e} {np.abs(b).max():>11.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
