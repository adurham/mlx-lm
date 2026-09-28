"""bf16 correctness control for the DSv4.1 vision tower.

The tower's residual stream amplifies by ~40-70x over 32 blocks (fp32 max |x|
grows 1.8 -> 1650), so ANY bf16 implementation differs from the fp32 answer by
a lot; that is a property of the network, not of the port. The meaningful
question -- and the criterion the existing DSv4 vision tests use -- is whether
MLX's bf16 result is as close to the exact fp32 answer as torch's own bf16
result is.

For each fixture this reports, at the aligner output and at the final ViT norm:

  err_torch = ||torch_bf16 - torch_fp32|| / ||torch_fp32||
  err_mlx   = ||mlx_bf16   - torch_fp32|| / ||torch_fp32||
  ratio     = err_mlx / err_torch         (bar: <= 1.5)

plus cos(torch_bf16, mlx_bf16) and cos of the fp32 pair for reference.
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dsv41-ws/H"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import mlx.core as mx  # noqa: E402
from p97_trace import fixtures, load_ref_module, run_mlx, run_torch, cos  # noqa: E402
from mlx_lm.models.deepseek_v41 import vision as mv  # noqa: E402

REF_DIR = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/docs/reference")
CKPT = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)


def main() -> int:
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
        wnp = {n: ck.np(n) for n in mv.vision_weight_names(ck.index)}

    tower_bf, cfg = mv.load_vision_tower(CKPT, dtype=mx.bfloat16)
    tower_f32, _ = mv.load_vision_tower(CKPT, dtype=mx.float32)

    out = []
    for i, arr in enumerate(fixtures()):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        rec = {"data": buf.getvalue()}

        t_f32, geo = run_torch(ref_vis, ref_ip, rec, args, wnp, torch.float32)
        t_bf, geo2 = run_torch(ref_vis, ref_ip, rec, args, wnp, torch.bfloat16)
        m_bf, geo3 = run_mlx(tower_bf, cfg, rec, mx.bfloat16)
        m_f32, geo4 = run_mlx(tower_f32, cfg, rec, mx.float32)
        assert geo == geo2 == geo3 == geo4, (geo, geo2, geo3, geo4)

        row = dict(image=i, grid=[geo[0], geo[1]])
        for stage in ("vit_norm", "aligner_out"):
            ref = t_f32[stage]
            n = np.linalg.norm(ref)
            e_t = float(np.linalg.norm(t_bf[stage] - ref) / n)
            e_m = float(np.linalg.norm(m_bf[stage] - ref) / n)
            e_mf = float(np.linalg.norm(m_f32[stage] - ref) / n)
            row[stage] = dict(
                err_torch_bf16=e_t,
                err_mlx_bf16=e_m,
                err_mlx_f32=e_mf,
                ratio_mlx_over_torch=e_m / max(e_t, 1e-30),
                cos_bf16_pair=cos(t_bf[stage], m_bf[stage]),
                cos_f32_pair=cos(t_f32[stage], m_f32[stage]),
                ref_absmax=float(np.abs(ref).max()),
            )
        out.append(row)
        a = row["aligner_out"]
        b = row["vit_norm"]
        print(
            f"img{i} grid {geo[0]}x{geo[1]} | vit_norm: err_t={b['err_torch_bf16']:.3e} "
            f"err_m={b['err_mlx_bf16']:.3e} ratio={b['ratio_mlx_over_torch']:.2f} "
            f"cos_bf={b['cos_bf16_pair']:.8f} cos_f32={b['cos_f32_pair']:.10f}\n"
            f"        | aligner: err_t={a['err_torch_bf16']:.3e} "
            f"err_m={a['err_mlx_bf16']:.3e} err_m_f32={a['err_mlx_f32']:.3e} "
            f"ratio={a['ratio_mlx_over_torch']:.2f} cos_bf={a['cos_bf16_pair']:.8f} "
            f"cos_f32={a['cos_f32_pair']:.10f} refmax={a['ref_absmax']:.1f}"
        )

    worst_ratio = max(r["aligner_out"]["ratio_mlx_over_torch"] for r in out)
    worst_cos_f32 = min(r["aligner_out"]["cos_f32_pair"] for r in out)
    print(json.dumps({
        "n_images": len(out),
        "worst_mlx_over_torch_bf16_error_ratio": worst_ratio,
        "ratio_bar": 1.5,
        "pass_ratio": bool(worst_ratio <= 1.5),
        "worst_fp32_pair_cos_aligner": worst_cos_f32,
        "fp32_parity_bar": 0.9998,
        "pass_fp32_parity": bool(worst_cos_f32 >= 0.9998),
    }, indent=2))
    if os.environ.get("P97_JSON"):
        with open(os.path.expanduser(os.environ["P97_JSON"]), "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
