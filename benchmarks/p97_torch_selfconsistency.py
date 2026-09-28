"""Is torch's own bf16 vision tower self-consistent at the 0.9998 cosine bar?

The DSv4.1 ViT's residual stream grows from max|x| ~ 1.8 to ~1650 over its 32
blocks, so bf16 rounding differences between two kernels amplify enormously. If
torch-bf16 itself (same weights, same patches, two different kernel paths)
cannot hit cos >= 0.9998 against another torch-bf16 run, then no independent MLX
implementation can either -- the bar would be unsatisfiable at bf16 for ANY
implementation, and the meaningful criteria become fp32 parity plus
"MLX-bf16 no worse than torch-bf16".

Paths compared, all at the aligner output of the SAME reference code:
  cpu       -- torch bf16 on CPU (default)
  mps       -- torch bf16 on MPS (different kernels)
  cpu1t     -- torch bf16 on CPU with one thread (different reduction order)
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dsv41-ws/H"))

import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from p97_trace import fixtures, load_ref_module, cos  # noqa: E402

REF_DIR = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/docs/reference")
CKPT = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)


def build(ref_vis, args, wnp, torch_dtype):
    torch.set_default_dtype(torch_dtype)
    tv, ta = ref_vis.ViT(args), ref_vis.Aligner(args)
    tv.load_state_dict(
        {k[len("vision."):]: torch.from_numpy(v).to(torch_dtype)
         for k, v in wnp.items() if k.startswith("vision.")}, strict=True)
    ta.load_state_dict(
        {k[len("aligner."):]: torch.from_numpy(v).to(torch_dtype)
         for k, v in wnp.items() if k.startswith("aligner.")}, strict=True)
    tv.eval(), ta.eval()
    return tv, ta


def encode(tv, ta, ref_vis, ref_ip, rec, args, device, torch_dtype, n_threads=None):
    if n_threads is not None:
        torch.set_num_threads(n_threads)
    tv.to(device)
    ta.to(device)
    patches, nh, nw, nlh, nlw = ref_ip.load_image(rec, args)
    with torch.inference_mode():
        p = patches.to(device=device, dtype=torch_dtype)
        x = tv.patch_embed(p)
        c, s = ref_vis.get_vision_cos_sin(nh, nw, tv.rope_dim, tv.rope_theta)
        c, s = c.to(device), s.to(device)
        for blk in tv.blocks:
            x = x + blk.attn(blk.norm1(x), c, s)
            x = x + blk.mlp(blk.norm2(x))
        xn = tv.norm(x)
        r = ta.downsample_ratio
        flat = TF.pad(xn.view(nh, nw, -1).permute(2, 0, 1), (0, -nw % r, 0, -nh % r))
        flat = TF.unfold(flat.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        out = ta.w2(TF.gelu(ta.w1(flat)))
    return out.float().cpu().numpy()


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
        wnp = {n: ck.np(n) for n in
               sorted(k for k in ck.index if k.startswith(("vision.", "aligner.")))}

    arr = fixtures()[0]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    rec = {"data": buf.getvalue()}

    tv, ta = build(ref_vis, args, wnp, torch.bfloat16)
    a = encode(tv, ta, ref_vis, ref_ip, rec, args, "cpu", torch.bfloat16, n_threads=8)
    b = encode(tv, ta, ref_vis, ref_ip, rec, args, "cpu", torch.bfloat16, n_threads=1)
    mps_ok = torch.backends.mps.is_available()
    c = None
    if mps_ok:
        try:
            c = encode(tv, ta, ref_vis, ref_ip, rec, args, "mps", torch.bfloat16)
        except Exception as exc:  # noqa: BLE001
            print("mps failed:", exc)
            mps_ok = False

    def rep(x, y, name):
        rel = float(np.linalg.norm(x - y) / np.linalg.norm(y))
        print(f"{name:>22}: cos {cos(x, y):.10f} rel {rel:.4e} maxabs {np.abs(x - y).max():.4e}")

    rep(a, b, "torch bf16 cpu vs cpu1t")
    if c is not None:
        rep(a, c, "torch bf16 cpu vs mps")
        rep(b, c, "torch bf16 cpu1t vs mps")

    # fp32 self-consistency for contrast
    tv32, ta32 = build(ref_vis, args, wnp, torch.float32)
    a32 = encode(tv32, ta32, ref_vis, ref_ip, rec, args, "cpu", torch.float32, n_threads=8)
    b32 = encode(tv32, ta32, ref_vis, ref_ip, rec, args, "cpu", torch.float32, n_threads=1)
    rep(a32, b32, "torch fp32 cpu vs cpu1t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
