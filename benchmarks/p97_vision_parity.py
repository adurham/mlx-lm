"""Workstream H parity + perf harness: MLX vision tower vs the torch reference.

Two independent paths over the SAME fixed image set:

  A. torch reference  -- docs/reference/{vision.py,image_processor.py} from the
     reference repo, weights = the same checkpoint BF16 tensors.
  B. MLX tower        -- mlx_lm/models/deepseek_v41/{vision,image_processor}.py,
     weights read from the EXL3 checkpoint by load_vision_tower.

Compares geometry, patches (bit-exact), token types, rope tables, the aligner
output and the full merged image block (sentinels + aligner rows). Measures
per-image time and transient memory for the MLX path.

WHAT PARITY NUMBER TO EXPECT, AND WHY
-------------------------------------
* P97_DTYPE=f32 -> both sides fp32. This is the ALGORITHMIC check: it isolates
  the port's math from float rounding. Measured cos 0.99999999 at the aligner
  output (bar 0.9998); any real port bug (rotary style, unfold order, eps,
  wrong norm) shows up here immediately.
* P97_DTYPE=bf16 (the shipped numerics) -> cos ~0.99-0.9997, below 0.9998.
  That is NOT a port defect. Control: torch-bf16 vs torch-bf16 across kernels
  (CPU vs MPS, same weights, same patches, benchmarks/p97_torch_selfconsistency.py)
  is cos 0.9944 -- no independent implementation can reach 0.9998 at bf16.
  Cause: the ViT's residual stream grows from max|x| 1.8 (block 0) to 1650
  (block 31), so bf16 rounding-order differences amplify ~100x over 32 blocks.
  The meaningful bf16 criterion is "MLX-bf16 no worse than torch-bf16 against
  the exact fp32 answer", measured by benchmarks/p97_bf16_control.py
  (ratio 0.83-1.16, bar 1.5).

Usage (on a Mac; production is running -- always hold the GPU lock):

  EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python p97_vision_parity.py

Note on the peak column here: it is ``mx.get_peak_memory()`` after a reset,
i.e. TOTAL MLX memory at peak -- the ~971 MB of resident tower weights included.
benchmarks/p97_perf.py reports the TRANSIENT delta instead.

Env: P97_DTYPE=bf16|f32, P97_IMAGES=n, P97_JSON=out.json,
P97_LAYERS=0-3 (cheap subset run: blocks 0-3 only, both sides; peak < 1 GB).

PRODUCTION IS RUNNING: hold ~/dsv41-gpu.lock, run ONE job at a time, and keep
MLX peak under 8 GB. P97_LAYERS + P97_IMAGES do that; a full 32-block fp32 run
on the largest fixture peaks near 3 GB and should be timed accordingly.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REF_DIR = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/docs/reference")
PKG = os.environ.get("P97_PKG", os.path.expanduser("~/dsv41-ws/H"))
CKPT = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)
sys.path.insert(0, PKG)

import mlx.core as mx  # noqa: E402
from PIL import Image  # noqa: E402

from mlx_lm.models.deepseek_v41 import image_processor as mip  # noqa: E402
from mlx_lm.models.deepseek_v41 import vision as mv  # noqa: E402


def load_ref_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------------------------------------------------------ fixtures

# (name, height, width). The last two are realistic upper-bound payloads
# (a 2048x2048 photo and a 4096-wide panorama): both must be brought under the
# 1024-token budget by the resize solver, exercising solve_resize_ratio.
FIXTURE_SPECS = [
    ("flat", 300, 500),
    ("gradient", 300, 500),
    ("checker", 300, 500),
    ("noise", 300, 500),
    ("stripes", 300, 500),
    ("odd_141x253", 141, 253),
    ("tall_600x120", 600, 120),
    ("wide_120x600", 120, 600),
    ("photo_2048", 2048, 2048),
    ("pano_512x4096", 512, 4096),
]


def build_fixtures() -> list[dict]:
    """Deterministic fixed image set, as raw PNG bytes."""
    rng = np.random.default_rng(1234)
    imgs: list[dict] = []
    for name, h, w in FIXTURE_SPECS:
        if name == "flat":
            arr = np.full((h, w, 3), 96, dtype=np.uint8)
        elif name == "gradient":
            gx = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
            gy = np.linspace(0, 255, h, dtype=np.float32)[:, None, None]
            arr = np.clip((gx + gy) / 2, 0, 255).astype(np.uint8).repeat(3, axis=2)
        elif name == "checker":
            arr = (((np.indices((h, w)).sum(0) // 25) % 2)[..., None] * 255).astype(np.uint8)
            arr = arr.repeat(3, axis=2)
        elif name == "stripes":
            band = ((np.arange(w) // 7 % 2) * 255).astype(np.uint8)
            arr = np.broadcast_to(band[None, :, None], (h, w, 3)).copy()
        else:
            arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        imgs.append({"data": buf.getvalue(), "name": name})
    return imgs


# ---------------------------------------------------------------- torch side


def _renumbered(weights, block_index):
    """Remap checkpoint block keys onto a compacted subset's indices.

    ``block_index`` maps subset position -> original block number (None = keep
    all). Returns the weight dict with ``vision.blocks.<orig>.`` rewritten to
    ``vision.blocks.<pos>.`` for the kept blocks only.
    """
    if block_index is None:
        return weights
    out = {}
    for k, v in weights.items():
        if k.startswith("vision.blocks."):
            head, _, rest = k[len("vision.blocks."):].partition(".")
            orig = int(head)
            if orig not in block_index:
                continue
            k = f"vision.blocks.{block_index.index(orig)}.{rest}"
        out[k] = v
    return out


def torch_encode(ref_vis, patches_np, nh, nw, weights, cfg_holder, torch_dtype,
                 n_layers=None, block_index=None):
    """Reference ViT+Aligner on numpy patches with checkpoint weights."""
    import torch

    torch.set_default_dtype(torch_dtype)
    vit, alg = ref_vis.ViT(cfg_holder), ref_vis.Aligner(cfg_holder)
    if n_layers is not None:
        vit.blocks = vit.blocks[:n_layers]
    weights = _renumbered(weights, block_index)
    vit.load_state_dict(
        {k[len("vision."):]: torch.from_numpy(v).to(torch_dtype)
         for k, v in weights.items() if k.startswith("vision.")}, strict=True)
    alg.load_state_dict(
        {k[len("aligner."):]: torch.from_numpy(v).to(torch_dtype)
         for k, v in weights.items() if k.startswith("aligner.")}, strict=True)
    vit.eval(), alg.eval()
    with torch.inference_mode():
        out = alg(vit(patches_np.to(torch_dtype), nh, nw), nh, nw)
    return out.float().numpy()


# ------------------------------------------------------------------- harness


def main() -> int:
    import torch

    dtype_name = os.environ.get("P97_DTYPE", "bf16")
    mlx_dtype = mx.float32 if dtype_name == "f32" else mx.bfloat16
    torch_dtype = torch.float32 if dtype_name == "f32" else torch.bfloat16
    n_images = int(os.environ.get("P97_IMAGES", str(len(FIXTURE_SPECS))))

    ref_vis = load_ref_module("vision", os.path.join(REF_DIR, "vision.py"))
    ref_ip = load_ref_module("image_processor", os.path.join(REF_DIR, "image_processor.py"))

    rargs = type("RA", (), {})()
    for k, v in dict(vision_patch_size=14, vision_dim=1024, vision_n_heads=16,
                     vision_inter_dim=2816, vision_n_layers=32, vision_rope_theta=10000.0,
                     vision_downsample_ratio=3, vision_max_n_token=1024,
                     vision_min_pixels=295936, vision_max_wh_ratio=None, dim=5120).items():
        setattr(rargs, k, v)

    from mlx_lm.models.exl3.loader import Exl3Checkpoint

    with Exl3Checkpoint(CKPT) as ck:
        names = mv.vision_weight_names(ck.index)
        weights_np = {n: ck.np(n) for n in names}
    tower, cfg = mv.load_vision_tower(CKPT, dtype=mlx_dtype)
    layers = os.environ.get("P97_LAYERS")
    keep_blocks = None
    if layers:  # cheap subset run: keep only a few blocks (both sides)
        lo, _, hi = layers.partition("-")
        keep_blocks = list(range(int(lo), int(hi) + 1))
        tower.vision.blocks = [tower.vision.blocks[i] for i in keep_blocks]
        weights_np = _renumbered(weights_np, keep_blocks)
        rargs.vision_n_layers = len(keep_blocks)
        print(f"[setup] LAYER SUBSET {keep_blocks} ({len(keep_blocks)} of {cfg.n_layers} blocks)")
    mx.eval(tower.parameters())
    print(f"[setup] {len(weights_np)} vision tensors, dtype={dtype_name}, "
          f"tower resident {mx.get_active_memory() / 1e6:.1f} MB, "
          f"torch {torch.__version__}")

    images = build_fixtures()[:n_images]
    results = []
    worst_aligner_cos = 1.0
    worst_block_cos = 1.0
    for i, rec in enumerate(images):
        r_patches, r_nh, r_nw, r_nlh, r_nlw = ref_ip.load_image(rec, rargs)
        m_patches, m_nh, m_nw, m_nlh, m_nlw = mip.load_image(rec, cfg.preprocess)
        if (r_nh, r_nw, r_nlh, r_nlw) != (m_nh, m_nw, m_nlh, m_nlw):
            raise AssertionError(
                f"geometry mismatch image {i}: ref {(r_nh, r_nw, r_nlh, r_nlw)} "
                f"mlx {(m_nh, m_nw, m_nlh, m_nlw)}")
        patch_diff = float(np.abs(r_patches.float().numpy() - m_patches).max())
        ref_types = ref_ip.image_token_types(m_nlh, m_nlw).numpy()
        mlx_types = mip.image_token_types(m_nlh, m_nlw)
        if not np.array_equal(ref_types, mlx_types):
            raise AssertionError(f"token types differ on image {i}")
        # rope tables: built by different code paths, must agree
        t_cos, t_sin = ref_vis.get_vision_cos_sin(m_nh, m_nw, 32, 10000.0)
        rope_diff = max(
            float(np.abs(np.array(mv.get_vision_cos_sin(m_nh, m_nw, 32, 10000.0)[0]
                                  .astype(mx.float32)) - t_cos.numpy()).max()),
            float(np.abs(np.array(mv.get_vision_cos_sin(m_nh, m_nw, 32, 10000.0)[1]
                                  .astype(mx.float32)) - t_sin.numpy()).max()),
        )

        ref_out = torch_encode(ref_vis, r_patches, r_nh, r_nw, weights_np, rargs, torch_dtype,
                               n_layers=len(tower.vision.blocks),
                               block_index=None if keep_blocks is None else list(range(len(keep_blocks))))

        mx.reset_peak_memory()
        t0 = time.perf_counter()
        mlx_out = tower.encode_image(mip.patches_to_mlx(m_patches).astype(mlx_dtype), m_nh, m_nw)
        mx.eval(mlx_out)
        t_mlx = time.perf_counter() - t0
        peak_mb = mx.get_peak_memory() / 1e6  # MLX TOTAL peak (weights included)

        a = np.array(mlx_out.astype(mx.float32))
        c = cos(a, ref_out)
        maxabs = float(np.abs(a - ref_out).max())
        rel = float(np.linalg.norm(a - ref_out) / np.linalg.norm(ref_out))
        worst_aligner_cos = min(worst_aligner_cos, c)

        # full block: aligner rows in the IMAGE slots, learned rows elsewhere
        img_in = mip.ImageInput(0, m_patches, m_nh, m_nw, mlx_types)
        blk = np.array(tower.build_image_block(img_in).astype(mx.float32))
        rb = np.zeros_like(blk)
        rb[mlx_types == mip.IMAGE] = ref_out
        rb[mlx_types == mip.IMAGE_START] = weights_np["image_start"]
        rb[mlx_types == mip.IMAGE_END] = weights_np["image_end"]
        rb[mlx_types == mip.IMAGE_NEW_LINE] = weights_np["image_newline"]
        bcos = cos(blk, rb)
        worst_block_cos = min(worst_block_cos, bcos)

        results.append(dict(
            image=i, name=rec["name"], vit_grid=[m_nh, m_nw], llm_grid=[m_nlh, m_nlw],
            n_patches=int(m_patches.shape[0]), n_tokens=int(mlx_types.size),
            patches_bit_exact=patch_diff == 0.0, patch_maxdiff=patch_diff,
            rope_table_maxdiff=rope_diff, aligner_cos=c, aligner_maxabs=maxabs,
            aligner_rel=rel, block_cos=bcos, mlx_ms=t_mlx * 1e3,
            mlx_peak_mb=peak_mb,
        ))
        print(f"[{i}] {rec['name']:>14} vit {m_nh:>3}x{m_nw:<3} llm {m_nlh:>2}x{m_nlw:<2} "
              f"{mlx_types.size:>4} tok  patches_exact={patch_diff == 0.0!s:>5} "
              f"rope_ok={rope_diff == 0.0!s:>5}  cos={c:.10f} block_cos={bcos:.10f} "
              f"maxabs={maxabs:.4f} rel={rel:.3e}  {t_mlx * 1e3:6.0f} ms  peak {peak_mb:6.1f} MB")

    summary = dict(
        dtype=dtype_name, n_images=len(images),
        worst_aligner_cos=worst_aligner_cos, worst_block_cos=worst_block_cos,
        bar=0.9998,
        pass_aligner=bool(worst_aligner_cos >= 0.9998),
        pass_block=bool(worst_block_cos >= 0.9998),
        results=results,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    if os.environ.get("P97_JSON"):
        Path(os.path.expanduser(os.environ["P97_JSON"])).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
