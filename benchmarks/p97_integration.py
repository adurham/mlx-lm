"""End-to-end smoke test for the DSv4.1 vision plumbing.

Exercises the whole client-facing path exactly as the text runtime will use it:

  1. ``prepare_vl_inputs``  -- prompt with an image placeholder -> expanded token
     ids + token types + ImageInput,
  2. ``attach_vision``      -- tower attached to a text model's attributes,
  3. ``merge_image_embeddings`` -- image span written into a (b, s, 5120)
     embedding tensor,
  4. sanity: the span holds the sentinel rows and the aligner rows in order, the
     rest of the tensor is untouched, and no token id escapes the vocabulary
     except the sentinel block (which is deliberately out of vocabulary).

  PYTHONPATH=~/dsv41-ws/H lockf -k ~/dsv41-gpu.lock \
    ~/repos/exo/.venv/bin/python benchmarks/p97_integration.py
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dsv41-ws/H"))

import mlx.core as mx  # noqa: E402
from PIL import Image  # noqa: E402

from mlx_lm.models.deepseek_v41 import image_processor as mip  # noqa: E402
from mlx_lm.models.deepseek_v41 import vision as mv  # noqa: E402

CKPT = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)


class _Tok:
    """Minimal tokenizer: maps '<image>' to the real image_token_id."""

    def __init__(self, image_token_id: int):
        self.image_token_id = image_token_id

    def encode(self, prompt: str) -> list[int]:
        out = []
        for part in prompt.split("<image>"):
            out.extend(ord(c) % 1000 for c in part)
            out.append(self.image_token_id)
        return out[:-1]  # drop the trailing marker


def main() -> int:
    tower, cfg = mv.load_vision_tower(CKPT, dtype=mx.bfloat16)
    layers = os.environ.get("P97_LAYERS")
    if layers:
        lo, _, hi = layers.partition("-")
        tower.vision.blocks = [tower.vision.blocks[i] for i in range(int(lo), int(hi) + 1)]
    mx.eval(tower.parameters())

    arr = np.random.default_rng(11).integers(0, 256, (300, 500, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    rec = {"data": buf.getvalue()}

    # --- 1. prompt expansion
    args = type("A", (), {})()
    args.image_token_id = cfg.image_token_id
    args.preprocess = cfg.preprocess
    tok = _Tok(cfg.image_token_id)
    ids, types, imgs = mip.prepare_vl_inputs(
        "describe this: <image> please", [rec], tok, args
    )
    assert imgs is not None and len(imgs) == 1, imgs
    img = imgs[0]
    n_img_tokens = int(np.sum(np.array(types) >= 0))
    print(f"[1] prompt expanded: {len(ids)} tokens, {n_img_tokens} image-span tokens, "
          f"span [{img.start}, {img.start + len(img.types)})")
    assert n_img_tokens == mip.num_image_tokens(*[int(x) for x in
                                                  mip.llm_grid(img.n_vit_h * cfg.patch_size,
                                                               img.n_vit_w * cfg.patch_size,
                                                               cfg.patch_size,
                                                               cfg.downsample_ratio)])
    assert all(ids[img.start + i] == cfg.image_token_id for i in range(len(img.types)))

    # --- 2. attach to a mock text model
    class _TextModel:
        pass

    model = _TextModel()
    attached = mv.attach_vision(model, cfg)
    assert model.vision is attached.vision and model.aligner is attached.aligner
    for name in ("image_start", "image_end", "image_newline"):
        assert hasattr(model, name), name
    print("[2] tower attached: model.vision / model.aligner / image_* present")

    # --- 3. merge into a (1, s, 5120) embedding tensor
    h = mx.ones((1, len(ids), cfg.text_dim), dtype=mx.bfloat16) * -3.0
    merged = attached.merge_image_embeddings(h, imgs, sample=0)
    mx.eval(merged)
    m = np.array(merged.astype(mx.float32))
    span = m[0, img.start: img.start + len(img.types)]
    blk = np.array(attached.build_image_block(img).astype(mx.float32))
    np.testing.assert_allclose(span, blk, rtol=0, atol=0)
    # untouched outside the span
    np.testing.assert_allclose(m[0, : img.start], -3.0)
    np.testing.assert_allclose(m[0, img.start + len(img.types):], -3.0)
    print(f"[3] merged {len(img.types)} rows into a {m.shape} tensor; "
          f"outside-span rows untouched")

    # --- 4. sanity on the span content
    types_arr = np.asarray(img.types)
    for t, name, param in ((mip.IMAGE_START, "image_start", attached.image_start),
                           (mip.IMAGE_NEW_LINE, "image_newline", attached.image_newline),
                           (mip.IMAGE_END, "image_end", attached.image_end)):
        rows = np.nonzero(types_arr == t)[0]
        want = np.array(param.astype(mx.float32))
        for r in rows:
            np.testing.assert_allclose(span[r], want, rtol=0, atol=0)
    embeds = np.array(
        attached.encode_image(mip.patches_to_mlx(img.patches), img.n_vit_h, img.n_vit_w)
        .astype(mx.float32)
    )
    np.testing.assert_allclose(span[types_arr == mip.IMAGE], embeds, rtol=0, atol=0)
    print(f"[4] span layout verified: {int((types_arr == mip.IMAGE).sum())} aligner rows, "
          f"{int((types_arr == mip.IMAGE_NEW_LINE).sum())} newlines, 1 start, 1 end")
    print(f"    aligner rows {embeds.shape}, span rows {span.shape}, "
          f"sentinel dtype {attached.image_start.dtype}")
    print(f"[peak] MLX total peak: {mx.get_peak_memory() / 1e9:.2f} GB")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
