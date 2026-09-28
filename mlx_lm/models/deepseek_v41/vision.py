"""DeepSeek-V4.1-Flash vision tower (ViT + Aligner) for MLX -- native reimplementation.

Port of the release's torch reference:

    docs/reference/vision.py            (ViT, Aligner, 2D RoPE, RMSNorm)
    docs/reference/image_processor.py   (the DSv4.1 four-sentinel variant)
    docs/reference/model.py             (Transformer.encode_image /
                                         Transformer.merge_image_embeddings)

The tower is not part of the text stack's EXL3 quantization: every
``vision.*`` / ``aligner.*`` / ``image_start`` / ``image_end`` /
``image_newline`` tensor is stored BF16 in the checkpoint and is loaded here
as bf16 (no trellis decode involved). The text-side integration points are:

* :meth:`VisionTower.encode_image` -- patches -> ``(n_rows, text_dim)``
  aligner features for one image;
* :meth:`VisionTower.merge_image_embeddings` -- overwrite an image's span of a
  ``(b, s, text_dim)`` embedding tensor in place, exactly like the reference;
* :meth:`VisionTower.build_image_block` -- one image's full span block
  (aligner rows into the IMAGE slots, learned embeddings on the delimiters).

Differences from the V4-Flash vision path already in this repo (``deepseek_v4.py``)
that matter, all inherited from the release:

* FOUR sentinel types, not five: there is no ``IMAGE_PAD`` and no
  position-dependent compress-alignment padding. The checkpoint has exactly
  three sentinel tensors, and the span is
  ``[IMAGE_START] + ([IMAGE] * n_w + [IMAGE_NEW_LINE]) * n_h + [IMAGE_END]``.
* The aligner grid uses ``ceil`` with zero padding folded into the LAST row and
  column of aligner outputs (the padded ViT rows are assigned to real IMAGE
  tokens), so ``n_llm_h * n_llm_w`` always equals the aligner output row count.

Numerics: modules are constructed BF16, activations stay BF16, and the two
places the reference does float32 work (RMSNorm, and the rotary multiply) are
kept in float32 here too. Parity against the torch reference is measured, not
assumed -- see ``benchmarks/p97_vision_parity.py``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .image_processor import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    ImageInput,
    ImagePreprocessConfig,
    patches_to_mlx,
)

__all__ = [
    "Aligner",
    "Attention",
    "Block",
    "MLP",
    "PatchEmbed",
    "RMSNorm",
    "ViT",
    "VisionConfig",
    "VisionTower",
    "attach_vision",
    "get_vision_cos_sin",
    "load_vision_tower",
]


@dataclass(frozen=True)
class VisionConfig:
    """Vision-relevant subset of the checkpoint config.

    Read from the EXL3/HF layout (``vision_config`` sub-dict + top-level
    ``hidden_size`` / ``image_token_id``) or from the reference's flat
    ``vision_*`` keys.
    """

    dim: int = 1024
    n_layers: int = 32
    n_heads: int = 16
    inter_dim: int = 2816
    patch_size: int = 14
    rope_theta: float = 10000.0
    downsample_ratio: int = 3
    max_image_tokens: int = 1024
    min_pixels: int = 544 * 544
    max_wh_ratio: int | None = None
    text_dim: int = 5120
    image_token_id: int = 129264

    @property
    def rope_dim(self) -> int:
        """Per-axis rotary width: half of one head's channels, split h/w."""
        return self.dim // self.n_heads // 2

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def vision_enabled(self) -> bool:
        return self.n_layers > 0

    @property
    def preprocess(self) -> ImagePreprocessConfig:
        return ImagePreprocessConfig(
            patch_size=self.patch_size,
            downsample_ratio=self.downsample_ratio,
            max_token_count=self.max_image_tokens,
            min_pixel_count=self.min_pixels,
            max_width_height_ratio=self.max_wh_ratio,
        )

    @classmethod
    def from_config(cls, c: dict[str, Any]) -> "VisionConfig":
        vc = c.get("vision_config") or {}
        top = c.get("text_config", c)

        def pick(vc_key, *flat_keys):
            if vc.get(vc_key) is not None:
                return vc[vc_key]
            for k in flat_keys:
                if c.get(k) is not None:
                    return c[k]
            return None

        n_layers = pick("num_hidden_layers", "vision_n_layers")
        dim = pick("hidden_size", "vision_dim")
        n_heads = pick("num_attention_heads", "vision_n_heads")
        return cls(
            dim=dim if dim is not None else 1024,
            n_layers=n_layers if n_layers is not None else 0,
            n_heads=n_heads if n_heads is not None else 16,
            inter_dim=pick("intermediate_size", "vision_inter_dim") or 2816,
            patch_size=pick("patch_size", "vision_patch_size") or 14,
            rope_theta=float(pick("rope_theta", "vision_rope_theta") or 10000.0),
            downsample_ratio=pick("downsample_ratio", "vision_downsample_ratio") or 3,
            max_image_tokens=pick("max_image_tokens", "vision_max_n_token") or 1024,
            min_pixels=pick("min_pixels", "vision_min_pixels") or 544 * 544,
            max_wh_ratio=pick("max_wh_ratio", "vision_max_wh_ratio"),
            text_dim=int(pick("text_hidden_size", "dim", "hidden_size") or top.get("hidden_size", 5120)),
            image_token_id=int(c.get("image_token_id") or 129264),
        )


class RMSNorm(nn.Module):
    """Reference RMSNorm: float32 compute and a float32 weight, bf16 output.

    ``eps=1e-6`` is the reference constructor default and is unrelated to the
    text model's ``rms_norm_eps=1e-20``; wiring the text epsilon in here would
    be silently wrong.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        xf = x.astype(mx.float32)
        xf = xf * mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + self.eps)
        return (self.weight * xf).astype(dtype)


@lru_cache(maxsize=8)
def get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float
) -> tuple[mx.array, mx.array]:
    """2D rotary tables for an ``n_h x n_w`` patch grid, shape ``(n, 1, dim)``.

    The frequency layout is the reference's, exactly: per position the vector is
    ``[h * inv_freq, w * inv_freq]`` (h-half first), with
    ``inv_freq = theta ** -(arange(0, dim, 2) / dim)``. Computed in float32 and
    hoisted once -- ``lru_cache`` keyed on the grid keeps it out of the per-image
    path. The ``(n, 1, dim)`` shape is what lets the same table broadcast over
    the head axis in :func:`apply_rotary`.
    """
    inv_freq = (
        1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)))
    ).astype(np.float32)
    hpos = np.broadcast_to(np.arange(n_h, dtype=np.float32)[:, None], (n_h, n_w))
    wpos = np.broadcast_to(np.arange(n_w, dtype=np.float32)[None, :], (n_h, n_w))
    freqs = (np.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1) * inv_freq.reshape(1, 1, -1))
    freqs = freqs.reshape(n_h * n_w, -1)
    return (
        mx.array(np.cos(freqs).astype(np.float32))[:, None, :],
        mx.array(np.sin(freqs).astype(np.float32))[:, None, :],
    )


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Reference rotary application: float32 multiply, back to the input dtype."""
    dtype = x.dtype
    x1, x2 = mx.split(x.astype(mx.float32), 2, axis=-1)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(dtype)


def _linear(in_dim: int, out_dim: int, *, bias: bool = True, dtype=mx.bfloat16) -> nn.Linear:
    """``nn.Linear`` with its weights stored at ``dtype``.

    This MLX build's ``nn.Linear`` has no ``dtype`` argument, so the weight and
    bias are built float32 and cast. Both are plain parameters (no quantization
    state), so the cast is the whole story. Note that ``bias=False`` leaves no
    ``bias`` ATTRIBUTE at all in this build, hence the ``getattr``.
    """
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    lin.weight = lin.weight.astype(dtype)
    bias_value = getattr(lin, "bias", None)
    if bias_value is not None:
        lin.bias = bias_value.astype(dtype)
    return lin


class PatchEmbed(nn.Module):
    """Linear over flattened patches -- the reference has no Conv2d here."""

    def __init__(self, cfg: VisionConfig, dtype=mx.bfloat16):
        super().__init__()
        self.proj = _linear(3 * cfg.patch_size**2, cfg.dim, dtype=dtype)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x.reshape(x.shape[0], -1))


class Attention(nn.Module):
    """Full bidirectional attention over one image, 2D RoPE on q and k."""

    def __init__(self, cfg: VisionConfig, dtype=mx.bfloat16):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        self.wqkv = _linear(cfg.dim, 3 * cfg.dim, dtype=dtype)
        self.wo = _linear(cfg.dim, cfg.dim, dtype=dtype)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        n = x.shape[0]
        q, k, v = (
            t.reshape(n, self.n_heads, self.head_dim)
            for t in mx.split(self.wqkv(x), 3, axis=-1)
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        # The reference feeds torch SDPA 3-D (H, n, D) tensors, which torch reads
        # as an unbatched multi-head attention with scale 1/sqrt(D). MLX wants
        # rank 4: the same math with an explicit batch axis.
        out = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            k.transpose(1, 0, 2)[None],
            v.transpose(1, 0, 2)[None],
            scale=self.scale,
            mask=None,
        )
        return self.wo(out[0].transpose(1, 0, 2).reshape(n, -1))


class MLP(nn.Module):
    """SwiGLU with a fused gate+up projection (``w1`` emits ``2 * inter_dim``)."""

    def __init__(self, cfg: VisionConfig, dtype=mx.bfloat16):
        super().__init__()
        self.w1 = _linear(cfg.dim, 2 * cfg.inter_dim, bias=False, dtype=dtype)
        self.w2 = _linear(cfg.inter_dim, cfg.dim, bias=False, dtype=dtype)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.w1(x), 2, axis=-1)
        return self.w2(nn.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: VisionConfig, dtype=mx.bfloat16):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg, dtype=dtype)
        self.norm2 = RMSNorm(cfg.dim)
        self.mlp = MLP(cfg, dtype=dtype)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


#: Evaluate every ``fence_every`` blocks during the ViT forward.
#:
#: MLX builds one lazy graph for the whole tower, so with no fence the entire
#: 32-block chain plus its intermediates is live at once. Measured on a
#: 2048x2048 image (8649 patches) the fence is FREE -- 1366 ms fenced vs
#: 1367 ms unfenced -- while cutting the transient peak from 1530 MB to 873 MB.
#: On a 1581-patch image it is a 8.4x cut (1341 MB -> 160 MB) and slightly
#: FASTER (148 ms vs 153 ms), because it lets MLX recycle buffers instead of
#: growing one graph-wide allocation pool. 1 (every block) is the default.
VIT_FENCE_EVERY = int(os.environ.get("DSV41_VIT_FENCE", "1"))


class ViT(nn.Module):
    """DeepSeek ViT: patches -> patch_embed -> N blocks -> final norm."""

    def __init__(self, cfg: VisionConfig, dtype=mx.bfloat16, fence_every: int | None = None):
        super().__init__()
        self.rope_dim = cfg.rope_dim
        self.rope_theta = cfg.rope_theta
        self.patch_embed = PatchEmbed(cfg, dtype=dtype)
        self.blocks = [Block(cfg, dtype=dtype) for _ in range(cfg.n_layers)]
        self.norm = RMSNorm(cfg.dim)
        self.fence_every = VIT_FENCE_EVERY if fence_every is None else fence_every

    def __call__(self, patches: mx.array, n_h: int, n_w: int) -> mx.array:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        fence = self.fence_every
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin)
            if fence and (i + 1) % fence == 0:
                mx.eval(x)
        return self.norm(x)


def _unfold(x: mx.array, r: int) -> mx.array:
    """``F.unfold(x_chw[None], r, stride=r)[0].transpose(0, 1)`` in MLX.

    Input ``(H, W, C)`` with H, W divisible by ``r``; output
    ``(H//r * W//r, C * r * r)``. PyTorch's unfold emits channels first, then
    kernel row, then kernel col -- flat index ``c * r*r + kh * r + kw`` -- which
    is why the channel axis is transposed BEFORE the two kernel axes.
    """
    h, w, c = x.shape
    nbh, nbw = h // r, w // r
    return x.reshape(nbh, r, nbw, r, c).transpose(0, 2, 4, 1, 3).reshape(nbh * nbw, -1)


class Aligner(nn.Module):
    """Patch features -> ``r x r`` downsample -> 2-layer GELU MLP into text dim."""

    def __init__(self, cfg: VisionConfig, dtype=mx.bfloat16):
        super().__init__()
        self.downsample_ratio = cfg.downsample_ratio
        in_dim = cfg.dim * self.downsample_ratio**2
        self.w1 = _linear(in_dim, cfg.text_dim, dtype=dtype)
        self.w2 = _linear(cfg.text_dim, cfg.text_dim, dtype=dtype)

    def __call__(self, x: mx.array, n_h: int, n_w: int) -> mx.array:
        r = self.downsample_ratio
        x = x.reshape(n_h, n_w, -1)
        pad_h, pad_w = -n_h % r, -n_w % r
        if pad_h or pad_w:
            x = mx.pad(x, [(0, pad_h), (0, pad_w), (0, 0)])
        x = _unfold(x, r)
        # F.gelu default is the EXACT erf GELU -- nn.gelu, never gelu_approx.
        return self.w2(nn.gelu(self.w1(x)))


class VisionTower(nn.Module):
    """ViT + Aligner + the three learned sentinel embeddings.

    Attribute names match the checkpoint keys one-for-one (``vision.*``,
    ``aligner.*``, ``image_start`` / ``image_end`` / ``image_newline``), so
    :func:`load_vision_tower` can hand checkpoint tensors straight to
    ``load_weights``.
    """

    def __init__(self, cfg: VisionConfig, dtype=mx.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.vision = ViT(cfg, dtype=dtype)
        self.aligner = Aligner(cfg, dtype=dtype)
        self.image_start = mx.zeros((cfg.text_dim,), dtype=dtype)
        self.image_end = mx.zeros((cfg.text_dim,), dtype=dtype)
        self.image_newline = mx.zeros((cfg.text_dim,), dtype=dtype)

    # -- encoding ---------------------------------------------------------
    def encode_image(self, patches: mx.array, n_vit_h: int, n_vit_w: int) -> mx.array:
        """Reference ``Transformer.encode_image``: ViT then aligner."""
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def sentinel_parameters(self) -> mx.array:
        """``(4, text_dim)`` rows indexed by sentinel TYPE.

        Slot 1 (``IMAGE``) is never read: ``build_image_block`` overwrites those
        rows with aligner output. It exists so the table can be indexed by raw
        type value, as in the reference.
        """
        return mx.stack(
            [self.image_start, mx.zeros_like(self.image_start), self.image_newline, self.image_end]
        )

    def build_image_block(self, img: ImageInput) -> mx.array:
        """One image's ``(n_tokens, text_dim)`` span: aligner rows + sentinels."""
        embeds = self.encode_image(patches_to_mlx(img.patches), img.n_vit_h, img.n_vit_w)
        types = np.asarray(img.types)
        is_image = types == IMAGE
        n_image = int(is_image.sum())
        if n_image != embeds.shape[0]:
            raise ValueError(
                f"image block has {n_image} IMAGE slots but the aligner produced "
                f"{embeds.shape[0]} rows -- the patch geometry and the emitted block "
                "were computed from different image dimensions"
            )
        block = self.sentinel_parameters()[mx.array(types)]
        # cumsum - 1 numbers the IMAGE slots 0..n-1 in order; non-IMAGE rows read
        # a clamped (discarded) index.
        slot = mx.array(np.clip(np.cumsum(is_image) - 1, 0, n_image - 1))
        return mx.where(mx.array(is_image)[:, None], embeds[slot], block)

    def merge_image_embeddings(
        self, h: mx.array, image_inputs: list[ImageInput] | None, sample: int = 0
    ) -> mx.array:
        """Overwrite every image's token span in ``h`` with its features, in place.

        ``h`` is ``(s, dim)`` or ``(b, s, dim)`` -- the reference's
        ``merge_image_embeddings`` for one sample. Returns ``h`` (modified).
        Callers must have ``h`` materialized before the layers read it; MLX
        handles the write ordering, but the spans must lie inside the prompt.
        """
        if not image_inputs:
            return h
        n_tokens = h.shape[-2]
        for img in image_inputs:
            block = self.build_image_block(img)
            start = img.start
            end = start + block.shape[0]
            if end > n_tokens:
                raise ValueError(
                    f"image block at {start} runs to {end}, past the {n_tokens}-token prompt"
                )
            if h.ndim == 2:
                h[start:end, :] = block.astype(h.dtype)
            else:
                h[sample, start:end, :] = block.astype(h.dtype)
        return h


#: Checkpoint key prefixes this module owns.
VISION_PREFIXES = ("vision.", "aligner.", "image_start", "image_end", "image_newline")


def vision_weight_names(index: dict[str, str]) -> list[str]:
    """Every checkpoint key the vision tower consumes, sorted."""
    return sorted(k for k in index if k.startswith(VISION_PREFIXES))


def load_vision_tower(
    model_dir: str, *, dtype=mx.bfloat16, verbose: bool = False
) -> tuple[VisionTower, VisionConfig]:
    """Build the tower and load its BF16 tensors from an EXL3 checkpoint.

    Reads only the ``vision.*`` / ``aligner.*`` / ``image_*`` entries of the
    checkpoint (266 tensors in the release) with positional pread, converts
    BF16 -> bf16 exactly (via float32), and asserts strict accounting: every
    checkpoint key of this group must land, and no tower parameter may be left
    at its initial value.
    """
    from ..exl3.loader import Exl3Checkpoint

    with Exl3Checkpoint(model_dir) as ck:
        cfg = VisionConfig.from_config(ck.config)
        if not cfg.vision_enabled:
            raise ValueError(
                f"{model_dir}: config has no vision tower (vision_n_layers == 0)"
            )
        tower = VisionTower(cfg, dtype=dtype)
        names = vision_weight_names(ck.index)
        missing = [
            n
            for n in names
            if not _tower_has_name(tower, n)
        ]
        if missing:
            raise KeyError(f"checkpoint vision tensors with no tower parameter: {missing}")
        weights = [
            (name, mx.array(ck.np(name)).astype(dtype)) for name in names
        ]
        tower.load_weights(weights)
        expect = {n for n, _ in nn.utils.tree_flatten(tower.parameters())}
        landed = {name for name, _ in weights}
        if expect != landed:
            raise KeyError(
                "vision tower accounting mismatch: "
                f"unloaded={sorted(expect - landed)} extra={sorted(landed - expect)}"
            )
        mx.eval(tower.parameters())
        if verbose:
            print(f"vision tower: {len(names)} tensors, cfg={cfg}")
        return tower, cfg


def _tower_has_name(tower: VisionTower, name: str) -> bool:
    try:
        node: Any = tower
        for part in name.split("."):
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        return True
    except (AttributeError, IndexError, KeyError, TypeError):
        return False


def attach_vision(model: Any, cfg: VisionConfig | None = None, *, dtype=mx.bfloat16) -> VisionTower:
    """Attach an (unloaded) vision tower to a text model, V4-style.

    Sets ``model.vision``, ``model.aligner`` and the three sentinel parameters so
    the tower participates in ``model.parameters()`` / ``load_weights`` like it
    does in the reference ``Transformer``. Weight loading stays the caller's job
    (``load_vision_tower`` reads them straight from the checkpoint).
    """
    cfg = cfg or VisionConfig()
    tower = VisionTower(cfg, dtype=dtype)
    model.vision = tower.vision
    model.aligner = tower.aligner
    model.image_start = tower.image_start
    model.image_end = tower.image_end
    model.image_newline = tower.image_newline
    return tower
