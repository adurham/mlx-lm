"""Image preprocessing for the DeepSeek-V4.1 vision tower (torch-free).

Ported from the release's ``inference/image_processor.py`` -- the DSv4.1
revision, which uses FOUR sentinel token types with no compress-alignment
padding. (The earlier DSv4-Flash-Vision release used five types plus a
position-dependent pad; that is a different model and must not be mixed in
here.)

The pipeline is:

* resize/letterbox the image to a patch-aligned pixel grid,
* normalize to ``[-1, 1]`` and cut it into ``patch_size x patch_size``
  patches in row-major order,
* expand each ``<|deepseek_image|>`` placeholder token into the sentinel block

    ``[IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]``

Only ``numpy`` and ``PIL`` are used: this module must stay importable without
torch. Numbers are reproduced bit-for-bit against the reference -- in
particular the bfloat16 rounding of the normalized pixels, which
:func:`round_to_bfloat16_precision` performs in float32 storage because numpy
has no native bfloat16 (see that function's docstring for the verification
record).
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps

__all__ = [
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_START",
    "TEXT",
    "ImageInput",
    "ImagePreprocessConfig",
    "image_token_types",
    "llm_grid",
    "load_image",
    "load_image_bytes",
    "num_image_tokens",
    "plan_image_grid",
    "prepare_vl_inputs",
    "round_to_bfloat16_precision",
    "safe_resize",
    "solve_resize_ratio",
]

#: Sentinel token TYPES. ``TEXT`` marks ordinary text positions; the sentinel
#: values are 0..3 in EMISSION order and are load-bearing: the merge helper
#: indexes its per-type embeddings by them.
TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


@dataclass(frozen=True)
class ImagePreprocessConfig:
    """The preprocessing-relevant subset of the vision config."""

    patch_size: int
    downsample_ratio: int
    max_token_count: int
    min_pixel_count: int
    max_width_height_ratio: int | None


@dataclass
class ImageInput:
    """One image's ViT input and its position in the expanded token stream.

    Attributes:
        start: Index of the image block's first token within the prompt's token
            stream (the ``len(tokens)`` captured before the block is appended).
        patches: ``(n_vit_h * n_vit_w, 3, patch_size, patch_size)`` float32
            array holding bfloat16-precision values (numpy has no bfloat16;
            see :func:`round_to_bfloat16_precision`).
        n_vit_h: ViT patch-grid height.
        n_vit_w: ViT patch-grid width.
        types: Sentinel token TYPES in final emission order (``int64``).
    """

    start: int
    patches: np.ndarray
    n_vit_h: int
    n_vit_w: int
    types: np.ndarray


_PIXEL_VALUE_SCALE = np.float32(255.0)
_NORMALIZE_SHIFT = np.float32(0.5)
_NORMALIZE_SCALE = np.float32(0.5)
_PAD_FILL_COLOR = (127, 127, 127)


def round_to_bfloat16_precision(values: np.ndarray) -> np.ndarray:
    """Reduce float32 values to bfloat16 precision, keeping float32 storage.

    Reproduces ``torch.Tensor.to(torch.bfloat16)`` for the pixel domain: the
    cast keeps the top 8 mantissa bits and rounds half-to-even, i.e. it adds
    ``0x7FFF + lsb_of_kept_bits`` and truncates the low 16 bits.

    Verified equal to the torch cast on this pipeline's exact value domain
    (all values are ``(k/255 - 0.5) / 0.5`` for integer ``k`` in 0..255, so the
    domain is 256 distinct magnitudes per sign): the rounding path is exact for
    every one of them, and there are no subnormals or NaN/Inf. An ``mlx.core``
    bfloat16 cast was checked and diverges only on subnormal inputs, which this
    domain cannot produce.
    """
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    bias = ((bits >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32).copy()


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """Token count of an image span: rows of ``n_llm_w`` images + newline, +2."""
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(
    height: int | float,
    width: int | float,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
):
    """Largest aspect-preserving pixel size whose token grid fits max_n_token.

    Returns ``(best_height, best_width)``, both multiples of ``patch_size``.
    """
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def safe_resize(
    height: int | float,
    width: int | float,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
):
    """Shrink the pixel size until the image costs at most max_n_token tokens.

    Returns ``(n_llm_h, n_llm_w, best_height, best_width)``.
    """
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token
        )
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
            raise ValueError(
                f"resize solve overflowed the token budget: {n_llm_h}x{n_llm_w} "
                f"= {num_image_tokens(n_llm_h, n_llm_w)} > {max_n_token}"
            )
    return n_llm_h, n_llm_w, best_height, best_width


def load_image_bytes(record) -> bytes:
    """Load image bytes from raw/base64 data, an Anthropic source, URL, or path."""
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)

    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]})

    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as response:  # noqa: S310
                return response.read()
        with open(url, "rb") as file:
            return file.read()

    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def plan_image_grid(width: int, height: int, config: ImagePreprocessConfig):
    """Resize plan for an image of the given original size; a pure function of its arguments.

    Returns ``(n_llm_h, n_llm_w, best_height, best_width)``.
    """
    p = config.patch_size
    if config.max_width_height_ratio is not None and width > height * config.max_width_height_ratio:
        width = height * config.max_width_height_ratio
    if 0 < width * height < config.min_pixel_count:
        ratio = (config.min_pixel_count / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(
        height, width, best_height, best_width, p, config.downsample_ratio, config.max_token_count
    )


def load_image(record, config: ImagePreprocessConfig):
    """Load and transform one image record into ViT patches.

    Returns ``(patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w)`` with ``patches``
    shaped ``(n_vit_h * n_vit_w, 3, patch_size, patch_size)`` in row-major
    patch order.

    Two reference details are preserved verbatim because they are load-bearing:
    the aspect clamp mutates only the LOCAL ``width`` used for grid solving
    (the letterbox branch below re-reads the unclamped ``image.width``), and an
    image at least ``max_width_height_ratio`` times wider than tall is resized
    outright rather than letterboxed.
    """
    p = config.patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, config)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if config.max_width_height_ratio is not None and image.width >= config.max_width_height_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=_PAD_FILL_COLOR)
    x = np.asarray(image, dtype=np.float32)
    x = x.transpose(2, 0, 1) / _PIXEL_VALUE_SCALE
    x = round_to_bfloat16_precision((x - _NORMALIZE_SHIFT) / _NORMALIZE_SCALE)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .transpose(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> np.ndarray:
    """Default layout: the aligner grid in reading order, one IMAGE_NEW_LINE per row."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return np.array(types, dtype=np.int64)


def patches_to_mlx(patches: np.ndarray):
    """``(n, 3, p, p)`` bfloat16-precision float32 patches -> mlx bfloat16.

    The values are exactly representable in bfloat16, so the cast is lossless.
    """
    import mlx.core as mx

    return mx.array(patches).astype(mx.bfloat16)


def _as_preprocess_config(args) -> ImagePreprocessConfig:
    """Accept either an ``ImagePreprocessConfig``, an object exposing one as
    ``.preprocess``, or the reference's flat ``vision_*`` attribute bag."""
    if isinstance(args, ImagePreprocessConfig):
        return args
    inner = getattr(args, "preprocess", None)
    if isinstance(inner, ImagePreprocessConfig):
        return inner
    return ImagePreprocessConfig(
        patch_size=args.vision_patch_size,
        downsample_ratio=args.vision_downsample_ratio,
        max_token_count=args.vision_max_n_token,
        min_pixel_count=args.vision_min_pixels,
        max_width_height_ratio=args.vision_max_wh_ratio,
    )


def prepare_vl_inputs(prompt, images, tokenizer, args) -> tuple[list[int], list[int], list[ImageInput] | None]:
    """Tokenize ``prompt``, expanding each image placeholder into its image span.

    Returns ``(tokens, token_types, image_inputs)``. Image-span positions carry
    ``args.image_token_id`` in ``tokens`` and are distinguished only by
    ``token_types`` (``TEXT`` elsewhere). ``image_inputs`` is None with no images.
    """
    config = _as_preprocess_config(args)
    image_token_id = args.image_token_id
    prompt_tokens = tokenizer.encode(prompt)
    num_placeholders = sum(token == image_token_id for token in prompt_tokens)
    if num_placeholders != len(images):
        raise ValueError(f"Found {num_placeholders} image tokens but got {len(images)} images")

    tokens, token_types, image_inputs = [], [], []
    image_iter = iter(images)
    for tok in prompt_tokens:
        if tok != image_token_id:
            tokens.append(tok)
            token_types.append(TEXT)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(next(image_iter), config)
        types = image_token_types(n_llm_h, n_llm_w)
        image_inputs.append(ImageInput(len(tokens), patches, n_vit_h, n_vit_w, types))
        tokens += [image_token_id] * int(types.size)
        token_types += [int(t) for t in types]
    return tokens, token_types, image_inputs or None
