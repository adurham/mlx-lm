# Copyright © 2026 Apple Inc.

"""Pure-torch reference logic for DeepSeek-V4-Flash-Vision-Exp Phase 3.

Why this file exists
--------------------
The parity target is DeepSeek's own ``inference/model.py`` from the
``DeepSeek-V4-Flash-Vision-Exp`` repo. That file **cannot be imported**: its
first import is

    from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn

and ``kernel`` is DeepSeek's closed proprietary CUDA module — it is not
published on HuggingFace and is not obtainable. Instantiating ``Gate`` or
``Transformer`` as-is is therefore impossible, on any machine.

What IS possible, and is what this file does: the three pieces of logic Phase 3
ports are **pure torch** and touch none of that. ``get_image_visible`` and
``get_window_topk_idxs_visible`` are verbatim-copyable (they use only
``arange``/``cumsum``/``cummax``/``cummin``/``clamp``/``where``).
``Gate.forward``'s routing is torch ops plus one call to the reference's own
``linear()`` helper, which for an unquantized float weight is exactly
``F.linear`` — see ``model.py``'s ``linear()``: it branches on
``weight.dtype in (float4_e2m1fn_x2, float8_e4m3fn)`` and otherwise returns
``F.linear(x, weight, bias)``. All reference tensors here are float32, so the
``F.linear`` branch is the one the real model takes for these ops too.

Everything below is transcribed from ``model.py`` with NO semantic edits. The
line references are to the Vision-Exp ``inference/model.py``.

This module is TEST-ONLY. It is not imported by, and must never be imported
by, anything under ``mlx_lm/``.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F  # noqa: N812  (universal PyTorch convention)

# `from image_processor import IMAGE, IMAGE_START, IMAGE_END` (model.py:13).
# image_processor.py defines `IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE,
# IMAGE_END = range(5)`. Mirrored by exo's vendored port at
# src/exo/worker/engines/mlx/vendor/deepseek_v4_image_processor.py:72-76.
IMAGE_START = 0
IMAGE_PAD = 1
IMAGE = 2
IMAGE_NEW_LINE = 3
IMAGE_END = 4

__all__ = [
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_PAD",
    "IMAGE_START",
    "RefGate",
    "get_image_visible",
    "get_window_topk_idxs",
    "get_window_topk_idxs_visible",
]


# ─────────────────────────── model.py:270-280, verbatim ───────────────────────
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1
        )
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size)
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


# ─────────────────────────── model.py:283-294, verbatim ───────────────────────
def get_image_visible(input_ids: torch.Tensor, vocab_size: int, max_image_tokens: int):
    """Per-token visible counts to the left/right within each [IMAGE_START, IMAGE_END] span."""
    seqlen = input_ids.size(1)
    idx = torch.arange(seqlen, dtype=torch.int32).unsqueeze(0)
    is_start = input_ids == vocab_size + IMAGE_START
    is_end = input_ids == vocab_size + IMAGE_END
    valid = (is_start.cumsum(1) > is_end.cumsum(1)) | is_end
    starts = torch.where(is_start, idx, 0).cummax(1)[0]
    left = (idx - starts) * valid
    ends = torch.where(is_end, idx, seqlen).flip(1).cummin(1)[0].flip(1)
    right = (ends - idx) * valid
    return left.clamp(max=max_image_tokens - 1), right.clamp(max=max_image_tokens)


# ─────────────────────────── model.py:297-305, verbatim ───────────────────────
def get_window_topk_idxs_visible(
    window_size: int,
    seqlen: int,
    left: torch.Tensor,
    right: torch.Tensor,
    max_image_tokens: int,
):
    width = min(seqlen, window_size + max_image_tokens)
    idx = torch.arange(seqlen).unsqueeze(0)
    left_add = (left - (window_size - 1)).clamp(min=0)
    starts = (idx - (window_size - 1) - left_add).clamp(min=0)
    matrix = starts.unsqueeze(-1) + torch.arange(width)
    matrix = torch.where(matrix > (idx + right).unsqueeze(-1), -1, matrix)
    return matrix.int().contiguous()


class RefGate:
    """``model.py:589-639`` ``Gate``, reduced to its pure-torch routing logic.

    ``nn.Module``/``nn.Parameter`` and the FP8 ``Linear`` are dropped (the
    parameters are passed in as plain tensors); ``linear(x.float(),
    self.weight.float())`` becomes ``F.linear`` for float32 weights, which is
    what ``model.py``'s own ``linear()`` dispatches to when the weight is not
    a quantized dtype. Nothing else is changed: the branch structure, the
    order of operations, and which tensor each op reads are transcribed
    literally from ``Gate.forward``.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        topk: int,
        score_func: str,
        route_scale: float,
        is_hash: bool,
        vocab_size: int,
        tid2eid: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        bias_vl: Optional[torch.Tensor] = None,
    ):
        self.weight = weight
        self.topk = topk
        self.score_func = score_func
        self.route_scale = route_scale
        self.hash = is_hash
        self.vocab_size = vocab_size
        self.tid2eid = tid2eid
        self.bias = bias
        self.bias_vl = bias_vl

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        image_mask = (
            (input_ids >= self.vocab_size) if self.bias_vl is not None else None
        )
        # Bias shifts scores for expert selection (topk) but does not affect routing weights.
        if self.hash:
            if image_mask is None:
                indices = self.tid2eid[input_ids]
            else:
                indices = self.tid2eid[torch.where(image_mask, 0, input_ids)]
                vl_indices = (scores + self.bias_vl).topk(self.topk, dim=-1)[1]
                indices = torch.where(
                    image_mask.unsqueeze(-1), vl_indices.to(indices.dtype), indices
                )
        else:
            if image_mask is None:
                scores = scores + self.bias
            else:
                scores = scores + torch.where(
                    image_mask.unsqueeze(-1), self.bias_vl, self.bias
                )
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices
