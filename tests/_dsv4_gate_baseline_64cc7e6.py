# Copyright © 2026 Apple Inc.

"""FROZEN pre-Phase-3 baseline of the DeepSeek-V4 MoE gate.

Extracted VERBATIM by `git show 64cc7e6:mlx_lm/models/deepseek_v4.py` -- the
mlx-lm submodule HEAD immediately before any Phase 3 (bias_vl / image
visibility) change landed. This exists for exactly one purpose: to give
`test_deepseek_v4_gate.py` a genuine, independently-executable "before"
implementation to assert BITWISE identity against, rather than comparing the
post-change code to itself.

DO NOT EDIT. DO NOT IMPORT FROM SHIPPED CODE. If this file ever needs to
change, the text-only no-regression guarantee it underwrites has been broken
and that is the bug, not this file.

Regenerate with:

    git show 64cc7e6:mlx_lm/models/deepseek_v4.py

and re-extract `_score_func`, `_gate_route`, `_hash_gate_route`, `MoEGate`.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.deepseek_v4 import ModelArgs

__all__ = ["MoEGate", "_gate_route", "_hash_gate_route", "_score_func"]


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    if func == "sqrtsoftplus":
        return mx.sqrt(nn.softplus(scores))
    raise ValueError(f"Unsupported DeepSeek-V4 scoring function: {func}")


@mx.compile
def _gate_route(
    x: mx.array,
    weight: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    """Phase H: fold the gate matmul into the compiled expert-select chain.

    Was 2 dispatches (matmul + compiled chain). The matmul output is small
    (B, L, n_experts) so MLX can keep it in registers across the cast +
    score-func + argpartition + take_along_axis chain. Bit-equivalent.
    """
    logits = (x @ weight.T).astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    biased = scores + e_score_correction_bias
    inds = mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _hash_gate_route(
    input_ids: mx.array,
    x: mx.array,
    weight: mx.array,
    tid2eid: mx.array,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    """Phase H: hash-routing variant of `_gate_route` with matmul folded in."""
    logits = (x @ weight.T).astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    inds = tid2eid[input_ids]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.hash = layer_idx < config.num_hash_layers
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.num_experts, self.hidden_dim))
        if self.hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros(
                (self.num_experts,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        if self.hash:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash routing requires input_ids.")
            inds, weights = _hash_gate_route(
                input_ids,
                x,
                self.weight,
                self.tid2eid,
                self.routed_scaling_factor,
                self.norm_topk_prob,
                self.scoring_func,
            )
        else:
            inds, weights = _gate_route(
                x,
                self.weight,
                self.e_score_correction_bias,
                self.top_k,
                self.routed_scaling_factor,
                self.norm_topk_prob,
                self.scoring_func,
            )

        return inds, weights
