# Copyright © 2026 Apple Inc.

import math
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_flatten

from ..profiler import finalize, span
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention

# CPU build-time probe (env-gated MLX_BUILD_PROBE=1). Accumulates per-section
# CPU-wall time across all V4Block fast-path invocations. Reported by
# DeepseekV4Model.__call__ at MLX_BUILD_PROBE_LOG_EVERY decode cycles.
import sys as _bp_sys
import time as _bp_time
_BUILD_PROBE_ENABLED = bool(os.environ.get("MLX_BUILD_PROBE"))
_BUILD_PROBE_LOG_EVERY = int(os.environ.get("MLX_BUILD_PROBE_LOG_EVERY", "8"))
_BUILD_PROBE_PERF = _bp_time.perf_counter
_BUILD_PROBE_ACC: Dict[str, float] = {
    "attn_pre": 0.0,
    "attn": 0.0,
    "post_attn": 0.0,
    "ffn_pre": 0.0,
    "ffn": 0.0,
    "post_ffn": 0.0,
    "layer_count": 0,
    "model_forward_total": 0.0,
    "embed": 0.0,
    "attn_mask": 0.0,
    "final_norm": 0.0,
    "step_count": 0,
}

# Per-mlx-op CPU-dispatch probe (env-gated MLX_OP_PROBE=1).
# Monkey-patches a curated set of hot mlx primitives at module load so every
# call accumulates wall-time (Python wrapper + pybind cross + C++-side
# eager submit, NOT GPU compute — GPU runs async). Reported alongside
# BUILD_PROBE at MLX_BUILD_PROBE_LOG_EVERY steps. Use this to find which
# op classes dominate the un-compiled CPU dispatch budget identified by
# build_probe.
#
# Probed ops (chosen as the hot path through V4Attention.__call__ and the
# FFN body): mx.fast.rope, mx.fast.scaled_dot_product_attention,
# mx.fast.rms_norm, mx.quantized_matmul, mx.matmul, mx.softmax,
# mx.logsumexp, mx.logaddexp, mx.take_along_axis, mx.argpartition,
# mx.einsum, mx.distributed.all_sum.
#
# Each tracked op also gets a call-count so we can compute mean-per-call.
_OP_PROBE_ENABLED = bool(os.environ.get("MLX_OP_PROBE"))

# 2026-05-18 per-fence-point all_sum CPU-wall probe (env-gated EXO_DSV4_ALLSUM_PROBE=1).
# Measures the CPU wall-clock spent inside mx.eval(y) at each fence-taken layer
# in DeepseekV4MoE.__call__ (both the fast path and the span path). Used to
# characterize verify-phase tail variance — see
# .hermes/plans/2026-05-18_1830-dsv4-verify-tail-investigation.md.
#
# Cost when off: a single bool read (the global ref) per fence-taken layer.
# Zero env-var reads in the hot path. Zero impact on the mx.eval call itself.
#
# Cost when on: ~1 us per fence-taken layer (perf_counter pair + dict
# setdefault + list append). The mx.eval CPU-wall it measures is typically
# 0.1-10 ms; ~1 us probe overhead is well within noise.
#
# Output format (per dump cycle, every _ALLSUM_PROBE_LOG_EVERY full forward
# passes through the last layer):
#   [ALLSUM-PROBE pid=N] cycles=N layer=L n=N p50=X.XXms p99=X.XXms max=X.XXms ...
# One line per fence-taken layer. Stats are over the most recent
# _ALLSUM_PROBE_LOG_EVERY forward passes; _ACC is cleared after each dump.
_ALLSUM_PROBE_ENABLED = bool(os.environ.get("EXO_DSV4_ALLSUM_PROBE"))
_ALLSUM_PROBE_LOG_EVERY = int(os.environ.get("EXO_DSV4_ALLSUM_PROBE_LOG_EVERY", "50"))
_ALLSUM_PROBE_ACC: Dict[int, List[float]] = {}    # layer_idx -> list[ms]
_ALLSUM_PROBE_CYCLES: int = 0


def _allsum_probe_dump() -> None:
    """Format and emit per-layer p50/p99/max from _ALLSUM_PROBE_ACC, then reset."""
    import statistics as _ap_stats
    if not _ALLSUM_PROBE_ACC:
        return
    layers = sorted(_ALLSUM_PROBE_ACC.keys())
    lines = [
        f"[ALLSUM-PROBE pid={os.getpid()}] cycles={_ALLSUM_PROBE_CYCLES} "
        f"window={_ALLSUM_PROBE_LOG_EVERY} fence_layers={len(layers)}"
    ]
    for L in layers:
        vals = _ALLSUM_PROBE_ACC[L]
        n = len(vals)
        if n == 0:
            continue
        s = sorted(vals)
        p50 = s[n // 2]
        p99 = s[min(n - 1, int(n * 0.99))]
        mn = s[0]
        mx_ = s[-1]
        mean = sum(vals) / n
        lines.append(
            f"[ALLSUM-PROBE pid={os.getpid()}]   layer={L:3d} n={n:4d} "
            f"mean={mean:6.3f}ms min={mn:6.3f}ms p50={p50:6.3f}ms "
            f"p99={p99:6.3f}ms max={mx_:6.3f}ms"
        )
    _bp_sys.stderr.write("\n".join(lines) + "\n")
    _bp_sys.stderr.flush()
    _ALLSUM_PROBE_ACC.clear()


# 2026-05-13 NOP-probe diagnostic. File-based switch (changed between bench
# runs without restarting the cluster). Reads /tmp/dsv4_nop_targets; cached
# for 1 sec to avoid file IO per layer call. Targets are comma-separated:
#   "indexer"         -> Indexer.__call__ returns zeros (shape (B, L, k))
#   "sparse_attn"     -> SparseCompressedAttention attention output = zeros
#   "compressed_attn" -> CompressedAttention attention output = zeros
#   "moe"             -> DeepseekV4MoE.__call__ returns zeros (shape (B, L, hidden))
#   "all_sum"         -> mx.distributed.all_sum becomes identity (skip reduce)
# Output is GARBAGE — bench tok/s only. Quality intentionally broken.
import time as _nop_time
_NOP_FILE = "/tmp/dsv4_nop_targets"
_nop_cache = [0.0, set()]


def _get_nop_targets():
    now = _nop_time.time()
    if now - _nop_cache[0] < 1.0:
        return _nop_cache[1]
    targets = set()
    try:
        with open(_NOP_FILE) as f:
            targets = {s.strip() for s in f.read().split(",") if s.strip()}
    except Exception:
        pass
    _nop_cache[0] = now
    _nop_cache[1] = targets
    return targets


# Monkey-patch mx.distributed.all_sum to honor the NOP flag.
# Idempotent: guarded by _all_sum_nop_wrapped attribute.
_orig_all_sum = mx.distributed.all_sum
if not getattr(mx.distributed.all_sum, "_all_sum_nop_wrapped", False):
    def _all_sum_nop_aware(x, *args, **kwargs):
        if "all_sum" in _get_nop_targets():
            return x  # NOP: pass through, skip cross-rank reduce
        return _orig_all_sum(x, *args, **kwargs)
    _all_sum_nop_aware._all_sum_nop_wrapped = True
    mx.distributed.all_sum = _all_sum_nop_aware


# Token-tree drafting verify-pass side channel. Set by the exo
# DSv4MTPBatchGenerator BEFORE calling model(verify_input, ...) when a tree
# verify is desired; cleared AFTER. When `tree_mask` is non-None, the
# DeepseekV4Model.__call__ uses it instead of the standard causal mask; and
# the three Attention.__call__ classes use `tree_positions` (instead of the
# implicit `cache.offset + arange(L_q)`) for RoPE.
#
# This is a side-channel rather than a kwarg so we don't have to thread it
# through every layer + attention class signature -- mirrors the existing
# `_captured["pre_norm"]` pattern used by MTPBatchGenerator's wrapped final
# norm. Single-threaded per process: each worker has one inference thread.
#
# tree_mask shape: (L_q, L_kv + L_q) additive (-inf at do-not-attend, 0 at
#   attend). Broadcasts to (B=1, n_heads, L_q, L_k).
# tree_positions shape: (L_q,) int -- the RoPE position of each tree node.
#   Same-depth siblings share a position; depth-d node has position L_kv + d.
_TREE_VERIFY_CTX: Dict[str, Any] = {"mask": None, "positions": None}


def _set_tree_verify_ctx(mask: Optional[mx.array],
                          positions: Optional[mx.array]) -> None:
    """Caller-side helper: install or clear the tree-verify side channel."""
    _TREE_VERIFY_CTX["mask"] = mask
    _TREE_VERIFY_CTX["positions"] = positions


def _tree_pmask(pool_cache, positions: mx.array):
    """Tree-aware drop-in replacement for ``PoolingCache.make_mask``.

    The stock ``PoolingCache.make_mask(L, offset)`` builds a row-causal
    pmask whose row ``j`` uses ``query_idx = offset + j + 1``. That's
    correct for linear-causal input where each query row sits at a
    monotonically increasing absolute position. Token-tree drafting
    violates that assumption: same-depth siblings share an absolute
    position, deeper siblings share a position with each other, so the
    row index is NOT the right cutoff -- the depth (= position - offset)
    is.

    This helper uses the per-token absolute positions from
    ``_TREE_VERIFY_CTX["positions"]`` to build a pmask whose row ``i``
    uses cutoff ``(positions[i] + 1) // pool_cache.ratio``. Same-depth
    siblings get IDENTICAL pmask rows -- which is what the linear-causal
    case satisfies trivially but the tree case must enforce explicitly.

    Returns ``(L_q, P)`` bool mask, or ``None`` when the pool is empty
    (matches ``make_mask`` semantics).
    """
    if pool_cache is None or pool_cache.pooled is None:
        return None
    P = pool_cache.pooled.shape[1]
    pool_idx = mx.arange(P)
    # (positions + 1) // ratio gives each row's per-token cutoff. Cast to
    # int32 to match make_mask's arange dtype and keep the comparison cheap.
    query_idx = (positions + 1).astype(mx.int32)
    return pool_idx < (query_idx[:, None] // pool_cache.ratio)


def _dispatch_pmask(pool_cache, L: int, offset):
    """Pick the right pmask builder: tree-aware when the verify side
    channel is active and the L_q matches, otherwise the stock
    row-causal ``PoolingCache.make_mask``.

    Called at the three sites that consume the pool's row-causal mask:
    CompressedAttention.__call__, SparseCompressedAttention.__call__,
    and Indexer.__call__. Keeps the linear path bit-exact (returns the
    same object the stock call returns) when the side channel is None.
    """
    if pool_cache is None:
        return None
    positions = _TREE_VERIFY_CTX.get("positions")
    if positions is not None and L == positions.shape[0]:
        return _tree_pmask(pool_cache, positions)
    return pool_cache.make_mask(L, offset)


# ROUTE_HIST: per-(layer, expert) routing histogram probe.
_ROUTE_HIST_DIR = "/tmp/dsv4_route_hist"
_route_hist_counts: dict = {}
_route_hist_n_calls: dict = {}

def _route_hist_record(layer_idx: int, inds) -> None:
    import os as _osh, numpy as _np
    try:
        _osh.makedirs(_ROUTE_HIST_DIR, exist_ok=True)
    except Exception:
        return
    try:
        idx_np = _np.asarray(inds, dtype=_np.int64).ravel()
    except Exception:
        try:
            idx_np = _np.asarray(inds.tolist(), dtype=_np.int64).ravel()
        except Exception:
            return
    if layer_idx not in _route_hist_counts:
        _route_hist_counts[layer_idx] = _np.zeros(256, dtype=_np.int64)
        _route_hist_n_calls[layer_idx] = 0
    counts = _route_hist_counts[layer_idx]
    mask = (idx_np >= 0) * (idx_np < counts.shape[0])
    idx_np = idx_np[mask.astype(bool)]
    bc = _np.bincount(idx_np, minlength=counts.shape[0])[: counts.shape[0]]
    counts += bc
    _route_hist_n_calls[layer_idx] += 1
    if (_route_hist_n_calls[layer_idx] % 64) == 0:
        _route_hist_flush(layer_idx)

def _route_hist_flush(layer_idx: int) -> None:
    import os as _osh, numpy as _np
    try:
        counts = _route_hist_counts[layer_idx]
        pid = _osh.getpid()
        out = f"{_ROUTE_HIST_DIR}/L{layer_idx:02d}_pid{pid}.npy"
        _np.save(out, counts)
    except Exception:
        pass

def _route_hist_flush_all() -> None:
    for li in list(_route_hist_counts.keys()):
        _route_hist_flush(li)

import atexit as _atexit
_atexit.register(_route_hist_flush_all)

_OP_PROBE_ACC: Dict[str, float] = {}
_OP_PROBE_COUNT: Dict[str, int] = {}


def _op_probe_install() -> None:
    """Monkey-patch hot mlx primitives to accumulate per-op CPU-wall time.

    Idempotent — guarded by the _patched flag on each wrapped callable so
    repeated calls (e.g. from multiple model instances) don't double-wrap.

    Notes:
      * The wrappers do their own try/finally to be exception-safe.
      * They write to module-level dicts (not thread-local). Single-process
        decode is the only target, so no locking needed.
      * Wall-time captured includes the Python wrapper itself; in practice
        that is sub-microsecond and dwarfed by the pybind cross + C++ side
        eager-submit / shape-validation work.
    """

    def _wrap(mod, name: str, label: str) -> None:
        fn = getattr(mod, name, None)
        if fn is None or getattr(fn, "_op_probe_wrapped", False):
            return

        def _wrapped(*args, _fn=fn, _label=label, **kwargs):
            _t0 = _BUILD_PROBE_PERF()
            try:
                return _fn(*args, **kwargs)
            finally:
                _t1 = _BUILD_PROBE_PERF()
                _OP_PROBE_ACC[_label] = _OP_PROBE_ACC.get(_label, 0.0) + (_t1 - _t0)
                _OP_PROBE_COUNT[_label] = _OP_PROBE_COUNT.get(_label, 0) + 1

        _wrapped._op_probe_wrapped = True
        setattr(mod, name, _wrapped)

    # Top-level mx.* ops
    _wrap(mx, "matmul", "matmul")
    _wrap(mx, "quantized_matmul", "quantized_matmul")
    _wrap(mx, "softmax", "softmax")
    _wrap(mx, "logsumexp", "logsumexp")
    _wrap(mx, "logaddexp", "logaddexp")
    _wrap(mx, "take_along_axis", "take_along_axis")
    _wrap(mx, "argpartition", "argpartition")
    _wrap(mx, "einsum", "einsum")
    _wrap(mx, "concatenate", "concatenate")
    _wrap(mx, "broadcast_to", "broadcast_to")
    _wrap(mx, "where", "where")
    _wrap(mx, "exp", "exp")
    # mx.fast.*
    _wrap(mx.fast, "rope", "fast.rope")
    _wrap(mx.fast, "scaled_dot_product_attention", "fast.sdpa")
    _wrap(mx.fast, "rms_norm", "fast.rms_norm")
    # mx.distributed.*
    _wrap(mx.distributed, "all_sum", "dist.all_sum")
    _wrap(mx.distributed, "all_gather", "dist.all_gather")
    _wrap(mx.distributed, "send", "dist.send")
    _wrap(mx.distributed, "recv_like", "dist.recv_like")


if _OP_PROBE_ENABLED:
    _op_probe_install()


def _op_probe_report() -> str:
    """Build a one-line report of per-op CPU-wall accumulation.

    Sorted by total wall (descending) so the dominant ops appear first.
    """
    if not _OP_PROBE_ENABLED or not _OP_PROBE_ACC:
        return ""
    items = sorted(_OP_PROBE_ACC.items(), key=lambda kv: -kv[1])
    parts = []
    for label, total_s in items:
        count = _OP_PROBE_COUNT.get(label, 0)
        mean_us = (total_s / count * 1e6) if count else 0.0
        parts.append(f"{label}={total_s * 1000:.1f}ms/n={count}/mean={mean_us:.1f}us")
    return " ".join(parts)


from .cache import CacheList, PoolingCache, RotatingKVCache
from .hyper_connection import HyperConnection, HyperHead, hc_expand
from .mla import MultiLinear
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 1.5
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 512
    scoring_func: str = "sqrtsoftplus"
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    tie_word_embeddings: bool = False
    topk_method: str = "noaux_tc"

    def __post_init__(self):
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = (
                [0]
                + [4 if i % 2 else 128 for i in range(max(n - 2, 0))]
                + ([0] if n >= 2 else [])
            )
        self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError(
                "`compress_ratios` must have one entry per hidden layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers."
            )
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {bad}")


def make_quantization_config(model):
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

    flat_modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    experts = {
        k: mxfp4
        for k, _ in flat_modules
        if ".ffn.switch_mlp." in k and k.endswith("_proj")
    }
    shared_experts = {k: mxfp8 for k, _ in flat_modules if ".ffn.shared_experts." in k}
    attn = {
        k: mxfp8 for k, _ in flat_modules if ".attn.w" in k or ".attn.indexer.wq" in k
    }
    # MTP block has two extra Linear projections (e_proj / h_proj) that
    # fuse the embedding and prev-hidden inputs. Upstream stores them
    # in the same FP8 format as attention weights, so apply the same
    # mxfp8 quantization override.
    mtp_proj = {
        k: mxfp8
        for k, _ in flat_modules
        if k.startswith("model.mtp.") and (k.endswith(".e_proj") or k.endswith(".h_proj"))
    }

    return {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        **experts,
        **shared_experts,
        **attn,
        **mtp_proj,
    }


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


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


@mx.compile
def _q_finalize(
    q_proj_out: mx.array,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
    eps: float,
) -> mx.array:
    """Phase H: fuse q's reshape + rms_norm + transpose into one compiled op.

    Was 3 separate dispatches per layer. Bit-equivalent.
    """
    q = q_proj_out.reshape(batch_size, seq_len, n_heads, head_dim)
    q = mx.fast.rms_norm(q, None, eps)
    return q.transpose(0, 2, 1, 3)


@mx.compile
def _o_pre_a(
    out: mx.array,
    batch_size: int,
    o_groups: int,
    seq_len: int,
    head_dim: int,
) -> mx.array:
    """Phase H: fuse pre-wo_a reshape + transpose + flatten."""
    out = out.reshape(batch_size, o_groups, -1, seq_len, head_dim)
    return out.transpose(0, 1, 3, 2, 4).flatten(-2)


@mx.compile
def _o_pre_b(out: mx.array) -> mx.array:
    """Phase H: fuse pre-wo_b transpose + flatten."""
    return out.transpose(0, 2, 1, 3).flatten(-2)


def _try_fuse_two_quantized_linears(
    holder: nn.Module,
    name_a: str,
    name_b: str,
    fused_prefix: str,
) -> bool:
    """Concatenate two same-input QuantizedLinears along the output axis.

    Stores fused weights as ``f"_{fused_prefix}_w" / "_s" / "_b"`` on
    ``holder`` along with ``f"_{fused_prefix}_n"`` (the size of the first
    half) and ``_fused_group_size / _fused_bits / _fused_mode``. Frees
    the original sub-linears by replacing them with empty modules.

    Returns True on success, False if the projections aren't both
    quantized or share incompatible modes/group_sizes/bits. Idempotent.
    """
    a: Any = getattr(holder, name_a)
    b: Any = getattr(holder, name_b)
    for proj_name, proj in ((name_a, a), (name_b, b)):
        for attr in ("weight", "scales", "group_size", "bits", "mode"):
            if not hasattr(proj, attr):
                return False
    if a.group_size != b.group_size or a.bits != b.bits or a.mode != b.mode:
        return False

    a_w = a["weight"]
    b_w = b["weight"]
    a_s = a["scales"]
    b_s = b["scales"]
    a_bias = a.get("biases") if hasattr(a, "get") else None
    b_bias = b.get("biases") if hasattr(b, "get") else None

    setattr(holder, f"_{fused_prefix}_w", mx.concatenate([a_w, b_w], axis=0))
    setattr(holder, f"_{fused_prefix}_s", mx.concatenate([a_s, b_s], axis=0))
    fused_b = (
        mx.concatenate([a_bias, b_bias], axis=0)
        if a_bias is not None and b_bias is not None
        else None
    )
    setattr(holder, f"_{fused_prefix}_b", fused_b)
    setattr(holder, f"_{fused_prefix}_n", int(a_w.shape[0]))
    holder._fused_group_size = int(a.group_size)  # type: ignore[attr-defined]
    holder._fused_bits = int(a.bits)  # type: ignore[attr-defined]
    holder._fused_mode = a.mode  # type: ignore[attr-defined]

    fused_w = getattr(holder, f"_{fused_prefix}_w")
    fused_s = getattr(holder, f"_{fused_prefix}_s")
    mx.eval(fused_w, fused_s)
    if fused_b is not None:
        mx.eval(fused_b)

    setattr(holder, name_a, nn.Module())
    setattr(holder, name_b, nn.Module())
    return True


def _fused_quantized_matmul(holder: nn.Module, fused_prefix: str, x: mx.array):
    """Issue a single quantized_matmul against fused weights and split.

    Returns ``(first_half, second_half)`` where ``first_half`` has
    ``..._{fused_prefix}_n`` columns and ``second_half`` has the rest.
    """
    fused_w = getattr(holder, f"_{fused_prefix}_w")
    fused_s = getattr(holder, f"_{fused_prefix}_s")
    fused_b = getattr(holder, f"_{fused_prefix}_b")
    n = getattr(holder, f"_{fused_prefix}_n")
    out = mx.quantized_matmul(
        x,
        fused_w,
        scales=fused_s,
        biases=fused_b,
        transpose=True,
        group_size=holder._fused_group_size,  # type: ignore[attr-defined]
        bits=holder._fused_bits,  # type: ignore[attr-defined]
        mode=holder._fused_mode,  # type: ignore[attr-defined]
    )
    return out[..., :n], out[..., n:]


@mx.compile
def _moe_post_combine(
    y: mx.array, scores: mx.array, shared_out: mx.array
) -> mx.array:
    """Phase H: fuse moe.weighted_reduce + moe.shared_experts add.

    Was two separate spans (~5.9% + 7.7% = ~13.6% of decode profile each
    with their own dispatches and intermediate eval). Combining into one
    compiled function lets MLX fuse the elementwise multiply, sum-reduce,
    and addition into a single graph node with downstream dispatch.
    Bit-equivalent: same ops, same order.
    """
    return (y * scores[..., None].astype(y.dtype)).sum(-2) + shared_out


class LimitedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _limited_swiglu(gate, x, self.limit)


class DeepseekV4RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
        max_position_embeddings: int = 1048576,
        freq_scale: int = 1,
    ):
        super().__init__()
        self.dims = dims
        self.freq_scale = freq_scale

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max_position_embeddings = scaling_config[
                "original_max_position_embeddings"
            ]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(
                        original_max_position_embeddings / (num_rotations * 2 * math.pi)
                    )
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type: {rope_type}")

        self._freqs = 1.0 / inv_freq
        self._freqs_cache = {}

    def _get_freqs(self, head_dim: int, inverse: bool):
        key = (head_dim, inverse)
        if key not in self._freqs_cache:
            f = self._freqs
            if self.freq_scale != 1:
                f = f / self.freq_scale
            if inverse:
                f = -f
            nope_pairs = (head_dim - self.dims) // 2
            if nope_pairs > 0:
                f = mx.concatenate([mx.full((nope_pairs,), mx.inf), f])
            self._freqs_cache[key] = f
        return self._freqs_cache[key]

    def __call__(
        self,
        x: mx.array,
        offset: Any = 0,
        inverse: bool = False,
    ) -> mx.array:
        head_dim = x.shape[-1]
        freqs = self._get_freqs(head_dim, inverse)
        offset = offset // self.freq_scale if self.freq_scale != 1 else offset
        return mx.fast.rope(
            x,
            head_dim,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=freqs,
        )


def _apply_score_mask(scores: mx.array, mask: Optional[mx.array]) -> mx.array:
    if mask is None:
        return scores
    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask.astype(scores.dtype)


def _rope_with_positions(
    rope: "DeepseekV4RoPE",
    x: mx.array,
    positions: mx.array,
    inverse: bool = False,
) -> mx.array:
    """Apply RoPE with PER-TOKEN positions (not contiguous from offset).

    mx.fast.rope's `offset` arg accepts a scalar OR a per-BATCH vector --
    NOT a per-token vector. For tree-attention we need per-token positions
    (same-depth tree siblings share a position). Trick: reshape so L_q is
    the batch axis, pass per-batch offsets, reshape back.

    Args:
        rope: DeepseekV4RoPE instance (for `_get_freqs` and `freq_scale`).
        x: shape `(B, n_heads, L_q, head_dim)` -- the standard attention
            tensor layout (Q or K or V or attention output).
        positions: shape `(L_q,)` int -- per-token RoPE positions.
        inverse: True to apply the inverse rotation (used post-SDPA on
            output to undo the Q-side rotation; see deepseek_v4 attention
            classes' `out = self.rope(out, offset, inverse=True)` calls).

    Returns: rotated tensor with same shape as `x`.
    """
    B, H, L_q, D = x.shape
    if B != 1:
        raise NotImplementedError(
            "_rope_with_positions: B>1 not supported in v1 (tree drafting "
            "currently is c=1 only)."
        )
    head_dim = D
    freqs = rope._get_freqs(head_dim, inverse)
    # Move L_q to the batch axis so mx.fast.rope's per-batch `offset` vector
    # gives each token its own RoPE position.
    # (1, H, L_q, D) -> (L_q, H, 1, D)
    x_re = x.transpose(2, 1, 0, 3)  # (L_q, H, 1, D)
    # Apply rope; offset shape (L_q,) matches the new batch dim.
    pos = positions
    if rope.freq_scale != 1:
        pos = pos // rope.freq_scale
    out = mx.fast.rope(
        x_re,
        head_dim,
        traditional=True,
        base=None,
        scale=1.0,
        offset=pos,
        freqs=freqs,
    )
    # Reshape back: (L_q, H, 1, D) -> (1, H, L_q, D).
    return out.transpose(2, 1, 0, 3)


def _rope_dispatch(
    rope: "DeepseekV4RoPE",
    x: mx.array,
    offset: Any,
    inverse: bool = False,
) -> mx.array:
    """Dispatch RoPE: tree positions when set, fall through to standard rope.

    Used at attention Q/K/V/out RoPE sites in the 3 DSv4 attention classes
    and the Indexer's Q RoPE. When the tree-verify side channel has
    `positions` set, route to `_rope_with_positions` (per-token positions);
    otherwise call `rope(x, offset, inverse)` as before.

    `x` is expected to be in `(B, H, L_q, D)` layout (the same layout that
    attention classes hand to `self.rope`). For the indexer's Q tensor
    (also `(B, H, L, D)`) the same dispatch applies.
    """
    positions = _TREE_VERIFY_CTX.get("positions")
    if positions is not None and x.shape[2] == positions.shape[0]:
        return _rope_with_positions(rope, x, positions, inverse=inverse)
    return rope(x, offset, inverse=inverse) if inverse else rope(x, offset)


def _extend_mask(mask: Optional[mx.array], pool_mask: Optional[mx.array], N: int):
    if mask is None:
        return None

    if mask.ndim == 2:
        mask = mask[None, None]
    B, H, L, S = mask.shape

    if pool_mask is None:
        pool_mask = mx.ones((B, H, L, N - S), dtype=mx.bool_)
    elif pool_mask.ndim == 2:
        pool_mask = mx.broadcast_to(pool_mask, (B, H, L, N - S))
    elif pool_mask.ndim == 3:
        pool_mask = mx.broadcast_to(pool_mask[:, None], (B, H, L, N - S))

    full_mask = mx.concatenate([mask, pool_mask], axis=-1)

    return full_mask


@partial(mx.compile, shapeless=True)
def _simple_compress_kv(kv, gate, ape, head_dim):
    weights = mx.softmax(gate.astype(mx.float32) + ape, axis=-2)
    weights = weights.astype(kv.dtype)
    return (kv * weights).sum(axis=-2)


@mx.compile
def _overlap_compress_kv(kv, gate, ape, head_dim):
    B, L, R, D = kv.shape

    gate = gate + ape.astype(gate.dtype)

    kv_0 = mx.zeros((B, 1, R, D // 2), dtype=kv.dtype)
    kv_a, kv_b = mx.split(kv, 2, axis=-1)
    kv_a = mx.concatenate([kv_0, kv_a[:, :-1]], axis=1)
    kv = mx.concatenate([kv_a, kv_b], axis=2)

    gate_0 = mx.full((B, 1, R, D // 2), -mx.inf, dtype=kv.dtype)
    gate_a, gate_b = mx.split(gate, 2, axis=-1)
    gate_a = mx.concatenate([gate_0, gate_a[:, :-1]], axis=1)
    gate = mx.concatenate([gate_a, gate_b], axis=2)

    weights = mx.softmax(gate, axis=-2, precise=True)
    return (kv * weights).sum(axis=-2)


@partial(mx.compile, shapeless=True)
def _split_softmax(log_normalizer, logits_a, logits_b, sinks=None):
    if sinks is not None:
        log_normalizer = mx.logaddexp(log_normalizer, sinks)
    weights_a = mx.exp(logits_a - log_normalizer)
    weights_b = mx.exp(logits_b - log_normalizer)
    return weights_a, weights_b


@partial(mx.compile, shapeless=True)
def _sparse_pooled_attention_inner(
    q_scaled: mx.array,
    local_kv: mx.array,
    pooled_gathered: mx.array,
    local_mask: Optional[mx.array],
    pooled_mask: Optional[mx.array],
    sinks_expanded: Optional[mx.array],
) -> mx.array:
    """Inner kernel of _sparse_pooled_attention with all-static shapes.

    All shape variation (pooled.shape[1] grows with decode position) is
    handled by the outer wrapper's take_along_axis. Inside this function:
      - q_scaled: (B, H, L_q, D) — H, L_q, D fixed per workload
      - local_kv: (B, 1, sliding_window, D) — sliding_window fixed
      - pooled_gathered: (B, 1, L_q, k, D) — k=index_topk fixed
      - masks: shapes derived from above, all fixed

    Microbench (sparse_pooled_attn_microbench.py) shows ~13% speedup over
    the un-compiled chain by collapsing ~15 separate op constructions
    into one compile-cache lookup per cycle.
    """
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = _apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)

    pooled_sq = pooled_gathered.squeeze(1)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True)
    )

    local_weights, pooled_weights = _split_softmax(
        normalizer, local_scores, pooled_scores, sinks_expanded,
    )

    out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(q_scaled.dtype)


def _sparse_pooled_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    topk: mx.array,
    local_mask: Optional[mx.array],
    pooled_mask: Optional[mx.array],
    scale: float,
    sinks: Optional[mx.array],
) -> mx.array:
    """Sparse-pooled attention dispatch.

    Pulls take_along_axis (the only op with dynamic shape via pooled.shape[1])
    OUT of the @mx.compile boundary, then calls the static-shape inner
    kernel. This avoids the shapeless broadcast bug that bit the May 9
    attempt at compiling the whole function (97f87c0 → 6c4112a/25a47a1b).

    L_q=1 fast path (May 13 2026): when there's exactly one query position
    (canonical MTP-off decode), concatenate local + per-query-pooled K/V
    into one tensor and dispatch through ``mx.fast.scaled_dot_product_attention``
    instead of the hand-rolled split-softmax in the inner kernel. Apple's
    optimized SDPA Metal kernel fuses score-matmul + softmax + value-matmul,
    and using fp32 internally for the accumulation makes the bf16 output
    numerically CLOSER to fp32 reference than the manual code (microbench
    bench/sparse_pooled_refactor_microbench.py: max abs diff vs fp32 ref
    drops from 0.012 (current) to 0.004 (proposed), and wall drops 1.24x).

    The fast path is gated on L_q==1 because at L_q>1 each query position
    has its OWN gathered pooled K/V (different topk per row), which the
    single-SDPA approach can't express without expensive broadcasting.
    MTP-on (L_q=γ+1>=2) falls through to the inner kernel as before.
    """
    B, H, L, D = q.shape

    # L_q=1 fast path: concat-and-fused-sdpa
    if L == 1:
        # pooled_gathered shape: (B, 1, 1, k, D) — squeeze the L axis at axis=2
        idx = topk[:, None, :, :, None]  # (B, 1, L=1, k, 1)
        pooled_gathered = mx.take_along_axis(
            mx.broadcast_to(pooled[:, None, None], (B, 1, L, pooled.shape[1], D)),
            mx.broadcast_to(idx, idx.shape[:-1] + (D,)),
            axis=3,
        )
        # (B, 1, 1, k, D) -> (B, 1, k, D)
        pooled_kv = pooled_gathered.squeeze(2)
        # Concat along seq axis: local_kv (B, 1, sw, D) + pooled_kv (B, 1, k, D)
        combined_kv = mx.concatenate([local_kv, pooled_kv], axis=2)

        # Merge masks if either is present. local_mask comes from the
        # global attention mask (B, 1, L, sw)-ish — broadcast across H.
        # pooled_mask is per-head (B, H, L, k). Both need to be expanded
        # to (B, H, L, *) before concat. Also handle additive (fp) vs
        # boolean masks — fast.sdpa accepts either as long as types match.
        combined_mask: Optional[mx.array] = None
        if local_mask is not None or pooled_mask is not None:
            sw = local_kv.shape[2]
            k = pooled_kv.shape[2]
            target_dtype = (
                local_mask.dtype if local_mask is not None else pooled_mask.dtype
            )
            target_is_bool = target_dtype == mx.bool_

            def _full(shape):
                if target_is_bool:
                    return mx.ones(shape, dtype=mx.bool_)
                return mx.zeros(shape, dtype=target_dtype)

            lm = local_mask if local_mask is not None else _full((B, H, L, sw))
            pm = pooled_mask if pooled_mask is not None else _full((B, H, L, k))
            # Broadcast head axis if needed
            if lm.shape[1] == 1 and H > 1:
                lm = mx.broadcast_to(lm, (B, H, L, sw))
            if pm.shape[1] == 1 and H > 1:
                pm = mx.broadcast_to(pm, (B, H, L, k))
            # Coerce dtypes to match for concat
            if lm.dtype != pm.dtype:
                if target_is_bool:
                    lm = lm.astype(mx.bool_)
                    pm = pm.astype(mx.bool_)
                else:
                    lm = lm.astype(target_dtype)
                    pm = pm.astype(target_dtype)
            combined_mask = mx.concatenate([lm, pm], axis=-1)

        return mx.fast.scaled_dot_product_attention(
            q,
            combined_kv,
            combined_kv,
            scale=scale,
            mask=combined_mask,
            sinks=sinks,
        )

    # L_q>1 fallback (e.g. MTP-on verify pass): use inner kernel as before
    idx = topk[:, None, :, :, None]
    pooled_gathered = mx.take_along_axis(
        mx.broadcast_to(pooled[:, None, None], (B, 1, L, pooled.shape[1], D)),
        mx.broadcast_to(idx, idx.shape[:-1] + (D,)),
        axis=3,
    )
    sinks_expanded = sinks[None, :, None, None] if sinks is not None else None
    return _sparse_pooled_attention_inner(
        q * scale,
        local_kv,
        pooled_gathered,
        local_mask,
        pooled_mask,
        sinks_expanded,
    )


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


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        intermediate_size: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def fuse_gate_up_weights(self) -> None:
        """Concatenate gate_proj + up_proj weights along the output axis.

        Phase H: lets ``__call__`` issue a single quantized matmul instead
        of two for the gate/up projections. Saves one Metal dispatch per
        DSv4 layer per decode token (60 dispatches/cycle on DSv4-Flash).
        Bit-equivalent: ``concat(x@G.T, x@U.T) == x @ concat(G, U).T``.

        Idempotent. Requires both projections quantized with the same
        group_size / bits / mode (true for DSv4: both mxfp8). Frees
        gate_proj/up_proj weights after fusion to keep memory flat.
        """
        gp: Any = self.gate_proj
        up: Any = self.up_proj
        for proj_name, proj in (("gate_proj", gp), ("up_proj", up)):
            for attr in ("weight", "scales", "group_size", "bits", "mode"):
                if not hasattr(proj, attr):
                    return  # not quantized — skip silently
        if (
            gp.group_size != up.group_size
            or gp.bits != up.bits
            or gp.mode != up.mode
        ):
            return

        gp_w = gp["weight"]
        up_w = up["weight"]
        gp_s = gp["scales"]
        up_s = up["scales"]
        gp_b = gp.get("biases") if hasattr(gp, "get") else None
        up_b = up.get("biases") if hasattr(up, "get") else None

        self._fused_gu_w = mx.concatenate([gp_w, up_w], axis=0)
        self._fused_gu_s = mx.concatenate([gp_s, up_s], axis=0)
        self._fused_gu_b = (
            mx.concatenate([gp_b, up_b], axis=0)
            if gp_b is not None and up_b is not None
            else None
        )
        self._fused_gu_n = int(gp_w.shape[0])
        self._fused_group_size = int(gp.group_size)
        self._fused_bits = int(gp.bits)
        self._fused_mode = gp.mode
        mx.eval(self._fused_gu_w, self._fused_gu_s)
        if self._fused_gu_b is not None:
            mx.eval(self._fused_gu_b)

        # Free originals — gate_proj/up_proj are now dead weight.
        self.gate_proj = nn.Module()
        self.up_proj = nn.Module()

    def __call__(self, x: mx.array) -> mx.array:
        if hasattr(self, "_fused_gu_w"):
            gu = mx.quantized_matmul(
                x,
                self._fused_gu_w,
                scales=self._fused_gu_s,
                biases=self._fused_gu_b,
                transpose=True,
                group_size=self._fused_group_size,
                bits=self._fused_bits,
                mode=self._fused_mode,
            )
            n = self._fused_gu_n
            x_gate = gu[..., :n]
            x_up = gu[..., n:]
            return self.down_proj(_limited_swiglu(x_gate, x_up, self.swiglu_limit))
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Lever 6 (2026-05-09): cross-rank fence batching.
        # By default (N=1), fence after every layer's all_sum — Phase H Lever 1
        # behavior, required for cross-rank lockstep at long c=2 context.
        # With N>=2, fence only every Nth layer (plus the final layer,
        # whose output flows to lm_head). Trades cross-rank lockstep
        # frequency for fewer GPU/CPU sync points per cycle. Safe ceiling
        # is empirical — too aggressive (N too large) lets graph-position
        # drift accumulate across ranks and JACCL ack barriers wedge.
        # See `dsv4_v4block_compile_2026_05_08.md` for the all_sum-inside-
        # compile case (effectively N=∞) collapsing c=2 100K to 7.7 tok/s.
        import os as _os
        self._fence_every_n = max(1, int(
            _os.environ.get("EXO_DSV4_FENCE_EVERY_N_LAYERS", "1")
        ))
        self._num_total_layers = int(config.num_hidden_layers)
        self.gate = MoEGate(config, layer_idx)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = DeepseekV4MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.sharding_group = None
        self._compiled_forward: Optional[Any] = None

    def install_compiled_forward(self) -> None:
        """Phase H mx.compile: trace the FFN body once and reuse on every
        subsequent decode step.

        The compile boundary covers the **pure local compute** —
        gate / switch_mlp / shared_experts / post_combine. The cross-rank
        ``mx.distributed.all_sum`` is held outside compile so the
        post-allreduce ``mx.eval`` fence (the same one in the span path)
        can fire and force cross-rank lockstep. At c=2 long context
        without the fence, JACCL ack barriers wait on the slowest rank
        and per-stream throughput collapses (~7-8 tok/s vs ~17 with
        fence). Idempotent. Call after weights are loaded and
        ``sharding_group`` is set so the compile boundary closes over the
        final weight identities.
        """
        if self._compiled_forward is not None:
            return
        self._compiled_forward = mx.compile(self._raw_local)

    def _raw_local(self, x: mx.array, input_ids: mx.array) -> mx.array:
        """Pure local compute portion of the MoE forward — no collective.

        This is the part inside the ``mx.compile`` boundary. The
        cross-rank ``all_sum`` happens outside (in ``__call__`` and in
        ``DeepseekV4Block`` callers) so the post-allreduce eval fence can
        fire without poisoning the compile cache.
        """
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, scores = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds)
        shared_out = self.shared_experts(x)
        return _moe_post_combine(y, scores, shared_out)

    def _raw_forward(self, x: mx.array, input_ids: mx.array) -> mx.array:
        """Pure compute + cross-rank allreduce, eval-free.

        Kept for callers (e.g. ``DeepseekV4Block._raw_ffn_section``) that
        wrap the entire FFN in their own outer compile and provide the
        cross-rank fence themselves.
        """
        y = self._raw_local(x, input_ids)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        # 2026-05-18 allsum probe: declare _ALLSUM_PROBE_CYCLES as global so
        # the per-branch increments at the two fence sites below are well-
        # formed regardless of which branch runs first. Free when probe off.
        global _ALLSUM_PROBE_CYCLES
        if "moe" in _get_nop_targets():
            return mx.zeros(x.shape, dtype=x.dtype)
        # ROUTE_HIST probe: EXO_DSV4_ROUTE_HIST=1 records expert routing.
        # EXO_DSV4_ROUTE_HIST_DECODE_ONLY=1 records only L==1 calls (decode path).
        # Prefill calls have L > 1; decode calls have L == 1 (or sometimes L == γ+1 for MTP).
        import os as _ros
        if _ros.environ.get("EXO_DSV4_ROUTE_HIST", "0") == "1":
            try:
                _decode_only = _ros.environ.get("EXO_DSV4_ROUTE_HIST_DECODE_ONLY", "0") == "1"
                _record = True
                if _decode_only:
                    # x shape is (B, L, D). Decode = L == 1.
                    _record = (x.ndim >= 2 and x.shape[-2] == 1)
                if _record:
                    _ri, _ = self.gate(x, input_ids)
                    _route_hist_record(self.layer_idx, _ri)
            except Exception:
                pass
        # Fast path: pre-compiled local trace + Python-level allreduce
        # with the fence eval. Mirrors the span-path semantics.
        if self._compiled_forward is not None:
            y = self._compiled_forward(x, input_ids)
            if self.sharding_group is not None:
                y = mx.distributed.all_sum(y, group=self.sharding_group)
                # Cross-rank lockstep fence — see install_compiled_forward
                # docstring. Without this, JACCL ack barriers serialize
                # on the slowest rank at long c=2 context.
                # Lever 6: with EXO_DSV4_FENCE_EVERY_N_LAYERS>=2, only fence
                # every Nth layer (plus the final layer). Reduces sync
                # points per cycle.
                _is_last = self.layer_idx == self._num_total_layers - 1
                _is_fence_idx = (self.layer_idx % self._fence_every_n) == (
                    self._fence_every_n - 1
                )
                if _is_last or _is_fence_idx:
                    if _ALLSUM_PROBE_ENABLED:
                        import time as _ap_t
                        _t0 = _ap_t.perf_counter()
                        mx.eval(y)
                        _ms = (_ap_t.perf_counter() - _t0) * 1000.0
                        _ALLSUM_PROBE_ACC.setdefault(
                            self.layer_idx, []
                        ).append(_ms)
                        if _is_last:
                            _ALLSUM_PROBE_CYCLES += 1
                            if (
                                _ALLSUM_PROBE_CYCLES
                                % _ALLSUM_PROBE_LOG_EVERY
                                == 0
                            ):
                                _allsum_probe_dump()
                    else:
                        mx.eval(y)
            return y

        with span("ffn"):
            if self.sharding_group is not None:
                x = sum_gradients(self.sharding_group)(x)

            with span("moe.gate"):
                inds, scores = self.gate(x, input_ids)
                finalize(inds)
                finalize(scores)

            with span("moe.switch_mlp"):
                y = finalize(self.switch_mlp(x, inds))

            with span("moe.post_combine"):
                # Phase H: fused weighted_reduce + shared_experts add via
                # @mx.compile (_moe_post_combine). Was two separate spans
                # before. shared_experts forward (the matmul itself) stays
                # separate; we fuse only the y-side combine, which is
                # the elementwise + sum + add path.
                shared_out = self.shared_experts(x)
                y = finalize(_moe_post_combine(y, scores, shared_out))

            if self.sharding_group is not None:
                with span("moe.all_sum"):
                    y = mx.distributed.all_sum(y, group=self.sharding_group)
                    # Phase H Lever 1 (2026-05-06): force evaluation of the
                    # collective output before any subsequent layer reads
                    # `y`. The all_sum itself is bit-deterministic across
                    # ranks, but a lazy graph can let two ranks dispatch
                    # the next MoE layer with subtly-different inputs if
                    # GPU stragglers cause the all_sum to be evaluated at
                    # different graph positions per rank. mx.eval flushes
                    # that ordering window. Required for the re-sharded
                    # MTP MoE path (auto_parallel.py:935 Phase H Lever 1)
                    # to remain bit-equivalent across ranks at c=2 temp=0.
                    if _ALLSUM_PROBE_ENABLED:
                        import time as _ap_t
                        _t0 = _ap_t.perf_counter()
                        mx.eval(y)
                        _ms = (_ap_t.perf_counter() - _t0) * 1000.0
                        _ALLSUM_PROBE_ACC.setdefault(
                            self.layer_idx, []
                        ).append(_ms)
                        _is_last_span = (
                            self.layer_idx
                            == self._num_total_layers - 1
                        )
                        if _is_last_span:
                            _ALLSUM_PROBE_CYCLES += 1
                            if (
                                _ALLSUM_PROBE_CYCLES
                                % _ALLSUM_PROBE_LOG_EVERY
                                == 0
                            ):
                                _allsum_probe_dump()
                    else:
                        mx.eval(y)
                    y = finalize(y)
            return y


class Compressor(nn.Module):

    def __init__(self, config: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
            freq_scale=compress_ratio,
        )

    def fuse_kv_gate_weights(self) -> None:
        """Concatenate wkv + wgate weights along the output axis.

        Phase H: lets ``__call__`` issue a single quantized matmul instead
        of two for the kv/gate projections. Saves one Metal dispatch per
        compressor per decode token. Compressors are NOT sharded, so this
        runs as one local quantized_matmul per rank. Bit-equivalent.
        """
        wkv: Any = self.wkv
        wgate: Any = self.wgate
        for proj_name, proj in (("wkv", wkv), ("wgate", wgate)):
            for attr in ("weight", "scales", "group_size", "bits", "mode"):
                if not hasattr(proj, attr):
                    return  # not quantized — skip
        if (
            wkv.group_size != wgate.group_size
            or wkv.bits != wgate.bits
            or wkv.mode != wgate.mode
        ):
            return

        kv_w = wkv["weight"]
        g_w = wgate["weight"]
        kv_s = wkv["scales"]
        g_s = wgate["scales"]
        kv_b = wkv.get("biases") if hasattr(wkv, "get") else None
        g_b = wgate.get("biases") if hasattr(wgate, "get") else None

        self._fused_kg_w = mx.concatenate([kv_w, g_w], axis=0)
        self._fused_kg_s = mx.concatenate([kv_s, g_s], axis=0)
        self._fused_kg_b = (
            mx.concatenate([kv_b, g_b], axis=0)
            if kv_b is not None and g_b is not None
            else None
        )
        self._fused_kg_n = int(kv_w.shape[0])
        self._fused_group_size = int(wkv.group_size)
        self._fused_bits = int(wkv.bits)
        self._fused_mode = wkv.mode
        mx.eval(self._fused_kg_w, self._fused_kg_s)
        if self._fused_kg_b is not None:
            mx.eval(self._fused_kg_b)

        self.wkv = nn.Module()
        self.wgate = nn.Module()

    def _project_kv_gate(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        if hasattr(self, "_fused_kg_w"):
            gu = mx.quantized_matmul(
                x,
                self._fused_kg_w,
                scales=self._fused_kg_s,
                biases=self._fused_kg_b,
                transpose=True,
                group_size=self._fused_group_size,
                bits=self._fused_bits,
                mode=self._fused_mode,
            )
            n = self._fused_kg_n
            return gu[..., :n], gu[..., n:]
        return self.wkv(x), self.wgate(x)

    def __call__(
        self,
        x: mx.array,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
    ) -> mx.array:
        B, _, _ = x.shape
        kv, gate = self._project_kv_gate(x)
        if pool_cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = offset
        else:
            ready_kv, ready_gate, pool_base = pool_cache.accumulate_windows(
                kv, gate, offset
            )

        if ready_kv.size == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            compress_func = (
                _overlap_compress_kv if self.overlap else _simple_compress_kv
            )
            kv = mx.unflatten(ready_kv, 1, (-1, self.compress_ratio))
            gate = mx.unflatten(ready_gate, 1, (-1, self.compress_ratio))
            new_pooled = compress_func(kv, gate, self.ape, self.head_dim)
            new_pooled = self.norm(new_pooled)
            # Phase I.b (2026-05-12): rope expects (..., L, D); the leading
            # axes can be any rank. The original code did
            # ``self.rope(new_pooled[:, None], offset=...).squeeze(1)``
            # which inserts then removes a unit axis. mx.fast.rope works
            # on (B, L, D) directly, so we can drop the unsqueeze/squeeze
            # pair and save two array ops per call.
            new_pooled = self.rope(new_pooled, offset=pool_base)

        if pool_cache is not None:
            new_pooled = pool_cache.update_and_fetch(new_pooled)

        return new_pooled




# ─────────── Fused top-K kernel (EXO_DSV4_TOPK_FUSED=1) ───────────
# Replaces ``mx.argsort(-scores, axis=-1)[..., :k]`` in Indexer.__call__.
# Numerical accuracy: 99.8% top-K overlap vs argsort (bf16 ULP drift at the
# K boundary, same character as variant_d _indexer_score transform).
# Microbench at production shape (B=1, L=1, P=25000, K=160) shows ~5.5x
# pipelined chain speedup on m4-1 (60us/call -> 11us/call).
# Pool size P is passed as a runtime uniform so a single Metal pipeline
# handles all pool sizes — no shapeless-compile cache blowup.

import os as _topk_os

_TOPK_KERNEL_CACHE = {}

def _topk_kernel_metal_source():
    return """uint tid = thread_position_in_threadgroup.x;
uint bl  = threadgroup_position_in_grid.x;
uint b   = bl / L_;
uint l   = bl % L_;

uint P_RT = p_runtime[0];

uint sc_off  = b * (L_ * P_RT) + l * P_RT;
uint out_off = b * (L_ * K_) + l * K_;

float local_score[K_LOCAL_];
int   local_idx  [K_LOCAL_];
for (uint i = 0; i < K_LOCAL_; ++i) {
    local_score[i] = -INFINITY;
    local_idx[i]   = -1;
}
float local_min = -INFINITY;
uint  local_min_slot = 0;

for (uint p = tid; p < P_RT; p += T_) {
    float sc = float(scores[sc_off + p]);
    if (sc > local_min) {
        local_score[local_min_slot] = sc;
        local_idx  [local_min_slot] = (int)p;
        local_min = local_score[0];
        local_min_slot = 0;
        for (uint i = 1; i < K_LOCAL_; ++i) {
            if (local_score[i] < local_min) {
                local_min = local_score[i];
                local_min_slot = i;
            }
        }
    }
}

threadgroup float tg_score[CANDIDATES_];
threadgroup int   tg_idx  [CANDIDATES_];
for (uint i = 0; i < K_LOCAL_; ++i) {
    uint slot = tid * K_LOCAL_ + i;
    tg_score[slot] = local_score[i];
    tg_idx  [slot] = local_idx[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint k_stride = 2; k_stride <= CANDIDATES_; k_stride <<= 1) {
    for (uint j_stride = k_stride >> 1; j_stride > 0; j_stride >>= 1) {
        for (uint i = tid; i < CANDIDATES_; i += T_) {
            uint ixj = i ^ j_stride;
            if (ixj > i) {
                bool ascending = ((i & k_stride) == 0);
                float si = tg_score[i];
                float sj = tg_score[ixj];
                bool swap_it = ascending ? (si < sj) : (si > sj);
                if (swap_it) {
                    tg_score[i]   = sj;
                    tg_score[ixj] = si;
                    int ti = tg_idx[i];
                    tg_idx[i]   = tg_idx[ixj];
                    tg_idx[ixj] = ti;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

for (uint i = tid; i < K_; i += T_) {
    out_idx[out_off + i] = tg_idx[i];
}
"""

def _get_topk_kernel(K_baked: int):
    """Build (or fetch cached) top-K kernel for the given K."""
    if K_baked in _TOPK_KERNEL_CACHE:
        return _TOPK_KERNEL_CACHE[K_baked]
    T_threads = 256
    K_local = 4
    candidates = T_threads * K_local
    if K_baked > candidates:
        # K too large — fall back to argsort path (return None)
        _TOPK_KERNEL_CACHE[K_baked] = None
        return None
    k = mx.fast.metal_kernel(
        name=f"dsv4_topk_K{K_baked}",
        input_names=["scores", "p_runtime"],
        output_names=["out_idx"],
        source=_topk_kernel_metal_source(),
        header=f"""
        constant uint L_ = 1;
        constant uint K_ = {K_baked};
        constant uint T_ = {T_threads};
        constant uint K_LOCAL_ = {K_local};
        constant uint CANDIDATES_ = {candidates};
        """,
        ensure_row_contiguous=True,
    )
    _TOPK_KERNEL_CACHE[K_baked] = k
    return k

def _fused_topk(scores: mx.array, k: int):
    """Return (B, L, k) int32 indices of top-k scores along axis -1.

    Assumes scores shape (B, L=1, P). Falls back to None if k > 1024
    (kernel can't support it — caller must use argsort).
    """
    B_runtime = scores.shape[0]
    P_runtime = scores.shape[-1]
    kernel = _get_topk_kernel(k)
    if kernel is None:
        return None
    p_arr = mx.array([P_runtime], dtype=mx.uint32)
    outs = kernel(
        inputs=[scores, p_arr],
        grid=(256 * B_runtime, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B_runtime, 1, k)],
        output_dtypes=[mx.int32],
    )
    return outs[0]


@partial(mx.compile, shapeless=True)
def _indexer_score(
    q: mx.array,
    pooled: mx.array,
    weights_x: mx.array,
    scale: float,
    n_heads_inv_sqrt: float,
):
    """Compiled score-and-collapse for the DSv4 Indexer hot path.

    `pooled.shape[1]` grows by 1 every `compress_ratio` decode tokens, so
    `shapeless=True` is required — without it MLX recompiles a fresh Metal
    pipeline per distinct pool size, accumulates all of them in the
    process-wide compile cache (no eviction), and OOMs at ~94K decoded
    Think-mode tokens with ~24K cached pipelines.

    Replaces lines:
      scores = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
      scores = mx.maximum(scores, 0) * self.scale
      weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads**-0.5)
      scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
    Runs every decode step on every indexer-equipped layer (~21 layers).
    """
    # Drops three explicit .astype(mx.float32) casts. MLX's bf16 GEMM
    # accumulates in fp32 internally so matmul precision is preserved up
    # to the final downcast. mx.argpartition top-k is robust to small
    # score perturbations at the cutoff (microbench: 98.6% top-192 overlap
    # vs fp32 reference, max abs diff ~0.015 at bf16 epsilon scale).
    #
    # This restores the fix originally landed in f4dd9e7 (+3.4% decode at
    # 100K per fork-notes) — 2e099bd silently undid it when wrapping in
    # mx.compile. See bench/indexer_score_microbench.py for microbench.
    #
    # CAUTION: under MTP self-spec (EXO_DSV4_MTP=1), this perturbation
    # reduces draft/verify agreement by ~9% (1.04 → 0.95 mean acceptance
    # on c=1 100K) — the cycle is 1.6% faster but generates 4.6% fewer
    # accepted tokens, net -1.4% decode tps. Under MTP-off (the canonical
    # tuning configuration), it is a pure win: same kernel speedup, no
    # acceptance to lose.
    # 2026-05-13 refactor: replace the H-reduce elementwise-mul+sum with a
    # batched matmul, and pre-multiply scale*n_heads_inv_sqrt into weights
    # once (instead of once on the (B,H,L,P) scores tensor and once on the
    # (B,L,H) weights). Microbench at production shape (B=1 H=64 L=1 D=128
    # P=25000 bf16):
    #   baseline:                       0.446 ms/call
    #   variant_d (this code):          0.213 ms/call  (~2.1x faster)
    # Top-K agreement at the cutoff: 159-160/160 across 15 random trials,
    # with all disagreements being score-ties within 1% of the cutoff score
    # (i.e. the partition arbitrarily picks one of two ties -- same character
    # as the previously-validated bf16 cast removal). Bit-equivalent at bf16
    # precision; max abs diff vs baseline = 1.5e-2 = 1 bf16 ulp.
    # See bench/indexer_score_microbench.py and bench/indexer_fused_microbench.py.
    pf = pooled[:, None]
    scores = q @ pf.swapaxes(-1, -2)               # (B, H, L, P)
    scores = mx.maximum(scores, 0)                  # (B, H, L, P), ReLU
    # Fold scale*n_heads_inv_sqrt into the per-head weights (cheap (B,L,H))
    # instead of multiplying the larger (B,H,L,P) scores tensor by scale.
    w = (weights_x * (scale * n_heads_inv_sqrt))[..., None]   # (B, L, H, 1)
    # H reduce via a batched matmul: (B,L,P,H) @ (B,L,H,1) -> (B,L,P,1)
    scores_blph = scores.transpose(0, 2, 3, 1)      # (B, L, P, H)
    return (scores_blph @ w).squeeze(-1)            # (B, L, P)


class Indexer(nn.Module):
    def __init__(self, config: ModelArgs, compress_ratio: int):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        # EXO_DSV4_INDEX_TOPK overrides the model-config default. Useful for
        # decode perf tuning — lower topk reduces SDPA work per indexer step
        # at the cost of attention coverage. Validated quality-neutral at 192
        # on AIME for DSv4-Flash-6bit.
        import os as _os
        _topk_env = _os.environ.get("EXO_DSV4_INDEX_TOPK")
        self.index_topk = int(_topk_env) if _topk_env else config.index_topk
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        position_rope: DeepseekV4RoPE,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
    ):
        B, L, _ = x.shape
        pooled = self.compressor(x, pool_cache, offset)
        if pooled.shape[1] == 0:
            return None

        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = _rope_dispatch(position_rope, q, offset)

        scores = _indexer_score(
            q,
            pooled,
            self.weights_proj(x),
            self.scale,
            self.n_heads**-0.5,
        )
        # Tree-aware pmask dispatch: same-depth tree siblings need IDENTICAL
        # pmask rows, but make_mask is row-causal-by-row-index. See
        # ``_tree_pmask`` for the fix; linear path is bit-exact.
        pmask = _dispatch_pmask(pool_cache, L, offset)
        if pmask is not None:
            scores = mx.where(
                pmask if pmask.ndim == 3 else pmask[None],
                scores,
                mx.finfo(scores.dtype).min,
            )
        k = min(self.index_topk, pooled.shape[1])
        # EXO_DSV4_TOPK_FUSED=1: use fused Metal top-K kernel that beats
        # argsort+slice by ~5x at the pipelined chain level (microbench
        # at B=1 L=1 P=25000 K=160: 60us/call -> 11us/call). Falls back
        # to argsort when fused path can't run (large k, or pmask gating
        # for L>1 which the fast-path kernel doesn't handle).
        # File toggle: putting "topk_fused" in /tmp/dsv4_nop_targets enables
        # the fused path live (without restart). "topk_off" disables.
        _topk_targets = _get_nop_targets()
        _topk_enabled = (
            "topk_fused" in _topk_targets
            or (_topk_os.environ.get("EXO_DSV4_TOPK_FUSED", "0") == "1"
                and "topk_off" not in _topk_targets)
        )
        if (_topk_enabled
                and scores.shape[1] == 1
                and pmask is None
                and k <= 1024):
            fused = _fused_topk(scores, k)
            if fused is not None:
                return fused
        # Fallback: 2026-05-13 argsort+slice. Bit-equivalent to argpartition
        # +slice for this shape and ~5% faster on Apple's Metal kernel.
        return mx.argsort(-scores, axis=-1)[..., :k]


class LocalAttention(nn.Module):
    """DeepSeek V4 attention with no KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = 0
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.rope_theta,
            None,
            config.max_position_embeddings,
        )

        self.sharding_group = None

    def fuse_qa_kv_weights(self) -> bool:
        """Phase H: fuse wq_a + wkv weights into a single quantized matmul.

        Both consume the same input ``x`` and are NOT sharded
        (q_lora_rank/head_dim are per-rank duplicated). Concatenating
        along the output axis collapses two ``mx.quantized_matmul``
        dispatches into one per attention block per decode token. The
        downstream q_norm/wq_b path consumes the q_lora half; kv_norm
        the kv half — both unchanged. Bit-equivalent.
        """
        return _try_fuse_two_quantized_linears(self, "wq_a", "wkv", "fused_qkv")

    def _project_qa_kv(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        if hasattr(self, "_fused_qkv_w"):
            return _fused_quantized_matmul(self, "fused_qkv", x)
        return self.wq_a(x), self.wkv(x)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        with span("attn"):
            B, L, _ = x.shape
            offset = cache.offset if cache is not None else 0
            offset = mx.array(offset) if isinstance(offset, mx.array) else offset

            q_lora, kv_pre = self._project_qa_kv(x)
            q = _q_finalize(
                self.wq_b(self.q_norm(q_lora)),
                B, L, self.n_heads, self.head_dim,
                self.config.rms_norm_eps,
            )
            q = _rope_dispatch(self.rope, q, offset)

            kv = self.kv_norm(kv_pre).reshape(B, 1, L, self.head_dim)
            kv = _rope_dispatch(self.rope, kv, offset)
            if cache is not None:
                kv, _ = cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

            with span("attn.sdpa"):
                out = finalize(
                    scaled_dot_product_attention(
                        q,
                        kv,
                        kv,
                        cache=cache,
                        scale=self.scale,
                        mask=mask,
                        sinks=self.attn_sink.astype(q.dtype),
                    )
                )
            out = _rope_dispatch(self.rope, out, offset, inverse=True)

            out = _o_pre_a(out, B, self.o_groups, L, self.head_dim)
            out = self.wo_a(out)
            out = _o_pre_b(out)
            out = self.wo_b(out)

            if self.sharding_group is not None:
                with span("attn.all_sum"):
                    out = finalize(
                        mx.distributed.all_sum(out, group=self.sharding_group)
                    )

            return finalize(out)


class CompressedAttention(nn.Module):
    """DeepSeek V4 attention with pooled KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        # Compressed layers use Yarn-scaled RoPE
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)

        self.sharding_group = None

    def fuse_qa_kv_weights(self) -> bool:
        """Phase H: fuse wq_a + wkv into a single quantized matmul.
        See LocalAttention.fuse_qa_kv_weights for details. Bit-equivalent.
        """
        return _try_fuse_two_quantized_linears(self, "wq_a", "wkv", "fused_qkv")

    def _project_qa_kv(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        if hasattr(self, "_fused_qkv_w"):
            return _fused_quantized_matmul(self, "fused_qkv", x)
        return self.wq_a(x), self.wkv(x)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        with span("attn"):
            B, L, _ = x.shape
            local_cache = cache[0] if cache is not None else None
            pool_cache = cache[1] if cache is not None else None
            offset = local_cache.offset if local_cache is not None else 0
            offset = mx.array(offset) if isinstance(offset, mx.array) else offset

            q_lora, kv_pre = self._project_qa_kv(x)
            q = _q_finalize(
                self.wq_b(self.q_norm(q_lora)),
                B, L, self.n_heads, self.head_dim,
                self.config.rms_norm_eps,
            )
            q = _rope_dispatch(self.rope, q, offset)

            kv = self.kv_norm(kv_pre).reshape(B, 1, L, self.head_dim)
            kv = _rope_dispatch(self.rope, kv, offset)
            if local_cache is not None:
                kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

            # Pool tokens into compressed KV and concatenate with local KV
            with span("attn.compressor"):
                pooled = finalize(self.compressor(x, pool_cache, offset))
            pooled_mask = None
            if pooled.shape[1] > 0:
                # Tree-aware pmask dispatch: see _tree_pmask docstring.
                pooled_mask = _dispatch_pmask(pool_cache, L, offset)
                kv = mx.concatenate([kv, pooled[:, None]], axis=2)

            mask = _extend_mask(mask, pooled_mask, kv.shape[2])

            with span("attn.sdpa"):
                if "compressed_attn" in _get_nop_targets():
                    out = mx.zeros(q.shape, dtype=q.dtype)
                else:
                    out = finalize(
                        scaled_dot_product_attention(
                            q,
                            kv,
                            kv,
                            cache=local_cache,
                            scale=self.scale,
                            mask=mask,
                            sinks=self.attn_sink.astype(q.dtype),
                        )
                    )
            out = _rope_dispatch(self.rope, out, offset, inverse=True)

            out = _o_pre_a(out, B, self.o_groups, L, self.head_dim)
            out = self.wo_a(out)
            out = _o_pre_b(out)
            out = self.wo_b(out)

            if self.sharding_group is not None:
                with span("attn.all_sum"):
                    out = finalize(
                        mx.distributed.all_sum(out, group=self.sharding_group)
                    )

            return finalize(out)


class SparseCompressedAttention(nn.Module):
    """DeepSeek V4 attention with sparse indexed pooled KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
        self.indexer = Indexer(config, self.compress_ratio)

        self.sharding_group = None

    def fuse_qa_kv_weights(self) -> bool:
        """Phase H: fuse wq_a + wkv into a single quantized matmul.
        See LocalAttention.fuse_qa_kv_weights for details. Bit-equivalent.
        """
        return _try_fuse_two_quantized_linears(self, "wq_a", "wkv", "fused_qkv")

    def _project_qa_kv(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        if hasattr(self, "_fused_qkv_w"):
            return _fused_quantized_matmul(self, "fused_qkv", x)
        return self.wq_a(x), self.wkv(x)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        with span("attn"):
            B, L, _ = x.shape
            local_cache = cache[0] if cache is not None else None
            comp_cache = cache[1] if cache is not None else None
            idx_cache = cache[2] if cache is not None else None
            offset = local_cache.offset if local_cache is not None else 0
            offset = mx.array(offset) if isinstance(offset, mx.array) else offset

            q_lora, kv_pre = self._project_qa_kv(x)
            q_residual = self.q_norm(q_lora)
            q = _q_finalize(
                self.wq_b(q_residual),
                B, L, self.n_heads, self.head_dim,
                self.config.rms_norm_eps,
            )
            q = _rope_dispatch(self.rope, q, offset)

            kv = self.kv_norm(kv_pre).reshape(B, 1, L, self.head_dim)
            kv = _rope_dispatch(self.rope, kv, offset)
            if local_cache is not None:
                kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

            with span("attn.compressor"):
                pooled = finalize(self.compressor(x, comp_cache, offset))
            # Tree-aware pmask dispatch: see _tree_pmask docstring.
            pmask = _dispatch_pmask(comp_cache, L, offset)
            with span("attn.indexer"):
                if "indexer" in _get_nop_targets():
                    # Indexer returns argsort(-scores)[..., :k] over scores shaped
                    # (B, L, P) so output is (B, L, k). Return deterministic
                    # in-range indices [0, k) so downstream sparse_sdpa doesn't OOB.
                    _topk = self.indexer.index_topk
                    _pool_len = pooled.shape[1] if pooled.shape[1] > 0 else _topk
                    _take = min(_topk, _pool_len)
                    _B, _, _L, _ = q.shape
                    topk = mx.broadcast_to(
                        mx.arange(_take, dtype=mx.int32)[None, None, :],
                        (_B, _L, _take),
                    )
                else:
                    topk = finalize(
                        self.indexer(x, q_residual, self.rope, idx_cache, offset)
                    )
            sinks = self.attn_sink.astype(q.dtype)

            with span("attn.sdpa"):
                # Local attention
                if pooled.shape[1] == 0:
                    out = scaled_dot_product_attention(
                        q,
                        kv,
                        kv,
                        cache=local_cache,
                        scale=self.scale,
                        mask=mask,
                        sinks=sinks,
                    )

                # Compressed attention
                elif pooled.shape[1] <= self.indexer.index_topk:
                    full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
                    mask = _extend_mask(mask, pmask, full_kv.shape[2])
                    out = scaled_dot_product_attention(
                        q,
                        full_kv,
                        full_kv,
                        cache=local_cache,
                        scale=self.scale,
                        mask=mask,
                        sinks=sinks,
                    )

                # Sparse compressed attention
                else:
                    # Per-layer NOP: targets file or env can specify which sparse layers to NOP.
                    # File: any token "sparse_layers:2,4,6" in /tmp/dsv4_nop_targets (1-sec TTL).
                    # Env:  EXO_DSV4_NOP_SPARSE_LAYERS="2,4,6"
                    # Falls back to the global "sparse_attn" NOP target.
                    _targets = _get_nop_targets()
                    _layer_nop = False
                    for _t in _targets:
                        if _t.startswith("sparse_layers:"):
                            try:
                                _ids = set(int(x) for x in _t[len("sparse_layers:"):].split(",") if x.strip())
                                if self.layer_idx in _ids:
                                    _layer_nop = True
                                    break
                            except Exception:
                                pass
                    if not _layer_nop:
                        import os as _ronl
                        _env = _ronl.environ.get("EXO_DSV4_NOP_SPARSE_LAYERS", "")
                        if _env:
                            try:
                                _ids = set(int(x) for x in _env.split(",") if x.strip())
                                _layer_nop = self.layer_idx in _ids
                            except Exception:
                                pass
                    if _layer_nop or "sparse_attn" in _targets:
                        # Skip the expensive sparse SDPA — just return zeros of q shape.
                        out = mx.zeros(q.shape, dtype=q.dtype)
                    else:
                        sparse_mask = None
                        if pmask is not None:
                            sparse_mask = mx.take_along_axis(
                                pmask[None] if pmask.ndim == 2 else pmask,
                                topk,
                                axis=2,
                            )[:, None]
                        out = _sparse_pooled_attention(
                            q,
                            kv,
                            pooled,
                            topk,
                            mask,
                            sparse_mask,
                            self.scale,
                            sinks,
                        )
                out = finalize(out)

            out = _rope_dispatch(self.rope, out, offset, inverse=True)

            out = _o_pre_a(out, B, self.o_groups, L, self.head_dim)
            out = self.wo_a(out)
            out = _o_pre_b(out)
            out = self.wo_b(out)

            if self.sharding_group is not None:
                with span("attn.all_sum"):
                    out = finalize(
                        mx.distributed.all_sum(out, group=self.sharding_group)
                    )

            return finalize(out)


def v4_attention_factory(config: ModelArgs, layer_idx: int) -> nn.Module:
    """Instantiate the appropriate attention module for a given layer."""
    ratio = config.compress_ratios[layer_idx]
    if ratio == 0:
        return LocalAttention(config, layer_idx)
    if ratio == 128:
        return CompressedAttention(config, layer_idx)
    return SparseCompressedAttention(config, layer_idx)


class DeepseekV4Block(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn = v4_attention_factory(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        self._compiled_attn_pre: Optional[Any] = None
        self._compiled_post_attn: Optional[Any] = None
        self._compiled_ffn_pre: Optional[Any] = None
        self._compiled_post_ffn: Optional[Any] = None

    def install_compiled_forward(self) -> None:
        """Phase H+ mx.compile of the layer's pure subsections.

        Splits the layer body around the cache-mutating attention call
        AND around the FFN's cross-rank allreduce so the post-allreduce
        eval fence (in DeepseekV4MoE.__call__) can fire:

          * ``_raw_attn_pre``   — attn_hc + attn_norm
          * ``[uncompiled]``    — attention proper (cache.update_and_fetch)
          * ``_raw_post_attn``  — hc_expand back into the residual
          * ``_raw_ffn_pre``    — ffn_hc + ffn_norm
          * ``[ffn.__call__]``  — MoE body via its own compile + eval fence
          * ``_raw_post_ffn``   — hc_expand back into the residual

        Earlier (76016ec) we tried a single ``_raw_ffn_section`` compile
        that called ``self.ffn._raw_forward`` directly, but that put the
        all_sum inside the V4Block compile boundary and lost the
        cross-rank eval fence — c=2 100K collapsed to ~7.7 tok/s vs
        ~17 with the fence intact. Splitting the FFN restores the
        fence at the cost of one extra compile-cache lookup per layer
        (acceptable: each call is microseconds).

        Idempotent — safe to call multiple times.
        """
        if self._compiled_attn_pre is not None:
            return
        self._compiled_attn_pre = mx.compile(self._raw_attn_pre)
        self._compiled_post_attn = mx.compile(self._raw_post_attn)
        self._compiled_ffn_pre = mx.compile(self._raw_ffn_pre)
        self._compiled_post_ffn = mx.compile(self._raw_post_ffn)

    def _raw_attn_pre(
        self, h: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """HC + RMSNorm fused into one compiled trace.

        Returns ``(normed, residual, post, comb)`` where ``residual``
        is the original ``h`` (kept inside the trace so the post-attn
        hc_expand can read it without a separate Python ref).
        """
        x, post, comb = self.attn_hc(h)
        normed = self.attn_norm(x)
        return normed, h, post, comb

    def _raw_post_attn(
        self,
        attn_out: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
    ) -> mx.array:
        return hc_expand(attn_out, residual, post, comb)

    def _raw_ffn_pre(
        self, h: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """HC + RMSNorm fused into one compiled trace, FFN side."""
        x, post, comb = self.ffn_hc(h)
        normed = self.ffn_norm(x)
        return normed, h, post, comb

    def _raw_post_ffn(
        self,
        ffn_out: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
    ) -> mx.array:
        return hc_expand(ffn_out, residual, post, comb)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        input_ids: mx.array,
    ) -> mx.array:
        if "v4block" in _get_nop_targets():
            return h  # NOP: pass residual through unchanged
        # Fast path — pre-traced compile graphs skip the per-step
        # Python lazy-graph build for the layer's pure chunks. The
        # FFN goes through ``self.ffn`` (MoE.__call__) so its post-allreduce
        # mx.eval fence fires — required for cross-rank lockstep at
        # c=2 long context.
        if self._compiled_attn_pre is not None:
            # Per-section CPU build-time probe (env-gated MLX_BUILD_PROBE=1).
            # Times each of the 6 calls and accumulates into a process-global
            # dict. The probe is no-op when the env var is unset.
            _bp = _BUILD_PROBE_ENABLED
            if _bp:
                _bt0 = _BUILD_PROBE_PERF()
            normed, residual, post, comb = self._compiled_attn_pre(h)
            if _bp:
                _bt1 = _BUILD_PROBE_PERF()
            x = self.attn(normed, mask=mask, cache=cache)
            if _bp:
                _bt2 = _BUILD_PROBE_PERF()
            h = self._compiled_post_attn(x, residual, post, comb)
            if _bp:
                _bt3 = _BUILD_PROBE_PERF()
            normed, residual, post, comb = self._compiled_ffn_pre(h)
            if _bp:
                _bt4 = _BUILD_PROBE_PERF()
            x = self.ffn(normed, input_ids)
            if _bp:
                _bt5 = _BUILD_PROBE_PERF()
            out = self._compiled_post_ffn(x, residual, post, comb)
            if _bp:
                _bt6 = _BUILD_PROBE_PERF()
                _BUILD_PROBE_ACC["attn_pre"] += (_bt1 - _bt0)
                _BUILD_PROBE_ACC["attn"] += (_bt2 - _bt1)
                _BUILD_PROBE_ACC["post_attn"] += (_bt3 - _bt2)
                _BUILD_PROBE_ACC["ffn_pre"] += (_bt4 - _bt3)
                _BUILD_PROBE_ACC["ffn"] += (_bt5 - _bt4)
                _BUILD_PROBE_ACC["post_ffn"] += (_bt6 - _bt5)
                _BUILD_PROBE_ACC["layer_count"] += 1
            return out

        residual = h
        with span("layer.attn_hc"):
            x, post, comb = self.attn_hc(h)
            finalize(x)
        with span("layer.attn_norm"):
            normed = finalize(self.attn_norm(x))
        x = self.attn(normed, mask=mask, cache=cache)
        with span("layer.attn_residual"):
            h = finalize(hc_expand(x, residual, post, comb))

        residual = h
        with span("layer.ffn_hc"):
            x, post, comb = self.ffn_hc(h)
            finalize(x)
        with span("layer.ffn_norm"):
            normed = finalize(self.ffn_norm(x))
        x = self.ffn(normed, input_ids)
        with span("layer.ffn_residual"):
            return finalize(hc_expand(x, residual, post, comb))


class DeepseekV4MTPModule(nn.Module):
    """Single Multi-Token-Prediction head for DSv4 self-speculative decode.

    Structure (matches upstream `mtp.{idx}.*` weights):

      enorm/hnorm  → RMSNorms applied to (embedding, prev_hidden) inputs
      e_proj/h_proj → Linear projections of the normed inputs
      norm         → RMSNorm of (e_proj_out + h_proj_out)
      <body>       → standard DSv4 decoder block:
                       attn_hc, attn_norm, LocalAttention,
                       ffn_hc, ffn_norm, DeepseekV4MoE
      hc_head      → HyperHead reducing hc_mult → 1

    The embedding lookup, final RMSNorm, and lm_head are SHARED with the
    target model and passed in at __call__ time (not owned by this module).

    Forward contract:
      Input:  prev_hidden (B, L, hidden_size) — the target model's
                          post-hc_head, pre-final-norm output at the
                          previous decode step (captured via the
                          MTPBatchGenerator's `_setup_hidden_capture`).
              next_token  (B, L) — the token id sampled at that step.
      Output: logits      (B, L, vocab_size) — predictions for position+1.
    """

    def __init__(self, config: ModelArgs, mtp_idx: int):
        super().__init__()
        self.config = config
        self.mtp_idx = mtp_idx

        self.enorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Body — a standard DSv4 decoder block. Use a layer_idx past
        # num_hash_layers so MoEGate is in non-hash mode (matches the
        # upstream `mtp.0.ffn.gate.bias` weight layout).
        body_layer_idx = config.num_hidden_layers + mtp_idx
        self.attn = LocalAttention(config, body_layer_idx)
        self.ffn = DeepseekV4MoE(config, body_layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

        # Output HyperHead — reduces (B, L, hc_mult, hidden_size) → (B, L, hidden_size).
        self.hc_head = HyperHead(config)

    def make_cache(self):
        """MTP attention is a LocalAttention (compress_ratio=0)."""
        return RotatingKVCache(max_size=self.config.sliding_window)

    def __call__(
        self,
        prev_hidden: mx.array,
        next_token: mx.array,
        embed_tokens: nn.Embedding,
        final_norm: nn.RMSNorm,  # unused for DSv4 — kept for API parity
        lm_head: nn.Linear,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        return_hidden: bool = False,
    ) -> Any:
        del final_norm  # MTP uses its OWN self.norm as the final norm
        # 1. Embed the "current" token and project both inputs.
        #    (Equivalent to DSv3's M_k @ concat(hnorm(h), enorm(e)) when
        #    h_proj + e_proj are seen as the two halves of M_k.) No
        #    intermediate norm here — Qwen3.5's MTPPredictor uses the
        #    same pattern and treats `mtp.norm` as the FINAL norm.
        emb = embed_tokens(next_token)  # (B, L, hidden_size)
        e_normed = self.enorm(emb)
        h_normed = self.hnorm(prev_hidden)
        x = self.e_proj(e_normed) + self.h_proj(h_normed)

        # 2. Broadcast into hc_mult parallel streams (matching main model).
        x = mx.broadcast_to(
            x[:, :, None, :],
            (x.shape[0], x.shape[1], self.config.hc_mult, x.shape[2]),
        )
        x = mx.contiguous(x)

        # 3. Standard DSv4-block body: attn + ffn with hyperconnection.
        residual = x
        x_in, post, comb = self.attn_hc(x)
        x_attn = self.attn(self.attn_norm(x_in), mask=mask, cache=cache)
        x = hc_expand(x_attn, residual, post, comb)

        residual = x
        x_in, post, comb = self.ffn_hc(x)
        x_ffn = self.ffn(self.ffn_norm(x_in), next_token)
        x = hc_expand(x_ffn, residual, post, comb)

        # 4. Reduce hc_mult → 1 via this MTP block's own HyperHead.
        x = self.hc_head(x)  # (B, L, hidden_size)
        # post-hc_head, pre-final-norm hidden state — matches the
        # `pre_norm` capture from the target model so chained draft
        # steps can feed it back in as `prev_hidden`.
        pre_norm_out = x

        # 5. Apply MTP's OWN final norm (loaded from `mtp.{idx}.norm.weight`)
        #    and the shared lm_head. The target model's final_norm is
        #    deliberately bypassed — that one's for the target's main
        #    hidden state, not the MTP head's output. Same pattern as
        #    Qwen3.5 MTPPredictor.predict() in the speculative module.
        x = self.norm(x)
        logits = lm_head(x)

        if return_hidden:
            return logits, pre_norm_out
        return logits


class DeepseekV4Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV4Block(config, idx) for idx in range(config.num_hidden_layers)
        ]
        # MTP heads — created only when the checkpoint actually contains
        # mtp.* weights AND the user has opted in via EXO_DSV4_MTP=1.
        #
        # Two reasons for the env-var gate:
        #   1) The mlx-community 8bit/6bit/4bit conversions ship with
        #      `num_nextn_predict_layers: 1` in config.json but have ZERO
        #      mtp.* keys in the safetensors (sanitize stripped them).
        #      Loading those checkpoints without the gate would create
        #      zero-weight MTP modules — broken if EXO_SPECULATIVE=1.
        #   2) An MTP-included MLX conversion (with `mtp.*` weights)
        #      requires a custom run of mlx_lm.convert that we control;
        #      the gate ensures MTP only activates when the user has
        #      switched to that variant.
        if (
            config.num_nextn_predict_layers > 0
            and os.environ.get("EXO_DSV4_MTP", "0") == "1"
        ):
            self.mtp = [
                DeepseekV4MTPModule(config, i)
                for i in range(config.num_nextn_predict_layers)
            ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        _bp = _BUILD_PROBE_ENABLED
        if _bp:
            _bp_t_start = _BUILD_PROBE_PERF()
        with span("model.embed"):
            h = self.embed_tokens(inputs)
            h = mx.broadcast_to(
                h[:, :, None, :],
                (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
            )
            h = finalize(mx.contiguous(h))
        if _bp:
            _bp_t_post_embed = _BUILD_PROBE_PERF()
            _BUILD_PROBE_ACC["embed"] += (_bp_t_post_embed - _bp_t_start)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList) else first_cache
        )
        with span("model.attn_mask"):
            # Token-tree drafting side channel: when set, skip the standard
            # causal mask and use the caller-supplied per-node tree mask.
            # The tree mask is (L_q, L_kv + L_q) additive; we broadcast it
            # to (1, 1, L_q, L_k) so it works with mx.fast.scaled_dot_product_attention.
            _tree_mask = _TREE_VERIFY_CTX.get("mask")
            if _tree_mask is not None:
                # Shape into the broadcast-friendly 4D layout SDPA expects.
                if _tree_mask.ndim == 2:
                    mask = _tree_mask[None, None, :, :]
                else:
                    mask = _tree_mask
                finalize(mask)
            else:
                mask = create_attention_mask(
                    h[:, :, 0, :],
                    mask_cache,
                    window_size=self.args.sliding_window,
                    return_array=True,
                )
                if mask is not None:
                    finalize(mask)
        if _bp:
            _bp_t_post_mask = _BUILD_PROBE_PERF()
            _BUILD_PROBE_ACC["attn_mask"] += (_bp_t_post_mask - _bp_t_post_embed)

        # NOP TARGET "pipeline": skip all pipeline collectives (recv/send/all_gather).
        # Only valid for diagnostics — output text is meaningless because each rank
        # only sees its own layers' contribution. Used to quantify how much per-token
        # wall is RDMA pipeline sync vs actual compute.
        _nop_pipeline = "pipeline" in _get_nop_targets()

        if not _nop_pipeline and pipeline_rank < pipeline_size - 1:
            with span("model.recv"):
                h = finalize(mx.distributed.recv_like(h, (pipeline_rank + 1)))

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            h = layer(h, mask, layer_cache, inputs)

        if not _nop_pipeline and pipeline_rank != 0:
            with span("model.send"):
                h = finalize(mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size))
                cache_item = cache[-1]
                if isinstance(cache_item, CacheList):
                    cache_item = cache_item[0]
                if cache_item is not None:
                    cache_item.keys = mx.depends(cache_item.keys, h)

        if not _nop_pipeline and pipeline_size > 1:
            with span("model.all_gather"):
                h = finalize(mx.distributed.all_gather(h)[: h.shape[0]])

        with span("model.final_norm"):
            out = finalize(self.norm(self.hc_head(h)))
        if _bp:
            _bp_t_end = _BUILD_PROBE_PERF()
            _BUILD_PROBE_ACC["final_norm"] += (_bp_t_end - _bp_t_post_mask)
            _BUILD_PROBE_ACC["model_forward_total"] += (_bp_t_end - _bp_t_start)
            _BUILD_PROBE_ACC["step_count"] += 1
            sc = _BUILD_PROBE_ACC["step_count"]
            if sc % _BUILD_PROBE_LOG_EVERY == 0:
                lc = max(_BUILD_PROBE_ACC["layer_count"], 1)
                # Per-step averages (sum across all layers, divided by step_count)
                pms = lambda k: _BUILD_PROBE_ACC[k] / sc * 1000
                # Per-layer-call averages (sum / layer_count, in ms)
                pml = lambda k: _BUILD_PROBE_ACC[k] / lc * 1000
                _bp_sys.stderr.write(
                    f"[BUILD_PROBE pid={os.getpid()}] "
                    f"steps={sc} "
                    f"total={pms('model_forward_total'):.2f} "
                    f"embed={pms('embed'):.3f} "
                    f"mask={pms('attn_mask'):.3f} "
                    f"final={pms('final_norm'):.3f} "
                    f"layers/step="
                    f"attn_pre={pms('attn_pre'):.2f} "
                    f"attn={pms('attn'):.2f} "
                    f"post_attn={pms('post_attn'):.2f} "
                    f"ffn_pre={pms('ffn_pre'):.2f} "
                    f"ffn={pms('ffn'):.2f} "
                    f"post_ffn={pms('post_ffn'):.2f} "
                    f"per_layer="
                    f"attn_pre={pml('attn_pre'):.3f} "
                    f"attn={pml('attn'):.3f} "
                    f"post_attn={pml('post_attn'):.3f} "
                    f"ffn_pre={pml('ffn_pre'):.3f} "
                    f"ffn={pml('ffn'):.3f} "
                    f"post_ffn={pml('post_ffn'):.3f}\n"
                )
                if _OP_PROBE_ENABLED:
                    _op_line = _op_probe_report()
                    if _op_line:
                        _bp_sys.stderr.write(
                            f"[OP_PROBE pid={os.getpid()}] steps={sc} {_op_line}\n"
                        )
                _bp_sys.stderr.flush()
        return out


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        if "model_call" in _get_nop_targets():
            B = inputs.shape[0]
            L = inputs.shape[1] if inputs.ndim > 1 else 1
            return mx.zeros((B, L, self.args.vocab_size), dtype=mx.bfloat16)
        h = self.model(inputs, cache)
        with span("model.lm_head"):
            if "lm_head" in _get_nop_targets():
                # Return zeros of the expected output shape (B, L, vocab_size).
                # The shape needs to match what the BatchGenerator expects so
                # logsumexp / argmax don't blow up. Output text will be garbage.
                B = h.shape[0]
                L = h.shape[1]
                return finalize(mx.zeros((B, L, self.args.vocab_size), dtype=mx.bfloat16))
            return finalize(self.lm_head(h))

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return not (
                "attn_sink" in k
                or "e_score_correction_bias" in k
                or ".attn_hc." in k
                or ".ffn_hc." in k
                or ".hc_head." in k
            )

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            ratio = layer.attn.compress_ratio
            if ratio == 0:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            elif isinstance(layer.attn, SparseCompressedAttention):
                # local + compressor pool + indexer pool
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=self.args.sliding_window),
                        PoolingCache(ratio),
                        PoolingCache(ratio),
                    )
                )
            else:
                # local + compressor pool
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=self.args.sliding_window),
                        PoolingCache(ratio),
                    )
                )
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers
        # Only KEEP mtp.* weights when we'll actually have modules to
        # absorb them — see the matching gate in DeepseekV4Model.__init__
        # for the rationale (mlx-community variants advertise
        # num_nextn_predict_layers=1 but ship zero mtp.* keys; the
        # MTP-included variant is opt-in via EXO_DSV4_MTP=1).
        mtp_enabled = (
            self.args.num_nextn_predict_layers > 0
            and os.environ.get("EXO_DSV4_MTP", "0") == "1"
        )
        n_mtp = self.args.num_nextn_predict_layers if mtp_enabled else 0

        new_weights = {}
        for k, v in weights.items():
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    if int(parts[1]) >= n_layers:
                        continue
                except ValueError:
                    pass
            elif len(parts) >= 2 and parts[0] == "mtp":
                # Drop mtp.* if MTP isn't enabled (mlx-community
                # variants without mtp weights, or user hasn't opted
                # in via EXO_DSV4_MTP=1) or the index is out-of-range.
                try:
                    if not mtp_enabled or int(parts[1]) >= n_mtp:
                        continue
                except ValueError:
                    pass
            new_weights[k] = v
        weights = new_weights

        new_weights = {}
        for k, v in weights.items():
            if "tid2eid" in k:
                new_weights[k] = v.astype(mx.int32)

            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".ffn.experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[k + "s"] = v
                new_weights[wk] = weight.view(mx.uint32)
            elif weight.dtype == mx.uint8:
                new_weights[k + "s"] = mx.repeat(mx.repeat(v, 4, -1), 128, 0)
                new_weights[wk] = weight.view(mx.uint32)
            else:
                new_weights[k] = v
        weights = new_weights

        top_remap = {
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        for old, new in top_remap.items():
            if old in weights:
                weights[new] = weights.pop(old)

        # MTP-specific top-level renames (parallel to the main-model
        # hc_head_* renames in `top_remap` above).
        for mtp_idx in range(n_mtp):
            mtp_remap = {
                f"mtp.{mtp_idx}.hc_head_fn": f"model.mtp.{mtp_idx}.hc_head.fn",
                f"mtp.{mtp_idx}.hc_head_base": f"model.mtp.{mtp_idx}.hc_head.base",
                f"mtp.{mtp_idx}.hc_head_scale": f"model.mtp.{mtp_idx}.hc_head.scale",
            }
            for old, new in mtp_remap.items():
                if old in weights:
                    weights[new] = weights.pop(old)

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            if k.startswith("layers.") or k.startswith("mtp."):
                nk = "model." + k
            else:
                nk = k
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[nk] = v
        weights = remapped

        # Stack expert weights for both main layers and MTP blocks.
        for prefix_root, count in (
            ("model.layers", n_layers),
            ("model.mtp", n_mtp),
        ):
            for idx in range(count):
                prefix = f"{prefix_root}.{idx}.ffn.experts"
                for src, dst in (
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ):
                    for suffix in ("weight", "scales"):
                        key0 = f"{prefix}.0.{src}.{suffix}"
                        if key0 in weights:
                            stacked = [
                                weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                                for e in range(self.args.n_routed_experts)
                            ]
                            weights[
                                f"{prefix_root}.{idx}.ffn.switch_mlp.{dst}.{suffix}"
                            ] = mx.stack(stacked)

        # Reshape wo_a from nn.Linear (2D) to MultiLinear (3D) for all
        # layers — including the MTP block(s), whose attention is a
        # LocalAttention with the same wo_a structure.
        for prefix_root, count in (
            ("model.layers", n_layers),
            ("model.mtp", n_mtp),
        ):
            for idx in range(count):
                prefix = f"{prefix_root}.{idx}.attn.wo_a"
                for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                    if key in weights and weights[key].ndim == 2:
                        weights[key] = weights[key].reshape(
                            self.args.o_groups, self.args.o_lora_rank, -1
                        )

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()
        for layer in self.model.layers:
            layer.attn.sharding_group = group
            layer.attn.wq_b = shard_linear(
                layer.attn.wq_b,
                "all-to-sharded",
                segments=self.args.o_groups,
                group=group,
            )
            shard_inplace(layer.attn.wo_a, "sharded-to-all", group=group)
            layer.attn.attn_sink = mx.split(layer.attn.attn_sink, N)[rank]
            layer.attn.n_heads //= N

            layer.ffn.sharding_group = group
            shard_inplace(
                layer.ffn.shared_experts.gate_proj, "all-to-sharded", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.down_proj, "sharded-to-all", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.up_proj, "all-to-sharded", group=group
            )
            shard_inplace(layer.ffn.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
            shard_inplace(layer.ffn.switch_mlp.up_proj, "all-to-sharded", group=group)
