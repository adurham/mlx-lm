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

# 2026-07-02 decode-fence overlap experiment. The Phase H Lever 1 fence
# below is a BLOCKING mx.eval(y): the CPU waits for the GPU to finish each
# layer before encoding the next, so a decode cycle pays
# (graph-build + GPU) serially 44 times (MTP-PROF: verify = 90% of the
# 62 ms cycle; ALLSUM probe: ~1.1 ms fence wall per layer vs ~0.5 ms
# weight-read floor). mx.async_eval(y) commits the graph at the same
# per-layer points — the cross-rank dispatch ORDER that Lever 1 needs is
# still pinned — but does not block, letting the CPU encode layer n+1
# while the GPU runs layer n. This is DIFFERENT from OPT-7 (which removed
# the per-layer eval entirely and paid a bigger batched-graph cost).
# Default OFF until A/B'd for throughput, bit-determinism across ranks,
# and c=2 stability. EXO_DSV4_FENCE_ASYNC=1 to enable.
_FENCE_ASYNC = bool(int(os.environ.get("EXO_DSV4_FENCE_ASYNC", "0")))
_ALLSUM_PROBE_ACC: Dict[int, List[float]] = {}    # layer_idx -> list[ms]
_ALLSUM_PROBE_CYCLES: int = 0


# ── GPU-time section probe (env-gated EXO_DSV4_SECTION_TIME=1) ──────────────
# The MLX_BUILD_PROBE above only times CPU graph-BUILD wall (no mx.synchronize),
# and the SpanProfilerHook collapses everything onto the un-compiled all_sum
# collective — neither can attribute real prefill GPU time to attention vs MoE.
# This probe wraps the two REAL compute calls in DeepseekV4Block.__call__'s
# fast path (self.attn / self.ffn) with mx.synchronize() boundaries, so each
# section's accumulated time is true device kernel time, not lazy-build time.
#
# Cost when off: one bool read per layer. When on: 4 mx.synchronize() per layer
# (serializes the pipeline, so absolute throughput drops ~10-20% — but the
# per-section SHARE is accurate, which is the point). Dumps on SIGUSR2 or via
# DeepseekV4Model.__call__ every _SECTION_TIME_LOG_EVERY forward passes.
_SECTION_TIME_ENABLED = bool(os.environ.get("EXO_DSV4_SECTION_TIME"))
_SECTION_TIME_LOG_EVERY = int(os.environ.get("EXO_DSV4_SECTION_TIME_LOG_EVERY", "0"))
_SECTION_TIME_ACC: Dict[str, float] = {
    "attn": 0.0, "ffn": 0.0, "other": 0.0, "layer_count": 0,
}
_SECTION_TIME_CYCLES: int = 0

# Sub-section attn attribution (same gate). When on, SparseCompressedAttention
# accumulates true GPU wall (mx.synchronize boundaries) for the three big attn
# blocks — compressor / indexer / sdpa — plus the remaining projections/rope.
# This answers "within attn's ~44% of prefill, what dominates?" — i.e. is the
# sparse indexer (the suspected cubic-ish blowup) the hot spot worth rewriting.
_ATTN_SUB_ACC: Dict[str, float] = {
    "compressor": 0.0, "proj_qkv": 0.0, "qk_prep": 0.0, "indexer": 0.0,
    "sdpa": 0.0, "out_proj": 0.0, "n": 0,
}

# OPT-3: sequence-split attention (env-gated EXO_DSV4_SEQ_SPLIT=1). Attention is
# replicated across both TP ranks today — both compute the full ~46% redundantly.
# In prefill (L>1) we keep compressor / kv-cache / indexer FULL on both ranks
# (so every cache + pool stays bit-identical, zero coherence risk), then slice
# the QUERY side (q, topk, mask, pmask) to this rank's contiguous row band, run
# sdpa + o_proj on L/N rows, and all_gather the output halves back to full L.
# Halves the two largest attn sub-blocks (sdpa ~31%, out_proj ~23%) at the cost
# of one all_gather/layer. Decode (L==1) and MTP verify (tiny L) skip it via the
# length gate, so decode is untouched by construction. Quality is exact: the
# gather reconstructs the identical full-sequence attention output.
# Default ON: validated +18-19% prefill (236 -> ~280 tok/s) at 20-25K ctx,
# quality-exact, decode untouched (length-gated). Set EXO_DSV4_SEQ_SPLIT=0 to
# disable (falls back to fully-replicated attention).
_SEQ_SPLIT_ENABLED = os.environ.get("EXO_DSV4_SEQ_SPLIT", "1") == "1"
_SEQ_SPLIT_MIN_L = int(os.environ.get("EXO_DSV4_SEQ_SPLIT_MIN_L", "16"))

# OPT-4 two-level chunking: max query-row width for the sparse SDPA's gathered
# (B,H,L_q,k,D) tensor. The rest of the layer (proj_qkv/indexer/o_proj/MoE) runs
# at the full prefill super-chunk width for weight-bandwidth amortization, but
# the sparse SDPA is tiled to this width so its gathered tensor never blows up.
# This is what makes larger EXO_PREFILL_STEP_SIZE viable (raw chunk 256 was
# catastrophic: 290->120 tok/s, all in this gathered tensor). 0 disables tiling.
_SPARSE_SDPA_TILE = int(os.environ.get("EXO_DSV4_SPARSE_SDPA_TILE", "128"))

# Tiled-P indexer score: when > 0 and the pooled length P exceeds this block
# size, _indexer_score is computed in contiguous P-blocks and concatenated, so
# the full (B, 64, L, P) pre-collapse scores tensor never materializes (only one
# (B,64,L,p_block) transient at a time). Bounds the per-call peak allocation that
# drives the high-context prefill stall spikes (profiler 2026-06-21: attn.indexer
# max/avg ~4x, ~22ms spikes at 360K ctx, the dominant prefill-cliff cost). 0
# (default) = OFF = full-P path, zero behaviour change. Bit-identical output;
# see bench/indexer_score_microbench.py. Tune block size for the alloc/overhead
# tradeoff (smaller = lower peak, more kernel launches).
_INDEXER_PBLOCK = int(os.environ.get("EXO_DSV4_INDEXER_PBLOCK", "0"))

# OPT-2 correctness threshold: minimum L for the lm_head last-row shortcut
# (EXO_DSV4_LMHEAD_LASTROW). Must sit ABOVE the largest small-L forward whose
# multi-row logits are consumed (MTP verify L=gamma+1, tree verify <= 16) and
# BELOW the prefill chunk width (default 128). See Model.__call__ for the
# 2026-07-01 degeneration post-mortem.
_LMHEAD_LASTROW_MIN_L = int(os.environ.get("EXO_DSV4_LMHEAD_LASTROW_MIN_L", "32"))


# FFN sub-attribution (same gate): expert compute vs the cross-rank all_sum
# (RDMA reduction). Quantifies how much of the ~50%-of-prefill MoE bucket is
# communication — i.e. the upside ceiling of switching to Pipeline sharding
# (which eliminates the per-layer all_sum).
_FFN_SUB_ACC: Dict[str, float] = {"experts": 0.0, "all_sum": 0.0, "n": 0}


def _section_time_dump() -> None:
    """Emit accumulated per-section GPU wall time (attn vs ffn) and reset."""
    acc = _SECTION_TIME_ACC
    n = acc["layer_count"]
    if not n:
        return
    attn_ms = acc["attn"] * 1000.0
    ffn_ms = acc["ffn"] * 1000.0
    other_ms = acc["other"] * 1000.0
    total_ms = attn_ms + ffn_ms + other_ms
    if total_ms <= 0:
        return
    lines = [
        f"[SECTION-TIME pid={os.getpid()}] layer_invocations={int(n)} "
        f"total={total_ms:.1f}ms",
        f"[SECTION-TIME pid={os.getpid()}]   attn  = {attn_ms:9.1f}ms "
        f"({100.0 * attn_ms / total_ms:5.1f}%)  avg/layer={attn_ms / n:6.3f}ms",
        f"[SECTION-TIME pid={os.getpid()}]   ffn   = {ffn_ms:9.1f}ms "
        f"({100.0 * ffn_ms / total_ms:5.1f}%)  avg/layer={ffn_ms / n:6.3f}ms",
        f"[SECTION-TIME pid={os.getpid()}]   other = {other_ms:9.1f}ms "
        f"({100.0 * other_ms / total_ms:5.1f}%)  avg/layer={other_ms / n:6.3f}ms",
    ]
    # Attn sub-breakdown: within the attn bucket, where does the time go?
    sub = _ATTN_SUB_ACC
    sn = sub["n"]
    if sn:
        parts = ("compressor", "proj_qkv", "qk_prep", "indexer", "sdpa", "out_proj")
        ms = {k: sub[k] * 1000.0 for k in parts}
        sub_total = sum(ms.values())
        if sub_total > 0:
            frag = "  ".join(
                f"{k}={ms[k]:.1f}ms ({100.0 * ms[k] / sub_total:.1f}%)"
                for k in parts
            )
            lines.append(
                f"[SECTION-TIME pid={os.getpid()}]   attn-sub (n={int(sn)}): {frag}"
            )
    fsub = _FFN_SUB_ACC
    if fsub["n"]:
        e_ms = fsub["experts"] * 1000.0
        a_ms = fsub["all_sum"] * 1000.0
        ft = e_ms + a_ms
        if ft > 0:
            lines.append(
                f"[SECTION-TIME pid={os.getpid()}]   ffn-sub (n={int(fsub['n'])}): "
                f"experts={e_ms:.1f}ms ({100.0 * e_ms / ft:.1f}%)  "
                f"all_sum={a_ms:.1f}ms ({100.0 * a_ms / ft:.1f}%)"
            )
    _bp_sys.stderr.write("\n".join(lines) + "\n")
    _bp_sys.stderr.flush()
    for k in ("attn", "ffn", "other", "layer_count"):
        _SECTION_TIME_ACC[k] = 0.0
    for k in ("compressor", "proj_qkv", "qk_prep", "indexer", "sdpa", "out_proj", "n"):
        _ATTN_SUB_ACC[k] = 0.0
    for k in ("experts", "all_sum", "n"):
        _FFN_SUB_ACC[k] = 0.0


def _install_section_time_sigdump() -> None:
    """Wire SIGUSR2 + atexit to dump the section-time probe."""
    if not _SECTION_TIME_ENABLED:
        return
    import atexit as _st_atexit
    import signal as _st_signal
    try:
        _st_signal.signal(_st_signal.SIGUSR2, lambda *_a: _section_time_dump())
    except (ValueError, OSError):
        pass  # not on main thread
    _st_atexit.register(_section_time_dump)
    _bp_sys.stderr.write(
        f"[SECTION-TIME pid={os.getpid()}] enabled; dump on SIGUSR2 or exit.\n"
    )
    _bp_sys.stderr.flush()


_install_section_time_sigdump()


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

# Pool-freeze flag for linear speculative verify. When True, the
# Compressor skips accumulate_windows + compress entirely and returns the
# current committed pool prefix — same freeze the tree path uses via
# _TREE_VERIFY_CTX, but without the tree-specific mask/position overrides.
# Set by the spec orchestrator around the verify forward; cleared after.
_POOL_FREEZE: bool = False


def _set_pool_freeze(freeze: bool) -> None:
    global _POOL_FREEZE
    _POOL_FREEZE = freeze


def _set_tree_verify_ctx(mask: Optional[mx.array],
                          positions: Optional[mx.array]) -> None:
    """Caller-side helper: install or clear the tree-verify side channel."""
    _TREE_VERIFY_CTX["mask"] = mask
    _TREE_VERIFY_CTX["positions"] = positions


# Eagle-style soft-embedding side channel for chained MTP drafting.
# When ``_EAGLE_CTX["soft_emb"]`` is set to a (B, S, hidden) array, the
# MTP module skips its hard-argmax ``embed_tokens(next_token)`` lookup
# and uses the supplied embedding mixture instead. Caller computes the
# mixture from the previous draft step's full logit distribution
# (probability-weighted top-K of ``embed_tokens(topk_ids)``) and clears
# the channel after the predict() call so subsequent forwards revert
# to the hard-embed path. Same module-level side-channel pattern as
# ``_TREE_VERIFY_CTX`` — single-threaded per worker process.
#
# Gated by ``EXO_DSV4_MTP_EAGLE_K`` on the exo side; mlx-lm just honors
# the channel when it's populated. When unset (default), behavior is
# bit-exact with the prior hard-embed path. See Phase 14 plan B.2.
_EAGLE_CTX: Dict[str, Any] = {"soft_emb": None}


def _set_eagle_soft_emb(soft_emb: Optional[mx.array]) -> None:
    """Caller-side helper: install or clear the Eagle soft-embedding."""
    _EAGLE_CTX["soft_emb"] = soft_emb


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


# _try_fuse_two_quantized_linears / _fused_quantized_matmul REMOVED 2026-06-18:
# helpers for the DSv4 wq_a+wkv / kv+gate fusions, which batch-mis-specialized
# at BS>1 (concurrent MTP verify → repetition degeneration). All callers
# removed. Redo batch-correctly later. See module/auto_parallel header.


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


def _clamp_mask_to_kv(mask: Optional[mx.array], kv_len: int):
    """Clamp an attention mask's trailing KV dimension to ``kv_len``.

    LocalAttention runs against a RotatingKVCache capped at
    ``sliding_window``, so once the sequence grows past the window the actual
    KV length is ``sliding_window`` — but the model-level / speculative
    tree mask is sized for the full (compressed) cache and carries a larger
    KV width (e.g. mask S=134 vs local KV=128). Passing that oversized mask
    straight to SDPA raises
    ``[broadcast_shapes] Shapes (L, S) and (B, H, L, kv_len) cannot be
    broadcast``. Sliding-window attention attends to the most-recent KV
    positions, so the correct slice is the trailing ``kv_len`` columns.

    String ("causal") and ``None`` masks are returned unchanged — the kernel
    sizes those itself. Array masks already matching ``kv_len`` (or smaller)
    pass through untouched.
    """
    if mask is None or isinstance(mask, str):
        return mask
    s = mask.shape[-1]
    if s <= kv_len:
        return mask
    return mask[..., -kv_len:]


def _extend_mask(mask: Optional[mx.array], pool_mask: Optional[mx.array], N: int):
    if mask is None:
        return None

    if mask.ndim == 2:
        mask = mask[None, None]
    B, H, L, S = mask.shape

    # The incoming mask is the model-level windowed/causal mask, sized for the
    # full sequence. This attention runs against a rotating sliding-window
    # local cache plus an optional pooled tail, so the local portion of the
    # mask must be exactly ``N - pooled_width`` columns. When the sequence has
    # grown past the window the model mask is WIDER than that (S > local_len),
    # which made ``N - S`` go negative and crash mx.ones/broadcast with
    # "[full] Negative dimensions not allowed". Sliding-window attention keeps
    # the most-recent keys, so clamp the mask to its trailing local columns.
    pooled_width = pool_mask.shape[-1] if pool_mask is not None else 0
    local_len = N - pooled_width
    if local_len >= 0 and S > local_len:
        mask = mask[..., -local_len:] if local_len > 0 else mask[..., :0]
        S = local_len

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


# Max query length L routed through the accurate per-position fused sparse
# SDPA (the MTP speculative verify, L == gamma+1, tiny). Larger L (prefill
# chunks, L == EXO_PREFILL_STEP_SIZE = 128/4096) uses the batched inner
# kernel — looping hundreds of fused SDPAs would be far too slow. 16 cleanly
# separates verify (<=~8) from prefill. See _sparse_pooled_attention.
_SPARSE_VERIFY_MAX_L = 16


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
        # L_q=1 fast path: concat-and-fused-sdpa
        # OPT-10 (2026-06-24): reshape+gather instead of take_along_axis on
        # broadcast. The broadcast (B,1,L,P,D) materializes O(B*L*P*D) memory
        # and take_along_axis iterates O(P) per call. The reshape+gather
        # flattens pooled to (B*P, D), offsets topk by b*P, does a 1D gather
        # — touches only k entries per query, O(B*L*k*D). 14× faster at
        # B=2 P=95000 (1.4ms vs 19.3ms) and does NOT scale with P. B-general.
        P_dim = pooled.shape[1]
        k_dim = topk.shape[2]
        with span("attn.gather"):
            pooled_flat = pooled.reshape(B * P_dim, D)
            offset = (mx.arange(B) * P_dim).reshape(B, 1, 1)
            topk_flat = (topk + offset).reshape(-1)
            pooled_kv = pooled_flat[topk_flat].reshape(B, L, k_dim, D)
        # Match local_kv's ndim. local_kv is normally (B, 1, sw, D) = 4D,
        # but in some MTP verify paths it can be 5D (B, 1, L, sw, D).
        # Insert a singleton at axis 1 to get (B, 1, L, k, D) = 5D, then
        # squeeze the L axis if local_kv is 4D.
        pooled_kv = pooled_kv[:, None, :, :, :]  # (B, 1, L, k, D) = 5D
        if local_kv.ndim == 4:
            pooled_kv = pooled_kv.squeeze(2)  # (B, 1, k, D) = 4D
        # Concat along seq axis: local_kv + pooled_kv
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
            # The local mask is sliced from the model-level windowed mask and
            # can be wider than the rotating local cache once the sequence
            # crosses the sliding-window boundary (e.g. decode mask local-width
            # 129 vs local_kv sw=128) → [broadcast_shapes] crash at the SDPA
            # below. Same root cause as the LocalAttention / _extend_mask
            # clamps: sliding-window attention keeps the most-recent keys, so
            # clamp the local mask's trailing columns to match local_kv (sw).
            # ``lm`` is always a real array here (local_mask or _full(...)), so
            # slice directly rather than via the Optional-typed helper.
            if lm.shape[-1] > sw:
                lm = lm[..., -sw:] if sw > 0 else lm[..., :0]
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

    # Small L_q>1 == MTP-on speculative VERIFY (L == gamma+1, tiny). Each
    # query position has its OWN top-k-gathered pooled K/V, so a single fused
    # SDPA can't express it. The legacy inner kernel (hand-rolled split-
    # softmax) accumulates in bf16 and is ~3x LESS accurate than the fused
    # fp32 SDPA (max abs diff vs fp32 ref 0.012 vs 0.004 — see L_q==1
    # docstring). Across the 21 sparse layers that error compounds into a
    # ~0.6-logit shift at the verify, enough to flip near-tie final tokens
    # (EOS vs the real next token) → c>=2 spec drops/over-stops the last
    # token vs the bit-correct non-spec batched decode. Fix: run each of the
    # few verify query positions through the accurate fused L_q==1 path and
    # stack. Gated to small L so large PREFILL chunks (L == step size, 128/
    # 4096) keep the batched inner kernel (looping hundreds of fused SDPAs
    # would be catastrophically slow, and prefill accuracy isn't tie-critical).
    if L <= _SPARSE_VERIFY_MAX_L:
        # _DSV4_PREFILL_MASK_FIX: a small PREFILL remainder chunk (1<L<=16)
        # reaches this branch with a 2-D (L,S) causal mask, but the per-
        # position slicing below assumes a 4-D mask (verify tree mask).
        # Normalize 2-D masks to 4-D so prefill chunks no longer crash with
        # 'Too many indices for array with 2 dimensions'. Verify passes
        # already supply 4-D/None masks and are unaffected.
        if local_mask is not None and local_mask.ndim == 2:
            local_mask = local_mask[None, None]
        if pooled_mask is not None and pooled_mask.ndim == 2:
            pooled_mask = pooled_mask[None, None]
        outs = []
        for li in range(L):
            out_l = _sparse_pooled_attention(
                q[:, :, li : li + 1, :],
                local_kv,
                pooled,
                topk[:, li : li + 1, :],
                None if local_mask is None else local_mask[:, :, li : li + 1, :],
                None if pooled_mask is None else pooled_mask[:, :, li : li + 1, :],
                scale,
                sinks,
            )
            outs.append(out_l)
        return mx.concatenate(outs, axis=2)

    # Large L_q (prefill chunk): batched inner kernel.
    # OPT-10 (2026-06-24): reshape+gather instead of take_along_axis on
    # broadcast. 14× faster, does NOT scale with P, B-general.
    P_dim = pooled.shape[1]
    k_dim = topk.shape[2]
    with span("attn.gather"):
        pooled_flat = pooled.reshape(B * P_dim, D)
        offset = (mx.arange(B) * P_dim).reshape(B, 1, 1)
        topk_flat = (topk + offset).reshape(-1)
        pooled_gathered = pooled_flat[topk_flat].reshape(B, L, k_dim, D)
    # Need (B, 1, L, k, D) for the inner kernel
    pooled_gathered = pooled_gathered[:, None, :, :, :]  # (B, 1, L, k, D)
    sinks_expanded = sinks[None, :, None, None] if sinks is not None else None
    # local_scores below are (B, H, L, sliding_window); the local mask sliced
    # from the model-level windowed mask can be wider than local_kv once the
    # sequence crosses the window boundary, so clamp its trailing columns to
    # the local cache width (same root cause / fix as the decode + extend
    # paths). Sliding-window attention keeps the most-recent keys.
    sw_pref = local_kv.shape[2]
    if local_mask is not None and local_mask.shape[-1] > sw_pref:
        local_mask = local_mask[..., -sw_pref:] if sw_pref > 0 else local_mask[..., :0]
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

    # fuse_gate_up_weights REMOVED 2026-06-18: the gate+up fusion path
    # batch-mis-specialized at BS>1 (concurrent MTP verify → repetition
    # degeneration). See module/auto_parallel header. Redo batch-correctly.

    def __call__(self, x: mx.array) -> mx.array:
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

    # install_compiled_forward / _raw_local / _raw_forward + the compiled
    # fast-path in __call__ REMOVED 2026-06-18: the mx.compile FFN-body path
    # batch-mis-specialized at BS>1 (concurrent MTP verify → repetition
    # degeneration). The span path below is the sole, unfused forward. Redo
    # batch-correctly later. See module/auto_parallel header + diagnosis doc.

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        # 2026-05-18 allsum probe: declare _ALLSUM_PROBE_CYCLES as global so
        # the per-branch increments at the fence site below are well-formed.
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
                    #
                    # NOTE: OPT-7 (gating mx.eval on _fence_every_n) was
                    # tested and REVERTED — it made B=2 prefill 23% SLOWER
                    # (111 vs 144 t/s). Without the per-layer eval, MLX
                    # builds a larger lazy graph that's more expensive to
                    # evaluate at the fence point than incremental evals.
                    # The overlap benefit doesn't materialize; the graph
                    # accumulation cost dominates. Keep per-layer eval.
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
                    elif _FENCE_ASYNC and y.shape[0] == 1 and y.shape[1] <= 8:
                        # Async fence is ONLY safe for c=1 decode/verify
                        # (B==1, short L). A/B'd 2026-07-02: c=1 decode
                        # 28.9 -> 37.0 t/s, outputs byte-identical; but a
                        # c=2 (B=2 batched verify) run under async fence
                        # degenerated within ~1s and wedged both ranks —
                        # the blocking eval is load-bearing for cross-rank
                        # lockstep exactly as the Phase H Lever 1 note
                        # says. Prefill (L large) and any B>1 keep the
                        # blocking fence unconditionally.
                        mx.async_eval(y)
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

    # fuse_kv_gate_weights REMOVED 2026-06-18 (BS>1 fusion degeneration;
    # see module/auto_parallel header). _project_kv_gate keeps the unfused path.

    def _project_kv_gate(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        return self.wkv(x), self.wgate(x)

    def __call__(
        self,
        x: mx.array,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
    ) -> mx.array:
        B, _, _ = x.shape
        # W4 sub-op NOP toggles — each independent of the others. Replaces the
        # output with a zero-shaped placeholder that downstream code accepts.
        # Quality intentionally broken when any sub-toggle is on. Toggles:
        #   compressor_proj   → skip _project_kv_gate (fake kv/gate as zeros)
        #   compressor_accum  → skip accumulate_windows (treat as no ready chunk)
        #   compressor_compress → skip compress+norm+rope (return zeros for new_pooled)
        #   compressor_pool   → skip pool_cache.update_and_fetch (drop the pool write)
        _nop = _get_nop_targets()
        # W4 path-1: when running the deferred-update path, apply any
        # offset bump staged by the prior step's call BEFORE we read
        # pool_cache state for this step. This makes the just-written
        # entry from the prior step's deferred update visible to this
        # step's pooled view (one step of staleness, by design).
        if pool_cache is not None:
            pool_cache.commit_pending()
        if "compressor_proj" in _nop:
            kv = mx.zeros((B, x.shape[1], self.out_dim), dtype=x.dtype)
            gate = mx.zeros((B, x.shape[1], self.out_dim), dtype=x.dtype)
        else:
            kv, gate = self._project_kv_gate(x)

        # Tree-verify path: do NOT mutate pool_cache. The pre-verify pool
        # represents prefill-derived KV summaries; mixing tree-input KV
        # (which packs same-depth siblings into contradictory positions)
        # corrupts BOTH the within-cycle attention (the new "pooled"
        # entries return contradictory siblings to the SDPA Q) AND
        # subsequent cycles (the committed pool stays contaminated even
        # after the local-cache rollback to L_kv).
        #
        # During tree verify the verify-tokens are TRANSIENT -- they're
        # rolled back from local_cache anyway. The pool must stay frozen
        # so subsequent cycles see only causally-consistent summaries.
        # Just return the current committed pool prefix (or None when
        # the pool is empty), bypassing accumulate_windows entirely.
        if pool_cache is not None and (
            _TREE_VERIFY_CTX.get("positions") is not None or _POOL_FREEZE
        ):
            if pool_cache.pooled is None:
                return mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
            return pool_cache.pooled

        if "compressor_accum" in _nop:
            # Pretend no ready chunk this step. Compress kernel never fires.
            ready_kv = mx.zeros((B, 0, kv.shape[-1]), dtype=x.dtype)
            ready_gate = mx.zeros((B, 0, gate.shape[-1]), dtype=x.dtype)
            pool_base = 0
        elif pool_cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = offset
        else:
            ready_kv, ready_gate, pool_base = pool_cache.accumulate_windows(
                kv, gate, offset
            )

        if "compressor_compress" in _nop or ready_kv.size == 0:
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

        if pool_cache is not None and "compressor_pool" not in _nop:
            # W4 path-1 (2026-05-25, promoted to default): write the
            # just-pooled entry to pool storage with the offset bump
            # DEFERRED to the next step's commit_pending(). SDPA reads
            # the PRE-WRITE prefix, breaking the compress→SDPA serialization
            # chain. The slice-assign of new_pooled into the deferred slot
            # still depends on the compress kernel but SDPA's lazy graph
            # never touches that slot.
            #
            # Quality cost: the just-pooled entry becomes attendable NEXT
            # decode step instead of the current one. Pool only updates
            # every `compress_ratio` (4 or 128) decode steps per layer,
            # so this is at most one stale entry out of N pooled per
            # layer per step. Validated 2026-05-25: 100K needle ✓,
            # short-prompt smoke ✓, +0.86 t/s (+3.0%, Welch t=13.96
            # p<<0.001) over the K=8 baseline.
            #
            # Escape hatch: putting "compressor_defer_off" in
            # /tmp/dsv4_nop_targets reverts to the synchronous path
            # for forensic A/B if a regression surfaces.
            if "compressor_defer_off" in _nop:
                new_pooled = pool_cache.update_and_fetch(new_pooled)
            else:
                new_pooled = pool_cache.update_and_fetch_deferred(new_pooled)

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
    # OPT-6 (2026-06-22): fold the per-head weights into q BEFORE the GEMM,
    # collapsing 64 heads to 1 and doing a SINGLE (L,D)@(D,P) matmul instead
    # of 64 head GEMMs + a collapse. Mathematically identical:
    #   out[b,l,p] = sum_h w[b,l,h] * sum_d q[b,h,l,d]*pooled[b,p,d]
    #             = sum_d (sum_h w[b,l,h]*q[b,h,l,d]) * pooled[b,p,d]
    #             = sum_d q_w[b,l,d] * pooled[b,p,d]
    # 64x less compute (130 GFLOP -> 2 GFLOP/chunk at 380K) and the (B,H,L,P)
    # transient is never materialized. Bit-equivalent (fewer ops = more
    # accurate). Max diff 6e-5 at fp32, <1 bf16 ulp.
    w = (mx.sigmoid(weights_x) * (scale * n_heads_inv_sqrt))  # (B, L, H)
    # q is (B, H, L, D). Fold w into q over H: q_w[b,l,d] = sum_h w[b,l,h]*q[b,h,l,d]
    q_blhd = q.transpose(0, 2, 1, 3)  # (B, L, H, D)
    q_weighted = (w[..., None] * q_blhd).sum(axis=2)  # (B, L, D)
    # Single GEMM: (B, L, D) @ (B, D, P) -> (B, L, P)
    return q_weighted @ pooled.swapaxes(-1, -2)  # (B, L, P)


@partial(mx.compile, shapeless=True)
def _indexer_score_tile(
    q_weighted: mx.array,
    pooled_tile: mx.array,
):
    """One P-tile of the indexer score. Bit-identical to the corresponding
    P-slice of ``_indexer_score``: same folded q_weighted @ pooled_tile math,
    restricted to a contiguous block of the pooled (P) axis. ``q_weighted`` is
    the already-folded (B, L, D) query — weights collapsed over H once by the
    caller — so the only per-tile work is the single GEMM.

    ``shapeless=True`` so the single compiled kernel serves every tile width
    (including the ragged final tile) without per-size recompilation.
    """
    return q_weighted @ pooled_tile.swapaxes(-1, -2)  # (B, L, P_blk)


def _indexer_score_tiled(
    q: mx.array,
    pooled: mx.array,
    weights_x: mx.array,
    scale: float,
    n_heads_inv_sqrt: float,
    p_block: int,
):
    """Tiled-P variant of ``_indexer_score``.

    Processes the pooled (P) axis in contiguous blocks of ``p_block`` and
    concatenates the collapsed ``(B, L, P_blk)`` results along P, so the full
    pre-collapse ``(B, H=64, L, P)`` scores tensor (and its transpose) is never
    materialized — only one ``(B, 64, L, p_block)`` transient exists at a time.
    This bounds the per-call peak allocation that drives the high-context
    prefill stall spikes (profiler: attn.indexer max/avg ~4x, ~22ms spikes at
    360K ctx) while keeping the output mathematically identical to the
    full-P path: concatenating per-block collapses of an op that is independent
    across the P axis equals the full op. Bit-exactness is asserted in
    bench/indexer_score_microbench.py.

    Falls back to the full-P kernel when ``P <= p_block`` (single tile) so small
    contexts (and decode, L==1, P small) pay zero overhead.
    """
    P = pooled.shape[1]
    if P <= p_block:
        return _indexer_score(q, pooled, weights_x, scale, n_heads_inv_sqrt)
    # OPT-6 (2026-06-22): fold w into q ONCE (before the tile loop), then each
    # tile is a single (B,L,D)@(B,D,P_blk) GEMM. 64x less compute per tile and
    # the (B,H,L,P_blk) transient is never materialized.
    w = (mx.sigmoid(weights_x) * (scale * n_heads_inv_sqrt))  # (B, L, H)
    q_blhd = q.transpose(0, 2, 1, 3)  # (B, L, H, D)
    q_weighted = (w[..., None] * q_blhd).sum(axis=2)  # (B, L, D)
    out_tiles = []
    for p0 in range(0, P, p_block):
        pooled_tile = pooled[:, p0 : p0 + p_block]
        tile = _indexer_score_tile(q_weighted, pooled_tile)
        # Force-materialize this tile's (B,L,p_block) collapse BEFORE building the
        # next tile's graph. Without this, MLX's lazy evaluation keeps every
        # tile's large (B,64,L,p_block) pre-collapse transient alive until the
        # final concatenate evals — so peak memory equals the full-P path (no
        # win). Evaluating per tile frees each transient first: measured peak
        # 4.36GB (full-P) -> 0.46GB (block=16384) at P=250K, L=128. The collapsed
        # tile kept across iterations is small ((B,L,p_block), no H axis).
        mx.eval(tile)
        out_tiles.append(tile)
    return mx.concatenate(out_tiles, axis=-1)        # (B, L, P)


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
        seq_band: Optional[Tuple[int, int]] = None,
    ):
        B, L_full, _ = x.shape
        # OPT-3 seq-split v2: the compressor MUST see full x (it builds the pool
        # and mutates pool_cache — coherence across ranks). But the score GEMM,
        # q-projection, weights_proj, pmask and topk are per-query-row, so when a
        # row band is given we run those on the band only (eliminating the
        # full-L work this rank would otherwise duplicate). rope offset shifts by
        # the band lo so each banded row keeps its true sequence position.
        pooled = self.compressor(x, pool_cache, offset)
        if pooled.shape[1] == 0:
            return None

        # Build the full row-causal pmask first (row index == query position),
        # then slice to the band so the kept rows carry their correct masking.
        pmask = _dispatch_pmask(pool_cache, L_full, offset)

        if seq_band is not None:
            lo, hi = seq_band
            x = x[:, lo:hi, :]
            q_residual = q_residual[:, lo:hi, :]
            L = hi - lo
            q_off = offset + lo
            if pmask is not None:
                pmask = pmask[lo:hi, :] if pmask.ndim == 2 else pmask[..., lo:hi, :]
        else:
            L = L_full
            q_off = offset

        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = _rope_dispatch(position_rope, q, q_off)

        with span("indexer.score"):
            if _INDEXER_PBLOCK > 0:
                scores = _indexer_score_tiled(
                    q,
                    pooled,
                    self.weights_proj(x),
                    self.scale,
                    self.n_heads**-0.5,
                    _INDEXER_PBLOCK,
                )
            else:
                scores = _indexer_score(
                    q,
                    pooled,
                    self.weights_proj(x),
                    self.scale,
                    self.n_heads**-0.5,
                )
        if pmask is not None:
            # OPT-12 (env-gated EXO_DSV4_TAIL_PMASK=1, default ON): tail-
            # restricted pmask apply. The row-causal pmask row j is
            # ``pool_idx < (q_off + j + 1) // ratio`` — monotone in j. Pool
            # columns below vis_min = (q_off+1)//ratio are visible to EVERY
            # row of this chunk; columns >= vis_max = (q_off+L)//ratio + 1
            # are visible to NONE. Only the tiny [vis_min, vis_max) band
            # (≈ L/ratio + 1 columns, e.g. ~65 at L=256 ratio=4) is row-
            # dependent. Applying the full (L, P) where() drags an O(L·P)
            # bool tensor through memory per indexer layer per chunk —
            # ~127 MB at P=124K — when all but the band is constant.
            # Restricting the where() to the band is EXACT: outside the
            # band the mask is constant-true (keep score) or constant-false
            # (min-fill), applied as a cheap slice fill. Tree-verify passes
            # a non-monotone mask — detected via _TREE_VERIFY_CTX — and
            # keeps the full-P path. Decode/verify (pmask None) unaffected.
            _tail_ok = (
                _topk_os.environ.get("EXO_DSV4_TAIL_PMASK", "1") == "1"
                and pmask.ndim == 2
                and _TREE_VERIFY_CTX.get("positions") is None
                and not isinstance(q_off, mx.array)
            )
            if _tail_ok:
                P_len = scores.shape[-1]
                L_rows = scores.shape[1]
                ratio = self.compressor.compress_ratio
                vis_min = min((q_off + 1) // ratio, P_len)
                vis_max = min((q_off + L_rows) // ratio + 1, P_len)
                neg = mx.finfo(scores.dtype).min
                parts = [scores[..., :vis_min]]
                if vis_max > vis_min:
                    parts.append(mx.where(
                        pmask[None, :, vis_min:vis_max],
                        scores[..., vis_min:vis_max],
                        neg,
                    ))
                if P_len > vis_max:
                    parts.append(mx.full(
                        (scores.shape[0], L_rows, P_len - vis_max),
                        neg, dtype=scores.dtype,
                    ))
                scores = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
            else:
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
        with span("indexer.topk"):
            if (_topk_enabled
                    and scores.shape[1] == 1
                    and pmask is None
                    and k <= 1024):
                fused = _fused_topk(scores, k)
                if fused is not None:
                    return fused
            # OPT-1 (env-gated EXO_DSV4_PREFILL_ARGPARTITION=1): in PREFILL (L>1)
            # the argsort below is a full O(P log P) sort over the pool just to take
            # the top-k. argpartition is O(P) and the top-k SET is identical; the
            # downstream take_along_axis → gathered-KV attention is order-invariant
            # (softmax sums over all gathered positions), so unordered top-k is
            # quality-equivalent. Decode (L==1) keeps argsort untouched (the
            # ~5%-faster-on-Metal claim was measured at L=1), so decode is unaffected
            # by construction. Gated for clean A/B against the section-time harness.
            #
            # P-threshold (2026-06-21): argpartition is SLOWER than argsort on Metal
            # at small P (kernel launch overhead dominates the O(P log P)->O(P) win).
            # Measured: at P=500 (2K context) argpartition drops throughput 295->163
            # t/s. Only fire when P exceeds EXO_DSV4_ARGPARTITION_MIN_P (default 0 =
            # always fire when env enabled; set e.g. 20000 to only fire past ~80K ctx).
            if (scores.shape[1] > 1
                    and _topk_os.environ.get("EXO_DSV4_PREFILL_ARGPARTITION", "0") == "1"
                    and pooled.shape[1] >= int(_topk_os.environ.get("EXO_DSV4_ARGPARTITION_MIN_P", "0"))):
                return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
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

    # fuse_qa_kv_weights REMOVED 2026-06-18 (BS>1 fusion degeneration;
    # see module/auto_parallel header). _project_qa_kv keeps the unfused path.

    def _project_qa_kv(self, x: mx.array) -> Tuple[mx.array, mx.array]:
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

            # Sub-span attribution for the 2026-05-25 "16% unaccounted attn
            # wall" investigation: project_qa_kv + q_norm + wq_b + _q_finalize +
            # kv_norm + reshape. Bracketed by finalize() so the perf_counter
            # measures real compute, not lazy graph build.
            with span("attn.proj_qkv"):
                q_lora, kv_pre = self._project_qa_kv(x)
                q = _q_finalize(
                    self.wq_b(self.q_norm(q_lora)),
                    B, L, self.n_heads, self.head_dim,
                    self.config.rms_norm_eps,
                )
                kv = self.kv_norm(kv_pre).reshape(B, 1, L, self.head_dim)
                q = finalize(q)
                kv = finalize(kv)
            with span("attn.rope_in"):
                q = _rope_dispatch(self.rope, q, offset)
                kv = _rope_dispatch(self.rope, kv, offset)
                q = finalize(q)
                kv = finalize(kv)
            if cache is not None:
                with span("attn.kv_cache"):
                    kv, _ = cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
                    kv = finalize(kv)

            # The model-level / speculative tree mask is sized for the full
            # (compressed) cache; this attention runs against a rotating
            # sliding-window cache whose KV length is capped at
            # ``sliding_window``. Clamp the mask's trailing KV dimension to the
            # actual KV length so SDPA can broadcast it (otherwise e.g. a
            # mask S=134 vs local KV=128 raises [broadcast_shapes]).
            mask = _clamp_mask_to_kv(mask, kv.shape[2])

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
            with span("attn.rope_out"):
                out = _rope_dispatch(self.rope, out, offset, inverse=True)
                out = finalize(out)

            with span("attn.o_proj"):
                out = _o_pre_a(out, B, self.o_groups, L, self.head_dim)
                out = self.wo_a(out)
                out = _o_pre_b(out)
                out = self.wo_b(out)
                out = finalize(out)

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

    # fuse_qa_kv_weights REMOVED 2026-06-18 (BS>1 fusion degeneration;
    # see module/auto_parallel header). _project_qa_kv keeps the unfused path.

    def _project_qa_kv(self, x: mx.array) -> Tuple[mx.array, mx.array]:
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

            # W4 path-1 (2026-05-24): issue the compressor BEFORE the q/k/v
            # projections so its independent kernel chain (project_kv_gate →
            # accumulate_windows → optional compress+norm+rope) is queued
            # first in the lazy graph. The compressor consumes only `x`; the
            # main attention's projections also consume only `x`. With the
            # compressor queued first, mlx's async dispatch can overlap the
            # rare-but-heavy compress kernel with the always-on q/k/v
            # projections that follow. Previously the compressor was
            # serialized after kv was ready, even though it has no data
            # dependency on q/k/v.
            with span("attn.compressor"):
                if "compressor" in _get_nop_targets():
                    # NOP: emit an empty pooled tensor with the same dtype and
                    # batch dim. attn proceeds with only local KV. Quality
                    # intentionally broken — bench tok/s only.
                    pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
                else:
                    pooled = finalize(self.compressor(x, pool_cache, offset))

            # Sub-span attribution for the 2026-05-25 "16% unaccounted attn
            # wall" investigation — see LocalAttention.__call__ for the same
            # set of spans.
            with span("attn.proj_qkv"):
                q_lora, kv_pre = self._project_qa_kv(x)
                q = _q_finalize(
                    self.wq_b(self.q_norm(q_lora)),
                    B, L, self.n_heads, self.head_dim,
                    self.config.rms_norm_eps,
                )
                kv = self.kv_norm(kv_pre).reshape(B, 1, L, self.head_dim)
                q = finalize(q)
                kv = finalize(kv)
            with span("attn.rope_in"):
                q = _rope_dispatch(self.rope, q, offset)
                kv = _rope_dispatch(self.rope, kv, offset)
                q = finalize(q)
                kv = finalize(kv)
            if local_cache is not None:
                with span("attn.kv_cache"):
                    kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
                    kv = finalize(kv)
            pooled_mask = None
            with span("attn.mask"):
                if pooled.shape[1] > 0:
                    # Tree-aware pmask dispatch: see _tree_pmask docstring.
                    pooled_mask = _dispatch_pmask(pool_cache, L, offset)
                    kv = mx.concatenate([kv, pooled[:, None]], axis=2)
                mask = _extend_mask(mask, pooled_mask, kv.shape[2])
                kv = finalize(kv)

            # OPT-3b sequence-split (CompressedAttention): same pattern as the
            # sparse class. compressor + kv-cache above ran FULL on both ranks
            # (kv now holds full local+pooled, coherent); slice the query side
            # to this rank's row band, run sdpa + o_proj on L/N rows, gather
            # back after o_proj. kv is full-width so each band attends correctly.
            _sg = self.sharding_group
            _seq = (
                _sg is not None
                and _SEQ_SPLIT_ENABLED
                and L >= _SEQ_SPLIT_MIN_L
                and L % _sg.size() == 0
            )
            _seq_lo = 0
            if _seq and _sg is not None:
                _N = _sg.size()
                _band = L // _N
                _seq_lo = _sg.rank() * _band
                _seq_hi = _seq_lo + _band
                q = q[:, :, _seq_lo:_seq_hi, :]
                if mask is not None and not isinstance(mask, str):
                    mask = mask[..., _seq_lo:_seq_hi, :]

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
            with span("attn.rope_out"):
                # seq-split: band sits at positions offset+_seq_lo (rope increments
                # per row from offset) — shift the inverse-rope offset to match.
                _rope_off = (offset + _seq_lo) if _seq else offset
                out = _rope_dispatch(self.rope, out, _rope_off, inverse=True)
                out = finalize(out)

            with span("attn.o_proj"):
                # seq-split: out has only the band rows; reshape with band length.
                _o_len = out.shape[2] if _seq else L
                out = _o_pre_a(out, B, self.o_groups, _o_len, self.head_dim)
                out = self.wo_a(out)
                out = _o_pre_b(out)
                out = self.wo_b(out)
                out = finalize(out)

            if _seq and _sg is not None:
                # Reconstruct full sequence from per-rank bands. out is
                # (B, band, H); all_gather concatenates each rank's full
                # B-batch along axis 0 in rank order → (N*B, band, H) with
                # memory layout [r0s0, r0s1, ..., r1s0, r1s1, ...] (rank-major).
                # The naive reshape(B, L, H) is row-major and would interpret
                # that as [s0_band0, s0_band1, s1_band0, ...] — scrambling
                # which band lands in which stream at B>1. Fix: view as
                # (N, B, band, H), transpose to (B, N, band, H), then flatten
                # N and band into L so each stream's L axis is the concat of
                # ITS OWN bands across ranks. Bit-exact vs unsharded at B=1.
                with span("attn.all_gather"):
                    _B = out.shape[0]
                    _H = out.shape[-1]
                    _N = _sg.size()
                    _band = L // _N
                    _g = mx.distributed.all_gather(out, group=_sg)
                    out = finalize(
                        _g.reshape(_N, _B, _band, _H)
                        .transpose(1, 0, 2, 3)
                        .reshape(_B, L, _H)
                    )
            elif self.sharding_group is not None:
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

    # fuse_qa_kv_weights REMOVED 2026-06-18 (BS>1 fusion degeneration;
    # see module/auto_parallel header). _project_qa_kv keeps the unfused path.

    def _project_qa_kv(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        return self.wq_a(x), self.wkv(x)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        with span("attn"):
            _stsub = _SECTION_TIME_ENABLED
            _sub_t = 0.0
            if _stsub:
                mx.synchronize()
                _sub_t = _BUILD_PROBE_PERF()
            B, L, _ = x.shape
            local_cache = cache[0] if cache is not None else None
            comp_cache = cache[1] if cache is not None else None
            idx_cache = cache[2] if cache is not None else None
            offset = local_cache.offset if local_cache is not None else 0
            offset = mx.array(offset) if isinstance(offset, mx.array) else offset

            # OPT-3 seq-split v2: compute this rank's contiguous query row band
            # ONCE, up front. The kv side (wkv/kv_norm/kv-cache) and compressor
            # /pool stay FULL on every rank (coherence); the q side (wq_b main q,
            # indexer score, sdpa, o_proj) runs on the band [_seq_lo:_seq_hi].
            # Outputs are all_gathered back to full L after o_proj.
            _sg = self.sharding_group
            _seq = (
                _sg is not None
                and _SEQ_SPLIT_ENABLED
                and L >= _SEQ_SPLIT_MIN_L
                and L % _sg.size() == 0
            )
            _seq_lo = 0
            _seq_hi = L
            _seq_band = None
            if _seq and _sg is not None:
                _band = L // _sg.size()
                _seq_lo = _sg.rank() * _band
                _seq_hi = _seq_lo + _band
                _seq_band = (_seq_lo, _seq_hi)

            # W4 path-1 (2026-05-24): issue compressor first; see
            # CompressedAttention.__call__ for rationale.
            with span("attn.compressor"):
                if "compressor" in _get_nop_targets():
                    # NOP: see CompressedAttention.__call__ above. Quality
                    # intentionally broken — bench tok/s only.
                    pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
                else:
                    pooled = finalize(self.compressor(x, comp_cache, offset))
            if _stsub:
                # Explicit eval — finalize() is a no-op unless EXO_PROFILER is
                # set, so we must force materialization ourselves to time GPU.
                mx.eval(pooled)
                mx.synchronize()
                _t_comp = _BUILD_PROBE_PERF()
                _ATTN_SUB_ACC["compressor"] += (_t_comp - _sub_t)
                _sub_t = _t_comp

            # Sub-span attribution for the 2026-05-25 "16% unaccounted attn
            # wall" investigation — see LocalAttention.__call__ for the same
            # set of spans. q_residual is preserved because the indexer needs
            # it (it consumes the pre-wq_b q lora).
            with span("attn.proj_qkv"):
                q_lora, kv_pre = self._project_qa_kv(x)
                q_residual = self.q_norm(q_lora)
                # seq-split v2: the main-q projection wq_b is per-row → run it on
                # the band only. q_residual is kept FULL for the indexer call
                # below (the indexer does its own banded slice internally).
                _q_res_band = q_residual[:, _seq_lo:_seq_hi, :] if _seq_band is not None else q_residual
                _Lq = (_seq_hi - _seq_lo) if _seq_band is not None else L
                q = _q_finalize(
                    self.wq_b(_q_res_band),
                    B, _Lq, self.n_heads, self.head_dim,
                    self.config.rms_norm_eps,
                )
                kv = self.kv_norm(kv_pre).reshape(B, 1, L, self.head_dim)
                q = finalize(q)
                kv = finalize(kv)
            if _stsub:
                mx.eval(q, kv)
                mx.synchronize()
                _t_proj = _BUILD_PROBE_PERF()
                _ATTN_SUB_ACC["proj_qkv"] += (_t_proj - _sub_t)
                _sub_t = _t_proj
            with span("attn.rope_in"):
                # q rows sit at positions offset+_seq_lo (band); kv is full.
                _q_rope_off = (offset + _seq_lo) if _seq_band is not None else offset
                q = _rope_dispatch(self.rope, q, _q_rope_off)
                kv = _rope_dispatch(self.rope, kv, offset)
                q = finalize(q)
                kv = finalize(kv)
            if local_cache is not None:
                with span("attn.kv_cache"):
                    kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
                    kv = finalize(kv)
            with span("attn.mask"):
                # Tree-aware pmask dispatch: see _tree_pmask docstring. Built for
                # full L (row index == query position), then sliced to the band so
                # the kept rows carry their correct masking under seq-split v2.
                pmask = _dispatch_pmask(comp_cache, L, offset)
                if pmask is not None and _seq_band is not None:
                    pmask = pmask[_seq_lo:_seq_hi, :] if pmask.ndim == 2 else pmask[..., _seq_lo:_seq_hi, :]
                if pmask is not None:
                    pmask = finalize(pmask)
            if _stsub:
                mx.eval(q, kv)
                mx.synchronize()
                _t_pre_idx = _BUILD_PROBE_PERF()
                # rope_in + kv_cache + mask (q/k prep before the indexer)
                _ATTN_SUB_ACC["qk_prep"] += (_t_pre_idx - _sub_t)
                _sub_t = _t_pre_idx
            with span("attn.indexer"):
                if "indexer" in _get_nop_targets():
                    # Indexer returns argsort(-scores)[..., :k] over scores shaped
                    # (B, L, P) so output is (B, L, k). Return deterministic
                    # in-range indices [0, k) so downstream sparse_sdpa doesn't OOB.
                    # q is already banded here, so _L = band length.
                    _topk = self.indexer.index_topk
                    _pool_len = pooled.shape[1] if pooled.shape[1] > 0 else _topk
                    _take = min(_topk, _pool_len)
                    _B, _, _L, _ = q.shape
                    topk = mx.broadcast_to(
                        mx.arange(_take, dtype=mx.int32)[None, None, :],
                        (_B, _L, _take),
                    )
                else:
                    # seq-split v2: pass the band so the indexer's compressor/pool
                    # run FULL (coherent) but its score GEMM + topk run banded.
                    topk = finalize(
                        self.indexer(x, q_residual, self.rope, idx_cache, offset,
                                     seq_band=_seq_band)
                    )
            if _stsub:
                mx.eval(topk)
                mx.synchronize()
                _t_idx = _BUILD_PROBE_PERF()
                # The indexer block alone (pre-indexer fence above isolated it).
                _ATTN_SUB_ACC["indexer"] += (_t_idx - _sub_t)
                _sub_t = _t_idx
            sinks = self.attn_sink.astype(q.dtype)

            # seq-split v2: q, topk, and pmask are ALREADY banded above (the
            # q-projection, indexer score, and pmask all ran on the band). Only
            # the attention `mask` is still full-L here, so slice it to the band
            # rows to match q. kv/pooled stay full (each banded row attends them).
            if _seq_band is not None and mask is not None and not isinstance(mask, str):
                mask = mask[..., _seq_lo:_seq_hi, :]

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
                        # OPT-4 two-level chunking: the sparse SDPA builds a
                        # gathered (B,H,L_q,k,D) tensor — the term that blows up
                        # ~cubically with chunk width and made bigger prefill
                        # chunks catastrophic. proj_qkv / indexer / o_proj / MoE
                        # all WANT a big batch (weight-bandwidth amortization),
                        # but THIS step must stay narrow. So tile ONLY the sparse
                        # SDPA over query-row sub-chunks of <= _SPARSE_SDPA_TILE,
                        # keeping the gathered tensor small while the rest of the
                        # layer runs at the full super-chunk width. Each sub-chunk
                        # is per-query-row independent (own topk gather + shared
                        # local window), so slicing q/topk/masks by row and
                        # concatenating the outputs is bit-exact. No cache
                        # mutation here (kv/pooled already built).
                        _Lq = q.shape[2]
                        _tile = _SPARSE_SDPA_TILE
                        if _tile > 0 and _Lq > _tile:
                            _parts = []
                            for _s in range(0, _Lq, _tile):
                                _e = min(_s + _tile, _Lq)
                                _qm = mask
                                if _qm is not None and not isinstance(_qm, str):
                                    _qm = _qm[..., _s:_e, :]
                                _sm = sparse_mask
                                if _sm is not None:
                                    _sm = _sm[:, :, _s:_e, :]
                                _parts.append(_sparse_pooled_attention(
                                    q[:, :, _s:_e, :],
                                    kv,
                                    pooled,
                                    topk[:, _s:_e, :],
                                    _qm,
                                    _sm,
                                    self.scale,
                                    sinks,
                                ))
                            out = mx.concatenate(_parts, axis=2)
                        else:
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
            if _stsub:
                mx.eval(out)
                mx.synchronize()
                _t_sdpa = _BUILD_PROBE_PERF()
                _ATTN_SUB_ACC["sdpa"] += (_t_sdpa - _sub_t)
                _sub_t = _t_sdpa

            with span("attn.rope_out"):
                # seq-split: this rank's band sits at sequence positions
                # [offset+_seq_lo : ...], so the inverse RoPE offset must shift
                # by _seq_lo (mx.fast.rope increments per row from `offset`).
                _rope_off = (offset + _seq_lo) if _seq else offset
                out = _rope_dispatch(self.rope, out, _rope_off, inverse=True)
                out = finalize(out)

            with span("attn.o_proj"):
                # seq-split: out has only this rank's row band, so the o_proj
                # reshape must use the band length, not the full L.
                _o_len = out.shape[2] if _seq else L
                out = _o_pre_a(out, B, self.o_groups, _o_len, self.head_dim)
                out = self.wo_a(out)
                out = _o_pre_b(out)
                out = self.wo_b(out)
                out = finalize(out)
            if _stsub:
                mx.eval(out)
                mx.synchronize()
                _t_oproj = _BUILD_PROBE_PERF()
                # rope_out + o_proj
                _ATTN_SUB_ACC["out_proj"] += (_t_oproj - _sub_t)
                _sub_t = _t_oproj
                _ATTN_SUB_ACC["n"] += 1

            if _seq and _sg is not None:
                # Reconstruct the full sequence from per-rank row bands. out is
                # (B, band, hidden); all_gather concatenates each rank's full
                # B-batch along axis 0 in rank order → (N*B, band, hidden),
                # memory layout [r0s0, r0s1, ..., r1s0, r1s1, ...] (rank-major).
                # The naive reshape(B, L, H) is row-major and would interpret
                # that as [s0_band0, s0_band1, s1_band0, ...] — scrambling
                # which band lands in which stream at B>1. Fix: view as
                # (N, B, band, H), transpose to (B, N, band, H), then flatten
                # N and band into L so each stream's L axis is the concat of
                # ITS OWN bands across ranks. Bit-exact vs unsharded at B=1.
                with span("attn.all_gather"):
                    _B = out.shape[0]
                    _H = out.shape[-1]
                    _N = _sg.size()
                    _band = L // _N
                    _g = mx.distributed.all_gather(out, group=_sg)
                    out = finalize(
                        _g.reshape(_N, _B, _band, _H)
                        .transpose(1, 0, 2, 3)
                        .reshape(_B, L, _H)
                    )
            elif self.sharding_group is not None:
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
    # install_compiled_forward + _raw_attn_pre/_raw_post_attn/_raw_ffn_pre/
    # _raw_post_ffn + the compiled fast-path in __call__ REMOVED 2026-06-18:
    # the V4Block-level mx.compile path batch-mis-specialized at BS>1
    # (concurrent MTP verify → repetition degeneration). The span path below
    # is the sole forward. Redo batch-correctly. See module/auto_parallel header.

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        input_ids: mx.array,
    ) -> mx.array:
        if "v4block" in _get_nop_targets():
            return h  # NOP: pass residual through unchanged

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
        #
        # Eagle soft-embedding override: when the caller has stashed a
        # (B, L, hidden_size) probability-weighted embedding mixture in
        # ``_EAGLE_CTX["soft_emb"]``, use it instead of the hard
        # ``embed_tokens(next_token)`` lookup. ``next_token`` is still
        # passed by the caller for cache bookkeeping / signature parity,
        # but its embedding is ignored. The caller is responsible for
        # clearing the channel after the predict() call so the next
        # forward reverts to the hard-embed path. Default-off — when
        # ``_EAGLE_CTX["soft_emb"]`` is None the path is bit-exact with
        # the prior implementation.
        _eagle_soft = _EAGLE_CTX.get("soft_emb")
        if _eagle_soft is not None:
            emb = _eagle_soft  # (B, L, hidden_size)
        else:
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

        _actprobe = os.environ.get("EXO_DSV4_ACT_PROBE") == "1"
        if _actprobe:
            import sys as _ap_sys
            def _ap_std(t):
                v = t.astype(mx.float32)
                return float(mx.sqrt(mx.mean(v * v)).item())
            mx.eval(h)
            _ap_sys.stderr.write(f"[ACTPROBE] embed rms={_ap_std(h):.4f}\n")
        for _ap_i, (layer, layer_cache) in enumerate(zip(self.pipeline_layers, cache)):
            h = layer(h, mask, layer_cache, inputs)
            if _actprobe:
                mx.eval(h)
                _r = _ap_std(h)
                _ap_sys.stderr.write(f"[ACTPROBE] layer={_ap_i:2d} rms={_r:.4f}\n")
                _ap_sys.stderr.flush()

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
            if _actprobe:
                _hc = self.hc_head(h)
                mx.eval(_hc)
                _ap_sys.stderr.write(f"[ACTPROBE] hc_head rms={_ap_std(_hc):.4f} shape={tuple(_hc.shape)}\n")
                _normed = self.norm(_hc)
                mx.eval(_normed)
                _ap_sys.stderr.write(f"[ACTPROBE] post_norm rms={_ap_std(_normed):.4f}\n")
                out = finalize(_normed)
            else:
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
        # Section-time probe: dump from the inference thread itself every
        # _SECTION_TIME_LOG_EVERY forward passes. This is the reliable dump
        # path — SIGUSR2 only wires if the module imported on the main thread,
        # which is not guaranteed for runner subprocesses. Default 1 = dump
        # after every forward (one prefill forward already accumulates all
        # layers, so a single prefill yields a complete attribution).
        if _SECTION_TIME_ENABLED and _SECTION_TIME_ACC["layer_count"]:
            global _SECTION_TIME_CYCLES
            _SECTION_TIME_CYCLES += 1
            if _SECTION_TIME_CYCLES % max(1, _SECTION_TIME_LOG_EVERY) == 0:
                _section_time_dump()
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
            # OPT-2 (env-gated EXO_DSV4_LMHEAD_LASTROW=1): during PREFILL the
            # caller (mlx_lm stream_generate prefill loop) DISCARDS this output —
            # it keeps only the KV cache — and the decode _step only ever reads
            # logits[:, -1, :]. So projecting all L rows through lm_head
            # (L × vocab_size ≈ 128 × 129K) is wasted work every prefill chunk.
            # When L is prefill-sized, project only the last row.
            #
            # CORRECTNESS GATE (2026-07-01): the original `L > 1` gate was WRONG
            # under MTP — the speculative VERIFY forward routes through here at
            # L == gamma+1 (small), and the acceptance check consumes ALL rows
            # of verify_logits. Slicing it to the last row broke acceptance →
            # repetition-loop degeneration (reproduced on-cluster, cluster
            # smoke 2026-07-01). Prefill chunks are >= 32 in practice (default
            # 128); verify is gamma+1 (2-9) and tree verify <= 16
            # (_SPARSE_VERIFY_MAX_L). Gate on L > _LMHEAD_LASTROW_MIN_L
            # (default 32) so ONLY true prefill chunks take the last-row path.
            # Remainder chunks 1 < L <= 32 harmlessly keep the full projection.
            if (h.shape[1] > _LMHEAD_LASTROW_MIN_L
                    and os.environ.get("EXO_DSV4_LMHEAD_LASTROW", "0") == "1"):
                h = h[:, -1:, :]
            _logits = self.lm_head(h)
            if os.environ.get("EXO_DSV4_ACT_PROBE") == "1":
                import sys as _lp_sys
                mx.eval(_logits)
                _row = _logits[0, -1].astype(mx.float32)
                _top = mx.argsort(_row)[-5:]
                _ids = [int(x) for x in _top.tolist()][::-1]
                _vals = [round(float(_row[i]), 3) for i in _ids]
                _lp_sys.stderr.write(f"[ACTPROBE] logits top5_ids={_ids} top5={_vals} rms={float(mx.sqrt(mx.mean(_row*_row)).item()):.3f}\n")
                _lp_sys.stderr.flush()
            return finalize(_logits)

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

        # Checkpoint names per-layer Hyper-Connection modules hc_attn /
        # hc_ffn, but this model defines them as attn_hc / ffn_hc. Without
        # this rename the HC weights silently fail to load (strict=False)
        # and stay at mx.zeros init, so each layer hyper-connection runs
        # with zero mix weights -> healthy-magnitude but semantically
        # scrambled residual stream -> confident-garbage output.
        renamed = {}
        for k, v in weights.items():
            nk = k.replace(".hc_attn.", ".attn_hc.").replace(".hc_ffn.", ".ffn_hc.")
            renamed[nk] = v
        weights = renamed

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
