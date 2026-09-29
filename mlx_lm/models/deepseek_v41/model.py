# Vendored from PipeNetwork/deepseek-v41-mlx (Apache-2.0), upstream a9567ae plus the
# node-local MTP/DSpark port edits (exo docs phase 3). See README.md here.
"""DeepSeek-V4.1 block and full model (text stack).

The block wires Hyper-Connections in V4.1's *staggered* form: the coefficient
set a sub-layer computes is consumed by the **next** sub-layer's collapse.
Attention collapses with the previous layer's FFN ``pre`` (identity one-hot at
the start), the FFN collapses with this layer's attention ``pre``, and the LM
head collapses with the last layer's FFN ``pre``.

Cross-layer sharing flows through a per-forward :class:`SharedState`, mirroring
the reference's process-global ``SharedAttentionRuntime``: kv sources publish
their compressed-KV cache and index-key cache, index sources publish their
top-k selection, the candidate source publishes its block mask, and every layer
in between consumes the most recent value. Layers run top-down, so every source
writes before its consumers read.

Engram layers apply their gated n-gram lookup to the hc-expanded stream
*before* the block runs, exactly as the reference does in ``Transformer.forward``.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .attention import Attention
from .cache import ModelCache
from .config import ModelArgs
from .engram import Engram, EngramHasher
from .hyper_connections import hc_mixes, hc_post, hc_pre, make_identity_pre_mix
from .layers import RMSNorm
from .moe import MoE
from .hc_fused import hc_expand, mixes_and_collapse

import os

# Fused hyper-connection kernels (hc_fused.py). "0" selects the reference ops.
_HC_FUSED = os.environ.get("DSV41_HC_FUSED", "1") == "1"
# Queue each block's GPU work as soon as it is built (same results, earlier start).
_ASYNC_EVAL = os.environ.get("DSV41_ASYNC_EVAL", "1") == "1"


class SharedState:
    """What attention layers hand down the stack instead of recomputing.

    Fresh per forward; every source writes before its consumers read."""

    def __init__(self):
        self.kv_src_cache = None       # LayerCache of the most recent kv source
        self.index_src_cache = None    # LayerCache of the most recent index-key owner
        self.topk_idxs = None          # [b, n, k] from the most recent index source
        self.candidates = None         # [b, n, nb] bool from the candidate source


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.hc_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps

        self.attn = Attention(layer_id, args)
        self.ffn = MoE(args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

        mix_hc = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.dim
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_attn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_ffn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_attn_scale = mx.zeros((3,), dtype=mx.float32)
        self.hc_ffn_scale = mx.zeros((3,), dtype=mx.float32)

        if layer_id in args.engram_layer_ids:
            self.engram = Engram(args, args.engram_layer_ids.index(layer_id))
        else:
            self.engram = None

    def __call__(self, x: mx.array, pre_mix: mx.array, start_pos: int,
                 cache, shared):
        """x [b, s, hc, d]; pre_mix [b, s, hc] from the previous sub-layer.
        Returns (x, ffn_pre) — ffn_pre feeds the next layer (or the head)."""
        if _HC_FUSED and self.hc_mult == 4:
            return self._fused_call(x, pre_mix, start_pos, cache, shared)
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            self.hc_mult, self.hc_iters, self.norm_eps, self.hc_eps)
        h = hc_pre(x, pre_mix)
        h = self.attn(self.attn_norm(h), start_pos, cache, shared)
        x = hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            self.hc_mult, self.hc_iters, self.norm_eps, self.hc_eps)
        h = hc_pre(x, attn_pre)
        h = self.ffn(self.ffn_norm(h))
        x = hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def _fused_call(self, x, pre_mix, start_pos, cache, shared):
        h, attn_pre, attn_post, attn_comb = mixes_and_collapse(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, pre_mix,
            self.hc_iters, self.norm_eps, self.hc_eps)
        h = self.attn(self.attn_norm(h), start_pos, cache, shared)
        x = hc_expand(h, x, attn_post, attn_comb)
        h, ffn_pre, ffn_post, ffn_comb = mixes_and_collapse(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, attn_pre,
            self.hc_iters, self.norm_eps, self.hc_eps)
        h = self.ffn(self.ffn_norm(h))
        x = hc_expand(h, x, ffn_post, ffn_comb)
        return x, ffn_pre


class Model(nn.Module):
    """embed -> expand to hc copies -> blocks -> collapse (last ffn_pre) -> logits."""

    def __init__(self, args: ModelArgs, token_map=None):
        super().__init__()
        self.args = args
        self.hc_mult = args.hc_mult
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [Block(i, args) for i in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.mtp = None
        if args.n_mtp_layers:
            from .mtp import DSparkHead
            self.mtp = DSparkHead(args)
        self.engram_hasher = None
        if args.engram_layer_ids and token_map is not None:
            self.set_token_map(token_map)
        # test hook: remap which source each consumer reads (negative control)
        self._break_sharing = False
        # prefill orchestration (prefill.py): commit the chunk every N layers of
        # a multi-row forward by evaluating (h, pre_mix). 0 disables. Set only
        # by prefill.prefill()/warmup(); single-row forwards are never fenced.
        self._fence_every = 0

    def set_token_map(self, token_map):
        self.engram_hasher = EngramHasher(self.args, token_map)

    def make_cache(self, bsz: int = 1, max_seq_len: int | None = None,
                   dtype=mx.float32) -> ModelCache:
        return ModelCache(self.args, bsz, max_seq_len, dtype)

    def __call__(self, input_ids: mx.array, cache: ModelCache,
                 last_logit_only: bool = False, return_taps: bool = False,
                 argmax: bool = False):
        # First calls per input shape host-sync every collective, so a rank
        # still JIT-building kernels for the shape cannot leave its peer's
        # GPU waiting inside a command buffer past the Metal watchdog
        # (collective.py). Keyed on (rows, flags); decode/verify shapes warm
        # after 2 calls each.
        from . import collective as _coll
        key = ("body", int(input_ids.shape[1]), bool(last_logit_only),
               bool(return_taps), bool(argmax))
        with _coll.warm_guard(key):
            return self._forward(input_ids, cache, last_logit_only=last_logit_only,
                                 return_taps=return_taps, argmax=argmax)

    def _forward(self, input_ids: mx.array, cache: ModelCache,
                 last_logit_only: bool = False, return_taps: bool = False,
                 argmax: bool = False):
        """input_ids [b, n] continue the sequence at cache.offset. Advances the cache.

        ``return_taps`` additionally returns ``{layer_id: hc_mean_hidden}`` for
        every layer in ``args.dspark_target_layer_ids`` -- the DSpark context
        feed. Production captures the same quantity (``h.mean(axis=2)`` after
        each tapped layer) into a module-level side channel; returning it is the
        same information without the global.
        """
        start_pos = cache.offset
        b, n = input_ids.shape

        hashes = None
        if self.engram_hasher is not None:
            ids_np = np.array(input_ids, dtype=np.int64)
            hashes = self.engram_hasher(ids_np, start_pos, cache.engram_ids)
            if getattr(self, "_host_engram", False):
                for layer in self.layers:
                    emb = getattr(layer.engram, "embed", None) if layer.engram is not None else None
                    if hasattr(emb, "prefetch"):
                        emb.prefetch(hashes[:, :, layer.engram.layer_hash_index])
            else:
                hashes = mx.array(hashes)            # [b, n, n_engram_layers, cols]
        elif self.args.engram_layer_ids:
            raise RuntimeError(
                "model has engram layers but no token map — call "
                "set_token_map() (load.py builds it from the release tokenizer)")

        h = self.embed(input_ids)
        h = mx.broadcast_to(h[:, :, None, :], (b, n, self.hc_mult, h.shape[-1]))

        pre_mix = make_identity_pre_mix(b, n, self.hc_mult)
        shared = SharedState()
        tap_ids = set(self.args.dspark_target_layer_ids) if return_taps else set()
        taps = {}
        # Prefill fence (prefill.py): during a multi-row forward, commit the
        # chunk every ``_fence_every`` layers. Same ops, same dtypes -- only the
        # command-buffer boundaries move, so results are bit-identical to an
        # unfenced run; the point is to bound the transient memory a chunk's
        # lazy graph holds (indexer scores / sparse-attn gathers).
        fence = getattr(self, "_fence_every", 0) if n > 1 else 0
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index])
            # the draft head reads the INPUT of its target layers (reference
            # Transformer.forward), not their output
            if layer.layer_id in tap_ids:
                taps[layer.layer_id] = h.mean(axis=2)
            if self._break_sharing and not layer.attn.is_kv_source and layer.attn.ratio:
                shared_use = SharedState()           # sever the link: consumers see nothing
                shared_use.kv_src_cache = shared.kv_src_cache
                shared_use.index_src_cache = shared.index_src_cache
                zero = mx.full(shared.topk_idxs.shape, -1, dtype=mx.int32) \
                    if shared.topk_idxs is not None else None
                shared_use.topk_idxs = zero
                h, pre_mix = layer(h, pre_mix, start_pos, cache, shared_use)
            else:
                h, pre_mix = layer(h, pre_mix, start_pos, cache, shared)
            if _ASYNC_EVAL:
                mx.async_eval(h, pre_mix)
            if fence and layer.layer_id % fence == fence - 1:
                mx.eval(h, pre_mix)

        h = hc_pre(h, pre_mix)                       # collapse with the last ffn_pre
        h = self.norm(h)
        if last_logit_only:
            h = h[:, -1:]
        if argmax and hasattr(self.head, "argmax"):
            logits = self.head.argmax(h.astype(mx.float32))   # token ids [b, n]
        elif argmax:
            logits = mx.argmax(self.head(h.astype(mx.float32)), axis=-1).astype(mx.int32)
        else:
            logits = self.head(h.astype(mx.float32))   # fp32 logits, as the reference
        cache.offset = start_pos + n
        if return_taps:
            return logits, taps
        return logits
