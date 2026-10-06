# Vendored from PipeNetwork/deepseek-v41-mlx (Apache-2.0), upstream a9567ae plus the
# node-local MTP/DSpark port edits (exo docs phase 3). See README.md here.
"""V4.1 attention — MLA over a sliding window plus shared compressed positions.

One layer:

* queries through a low-rank bottleneck (``wq_a`` -> ``q_norm`` -> ``wq_b``),
  rope on the last 64 channels (no per-head q normalization — that was V4);
* a **single** shared 512-d KV vector per position (key and value are the same
  tensor), FP8 fake-quantized whole — rope tail included — before caching;
* every layer attends over the last ``window_size`` tokens; layers with
  ``compress_ratio > 0`` additionally attend over ``index_topk`` compressed
  positions. Only kv_source layers *produce* the compressed cache and only
  index_source layers *select* — everything else consumes through the per-forward
  ``shared`` state, exactly mirroring the reference's SharedAttentionRuntime;
* the attention output gets the rope **inverse** applied to its tail, then the
  grouped block-diagonal ``wo_a`` (einsum over a reshape — which is why wo_a can
  never be MLX-quantized) and the shared ``wo_b``.

Rope: compressing layers use YaRN on ``compress_rope_theta``; pure sliding-window
layers use the base theta with YaRN off. One freqs table serves the queries, the
window KV, the compressed latents (at each group's first-token position), the
indexer, and the inverse on the way out.

The forward is chunk-general: full prefill, chunked prefill and 1-token decode
are the same code path.
"""

from __future__ import annotations

import os

import mlx.core as mx

from . import collective as _coll
import mlx.nn as nn

from .compressor import Compressor
from .config import ModelArgs
from .fakequant import fake_quant_fp4_e4m3, fake_quant_fp8_ue8m0
from .indexer import Indexer
from .layers import RMSNorm, cos_sin_at, rope_freqs, rope_tail
from .sparse_attention import sparse_attn
from ...profiler import span

# Prefill query-row cap handed to ``sparse_attn``'s tiled loop. The loop fences
# (mx.eval) once per query tile, so this is BOTH the compile/tile size and the
# per-call sync count at prefill: at 2048-row chunks, chunk=64 => 32 tiles+syncs
# per layer per call; chunk=256 => 8. Read once at import (like the sparse
# module's own knobs). Does NOT touch decode: the fence is skipped for
# single-row forwards regardless. A/B knob: DSV41_SPARSE_PREFILL_CHUNK.
_PREFILL_CHUNK = int(os.environ.get("DSV41_SPARSE_PREFILL_CHUNK", "64"))


def window_idx_matrix(wp: int, n: int, window: int) -> mx.array:
    """[n, W_eff] window indices into concat([prev_window(wp), chunk(n)]).

    Query i (concat end position wp+i) sees the last ``window`` concat positions
    that exist; -1 pads. W_eff = min(window, wp+n).
    """
    w_eff = min(window, wp + n)
    end = mx.arange(n) + wp
    start = mx.maximum(end - window + 1, 0)
    idx = start[:, None] + mx.arange(w_eff)[None, :]
    return mx.where(idx > end[:, None], mx.array(-1, mx.int32), idx.astype(mx.int32))


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.ratio = args.compress_ratio(layer_id)
        self.eps = args.norm_eps
        self.softmax_scale = args.head_dim ** -0.5

        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = nn.Linear(self.n_heads * self.head_dim // self.n_groups,
                              self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

        # Tensor-parallel group; set when heads / wo_a groups / wo_b inputs
        # hold one rank's slice (n_heads and n_groups are then per-rank).
        self.group = None
        self.is_kv_source = layer_id in args.kv_source_layers
        self.is_index_source = layer_id in args.index_source_layers
        if self.is_kv_source:
            self.compressor = Compressor(args, layer_id)
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id)

        if self.ratio:
            orig_len, theta = args.original_seq_len, args.compress_rope_theta
        else:
            orig_len, theta = 0, args.rope_theta
        self._rope = (args.rope_head_dim, orig_len, theta, args.rope_factor,
                      args.beta_fast, args.beta_slow)
        # One [32] fp32 YaRN frequency vector per layer (was a [2*end_pos, 32]
        # cos/sin table per layer -- ~21 GiB at 1M context). cos/sin are now
        # computed per call for exactly the positions the call needs.
        self._freqvec = rope_freqs(*self._rope)

        # test hooks (negative controls)
        self._break_rope_inverse = False
        self._break_sink = False

    def _cos_sin(self, positions: mx.array):
        """cos/sin for an arbitrary token-position array (bit-exact vs the table)."""
        return cos_sin_at(self._freqvec, positions)

    def __call__(self, x: mx.array, start_pos: int, cache, shared) -> mx.array:
        """x [b, n, dim] (post attn_norm), absolute positions [start_pos, start_pos+n)."""
        bsz, n, _ = x.shape
        rd = self.rope_head_dim
        end_pos = start_pos + n
        # Query rows: exactly the n positions this forward's rows occupy.
        c_q, s_q = self._cos_sin(mx.arange(start_pos, end_pos))

        with span("attn.proj_qkv"):
            # --- queries ---
            qr = self.q_norm(self.wq_a(x))
            q = self.wq_b(qr).reshape(bsz, n, self.n_heads, self.head_dim)
            q = rope_tail(q, rd, c_q, s_q)

            # --- window KV: rope tail, FP8 fake-quant over the whole vector ---
            kv = self.kv_norm(self.wkv(x))
            kv = rope_tail(kv, rd, c_q, s_q)
            kv = fake_quant_fp8_ue8m0(kv, 32)

        with span("attn.kv_cache"):
            lc = cache.layers[self.layer_id]
            prev = lc.window_chrono(start_pos)                      # [b, Wp, hd]
            wp = prev.shape[1]
            kv_all = mx.concatenate([prev.astype(kv.dtype), kv], axis=1) if wp else kv
            idxs = mx.broadcast_to(window_idx_matrix(wp, n, self.window_size)[None],
                                   (bsz, n, min(self.window_size, wp + n)))
            lc.write_window(start_pos, kv)
        offset = wp + n                                          # compressed entries follow

        kv2 = None                                               # second K/V source
        if self.ratio:
            src = shared.kv_src_cache                            # the source layer's LayerCache
            compress_len = end_pos // self.ratio

            latents = None
            if self.is_kv_source:
                with span("attn.compressor"):
                    latents = self.compressor(x, start_pos, lc.comp_state)
                shared.kv_src_cache = src = lc

            # indexer runs on the pre-RoPE latents, before the cache write
            if self.is_index_source:
                if self.indexer.owns_k:
                    if latents is not None:
                        self.indexer.publish_keys(latents, start_pos, self._freqvec, lc)
                    shared.index_src_cache = lc   # an owner always reads its own keys
                if compress_len == 0:
                    cidx = mx.zeros((bsz, n, 0), dtype=mx.int32)
                else:
                    index_k = shared.index_src_cache.index_k[:bsz, :compress_len]
                    with span("attn.indexer"):
                        cidx = self.indexer(x, qr, start_pos, offset, self._freqvec,
                                            index_k, shared)
                shared.topk_idxs = cidx
            else:
                cidx = shared.topk_idxs

            if latents is not None:
                g0 = start_pos // self.ratio
                g = latents.shape[1]
                pos = (g0 + mx.arange(g)) * self.ratio           # group j at position j*ratio
                # The compressor's NON-contiguous latent positions, computed on
                # demand (no table row lookup, no max(upto*2,4096) headroom).
                l_cos, l_sin = self._cos_sin(pos)
                latents = rope_tail(latents, rd, l_cos, l_sin)
                latents = fake_quant_fp4_e4m3(latents, 16)
                lc.comp_kv[:bsz, g0:g0 + g] = latents.astype(lc.dtype)

            if compress_len:
                # Gather-direct: hand the shared comp_kv to sparse_attn as a
                # second source addressed by cidx (which the indexer emitted in
                # the concat space via `offset`) instead of materializing
                # `concat(kv_all, comp)` per layer per forward. comp row i is
                # concat position offset+i, and |kv_all| == offset (split).
                idxs = mx.concatenate([idxs, cidx], axis=-1)
                kv2 = src.comp_kv[:bsz, :compress_len]
        sink = mx.zeros_like(self.attn_sink) if self._break_sink else self.attn_sink
        with span("attn.sdpa"):
            o = sparse_attn(q, kv_all, sink, idxs, self.softmax_scale,
                            chunk=_PREFILL_CHUNK, kv2=kv2, split=offset)

        # --- inverse rope, grouped block-diagonal output LoRA ---
        with span("attn.o_proj"):
            if not self._break_rope_inverse:
                o = rope_tail(o, rd, c_q, s_q, inverse=True)
            o = o.reshape(bsz, n, self.n_groups, -1)
            if isinstance(self.wo_a, nn.Linear):
                wo_a = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)
                o = mx.einsum("bsgd,grd->bsgr", o.astype(mx.float32), wo_a.astype(mx.float32))
            else:
                o = self.wo_a(o)                                 # grouped module -> [b, s, g, r]
            out = self.wo_b(o.reshape(bsz, n, -1).astype(x.dtype))
        if self.group is not None:                   # heads sharded: sum partials
            with span("attn.all_sum"):
                out = _coll.all_sum(out, group=self.group)
        return out
