# Vendored from PipeNetwork/deepseek-v41-mlx (Apache-2.0), upstream a9567ae plus the
# node-local MTP/DSpark port edits (exo docs phase 3). See README.md here.
"""DSpark 3-stage semi-autoregressive speculative draft head (MTP).

Structure read off the checkpoint index (2,401 `mtp.*` tensors in the native
release, shards 44-46), not guessed:

  mtp.{0,1,2}.attn.{wq_a,q_norm,wq_b,wkv,kv_norm,wo_a,wo_b,attn_sink}
  mtp.{0,1,2}.ffn.{gate,shared_experts,experts.{0..127}}
  mtp.{0,1,2}.{attn,ffn}_norm, hc_{attn,ffn}_{fn,base,scale}
  mtp.0.main_proj, mtp.0.main_norm                 (stage-0 target-ctx projection)
  mtp.2.norm, mtp.2.markov_head.{embed,head}, mtp.2.confidence_head.proj

Differences from the main stack that must not be glossed over:

* the draft MoE is **128 routed experts, top-3** (`dspark_n_routed_experts` /
  `dspark_num_experts_per_tok`), not 384/top-6 like the body;
* draft attention is LOCAL ONLY — pure sliding window, no compressed-KV /
  indexer / candidate machinery (`compress_ratios[40:43] == [0,0,0]`);
* there is **no `hc_head` tensor anywhere in the release**. The main stack's
  final collapse is done with the last block's `ffn_pre` (see `model.py`), and
  the draft follows the same convention: each stage returns its `ffn_pre` and
  the module collapses with the last stage's. `HyperHead` is therefore not
  needed and its absence is consistent, not a missing weight.

The draft is ONE parallel forward over ``[anchor, noise x (block-1)]`` producing
base logits for every block position, then a rank-``markov_rank`` first-order
transition bias injects intra-block dependency during a sequential sampling
loop. Context conditioning: the target model's hc-mean hidden states at
``dspark_target_layer_ids`` are concatenated, projected by stage-0's main_proj,
and appended to every stage's rotating window as KV.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs
from .hyper_connections import hc_mixes, hc_post, hc_pre, make_identity_pre_mix
from .layers import RMSNorm, precompute_freqs_cis, rope_tail
from .moe import ClampedSwiGLU, Gate, SharedExpert
from mlx_lm.models.switch_layers import SwitchGLU


class DraftWindow:
    """Rotating window KV for one draft stage, plus the draft block's own KV.

    ``win`` holds the context entries (positions written by ``append_ctx``);
    ``blk`` holds the current block's KV, valid only for the duration of one
    draft call — the caller trims it back out, exactly as production does.
    """

    def __init__(self, bsz, window, head_dim, dtype=mx.float32):
        self.window = window
        self.head_dim = head_dim
        self.dtype = dtype
        self.win_kv = mx.zeros((bsz, window, head_dim), dtype=dtype)
        self.n_ctx = 0

    def append(self, kv: mx.array):
        """Append L context KV rows into the ring; returns the current ring."""
        b, l, _ = kv.shape
        pos = self.n_ctx
        slots = (pos + mx.arange(l)) % self.window
        # scatter through a take/put pair so wrap-around is correct
        cur = self.win_kv
        for i in range(l):
            idx = int(slots[i])
            cur[:, idx] = kv[:, i]
        self.win_kv = cur
        self.n_ctx = pos + l
        return self.win_kv

    def chrono(self):
        """Context window in chronological order [b, Wp, hd]."""
        wp = min(self.n_ctx, self.window)
        if wp == 0:
            return self.win_kv[:, :0]
        first = self.n_ctx - wp
        slots = (first + mx.arange(wp)) % self.window
        return self.win_kv[:, slots]


class DraftAttention(nn.Module):
    """Local attention for a draft stage: projections shared by both entry points."""

    def __init__(self, args: ModelArgs, stage_idx: int):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.n_groups = args.o_groups
        self.eps = args.norm_eps
        self.softmax_scale = args.head_dim ** -0.5
        self.stage_idx = stage_idx

        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = nn.Linear(self.n_heads * self.head_dim // self.n_groups,
                              self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

        # draft attention never compresses: YaRN off, base theta
        self._rope = (args.rope_head_dim, 0, args.rope_theta, args.rope_factor,
                      args.beta_fast, args.beta_slow)
        self._cos = None
        self._sin = None

    def _freqs(self, upto: int):
        if self._cos is None or self._cos.shape[0] < upto:
            rd, orig_len, theta, factor, bf, bs = self._rope
            self._cos, self._sin = precompute_freqs_cis(
                rd, max(upto * 2, 4096), orig_len, theta, factor, bf, bs)
        return self._cos, self._sin

    def _kv(self, x: mx.array, start: int):
        rd = self.rope_head_dim
        cos, sin = self._freqs(start + x.shape[1])
        c, s = cos[start:start + x.shape[1]], sin[start:start + x.shape[1]]
        kv = self.kv_norm(self.wkv(x))
        return rope_tail(kv, rd, c, s)

    def append_ctx(self, main_x: mx.array, cache: DraftWindow):
        """Push context KV (from projected target hiddens) into the window."""
        kv = self._kv(main_x, cache.n_ctx)
        cache.append(kv)

    def draft_block(self, x: mx.array, cache: DraftWindow) -> mx.array:
        """Bidirectional attention of the draft block over [ctx window; block]."""
        b, l, _ = x.shape
        rd = self.rope_head_dim
        start = cache.n_ctx

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(b, l, self.n_heads, self.head_dim)
        cos, sin = self._freqs(start + l)
        q = rope_tail(q, rd, cos[start:start + l], sin[start:start + l])
        kv_blk = self._kv(x, start)

        ctx = cache.chrono()
        kv = mx.concatenate([ctx.astype(kv_blk.dtype), kv_blk], axis=1)

        # dense bidirectional attention with a learned sink per head
        qf = q.astype(mx.float32)
        kf = kv.astype(mx.float32)
        logits = mx.einsum("blhd,bkd->blhk", qf, kf) * self.softmax_scale
        sink = self.attn_sink.astype(mx.float32).reshape(1, 1, self.n_heads, 1)
        mmax = mx.maximum(mx.max(logits, axis=-1, keepdims=True), sink)
        w = mx.exp(logits - mmax)
        denom = mx.sum(w, axis=-1, keepdims=True) + mx.exp(sink - mmax)
        o = mx.einsum("blhk,bkd->blhd", w, kf) / denom

        o = rope_tail(o, rd, cos[start:start + l], sin[start:start + l], inverse=True)
        o = o.reshape(b, l, self.n_groups, -1)
        wo_a = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)
        o = mx.einsum("blgd,grd->blgr", o.astype(mx.float32), wo_a.astype(mx.float32))
        return self.wo_b(o.reshape(b, l, -1).astype(x.dtype))


class DraftMoE(nn.Module):
    """The draft MoE: 128 routed experts, top-3 (NOT the body's 384/top-6)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_experts = args.dspark_n_experts or args.n_routed_experts
        self.topk = args.dspark_topk or args.n_activated_experts
        self.gate = Gate(args, n_experts=self.n_experts, topk=self.topk)
        self.experts = SwitchGLU(args.dim, args.moe_inter_dim, self.n_experts,
                                 activation=ClampedSwiGLU(args.swiglu_limit), bias=False)
        self.shared_experts = SharedExpert(args.dim, args.moe_inter_dim, args.swiglu_limit)

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        xf = x.reshape(-1, self.dim)
        weights, indices = self.gate(xf)
        y = self.experts(xf, indices)
        y = mx.sum(y.astype(mx.float32) * weights[..., None], axis=-2)
        y = y + self.shared_experts(xf).astype(mx.float32)
        return y.reshape(shape).astype(x.dtype)


class DraftStage(nn.Module):
    """One draft block: attn + ffn with the same staggered hyper-connections."""

    def __init__(self, args: ModelArgs, stage_idx: int):
        super().__init__()
        self.stage_idx = stage_idx
        self.hc_mult = args.hc_mult
        self.hc_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.norm_eps = args.norm_eps

        self.attn = DraftAttention(args, stage_idx)
        self.ffn = DraftMoE(args)
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

    def __call__(self, x, pre_mix, cache):
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            self.hc_mult, self.hc_iters, self.norm_eps, self.hc_eps)
        h = hc_pre(x, pre_mix)
        h = self.attn.draft_block(self.attn_norm(h), cache)
        x = hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            self.hc_mult, self.hc_iters, self.norm_eps, self.hc_eps)
        h = hc_pre(x, attn_pre)
        h = self.ffn(self.ffn_norm(h))
        x = hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre


class DSparkHead(nn.Module):
    """The 3-stage draft head: anchor + noise block -> logits + confidence."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_stages = args.n_mtp_layers or 3
        self.block_size = args.dspark_block_size or 5
        self.noise_token_id = args.dspark_noise_token_id or 128799
        self.markov_rank = args.dspark_markov_rank or 256
        self.hc_mult = args.hc_mult
        self.eps = args.norm_eps

        self.stages = [DraftStage(args, i) for i in range(self.n_stages)]

        h = args.dim
        n_tap = len(args.dspark_target_layer_ids) or 3
        self.main_proj = nn.Linear(n_tap * h, h, bias=False)
        self.main_norm = RMSNorm(h, self.eps)

        self.norm = RMSNorm(h, self.eps)
        self.markov_embed = nn.Embedding(args.vocab_size, self.markov_rank)
        self.markov_head = nn.Linear(self.markov_rank, args.vocab_size, bias=False)
        self.confidence_proj = nn.Linear(h + self.markov_rank, 1, bias=False)

    def make_cache(self, bsz=1):
        return [DraftWindow(bsz, self.args.window_size, self.args.head_dim)
                for _ in self.stages]

    def append_ctx(self, main_hidden_cat: mx.array, caches) -> None:
        """Project the concatenated target hiddens and push KV into every stage."""
        main_x = self.main_norm(self.main_proj(main_hidden_cat))
        for stage, c in zip(self.stages, caches):
            stage.attn.append_ctx(main_x, c)

    def draft(self, anchor_tokens: mx.array, embed, head, caches, width=None):
        """One parallel draft round -> (draft_tokens, confidence).

        ``anchor_tokens`` [b]. Position 0 of the block IS the anchor, so the
        returned draft for position k predicts anchor+k+1.
        """
        b = anchor_tokens.shape[0]
        bs = width or self.block_size
        block_ids = mx.concatenate([
            anchor_tokens[:, None],
            mx.full((b, bs - 1), self.noise_token_id, dtype=anchor_tokens.dtype),
        ], axis=1)

        x = embed(block_ids)
        x = mx.broadcast_to(x[:, :, None, :], (b, bs, self.hc_mult, x.shape[-1]))
        x = mx.contiguous(x)

        pre_mix = make_identity_pre_mix(b, bs, self.hc_mult)
        for stage, c in zip(self.stages, caches):
            x, pre_mix = stage(x, pre_mix, c)
        x = hc_pre(x, pre_mix)            # collapse with the last stage's ffn_pre

        base_logits = head(self.norm(x))  # [b, bs, V]

        # sequential first-order Markov sampling left -> right
        prev = anchor_tokens
        toks, m_embeds = [], []
        for k in range(bs):
            m_emb = self.markov_embed(prev)
            step_logits = base_logits[:, k, :] + self.markov_head(m_emb)
            nxt = mx.argmax(step_logits, axis=-1)
            toks.append(nxt)
            m_embeds.append(m_emb[:, None, :])
            prev = nxt

        draft_tokens = mx.stack(toks, axis=1)             # [b, bs]
        m_embed = mx.concatenate(m_embeds, axis=1)        # [b, bs, r]
        conf_in = mx.concatenate([x.astype(mx.float32),
                                  m_embed.astype(mx.float32)], axis=-1)
        confidence = mx.sigmoid(self.confidence_proj(conf_in).squeeze(-1))
        return draft_tokens, confidence
