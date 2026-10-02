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

from . import collective as _coll
import mlx.nn as nn

from .config import ModelArgs
from .hyper_connections import hc_mixes, hc_post, hc_pre, make_identity_pre_mix
from .fakequant import fake_quant_fp8_ue8m0
from .layers import RMSNorm, precompute_freqs_cis, rope_tail
from .moe import ClampedSwiGLU, Gate, SharedExpert
from mlx_lm.models.switch_layers import SwitchGLU

import os

from .hc_fused import hc_expand, mixes_and_collapse

_HC_FUSED = os.environ.get("DSV41_HC_FUSED", "1") == "1"

# DSV41_DRAFT_COMPILE=1: run the draft attention block's pure tensor math
# under ``mx.compile`` (same convention as ``layers.py:110``), so the ~130
# elementwise/reshape dispatches of one draft-stage attention collapse into a
# handful of fused kernels. Default OFF.
#
# Scope, exactly: only the region between the projections and the output
# projection is compiled -- rope, the fp8 fake-quant of the block KV, the
# softmax-with-sink and both einsums. The pieces deliberately left OUT, with
# reasons:
#
# * the q/k/v/wo projections (``wq_a`` ... ``wo_b``): they are EXL3Linear /
#   Exl3Proj modules whose weights are closure-baked at trace time (a compiled
#   function cannot take an nn.Module argument: "Function arguments must be
#   trees of arrays or constants"). Compiling them again per shape would be
#   redundant -- ``exl3_build.py:44`` already compiles every EXL3 projection's
#   own ``__call__`` -- and would bake the module into a second cache keyed
#   only by shape, which is unsafe for any other module instance with the same
#   shapes. See REPORT §"scope" for the probe.
# * ``DraftWindow.chrono`` / the ring write in ``append``: this is cache state
#   mutation (``self.win_kv[:, slots] = ...``) driven by ``n_ctx``, which is a
#   host-side Python int, not an array. It stays eager; the compiled region
#   receives the already-materialised ``ctx`` array.
# * the markov sampling loop (``DSparkHead._draft``): it cannot be compiled as
#   a whole because the SHARDED path calls ``head.combine_argmax`` inside the
#   loop (mtp.py:321-336), a collective whose watchdog sync does
#   ``mx.eval`` -- and ``mx.eval`` inside a compiled trace raises
#   ``ValueError: [eval] Attempting to eval an array during function
#   transformations like compile or vmap is not allowed`` (probed, 2026-10-02).
#   The one-step body (markov_embed -> matmul -> add -> argmax) is pure and is
#   compiled separately in ``DSparkHead._draft`` below; the collectives stay
#   between the compiled steps, exactly as they are eager today.
_DRAFT_COMPILE = os.environ.get("DSV41_DRAFT_COMPILE", "0") == "1"

# Per-shape compiled bodies (see ``_COMPILED_DRAFT_ATTN``; MLX keys its own
# cache per input shape, and this explicit registry mirrors the convention in
# ``indexer.py:_COMPILED_BODIES`` / ``sparse_attention.py:_COMPILED``).
_COMPILED_DRAFT_ATTN: dict = {}
_COMPILED_MARKOV_STEP: dict = {}


def _markov_step(prev, base_row, embed_w, head_w):
    """One Markov sampling step's pure math: embed -> logits row.

    ``prev`` [b] int ids, ``base_row`` [b, V] fp32, ``embed_w`` [V, r] and
    ``head_w`` [V_head, r] are the markov embed / head weights passed as
    arrays (never closure-baked, so a weight reload can never go stale). The
    selection (``combine_argmax`` collective or ``argmax``) stays outside: the
    collective syncs with the host and cannot live in a trace.
    """
    m_emb = embed_w[prev]
    return m_emb, base_row + m_emb @ head_w.T


def _markov_step_c(prev, base_row, embed_w, head_w):
    """Dispatch the markov step: eager (default) or compiled."""
    if not _DRAFT_COMPILE:
        return _markov_step(prev, base_row, embed_w, head_w)
    key = (tuple(prev.shape), tuple(base_row.shape), str(prev.dtype))
    fn = _COMPILED_MARKOV_STEP.get(key)
    if fn is None:
        fn = _COMPILED_MARKOV_STEP[key] = mx.compile(_markov_step)
    return fn(prev, base_row, embed_w, head_w)


def _draft_attn_math(q_pre, kv_pre, ctx, c, s, sink, rd, scale):
    """The draft block's pure tensor math: rope -> fp8 KV -> softmax w/ sink.

    All inputs are arrays (no module state, so one compiled body serves all
    three stages); ``rd`` is a Python constant. Op-for-op the body of the
    eager ``DraftAttention.draft_block`` region between the projections and
    the output projection, so eager and compiled are bit-identical.

    ``scale`` MUST be an ``mx.array``, not a Python float: MLX folds a traced
    float constant into a matmul epilogue with a different rounding than the
    eager ``array * python_float`` promotion (~1 ulp, measured on the CPU
    wheel 0.31.2), which breaks bit-exactness. As a runtime array operand the
    multiply keeps the eager order.
    """
    q = rope_tail(q_pre, rd, c, s)
    kv_blk = fake_quant_fp8_ue8m0(rope_tail(kv_pre, rd, c, s), 32)
    kv = mx.concatenate([ctx.astype(kv_blk.dtype), kv_blk], axis=1)

    # dense bidirectional attention with a learned sink per head
    qf = q.astype(mx.float32)
    kf = kv.astype(mx.float32)
    logits = mx.einsum("blhd,bkd->blhk", qf, kf) * scale
    sinkf = sink.astype(mx.float32).reshape(1, 1, sink.shape[0], 1)
    mmax = mx.maximum(mx.max(logits, axis=-1, keepdims=True), sinkf)
    w = mx.exp(logits - mmax)
    denom = mx.sum(w, axis=-1, keepdims=True) + mx.exp(sinkf - mmax)
    o = mx.einsum("blhk,bkd->blhd", w, kf) / denom
    return rope_tail(o, rd, c, s, inverse=True)


def _draft_attn_math_c(q_pre, kv_pre, ctx, c, s, sink, rd, scale):
    """Dispatch the draft attention math: eager (default) or compiled.

    The eager path takes the same ``mx.array`` scale as the compiled one; both
    are bitwise equal to the pre-refactor ``array * float`` expression (pinned
    by ``test_eager_refactor_unchanged_baseline``).
    """
    scale = mx.array(float(scale), mx.float32)
    if not _DRAFT_COMPILE:
        return _draft_attn_math(q_pre, kv_pre, ctx, c, s, sink, rd, scale)
    key = (tuple(q_pre.shape), tuple(kv_pre.shape), tuple(ctx.shape),
           tuple(c.shape), tuple(s.shape), tuple(sink.shape),
           str(q_pre.dtype), str(ctx.dtype), int(rd), float(scale))
    fn = _COMPILED_DRAFT_ATTN.get(key)
    if fn is None:
        fn = _COMPILED_DRAFT_ATTN[key] = mx.compile(_draft_attn_math)
    return fn(q_pre, kv_pre, ctx, c, s, sink, rd, scale)


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
        keep = min(l, self.window)
        slots = (pos + l - keep + mx.arange(keep)) % self.window
        self.win_kv[:, slots] = kv[:, l - keep:].astype(self.dtype)
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
        kv = rope_tail(kv, rd, c, s)
        return fake_quant_fp8_ue8m0(kv, 32)      # reference act_quant on draft KV

    def append_ctx(self, main_x: mx.array, cache: DraftWindow):
        """Push context KV (from projected target hiddens) into the window."""
        kv = self._kv(main_x, cache.n_ctx)
        cache.append(kv)

    def draft_block(self, x: mx.array, cache: DraftWindow) -> mx.array:
        """Bidirectional attention of the draft block over [ctx window; block].

        The rope + fp8-KV + softmax + einsum region runs through
        ``_draft_attn_math_c`` (eager by default, compiled behind
        ``DSV41_DRAFT_COMPILE=1``); the projections and the output projection
        stay on their own (already compiled) modules, and the cache
        materialisation stays eager. See ``_DRAFT_COMPILE`` for the scope.
        """
        b, l, _ = x.shape
        rd = self.rope_head_dim
        start = cache.n_ctx

        q_pre = self.wq_b(self.q_norm(self.wq_a(x))).reshape(
            b, l, self.n_heads, self.head_dim)
        kv_pre = self.kv_norm(self.wkv(x))
        cos, sin = self._freqs(start + l)
        c, s = cos[start:start + l], sin[start:start + l]
        ctx = cache.chrono()

        o = _draft_attn_math_c(q_pre, kv_pre, ctx, c, s, self.attn_sink, rd,
                               self.softmax_scale)

        o = o.reshape(b, l, self.n_groups, -1)
        if isinstance(self.wo_a, nn.Linear):
            wo_a = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)
            o = mx.einsum("blgd,grd->blgr", o.astype(mx.float32), wo_a.astype(mx.float32))
        else:
            o = self.wo_a(o.astype(x.dtype))
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
        self.group = None

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        xf = x.reshape(-1, self.dim)
        weights, indices = self.gate(xf)
        y = self.experts(xf, indices)
        y = mx.sum(y.astype(mx.float32) * weights[..., None], axis=-2)
        if self.group is not None:          # experts hold one rank's width slice
            y = _coll.all_sum(y, group=self.group)
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
        if _HC_FUSED and self.hc_mult == 4:
            h, attn_pre, attn_post, attn_comb = mixes_and_collapse(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, pre_mix,
                self.hc_iters, self.norm_eps, self.hc_eps)
            h = self.attn.draft_block(self.attn_norm(h), cache)
            x = hc_expand(h, x, attn_post, attn_comb)
            h, ffn_pre, ffn_post, ffn_comb = mixes_and_collapse(
                x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, attn_pre,
                self.hc_iters, self.norm_eps, self.hc_eps)
            h = self.ffn(self.ffn_norm(h))
            x = hc_expand(h, x, ffn_post, ffn_comb)
            return x, ffn_pre
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
    """The 3-stage draft head: anchor + noise block -> logits + confidence.

    ``DSV41_RD_MARKOV_REP=1`` (default OFF, LOCAL ADDITION for the D2 decode
    sync-reduction work) replicates the markov head's vocab projection on
    every rank instead of slicing it: at build time
    (``exl3_build.build_mtp``) the flag keeps ``markov_head`` full-width and
    leaves ``vocab_sharded`` unset, so ``_draft`` below takes the LOCAL
    ``mx.argmax`` per markov step instead of ``head.combine_argmax``.

    The TRADE, stated honestly: the unflagged path's per-step
    ``combine_argmax`` collectives are TINY (a [world, rows, 2] pair buffer,
    ~16 B) but SERIAL (the markov chain needs step k's token to build step
    k+1), i.e. gamma serial collective waits per draft. The flagged path
    needs the FULL-vocab base row on every rank, which on a TP build comes
    from ``ShardedHead.__call__`` -- ONE padded [b, bs, vocab] fp32 all_sum
    (~1.5 MB at b=1, bs=3) per draft. So the flag collapses gamma serial
    waits into one large collective: count 88 -> 86/round, wire bytes UP,
    direction unknown until the Mac A/B. It stays bit-exact either way:
    each logit element is an independent dot product, the padded all_sum
    adds exact zeros, and ``combine_argmax`` returns exactly the global
    first-occurrence argmax that the full-row ``mx.argmax`` computes (ties ->
    lowest vocab id). The draft's OTHER sharded parts (routed expert all_sum)
    are untouched by the flag.
    """

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
        from . import collective as _coll
        with _coll.warm_guard(("draft", int(width or self.block_size))):
            return self._draft(anchor_tokens, embed, head, caches, width=width)

    def _draft(self, anchor_tokens: mx.array, embed, head, caches, width=None):
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

        sharded = getattr(self, "vocab_sharded", False) and hasattr(head, "combine_argmax")
        base_logits = head.local(self.norm(x)) if sharded else head(self.norm(x))

        # sequential first-order Markov sampling left -> right
        prev = anchor_tokens
        toks, m_embeds = [], []
        embed_w = self.markov_embed.weight
        head_w = self.markov_head.weight
        for k in range(bs):
            m_emb, step_logits = _markov_step_c(
                prev, base_logits[:, k, :], embed_w, head_w)
            if sharded:
                nxt = head.combine_argmax(step_logits.astype(mx.float32))
            else:
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
