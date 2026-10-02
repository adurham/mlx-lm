# Copyright © 2026 Adam Durham (hermes-gw)
"""DSV41_DRAFT_COMPILE: the compiled draft math must be BIT-IDENTICAL to eager.

The flag compiles two pure regions of the DSpark draft:
  * the draft attention block's rope/fp8-KV/softmax/einsum body
    (``mtp._draft_attn_math``); and
  * the per-step Markov sampling math (``mtp._markov_step``).

Both are pure functions of their array arguments, so eager vs compiled must be
bitwise equal (this is the ``layers.py:110`` convention, not an approximation).
The tests drive the REAL ``DraftAttention.draft_block`` / ``DSparkHead._draft``
(the eager path must be untouched by the refactor as well), toggle the module
flag, and compare raw bits at several shapes and both entry contexts
(empty window / primed window), for fp32 and bf16 stage inputs.

The collectives stay between compiled steps (``combine_argmax`` in the sharded
path): the tests assert the compiled region contains no collective call and
that the flag-off path is the identity-eager path.
"""

from __future__ import annotations

import os
import unittest

import mlx.core as mx
import mlx.nn as nn

os.environ.setdefault("DSV41_HC_FUSED", "0")  # ops path; the fused path needs Metal

from mlx_lm.models.deepseek_v41 import mtp  # noqa: E402
from mlx_lm.models.deepseek_v41.config import ModelArgs  # noqa: E402


def tiny_args(**over):
    a = ModelArgs(
        dim=64, n_heads=2, head_dim=32, rope_head_dim=8, q_lora_rank=16,
        o_lora_rank=8, o_groups=2, norm_eps=1e-5, moe_inter_dim=32,
        n_routed_experts=4, n_activated_experts=2, swiglu_limit=7.0,
        vocab_size=64, window_size=8, hc_mult=4, hc_sinkhorn_iters=2, hc_eps=1e-6,
        n_mtp_layers=1, dspark_block_size=3, dspark_markov_rank=8,
        dspark_target_layer_ids=(1,), dspark_n_experts=4, dspark_topk=2,
        dspark_noise_token_id=63,
        rope_theta=10000.0, rope_factor=1.0, beta_fast=32.0, beta_slow=1.0,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _bits(a: mx.array, b: mx.array) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == mx.bfloat16:
        return bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)))
    return bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)))


def _force_flag(value: bool) -> bool:
    prev = mtp._DRAFT_COMPILE
    mtp._DRAFT_COMPILE = value
    return prev


def _make_attn(args):
    mx.random.seed(23)
    attn = mtp.DraftAttention(args, 0)
    _randomize(attn)
    return attn


def _make_head(args):
    mx.random.seed(29)
    head = mtp.DSparkHead(args)
    _randomize(head)
    return head


def _randomize(module) -> None:
    from mlx.utils import tree_flatten, tree_unflatten

    items = []
    for name, arr in tree_flatten(module.parameters()):
        if isinstance(arr, mx.array):
            items.append((name, (mx.random.normal(arr.shape) * 0.2).astype(arr.dtype)))
    module.update(tree_unflatten(items))


class _NoCollective:
    """Assert no collective is called inside the block (compiled regions are pure)."""

    def __init__(self):
        self.hits = 0
        self._orig = mx.distributed.all_sum

    def __enter__(self):
        outer = self

        def wrapped(*a, **k):
            outer.hits += 1
            return outer._orig(*a, **k)

        mx.distributed.all_sum = wrapped
        return self

    def __exit__(self, *exc):
        mx.distributed.all_sum = self._orig
        return False


class TestDraftCompile(unittest.TestCase):
    # ---- contract ---------------------------------------------------------
    def test_kill_switch_default_off(self):
        env = os.environ.get("DSV41_DRAFT_COMPILE", "0")
        self.assertEqual(mtp._DRAFT_COMPILE, env == "1")
        if env != "1":
            self.assertFalse(mtp._DRAFT_COMPILE,
                             "flag must default OFF when the env var is unset")

    def test_eager_refactor_unchanged_baseline(self):
        """Flag OFF: draft_block output equals the inline baseline expression."""
        args = tiny_args()
        attn = _make_attn(args)
        cache = mtp.DraftWindow(1, args.window_size, args.head_dim)
        mx.random.seed(31)
        x = (mx.random.normal((1, 3, args.dim)) * 0.5).astype(mx.float32)

        prev = _force_flag(False)
        try:
            got = attn.draft_block(x, cache)
            mx.eval(got)

            # Inline baseline: the pre-refactor op sequence, verbatim.
            b, l = 1, 3
            rd = attn.rope_head_dim
            start = 0
            qr = attn.q_norm(attn.wq_a(x))
            q = attn.wq_b(qr).reshape(b, l, attn.n_heads, attn.head_dim)
            cos, sin = attn._freqs(start + l)
            c_s = cos[start:start + l], sin[start:start + l]
            q = mtp.rope_tail(q, rd, *c_s)
            kv_blk = attn._kv(x, start)
            ctx = cache.chrono()
            kv = mx.concatenate([ctx.astype(kv_blk.dtype), kv_blk], axis=1)
            qf = q.astype(mx.float32)
            kf = kv.astype(mx.float32)
            logits = mx.einsum("blhd,bkd->blhk", qf, kf) * attn.softmax_scale
            sink = attn.attn_sink.astype(mx.float32).reshape(1, 1, attn.n_heads, 1)
            mmax = mx.maximum(mx.max(logits, axis=-1, keepdims=True), sink)
            w = mx.exp(logits - mmax)
            denom = mx.sum(w, axis=-1, keepdims=True) + mx.exp(sink - mmax)
            o = mx.einsum("blhk,bkd->blhd", w, kf) / denom
            o = mtp.rope_tail(o, rd, *c_s, inverse=True)
            o = o.reshape(b, l, attn.n_groups, -1)
            wo_a = attn.wo_a.weight.reshape(attn.n_groups, attn.o_lora_rank, -1)
            o = mx.einsum("blgd,grd->blgr", o.astype(mx.float32), wo_a.astype(mx.float32))
            base = attn.wo_b(o.reshape(b, l, -1).astype(x.dtype))
            mx.eval(base)
            self.assertTrue(_bits(got, base), "eager draft_block drifted from baseline")
        finally:
            mtp._DRAFT_COMPILE = prev

    def test_compiled_draft_block_bitwise(self):
        """Flag ON: draft_block bitwise equals flag-OFF, at several contexts."""
        args = tiny_args()
        attn = _make_attn(args)

        contexts = [
            ("empty-window", 0),
            ("primed-window", 5),
        ]
        for label, n_ctx in contexts:
            for dtype in (mx.float32, mx.bfloat16):
                prev = _force_flag(False)
                try:
                    cache_off = mtp.DraftWindow(1, args.window_size, args.head_dim)
                    if n_ctx:
                        mx.random.seed(37)
                        cache_off.append((mx.random.normal((1, n_ctx, args.head_dim)) * 0.3).astype(mx.float32))
                    mx.random.seed(41)
                    x = (mx.random.normal((1, 3, args.dim)) * 0.5).astype(dtype)
                    off = attn.draft_block(x, cache_off)
                    mx.eval(off)
                finally:
                    mtp._DRAFT_COMPILE = prev

                _force_flag(True)
                try:
                    cache_on = mtp.DraftWindow(1, args.window_size, args.head_dim)
                    if n_ctx:
                        mx.random.seed(37)
                        cache_on.append((mx.random.normal((1, n_ctx, args.head_dim)) * 0.3).astype(mx.float32))
                    mx.random.seed(41)
                    x2 = (mx.random.normal((1, 3, args.dim)) * 0.5).astype(dtype)
                    with _NoCollective() as nc:
                        on = attn.draft_block(x2, cache_on)
                    mx.eval(on)
                finally:
                    mtp._DRAFT_COMPILE = prev

                self.assertEqual(nc.hits, 0,
                                 f"collective inside the compiled region ({label})")
                self.assertTrue(_bits(off, on),
                                f"compiled draft_block differs from eager ({label}, {dtype})")

    def test_compiled_markov_step_bitwise(self):
        """Flag ON: the markov step math is bitwise equal to eager."""
        args = tiny_args()
        head = _make_head(args)
        b, V = 1, args.vocab_size
        mx.random.seed(43)
        prev_id = mx.array([7], dtype=mx.int32)
        base_row = mx.random.normal((b, V)).astype(mx.float32)
        embed_w = head.markov_embed.weight
        head_w = head.markov_head.weight

        flag = _force_flag(False)
        try:
            e_emb, e_log = mtp._markov_step(prev_id, base_row, embed_w, head_w)
            mx.eval(e_emb, e_log)
        finally:
            mtp._DRAFT_COMPILE = flag

        _force_flag(True)
        try:
            c_emb, c_log = mtp._markov_step_c(prev_id, base_row, embed_w, head_w)
            mx.eval(c_emb, c_log)
        finally:
            mtp._DRAFT_COMPILE = flag

        self.assertTrue(_bits(e_emb, c_emb), "markov embed differs")
        self.assertTrue(_bits(e_log, c_log), "markov logits differ")

    def test_full_draft_bitwise(self):
        """The whole DSparkHead._draft (unsharded path) is bit-identical."""
        args = tiny_args()
        head = _make_head(args)

        def run(flag_on: bool):
            _force_flag(flag_on)
            caches = head.make_cache(1)
            mx.random.seed(47)
            ctx = (mx.random.normal((1, 4, len(args.dspark_target_layer_ids) * args.dim)) * 0.4).astype(mx.float32)
            head.append_ctx(ctx, caches)
            embed = nn.Embedding(args.vocab_size, args.dim)
            mx.random.seed(53)
            embed.weight = (mx.random.normal((args.vocab_size, args.dim)) * 0.3).astype(mx.float32)
            head_out = nn.Linear(args.dim, args.vocab_size, bias=False)
            mx.random.seed(59)
            head_out.weight = (mx.random.normal((args.vocab_size, args.dim)) * 0.3).astype(mx.float32)
            anchor = mx.array([3], dtype=mx.int32)
            toks, conf = head.draft(anchor, embed, head_out, caches, width=3)
            mx.eval(toks, conf)
            return toks, conf

        prev = _force_flag(False)
        try:
            t_off, c_off = run(False)
            t_on, c_on = run(True)
        finally:
            mtp._DRAFT_COMPILE = prev

        self.assertTrue(_bits(t_off, t_on), f"draft tokens differ: {t_off} vs {t_on}")
        self.assertTrue(_bits(c_off, c_on), "confidence differs")

    def test_compiled_region_is_actually_compiled(self):
        """The flag must change the code path (an inert flag also 'passes' equality)."""
        args = tiny_args()
        attn = _make_attn(args)
        cache = mtp.DraftWindow(1, args.window_size, args.head_dim)
        mx.random.seed(61)
        x = (mx.random.normal((1, 3, args.dim)) * 0.5).astype(mx.float32)

        prev = _force_flag(False)
        try:
            mtp._COMPILED_DRAFT_ATTN.clear()
            attn.draft_block(x, cache)
            mx.eval(attn.draft_block(x, cache))
            self.assertEqual(len(mtp._COMPILED_DRAFT_ATTN), 0,
                             "flag OFF must not populate the compiled registry")
        finally:
            mtp._DRAFT_COMPILE = prev

        _force_flag(True)
        try:
            mtp._COMPILED_DRAFT_ATTN.clear()
            out = attn.draft_block(x, cache)
            mx.eval(out)
            self.assertEqual(len(mtp._COMPILED_DRAFT_ATTN), 1,
                             "flag ON must populate exactly one compiled body per shape")
            key = next(iter(mtp._COMPILED_DRAFT_ATTN))
            self.assertEqual(key[0], (1, 3, attn.n_heads, attn.head_dim))
        finally:
            mtp._DRAFT_COMPILE = prev


if __name__ == "__main__":
    unittest.main()
