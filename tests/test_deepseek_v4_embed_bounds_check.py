# Copyright © 2026 Apple Inc.

"""Defense-in-depth bounds-check tests for DeepseekV4Model.embed_tokens.

MLX's array-index gather does not bounds-check: an out-of-range token id
silently returns a zero row. DeepSeek-V4's vision scheme uses SENTINEL ids
(vocab_size + {0..4}) that must never reach the real ``nn.Embedding`` gather
directly -- they are spliced in from the vision tower instead. This module
proves both directions:

  (i)  an out-of-range id now raises ``ValueError`` with an actionable
       message, on both the L=1 (decode-shaped) and L>1 (prefill-shaped)
       forward calls.
  (ii) every legitimate path still works unchanged:
         - plain text-only forward (no sentinel ids at all)
         - the clamped vision path (`deepseek_v4_vision.build_embeddings`,
           which clamps BEFORE the gather)
         - the `patch_embed_tokens`-monkeypatched path (raw sentinel ids
           reach `_inject`, which is expected to handle them -- and must
           NOT trip the model-level check, because `_inject` is marked
           `handles_out_of_range_ids = True`)

Run with the SUBMODULE on PYTHONPATH so this exercises the working tree
rather than an installed copy:

    PYTHONPATH=/Users/adam.durham/scratch/dsv4-taskB/exo/src:/Users/adam.durham/scratch/dsv4-taskB/exo/mlx-lm \
      /Users/adam.durham/repos/exo/.venv/bin/python -m pytest \
      /Users/adam.durham/scratch/dsv4-taskB/exo/mlx-lm/tests/test_deepseek_v4_embed_bounds_check.py -v
"""

import sys
import unittest

import mlx.core as mx
from mlx_lm.models import deepseek_v4 as dsv4


def _tiny_args(vocab_size: int = 32) -> dsv4.ModelArgs:
    """A tiny synthetic (untrained, random-weight) config -- CPU-cheap,
    seconds not minutes, no real checkpoint involved."""
    return dsv4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=1,
        head_dim=8,
        qk_rope_head_dim=4,
        sliding_window=8,
        compress_ratios=[0],
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        moe_intermediate_size=8,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        num_hash_layers=1,
        hc_mult=1,
        hc_sinkhorn_iters=1,
        num_nextn_predict_layers=0,
    )


class TestEmbedBoundsCheckProvenance(unittest.TestCase):
    def test_running_against_submodule_copy(self):
        path = dsv4.__file__
        print(f"\n[provenance] mlx_lm.models.deepseek_v4.__file__ = {path}")
        print(f"[provenance] mlx_lm package             = {sys.modules['mlx_lm'].__file__}")
        self.assertNotIn(
            "site-packages",
            path,
            "tests are importing an INSTALLED mlx_lm copy (site-packages), not "
            "the working tree; re-run with PYTHONPATH pointed at the mlx-lm "
            "checkout you are actually editing.",
        )


class TestOutOfRangeRaises(unittest.TestCase):
    """(i) an out-of-range id must now raise, loudly, with the expected message."""

    def setUp(self):
        self.args = _tiny_args(vocab_size=32)
        self.model = dsv4.Model(self.args)
        self.cache = self.model.make_cache()

    def test_prefill_shaped_out_of_range_raises(self):
        # L=5, one id (32) at exactly vocab_size -- the first invalid id.
        inputs = mx.array([[1, 2, 32, 4, 5]], dtype=mx.int32)
        with self.assertRaises(ValueError) as ctx:
            self.model(inputs, cache=self.cache)
        msg = str(ctx.exception)
        print(f"\n[raised] {msg}")
        self.assertIn("32", msg)  # embedding table size
        self.assertIn("sentinel", msg.lower())
        self.assertIn("embed_tokens", msg)

    def test_decode_shaped_out_of_range_raises(self):
        # L=1 decode-shaped call also must not silently pass a sentinel id.
        inputs = mx.array([[32]], dtype=mx.int32)
        with self.assertRaises(ValueError) as ctx:
            self.model(inputs, cache=self.cache)
        msg = str(ctx.exception)
        print(f"\n[raised, L=1] {msg}")
        self.assertIn("32", msg)

    def test_multiple_sentinel_ids_all_named(self):
        # DSv4 has 5 sentinel types: vocab_size + {0..4}.
        inputs = mx.array([[32, 33, 34, 35, 36]], dtype=mx.int32)
        with self.assertRaises(ValueError) as ctx:
            self.model(inputs, cache=self.cache)
        msg = str(ctx.exception)
        print(f"\n[raised, multi] {msg}")
        for sentinel in (32, 33, 34, 35, 36):
            self.assertIn(str(sentinel), msg)

    def test_far_out_of_range_also_raises(self):
        inputs = mx.array([[999999]], dtype=mx.int32)
        with self.assertRaises(ValueError):
            self.model(inputs, cache=self.cache)


class TestLegitimatePathsUnaffected(unittest.TestCase):
    """(ii) every legitimate path must still work UNCHANGED."""

    def setUp(self):
        self.args = _tiny_args(vocab_size=32)
        self.model = dsv4.Model(self.args)

    def test_text_only_forward_unaffected(self):
        cache = self.model.make_cache()
        inputs = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)
        logits = self.model(inputs, cache=cache)
        mx.eval(logits, [c.state for c in cache])
        self.assertEqual(logits.shape, (1, 8, self.args.vocab_size))
        self.assertFalse(bool(mx.any(mx.isnan(logits)).item()))

    def test_text_only_decode_step_unaffected(self):
        cache = self.model.make_cache()
        # Prefill then a decode-shaped (L=1) step -- must not raise.
        prefill = mx.array([[1, 2, 3]], dtype=mx.int32)
        mx.eval(self.model(prefill, cache=cache))
        decode = mx.array([[4]], dtype=mx.int32)
        logits = self.model(decode, cache=cache)
        mx.eval(logits, [c.state for c in cache])
        self.assertEqual(logits.shape, (1, 1, self.args.vocab_size))

    def test_clamped_call_directly_at_gather_does_not_raise(self):
        """Directly exercises the model-level `embed_tokens` call the way
        `deepseek_v4_vision.build_embeddings` does: clamp BEFORE calling
        `inner.embed_tokens(clamped)`. This must never raise -- the whole
        point of clamping is to make the id valid before the real gather."""
        inner = self.model.model
        vocab_size = self.args.vocab_size
        raw_ids = mx.array([[1, 2, vocab_size, vocab_size + 4, 5]], dtype=mx.int32)
        clamped = mx.minimum(raw_ids, vocab_size - 1)
        # Must NOT raise, and must produce real (non-degenerate) embeddings.
        embeddings = inner.embed_tokens(clamped)
        mx.eval(embeddings)
        self.assertEqual(embeddings.shape, (1, 5, self.args.hidden_size))

    def test_full_model_forward_with_clamped_ids_unaffected(self):
        """The `_apply_image_visibility` bounds check inside `_forward_steps`
        looks at RAW `inputs`, but `self.embed_tokens(inputs)` only ever
        needs to see the CLAMPED ids -- so this exercises the whole
        `_forward_steps` including `_assert_embeddable`, but with an inputs
        array that is already <vocab_size (i.e. this simulates the actual
        clamp-then-call ordering `build_embeddings` uses, verifying the
        model-level check does not object to legitimately in-range ids that
        merely LOOK like they were once out of range)."""
        cache = self.model.make_cache()
        vocab_size = self.args.vocab_size
        raw_ids = [1, 2, vocab_size, vocab_size + 4, 5]
        clamped_ids = [min(t, vocab_size - 1) for t in raw_ids]
        inputs = mx.array([clamped_ids], dtype=mx.int32)
        logits = self.model(inputs, cache=cache)
        mx.eval(logits, [c.state for c in cache])
        self.assertEqual(logits.shape, (1, len(clamped_ids), vocab_size))

    def test_patch_embed_tokens_style_monkeypatch_bypasses_check(self):
        """Simulates exo's `patch_embed_tokens`: install a callable on
        `inner.embed_tokens` that is NOT an `nn.Embedding`, marked
        `handles_out_of_range_ids = True`, and verify the model-level check
        in `_forward_steps` defers to it (does not raise) even though the
        raw `inputs` carries sentinel ids -- exactly the situation
        `patch_embed_tokens` creates during a real vision prefill chunk."""
        inner = self.model.model
        original_embed = inner.embed_tokens
        vocab_size = self.args.vocab_size
        hidden_size = self.args.hidden_size

        def _inject(input_ids):
            # A trivial stand-in for the real splice: clamp, embed, done.
            clamped = mx.minimum(input_ids, vocab_size - 1)
            return original_embed(clamped)

        _inject.handles_out_of_range_ids = True
        inner.embed_tokens = _inject
        try:
            cache = self.model.make_cache()
            inputs = mx.array(
                [[1, 2, vocab_size, vocab_size + 4, 5]], dtype=mx.int32
            )
            # Must NOT raise -- the marker tells _forward_steps this
            # callable already handles out-of-range ids.
            logits = self.model(inputs, cache=cache)
            mx.eval(logits, [c.state for c in cache])
            self.assertEqual(logits.shape, (1, 5, vocab_size))
        finally:
            inner.embed_tokens = original_embed

    def test_unmarked_monkeypatch_without_flag_still_checked(self):
        """Sanity: if a monkeypatch does NOT set the marker attribute, the
        model-level check still fires (proves the guard is opt-in, not
        opt-out-by-default -- i.e. safety is the default posture)."""
        inner = self.model.model
        original_embed = inner.embed_tokens
        vocab_size = self.args.vocab_size

        def _unmarked_inject(input_ids):
            clamped = mx.minimum(input_ids, vocab_size - 1)
            return original_embed(clamped)

        # Deliberately NOT setting handles_out_of_range_ids.
        inner.embed_tokens = _unmarked_inject
        try:
            cache = self.model.make_cache()
            inputs = mx.array([[1, 2, vocab_size, 5]], dtype=mx.int32)
            with self.assertRaises(ValueError):
                self.model(inputs, cache=cache)
        finally:
            inner.embed_tokens = original_embed


if __name__ == "__main__":
    unittest.main()
