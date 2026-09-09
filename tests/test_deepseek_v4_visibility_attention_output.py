# Copyright © 2026 Apple Inc.

"""Phase 3b POSITIVE correctness tests: image-span visibility in real attention.

Gap-closing follow-up to ``test_deepseek_v4_attention_visibility.py``. That file
proves the visibility MASK carries the reference's visible set, and that the
flag is inert on text-only input. What it does not prove — and what this file
does — is that **the attention OUTPUT is correct when a real image span is
present**, through the fork's actual attention modules and every optimized
fast-path that can be reached while visibility is active.

The oracle here is deliberately *not* the fork's mask. It is a naive,
row-by-row, gather-based attention written straight from the reference's
visible-set definition::

    left_i   = clamp(i - span_start, max=max_image_tokens-1)  if i in a span else 0
    right_i  = clamp(span_end - i,   max=max_image_tokens)    if i in a span else 0
    start_i  = max(0, i - max(window_size - 1, left_i))
    visible_i = { j : start_i <= j <= i + right_i and j < start_i + width }
    width    = min(seqlen, window_size + max_image_tokens)

For each query row it explicitly builds that key set, gathers those keys, and
computes ``softmax(q·kᵀ/√d + sink) @ v`` in float64. Nothing in the oracle
reads the fork's mask code, so agreement is real evidence rather than a
tautology.

Run on a machine with MLX::

    PYTHONPATH=<repo>/mlx-lm <venv>/bin/python -m pytest \\
        tests/test_deepseek_v4_visibility_attention_output.py -q
"""

import importlib
import os
import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models import deepseek_v4 as dsv4
from mlx_lm.models.base import create_causal_mask

from _dsv4_torch_reference import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_START,
)

VOCAB = 512
WINDOW = 128
MAX_IMAGE_TOKENS = 384


def build_ids(seqlen, spans, rng, vocab=VOCAB):
    """Text ids with ``spans`` = [(start, length), ...] image spans laid in."""
    ids = rng.integers(0, vocab, size=(1, seqlen)).astype(np.int64)
    for start, length in spans:
        assert length >= 2
        ids[0, start] = vocab + IMAGE_START
        ids[0, start + 1 : start + length - 1] = rng.choice(
            [vocab + IMAGE, vocab + IMAGE_PAD, vocab + IMAGE_NEW_LINE],
            size=length - 2,
        )
        ids[0, start + length - 1] = vocab + IMAGE_END
    return ids


# ───────────────────────── the independent oracle ─────────────────────────────
def oracle_visible_sets(ids, window_size=WINDOW, max_image_tokens=MAX_IMAGE_TOKENS):
    """Per-row visible key sets, from the reference definition, in pure Python.

    Written from the span semantics directly — no cumsum/cummax tricks, no MLX,
    no torch, and no reference to `_get_image_visible`. A plain scan marks which
    positions are inside a span and where that span starts and ends; the rest is
    the reference's arithmetic transcribed literally.
    """
    seqlen = ids.shape[1]
    row = ids[0]
    inside = [None] * seqlen  # (span_start, span_end) or None
    open_start = None
    for i in range(seqlen):
        if row[i] == VOCAB + IMAGE_START:
            open_start = i
        if open_start is not None:
            inside[i] = open_start
        if row[i] == VOCAB + IMAGE_END:
            open_start = None
    # Next IMAGE_END at or after i (the reference's `ends`, defaulting to seqlen)
    next_end = [seqlen] * seqlen
    nxt = seqlen
    for i in range(seqlen - 1, -1, -1):
        if row[i] == VOCAB + IMAGE_END:
            nxt = i
        next_end[i] = nxt

    width = min(seqlen, window_size + max_image_tokens)
    sets = []
    for i in range(seqlen):
        if inside[i] is None:
            left = right = 0
        else:
            left = min(i - inside[i], max_image_tokens - 1)
            right = min(next_end[i] - i, max_image_tokens)
        left_add = max(0, left - (window_size - 1))
        start = max(0, i - (window_size - 1) - left_add)
        hi = min(i + right, start + width - 1, seqlen - 1)
        sets.append(list(range(start, hi + 1)))
    return sets


def oracle_attention(q, kv, ids, sinks, scale, window_size=WINDOW):
    """Naive per-row gather attention over the oracle's visible key sets.

    ``q``: (B, H, L, D)  ``kv``: (B, 1, S, D)  ``sinks``: (H,) or None.
    Computed in float64 and returned as float64 (B, H, L, D).
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(kv, dtype=np.float64)[:, 0]  # (B, S, D)
    B, H, L, D = q.shape
    sets = oracle_visible_sets(ids, window_size=window_size)
    sinks_np = None if sinks is None else np.asarray(sinks, dtype=np.float64)
    out = np.zeros((B, H, L, D), dtype=np.float64)
    for b in range(B):
        for i in range(L):
            cols = np.asarray(sets[i], dtype=np.int64)
            kk = k[b, cols]                              # (n_vis, D)
            logits = (q[b, :, i, :] @ kk.T) * scale      # (H, n_vis)
            if sinks_np is not None:
                logits = np.concatenate(
                    [logits, sinks_np[:, None]], axis=-1
                )
            m = logits.max(axis=-1, keepdims=True)
            e = np.exp(logits - m)
            p = e / e.sum(axis=-1, keepdims=True)
            if sinks_np is not None:
                p = p[:, :-1]                            # sink absorbs mass only
            out[b, :, i, :] = p @ kk
    return out


def _report(tag, got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    diff = np.abs(got - want)
    denom = np.abs(want).max() or 1.0
    print(
        f"\n[{tag}] shape={got.shape}\n"
        f"    max abs diff {diff.max():.3e}   mean abs diff {diff.mean():.3e}\n"
        f"    max REL diff {diff.max() / denom:.3e} "
        f"(vs |oracle|max {np.abs(want).max():.4f})"
    )
    return float(diff.max()), float(diff.max() / denom)


def _config(n_layers=4, ratios=(0, 4, 128, 0), vision_n_layers=32):
    return dsv4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        head_dim=32,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        qk_rope_head_dim=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        sliding_window=WINDOW,
        compress_ratios=list(ratios)[:n_layers],
        index_topk=8,
        index_n_heads=4,
        index_head_dim=16,
        vision_n_layers=vision_n_layers,
        vision_max_n_token=MAX_IMAGE_TOKENS,
    )


class TestOracleIsIndependentlyCorrect(unittest.TestCase):
    """Sanity-check the oracle itself before trusting it as a reference.

    On TEXT-ONLY input the oracle's visible sets must be exactly the plain
    128-wide causal window — if they were not, every downstream comparison
    would be measuring the oracle's bug.
    """

    def test_text_only_sets_equal_causal_window(self):
        seqlen = 384
        ids = (
            np.random.default_rng(1).integers(0, VOCAB, size=(1, seqlen))
        ).astype(np.int64)
        sets = oracle_visible_sets(ids)
        causal = np.asarray(create_causal_mask(seqlen, 0, window_size=WINDOW))
        bad = 0
        for i in range(seqlen):
            if set(sets[i]) != set(np.flatnonzero(causal[i]).tolist()):
                bad += 1
        print(
            f"\n[oracle self-check] text-only, seqlen={seqlen}: "
            f"{seqlen - bad}/{seqlen} rows equal create_causal_mask's window"
        )
        self.assertEqual(bad, 0)

    def test_span_rows_are_wider_and_noncausal(self):
        """With a span the oracle must add keys, incl. keys AHEAD of the row."""
        seqlen, start, length = 512, 100, 200
        ids = build_ids(seqlen, [(start, length)], np.random.default_rng(2))
        sets = oracle_visible_sets(ids)
        span_end = start + length - 1
        # The span's first token must see the span's LAST token (non-causal).
        self.assertIn(span_end, sets[start])
        # A row inside the span reaches further back than the window allows.
        self.assertLess(min(sets[span_end]), span_end - (WINDOW - 1))
        total = sum(len(s) for s in sets)
        causal = np.asarray(create_causal_mask(seqlen, 0, window_size=WINDOW))
        print(
            f"\n[oracle span check] span=[{start},{span_end}]: "
            f"{total} visible keys vs {int(causal.sum())} causal-window keys "
            f"(+{total - int(causal.sum())}); row {start} sees key {span_end} "
            f"({span_end - start} AHEAD of itself); row {span_end} reaches back "
            f"to {min(sets[span_end])} (window would stop at "
            f"{span_end - WINDOW + 1})"
        )


class TestForkMaskMatchesOracleSets(unittest.TestCase):
    """The fork's runtime mask must be exactly the oracle's visible sets."""

    CASES = {
        "one_span": (384, [(50, 200)]),
        "two_spans": (384, [(20, 40), (200, 100)]),
        "span_at_zero": (256, [(0, 64)]),
        "span_at_end": (256, [(200, 56)]),
        "span_over_max_tokens": (900, [(100, 500)]),
        "unterminated_span": (256, [(100, 40)]),  # END stripped below
    }

    def test_all_cases(self):
        os.environ["EXO_DSV4_IMAGE_VISIBILITY"] = "1"
        try:
            importlib.reload(dsv4)
            model = dsv4.Model(_config()).model
            for name, (seqlen, spans) in self.CASES.items():
                with self.subTest(case=name):
                    rng = np.random.default_rng(abs(hash(name)) % 2**31)
                    ids = build_ids(seqlen, spans, rng)
                    if name == "unterminated_span":
                        ids[0, spans[0][0] + spans[0][1] - 1] = 7  # drop IMAGE_END
                    mask = create_causal_mask(seqlen, 0, window_size=WINDOW)
                    out = model._apply_image_visibility(
                        mask, mx.array(ids.astype(np.int32)), None
                    )
                    mx.eval(out)
                    self.assertEqual(
                        out.ndim, 4, "mask must be 4-D (B,H,L,S) for downstream"
                    )
                    got = np.asarray(out)[0, 0]
                    want = oracle_visible_sets(ids)
                    bad = sum(
                        1
                        for i in range(seqlen)
                        if set(np.flatnonzero(got[i]).tolist()) != set(want[i])
                    )
                    print(
                        f"\n[mask == oracle] {name}: seqlen={seqlen} "
                        f"spans={spans} -> {seqlen - bad}/{seqlen} rows' visible "
                        f"key sets identical ({int(got.sum())} keys total)"
                    )
                    self.assertEqual(bad, 0)
        finally:
            del os.environ["EXO_DSV4_IMAGE_VISIBILITY"]
            importlib.reload(dsv4)


class TestLocalAttentionOutputMatchesOracle(unittest.TestCase):
    """GAP 1, core: real LocalAttention SDPA output vs the gather oracle.

    Runs the fork's actual ``LocalAttention.__call__`` with the visibility mask
    and compares the post-SDPA attention output against a naive per-row gather
    attention. Also asserts the flag-OFF output DIFFERS (so the test cannot
    pass vacuously) and that rows OUTSIDE the span are unchanged by the flag.
    """

    def _run(self, seqlen, spans, seed):
        os.environ["EXO_DSV4_IMAGE_VISIBILITY"] = "1"
        try:
            importlib.reload(dsv4)
            rng = np.random.default_rng(seed)
            ids = build_ids(seqlen, spans, rng)
            cfg = _config(n_layers=1, ratios=(0,))
            model = dsv4.Model(cfg).model
            attn = dsv4.LocalAttention(cfg, 0)

            H, D = cfg.num_attention_heads, cfg.head_dim
            q = mx.array((rng.standard_normal((1, H, seqlen, D)) * 0.5).astype(np.float32))
            kv = mx.array((rng.standard_normal((1, 1, seqlen, D)) * 0.5).astype(np.float32))
            sinks = mx.array((rng.standard_normal(H) * 0.3).astype(np.float32))
            scale = float(D**-0.5)

            base = create_causal_mask(seqlen, 0, window_size=WINDOW)
            vis = model._apply_image_visibility(
                base, mx.array(ids.astype(np.int32)), None
            )
            mx.eval(vis)

            from mlx_lm.models.base import scaled_dot_product_attention as sdpa

            on = sdpa(q, kv, kv, cache=None, scale=scale, mask=vis, sinks=sinks)
            off = sdpa(
                q, kv, kv, cache=None, scale=scale, mask=base[None, None], sinks=sinks
            )
            mx.eval(on, off)

            oracle = oracle_attention(q, kv, ids, sinks, scale)
            return np.asarray(on), np.asarray(off), oracle, ids
        finally:
            del os.environ["EXO_DSV4_IMAGE_VISIBILITY"]
            importlib.reload(dsv4)

    def _check(self, tag, seqlen, spans, seed):
        on, off, oracle, ids = self._run(seqlen, spans, seed)
        max_abs, max_rel = _report(f"GAP1 {tag}: fork SDPA vs gather ORACLE", on, oracle)
        self.assertLess(max_rel, 2e-6, "fork attention output != independent oracle")

        # Control: the flag MUST change the answer, or this proves nothing.
        d_flag = np.abs(on.astype(np.float64) - off.astype(np.float64))
        changed_rows = int((d_flag.max(axis=(1, 3)) > 0).sum())
        span_rows = {
            i for i, s in enumerate(oracle_visible_sets(ids))
            if len(s) > len(range(max(0, i - WINDOW + 1), i + 1))
        }
        outside_changed = [
            i for i in range(seqlen)
            if i not in span_rows and d_flag[0, :, i, :].max() > 0
        ]
        print(
            f"[GAP1 {tag} control] flag ON vs OFF: max abs diff "
            f"{d_flag.max():.3e} on {changed_rows}/{seqlen} rows; "
            f"{len(span_rows)} rows have a widened visible set; rows OUTSIDE "
            f"any widened span that changed: {len(outside_changed)} (want 0)"
        )
        self.assertGreater(d_flag.max(), 0.0, "flag ON changed nothing — vacuous")
        self.assertEqual(
            outside_changed, [], "non-span rows must be bit-identical to flag OFF"
        )

    def test_single_span(self):
        self._check("single span [50,249] seqlen=384", 384, [(50, 200)], 4001)

    def test_two_spans(self):
        self._check("two spans seqlen=384", 384, [(20, 40), (200, 100)], 4002)

    def test_span_at_zero(self):
        self._check("span at 0 seqlen=256", 256, [(0, 64)], 4003)

    def test_span_longer_than_max_image_tokens(self):
        self._check("span 500 > max_image_tokens seqlen=900", 900, [(100, 500)], 4004)


class TestFullModelForwardWithRealSpan(unittest.TestCase):
    """GAP 1, end-to-end: a real multi-layer forward with a real span.

    Exercises the whole stack — create_attention_mask, _apply_image_visibility,
    LocalAttention + CompressedAttention + SparseCompressedAttention, the MoE
    gate — on input that actually contains an [IMAGE_START..IMAGE_END] span.
    Regression guard for the 4-D mask rank bug: with a 3-D mask this raises
    "not enough values to unpack (expected 4, got 3)" in `_extend_mask`.
    """

    @staticmethod
    def _forward(seed, ids, ratios=(0, 4, 128, 0)):
        model = dsv4.Model(_config(n_layers=len(ratios), ratios=ratios))
        rng = np.random.default_rng(seed)

        def fill(t):
            if isinstance(t, dict):
                return {k: fill(v) for k, v in t.items()}
            if isinstance(t, list):
                return [fill(v) for v in t]
            if isinstance(t, mx.array):
                if t.dtype == mx.int32:
                    return mx.array(rng.integers(0, 8, size=t.shape).astype(np.int32))
                return mx.array(
                    (rng.standard_normal(t.shape) * 0.05).astype(np.float32)
                )
            return t

        model.update(fill(model.parameters()))
        logits = model(mx.array(ids), cache=model.make_cache())
        mx.eval(logits)
        return np.asarray(logits)

    def test_forward_runs_and_changes_logits(self):
        seqlen = 384
        ids = build_ids(seqlen, [(50, 200)], np.random.default_rng(556)).astype(
            np.int32
        )
        # Image ids are >= vocab_size; the embedding is only defined below that.
        # Phase 4 merges real image embeddings in place. Here clamp the EMBED
        # input while keeping the true ids for the MASK, which is what this
        # test is about — mirroring the reference's merge_image_embeddings.
        n_img = int((ids >= VOCAB).sum())
        embed_ids = np.minimum(ids, VOCAB - 1)

        self.assertFalse(dsv4._IMAGE_VISIBILITY)
        off = self._forward(777, embed_ids)

        os.environ["EXO_DSV4_IMAGE_VISIBILITY"] = "1"
        try:
            importlib.reload(dsv4)
            # The mask reads the CLAMPED ids here (same array), so this run is
            # the flag-ON/no-span control; the real-span run is below.
            on_nospan = self._forward(777, embed_ids)
        finally:
            del os.environ["EXO_DSV4_IMAGE_VISIBILITY"]
            importlib.reload(dsv4)

        eq = int((off == on_nospan).sum())
        print(
            f"\n[GAP1 e2e control] clamped (no image ids) flag OFF vs ON: "
            f"{eq}/{off.size} logits identical (expect all — no span visible)"
        )
        self.assertEqual(eq, off.size)

    def test_forward_with_real_image_ids_runs_clean(self):
        """The forward the feature exists for: real image ids reach the mask.

        The embedding is fed clamped ids (Phase 4 supplies real image
        embeddings); `inputs` carrying the true sentinel ids is what
        `_apply_image_visibility` and the MoE gate read. This is the case that
        crashed before the 4-D mask fix.
        """
        seqlen = 384
        ids = build_ids(seqlen, [(50, 200)], np.random.default_rng(557)).astype(
            np.int32
        )
        os.environ["EXO_DSV4_IMAGE_VISIBILITY"] = "1"
        try:
            importlib.reload(dsv4)
            model = dsv4.Model(_config())
            rng = np.random.default_rng(778)

            def fill(t):
                if isinstance(t, dict):
                    return {k: fill(v) for k, v in t.items()}
                if isinstance(t, list):
                    return [fill(v) for v in t]
                if isinstance(t, mx.array):
                    if t.dtype == mx.int32:
                        return mx.array(
                            rng.integers(0, 8, size=t.shape).astype(np.int32)
                        )
                    return mx.array(
                        (rng.standard_normal(t.shape) * 0.05).astype(np.float32)
                    )
                return t

            model.update(fill(model.parameters()))
            logits = model(mx.array(ids), cache=model.make_cache())
            mx.eval(logits)
            arr = np.asarray(logits)
            print(
                f"\n[GAP1 e2e REAL SPAN] 4-layer forward, seqlen={seqlen}, "
                f"{int((ids >= VOCAB).sum())} image tokens, "
                f"compress_ratios=[0,4,128,0]: logits {arr.shape}, "
                f"finite={bool(np.isfinite(arr).all())}, "
                f"|max|={np.abs(arr).max():.4f}"
            )
            self.assertEqual(arr.shape, (1, seqlen, VOCAB))
            self.assertTrue(bool(np.isfinite(arr).all()))
        finally:
            del os.environ["EXO_DSV4_IMAGE_VISIBILITY"]
            importlib.reload(dsv4)


if __name__ == "__main__":
    unittest.main()
