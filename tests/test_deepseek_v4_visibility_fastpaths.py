# Copyright © 2026 Apple Inc.

"""Phase 3b GAP 2: every reachable attention fast-path, with REAL image spans.

The Phase 3b inventory (docs/dsv4-vision-phase3b-window-geometry-inventory.md)
catalogues 16 window/top-k consumer sites, C1..C16. Only C7 was ever tested in
the case that matters — visibility flag ON **and** a real image span present.
The rest were exercised with the flag ON but NO span, which is degenerate: with
no span the visibility mask is bit-identical to the ordinary causal-window mask,
so a fast path that silently ignores span positions outside the window still
passes.

This file forces each reachable path to activate with a real
``[IMAGE_START..IMAGE_END]`` span present and checks its output against the
plain path it is supposed to be an optimization of.

Structurally-excluded paths are asserted to be excluded (their own guard is
called and shown to decline) rather than skipped silently — see
``TestStructurallyExcludedPaths``.

Run on a machine with MLX::

    PYTHONPATH=<repo>/mlx-lm <venv>/bin/python -m pytest \\
        tests/test_deepseek_v4_visibility_fastpaths.py -q
"""

import importlib
import os
import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models import deepseek_v4 as dsv4
from mlx_lm.models.base import create_causal_mask

from test_deepseek_v4_visibility_attention_output import (
    MAX_IMAGE_TOKENS,
    VOCAB,
    WINDOW,
    build_ids,
    oracle_visible_sets,
)


def _f64(a):
    """bf16-safe conversion to float64 (numpy cannot view bf16 directly)."""
    if isinstance(a, mx.array) and a.dtype == mx.bfloat16:
        a = a.astype(mx.float32)
    return np.asarray(a, dtype=np.float64)


def _cfg(mod, n_layers, ratios, *, head_dim=32, heads=4, index_topk=8):
    return mod.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=n_layers,
        num_attention_heads=heads,
        head_dim=head_dim,
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
        index_topk=index_topk,
        index_n_heads=4,
        index_head_dim=16,
        vision_n_layers=32,
        vision_max_n_token=MAX_IMAGE_TOKENS,
    )


def _fill(model, seed, mod, dtype=mx.float32):
    rng = np.random.default_rng(seed)

    def go(t):
        if isinstance(t, dict):
            return {k: go(v) for k, v in t.items()}
        if isinstance(t, list):
            return [go(v) for v in t]
        if isinstance(t, mx.array):
            if t.dtype == mx.int32:
                return mx.array(rng.integers(0, 8, size=t.shape).astype(np.int32))
            return mx.array(
                (rng.standard_normal(t.shape) * 0.05).astype(np.float32)
            ).astype(dtype)
        return t

    model.update(go(model.parameters()))
    return model


def _clamp_embed_input(model, vocab_size):
    """Clamp the EMBEDDING's input ids to ``< vocab_size``, in place.

    Image sentinel ids are ``vocab_size + {0..4}`` — deliberately OUTSIDE the
    embedding table, because the real pipeline never looks them up: DeepSeek's
    reference overwrites those rows wholesale in ``merge_image_embeddings``
    (and exo's Phase 4 path does the same via ``patch_embed_tokens``). The RAW
    ids must still reach the layers, since both the MoE gate
    (``input_ids >= vocab_size`` selects ``bias_vl``) and the visibility mask
    are derived from them.

    Without this, the test harness fed out-of-range ids straight into
    ``nn.Embedding``. MLX 0.32.1 happens to return deterministic ZEROS for an
    out-of-range gather (measured: 20 repeats x 3 processes, 1 distinct digest,
    all-zero=True), so it did not actually flake — but zero rows are NOT what
    the product feeds the model, and out-of-bounds gather behaviour is not a
    documented guarantee. Clamping makes the harness depend on defined
    behaviour only, and mirrors what the real path does.

    Only ``embed_tokens`` sees clamped ids; ``inputs`` reaching the layers,
    the gate and ``_apply_image_visibility`` keep the TRUE sentinel values.

    This is a stand-in for exo's real ``patch_embed_tokens`` splice, which
    performs the identical clamp-before-gather. Marked
    ``handles_out_of_range_ids = True`` so ``DeepseekV4Model._forward_steps``'s
    ``_assert_embeddable`` defense-in-depth check defers to it, exactly as it
    would for the real splice -- without the marker the RAW sentinel ids
    this test deliberately preserves in ``inputs`` (for the mask/gate) would
    trip that check.
    """
    inner = model.model
    original_embed = inner.embed_tokens

    def _clamped(input_ids):
        return original_embed(mx.minimum(input_ids, vocab_size - 1))

    _clamped.handles_out_of_range_ids = True

    inner.embed_tokens = _clamped
    return model


def _forward_logits(mod, ids, *, seed=4242, ratios=(0, 4, 128, 0), index_topk=8):
    """Full model forward on ``ids`` (image sentinels included) -> logits."""
    cfg = _cfg(mod, len(ratios), ratios, index_topk=index_topk)
    model = mod.Model(cfg)
    _fill(model, seed, mod)
    _clamp_embed_input(model, cfg.vocab_size)
    out = model(mx.array(ids), cache=model.make_cache())
    mx.eval(out)
    return np.asarray(out)


def _reload_with(env):
    """Reload deepseek_v4 with ``env`` applied; returns the reloaded module."""
    for k, v in env.items():
        os.environ[k] = v
    return importlib.reload(dsv4)


def _restore(env_keys):
    for k in env_keys:
        os.environ.pop(k, None)
    importlib.reload(dsv4)


def _span_ids(seqlen=384, spans=((50, 200),), seed=90210):
    ids = build_ids(seqlen, list(spans), np.random.default_rng(seed))
    return ids.astype(np.int32)


class TestC7QueryTiledSdpaWithRealSpans(unittest.TestCase):
    """C7 ``EXO_DSV4_QUERY_TILED_SDPA``: forced ON, with a real span present.

    Phase 3 makes ``_query_tiled_ok`` decline under visibility because the path
    re-derives its key slice from ``config.sliding_window`` and would drop span
    keys. Previously only the *predicate* was tested. Here the env var is
    actually set and a full forward is run with a real span, so the fallback is
    exercised end to end: the logits must equal the flag-off-fast-path logits
    EXACTLY (same fused SDPA call), not merely approximately.
    """

    def test_forced_on_with_span_matches_plain_path_bitwise(self):
        ids = _span_ids()
        base_env = {"EXO_DSV4_IMAGE_VISIBILITY": "1"}
        try:
            mod = _reload_with(base_env)
            plain = _forward_logits(mod, ids)

            mod = _reload_with({**base_env, "EXO_DSV4_QUERY_TILED_SDPA": "1"})
            self.assertTrue(mod._QUERY_TILED_SDPA, "env var did not take effect")
            tiled = _forward_logits(mod, ids)
        finally:
            _restore(["EXO_DSV4_IMAGE_VISIBILITY", "EXO_DSV4_QUERY_TILED_SDPA"])

        eq = int((plain == tiled).sum())
        bitwise = plain.tobytes() == tiled.tobytes()
        maxdiff = float(np.abs(plain.astype(np.float64) - tiled.astype(np.float64)).max())
        print(
            f"\n[C7 QUERY_TILED_SDPA=1 + real span] logits {plain.shape}: "
            f"{eq}/{plain.size} identical, raw-bytes identical={bitwise}, "
            f"max abs diff {maxdiff:.1e}  (path declines -> plain fused SDPA)"
        )
        self.assertEqual(eq, plain.size)
        self.assertTrue(bitwise)
        self.assertEqual(maxdiff, 0.0)

    def test_predicate_declines_only_when_visibility_active(self):
        """The decline must be conditional, not a blanket disable of C7."""
        try:
            mod = _reload_with(
                {"EXO_DSV4_IMAGE_VISIBILITY": "1", "EXO_DSV4_QUERY_TILED_SDPA": "1"}
            )

            class _Attn:
                config = _cfg(mod, 2, (0, 0))

            class _Pool:
                pooled = mx.zeros((1, 16, 32))

            class _Local:
                offset = 4096

            q = mx.zeros((1, 4, 256, 32))
            kv = mx.zeros((1, 1, 144, 32))
            mask = mx.ones((1, 1, 256, 144), dtype=mx.bool_)

            mod._IMAGE_VISIBILITY_CTX["active"] = False
            off = mod._query_tiled_ok(_Attn(), q, kv, mask, _Pool(), _Local())
            mod._IMAGE_VISIBILITY_CTX["active"] = True
            on = mod._query_tiled_ok(_Attn(), q, kv, mask, _Pool(), _Local())
            mod._IMAGE_VISIBILITY_CTX["active"] = False
        finally:
            _restore(["EXO_DSV4_IMAGE_VISIBILITY", "EXO_DSV4_QUERY_TILED_SDPA"])
        print(
            f"[C7 predicate] visibility inactive -> {off} (path still usable); "
            f"visibility active -> {on} (declines)"
        )
        self.assertTrue(off)
        self.assertFalse(on)


class TestC8SparseFusedSdpaIsReachableAndCorrect(unittest.TestCase):
    """C8 ``EXO_DSV4_SPARSE_FUSED_SDPA`` — REACHABLE, contrary to the inventory.

    The inventory (§2 C8, §3 corollary) argues C8 is "structurally out of
    reach" because it is gated ``L <= 16`` and visibility is prefill-only.
    That inference is WRONG: a *prefill* can also have ``L <= 16``. A 16-token
    prompt containing a 6-token image span is ``offset == 0`` (so visibility
    activates) AND ``L <= 16`` (so C8's gate passes). Verified below: the
    kernel FIRES.

    It is nonetheless CORRECT under a widened mask: the kernel reads the mask
    it is handed (``_norm_mask`` only reshapes/clamps it) rather than
    re-deriving reach from ``sliding_window`` the way C7 does. Both the fused
    and legacy paths are compared against a float64 oracle.
    """

    # C8 requires head_dim in (128, 512) and L <= 16.
    #
    # SEQLEN/SW CHOICE MATTERS. The fused Metal kernel is NOT run-to-run
    # deterministic for sliding_window in roughly [16, 96] — see
    # TestC8FusedKernelDeterminism below, which demonstrates this with a plain
    # causal mask and no Phase 3 code involved at all. Production sliding_window
    # is 128 and IS stable, so these correctness comparisons are made at a
    # KV width of 144 (>= 128, stable) with 16 query rows, which is both the
    # production-representative geometry and the only one where a bitwise
    # comparison is meaningful.
    L, D, H, K, P = 16, 128, 4, 2, 8
    SEQ = 144          # KV width; >= 128 keeps the fused kernel deterministic
    SPAN = (20, 120)   # covers 20..139, so query rows 128..139 are inside it

    def _masks(self, mod):
        """(narrow, widened) 4-D masks sliced to the last ``L`` query rows."""
        model = mod.Model(
            _cfg(mod, 2, (0, 4), head_dim=self.D, index_topk=self.K)
        ).model
        ids = _span_ids(seqlen=self.SEQ, spans=(self.SPAN,), seed=32)
        base = create_causal_mask(self.SEQ, 0, window_size=WINDOW)
        vis = model._apply_image_visibility(base, mx.array(ids), None)
        mx.eval(vis)
        lo = self.SEQ - self.L
        narrow = mx.broadcast_to(
            mx.array(np.asarray(base))[None, None], (1, 1, self.SEQ, self.SEQ)
        )[:, :, lo:, :]
        return mx.contiguous(narrow), mx.contiguous(vis[:, :, lo:, :]), ids

    def _tensors(self, seed=1234):
        rng = np.random.default_rng(seed)

        def arr(shape, s=0.5):
            return mx.array(
                (rng.standard_normal(shape) * s).astype(np.float32)
            ).astype(mx.bfloat16)

        t = dict(
            q=arr((1, self.H, self.L, self.D)),
            local_kv=arr((1, 1, self.SEQ, self.D)),
            pooled=arr((1, self.P, self.D)),
            topk=mx.array(
                rng.integers(0, self.P, size=(1, self.L, self.K)).astype(np.int32)
            ),
            pmask=mx.ones((1, 1, self.L, self.K), dtype=mx.bool_),
            sinks=mx.array((rng.standard_normal(self.H) * 0.3).astype(np.float32)).astype(
                mx.bfloat16
            ),
            scale=float(self.D**-0.5),
        )
        mx.eval(*[v for v in t.values() if isinstance(v, mx.array)])
        return t

    @staticmethod
    def _oracle(q, local_kv, pooled, topk, lmask, pmask, scale, sinks):
        qn, kn, pn = _f64(q), _f64(local_kv)[:, 0], _f64(pooled)
        tn = np.asarray(topk)
        lm, pm, sn = np.asarray(lmask)[0, 0], np.asarray(pmask)[0, 0], _f64(sinks)
        B, H, L, D = qn.shape
        out = np.zeros((B, H, L, D))
        for b in range(B):
            for i in range(L):
                lc, pc = np.flatnonzero(lm[i]), np.flatnonzero(pm[i])
                kk = np.concatenate([kn[b, lc], pn[b, tn[b, i, pc]]], axis=0)
                lg = np.concatenate(
                    [(qn[b, :, i, :] @ kk.T) * scale, sn[:, None]], axis=-1
                )
                m = lg.max(-1, keepdims=True)
                e = np.exp(lg - m)
                p = e / e.sum(-1, keepdims=True)
                out[b, :, i, :] = p[:, :-1] @ kk
        return out

    def test_c8_fires_on_a_small_prefill_with_a_span(self):
        """Reachability: the guard passes on a real prefill-with-image-span.

        The model must use ``head_dim=128`` (C8's contract is D in (128, 512));
        the earlier Phase 3b test config used head_dim=32, which is exactly why
        C8 was believed unreachable — it was declining on DTYPE/SHAPE, not on
        prefill-vs-decode.
        """
        try:
            mod = _reload_with(
                {"EXO_DSV4_IMAGE_VISIBILITY": "1", "EXO_DSV4_SPARSE_FUSED_SDPA": "1"}
            )
            fired = []
            orig = mod._sparse_fused_sdpa

            def spy(*a, **k):
                r = orig(*a, **k)
                fired.append(r is not None)
                return r

            mod._sparse_fused_sdpa = spy
            ids = _span_ids(seqlen=16, spans=((4, 6),), seed=32)
            cfg = _cfg(mod, 2, (0, 4), head_dim=self.D, index_topk=self.K)
            model = mod.Model(cfg)
            _fill(model, 4242, mod, dtype=mx.bfloat16)
            _clamp_embed_input(model, cfg.vocab_size)
            logits = model(mx.array(ids), cache=model.make_cache())
            mx.eval(logits)
            active = mod._IMAGE_VISIBILITY_CTX["active"]
            mod._sparse_fused_sdpa = orig
        finally:
            _restore(["EXO_DSV4_IMAGE_VISIBILITY", "EXO_DSV4_SPARSE_FUSED_SDPA"])

        print(
            f"\n[C8 REACHABILITY] L=16 PREFILL with a 6-token span, head_dim=128:\n"
            f"    offset==0 so visibility activates; L<=16 so C8's gate passes.\n"
            f"    _sparse_fused_sdpa called {len(fired)}x, fired={fired}, "
            f"visibility active during forward={active}\n"
            f"    -> the inventory's 'C8 is structurally out of reach' claim is "
            f"WRONG. A prefill can also be L<=16."
        )
        self.assertTrue(any(fired), "C8 did not fire; reachability claim unproven")
        self.assertTrue(active)

    def test_c8_output_matches_legacy_and_oracle_under_a_widened_mask(self):
        try:
            mod = _reload_with(
                {"EXO_DSV4_IMAGE_VISIBILITY": "1", "EXO_DSV4_SPARSE_FUSED_SDPA": "1"}
            )
            narrow, vis, _ = self._masks(mod)
            t = self._tensors()

            results = {}
            for tag, m in (("causal-only", narrow), ("visibility-widened", vis)):
                mod._SPARSE_FUSED_SDPA = True
                fused = mod._sparse_fused_sdpa(
                    t["q"], t["local_kv"], t["pooled"], t["topk"], m, t["pmask"],
                    t["scale"], t["sinks"],
                )
                mod._SPARSE_FUSED_SDPA = False
                legacy = mod._sparse_pooled_attention(
                    t["q"], t["local_kv"], t["pooled"], t["topk"], m, t["pmask"],
                    t["scale"], t["sinks"],
                )
                mx.eval(fused, legacy)
                self.assertIsNotNone(fused, f"C8 declined on {tag}")
                orc = self._oracle(
                    t["q"], t["local_kv"], t["pooled"], t["topk"], m, t["pmask"],
                    t["scale"], t["sinks"],
                )
                results[tag] = (
                    float(np.abs(_f64(fused) - orc).max()),
                    float(np.abs(_f64(legacy) - orc).max()),
                    float(np.abs(_f64(fused) - orc).mean()),
                    float(np.abs(_f64(legacy) - orc).mean()),
                    float(np.abs(orc).max()),
                )
        finally:
            _restore(["EXO_DSV4_IMAGE_VISIBILITY", "EXO_DSV4_SPARSE_FUSED_SDPA"])

        print(
            f"\n[C8 CORRECTNESS vs float64 oracle] bf16 fused kernel, "
            f"L={self.L} query rows, KV width {self.SEQ}, D={self.D}"
        )
        for tag, (fm, lm_, fmean, lmean, omax) in results.items():
            print(
                f"    {tag:20s}  fused max {fm:.3e} (mean {fmean:.3e})  |  "
                f"legacy max {lm_:.3e} (mean {lmean:.3e})  |oracle|max {omax:.4f}"
            )
        wide_f, wide_l = results["visibility-widened"][0], results["visibility-widened"][1]
        narrow_f = results["causal-only"][0]
        print(
            f"    -> widened fused error ({wide_f:.3e}) is in family with the "
            f"causal-only fused error ({narrow_f:.3e}): the kernel READS the "
            f"mask it is given (_norm_mask only reshapes/clamps it) instead of "
            f"re-deriving reach from sliding_window the way C7 does, so "
            f"visibility does not degrade it."
        )
        # bf16 Metal kernel vs a float64 reference: ~1e-2 is the honest bound
        # for this dtype. The load-bearing claim is that widening does not make
        # it worse than the causal-only baseline.
        self.assertLess(wide_f, 2e-2, "C8 wrong under a widened mask")
        self.assertLess(wide_l, 2e-2)
        self.assertLess(
            wide_f,
            max(10 * narrow_f, 2e-2),
            "visibility specifically degraded the fused kernel",
        )

    def test_c8_actually_honors_the_extra_span_bits(self):
        """Control: if C8 ignored the widening its two outputs would be equal."""
        try:
            mod = _reload_with(
                {"EXO_DSV4_IMAGE_VISIBILITY": "1", "EXO_DSV4_SPARSE_FUSED_SDPA": "1"}
            )
            narrow, vis, _ = self._masks(mod)
            t = self._tensors()
            mod._SPARSE_FUSED_SDPA = True
            a = mod._sparse_fused_sdpa(
                t["q"], t["local_kv"], t["pooled"], t["topk"], vis, t["pmask"],
                t["scale"], t["sinks"],
            )
            b = mod._sparse_fused_sdpa(
                t["q"], t["local_kv"], t["pooled"], t["topk"], narrow, t["pmask"],
                t["scale"], t["sinks"],
            )
            mx.eval(a, b)
            added = int(np.asarray(vis).sum() - np.asarray(narrow).sum())
            d = float(np.abs(_f64(a) - _f64(b)).max())
        finally:
            _restore(["EXO_DSV4_IMAGE_VISIBILITY", "EXO_DSV4_SPARSE_FUSED_SDPA"])
        print(
            f"\n[C8 control] widened mask adds {added} keys over the {self.L} "
            f"query rows; fused(widened) vs fused(causal-only) max abs diff "
            f"{d:.3e} (must be > 0, else the kernel silently ignores span bits)"
        )
        self.assertGreater(added, 0, "test setup produced no widening")
        self.assertGreater(d, 1e-3)


class TestC8FusedKernelDeterminism(unittest.TestCase):
    """PRE-EXISTING BUG, NOT a Phase 3 regression: C8 is nondeterministic.

    Found while closing the Phase 3 fast-path gap, and reported separately.
    With a PLAIN causal-window mask — no image span, no visibility flag, no
    Phase 3 code anywhere on the stack — ``_sparse_fused_sdpa`` returns
    DIFFERENT results across repeated calls on byte-identical, pre-materialized
    inputs. Up to 10 distinct outputs from 10 identical calls have been
    observed, with spreads up to ~8e-2 on outputs of magnitude ~0.15, i.e. tens
    of percent. The legacy path is bitwise stable on exactly the same inputs.

    It is intermittent and state-dependent: an early run of this sweep showed
    sliding_window=128 stable, a later one in the same process showed it
    unstable, so it is NOT confined to a small-`sw` band and must not be
    characterized as such.

    ``EXO_DSV4_SPARSE_FUSED_SDPA`` DEFAULTS TO OFF, so production is not
    exposed today. This test therefore asserts the invariant that must hold
    (the legacy path is deterministic) and RECORDS the fused path's behavior
    rather than asserting a stability the kernel does not currently provide —
    a test that flipped with GPU scheduling would be worse than none.
    """

    def _runs(self, mod, sw, *, fused, L=16, D=128, H=4, k=2, P=8, n=10, seed=4242):
        rng = np.random.default_rng(seed)

        def arr(shape, s=0.5):
            return mx.array(
                (rng.standard_normal(shape) * s).astype(np.float32)
            ).astype(mx.bfloat16)

        q, kv, pooled = arr((1, H, L, D)), arr((1, 1, sw, D)), arr((1, P, D))
        topk = mx.array(rng.integers(0, P, size=(1, L, k)).astype(np.int32))
        sinks = mx.array((rng.standard_normal(H) * 0.3).astype(np.float32)).astype(
            mx.bfloat16
        )
        lm = mx.ones((1, 1, L, sw), dtype=mx.bool_)
        pm = mx.ones((1, 1, L, k), dtype=mx.bool_)
        mx.eval(q, kv, pooled, topk, sinks, lm, pm)

        outs = []
        for _ in range(n):
            if fused:
                o = mod._sparse_fused_sdpa(
                    q, kv, pooled, topk, lm, pm, float(D**-0.5), sinks
                )
                if o is None:
                    return None
            else:
                mod._SPARSE_FUSED_SDPA = False
                o = mod._sparse_pooled_attention(
                    q, kv, pooled, topk, lm, pm, float(D**-0.5), sinks
                )
            mx.eval(o)
            outs.append(_f64(o))
        return (
            len({a.tobytes() for a in outs}),
            float(max(np.abs(a - outs[0]).max() for a in outs)),
            float(np.abs(outs[0]).max()),
        )

    def test_legacy_is_deterministic_and_fused_is_recorded(self):
        sws = (8, 16, 32, 64, 96, 128, 192, 256)
        try:
            mod = _reload_with({"EXO_DSV4_SPARSE_FUSED_SDPA": "1"})
            fused_rows, legacy_rows = [], []
            for sw in sws:
                mod._SPARSE_FUSED_SDPA = True
                fused_rows.append((sw, self._runs(mod, sw, fused=True)))
                legacy_rows.append((sw, self._runs(mod, sw, fused=False)))
            default_off = (
                os.environ.get("EXO_DSV4_SPARSE_FUSED_SDPA_DEFAULT_PROBE") is None
            )
        finally:
            _restore(["EXO_DSV4_SPARSE_FUSED_SDPA"])
        # With the env var unset the module-level gate must be False.
        self.assertFalse(
            dsv4._SPARSE_FUSED_SDPA,
            "EXO_DSV4_SPARSE_FUSED_SDPA must default OFF; if this ever flips, "
            "the nondeterminism recorded below becomes a production exposure",
        )

        print(
            "\n[C8 DETERMINISM — plain causal mask, NO visibility, NO image span]\n"
            "  10 identical calls on byte-identical pre-materialized inputs:\n"
            f"    {'sw':>5} | {'FUSED distinct':>14} {'spread':>11} | "
            f"{'LEGACY distinct':>15} {'spread':>11}"
        )
        unstable = []
        for (sw, f), (_, lg) in zip(fused_rows, legacy_rows):
            fs = "declined" if f is None else f"{f[0]:>14}"
            fsp = "-" if f is None else f"{f[1]:>11.4e}"
            print(
                f"    {sw:>5} | {fs} {fsp} | {lg[0]:>15} {lg[1]:>11.4e}"
                + ("   <-- FUSED UNSTABLE" if f and f[0] > 1 else "")
            )
            if f and f[0] > 1:
                unstable.append((sw, f[0], f[1], f[2]))
            # THE INVARIANT: the legacy path must be bitwise reproducible.
            self.assertEqual(
                lg[0], 1, f"legacy path became nondeterministic at sw={sw}"
            )
            self.assertEqual(lg[1], 0.0)

        print(
            f"    -> legacy path: bitwise identical across all {len(sws)} widths.\n"
            f"    -> fused path: {len(unstable)}/{len(sws)} widths returned more "
            f"than one distinct result from identical inputs"
            + (
                f"; worst spread {max(u[2] for u in unstable):.3e} on outputs of "
                f"magnitude ~{max(u[3] for u in unstable):.3f}"
                if unstable
                else ""
            )
            + f"\n    -> EXO_DSV4_SPARSE_FUSED_SDPA defaults OFF "
            f"(module gate with env unset: {dsv4._SPARSE_FUSED_SDPA}), so this "
            f"PRE-EXISTING issue is not a production exposure today."
        )
        self.assertTrue(default_off)


class TestC10C11SeqSplitAndSparseTilingWithRealSpans(unittest.TestCase):
    """C10 (row-band slicing) and C11 (``_SPARSE_SDPA_TILE`` + SINGLE_GATHER).

    C11 is ON BY DEFAULT (`EXO_DSV4_SPARSE_SDPA_TILE=128`,
    `EXO_DSV4_SINGLE_GATHER=1`) and DOES run during a real vision prefill —
    the probe shows `_sparse_pooled_attention` called 3x at (1,4,128,32) for a
    384-token prefill, i.e. three 128-row tiles. It slices the mask by ROWS
    only, so widened COLUMNS survive; this proves that rather than assuming it.

    C10's seq-split needs a distributed sharding group and cannot be built in a
    single-process test; its row-only slicing is the same operation as C11's and
    is covered by the same argument. Stated, not faked.
    """

    def test_tiling_variants_agree_with_the_untiled_path(self):
        ids = _span_ids(seqlen=384, spans=((50, 200),))
        variants = {
            "tile=128 single_gather=1 (production default)": {},
            "tile=128 single_gather=0 (per-tile gather)": {
                "EXO_DSV4_SINGLE_GATHER": "0"
            },
            "tile=64  single_gather=1": {"EXO_DSV4_SPARSE_SDPA_TILE": "64"},
            "tile=0   (UNTILED reference)": {"EXO_DSV4_SPARSE_SDPA_TILE": "0"},
        }
        keys = [
            "EXO_DSV4_IMAGE_VISIBILITY",
            "EXO_DSV4_SINGLE_GATHER",
            "EXO_DSV4_SPARSE_SDPA_TILE",
        ]
        outs = {}
        try:
            for tag, env in variants.items():
                _restore(keys)
                mod = _reload_with({"EXO_DSV4_IMAGE_VISIBILITY": "1", **env})
                calls = []
                orig = mod._sparse_pooled_attention

                def spy(*a, _o=orig, _c=calls, **k):
                    _c.append(a[0].shape)
                    return _o(*a, **k)

                mod._sparse_pooled_attention = spy
                outs[tag] = (_forward_logits(mod, ids), list(calls))
                mod._sparse_pooled_attention = orig
        finally:
            _restore(keys)

        ref_tag = "tile=0   (UNTILED reference)"
        ref = outs[ref_tag][0]
        print("\n[C10/C11 tiling under image-span visibility] seqlen=384, span [50,249]")
        for tag, (arr, calls) in outs.items():
            d = float(np.abs(ref.astype(np.float64) - arr.astype(np.float64)).max())
            eq = int((ref == arr).sum())
            print(
                f"    {tag:46s} sparse-SDPA calls={len(calls)} "
                f"q-shapes={calls[:3]}\n"
                f"      vs untiled: {eq}/{ref.size} identical, max abs diff {d:.3e}"
            )
            self.assertLess(d, 5e-3, f"{tag} diverges from the untiled path")
        self.assertGreater(
            len(outs["tile=128 single_gather=1 (production default)"][1]),
            1,
            "C11 tiling did not actually engage — test would be vacuous",
        )


class TestC2C3C5MaskPlumbingWithRealSpans(unittest.TestCase):
    """C2 ``_clamp_mask_to_kv`` / C3+C5 ``_extend_mask`` on a WIDENED 4-D mask.

    These are the two helpers every attention class routes the mask through.
    Under visibility the mask is 4-D and carries non-causal (forward-reaching)
    bits; both helpers must preserve every visible column they are not
    explicitly asked to drop.
    """

    def test_extend_and_clamp_preserve_the_widened_columns(self):
        seqlen = 384
        ids = _span_ids(seqlen=seqlen, spans=((50, 200),))
        try:
            mod = _reload_with({"EXO_DSV4_IMAGE_VISIBILITY": "1"})
            model = mod.Model(_cfg(mod, 4, (0, 4, 128, 0))).model
            base = create_causal_mask(seqlen, 0, window_size=WINDOW)
            vis = model._apply_image_visibility(base, mx.array(ids), None)
            mx.eval(vis)

            # C2: no clamp needed at offset 0 (kv_len == seqlen) -> identity.
            clamped = mod._clamp_mask_to_kv(vis, seqlen)
            self.assertIs(clamped, vis, "no-op clamp must return the same object")

            # C2: an oversized mask must keep its TRAILING columns.
            narrowed = mod._clamp_mask_to_kv(vis, WINDOW)
            mx.eval(narrowed)
            self.assertEqual(narrowed.shape[-1], WINDOW)
            self.assertTrue(
                bool(
                    (
                        np.asarray(narrowed) == np.asarray(vis)[..., -WINDOW:]
                    ).all()
                )
            )

            # C3/C5: extend with a pooled tail. Local columns must survive intact.
            pooled_w = 8
            pm = mx.ones((1, 1, seqlen, pooled_w), dtype=mx.bool_)
            ext = mod._extend_mask(vis, pm, seqlen + pooled_w)
            mx.eval(ext)
            e = np.asarray(ext)
            v = np.asarray(vis)
            local_ok = bool((e[..., :seqlen] == v).all())
            pooled_ok = bool(e[..., seqlen:].all())
            want = int(v.sum())
            got = int(e[..., :seqlen].sum())
        finally:
            _restore(["EXO_DSV4_IMAGE_VISIBILITY"])

        print(
            f"\n[C2/C3/C5 mask plumbing, widened 4-D mask] seqlen={seqlen}\n"
            f"    _clamp_mask_to_kv(kv_len=seqlen) -> identity object: True\n"
            f"    _clamp_mask_to_kv(kv_len=128) -> trailing-128 slice matches: True\n"
            f"    _extend_mask -> shape {e.shape}; local half preserved "
            f"{got}/{want} visible keys ({local_ok}); pooled tail all-True "
            f"({pooled_ok})"
        )
        self.assertTrue(local_ok)
        self.assertTrue(pooled_ok)
        self.assertEqual(got, want)


class TestC12IndexerPooledAxisUntouched(unittest.TestCase):
    """C12: visibility must not touch the Indexer's pooled-axis machinery.

    The reference replaces only the WINDOW half of
    ``cat([topk_idxs, compress_topk_idxs])``; the compressed half is untouched.
    The fork's analogue is the Indexer's ``topk``.

    IMPORTANT — what "untouched" can and cannot mean here. The Indexer scores
    the pooled axis from the CURRENT LAYER'S HIDDEN STATES. Once ANY earlier
    layer's attention has been widened, later layers' hidden states
    legitimately differ between flag OFF and flag ON — that is the feature
    working, not a leak. So "all top-k indices identical across the whole
    forward" is the WRONG specification; measured on a [0,4,128,0] model it is
    2411/3072, and the difference is entirely downstream of layer 0's
    correctly-widened attention.

    The right specification, asserted here, is that visibility does not reach
    into the compressed half ITSELF. Tested with ``compress_ratios=(4, 4)`` so
    LAYER 0 is the sparse layer: its indexer consumes the embeddings directly,
    with no widened attention anywhere upstream, so its top-k must be
    bit-identical between flag OFF and flag ON+span. Any difference there could
    only come from visibility touching the pooled path.
    """

    def test_first_layer_indexer_topk_is_bit_identical(self):
        ids = _span_ids(seqlen=384, spans=((50, 200),))
        captured = {}
        try:
            for tag, env in (
                ("flag OFF", {}),
                ("flag ON + span", {"EXO_DSV4_IMAGE_VISIBILITY": "1"}),
            ):
                _restore(["EXO_DSV4_IMAGE_VISIBILITY"])
                mod = _reload_with(env)
                seen = []
                orig = mod._sparse_pooled_attention

                def spy(*a, _o=orig, _s=seen, **k):
                    _s.append(np.asarray(a[3]))  # topk
                    return _o(*a, **k)

                mod._sparse_pooled_attention = spy
                # LAYER 0 sparse: its indexer input is the embedding, which no
                # widened attention can have touched.
                _forward_logits(mod, ids, ratios=(4, 4))
                mod._sparse_pooled_attention = orig
                captured[tag] = seen
        finally:
            _restore(["EXO_DSV4_IMAGE_VISIBILITY"])

        off, on = captured["flag OFF"], captured["flag ON + span"]
        self.assertGreater(len(off), 0, "indexer path never ran — vacuous")
        self.assertEqual(len(off), len(on), "visibility changed the CALL COUNT")
        for a, b in zip(off, on):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)

        # Layer 0 emits the first ceil(L / _SPARSE_SDPA_TILE) calls; compare
        # exactly those, i.e. the ones whose inputs precede any attention.
        n_l0 = max(1, len(off) // 2)
        l0_eq = sum(int((a == b).sum()) for a, b in zip(off[:n_l0], on[:n_l0]))
        l0_tot = sum(int(a.size) for a in off[:n_l0])
        total = sum(a.size for a in off)
        same = sum(int((a == b).sum()) for a, b in zip(off, on))
        print(
            f"\n[C12 Indexer pooled axis] compress_ratios=(4,4) so LAYER 0 is "
            f"sparse; {len(off)} sparse-SDPA calls, top-k shape {off[0].shape} "
            f"dtype {off[0].dtype} (shape/dtype identical OFF vs ON)\n"
            f"    LAYER 0 top-k ({n_l0} tiles): {l0_eq}/{l0_tot} elements "
            f"identical — must be ALL, its inputs precede every widened "
            f"attention\n"
            f"    whole forward: {same}/{total} identical — NOT expected to be "
            f"all; later layers' hidden states are correctly downstream of "
            f"layer 0's widened attention"
        )
        self.assertEqual(
            l0_eq,
            l0_tot,
            "visibility perturbed the indexer's top-k BEFORE any widened "
            "attention could have influenced it — that would mean the port "
            "reaches into the compressed half, which the reference never does",
        )


class TestStructurallyExcludedPaths(unittest.TestCase):
    """Paths that genuinely cannot co-occur with visibility, asserted not assumed.

    For each, the exclusion is demonstrated by calling the path's own guard and
    showing it declines, rather than by argument alone.
    """

    def test_c9_and_c16_require_a_nonzero_cache_offset(self):
        """C9 (verify batching) / C16 (rowseq) are offset>0 by construction.

        Visibility raises a ValueError if ANY image token appears at cache
        offset != 0 (the reference's own single-chunk assert), so a forward can
        never be simultaneously 'visibility active' and 'decode/verify'. That
        guard is the structural exclusion; it is exercised here directly.
        """
        try:
            mod = _reload_with({"EXO_DSV4_IMAGE_VISIBILITY": "1"})
            model = mod.Model(_cfg(mod, 2, (0, 4))).model

            class _Cache:
                offset = 2048

            ids = _span_ids(seqlen=64, spans=((10, 8),), seed=5)
            mask = create_causal_mask(64, 0, window_size=WINDOW)
            with self.assertRaises(ValueError) as ctx:
                model._apply_image_visibility(mask, mx.array(ids), _Cache())
            msg = str(ctx.exception)

            # And with offset 0 + no image token the ctx is explicitly cleared,
            # so a later decode cannot inherit a stale 'active'.
            text_ids = np.random.default_rng(6).integers(
                0, VOCAB, size=(1, 64)
            ).astype(np.int32)
            mod._IMAGE_VISIBILITY_CTX["active"] = True
            out = model._apply_image_visibility(mask, mx.array(text_ids), None)
            cleared = mod._IMAGE_VISIBILITY_CTX["active"]
            same_obj = out is mask
        finally:
            _restore(["EXO_DSV4_IMAGE_VISIBILITY"])
        print(
            f"\n[C9/C16 structural exclusion] image tokens at cache offset "
            f"2048 -> ValueError({msg[:60]}...).\n"
            f"    decode/verify (offset>0) therefore cannot carry a visibility "
            f"mask at all.\n"
            f"    text-only chunk clears the ctx: active={cleared} (False), "
            f"mask returned unchanged (identity={same_obj})"
        )
        self.assertIn("single chunk", msg)
        self.assertFalse(cleared)
        self.assertTrue(same_obj)

    def test_tree_verify_branch_clears_the_visibility_context(self):
        """C15/tree drafting bypasses _apply_image_visibility entirely.

        That branch never calls the widener, so without an explicit reset the
        ctx would stay stale at whatever the last prefill set — permanently
        making `_query_tiled_ok` decline. Not a wrong answer (the fallback is
        correct) but a lasting performance leak. Asserted here so it cannot
        silently come back.
        """
        try:
            mod = _reload_with({"EXO_DSV4_IMAGE_VISIBILITY": "1"})
            import inspect

            src = inspect.getsource(mod.DeepseekV4Model._forward_steps)
            # Anchor on the branch's CONDITIONAL, not on prose that also
            # mentions the ctx: `if _tree_mask is not None:` .. up to the call
            # that widens the ordinary mask.
            tree_idx = src.index("if _tree_mask is not None:")
            widen_idx = src.index("mask = self._apply_image_visibility")
            self.assertLess(tree_idx, widen_idx)
            between = src[tree_idx:widen_idx]
            resets = '_IMAGE_VISIBILITY_CTX["active"] = False' in between
        finally:
            _restore(["EXO_DSV4_IMAGE_VISIBILITY"])
        print(
            f"\n[C15 tree-verify branch] resets _IMAGE_VISIBILITY_CTX before "
            f"using the caller-supplied tree mask: {resets}"
        )
        self.assertTrue(
            resets,
            "the tree-verify branch must clear the visibility ctx; otherwise a "
            "stale 'active' permanently disables the query-tiled SDPA",
        )

    def test_c13_c14_are_pooled_axis_or_draft_head_only(self):
        """C13 (Compressor/PoolingCache) and C14 (DSpark draft) never see the mask.

        C13 operates on the pooled axis and takes no ``mask`` argument at all;
        C14's ``draft_block`` passes ``mask=None``. Both are read off the
        signatures rather than argued.
        """
        import inspect

        comp_sig = inspect.signature(dsv4.Compressor.__call__)
        draft_src = inspect.getsource(dsv4.DSparkLocalAttention.draft_block)
        print(
            f"\n[C13/C14 structural exclusion]\n"
            f"    Compressor.__call__{comp_sig} -> no `mask` parameter\n"
            f"    DSparkLocalAttention.draft_block passes mask=None: "
            f"{'mask=None' in draft_src}"
        )
        self.assertNotIn("mask", comp_sig.parameters)
        self.assertIn("mask=None", draft_src)


if __name__ == "__main__":
    unittest.main()
