# Copyright © 2026 Apple Inc.

"""Phase 3a GAPS 4 + 5: hermetic text-only no-regression guard for `bias_vl`.

GAP 4 — is the text-only guard VACUOUS?
    The concern raised in review: if the synthetic gate's ``bias_vl`` were
    ``mx.zeros(...)``, then "text tokens ignore bias_vl" proves nothing, since a
    zero bias changes nothing by construction.

    Answer, asserted below rather than argued: it was never zero.
    ``tests/test_deepseek_v4_gate.py`` seeds it with
    ``rng.standard_normal(N_EXPERTS) * 0.5`` (nonzero random), and the
    text-only-CHECKPOINT test is stronger still — with ``vision_n_layers=0``
    the parameter is not allocated AT ALL, so there is no zero to hide behind.
    ``TestBiasVlIsGenuinelyNonzero`` measures the seeded values, and every test
    here re-runs the guard with a DELIBERATELY HUGE ``bias_vl`` (±50, i.e. two
    orders of magnitude above the ~1.0 score range) that would certainly
    reorder the top-k if it leaked onto a text token.

GAP 5 — permanent hermetic regression test for the core safety invariant.
    A reviewer's throwaway script once proved: ``*_vl`` routing with an
    all-False image mask == the production ``_gate_route`` / ``_hash_gate_route``
    output, exactly. That script was deleted. This restores it as a committed
    test.

    It does NOT extract a frozen baseline from git history (fragile; needs git
    surgery at test time). It calls the STILL-PRESENT production functions
    directly and asserts exact equality against their ``*_vl`` siblings on
    all-text input, across batch sizes, sequence lengths, and hash/non-hash
    layers — plus the end-to-end model-level flag OFF-vs-ON bitwise check.

Run on a machine with MLX::

    PYTHONPATH=<repo>/mlx-lm <venv>/bin/python -m pytest \\
        tests/test_deepseek_v4_gate_vl_invariant.py -q
"""

import importlib
import os
import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models import deepseek_v4 as dsv4

VOCAB = 512
N_EXPERTS = 32
TOP_K = 6
HIDDEN = 64
SCORING = "sqrtsoftplus"
ROUTE_SCALE = 1.5
NORM_TOPK = True

#: Deliberately enormous relative to the score scale (sqrt(softplus(.)) of a
#: small dot product, i.e. O(1)). If ANY of this leaked onto a text token's
#: selection, the top-k would reorder and the exact-equality assertions below
#: would fail loudly. A zero bias could not make that claim.
_HUGE = 50.0


def _config(vision_n_layers):
    return dsv4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        n_routed_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=8,
        num_hash_layers=3,
        scoring_func=SCORING,
        routed_scaling_factor=ROUTE_SCALE,
        norm_topk_prob=NORM_TOPK,
        vision_n_layers=vision_n_layers,
    )


def _bias_vl(rng, *, huge=True):
    """A bias_vl that is unmistakably NOT a no-op."""
    v = rng.standard_normal(N_EXPERTS).astype(np.float32)
    if huge:
        v = (v * _HUGE).astype(np.float32)
    return v


class TestBiasVlIsGenuinelyNonzero(unittest.TestCase):
    """GAP 4: measure the bias_vl the existing guard actually used."""

    def test_seeded_bias_vl_is_nonzero(self):
        """The gate-test helper seeds a real random bias, not zeros."""
        from test_deepseek_v4_gate import _seed_gate  # same tests/ dir

        rng = np.random.default_rng(11)
        gate = dsv4.MoEGate(_config(vision_n_layers=32), 0)
        _, _, bias, bias_vl = _seed_gate(gate, rng, hash_layer=True, vl=True)
        self.assertIsNotNone(bias_vl)
        nz = int(np.count_nonzero(bias_vl))
        print(
            f"\n[GAP 4 — is bias_vl vacuous?] tests/test_deepseek_v4_gate.py's "
            f"_seed_gate produced bias_vl with:\n"
            f"    {nz}/{bias_vl.size} nonzero entries, min {bias_vl.min():+.4f}, "
            f"max {bias_vl.max():+.4f}, mean |x| {np.abs(bias_vl).mean():.4f}, "
            f"L2 {np.linalg.norm(bias_vl):.4f}\n"
            f"    -> NOT mx.zeros; the parity assertions were never vacuous."
        )
        self.assertEqual(nz, bias_vl.size)
        self.assertGreater(float(np.abs(bias_vl).mean()), 0.05)

    def test_text_only_checkpoint_has_no_bias_vl_at_all(self):
        """Stronger than nonzero: with vision_n_layers=0 it is not allocated.

        The text-only bitwise guard therefore cannot be vacuous "because the
        bias happened to be zero" — there is no bias_vl parameter to be zero.
        """
        text_gate = dsv4.MoEGate(_config(vision_n_layers=0), 0)
        vis_gate = dsv4.MoEGate(_config(vision_n_layers=32), 0)
        print(
            f"\n[GAP 4] text-only checkpoint (vision_n_layers=0): "
            f"hasattr(gate,'e_score_correction_bias_vl') = "
            f"{hasattr(text_gate, 'e_score_correction_bias_vl')}\n"
            f"    vision checkpoint (vision_n_layers=32): "
            f"{hasattr(vis_gate, 'e_score_correction_bias_vl')} "
            f"shape {vis_gate.e_score_correction_bias_vl.shape}"
        )
        self.assertFalse(hasattr(text_gate, "e_score_correction_bias_vl"))
        self.assertTrue(hasattr(vis_gate, "e_score_correction_bias_vl"))


class TestVlRoutingEqualsProductionOnAllTextInput(unittest.TestCase):
    """GAP 5: the core safety invariant, hermetically.

    ``*_vl`` with an ALL-FALSE image mask (every id < vocab_size) must equal the
    STILL-PRESENT production ``_gate_route`` / ``_hash_gate_route`` EXACTLY —
    same indices, raw-byte-identical weights — even with a huge ``bias_vl``.

    Compared against the live production functions, not a git-extracted frozen
    copy: nothing to regenerate, nothing to drift.
    """

    SHAPES = [(1, 1), (1, 16), (1, 128), (1, 512), (2, 64), (4, 33)]

    def _one(self, batch, length, seed, *, hash_path, huge=True):
        rng = np.random.default_rng(seed)
        ids = mx.array(rng.integers(0, VOCAB, size=(batch, length)).astype(np.int32))
        x = mx.array(
            (rng.standard_normal((batch, length, HIDDEN)) * 0.5).astype(np.float32)
        )
        w = mx.array(
            (rng.standard_normal((N_EXPERTS, HIDDEN)) * 0.1).astype(np.float32)
        )
        bvl = mx.array(_bias_vl(rng, huge=huge))

        if hash_path:
            tid2eid = mx.array(
                rng.integers(0, N_EXPERTS, size=(VOCAB, TOP_K)).astype(np.int32)
            )
            pi, pw = dsv4._hash_gate_route(
                ids, x, w, tid2eid, ROUTE_SCALE, NORM_TOPK, SCORING
            )
            vi, vw = dsv4._hash_gate_route_vl(
                ids, x, w, tid2eid, bvl, VOCAB, TOP_K, ROUTE_SCALE, NORM_TOPK, SCORING
            )
        else:
            bias = mx.array(
                (rng.standard_normal(N_EXPERTS) * 0.5).astype(np.float32)
            )
            pi, pw = dsv4._gate_route(
                x, w, bias, TOP_K, ROUTE_SCALE, NORM_TOPK, SCORING
            )
            vi, vw = dsv4._gate_route_vl(
                ids, x, w, bias, bvl, VOCAB, TOP_K, ROUTE_SCALE, NORM_TOPK, SCORING
            )
        mx.eval(pi, pw, vi, vw)
        pi, pw, vi, vw = map(np.asarray, (pi, pw, vi, vw))
        return pi, pw, vi, vw, np.asarray(bvl)

    def _check(self, hash_path):
        label = "_hash_gate_route" if hash_path else "_gate_route"
        print(
            f"\n[GAP 5 — {label} vs {label}_vl, ALL-TEXT input, "
            f"bias_vl scaled to ±{_HUGE:g}]"
        )
        for i, (b, l) in enumerate(self.SHAPES):
            pi, pw, vi, vw, bvl = self._one(b, l, 700 + i, hash_path=hash_path)
            idx_eq = int((pi == vi).sum())
            idx_tot = int(pi.size)
            w_bytes = pw.tobytes() == vw.tobytes()
            w_max = float(
                np.abs(pw.astype(np.float64) - vw.astype(np.float64)).max()
            )
            print(
                f"    B={b:<2} L={l:<4} shape={tuple(pi.shape)}: "
                f"indices {idx_eq}/{idx_tot} identical, weights raw-bytes "
                f"identical={w_bytes}, max abs diff {w_max:.1e}  "
                f"(|bias_vl|max {np.abs(bvl).max():.2f})"
            )
            self.assertEqual(idx_eq, idx_tot, f"B={b} L={l}: indices diverged")
            self.assertEqual(pi.dtype, vi.dtype)
            self.assertTrue(w_bytes, f"B={b} L={l}: weights not bitwise equal")
            self.assertEqual(w_max, 0.0)

    def test_hash_path_invariant(self):
        self._check(hash_path=True)

    def test_non_hash_path_invariant(self):
        self._check(hash_path=False)

    def test_the_huge_bias_is_not_secretly_inert(self):
        """Control: the same huge bias_vl MUST change IMAGE-token routing.

        Without this, "text is unaffected" could be explained by bias_vl being
        ignored everywhere — which would mean the feature does nothing.
        """
        rng = np.random.default_rng(4242)
        b, l = 1, 32
        ids_img = mx.array(np.full((b, l), VOCAB + 2, dtype=np.int32))
        x = mx.array((rng.standard_normal((b, l, HIDDEN)) * 0.5).astype(np.float32))
        w = mx.array((rng.standard_normal((N_EXPERTS, HIDDEN)) * 0.1).astype(np.float32))
        tid2eid = mx.array(
            rng.integers(0, N_EXPERTS, size=(VOCAB, TOP_K)).astype(np.int32)
        )
        bvl = mx.array(_bias_vl(rng))

        pi, _ = dsv4._hash_gate_route(
            ids_img, x, w, tid2eid, ROUTE_SCALE, NORM_TOPK, SCORING
        )
        vi, _ = dsv4._hash_gate_route_vl(
            ids_img, x, w, tid2eid, bvl, VOCAB, TOP_K, ROUTE_SCALE, NORM_TOPK, SCORING
        )
        mx.eval(pi, vi)
        pi, vi = np.asarray(pi), np.asarray(vi)
        differing = int((np.sort(pi, -1) != np.sort(vi, -1)).any(-1).sum())
        print(
            f"\n[GAP 5 control] SAME huge bias_vl on ALL-IMAGE input "
            f"(ids = vocab_size+IMAGE): {differing}/{pi.shape[0] * pi.shape[1]} "
            f"rows route to a DIFFERENT expert set than the plain hash route "
            f"(must be > 0, else bias_vl is inert everywhere and the text-only "
            f"guard above is trivially satisfiable)"
        )
        self.assertGreater(differing, 0)

    def test_production_functions_are_distinct_objects(self):
        """Structural half: the vl variants are siblings, not replacements."""
        self.assertIsNot(dsv4._gate_route, dsv4._gate_route_vl)
        self.assertIsNot(dsv4._hash_gate_route, dsv4._hash_gate_route_vl)
        print(
            "\n[GAP 5 structural] _gate_route / _hash_gate_route are still "
            "distinct objects from their _vl siblings, so a text-only "
            "checkpoint calls the SAME compiled function it did pre-Phase-3."
        )


class TestMoEGateDispatchIsConfigDriven(unittest.TestCase):
    """A text-only checkpoint must reach the production route, not the vl one.

    Dispatch is on CONFIG (does this checkpoint have a bias_vl?), never on DATA
    (does this batch contain image tokens?) — matching the reference's
    ``self.bias_vl is not None``. Verified by observing which compiled function
    the gate actually calls.
    """

    def _which_route(self, vision_n_layers, layer_idx):
        called = []
        names = ("_gate_route", "_hash_gate_route", "_gate_route_vl", "_hash_gate_route_vl")
        originals = {n: getattr(dsv4, n) for n in names}
        try:
            for n in names:
                def spy(*a, _n=n, _o=originals[n], **k):
                    called.append(_n)
                    return _o(*a, **k)

                setattr(dsv4, n, spy)
                
            rng = np.random.default_rng(31)
            gate = dsv4.MoEGate(_config(vision_n_layers), layer_idx)
            gate.weight = mx.array(
                (rng.standard_normal((N_EXPERTS, HIDDEN)) * 0.1).astype(np.float32)
            )
            if layer_idx < 3:
                gate.tid2eid = mx.array(
                    rng.integers(0, N_EXPERTS, size=(VOCAB, TOP_K)).astype(np.int32)
                )
            ids = mx.array(rng.integers(0, VOCAB, size=(1, 8)).astype(np.int32))
            x = mx.array((rng.standard_normal((1, 8, HIDDEN)) * 0.5).astype(np.float32))
            out = gate(x, ids)
            mx.eval(*out)
        finally:
            for n, o in originals.items():
                setattr(dsv4, n, o)
        return called

    def test_dispatch_matrix(self):
        cases = [
            (0, 0, "_hash_gate_route"),
            (0, 3, "_gate_route"),
            (32, 0, "_hash_gate_route_vl"),
            (32, 3, "_gate_route_vl"),
        ]
        print("\n[gate dispatch] vision_n_layers x layer_idx -> compiled route")
        for vnl, li, want in cases:
            got = self._which_route(vnl, li)
            print(
                f"    vision_n_layers={vnl:<3} layer_idx={li} "
                f"({'hash' if li < 3 else 'non-hash'}) -> {got}  (want {want})"
            )
            self.assertEqual(got, [want])


class TestEndToEndFlagOffVsOnBitwise(unittest.TestCase):
    """GAP 5, end-to-end half: a real multi-layer forward, flag OFF vs ON.

    Text-only input through a REAL 4-layer DeepseekV4Model spanning every
    attention class (compress_ratios [0,4,128,0]) must be BITWISE identical
    with ``EXO_DSV4_IMAGE_VISIBILITY`` off and on. Complements the gate-level
    invariant above: that one isolates the routing functions, this one covers
    the whole stack including mask construction and every attention path.
    """

    @staticmethod
    def _config(n_layers=4):
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
            sliding_window=128,
            compress_ratios=[0, 4, 128, 0][:n_layers],
            index_topk=8,
            index_n_heads=4,
            index_head_dim=16,
            vision_n_layers=32,
            vision_max_n_token=384,
        )

    def _logits(self, mod, seed, ids):
        model = mod.Model(self._config())
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
        out = model(mx.array(ids), cache=model.make_cache())
        mx.eval(out)
        return np.asarray(out)

    def test_text_only_forward_is_bitwise_identical(self):
        for seqlen in (128, 384):
            with self.subTest(seqlen=seqlen):
                ids = (
                    np.random.default_rng(1000 + seqlen)
                    .integers(0, VOCAB, size=(1, seqlen))
                    .astype(np.int32)
                )
                self.assertFalse(dsv4._IMAGE_VISIBILITY)
                off = self._logits(dsv4, 777, ids)
                try:
                    os.environ["EXO_DSV4_IMAGE_VISIBILITY"] = "1"
                    mod = importlib.reload(dsv4)
                    self.assertTrue(mod._IMAGE_VISIBILITY)
                    on = self._logits(mod, 777, ids)
                finally:
                    os.environ.pop("EXO_DSV4_IMAGE_VISIBILITY", None)
                    importlib.reload(dsv4)

                eq = int((off == on).sum())
                bitwise = off.tobytes() == on.tobytes()
                mx_ = float(
                    np.abs(off.astype(np.float64) - on.astype(np.float64)).max()
                )
                print(
                    f"\n[GAP 5 END-TO-END] 4-layer forward, compress_ratios "
                    f"[0,4,128,0], TEXT-ONLY, seqlen={seqlen}: "
                    f"{eq}/{off.size} logits identical, raw-bytes "
                    f"identical={bitwise}, max abs diff {mx_:.1e}"
                )
                self.assertEqual(eq, off.size)
                self.assertTrue(bitwise)
                self.assertEqual(mx_, 0.0)


if __name__ == "__main__":
    unittest.main()
