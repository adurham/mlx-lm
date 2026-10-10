# Copyright © 2026 Adam Durham (hermes-gw)
"""Per-tensor dense-quant policy (``DSV41_DENSE_POLICY``): parse + resolve + wire.

``DSV41_DENSE_POLICY`` overrides the single global ``DSV41_DENSE`` format on a
per-tensor basis, so a mixed-precision dense roster (e.g. attention at q8g64
while the shared experts stay q6g64) can be built. The spec is either an inline
``selector=mode,...`` string or a path to a JSON file; selectors are ``fnmatch``
globs on the FULL tensor name and the LAST match wins. Anything unmatched --
and an unset/empty policy -- falls back to the global ``DSV41_DENSE`` behavior.

This file pins, with no checkpoint and no import-time env mutation (the parser
and resolver are pure functions):

1. **inline parsing**: ``selector=mode`` pairs, order preserved, whitespace ok,
   empty/None -> [];
2. **JSON parsing**: a dict and a list of ``[selector, mode]`` pairs;
3. **mode validation**: an unknown mode raises ``ValueError`` (inline OR JSON);
4. **resolution**: glob match on real-shaped names, last-match-wins, and the
   fallback to the global base mode when nothing matches;
5. **byte-identity**: with the policy unset, ``_dense``/``_dense_slice`` build
   EXACTLY the same modules (same class, bits, group, quantized tuples) as the
   pre-policy engine;
6. **wiring**: ``_dense``/``_dense_slice`` consult the policy per tensor.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_dense_policy.py -q
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import EXL3Linear
from mlx_lm.models.exl3.ref.layer import EXL3Layer
from mlx_lm.models.deepseek_v41 import exl3_build as eb

IN_TILES, OUT_TILES, K = 16, 32, 2                      # in 256, out 512
IN_F, OUT_F = IN_TILES * 16, OUT_TILES * 16
WORLD = 2
LOAD_DENSE_LAYER = "mlx_lm.models.exl3.loader.load_dense_layer"

# real-shaped dense tensor names (from a V4.1 checkpoint index)
ATTN = "layers.20.attn.wq_b"
WO_A = "layers.20.attn.wo_a.slice.0"
SHARED = "layers.33.ffn.shared_experts.w2"
OTHER = "layers.5.ffn.gate.weight"     # matches neither attn nor shared_experts


def _synth(key: str = "synth", seed: int = 0) -> EXL3Layer:
    rng = np.random.default_rng(seed)
    packed = 256 * K // 16
    return EXL3Layer(
        key=key,
        in_features=IN_F,
        out_features=OUT_F,
        k=K,
        trellis=rng.integers(0, 2 ** 16, size=(IN_TILES, OUT_TILES, packed), dtype=np.uint16),
        suh=rng.choice(np.array([-1.0, 1.0], np.float16), size=IN_F),
        svh=rng.choice(np.array([-1.0, 1.0], np.float16), size=OUT_F),
        mul1=True,
    )


def _tuples_equal(a, b) -> bool:
    mx.eval(*a, *b)
    return all(mx.array_equal(x, y) for x, y in zip(a, b))


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

class InlineParseTest(unittest.TestCase):
    def test_pairs_in_order(self):
        got = eb._parse_dense_policy("layers.*.attn.*=q8g64,layers.*.ffn.shared_experts.*=exl3")
        self.assertEqual(got, [("layers.*.attn.*", "q8g64"),
                               ("layers.*.ffn.shared_experts.*", "exl3")])

    def test_whitespace_tolerated_and_blank_dropped(self):
        got = eb._parse_dense_policy("  layers.*.attn.* = q6g32 ,, head=q8g64 ")
        self.assertEqual(got, [("layers.*.attn.*", "q6g32"), ("head", "q8g64")])

    def test_empty_specs_return_empty(self):
        for spec in (None, "", "   ", ","):
            self.assertEqual(eb._parse_dense_policy(spec), [], repr(spec))

    def test_unknown_mode_rejected_inline(self):
        with self.assertRaises(ValueError) as cm:
            eb._parse_dense_policy("layers.*.attn.*=q4g64")
        self.assertIn("q4g64", str(cm.exception))
        self.assertIn("q6g64", str(cm.exception))       # message lists allowed modes

    def test_entry_without_equals_rejected(self):
        with self.assertRaises(ValueError):
            eb._parse_dense_policy("layers.*.attn.*")


class JsonParseTest(unittest.TestCase):
    def _write(self, obj) -> str:
        d = tempfile.mkdtemp(prefix="dsv41_policy_")
        path = os.path.join(d, "policy.json")
        with open(path, "w") as fh:
            json.dump(obj, fh)
        return path

    def test_dict(self):
        path = self._write({"layers.*.attn.*": "q8g64", "head": "exl3"})
        self.assertEqual(eb._parse_dense_policy(path),
                         [("layers.*.attn.*", "q8g64"), ("head", "exl3")])

    def test_list_of_pairs(self):
        path = self._write([["layers.*.attn.*", "q6g64"], ["layers.*.ffn.*", "q6g32"]])
        self.assertEqual(eb._parse_dense_policy(path),
                         [("layers.*.attn.*", "q6g64"), ("layers.*.ffn.*", "q6g32")])

    def test_json_unknown_mode_rejected(self):
        path = self._write({"layers.*.attn.*": "bf16"})
        with self.assertRaises(ValueError) as cm:
            eb._parse_dense_policy(path)
        self.assertIn("bf16", str(cm.exception))

    def test_json_wrong_type_rejected(self):
        path = self._write("not a spec")
        with self.assertRaises(ValueError):
            eb._parse_dense_policy(path)


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

class ResolveTest(unittest.TestCase):
    def test_glob_match_on_full_name(self):
        policy = eb._parse_dense_policy("layers.*.attn.*=q8g64")
        # '*' crosses '.': both attn names match, the shared-expert name does not
        self.assertEqual(eb._resolve_dense_mode(ATTN, policy, "exl3"),
                         ("affine", 8, 64))
        self.assertEqual(eb._resolve_dense_mode(WO_A, policy, "exl3"),
                         ("affine", 8, 64))
        self.assertEqual(eb._resolve_dense_mode(SHARED, policy, "exl3"),
                         ("exl3", None, None))            # no match -> base exl3

    def test_every_policy_mode_maps(self):
        for mode, want in (("exl3", ("exl3", None, None)),
                           ("q6g64", ("affine", 6, 64)),
                           ("q6g32", ("affine", 6, 32)),
                           ("q8g64", ("affine", 8, 64))):
            policy = eb._parse_dense_policy(f"layers.*= {mode}")
            self.assertEqual(eb._resolve_dense_mode(ATTN, policy, "exl3"), want, mode)

    def test_last_match_wins(self):
        policy = eb._parse_dense_policy(
            "layers.*=q8g64,layers.*.attn.*=exl3,layers.20.*=q6g32")
        # all three match ATTN; the last one wins
        self.assertEqual(eb._resolve_dense_mode(ATTN, policy, "affine6"),
                         ("affine", 6, 32))
        # only the first two match layer 21 -> the last of those wins
        self.assertEqual(eb._resolve_dense_mode("layers.21.attn.wq_b", policy, "affine6"),
                         ("exl3", None, None))
        # only the first matches a shared expert
        self.assertEqual(eb._resolve_dense_mode(SHARED, policy, "affine6"),
                         ("affine", 8, 64))

    def test_fallback_uses_base_mode(self):
        policy = eb._parse_dense_policy("layers.*.attn.*=q8g64")
        self.assertEqual(eb._resolve_dense_mode(OTHER, policy, "exl3"),
                         ("exl3", None, None))
        self.assertEqual(eb._resolve_dense_mode(OTHER, policy, "affine6"),
                         ("affine", 6, 64))
        self.assertEqual(eb._resolve_dense_mode(OTHER, policy, "affine5"),
                         ("affine", 5, 64))


class DefaultUnsetTest(unittest.TestCase):
    """Requirement: with the env unset the resolver returns the global behavior."""

    def test_unset_env_parses_to_empty(self):
        # the module does exactly this at import when DSV41_DENSE_POLICY is unset
        self.assertEqual(eb._parse_dense_policy(os.environ.get("DSV41_DENSE_POLICY", "")),
                         [] if not os.environ.get("DSV41_DENSE_POLICY") else
                         eb._parse_dense_policy(os.environ["DSV41_DENSE_POLICY"]))

    def test_module_policy_is_a_list(self):
        self.assertIsInstance(eb._DENSE_POLICY, list)

    def test_empty_policy_returns_base_for_every_base(self):
        for base, want in (("exl3", ("exl3", None, None)),
                           ("affine8", ("affine", 8, 64)),
                           ("affine6", ("affine", 6, 64)),
                           ("affine5", ("affine", 5, 64))):
            self.assertEqual(eb._resolve_dense_mode(ATTN, [], base), want, base)


# --------------------------------------------------------------------------
# byte-identity when the policy is unset
# --------------------------------------------------------------------------

class ByteIdentityTest(unittest.TestCase):
    """An unset policy builds EXACTLY the pre-policy modules."""

    def setUp(self):
        self.layer = _synth(seed=7)

    def test_dense_unset_matches_global_affine6(self):
        ref = eb.AffineProj(self.layer, 6, 64)            # the pre-policy path
        with mock.patch.object(eb, "DENSE_MODE", "affine6"), \
             mock.patch.object(eb, "_DENSE_POLICY", []), \
             mock.patch(LOAD_DENSE_LAYER, lambda ck, n: self.layer):
            got = eb._dense(None, ATTN)
        self.assertIsInstance(got, eb.AffineProj)
        self.assertEqual((got._bits, got._group), (6, 64))
        self.assertTrue(_tuples_equal(got._q, ref._q))

    def test_dense_unset_matches_global_exl3(self):
        with mock.patch.object(eb, "DENSE_MODE", "exl3"), \
             mock.patch.object(eb, "_DENSE_POLICY", []), \
             mock.patch.object(eb, "load_dense_linear",
                               lambda ck, n: EXL3Linear(self.layer)):
            got = eb._dense(None, ATTN)
        self.assertIsInstance(got, eb.Exl3Proj)

    def test_dense_slice_unset_matches_global_affine6(self):
        for axis in ("out", "in"):
            w = eb._slice_weight(self.layer, axis=axis, rank=0, world=WORLD)
            expect = ((w.shape[0], self.layer.in_features) if axis == "out"
                      else (self.layer.out_features, w.shape[1]))
            ref = eb.AffineProj.from_weight(w, 6, 64, expect=expect)
            with mock.patch.object(eb, "DENSE_MODE", "affine6"), \
                 mock.patch.object(eb, "_DENSE_POLICY", []), \
                 mock.patch(LOAD_DENSE_LAYER, lambda ck, n: self.layer):
                got = eb._dense_slice(None, ATTN, axis=axis, rank=0, world=WORLD)
            self.assertIsInstance(got, eb.AffineProj)
            self.assertEqual((got._bits, got._group), (6, 64))
            self.assertTrue(_tuples_equal(got._q, ref._q), axis)

    def test_dense_slice_unset_matches_global_exl3(self):
        with mock.patch.object(eb, "DENSE_MODE", "exl3"), \
             mock.patch.object(eb, "_DENSE_POLICY", []), \
             mock.patch(LOAD_DENSE_LAYER, lambda ck, n: self.layer):
            got = eb._dense_slice(None, ATTN, axis="out", rank=0, world=WORLD)
        self.assertIsInstance(got, eb.Exl3Proj)


# --------------------------------------------------------------------------
# wiring: the policy is consulted per tensor by _dense / _dense_slice
# --------------------------------------------------------------------------

class DispatchTest(unittest.TestCase):
    def test_dense_routes_per_tensor(self):
        policy = eb._parse_dense_policy(
            "layers.*.attn.*=q8g64,layers.*.ffn.shared_experts.*=exl3")
        # a fresh synthetic layer per call: EXL3Proj.release_source() nulls the
        # source layer's trellis, so a shared instance cannot be reused.
        with mock.patch.object(eb, "DENSE_MODE", "affine6"), \
             mock.patch.object(eb, "_DENSE_POLICY", policy), \
             mock.patch(LOAD_DENSE_LAYER, lambda ck, n: _synth(seed=8)), \
             mock.patch.object(eb, "load_dense_linear",
                               lambda ck, n: EXL3Linear(_synth(seed=8))):
            attn = eb._dense(None, ATTN)                   # -> q8g64
            shared = eb._dense(None, SHARED)               # -> exl3
            other = eb._dense(None, OTHER)                 # -> base affine6
        self.assertIsInstance(attn, eb.AffineProj)
        self.assertEqual((attn._bits, attn._group), (8, 64))
        self.assertIsInstance(shared, eb.Exl3Proj)
        self.assertIsInstance(other, eb.AffineProj)
        self.assertEqual((other._bits, other._group), (6, 64))

    def test_dense_slice_routes_per_tensor(self):
        policy = eb._parse_dense_policy("layers.*.ffn.shared_experts.*=q6g32")
        with mock.patch.object(eb, "DENSE_MODE", "affine6"), \
             mock.patch.object(eb, "_DENSE_POLICY", policy), \
             mock.patch(LOAD_DENSE_LAYER, lambda ck, n: _synth(seed=9)):
            sliced = eb._dense_slice(None, SHARED, axis="in", rank=0, world=WORLD)
            base = eb._dense_slice(None, ATTN, axis="out", rank=0, world=WORLD)
        self.assertEqual((sliced._bits, sliced._group), (6, 32))
        self.assertEqual((base._bits, base._group), (6, 64))


if __name__ == "__main__":
    unittest.main()
