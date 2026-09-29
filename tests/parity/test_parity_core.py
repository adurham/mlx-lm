# Unit tests for the MLX-free core of the parity harness.
#
#   python3 -m unittest tests.parity.test_parity_core -v     (or)
#   python3 tests/parity/test_parity_core.py
#
# These run anywhere (no MLX, no GPU): they cover the comparison / gate /
# fingerprint / reference-file logic that the GPU runner is built on, so a
# broken verdict path is caught before spending a GPU run on it.

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_core as pc  # noqa: E402


class TestParse(unittest.TestCase):
    def test_parse_layers(self):
        self.assertEqual(pc.parse_layers("2,20,24"), [2, 20, 24])
        self.assertEqual(pc.parse_layers("2-5"), [2, 3, 4, 5])
        self.assertEqual(pc.parse_layers("all", 4), [0, 1, 2, 3])
        self.assertEqual(pc.parse_layers("1,1,0"), [0, 1])
        with self.assertRaises(ValueError):
            pc.parse_layers("40", 40)

    def test_parse_perturb(self):
        self.assertEqual(pc.parse_perturb("none"), {"kind": "none"})
        self.assertEqual(pc.parse_perturb(None), {"kind": "none"})
        self.assertEqual(pc.parse_perturb("argmax:3"), {"kind": "argmax", "step": 3})
        self.assertEqual(pc.parse_perturb("noise:0.02"), {"kind": "noise", "rel": 0.02})
        with self.assertRaises(ValueError):
            pc.parse_perturb("spike")


class TestCompare(unittest.TestCase):
    def test_equal(self):
        r = pc.compare_tokens([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertTrue(r["ok"])
        self.assertIsNone(r["first_div"])
        self.assertEqual(r["n_match"], 4)

    def test_flip_at_2(self):
        r = pc.compare_tokens([1, 2, 3, 4], [1, 2, 9, 4])
        self.assertFalse(r["ok"])
        self.assertEqual(r["first_div"], 2)
        self.assertEqual(r["n_match"], 3)
        self.assertEqual(r["n_common"], 4)

    def test_length_diff(self):
        r = pc.compare_tokens([1, 2, 3], [1, 2, 3, 4])
        self.assertFalse(r["ok"])
        self.assertTrue(r["length_diff"])
        self.assertIsNone(r["first_div"])

    def test_compare_runs_input_mismatch(self):
        ref = [{"name": "a", "ids": [1, 2], "tokens": [5]}]
        cur = [{"name": "a", "ids": [1, 3], "tokens": [5]}]
        r = pc.compare_runs(ref, cur)
        self.assertFalse(r["ok"])
        self.assertEqual(r["prompts"][0]["error"], "prompt ids differ")

    def test_compare_runs_count_mismatch(self):
        r = pc.compare_runs([], [{"name": "a", "ids": [1], "tokens": [5]}])
        self.assertFalse(r["ok"])
        self.assertIn("prompt count", r["error"])


class TestGates(unittest.TestCase):
    def test_cos_gate_pass(self):
        items = [{"layer": 2, "cos": 0.99990}, {"layer": 20, "cos": 0.99983}]
        g = pc.cos_gate(items)
        self.assertTrue(g["ok"])
        self.assertEqual(g["worst_layer"], 20)

    def test_cos_gate_fail(self):
        g = pc.cos_gate([{"layer": 2, "cos": 0.99990}, {"layer": 20, "cos": 0.812}])
        self.assertFalse(g["ok"])
        self.assertEqual(g["failed"], [20])

    def test_cos_gate_reference_calibrated(self):
        # above the absolute bar but well below the recorded clean value: the
        # deviation gate must fail it even though the floor passes
        items = [{"layer": 2, "cos": 0.99981}]
        self.assertTrue(pc.cos_gate(items)["ok"])
        g = pc.cos_gate(items, ref_cos={2: 0.99999})
        self.assertFalse(g["ok"])
        self.assertLess(g["worst_deviation"], 0)
        # a clean run near the recorded value passes both
        g2 = pc.cos_gate([{"layer": 2, "cos": 0.99998}], ref_cos={2: 0.99999})
        self.assertTrue(g2["ok"])

    def test_nll_gate(self):
        self.assertTrue(pc.nll_gate(1.003, 0.0288, 78.3)["ok"])
        # the measured chained-p46 run: over the bars -> FAIL
        self.assertFalse(pc.nll_gate(1.0231, 0.0325, 76.8)["ok"])

    def test_negctl_verdict(self):
        good = [{"name": "a", "expect_detected": False, "detected": False},
                {"name": "b", "expect_detected": True, "detected": True}]
        bad = [{"name": "a", "expect_detected": True, "detected": False}]
        self.assertTrue(pc.negctl_verdict(good)["ok"])
        self.assertFalse(pc.negctl_verdict(bad)["ok"])

    def test_negctl_ladder_policy(self):
        # a harness that always reports a mismatch must NOT pass: the quiet
        # rung has to stay quiet and the loud rung has to fire
        events = [{"name": "clean", "expect_detected": False, "detected": False}]
        alldetect = [{"name": "clean", "expect_detected": False, "detected": True}]
        low = {"rel": 0.01, "detected": False}
        high = {"rel": 0.64, "detected": True}
        self.assertTrue(pc.negctl_verdict(events, low_rung=low,
                                          high_rung=high)["ok"])
        self.assertFalse(pc.negctl_verdict(alldetect, low_rung=low,
                                           high_rung=high)["ok"])
        self.assertFalse(pc.negctl_verdict(
            events, low_rung={"rel": 0.01, "detected": True},
            high_rung=high)["ok"])

    def test_overall(self):
        ok, s = pc.overall({"x": True, "y": False})
        self.assertFalse(ok)
        self.assertIn("y=FAIL", s)


class TestFingerprint(unittest.TestCase):
    def _prompts(self):
        return [{"name": "p", "ids": [1, 2, 3]}]

    def _cfg(self, **kw):
        c = {"layers": [2, 20], "dist": "none", "world": 2, "max_new": 16,
             "chunk": 0, "model_dir": "/m", "native_dir": "/n",
             "token_map_digest": "aa", "trace_digest": "tt", "pkg_digest": "deadbeef"}
        c.update(kw)
        return c

    def test_structural_identity(self):
        a = pc.make_fingerprint(self._cfg(), self._prompts())
        b = pc.make_fingerprint(self._cfg(), self._prompts())
        self.assertEqual(a["structural"], b["structural"])
        self.assertEqual(pc.fingerprint_mismatch(a, b), ({}, False))

    def test_structural_diff_is_hard(self):
        a = pc.make_fingerprint(self._cfg(), self._prompts())
        b = pc.make_fingerprint(self._cfg(layers=[2]), self._prompts())
        struct, digest_diff = pc.fingerprint_mismatch(a, b)
        self.assertIn("layers", struct)
        self.assertFalse(digest_diff)

    def test_pkg_digest_is_a_gate(self):
        # the code digest is a gate (check mode fails on drift unless
        # --allow-code-drift); fingerprint_mismatch reports it separately
        a = pc.make_fingerprint(self._cfg(), self._prompts())
        b = pc.make_fingerprint(self._cfg(pkg_digest="cafe"), self._prompts())
        struct, digest_diff = pc.fingerprint_mismatch(a, b)
        self.assertEqual(struct, {})       # same experiment, different code
        self.assertTrue(digest_diff)       # -> gate on the code digest

    def test_trace_digest_is_structural(self):
        a = pc.make_fingerprint(self._cfg(), self._prompts())
        b = pc.make_fingerprint(self._cfg(trace_digest="other"), self._prompts())
        struct, _ = pc.fingerprint_mismatch(a, b)
        self.assertIn("trace_digest", struct)

    def test_digest_ignores_rank(self):
        a = pc.make_fingerprint(self._cfg(), self._prompts())
        b = pc.make_fingerprint(self._cfg(rank=1), self._prompts())
        struct, _ = pc.fingerprint_mismatch(a, b)
        self.assertNotIn("rank", struct)

    def test_prompt_digest_tracks_prompt_change(self):
        a = pc.make_fingerprint(self._cfg(), self._prompts())
        b = pc.make_fingerprint(self._cfg(),
                                [{"name": "p", "ids": [1, 2, 4]}])
        struct, _ = pc.fingerprint_mismatch(a, b)
        self.assertIn("prompts_digest", struct)


class TestReferenceIO(unittest.TestCase):
    def test_roundtrip_and_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            fp = pc.make_fingerprint({"layers": [2], "dist": "none", "world": 2,
                                      "max_new": 4, "chunk": 0, "model_dir": "m",
                                      "native_dir": "n", "token_map_digest": "t",
                                      "trace_digest": "tr", "pkg_digest": "p"},
                                     [{"name": "p", "ids": [1]}])
            prompts = [{"name": "p", "ids": [1], "tokens": [7, 8],
                        "stats": {"ms_step": 1.0}}]
            pc.save_reference(path, {"created": "now", "timings": {"total_s": 1.0},
                                     "memory": {"peak_gb": 5.0},
                                     "nll": {"mean": 1.0}}, fp, prompts, note="x")
            doc = pc.load_reference(path)
            self.assertEqual(doc["schema"], pc.SCHEMA)
            self.assertEqual(doc["prompts"][0]["tokens"], [7, 8])
            self.assertEqual(doc["nll"]["mean"], 1.0)
            self.assertEqual(doc["note"], "x")
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_ref_is_not_overwritten_silently(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            fp = pc.make_fingerprint({"layers": [2], "dist": "none", "world": 2,
                                      "max_new": 4, "chunk": 0, "model_dir": "m",
                                      "native_dir": "n", "token_map_digest": "t",
                                      "trace_digest": "tr", "pkg_digest": "p"},
                                     [{"name": "p", "ids": [1]}])
            pc.save_reference(path, {}, fp, [{"name": "p", "ids": [1],
                                              "tokens": [1]}])
            with self.assertRaises(FileExistsError):
                pc.save_reference(path, {}, fp, [{"name": "p", "ids": [1],
                                                  "tokens": [2]}])
            # the original survived
            self.assertEqual(pc.load_reference(path)["prompts"][0]["tokens"], [1])
            # explicit overwrite works
            pc.save_reference(path, {}, fp, [{"name": "p", "ids": [1],
                                              "tokens": [2]}], overwrite=True)
            self.assertEqual(pc.load_reference(path)["prompts"][0]["tokens"], [2])

    def test_bad_schema_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            with open(path, "w") as f:
                json.dump({"schema": "nope"}, f)
            with self.assertRaises(ValueError):
                pc.load_reference(path)


class TestDigests(unittest.TestCase):
    def test_pkg_digest_missing_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("missing", pc.pkg_digest(d, ["nope.py"]))

    def test_file_digest_stable(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f")
            with open(p, "wb") as f:
                f.write(b"abc")
            self.assertEqual(pc.file_digest(p), pc.sha256_text("abc"))

    def test_pkg_digest_changes_with_content(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.py")
            with open(p, "w") as f:
                f.write("x = 1\n")
            a = pc.pkg_digest(d, ["f.py"])
            with open(p, "w") as f:
                f.write("x = 2\n")
            self.assertNotEqual(a, pc.pkg_digest(d, ["f.py"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
