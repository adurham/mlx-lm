# Copyright © 2026 Apple Inc.

"""GAP 3: prove `_dsv4_torch_reference.py` is a faithful transcription.

The concern raised in review: the Phase 3a parity numbers only mean something if
the torch oracle was derived from DeepSeek's ACTUAL Vision-Exp ``model.py``. If
it were reconstructed from the fork's own MLX implementation (or from a prior
worker's paraphrase), matching parity would prove nothing beyond a shared
misreading.

This test settles it MECHANICALLY rather than by assertion: it parses both the
oracle and the reference source and compares their **abstract syntax trees**.
An AST comparison ignores whitespace, comments, line wrapping and docstrings,
but is exact about operations, operands, argument order, and branch structure —
so it cannot be satisfied by something that merely looks similar.

Checked (all must be AST-IDENTICAL):
  * ``get_image_visible``
  * ``get_window_topk_idxs_visible``
  * ``get_window_topk_idxs`` (the non-visible baseline)
  * ``Gate.forward``, modulo ONE documented substitution:
    ``linear(...)`` -> ``F.linear(...)``. The reference's own ``linear()``
    helper dispatches to ``fp4_gemm`` / ``fp8_gemm`` for quantized weight dtypes
    and otherwise ``return F.linear(x, weight)``. Every tensor in the oracle is
    float32, so ``F.linear`` is the branch the real model takes for this op too.
    The substitution is asserted to be the ONLY textual difference.

The reference file is DeepSeek's ``inference/model.py`` from the
``DeepSeek-V4-Flash-Vision-Exp`` repo, kept out-of-tree at
``$DSV4_VISION_REF_MODEL`` (default ``/home/hermes/work/dsv4-refs/model_vision.py``).
It is not vendored here: it is third-party source obtained for reference, and
the fork does not redistribute it. When it is absent the test SKIPS with a
message naming the file, so this never silently degrades into a no-op — a skip
is visible in pytest output, an absent test is not.

Run on any machine with torch (MLX not required)::

    PYTHONPATH=<repo>/mlx-lm <venv>/bin/python -m pytest \\
        tests/test_deepseek_v4_reference_provenance.py -q -s
"""

import ast
import os
import re
import textwrap
import unittest

_DEFAULT_REF = "/home/hermes/work/dsv4-refs/model_vision.py"
_REF_PATH = os.environ.get("DSV4_VISION_REF_MODEL", _DEFAULT_REF)
_ORACLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_dsv4_torch_reference.py")

#: The single permitted substitution, and why it is semantically inert.
_LINEAR_SUB = (
    "linear(x.float(), self.weight.float())",
    "F.linear(x.float(), self.weight.float())",
)


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _top_level_func(src, name):
    """Source of a module-level ``def name`` (decorators excluded)."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise LookupError(f"no top-level def {name}")


def _method(src, cls, name):
    """Source of ``cls.name``, dedented to module level."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    return textwrap.dedent(ast.get_source_segment(src, sub))
    raise LookupError(f"no {cls}.{name}")


def _norm_ast(src, *, drop_docstring=True):
    """AST dump with docstrings dropped, so prose cannot mask a difference."""
    tree = ast.parse(src)
    fn = tree.body[0]
    if (
        drop_docstring
        and getattr(fn, "body", None)
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        fn.body = fn.body[1:]
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _stmt_lines(src):
    """Comment-free, whitespace-normalized statements (for the report)."""
    out = []
    for line in src.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(re.sub(r"\s+", " ", s))
    return out


@unittest.skipUnless(
    os.path.exists(_REF_PATH),
    f"DeepSeek Vision-Exp reference model.py not found at {_REF_PATH}; set "
    f"DSV4_VISION_REF_MODEL to run the provenance check",
)
class TestOracleIsTranscribedFromTheReference(unittest.TestCase):
    """Every ported function must be AST-identical to DeepSeek's source."""

    @classmethod
    def setUpClass(cls):
        cls.ref = _read(_REF_PATH)
        cls.oracle = _read(_ORACLE_PATH)

    def _compare_function(self, name, label):
        r = _top_level_func(self.ref, name)
        o = _top_level_func(self.oracle, name)
        same = _norm_ast(r) == _norm_ast(o)
        print(
            f"\n[GAP 3 provenance] {label}\n"
            f"    reference: {_REF_PATH}\n"
            f"    oracle:    tests/_dsv4_torch_reference.py\n"
            f"    statements: {len(_stmt_lines(r))} (ref) vs "
            f"{len(_stmt_lines(o))} (oracle)\n"
            f"    AST IDENTICAL (ignoring comments/whitespace/docstrings, "
            f"exact on ops/operands/order/branches): {same}"
        )
        self.assertTrue(
            same,
            f"{name} in the oracle is NOT an op-for-op transcription of "
            f"{_REF_PATH}; the Phase 3 parity numbers would be meaningless",
        )

    def test_get_image_visible(self):
        self._compare_function("get_image_visible", "get_image_visible")

    def test_get_window_topk_idxs_visible(self):
        self._compare_function(
            "get_window_topk_idxs_visible", "get_window_topk_idxs_visible"
        )

    def test_get_window_topk_idxs(self):
        self._compare_function(
            "get_window_topk_idxs", "get_window_topk_idxs (non-visible baseline)"
        )

    def test_gate_forward_modulo_the_documented_linear_substitution(self):
        ref = _method(self.ref, "Gate", "forward")
        ora = _method(self.oracle, "RefGate", "forward")

        self.assertIn(
            _LINEAR_SUB[0],
            ref,
            "the reference's Gate.forward no longer calls linear(...) as "
            "expected; re-derive the substitution before trusting this test",
        )
        subbed = ref.replace(*_LINEAR_SUB)
        same = _norm_ast(subbed) == _norm_ast(ora)

        # The substitution must be the ONLY difference: with it applied the
        # ASTs match; without it they must NOT (else the test proves nothing
        # about that line).
        unsubbed_same = _norm_ast(ref) == _norm_ast(ora)

        ref_stmts, ora_stmts = _stmt_lines(subbed), _stmt_lines(ora)
        print(
            f"\n[GAP 3 provenance] Gate.forward\n"
            f"    reference: {_REF_PATH} (class Gate)\n"
            f"    oracle:    tests/_dsv4_torch_reference.py (class RefGate)\n"
            f"    statements: {len(ref_stmts)} (ref) vs {len(ora_stmts)} (oracle)\n"
            f"    permitted substitution: {_LINEAR_SUB[0]!r} -> "
            f"{_LINEAR_SUB[1]!r}\n"
            f"      (model.py's own linear() returns F.linear(x, weight) for "
            f"non-quantized weight dtypes; all oracle tensors are float32)\n"
            f"    AST IDENTICAL after substitution: {same}\n"
            f"    AST identical WITHOUT substitution: {unsubbed_same} "
            f"(expected False — proves the substitution is real, not vacuous)"
        )
        self.assertTrue(
            same,
            "RefGate.forward is NOT an op-for-op transcription of the "
            "reference Gate.forward",
        )
        self.assertFalse(unsubbed_same)

    def test_reference_linear_helper_dispatches_to_f_linear(self):
        """Justify the one substitution by reading the reference's own helper."""
        src = _top_level_func(self.ref, "linear")
        stmts = _stmt_lines(src)
        fallback = [s for s in stmts if s.startswith("return F.linear")]
        print(
            f"\n[GAP 3 provenance] reference linear() helper, "
            f"non-quantized fallback branch:\n"
            f"    " + "\n    ".join(stmts)
        )
        self.assertEqual(
            fallback,
            ["return F.linear(x, weight)"],
            "the reference's linear() no longer falls back to F.linear; the "
            "oracle's substitution is no longer justified",
        )

    def test_image_sentinel_constants_match_the_reference_import(self):
        """IMAGE_* ids must be image_processor.py's, not invented."""
        self.assertIn("from image_processor import IMAGE, IMAGE_START, IMAGE_END", self.ref)
        ns = {}
        exec(  # noqa: S102 - reading our own test helper's constants
            "\n".join(
                l for l in self.oracle.splitlines()
                if re.match(r"^IMAGE\w* = \d+$", l)
            ),
            ns,
        )
        got = {k: v for k, v in ns.items() if k.startswith("IMAGE")}
        want = {
            "IMAGE_START": 0,
            "IMAGE_PAD": 1,
            "IMAGE": 2,
            "IMAGE_NEW_LINE": 3,
            "IMAGE_END": 4,
        }
        print(
            f"\n[GAP 3 provenance] image sentinels: oracle {got}\n"
            f"    expected from image_processor.py's "
            f"`IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)`: "
            f"{want}"
        )
        self.assertEqual(got, want)


class TestProvenanceCheckIsNotSilentlySkipped(unittest.TestCase):
    """Make the skip condition itself visible, so a missing ref is obvious."""

    def test_reference_path_is_reported(self):
        present = os.path.exists(_REF_PATH)
        print(
            f"\n[GAP 3] reference model.py at {_REF_PATH}: "
            f"{'PRESENT — provenance asserted above' if present else 'ABSENT — provenance tests SKIPPED'}"
        )
        self.assertTrue(
            os.path.exists(_ORACLE_PATH),
            "the oracle itself must exist regardless of the reference",
        )


if __name__ == "__main__":
    unittest.main()
