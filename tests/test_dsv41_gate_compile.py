# Copyright © 2026 Adam Durham (hermes-gw)
"""DSV41 MoE Gate: compiled vs eager routing must be bit-identical.

Covers both score branches, the kill-switch contract (DSV41_GATE_COMPILE,
default off), and the two activation dtypes that reach the gate at runtime.
"""

import os
import unittest

import mlx.core as mx

from mlx_lm.models.deepseek_v41 import moe
from mlx_lm.models.deepseek_v41.config import ModelArgs


def _make_gate(**overrides):
    kwargs: dict = {"dim": 64, "n_routed_experts": 96, "n_activated_experts": 6}
    kwargs.update(overrides)
    args = ModelArgs(**kwargs)
    gate = moe.Gate(args)
    gate.weight = (mx.random.normal(gate.weight.shape) * 0.05).astype(mx.float32)
    gate.bias = (mx.random.normal(gate.bias.shape) * 0.05).astype(mx.float32)
    return gate


def _routing(gate, x):
    """(eager, compiled) routing for one input, restoring the switch after."""
    prev = moe._GATE_COMPILE
    try:
        moe._GATE_COMPILE = False
        eager = gate(x)
        moe._GATE_COMPILE = True
        compiled = gate(x)
    finally:
        moe._GATE_COMPILE = prev
    return eager, compiled


class TestGateCompile(unittest.TestCase):
    def assert_routing_equal(self, gate, x, msg=""):
        (w_e, i_e), (w_c, i_c) = _routing(gate, x)
        mx.eval(w_e, i_e, w_c, i_c)
        self.assertTrue(bool(mx.array_equal(w_e, w_c)), f"weights differ {msg}")
        self.assertTrue(bool(mx.array_equal(i_e, i_c)), f"top-k indices differ {msg}")

    def test_kill_switch_contract(self):
        # The module flag must track the env var exactly (default off).
        self.assertEqual(moe._GATE_COMPILE,
                         os.environ.get("DSV41_GATE_COMPILE", "0") == "1")

    def test_bit_equal_default_config(self):
        mx.random.seed(7)
        gate = _make_gate()
        for dtype in (mx.float32, mx.bfloat16):
            mx.random.seed(11)
            x = mx.random.normal((37, 64)).astype(dtype)
            self.assert_routing_equal(gate, x, msg=f"dtype={dtype}")

    def test_bit_equal_sigmoid_no_norm(self):
        mx.random.seed(13)
        gate = _make_gate(score_func="sigmoid", n_activated_experts=3,
                          norm_topk_prob=False)
        mx.random.seed(17)
        x = mx.random.normal((37, 64)).astype(mx.float32)
        self.assert_routing_equal(gate, x)


if __name__ == "__main__":
    unittest.main()
