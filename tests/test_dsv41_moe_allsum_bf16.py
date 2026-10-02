# Copyright © 2026 Adam Durham (hermes-gw)
"""DSV41_MOE_ALLSUM_BF16: flag-off is the exact baseline; flag-on is bf16-bounded.

The flag halves the MoE routed-sum collective's payload on the wire (fp32 ->
bf16). It is NOT bit-exact by construction, so this file pins the contract that
survives an A/B:

* flag OFF -> the payload on the wire is fp32 and the result is BITWISE the
  baseline `collective.all_sum` result (the flag must be a pure no-op);
* flag ON  -> the payload is bf16, the widened result is fp32, and the
  elementwise error against the input is bounded by bf16 round-to-nearest
  (|err| <= 2**-8 * |x| + 2**-134, the subnormal floor); the test prints the
  observed max absolute / relative error so the A/B owner has the number;
* the MoE call site (moe.py) actually takes the lowp path when the flag is on
  (equality alone cannot prove wiring), and the draft MoE (mtp.py) does NOT
  (it was deliberately left exact -- see the report).

Everything runs world-1 on the CPU wheel: the real `mx.distributed.all_sum`
with an uninitialised or world-1 group is the identity, so the tests observe
the wire payload through a spy without JACCL.
"""

from __future__ import annotations

import inspect
import os
import unittest

import mlx.core as mx

from mlx_lm.models.deepseek_v41 import collective
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41 import moe as moe_mod


class _PayloadSpy:
    """Records the dtype/shape of every payload handed to the real collective.

    Instrument at ``collective._raw`` -- the single chokepoint every collective
    in this module goes through. Patching ``mx.distributed.all_sum`` directly
    does NOT work: ``_raw`` calls ``inspect.unwrap``, so any Python wrapper
    installed on the attribute is unwrapped away before the call and the spy
    would never fire. ``_raw`` returns the callable that actually runs, so
    wrapping ITS return value sees the exact payload that goes on the wire.
    """

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self._orig_raw = collective._raw

    def __enter__(self):
        spy = self
        orig = self._orig_raw

        def raw_spy(fn):
            real = orig(fn)

            def caller(x, *args, **kwargs):
                spy.calls.append((str(x.dtype), tuple(x.shape)))
                return real(x, *args, **kwargs)

            return caller

        collective._raw = raw_spy
        return self

    def __exit__(self, *exc):
        collective._raw = self._orig_raw
        return False


def _force_flag(value: bool) -> bool:
    prev = collective._ALLSUM_BF16
    collective._ALLSUM_BF16 = value
    return prev


def _bitwise(a: mx.array, b: mx.array) -> bool:
    """Exact bit comparison (dtype-agnostic: compare raw bits, not values)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == mx.bfloat16:
        return bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)))
    if a.dtype == mx.float32:
        return bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)))
    return bool(mx.array_equal(a, b))


def _make_moe(group):
    args = ModelArgs(
        dim=32,
        moe_inter_dim=16,
        n_routed_experts=8,
        n_activated_experts=2,
        n_shared_experts=1,
        swiglu_limit=10.0,
    )
    mx.random.seed(5)
    m = moe_mod.MoE(args)
    _randomize(m)
    m.group = group
    return m


def _randomize(module) -> None:
    """Nudge every leaf parameter off its initialization (nested dict-safe)."""
    from mlx.utils import tree_flatten, tree_unflatten

    items = tree_flatten(module.parameters())
    new = []
    for name, arr in items:
        if not isinstance(arr, mx.array):
            continue
        new.append((name, (mx.random.normal(arr.shape) * 0.1).astype(arr.dtype)))
    module.update(tree_unflatten(new))


class TestMoeAllSumBf16(unittest.TestCase):
    # ---- contract ---------------------------------------------------------
    def test_kill_switch_default_off(self):
        """The module flag tracks the env var exactly (default off)."""
        env = os.environ.get("DSV41_MOE_ALLSUM_BF16", "0")
        self.assertEqual(collective._ALLSUM_BF16, env == "1")
        if env != "1":
            self.assertFalse(collective._ALLSUM_BF16,
                             "flag must default OFF when the env var is unset")

    def test_flag_off_is_fp32_and_bitwise_baseline(self):
        """Flag OFF: fp32 on the wire, result bitwise equal to all_sum."""
        mx.random.seed(11)
        prev = _force_flag(False)
        try:
            for shape in [(1, 5120), (4, 5120), (2048, 5120), (1, 3, 5120)]:
                x = mx.random.normal(shape).astype(mx.float32)
                with _PayloadSpy() as spy:
                    got = collective.all_sum_lowp(x)
                mx.eval(got)
                self.assertEqual(spy.calls, [("mlx.core.float32", shape)],
                                 f"payload dtype changed at {shape}")
                self.assertTrue(_bitwise(got, x), f"not identity at {shape}")
                with _PayloadSpy() as spy2:
                    base = collective.all_sum(x)
                mx.eval(base)
                self.assertEqual(spy2.calls, [("mlx.core.float32", shape)])
                self.assertTrue(_bitwise(got, base),
                                f"flag-off result differs from baseline at {shape}")
        finally:
            collective._ALLSUM_BF16 = prev

    def test_flag_on_is_bf16_and_error_bounded(self):
        """Flag ON: bf16 on the wire, fp32 out, |err| <= 2**-8|x| + 2**-134."""
        mx.random.seed(13)
        prev = _force_flag(True)
        try:
            worst_abs = 0.0
            worst_rel = 0.0
            for shape in [(1, 5120), (4, 5120), (2048, 5120)]:
                x = mx.random.normal(shape).astype(mx.float32)
                with _PayloadSpy() as spy:
                    got = collective.all_sum_lowp(x)
                mx.eval(got)
                self.assertEqual(spy.calls, [("mlx.core.bfloat16", shape)],
                                 f"payload not bf16 at {shape}")
                self.assertEqual(got.dtype, mx.float32)
                self.assertTrue(
                    _bitwise(got, x.astype(mx.bfloat16).astype(mx.float32)),
                    f"widened bf16 payload != result at {shape}",
                )
                err = mx.abs(got - x)
                bound = (2.0 ** -8) * mx.abs(x) + 2.0 ** -134
                self.assertTrue(bool(mx.all(err <= bound)),
                                f"error exceeds bf16 round-to-nearest at {shape}")
                worst_abs = max(worst_abs, float(mx.max(err)))
                rel = float(mx.max(err / mx.maximum(mx.abs(x), mx.array(1e-30))))
                worst_rel = max(worst_rel, rel)
            print(f"[allsum-bf16] world-1 max abs err={worst_abs:.6e} "
                  f"max rel err={worst_rel:.6e} (bound 2^-8={2.0 ** -8:.6e})")
            self.assertLessEqual(worst_rel, 2.0 ** -8)
        finally:
            collective._ALLSUM_BF16 = prev

    # ---- wiring: the MoE call site takes the flag, the draft one does not --
    def test_moe_site_takes_lowp_path_when_on(self):
        group = mx.distributed.init()
        m = _make_moe(group)
        mx.random.seed(17)
        x = mx.random.normal((2, 3, 32)).astype(mx.float32)

        prev = _force_flag(False)
        try:
            # Baseline: the exact pre-change call (collective.all_sum at the site).
            with _PayloadSpy() as spy_base:
                base = m(x)
            mx.eval(base)
            self.assertEqual([c[0] for c in spy_base.calls], ["mlx.core.float32"])

            with _PayloadSpy() as spy_off:
                off = m(x)
            mx.eval(off)
            self.assertEqual([c[0] for c in spy_off.calls], ["mlx.core.float32"])
            self.assertTrue(_bitwise(off, base),
                            "flag-off MoE forward differs from the exact baseline")

            _force_flag(True)
            with _PayloadSpy() as spy_on:
                on = m(x)
            mx.eval(on)
            self.assertEqual([c[0] for c in spy_on.calls], ["mlx.core.bfloat16"],
                             "flag-on MoE forward did not take the lowp path")
            self.assertFalse(_bitwise(on, base),
                             "flag-on MoE forward should differ (bf16 payload)")
            err = mx.abs(on - base)
            self.assertLess(float(mx.max(err)), 0.05 * float(mx.max(mx.abs(base))) + 1e-3)
        finally:
            collective._ALLSUM_BF16 = prev

    def test_draft_moe_stays_exact(self):
        """The draft head's MoE collective is deliberately NOT routed via lowp."""
        src = inspect.getsource(moe_mod.MoE.__call__)
        self.assertIn("all_sum_lowp", src)
        import mlx_lm.models.deepseek_v41.mtp as mtp_mod

        draft_src = inspect.getsource(mtp_mod.DraftMoE.__call__)
        self.assertIn("_coll.all_sum", draft_src)
        self.assertNotIn("all_sum_lowp", draft_src)


if __name__ == "__main__":
    unittest.main()
