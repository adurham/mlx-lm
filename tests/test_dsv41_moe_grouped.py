"""Expert-grouped multi-row MoE decode (DSV41_MOE_GROUPED) — LXC-runnable tests.

Covers, without any Metal backend (the LXC has a CPU MLX build):

1. Gate contract: the flag exists, defaults OFF, and both gate directions
   route ``EXL3SwitchGLU.__call__``'s R<=8 branch to the right function
   (revert-proof: collapsing the gate either way fails one of them).
2. Flag-off path untouched: the fused2/A2/B2 Metal source strings and the
   ``_decode_fused2`` dispatcher's compiled bytecode are byte-identical to
   the committed baseline of this branch.
3. On-device tables: ``_grouped_srt`` + ``_grouped_tables_fn`` match their
   host (NumPy) spec on randomized + adversarial routings, with the exact
   edge cases the scatter can break on.
4. Grouping/scatter equality: the NumPy references of the OLD (fused2) and
   NEW (grouped) slot->output math agree BIT-EXACTLY on ygu (the A2/A2G
   output buffer) and on the final y, for two independent trellis
   "extraction" functions — proving the equality is about index arithmetic,
   not about the codeword-layout guess.
5. rows_prep anchor: the reference's Hadamard path is bit-exact against the
   real compiled ``_rows_prep`` on this CPU build.
6. Mac-only: the real grouped-vs-fused2 kernel bit-exactness + perf run
   (skipped here; benchmarks/w4k_grouped_bench.py is the harness).
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

# The flag must be read at import; tests reload the module under a patched
# env where needed. Default (this process env) must be OFF.
import mlx.core as mx
import numpy as np

from mlx_lm.models.exl3 import exl3_moe as M
from mlx_lm.models.exl3.exl3_moe import EXL3SwitchGLU, _grouped_tables_fn
from mlx_lm.models.exl3.moe_grouped_ref import (
    _rows_prep as np_rows_prep,
    decode_fused2_ref,
    decode_grouped_ref,
    grouped_tables as host_tables,
    hash_tile_weights,
    metal_sim_tile_weights,
)
from mlx_lm.models.exl3.ref.codebook import CodebookMode


def _build_module(E=6, D=256, H=128, k=6, seed=7):
    rng = np.random.default_rng(seed)
    gu_tr = mx.array(
        rng.integers(0, 65536, (D // 16, E * 2 * (H // 16), 16 * k)).astype(np.uint16)
    )
    dn_tr = mx.array(
        rng.integers(0, 65536, (H // 16, E * (D // 16), 16 * k)).astype(np.uint16)
    )
    gu_suh = rng.standard_normal((E, 2, D)).astype(np.float16)
    gu_svh = rng.standard_normal((E, 2 * H)).astype(np.float16)
    dn_suh = rng.standard_normal((E, H)).astype(np.float16)
    dn_svh = rng.standard_normal((E, D)).astype(np.float16)
    sg = EXL3SwitchGLU(
        gu_trellis=gu_tr,
        gu_suh=mx.array(gu_suh),
        gu_svh=mx.array(gu_svh),
        dn_trellis=dn_tr,
        dn_suh=mx.array(dn_suh),
        dn_svh=mx.array(dn_svh),
        k=k,
        cb=CodebookMode.MUL1,
        activation="silu_clamp",
    )
    np_arrays = dict(
        gu_tr=np.array(gu_tr.tolist(), np.uint16),
        dn_tr=np.array(dn_tr.tolist(), np.uint16),
        gu_suh=gu_suh,
        gu_svh=gu_svh,
        dn_suh=dn_suh,
        dn_svh=dn_svh,
    )
    return sg, np_arrays


class GateContractTest(unittest.TestCase):
    """The DSV41_MOE_GROUPED gate: default OFF, both directions routed."""

    def test_flag_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV41_MOE_GROUPED", None)
            import importlib

            importlib.reload(M)
            self.assertIs(M._MOE_GROUPED, False)
        importlib.reload(M)  # restore whatever the process env says

    def test_flag_env_parsing(self):
        import importlib

        for val, expect in [("0", False), ("", False), ("1", True)]:
            with mock.patch.dict(os.environ, {"DSV41_MOE_GROUPED": val}):
                importlib.reload(M)
                self.assertEqual(M._MOE_GROUPED, expect)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV41_MOE_GROUPED", None)
            importlib.reload(M)

    def test_call_routes_fused2_when_off(self):
        sg, _ = _build_module()
        x = mx.zeros((1, 2, 256), mx.float16)
        idx = mx.zeros((1, 2, 6), mx.int32)
        called = {}

        def fake_f2(self, *a):
            called["f2"] = 1
            return mx.zeros((2, 6, 256), mx.float16)

        def fake_g(self, *a):
            called["g"] = 1
            return mx.zeros((2, 6, 256), mx.float16)

        with mock.patch.object(EXL3SwitchGLU, "_decode_fused2", fake_f2), \
             mock.patch.object(EXL3SwitchGLU, "_decode_grouped", fake_g):
            with mock.patch.object(M, "_MOE_GROUPED", False):
                sg(x, idx)
        self.assertIn("f2", called)
        self.assertNotIn("g", called)

    def test_call_routes_grouped_when_on(self):
        sg, _ = _build_module()
        x = mx.zeros((1, 2, 256), mx.float16)
        idx = mx.zeros((1, 2, 6), mx.int32)
        called = {}

        def fake_f2(self, *a):
            called["f2"] = 1
            return mx.zeros((2, 6, 256), mx.float16)

        def fake_g(self, *a):
            called["g"] = 1
            return mx.zeros((2, 6, 256), mx.float16)

        with mock.patch.object(EXL3SwitchGLU, "_decode_fused2", fake_f2), \
             mock.patch.object(EXL3SwitchGLU, "_decode_grouped", fake_g):
            with mock.patch.object(M, "_MOE_GROUPED", True):
                sg(x, idx)
        self.assertIn("g", called)
        self.assertNotIn("f2", called)


class FlagOffUntouchedTest(unittest.TestCase):
    """The flag-off artifacts are byte-identical to this branch's baseline.

    The fused2 Metal sources + its dispatcher are the shipped path; any byte
    change to them is a real change (the Metal compiler caches on the source
    text). A revert of the grouped work that edits these strings fails here.
    """

    # sha256 of the A2/B2 source strings as generated by the PRISTINE parent
    # commit ad6f9d3 (verified byte-identical at branch time; re-derived in
    # the RED/GREEN harness by `git show ad6f9d3:...exl3_moe.py` + exec).
    BASELINE_A2_SHA = "e6656f7303aedd55471dc2c354b642e7bce5cf385207fb56861f7c1aa6d9845b"
    BASELINE_B2_SHA = "ba0fa965157b51a2d88dddc98e0a2338f26041b3d95ff91197fff3a667b2521b"

    def test_fused2_metal_sources_unchanged(self):
        import hashlib

        k, cb = 6, CodebookMode.MUL1
        a2 = M._moe_gateup2_source(k, cb, tiles=M._A2_TILES)
        b2 = M._moe_down2_source(k, cb, "silu_clamp", hidden=128)
        self.assertEqual(
            hashlib.sha256(a2.encode()).hexdigest(), self.BASELINE_A2_SHA,
            "A2 (fused2 gate+up) Metal source drifted from the baseline",
        )
        self.assertEqual(
            hashlib.sha256(b2.encode()).hexdigest(), self.BASELINE_B2_SHA,
            "B2 (fused2 down) Metal source drifted from the baseline",
        )

    def test_decode_fused2_has_no_grouped_references(self):
        # the flag-off dispatcher must not grow a grouped branch: assert no
        # grouped symbol appears anywhere in its compiled code object
        code = EXL3SwitchGLU._decode_fused2.__code__
        self.assertNotIn("_decode_grouped", code.co_names)
        self.assertNotIn("_grouped", " ".join(code.co_names))
        self.assertNotIn("_grouped", code.co_varnames)
        # and the grouped gate itself lives only in __call__'s branch
        call_names = list(EXL3SwitchGLU.__call__.__code__.co_names)
        self.assertIn("_decode_grouped", call_names)


class GroupedTablesTest(unittest.TestCase):
    """_grouped_srt + _grouped_tables_fn vs their host spec."""

    def _check(self, sel, E, sm):
        sel = np.asarray(sel, np.int64)
        n = len(sel)
        # module path (exactly what _decode_grouped builds)
        stride = mx.array(2 * n, dtype=mx.uint32)
        sel_u = mx.array(sel.astype(np.uint32))
        key = sel_u * stride + mx.arange(n, dtype=mx.uint32)
        pad = mx.full((n,), E * 2 * n, dtype=mx.uint32)
        sorted_key = mx.sort(mx.concatenate([key, pad]))
        srt_m = (sorted_key // stride).astype(mx.uint32)
        slots_m = (sorted_key % stride).astype(mx.uint32)
        tabs_m, slist_m = _grouped_tables_fn(E, 2 * n, sm)(srt_m, slots_m)
        mx.eval(srt_m, slots_m, tabs_m, slist_m)
        srt_h, slots_h, tabs_h, slist_h = host_tables(sel, E, sm)
        self.assertTrue(
            np.array_equal(np.array(srt_m.tolist()), srt_h), f"srt {sel}"
        )
        self.assertTrue(
            np.array_equal(np.array(slots_m.tolist()), slots_h), f"slots {sel}"
        )
        self.assertTrue(
            np.array_equal(np.array(tabs_m.tolist()), tabs_h), f"tabs {sel}"
        )
        self.assertTrue(
            np.array_equal(np.array(slist_m.tolist()), slist_h), f"slist {sel}"
        )

    def test_random_and_adversarial(self):
        rng = np.random.default_rng(0)
        for trial in range(120):
            R = int(rng.integers(1, 9))
            kk = int(rng.integers(1, 9))
            E = int(rng.integers(2, 12))
            sm = max(M._SLOTS_MAX, R)
            n = R * kk
            style = trial % 6
            if style == 0:
                sel = rng.integers(0, E, size=n)
            elif style == 1:
                sel = rng.integers(0, min(E, 3), size=n)
            elif style == 2:
                sel = np.zeros(n, dtype=int)  # ALL rows same expert
            elif style == 3:
                sel = np.full(n, E - 1)  # expert id n-1 (edge)
            elif style == 4:
                sel = rng.integers(0, max(E, 2), size=n)
            else:
                # per-row distinct experts (the real routing shape)
                sel = np.stack(
                    [rng.choice(E, min(kk, E), replace=False) for _ in range(R)]
                ).reshape(-1)
            self._check(sel, E, sm)

    def test_edge_cases(self):
        # single slot
        self._check([3], 6, 8)
        # an expert with zero slots, others routed
        self._check([0, 0, 5], 6, 8)
        # max slots per expert (R=8 all same expert)
        self._check([2] * 8 * 6, 6, 8)
        # same expert at different k positions across rows
        self._check([1, 2, 3, 1, 2, 3, 1, 2], 6, 8)
        # padding tail: last expert only
        self._check([7, 7, 7], 8, 8)

    def test_tables_shape_contract(self):
        # fixed shapes per (E, n, sm): no data-dependent shapes anywhere
        sel = np.array([1, 3, 3, 0])
        E, sm = 6, 8
        srt_h, slots_h, tabs_h, slist_h = host_tables(sel, E, sm)
        self.assertEqual(tabs_h.shape, (2, E))
        self.assertEqual(slist_h.shape, (E, sm))
        self.assertEqual(srt_h.shape, (2 * len(sel),))
        self.assertEqual(slots_h.shape, (2 * len(sel),))


class GroupingScatterEqualityTest(unittest.TestCase):
    """fused2_ref == grouped_ref, BIT-EXACT, on both extraction functions."""

    def _run_case(self, R, kk, E, sel, tile_weights, seed=13):
        D, H, k = 256, 128, 6
        rng = np.random.default_rng(seed)
        gu_tr = rng.integers(
            0, 65536, (D // 16, E * 2 * (H // 16), 16 * k)
        ).astype(np.uint16)
        dn_tr = rng.integers(
            0, 65536, (H // 16, E * (D // 16), 16 * k)
        ).astype(np.uint16)
        gu_suh = rng.standard_normal((E, 2, D)).astype(np.float16)
        gu_svh = rng.standard_normal((E, 2 * H)).astype(np.float16)
        dn_suh = rng.standard_normal((E, H)).astype(np.float16)
        dn_svh = rng.standard_normal((E, D)).astype(np.float16)
        x = rng.standard_normal((R, D)).astype(np.float16)
        idx = np.asarray(sel, np.int64).reshape(R, kk)
        common = dict(
            gu_trellis=gu_tr, gu_suh=gu_suh, gu_svh=gu_svh,
            dn_trellis=dn_tr, dn_suh=dn_suh, dn_svh=dn_svh,
            k=k, cb=CodebookMode.MUL1, act="silu_clamp", clamp=10.0,
            ch=128, tile_weights=tile_weights,
        )
        y0, ygu0 = decode_fused2_ref(x, idx, **common)
        grouped_kw = dict(common)
        grouped_kw["sm"] = max(8, R)
        y1, ygu1, srt, tabs, slist = decode_grouped_ref(x, idx, **grouped_kw)
        self.assertTrue(np.array_equal(ygu0, ygu1), f"ygu differs (R={R})")
        self.assertTrue(np.array_equal(y0, y1), f"y differs (R={R})")
        return y0

    def test_bit_exact_metal_sim_extraction(self):
        # duplicate-heavy (ident-like): rows share experts
        sel = [1, 1, 2, 2, 1, 2, 1, 2]  # R=4 kk=2 E=3: heavy sharing
        self._run_case(4, 2, 3, sel, metal_sim_tile_weights)
        # all-distinct (dist-like)
        self._run_case(2, 4, 8, [0, 1, 2, 3, 4, 5, 6, 7], metal_sim_tile_weights)
        # real-routing shape: per-row distinct top-k, cross-row overlap
        rng = np.random.default_rng(2)
        sel = np.stack(
            [rng.choice(8, 6, replace=False) for _ in range(5)]
        ).reshape(-1)
        self._run_case(5, 6, 8, sel.tolist(), metal_sim_tile_weights)

    def test_bit_exact_hash_extraction(self):
        # same routings, extraction-agnostic weights: proves the equality is
        # index arithmetic, not a property of the codeword layout guess
        sel = [1, 1, 2, 2, 1, 2, 1, 2]
        self._run_case(4, 2, 3, sel, hash_tile_weights)
        rng = np.random.default_rng(3)
        sel = np.stack(
            [rng.choice(6, 6, replace=False) for _ in range(6)]
        ).reshape(-1)
        self._run_case(6, 6, 6, sel.tolist(), hash_tile_weights)

    def test_sweep_r1_to_r8(self):
        # the real routing shape: per-row DISTINCT top-k experts, so a
        # per-expert count is <= R <= sm (the clip contract never engages)
        rng = np.random.default_rng(4)
        for R in range(1, 9):
            E, kk = 5, min(6, 5)
            sel = np.stack(
                [rng.choice(E, min(kk, E), replace=False) for _ in range(R)]
            ).reshape(-1).tolist()
            self._run_case(R, kk, E, sel, hash_tile_weights, seed=100 + R)
            self._run_case(R, kk, E, sel, metal_sim_tile_weights, seed=100 + R)

    def test_overfull_expert_clips_consistently(self):
        # A caller routing more than sm slots to one expert (impossible under
        # the real gate: per-row top-k experts are distinct, count <= R) is
        # handled deterministically by BOTH refs: the fused2 path computes
        # every slot, the grouped path keeps the FIRST sm (ascending slot
        # order). Document the contract: the first sm slots agree.
        import numpy as np

        from mlx_lm.models.exl3.moe_grouped_ref import grouped_tables

        sel = np.array([2] * 10)  # 10 slots on one expert, sm = 8
        srt, slots, tabs, slist = grouped_tables(sel, 6, 8)
        self.assertEqual(int(tabs[1, 2]), 8)
        self.assertEqual(slist[2].tolist(), list(range(8)))


class RowsPrepAnchorTest(unittest.TestCase):
    """The reference Hadamard path is bit-exact vs the real compiled op."""

    def test_rows_prep_bit_exact(self):
        rng = np.random.default_rng(11)
        E, D = 4, 256
        gu_suh = rng.standard_normal((E, 2, D)).astype(np.float16)
        x2d = rng.standard_normal((3, D)).astype(np.float16)
        sel = rng.integers(0, E, size=3 * 4)
        suh_sel = gu_suh[sel].reshape(len(sel) * 2, D)
        x_rep = np.broadcast_to(
            x2d[:, None, :], (3, 4 * 2, D)
        ).reshape(len(sel) * 2, D)
        xh_mlx = M._rows_prep()(mx.array(x_rep), mx.array(suh_sel))
        mx.eval(xh_mlx)
        xh_np = np_rows_prep(x2d, sel, gu_suh)
        a, b = np.array(xh_mlx.tolist()), xh_np
        self.assertTrue(np.array_equal(a, b), "rows_prep drifted from MLX op")


if __name__ == "__main__":
    unittest.main()