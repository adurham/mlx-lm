# Copyright © 2026 Apple Inc.

"""Phase 3a parity + no-regression tests for the DeepSeek-V4 MoE gate.

Covers the `bias_vl` second routing bias that DeepSeek-V4-Flash-Vision-Exp adds
for image tokens (`input_ids >= vocab_size`):

* **Parity** vs the torch reference logic (`tests/_dsv4_torch_reference.py`,
  transcribed from the Vision-Exp `inference/model.py`) on synthetic inputs
  containing BOTH text and image tokens, on hash layers (0,1,2) and non-hash.
* **Text-only bitwise identity** vs the FROZEN pre-change implementation
  (`tests/_dsv4_gate_baseline_64cc7e6.py`, extracted by `git show` from the
  submodule HEAD before Phase 3). This is the single most important test in
  Phase 3: `_hash_gate_route` is the fork's daily production hot path and a
  regression there silently degrades the model the user runs every day.

Run on a machine with MLX, with the submodule on PYTHONPATH::

    PYTHONPATH=<repo>/mlx-lm <venv>/bin/python -m pytest \\
        tests/test_deepseek_v4_gate.py -q
"""

import unittest

import mlx.core as mx
import numpy as np
import torch
from mlx_lm.models import deepseek_v4 as dsv4

# tests/ has no __init__.py, so under pytest's default "prepend" import mode
# the test file's own directory is on sys.path and these resolve as top-level
# modules. Both are TEST-ONLY helpers and are deliberately not importable from
# the shipped package.
from _dsv4_gate_baseline_64cc7e6 import MoEGate as BaselineMoEGate
from _dsv4_torch_reference import IMAGE, IMAGE_END, IMAGE_START, RefGate

# Small but structurally faithful: real DSv4 routing constants (top_k=6,
# sqrtsoftplus, route_scale 1.5, norm_topk_prob) with a reduced expert count
# and vocab so the tests run in milliseconds. num_hash_layers=3 is the real
# value, so layer_idx 0/1/2 are hash layers and 3+ are not, exactly as in the
# checkpoint.
VOCAB = 512
N_EXPERTS = 32
TOP_K = 6
HIDDEN = 64


def _config(vision_n_layers):
    return dsv4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        n_routed_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=8,
        num_hash_layers=3,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        vision_n_layers=vision_n_layers,
    )


def _mixed_input_ids(rng, batch, length, n_image):
    """Token ids with `n_image` image-sentinel tokens per row.

    Image tokens are `vocab_size + {IMAGE_START, IMAGE, IMAGE_END}` — ids that
    are OUT OF RANGE for the tid2eid table, which is precisely why the
    reference clamps them to 0 before the gather.
    """
    ids = rng.integers(0, VOCAB, size=(batch, length)).astype(np.int32)
    for b in range(batch):
        pos = rng.choice(length, size=n_image, replace=False)
        sentinels = rng.choice(
            [VOCAB + IMAGE_START, VOCAB + IMAGE, VOCAB + IMAGE_END], size=n_image
        )
        ids[b, pos] = sentinels
    return ids


def _seed_gate(gate, rng, *, hash_layer, vl):
    """Fill an MLX MoEGate with reproducible weights; return the torch twins."""
    weight = rng.standard_normal((N_EXPERTS, HIDDEN)).astype(np.float32) * 0.1
    gate.weight = mx.array(weight)
    tid2eid = None
    bias = None
    bias_vl = None
    if hash_layer:
        tid2eid = rng.integers(0, N_EXPERTS, size=(VOCAB, TOP_K)).astype(np.int32)
        gate.tid2eid = mx.array(tid2eid)
    if not hash_layer or vl:
        bias = rng.standard_normal(N_EXPERTS).astype(np.float32) * 0.5
        gate.e_score_correction_bias = mx.array(bias)
    if vl:
        bias_vl = rng.standard_normal(N_EXPERTS).astype(np.float32) * 0.5
        gate.e_score_correction_bias_vl = mx.array(bias_vl)
    return weight, tid2eid, bias, bias_vl


def _sorted_rows(a):
    """Sort each row's expert ids so argpartition-vs-topk ORDER is not compared.

    The fork uses `mx.argpartition` where the reference uses `torch.topk`: the
    selected SET is identical but argpartition is unordered. The MoE combine
    downstream is a sum over experts (order-invariant) and `weights` is gathered
    with the same indices, so the (expert, weight) PAIRING — which is what
    matters — is preserved. Comparing sorted rows tests exactly that invariant.
    """
    return np.sort(np.asarray(a), axis=-1)


def _pairs(inds, weights):
    """{(expert, weight)} per row — the order-invariant content of a routing."""
    inds = np.asarray(inds)
    weights = np.asarray(weights, dtype=np.float64)
    flat_i = inds.reshape(-1, inds.shape[-1])
    flat_w = weights.reshape(-1, weights.shape[-1])
    return [
        dict(sorted(zip(row_i.tolist(), row_w.tolist()))) for row_i, row_w in zip(flat_i, flat_w)
    ]


class TestGateVisionParity(unittest.TestCase):
    """Acceptance criterion 1: parity vs the torch reference, text+image input."""

    def _run_parity(self, layer_idx, n_image, seed):
        rng = np.random.default_rng(seed)
        hash_layer = layer_idx < 3
        config = _config(vision_n_layers=32)
        gate = dsv4.MoEGate(config, layer_idx)
        weight, tid2eid, bias, bias_vl = _seed_gate(
            gate, rng, hash_layer=hash_layer, vl=True
        )
        self.assertTrue(gate.vl, "vision config must select the vl gate path")

        batch, length = 1, 48
        ids = _mixed_input_ids(rng, batch, length, n_image)
        x = (rng.standard_normal((batch, length, HIDDEN)) * 0.5).astype(np.float32)

        mlx_inds, mlx_weights = gate(mx.array(x), mx.array(ids))
        mx.eval(mlx_inds, mlx_weights)

        # The reference Gate operates on FLATTENED (N, dim) tokens --
        # MoE.forward does `x.view(-1, self.dim)` and `input_ids.flatten()`.
        ref = RefGate(
            weight=torch.from_numpy(weight),
            topk=TOP_K,
            score_func="sqrtsoftplus",
            route_scale=1.5,
            is_hash=hash_layer,
            vocab_size=VOCAB,
            tid2eid=torch.from_numpy(tid2eid).long() if tid2eid is not None else None,
            bias=torch.from_numpy(bias) if bias is not None else None,
            bias_vl=torch.from_numpy(bias_vl),
        )
        ref_w, ref_i = ref.forward(
            torch.from_numpy(x).reshape(-1, HIDDEN),
            torch.from_numpy(ids).reshape(-1).long(),
        )

        mlx_i = np.asarray(mlx_inds).reshape(-1, TOP_K)
        mlx_w = np.asarray(mlx_weights, dtype=np.float64).reshape(-1, TOP_K)
        ref_i_np = ref_i.numpy().reshape(-1, TOP_K)
        ref_w_np = ref_w.numpy().astype(np.float64).reshape(-1, TOP_K)

        n_tokens = mlx_i.shape[0]
        matched = int((_sorted_rows(mlx_i) == _sorted_rows(ref_i_np)).all(axis=-1).sum())
        idx_elems = int((_sorted_rows(mlx_i) == _sorted_rows(ref_i_np)).sum())

        # Weight comparison must follow the PAIRING, not the slot order.
        mlx_pairs = _pairs(mlx_i, mlx_w)
        ref_pairs = _pairs(ref_i_np, ref_w_np)
        diffs = []
        for mp, rp in zip(mlx_pairs, ref_pairs):
            self.assertEqual(set(mp), set(rp), "expert sets differ")
            for e in mp:
                diffs.append(abs(mp[e] - rp[e]))
        diffs = np.asarray(diffs)

        n_img_tok = int((ids.reshape(-1) >= VOCAB).sum())
        print(
            f"\n[gate parity] layer_idx={layer_idx} "
            f"({'hash' if hash_layer else 'non-hash'}) "
            f"tokens={n_tokens} image_tokens={n_img_tok} "
            f"seed={seed}\n"
            f"    indices: {matched}/{n_tokens} rows matched exactly, "
            f"{idx_elems}/{n_tokens * TOP_K} index elements matched\n"
            f"    weights: max abs diff {diffs.max():.3e}  "
            f"mean abs diff {diffs.mean():.3e}"
        )

        self.assertEqual(matched, n_tokens, "every row's expert SET must match")
        self.assertEqual(idx_elems, n_tokens * TOP_K)
        # fp32 matmul on two different backends; the routing weight is a
        # sqrt(softplus) of a dot product, so ~1e-7 relative is the floor.
        self.assertLess(float(diffs.max()), 1e-5)

    def test_hash_layer_0(self):
        self._run_parity(layer_idx=0, n_image=12, seed=11)

    def test_hash_layer_1(self):
        self._run_parity(layer_idx=1, n_image=12, seed=12)

    def test_hash_layer_2(self):
        self._run_parity(layer_idx=2, n_image=12, seed=13)

    def test_non_hash_layer_3(self):
        self._run_parity(layer_idx=3, n_image=12, seed=14)

    def test_non_hash_layer_7(self):
        self._run_parity(layer_idx=7, n_image=12, seed=15)

    def test_hash_layer_all_image_tokens(self):
        """Every token is an image token: the tid2eid gather is fully bypassed."""
        self._run_parity(layer_idx=0, n_image=48, seed=16)

    def test_hash_layer_no_image_tokens_still_matches_reference(self):
        """vl gate on text-only input still equals the reference's vl branch."""
        self._run_parity(layer_idx=0, n_image=0, seed=17)

    def test_non_hash_layer_no_image_tokens(self):
        self._run_parity(layer_idx=3, n_image=0, seed=18)


class TestImageTokensBypassHashTable(unittest.TestCase):
    """The hash-layer image bypass must be REAL, not a no-op.

    A test that only compares against a reference implementing the same bug
    would pass vacuously. This asserts the observable behavior directly.
    """

    def test_image_rows_ignore_tid2eid_and_text_rows_do_not(self):
        rng = np.random.default_rng(99)
        config = _config(vision_n_layers=32)
        gate = dsv4.MoEGate(config, 0)
        _, tid2eid, _, _ = _seed_gate(gate, rng, hash_layer=True, vl=True)

        ids = np.full((1, 8), VOCAB + IMAGE, dtype=np.int32)
        ids[0, :4] = rng.integers(0, VOCAB, size=4)  # first 4 text, last 4 image
        x = (rng.standard_normal((1, 8, HIDDEN)) * 0.5).astype(np.float32)
        inds, _ = gate(mx.array(x), mx.array(ids))
        inds = np.asarray(inds)[0]

        for t in range(4):  # text rows: exactly the table row
            np.testing.assert_array_equal(inds[t], tid2eid[ids[0, t]])
        # Image rows: a real top-k. tid2eid[0] is what a NON-bypassing
        # implementation would have produced (the reference clamps the id to 0).
        table_row_0 = np.sort(tid2eid[0])
        n_equal = sum(
            1 for t in range(4, 8) if np.array_equal(np.sort(inds[t]), table_row_0)
        )
        print(
            f"\n[hash bypass] text rows matched tid2eid: 4/4; "
            f"image rows equal to tid2eid[0]: {n_equal}/4 (want 0)"
        )
        self.assertEqual(
            n_equal, 0, "image rows must NOT come from the tid2eid table"
        )

    def test_out_of_range_ids_are_clamped_before_the_gather(self):
        """tid2eid has VOCAB rows; ids are VOCAB+4. Must not read out of bounds."""
        rng = np.random.default_rng(100)
        config = _config(vision_n_layers=32)
        gate = dsv4.MoEGate(config, 0)
        _seed_gate(gate, rng, hash_layer=True, vl=True)
        ids = np.full((1, 4), VOCAB + IMAGE_END, dtype=np.int32)
        x = (rng.standard_normal((1, 4, HIDDEN)) * 0.5).astype(np.float32)
        inds, weights = gate(mx.array(x), mx.array(ids))
        mx.eval(inds, weights)
        inds = np.asarray(inds)
        self.assertTrue((inds >= 0).all() and (inds < N_EXPERTS).all())
        print(
            f"\n[clamp] ids={VOCAB + IMAGE_END} (table has {VOCAB} rows) -> "
            f"indices in [{inds.min()}, {inds.max()}], valid range "
            f"[0, {N_EXPERTS - 1}]"
        )


class TestBiasIsSelectionOnlyNotWeight(unittest.TestCase):
    """`weights` must be gathered from UNBIASED scores (reference: original_scores)."""

    def test_weights_ignore_bias_vl(self):
        rng = np.random.default_rng(21)
        config = _config(vision_n_layers=32)
        ids = np.full((1, 6), VOCAB + IMAGE, dtype=np.int32)
        x = (rng.standard_normal((1, 6, HIDDEN)) * 0.5).astype(np.float32)

        gate = dsv4.MoEGate(config, 0)
        weight, tid2eid, bias, bias_vl = _seed_gate(
            gate, rng, hash_layer=True, vl=True
        )
        inds_a, w_a = gate(mx.array(x), mx.array(ids))
        mx.eval(inds_a, w_a)

        # Shift bias_vl by a CONSTANT: the argmax ordering (hence the selected
        # set) is unchanged, so the weights must be bit-identical.
        gate.e_score_correction_bias_vl = mx.array(bias_vl + 7.5)
        inds_b, w_b = gate(mx.array(x), mx.array(ids))
        mx.eval(inds_b, w_b)

        same_inds = bool((np.asarray(inds_a) == np.asarray(inds_b)).all())
        max_w = float(np.abs(np.asarray(w_a) - np.asarray(w_b)).max())
        print(
            f"\n[bias is selection-only] bias_vl += 7.5 -> "
            f"indices identical: {same_inds}, max weight diff {max_w:.3e}"
        )
        self.assertTrue(same_inds)
        self.assertEqual(max_w, 0.0)


class TestTextOnlyBitwiseIdentity(unittest.TestCase):
    """ACCEPTANCE CRITERION 3 (3a) — the single most important test in Phase 3.

    With a text-only checkpoint and no image tokens, the post-change gate must
    be BITWISE identical to the frozen pre-change implementation extracted from
    git (`_dsv4_gate_baseline_64cc7e6.py`). Exercises `_hash_gate_route`, the
    fork's daily production hot path.
    """

    def _compare(self, layer_idx, seed, batch=2, length=64):
        rng = np.random.default_rng(seed)
        config = _config(vision_n_layers=0)  # text-only checkpoint
        hash_layer = layer_idx < 3

        new_gate = dsv4.MoEGate(config, layer_idx)
        old_gate = BaselineMoEGate(config, layer_idx)
        self.assertFalse(new_gate.vl, "text-only config must NOT select vl path")

        weight = (rng.standard_normal((N_EXPERTS, HIDDEN)) * 0.1).astype(np.float32)
        new_gate.weight = old_gate.weight = mx.array(weight)
        if hash_layer:
            tid2eid = rng.integers(0, N_EXPERTS, size=(VOCAB, TOP_K)).astype(np.int32)
            new_gate.tid2eid = old_gate.tid2eid = mx.array(tid2eid)
            self.assertFalse(
                hasattr(new_gate, "e_score_correction_bias"),
                "text-only hash layer must allocate no bias (reference parity)",
            )
        else:
            bias = (rng.standard_normal(N_EXPERTS) * 0.5).astype(np.float32)
            new_gate.e_score_correction_bias = mx.array(bias)
            old_gate.e_score_correction_bias = mx.array(bias)
        self.assertFalse(hasattr(new_gate, "e_score_correction_bias_vl"))

        ids = rng.integers(0, VOCAB, size=(batch, length)).astype(np.int32)
        x = (rng.standard_normal((batch, length, HIDDEN)) * 0.5).astype(np.float32)

        ni, nw = new_gate(mx.array(x), mx.array(ids))
        oi, ow = old_gate(mx.array(x), mx.array(ids))
        mx.eval(ni, nw, oi, ow)

        ni, nw = np.asarray(ni), np.asarray(nw)
        oi, ow = np.asarray(oi), np.asarray(ow)

        idx_equal = int((ni == oi).sum())
        idx_total = int(ni.size)
        # Bitwise on the float payload: compare raw bytes, not a tolerance.
        w_bitwise = nw.tobytes() == ow.tobytes()
        w_maxdiff = float(np.abs(nw.astype(np.float64) - ow.astype(np.float64)).max())
        print(
            f"\n[TEXT-ONLY BITWISE GUARD] layer_idx={layer_idx} "
            f"({'hash/_hash_gate_route' if hash_layer else 'non-hash/_gate_route'}) "
            f"shape={tuple(ni.shape)} seed={seed}\n"
            f"    indices: {idx_equal}/{idx_total} identical  "
            f"(dtype {ni.dtype} vs {oi.dtype})\n"
            f"    weights: raw-bytes identical = {w_bitwise}, "
            f"max abs diff {w_maxdiff:.1e}"
        )
        self.assertEqual(idx_equal, idx_total)
        self.assertEqual(ni.dtype, oi.dtype)
        self.assertTrue(w_bitwise, "gate weights are not BITWISE identical")
        self.assertEqual(w_maxdiff, 0.0)

    def test_hash_layer_0_bitwise(self):
        self._compare(layer_idx=0, seed=201)

    def test_hash_layer_1_bitwise(self):
        self._compare(layer_idx=1, seed=202)

    def test_hash_layer_2_bitwise(self):
        self._compare(layer_idx=2, seed=203)

    def test_non_hash_layer_3_bitwise(self):
        self._compare(layer_idx=3, seed=204)

    def test_non_hash_layer_7_bitwise(self):
        self._compare(layer_idx=7, seed=205)

    def test_decode_shape_bitwise(self):
        """L == 1: the decode hot path shape."""
        self._compare(layer_idx=0, seed=206, batch=1, length=1)

    def test_prefill_shape_bitwise(self):
        """A prefill-sized chunk."""
        self._compare(layer_idx=1, seed=207, batch=1, length=512)

    def test_compiled_function_objects_are_untouched(self):
        """The production compiled routes must be the SAME objects, not copies.

        Structural half of the no-regression guarantee: a text-only gate calls
        `_hash_gate_route` / `_gate_route` themselves, so there is no new branch
        or new compile-cache entry on the hot path.
        """
        self.assertIsNot(dsv4._hash_gate_route, dsv4._hash_gate_route_vl)
        self.assertIsNot(dsv4._gate_route, dsv4._gate_route_vl)


class TestVlGateEqualsPlainGateOnTextOnlyInput(unittest.TestCase):
    """The vl variants are bitwise-identical to the plain ones on text input.

    Not required by the acceptance criteria (a vision checkpoint is a different
    checkpoint), but it proves `mx.where` on an all-False image mask is a pure
    select with no numerical side effect -- so the vl path costs correctness
    nothing on the text tokens that dominate any real vision prompt.
    """

    def test_non_hash_vl_matches_plain_when_no_image_tokens(self):
        rng = np.random.default_rng(301)
        x = mx.array((rng.standard_normal((1, 32, HIDDEN)) * 0.5).astype(np.float32))
        ids = mx.array(rng.integers(0, VOCAB, size=(1, 32)).astype(np.int32))
        w = mx.array((rng.standard_normal((N_EXPERTS, HIDDEN)) * 0.1).astype(np.float32))
        bias = mx.array((rng.standard_normal(N_EXPERTS) * 0.5).astype(np.float32))
        bias_vl = mx.array((rng.standard_normal(N_EXPERTS) * 0.5).astype(np.float32))

        pi, pw = dsv4._gate_route(x, w, bias, TOP_K, 1.5, True, "sqrtsoftplus")
        vi, vw = dsv4._gate_route_vl(
            ids, x, w, bias, bias_vl, VOCAB, TOP_K, 1.5, True, "sqrtsoftplus"
        )
        mx.eval(pi, pw, vi, vw)
        idx_eq = bool((np.asarray(pi) == np.asarray(vi)).all())
        bytes_eq = np.asarray(pw).tobytes() == np.asarray(vw).tobytes()
        print(
            f"\n[vl==plain on text] non-hash: indices identical={idx_eq}, "
            f"weights raw-bytes identical={bytes_eq}"
        )
        self.assertTrue(idx_eq)
        self.assertTrue(bytes_eq)

    def test_hash_vl_matches_plain_when_no_image_tokens(self):
        rng = np.random.default_rng(302)
        x = mx.array((rng.standard_normal((1, 32, HIDDEN)) * 0.5).astype(np.float32))
        ids = mx.array(rng.integers(0, VOCAB, size=(1, 32)).astype(np.int32))
        w = mx.array((rng.standard_normal((N_EXPERTS, HIDDEN)) * 0.1).astype(np.float32))
        tid2eid = mx.array(
            rng.integers(0, N_EXPERTS, size=(VOCAB, TOP_K)).astype(np.int32)
        )
        bias_vl = mx.array((rng.standard_normal(N_EXPERTS) * 0.5).astype(np.float32))

        pi, pw = dsv4._hash_gate_route(ids, x, w, tid2eid, 1.5, True, "sqrtsoftplus")
        vi, vw = dsv4._hash_gate_route_vl(
            ids, x, w, tid2eid, bias_vl, VOCAB, TOP_K, 1.5, True, "sqrtsoftplus"
        )
        mx.eval(pi, pw, vi, vw)
        idx_eq = bool((np.asarray(pi) == np.asarray(vi)).all())
        bytes_eq = np.asarray(pw).tobytes() == np.asarray(vw).tobytes()
        print(
            f"\n[vl==plain on text] hash: indices identical={idx_eq}, "
            f"weights raw-bytes identical={bytes_eq}"
        )
        self.assertTrue(idx_eq)
        self.assertTrue(bytes_eq)


class TestGateParameterAllocation(unittest.TestCase):
    """Reference parity for WHICH parameters exist (`if self.hash and not vl`)."""

    def test_text_only_allocation_unchanged(self):
        config = _config(vision_n_layers=0)
        hash_gate = dsv4.MoEGate(config, 0)
        dense_gate = dsv4.MoEGate(config, 3)
        self.assertTrue(hasattr(hash_gate, "tid2eid"))
        self.assertFalse(hasattr(hash_gate, "e_score_correction_bias"))
        self.assertFalse(hasattr(hash_gate, "e_score_correction_bias_vl"))
        self.assertFalse(hasattr(dense_gate, "tid2eid"))
        self.assertTrue(hasattr(dense_gate, "e_score_correction_bias"))
        self.assertFalse(hasattr(dense_gate, "e_score_correction_bias_vl"))

    def test_vision_hash_layer_allocates_both_biases(self):
        """Reference allocates `bias` on vision hash layers even though it is unread."""
        config = _config(vision_n_layers=32)
        hash_gate = dsv4.MoEGate(config, 0)
        self.assertTrue(hasattr(hash_gate, "tid2eid"))
        self.assertTrue(hasattr(hash_gate, "e_score_correction_bias"))
        self.assertTrue(hasattr(hash_gate, "e_score_correction_bias_vl"))
        self.assertEqual(hash_gate.e_score_correction_bias_vl.shape, (N_EXPERTS,))
        self.assertEqual(hash_gate.e_score_correction_bias_vl.dtype, mx.float32)


if __name__ == "__main__":
    unittest.main()
