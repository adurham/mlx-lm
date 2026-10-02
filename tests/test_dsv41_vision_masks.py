# Copyright © 2026 Adam Durham (hermes-gw)
"""Image-row semantics of the DSv4.1 body, pinned to the reference.

The reference ``Transformer.forward`` (release ``inference/model.py``) treats an
image span differently from text in exactly three places:

1. ``image_mask = token_types >= 0`` picks ``gate.bias_vl`` instead of
   ``gate.bias`` for the expert top-k of every image row (``Gate.forward``);
2. ``engram_mask = ~image_mask`` caches image positions as ``DEAD`` in the
   n-gram hasher, so no n-gram ever spans an image row (``NgramHashState``);
3. the same mask shuts the engram gate on image rows (``Engram.forward``), so
   those rows get no engram write.

The MLX port had none of the three (it was written as a text-only runtime and
the vision splice only replaced the embeddings), and the served EXL3 quant also
dropped all 43 ``gate.bias_vl`` tensors. Every test here fails on the pre-fix
code (missing args / attributes or wrong numbers), and the text path (no mask)
is checked to stay bit-identical.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_vision_masks.py -v
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_lm.models.deepseek_v41 import engram as E
from mlx_lm.models.deepseek_v41 import model as M
from mlx_lm.models.deepseek_v41 import moe as MO


def _hasher(n_compressed=50, layers=(1,), ngram=4, heads=2, vocab=97):
    """An EngramHasher over a tiny synthetic layout (identity-ish token map)."""
    h = E.EngramHasher.__new__(E.EngramHasher)
    h.max_ngram = ngram
    h.token_map = np.arange(n_compressed, dtype=np.int64)
    h.pad_id = 2
    rng = np.random.default_rng(0)
    h.primes = np.array([[[vocab + 2 * i + 2 * j * heads for j in range(heads)]
                          for i in range(ngram - 1)] for _ in layers], dtype=np.int64)
    h.offsets = np.zeros((len(layers), (ngram - 1) * heads), dtype=np.int64)
    h.multipliers = (rng.integers(1, 1 << 20, size=(len(layers), ngram)) * 2 + 1).astype(np.int64)
    return h


def _reference_hash(h, ids, mask):
    """Straight port of the reference NgramHashState.forward (torch -> numpy)."""
    ids = np.asarray(ids)
    b, s = ids.shape
    comp = h.token_map[ids]
    comp = np.where(mask, comp, -1)
    pos = np.arange(s)[None].repeat(b, 0)
    toks, blocked = [], np.zeros_like(pos, dtype=bool)
    for shift in range(h.max_ngram):
        src = np.take_along_axis(comp, np.clip(pos - shift, 0, None), axis=1)
        blocked = blocked | (pos < shift) | (src == -1)
        toks.append(np.where(blocked, h.pad_id, src))
    toks = np.stack(toks, -1)
    prod = toks[:, :, None, :] * h.multipliers
    rolling, out = prod[..., 0], []
    for i in range(1, h.max_ngram):
        rolling = np.bitwise_xor(rolling, prod[..., i])
        out.append(rolling[..., None] % h.primes[:, i - 1])
    return np.concatenate(out, -1) + h.offsets


class HasherImageSpan(unittest.TestCase):
    def setUp(self):
        self.h = _hasher()
        rng = np.random.default_rng(1)
        self.ids = rng.integers(3, 50, size=(1, 40))
        self.mask = np.ones((1, 40), dtype=bool)
        self.mask[:, 10:22] = False                    # an image span

    def test_matches_reference_one_shot(self):
        got = self.h(self.ids, 0, np.zeros((1, 64), np.int64), self.mask)
        np.testing.assert_array_equal(got, _reference_hash(self.h, self.ids, self.mask))

    def test_matches_reference_across_chunks_and_decode(self):
        cache = np.zeros((1, 64), np.int64)
        parts = [self.h(self.ids[:, :25], 0, cache, self.mask[:, :25]),
                 self.h(self.ids[:, 25:30], 25, cache, None)]   # later piece / text: no mask
        for p in range(30, 40):                                 # decode rows
            parts.append(self.h(self.ids[:, p:p + 1], p, cache, None))
        np.testing.assert_array_equal(np.concatenate(parts, 1),
                                      _reference_hash(self.h, self.ids, self.mask))

    def test_text_rows_after_span_do_not_hash_image_ids(self):
        # position 22 (first text row after the span) must see pad for every
        # look-back into the span, i.e. equal to a 1-gram-only context
        got = self.h(self.ids, 0, np.zeros((1, 64), np.int64), self.mask)
        legacy = self.h(self.ids, 0, np.zeros((1, 64), np.int64), None)
        self.assertFalse(np.array_equal(got[:, 22], legacy[:, 22]))

    def test_no_mask_is_bit_identical_to_text_path(self):
        a = self.h(self.ids, 0, np.zeros((1, 64), np.int64), None)
        b = self.h(self.ids, 0, np.zeros((1, 64), np.int64), np.ones((1, 40), bool))
        np.testing.assert_array_equal(a, b)


class EngramGateMask(unittest.TestCase):
    def _engram(self, dim=16, hc=2, cols=6, hd=4):
        args = SimpleNamespace(dim=dim, hc_mult=hc, norm_eps=1e-6, engram_max_ngram_size=4,
                               engram_n_heads=2, engram_head_dim=hd,
                               engram_num_embeddings=(200,))
        e = E.Engram(args, 0)
        rng = np.random.default_rng(2)
        e.embed = E.EngramEmbedding(200, hd)
        e.embed.weight = mx.array(rng.standard_normal((200, hd)).astype(np.float32))
        e.embed.scale = mx.ones((200, 1), dtype=mx.float32)
        e.embed.block = hd
        e.wkv.weight = mx.array(rng.standard_normal(((hc + 1) * dim, cols * hd)).astype(np.float32) * 0.1)
        return e, args

    def test_masked_rows_pass_through_and_text_rows_unchanged(self):
        e, a = self._engram()
        rng = np.random.default_rng(3)
        x = mx.array(rng.standard_normal((1, 9, a.hc_mult, a.dim)).astype(np.float32))
        ids = mx.array(rng.integers(0, 200, size=(1, 9, 6)))
        mask = np.ones((1, 9), bool)
        mask[:, 2:6] = False
        out_m = np.array(e(x, ids, mx.array(mask)))
        out_t = np.array(e(x, ids))
        np.testing.assert_array_equal(out_m[:, 2:6], np.array(x)[:, 2:6])     # untouched
        np.testing.assert_array_equal(out_m[:, mask[0]], out_t[:, mask[0]])   # text rows identical
        self.assertFalse(np.array_equal(out_t[:, 2:6], np.array(x)[:, 2:6]))  # the write is real


class GateVlBias(unittest.TestCase):
    def _gate(self, n=16, k=3, dim=8):
        args = SimpleNamespace(n_routed_experts=n, n_activated_experts=k, score_func="sqrtsoftplus",
                               gate_temp=1.0, norm_topk_prob=True, route_scale=1.5, dim=dim)
        g = MO.Gate(args)
        rng = np.random.default_rng(4)
        g.weight = mx.array(rng.standard_normal((n, dim)).astype(np.float32))
        g.bias = mx.array(rng.standard_normal(n).astype(np.float32) * 0.01)
        vl = np.zeros(n, np.float32)
        vl[[1, 5, 9]] = 50.0                                  # forces experts 1,5,9 on image rows
        g.bias_vl = mx.array(vl)
        return g

    def test_image_rows_route_with_bias_vl_text_rows_with_bias(self):
        g = self._gate()
        x = mx.array(np.random.default_rng(5).standard_normal((6, 8)).astype(np.float32))
        mask = mx.array(np.array([False, True, True, False, True, False]))
        w_m, i_m = g(x, mask)
        w_t, i_t = g(x)
        im, it = np.sort(np.array(i_m), -1), np.sort(np.array(i_t), -1)
        for r in (1, 2, 4):
            self.assertEqual(im[r].tolist(), [1, 5, 9])
        for r in (0, 3, 5):
            self.assertEqual(im[r].tolist(), it[r].tolist())
            np.testing.assert_array_equal(np.array(w_m)[r], np.array(w_t)[r])

    def test_bias_vl_never_scales_the_weights(self):
        g = self._gate()
        x = mx.array(np.random.default_rng(6).standard_normal((2, 8)).astype(np.float32))
        w, i = g(x, mx.array(np.array([True, True])))
        s = np.sqrt(np.log1p(np.exp(np.array(x) @ np.array(g.weight).T)))
        sel = np.take_along_axis(s, np.array(i), -1)
        np.testing.assert_allclose(np.array(w), sel / sel.sum(-1, keepdims=True) * 1.5, rtol=1e-5)


class ImageRows(unittest.TestCase):
    def test_slices_by_absolute_position(self):
        vl = np.zeros((1, 30), bool)
        vl[:, 5:12] = True
        self.assertIsNone(M.image_rows(None, 0, 1, 30))
        self.assertIsNone(M.image_rows(vl, 12, 1, 8))         # piece past the span
        self.assertIsNone(M.image_rows(vl, 40, 1, 1))         # decode row
        seg = M.image_rows(vl, 0, 1, 30)
        np.testing.assert_array_equal(seg, vl)
        seg = M.image_rows(vl, 10, 1, 4)
        self.assertEqual(seg.tolist(), [[True, True, False, False]])


class BiasVlSidecar(unittest.TestCase):
    def test_sidecar_path_and_override(self):
        from mlx_lm.models.deepseek_v41 import exl3_build as eb
        p = eb.bias_vl_sidecar_path("/x/models/foo-EXL3/")
        self.assertTrue(p.endswith("/.exo/dsv41/gate_bias_vl_foo-EXL3.npz"), p)
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "s.npz")
            np.savez(f, **{"layers.3.ffn.gate.bias_vl": np.arange(4, dtype=np.float32)})
            os.environ["DSV41_BIAS_VL"] = f
            try:
                self.assertEqual(eb.bias_vl_sidecar_path("/whatever"), f)
                ck = SimpleNamespace(model_dir="/whatever")
                eb._BIAS_VL_CACHE.clear()
                side = eb._bias_vl_sidecar(ck)
                np.testing.assert_array_equal(side["layers.3.ffn.gate.bias_vl"], np.arange(4))
            finally:
                del os.environ["DSV41_BIAS_VL"]
                eb._BIAS_VL_CACHE.clear()


if __name__ == "__main__":
    unittest.main()
