# Copyright © 2026 Adam Durham (hermes-gw)
"""P20-P1 tests — the indexer QUERY-BAND split across the 2 TP ranks.

``DSV41_INDEXER_QSPLIT`` (default OFF) splits the indexer's query side: each rank
scores / top-ks / emits half the chunk's query rows through the whole indexer
(hier / tiled / untiled), while the KEY side (``index_k``) stays FULL on every
rank. A small ``cidx`` [b, n, index_topk] int32 exchange — a zero-padded ``all_sum``
over the TP group, the v4-default ARQ-reliable path — reconstructs the full
selection on both ranks so the ``sparse_attn`` / candidate-mask consumer is
unchanged.

What these tests pin:

1. GATE SHAPE — ``query_band`` returns the correct half for world=2 rank 0/1 and
   ``None`` (replicated) for: the gate off, no group, world=1, decode/verify-sized
   n (``n <= _FENCE_MIN_ROWS``) and a row count not divisible by the world.
2. ROW-INDEPENDENCE (exactness) — a rank's banded indexer output equals the
   full-run output's row slice, on the SOURCE role (candidate publishing) and a
   plain role, on BOTH the hierarchical and the tiled branches. Each rank's padded
   tensor is captured through a spied ``all_sum`` and summed: the reconstruction is
   asserted ``np.array_equal`` to the full run AND the top-k OVERLAP fraction is
   asserted == 1.0 (bit-exact on these continuous synthetic scores).
3. B>1 GUARD — the same on a b=2 forward. The band is a slice of the QUERY axis
   only, so each batch bands identically and the padded exchange is batch-safe
   (the historical seq-split "batch-unsafe" corruption class).
4. FLAG-OFF IDENTITY — with the gate off ``query_band`` is always ``None``, and a
   ``band=None`` forward never issues a collective (spied ``all_sum`` raising).

Run (mlx-lm worktree root, off-repo copy so the outer ``tests/`` package cannot win
the import race):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_indexer_qsplit.py -v
"""

from __future__ import annotations

import json
import unittest
from typing import Any

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as IX
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import SharedState

RESULTS: list[dict[str, Any]] = []


def record(row: dict[str, Any]) -> None:
    RESULTS.append(row)
    print("  " + json.dumps(row))


class FakeGroup:
    """Minimal stand-in for ``mx.distributed.Group`` (rank / size only)."""

    def __init__(self, rank: int, world: int = 2):
        self._rank = rank
        self._world = world

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._world


def _args(**over: Any) -> ModelArgs:
    base: dict[str, Any] = dict(
        dim=64, n_layers=1, n_heads=4, head_dim=128, rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
        compress_ratios=(2, 2), kv_source_layers=(0,), index_source_layers=(0,),
        index_n_heads=2, index_head_dim=32, index_topk=6, max_seq_len=4096,
        candidate_source_layer=0, candidate_topk_blocks=2, candidate_block_size=8)
    base.update(over)
    return ModelArgs(**base)


def _indexer(args: ModelArgs, layer_id: int) -> IX.Indexer:
    mx.random.seed(0)
    idx = IX.Indexer(args, layer_id)
    mx.eval(idx.parameters())
    return idx


def _inputs(rng, args, *, b=1, n=64, nb=120, sp=None):
    if sp is None:
        sp = 2 * nb                               # every column visible
    x = mx.array(rng.standard_normal((b, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((b, n, args.q_lora_rank)).astype(np.float32))
    ik = mx.array(rng.standard_normal((b, nb, args.index_head_dim)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    return x, qr, sp, fv, ik


def _overlap(got: np.ndarray, ref: np.ndarray) -> float:
    """Mean per-row top-k overlap fraction between two [., n, k] index tensors."""
    a = got.reshape(-1, got.shape[-1])
    b = ref.reshape(-1, ref.shape[-1])
    inter = [len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b)]
    return float(np.mean(inter)) / a.shape[-1]


class _GateFixture(unittest.TestCase):
    """Save/restore the import-read module constants the way the sibling tests do."""

    def setUp(self) -> None:
        self._saved = {"qsplit": IX._QSPLIT, "tile": IX._TILE,
                       "min_nb": IX._TILE_MIN_NB, "force": IX._TILE_FORCE,
                       "hier": IX._HIER, "fence": IX._FENCE_MIN_ROWS}

    def tearDown(self) -> None:
        IX._QSPLIT = self._saved["qsplit"]
        IX._TILE = self._saved["tile"]
        IX._TILE_MIN_NB = self._saved["min_nb"]
        IX._TILE_FORCE = self._saved["force"]
        IX._HIER = self._saved["hier"]
        IX._FENCE_MIN_ROWS = self._saved["fence"]


# --------------------------------------------------------------------------
# 1. gate shape
# --------------------------------------------------------------------------
class GateShapeTest(_GateFixture):
    def test_band_shape_world2(self) -> None:
        IX._QSPLIT = True
        self.assertEqual(IX.query_band(64, FakeGroup(0)), (0, 32))
        self.assertEqual(IX.query_band(64, FakeGroup(1)), (32, 64))
        record({"case": "band_world2", "n": 64, "rank0": (0, 32), "rank1": (32, 64)})

    def test_none_when_gate_off(self) -> None:
        IX._QSPLIT = False
        self.assertIsNone(IX.query_band(64, FakeGroup(0)))
        record({"case": "band_none_gate_off"})

    def test_none_when_no_group(self) -> None:
        IX._QSPLIT = True
        self.assertIsNone(IX.query_band(64, None))
        record({"case": "band_none_no_group"})

    def test_none_when_world_is_one(self) -> None:
        IX._QSPLIT = True
        self.assertIsNone(IX.query_band(64, FakeGroup(0, world=1)))
        record({"case": "band_none_world1"})

    def test_none_at_decode_and_small_n(self) -> None:
        IX._QSPLIT = True                                    # _FENCE_MIN_ROWS == 16
        for n in (1, 4, 16):
            self.assertIsNone(IX.query_band(n, FakeGroup(0)),
                              f"band must stay replicated at n={n}")
        record({"case": "band_none_small_n", "n": [1, 4, 16]})

    def test_none_when_rows_not_divisible(self) -> None:
        IX._QSPLIT = True
        self.assertIsNone(IX.query_band(65, FakeGroup(0)))  # odd
        record({"case": "band_none_uneven"})


# --------------------------------------------------------------------------
# 2/3. row-independence (exactness) + reconstruction
# --------------------------------------------------------------------------
def _run_banded(idx, x, qr, sp, fv, ik, *, band, group, offset=0):
    """Run the indexer for one rank, capturing the padded ``all_sum`` input.

    The spied collective is an identity so the returned tensor IS the rank's
    zero-padded band; two ranks' pads sum to the full selection.
    """
    captured: dict[str, Any] = {}
    real = IX._coll.all_sum

    def fake_all_sum(t, group=None):                     # noqa: ANN001
        captured["padded"] = t
        return t

    IX._coll.all_sum = fake_all_sum
    try:
        shared = SharedState()
        out = idx(x, qr, sp, offset, fv, ik, shared, band=band, group=group)
        mx.eval(out)
    finally:
        IX._coll.all_sum = real
    return np.array(out), np.array(captured["padded"]), shared


def _run_full(idx, x, qr, sp, fv, ik, *, offset=0):
    """Replicated run: no band, no collective (all_sum would raise if touched)."""
    real = IX._coll.all_sum

    def boom(*_a, **_k):
        raise AssertionError("band=None must never issue a collective")

    IX._coll.all_sum = boom
    try:
        shared = SharedState()
        out = idx(x, qr, sp, offset, fv, ik, shared)
        mx.eval(out)
        cand = shared.candidates
    finally:
        IX._coll.all_sum = real
    return np.array(out), cand


class RowIndependenceTest(_GateFixture):
    def _check_band_equals_slice(self, *, hier: bool, source: bool, b: int,
                                 n: int, nb: int, seed: int,
                                 sp: int | None = None,
                                 force_untiled: bool = False,
                                 check_overlap: bool = True) -> None:
        IX._QSPLIT = True
        if force_untiled:
            # tile >= nb -> the UNTILED reference branch (`if tile < nb` false)
            IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 512, 0, False
        else:
            IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 8, 0, True
        IX._HIER = hier
        args = _args(candidate_source_layer=(0 if source else -1))
        idx = _indexer(args, 0)
        x, qr, sp, fv, ik = _inputs(np.random.default_rng(seed), args,
                                    b=b, n=n, nb=nb, sp=sp)
        offset = 37                                     # exercise the +offset path
        g0, g1 = FakeGroup(0), FakeGroup(1)

        full, _ = _run_full(idx, x, qr, sp, fv, ik, offset=offset)
        _o0, pad0, _ = _run_banded(idx, x, qr, sp, fv, ik,
                                   band=(0, n // 2), group=g0, offset=offset)
        _o1, pad1, _ = _run_banded(idx, x, qr, sp, fv, ik,
                                   band=(n // 2, n), group=g1, offset=offset)
        recon = pad0 + pad1                             # two ranks' disjoint pads

        # reconstruction is exact vs the full replicated run ...
        self.assertTrue(np.array_equal(recon, full),
                        f"reconstruction != full (hier={hier} source={source} b={b})")
        # ... and each rank's own rows equal the full run's slice (row-independence)
        half = n // 2
        self.assertTrue(np.array_equal(pad0[:, :half, :], full[:, :half, :]),
                        "rank 0 band != full slice")
        self.assertTrue(np.array_equal(pad1[:, half:, :], full[:, half:, :]),
                        "rank 1 band != full slice")
        # the padded tensor is zero outside this rank's band (disjoint contributors)
        self.assertTrue(np.all(pad0[:, half:, :] == 0))
        self.assertTrue(np.all(pad1[:, :half, :] == 0))
        if check_overlap:
            # a redundant, weaker check than np.array_equal above: it counts the
            # DISTINCT top-k indices, so it is only meaningful when a row's top-k
            # holds distinct values. Heavily-masked frames fill rows with the -1
            # sentinel (a duplicate), collapsing the fraction -- those cases rely
            # on the byte-equality assertion instead (check_overlap=False).
            overlap = _overlap(recon, full)
            self.assertEqual(overlap, 1.0, "top-k overlap < 1.0 vs the full run")
            record({"case": "band_eq_slice", "hier": hier, "source": source,
                    "b": b, "n": n, "nb": nb, "overlap": overlap})
        else:
            record({"case": "band_eq_slice", "hier": hier, "source": source,
                    "b": b, "n": n, "nb": nb, "overlap": "byte-exact"})

    def test_hier_source(self) -> None:
        self._check_band_equals_slice(hier=True, source=True, b=1, n=64, nb=120, seed=3)

    def test_hier_plain_role(self) -> None:
        self._check_band_equals_slice(hier=True, source=False, b=1, n=64, nb=120, seed=4)

    def test_tiled_source(self) -> None:
        self._check_band_equals_slice(hier=False, source=True, b=1, n=64, nb=120, seed=5)

    def test_tiled_plain_role(self) -> None:
        self._check_band_equals_slice(hier=False, source=False, b=1, n=64, nb=120, seed=6)

    def test_uneven_nb(self) -> None:
        self._check_band_equals_slice(hier=True, source=True, b=1, n=64, nb=117, seed=7)

    def test_batch_two_is_batch_safe(self) -> None:
        # B>1 hazard guard: the band is the QUERY axis (per batch), and the padded
        # exchange keeps axis 0 intact, so a b=2 forward reconstructs exactly.
        self._check_band_equals_slice(hier=True, source=True, b=2, n=64, nb=120, seed=8)
        self._check_band_equals_slice(hier=False, source=True, b=2, n=64, nb=120, seed=9)

    def test_negative_sentinel_survives_exchange(self) -> None:
        IX._QSPLIT = True
        # rows whose top-k is all-masked (-1) must reconstruct to -1, not 0.
        banded = np.full((1, 2, 3), -1, np.int32)
        real = IX._coll.all_sum
        IX._coll.all_sum = lambda t, group=None: t        # noqa: ARG005
        try:
            full = np.array(IX._reconstruct_rows(mx.array(banded), 4, 0, 2, None))
        finally:
            IX._coll.all_sum = real
        self.assertTrue(np.all(full[:, 0:2, :] == -1))
        self.assertTrue(np.all(full[:, 2:4, :] == 0))
        record({"case": "negative_sentinel"})

    def test_causal_frame_visibility_is_absolute(self) -> None:
        """T1 (audit gap): with a band, the visibility ``lens`` must use ABSOLUTE
        positions (like the rope), not band-relative ones. Every other case here
        uses ``sp = 2*nb`` (all columns visible), so ``lens`` is inert and the
        mutation M5 ('lens band-relative, rope absolute') escaped the suite. A
        causal frame (``sp=0``, ``nb=n/2``) leaves many columns invisible, so the
        two ranks' frames genuinely differ and a band-relative ``lens`` breaks the
        reconstruction. The masked rows carry duplicate ``-1`` sentinels, so the
        distinct-index overlap metric is degenerate here -- byte-equality is the
        exactness assertion."""
        self._check_band_equals_slice(hier=True, source=True, b=1, n=64, nb=32,
                                      seed=21, sp=0, check_overlap=False)
        self._check_band_equals_slice(hier=False, source=True, b=1, n=64, nb=32,
                                      seed=22, sp=0, check_overlap=False)

    def test_untiled_path_is_banded_and_reconstructed(self) -> None:
        """T2 (audit gap): the UNTILED reference return site must wrap its ``cidx``
        in ``_finish_rows`` too. Every other banded case forces tiling or the
        hierarchy, so this site was uncovered (mutation M10 -- drop the untiled
        wrap -- escaped; a rank would return a ``[b, n/2, k]`` band
        unreconstructed, with no collective issued). Until it is wrapped this test
        fails with ``KeyError 'padded'``."""
        self._check_band_equals_slice(hier=False, source=True, b=1, n=64, nb=120,
                                      seed=31, force_untiled=True)


# --------------------------------------------------------------------------
# 4. flag-off identity
# --------------------------------------------------------------------------
class FlagOffTest(_GateFixture):
    def test_gate_off_returns_none_for_every_shape(self) -> None:
        IX._QSPLIT = False
        for n in (1, 4, 16, 64, 2048):
            for world in (1, 2):
                self.assertIsNone(IX.query_band(n, FakeGroup(0, world=world)))
        record({"case": "flag_off_none_all_shapes"})

    def test_band_none_never_collects(self) -> None:
        IX._QSPLIT = False
        IX._HIER = True
        args = _args()
        idx = _indexer(args, 0)
        x, qr, sp, fv, ik = _inputs(np.random.default_rng(11), args, b=1, n=64, nb=120)
        # _run_full installs a raising all_sum stub; a band=None forward must not
        # touch it (this is the flag-off byte-identity contract).
        out, _ = _run_full(idx, x, qr, sp, fv, ik, offset=5)
        self.assertEqual(out.shape, (1, 64, args.index_topk))
        record({"case": "band_none_never_collects"})


# --------------------------------------------------------------------------
# 5. D2 — the band fence must use the BAND's row count, not the full n
# --------------------------------------------------------------------------
class D2BandFenceTest(_GateFixture):
    """Audit D2 regression: ``query_band`` gated on the FULL ``n``
    (``n <= _FENCE_MIN_ROWS``) while the indexer routes on the BAND's row count
    (``indexer.py`` ``n = qpos.shape[0]``). For ``17 <= n_full <= 32`` at world=2
    the band's 9..16 rows took the small-n fp32 fallback while the replicated run
    took the hier/bf16 path -> not byte-identical. The fix fences on ``n // world``;
    these tests pin the boundary the shipped suite left open (the audit's M16 was
    an expected-green control because nothing exercised it)."""

    def test_fence_applies_to_the_band_row_count(self) -> None:
        IX._QSPLIT = True
        fence = IX._FENCE_MIN_ROWS                              # 16
        # n_full 17..32 at world=2 -> band rows 9..16 <= fence -> replicated
        for n in range(fence + 1, fence * 2 + 1):
            self.assertIsNone(IX.query_band(n, FakeGroup(0)), f"rank0 n={n}")
            self.assertIsNone(IX.query_band(n, FakeGroup(1)), f"rank1 n={n}")
        # just above the boundary: band rows = fence + 1 -> banded
        n = fence * 2 + 2                                      # 34
        self.assertEqual(IX.query_band(n, FakeGroup(0)), (0, fence + 1))
        self.assertEqual(IX.query_band(n, FakeGroup(1)), (fence + 1, n))
        # decode / verify sized stays replicated
        for small in (1, 4, fence):
            self.assertIsNone(IX.query_band(small, FakeGroup(0)))
        record({"case": "d2_fence_on_band_rows", "fence": fence})

    def test_pre_fix_band_splits_the_regime(self) -> None:
        """The D2 mechanism, deterministically: at ``n_full=32`` HIER-on the
        PRE-FIX band (rows=16) does NOT reach the hierarchical pass while the full
        ``n=32`` does -- the regime split D2 reports. The gate now returns ``None``
        for this shape (so the split cannot occur); the replicated output equals the
        full run's."""
        IX._QSPLIT = True
        IX._HIER = True
        args = _args()
        idx = _indexer(args, 0)
        x, qr, sp, fv, ik = _inputs(np.random.default_rng(7), args, b=1, n=32, nb=64)
        calls = {"hier": 0}
        real_hier = IX._hier.hierarchical_topk_prod
        real_sum = IX._coll.all_sum

        def hier_spy(*a, **k):                                 # noqa: ANN002, ANN003
            calls["hier"] += 1
            return real_hier(*a, **k)

        IX._hier.hierarchical_topk_prod = hier_spy
        IX._coll.all_sum = lambda t, group=None: t             # noqa: ARG005
        try:
            out_full = idx(x, qr, sp, 0, fv, ik, SharedState())        # band=None
            mx.eval(out_full)
            hier_full = calls["hier"]
            calls["hier"] = 0
            out_band = idx(x, qr, sp, 0, fv, ik, SharedState(),
                           band=(0, 16), group=FakeGroup(0))          # pre-fix band
            mx.eval(out_band)
            hier_band = calls["hier"]
        finally:
            IX._hier.hierarchical_topk_prod = real_hier
            IX._coll.all_sum = real_sum
        self.assertEqual(hier_full, 1, "full n=32 must take the hierarchical pass")
        self.assertEqual(hier_band, 0, "band rows=16 must NOT take the hier pass")
        # the gate now refuses to band this shape at all (the D2 fix)
        self.assertIsNone(IX.query_band(32, FakeGroup(0)))
        record({"case": "d2_pre_fix_regime_split", "hier_full": hier_full,
                "hier_band": hier_band})


# --------------------------------------------------------------------------
# 6. D1 + T3 — the attention.py CALL SITE
# --------------------------------------------------------------------------
class AttentionCallSiteTest(_GateFixture):
    """D1 + T3/M12: exercise the real ``Attention.__call__`` call site, which the
    direct Indexer-level tests never touch.

    D1: attention.py must not pass ``band=``/``group=`` when the flag is off, or a
        FIXED positional-signature monkeypatch of ``Indexer.__call__`` (the
        supported ``DSV41_NEXT18_CAPTURE`` hook, ``bench/next18_capture.py:222``)
        raises ``TypeError`` on every indexer call.
    T3/M12: with the flag + a TP group ON and a prefill-sized forward, the call
        site must actually pass the band (mutation M12, 'attention never bands',
        is not caught by any direct Indexer test).
    """

    def setUp(self) -> None:
        super().setUp()
        from mlx_lm.models.deepseek_v41.attention import Attention
        from mlx_lm.models.deepseek_v41.cache import ModelCache
        self._Attention = Attention
        self._ModelCache = ModelCache

    def _build(self, *, n0: int):
        args = _args(n_layers=1, max_seq_len=64)
        mx.random.seed(0)
        att = self._Attention(0, args)
        mx.eval(att.parameters())
        cache = self._ModelCache(args, 1, 64)
        cache.ensure_capacity(n0)
        return att, cache

    @staticmethod
    def _indexer_class(att):
        """The ``Indexer`` class THIS Attention instance actually calls.

        Deliberately ``type(att.indexer)``, not ``IX.Indexer``:
        ``tests/test_dsv41_indexer_bf16_row.py`` ends with an
        ``importlib.reload(indexer)`` (it models the import-time env read that
        way), which rebinds a SECOND ``Indexer`` class in the reloaded module
        while ``attention`` still holds the first. When that test runs earlier in
        the directory (alphabetically it does), ``IX.Indexer`` is the new class and
        ``attention.Indexer`` the old -- patching ``IX.Indexer`` would silently
        miss the instance. Patching the instance's own class is correct under
        either state."""
        return type(att.indexer)

    def test_flag_off_survives_a_fixed_positional_wrapper(self) -> None:
        """D1 regression: a fixed-arity capture-style wrapper still binds with the
        flag OFF -- the base call form, with no keyword at all."""
        IX._QSPLIT = False
        att, cache = self._build(n0=8)
        idx_cls = self._indexer_class(att)
        orig = idx_cls.__call__
        calls = {"n": 0}

        def fixed(self_ix, x, qr, start_pos, offset, freqvec, index_k, shared):
            calls["n"] += 1
            return orig(self_ix, x, qr, start_pos, offset, freqvec, index_k, shared)

        setattr(idx_cls, "__call__", fixed)
        try:
            mx.random.seed(3)
            x = mx.random.normal((1, 8, att.dim)) * 0.5
            out = att(x, 0, cache, SharedState())
            mx.eval(out)
        finally:
            setattr(idx_cls, "__call__", orig)
        self.assertGreater(calls["n"], 0, "the indexer was never reached")
        self.assertEqual(out.shape, (1, 8, att.dim))
        record({"case": "d1_fixed_positional_wrapper", "flag": "off",
                "indexer_calls": calls["n"]})

    def test_attention_passes_the_band_when_flag_and_group_on(self) -> None:
        """T3/M12: with the flag + a TP group ON and a prefill-sized forward the
        call site must pass ``band`` (and ``group``)."""
        IX._QSPLIT = True
        IX._HIER = True
        att, cache = self._build(n0=40)
        setattr(att, "group", FakeGroup(0))
        att_group = getattr(att, "group")
        seen: list[dict[str, Any]] = []
        idx_cls = self._indexer_class(att)
        orig = idx_cls.__call__
        real_sum = IX._coll.all_sum

        def recording(self_ix, *a, **kw):                      # noqa: ANN002, ANN003
            seen.append(kw)
            return orig(self_ix, *a, **kw)

        setattr(idx_cls, "__call__", recording)
        IX._coll.all_sum = lambda t, group=None: t             # noqa: ARG005
        try:
            mx.random.seed(4)
            x = mx.random.normal((1, 40, att.dim)) * 0.5
            out = att(x, 0, cache, SharedState())
            mx.eval(out)
        finally:
            setattr(idx_cls, "__call__", orig)
            IX._coll.all_sum = real_sum
        self.assertTrue(seen, "the indexer was never called")
        self.assertEqual(seen[0].get("band"), (0, 20),
                         "attention did not band (M12)")
        self.assertIs(seen[0].get("group"), att_group)
        record({"case": "t3_attention_bands", "band": list(seen[0]["band"])})


# --------------------------------------------------------------------------
# 7. live-A/B remediation — banded NON-FIRST chunks, row-count-dependent GEMMs
# --------------------------------------------------------------------------
# Live gate (postmortem 2026-10-11): prompt identity_3 = 2 prefill chunks (2048 +
# 78); only chunk 2 (even n=78 -> bands of 39) engaged the band, and the output
# diverged at token 4. Root cause: the band ran the indexer's two QUERY
# projections (``wq_b``, ``weights_proj``) on ITS rows only. Neither GEMM is
# row-count invariant -- production ``wq_b`` is an EXL3Linear whose kernel is
# picked by the row count (fused trellis GEMM for rows <= 64, decode_full +
# native matmul above), and MLX's bf16 matmul is M-dependent as well -- so a
# band row's q/w differed from the replicated run's, and the k-th boundary
# flipped. The suite missed it because nn.Linear fp32 at n=64 is M-invariant.
class _RowCountSensitive:
    """Projection whose output depends on the CALL's row count (like EXL3's
    rows<=64 vs rows>64 kernel switch, or an M-dependent GEMM): ``lin(x) +
    noise * rows``. Deterministic, and guaranteed to differ between a band call
    (n/2 rows) and the full call (n rows) -- so a band that projects only its own
    rows CANNOT reproduce the replicated selection."""

    def __init__(self, lin, seed: int):
        self._lin = lin
        rng = np.random.default_rng(seed)
        self._noise = mx.array(rng.standard_normal(lin.weight.shape[0])
                               .astype(np.float32) * 2e-3)
        self.rows_seen: list[int] = []

    def __call__(self, x: mx.array) -> mx.array:
        rows = x.size // x.shape[-1]
        self.rows_seen.append(rows)
        return self._lin(x) + self._noise * float(rows)


def _synth_exl3(inf: int, outf: int, seed: int, k: int = 3):
    """Synthetic EXL3 layer (unique runtime-cache key per shape+seed)."""
    from mlx_lm.models.exl3.ref.layer import EXL3Layer
    rng = np.random.default_rng(seed)
    return EXL3Layer(
        key=f"qsplit-test-{seed}-{inf}x{outf}", in_features=inf, out_features=outf,
        k=k, trellis=rng.integers(0, 2 ** 16, size=(inf // 16, outf // 16, 256 * k // 16),
                                  dtype=np.uint16),
        suh=rng.choice(np.array([-1.0, 1.0], np.float16), size=inf),
        svh=rng.choice(np.array([-1.0, 1.0], np.float16), size=outf), mul1=True)


def _band_vs_full(idx, x, qr, sp, fv, ik, *, offset: int):
    """(reconstructed two-rank cidx, replicated cidx) for an even chunk."""
    n = x.shape[1]
    full, _ = _run_full(idx, x, qr, sp, fv, ik, offset=offset)
    _o0, p0, _ = _run_banded(idx, x, qr, sp, fv, ik, band=(0, n // 2),
                             group=FakeGroup(0), offset=offset)
    _o1, p1, _ = _run_banded(idx, x, qr, sp, fv, ik, band=(n // 2, n),
                             group=FakeGroup(1), offset=offset)
    return p0 + p1, full


class NonFirstChunkBandTest(_GateFixture):
    """Banded NON-FIRST chunks: nonzero chunk base, even/odd band boundaries,
    row-count-dependent projections. Each case compares the two-rank
    reconstruction to the replicated run byte-for-byte."""

    def _case(self, *, hier: bool, n: int, sp: int, nb: int, seed: int,
              proj: str, source: bool = True) -> None:
        IX._QSPLIT = True
        IX._HIER = hier
        IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 8, 0, not hier
        args = _args(candidate_source_layer=(0 if source else -1), index_topk=6)
        idx = _indexer(args, 0)
        if proj in ("wq_b", "both"):
            setattr(idx, "wq_b", _RowCountSensitive(idx.wq_b, seed))
        if proj in ("weights_proj", "both"):
            setattr(idx, "weights_proj", _RowCountSensitive(idx.weights_proj, seed + 1))
        rng = np.random.default_rng(seed)
        x, qr, _sp, fv, ik = _inputs(rng, args, n=n, nb=nb)
        offset = args.window_size + n                    # concat-space, like attention
        recon, full = _band_vs_full(idx, x, qr, sp, fv, ik, offset=offset)
        rows = int((recon != full).any(-1).sum())
        record({"case": "nonfirst_chunk", "hier": hier, "n": n, "start_pos": sp,
                "nb": nb, "proj": proj, "diff_rows": rows})
        self.assertEqual(rows, 0, f"{rows} band rows != replicated (proj={proj}, "
                                  f"hier={hier}, n={n}, start_pos={sp})")

    # identity_3's chunk-2 geometry, scaled: even n=78 -> bands of 39 (odd)
    def test_chunk2_n78_rowcount_wq_b_hier(self) -> None:
        self._case(hier=True, n=78, sp=2048, nb=(2048 + 78) // 2 // 8, seed=41,
                   proj="wq_b")

    def test_chunk2_n78_rowcount_wq_b_tiled(self) -> None:
        self._case(hier=False, n=78, sp=2048, nb=(2048 + 78) // 2 // 8, seed=42,
                   proj="wq_b")

    def test_chunk2_n78_rowcount_weights_proj_hier(self) -> None:
        self._case(hier=True, n=78, sp=2048, nb=(2048 + 78) // 2 // 8, seed=43,
                   proj="weights_proj")

    def test_chunk2_n78_rowcount_weights_proj_tiled(self) -> None:
        self._case(hier=False, n=78, sp=2048, nb=(2048 + 78) // 2 // 8, seed=44,
                   proj="weights_proj")

    def test_even_band_rows_n80(self) -> None:
        # band rows 40 (even) -- the other parity of the band boundary
        self._case(hier=True, n=80, sp=512, nb=150, seed=45, proj="both")

    def test_plain_role_nonfirst(self) -> None:
        self._case(hier=True, n=78, sp=1024, nb=270, seed=46, proj="both",
                   source=False)

    def test_band_boundaries_even_odd(self) -> None:
        """The live table's chunk-2 sizes: 3 / 129 (odd) stay replicated; 78
        bands at 39 (odd band rows); 2048 at 1024; 80 at 40 (even band rows)."""
        IX._QSPLIT = True
        for n in (3, 129, 79):
            self.assertIsNone(IX.query_band(n, FakeGroup(0)), f"n={n}")
        for n, h in ((78, 39), (80, 40), (2048, 1024), (34, 17)):
            self.assertEqual(IX.query_band(n, FakeGroup(0)), (0, h))
            self.assertEqual(IX.query_band(n, FakeGroup(1)), (h, n))
        record({"case": "band_boundaries_even_odd"})

    def test_projections_see_the_full_chunk(self) -> None:
        """Structural: a banded call projects ALL n_full rows (then slices), so
        each band row is bitwise the replicated run's row whatever kernel the
        row count selects."""
        IX._QSPLIT = True
        IX._HIER = True
        args = _args()
        idx = _indexer(args, 0)
        wq = _RowCountSensitive(idx.wq_b, 1)
        wp = _RowCountSensitive(idx.weights_proj, 2)
        setattr(idx, "wq_b", wq)
        setattr(idx, "weights_proj", wp)
        x, qr, sp, fv, ik = _inputs(np.random.default_rng(5), args, n=78, nb=120)
        _run_banded(idx, x, qr, 2048, fv, ik, band=(39, 78), group=FakeGroup(1))
        self.assertEqual(wq.rows_seen, [78])
        self.assertEqual(wp.rows_seen, [78])
        record({"case": "projections_full_rows", "wq_b": wq.rows_seen,
                "weights_proj": wp.rows_seen})


class MarginZeroTieTest(_GateFixture):
    """Margin-0 ties AT the k=512 boundary: a third of the key columns are one
    duplicated key, so many rows' 512th/513th scores are bitwise equal and the
    pick is argpartition's tie rule. Band rows are bitwise the replicated rows
    (same q, w, keys, same top-k op on the same row shape [nb]), so the tie
    break is applied to identical inputs and must agree exactly. Control: green
    pre- and post-fix on nn.Linear (row-count invariant); the RED proof is
    the row-count-sensitive variant below."""

    def _tie_case(self, *, hier: bool, rowcount: bool, seed: int) -> None:
        IX._QSPLIT = True
        IX._HIER = hier
        IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 512, 0, False
        # no candidate pre-filter: the k=512 boundary must sit inside the tie class
        args = _args(compress_ratios=(4, 4), index_topk=512, index_n_heads=4,
                     max_seq_len=1 << 14, candidate_source_layer=-1)
        idx = _indexer(args, 0)
        if rowcount:
            setattr(idx, "wq_b", _RowCountSensitive(idx.wq_b, seed))
        rng = np.random.default_rng(seed)
        n, sp = 78, 2048
        nb = (sp + n) // 4                                     # 531 > k=512
        x, qr, _sp, fv, _ik = _inputs(rng, args, n=n, nb=nb)
        keys = rng.standard_normal((nb, args.index_head_dim)).astype(np.float32)
        dup = rng.choice(nb, size=nb // 3, replace=False)
        keys[dup] = keys[dup[0]]
        ik = mx.array(keys[None])
        recon, full = _band_vs_full(idx, x, qr, sp, fv, ik, offset=8 + n)
        rows = int((recon != full).any(-1).sum())
        record({"case": "margin0_ties_k512", "hier": hier, "rowcount": rowcount,
                "diff_rows": rows})
        self.assertEqual(rows, 0)

    def test_ties_hier(self) -> None:
        self._tie_case(hier=True, rowcount=False, seed=51)

    def test_ties_untiled(self) -> None:
        self._tie_case(hier=False, rowcount=False, seed=52)

    def test_ties_rowcount_sensitive(self) -> None:
        self._tie_case(hier=True, rowcount=True, seed=53)


class ProductionExl3ChunkTwoTest(_GateFixture):
    """The deterministic OFFLINE repro of identity_3's class at production
    indexer geometry: q_lora 1024 -> EXL3 ``wq_b`` 8192 (64 heads x 128), bf16
    ``weights_proj`` 4096 -> 64, k=512, ratio 4, chunk 2 at start_pos=2048 with
    n=78 (bands of 39 rows: the fused-trellis-GEMM side of
    ``EXL3_FUSED_ROW_LIMIT=64``; the full 78 rows take decode_full + matmul).
    Pre-fix: seed 4 diverges at row 38 (1 swapped slot; 8/32 seeds diverge).
    Post-fix: identical."""

    def test_identity3_shape_real_exl3(self) -> None:
        from mlx_lm.models.deepseek_v41.exl3_build import Exl3Proj
        from mlx_lm.models.exl3 import EXL3Linear
        IX._QSPLIT = True
        IX._HIER = True
        seed = 4
        a = ModelArgs(dim=4096, n_layers=1, n_heads=4, head_dim=512, rope_head_dim=64,
                      q_lora_rank=1024, o_lora_rank=32, o_groups=4, window_size=128,
                      compress_ratios=(4,), kv_source_layers=(0,),
                      index_source_layers=(0,), index_n_heads=64, index_head_dim=128,
                      index_topk=512, max_seq_len=8192, candidate_source_layer=0,
                      candidate_topk_blocks=64, candidate_block_size=8)
        mx.random.seed(seed)
        idx = IX.Indexer(a, 0)
        idx.set_dtype(mx.bfloat16)
        setattr(idx, "wq_b", Exl3Proj(EXL3Linear(_synth_exl3(1024, 8192, seed))))
        mx.eval(idx.parameters())
        rng = np.random.default_rng(seed)
        n, sp = 78, 2048
        nb = (sp + n) // 4
        x = mx.array(rng.standard_normal((1, n, a.dim)).astype(np.float32)).astype(mx.bfloat16)
        qr = mx.array(rng.standard_normal((1, n, a.q_lora_rank)).astype(np.float32)
                      ).astype(mx.bfloat16)
        ik = IX.fake_quant_fp4_ue8m0(
            mx.array(rng.standard_normal((1, nb, a.index_head_dim)).astype(np.float32) * 0.5),
            32).astype(mx.bfloat16)
        fv = mx.array(np.linspace(1.0, 1e-4, a.rope_head_dim // 2).astype(np.float32))
        recon, full = _band_vs_full(idx, x, qr, sp, fv, ik, offset=128 + n)
        rows = [int(r) for r in np.nonzero((recon != full).any(-1))[1]]
        record({"case": "identity3_real_exl3", "diff_rows": rows})
        self.assertEqual(rows, [], f"band != replicated at rows {rows}")


class MultiChunkAttentionTest(_GateFixture):
    """Multi-chunk prompt through the real ``Attention.__call__`` (window +
    compressed KV + indexer + sparse_attn), flag ON with a simulated 2-rank cidx
    exchange vs flag OFF: the chunk outputs (the DOWNSTREAM effect, not only the
    selection) must be bitwise equal on every chunk, including the non-first
    banded chunk. The indexer's ``wq_b`` is row-count sensitive (the production
    EXL3 property)."""

    def test_two_chunk_prompt_outputs_identical(self) -> None:
        from mlx_lm.models.deepseek_v41.attention import Attention
        from mlx_lm.models.deepseek_v41.cache import ModelCache
        IX._HIER = True
        args = _args(n_layers=1, max_seq_len=256)
        chunks = (40, 78)                                 # bands 20 then 39
        mx.random.seed(9)
        xs = [mx.random.normal((1, c, args.dim)) * 0.5 for c in chunks]

        def prompt(flag: bool):
            IX._QSPLIT = flag
            mx.random.seed(0)
            att = Attention(0, args)
            mx.eval(att.parameters())
            setattr(att.indexer, "wq_b", _RowCountSensitive(att.indexer.wq_b, 61))
            setattr(att, "group", FakeGroup(0))
            cache = ModelCache(args, 1, 256)
            cache.ensure_capacity(sum(chunks))
            idx_cls = type(att.indexer)
            orig = idx_cls.__call__
            bands: list[tuple[int, int]] = []

            def two_rank(self_ix, x, qr, sp, off, fv, ik, shared, **kw):
                # simulate both ranks' bands + the padded all_sum in-process
                if kw.get("band") is None:
                    return orig(self_ix, x, qr, sp, off, fv, ik, shared, **kw)
                nf = x.shape[1]
                bands.append(kw["band"])
                pads = []
                for r, b in enumerate(((0, nf // 2), (nf // 2, nf))):
                    pads.append(orig(self_ix, x, qr, sp, off, fv, ik, shared,
                                     band=b, group=FakeGroup(r)))
                return pads[0] + pads[1]

            real_sum = IX._coll.all_sum
            setattr(idx_cls, "__call__", two_rank)
            IX._coll.all_sum = lambda t, group=None: t     # noqa: ARG005
            outs, sels = [], []
            try:
                sp = 0
                for x in xs:
                    shared = SharedState()
                    o = att(x, sp, cache, shared)
                    mx.eval(o, shared.topk_idxs)
                    outs.append(np.array(o))
                    sels.append(np.array(shared.topk_idxs))
                    sp += x.shape[1]
            finally:
                setattr(idx_cls, "__call__", orig)
                IX._coll.all_sum = real_sum
            return outs, sels, bands

        off_o, off_s, off_b = prompt(False)
        on_o, on_s, on_b = prompt(True)
        self.assertEqual(off_b, [])
        self.assertEqual(on_b, [(0, 20), (0, 39)], "both chunks must band")
        sel_rows = [int((a != b).any(-1).sum()) for a, b in zip(on_s, off_s)]
        out_eq = [bool(np.array_equal(a, b)) for a, b in zip(on_o, off_o)]
        record({"case": "two_chunk_attention", "chunks": list(chunks),
                "sel_diff_rows": sel_rows, "out_equal": out_eq})
        self.assertEqual(sel_rows, [0, 0], "cidx differs on a banded chunk")
        self.assertEqual(out_eq, [True, True], "attention output differs")


if __name__ == "__main__":
    unittest.main()
