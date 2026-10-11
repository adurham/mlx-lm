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


def _inputs(rng, args, *, b=1, n=64, nb=120):
    sp = 2 * nb                                   # every column visible
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
                                 n: int, nb: int, seed: int) -> None:
        IX._QSPLIT = True
        IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 8, 0, True
        IX._HIER = hier
        args = _args(candidate_source_layer=(0 if source else -1))
        idx = _indexer(args, 0)
        x, qr, sp, fv, ik = _inputs(np.random.default_rng(seed), args, b=b, n=n, nb=nb)
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
        overlap = _overlap(recon, full)
        self.assertEqual(overlap, 1.0, "top-k overlap < 1.0 vs the full run")
        record({"case": "band_eq_slice", "hier": hier, "source": source, "b": b,
                "n": n, "nb": nb, "overlap": overlap})

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


if __name__ == "__main__":
    unittest.main()
