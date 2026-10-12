# Copyright © 2026 Adam Durham (hermes-gw)
"""R-B small-n wide tile (``DSV41_INDEXER_SMALLN_WIDE_TILE``, idxprof5 §6 / idxprof7).

For a forward with ``n <= _FENCE_MIN_ROWS`` rows (n = the FULL chunk, D3) that the
budget sends down the tiled branch, :func:`indexer.tile_width` now returns the
aligned budget cap instead of ``min(_TILE, cap)``. Nothing else moves: the
tiled/untiled decision, the budget, the L2-full fp32 row, the branch and the top-k.

The budget is shrunk (``_TILE_BUDGET``, the import-read value of
``DSV41_INDEXER_TILE_MB``) so the crossing happens at a few thousand columns, and
``_TILE`` is 64 so the "old" arm runs many tiles. Arms, all through the REAL
``Indexer.__call__``:

  NEW  switch on (default)        -> tiled at the aligned cap
  OLD  switch off (kill switch)   -> tiled at min(_TILE, cap)  (today's rule)
  UNT  ``_TILE = 0``              -> untiled reference branch

What these tests pin:

1. RULE — the width is the aligned cap at every n <= 16, unchanged for n > 16, the
   kill switch reproduces the old formula byte-for-byte on a dense grid, the
   ``tiled()`` decision is identical with the switch on/off, and the wide tile's
   transient + row stays inside the budget.
2. ENV — the switch is read once at import: unset -> on, ``=0`` -> off.
3. IDENTITY — cidx (and layer-20-style candidate mask) identical NEW / OLD / UNT on
   random keys, periodic tie-injected keys (asserted: exact ties AT rank k),
   a targeted cap-boundary tie, and remainder edges (cap+1, 2cap+1, 3cap-1, a
   tail shorter than ``align``), for plain owner, candidate source and consumer.
4. D3 — a band of a small-n chunk (explicit ``band``) uses the FULL chunk's width
   (a band routed on its own rows would go UNTILED here), and the two-rank
   reconstruction equals the replicated run, source and consumer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from typing import Any

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as IX
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import SharedState

RESULTS: list[dict[str, Any]] = []
FMR = IX._FENCE_MIN_ROWS                      # 16 unless the env overrode it

_KNOBS = ("_SMALLN_WIDE_TILE", "_TILE", "_TILE_BUDGET", "_TILE_MIN_NB",
          "_TILE_FORCE", "_HIER", "_QSPLIT", "_L2_FULL", "_SMALLN_ROW_BF16",
          "_TILED_IMPL", "_FENCE_MIN_ROWS")

H = 2                                         # index_n_heads in the toy config
ROW = 4                                       # fp32 small-n row (L2-full)
BUDGET = 49_152                              # bytes -> cap(n) = 4096 // n (x8 at n=1,4,8,16)
TILE_SMALL = 64


def record(row: dict[str, Any]) -> None:
    RESULTS.append(row)
    print("  " + json.dumps(row), flush=True)


def old_tile_width(bsz: int, n: int, n_heads: int, nb: int, align: int, *,
                   tile: int, budget: int, row_size: int) -> int:
    """Today's rule, transcribed verbatim from 6a5c78f4 (the reference)."""
    if tile <= 0:
        return nb
    cap = max(align, budget // max(1, bsz * n * (n_heads * 4 + row_size)))
    t = max(align, (min(tile, cap) // align) * align)
    return min(nb, t)


def cap_of(n: int, align: int, *, budget: int = BUDGET, heads: int = H,
           row: int = ROW, bsz: int = 1) -> int:
    cap = max(align, budget // (bsz * n * (heads * 4 + row)))
    return max(align, (cap // align) * align)


def _args(**over: Any) -> ModelArgs:
    base: dict[str, Any] = dict(
        dim=64, n_layers=3, n_heads=4, head_dim=128, rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
        compress_ratios=(2, 2, 2), kv_source_layers=(0,), index_source_layers=(0,),
        index_n_heads=H, index_head_dim=32, index_topk=128, max_seq_len=1 << 16,
        candidate_source_layer=0, candidate_topk_blocks=32, candidate_block_size=8)
    base.update(over)
    return ModelArgs(**base)


def _indexer(args: ModelArgs, layer_id: int, seed: int) -> IX.Indexer:
    mx.random.seed(seed)
    idx = IX.Indexer(args, layer_id)
    mx.eval(idx.parameters())
    return idx


def _inputs(seed: int, args: ModelArgs, *, n: int, nb: int, keys: str = "random"):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
    d = args.index_head_dim
    if keys == "random":
        k = rng.standard_normal((nb, d)).astype(np.float32)
    elif keys == "ties":
        # periodic keys: column j == column j % p -> every score repeats ~nb/p times
        p = 97
        base = rng.standard_normal((p, d)).astype(np.float32)
        k = base[np.arange(nb) % p]
    else:
        raise ValueError(keys)
    ik = mx.array(k[None])
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    sp = 2 * nb - n                     # ratio 2: the last row sees exactly nb groups
    return x, qr, sp, fv, ik


class _TileSpy:
    """Record the ``tile`` each ``_tiled_scores_buffer`` call receives (None = untiled)."""

    def __init__(self) -> None:
        self.tiles: list[int] = []
        self._real = IX._tiled_scores_buffer

    def __enter__(self):
        real = self._real

        def wrapper(*a, **k):
            self.tiles.append(int(a[6]))
            return real(*a, **k)

        IX._tiled_scores_buffer = wrapper
        return self

    def __exit__(self, *exc):
        IX._tiled_scores_buffer = self._real


class _Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {k: getattr(IX, k) for k in _KNOBS}
        self._real_all_sum = IX._coll.all_sum

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            setattr(IX, k, v)
        IX._coll.all_sum = self._real_all_sum

    def shrink(self) -> None:
        IX._TILE, IX._TILE_BUDGET = TILE_SMALL, BUDGET
        IX._TILE_MIN_NB, IX._TILE_FORCE = 0, False
        IX._HIER, IX._QSPLIT = True, False
        IX._L2_FULL, IX._SMALLN_ROW_BF16 = True, False
        IX._TILED_IMPL = "buffer"


# --------------------------------------------------------------------------
# 1. the rule
# --------------------------------------------------------------------------
class RuleTest(_Fixture):
    def test_production_constants_widths(self) -> None:
        """Production geometry (TILE 512, 128 MiB, 32 heads, fp32 small-n row):
        the widths idxprof5 derived, e.g. n=4 -> 254,200 at align 8."""
        IX._TILE, IX._TILE_BUDGET = 512, 128 << 20
        IX._L2_FULL, IX._SMALLN_ROW_BF16 = True, False
        want = {1: 1_016_800, 4: 254_200, 8: 127_096, 12: 84_728, 16: 63_544}
        for n, w in want.items():
            IX._SMALLN_WIDE_TILE = True
            got = IX.tile_width(1, n, 32, 2_000_000, 8)
            self.assertEqual(got, w, f"n={n}")
            IX._SMALLN_WIDE_TILE = False
            self.assertEqual(IX.tile_width(1, n, 32, 2_000_000, 8), 512, f"old n={n}")
        # align 1 (owners / consumers): unaligned cap
        IX._SMALLN_WIDE_TILE = True
        self.assertEqual(IX.tile_width(1, 4, 32, 2_000_000, 1), 254_200)
        self.assertEqual(IX.tile_width(1, 12, 32, 2_000_000, 1), 84_733)
        # n > 16 untouched
        for n in (17, 32, 78, 512, 2048):
            self.assertEqual(IX.tile_width(1, n, 32, 2_000_000, 8),
                             old_tile_width(1, n, 32, 2_000_000, 8, tile=512,
                                            budget=128 << 20,
                                            row_size=IX._row_dtype(n).size), f"n={n}")
        record({"case": "production_widths", "widths": want})

    def test_kill_switch_is_old_rule_and_decision_unchanged(self) -> None:
        """Dense grid: switch OFF == the transcribed old formula byte-for-byte;
        switch ON == old for n > 16; the tiled/untiled DECISION never changes."""
        grid = 0
        IX._TILE_MIN_NB, IX._TILE_FORCE = 0, False
        for tile, budget in ((512, 128 << 20), (64, BUDGET), (0, BUDGET), (512, 4096)):
            IX._TILE, IX._TILE_BUDGET = tile, budget
            for l2 in (True, False):
                IX._L2_FULL = l2
                for bsz in (1, 2):
                    for heads in (2, 32, 64):
                        for n in (1, 2, 3, 4, 5, 8, 12, 15, 16, 17, 18, 33, 78, 512):
                            for nb in (1, 7, 8, 63, 511, 512, 513, 999, 1000, 1001,
                                       4097, 254_199, 254_200, 254_201, 300_000,
                                       1_000_003):
                                for align in (1, 8):
                                    rs = IX._row_dtype(n).size
                                    ref = old_tile_width(bsz, n, heads, nb, align,
                                                         tile=tile, budget=budget,
                                                         row_size=rs)
                                    IX._SMALLN_WIDE_TILE = False
                                    w_off = IX.tile_width(bsz, n, heads, nb, align)
                                    d_off = IX.tiled(bsz, n, heads, nb, align) < nb
                                    IX._SMALLN_WIDE_TILE = True
                                    w_on = IX.tile_width(bsz, n, heads, nb, align)
                                    d_on = IX.tiled(bsz, n, heads, nb, align) < nb
                                    self.assertEqual(w_off, ref)
                                    self.assertEqual(d_on, d_off,
                                                     (tile, budget, bsz, heads, n, nb))
                                    if n > FMR or tile <= 0:
                                        self.assertEqual(w_on, ref)
                                    else:
                                        cap = max(align, budget // max(
                                            1, bsz * n * (heads * 4 + rs)))
                                        self.assertEqual(
                                            w_on, min(nb, max(align, (cap // align) * align)))
                                        self.assertGreaterEqual(w_on, w_off)
                                    grid += 1
        record({"case": "kill_switch_grid", "cells": grid})

    def test_tile_force_hook_keeps_tile_width(self) -> None:
        """``DSV41_INDEXER_TILE_FORCE`` (test hook, off in production) keeps its
        contract -- tiled at ``_TILE`` -- so the sibling suites that force the
        tiled branch at toy shapes still exercise it."""
        IX._TILE, IX._TILE_BUDGET, IX._TILE_MIN_NB = 8, 128 << 20, 0
        IX._TILE_FORCE, IX._SMALLN_WIDE_TILE = True, True
        for n in (1, 4, 16):
            self.assertEqual(IX.tile_width(1, n, 2, 120, 8), 8)
            self.assertEqual(IX.tiled(1, n, 2, 120, 8), 8)
        record({"case": "tile_force_hook"})

    def test_wide_tile_within_budget(self) -> None:
        """The wide tile's [b,n,h,t] fp32 transient + [b,n,t] row stays <= budget
        whenever the cap is not floored at align."""
        IX._TILE, IX._TILE_BUDGET, IX._SMALLN_WIDE_TILE = 512, 128 << 20, True
        for n in range(1, FMR + 1):
            for align in (1, 8):
                t = IX.tile_width(1, n, 32, 10_000_000, align)
                self.assertLessEqual(n * t * (32 * 4 + IX._row_dtype(n).size), 128 << 20)
        record({"case": "wide_within_budget"})


class EnvReadTest(unittest.TestCase):
    def _flag(self, value: str | None) -> bool:
        env = {k: v for k, v in os.environ.items()
               if k != "DSV41_INDEXER_SMALLN_WIDE_TILE"}
        if value is not None:
            env["DSV41_INDEXER_SMALLN_WIDE_TILE"] = value
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run(
            [sys.executable, "-c",
             "from mlx_lm.models.deepseek_v41 import indexer as I;"
             "print('FLAG', int(I._SMALLN_WIDE_TILE))"],
            env=env, capture_output=True, text=True, check=True, cwd=root)
        line = [s for s in out.stdout.splitlines() if s.startswith("FLAG")][-1]
        return line.split()[1] == "1"

    def test_default_on_and_zero_off(self) -> None:
        self.assertTrue(self._flag(None))
        self.assertTrue(self._flag("1"))
        self.assertFalse(self._flag("0"))
        record({"case": "env_read", "default": True, "zero": False})


# --------------------------------------------------------------------------
# 3. index identity NEW / OLD / UNT through the real __call__
# --------------------------------------------------------------------------
def _call(idx: IX.Indexer, inp, shared: SharedState):
    x, qr, sp, fv, ik = inp
    out = idx(x, qr, sp, 0, fv, ik, shared)
    mx.eval(out)
    return np.array(out)


class IdentityTest(_Fixture):
    ARMS = ("NEW", "OLD", "UNT")

    def _set_arm(self, arm: str) -> None:
        self.shrink()
        if arm == "NEW":
            IX._SMALLN_WIDE_TILE = True
        elif arm == "OLD":
            IX._SMALLN_WIDE_TILE = False
        else:
            IX._TILE = 0

    def _run_arms(self, role: str, n: int, nb: int, keys: str, seed: int,
                  boundary_tie: bool = False) -> dict[str, Any]:
        args = _args(candidate_source_layer=(-1 if role == "plain" else 0))
        src = _indexer(args, 0, seed)
        cons = _indexer(args, 1, seed + 1) if role == "consumer" else None
        inp = list(_inputs(seed, args, n=n, nb=nb, keys=keys))
        align = 8 if role in ("source", "consumer") else 1
        cap = cap_of(n, align)
        if boundary_tie and cap < nb:
            k = np.array(inp[4])
            k[0, cap] = k[0, cap - 1]                     # equal scores across the seam
            inp[4] = mx.array(k)
        out, masks, tiles = {}, {}, {}
        for arm in self.ARMS:
            self._set_arm(arm)
            sh = SharedState()
            with _TileSpy() as spy:
                o = _call(src, inp, sh)
                if cons is not None:
                    o = _call(cons, inp, sh)           # consumes src's mask
            out[arm] = o
            masks[arm] = (np.array(sh.candidates) if sh.candidates is not None else None)
            tiles[arm] = spy.tiles
        # each arm really took its intended route/width
        self.assertEqual(tiles["UNT"], [])
        # source layer aligns to candidate_block_size (8); plain and consumer to 1
        want_new = [min(nb, cap_of(n, 8 if role != "plain" else 1))]
        if role == "consumer":
            want_new.append(min(nb, cap_of(n, 1)))
        self.assertEqual(tiles["NEW"], want_new, (role, n, nb))
        self.assertTrue(all(t == min(TILE_SMALL, cap) // align * align or t == align
                            for t in tiles["OLD"]), tiles)
        self.assertTrue(tiles["NEW"] and tiles["OLD"], "shape must be on the tiled route")
        rec = dict(role=role, n=n, nb=nb, keys=keys, seed=seed, cap=cap,
                   boundary_tie=boundary_tie, new_tile=tiles["NEW"][0],
                   old_tile=tiles["OLD"][0])
        for arm in ("OLD", "UNT"):
            rec[f"NEW_vs_{arm}_rows"] = int((out["NEW"] != out[arm]).any(-1).sum())
            if masks["NEW"] is not None:
                rec[f"NEW_vs_{arm}_mask"] = bool(np.array_equal(masks["NEW"], masks[arm]))
        record(rec)
        return rec

    def _assert_identical(self, rec: dict[str, Any]) -> None:
        for arm in ("OLD", "UNT"):
            self.assertEqual(rec[f"NEW_vs_{arm}_rows"], 0, rec)
            if f"NEW_vs_{arm}_mask" in rec:
                self.assertTrue(rec[f"NEW_vs_{arm}_mask"], rec)

    def test_random_keys_all_roles(self) -> None:
        bad = []
        for role in ("plain", "source", "consumer"):
            for n in (1, 4, 8, 12, 16):
                for mult in (2.5, 4.0):
                    nb = max(int(cap_of(n, 8) * mult), 300)
                    rec = self._run_arms(role, n, nb, "random", 1000 + n)
                    if rec["NEW_vs_OLD_rows"] or rec["NEW_vs_UNT_rows"] or \
                            rec.get("NEW_vs_OLD_mask") is False or \
                            rec.get("NEW_vs_UNT_mask") is False:
                        bad.append(rec)
        self.assertFalse(bad, bad[:3])

    def test_tie_injected_keys_ties_at_k(self) -> None:
        """Periodic keys put exact ties ON the k boundary (census asserted on the
        fp32 row); the three routes still return identical sets."""
        tied_rows_total = 0
        for role in ("plain", "source", "consumer"):
            for n in (1, 4, 16):
                nb = cap_of(n, 8) * 3 + 5
                rec = self._run_arms(role, n, nb, "ties", 2000 + n)
                self._assert_identical(rec)
        # census: score the fp32 row once (plain role, n=4) and count kth==(k+1)th
        self.shrink()
        args = _args(candidate_source_layer=-1)
        idx = _indexer(args, 0, 2004)
        for n in (1, 4, 16):
            nb = cap_of(n, 8) * 3 + 5
            x, qr, sp, fv, ik = _inputs(2000 + n, args, n=n, nb=nb, keys="ties")
            q32, w32, lens = self._qw(idx, x, qr, sp, fv)
            s = IX._score_body_eager(q32, ik.astype(mx.float32), w32)
            s = mx.where(mx.arange(nb)[None, :] < lens, s, IX.NEG_INF)
            srt = np.sort(np.array(s), axis=-1)[..., ::-1]
            k = args.index_topk
            tied = int((srt[..., k - 1] == srt[..., k]).sum())
            tied_rows_total += tied
            record({"case": "tie_census", "n": n, "nb": nb, "rows_tied_at_k": tied,
                    "rows": n})
        self.assertGreater(tied_rows_total, 0, "tie injection did not reach rank k")

    @staticmethod
    def _qw(idx: IX.Indexer, x, qr, sp, fv):
        """The indexer's own q/w/lens for a replicated forward (mirrors __call__)."""
        n = x.shape[1]
        q = idx.wq_b(qr).reshape(1, n, idx.n_heads, idx.head_dim)
        qpos = mx.arange(sp, sp + n)
        q = IX.rope_tail(q, idx.rope_head_dim, *IX.cos_sin_at(fv, qpos))
        q = IX.fake_quant_fp4_ue8m0(q, 32)
        w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
        lens = ((qpos + 1) // idx.ratio)[:, None]
        return q.astype(mx.float32), w.astype(mx.float32), lens

    def test_remainder_edges_and_boundary_tie(self) -> None:
        """nb = cap+1, 2cap+1, 3cap-1 and a tail < align (k*cap + 3), each with a
        targeted equal-score pair straddling the first tile seam."""
        for role in ("source", "consumer", "plain"):
            for n in (1, 4, 8, 16):
                c = cap_of(n, 8)
                for nb in (c + 1, 2 * c + 1, 3 * c - 1, 2 * c + 3):
                    if nb <= 128:
                        continue
                    rec = self._run_arms(role, n, nb, "random", 3000 + n + nb,
                                         boundary_tie=True)
                    self._assert_identical(rec)


# --------------------------------------------------------------------------
# 4. D3: a band of a small-n chunk keys on the FULL chunk
# --------------------------------------------------------------------------
class FakeGroup:
    def __init__(self, rank: int, world: int = 2):
        self._rank, self._world = rank, world

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._world


class D3KeyingTest(_Fixture):
    def test_band_uses_full_chunk_width(self) -> None:
        self.shrink()
        IX._SMALLN_WIDE_TILE = True
        n_full = 4
        cap4 = cap_of(n_full, 8)
        nb = cap4 + 1                    # full chunk TILED at cap4 ...
        # ... while a 2-row band routed on its OWN rows would be UNTILED (D3 window)
        self.assertLess(IX.tiled(1, n_full, H, nb, 8), nb)
        self.assertGreaterEqual(IX.tiled(1, n_full // 2, H, nb, 8), nb)
        for hier in (True, False):
            IX._HIER = hier
            args = _args()
            src = _indexer(args, 0, 71)
            cons = _indexer(args, 1, 72)
            x, qr, sp, fv, ik = _inputs(73, args, n=n_full, nb=nb)

            def boom(*_a, **_k):
                raise AssertionError("band=None must never issue a collective")

            IX._coll.all_sum = boom
            with _TileSpy() as spy_full:
                sh = SharedState()
                full = np.array(src(x, qr, sp, 0, fv, ik, sh))
                fcons = np.array(cons(x, qr, sp, 0, fv, ik, sh))
            IX._coll.all_sum = lambda t, group=None: t
            recon, rcons = np.zeros_like(full), np.zeros_like(fcons)
            with _TileSpy() as spy_band:
                for r in range(2):
                    lo, hi = r * 2, (r + 1) * 2
                    sh = SharedState()
                    recon += np.array(src(x, qr, sp, 0, fv, ik, sh, band=(lo, hi),
                                          group=FakeGroup(r)))
                    rcons += np.array(cons(x, qr, sp, 0, fv, ik, sh, band=(lo, hi),
                                           group=FakeGroup(r)))
            IX._coll.all_sum = self._real_all_sum
            src_rows = int((recon != full).any(-1).sum())
            cons_rows = int((rcons != fcons).any(-1).sum())
            rec = {"case": "d3_small_n_band", "hier": hier, "n_full": n_full,
                   "nb": nb, "full_tiles": spy_full.tiles, "band_tiles": spy_band.tiles,
                   "src_diff_rows": src_rows, "cons_diff_rows": cons_rows}
            record(rec)
            # source aligns to 8 (cap4), consumer to 1 (unaligned cap)
            self.assertEqual(spy_full.tiles, [cap4, cap_of(n_full, 1)])
            self.assertEqual(spy_band.tiles, spy_full.tiles * 2,
                             "band must route/width on n_full, not its own rows")
            self.assertEqual(src_rows, 0)
            self.assertEqual(cons_rows, 0)


if __name__ == "__main__":
    unittest.main()
