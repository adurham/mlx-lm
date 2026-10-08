# Copyright © 2026 Adam Durham (hermes-gw)
"""Exhaustive INDEX-IDENTITY proof suite for the "lever-2" row-count guard.

WHAT IS UNDER TEST (frozen; this file does NOT edit any source)
--------------------------------------------------------------
``mlx_lm/models/deepseek_v41/indexer.py::Indexer.__call__`` now branches::

    if _HIER and n > _FENCE_MIN_ROWS:      # was: `if _HIER:`
        ... hierarchical / streamed exact pass ...
    # else: tiled / untiled FALLBACK (exact-scores ALL columns)

with ``n = x.shape[1]`` (the forward's query-row count; == the ``m`` lever-1
guards on) and ``_FENCE_MIN_ROWS`` the single shared symbol read from
``_gates`` (default 16) and also imported by ``sparse_attention.py``.

WHY AN IDENTITY PROOF IS MANDATORY
----------------------------------
The hierarchical path ranks blocks by their **bf16 coarse** maxima and then
exact-rescores (fp32) only the top ``k + overfetch`` blocks; the fallback
exact-scores **every** column. Both are correct top-k *value* selectors, but the
returned *indices* (and the ``-1`` mask / position order) coincide only if no
column that belongs to the true top-k lives in a dropped block. ``overfetch=16``
is a HEURISTIC margin against bf16 coarse mis-ranking, not a proof. If the two
paths disagree at small ``n``, then shipping the guard changes the decode
(n=1) / verify (n=4) output versus the shipped next17 build -> ABORT.

METHOD
------
The two paths are compared at the level of the ``[b, n, k]`` int32 tensor
returned by ``Indexer.__call__`` (so the ``+offset`` remapping, the ``-1`` mask
pattern AND the position order are all exercised end-to-end). The guard is
forced by monkeypatching the module constant, exactly as the task prescribes::

    I._FENCE_MIN_ROWS = -1   # force HIER for all n  -> path_hier
    I._FENCE_MIN_ROWS = 16   # shipped default       -> n<=16 fallback

Any elementwise int32 mismatch (a differing index, a differing ``-1`` slot, or a
differing slot ORDER) counts as a DIVERGENCE. Ties count as divergence.

SABOTAGE RED/GREEN
------------------
A positive control proves the comparator itself detects a known difference: a
GREEN (wide-margin, well-separated) case must PASS, and two deliberate
sabotages -- ``overfetch=0`` on a bf16 near-tie target, and a coarse-pass
corruption -- must make cases FAIL.

Run (mlx-lm worktree root; ~/repos/exo/.venv/bin/python has GPU mlx)::

    PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python -m pytest \\
        tests/test_dsv41_indexer_smallm_hier.py -q
    PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python tests/test_dsv41_indexer_smallm_hier.py

The module prints one JSON record per case and a final ``DSV41_SMALLM_HIER``
verdict line with exact totals (cases / equal / divergent).
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as H
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import SharedState

RESULTS: list[dict[str, Any]] = []
#: default threshold as read at import (16 unless the env overrode it).
THRESH = I._FENCE_MIN_ROWS

_KNOBS = ("_FENCE_MIN_ROWS", "_HIER", "_HIER_BLOCK", "_HIER_STRIP",
          "_HIER_OVERFETCH", "_HIER_EXACT_STRIP", "_HIER_CONSUMER_SKIP",
          "_ROW_DTYPE", "_TILE", "_TILE_MIN_NB", "_TILE_FORCE")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def record(row: dict[str, Any]) -> None:
    RESULTS.append(row)
    print("  " + json.dumps(row), flush=True)


def _args(**over) -> ModelArgs:
    base: dict[str, Any] = dict(
        dim=64, n_layers=3, n_heads=4, head_dim=128, rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
        compress_ratios=(2, 2, 2), kv_source_layers=(0,), index_source_layers=(0,),
        index_n_heads=2, index_head_dim=32, index_topk=512, max_seq_len=4096,
        candidate_source_layer=0, candidate_topk_blocks=4, candidate_block_size=8)
    base.update(over)
    return ModelArgs(**base)


def _make_indexer(args: ModelArgs, layer_id: int, seed: int = 0) -> I.Indexer:
    mx.random.seed(seed)
    idx = I.Indexer(args, layer_id)
    mx.eval(idx.parameters())
    return idx


def _inputs(seed: int, args: ModelArgs, *, b: int = 1, n: int = 8, nb: int = 120,
            sp: int | None = None):
    """Random x/qr/index_k/freqvec; ``sp`` (start_pos) drives the causal ``lens``.

    ``sp=None`` -> every column visible. ``sp`` small -> ``lens`` spans only the
    first few compressed groups, so most columns (and whole blocks) are
    ``-inf``/invisible -- the fully-masked stressor.
    """
    rng = np.random.default_rng(seed)
    if sp is None:
        sp = 2 * nb                                # every group visible
    x = mx.array(rng.standard_normal((b, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((b, n, args.q_lora_rank)).astype(np.float32))
    ik = mx.array(rng.standard_normal((b, nb, args.index_head_dim)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    return x, qr, sp, fv, ik


# --------------------------------------------------------------------------
# path spy: count how many times the HIER entry point is actually entered
# --------------------------------------------------------------------------
class _PathSpy:
    def __init__(self):
        self.calls = 0
        self._real = H.hierarchical_topk_prod

        def wrapper(*a, **k):
            self.calls += 1
            return self._real(*a, **k)

        self._wrapper = wrapper

    def install(self):
        H.hierarchical_topk_prod = self._wrapper
        return self

    def restore(self):
        H.hierarchical_topk_prod = self._real


def _run(idx: I.Indexer, inp, shared: SharedState | None = None):
    x, qr, sp, fv, ik = inp
    sh = shared if shared is not None else SharedState()
    out = idx(x, qr, sp, 0, fv, ik, sh)
    mx.eval(out)
    if sh.candidates is not None:
        mx.eval(sh.candidates)
    return np.array(out), sh


# --------------------------------------------------------------------------
# base fixture
# --------------------------------------------------------------------------
class _Fixture(unittest.TestCase):
    """Save/restore every module constant we touch, plus the path spy."""

    def setUp(self):
        self._saved = {n: getattr(I, n) for n in _KNOBS}
        self._spy = _PathSpy().install()

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(I, n, v)
        self._spy.restore()

    # --- one comparison of the two paths at a given cell -------------------
    def compare(self, role: str, n: int, nb: int, k: int, seed: int, *,
                sp: int | None = None, layer_id: int = 0, compare_mask: bool = False,
                consumer_skip: bool | None = None) -> bool:
        args = _args(candidate_source_layer=(0 if role != "plain" else -1))
        idx = _make_indexer(args, layer_id, seed=len(str(nb)))
        idx.index_topk = int(k)
        inp = _inputs(seed, args, n=n, nb=nb, sp=sp)

        if consumer_skip is not None:
            I._HIER_CONSUMER_SKIP = bool(consumer_skip)

        # --- default guard (n<=THRESH -> fallback) ---
        I._FENCE_MIN_ROWS = THRESH
        self._spy.calls = 0
        out_d, sh_d = _run(idx, inp)
        path_d = self._spy.calls

        # --- forced HIER ---
        I._FENCE_MIN_ROWS = -1
        self._spy.calls = 0
        out_h, sh_h = _run(idx, inp)
        path_h = self._spy.calls

        eq = bool(np.array_equal(out_d, out_h))
        ndiff = int((out_d != out_h).sum())
        mask_eq = None
        if compare_mask:
            cd, ch = sh_d.candidates, sh_h.candidates
            if cd is None or ch is None:
                mask_eq = (cd is None and ch is None)
            else:
                mask_eq = bool(np.array_equal(np.array(cd), np.array(ch)))

        want_path = 1 if n > THRESH else 0        # default-guard path expectation
        row = {"role": role, "n": n, "nb": nb, "k": k, "seed": seed,
               "equal": eq, "ndiff": ndiff,
               "path_default": path_d, "path_hier": path_h,
               "path_ok": (path_d == want_path)}
        if compare_mask:
            row["mask_equal"] = mask_eq
        if ndiff:
            pos = np.argwhere(out_d != out_h)[:8].tolist()
            row["first_diff_pos"] = pos
        record(row)
        return eq and (mask_eq is not False) and row["path_ok"]


# --------------------------------------------------------------------------
# 1. the plain exact-top-k role, full grid
# --------------------------------------------------------------------------
class IdentityPlainGrid(_Fixture):
    N = (1, 2, 3, 4, 16, 17)
    NB = (64, 512, 4096, 16384)
    SEEDS = 24

    def _klist(self, nb: int):
        return (nb,) if nb < 512 else (511, 512, 513)

    def test_plain_grid(self):
        bad = []
        for n in self.N:
            for nb in self.NB:
                for k in self._klist(nb):
                    for seed in range(self.SEEDS):
                        s = 90000 + n * 977 + nb * 17 + k * 3 + seed
                        ok = self.compare("plain", n, nb, k, s)
                        if not ok:
                            bad.append((n, nb, k, s))
        self.assertFalse(bad, f"{len(bad)} divergent/bad cells in plain grid; "
                              f"first: {bad[:6]}")


# --------------------------------------------------------------------------
# 2. causal / fully-masked rows and blocks
# --------------------------------------------------------------------------
class IdentityMasking(_Fixture):
    def test_heavily_masked_and_all_masked_rows(self):
        """start_pos drives ``lens``; small sp masks whole rows and blocks.

        ratio=2 so ``lens = (sp + i + 1)//2``. ``sp=0`` -> the first query rows
        have ``lens=0`` (every column invisible -> an all ``-1`` row); a mid ``sp``
        masks most blocks; ``sp=2*nb`` is fully visible. Every arm must be
        path-identical.
        """
        bad = []
        for n in (4, 16):
            for nb in (512, 4096, 16384):
                for tag, sp in (("all_masked_rows", 0),
                                ("mostly_masked", max(0, nb // 32 - n)),
                                ("half_visible", nb // 2),
                                ("full_visible", 2 * nb)):
                    for seed in range(8):
                        s = 30000 + n * 71 + nb + seed
                        ok = self.compare("plain", n, nb, 64, s, sp=sp)
                        if not ok:
                            bad.append((n, nb, 64, tag, s))
        self.assertFalse(bad, f"{len(bad)} divergent masked cells; first {bad[:6]}")


# --------------------------------------------------------------------------
# 3. candidate-source role (layer 20): published mask must match
# --------------------------------------------------------------------------
class IdentityCandidateSource(_Fixture):
    def test_source_mask_and_indices(self):
        bad = []
        for n in (1, 4, 16, 17):
            for nb in (64, 512, 4096, 16384):
                for seed in range(8):
                    s = 51000 + n * 313 + nb + seed
                    ok = self.compare("source", n, nb, 32, s, compare_mask=True)
                    if not ok:
                        bad.append((n, nb, 32, s))
        self.assertFalse(bad, f"{len(bad)} divergent source cells; first {bad[:6]}")


# --------------------------------------------------------------------------
# 4. consumer role (layers 24..36): cand_mask set, consumer_skip ON and OFF
# --------------------------------------------------------------------------
def _block_mask(rng, b, n, nb, block, n_keep, zero_rows=()):
    NB = -(-nb // block)
    keep = np.zeros((b, n, NB), bool)
    for bi in range(b):
        for r in range(n):
            keep[bi, r, rng.choice(NB, size=min(n_keep, NB), replace=False)] = True
    for r in zero_rows:
        keep[:, r] = False
    return mx.array(np.repeat(keep, block, axis=-1)[..., :nb])


class IdentityConsumer(_Fixture):
    def test_consumer_with_precomputed_mask(self):
        """Consumer path with an externally-set shared.candidates, skip on/off."""
        bad = []
        for n in (1, 4, 16, 17):
            for nb in (512, 4096):
                cmask = _block_mask(np.random.default_rng(4321), 1, n, nb, 8,
                                    max(1, nb // 16), zero_rows=(
                                        0,) if n > 1 else ())
                for k in (64, 128):
                    for skip in (True, False):
                        for seed in range(8):
                            s = 77000 + n * 211 + nb + k + seed
                            args = _args(candidate_source_layer=0)
                            idx = _make_indexer(args, 1, seed=7)   # consumer layer
                            idx.index_topk = k
                            inp = _inputs(s, args, n=n, nb=nb)
                            I._HIER_CONSUMER_SKIP = skip

                            sh = SharedState()
                            sh.candidates = cmask
                            I._FENCE_MIN_ROWS = THRESH
                            out_d, _ = _run(idx, inp, sh)
                            I._FENCE_MIN_ROWS = -1
                            sh2 = SharedState()
                            sh2.candidates = cmask
                            out_h, _ = _run(idx, inp, sh2)
                            eq = bool(np.array_equal(out_d, out_h))
                            ndiff = int((out_d != out_h).sum())
                            record({"role": "consumer", "n": n, "nb": nb, "k": k,
                                    "skip": skip, "seed": seed, "equal": eq,
                                    "ndiff": ndiff})
                            if not eq:
                                bad.append((n, nb, k, skip, s))
        self.assertFalse(bad, f"{len(bad)} divergent consumer cells; first {bad[:6]}")


# --------------------------------------------------------------------------
# 5. deliberate bf16 near-tie mis-ranks + overfetch-boundary placement
# --------------------------------------------------------------------------
def _one_hot_keys(target: np.ndarray):
    """(q, index_k, w) whose fp32 score row equals ``target`` [n, nb]."""
    n, nb = target.shape
    q = np.zeros((1, n, 1, n), np.float32)
    q[0, :, 0, :] = np.eye(n, dtype=np.float32)
    kk = np.zeros((1, nb, n), np.float32)
    kk[0, :, :] = target.T
    w = np.ones((1, n, 1), np.float32)
    return mx.array(q), mx.array(kk).astype(mx.bfloat16), mx.array(w)


def _exact_row_topk(index_k, k):
    row = mx.swapaxes(index_k.astype(mx.float32), 1, 2)     # [b, n, nb]
    v, i = I.topk_from_row(row, k)
    mx.eval(v, i)
    return np.array(i)


class TieAndBoundary(_Fixture):
    """Engineered bf16 near-tie and overfetch-boundary, at the function level."""

    def _alias_target(self, n, nb, n_hot_blocks, gap=0.004, seed=0, scale=100.0):
        """Per-block single hot column, fp32-distinct but bf16-aliased magnitudes.

        At scale=100 the bf16 ULP is ~0.5, so ``gap=0.004`` magnitudes collapse to
        the SAME bf16 value: the coarse pass sees block-max ties and its
        argpartition tie order is arbitrary -> a true-top-k block can be pruned.
        """
        rng = np.random.default_rng(seed)
        target = (rng.random((n, nb)) * 0.05).astype(np.float32)
        blk = I._HIER_BLOCK
        cols = np.arange(n_hot_blocks) * blk + (blk - 1)
        target[:, cols] = (scale - np.arange(n_hot_blocks) * gap)[None, :]
        return target

    def test_bf16_near_tie_blocks_and_overfetch_boundary(self):
        n, nb, k = 16, 1024, 64           # 128 blocks; k+of = 80 blocks kept
        block = I._HIER_BLOCK
        for seed in range(6):
            # one hot column in each of the first (k+of+8) blocks -> the cut sits
            # inside a tied region; block #(k+of) genuinely holds a top-k column.
            n_hot = k + I._HIER_OVERFETCH + 8
            target = self._alias_target(n, nb, n_hot, seed=seed)
            q, kk, w = _one_hot_keys(target)
            lens = mx.full((n, 1), nb, mx.int32)
            ri = _exact_row_topk(kk, k)

            for tag, overfetch in (("of=0", 0), ("of=16", 16), ("of=all", 1 << 30)):
                hv, hi, _ = H.hierarchical_topk_prod(
                    q, kk, w, lens, k, block, 1024, 128, overfetch)
                mx.eval(hv, hi)
                ndiff = int((np.array(hi) != ri).sum())
                record({"role": "tie_boundary", "n": n, "nb": nb, "k": k,
                        "seed": seed, "overfetch": tag, "ndiff_vs_exact": ndiff})
                if tag == "of=16":
                    # The shipped margin is what the guard ships: it must equal
                    # the exact production reference, or the guard is unsound.
                    self.assertEqual(ndiff, 0,
                                     f"shipped overfetch=16 loses {ndiff} of "
                                     f"{ri.size} exact slots (seed={seed})")


# --------------------------------------------------------------------------
# 6. path-taken boundary assertions (16 -> fallback, 17 -> HIER)
# --------------------------------------------------------------------------
class PathBoundary(_Fixture):
    def test_boundary_16_fallback_17_hier(self):
        args = _args(candidate_source_layer=-1)
        idx = _make_indexer(args, 0, seed=0)
        idx.index_topk = 64
        for n, want in ((1, 0), (4, 0), (15, 0), (16, 0), (17, 1), (18, 1), (32, 1)):
            x, qr, sp, fv, ik = _inputs(100 + n, args, n=n, nb=2048)
            I._FENCE_MIN_ROWS = THRESH
            self._spy.calls = 0
            mx.eval(idx(x, qr, sp, 0, fv, ik, SharedState()))
            got = self._spy.calls
            record({"case": f"path_boundary/n{n}", "hier_calls": got, "want": want})
            self.assertEqual(got, want,
                             f"n={n}: default path took {'HIER' if got else 'fallback'} "
                             f"(want {'HIER' if want else 'fallback'})")
        # n=17 must ALSO be HIER when forced, and n=16 must take fallback when
        # the guard is at its shipped 16 even with the feature ON.
        self.assertTrue(I._HIER, "feature default must be ON for this proof")


# --------------------------------------------------------------------------
# 7. sabotage RED/GREEN self-check
# --------------------------------------------------------------------------
class RedGreen(_Fixture):
    def test_comparator_positive_control(self):
        """The comparator MUST flag two different index tensors as unequal."""
        a = np.zeros((1, 4, 6), np.int32)
        b = np.zeros((1, 4, 6), np.int32)
        b[0, 0, 5] = 7                     # a single differing slot
        self.assertFalse(np.array_equal(a, b), "comparator failed a 1-slot diff")
        record({"case": "redgreen/comparator_control", "detects_1slot_diff": True})

    def test_green_wide_margin_passes(self):
        """GREEN: a well-separated case must take IDENTICAL paths (no divergence)."""
        n, nb, k, block = 16, 1024, 64, I._HIER_BLOCK   # 128 blocks, k+of=80 kept
        rng = np.random.default_rng(0)
        target = (rng.random((n, nb)) * 0.01).astype(np.float32)
        target[:, :k] = 10.0                       # k unconditional top columns
        q, kk, w = _one_hot_keys(target)
        lens = mx.full((n, 1), nb, mx.int32)
        ri = _exact_row_topk(kk, k)
        hv, hi, _ = H.hierarchical_topk_prod(q, kk, w, lens, k, block, 1024, 128, 16)
        mx.eval(hv, hi)
        ndiff = int((np.array(hi) != ri).sum())
        record({"case": "redgreen/green_wide_margin", "ndiff": ndiff})
        self.assertEqual(ndiff, 0, "GREEN arm must pass; comparator is broken")

    def _real_keys(self, seed, n, nb, h=8, d=64):
        rng = np.random.default_rng(seed)
        q = mx.array((rng.standard_normal((1, n, h, d)) * 0.3).astype(np.float32))
        kk = mx.array(rng.standard_normal((1, nb, d)).astype(np.float32)).astype(mx.bfloat16)
        w = mx.array((rng.random((1, n, h)) + 0.5).astype(np.float32))
        return q, kk, w

    def test_red_sabotage_overfetch_zero(self):
        """RED: forcing overfetch=0 on REAL keys (bf16 coarse mis-rank) MUST FAIL.

        With real keys the bf16 coarse accumulation genuinely reorders close
        blocks; ``overfetch=16`` recovers them, ``overfetch=0`` does not. seed 3
        is a known mis-ranking seed (7 lost exact slots at of=0, 0 at of=16).
        """
        n, nb, k, block = 1, 4096, 64, I._HIER_BLOCK
        q, kk, w = self._real_keys(3, n, nb)
        lens = mx.full((n, 1), nb, mx.int32)
        ri = _exact_row_topk(kk, k)
        hv, hi, _ = H.hierarchical_topk_prod(q, kk, w, lens, k, block, 1024, 128, 0)
        mx.eval(hv, hi)
        ndiff = int((np.array(hi) != ri).sum())
        record({"case": "redgreen/red_overfetch_zero", "sabotage": "overfetch=0",
                "ndiff": ndiff})
        self.assertGreater(ndiff, 0, "overfetch=0 sabotage was NOT detected")

    def test_red_sabotage_coarse_corruption(self):
        """RED: corrupting the coarse maxima MUST make a green case diverge.

        Requires pruning to be REAL: ``nb/block`` blocks must exceed
        ``k + overfetch`` (nb=1024, block=8 -> 128 blocks, k+of=80 kept), else
        every block is rescored exact and no coarse error can matter.
        """
        n, nb, k, block = 16, 1024, 64, I._HIER_BLOCK
        assert nb // block > k + I._HIER_OVERFETCH, "fixture: prune must be real"
        rng = np.random.default_rng(1)
        target = (rng.random((n, nb)) * 0.01).astype(np.float32)
        target[:, :k] = 10.0
        q, kk, w = _one_hot_keys(target)
        lens = mx.full((n, 1), nb, mx.int32)
        ri = _exact_row_topk(kk, k)

        real = H.coarse_block_scores

        def sabotage(*a, **kw):
            bm = real(*a, **kw)
            # wipe block 0 (holds 8 true top columns) out of the coarse ranking
            return bm.at[:, :, 0].add(-1e30)

        H.coarse_block_scores = sabotage
        try:
            hv, hi, _ = H.hierarchical_topk_prod(q, kk, w, lens, k, block, 1024, 128, 16)
            mx.eval(hv, hi)
        finally:
            H.coarse_block_scores = real
        ndiff = int((np.array(hi) != ri).sum())
        record({"case": "redgreen/red_coarse_corruption",
                "sabotage": "coarse block 0 maxima -= 1e30", "ndiff": ndiff})
        self.assertGreater(ndiff, 0, "coarse-corruption sabotage was NOT detected")


# --------------------------------------------------------------------------
# 8. verdict summary (defined last => runs last)
# --------------------------------------------------------------------------
class Summary(unittest.TestCase):
    def test_zz_verdict(self):
        total = len(RESULTS)
        diverged = [r for r in RESULTS if not r.get("equal", True)]
        mask_bad = [r for r in RESULTS if r.get("mask_equal") is False]
        path_bad = [r for r in RESULTS if r.get("path_ok") is False]

        # breakdown by (role, n) so the report can name exactly which small-m
        # cells diverge -- the ship/abort decision turns on n in {1,4}.
        by_role_n: dict[str, dict[str, int]] = {}
        for r in RESULTS:
            role = r.get("role", r.get("case", "meta"))
            nn = r.get("n")
            key = f"{role}/n={nn}" if nn is not None else str(role)
            d = by_role_n.setdefault(key, {"total": 0, "divergent": 0})
            d["total"] += 1
            if not r.get("equal", True):
                d["divergent"] += 1

        summary = {"total_cases": total, "equal": total - len(diverged),
                   "divergent": len(diverged),
                   "mask_mismatch": len(mask_bad),
                   "path_violations": len(path_bad),
                   "by_role_n": by_role_n}
        print("DSV41_SMALLM_HIER " + json.dumps(summary), flush=True)
        if diverged:
            print("  divergent cells (first 12):")
            for r in diverged[:12]:
                print("    " + json.dumps(r))
        self.assertEqual(len(diverged), 0,
                         f"INDEX IDENTITY FAILS: {len(diverged)}/{total} cases diverge "
                         f"between the fallback and the hierarchical path")
        self.assertEqual(len(mask_bad), 0, f"{len(mask_bad)} candidate-mask mismatches")


if __name__ == "__main__":
    unittest.main(verbosity=2)
    total = len(RESULTS)
    div = sum(1 for r in RESULTS if not r.get("equal", True))
    print(f"DSV41_SMALLM_HIER {json.dumps({'cases': total, 'divergent': div})}")
    sys.exit(1 if div else 0)
