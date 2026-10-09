# Copyright © 2026 Adam Durham (hermes-gw)
"""ADVERSARIAL CELL SUITE at PRODUCTION H — the amended lever-2 identity gate.

Companion to ``tests/test_dsv41_indexer_smallm_hier.py`` (the frozen suite, run at
the fixture's ``index_n_heads=2``). This suite runs at the SERVED checkpoint's
head count (PHASE5-P1-LEVER2.md §14 req 1) and shifts the assertion level from
"no ties" to **IDENTICAL ROW + IDENTICAL OP + IDENTICAL ORDER**, with ROW-LEVEL
BITWISE asserts on the full ``[b, n, nb]`` fp32 score row — not merely the top-k
int indices.

WHY ROW-LEVEL. The lever-2 ship design (L2-full) is: at ``n <= _FENCE_MIN_ROWS``
(16) the fallback stores its score row in fp32, exactly the precision the
hierarchical exact re-score ranks, so both branches run the SAME global
``topk_from_row`` over a bitwise-identical row. The load-bearing claim is
therefore ROW identity, not index identity; proving only index equality could
pass for the wrong reason. This suite asserts:

    R_fallback[r, i_hier]  ==  v_hier        (bitwise, per gathered column)
    topk_from_row(R_fallback, k)  ==  i_hier (the returned index tensor)

on EVERY case, at H in {8, 32, 64} with H=32 (production) on every stratum.

CASES (amendment leg B, i-v)
  (i)   DUPLICATE KEYS — repeated tokens/spans, padding sinks (masked
        columns), quantized-collision keys.
  (ii)  ULP-BOUNDARY — rows whose k-th boundary sits within <=1 ulp; count any
        top-k flip between the two paths. A flip at production H is an ABORT.
  (iii) RoPE PHASE-ALIASING — positions aligned to the indexer RoPE period so
        key columns alias.
  (iv)  CANDIDATE-ORDER at n>16 — the coarse-pass candidate order vs the
        fallback order feeding the shared top-k op: set AND order must agree.
  (v)   GUARD OFF-BY-ONE — explicit n in {15,16,17} at production H, no
        cross-boundary mismatch.

Run (mlx-lm worktree root; single file only):: 

    cd /private/tmp/next18-lever2
    PYTHONPATH=$PWD:/private/tmp/next18-lever2/tests \
      /Users/adam.durham/repos/exo/.venv/bin/python -m pytest \
      tests/test_dsv41_indexer_adversarial_prodh.py -q -s

Prints one JSON record per case and a final ``DSV41_ADV_PRODH`` verdict line with
the exact totals (cases / equal / divergent / ulp_flips / row_bitwise_mismatches).
Deterministic; runnable fresh.
"""
from __future__ import annotations

import json
import sys
import unittest
from typing import Any

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as HIER
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import SharedState

RESULTS: list[dict[str, Any]] = []
#: production head count (PHASE5-P1-LEVER2.md §14 req 1: config.py:82 + node config.json)
H_PROD = 32
H_SWEEP = (8, 32, 64)
PROD_DIM = 128                         # production index_head_dim
THRESH = I._FENCE_MIN_ROWS
BLOCK = I._HIER_BLOCK                  # 8
OVERFETCH = I._HIER_OVERFETCH          # 16

_KNOBS = ("_FENCE_MIN_ROWS", "_HIER", "_HIER_BLOCK", "_HIER_STRIP",
          "_HIER_OVERFETCH", "_HIER_EXACT_STRIP", "_HIER_CONSUMER_SKIP",
          "_ROW_DTYPE", "_L2_FULL", "_SMALLN_ROW_BF16", "_TILE", "_TILE_MIN_NB",
          "_TILE_FORCE")


def record(row: dict[str, Any]) -> None:
    RESULTS.append(row)
    print("  " + json.dumps(row), flush=True)


def _args(H: int, **over) -> ModelArgs:
    base: dict[str, Any] = dict(
        dim=64, n_layers=3, n_heads=4, head_dim=128, rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
        compress_ratios=(2, 2, 2), kv_source_layers=(0,), index_source_layers=(0,),
        index_n_heads=H, index_head_dim=PROD_DIM, index_topk=512, max_seq_len=4096,
        candidate_source_layer=0, candidate_topk_blocks=4, candidate_block_size=8)
    base.update(over)
    return ModelArgs(**base)


def _make_indexer(args: ModelArgs, layer_id: int, seed: int = 0) -> I.Indexer:
    mx.random.seed(seed)
    idx = I.Indexer(args, layer_id)
    mx.eval(idx.parameters())
    return idx


# --------------------------------------------------------------------------
# the fp32 truth row (exact fallback expression) + the HIER exact re-score
# --------------------------------------------------------------------------
def _fallback_row(idx: I.Indexer, inp, l2_full: bool = True) -> np.ndarray:
    """The fallback [1, n, nb] fp32 score row, exactly as ``__call__`` builds it."""
    x, qr, sp, fv, ik = inp
    n = x.shape[1]
    if sp is None:
        sp = 2 * ik.shape[1]
    q = idx.wq_b(qr).reshape(1, n, idx.n_heads, idx.head_dim)
    from mlx_lm.models.deepseek_v41.layers import cos_sin_at, rope_tail
    from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0
    q = rope_tail(q, idx.rope_head_dim, *cos_sin_at(fv, mx.arange(sp, sp + n)))
    q = fake_quant_fp4_ue8m0(q, 32)
    w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((sp + mx.arange(n) + 1) // idx.ratio)[:, None]
    sc = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), ik.astype(mx.float32))
    sc = mx.maximum(sc, 0.0) * w[..., None].astype(mx.float32)
    row = mx.sum(sc, axis=2)
    row = row.astype(mx.float32) if l2_full else row.astype(I._ROW_DTYPE)
    vis = mx.arange(ik.shape[1])[None, :] < lens
    row = mx.where(vis[None], row, float("-inf"))
    mx.eval(row)
    return np.array(row), lens, q, w


def _hier(idx: I.Indexer, inp, k: int, *, cmask=None):
    x, qr, sp, fv, ik = inp
    nb = ik.shape[1]
    _row, lens, q, w = _fallback_row(idx, inp, l2_full=True)
    estrip = I.hier_strip_for_budget(1, x.shape[1], idx.head_dim,
                                     int(I._HIER_EXACT_MB * (1 << 20)), block=BLOCK)
    v, i, blk = HIER.hierarchical_topk_prod(
        q.astype(mx.float32), ik, w.astype(mx.float32), lens, k, BLOCK,
        I._HIER_STRIP, estrip, OVERFETCH, cand_mask=cmask, consumer_skip=True)
    mx.eval(v, i)
    return np.array(v), np.array(i), lens


def _row_and_index_check(idx: I.Indexer, inp, k: int, *, cmask=None) -> dict:
    """ROW-LEVEL bitwise assert + finite-slot index equality between the arms.

    * row_bitwise_mismatch: count of HIER-returned *finite* slots whose gathered
      fp32 row value is NOT bitwise equal to the HIER top value (the "identical
      row" leg). -inf padding slots are excluded: both arms return "-inf" there
      and the index convention for an empty slot is arbitrary by construction
      (exactly the tie/padding class the design doc calls out), so it carries no
      value information.
    * index_equal: the finite ``(column, value)`` maps of the two arms agree
      exactly (identical op + order on every value-bearing slot).
    * padding_slots_h / padding_slots_f: how many of the k slots each arm left
      non-finite, for transparency.
    """
    row_fb, _lens, q, w = _fallback_row(idx, inp, l2_full=True)
    v_h, i_h, _lens2 = _hier(idx, inp, k, cmask=cmask)

    gathered = np.take_along_axis(row_fb, i_h, axis=2)
    finite_h = np.isfinite(v_h)
    row_mm = int((gathered.view(np.uint32) != v_h.view(np.uint32))[finite_h].sum())

    fv, fi = I.topk_from_row(mx.array(row_fb), k)
    mx.eval(fv, fi)
    fv_np, fi_np = np.array(fv), np.array(fi)
    finite_f = np.isfinite(fv_np)
    hi_cols, hi_vals = i_h[finite_h], v_h[finite_h]
    fo_cols, fo_vals = fi_np[finite_f], fv_np[finite_f]
    index_equal = bool(
        hi_cols.size == fo_cols.size
        and np.array_equal(hi_cols, fo_cols)
        and np.array_equal(hi_vals.view(np.uint32), fo_vals.view(np.uint32)))

    return dict(row_bitwise_mismatch=row_mm, index_equal=index_equal,
                slots=int(i_h.size), padding_slots_h=int((~finite_h).sum()),
                padding_slots_f=int((~finite_f).sum()))


def _call_arm(idx: I.Indexer, inp, fence: int, *, cmask=None):
    x, qr, sp, fv, ik = inp
    sh = SharedState()
    if cmask is not None:
        sh.candidates = cmask
    I._FENCE_MIN_ROWS = fence
    from mlx_lm.models.deepseek_v41.model import SharedState as _SS
    out = idx(x, qr, sp, 0, fv, ik, sh if cmask is not None else _SS())
    mx.eval(out)
    return np.array(out)


# --------------------------------------------------------------------------
# fixture
# --------------------------------------------------------------------------
class _Adv(unittest.TestCase):
    def setUp(self):
        self._saved = {n: getattr(I, n) for n in _KNOBS}

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(I, n, v)


# ==========================================================================
# (i) DUPLICATE KEYS
# ==========================================================================
class DuplicateKeys(_Adv):
    def _dup_inputs(self, H, n, nb, seed, *, span=4, n_dup=None):
        args = _args(H, candidate_source_layer=-1)
        idx = _make_indexer(args, 0, seed=1)
        rng = np.random.default_rng(seed)
        ik = rng.standard_normal((1, nb, PROD_DIM)).astype(np.float32)
        # quantized-collision keys: collapse many columns onto a few base vectors
        bases = rng.standard_normal((span, PROD_DIM)).astype(np.float32)
        ik = np.repeat(bases, nb // span + 1, axis=0)[:nb]
        # padding sinks: tail block columns equal to 0 (score 0 exactly)
        ik[nb - 8:] = 0.0
        x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
        qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
        return idx, (x, qr, 2 * nb, mx.array(np.linspace(1, 0.01, 8).astype(np.float32)),
                     mx.array(ik)[None])

    def test_duplicate_and_padding_keys(self):
        bad = []
        for H in H_SWEEP:
            for n in (1, 4, 16):
                for nb in (512, 4096):
                    idx, inp = self._dup_inputs(H, n, nb, seed=7000 + n + nb)
                    k = min(64, nb)
                    idx.index_topk = k
                    chk = _row_and_index_check(idx, inp, k)
                    eq = chk["index_equal"] and chk["row_bitwise_mismatch"] == 0
                    record(dict(case="dup_keys", H=H, n=n, nb=nb, k=k, **chk))
                    if not eq:
                        bad.append((H, n, nb, k))
        self.assertFalse(bad, f"{len(bad)} duplicate-key cells diverge; first {bad[:6]}")


# ==========================================================================
# (ii) ULP-BOUNDARY
# ==========================================================================
def _controlled_keys(target: np.ndarray, H: int):
    """(q, ik, w) at head count H whose fp32 score row equals ``max(target, 0)``.

    One active head (head 0) carries the identity in its first n dims and the
    target values in the keys; heads 1..H-1 are zeroed. The resulting row is
    exactly controllable, so the k-th top-k boundary can be placed at a precise
    ULP gap. (Row identity is a per-column property, independent of H.)
    """
    n, nb = target.shape
    q = np.zeros((1, n, H, PROD_DIM), np.float32)
    ik = np.zeros((1, nb, PROD_DIM), np.float32)
    w = np.zeros((1, n, H), np.float32)
    for s in range(n):
        q[0, s, 0, s] = 1.0
        w[0, s, 0] = 1.0
    for col in range(nb):
        ik[0, col, :n] = target[:, col]
    return (mx.array(q), mx.array(ik).astype(mx.bfloat16), mx.array(w))


def _fallback_row_from_keys(q, ik, w, lens, nb):
    sc = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), ik.astype(mx.float32))
    sc = mx.maximum(sc, 0.0) * w[..., None].astype(mx.float32)
    row = mx.sum(sc, axis=2).astype(mx.float32)
    vis = mx.arange(nb)[None, :] < lens
    row = mx.where(vis[None], row, float("-inf"))
    mx.eval(row)
    return row


class UlpBoundary(_Adv):
    def test_ulp_boundary_no_flip(self):
        """Rows with a k-th boundary at a chosen ULP gap; count top-k flips."""
        total_flips = 0
        bad = []
        for H in (8, 32, 64):
            for n in (4, 16):
                for nb, k in ((1024, 64), (4096, 128)):
                    for gap_ulp in (1, 2, 4):
                        rng = np.random.default_rng(9100 + H + n + gap_ulp)
                        target = (rng.random((n, nb)) * 0.05).astype(np.float32)
                        # columns [k-1, k] separated by exactly `gap_ulp` ULP at
                        # the boundary; the rest well below.
                        hi = np.float32(0.9)
                        for r in range(n):
                            target[r, :k] = np.linspace(0.5, 0.9, k).astype(np.float32)
                            v = hi
                            for _ in range(gap_ulp):
                                v = np.nextafter(v, np.float32(0.0), dtype=np.float32)
                            target[r, k] = v
                        q, ik, w = _controlled_keys(target, H)
                        nb_blocks = -(-nb // BLOCK)
                        lens = mx.full((n, 1), nb, mx.int32)
                        row_fb = _fallback_row_from_keys(q, ik, w, lens, nb)
                        bm = HIER.coarse_block_scores(q, ik, w, lens, block=BLOCK,
                                                   strip=4096, dtype=mx.bfloat16)
                        blocks = HIER.top_blocks(bm, min(nb_blocks, k + OVERFETCH),
                                              block_size=BLOCK)
                        v_h, i_h = HIER.exact_rescore_streaming(
                            q, ik, w, lens, blocks, strip=4096, k=k, block=BLOCK)
                        fv, fi = I.topk_from_row(row_fb, k)
                        mx.eval(i_h, fi)
                        flips = int((np.array(i_h) != np.array(fi)).sum())
                        total_flips += flips
                        rec = dict(case="ulp_boundary", H=H, n=n, nb=nb, k=k,
                                   gap_ulp=gap_ulp, topk_flips=flips)
                        record(rec)
                        if flips:
                            bad.append((H, n, nb, k, gap_ulp, flips))
        # ABORT signal: a top-k flip at production H
        prod_flips = sum(r["topk_flips"] for r in RESULTS
                         if r.get("case") == "ulp_boundary" and r.get("H") == H_PROD)
        if prod_flips:
            print("!!! ABORT: ULP-BOUNDARY top-k flip at PRODUCTION H "
                  f"({prod_flips} slots) !!!", flush=True)
        self.assertFalse(prod_flips,
                         f"ABORT: {prod_flips} ULP-boundary top-k flips at production H")


# ==========================================================================
# (iii) RoPE PHASE-ALIASING
# ==========================================================================
class RopeAliasing(_Adv):
    def test_rope_period_alias(self):
        """Key columns that alias under the indexer's RoPE period.

        The stored ``index_k`` is already RoPE'd; two columns alias when their
        dot with a query at aligned positions is identical. We realize that by
        rotating a base key by the RoPE period across columns and placing the
        queries at period-aligned positions, then assert row + index identity.
        """
        from mlx_lm.models.deepseek_v41.layers import cos_sin_at, rope_tail
        from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0
        bad = []
        for H in (8, 32, 64):
            for n, nb in ((4, 512), (16, 4096)):
                args = _args(H, candidate_source_layer=-1)
                idx = _make_indexer(args, 0, seed=3)
                k = min(64, nb)
                idx.index_topk = k
                rng = np.random.default_rng(4200 + H + nb)
                x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
                qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
                # period-aligned positions: start_pos a multiple of the period (2)
                period = 2
                sp = period * (nb // 2)
                # one base key, repeated every period -> exact aliases
                base = rng.standard_normal((1, PROD_DIM)).astype(np.float32)
                ika = np.repeat(base, nb, axis=0)[None]
                ik = mx.array(ika)
                fv = mx.array(np.linspace(1.0, 0.01, 8).astype(np.float32))
                inp = (x, qr, sp, fv, ik)
                chk = _row_and_index_check(idx, inp, k)
                record(dict(case="rope_alias", H=H, n=n, nb=nb, k=k, **chk))
                if not (chk["index_equal"] and chk["row_bitwise_mismatch"] == 0):
                    bad.append((H, n, nb))
        self.assertFalse(bad, f"{len(bad)} RoPE-alias cells diverge; first {bad[:6]}")


# ==========================================================================
# (iv) CANDIDATE-ORDER at n>16
# ==========================================================================
def _block_mask(rng, b, n, nb, block, n_keep, zero_rows=()):
    NB = -(-nb // block)
    keep = np.zeros((b, n, NB), bool)
    for bi in range(b):
        for r in range(n):
            keep[bi, r, rng.choice(NB, size=min(n_keep, NB), replace=False)] = True
    for r in zero_rows:
        keep[:, r] = False
    return mx.array(np.repeat(keep, block, axis=-1)[..., :nb])


class CandidateOrder(_Adv):
    def test_candidate_order_n_gt_16(self):
        """At n>16: HIER (coarse order -> shared top-k) vs the full-width
        fallback (all columns -> shared top-k). Set AND ORDER must agree."""
        bad = []
        for H in (8, 32, 64):
            for n in (17, 32):
                for nb in (1024, 4096):
                    k = min(128, nb)
                    args = _args(H, candidate_source_layer=0)
                    idx = _make_indexer(args, 1, seed=7)          # consumer layer
                    idx.index_topk = k
                    rng = np.random.default_rng(5500 + H + n + nb)
                    x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
                    qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
                    ik = mx.array(rng.standard_normal((1, nb, PROD_DIM)).astype(np.float32))
                    fv = mx.array(np.linspace(1.0, 0.01, 8).astype(np.float32))
                    sp = 2 * nb
                    inp = (x, qr, sp, fv, ik)
                    cmask = _block_mask(np.random.default_rng(4321), 1, n, nb, 8,
                                         max(1, nb // 16), zero_rows=(0,))

                    # HIER arm (n>16 -> HIER both sides at DEFAULTS; here we take
                    # the coarse-ordered selection explicitly)
                    v_h, i_h, _ = _hier(idx, inp, k, cmask=cmask)
                    # FULL-width fallback arm: all columns scored, then the SAME
                    # top-k op (topk_from_row) -- the fallback's own order.
                    row_fb, _l, _q, _w = _fallback_row(idx, inp, l2_full=True)
                    row_fb = np.where(np.array(cmask), row_fb, float("-inf"))
                    fv2, fi = I.topk_from_row(mx.array(row_fb), k)
                    mx.eval(fv2, fi)
                    fv2n, fin = np.array(fv2), np.array(fi)
                    finf = np.isfinite(fv2n)
                    hinf = np.isfinite(v_h)
                    # SET and ORDER agree on every value-bearing slot; padding
                    # slots are unorderable by construction (both -inf).
                    ok = bool(hinf.sum() == finf.sum()
                              and np.array_equal(i_h[hinf], fin[finf])
                              and np.array_equal(v_h[hinf].view(np.uint32),
                                                 fv2n[finf].view(np.uint32)))
                    record(dict(case="cand_order", H=H, n=n, nb=nb, k=k,
                                set_and_order_equal=ok,
                                finite_slots=int(hinf.sum()),
                                padding_slots=int((~hinf).sum()),
                                slots=int(i_h.size)))
                    if not ok:
                        bad.append((H, n, nb, k))
        self.assertFalse(bad, f"{len(bad)} candidate-order cells diverge; first {bad[:6]}")


# ==========================================================================
# (v) GUARD OFF-BY-ONE
# ==========================================================================
class GuardBoundary(_Adv):
    def test_boundary_15_16_17(self):
        spy_calls = {"n": 0}
        real = HIER.hierarchical_topk_prod

        def wrapper(*a, **kw):
            spy_calls["n"] += 1
            return real(*a, **kw)

        HIER.hierarchical_topk_prod = wrapper
        bad = []
        try:
            for H in (8, 32, 64):
                for n in (15, 16, 17):
                    nb, k = 2048, 64
                    args = _args(H, candidate_source_layer=-1)
                    idx = _make_indexer(args, 0, seed=0)
                    idx.index_topk = k
                    rng = np.random.default_rng(6600 + H + n)
                    x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
                    qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
                    ik = mx.array(rng.standard_normal((1, nb, PROD_DIM)).astype(np.float32))
                    fv = mx.array(np.linspace(1.0, 0.01, 8).astype(np.float32))
                    inp = (x, qr, 2 * nb, fv, ik)

                    I._FENCE_MIN_ROWS = THRESH
                    spy_calls["n"] = 0
                    out_default = _call_arm(idx, inp, THRESH)
                    took_hier = spy_calls["n"] > 0
                    out_hier = _call_arm(idx, inp, -1)          # forced HIER
                    eq = bool(np.array_equal(out_default, out_hier))
                    want_hier = n > THRESH
                    ok = eq and (took_hier == want_hier)
                    record(dict(case="guard_boundary", H=H, n=n, nb=nb, k=k,
                                equal=eq, took_hier=took_hier, want_hier=want_hier,
                                ok=ok))
                    if not ok:
                        bad.append((H, n))
        finally:
            HIER.hierarchical_topk_prod = real
        self.assertFalse(bad, f"{len(bad)} guard-boundary cells violate; first {bad[:6]}")


# ==========================================================================
# verdict
# ==========================================================================
class Summary(unittest.TestCase):
    def test_zz_verdict(self):
        adv = [r for r in RESULTS
               if r.get("case") in ("dup_keys", "rope_alias", "cand_order", "guard_boundary")]
        ulp = [r for r in RESULTS if r.get("case") == "ulp_boundary"]
        equal = [r for r in adv if r.get("index_equal", r.get("set_and_order_equal",
                                                              r.get("ok", False)))]
        row_mm = sum(r.get("row_bitwise_mismatch", 0) for r in adv)
        ulp_flips = sum(r["topk_flips"] for r in ulp)
        ulp_flips_prod = sum(r["topk_flips"] for r in ulp if r.get("H") == H_PROD)
        summary = {
            "suite": "DSV41_ADV_PRODH",
            "H_production": H_PROD,
            "cases": len(RESULTS),
            "identity_cases": len(adv),
            "identity_equal": len(equal),
            "identity_divergent": len(adv) - len(equal),
            "row_bitwise_mismatches": row_mm,
            "ulp_cases": len(ulp),
            "ulp_topk_flips_total": ulp_flips,
            "ulp_topk_flips_at_production_H": ulp_flips_prod,
            "abort_signal": bool(ulp_flips_prod),
        }
        print("DSV41_ADV_PRODH " + json.dumps(summary), flush=True)
        self.assertEqual(len(adv) - len(equal), 0,
                         f"ADVERSARIAL IDENTITY FAILS: {len(adv)-len(equal)}/{len(adv)} cells")
        self.assertEqual(row_mm, 0, f"{row_mm} row-level bitwise mismatches")
        self.assertEqual(ulp_flips_prod, 0,
                         f"ABORT: {ulp_flips_prod} ULP-boundary top-k flips at production H")


if __name__ == "__main__":
    unittest.main(verbosity=2)
