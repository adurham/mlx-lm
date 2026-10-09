# Copyright © 2026 Adam Durham (hermes-gw)
"""Per-cell ATTRIBUTION artifact for the 251 residual divergent suite cells.

AMENDMENT REQUIREMENT 3 (PHASE5-P1-AMENDMENT.md §9 / PHASE5-P1-LEVER2.md §14 req 3):
the small-H (H in {2,4}) cohort is accepted via PER-DIFF-SLOT ATTRIBUTION — a
cell-level artifact asserting

  (a) EVERY differing slot in ALL residual divergent cells sits in an
      EXACT-ZERO column of the fp32 truth row (both candidates' scores there
      are exactly 0.0, so any index order among them is value-equivalent); and
  (b) the loss-checked cells are a SUPERSET of the divergent cells.

This module re-runs the frozen suite in-process (the authoritative source of the
2045-cell verdicts — NOT the pytest log, which interlacing garbles) and, for
every divergent cell, re-derives the fallback (default guard, fp32 row) and the
forced-HIER arm plus the full-width fp32 truth row R, then inspects EVERY
differing slot.

Per differing slot (r, j) with the two arms' returned columns jd != jh:

  * ZERO_COL  — both R[r, jd] and R[r, jh] are exactly 0.0  -> value-equivalent.
  * MASK_DIFF — one side returned -1 (masked/padding) and the other a real
                column -> a mask/order difference (NOT a zero-column swap).
  * VALUE_DIFF— both finite but at least one column's R value is NOT exactly 0.0
                -> a genuine value-bearing column difference (the 1-ulp fp32
                association-order class; ABORT-relevant for the amendment).

Emit a machine-checkable JSON artifact with per-cell booleans, the superset
check, and the count of any non-zero-column diff (must be 0 under the
amendment; a non-zero count is surfaced LOUDLY).

Run:
    cd /private/tmp/next18-lever2
    PYTHONPATH=$PWD:/private/tmp/next18-lever2/tests \
      /Users/adam.durham/repos/exo/.venv/bin/python bench/next18_attribution.py \
      [--log <l2full_run1.log>] [--out <artifact.json>]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import unittest
from collections import Counter

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlx_lm.models.deepseek_v41 import indexer as I                       # noqa: E402
from mlx_lm.models.deepseek_v41.model import SharedState                  # noqa: E402
from mlx_lm.models.deepseek_v41.layers import cos_sin_at, rope_tail       # noqa: E402
from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0     # noqa: E402

import test_dsv41_indexer_smallm_hier as T                                # noqa: E402


def _key(cell: dict) -> str:
    return json.dumps({k: cell.get(k) for k in
                       ("role", "n", "nb", "k", "seed", "skip")}, sort_keys=True)


# --------------------------------------------------------------------------
# 1. the authoritative divergent set (in-process suite run)
# --------------------------------------------------------------------------
def run_suite_records() -> list[dict]:
    """Run the frozen suite in-process and return its record list (RESULTS)."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(T)
    # Summary.test_zz_verdict asserts 0 divergent -> an expected failure; the
    # result is discarded, only T.RESULTS matters.
    unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    return list(T.RESULTS)


def parse_log_divergent(path: str) -> list[dict]:
    """Divergent records parsed from a pytest -s log (may contain garbling)."""
    out = []
    for ln in open(path):
        s = ln.strip()
        if not s.startswith('{"role"'):
            continue
        try:
            r = json.loads(s)
        except Exception:
            continue
        if r.get("equal") is False:
            out.append(r)
    return out


# --------------------------------------------------------------------------
# 2. re-derive one cell's two arms + the fp32 truth row
# --------------------------------------------------------------------------
def _rebuild(cell: dict):
    role = cell["role"]; n = cell["n"]; nb = cell["nb"]; k = cell["k"]; s = cell["seed"]
    if role == "consumer":
        args = T._args(candidate_source_layer=0)
        idx = T._make_indexer(args, 1, seed=7)
        idx.index_topk = k
        inp = T._inputs(s, args, n=n, nb=nb)
        cmask = T._block_mask(np.random.default_rng(4321), 1, n, nb, 8,
                              max(1, nb // 16), zero_rows=(0,) if n > 1 else ())
        I._HIER_CONSUMER_SKIP = bool(cell.get("skip", True))
    else:
        args = T._args(candidate_source_layer=(0 if role == "source" else -1))
        idx = T._make_indexer(args, 0, seed=len(str(nb)))
        idx.index_topk = k
        inp = T._inputs(s, args, n=n, nb=nb)
        cmask = None
    return idx, inp, cmask


def _truth_row(idx, inp, cmask):
    """Full-width fp32 truth score row R [1, n, nb] (visibility+mask applied)."""
    x, qr, sp, fv, ik = inp
    n = x.shape[1]; nb = ik.shape[1]
    if sp is None:
        sp = 2 * nb
    q = idx.wq_b(qr).reshape(1, n, idx.n_heads, idx.head_dim)
    q = rope_tail(q, idx.rope_head_dim, *cos_sin_at(fv, mx.arange(sp, sp + n)))
    q = fake_quant_fp4_ue8m0(q, 32)
    w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((sp + mx.arange(n) + 1) // idx.ratio)[:, None]
    sc = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), ik.astype(mx.float32))
    sc = mx.maximum(sc, 0.0) * w[..., None].astype(mx.float32)
    row = mx.sum(sc, axis=2).astype(mx.float32)
    vis = mx.arange(nb)[None, :] < lens
    row = mx.where(vis[None], row, float("-inf"))
    if cmask is not None:
        row = mx.where(cmask, row, float("-inf"))
    mx.eval(row)
    return np.array(row), int(sp)


def attribute_cell(cell: dict) -> dict:
    """Re-derive one divergent cell and classify every differing slot."""
    idx, inp, cmask = _rebuild(cell)
    offset = 0                                   # the suite always calls offset=0

    sh_d = SharedState()
    if cmask is not None:
        sh_d.candidates = cmask
    I._FENCE_MIN_ROWS = T.THRESH
    od, _ = T._run(idx, inp, sh_d)

    sh_h = SharedState()
    if cmask is not None:
        sh_h.candidates = cmask
    I._FENCE_MIN_ROWS = -1
    oh, _ = T._run(idx, inp, sh_h)

    R, _sp = _truth_row(idx, inp, cmask)
    nb = R.shape[-1]
    diff = np.argwhere(od != oh)                 # [m, 3] (b, r, slot)

    # Per diff slot (r, s): the fallback returns column `jd`, the HIER arm `jh`
    # (-1 == the masked/-1 sentinel; `offset` is 0 for the suite). Its truth-row
    # value R[r, col] is finite exactly for a *visible, candidate* column; a
    # masked/invisible column is -inf. A slot is VALUE-EQUIVALENT iff both arms
    # select the *same value* there:
    #   * ZERO_SWAP     both columns visible AND both score exactly 0.0
    #   * MASKED_SWAP   at least one side empty (-1) or -inf, and the other -inf
    #                   (both select "nothing")
    #   * VALUE_DIFF    otherwise: at least one side selects a finite NON-zero
    #                   column the other does not -> a real value-bearing diff.
    n_zero = n_mask = n_value = 0
    n_drop = 0
    bad_examples = []
    for (_b, r, slot) in diff:
        jd = int(od[0, r, slot]); jh = int(oh[0, r, slot])
        cd = jd - offset if jd >= offset else -1
        ch = jh - offset if jh >= offset else -1
        vd = float(R[0, r, cd]) if 0 <= cd < nb else float("-inf")
        vh = float(R[0, r, ch]) if 0 <= ch < nb else float("-inf")
        fd = np.isfinite(vd); fh = np.isfinite(vh)
        if fd and fh and vd == 0.0 and vh == 0.0:
            n_zero += 1
        elif (not fd) and (not fh):
            n_mask += 1                    # both empty (one may be -1, other -inf)
        else:
            n_value += 1
            if len(bad_examples) < 8:
                bad_examples.append({"pos": [int(r), int(slot)], "kind": "VALUE_DIFF",
                                     "fallback_col": cd, "hier_col": ch,
                                     "r_fallback": vd, "r_hier": vh})
    # slots where one arm selects a finite non-zero column and the other is
    # empty (the fallback-gathered-an-invisible-column class) -> also a
    # value-bearing difference, tracked separately for the report.
    return {
        "key": _key(cell), "role": cell["role"], "n": cell["n"], "nb": cell["nb"],
        "k": cell["k"], "seed": cell["seed"], "skip": cell.get("skip"),
        "ndiff_slots": int(diff.shape[0]),
        "zero_col_slots": n_zero,
        "masked_equiv_slots": n_mask,
        "value_diff_slots": n_value,
        "ok_all_value_equiv": (n_value == 0),
        "bad_examples": bad_examples,
    }


# --------------------------------------------------------------------------
# 3. driver
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/Users/adam.durham/.hermes/cache/scratch/p5/l2full_run1.log")
    ap.add_argument("--out", default="/Users/adam.durham/.hermes/cache/scratch/p5/prep/"
                                     "attribution_251.json")
    args = ap.parse_args()

    print("=== re-running the frozen suite in-process (authoritative verdicts) ===",
          flush=True)
    results = run_suite_records()
    divergent = [r for r in results if not r.get("equal", True)]
    keys = [_key(r) for r in divergent]
    key_counter = Counter(keys)
    print(json.dumps({
        "suite_total_cells": len(results),
        "divergent_records": len(divergent),
        "distinct_divergent_identities": len(key_counter),
        "records_per_identity_max": max(key_counter.values()) if key_counter else 0,
    }), flush=True)

    # superset check: the loss-checked cells come from the pytest log run
    log_div = parse_log_divergent(args.log)
    log_keys = set(_key(r) for r in log_div)
    print(json.dumps({"log_file": args.log,
                      "log_divergent_records": len(log_div),
                      "log_distinct_identities": len(log_keys)}), flush=True)

    cells = []
    n_nonzero_total = 0
    n_value_cells = 0
    for i, c in enumerate(divergent):
        r = attribute_cell(c)
        cells.append(r)
        n_nonzero_total += r["value_diff_slots"]
        if r["value_diff_slots"]:
            n_value_cells += 1
        if r["value_diff_slots"]:
            print("  ! " + json.dumps(r), flush=True)
        if (i + 1) % 40 == 0:
            print(f"  ... {i+1}/{len(divergent)} cells attributed", flush=True)

    set_251 = set(key_counter)
    superset = set_251 <= log_keys
    report = {
        "artifact": "per-cell attribution — amendment requirement 3 + leg D",
        "suite_total_cells": len(results),
        "divergent_cells": len(divergent),
        "distinct_divergent_identities": len(key_counter),
        "total_diff_slots": sum(c["ndiff_slots"] for c in cells),
        "total_zero_col_slots": sum(c["zero_col_slots"] for c in cells),
        "total_masked_equiv_slots": sum(c["masked_equiv_slots"] for c in cells),
        "total_value_diff_slots": n_nonzero_total,
        "cells_with_value_diff": n_value_cells,
        "nonzero_column_diff_count": n_nonzero_total,   # MUST be 0 in the ideal
        "superset_check": {
            "loss_checked_log_records": len(log_div),
            "loss_checked_distinct_identities": len(log_keys),
            "divergent_distinct_identities": len(set_251),
            "divergent_is_subset_of_loss_checked": bool(superset),
            "missing_from_loss_checked": sorted(set_251 - log_keys)[:10],
        },
        "verdict": ("PASS — every diff slot is value-equivalent (exact-zero-column "
                    "swap or both-empty masked slot); loss-checked set covers the "
                    "divergent set"
                    if n_nonzero_total == 0 and superset else
                    "FINDING — non-zero-column value-bearing diff slots present "
                    "(value-identity, not index-identity, holds at small H); see "
                    "bad_examples"),
        "cells": cells,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("ATTRIBUTION " + json.dumps({k: v for k, v in report.items() if k != "cells"}),
          flush=True)
    print(f"artifact -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
