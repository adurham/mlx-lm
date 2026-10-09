# Copyright © 2026 Adam Durham (hermes-gw)
"""Correct classifier over ALL residual divergent cells (value-multiset metric).

The two arms return index sets SORTED ASCENDING BY INDEX, so a slot-wise
comparison compares different columns and is meaningless when the sets differ.
The sound metric: per query row, sort each arm's returned VALUES descending
(using the full-width fp32 row R as the score oracle) and compare the rank-wise
multisets, plus each arm's loss against the true top-k multiset of R.

Classification per cell:
  * EQUAL          -- index sets identical.
  * VALUE-TIE      -- value multisets agree within `tol` (differences are exact
                      ties or float-rounding ulp; no picked column is worse).
  * VALUE-LOSS     -- some rank where the fallback's value exceeds hier's by
                      > tol (hier dropped a column the fallback kept at a
                      strictly higher score).

Run: PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_classify_all.py <suite.log> [--tol REL]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests")))
from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41.model import SharedState
from mlx_lm.models.deepseek_v41.layers import cos_sin_at, rope_tail
from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0

import test_dsv41_indexer_smallm_hier as T


def parse_divergent(path):
    cells = []
    for ln in open(path):
        s = ln.strip()
        if not s.startswith('{"role"'):
            continue
        try:
            r = json.loads(s)
        except Exception:
            continue
        if r.get("equal") is False:
            cells.append(r)
    return cells


def values_for_arm(o, R, nb):
    """[n_rows, k] descending value multiset of one arm's returned indices."""
    n = o.shape[1]; k = o.shape[2]
    out = np.empty((n, k), np.float64)
    for r in range(n):
        row = []
        for s in range(k):
            j = int(o[0, r, s])
            row.append(float(R[0, r, j]) if 0 <= j < nb else float("-inf"))
        out[r] = np.sort(np.array(row))[::-1]
    return out


def classify(cell, tol):
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
    nd = int((od != oh).sum())
    if nd == 0:
        return dict(role=role, n=n, nb=nb, k=k, seed=s, ndiff=0, verdict="EQUAL")

    x, qr, sp, fv, ik = inp
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
    R = np.array(row)

    truth = np.stack([np.sort(R[0, r])[::-1][:k] for r in range(n)])
    vf = values_for_arm(od, R, nb)      # fallback (full-width)
    vh = values_for_arm(oh, R, nb)      # hier
    scale = max(1.0, float(np.abs(truth[np.isfinite(truth)]).max()) if np.isfinite(truth).any() else 1.0)

    def loss_vs_truth(v):
        # a "loss" is a finite truth rank where this arm's value is strictly
        # below truth beyond tol. (-inf arm values where truth is finite count;
        # -inf truth ranks are unmatchable by construction and never count.)
        finite_t = np.isfinite(truth)
        return int((finite_t & ((truth - v) > tol)).sum())

    # fb-vs-hier per rank, NaN-safe (both -inf => equal, no worse)
    d = vf - vh
    d = np.where(np.isnan(d), 0.0, d)
    fb_minus_hier = float(d.max()) if d.size else 0.0
    fb_loss_vs_truth = loss_vs_truth(vf)
    hier_loss_vs_truth = loss_vs_truth(vh)
    if fb_loss_vs_truth > 0:
        verdict = "FALLBACK-LOSS"
    elif fb_minus_hier > tol:
        verdict = "VALUE-LOSS"
    else:
        verdict = "VALUE-TIE"
    return dict(role=role, n=n, nb=nb, k=k, seed=s, ndiff=nd,
                skip=cell.get("skip"),
                fb_minus_hier_max=round(fb_minus_hier, 8),
                fb_loss_vs_truth=fb_loss_vs_truth,
                hier_loss_vs_truth=hier_loss_vs_truth,
                verdict=verdict)


if __name__ == "__main__":
    path = sys.argv[1]
    tol = 1e-6
    if "--tol" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tol") + 1])
    cells = parse_divergent(path)
    print(f"# {len(cells)} divergent cells in {path} (tol={tol:g})", flush=True)
    res = []
    for i, c in enumerate(cells):
        r = classify(c, tol)
        res.append(r)
        if r["verdict"] != "VALUE-TIE" or r["ndiff"] == 0:
            print("  ! " + json.dumps(r), flush=True)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(cells)}", flush=True)
    tie = sum(1 for r in res if r["verdict"] == "VALUE-TIE")
    loss = [r for r in res if r["verdict"] == "VALUE-LOSS"]
    fbloss = [r for r in res if r["verdict"] == "FALLBACK-LOSS"]
    eq = sum(1 for r in res if r["verdict"] == "EQUAL")
    print("CLASSIFY_ALL " + json.dumps({
        "cells": len(res), "equal_on_rerun": eq, "value_tie": tie,
        "value_loss_hier_worse": len(loss),
        "fallback_loss_vs_truth_cells": len(fbloss),
        "fallback_loss_slots_vs_truth": sum(r.get("fb_loss_vs_truth", 0) for r in res),
        "hier_loss_slots_vs_truth": sum(r.get("hier_loss_vs_truth", 0) for r in res),
        "max_fb_minus_hier": max((r.get("fb_minus_hier_max", 0) for r in res), default=0),
        "verdict": ("FALLBACK (L2-full) NEVER LOSES vs truth; HIER loses "
                    f"{sum(r.get('hier_loss_vs_truth',0) for r in res)} slots"
                    if not fbloss else "FALLBACK LOSSES DETECTED")}),
        flush=True)
    for r in loss[:8]:
        print("  LOSS " + json.dumps(r), flush=True)
