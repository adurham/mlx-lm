#!/usr/bin/env python3
"""cap_gap.py -- value-level (L1) comparison on the REAL captures.

For a subset of rows: the fp32 score row (the reference the exact pass ranks)
vs the values the HIER path actually selected.  Answers the plan's §2.2 L1
question ("score values before selection") and prints the gap census that
decides which rows are at risk of a near-tie flip.

Run on studio2:  P1_CAP_OUT=/private/tmp/idxprof1/raw/cap python bench/cap_gap.py
"""
from __future__ import annotations

import json
import os
import sys

_repo = os.environ.get("P1_MLXLN")
if _repo:
    sys.path.insert(0, _repo)
import numpy as np                                   # noqa: E402
import mlx.core as mx                                # noqa: E402
from mlx_lm.models.deepseek_v41 import indexer as IX              # noqa: E402
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as HC  # noqa: E402
from mlx_lm.models.deepseek_v41 import collective as _coll        # noqa: E402

CAP = os.environ.get("P1_CAP_OUT", "/private/tmp/idxprof1/raw/cap")
TOPK, BLOCK = 512, 8
ROWS = int(os.environ.get("P1_GAP_ROWS", "64"))


def log(*a):
    print("[gap]", *a, flush=True)


def main():
    _coll.all_sum = lambda x, group=None: x
    out = []
    for L in (50000, 105000):
        f = os.path.join(CAP, f"cap_L{L}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        for lay in (8, 20):
            pre = f"L{lay}_"
            if pre + "ik16" not in z:
                continue
            sp, nb, ratio = (int(z[pre + k]) for k in ("start_pos", "nb", "ratio"))
            n_full = int(z[pre + "n_full"])
            ik = mx.array(z[pre + "ik16"]).astype(mx.bfloat16)
            x, qr = mx.array(z[pre + "x"]), mx.array(z[pre + "qr"])
            ix = None
            for rk in (0, 1):
                q = mx.array(z[f"{pre}q_f32_r{rk}"])
                w = mx.array(z[f"{pre}w_r{rk}"])
                lens = mx.array(z[f"{pre}lens_r{rk}"])
                n = min(ROWS, q.shape[1])
                # fp32 score row for the subset (production tiled expression)
                row = np.empty((n, nb), dtype=np.float32)
                T = 504
                for c0 in range(0, nb, T):
                    c1 = min(c0 + T, nb)
                    s = mx.einsum("bshd,btd->bsht", q[:, :n],
                                  ik[:, c0:c1].astype(mx.float32))
                    s = mx.maximum(s, 0.0) * w[:, :n, ..., None]
                    s = mx.sum(s, axis=2)
                    col = mx.arange(c0, c1, dtype=mx.int32)
                    s = mx.where(col[None, :] < lens[:n], s, IX.NEG_INF)
                    mx.eval(s)
                    row[:, c0:c1] = np.array(s)[0, :n]
                ln = np.array(lens).reshape(-1)[:n]
                # HIER selection on the same rows
                k = min(TOPK, nb)
                if ix is None:
                    from mlx_lm.models.deepseek_v41.exl3_build import _dense, _plain, _groups
                    from mlx_lm.models.exl3.loader import Exl3Checkpoint
                    from mlx_lm.models.deepseek_v41.config import ModelArgs
                    ck = Exl3Checkpoint(os.environ.get(
                        "P1_MODEL", os.path.expanduser("~") +
                        "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"))
                    args = ModelArgs.from_dict(ck.config)
                    ix = IX.Indexer(args, lay)
                    pre_w = f"layers.{lay}.attn.indexer."
                    items = [(g[len(pre_w):], _dense(ck, g)) for g in _groups(ck, pre_w)]
                    for kk in ck.index:
                        if kk.startswith(pre_w):
                            t = kk[len(pre_w):]
                            if t.rsplit(".", 1)[-1] in ("trellis", "suh", "svh", "mul1"):
                                continue
                            if f"{pre_w}{t}" in [g for g in _groups(ck, pre_w)]:
                                continue
                            items.append((t, _plain(ck, kk)))
                    ix.load_weights(items, strict=False)
                    mx.eval([v for _, v in items if isinstance(v, mx.array)])
                estrip = IX.hier_strip_for_budget(1, n, 128,
                                                 int(IX._HIER_EXACT_MB * (1 << 20)))
                vals, idx = HC.hierarchical_topk_prod(
                    q[:, :n].astype(mx.float32), ik, w[:, :n].astype(mx.float32),
                    lens[:n], k, BLOCK, IX._HIER_STRIP, estrip, IX._HIER_OVERFETCH,
                    cand_mask=None, cand_src=None,
                    consumer_skip=IX._HIER_CONSUMER_SKIP)[:2]
                idx = np.array(idx)[0]
                vals = np.array(vals)[0]
                # exact top-k of the fp32 row
                exp = np.sort(np.where(np.arange(nb)[None, :] < ln[:, None], row,
                                       -np.inf), axis=1)[:, -k:]
                got = np.sort(vals, axis=1)[:, -k:]
                d = np.abs(exp - got)
                d[~np.isfinite(exp) & ~np.isfinite(got)] = 0.0
                per_row = d.max(axis=1)
                # gap census: k-th minus (k+1)-th of the fp32 row
                part = np.partition(row, nb - k - 1, axis=1)
                kth = part[:, nb - k]
                kp1 = part[:, nb - k - 1]
                gap = kth - kp1
                at_risk = int(np.sum(per_row > 0))
                rec = dict(L=L, layer=lay, nb=nb, rank=rk, n_rows=n,
                           rows_with_value_diff=at_risk,
                           max_abs_value_diff=float(np.nanmax(per_row)),
                           p99_row_max_abs_diff=float(np.percentile(per_row, 99)),
                           min_gap_median=float(np.median(gap)),
                           rows_gap_below_1em6=int(np.sum(gap < 1e-6)),
                           rows_gap_below_1em5=int(np.sum(gap < 1e-5)),
                           score_scale_median=float(np.median(np.abs(row[np.isfinite(row)]))),
                           label="MEASURED(real activations)")
                out.append(rec)
                log(f"L{L} layer{lay} rank{rk}: rows_with_value_diff={at_risk}/{n} "
                    f"max|Δ|={rec['max_abs_value_diff']:.3e} "
                    f"p99_row_max|Δ|={rec['p99_row_max_abs_diff']:.3e} "
                    f"gap_med={rec['min_gap_median']:.3e} "
                    f"rows(gap<1e-6)={rec['rows_gap_below_1em6']} "
                    f"rows(gap<1e-5)={rec['rows_gap_below_1em5']} "
                    f"scale={rec['score_scale_median']:.3f}")
                del q, w, lens
    with open(os.path.join(CAP, "gap.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    log("GAP_DONE")


if __name__ == "__main__":
    main()
