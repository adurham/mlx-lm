# Copyright © 2026 Adam Durham (hermes-gw)
"""Reference-divergence harness for the dsv41 indexer's small-n WIDE TILE.

Why this exists (audit, 2026-10-14).  Between the clean baseline pin ``689e4ea``
(``deploy/next22-rankfix``, 144 clean calls) and the live pin ``a0573e3``
(``deploy/next26-limiter``) the COMPLETE mlx-lm change surface is 6 commits
(``dc0e79c -> f728bb6 -> e034062 -> de6caa2 -> 6a5c78f -> a0573e3``).  With
``DSV41_INDEXER_QSPLIT=0`` (the config of the second live degeneration event) every
qsplit hunk is a no-op (``query_band`` returns ``None`` -> the band kwargs are never
bound -> ``_finish_rows`` returns ``cidx`` unchanged), and the ONLY hunk that changes
serving math is the ``tile_width()`` small-n wide-tile branch
(``DSV41_INDEXER_SMALLN_WIDE_TILE``, default ON).  This harness measures that rule at
PRODUCTION geometry on three levels and decides whether it can move a selection.

Levels.  Token-level "it does not loop any more" proves nothing (greedy decode is
chaotic), so nothing here compares tokens.  Instead, per cell:

* ROUTE / ENGAGEMENT — the exact ``(n, ratio, context)`` set where the new width
  differs from the old one, and a proof that the tiled/untiled ROUTE cannot differ
  (``nb <= cap`` implies the budget clause, so the route is decided identically).
* INDEX SETS — the ``[b, n, index_topk]`` ``cidx`` returned by the REAL
  ``Indexer.__call__`` in the NEW / OLD / UNT arms (byte comparison), plus the mean
  per-row top-k overlap.
* SCORE ROW (the top-k's own input, i.e. the logit-proxy) — the stored
  ``[b, n, nb]`` row rebuilt at each arm's exact width through the SHIPPED compiled
  score body, compared bitwise: element count differing, max |delta|, and the k-th
  boundary gap so a near-tie flip is distinguishable from a real bug.
* CACHE CONTENTS — ``publish_keys`` is not touched by the 6 commits (it is
  query-side-invariant), pinned by the sibling suites.
* CANDIDATE MASK — the layer-20 mask is ONE global pass over the row
  (``select_candidate_blocks``) and is consumed by every later index layer, so a
  single flipped block-max amplifies across the whole forward; compared bitwise.

Production geometry used here: ``index_n_heads=32``, ``index_head_dim=128``,
``index_topk=512``, ``DSV41_INDEXER_TILE=512``, ``DSV41_INDEXER_TILE_MB=128``,
``_TILE_MIN_NB=0``, ``_FENCE_MIN_ROWS=16``, ``_L2_FULL=1`` (fp32 small-n row),
``_HIER=1``.  ``dim``/``q_lora_rank`` are shrunk: the score GEMM is
``(M = bsz*n, K = index_head_dim, N = tile)`` and neither dim nor q_lora_rank enters
it, so those two only make the projections cheaper.  The row dtype, the row width,
the top-k and the mask grid are production-exact.

Usage (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python bench/dsv41_indexer_refdiv.py \\
        [--out <json>] [--quick] [--compile 0|1]

Exit status 1 if any cell diverges (a divergence is a finding, not a test error).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as IX
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0
from mlx_lm.models.deepseek_v41.model import SharedState

HEADS = 32
HEAD_DIM = 128
INDEX_TOPK = 512
TILE = 512
BUDGET_MB = 128
BLOCK = 8

RESULTS: list[dict[str, Any]] = []


def record(row: dict[str, Any]) -> dict[str, Any]:
    RESULTS.append(row)
    print("  " + json.dumps(row, default=str), flush=True)
    return row


def args_of(role: str, *, block: int = BLOCK, topk_blocks: int = 32) -> ModelArgs:
    return ModelArgs(
        dim=256, n_layers=3, n_heads=4, head_dim=128, rope_head_dim=16,
        q_lora_rank=128, o_lora_rank=32, o_groups=4, window_size=8,
        compress_ratios=(1, 1, 1), kv_source_layers=(0,), index_source_layers=(0,),
        index_n_heads=HEADS, index_head_dim=HEAD_DIM, index_topk=INDEX_TOPK,
        max_seq_len=1 << 21,
        candidate_source_layer=(0 if role != "plain" else -1),
        candidate_topk_blocks=topk_blocks, candidate_block_size=block,
    )


def production_knobs() -> None:
    """The live knob values (env-unset defaults) at production geometry."""
    IX._TILE, IX._TILE_BUDGET = TILE, BUDGET_MB << 20
    IX._TILE_MIN_NB, IX._TILE_FORCE = 0, False
    IX._L2_FULL, IX._SMALLN_ROW_BF16 = True, False
    IX._HIER, IX._QSPLIT, IX._TILED_IMPL = True, False, "buffer"
    IX._SMALLN_WIDE_TILE = True
    IX._FENCE_MIN_ROWS = 16


def old_width(bsz: int, n: int, nb: int, align: int) -> int:
    """The pre-a0573e3 rule, transcribed verbatim from 6a5c78f4."""
    cap = max(align, (BUDGET_MB << 20) // max(1, bsz * n * (HEADS * 4 + IX._row_dtype(n).size)))
    return min(nb, max(align, (min(TILE, cap) // align) * align))


def new_width(bsz: int, n: int, nb: int, align: int) -> int:
    """The a0573e3 rule, transcribed: the wide branch is small-n only."""
    cap = max(align, (BUDGET_MB << 20) // max(1, bsz * n * (HEADS * 4 + IX._row_dtype(n).size)))
    if n <= IX._FENCE_MIN_ROWS and not IX._TILE_FORCE:
        return min(nb, max(align, (cap // align) * align))
    return min(nb, max(align, (min(TILE, cap) // align) * align))


# ---------------------------------------------------------------------------
# A. route / engagement
# ---------------------------------------------------------------------------
def context_grid() -> list[int]:
    """32K..1.1M with a dense inset across the 48K..82K band (the loop events)."""
    grid = list(range(32_000, 120_001, 4_000))
    grid += list(range(240_000, 1_100_001, 20_000))
    dense = list(range(48_000, 82_001, 512))          # 50-80K, ~1 per chunk row
    return sorted(set(grid + dense))


def section_a() -> dict[str, Any]:
    print("== A. route / engagement (production constants) ==", flush=True)
    production_knobs()
    n_values = (1, 2, 3, 4, 5, 8, 12, 16, 17, 32, 64, 78, 512)
    cells = differ = route_mismatch = transcription_mismatch = 0
    changed: list[dict[str, Any]] = []
    band_counts = {"50-80K": 0, "110-241K": 0, ">=254K": 0}
    band_changed = {"50-80K": 0, "110-241K": 0, ">=254K": 0}
    crossings: dict[str, int | None] = {}
    changed_n_in_band: set[int] = set()

    def band_of(ctx: int) -> str | None:
        if 48_000 <= ctx <= 82_000:
            return "50-80K"
        if 110_000 <= ctx <= 241_000:
            return "110-241K"
        if ctx >= 254_000:
            return ">=254K"
        return None

    for ratio in (1, 2, 4):
        for n in n_values:
            for align in (1, BLOCK):
                first_change = None
                for ctx in context_grid():
                    nb = max(1, ctx // ratio)
                    IX._SMALLN_WIDE_TILE = False
                    w_off = IX.tile_width(1, n, HEADS, nb, align)
                    route_old = IX.tiled(1, n, HEADS, nb, align) < nb
                    IX._SMALLN_WIDE_TILE = True
                    w_on = IX.tile_width(1, n, HEADS, nb, align)
                    route_new = IX.tiled(1, n, HEADS, nb, align) < nb
                    wo, wn = old_width(1, n, nb, align), new_width(1, n, nb, align)
                    cells += 1
                    transcription_mismatch += int(w_off != wo or w_on != wn)
                    route_mismatch += int(route_old != route_new)
                    band = band_of(ctx)
                    if band:
                        band_counts[band] += 1
                    # a width change is observable only on the tiled route
                    if route_new and wn != wo:
                        differ += 1
                        if first_change is None:
                            first_change = ctx
                        if band:
                            band_changed[band] += 1
                        if band == "50-80K":
                            changed_n_in_band.add(n)
                        if len(changed) < 16:
                            changed.append(dict(ratio=ratio, n=n, align=align, context=ctx,
                                                nb=nb, width_old=wo, width_new=wn))
                if n <= 16:
                    crossings[f"ratio{ratio}_n{n}_align{align}"] = first_change
    out = record(dict(case="route_engagement", cells=cells, width_changed_cells=differ,
                      route_mismatch=route_mismatch,
                      transcription_mismatch=transcription_mismatch,
                      band_cells=band_counts, band_cells_changed=band_changed,
                      first_change_context=crossings,
                      n_values_changed_in_50_80K=sorted(changed_n_in_band),
                      examples=changed))
    assert route_mismatch == 0, "the tiled/untiled ROUTE must be width-rule invariant"
    assert transcription_mismatch == 0, "transcribed rules must match the shipped widths"
    return out


# ---------------------------------------------------------------------------
# B/C. score row + index sets at each arm's exact width, production geometry
# ---------------------------------------------------------------------------
def keys_of(seed: int, nb: int, mode: str) -> mx.array:
    """``index_k`` as the cache holds it: fp4_ue8m0 fake-quantized (publish_keys)."""
    rng = np.random.default_rng(seed)
    if mode == "ties":
        p = 97
        base = rng.standard_normal((p, HEAD_DIM)).astype(np.float32)
        raw = mx.array(base[np.arange(nb) % p][None])
    else:
        raw = mx.array(rng.standard_normal((1, nb, HEAD_DIM)).astype(np.float32))
    return fake_quant_fp4_ue8m0(raw, 32)


def row_at_width(bsz: int, n: int, q32, ik, w32, lens, nb: int, width: int) -> np.ndarray:
    """The stored ``[b, n, nb]`` row exactly as the shipped path builds it.

    ``width=0`` = the untiled branch; otherwise the tiled loop in ``width`` columns
    through the SHIPPED (per-width compiled) score body, fenced per tile.
    """
    row_dtype = IX._row_dtype(n)
    if width <= 0:
        scores = mx.einsum("bshd,btd->bsht", q32, ik.astype(mx.float32))
        scores = mx.maximum(scores, 0.0) * w32[..., None]
        scores = mx.sum(scores, axis=2).astype(row_dtype)
        vis = mx.arange(nb)[None, :] < lens
        scores = mx.where(vis[None], scores, IX.NEG_INF)
        mx.eval(scores)
        return np.array(scores)
    body = IX._score_body(bsz, n, HEADS, HEAD_DIM, width)
    row = mx.zeros((bsz, n, nb), dtype=row_dtype)
    for c0 in range(0, nb, width):
        c1 = min(c0 + width, nb)
        s = body(q32, ik[:, c0:c1].astype(mx.float32), w32)
        cols = mx.arange(c0, c1, dtype=mx.int32)
        s = mx.where(cols[None, :] < lens, s, IX.NEG_INF)
        mx.eval(s)
        row[:, :, c0:c1] = s
    mx.eval(row)
    return np.array(row)


def topk_sets(row: np.ndarray, k: int) -> list[set[int]]:
    return [set(int(i) for i in np.argsort(-r, kind="stable")[:k])
            for r in row.reshape(-1, row.shape[-1])]


def mean_overlap(a: list[set[int]], b: list[set[int]]) -> float:
    return float(np.mean([len(x & y) / len(x) for x, y in zip(a, b)]))


def qw_of(idx: IX.Indexer, x, qr, sp, fv):
    n = x.shape[1]
    q = idx.wq_b(qr).reshape(1, n, idx.n_heads, idx.head_dim)
    qpos = mx.arange(sp, sp + n)
    q = IX.rope_tail(q, idx.rope_head_dim, *IX.cos_sin_at(fv, qpos))
    q = fake_quant_fp4_ue8m0(q, 32)
    w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((qpos + 1) // idx.ratio)[:, None]
    return q.astype(mx.float32), w.astype(mx.float32), lens, qpos


def inputs_for(seed: int, role: str, n: int, nb: int, mode: str):
    args = args_of(role)
    mx.random.seed(seed)
    idx = IX.Indexer(args, 1)
    mx.eval(idx.parameters())
    cons = IX.Indexer(args, 2) if role == "consumer" else None
    if cons is not None:
        mx.eval(cons.parameters())
    rng = np.random.default_rng(seed + 1)
    x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    ik = keys_of(seed + 2, nb, mode)
    return args, idx, cons, x, qr, fv, ik


def arm_row(arm: str, n: int, q32, ik, w32, lens, nb: int, align: int,
            impl: str = "buffer") -> tuple[np.ndarray, int, bool]:
    """The row the SHIPPED code produces for one arm, through its own route.

    Returns ``(row, effective_width, tiled)``.  The route is the shipped ``tiled()``
    decision for that arm: at ``nb`` close to ``cap`` the arm may be UNTILED, and then
    the width rule is irrelevant -- modelling those cells with a forced tiled row
    would over-state the change.
    """
    production_knobs()
    IX._TILED_IMPL = impl
    if arm == "UNT":
        IX._TILE = 0
    else:
        IX._SMALLN_WIDE_TILE = arm == "NEW"
    width = IX.tiled(1, n, HEADS, nb, align)
    tiled = width < nb
    production_knobs()
    IX._TILED_IMPL = impl
    return row_at_width(1, n, q32, ik, w32, lens, nb, width if tiled else 0), width, tiled


def score_cell(role: str, n: int, nb: int, mode: str, seed: int,
               impl: str = "buffer") -> dict[str, Any]:
    """Row/index-set/mask comparison across the three arms at production geometry."""
    production_knobs()
    IX._TILED_IMPL = impl
    args, idx, _, x, qr, fv, ik = inputs_for(seed, role, n, nb, mode)
    align = args.candidate_block_size if role == "source" else 1
    sp = nb - n                                    # ratio 1: the last row sees nb groups
    q32, w32, lens, _ = qw_of(idx, x, qr, sp, fv)
    wo, wn = old_width(1, n, nb, align), new_width(1, n, nb, align)
    rows, widths, routed = {}, {}, {}
    for arm in ("OLD", "NEW", "UNT"):
        rows[arm], widths[arm], routed[arm] = arm_row(arm, n, q32, ik, w32, lens, nb,
                                                       align, impl)
    r_old, r_new = rows["OLD"], rows["NEW"]
    sets = {k: topk_sets(r, INDEX_TOPK) for k, r in rows.items()}
    finite = np.isfinite(r_new) & np.isfinite(r_old)
    diff_mask = finite & (r_new != r_old)
    delta = np.abs(np.where(diff_mask, r_new, 0.0) - np.where(diff_mask, r_old, 0.0))
    # WHERE the arms differ and how close that element sits to the top-k boundary:
    # a difference only matters if it is inside the boundary margin of its row.
    cols = np.argwhere(diff_mask.reshape(-1, nb))
    rank_of = []
    for row_ix, col_ix in cols[:8]:
        row = r_new.reshape(-1, nb)[row_ix]
        rank_of.append(int((row > row[col_ix]).sum()))
    srt = np.sort(r_new, axis=-1)[..., ::-1]
    gap = (srt[..., INDEX_TOPK - 1] - srt[..., INDEX_TOPK]).ravel().tolist()
    min_gap = min(gap) if gap else float("inf")
    mask_diff = None
    if role == "source":
        m_old = np.array(IX.select_candidate_blocks(
            mx.array(r_old), mx.array(lens), args.candidate_topk_blocks,
            args.candidate_block_size))
        m_new = np.array(IX.select_candidate_blocks(
            mx.array(r_new), mx.array(lens), args.candidate_topk_blocks,
            args.candidate_block_size))
        mask_diff = int((m_old != m_new).sum())
    return record(dict(
        case="score_row", impl=impl, role=role, n=int(n), nb=int(nb), keys=mode,
        align=int(align), width_old=int(wo), width_new=int(wn),
        effective_old=int(widths["OLD"]), effective_new=int(widths["NEW"]),
        tiled_old=bool(routed["OLD"]), tiled_new=bool(routed["NEW"]),
        tiles_old=int(-(-nb // wo)), tiles_new=int(-(-nb // wn)),
        tail_old=int(nb % wo) or int(wo), tail_new=int(nb % wn) or int(wn),
        row_dtype="fp32" if IX._row_dtype(n) is mx.float32 else "bf16",
        row_elems_differing=int(diff_mask.sum()),
        max_abs_row_delta=float(delta.max()),
        differing_rank=rank_of,
        min_kth_gap=float(min_gap),
        delta_over_margin=(float(delta.max()) / min_gap if min_gap else float("inf")),
        sets_differing_new_vs_old=sum(1 for a, b in zip(sets["NEW"], sets["OLD"]) if a != b),
        sets_differing_new_vs_unt=sum(1 for a, b in zip(sets["NEW"], sets["UNT"]) if a != b),
        overlap_new_vs_old=round(mean_overlap(sets["NEW"], sets["OLD"]), 6),
        overlap_new_vs_unt=round(mean_overlap(sets["NEW"], sets["UNT"]), 6),
        kth_boundary_gap=[round(float(g), 12) for g in gap],
        mask_diff_bits=mask_diff))


def section_b(quick: bool) -> list[dict[str, Any]]:
    print("== B. score row / index sets at each arm's exact width ==", flush=True)
    n_grid = (1, 2, 4, 8) if quick else (1, 2, 3, 4, 5, 8, 12, 16)
    out = []
    for n in n_grid:
        for align in (1, BLOCK):
            # align only changes the aligned cap; probe near the crossing
            cap = max(align, (BUDGET_MB << 20) // (n * (HEADS * 4 + IX._row_dtype(n).size)))
            cap = max(align, (cap // align) * align)
            for mult in (1.0, 1.002):
                nb = int(cap * mult) + 1
                out.append(score_cell("source" if align != 1 else "plain",
                                      n, nb, "lattice", 100 + n))
    return out


def section_c() -> list[dict[str, Any]]:
    print("== C. the documented counterexample cells (deep, wide widths) ==", flush=True)
    out = []
    for n, nb in ((4, 1 << 20), (4, 1 << 19), (16, 1 << 20), (8, 1 << 20)):
        out.append(score_cell("plain", n, nb, "lattice", 300 + n))
    for n, nb in ((4, 1 << 20), (16, 1 << 20)):
        out.append(score_cell("plain", n, nb, "ties", 400 + n))
    return out


# ---------------------------------------------------------------------------
# D. the REAL Indexer.__call__ arms (end-to-end cidx + candidate mask)
# ---------------------------------------------------------------------------
def call_arms(role: str, n: int, nb: int, mode: str, seed: int) -> dict[str, Any]:
    production_knobs()
    args, idx, cons, x, qr, fv, ik = inputs_for(seed, role, n, nb, mode)
    sp = nb - n
    out: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray | None] = {}
    for arm in ("NEW", "OLD", "UNT"):
        production_knobs()
        IX._SMALLN_WIDE_TILE = arm != "OLD"
        if arm == "UNT":
            IX._TILE = 0
        sh = SharedState()
        src, consumer = (idx, cons) if cons is not None else (idx, None)
        a = np.array(src(x, qr, sp, 0, fv, ik, sh))
        if consumer is not None:
            a = np.array(consumer(x, qr, sp, 0, fv, ik, sh))
        out[arm] = a
        masks[arm] = None if sh.candidates is None else np.array(sh.candidates)
    production_knobs()

    def rows_differ(a: np.ndarray, b: np.ndarray) -> int:
        return int((a != b).any(-1).sum())

    return record(dict(
        case="real_call_arms", role=role, n=int(n), nb=int(nb), keys=mode,
        compile=int(IX._INDEXER_COMPILE),
        rows_new_vs_old=rows_differ(out["NEW"], out["OLD"]),
        rows_new_vs_unt=rows_differ(out["NEW"], out["UNT"]),
        bytes_equal_new_vs_old=bool((out["NEW"] == out["OLD"]).all()),
        bytes_equal_new_vs_unt=bool((out["NEW"] == out["UNT"]).all()),
        mask_bits_new_vs_old=(None if masks["NEW"] is None
                              else int((masks["NEW"] != masks["OLD"]).sum())),
        mask_bits_new_vs_unt=(None if masks["NEW"] is None
                              else int((masks["NEW"] != masks["UNT"]).sum()))))


def section_d(quick: bool) -> list[dict[str, Any]]:
    print("== D. the real Indexer.__call__ arms (cidx + candidate mask) ==", flush=True)
    out = []
    cells = (("plain", 4, 254_300, "lattice"), ("source", 4, 254_300, "ties"),
             ("consumer", 4, 254_300, "lattice"))
    if not quick:
        cells += (("plain", 16, 63_700, "lattice"), ("source", 16, 63_700, "ties"),
                  ("plain", 4, 900_000, "lattice"))
    for role, n, nb, mode in cells:
        out.append(call_arms(role, n, nb, mode, 500 + n))
    return out


# ---------------------------------------------------------------------------
# E. non-vacuity + the merge-impl arm
# ---------------------------------------------------------------------------
def section_e() -> dict[str, Any]:
    print("== E. non-vacuity controls + the merge implementation ==", flush=True)
    production_knobs()
    n, nb = 4, 254_300
    args, idx, _, x, qr, fv, ik = inputs_for(901, "plain", n, nb, "lattice")
    q32, w32, lens, _ = qw_of(idx, x, qr, nb - n, fv)
    r = row_at_width(1, n, q32, ik, w32, lens, nb, new_width(1, n, nb, 1))
    rows2 = r.reshape(-1, nb)
    base = topk_sets(r, INDEX_TOPK)
    order = np.argsort(-rows2, axis=-1)
    # (1) SENSITIVITY: raise the first OUT-of-set column above the k-th boundary by
    #     exactly its own margin (k-th gap) + 1 ulp -> that row's SET must change.
    #     Reports the smallest margin the harness can resolve, which is the scale a
    #     cross-arm difference would have to exceed to matter.
    flips = 0
    margins: list[float] = []
    for i in range(rows2.shape[0]):
        row = rows2[i].copy()
        j_in = int(order[i, INDEX_TOPK - 1])
        j_out = int(order[i, INDEX_TOPK])
        margin = float(row[j_in] - row[j_out])
        if margin > 0.0:
            margins.append(margin)
        bumped = row.copy()
        bumped[j_out] = np.nextafter(np.float32(row[j_in] + np.float32(margin)),
                                     np.float32(np.inf))
        s2 = set(int(t) for t in np.argsort(-bumped, kind="stable")[:INDEX_TOPK])
        flips += int(s2 != base[i])
    # (2) an injected EXACT tie at the boundary (periodic keys): the tie-break is
    #     what decides, and it is decided by the row the arms share.
    ik_t = keys_of(902, nb, "ties")
    rt = row_at_width(1, n, q32, ik_t, w32, lens, nb, new_width(1, n, nb, 1))
    srt_t = np.sort(rt, axis=-1)[..., ::-1]
    tied = int((srt_t[..., INDEX_TOPK - 1] == srt_t[..., INDEX_TOPK]).sum())
    # (3) the non-default running-merge implementation: its per-tile concatenation is
    #     k + width long, so the argpartition it runs has a WIDTH-DEPENDENT length --
    #     the one place a width change can move a tie pick even with bitwise-identical
    #     per-tile scores.  Recorded, not asserted: it is not the shipped default.
    m = score_cell("plain", n, nb, "lattice", 903, impl="merge")
    record(dict(case="non_vacuity", rows=int(rows2.shape[0]), margin_flips=int(flips),
                min_margin=(min(margins) if margins else 0.0),
                tie_rows_at_k=tied))
    return {"margin_flips": int(flips), "tie_rows_at_k": tied, "merge_cell": m}


def section_f() -> dict[str, Any]:
    print("== F. kill switch / env read ==", flush=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    flags = {}
    for value in (None, "1", "0"):
        env = {k: v for k, v in os.environ.items()
               if k != "DSV41_INDEXER_SMALLN_WIDE_TILE"}
        if value is not None:
            env["DSV41_INDEXER_SMALLN_WIDE_TILE"] = value
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run(
            [sys.executable, "-c",
             "from mlx_lm.models.deepseek_v41 import indexer as I;"
             "print('FLAG', int(I._SMALLN_WIDE_TILE))"],
            env=env, capture_output=True, text=True, check=True, cwd=root)
        line = [s for s in out.stdout.splitlines() if s.startswith("FLAG")][-1]
        flags[str(value)] = line.split()[1] == "1"
    production_knobs()
    IX._SMALLN_WIDE_TILE = False
    off_equals_old = all(IX.tile_width(1, n, HEADS, 300_000, 1) == old_width(1, n, 300_000, 1)
                         for n in (1, 4, 16))
    IX._SMALLN_WIDE_TILE = True
    return record(dict(case="kill_switch", flags=flags,
                       width_off_equals_old=bool(off_equals_old)))


def section_g() -> dict[str, Any]:
    """Census: at how many engagement SHAPES does the width change the row at all,
    and at every one of them, does anything downstream move?

    The arms differ in the TAIL tile's column count (``nb mod width``): the old arm's
    tail is ``nb mod 512`` wide, the new arm's is ``nb mod cap`` (often 1 column).  A
    different tail width means a different GEMM N for the tail and therefore a
    different kernel for it -- the shape class where an ulp difference can appear.
    """
    print("== G. shape census: how often does the width change the row at all? ==",
          flush=True)
    production_knobs()
    cells = row_diff = sel_diff = 0
    max_delta = 0.0
    worst_ratio = 0.0
    examples: list[dict[str, Any]] = []
    for n in (4, 16):
        cap = new_width(1, n, 10 ** 9, 1)
        offsets = [1, 2, 3, 4, 8, 16, 64, 256, 511, 512, 513, 1024]
        for off in offsets:
            for base in (cap, 2 * cap):
                nb = base + off
                rec = score_cell("plain", n, nb, "lattice", 700 + n + off)
                cells += 1
                if rec["row_elems_differing"]:
                    row_diff += 1
                    max_delta = max(max_delta, rec["max_abs_row_delta"])
                    worst_ratio = max(worst_ratio, float(rec["delta_over_margin"]))
                    if len(examples) < 6:
                        examples.append({k: rec[k] for k in
                                         ("n", "nb", "width_old", "width_new", "tail_old",
                                          "tail_new", "row_elems_differing",
                                          "max_abs_row_delta", "differing_rank",
                                          "min_kth_gap")})
                sel_diff += int(bool(rec["sets_differing_new_vs_old"]
                                     or rec["sets_differing_new_vs_unt"]
                                     or rec["mask_diff_bits"]))
    return record(dict(case="shape_census", cells=cells, cells_with_row_difference=row_diff,
                       max_abs_row_delta=max_delta,
                       max_delta_over_margin=worst_ratio,
                       cells_with_selection_difference=sel_diff, examples=examples))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("BH_AGENT_WORKSPACE", "."), "dsv41_indexer_refdiv.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--compile", default="1", choices=("0", "1"))
    cli = ap.parse_args()
    IX._INDEXER_COMPILE = cli.compile == "1"
    IX._COMPILED_BODIES.clear()

    section_a()
    section_b(cli.quick)
    if not cli.quick:
        section_c()
    section_d(cli.quick)
    section_e()
    section_f()
    if not cli.quick:
        section_g()

    # A row-level ulp difference is the documented behaviour of this change (the code
    # comment records it); it becomes a BLOCKER only when it moves a selection: the
    # index set, the candidate mask, or the real __call__ output.
    row_ulp = [r for r in RESULTS if r.get("row_elems_differing")]
    selection = [r for r in RESULTS if
                 r.get("sets_differing_new_vs_old") or r.get("sets_differing_new_vs_unt")
                 or r.get("rows_new_vs_old") or r.get("rows_new_vs_unt")
                 or (r.get("mask_diff_bits") or 0)
                 or (r.get("mask_bits_new_vs_old") or 0)
                 or (r.get("mask_bits_new_vs_unt") or 0)]
    with open(cli.out, "w") as fh:
        json.dump({"cells": RESULTS, "row_ulp_cells": row_ulp,
                   "selection_divergences": selection}, fh, indent=1)
    print(f"\n[{len(RESULTS)} cells | row-ulp-differing {len(row_ulp)} (INFORMATIONAL: the "
          f"documented ulp class) | SELECTION-diverging {len(selection)}] -> {cli.out}",
          flush=True)
    # Exit status is a SELECTION gate: a row-level ulp difference at the tail column is
    # the documented behaviour of this change (see the module docstring); only a moved
    # index set / candidate mask / cidx is a failure.
    return 1 if selection else 0


if __name__ == "__main__":
    sys.exit(main())
