# DSv4.1 hierarchical / streamed indexer exact pass — M2 integration

**Date:** 2026-10-05
**Branch:** `perf/dsv41-hier-integration`
**Base:** `d219882` (deploy/next2: origin/main + bf16 score row + fence hook + keep-0)
**Status:** integrated behind `DSV41_INDEXER_HIER` (default **OFF**). Not enabled.
**M1:** `docs/dsv41-hierarchical-indexer-design-2026-10-04.md`
**Tests:** `tests/test_dsv41_hier_integration.py` (21 tests).

---

## 1. What M2 does

Wires the M1 streamed exact pass into the production `Indexer.__call__` behind
`DSV41_INDEXER_HIER=1`. With the gate **off** the new branch is never entered
(spy-verified) and the production path is byte-for-byte what it was.

With the gate **on**, every index-source layer's exact top-k is produced by one
coarse sweep (bf16 block maxima, stripped) → top-`(k+overfetch)` blocks →
streamed fp32 re-score of only those blocks' columns. The `[b, n, nb]` score row
is never materialized (alloc-spy verified).

* **Owner roles (exact top-k, index sources 2/8/14/20):** the hierarchical pass
  **replaces** the row-materializing exact pass.
* **Layer 20 (candidate source):** candidate publishing is **fused** — the same
  coarse maxima buffer that feeds `top_blocks` also publishes the candidate mask
  via `indexer_hierarchical.candidate_mask_from_block_maxima`. No second pass.
  The published mask is exactly `[b, n, nb]` (truncated from the block-padded
  grid), matching production `select_candidate_blocks` shape.
* **Consumer roles (24..36):** the pass **composes** with `shared.candidates`:
  the coarse maxima already exclude non-candidate columns, so coarse → top-blocks
  → exact all run *within the candidate set* (two-level semantics preserved).

## 2. Strip sizing (the M1-review correction)

The exact-pass memory limiter is the per-row **key** gather `[b, n, c, d]` in
its bf16 gathered (2 B/elt) **and** fp32 working-cast (4 B/elt) forms, plus the
`[b, n, c]` fp32 score — i.e. `c <= M / (b*n*(6d + 4))`.
`indexer.hier_strip_for_budget(bsz, n, head_dim, budget_bytes)` implements this
and rounds **down** to a whole number of blocks. At `n=2048, d=128, b=1,
M=128 MiB` that is 80 columns ⇒ **10 blocks/strip**, vs the superseded
score-only `4 + 4d` bound's 127 columns/15 blocks. Coarse and exact strips are
separate knobs (coarse `[b,n,strip]` bf16 can be wide; exact is key-budgeted).

## 3. Knobs (read once at import, like the other `DSV41_*`)

* `DSV41_INDEXER_HIER` — master gate, default `0` (OFF)
* `DSV41_INDEXER_HIER_BLOCK` — coarse block = candidate block, default `8`
* `DSV41_INDEXER_HIER_STRIP` — coarse column strip, default `4096`
* `DSV41_INDEXER_HIER_OVERFETCH` — candidate-block margin beyond k, default `16`
* `DSV41_INDEXER_HIER_EXACT_MB` — exact-pass key budget MiB, default follows
  `DSV41_INDEXER_TILE_MB` (128)
* `DSV41_INDEXER_HIER_EXACT_STRIP` — direct exact-strip columns (0 = derive)

`DSV41_INDEXER_HIER_BLOCK` **must equal** `candidate_block_size` on the
candidate-source layer (the mask is derived from the coarse grid); the indexer
asserts this rather than silently mismatching. (Both are 8 in the release.)

## 4. What is verified (raw evidence in the commit / test output)

* **OFF == production, byte-identical** (indices *and* selected scores) through
  the real `__call__`; OFF never enters the hier branch.
* **ON vs exact, rank-weighted top-k recall = 1.0** on realistic one-hot
  (sparse_peaks/exp_tail/pareto_1p5/gauss_relu/top_heavy) and real multi-head
  keys at production k=512 (gate 0.99).
* **Candidate-mask parity:** fused mask == `select_candidate_blocks` fed the
  *same* coarse row — **exact**, and coarse blockmax bit-equal to the row's
  reshape-max (abs diff 0.0).
* **Composition:** every finite consumer index lies inside the mask; ON vs the
  production consumer path recall 1.0; consumer ON vs OFF overlap 1.0 through
  the real two-layer call, mask preserved.
* **Strip formula:** budget-respecting, block-aligned, ~10-11 blocks at the
  production point (not 15).
* **No `[b,n,nb]` allocation** when ON (owner/source/consumer), with a positive
  control.
* Suites: 52 passed (bf16_row + session_cache + hierarchical + integration);
  166 passed across all dsv41; ruff clean; basedpyright zero-new on the source
  files (same 4 pre-existing: unresolved mlx imports + one `dict` generic).

## 5. NOT verified

* No live-battery / NLL / top-k-overlap on real activations — the ship gate.
* No cluster run, no attribution measurement, no 1M-context soak.
* Synthetic recall only; the runtime certificate (M1 §8.1) is not implemented.
* M3 (removing the every-2-layer fence and the chunk-shrink policy) untouched.
* Not enabled by default; `DSV41_INDEXER_HIER` stays `0`.
