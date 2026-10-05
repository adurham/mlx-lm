# DSv4.1 hierarchical / streamed indexer exact pass — design (milestone 1)

**Date:** 2026-10-04
**Branch:** `perf/dsv41-hierarchical-indexer`
**Status:** M1 prototype complete (module + tests + this doc). **Not integrated.**
**Prototype:** `mlx_lm/models/deepseek_v41/indexer_hierarchical.py`
**Tests:** `tests/test_dsv41_indexer_hierarchical.py` (17 tests, all passing)

---

## 1. The problem

The indexer's exact top-k pass scores every compressed column for every query
row and *keeps the result as a materialized `[b, n, nb]` fp32 row*, then runs one
global `argpartition` over it
(`indexer._tiled_scores_buffer`; `nb = end_pos / ratio`, up to ~1M at 1M
context). At chunk `n=2048`, ratio 1, `nb=1048576` that row is
`1 * 2048 * 1048576 * 4` = **8.0 GiB per index layer**, materialized and then
exact-partitioned every chunk.

Everything the engine does to survive that row exists for it and nothing else:

* the **every-2-layer `mx.eval` fence** (bounds the lazy graph that would hold
  several rows live), and
* the **transient-budget chunk-shrink policy** (shrinks the chunk as context
  grows so the row stays within a memory budget).

This exact pass is inferred to be 40–55 % of deep incremental prefill time
(attribution measurement pending — see the M2 gate). Removing the row lets both
compensations be deleted (M3).

## 2. Mechanism

A two-level selection that mirrors the existing layer-20 candidate machinery
(`candidate_topk_blocks=2048`, `candidate_block_size=8`,
`indexer.select_candidate_blocks`):

1. **Coarse** — `coarse_block_scores`: stream `nb` in `strip`-wide column strips
   (block-aligned), score each strip **in bf16**, collapse to per-*block*
   (block = 8 contiguous columns) maxima, write into an `[b, n, nb/block]`
   buffer. The full `[b, n, nb]` is never allocated; the only score transient is
   `[b, n, strip]`.
2. **Top blocks** — `top_blocks`: `argpartition` the maxima for the top
   `k + overfetch` blocks (k = `index_topk` = 512, overfetch ≈ 16), ascending.
3. **Exact** — `exact_rescore_streaming`: re-score **only those blocks'
   columns**, at fp32, in strips, with a running top-k. Largest transients are
   `[b, n, c]` (scores) and `[b, n, c, d]` (gathered keys), `c = blocks/strip ×
   block`. Returns `(values, indices)` sorted ascending — the same contract as
   `indexer.topk_from_row`.

`hierarchical_topk` composes the three.

## 3. Why it fits

The row is the **top-k selection input**, not a compute chain. Nothing
downstream reads the score values except the selection itself: the selected
indices go to `sparse_attention._gather_split`, which gathers keys by index and
computes attention independently. Therefore:

* approximate *ranking* is tolerated as long as the selected set is right, so
  the coarse pass can run in bf16 and over a pruned block set; and
* the *values actually used* come from the exact fp32 re-score, so precision is
  preserved where it matters.

This is exactly the split the existing two-level candidate path already relies
on (coarse block max → mask → exact top-k), extended so that the row is never
materialized.

## 4. Exactness of the prune

Let `v*` be the k-th largest score of a row and `j = #{columns > v*}` (`j ≤ k−1`).
Each block containing a column `> v*` has block-max `> v*`, and each such block
contains ≥ 1 distinct such column, so **at most `j ≤ k−1` blocks have max
`> v*`**. The top-`k` blocks therefore always include every block holding a
column strictly greater than `v*`; the exact re-score then fills the remaining
`k − j` slots from blocks whose max `= v*`. **Under exact-arithmetic block maxima
the returned top-k values equal the global top-k values for any distribution — a
proof, not a heuristic.** (The bf16 *coarse pass* computes the maxima
approximately, so this guarantee is degraded to "exact iff the coarse top-B
blocks contain every true top-k block", which `overfetch` makes hold empirically;
the engineered bf16-alias case below shows value recall dipping to 0.86 at
overfetch 0 and recovering to 1.0 at overfetch 16 — the mitigation, not a proof.)

* Continuous scores (the generic case): `v*` is unique, one block has max `= v*`,
  it ranks at or above position k, and the top-k **indices** are recovered
  exactly. Measured rank-weighted recall is 1.0 on every realistic distribution
  even at overfetch 0.
* Ties at `v*`: the pick among tied blocks is arbitrary in *both* this scheme and
  production `argpartition`, independently — values match, indices within the tie
  class may not. Do not assert index equality on tied distributions.
* `overfetch` is **not** tie insurance (`B ≥ k` already covers ties). It insures
  against the bf16 coarse pass reordering blocks whose true maxima are genuinely
  close: loss needs a top-k-bearing block leapfrogged by `≥ B − rank` others, so
  a margin of `overfetch` absorbs up to `overfetch` such blocks.

## 5. Work / memory algebra

Per chunk, per index layer, `d = index_head_dim = 128`, `block = 8`,
`k = 512`, `overfetch = 16` (`B = 528` candidate blocks).

**Work.**

| term | production | hierarchical |
|---|---|---|
| columns scored, fp32 exact | `n·nb` | `n·B·block` (only candidate blocks) |
| columns scored, bf16 | 0 | `n·nb` (coarse sweep) |
| row materialization + global argpartition | `n·nb` fp32 | eliminated |

The exact fp32 work drops by ~`nb/(B·block)`: **~124×** at nb=524288, **~248×**
at nb=1048576. The coarse sweep still touches all `n·nb` columns — but in bf16,
without ever holding them. **Honest note on the task's work model:** the stated
`O(n·(nb/block + B·block))` coarse cost is achievable only with a *downsampled*
coarse (one representative per block), which is a lower bound on the true block
max and therefore **breaks the exactness proof in §4**. True block maxima require
reading every column (no exact shortcut for the max of arbitrary dot products),
so the real coarse cost is `Θ(n·nb)` at reduced precision. The achievable win is
**memory footprint** (row → maxima buffer, ~16×) plus the exact-pass work
reduction, not a reduction in coarse columns scored. This tension is inherent to
exactness-preserving two-level selection.

**Memory** (chunk `n=2048`, `b=1`; `2^30 = 1 GiB`):

| `nb` | fp32 row (production) | bf16 row (sibling) | hier maxima fp32 | hier maxima bf16 | exact-pass peak* |
|---|---|---|---|---|---|
| 16 384 | 128 MiB | 64 MiB | 16 MiB | 8 MiB | ~128 MiB |
| 131 072 | 1.0 GiB | 512 MiB | 128 MiB | 64 MiB | ~128 MiB |
| 524 288 | 4.0 GiB | 2.0 GiB | 512 MiB | 256 MiB | ~128 MiB |
| 1 048 576 | 8.0 GiB | 4.0 GiB | 1.0 GiB | 512 MiB | ~128 MiB |

\* exact-pass peak: the transient chain is gather(bf16) -> cast(fp32) -> score/reduce,
  so the limiter is the **key** gather pair: `b·n·c·(2d + 4d)` = `b·n·c·6d`
  (2 bytes/elt for the bf16 gather + 4 bytes/elt for the fp32 working copy per
  d-element, plus the `b·n·c·4` fp32 score — dominated by the keys). Independent of `nb` —
  the deep-context scaling term is gone. The maxima buffer scales as `nb/16`
  (bf16) vs `nb` (fp32 row): **16× smaller**; per-row (n=1) the row goes
  `64 KiB → 4 KiB` at nb=16384 and `4 MiB → 256 KiB` at nb=1048576.

**Exact-pass strip sizing (M2 finding).** The memory limiter in the exact pass is
the **key** gather `[b, n, c, d]` in BOTH the gathered dtype (bf16, 2 B/elt) and
its fp32 working cast (4 B/elt), not the score `[b, n, c]`. For a budget
`M`: `c ≤ M / (b·n·(6d + 4))`. At `n=2048, d=128, b=1, M=128 MiB` that is
`c ≈ 87` columns ⇒ ~10-11 blocks per strip (`bp ≈ 10`). Production `strip` must be
derived from this key budget, not chosen for the coarse score transient (which
can be far larger). See §8.

## 6. Precision split

* **Coarse (bf16):** ranking only. bf16 preserves the order of well-separated
  block maxima; `overfetch` absorbs the rest. The coarse error is *accumulated
  input rounding* (inputs rounded before the einsum, ~`√d·2⁻⁸` relative), not
  just final-value quantization (`2⁻⁹`).
* **Exact (fp32):** the returned values. Same expression order as
  `indexer._score_body_eager` on the same bf16 key buffer, so an fp32 call is
  elementwise-equal to the production per-column score.

## 7. Index space

Identical to production: returned indices are raw compressed-column positions in
`[0, nb)` — exactly what `indexer.topk_from_row` returns, *before* the
`+ offset` (`offset = wp + n`) and `-1` masking `Indexer.__call__` applies.
Confirmed against `sparse_attention._gather_split` (indices address
`concat(window+chunk, comp_kv)` rows) and by direct test:
hierarchical indices **exactly equal** `topk_from_row`'s on continuous
distributions; ascending; clamped to `[0, nb)`.

## 8. Risks

1. **Score-distribution skew → coarse misses.** Mitigated by §4 (values exact)
   plus `overfetch`. Measured adversarial recall is 1.0 (uniform, clustered
   near-ties, top-heavy, boundary-straddling) with real multi-head keys, because
   independent head variance prevents *per-row* block-max ties. The theoretical
   worst case is a **column-wise constant** (all `n` rows share a monotone max
   profile so coarse ranking errors correlate across rows) — not produced by
   real weights; the multi-head sum's per-row variance is what saves it. M2
   should add a **runtime certificate**: let `c` = the B-th coarse max and `v_k`
   the returned k-th exact value; if `v_k > c/(1−ε)` for a coarse-error bound
   `ε`, the result is provably exact, else expand `B` (the maxima buffer must be
   kept for that).
2. **Block-boundary effects.** Strips are rounded to block multiples;
   `nb` not divisible by `block` is padded with `-inf`; peaks sitting on block
   edges are tested (recall 1.0). `nb < block` and `nb ∈ {1,3,7}` verified.
3. **Index-space / padding correctness.** `-1`/`-inf` padding never leaks: masked
   columns are `-inf`, invisible columns filtered, index clamped `< nb`. Verified
   for all-masked rows, partial visibility, and tiny `nb`.
4. **Interaction with the layer-20 candidate path — does it REPLACE or COMPOSE?**
   **Both, in a specific sense:**
   * For layers that own a full-space exact top-k (index sources 2/8/14 and the
     candidate source 20): the hierarchical pass **REPLACES** the
     row-materializing exact pass. Layer 20's candidate publishing is preserved
     *for free* — its coarse stage already computes the per-block maxima that
     `select_candidate_blocks` derives its mask from, so the **same maxima
     buffer publishes the candidate mask and feeds the hierarchical top-blocks**.
     M2 can fuse candidate publishing and the exact pass into one coarse sweep
     (removing a redundant full-row pass over the row).
   * For consumer layers (24..36): the pass **COMPOSES** with the candidate path.
     It consumes `shared.candidates` and runs coarse → top-blocks → exact
     *within the candidate columns only*, preserving the exact two-level
     semantics (mask, then top-`index_topk`).
   * Net: the two-level machinery is **subsumed** — the hierarchical coarse stage
     is strictly the per-block-max information layer 20 already computes, so the
     separate candidate-only pass is no longer needed. It is not deleted; it is
     folded in.
5. **Compensation interactions (M3).** Removing the row lets the every-2-layer
   fence and the chunk-shrink policy go, but the new path still needs a per-strip
   `mx.eval` (to bound the lazy graph). That fence cadence is per-strip and far
   cheaper than every-2-layers; M3 must re-tune it. Gated on the soak below.
6. **bf16 coarse cost is still Θ(n·nb).** See §5 — an honest limitation of any
   exactness-preserving scheme; the win is footprint + exact-pass columns.

## 9. Milestones and gates

* **M1 (this):** prototype module + tests + design. Pure functions on synthetic
  tensors only; no production path touched.
* **M2 — integration into `indexer.py`.** **Gated on a cluster attribution
  measurement** showing the `[b,n,nb]` row at **≥ ~40 %** of deep incremental
  prefill time. If attribution is below that, M2 is not justified (the
  optimization targets the wrong term) — re-scope.
  * **Recall gate (synthetic):** ≥ 99 % **rank-weighted top-512 recall** vs exact
    at `nb = 524288` and an `nb = 1048576`-equivalent synthetic shape (small `n`
    for memory — the full shape does not fit locally).
  * **Live-battery gate:** NLL / top-k-overlap parity on a real activation
    battery (e.g. the `wsB-ab`-style captures) after integration; synthetic
    recall is necessary but not sufficient.
  * Add the runtime certificate (§8.1) and split the strip knob into
    coarse-strip vs exact-blocks-per-strip (§5).
* **M3 — remove the compensations.** Delete the every-2-layer eval fence and the
  transient-budget chunk-shrink policy. **Gated on a ≤ 120 GB soak** on the
  cluster (peak memory must stay within the M4 Max / 2× Mac Studio envelope
  across a full-context prefill).

## 10. M2 knobs

Module constants (no `os.environ` in M1, kept pure): `HIER_BLOCK = 8`,
`HIER_STRIP = 4096`, `HIER_OVERFETCH = 16`, `HIER_COARSE_DTYPE = bf16`.
Intended env names for the eventual integration, mirroring the
`DSV41_INDEXER_TILE*` neighbours:

* `DSV41_INDEXER_HIER` — master gate (default off until M2 lands)
* `DSV41_INDEXER_HIER_BLOCK`
* `DSV41_INDEXER_HIER_STRIP` (and, per §5, a separate
  `DSV41_INDEXER_HIER_EXACT_STRIP` / blocks-per-strip derived from the key budget)
* `DSV41_INDEXER_HIER_OVERFETCH`

## 11. M1 evidence (summary; raw in the commit message / test output)

* Tests: **17 passed** (`tests/test_dsv41_indexer_hierarchical.py`); **50
  passed** including the indexer-touching neighbours (gather_direct 10,
  cache_growth 14, rope_percall 9). 0 failures.
* Realistic recall (rank-weighted, gate 0.99): sparse_peaks / exp_tail /
  pareto_1.5 / gauss_relu / top_heavy = **1.0**; real multi-head keys
  (n64×16384×h8×d128, n32×32768×h4×d64) = **1.0**; big-nb emulation
  (nb=524288, n=16) = **0.9985**.
* Adversarial recall (gate 0.95, reported): uniform / clustered near-ties /
  top-heavy / boundary-straddling = **1.0** (real multi-head keys; independent
  head variance removes per-row block-max ties).
* Over-fetch sweep (engineered bf16-aliased near-ties: 528 blocks at the cut):
  of=0 → **0.9768**, of=4 → **0.9830**, of=16 → **1.0000**, of=64 → **1.0000**.
  Real keys (nb=32768): of=0 → 0.99992, of=4 → 0.99998, of=16 → 1.0.
* No-materialization: alloc spy confirms allocations are
  `[b,n,nb/block]`, `[b,n,k]` only — never `[b,n,nb]`; positive control catches a
  real full-row alloc. Max transient last-dim 8192 = `nb/block` < nb=65536.
* Index-space: exact equality with `topk_from_row`; ascending; in `[0, nb)`.
* Coarse vs materialized-row block-max: max relative diff **0.0**.

## 12. NOT verified (explicit)

* No cluster run, no real-model trace, no attribution measurement (by design —
  M2 gate).
* Recall measured on **synthetic distributions only**; live-battery / NLL gate
  pending M2.
* Integration, compensation removal (M3), and the soak are all pending.
* bf16 coarse error is characterized empirically, not bounded; the runtime
  certificate (§8.1) is proposed but not implemented.
