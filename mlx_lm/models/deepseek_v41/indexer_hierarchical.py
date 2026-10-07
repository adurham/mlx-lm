# Copyright © 2026 Adam Durham (hermes-gw)
"""PROTOTYPE (milestone 1) — hierarchical / streamed exact top-k for the indexer.

This module is **not wired into any production path**. It exists to prototype,
on synthetic tensors only, the exact-selection replacement for the indexer's
``[b, n, nb]`` score row (:func:`mlx_lm.models.deepseek_v41.indexer._tiled_scores_buffer`).
Integration is milestone 2, and is gated on a cluster attribution measurement
showing the row at >= ~40 % of deep incremental prefill time.

Why the row is the problem
--------------------------
The indexer's exact pass scores every one of the ``nb = end_pos // ratio``
compressed columns for every one of the ``n`` query rows of a chunk, at fp32,
and *keeps the whole thing* — as an ``[b, n, nb]`` fp32 row (the fp32
``[b, n, heads, nb]`` pre-collapse transient is tiled away, but the row is
written and then exact-``argpartition``ed in one global pass). At 1M context,
chunk 2048, ratio 1 that row is ``1 * 2048 * 1048576 * 4`` = 8.0 GiB per index
layer (matching the ~8.6 GB figure once heads/other layers are counted), or
~4.0 GiB at bf16. Everything the engine does to survive it — the every-2-layer
``mx.eval`` fence and the transient-budget chunk-shrink policy — exists *only*
because this row is materialized. Delete the row and both compensations go
(milestone 3).

Mechanism
---------
Two levels, mirroring the existing layer-20 candidate machinery
(``candidate_topk_blocks=2048, candidate_block_size=8``,
:func:`indexer.select_candidate_blocks`):

1. **Coarse** (:func:`coarse_block_scores`): stream ``nb`` in ``strip``-wide
   column strips, collapse each strip to per-*block* maxima (block = 8
   contiguous columns), write the maxima into an ``[b, n, nb/block]`` buffer.
   The full ``[b, n, nb]`` is NEVER materialized — only ``[b, n, strip]`` score
   transients exist. Run in bf16 (the coarse pass only *ranks*).
2. **Top blocks** (:func:`top_blocks`): ``argpartition`` the block-maxima row
   for the top ``k + overfetch`` blocks (the over-fetch absorbs near-tie
   ranking flips introduced by the bf16 coarse pass).
3. **Exact** (:func:`exact_rescore_streaming`): re-score ONLY the columns of
   those candidate blocks — at fp32 — in strips, with a running top-k, so the
   largest transient is ``[b, n, strip]`` (scores) plus ``[b, n, strip, d]``
   (gathered keys), never ``[b, n, nb]``. The final ``(values, indices)`` are
   sorted ascending by index, exactly like the production
   :func:`indexer.topk_from_row`.

Correctness of the coarse prune (why recall is ~exact)
------------------------------------------------------
Let ``v*`` be the k-th largest score of a row and ``j = #{columns > v*}``
(``j <= k-1``). Every column ``> v*`` sits in a block whose max is ``> v*``, and
one block holds at least one such column, so the number of blocks with
block-max ``> v*`` is ``<= j <= k-1 < k``. The top-``k`` blocks by block-max
therefore **always include every block that contains a column strictly greater
than** ``v*``; the exact re-score over those columns then picks the remaining
``k - j`` slots from blocks whose max is ``= v*``. The returned top-k *values*
are thus exactly the global top-k values, for **any** score distribution —
this is a proof, not a heuristic. Two caveats, both about *indices/value ties*,
not values:

* If scores are continuous (the generic case), ``v*`` is unique, exactly one
  block has max ``= v*``, it ranks at or above position ``k``, and the top-k
  *indices* are recovered exactly too. This is why recall is ~1.0 on realistic
  (continuous) logits even with ``overfetch=0``.
* If many columns are exactly tied at ``v*``, the pick among the tied blocks is
  arbitrary — in *both* this scheme and production ``argpartition``, and
  *independently* so. Values still match; indices within the tie class may not.
  Do not assert index equality against a reference on tied distributions.

``overfetch`` is **not** tie insurance (``B >= k`` already suffices for ties).
It is insurance against the *bf16 coarse pass* reordering blocks that are
genuinely close — the coarse scores are computed in bf16 (input rounding +
bf16 accumulation), so a block whose true max is just below the cut can rank
just above it. Loss occurs only when a top-k-bearing block is leapfrogged by
at least ``B - rank`` such blocks, so a margin of ``overfetch`` absorbs up to
``overfetch`` leapfroggers. (A runtime certificate — if the returned k-th exact
value exceeds the B-th coarse max by more than the coarse error bound, the
result is provably the global top-k — is a candidate M2 addition; see the
design doc.) See ``docs/dsv41-hierarchical-indexer-design-2026-10-04.md``.

Work / footprint algebra
------------------------
Per chunk, per index layer. Let ``B = k + overfetch`` (candidate blocks) and
``F`` = cost of scoring one column (a ``d``-wide dot per head; same for all
paths). ``L`` = bytes per fp32 number.

* Production (today): ``n * nb`` columns scored, ``n * nb * L`` row resident.
* This prototype: the coarse pass scores ``n * nb`` columns **but only in
  bf16** and materializes ``n * (nb/block)`` maxima, never the row; the exact
  pass scores ``n * B * block`` columns at fp32. So

      work  = n*nb*F (bf16)        + n*B*block*F (fp32)
      peak  = n*(nb/block)*L_mx     + n*strip*(L + d*L)   [maxima + strip]

  Concretely at block=8, k=512, B=528, fp32 row:

      nb         row (fp32)   row (bf16)   hier maxima (fp32)
      16384      64 MiB        32 MiB       8 MiB
      131072     512 MiB       256 MiB      64 MiB
      524288     2.0 GiB       1.0 GiB      256 MiB
      1048576    4.0 GiB       2.0 GiB      512 MiB      (per b=1, n=1 row)

  i.e. the resident footprint shrinks by ``block`` (8x) against the fp32 row,
  or 16x against it once the coarse maxima are kept bf16 — which is where the
  task's "~8x at 450 K, 15-20x at 1M" figures come from (8 = block, 16 =
  block x bf16). The quoted *work* model ``n*(nb/block + B*block)`` — coarse
  touching only ``nb/block`` columns — would additionally require a
  *downsampled* coarse (pool/sample one representative per block), which is
  exactly the thing that loses the exactness guarantee above; this prototype
  deliberately keeps true block maxima, so its coarse still reads all ``nb``
  columns, just cheaply and in bf16 and without ever holding them. See the
  design doc for the full table and the tradeoff.

Index space
-----------
Identical to production: the returned indices are compressed-column positions
in ``[0, nb)`` (the value :func:`indexer.topk_from_row` returns, *before* the
``+offset`` and the ``-1`` masking that :class:`indexer.Indexer` applies). Feed
them through the same ``idx + offset`` (``offset = wp + n``) mapping and mask
``-1`` where the value is non-finite.

Precision split (bf16 coarse / fp32 exact)
------------------------------------------
Coarse ranking is tolerant: bf16 preserves the *order* of well-separated block
maxima, and ``overfetch`` absorbs the rest. The exact re-score — whose values
are the ones actually gathered into the sparse attention — stays fp32 and uses
the same expression as :func:`indexer._score_body_eager`, so it is
elementwise-equal to the production score for those columns.

M2 knobs
--------
Module constants below (:data:`HIER_BLOCK`, :data:`HIER_STRIP`,
:data:`HIER_OVERFETCH`, :data:`HIER_COARSE_DTYPE`) are the intended knobs. M1
reads no ``os.environ`` (kept pure); the eventual integration should surface
them as ``DSV41_INDEXER_HIER_BLOCK`` / ``DSV41_INDEXER_HIER_STRIP`` /
``DSV41_INDEXER_HIER_OVERFETCH`` (and, if useful, gate the whole path on
``DSV41_INDEXER_HIER``), mirroring the ``DSV41_INDEXER_TILE*`` neighbours.
"""

from __future__ import annotations

import mlx.core as mx

NEG_INF = float("-inf")
POS_INF = float("inf")

# --- prototype knobs (see "M2 knobs" above) ---------------------------------
HIER_BLOCK = 8            # columns per coarse block (== candidate_block_size)
HIER_STRIP = 4096         # columns per coarse strip (rounded down to a block multiple)
HIER_OVERFETCH = 16       # candidate blocks beyond k, absorbing bf16 near-ties
HIER_COARSE_DTYPE = mx.bfloat16   # coarse ranking precision; exact pass is fp32

__all__ = [
    "HIER_BLOCK",
    "HIER_STRIP",
    "HIER_OVERFETCH",
    "HIER_COARSE_DTYPE",
    "coarse_block_scores",
    "coarse_block_scores_candidates",
    "coarse_gather_strip",
    "top_blocks",
    "exact_rescore_streaming",
    "candidate_mask_from_block_maxima",
    "hierarchical_topk",
]


# --------------------------------------------------------------------------
# score bodies (structurally identical to indexer._score_body_eager, with a
# dtype parameter so the coarse pass can run bf16 and the exact pass fp32)
# --------------------------------------------------------------------------
def _score_shared_columns(q: mx.array, keys: mx.array, w: mx.array) -> mx.array:
    """Score a *shared* key strip: q [b,n,h,d], keys [b,t,d] -> [b,n,t].

    Same expression order as :func:`indexer._score_body_eager`
    (``einsum -> relu -> *w -> head-sum``), so an fp32 call is elementwise equal
    to the production per-tile score for the same columns.
    """
    s = mx.einsum("bshd,btd->bsht", q, keys)
    s = mx.maximum(s, 0.0) * w[..., None]
    return mx.sum(s, axis=2)


def _score_gathered_columns(q: mx.array, keys: mx.array, w: mx.array) -> mx.array:
    """Score *per-row gathered* keys: q [b,n,h,d], keys [b,n,c,d] -> [b,n,c].

    Used by the exact pass, where each query row has its own candidate columns.
    """
    s = mx.einsum("bshd,bscd->bshc", q, keys)
    s = mx.maximum(s, 0.0) * w[..., None]
    return mx.sum(s, axis=2)


def _gather_rows(index_k: mx.array, cols: mx.array) -> mx.array:
    """index_k [b, nb, d], cols [b, n, c] int32 -> [b, n, c, d].

    A flat per-batch gather (same trick as ``sparse_attention._gather_kv``):
    ``mx.take(index_k, cols, axis=1)`` would *broadcast* the batch axis and
    yield ``[b, b, n, c, d]``, so we index the ``b*nb``-flat rows directly.
    ``cols`` must already be clamped to ``[0, nb)``.
    """
    b, nb, d = index_k.shape
    flat = index_k.reshape(b * nb, d)
    base = (mx.arange(b, dtype=mx.int32) * nb).reshape(b, 1, 1)
    flat_idx = (cols + base).reshape(-1)
    return flat[flat_idx].reshape(*cols.shape, d)


# --------------------------------------------------------------------------
# 1. coarse pass
# --------------------------------------------------------------------------
def coarse_block_scores(q32: mx.array, index_k: mx.array, w32: mx.array,
                        lens: mx.array, block: int = HIER_BLOCK,
                        strip: int = HIER_STRIP,
                        dtype=mx.bfloat16, col_mask: mx.array | None = None) -> mx.array:
    """Per-block maximum score, streamed — never materializes ``[b, n, nb]``.

    Shapes: ``q32`` [b, n, h, d] fp32, ``index_k`` [b, nb, d] (bf16 cache),
    ``w32`` [b, n, h] fp32, ``lens`` [n, 1] int32 (visible columns per query).
    ``col_mask`` [b, n, nb] bool, optional: restrict columns to a candidate set
    (the consumer-layer composition; a masked column is ``-inf`` and can never
    win a block max). Returns ``block_maxima`` [b, n, nb_blocks] fp32,
    ``nb_blocks = ceil(nb / block)``.

    ``nb`` is consumed in ``strip``-wide strips (round down to a multiple of
    ``block`` so block boundaries align across strips). Per strip: score
    ``[b, n, strip]`` in ``dtype``, mask invisible/masked columns to ``-inf``,
    reduce to ``[b, n, strip/block]`` block maxima, write into the buffer,
    ``mx.eval`` the strip (so MLX's lazy graph cannot keep every strip's
    transient alive). The largest score transient is ``[b, n, strip]``; the
    resident buffer is ``[b, n, nb/block]`` — the full row is never allocated.

    ``-inf`` block maxima mean "no visible column in this block" and are
    handled downstream (they rank last and contribute only ``-inf`` scores).
    """
    bsz, n, _h, _d = q32.shape
    nb = index_k.shape[1]
    block = max(1, min(int(block), nb))
    st = max(block, (int(strip) // block) * block)
    nb_blocks = -(-nb // block)

    block_maxima = mx.full((bsz, n, nb_blocks), NEG_INF, dtype=mx.float32)
    qc = q32.astype(dtype)
    wc = w32.astype(dtype)

    for c0 in range(0, nb, st):
        c1 = min(c0 + st, nb)
        t = c1 - c0
        keys = index_k[:, c0:c1].astype(dtype)                 # [b, t, d] view
        s = _score_shared_columns(qc, keys, wc)                # [b, n, t]
        cols = mx.arange(c0, c1, dtype=mx.int32)
        vis = cols[None, :] < lens
        if col_mask is not None:
            vis = vis & col_mask[:, :, c0:c1]
        s = mx.where(vis, s, mx.array(NEG_INF, dtype=dtype))
        mx.eval(s)                                             # bound the peak

        nt = -(-t // block)
        pad = nt * block - t
        if pad:
            s = mx.concatenate(
                [s, mx.full((bsz, n, pad), NEG_INF, dtype=dtype)], axis=-1)
        bv = s.reshape(bsz, n, nt, block).max(axis=-1).astype(mx.float32)
        b0 = c0 // block
        block_maxima[:, :, b0:b0 + nt] = bv
    mx.eval(block_maxima)
    return block_maxima


_BLOCK_ANY_STRIP = 16384   # mask columns reduced per step (bounds the bool copy)
_IDS_CHUNK_ELEMS = 1 << 23  # [rows, NB] elements per candidate-id argpartition


def _block_any(col_mask: mx.array, nb: int, block: int) -> mx.array:
    """``[b, n, nb]`` bool -> ``[b, n, ceil(nb/block)]``: does block j hold ANY
    masked-in column. Reduced in column strips so a non-contiguous ``col_mask``
    view never costs an ``nb``-wide copy (only a ``[b, n, strip]`` one); the
    tail block is reduced over its real ``< nb`` columns only.
    """
    bsz, n = col_mask.shape[0], col_mask.shape[1]
    st = max(block, (_BLOCK_ANY_STRIP // block) * block)
    parts = []
    for c0 in range(0, nb, st):
        c1 = min(c0 + st, nb)
        full = (c1 - c0) // block
        if full:
            parts.append(col_mask[:, :, c0:c0 + full * block]
                         .reshape(bsz, n, full, block).any(axis=-1))
        if c0 + full * block < c1:                       # partial tail block
            parts.append(col_mask[:, :, c0 + full * block:c1]
                         .any(axis=-1, keepdims=True))
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)


def coarse_gather_strip(coarse_strip: int, heads: int, head_dim: int,
                        block: int = HIER_BLOCK) -> int:
    """Column strip for :func:`coarse_block_scores_candidates`, peak-matched.

    The full-width coarse pass's per-strip transient is dominated by the
    ``[b, n, h, t]`` einsum output plus its ``relu * w`` (~``2h`` elements per
    row-column). The restricted pass instead holds a per-row key gather
    ``[b, n, c, d]`` plus the same ``[b, n, h, c]`` pair (~``d + 2h``). Matching
    the two peaks gives ``c = t * 2h / (d + 2h)``, rounded down to a block
    multiple (>= one block). At t=4096, h=32, d=128 that is 1360 columns.
    """
    block = max(1, int(block))
    c = (int(coarse_strip) * 2 * int(heads)) // (int(head_dim) + 2 * int(heads))
    return max(block, (c // block) * block)


def coarse_block_scores_candidates(q32: mx.array, index_k: mx.array, w32: mx.array,
                                   lens: mx.array, col_mask: mx.array,
                                   block: int = HIER_BLOCK,
                                   strip: int = HIER_STRIP,
                                   dtype=mx.bfloat16) -> mx.array:
    """:func:`coarse_block_scores` with ``col_mask``, scoring ONLY candidate blocks.

    Consumer index layers pass ``col_mask = shared.candidates``, which keeps
    ~``candidate_topk_blocks * block`` of ``nb`` columns per row. The
    full-width pass scores all ``nb`` columns and then sets the masked ones to
    ``-inf``; this pass skips every block that contains no masked-in column and
    scores the rest via a per-row key gather (same pattern as
    :func:`exact_rescore_streaming`). ``strip`` is in **columns** of the
    gather (see :func:`coarse_gather_strip`), rounded down to whole blocks.

    Returns ``block_maxima`` [b, n, ceil(nb/block)] fp32 that is
    **elementwise-identical** to ``coarse_block_scores(..., col_mask=col_mask)``
    for ANY bool mask (block-aligned or not):

    * a block with no masked-in column: every column is ``-inf`` in the old
      path, and the block is never written here (stays ``-inf``);
    * any other block: all ``block`` of its columns are scored with the same
      expression in the same dtype, masked with the same
      ``cols < lens & col_mask`` test (out-of-range tail columns ``>= nb`` are
      ``-inf``, exactly the old tail pad), and reduced with the same ``max``
      in ``dtype`` before the fp32 cast.

    Score-expression parity (shared-strip einsum vs per-row gathered einsum)
    is pinned bitwise by ``tests/test_dsv41_consumer_skip.py``.
    """
    bsz, n, _h, _d = q32.shape
    nb = index_k.shape[1]
    if nb == 0:
        return coarse_block_scores(q32, index_k, w32, lens, block=block,
                                   strip=strip, dtype=dtype, col_mask=col_mask)
    block = max(1, min(int(block), nb))      # same clamp as coarse_block_scores
    nb_blocks = -(-nb // block)

    blk = _block_any(col_mask, nb, block)                     # [b, n, NB]
    max_cand = int(blk.sum(axis=-1).max().item())

    # One spare slot (index NB) absorbs padding writes; sliced off at the end,
    # so padded slots can never overwrite a real block's maximum.
    bm = mx.full((bsz, n, nb_blocks + 1), NEG_INF, dtype=mx.float32)
    if max_cand == 0:
        return bm[:, :, :nb_blocks]

    # Candidate block ids per row, padded to max_cand: the max_cand smallest of
    # -blk include every candidate block (count <= max_cand); leftover slots
    # are non-candidates -> redirected to the dummy slot NB. Order within a
    # row is irrelevant (each block's max is written to its own slot). Done in
    # row chunks: argpartition returns a full [.., NB] index array, which at
    # deep offsets (NB ~ 94K, n = 2048) would be a ~0.75 GiB transient.
    rc = max(1, _IDS_CHUNK_ELEMS // max(1, bsz * nb_blocks))
    dummy = mx.array(nb_blocks, dtype=mx.int32)
    parts = []
    for r0 in range(0, n, rc):
        bk = blk[:, r0:r0 + rc]
        part = mx.argpartition(-bk.astype(mx.int32), max_cand - 1,
                               axis=-1)[..., :max_cand].astype(mx.int32)
        part = mx.where(mx.take_along_axis(bk, part, axis=-1), part, dummy)
        mx.eval(part)
        parts.append(part)
    ids = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
    mx.eval(ids)

    qc = q32.astype(dtype)
    wc = w32.astype(dtype)
    neg = mx.array(NEG_INF, dtype=dtype)
    bp = max(1, int(strip) // block)
    lane = mx.arange(block, dtype=mx.int32)
    for s0 in range(0, max_cand, bp):
        s1 = min(s0 + bp, max_cand)
        bs = s1 - s0
        bid = ids[:, :, s0:s1]                                   # [b, n, bs]
        gcols = (bid[..., None] * block + lane).reshape(bsz, n, bs * block)
        safe = mx.minimum(mx.maximum(gcols, 0), nb - 1)
        keys = _gather_rows(index_k, safe).astype(dtype)         # [b, n, c, d]
        s = _score_gathered_columns(qc, keys, wc)                # [b, n, c]
        vis = ((gcols < lens) & (gcols < nb)
               & mx.take_along_axis(col_mask, safe, axis=2))
        s = mx.where(vis, s, neg)
        bv = s.reshape(bsz, n, bs, block).max(axis=-1).astype(mx.float32)
        bm = mx.put_along_axis(bm, bid, bv, axis=-1)
        mx.eval(bm)                                              # bound the peak
    return bm[:, :, :nb_blocks]


# --------------------------------------------------------------------------
# 2. top blocks
# --------------------------------------------------------------------------
def top_blocks(block_maxima: mx.array, k_blocks: int,
               block_size: int = HIER_BLOCK) -> mx.array:
    """Top-``k_blocks`` block indices, ascending — the coarse selection.

    ``block_maxima`` [b, n, nb_blocks] -> indices [b, n, min(k_blocks, nb_blocks)]
    int32, sorted ascending (position order) so the exact pass is deterministic.
    Mirrors the ``argpartition`` style of :func:`indexer.topk_from_row` /
    :func:`indexer.select_candidate_blocks`. ``block_size`` is accepted for
    signature symmetry with the candidate path but does not affect indexing
    (block ``j`` covers columns ``[j*block_size, (j+1)*block_size)``).
    """
    nb_blocks = block_maxima.shape[-1]
    if k_blocks <= 0 or nb_blocks == 0:
        return mx.zeros((*block_maxima.shape[:-1], 0), dtype=mx.int32)
    kb = min(int(k_blocks), nb_blocks)
    idx = mx.argpartition(-block_maxima, kb - 1, axis=-1)[..., :kb].astype(mx.int32)
    return mx.sort(idx, axis=-1)


def candidate_mask_from_block_maxima(block_maxima: mx.array, lens: mx.array,
                                     topk_blocks: int,
                                     block_size: int) -> mx.array:
    """Layer-20 candidate mask, derived from the SAME coarse maxima buffer.

    This is the fused form of :func:`indexer.select_candidate_blocks`: instead
    of re-reading the ``[b, n, nb]`` score row, it consumes the per-block
    maxima the coarse pass already computed (``block_maxima`` [b, n, nb_blocks]
    fp32, unreachable/out-of-block columns already ``-inf``). Returns a bool
    mask ``[b, n, nb_blocks * block_size]`` truncated to ``nb`` — exactly what
    the production candidate mask is.

    Semantics match production: the block holding each query's newest position
    is pinned (its max forced to ``+inf``) before the top-``topk_blocks``
    argpartition, and a block is kept iff it contains a visible column (its
    max ``> -inf``). ``block_maxima`` may be built on a padded block grid
    (``nb_blocks = ceil(nb / block_size)``); the trailing block's own max
    already accounts for out-of-range columns, which the coarse pass masks to
    ``-inf``.
    """
    nb_blocks, block = block_maxima.shape[-1], int(block_size)
    width = nb_blocks * block
    # block j's true column span is [j*block, (j+1)*block); its stored max was
    # reduced over exactly that span (coarse pass), so no re-pad is needed.
    last = (lens - 1) // block                                   # [n, 1]
    pin = mx.arange(nb_blocks)[None, :] == last                  # [n, NB]
    blocks = mx.where(pin[None], POS_INF, block_maxima)

    k = min(int(topk_blocks), nb_blocks)
    top_idx = mx.argpartition(-blocks, k - 1, axis=-1)[..., :k]
    top_val = mx.take_along_axis(blocks, top_idx, axis=-1)
    keep = mx.zeros(blocks.shape, dtype=mx.bool_)
    keep = mx.put_along_axis(keep, top_idx, top_val > NEG_INF, axis=-1)
    return mx.repeat(keep, block, axis=-1)[..., :width]


# --------------------------------------------------------------------------
# 3. exact re-score, streamed
# --------------------------------------------------------------------------
def exact_rescore_streaming(q32: mx.array, index_k: mx.array, w32: mx.array,
                            lens: mx.array, blocks: mx.array,
                            strip: int = HIER_STRIP, k: int = 512,
                            block: int = HIER_BLOCK, col_mask: mx.array | None = None):
    """Exact fp32 top-k over only the candidate blocks' columns, streamed.

    ``blocks`` [b, n, kb] int32 (from :func:`top_blocks`). Processes
    ``bp = max(1, strip // block)`` blocks per strip: build the global column
    indices of those blocks (``bid*block + arange(block)``), gather their keys,
    score fp32, mask invisible/out-of-range columns to ``-inf``, and merge into
    a running top-k over global indices. ``col_mask`` [b, n, nb] bool, optional,
    additionally masks columns (consumer-layer composition). Returns ``(top_v
    [b,n,k], top_i [b,n,k] int32)`` sorted ascending by index — the same
    shape/ordering/index space as :func:`indexer.topk_from_row`. Non-finite
    values mark padding and an unusable index, exactly as production. Largest
    score transient: ``[b, n, bp*block]``; largest key transient:
    ``[b, n, bp*block, d]``.
    """
    bsz, n, _h, _d = q32.shape
    nb = index_k.shape[1]
    kk = min(max(0, int(k)), nb)
    best_v = mx.full((bsz, n, kk), NEG_INF, dtype=mx.float32)
    best_i = mx.zeros((bsz, n, kk), dtype=mx.int32)
    kb = blocks.shape[-1]
    if kb == 0 or kk == 0:
        return best_v, best_i

    block = max(1, int(block))
    bp = max(1, int(strip) // block)
    for s0 in range(0, kb, bp):
        s1 = min(s0 + bp, kb)
        bid = blocks[:, :, s0:s1]                              # [b, n, bs]
        bs = s1 - s0
        gcols = (bid[..., None] * block
                 + mx.arange(block, dtype=mx.int32)).reshape(bsz, n, bs * block)
        safe = mx.minimum(mx.maximum(gcols, 0), nb - 1)
        keys = _gather_rows(index_k, safe).astype(mx.float32)  # [b, n, c, d]
        s = _score_gathered_columns(q32, keys, w32)            # [b, n, c]
        vis = (gcols < lens) & (gcols < nb)
        if col_mask is not None:
            vis = vis & mx.take_along_axis(col_mask, safe, axis=2)
        s = mx.where(vis, s, NEG_INF)

        v = mx.concatenate([best_v, s], axis=-1)
        ii = mx.concatenate([best_i, gcols], axis=-1)
        part = mx.argpartition(-v, kk - 1, axis=-1)[..., :kk]
        best_v = mx.take_along_axis(v, part, axis=-1)
        best_i = mx.take_along_axis(ii, part, axis=-1)
        # ONE blocking eval per strip: the fused graph includes the gather,
        # score, mask, merge and the next-iteration's read of best_v/best_i,
        # which is exactly the set the old mid-strip `mx.eval(s)` bounded
        # (peak = max(score transient, merged [b,n,kk*2])). Halves the strip
        # sync count on the dominant per-chunk host-round-trip site.
        mx.eval(best_v, best_i)

    order = mx.argsort(best_i, axis=-1)                        # position order
    best_i = mx.take_along_axis(best_i, order, axis=-1)
    best_v = mx.take_along_axis(best_v, order, axis=-1)
    return best_v, best_i


# --------------------------------------------------------------------------
# 4. composition
# --------------------------------------------------------------------------
def hierarchical_topk(q32: mx.array, index_k: mx.array, w32: mx.array,
                      lens: mx.array, k: int = 512, block: int = HIER_BLOCK,
                      strip: int = HIER_STRIP, overfetch: int = HIER_OVERFETCH,
                      coarse_dtype=mx.bfloat16):
    """Coarse block-maxima -> top-``k+overfetch`` blocks -> streamed exact top-k.

    The one-call composition. Returns ``(top_v [b,n,k], top_i [b,n,k] int32)``
    in the production index space (sorted ascending; see the module docstring).
    ``k`` is clamped to ``nb``. ``overfetch`` is the candidate-block margin that
    absorbs bf16 coarse near-ties; ``0`` gives the pure ``k``-block prune.
    """
    nb = index_k.shape[1]
    nb_blocks = -(-nb // max(1, int(block)))
    k_blocks = min(nb_blocks, int(k) + int(overfetch))
    bm = coarse_block_scores(q32, index_k, w32, lens, block=block, strip=strip,
                             dtype=coarse_dtype)
    blocks = top_blocks(bm, k_blocks, block_size=block)
    return exact_rescore_streaming(q32, index_k, w32, lens, blocks, strip=strip,
                                   k=k, block=block)


# --------------------------------------------------------------------------
# 5. production integration entry point (M2)
# --------------------------------------------------------------------------
def hierarchical_topk_prod(q32: mx.array, index_k: mx.array, w32: mx.array,
                           lens: mx.array, k: int, block: int, coarse_strip: int,
                           exact_strip: int, overfetch: int,
                           coarse_dtype=mx.bfloat16, *,
                           cand_mask: mx.array | None = None,
                           cand_src: tuple[int, int] | None = None,
                           consumer_skip: bool = True):
    """The single coarse sweep, then top-blocks, exact re-score, and (if this
    layer is the candidate source) candidate-mask publishing — fused.

    This is the production entry point for the M2 hierarchical path
    (``DSV41_INDEXER_HIER=1``). It mirrors :func:`indexer._tiled_scores_buffer`'s
    contract exactly:

    * ``q32`` [b, n, h, d] fp32, ``index_k`` [b, nb, d], ``w32`` [b, n, h] fp32,
      ``lens`` [n, 1];
    * ``cand_mask`` [b, n, nb] bool or None restricts the columns (consumer
      layers 24..36 — the exact re-score runs *within* the candidate set);
      with ``consumer_skip`` (default) the coarse pass scores only the blocks
      holding a candidate column (:func:`coarse_block_scores_candidates`),
      elementwise-identical to the full-width score-then-mask sweep;
    * ``cand_src=(candidate_topk_blocks, candidate_block_size)`` makes this
      layer the candidate source (layer 20): the *same* coarse maxima buffer
      that feeds ``top_blocks`` also publishes the candidate mask, so there is
      no second full-row pass (the fusion the M1 design called for).

    The two strips are separate knobs (M1 design §9): ``coarse_strip`` is the
    coarse score transient width (``[b, n, coarse_strip]`` bf16), while
    ``exact_strip`` sizes the exact pass and must be derived from the *key*
    budget ``c <= M / (b*n*(6d+4))`` — see :func:`indexer.hier_strip_for_budget`.
    Both are interpreted as **columns** and rounded down to a block multiple
    internally.

    Returns ``(top_v [b,n,k], top_i [b,n,k] int32, block_mask [b,n,nb] bool or
    None)`` — the production return shape. ``top_i`` is raw compressed-column
    space (before ``+offset`` / ``-1`` masking by :class:`indexer.Indexer`).
    """
    bsz, n, _h, _d = q32.shape
    nb = index_k.shape[1]
    kk = min(max(0, int(k)), nb)
    block = max(1, int(block))
    nb_blocks = -(-nb // block)

    # ONE coarse sweep (bf16, stripped) -> [b, n, nb_blocks] fp32 maxima.
    # Never allocates [b, n, nb]; the resident buffer is nb/block-wide.
    if cand_mask is not None and consumer_skip:
        # Consumer layer: only candidate blocks can have a finite max, so score
        # only those (per-row gather) instead of all nb columns then masking.
        # Bit-identical maxima -> identical top_blocks -> identical top-k.
        bm = coarse_block_scores_candidates(
            q32, index_k, w32, lens, cand_mask, block=block,
            strip=coarse_gather_strip(coarse_strip, _h, _d, block),
            dtype=coarse_dtype)
    else:
        bm = coarse_block_scores(q32, index_k, w32, lens, block=block,
                                 strip=coarse_strip, dtype=coarse_dtype,
                                 col_mask=cand_mask)

    # Candidate publishing (layer 20) is derived from the SAME maxima buffer:
    # no score row, no second pass. (Fused here so the coarse sweep runs once.)
    # The mask is block-aligned to `block` (the coarse block), so the caller
    # must pass block == candidate_block_size when this layer is the source.
    mask = None
    if cand_src is not None:
        mask = candidate_mask_from_block_maxima(bm, lens, cand_src[0], block)
        mask = mask[:, :, :nb]          # production mask width is exactly nb

    # Exact top-k: top-(k+overfetch) blocks by coarse max, then a streamed fp32
    # re-score of only those blocks' columns. For a consumer layer the coarse
    # maxima already excluded non-candidate columns (col_mask above), so this
    # composes within shared.candidates — the two-level semantics are preserved
    # (block-prune, then exact top-k), just never materializing the row.
    k_blocks = min(nb_blocks, kk + int(overfetch))
    blocks = top_blocks(bm, k_blocks, block_size=block)
    top_v, top_i = exact_rescore_streaming(
        q32, index_k, w32, lens, blocks, strip=exact_strip, k=k, block=block,
        col_mask=cand_mask)
    return top_v, top_i, mask
