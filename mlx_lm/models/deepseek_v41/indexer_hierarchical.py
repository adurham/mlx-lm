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
    "top_blocks",
    "exact_rescore_streaming",
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
                        dtype=mx.bfloat16) -> mx.array:
    """Per-block maximum score, streamed — never materializes ``[b, n, nb]``.

    Shapes: ``q32`` [b, n, h, d] fp32, ``index_k`` [b, nb, d] (bf16 cache),
    ``w32`` [b, n, h] fp32, ``lens`` [n, 1] int32 (visible columns per query).
    Returns ``block_maxima`` [b, n, nb_blocks] fp32, ``nb_blocks =
    ceil(nb / block)``.

    ``nb`` is consumed in ``strip``-wide strips (round down to a multiple of
    ``block`` so block boundaries align across strips). Per strip: score
    ``[b, n, strip]`` in ``dtype``, mask invisible columns to ``-inf``, reduce
    to ``[b, n, strip/block]`` block maxima, write into the buffer, ``mx.eval``
    the strip (so MLX's lazy graph cannot keep every strip's transient alive).
    The largest score transient is ``[b, n, strip]``; the resident buffer is
    ``[b, n, nb/block]`` — the full row is never allocated.

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
        s = mx.where(cols[None, :] < lens, s, mx.array(NEG_INF, dtype=dtype))
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


# --------------------------------------------------------------------------
# 3. exact re-score, streamed
# --------------------------------------------------------------------------
def exact_rescore_streaming(q32: mx.array, index_k: mx.array, w32: mx.array,
                            lens: mx.array, blocks: mx.array,
                            strip: int = HIER_STRIP, k: int = 512,
                            block: int = HIER_BLOCK):
    """Exact fp32 top-k over only the candidate blocks' columns, streamed.

    ``blocks`` [b, n, kb] int32 (from :func:`top_blocks`). Processes
    ``bp = max(1, strip // block)`` blocks per strip: build the global column
    indices of those blocks (``bid*block + arange(block)``), gather their keys,
    score fp32, mask invisible/out-of-range columns to ``-inf``, and merge into
    a running top-k over global indices. Returns ``(top_v [b,n,k], top_i
    [b,n,k] int32)`` sorted ascending by index — the same shape/ordering/index
    space as :func:`indexer.topk_from_row`. Non-finite values mark padding and
    an unusable index, exactly as production. Largest score transient:
    ``[b, n, bp*block]``; largest key transient: ``[b, n, bp*block, d]``.
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
        s = mx.where(vis, s, NEG_INF)
        mx.eval(s)

        v = mx.concatenate([best_v, s], axis=-1)
        ii = mx.concatenate([best_i, gcols], axis=-1)
        part = mx.argpartition(-v, kk - 1, axis=-1)[..., :kk]
        best_v = mx.take_along_axis(v, part, axis=-1)
        best_i = mx.take_along_axis(ii, part, axis=-1)
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
