# Vendored from PipeNetwork/deepseek-v41-mlx (Apache-2.0), upstream a9567ae plus the
# node-local MTP/DSpark port edits (exo docs phase 3). See README.md here.
"""The sparse-attention indexer — a second attention deciding what the first reads.

The eight ``index_source_layers`` own one. Only the four that are *also*
``kv_source_layers`` (2, 8, 14, 20) own index **keys** (``wk``/``k_norm`` + a key
cache): keys are derived from the compressor's pre-RoPE latent. Index sources
that are not kv sources (24, 28, 32, 36) score with their own ``wq_b`` /
``weights_proj`` against the *most recent owner's* key cache (layer 20's).

Two-level selection: layer 20 is the candidate source — it scores all latents,
keeps the ``candidate_topk_blocks`` best blocks of ``candidate_block_size`` (the
block holding the newest position is pinned in), and publishes the boolean mask.
Sources after it (24..36) mask their own scores with those candidates before the
final per-query top-``index_topk``.

Queries and keys are FP4 fake-quantized (blocks of 32, power-of-two scales), no
Hadamard rotation (a V4 feature V4.1 dropped). Scores are ReLU'd, then collapsed
over heads by ``weights_proj(x) * head_dim**-0.5 * n_heads**-0.5``.

Reference artifact worth knowing (documented in docs/upstream-notes.md): the
reference reads keys through a process-global pointer that the *last* owner set.
During decode, a ratio-2 owner whose group is incomplete therefore scores
against layer 20's keys. This port always reads the owner's own cache.

Long-context tiling (local change, exo phase 19 prefill fix; reworked round 2)
-----------------------------------------------------------------------------
The pre-head-sum score tensor is ``[b, n, index_n_heads, nb]`` fp32: at an n=512
chunk and nb=16384 that is 1.07 GB for one index layer, and four index layers sit
in one lazy prefill graph — the allocation that drives the Metal GPU watchdog.
At ``nb >= DSV41_INDEXER_TILE_MIN_NB`` (default 8192) the columns are scored in
``DSV41_INDEXER_TILE``-wide chunks, so only one ``[b, n, heads, tile]`` transient
exists at a time (33 MB at tile=512, n=512), and the tile shrinks further to
honour ``DSV41_INDEXER_TILE_MB`` (default 128 MB).

Each tile's collapse is fenced (``mx.eval``) and kept as a 1 MB ``[b, n, t]``
row; the layer's top-k is then ONE exact ``argpartition`` over the whole row
(:func:`_tiled_scores_buffer`) — at nb=16384, n=512 that is ~2 ms where the
previous per-tile running merge spent ~7 ms re-partitioning 512-row windows, and
the candidate block selection (layer 20) is one global pass over per-tile block
maxima instead of a per-tile merge (measured 41.8 -> ~18 ms for the whole layer
call at nb=16384). ``DSV41_INDEXER_TILED_IMPL=merge`` restores the previous
per-tile running merge (:func:`_tiled_scores_merge`), kept for A/B and rollback.

Exactness. Score elements come from the same expressions on the same data, and
the per-tile score is bitwise equal to the untiled path at every production
width (``pB_indexer_test.py parity``), including under the per-shape compiled
score body (``DSV41_INDEXER_COMPILE``, default on, per-shape key so a width
change cannot silently mismatch). With one global exact top-k over those values,
the selected *value* multiset equals the untiled top-k; selected *indices* can
differ only where equal scores straddle the k-th boundary (argpartition's tie
pick is implementation-defined, the same class round 1 documented). Measured 0
differing rows on synthetic tensors (nb 4096..20000, tiles 256..1024) and on
real activations (49152 rows, ``wsB-ab``).

``DSV41_INDEXER_TILE=0`` restores the untiled path for any nb;
``DSV41_INDEXER_TILE_MIN_NB=0`` runs tiled at any nb (used by parity tests).
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs
from .fakequant import fake_quant_fp4_ue8m0
from .layers import RMSNorm, cos_sin_at, rope_tail

NEG_INF = float("-inf")
POS_INF = float("inf")

# --- tiled score path settings ----------------------------------------------
# DSV41_INDEXER_TILE_MIN_NB default 0 since 2026-10-02: always take the tiled
# path (bit-exact, max|dlogit|=0; -1.7 GB prefill peak). Set 8192 for the old gate.
_TILE = int(os.environ.get("DSV41_INDEXER_TILE", "512"))
_TILE_MIN_NB = int(os.environ.get("DSV41_INDEXER_TILE_MIN_NB", "0"))
_TILE_BUDGET = int(float(os.environ.get("DSV41_INDEXER_TILE_MB", "128")) * (1 << 20))
# Test hook: run the tiled path for every nb that exceeds one tile, ignoring the
# size gate entirely (parity / NLL gates force the new path at short context).
_TILE_FORCE = os.environ.get("DSV41_INDEXER_TILE_FORCE", "0") == "1"
# Selection structure: "buffer" (default) = one global exact top-k over the
# per-tile score row; "merge" = the previous per-tile running merge (A/B only).
_TILED_IMPL = os.environ.get("DSV41_INDEXER_TILED_IMPL", "buffer")
# Score body under mx.compile, keyed per (b, s, h, d, tile-width) so a width
# change can never silently reuse a mismatched pipeline. The body is a single
# einsum + relu*scale + head-sum; compiled it is ~15% faster per tile and
# bitwise equal to the eager expression (verified at every width the parity
# harness exercises). DSV41_INDEXER_COMPILE=0 falls back to the eager expression.
_INDEXER_COMPILE = os.environ.get("DSV41_INDEXER_COMPILE", "1") == "1"
_COMPILED_BODIES: dict = {}


def tile_width(bsz: int, n: int, n_heads: int, nb: int, align: int) -> int:
    """Column-tile width for a score pass, or ``nb`` for the untiled path.

    ``align`` keeps tiles block-aligned for the candidate path's block merge.
    This is the raw width; use :func:`tiled` for the routing decision.
    """
    if _TILE <= 0:
        return nb
    cap = max(align, _TILE_BUDGET // max(1, bsz * n * n_heads * 4))
    t = max(align, (min(_TILE, cap) // align) * align)
    return min(nb, t)


def tiled(bsz: int, n: int, n_heads: int, nb: int, align: int = 1) -> int:
    """Tile width if this shape takes the tiled path, else ``nb`` (untiled).

    Tiling engages only when both hold:

    * ``nb >= DSV41_INDEXER_TILE_MIN_NB`` (default 8192) — below it the
      argpartition path is kept exactly as it was; and
    * the untiled ``[b, n, heads, nb]`` fp32 transient would exceed
      ``DSV41_INDEXER_TILE_MB`` (default 128 MB) — so *decode* (n=1) stays on
      the fast path even at 100K context, where the transient is only a few MB
      and a tile loop would add ``nb/tile`` graph synchronizations per step.
    """
    tile = tile_width(bsz, n, n_heads, nb, align)
    if nb < _TILE_MIN_NB or tile >= nb:
        return nb
    if not _TILE_FORCE and bsz * n * n_heads * nb * 4 <= _TILE_BUDGET:
        return nb
    return tile


def merge_topk(best_v: mx.array, best_i: mx.array, tile_v: mx.array,
               tile_i: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Exact running top-k: the k largest values of (running best ∪ tile).

    ``best_v`` [..., k], ``tile_v`` [..., t] -> ([..., k], [..., k]). The values
    returned are exactly the k largest of everything merged so far, and the
    indices are the distinct columns they came from (``-inf``-valued slots carry
    an unusable padding index; callers filter on the value).

    Kept for the ``DSV41_INDEXER_TILED_IMPL=merge`` rollback path and the parity
    harness; the default buffer path does not use it.
    """
    v = mx.concatenate([best_v, tile_v], axis=-1)
    i = mx.concatenate([best_i, tile_i], axis=-1)
    part = mx.argpartition(-v, k - 1, axis=-1)[..., :k]
    return mx.take_along_axis(v, part, axis=-1), mx.take_along_axis(i, part, axis=-1)


def topk_from_row(s: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Exact top-k of a full ``[..., nb]`` score row, in position order.

    Returns ``(values [..., k], indices [..., k] int32)``; slots beyond the
    finite columns hold ``-inf`` and a caller-checkable index.
    """
    i = mx.argpartition(-s, k - 1, axis=-1)[..., :k].astype(mx.int32)
    i = mx.sort(i, axis=-1)                                  # position order
    v = mx.take_along_axis(s, i, axis=-1)
    return v, i


def _score_body(bsz: int, n: int, n_heads: int, head_dim: int, tile: int):
    """The per-tile score body: einsum(q, k) -> relu * w -> head sum.

    One entry per (b, s, h, d, tile-width) in a process-local cache, wrapping
    ``mx.compile`` of the eager expression. Compiled per tile width on purpose:
    a **shapeless** ``mx.compile`` of this einsum mis-tracks the output's final
    axis as soon as a second tile width arrives (``[reshape] Cannot reshape
    array of size ...`` — measured on this MLX build, widths 512/1024/128/2048),
    so the default path must not use it; a non-shapeless compile handles a new
    width by recompiling internally (MLX keys its own cache per input shape, so
    the ragged final tile costs one extra pipeline per width, bounded).
    ``DSV41_INDEXER_COMPILE=0`` returns the eager expression, bitwise identical.
    """
    if not _INDEXER_COMPILE:
        return _score_body_eager
    key = (bsz, n, n_heads, head_dim, tile)
    fn = _COMPILED_BODIES.get(key)
    if fn is None:
        fn = mx.compile(_score_body_eager)
        _COMPILED_BODIES[key] = fn
    return fn


def _score_body_eager(q32: mx.array, kt: mx.array, w32: mx.array) -> mx.array:
    """Score one tile: ``[b, n, h, t] -> relu -> * w -> head-sum -> [b, n, t]``.

    Elementwise-identical to the untiled path's expression (same ops, same order,
    same fp32 operands), so per-tile scores stay bitwise equal to the untiled
    score matrix (checked by the parity harness at every width).
    """
    s = mx.einsum("bshd,btd->bsht", q32, kt)
    s = mx.maximum(s, 0.0) * w32[..., None]
    return mx.sum(s, axis=2)


def _tiled_scores_buffer(q32: mx.array, index_k: mx.array, w32: mx.array,
                         lens: mx.array, nb: int, k: int, tile: int, *,
                         cand_mask=None, cand_src=None):
    """One pass over the nb columns, buffering the per-tile score row.

    ``q32`` [b, n, h, d] fp32, ``index_k`` [b, nb, d], ``w32`` [b, n, h] fp32,
    ``lens`` [n, 1]. ``cand_mask`` [b, n, nb] bool restricts columns (consumer
    layers 24..36); ``cand_src=(topk_blocks, block_size)`` makes this layer the
    candidate source (layer 20).

    Per tile: score ([b, n, h, tile] transient, fenced), mask visibility, append
    the ``[b, n, tile]`` collapse to a ``[b, n, nb]`` row (the untiled
    ``[b, n, h, nb]`` 1.07 GB transient becomes a 335 MB fp32 row at nb=16384,
    n=512, of which only the masked 1.07 GB-worth of head reductions are ever
    live one tile at a time). The top-k and the candidate block selection are
    then each ONE exact global pass over that row / over the per-tile block
    maxima.

    Returns ``(top_v [b,n,k], top_i [b,n,k], block_mask [b,n,nb] bool or None)``.
    ``top_i`` slots whose ``top_v`` is ``-inf`` are padding: drop them.
    """
    bsz, n = q32.shape[0], q32.shape[1]
    cand = cand_src is not None
    topk_blocks, block_size = cand_src if cand else (0, 1)
    body = _score_body(bsz, n, q32.shape[2], q32.shape[3], tile)

    # The whole buffer is written by the loop before anything reads it, so an
    # uninitialized allocation is safe here and skips a full zero-fill of
    # ``b*n*nb`` fp32 (335 MB at nb=16384, n=512 — measured ~0.8 ms).
    row = mx.empty((bsz, n, nb), dtype=mx.float32)
    for c0 in range(0, nb, tile):
        c1 = min(c0 + tile, nb)
        s = body(q32, index_k[:, c0:c1].astype(mx.float32), w32)
        cols = mx.arange(c0, c1, dtype=mx.int32)
        s = mx.where(cols[None, :] < lens, s, NEG_INF)
        # Materialize this tile's collapse BEFORE building the next one: MLX's
        # lazy evaluation would otherwise keep every tile's [b,n,heads,t]
        # pre-collapse transient alive until the end of the call, so the peak
        # would equal the untiled path and the whole exercise be pointless.
        # (Same finding, same fix as deepseek_v4.py _indexer_score_tiled.)
        mx.eval(s)
        row[:, :, c0:c1] = s
    # The candidate selection is ONE exact global pass over the completed row —
    # the same two-level block max / top-k the untiled path runs, just fed from
    # the buffered row instead of the full [b, n, heads, nb] tensor. This is
    # what took layer 20's call from ~42 ms (per-tile block merge) to ~20 ms.
    if cand:
        mask = select_candidate_blocks(row, lens, topk_blocks, block_size)
    else:
        mask = None
    if cand_mask is not None:
        row = mx.where(cand_mask, row, NEG_INF)
    v, i = topk_from_row(row, k)
    return v, i, mask


def select_candidate_blocks(scores: mx.array, lens: mx.array, topk_blocks: int,
                            block_size: int) -> mx.array:
    """Level one of the two-level top-k. scores [b, n, nb] with unreachable
    positions already at -inf; lens [n, 1] (positions visible per query).
    Returns a bool mask shaped like scores."""
    width = scores.shape[-1]
    pad = (-width) % block_size
    if pad:
        scores_p = mx.concatenate(
            [scores, mx.full((*scores.shape[:-1], pad), NEG_INF, dtype=scores.dtype)], axis=-1)
    else:
        scores_p = scores
    blocks = scores_p.reshape(*scores.shape[:-1], -1, block_size).max(axis=-1)  # [b, n, NB]
    nb_blocks = blocks.shape[-1]

    # pin the block holding each query's newest position: it is only partly
    # filled and could otherwise be outscored by an older, full block
    last = (lens - 1) // block_size                                # [n, 1]
    pin = mx.arange(nb_blocks)[None, :] == last                    # [n, NB]
    blocks = mx.where(pin[None], POS_INF, blocks)

    k = min(topk_blocks, nb_blocks)
    top_idx = mx.argpartition(-blocks, k - 1, axis=-1)[..., :k]
    top_val = mx.take_along_axis(blocks, top_idx, axis=-1)
    keep = mx.zeros(blocks.shape, dtype=mx.bool_)
    keep = mx.put_along_axis(keep, top_idx, top_val > NEG_INF, axis=-1)
    return mx.repeat(keep, block_size, axis=-1)[..., :width]


def _tiled_scores_merge(q32: mx.array, index_k: mx.array, w32: mx.array,
                        lens: mx.array, nb: int, k: int, tile: int, *,
                        cand_mask=None, cand_src=None):
    """Round-1 structure: per-tile running merge for both top-k and blocks.

    Kept verbatim for ``DSV41_INDEXER_TILED_IMPL=merge`` (A/B and rollback).
    ``q32`` [b, n, h, d] fp32, ``index_k`` [b, nb, d], ``w32`` [b, n, h] fp32,
    ``lens`` [n, 1]. ``cand_mask`` [b, n, nb] bool restricts columns (consumer
    layers 24..36); ``cand_src=(topk_blocks, block_size)`` makes this layer the
    candidate source (layer 20).

    Returns ``(top_v [b,n,k], top_i [b,n,k], block_mask [b,n,nb] bool or None)``.
    ``top_i`` entries whose ``top_v`` is -inf are padding: drop them.
    ``block_mask`` is ``None`` unless this layer is the candidate source.
    """
    bsz, n = q32.shape[0], q32.shape[1]
    best_v = mx.full((bsz, n, k), NEG_INF, dtype=mx.float32)
    best_i = mx.zeros((bsz, n, k), dtype=mx.int32)

    cand = cand_src is not None
    topk_blocks, block_size = cand_src if cand else (0, 1)
    nblocks = -(-nb // block_size) if cand else 0
    kb = min(topk_blocks, nblocks) if cand else 0
    # padding slots hold index `nblocks` (out of range) and write to a spare
    # bit, so a -inf-valued block can never be kept by a padding write
    keep_all = mx.zeros((bsz, n, nblocks + 1), dtype=mx.bool_)
    blk_v = mx.full((bsz, n, kb), NEG_INF, dtype=mx.float32)
    blk_i = mx.full((bsz, n, kb), nblocks, dtype=mx.int32)
    last = (lens - 1) // block_size if cand else None             # [n, 1]

    for c0 in range(0, nb, tile):
        c1 = min(c0 + tile, nb)
        t = c1 - c0
        s = mx.einsum("bshd,btd->bsht", q32, index_k[:, c0:c1].astype(mx.float32))
        s = mx.maximum(s, 0.0) * w32[..., None]
        s = mx.sum(s, axis=2)                                    # [b, n, t] fp32

        # visibility: group j is visible to query i once the query passed its
        # last token (the same per-column test the untiled path applies)
        cols = mx.arange(c0, c1, dtype=mx.int32)
        s = mx.where(cols[None, :] < lens, s, NEG_INF)
        if cand_mask is not None:
            s = mx.where(cand_mask[:, :, c0:c1], s, NEG_INF)
        # Materialize this tile's collapse BEFORE building the next one: MLX's
        # lazy evaluation would otherwise keep every tile's [b,n,heads,t]
        # pre-collapse transient alive until the end of the call, so the peak
        # would equal the untiled path and the whole exercise be pointless.
        # (Same finding, same fix as deepseek_v4.py _indexer_score_tiled.)
        mx.eval(s)

        best_v, best_i = merge_topk(
            best_v, best_i, s,
            mx.broadcast_to(cols[None, None, :], (bsz, n, t)), k)
        # and materialize the running best before the next tile: without this the
        # per-tile concat/argpartition outputs of every tile stay pending (their
        # inputs are pinned alive by the graph), which costs the win again
        # (measured at nb=16384, n=512, tile=512: 368 MB -> 48 MB peak).
        mx.eval(best_v, best_i)

        if cand and kb > 0:
            b0 = c0 // block_size
            nt = -(-t // block_size)                             # blocks in this tile
            pad = nt * block_size - t
            sp = s if not pad else mx.concatenate(
                [s, mx.full((bsz, n, pad), NEG_INF, dtype=s.dtype)], axis=-1)
            bv = sp.reshape(bsz, n, nt, block_size).max(axis=-1)  # [b, n, nt]
            bidx = mx.arange(b0, b0 + nt, dtype=mx.int32)
            bv = mx.where(bidx[None, :] == last, POS_INF, bv)     # newest block pinned
            blk_v, blk_i = merge_topk(
                blk_v, blk_i, bv,
                mx.broadcast_to(bidx[None, None, :], (bsz, n, nt)), kb)
            mx.eval(blk_v, blk_i)

    mask = None
    if cand:
        keep_all = mx.put_along_axis(keep_all, blk_i, blk_v > NEG_INF, axis=-1)
        mask = mx.repeat(keep_all[..., :nblocks], block_size, axis=-1)[..., :nb]
    return best_v, best_i, mask


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.owns_k = layer_id in args.kv_source_layers
        self.ratio = args.compress_ratio(layer_id)
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.softmax_scale = self.head_dim ** -0.5

        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(args.dim, self.n_heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(args.head_dim, self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.head_dim, args.norm_eps)

    def publish_keys(self, latents: mx.array, start_pos: int, freqvec: mx.array, cache):
        """Turn this chunk's pre-RoPE latents into index keys and cache them.

        Must run before Attention overwrites the latents with their RoPE'd,
        quantized form. ``latents`` [b, g, head_dim] for groups g0.., where
        g0 = start_pos // ratio; a latent's rope position is its group's first
        token, g*ratio. ``freqvec`` is the per-layer YaRN frequency vector:
        cos/sin are computed for exactly the non-contiguous group positions
        (no dense cos/sin table is threaded through).
        """
        rd = self.rope_head_dim
        g0 = start_pos // self.ratio
        g = latents.shape[1]
        pos = (g0 + mx.arange(g)) * self.ratio
        k = self.k_norm(self.wk(latents))
        k = rope_tail(k, rd, *cos_sin_at(freqvec, pos))
        k = fake_quant_fp4_ue8m0(k, 32)
        cache.index_k[:k.shape[0], g0:g0 + g] = k

    def __call__(self, x: mx.array, qr: mx.array, start_pos: int, offset: int,
                 freqvec: mx.array, index_k: mx.array, shared) -> mx.array:
        """Score and pick top-k compressed positions for each query.

        x [b, n, dim] (post-attn-norm), qr [b, n, q_lora_rank],
        index_k [b, nb, head_dim] — the owner's key cache, already sliced to the
        nb = (start_pos+n)//ratio complete groups. Returns [b, n, k] int32
        indices into the concatenated window+compressed KV, -1 = masked.
        """
        bsz, n, _ = x.shape
        ratio, rd = self.ratio, self.rope_head_dim
        nb = index_k.shape[1]

        q = self.wq_b(qr).reshape(bsz, n, self.n_heads, self.head_dim)
        # Query rows: exactly this forward's n positions, computed on demand.
        q = rope_tail(q, rd, *cos_sin_at(freqvec, mx.arange(start_pos, start_pos + n)))
        q = fake_quant_fp4_ue8m0(q, 32)

        w = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

        # visibility: group j is visible to query i once the query passed its last token
        lens = ((start_pos + mx.arange(n) + 1) // ratio)[:, None]      # [n, 1]

        align = self.candidate_block_size if self.is_candidate_source else 1
        tile = tiled(bsz, n, self.n_heads, nb, align)

        if tile < nb:
            # ---- tiled path: one [b, n, heads, tile] score transient at a time
            k = min(self.index_topk, nb)
            mask = shared.candidates if (self.uses_candidates
                                         and shared.candidates is not None) else None
            impl = _tiled_scores_merge if _TILED_IMPL == "merge" else _tiled_scores_buffer
            v, i, blk = impl(
                q.astype(mx.float32), index_k, w.astype(mx.float32), lens, nb, k, tile,
                cand_mask=mask,
                cand_src=((self.candidate_topk_blocks, self.candidate_block_size)
                          if self.is_candidate_source else None))
            if self.is_candidate_source:
                shared.candidates = blk
            if _TILED_IMPL == "merge":
                # round-1 result order: value-descending with finite slots first
                order = mx.argsort(i, axis=-1)
                i = mx.take_along_axis(i, order, axis=-1)
                v = mx.take_along_axis(v, order, axis=-1)
            valid = mx.isfinite(v) & (i < lens.astype(mx.int32)[None])
            return mx.where(valid, i + offset, mx.array(-1, mx.int32))

        # ---- untiled reference path (unchanged) ----
        scores = mx.einsum("bshd,btd->bsht", q.astype(mx.float32),
                           index_k.astype(mx.float32))
        scores = mx.maximum(scores, 0.0) * w[..., None].astype(mx.float32)
        scores = mx.sum(scores, axis=2)                          # [b, n, nb]

        vis = mx.arange(nb)[None, :] < lens                            # [n, nb]
        scores = mx.where(vis[None], scores, NEG_INF)

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                scores, lens, self.candidate_topk_blocks, self.candidate_block_size)
        elif self.uses_candidates and shared.candidates is not None:
            scores = mx.where(shared.candidates, scores, NEG_INF)

        k = min(self.index_topk, nb)
        idx = mx.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(mx.int32)
        idx = mx.sort(idx, axis=-1)                              # position order
        visible = idx < lens.astype(mx.int32)[None]
        return mx.where(visible, idx + offset, mx.array(-1, mx.int32))
