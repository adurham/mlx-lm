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

Long-context tiling (local change, exo phase 19 prefill fix)
------------------------------------------------------------
The pre-head-sum score tensor is ``[b, n, index_n_heads, nb]`` fp32: at an n=512
chunk and nb=16384 that is 1.07 GB for one index layer, and four index layers sit
in one lazy prefill graph — the allocation that drives the Metal GPU watchdog.
At ``nb >= DSV41_INDEXER_TILE_MIN_NB`` (default 8192) the columns are scored in
``DSV41_INDEXER_TILE``-wide chunks, so only one ``[b, n, heads, tile]`` transient
exists at a time (33 MB at tile=512, n=512), and the tile shrinks further to
honour ``DSV41_INDEXER_TILE_MB`` (default 128 MB). The top-k is an exact running
merge (``merge_topk``): each tile is unioned with the running best-k and
re-partitioned, and the layer-20 candidate block selection is merged the same
way — tiles are block-aligned so blocks never straddle a tile.

Exactness. Score elements come from the same expressions on the same data, so
they are bitwise equal to the untiled path (measured — ``pB_indexer_test.py
parity``). The running merge returns exactly the k largest values of the
row-union, so the selected *value* multiset equals the untiled top-k; selected
*indices* can differ only where equal scores straddle the k-th boundary. One
deliberate difference: the tiled path never reports a ``-inf`` column (it filters
on the merged values being finite), while the untiled path can return one before
its ``idx < lens`` test drops it. Both agree on the number of valid indices, and
that case needs fewer than ``index_topk`` finite columns — i.e. nb below
``index_topk + n_chunk/ratio``, never reachable in the tiled regime (nb >= 8192).

``DSV41_INDEXER_TILE=0`` restores the untiled path for any nb;
``DSV41_INDEXER_TILE_MIN_NB=0`` runs tiled at any nb (used by parity tests).
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs
from .fakequant import fake_quant_fp4_ue8m0
from .layers import RMSNorm, rope_tail

NEG_INF = float("-inf")
POS_INF = float("inf")

# --- tiled score path settings ----------------------------------------------
_TILE = int(os.environ.get("DSV41_INDEXER_TILE", "512"))
_TILE_MIN_NB = int(os.environ.get("DSV41_INDEXER_TILE_MIN_NB", "8192"))
_TILE_BUDGET = int(float(os.environ.get("DSV41_INDEXER_TILE_MB", "128")) * (1 << 20))
# Test hook: run the tiled path for every nb that exceeds one tile, ignoring the
# size gate entirely (parity / NLL gates force the new path at short context).
_TILE_FORCE = os.environ.get("DSV41_INDEXER_TILE_FORCE", "0") == "1"


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
    """
    v = mx.concatenate([best_v, tile_v], axis=-1)
    i = mx.concatenate([best_i, tile_i], axis=-1)
    part = mx.argpartition(-v, k - 1, axis=-1)[..., :k]
    return mx.take_along_axis(v, part, axis=-1), mx.take_along_axis(i, part, axis=-1)


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


def _tiled_scores(q32: mx.array, index_k: mx.array, w32: mx.array, lens: mx.array,
                  nb: int, k: int, tile: int, *, cand_mask=None, cand_src=None):
    """One pass over the nb columns in ``tile``-wide chunks.

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

    def publish_keys(self, latents: mx.array, start_pos: int, cos, sin, cache):
        """Turn this chunk's pre-RoPE latents into index keys and cache them.

        Must run before Attention overwrites the latents with their RoPE'd,
        quantized form. ``latents`` [b, g, head_dim] for groups g0.., where
        g0 = start_pos // ratio; a latent's rope position is its group's first
        token, g*ratio.
        """
        rd = self.rope_head_dim
        g0 = start_pos // self.ratio
        g = latents.shape[1]
        pos = (g0 + mx.arange(g)) * self.ratio
        k = self.k_norm(self.wk(latents))
        k = rope_tail(k, rd, cos[pos], sin[pos])
        k = fake_quant_fp4_ue8m0(k, 32)
        cache.index_k[:k.shape[0], g0:g0 + g] = k

    def __call__(self, x: mx.array, qr: mx.array, start_pos: int, offset: int,
                 cos, sin, index_k: mx.array, shared) -> mx.array:
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
        q = rope_tail(q, rd, cos[start_pos:start_pos + n], sin[start_pos:start_pos + n])
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
            v, i, blk = _tiled_scores(
                q.astype(mx.float32), index_k, w.astype(mx.float32), lens, nb, k, tile,
                cand_mask=mask,
                cand_src=((self.candidate_topk_blocks, self.candidate_block_size)
                          if self.is_candidate_source else None))
            if self.is_candidate_source:
                shared.candidates = blk
            order = mx.argsort(i, axis=-1)                           # position order
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
