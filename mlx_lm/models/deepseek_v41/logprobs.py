"""Exact top-k log-probabilities over a vocab-sharded head.

Each rank holds a slice of the logits row. It sends a small per-row summary
(log-sum-exp, max, argmax id, its top-k values and ids); one all_sum of that
buffer is enough to rebuild the exact global answer. The full vocab row never
crosses the wire.

Buffer layout per row: [lse, max, argmax_id, top_vals(k), top_ids(k)].
Ids are carried as fp32; vocab ids < 2**24 are exact in fp32.
"""
from __future__ import annotations

import mlx.core as mx


def local_buffer(y: mx.array, k: int, lo: int) -> mx.array:
    """Summary of local logits ``y`` [rows, w] (fp32); ids offset by ``lo``."""
    y = y.astype(mx.float32)
    k = max(1, min(int(k), y.shape[-1]))
    m = mx.max(y, axis=-1, keepdims=True)
    lse = m[..., 0] + mx.log(mx.sum(mx.exp(y - m), axis=-1))
    am = mx.argmax(y, axis=-1)
    part = mx.argpartition(-y, k - 1, axis=-1)[..., :k]
    vals = mx.take_along_axis(y, part, axis=-1)
    return mx.concatenate(
        [lse[..., None], m, (am + lo).astype(mx.float32)[..., None],
         vals, (part + lo).astype(mx.float32)], axis=-1)


def combine(allp: mx.array, k: int):
    """Global (ids, selected_logprob, top_ids, top_logprobs) from ``allp``.

    ``allp`` is [world, rows, 3 + 2k]: every rank's buffer in its own slot.
    The selected id follows ``mx.argmax`` over the full row (first rank with
    the max, lowest id inside a rank), so it equals the greedy token.
    """
    world, rows = allp.shape[0], allp.shape[1]
    k = (allp.shape[-1] - 3) // 2
    lse = mx.logsumexp(allp[..., 0], axis=0)
    best = mx.argmax(allp[..., 1], axis=0)
    ids = mx.take_along_axis(allp[..., 2], best[None], axis=0)[0].astype(mx.int32)
    sel = mx.max(allp[..., 1], axis=0) - lse
    cv = mx.moveaxis(allp[..., 3:3 + k], 0, 1).reshape(rows, world * k)
    ci = mx.moveaxis(allp[..., 3 + k:3 + 2 * k], 0, 1).reshape(rows, world * k)
    order = mx.argsort(-cv, axis=-1)[..., :k]
    top_lp = mx.take_along_axis(cv, order, axis=-1) - lse[..., None]
    top_ids = mx.take_along_axis(ci, order, axis=-1).astype(mx.int32)
    return ids, sel, top_ids, top_lp


def from_logits(row: mx.array, k: int):
    """The same answer from one full logits row (any leading shape)."""
    flat = row.reshape(-1, row.shape[-1])
    return combine(local_buffer(flat, k, 0)[None], k)
