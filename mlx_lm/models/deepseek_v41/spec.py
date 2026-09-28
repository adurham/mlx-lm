# LOCAL ADDITION. Speculative decode (DSpark draft + chunk verify) for V4.1.
"""Round loop, rollback and helpers.

Chunk verify runs the ``gamma + 1`` verify rows in ONE body forward. It is not
bit-identical to plain greedy (the body result depends on chunk shape, a
property of the body measured in exo phase 3), which is the same tradeoff
production DSv4 makes. Rollback: window ring and compressed caches are
position-addressed and are rewritten before they are read again; only the
compressor open-group carry must be rebuilt (from the saved carry plus the
rows the compressor stashed for this chunk).
"""

from __future__ import annotations

import mlx.core as mx


def snap(cache, start_pos: int):
    st = []
    for lc in cache.layers:
        cs = lc.comp_state
        if cs is None:
            st.append(None)
            continue
        m = start_pos % lc.ratio
        st.append((mx.array(cs.kv_state[:, :m]), mx.array(cs.score_state[:, :m]), m)
                  if m else (None, None, 0))
    return start_pos, st


def stashes(cache):
    return [(lc.comp_state.chunk_kv, lc.comp_state.chunk_score)
            if lc.comp_state is not None else (None, None) for lc in cache.layers]


def _rebuild(cs, ratio, chunk_start, target, saved, stash):
    s = ratio * (target // ratio)
    need = target - s
    if need == 0:
        return
    saved_kv, saved_sc, m = saved
    stash_kv, stash_sc = stash
    pk, ps = [], []
    if s < chunk_start:
        lo = max(s, chunk_start - m)
        pk.append(saved_kv[:, lo - (chunk_start - m): m])
        ps.append(saved_sc[:, lo - (chunk_start - m): m])
    lo2 = max(s, chunk_start)
    pk.append(stash_kv[:, lo2 - chunk_start: target - chunk_start])
    ps.append(stash_sc[:, lo2 - chunk_start: target - chunk_start])
    nk = mx.concatenate(pk, axis=1)
    ns = mx.concatenate(ps, axis=1)
    if nk.shape[1] != need:
        raise RuntimeError(f"carry rebuild produced {nk.shape[1]} rows, need {need}")
    cs.kv_state[:, :need] = nk
    cs.score_state[:, :need] = ns


def rollback(cache, sn, target: int, st) -> None:
    chunk_start, saved = sn
    cache.offset = target
    for lc, sv, stash in zip(cache.layers, saved, st):
        if lc.comp_state is None:
            continue
        _rebuild(lc.comp_state, lc.ratio, chunk_start, target, sv, stash)
