# LOCAL ADDITION (not vendored). Prefill orchestration for DeepSeek-V4.1.
"""Chunked, fenced prefill driver for ``deepseek_v41.Model``.

Callers (stream E ``session_cache.py``, the exo engine, bench harnesses)::

    from mlx_lm.models.deepseek_v41.prefill import prefill, warmup

    warmup(model)                                   # once, right after load
    logits = prefill(model, ids, cache, argmax=True)  # ids = prompt tokens

Contract of :func:`prefill` (kept deliberately small):

* ``ids`` -- the NEW tokens to append: a list/1-D array of length ``n``, or a
  ``[1, n]`` mx array. Batch must be 1.
* ``cache`` -- a ``ModelCache`` from ``model.make_cache(...)`` that has room for
  ``n`` more tokens. ``cache.offset`` is the context already cached (0 for a
  fresh prompt, or the length of the reused prefix for a session continuation).
* returns exactly what ``model(...)`` returns for the FINAL chunk:
  - default: fp32 logits ``[1, 1, vocab]`` (``last_logit_only=True``),
  - ``argmax=True``: token ids ``[1, 1]`` int32,
  - ``return_taps=True``: a ``(logits_or_ids, taps)`` tuple, taps = the DSpark
    tap dict for the final chunk's rows (same as ``model(..., return_taps=True)``).
  Every returned array is evaluated before return (one sync).
* ``cache.offset`` advances by ``n``. Model-side cache state (window ring,
  compressed KV, compressor carry, index keys, engram ids) is exactly the state
  a single full-length forward with the same chunk boundaries would leave --
  chunks are position-addressed and the compressor carry closes across
  boundaries (``Compressor`` is chunk-general).
* ``taps_out`` -- optional list; when given, the per-chunk tap dicts are
  appended to it (position order), so ``concat(taps_out, axis=1)`` is the tap
  stream of the whole prompt (the DSpark draft-head context feed). Needed
  because ``return_taps`` alone only covers the final chunk.

CHUNK BOUNDARIES (read before relying on "reuse == fresh"): the body's output
depends on the chunk SHAPE, a property of the model itself (exo phase 3 /
``spec.py``), not of this driver. Measured on the p48 layer set at n=1536:
chunking [512,512,512] reproduces a single 1536-row forward bit-exactly, and a
session split [1024]+[512] is bit-identical to [512,512,512] (max cache-state
|delta| = 0, cos 0.9999999). A split whose boundaries differ from the
reference chunking ([768]+[768] vs [512,512,512]) is NOT bit-identical
(cos 0.9896) -- and the plain model path shows the same effect (cos 0.9963), so
it is the model, not the driver. Stream E must therefore reconstruct the same
chunking when it compares a reused prefix against a fresh prefill, or compare
against the plain path's own chunking.

What the driver adds to a plain ``for chunk in range(...)`` loop:

* **eval fences** -- ``model._fence_every = K`` makes ``Model.__call__`` call
  ``mx.eval(h, pre_mix)`` every K layers of a multi-row forward (layer ids
  ``K-1, 2K-1, ...`` of the built set). That commits the chunk in K-layer
  command buffers instead of one giant lazy graph, so transient buffers (the
  multi-hundred-MB indexer scores / sparse-attn gathers) are released as the
  chunk progresses. Fences change no op and no dtype: fenced and unfenced runs
  are bit-identical (measured, every spacing 1/2/4, logits and cache state).
  Fences are skipped for single-row forwards, so decode is untouched.
* **context-aware chunk size** -- ``chunk`` (default 512) while the cached
  context is below ``long_threshold`` (8192), ``long_chunk`` (default 128) once
  ``cache.offset >= long_threshold``. Per-chunk transients of the indexer scale
  with rows x nb, so at long context the smaller chunk keeps the peak bounded.
* **bounded async queue** -- chunk outputs are ``mx.async_eval``'d and at most
  ``async_depth`` chunks are allowed in flight; the oldest is synced when the
  bound is exceeded. Prevents the host from queuing the whole prompt ahead of
  the GPU (which pins every chunk's intermediates in host/GPU memory).
* **periodic ``mx.clear_cache()``** -- every ``clear_cache_every`` chunks,
  returns the allocator's unused buffers to the OS.
* **:func:`warmup`** -- runs the prefill/decode branch shapes once at load time
  so Metal compiles them there instead of on the first user turn (the ~206 s
  first-chunk compile spike in exo phase 19).
"""

from __future__ import annotations

import collections
import os
import time

import mlx.core as mx
import numpy as np

# Defaults; every one is overridable per call. Env vars only seed the defaults.
BASE_CHUNK = int(os.environ.get("DSV41_PREFILL_CHUNK", "512"))
LONG_CHUNK = int(os.environ.get("DSV41_PREFILL_LONG_CHUNK", "128"))
LONG_THRESHOLD = int(os.environ.get("DSV41_PREFILL_LONG_THRESHOLD", "8192"))
FENCE_EVERY = int(os.environ.get("DSV41_FENCE_EVERY", "2"))
ASYNC_DEPTH = int(os.environ.get("DSV41_PREFILL_DEPTH", "2"))
CLEAR_EVERY = int(os.environ.get("DSV41_PREFILL_CLEAR", "4"))


def plan_step(pos: int, remaining: int, *, chunk: int, long_chunk: int,
              long_threshold: int) -> int:
    """Rows for the next chunk: ``long_chunk`` once ``pos >= long_threshold``."""
    step = long_chunk if pos >= long_threshold else chunk
    return min(step, remaining)


def _as_batch(ids) -> mx.array:
    if isinstance(ids, mx.array):
        arr = ids if ids.ndim == 2 else ids[None]
    else:
        arr = np.asarray(ids)
        if arr.ndim == 1:
            arr = arr[None]
        arr = mx.array(arr)
    if arr.ndim != 2 or arr.shape[0] != 1:
        raise ValueError(f"prefill: ids must be [n] or [1, n], got {arr.shape}")
    return arr


def prefill(model, ids, cache, *, chunk: int | None = None,
            long_chunk: int | None = None, long_threshold: int | None = None,
            fence_every: int | None = None, async_depth: int | None = None,
            clear_cache_every: int | None = None, last_logit_only: bool = True,
            argmax: bool = False, return_taps: bool = False, taps_out=None,
            progress=None):
    """Append ``ids`` to ``cache`` in fenced, size-adaptive chunks.

    Returns the final chunk's output (see the module docstring). ``fence_every=0``
    disables fences, ``async_depth=0`` disables the queue bound,
    ``clear_cache_every=0`` disables the periodic clear; ``progress`` is an
    optional ``fn(index, rows_done, elapsed_s)`` hook for logging.

    ``taps_out``: pass a list to collect the DSpark tap dict *per chunk*
    (``[{layer_id: [1, chunk_rows, dim]}, ...]`` in position order). The taps
    are per-position quantities taken at each tapped layer's input, so
    concatenating them along axis 1 over the chunks equals the taps of one
    full-length forward -- this is what the draft head's context feed needs
    when a prompt is prefilled in chunks.
    """
    ids_mx = _as_batch(ids)
    n = int(ids_mx.shape[1])
    if n == 0:
        raise ValueError("prefill: empty ids")

    chunk = BASE_CHUNK if chunk is None else int(chunk)
    long_chunk = LONG_CHUNK if long_chunk is None else int(long_chunk)
    long_threshold = LONG_THRESHOLD if long_threshold is None else int(long_threshold)
    fence_every = FENCE_EVERY if fence_every is None else int(fence_every)
    async_depth = ASYNC_DEPTH if async_depth is None else int(async_depth)
    clear_cache_every = (CLEAR_EVERY if clear_cache_every is None
                         else int(clear_cache_every))

    want_taps = return_taps or taps_out is not None
    fence_prev = getattr(model, "_fence_every", None)
    model._fence_every = fence_every
    pending: collections.deque = collections.deque()
    out = None
    taps = None
    done = 0
    nchunks = 0
    t0 = time.perf_counter()
    try:
        while done < n:
            step = plan_step(cache.offset, n - done, chunk=chunk,
                             long_chunk=long_chunk, long_threshold=long_threshold)
            stop = done + step
            piece = ids_mx[:, done:stop]
            last = stop == n
            if last:
                res = model(piece, cache, last_logit_only=last_logit_only,
                            return_taps=want_taps, argmax=argmax)
            else:
                # Intermediate chunks only need to be committed, not read: a
                # 1-row argmax through the head is the cheapest handle that
                # still forces the whole chunk's graph.
                res = model(piece, cache, last_logit_only=True,
                            return_taps=want_taps, argmax=True)
            if isinstance(res, tuple):
                handles = [res[0], *res[1].values()]
                if taps_out is not None:
                    taps_out.append(res[1])
                if last:
                    if return_taps:
                        out, taps = res
                    else:
                        out = res[0]
            else:
                handles = [res]
                if last:
                    out = res
            mx.async_eval(*handles)
            pending.append(handles)
            while async_depth > 0 and len(pending) > async_depth:
                mx.eval(*pending.popleft())
            nchunks += 1
            done = stop
            if clear_cache_every and nchunks % clear_cache_every == 0 and done < n:
                mx.clear_cache()
            if progress is not None:
                progress(nchunks, done, time.perf_counter() - t0)
    finally:
        model._fence_every = fence_prev if fence_prev is not None else 0

    if pending:
        # Drain everything still in flight so every returned/tapped array is
        # materialised (the caller may read any of them on the host).
        while pending:
            mx.eval(*pending.popleft())
    if taps is not None:
        mx.eval(out, *taps.values())
        return out, taps
    mx.eval(out)
    return out


def warmup(model, *, chunk: int | None = None, long_chunk: int | None = None,
           decode: bool = True, fence_every: int | None = None,
           clear: bool = True) -> dict:
    """Compile the prefill + decode branch shapes now, at load time.

    Runs a scratch cache through 2x``chunk`` rows, 1x``long_chunk`` rows and (by
    default) one 1-row decode step -- the row counts the prefill driver and the
    sampler actually use -- and returns per-stage seconds. The caller's caches
    are untouched. The model must be fully built (token map set for engram).

    ``fence_every`` defaults to the driver default (``FENCE_EVERY``) so the
    compiled command buffers are the ones a real prefill will use.
    """
    chunk = BASE_CHUNK if chunk is None else int(chunk)
    long_chunk = LONG_CHUNK if long_chunk is None else int(long_chunk)
    fence_prev = getattr(model, "_fence_every", None)
    model._fence_every = FENCE_EVERY if fence_every is None else int(fence_every)
    scratch = model.make_cache(1, max_seq_len=2 * chunk + long_chunk + 64)
    times: dict = {}
    dummy = mx.zeros((1, chunk), dtype=mx.int32)
    stages = [("chunk512_a", dummy), ("chunk512_b", dummy),
              ("chunk128", dummy[:, :long_chunk])]
    if decode:
        stages.append(("decode1", mx.zeros((1, 1), dtype=mx.int32)))
    try:
        for name, ids in stages:
            t0 = time.perf_counter()
            r = model(ids, scratch, last_logit_only=True, argmax=True)
            mx.eval(r)
            times[name] = time.perf_counter() - t0
    finally:
        model._fence_every = fence_prev if fence_prev is not None else 0
    del scratch
    if clear:
        mx.clear_cache()
    return times
