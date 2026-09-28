# LOCAL ADDITION (not vendored). Session / prefix KV reuse for DeepSeek-V4.1.
"""Keep a finished turn's :class:`~.cache.ModelCache` alive and prefill only
the delta on the next turn.

How it works
------------

A ``ModelCache`` is position-addressed almost everywhere, so a turn's state is
a complete description of the prefix it has seen:

* window ring -- slot ``p % window``, so a slot is a *class* of positions;
* compressed KV / index keys -- append-only per completed compressor group,
  and a group is rewritten by the chunk that completes it before anything
  reads it;
* compressor carry -- ``ratio`` rows of the open group, plus how many are live
  (``spec.snap`` / ``spec.rollback``'s format);
* engram id history -- an indexable host array.

The one buffer that cannot be left alone on a rewind is the **window ring**: a
discarded write at position ``q`` clobbers the slot that a live read at
``q - window`` needs, and the first row of the next forward reads the window at
the rollback point *before* any write of its own. (``spec.rollback`` is exact
for the speculative loop precisely because there the discarded rows are then
re-written at the same positions before anything reads them -- a rewind is
not.) So a session checkpoint stores a materialized ring copy per layer
(~256 KB each at ``window=128, head_dim=512`` fp32; ~10 MB for 40 layers) next
to the compressor carry, captured with :func:`spec.snap`.

Stale compressed rows above a rollback point are never read: ``compress_len =
end_pos // ratio`` only reaches a group once the chunk that closes it has
written it, and the same holds for the index-key cache. Stale engram ids above
the point are only ever read as lookback rows ``p - shift`` for positions
``p >= target``, which are either below ``target`` (reused prefix) or already
written by the calling forward.

API
---

``SessionCache.append_turn(ids)`` feeds the *whole* new token list of a
conversation turn: it finds the common prefix with what is cached, rewinds to
the newest checkpoint at or below it, and prefills only ``ids[boundary:]``.
Every turn end is checkpointed, so the normal multi-turn case (turn N+1's list
= turn N's list + the reply + new text) prefills exactly the reply + new text;
when the new list extends the cached one exactly, nothing is rewound at all.
A prefix mismatch rewinds to the newest checkpoint below the divergence point
and re-prefills from there; ``cancel()`` rewinds to the start of the current
turn. Reused-prefix output equals a fresh full prefill (measured bitwise where
the chunk plan matches, cos >= 0.99999 otherwise).

Generation inside a turn uses ``append_tokens`` (raw feed, no boundary
bookkeeping). The delta prefill runs through ``prefill`` from stream C's
``prefill.py`` when that module is present, otherwise through
:func:`chunked_prefill` -- same signature, same chunking policy, plain
synchronous chunks.

``SessionStore`` is the prefix-keyed LRU an engine holds: ``store.get(ids)``
returns the session whose cached tokens share the longest prefix with ``ids``
(or a new one) and appends the turn for you.
"""

from __future__ import annotations

import collections
import hashlib
import inspect
import time
from dataclasses import dataclass

import numpy as np
import mlx.core as mx

from . import spec as _spec

NEG_INF = float("-inf")

# Chunking policy defaults -- identical to prefill.py's, so switching between
# the two drivers changes nothing but the fences/queue.
BASE_CHUNK = 512
LONG_CHUNK = 128
LONG_THRESHOLD = 8192
CLEAR_EVERY = 4


class RollbackError(RuntimeError):
    """No checkpoint at (or below) the requested rewind target."""


class CapacityError(RuntimeError):
    """The turn does not fit in the cache's ``max_seq_len``."""


# --------------------------------------------------------------------------
# prefix helpers (host side, no MLX)
# --------------------------------------------------------------------------

def _as_ids(ids) -> np.ndarray:
    """Any of list / np / mx ``[n]`` or ``[1, n]`` -> contiguous int64 ``[n]``."""
    if isinstance(ids, mx.array):
        arr = np.array(ids)
    else:
        arr = np.asarray(ids)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"session ids must be [n] or [1, n], got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.int64)


def prefix_hash(ids, size: int = 16) -> str:
    """Stable hex digest of a token prefix (the session key).

    blake2b over the little-endian int32 token stream: platform independent,
    ~1 us for 1 K tokens, and cheap enough to recompute per turn.
    """
    arr = _as_ids(ids)
    return hashlib.blake2b(np.ascontiguousarray(arr, dtype="<i4").tobytes(),
                           digest_size=size).hexdigest()


def common_prefix_len(a, b) -> int:
    """Length of the shared leading token run of two sequences."""
    x, y = _as_ids(a), _as_ids(b)
    n = min(x.shape[0], y.shape[0])
    if n == 0:
        return 0
    diff = x[:n] != y[:n]
    if not diff.any():
        return n
    return int(np.argmax(diff))


def plan_step(pos: int, remaining: int, *, chunk: int, long_chunk: int,
              long_threshold: int) -> int:
    """Rows for the next chunk: ``long_chunk`` once ``pos >= long_threshold``."""
    return min(long_chunk if pos >= long_threshold else chunk, remaining)


# --------------------------------------------------------------------------
# delta prefill driver (stand-in for prefill.py, same signature)
# --------------------------------------------------------------------------

def chunked_prefill(model, ids, cache, *, chunk=None, long_chunk=None,
                    long_threshold=None, last_logit_only=True, argmax=False,
                    return_taps=False, progress=None, fence_every=None,
                    async_depth=None, clear_cache_every=CLEAR_EVERY, **rest):
    """Plain chunked ``model(...)`` loop.

    Drop-in for ``prefill.prefill`` (stream C): same keyword names, same
    chunk-size policy (``chunk`` rows, dropping to ``long_chunk`` once
    ``cache.offset >= long_threshold``), one sync per chunk. ``fence_every`` /
    ``async_depth`` are accepted and ignored -- the fenced driver in
    ``prefill.py`` is the real implementation of those; unknown keywords are
    ignored so a newer driver signature cannot break the session path.
    """
    step_base = BASE_CHUNK if chunk is None else int(chunk)
    step_long = LONG_CHUNK if long_chunk is None else int(long_chunk)
    threshold = LONG_THRESHOLD if long_threshold is None else int(long_threshold)
    ids_mx = mx.array(_as_ids(ids)[None, :].astype(np.int32))
    n = int(ids_mx.shape[1])
    if n == 0:
        raise ValueError("chunked_prefill: empty ids")

    done, chunks, out, taps = 0, 0, None, None
    t0 = time.perf_counter()
    while done < n:
        step = plan_step(cache.offset, n - done, chunk=step_base,
                         long_chunk=step_long, long_threshold=threshold)
        piece = ids_mx[:, done:done + step]
        last = done + step == n
        if last:
            res = model(piece, cache, last_logit_only=last_logit_only,
                        return_taps=return_taps, argmax=argmax)
            if isinstance(res, tuple):
                out, taps = res
            else:
                out = res
        else:
            # intermediate chunks only need to be committed; one argmax row
            # through the head is the cheapest way to hand the queue a handle
            res = model(piece, cache, last_logit_only=True, argmax=True)
        mx.eval(*([out, *taps.values()] if taps is not None else [out]))
        done += step
        chunks += 1
        if progress is not None:
            progress(chunks, done, time.perf_counter() - t0)
        if clear_cache_every and done < n and chunks % clear_cache_every == 0:
            mx.clear_cache()
    return (out, taps) if taps is not None else out


def resolve_prefill_fn(fn=None):
    """The fenced ``prefill.prefill`` when stream C's module is present."""
    if fn is not None:
        return fn
    try:
        from .prefill import prefill as _fenced
        return _fenced
    except Exception:                                    # not merged yet
        return chunked_prefill


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------

@dataclass
class Snapshot:
    """A restorable checkpoint of one ``ModelCache`` at ``pos``.

    ``rings`` is one materialized window-ring copy per layer, ``carries`` the
    per-layer compressor open-group rows in :func:`spec.snap`'s format
    ``(kv_rows, score_rows, m)`` or ``None`` for layers without a compressor.
    """

    pos: int
    rings: list
    carries: list

    def nbytes(self) -> int:
        return sum(r.nbytes for r in self.rings if r is not None)


def _restore_carry(lc, saved) -> None:
    """Write a checkpointed open-group carry back into ``lc``.

    Rows ``[m, ratio)`` are canonicalized (zero kv / NEG_INF score) because the
    aborted run left stale rows there: the compressor only ever reads ``[:m]``,
    but a canonical array makes checkpointed and fresh caches comparable.
    """
    cs = lc.comp_state
    if cs is None:
        return
    cs.chunk_kv = cs.chunk_score = None
    cs.chunk_start = 0
    if saved is None or saved[2] == 0:
        lc.reset_carry()
        return
    kv_rows, score_rows, m = saved
    b, ratio, hd = cs.kv_state.shape
    if m > ratio:
        raise RollbackError(f"carry row count {m} > ratio {ratio}")
    kv = mx.concatenate([kv_rows.astype(cs.kv_state.dtype),
                         mx.zeros((b, ratio - m, hd), dtype=cs.kv_state.dtype)], axis=1)
    sc = mx.concatenate([score_rows.astype(cs.score_state.dtype),
                         mx.full((b, ratio - m, hd), NEG_INF,
                                 dtype=cs.score_state.dtype)], axis=1)
    cs.kv_state, cs.score_state = kv, sc


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------

@dataclass
class TurnResult:
    """What one ``append_turn`` did."""

    logits: object                # prefill output for the last delta token, or None on a pure hit
    tokens_prefilled: int
    tokens_reused: int
    rolled_back_from: int | None   # cache offset the turn start rewound away from
    turn_start: int
    pos: int                       # cache offset after the turn
    hit: bool
    seconds: float

    def __str__(self) -> str:      # compact log line
        return (f"pos={self.pos} prefill={self.tokens_prefilled} "
                f"reuse={self.tokens_reused}"
                + (f" rewind={self.rolled_back_from}->{self.turn_start}"
                   if self.rolled_back_from is not None else "")
                + f" hit={self.hit} {self.seconds:.2f}s")


class SessionCache:
    """One conversation's ``ModelCache`` plus its rewind checkpoints."""

    def __init__(self, model, *, max_seq_len: int | None = None, bsz: int = 1,
                 dtype=mx.float32, max_snapshots: int = 4, prefill_fn=None,
                 chunk: int | None = None, long_chunk: int | None = None,
                 long_threshold: int | None = None,
                 clear_cache_every: int | None = None, progress=None):
        if bsz != 1:
            raise ValueError("session reuse is single-sequence (bsz=1)")
        self.model = model
        self.bsz, self.dtype = bsz, dtype
        args = getattr(model, "args", None)
        fallback = min(getattr(args, "max_seq_len", 4096), 4096)
        self.max_seq_len = int(max_seq_len or fallback)
        self.cache = model.make_cache(bsz, max_seq_len=self.max_seq_len, dtype=dtype)

        self._ids = np.zeros((0,), dtype=np.int64)
        self._snaps: "collections.OrderedDict[int, Snapshot]" = collections.OrderedDict()
        self.max_snapshots = max(2, int(max_snapshots))
        self._turn_start = 0
        self.n_turns = 0
        self.stats = collections.Counter()
        self.last_output = None          # most recent prefill output (pure hits reuse it)

        self._prefill = resolve_prefill_fn(prefill_fn)
        wanted = {"chunk": chunk, "long_chunk": long_chunk,
                  "long_threshold": long_threshold,
                  "clear_cache_every": clear_cache_every, "progress": progress}
        try:
            accepted = set(inspect.signature(self._prefill).parameters)
            if any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in inspect.signature(self._prefill).parameters.values()):
                accepted |= wanted.keys()
        except (TypeError, ValueError):
            accepted = set(wanted)
        self.prefill_kwargs = {k: v for k, v in wanted.items()
                               if v is not None and k in accepted}
        self.snapshot()                                  # the pos-0 boundary

    # ---- introspection ----

    @property
    def offset(self) -> int:
        return self.cache.offset

    @property
    def tokens(self) -> np.ndarray:
        """Exactly the tokens the cache has seen."""
        return self._ids

    @property
    def key(self) -> str:
        return prefix_hash(self._ids)

    @property
    def boundaries(self) -> list:
        return list(self._snaps)

    def summary(self) -> str:
        return (f"SessionCache(pos={self.offset}/{self.max_seq_len} turns={self.n_turns} "
                f"key={self.key[:10]} boundaries={self.boundaries} "
                f"snapshots={len(self._snaps)}x{self._snap_bytes()/1e6:.1f}MB "
                f"prefilled={self.stats['prefilled']} reused={self.stats['reused']} "
                f"rewinds={self.stats['rewinds']}")

    def _snap_bytes(self) -> int:
        return sum(s.nbytes() for s in self._snaps.values())

    # ---- checkpoints ----

    def snapshot(self) -> Snapshot:
        """Checkpoint the current state (materialized ring copies + carries)."""
        pos = self.cache.offset
        _, carries = _spec.snap(self.cache, pos)         # reuse spec's carry rows
        rings = [lc.ring_snapshot() for lc in self.cache.layers]
        mx.eval([r for r in rings if r is not None])
        snap = Snapshot(pos=pos, rings=rings, carries=carries)
        self._snaps[pos] = snap
        self._snaps.move_to_end(pos)
        while len(self._snaps) > self.max_snapshots:
            # oldest first -- the newest checkpoint (cancel's target) always stays
            del self._snaps[next(iter(self._snaps))]
        return snap

    def _boundary_le(self, target: int) -> int:
        best = None
        for p in self._snaps:
            if p <= target and (best is None or p > best):
                best = p
        if best is None:
            raise RollbackError(f"no checkpoint at or below {target} "
                                f"(have {self.boundaries})")
        return best

    def rewind(self, target: int) -> int:
        """Restore the checkpoint at exactly ``target``; returns rows discarded.

        Checkpoints above ``target`` are dropped: the next forwards rewrite
        those positions, possibly with different tokens.
        """
        snap = self._snaps.get(target)
        if snap is None:
            raise RollbackError(
                f"no checkpoint at {target}; rewind targets must be a "
                f"checkpoint (have {self.boundaries}) - use plan()/append_turn() "
                f"to rewind to the newest checkpoint below a divergence point")
        pos = self.cache.offset
        for lc, ring, carry in zip(self.cache.layers, snap.rings, snap.carries):
            lc.ring_restore(ring)
            _restore_carry(lc, carry)
        self.cache.offset = target
        self._ids = np.ascontiguousarray(self._ids[:target])
        for p in [p for p in self._snaps if p > target]:
            del self._snaps[p]
        self._snaps.move_to_end(target)
        if pos != target:
            self.stats["rewinds"] += 1
            self.stats["rewound"] += pos - target
        return pos - target

    def cancel(self) -> int:
        """Drop the in-flight generation: rewind to the newest checkpoint.

        Everything fed since the last checkpoint (the last completed turn end
        or an explicit :meth:`snapshot`) is discarded; the prompt/prefix state
        is exact again and the caller can re-run the generation. Returns the
        number of rows discarded (0 when there is nothing to drop).
        """
        newest = None
        for p in self._snaps:
            if p < self.cache.offset and (newest is None or p > newest):
                newest = p
        if newest is None:
            return 0
        return self.rewind(newest)

    def reset(self) -> None:
        """Forget everything (next turn is a cold full prefill)."""
        self.del_cache()
        self.cache = self.model.make_cache(self.bsz, max_seq_len=self.max_seq_len,
                                           dtype=self.dtype)
        self._ids = np.zeros((0,), dtype=np.int64)
        self._snaps.clear()
        self._turn_start = 0
        self.snapshot()

    def del_cache(self) -> None:
        """Release the ModelCache buffers (session no longer usable)."""
        self.cache = None
        self._snaps.clear()
        self._ids = np.zeros((0,), dtype=np.int64)
        mx.clear_cache()

    # ---- planning ----

    def plan(self, ids) -> tuple:
        """``(lcp, boundary, delta_len)`` for feeding ``ids`` now.

        An exact prefix hit needs no rewind at all (``boundary`` = the current
        offset). A mismatch rewinds to the newest checkpoint at or below the
        common prefix.
        """
        ids = _as_ids(ids)
        lcp = common_prefix_len(ids, self._ids)
        if lcp >= self.cache.offset:                     # cached tokens are a prefix of ids
            boundary = self.cache.offset
        else:
            boundary = self._boundary_le(lcp)
        return lcp, boundary, int(ids.shape[0]) - boundary

    def _check_capacity(self, n: int) -> None:
        if self.cache.offset + n > self.max_seq_len:
            raise CapacityError(
                f"session needs {self.cache.offset + n} tokens, cache holds "
                f"{self.max_seq_len}; size the session for the full context")

    # ---- feeding ----

    def _prefill_call(self, ids, *, argmax, return_taps):
        kw = dict(self.prefill_kwargs)
        kw["argmax"] = argmax
        kw["return_taps"] = return_taps
        return self._prefill(self.model, ids, self.cache, **kw)

    def append_tokens(self, ids, *, argmax: bool = False, return_taps: bool = False):
        """Feed tokens continuing the cached sequence. No boundary bookkeeping.

        This is the in-turn path (prompt chunk, generate step, verify batch).
        """
        ids = _as_ids(ids)
        if ids.shape[0] == 0:
            raise ValueError("append_tokens: empty ids")
        self._check_capacity(int(ids.shape[0]))
        out = self._prefill_call(ids, argmax=argmax, return_taps=return_taps)
        self._ids = np.concatenate([self._ids, ids])
        self.stats["prefilled"] += int(ids.shape[0])
        self.last_output = out
        return out

    def append_turn(self, ids, *, argmax: bool = False, return_taps: bool = False,
                    checkpoint: bool = True) -> TurnResult:
        """Feed a whole turn's token list, prefilling only what is new.

        ``ids`` is the conversation's token list for this turn; the cached
        prefix is rewound to the newest checkpoint at or below the common
        prefix and only ``ids[boundary:]`` is prefilled. The turn end is
        checkpointed (``checkpoint=False`` for streaming partial turns, where
        the next call continues the same turn).
        """
        ids = _as_ids(ids)
        if ids.shape[0] == 0:
            raise ValueError("append_turn: empty ids")
        _, boundary, _ = self.plan(ids)
        rolled_from = self.cache.offset if boundary < self.cache.offset else None
        if rolled_from is not None:
            self.rewind(boundary)
        self._turn_start = boundary
        self.n_turns += 1
        delta = ids[boundary:]
        t0 = time.perf_counter()
        out = None
        if delta.shape[0]:
            self._check_capacity(int(delta.shape[0]))
            try:
                out = self._prefill_call(delta, argmax=argmax,
                                         return_taps=return_taps)
            except Exception:
                # leave a cancelable cache behind: the tokens actually fed
                self._ids = np.concatenate([self._ids, delta[:self.cache.offset - boundary]])
                raise
        seconds = time.perf_counter() - t0
        self._ids = ids.copy()
        self.stats["prefilled"] += int(delta.shape[0])
        self.stats["reused"] += boundary
        self.stats["turns"] += 1
        if delta.shape[0]:
            self.last_output = out
            logits = out
        elif rolled_from is None:
            logits = self.last_output          # exact hit: the caller's next-token logits
            if logits is None:
                self.stats["hits_without_output"] += 1
        else:
            # pure truncation: nothing was computed and the old output belongs
            # to rows that were just discarded
            logits = None
        if checkpoint:
            self.snapshot()
        return TurnResult(logits=logits, tokens_prefilled=int(delta.shape[0]),
                          tokens_reused=boundary, rolled_back_from=rolled_from,
                          turn_start=boundary, pos=self.cache.offset,
                          hit=delta.shape[0] == 0, seconds=seconds)


# --------------------------------------------------------------------------
# prefix-keyed registry
# --------------------------------------------------------------------------

class SessionStore:
    """LRU of :class:`SessionCache`, keyed by the hash of the cached prefix.

    ``get(ids)`` picks the session whose cached tokens share the longest prefix
    with ``ids`` (at least ``reuse_min`` tokens, so a trivial overlap does not
    thrash a big session), appends the turn, and rekeys it. Sessions over
    ``max_sessions`` are dropped oldest-first.
    """

    def __init__(self, model, *, max_sessions: int = 2, reuse_min: int = 0,
                 **session_kw):
        self.model = model
        self.max_sessions = max(1, int(max_sessions))
        self.reuse_min = max(0, int(reuse_min))
        self._sessions: "collections.OrderedDict[str, SessionCache]" = collections.OrderedDict()
        self.kw = session_kw
        self.stats = collections.Counter()

    def keys(self) -> list:
        return list(self._sessions)

    def get(self, ids, *, argmax: bool = False):
        """``(session, TurnResult)`` for this turn's token list."""
        ids = _as_ids(ids)
        best, best_lcp = None, 0
        for s in self._sessions.values():
            if s.cache is None:
                continue
            lcp = common_prefix_len(ids, s.tokens)
            if lcp > best_lcp:
                best, best_lcp = s, lcp
        if best is None or best_lcp < max(self.reuse_min, 1):
            best = SessionCache(self.model, **self.kw)
            self.stats["cold"] += 1
            best_lcp = 0
        elif best_lcp < best.tokens.shape[0]:
            self.stats["prefix_rewind"] += 1
        res = best.append_turn(ids, argmax=argmax)
        self.put(best)
        return best, res

    def put(self, session: SessionCache) -> None:
        for k, s in list(self._sessions.items()):
            if s is session:
                del self._sessions[k]
        self._sessions[session.key] = session
        while len(self._sessions) > self.max_sessions:
            _, victim = self._sessions.popitem(last=False)
            victim.del_cache()
            self.stats["evicted"] += 1

    def __repr__(self) -> str:
        return (f"SessionStore(sessions={len(self._sessions)} "
                f"keys={[k[:8] for k in self._sessions]} stats={dict(self.stats)})")
