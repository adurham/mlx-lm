# LOCAL ADDITION (not vendored). DSv4.1 multi-turn serving entry point.
"""One object per conversation: ``Session.turn(...)`` serves a multi-turn chat
by reusing the cached prefix, prefilling only the delta, decoding
speculatively from the existing cache, and keeping ALL state for the next turn.

Why this shape
--------------

Hermes-style serving is multi-turn with a long shared prefix: turn N+1's token
list is turn N's list + the reply + the new user text. Everything that makes
that cheap has to survive the end of the turn, and three pieces of state do:

* the body ``ModelCache`` (window ring / compressed KV / compressor carries /
  index keys / engram ids) -- owned by
  :class:`~mlx_lm.models.deepseek_v41.session_cache.SessionCache`, which also
  knows how to rewind to a checkpoint;
* the DSpark draft head's context windows (one ``DraftWindow`` per stage, fed
  the body's tap stream) -- owned here, because ``spec`` only ever built them
  inside one ``generate`` call and threw them away at the end;
* the token history the body cache has actually seen -- after a speculative
  round this is NOT the token list the model emitted: a round drafts ``gamma``
  tokens and feeds ``gamma + 1`` rows, of which the rejected suffix is rolled
  back. ``spec.generate(cache_update=...)`` reports each round's surviving rows
  (``mark_seen``), so history stays in lockstep with ``cache.offset``.

Turn protocol
-------------

``turn(prompt_ids, max_new, ...)``:

1. ``SessionCache.append_turn(prompt_ids)`` -- common-prefix match, rewind to
   the newest checkpoint at or below it, prefill ONLY ``ids[boundary:]``. On a
   normal multi-turn extension nothing rewinds and exactly the reply tail + new
   text is prefilled. The delta's *per-chunk taps* are captured here
   (``taps_out``): the draft head must see the tap stream of the new rows and
   they are free at prefill time.
2. If the turn rewound (a prompt edit), the draft windows are restored to the
   matching checkpoint first, so no stale context rows survive.
3. ``spec.generate(..., cache=<the live body cache>, cache_state=<the live
   draft windows>, cache_update=<this session>, anchor=<the prefill's argmax>)``
   decodes from the existing cache. No fresh ``make_cache``, no re-prefill, and
   the first verify row is the anchor -- the same row the fresh path would feed.
4. On commit (the default) the body cache and the draft windows are
   checkpointed together. ``checkpoint=False`` leaves the turn uncommitted so
   :meth:`cancel` has something real to roll back.

State invariant at a turn boundary

* ``cache.offset == len(cache.tokens)`` (``SessionCache`` maintains this);
* ``generated[-1]`` is the next turn's anchor and has NOT been fed, so a
  Hermes-style next prompt (this turn's list + reply + new text) shares exactly
  ``cache.offset`` rows with the cache and its delta is
  ``[generated[-1]] + new_text``.

``cancel()`` rolls THE WHOLE TURN back: body cache to its last checkpoint,
draft windows to the matching ring copy, generated history to the length
recorded at that checkpoint -- so re-running the same turn produces the same
bytes.

Parity with a fresh cache
-------------------------

The body is chunk-shape sensitive (a position's result depends on the chunk it
was computed in: exo phase 3, measured again in ``prefill.py``/``spec.py``), so
"reused prefix == fresh prefill" is an *exact* claim only when the chunk
boundaries match. ``turn(..., chunk_plan=[...])`` forces the delta's piece
sizes, which is how ``tests/p67_serve.py`` checks parity: it records every body
forward and every speculative rollback the session makes (plus the tap stream
it feeds the draft head), replays those exact events onto a cold cache with a
fresh draft head, and requires the state and the token stream to be bitwise
equal.

Measured (2026-09-29, m4-1, 8-layer subset 0,1,2,3,20,21,24,25, single node,
production down; ``P67_LEN=512 P67_GEN=24 P67_NEW=32 P67_CHUNK=64``):
turn-1 prefill 512 tok, turn-2 prefill 33 tok = 6.45% (at ``P67_LEN=256``:
25/256 = 9.77%); replay state/tokens bitwise identical; cancel restores state
bitwise; peak 43.6 GB. Full log lines are in the harness's report.

Usage
-----

``serve`` holds no model-building code: ``Session(model, head, ...)`` takes a
built body + draft head, so it is testable on a small layer subset (single
node, one GPU job) and used unchanged by the parent's two-node run::

    from mlx_lm.models.deepseek_v41 import serve
    s = serve.Session(model, head, max_seq_len=8192, chunk=512, long_chunk=512)
    t1 = s.turn(prompt_ids_turn1, max_new=32)
    t2 = s.turn(prompt_ids_turn1 + t1.tokens + new_text_ids, max_new=32)
    print(t1.stats, t2.stats)      # t2 prefill = 1 + len(new_text_ids)
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field

import numpy as np
import mlx.core as mx

from . import spec as _spec
from .session_cache import SessionCache, TurnResult, CapacityError, RollbackError

__all__ = ["Session", "Turn", "TurnStats", "CapacityError", "RollbackError"]


@dataclass
class TurnStats:
    """What one :meth:`Session.turn` cost and produced.

    Prefill side: ``prompt_tokens`` (whole conversation this turn),
    ``prefill_tokens`` (rows actually fed to the body) and ``reused_tokens``
    (prefix rows served from the cache) -- the multi-turn win lives here.
    Decode side: ``gen_tokens`` (INCLUDES the token the prefill produced),
    ``decode_seconds``, ``tok_s`` and the speculative loop's ``n_rounds`` /
    ``mean_acc`` / ``gammas``.
    """

    prompt_tokens: int = 0
    prefill_tokens: int = 0
    reused_tokens: int = 0
    cache_offset: int = 0
    hit: bool = False
    gen_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    turn_seconds: float = 0.0
    n_rounds: int = 0
    mean_acc: float = 0.0
    gammas: list = field(default_factory=list)
    committed: bool = True
    rewound_from: int | None = None

    @property
    def tok_s(self) -> float:
        return self.gen_tokens / self.decode_seconds if self.decode_seconds else 0.0

    def __str__(self) -> str:
        return (f"prompt={self.prompt_tokens} prefill={self.prefill_tokens} "
                f"reuse={self.reused_tokens} -> cache={self.cache_offset} "
                f"gen={self.gen_tokens} @ {self.tok_s:.1f} tok/s "
                f"({self.decode_seconds:.2f}s, {self.n_rounds} rounds, "
                f"acc={self.mean_acc:.2f})"
                + ("" if self.committed else " UNCOMMITTED")
                + (f" rewind={self.rewound_from}"
                   if self.rewound_from is not None else ""))


@dataclass
class Turn:
    """Return value of :meth:`Session.turn`."""

    tokens: list          # this turn's generated ids (tokens[0] is the prefill anchor)
    stats: TurnStats

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)


@dataclass
class _Ckpt:
    """One draft-window checkpoint: ring copies + the generated-history length."""

    rings: list
    gen_len: int


class Session:
    """A multi-turn conversation on a built DSv4.1 body + DSpark draft head.

    ``model`` is an ``exl3_build.build_model`` result and ``head`` its
    ``build_mtp`` draft head (anything with ``make_cache`` / ``append_ctx`` /
    ``draft`` works, which is what the layer-subset test uses). A layer-subset
    body works as long as the tap layers are inside the subset.

    Keyword arguments mirror ``SessionCache``'s (chunking policy, snapshot cap,
    ``prefill_fn``) plus ``eos_id`` (1 for this release) and ``head_state``
    (pass existing draft windows; a fresh set is made by default).
    """

    def __init__(self, model, head, *, max_seq_len: int | None = None,
                 eos_id: int = 1, chunk: int | None = None,
                 long_chunk: int | None = None, long_threshold: int | None = None,
                 max_snapshots: int = 4, prefill_fn=None, head_state=None,
                 progress=None):
        self.model = model
        self.head = head
        self.eos_id = int(eos_id)
        self.cache = SessionCache(model, max_seq_len=max_seq_len, chunk=chunk,
                                  long_chunk=long_chunk,
                                  long_threshold=long_threshold,
                                  max_snapshots=max_snapshots,
                                  prefill_fn=prefill_fn, progress=progress)
        self.head_state = head_state if head_state is not None else head.make_cache(1)
        self._gen: list = []                       # every generated token, in order
        self._ckpts: "collections.OrderedDict[int, _Ckpt]" = collections.OrderedDict()
        self._max_ckpts = max(2, int(max_snapshots))
        self.n_turns = 0
        self.last_turn = None
        self._inflight = False
        self.stats = collections.Counter()
        self._checkpoint()                         # the pos-0 boundary

    # ------------------------------------------------------------ introspection

    @property
    def cache_offset(self) -> int:
        """Rows the body cache has seen (the cache's own offset)."""
        return self.cache.offset

    @property
    def tokens(self) -> list:
        """Every token the body cache has seen, in order (``cache_offset`` ids)."""
        return [int(t) for t in self.cache.tokens]

    @property
    def generated(self) -> list:
        """Every token generated in this session, in order (the greedy stream).

        One longer than ``cache_offset`` at a turn boundary: the last generated
        token is the next turn's anchor and has not been fed yet.
        """
        return list(self._gen)

    def head_ctx(self) -> int:
        """Context rows the draft windows hold (must equal ``cache_offset``)."""
        try:
            return int(self.head_state[0].n_ctx)
        except Exception:                                   # pragma: no cover
            return -1

    def summary(self) -> str:
        return (f"Session(turns={self.n_turns} cache={self.cache_offset} "
                f"generated={len(self._gen)} head_ctx={self.head_ctx()} "
                f"snapshots={len(self.cache.boundaries)}"
                + (" INFLIGHT" if self._inflight else "") + ")")

    # ------------------------------------------------------------------ helpers

    def _taps_ids(self) -> list:
        return list(self.model.args.dspark_target_layer_ids)

    def _append_ctx(self, taps) -> None:
        """Push one chunk's tap dict (per tapped layer ``[1, rows, dim]``) into the
        draft windows, in position order."""
        ids = self._taps_ids()
        cat = mx.concatenate([taps[L] for L in ids], axis=-1)
        self.head.append_ctx(cat, self.head_state)

    def _checkpoint(self) -> None:
        """Checkpoint body cache + draft windows + history length, all at
        ``cache.offset``."""
        self.cache.snapshot()
        pos = self.cache.offset
        rings = [mx.array(w.win_kv) for w in self.head_state]
        mx.eval(*rings)
        self._ckpts[pos] = _Ckpt(rings=rings, gen_len=len(self._gen))
        self._ckpts.move_to_end(pos)
        while len(self._ckpts) > self._max_ckpts:
            del self._ckpts[next(iter(self._ckpts))]
        self._inflight = False
        self.stats["checkpoints"] += 1

    def _restore_ckpt(self, pos: int) -> None:
        """Restore the draft windows + generated history to the checkpoint at
        ``pos``.

        The body cache's own buffers are rewound by ``SessionCache``; both sets
        of checkpoints are written together by :meth:`_checkpoint`. Checkpoints
        above ``pos`` are dropped, mirroring ``SessionCache.rewind``, so the two
        stay identical.
        """
        ck = self._ckpts.get(pos)
        if ck is None:
            raise RollbackError(
                f"no draft-window checkpoint at {pos} (have {list(self._ckpts)}); "
                "the body cache and the draft windows are checkpointed together, "
                "so this means the session's snapshot cap changed at runtime")
        for w, ring in zip(self.head_state, ck.rings):
            w.win_kv = mx.array(ring)
            w.n_ctx = pos
        for p in [p for p in self._ckpts if p > pos]:
            del self._ckpts[p]
        self._ckpts.move_to_end(pos)
        if len(self._gen) > ck.gen_len:
            del self._gen[ck.gen_len:]

    # --------------------------------------------------------------------- turn

    def turn(self, prompt_ids, max_new: int, *, gamma: int = 3,
             adaptive: bool = True, temperature: float = 0.0, policy=None,
             chunk_plan=None, checkpoint: bool = True) -> Turn:
        """Serve one conversation turn. See the module docstring.

        ``prompt_ids`` is the WHOLE conversation token list for this turn (turn
        N+1's list = turn N's list + this turn's reply + new user text), exactly
        like an engine's prompt: the prefix match happens inside. Returns a
        :class:`Turn` whose ``.tokens`` are this turn's generated ids
        (``.tokens[0]`` comes straight out of the prefill) and ``.stats`` the
        measured breakdown.

        ``gamma`` / ``adaptive`` / ``policy`` go to the speculative loop.
        ``chunk_plan`` forces the delta prefill's piece sizes (needed for a
        bitwise comparison against a cold run). ``checkpoint=False`` leaves the
        turn uncommitted for :meth:`cancel`.

        A turn that raises leaves the session cancellable: call :meth:`cancel`
        before retrying a different prompt.
        """
        if temperature and temperature > 0.0:
            raise NotImplementedError(
                "serve.Session.turn is greedy (temperature <= 0): the sampling "
                "loop needs the cache=/cache_state= plumbing that this entry "
                "already passes on the greedy path (see spec.generate).")
        if max_new <= 0:
            raise ValueError("max_new must be positive")
        if getattr(self.cache, "cache", None) is None:
            raise RuntimeError("session was closed")

        t0 = time.perf_counter()
        prompt_ids = np.asarray(prompt_ids).reshape(-1)

        # 1. delta prefill: only ids[boundary:] is fed (the whole point)
        taps: list = []
        try:
            res: TurnResult = self.cache.append_turn(
                prompt_ids, argmax=True, taps_out=taps, chunk_plan=chunk_plan,
                checkpoint=False)
        except Exception:
            self._inflight = True
            raise
        pre_s = time.perf_counter() - t0

        if res.hit:
            raise ValueError(
                "turn: the prompt is an exact cache hit (no new tokens); add the "
                "new user text to the prompt, or reset() to re-run from scratch")
        if res.logits is None:
            raise RollbackError(
                "turn: prefill produced no anchor logits (a truncated turn with "
                "no cached prefix); the session needs a prefill first")
        n_tap_rows = sum(int(t[self._taps_ids()[0]].shape[1]) for t in taps)
        if n_tap_rows != int(res.tokens_prefilled):
            raise RuntimeError(
                f"prefill driver reported {n_tap_rows} tap rows for "
                f"{res.tokens_prefilled} prefilled tokens: the draft head cannot "
                "be kept in step (does the installed driver support taps_out?)")

        # 2. draft windows: a rewound turn must not keep stale context rows
        if res.rolled_back_from is not None:
            self._restore_ckpt(int(res.turn_start))
        for t in taps:
            self._append_ctx(t)
        if self.head_ctx() != int(self.cache.offset):
            raise RuntimeError(
                f"draft windows hold {self.head_ctx()} rows but the body cache "
                f"is at {self.cache.offset} after the prefill")

        anchor = int(np.asarray(res.logits).reshape(-1)[0])

        # 3. speculative decode ON the existing cache: the prefill just fed the
        #    last prompt row, so `anchor` is this turn's first generated token
        #    and the cache + draft windows carry the whole prefix.
        self._inflight = True
        gen, st = _spec.generate(
            self.model, self.head, [], max_new, gamma=gamma, adaptive=adaptive,
            eos_id=self.eos_id, policy=policy, cache=self.cache.cache,
            cache_state=self.head_state, cache_update=self.cache, anchor=anchor)
        gen = [int(t) for t in gen]
        self._gen.extend(gen)
        dec_s = float(st["rounds"]) * float(st["ms_round"]) / 1e3

        # row accounting: from the turn's starting offset (post-rewind if the
        # turn rewound) the cache grew by the prefill delta + the decode's
        # surviving rows; the last generated token is the next anchor, un-fed
        start_off = int(res.turn_start)
        if int(self.cache.offset) != start_off + int(res.tokens_prefilled) + len(gen) - 1:
            raise RuntimeError(
                f"row accounting: cache {start_off}->{self.cache.offset} vs prefill "
                f"{res.tokens_prefilled} + decode {len(gen) - 1}")
        if self.cache.tokens.shape[0] != self.cache.offset:
            raise RuntimeError(
                f"history desync: {self.cache.tokens.shape[0]} tokens vs cache "
                f"offset {self.cache.offset}")

        # 4. keep every piece of state for the next turn
        if checkpoint:
            self._checkpoint()
        else:
            self._inflight = True
        self.n_turns += 1
        self.stats["turns"] += 1
        self.stats["tokens_prefilled"] += int(res.tokens_prefilled)
        self.stats["tokens_reused"] += int(res.tokens_reused)
        self.stats["tokens_generated"] += len(gen)

        stats = TurnStats(
            prompt_tokens=int(prompt_ids.shape[0]),
            prefill_tokens=int(res.tokens_prefilled),
            reused_tokens=int(res.tokens_reused),
            cache_offset=int(self.cache.offset),
            hit=bool(res.hit),
            gen_tokens=len(gen),
            prefill_seconds=pre_s,
            decode_seconds=dec_s,
            turn_seconds=time.perf_counter() - t0,
            n_rounds=int(st["rounds"]),
            mean_acc=float(st["mean_acc"]),
            gammas=list(st["gammas"]),
            committed=bool(checkpoint),
            rewound_from=res.rolled_back_from,
        )
        self.last_turn = Turn(tokens=gen, stats=stats)
        return self.last_turn

    # -------------------------------------------------------------- lifecycle

    def cancel(self) -> int:
        """Drop the in-flight turn; returns the number of body-cache rows discarded.

        Rewinds the body cache to its newest checkpoint and restores the draft
        windows + the generated history from the matching session checkpoint --
        the state before the in-flight turn, so re-running it produces the same
        tokens.
        """
        if not self._inflight:
            return 0
        dropped = self.cache.cancel()
        self._restore_ckpt(int(self.cache.offset))
        self._inflight = False
        self.stats["cancels"] += 1
        self.stats["rows_dropped"] += dropped
        return dropped

    def reset(self) -> None:
        """Forget the conversation (the next turn is a cold full prefill)."""
        self.cache.reset()
        self.head_state = self.head.make_cache(1)
        self._gen = []
        self._ckpts.clear()
        self._checkpoint()
        self.n_turns = 0
        self._inflight = False

    def close(self) -> None:
        """Release every buffer this session holds."""
        self.cache.del_cache()
        self.head_state = None
        self._ckpts.clear()
        mx.clear_cache()

    def __repr__(self) -> str:
        return self.summary()
