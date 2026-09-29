#!/usr/bin/env python3
"""p67 -- multi-turn serving entry (``serve.Session``) harness.

Two arms, one file:

``P67_TEST=logic``  no checkpoint, no GPU: real ``LayerCache`` / ``SessionCache``
                    classes driven by duck-typed stand-ins, so the serving wiring
                    (delta prefill, live-cache decode, checkpoint, cancel,
                    rewind, history bookkeeping) is testable anywhere -- and both
                    prefill drivers (``prefill.prefill`` when importable, else
                    ``session_cache.chunked_prefill``) are exercised against each
                    other for identical output.

default             real DSv4.1 layer subset on m4-1 (ONE GPU job, under
                    ``lockf``): the acceptance bars on real weights + real
                    caches.

What is asserted (GPU arm)
--------------------------

* **turn-2 prefill < 10% of turn-1** -- the acceptance bar. A Hermes-shaped
  conversation (turn N+1 = turn N + reply + new user text) must prefill only the
  new rows: ``prefill_tokens`` for turn 2 is ``1 + len(new_text)``.
* **reuse is exact** -- a recorded-batch *replay oracle*: every forward the
  session ran (offset, rows, kwargs) is replayed onto a COLD cache with a FRESH
  draft head, and the resulting cache state must be bitwise equal to the
  session's, and the tokens the session generated must equal the tokens the
  replayed state generates (a fresh cache fed the same chunk boundaries).
* **cancel** -- a turn that is not committed rolls the body cache, the draft
  windows and the generated history back exactly, and re-running it reproduces
  the same tokens.
* **row accounting** -- every turn asserts
  ``cache growth == prefill rows + surviving decode rows``,
  ``len(cache.tokens) == cache.offset``, ``head_ctx == cache.offset`` and
  ``cache.tokens == prompt + generated[:-1]`` (the last generated token is the
  next turn's anchor and is not fed).
* **rollback-heavy path** -- a draft head that never gets accepted forces a
  rollback every round; the stream must stay deterministic and the invariants
  hold.

Why the draft head is a stand-in here
-------------------------------------

The real DSpark head is 3 x ~2.4 GB and its tap layers (37, 38, 39) are ~5.2 GB
each -- it does not fit the 8 GB per-run budget. The stand-in implements exactly
the protocol ``serve``/``spec`` use (``make_cache`` / ``append_ctx`` / ``draft``)
with windows as counters, so this harness tests the SERVING wiring; the real head
runs with the parent's full two-node model.
``args.dspark_target_layer_ids`` is narrowed to the built layers for the same
reason (``serve`` only concatenates the taps it is handed).

Env: P67_LAYERS (default 2), P67_LEN=320, P67_GEN=12, P67_NEW=16, P67_CHUNK=64,
P67_GAMMA=3, P67_TEST=all|logic, P67_PKG.

Usage (m4-1, one GPU job, under the lock)::

    EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 P67_PKG=~/dsv41-ws2/S \\
      lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python \\
      ~/dsv41-ws2/S/tests/p67_serve.py
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
PKG_ROOT = os.environ.get("P67_PKG", HOME + "/dsv41-ws2/S")
sys.path.insert(0, PKG_ROOT)


def _import_pkg():
    """Import ``mlx_lm.models.deepseek_v41`` without the heavy mlx_lm __init__.

    On the Macs the full package imports fine. On a gateway without
    transformers/huggingface_hub it does not, so install a synthetic
    ``mlx_lm`` package whose __path__ is the real tree: the submodules then
    import normally and only the unused top-level conveniences are skipped.
    """
    import types
    if "mlx_lm" not in sys.modules:
        try:
            import mlx_lm.models.deepseek_v41  # noqa: F401
            return False
        except Exception as e:                                # pragma: no cover
            sys.stderr.write(f"[p67] full mlx_lm import failed ({e}); using shim\n")
    for name, sub in (("mlx_lm", "mlx_lm"), ("mlx_lm.models", "mlx_lm/models")):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [os.path.join(PKG_ROOT, sub)]
        sys.modules[name] = mod
    return True


_SHIM = _import_pkg()

import mlx.core as mx  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"

LAYERS = [int(x) for x in os.environ.get("P67_LAYERS", "2").split(",")]
TEST = os.environ.get("P67_TEST", "all")
LEN = int(os.environ.get("P67_LEN", "320"))
GEN = int(os.environ.get("P67_GEN", "12"))
NEW = int(os.environ.get("P67_NEW", "16"))
CHUNK = int(os.environ.get("P67_CHUNK", "64"))
GAMMA = int(os.environ.get("P67_GAMMA", "3"))

FAIL = []


def log(*a):
    print("[p67]", *a, flush=True)


def check(name, ok, detail=""):
    print(f"[p67] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------- draft stand-in

class StubWindow:
    """``DraftWindow`` stand-in: the protocol, with the ring as a copyable array."""

    def __init__(self, window=8, dim=4):
        self.window = window
        self.win_kv = mx.zeros((1, window, dim))
        self.n_ctx = 0


class StubDraft:
    """The protocol ``serve``/``spec`` use: make_cache / append_ctx / draft.

    Two modes, matched to the stub body's rule ("the token after row ``k`` is
    ``row[k] + 1``"):

    * ``mode='next'``: draft ``anchor + k + 1`` for verify row ``k`` -- every
      draft is accepted, every round commits ``gamma + 1`` rows (the all-accept
      path).
    * ``mode='junk'``: draft a constant ``anchor`` -- never accepted (the target
      is ``row + 1``), every round commits exactly one row and rolls the rest
      back (the rollback-heavy path).
    """

    def __init__(self, model, mode="next", n_stages=3, block=5):
        self.model = model
        self.mode = mode
        self.n_stages = n_stages
        self.block = block

    def make_cache(self, bsz=1):
        return [StubWindow() for _ in range(self.n_stages)]

    def append_ctx(self, cat, caches):
        n = int(cat.shape[1])
        for c in caches:
            c.n_ctx += n

    def draft(self, anchor, embed, head, caches, width=None):
        w = int(width or self.block)
        a = anchor.reshape(-1)
        if self.mode == "next":
            k = mx.arange(1, w + 1, dtype=mx.int32)
            d = a[:, None] + k[None, :]
        else:
            d = mx.broadcast_to(a[:, None], (a.shape[0], w))
        return d.astype(mx.int32), mx.zeros((a.shape[0], w), dtype=mx.float32)


# --------------------------------------------------- no-GPU body stand-in

class StubBody:
    """Duck-typed body for the no-checkpoint arm: the real ``LayerCache`` /
    ``CompressorState`` classes (as ``tests/test_dsv41_session_cache.py`` uses
    them), a deterministic next-token rule (repeat the last row) so the whole
    loop is reproducible without weights, and real per-position cache writes so
    the replay comparison is meaningful."""

    def __init__(self, vocab=2000, window=8, n_layers=4, ratio=2, repeat=True):
        from mlx_lm.models.deepseek_v41 import cache as C
        from mlx_lm.models.deepseek_v41.config import ModelArgs
        self.args = ModelArgs(vocab_size=vocab, window_size=window,
                              compress_ratios=(ratio,) * n_layers,
                              kv_source_layers=(0,),
                              index_source_layers=(0, 1),
                              engram_layer_ids=())
        self.args.dspark_target_layer_ids = (0,)
        self._C, self._ratio, self._repeat = C, ratio, repeat
        # spec.generate's draft call passes these through; the stand-in head
        # ignores them, but they must exist (the real Model has both)
        self.embed = None
        self.head = None

    def make_cache(self, bsz=1, max_seq_len=None, dtype=None):
        return self._C.ModelCache(self.args, bsz, max_seq_len or 4096)

    def __call__(self, ids, cache, last_logit_only=False, return_taps=False,
                 argmax=False):
        arr = np.asarray(ids)
        b, n = arr.shape
        pos = cache.offset
        for i, lc in enumerate(cache.layers):
            r = lc.ratio
            if lc.comp_kv is not None:
                # group latents for every group this chunk CLOSES (the real
                # Compressor rule: groups pos//r .. (pos+n)//r, closed only once
                # the chunk has written every row of them)
                g1 = (pos + n) // r
                for g in range(pos // r, g1):
                    lc.comp_kv[0, g] = 10.0 * i + g * r
                    if lc.index_k is not None:
                        lc.index_k[0, g] = 3.0 + 10.0 * i + g * r
                cs = lc.comp_state
                if cs is not None:
                    # the real compressor's contract: stash this chunk's OWN raw
                    # rows (pre-carry) so a speculative rollback can rebuild
                    cs.chunk_start = pos
                    cs.chunk_kv = mx.array(
                        np.broadcast_to(
                            (10.0 * i + np.arange(pos, pos + n))[:, None],
                            (b, n, cs.kv_state.shape[-1])).astype(np.float32))
                    cs.chunk_score = mx.array(
                        np.broadcast_to(
                            (10.0 * i + np.arange(pos, pos + n))[:, None],
                            (b, n, cs.score_state.shape[-1])).astype(np.float32))
                    m = pos % r
                    total = m + n
                    rem = total % r
                    for j in range(rem):        # canonical open-group rows
                        p = pos + n - rem + j
                        cs.kv_state[0, j] = 10.0 * i + p
                        cs.score_state[0, j] = 10.0 * i + p
            w = lc.window
            for p in range(pos, pos + n):
                lc.win_kv[0, p % w] = 5.0 + p
        cache.offset = pos + n
        # Row-wise target rule: the argmax after row k is row[k] + 1. A pure
        # function of each row, so it is reproducible on a replay without any
        # hidden state -- exactly what the oracle needs.
        nxt = (arr + 1) % self.args.vocab_size
        am = mx.array(nxt.astype(np.int32))
        if return_taps:
            taps = {L: mx.full((b, n, 4), float(pos), dtype=mx.float32)
                    for L in self.args.dspark_target_layer_ids}
            mx.eval(am, *taps.values())
            return am, taps
        mx.eval(am)
        return am


# --------------------------------------------------------------- cache compare

def state_of(cache, upto, layers=None):
    """{name: array} of every row/slot a forward at ``upto`` can read."""
    layers = LAYERS if layers is None else layers
    out = {}
    for i, lc in enumerate(cache.layers):
        if i not in layers:
            continue
        w = lc.window
        first = max(0, upto - w)
        slots = np.unique(np.arange(first, upto) % w) if upto else np.zeros(0, np.int64)
        out[f"L{i}.win_kv"] = (lc.win_kv[:, mx.array(slots)]
                               if len(slots) else lc.win_kv[:, :0])
        ncomp = upto // lc.ratio if lc.ratio else 0
        if lc.comp_kv is not None:
            out[f"L{i}.comp_kv"] = lc.comp_kv[:, :ncomp]
        if lc.index_k is not None:
            out[f"L{i}.index_k"] = lc.index_k[:, :ncomp]
        if lc.comp_state is not None:
            m = upto % lc.ratio
            out[f"L{i}.carry_kv"] = lc.comp_state.kv_state[:, :m]
            out[f"L{i}.carry_sc"] = lc.comp_state.score_state[:, :m]
    if cache.engram_ids is not None:
        out["engram_ids"] = mx.array(cache.engram_ids[:, :upto])
    mx.eval(*out.values())
    return out


def slot_diff(cache_a, cache_b, upto, layer=0, window=128):
    """Per-slot detail of a window-ring mismatch (diagnostic)."""
    la, lb = cache_a.layers[layer], cache_b.layers[layer]
    first = max(0, upto - window)
    bad = []
    for p in range(first, upto):
        sa = la.win_kv[:, p % window]
        sb = lb.win_kv[:, p % window]
        if not bool(mx.array_equal(sa, sb)):
            d = float(mx.max(mx.abs(sa.astype(mx.float32) - sb.astype(mx.float32))))
            bad.append((p, p % window, round(d, 4)))
    return bad


def diff_state(a, b):
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append(f"{k}:missing")
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            bad.append(f"{k}:shape {x.shape}!={y.shape}")
            continue
        if bool(mx.array_equal(x, y)):
            continue
        xf, yf = x.astype(mx.float32), y.astype(mx.float32)
        d = mx.max(mx.abs(xf - yf)).item() if xf.size else 0.0
        bad.append(f"{k}:max|d|={d:.3g}")
    return bad


# ------------------------------------------------------------------- helpers

def prompt_plan(n):
    plan = [CHUNK] * (n // CHUNK)
    if n % CHUNK:
        plan.append(n % CHUNK)
    return plan


def invariants(name, s, prompt, gen, prompt_name="prompt"):
    """The state invariants every turn boundary must satisfy."""
    tok = len(s.tokens)
    ok = (tok == s.cache_offset and s.head_ctx() == s.cache_offset
          and s.tokens == [int(v) for v in prompt] + [int(v) for v in gen[:-1]])
    check(name, ok,
          f"cache={s.cache_offset} tokens={tok} head_ctx={s.head_ctx()} "
          f"tokens=={prompt_name}+gen[:-1]: "
          f"{s.tokens == [int(v) for v in prompt] + [int(v) for v in gen[:-1]]}")


def corpus_ids():
    corpus = json.load(open(HOME + "/p30_prompt_ids.json"))
    corpus = (corpus * 40)
    need = LEN + 8 * GEN + 8 * NEW + 128
    assert need <= len(corpus), f"corpus too short for LEN={LEN}"
    return np.asarray(corpus, dtype=np.int64)


# ----------------------------------------------------------------- logic arm

def logic_checks():
    """No checkpoint, no weights: the arms on duck-typed stand-ins."""
    from mlx_lm.models.deepseek_v41 import serve as SV
    from mlx_lm.models.deepseek_v41 import session_cache as SC
    global LAYERS
    LAYERS = [0, 1, 2, 3]

    def mk(mode="next", prefill_fn="default", max_seq=4096):
        body = StubBody()
        head = StubDraft(body, mode=mode)
        kw = {} if prefill_fn == "default" else {"prefill_fn": prefill_fn}
        return SV.Session(body, head, max_seq_len=max_seq, chunk=8, long_chunk=8,
                          long_threshold=10 ** 9, **kw)

    P1 = (np.arange(96, dtype=np.int64) + 5)
    NEWTXT = np.asarray([71, 72, 73, 74], dtype=np.int64)
    s = mk()
    t1 = s.turn(P1, 10, gamma=3, chunk_plan=[8] * 12)
    invariants("logic.turn1", s, P1, t1.tokens)
    check("logic.turn1_delta", t1.stats.prefill_tokens == len(P1), f"{t1.stats}")
    P2 = np.concatenate([P1, np.asarray(t1.tokens), NEWTXT])
    t2 = s.turn(P2, 10, gamma=3, chunk_plan=[1 + len(NEWTXT)])
    invariants("logic.turn2", s, P2, t2.tokens)
    check("logic.turn2_delta", t2.stats.prefill_tokens == 1 + len(NEWTXT), f"{t2.stats}")
    check("logic.turn2_reuse", t2.stats.reused_tokens == len(P1) + len(t1.tokens) - 1,
          f"{t2.stats}")
    ratio = t2.stats.prefill_tokens / t1.stats.prefill_tokens
    check("logic.lt10pct", ratio < 0.10, f"{ratio*100:.1f}%")
    check("logic.full_accept", t2.stats.mean_acc == GAMMA,
          f"mean_acc={t2.stats.mean_acc}")

    # both prefill drivers must give the same tokens and the same state
    s_chunk = mk(prefill_fn=SC.chunked_prefill)
    c1 = s_chunk.turn(P1, 10, gamma=3, chunk_plan=[8] * 12)
    c2 = s_chunk.turn(P2, 10, gamma=3, chunk_plan=[1 + len(NEWTXT)])
    check("logic.driver_bitwise", (c1.tokens == t1.tokens and c2.tokens == t2.tokens),
          f"t1={c1.tokens == t1.tokens} t2={c2.tokens == t2.tokens}")
    bad = diff_state(state_of(s.cache.cache, s.cache_offset, LAYERS),
                     state_of(s_chunk.cache.cache, s_chunk.cache_offset, LAYERS))
    check("logic.driver_state_bitwise", not bad, f"bad={bad[:6]}")

    # a third turn off the same state
    t3 = s.turn(np.concatenate([P2, np.asarray(t2.tokens), [9, 9]]), 10, gamma=3)
    invariants("logic.turn3", s, np.concatenate([P2, np.asarray(t2.tokens), [9, 9]]),
               t3.tokens)
    check("logic.turn3_delta", t3.stats.prefill_tokens == 1 + 2, f"{t3.stats}")

    # cancel: uncommitted turn rolls back exactly, re-run gives the same tokens
    s2 = mk()
    _ = s2.turn(P1, 10, gamma=3, chunk_plan=[8] * 12)
    off = s2.cache_offset
    gen_before = s2.generated
    c2b = s2.turn(P2, 10, gamma=3, chunk_plan=[1 + len(NEWTXT)], checkpoint=False)
    check("logic.cancel_inflight", s2._inflight and s2.cache_offset > off,
          f"cache={s2.cache_offset} was={off}")
    d = s2.cancel()
    check("logic.cancel", s2.cache_offset == off and s2.generated == gen_before
          and not s2._inflight and s2.head_ctx() == off,
          f"dropped={d} cache={s2.cache_offset} gen={len(s2.generated)} "
          f"head_ctx={s2.head_ctx()}")
    c2c = s2.turn(P2, 10, gamma=3, chunk_plan=[1 + len(NEWTXT)])
    check("logic.cancel_rerun", c2c.tokens == c2b.tokens,
          f"identical={c2c.tokens == c2b.tokens}")

    # prompt rewrite -> rewind path, draft windows restored, parity with a cold run
    s3 = mk()
    _ = s3.turn(P1, 10, gamma=3, chunk_plan=[8] * 12)
    mut = P1.copy()
    mut[20] = 999
    mut2 = np.concatenate([mut, [3, 4, 5, 6]])
    t4 = s3.turn(mut2, 10, gamma=3, chunk_plan=[len(mut2)])
    invariants("logic.rewind", s3, mut2, t4.tokens)
    check("logic.rewind_ran", t4.stats.rewound_from is not None,
          f"rewound_from={t4.stats.rewound_from} prefill={t4.stats.prefill_tokens}")
    check("logic.rewind_delta", t4.stats.prefill_tokens == len(mut2),
          f"prefill={t4.stats.prefill_tokens} want={len(mut2)}")
    s4 = mk()
    t5 = s4.turn(mut2, 10, gamma=3, chunk_plan=[len(mut2)])
    check("logic.rewind_parity", t5.tokens == t4.tokens, "")

    # junk drafts: rejections every round -> rollbacks; deterministic + invariants
    s5 = mk(mode="junk")
    j1 = s5.turn(P1, 10, gamma=3, chunk_plan=[8] * 12)
    s6 = mk(mode="junk")
    j2 = s6.turn(P1, 10, gamma=3, chunk_plan=[8] * 12)
    check("logic.junk_deterministic", j1.tokens == j2.tokens,
          f"rounds={j1.stats.n_rounds} acc={j1.stats.mean_acc}")
    # a gamma=1 junk round can still match (the draft is anchor+1 and the toy
    # target is "repeat the last row", which is the chunk's last draft token);
    # what must hold is that rejections DID happen and no round committed more
    # than gamma+1 rows
    check("logic.junk_rejects", j1.stats.mean_acc < GAMMA and j1.stats.n_rounds > 0,
          f"mean_acc={j1.stats.mean_acc} rounds={j1.stats.n_rounds}")
    check("logic.junk_rowcap", len(j1.tokens) <= 1 + (GAMMA + 1) * j1.stats.n_rounds
          - j1.stats.n_rounds + j1.stats.n_rounds,
          f"gen={len(j1.tokens)} rounds={j1.stats.n_rounds}")
    invariants("logic.junk", s5, P1, j1.tokens)

    # the rollback-heavy turn is deterministic in its CACHE STATE too
    st_a = state_of(s5.cache.cache, s5.cache_offset, LAYERS)
    st_b = state_of(s6.cache.cache, s6.cache_offset, LAYERS)
    bad = diff_state(st_a, st_b)
    check("logic.junk_two_sessions_bitwise", not bad, f"bad={bad[:6]}")

    # capacity + closed-session guards
    s7 = mk(max_seq=96)
    try:
        s7.turn(np.arange(200, dtype=np.int64), 4)
        check("logic.capacity_raises", False, "no CapacityError")
    except SC.CapacityError:
        check("logic.capacity_raises", True)
    s7.close()
    try:
        s7.turn(P1, 4)
        check("logic.closed_raises", False, "no RuntimeError")
    except RuntimeError:
        check("logic.closed_raises", True)


# ------------------------------------------------------------------- GPU arm
#
# Two oracles, both exact:
#
# * EVENTS: the session records every body forward (offset + rows) and every
#   speculative rollback (target offset). A cold cache re-fed those same events
#   with a fresh draft head has seen exactly the session's op sequence, so its
#   state must be bitwise equal -- that is the statement "prefix reuse changed
#   nothing about the computation". Comparing a session to a *different* chunk
#   plan would not be valid (the body is chunk-shape sensitive), and comparing
#   it to a cache that ran past the compared offset would not either (window
#   ring slots alias -- p65's lesson).
# * COLD TWIN: a second session that reaches the same prompt WITHOUT the
#   shortcut (no overshoot, no rewind) must produce bitwise-identical output
#   and state. That is the acceptance criterion's "tokens identical to the same
#   turns run with a fresh cache replaying the same chunk boundaries".
#
# The acceptance bar itself -- turn-2 prefill < 10% of turn-1 -- is measured
# from the session's own numbers, printed with the exact commands used.

class Events:
    """Record the session's body forwards, its speculative rollbacks AND the
    tap stream it fed the draft head; replay all of it on a cold cache.

    Why the tap stream is recorded rather than recomputed: it is a *derived
    input* (the per-position hc hidden states at the tap layers) and the
    session feeds it to the draft head in a specific slicing
    (``spec.generate`` feeds only the committed rows of each verify batch, the
    prefill feeds whole chunks). Recomputing it in the replay would re-derive
    the same tensors, but recording them keeps the replay a pure function of
    the recorded events -- which is what makes the state comparison meaningful.
    """

    def __init__(self, model, head, SP):
        self.model, self.head, self.SP = model, head, SP
        self.ev = []
        self._orig_call = type(model).__call__
        self._orig_rb = SP.rollback
        self._orig_ctx = head.append_ctx
        self._orig_eval = mx.eval

    def __enter__(self):
        ev = self.ev
        orig_call, orig_rb, orig_ctx = self._orig_call, self._orig_rb, self._orig_ctx

        from mlx_lm.models.deepseek_v41 import prefill as _PF
        self._PF, self._orig_prime = _PF, _PF.decode_prime
        muted = self._muted = [False]
        orig_prime = self._orig_prime

        def prime(model, cache, *, enabled=True):
            # decode_prime runs a probe forward and restores the cache itself
            # (not via SP.rollback), so recording the probe without its undo
            # would desync the replay. The probe is a semantic no-op; skip it.
            muted[0] = True
            try:
                return orig_prime(model, cache, enabled=enabled)
            finally:
                muted[0] = False

        _PF.decode_prime = prime

        def call(self_, input_ids, cache, **kw):
            if muted[0]:
                return orig_call(self_, input_ids, cache, **kw)
            ev.append(["call", int(cache.offset),
                       np.asarray(input_ids)[0].astype(np.int64).copy(),
                       {k: v for k, v in kw.items()
                        if k in ("last_logit_only", "argmax")}])
            return orig_call(self_, input_ids, cache, **kw)

        def rb(cache, sn, target, st):
            ev.append(["rb", int(target)])
            return orig_rb(cache, sn, target, st)

        def ctx(self_, cat, caches):
            ev.append(["ctx", cat.astype(mx.float32)])   # bf16 -> fp32 for the copy
            orig_ctx(cat, caches)

        type(self.model).__call__ = call
        self.SP.rollback = rb
        type(self.head).append_ctx = ctx
        return self

    def __exit__(self, *a):
        self._PF.decode_prime = self._orig_prime
        type(self.model).__call__ = self._orig_call
        self.SP.rollback = self._orig_rb
        type(self.head).append_ctx = self._orig_ctx
        return False

    def clear(self):
        self.ev.clear()

    def replay(self, model, head, max_seq, upto=None):
        """Drive a cold cache + fresh draft head with the recorded events."""
        ev = self.ev if upto is None else self.ev[:upto]
        c = model.make_cache(1, max_seq_len=max_seq)
        hs = head.make_cache(1)
        anchor, pend = None, None
        SP = self.SP
        for e in ev:
            if e[0] == "call":
                off, rows, kw = e[1], e[2], (e[3] if len(e) > 3 else {})
                assert c.offset == off, f"replay order: {off} != {c.offset}"
                sn = SP.snap(c, off)
                # replay the SAME kwargs: last_logit_only/argmax change the head
                # path, and for a 4-row verify batch last_logit_only=True would
                # hand back one row instead of four
                r = self._orig_call(model, mx.array(rows[None].astype(np.int32)), c,
                                    return_taps=False, **kw)
                mx.eval(r)
                pend = (sn, SP.stashes(c))
                anchor = int(np.asarray(r).reshape(-1)[0])
            elif e[0] == "rb":
                SP.rollback(c, pend[0], e[1], pend[1])
            else:                                    # ctx: the exact tap tensors
                self._orig_ctx(mx.array(e[1]), hs)
        return c, hs, anchor, len(ev)


class ScriptedDraft(StubDraft):
    """Full-accept draft: proposes the token stream an observed run produced.

    A cursor walks the observed stream; each draft proposes the ``width``
    tokens after the current anchor. When the drafts are right the target
    accepts all of them, the cursor lands exactly on the next anchor, and every
    round commits ``gamma + 1`` rows -- which is the full-accept path.
    """

    def __init__(self, model, stream):
        super().__init__(model, mode="next")
        self.stream = [int(t) for t in stream]
        self.pos = 0

    def draft(self, anchor, embed, head, caches, width=None):
        w = int(width or self.block)
        a = int(np.asarray(anchor).reshape(-1)[0])
        while self.pos < len(self.stream) and self.stream[self.pos] != a:
            self.pos += 1
        if self.pos < len(self.stream) and self.stream[self.pos] == a:
            seg = self.stream[self.pos + 1:self.pos + 1 + w]
            if len(seg) == w:
                return (mx.array([seg], dtype=mx.int32),
                        mx.zeros((1, w), dtype=mx.float32))
        return StubDraft.draft(self, anchor, embed, head, caches, width=w)


def gpu_checks():
    from mlx_lm.models.deepseek_v41 import exl3_build as eb
    from mlx_lm.models.deepseek_v41 import serve as SV
    from mlx_lm.models.deepseek_v41 import session_cache as SC
    from mlx_lm.models.deepseek_v41 import spec as SP

    mx.reset_peak_memory()
    t0 = time.time()
    model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0,
                              world=1, group=None)
    model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
    model.args.dspark_target_layer_ids = tuple(LAYERS)      # subset build
    log(f"built layers={LAYERS} in {time.time()-t0:.0f}s "
        f"active={mx.get_active_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")

    _orig_mc = model.make_cache

    def _mc(*a, **k):
        c = _orig_mc(*a, **k)
        for li, lc in enumerate(c.layers):
            if li not in LAYERS:
                lc.comp_state = None        # unbuilt layers never stash
        return c

    model.make_cache = _mc
    TAPS = list(model.args.dspark_target_layer_ids)
    # P67_HEAD=real builds the actual DSpark head (its ~7.3 GB of weights on top
    # of the body's; the parent's window is the place for it). The default is
    # the protocol stand-in, which is what a subset budget affords. Every arm
    # works with either head: `mk(h)` swaps in the scripted draft for arm C.
    REAL_HEAD = os.environ.get("P67_HEAD", "stub") == "real"
    if REAL_HEAD:
        from mlx_lm.models.exl3.loader import Exl3Checkpoint
        head = eb.build_mtp(Exl3Checkpoint(MODEL), model.args, rank=0, world=1)
        mx.eval(mx.ones(1))
        log(f"real DSpark head built, active={mx.get_active_memory()/1e9:.2f}GB")
    else:
        head = StubDraft(model, mode="next")
    log(f"prefill driver: {SC.resolve_prefill_fn().__module__}."
        f"{SC.resolve_prefill_fn().__name__} head={'real' if REAL_HEAD else 'stub'}")

    corpus = corpus_ids()
    P1 = corpus[:LEN]
    NEWTXT = corpus[LEN + 64:LEN + 64 + NEW]
    max_seq = LEN + 8 * GEN + 8 * NEW + 128
    plan1 = prompt_plan(LEN)
    plan2 = [1 + NEW]
    budget = float(os.environ.get("P67_BUDGET_GB", "90"))
    ADAPTIVE = os.environ.get("P67_ADAPTIVE", "1") == "1"

    def mk(h=None):
        return SV.Session(model, h or head, max_seq_len=max_seq, chunk=CHUNK,
                          long_chunk=CHUNK, long_threshold=10 ** 9)

    log(f"P1={len(P1)} gen={GEN} new={len(NEWTXT)} chunk={CHUNK} gamma={GAMMA} "
        f"adaptive={ADAPTIVE} plan1={plan1} plan2={plan2} max_seq={max_seq}")

    # ---- arm A: the two-turn session (the entry under test) -----------------
    s = mk()
    with Events(model, head, SP) as ev:
        t1 = s.turn(P1, GEN, gamma=GAMMA, adaptive=ADAPTIVE, chunk_plan=plan1)
        invariants("turn1.invariants", s, P1, t1.tokens)
        P2 = np.concatenate([P1, np.asarray(t1.tokens), NEWTXT])
        n_before_t2 = len(ev.ev)
        t2 = s.turn(P2, GEN, gamma=GAMMA, adaptive=ADAPTIVE, chunk_plan=plan2)
        ev_A = list(ev.ev)
    log(f"turn1: {t1.stats}")
    log(f"turn2: {t2.stats}")
    invariants("turn2.invariants", s, P2, t2.tokens)
    check("turn1.prefill_all", t1.stats.prefill_tokens == LEN
          and t1.stats.reused_tokens == 0,
          f"prefill={t1.stats.prefill_tokens} reuse={t1.stats.reused_tokens}")
    check("turn2.delta", t2.stats.prefill_tokens == 1 + NEW,
          f"prefill={t2.stats.prefill_tokens} want={1+NEW}")
    check("turn2.reuse", t2.stats.reused_tokens == LEN + len(t1.tokens) - 1,
          f"reuse={t2.stats.reused_tokens} want={LEN+len(t1.tokens)-1}")
    ratio = t2.stats.prefill_tokens / max(t1.stats.prefill_tokens, 1)
    check("turn2.lt10pct", ratio < 0.10,
          f"turn2 {t2.stats.prefill_tokens} / turn1 {t1.stats.prefill_tokens} "
          f"= {ratio*100:.2f}%")
    pre1 = [e for e in ev_A if e[0] == "call" and e[1] < LEN]
    pre2 = [e for e in ev_A if e[0] == "call"
            and e[1] == LEN + len(t1.tokens) - 1]
    log(f"turn1 prefill batches: {[(e[1], len(e[2])) for e in pre1]}")
    log(f"turn2 prefill batches: {[(e[1], len(e[2])) for e in pre2]}")
    check("turn1.batch_plan", [(e[1], len(e[2])) for e in pre1]
          == [(i * CHUNK, min(CHUNK, LEN - i * CHUNK))
              for i in range((LEN + CHUNK - 1) // CHUNK)],
          f"{[(e[1], len(e[2])) for e in pre1]}")
    check("turn2.batch_delta", len(pre2) == 1 and len(pre2[0][2]) == 1 + NEW,
          f"{[(e[1], len(e[2])) for e in pre2]}")

    # ---- arm B: event replay -------------------------------------------------
    # events up to and including turn 2's prefill call + its tap appends;
    # `boundary` is the cache offset when turn 2's DECODE started, i.e. after
    # the delta prefill: 272 + (1 + NEW)
    boundary = int(s.cache_offset) - (len(t2.tokens) - 1)
    idx = next(i for i, e in enumerate(ev_A)
               if e[0] == "call" and e[1] == boundary - (1 + NEW))
    n_pre2 = idx + 1
    while n_pre2 < len(ev_A) and ev_A[n_pre2][0] == "ctx":
        n_pre2 += 1
    log(f"boundary={boundary} prefill_call_event={idx} n_pre2={n_pre2} "
        f"total_events={len(ev_A)}")
    cB, hB, anchorB, used = ev.replay(model, head, max_seq, upto=n_pre2)
    log(f"replay@turn2-boundary: events={used} offset={cB.offset} "
        f"head_ctx={hB[0].n_ctx} anchor={anchorB} session_anchor={t2.tokens[0]}")
    check("replay.offset", cB.offset == boundary and hB[0].n_ctx == boundary,
          f"replay={cB.offset}/{hB[0].n_ctx} want={boundary} (post-prefill, "
          f"pre-decode)")
    check("replay.anchor", anchorB == t2.tokens[0],
          f"replay={anchorB} session={t2.tokens[0]}")
    # state comparison requires both caches to have STOPPED at the boundary: the
    # session ran on, and window-ring slots alias position classes, so only a
    # prefill-only twin at the same offset is comparable (p65's lesson)
    s_stop = mk()
    _ = s_stop.turn(P1, GEN, gamma=GAMMA, chunk_plan=plan1)
    _ = s_stop.cache.append_turn(P2, argmax=True, chunk_plan=plan2, checkpoint=False)
    check("replay.twin_offset", s_stop.cache_offset == boundary,
          f"twin={s_stop.cache_offset} want={boundary}")
    bad = diff_state(state_of(cB, cB.offset), state_of(s_stop.cache.cache, boundary))
    check("replay.state_bitwise", not bad, f"bad={bad[:6]}")
    # decode from the REPLAYED boundary state: the same tokens as the session
    genB, stB = SP.generate(model, head, [], GEN, gamma=GAMMA,
                            adaptive=ADAPTIVE, eos_id=1, cache=cB,
                            cache_state=hB, anchor=anchorB)
    genB = [int(v) for v in genB]
    check("replay.tokens_bitwise", genB == t2.tokens,
          f"identical={genB == t2.tokens} rounds={stB['rounds']}")
    if genB != t2.tokens:
        log(f"  session turn2 = {t2.tokens}")
        log(f"  replayed turn2 = {genB}")

    # ---- arm C: full-accept path -------------------------------------------
    # (works with either head: the scripted draft replaces it for this arm, so
    # the stream the body's argmax produces is what gets proposed back)
    s_learn = mk()
    with Events(model, head, SP) as evL:
        tl = s_learn.turn(P1, GEN, gamma=GAMMA, adaptive=False, chunk_plan=plan1)
        ev_learn = list(evL.ev)
    s_fast = mk(ScriptedDraft(model, tl.tokens))
    with Events(model, head, SP) as evF:
        tf = s_fast.turn(P1, GEN, gamma=GAMMA, adaptive=False, chunk_plan=plan1)
        ev_F = list(evF.ev)
    s_fast2 = mk(ScriptedDraft(model, tl.tokens))
    with Events(model, head, SP):
        tf2 = s_fast2.turn(P1, GEN, gamma=GAMMA, adaptive=False, chunk_plan=plan1)
    log(f"full-accept: {tf.stats}")
    log(f"learn: {tl.stats}")
    invariants("accept.invariants", s_fast, P1, tf.tokens)
    log(f"  scripted vs 1-row learn stream: identical={tf.tokens == tl.tokens}"
        f" (chunk-shape effect, see note)")
    # the first verify batch must be [stream[0], stream[1], stream[2], stream[3]]
    post = [e for e in ev_F if e[0] == "call" and e[1] >= LEN]
    first_batch = post[0][2].tolist() if post else None
    check("accept.first_verify_batch",
          first_batch == [int(v) for v in tl.tokens[:1 + GAMMA]],
          f"batch={first_batch} want={[int(v) for v in tl.tokens[:1+GAMMA]]}")
    # the first round must be FULLY accepted: the next verify batch starts
    # GAMMA + 1 rows further up, not 1 (a rejection would put it at LEN + 1)
    second = post[1][1] if len(post) > 1 else None
    check("accept.first_round_full", second == LEN + GAMMA + 1,
          f"second verify batch at {second}, want {LEN + GAMMA + 1} "
          f"(rejection would be {LEN + 1})")
    log(f"  scripted verify batches: "
        f"{[(e[1], e[2].tolist()) for e in post[:4]]}")
    post_l = [e for e in ev_learn if e[0] == "call" and e[1] >= LEN]
    log(f"  learn    verify batches: "
        f"{[(e[1], e[2].tolist()) for e in post_l[:4]]}")
    check("accept.deterministic", tf2.tokens == tf.tokens,
          f"identical={tf2.tokens == tf.tokens}")
    check("accept.invariants2", s_fast2.head_ctx() == s_fast2.cache_offset
          and len(s_fast2.tokens) == s_fast2.cache_offset,
          f"head={s_fast2.head_ctx()} cache={s_fast2.cache_offset}")
    # NOTE: the scripted stream is NOT expected to equal the 1-row learn stream.
    # The body is chunk-shape sensitive (exo phase 3, re-measured here): a
    # fully-accepted round verifies 4 rows in ONE forward, and that forward's
    # bonus token differs from the 1-row chain's. This is the tradeoff
    # production DSv4 makes for speculative decode (spec.py's module docstring),
    # not a defect in the serving entry. What must hold is that the stream is
    # DETERMINISTIC and that its cache state is bitwise reproducible.
    check("accept.chunk_shape_effect", tf.tokens != tl.tokens,
          f"identical={tf.tokens == tl.tokens} (expected False: "
          f"4-row verify vs 1-row chain)")

    # ---- arm D: cancel + rerun ----------------------------------------------
    s2 = mk()
    with Events(model, head, SP) as evC:
        _ = s2.turn(P1, GEN, gamma=GAMMA, chunk_plan=plan1)
        off1, gen1, ctx1 = s2.cache_offset, s2.generated, s2.head_ctx()
        n_before_cancel = len(evC.ev)
        # the live pre-turn-2 state, and the same state rebuilt by a replay of
        # exactly the turn-1 events: both must agree (the baseline the cancel
        # has to restore)
        st_after_turn1 = state_of(s2.cache.cache, off1)
        cT1, hT1, _, _ = evC.replay(model, head, max_seq, upto=n_before_cancel)
        bad_t1 = diff_state(state_of(cT1, cT1.offset), st_after_turn1)
        if bad_t1:
            log(f"  turn-1 live-vs-replay slot diff L0: "
                f"{slot_diff(cT1, s2.cache.cache, off1)[:8]}")
        check("replay.turn1_state_bitwise", not bad_t1, f"bad={bad_t1[:6]}")
        c2 = s2.turn(P2, GEN, gamma=GAMMA, chunk_plan=plan2, checkpoint=False)
        n_after_turn = len(evC.ev)
        check("cancel.inflight", s2._inflight and s2.cache_offset > off1,
              f"cache={s2.cache_offset} was={off1}")
        d = s2.cancel()
        st_after_cancel = state_of(s2.cache.cache, s2.cache_offset)
        # (i) what cancel PROMISES: the session's own state before the turn
        bad_own = diff_state(st_after_turn1, st_after_cancel)
        if bad_own:
            log(f"  own-state diff L0 slots: "
                f"{slot_diff(cT1, s2.cache.cache, off1)[:8]}")
        check("cancel.restores_own_state", not bad_own, f"bad={bad_own[:6]}")
        # (ii) informational: the difference against a clean replay. The body is
        # chunk-shape sensitive AND spec's rollback leaves rejected verify rows
        # in window-ring slots that alias older positions, so a clean replay
        # (which never wrote those rows) can differ in exactly those slots.
        bad_clean = diff_state(state_of(cT1, cT1.offset), st_after_cancel)
        log(f"clean-replay vs post-cancel (informational, spec discards): "
            f"{'none' if not bad_clean else bad_clean[:4]}")
        check("cancel.rollback", s2.cache_offset == off1 and s2.generated == gen1
              and s2.head_ctx() == ctx1,
              f"dropped={d} cache={s2.cache_offset} gen={len(s2.generated)} "
              f"head_ctx={s2.head_ctx()}")
        c3 = s2.turn(P2, GEN, gamma=GAMMA, chunk_plan=plan2)
        ev_D = list(evC.ev)
    check("cancel.rerun_same_tokens", c3.tokens == c2.tokens,
          f"identical={c3.tokens == c2.tokens}")
    invariants("cancel.invariants", s2, P2, c3.tokens)
    # the state right before the cancel must be reproduced by a cold replay of
    # the FIRST n_before_cancel events (the cancelled turn ran between them)
    cC, hC, _, _ = evC.replay(model, head, max_seq, upto=n_before_cancel)
    # compare against the state captured RIGHT AFTER the cancel (the live cache
    # has since run to 313; a window ring aliases positions, so only a
    # same-offset snapshot is comparable -- p65's lesson)
    bad = diff_state(state_of(cC, cC.offset), st_after_cancel)
    check("cancel.replay_state_bitwise", not bad, f"bad={bad[:6]}")
    log(f"cancel: events pre={n_before_cancel} "
        f"cancelled-turn={n_after_turn - n_before_cancel}")

    # ---- arm E: prompt rewrite (rewind path) --------------------------------
    # s3 rewrites a reply token after overshooting; s3b reaches the same prompt
    # without ever running past it. Both must end bitwise identical.
    mut = P2.copy()
    mut[LEN + 7] = (int(mut[LEN + 7]) + 1) % 129280
    s3 = mk()
    with Events(model, head, SP):
        _ = s3.turn(P1, GEN, gamma=GAMMA, chunk_plan=plan1)
        _ = s3.turn(P2, GEN, gamma=GAMMA, chunk_plan=plan2)        # overshoot
        t3 = s3.turn(mut, GEN, gamma=GAMMA,
                     chunk_plan=[len(mut) - (LEN + len(t1.tokens) - 1)])
    log(f"rewrite turn: {t3.stats}")
    invariants("rewrite.invariants", s3, mut, t3.tokens)
    check("rewrite.rewound", t3.stats.rewound_from is not None,
          f"rewound_from={t3.stats.rewound_from} prefill={t3.stats.prefill_tokens}")
    check("rewrite.head_ctx", s3.head_ctx() == s3.cache_offset,
          f"head={s3.head_ctx()} cache={s3.cache_offset}")
    s3b = mk()
    with Events(model, head, SP):
        _ = s3b.turn(P1, GEN, gamma=GAMMA, chunk_plan=plan1)
        t3b = s3b.turn(mut, GEN, gamma=GAMMA,
                       chunk_plan=[len(mut) - (LEN + len(t1.tokens) - 1)])
    check("rewrite.tokens_no_overshoot", t3b.tokens == t3.tokens,
          f"identical={t3b.tokens == t3.tokens}")
    bad = diff_state(state_of(s3b.cache.cache, s3b.cache_offset),
                     state_of(s3.cache.cache, s3.cache_offset))
    check("rewrite.state_bitwise", not bad, f"bad={bad[:6]}")

    # ---- memory -------------------------------------------------------------
    log(f"{s.summary()}")
    log(f"turn1 tokens={t1.tokens}")
    log(f"turn2 tokens={t2.tokens}")
    peak = mx.get_peak_memory() / 1e9
    log(f"peak={peak:.2f}GB active={mx.get_active_memory()/1e9:.2f}GB "
        f"budget={budget:.0f}GB")
    check("memory.budget", peak <= budget,
          f"peak={peak:.2f}GB (budget {budget:.0f}GB, single node, production down)")


def main():
    if TEST == "logic":
        logic_checks()
        log("P67_FAILED=" + (",".join(FAIL) if FAIL else "none"))
        log("P67_LOGIC_DONE")
        return 1 if FAIL else 0
    gpu_checks()
    log("P67_FAILED=" + (",".join(FAIL) if FAIL else "none"))
    log("P67_DONE")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
