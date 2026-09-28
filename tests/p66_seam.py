#!/usr/bin/env python3
"""p66 -- the session driver seam: does session_cache use prefill.prefill?

Checks (no model needed):
  1. ``resolve_prefill_fn()`` picks ``prefill.prefill`` when the module is
     importable, and ``chunked_prefill`` when it is not.
  2. ``SessionCache`` accepts the fenced driver's kwargs and only forwards the
     ones it declares (``chunk`` / ``long_chunk`` / ``long_threshold`` /
     ``clear_cache_every`` / ``progress``; no ``fence_every``/``async_depth``
     invented).
  3. ``chunked_prefill`` is signature-compatible with ``prefill.prefill`` for
     every keyword the session passes.
  4. With a real ``prefill.prefill``-shaped stub, the session routes through it
     and the delta is still exact.

Runs on the gateway (mlx import works; no GPU work).
"""
import inspect
import os
import sys
import numpy as np

sys.path.insert(0, os.environ.get("P65_PKG", "/home/hermes/work/dsv41-ws/E"))
import mlx.core as mx  # noqa: E402
from mlx_lm.models.deepseek_v41 import session_cache as SC  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[p66] {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAIL.append(name)


# --- 1. resolution ---------------------------------------------------------
has_prefill = True
try:
    from mlx_lm.models.deepseek_v41 import prefill as PF
    sig_pf = inspect.signature(PF.prefill)
except ImportError:
    has_prefill = False
    sig_pf = None

resolved = SC.resolve_prefill_fn(None)
if has_prefill:
    check("resolve_picks_prefill", resolved is PF.prefill, f"{resolved.__module__}")
else:
    check("resolve_falls_back", resolved is SC.chunked_prefill, "no prefill.py present")

# --- 2. explicit fn wins ---------------------------------------------------
sentinel = lambda *a, **k: None
check("resolve_explicit", SC.resolve_prefill_fn(sentinel) is sentinel)

# --- 3. signature compatibility -------------------------------------------
sig_cp = inspect.signature(SC.chunked_prefill)
if has_prefill:
    shared = set(sig_pf.parameters) & set(sig_cp.parameters)
    session_kw = {"chunk", "long_chunk", "long_threshold", "clear_cache_every",
                  "progress", "argmax", "return_taps"}
    missing = session_kw - set(sig_pf.parameters)
    check("prefill_accepts_session_kwargs", not missing, f"missing={missing}")
    check("chunked_prefill_covers_prefill", session_kw <= set(sig_cp.parameters),
          f"shared={sorted(shared)}")
    # the session must not silently swallow unsupported kwargs
    extra = set(sig_pf.parameters) - set(sig_cp.parameters) - {"self"}
    check("extras_are_optional", all(
        sig_pf.parameters[p].default is not inspect.Parameter.empty
        for p in extra), f"extra={sorted(extra)}")
else:
    check("chunked_prefill_signature", {"chunk", "long_chunk", "long_threshold",
                                        "clear_cache_every", "argmax",
                                        "return_taps"} <= set(sig_cp.parameters))

# --- 4. routing + kwargs filtering ----------------------------------------
from mlx_lm.models.deepseek_v41.config import ModelArgs      # noqa: E402
from mlx_lm.models.deepseek_v41.cache import LayerCache      # noqa: E402

NLAYERS, WINDOW, RATIO = 3, 8, 2
SEEN = []


class StubCache:
    def __init__(self, max_seq_len=256):
        self.max_seq_len, self.offset = max_seq_len, 0
        a = ModelArgs(window_size=WINDOW, compress_ratios=(RATIO,) * NLAYERS,
                      kv_source_layers=(0,), index_source_layers=(0,),
                      engram_layer_ids=())
        self.layers = [LayerCache(1, a, i, max_seq_len) for i in range(NLAYERS)]
        self.engram_ids = None


class StubModel:
    args = ModelArgs()

    def make_cache(self, bsz=1, max_seq_len=None, dtype=None):
        return StubCache(max_seq_len or 256)

    def __call__(self, ids, cache, last_logit_only=True, return_taps=False,
                 argmax=False, **kw):
        """What chunked_prefill calls: commit rows, return a handle."""
        n = int(np.asarray(ids).shape[-1])
        for lc in cache.layers:
            for i in range(cache.offset, cache.offset + n):
                lc.win_kv[0, i % lc.window] = 5.0 + i
        cache.offset += n
        return mx.array([[float(n)]])


def fake_prefill(model, ids, cache, *, chunk=512, long_chunk=128,
                 long_threshold=8192, last_logit_only=True, argmax=False,
                 return_taps=False, progress=None, fence_every=2, async_depth=2,
                 clear_cache_every=4):
    """Same contract as prefill.prefill; records what the session forwarded."""
    SEEN.append({"n": len(ids), "chunk": chunk, "long_chunk": long_chunk,
                 "clear": clear_cache_every, "fence": fence_every})
    n = len(ids)
    pos = cache.offset
    for lc in cache.layers:                       # position-derived writes
        w = lc.window
        for i in range(pos, pos + n):
            lc.win_kv[0, i % w] = 5.0 + i
        if lc.comp_state is None:                 # consumers / ratio-0 layers
            continue
        for i in range(n):
            lc.comp_state.kv_state[0, (pos + i) % RATIO] = 3.0 + pos + i
            lc.comp_state.score_state[0, (pos + i) % RATIO] = 4.0 + pos + i
    cache.offset += n
    return mx.array([[float(n)]])


model = StubModel()
s = SC.SessionCache(model, max_seq_len=256, prefill_fn=fake_prefill, chunk=64,
                    long_chunk=32, long_threshold=100, clear_cache_every=2)
ids = np.arange(40, dtype=np.int64)
s.append_turn(ids[:20])
s.append_turn(ids[:32])
check("routed_through_fn", [x["n"] for x in SEEN] == [20, 12], f"{SEEN}")
check("session_kwargs_forwarded",
      all(x["chunk"] == 64 and x["long_chunk"] == 32 and x["clear"] == 2
          for x in SEEN), f"{SEEN}")
check("no_invented_kwargs",
      all(x["fence"] == 2 for x in SEEN), "driver default untouched")
check("delta_exact", s.offset == 32 and len(s.tokens) == 32, f"off={s.offset}")

# --- 5. chunk sizes follow plan_step when long_chunk is unset -------------
SEEN.clear()
s2 = SC.SessionCache(model, max_seq_len=512, prefill_fn=SC.chunked_prefill,
                     chunk=16, long_chunk=8, long_threshold=100)
s2.append_turn(np.arange(120, dtype=np.int64))
check("chunked_prefill_plan", SEEN == [], "chunked_prefill records nothing")
check("chunked_prefill_ran", s2.offset == 120, f"off={s2.offset}")
check("plan_step_switch", SC.plan_step(120, 10, chunk=16, long_chunk=8,
                                       long_threshold=100) == 8
      and SC.plan_step(96, 10, chunk=16, long_chunk=8, long_threshold=100) == 10)

print(f"[p66] {'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
sys.exit(1 if FAIL else 0)
