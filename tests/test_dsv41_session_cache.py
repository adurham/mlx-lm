# LOCAL ADDITION (not vendored). Offline bookkeeping tests for session KV reuse.
"""Standalone correctness tests for ``deepseek_v41.session_cache``.

No checkpoint, no GPU, no model build: a stub model + the real
``LayerCache`` / ``CompressorState`` classes exercise prefix matching, delta
prefill, boundary rewind, cancel, snapshot/restore and the store. The replayed
cache is the oracle -- same op sequence, so bitwise equality is the pass bar.

Usage::

    python tests/test_dsv41_session_cache.py      # exit 0 iff all pass
"""

from __future__ import annotations

import sys

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41.cache import LayerCache
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41 import session_cache as SC

FAIL: list = []


def check(name, ok, detail=""):
    print(f"[session] {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAIL.append(name)


WINDOW, RATIO, NLAYERS = 8, 2, 4


class StubCache:
    """Duck-typed ``ModelCache`` with the real per-layer classes."""

    def __init__(self, max_seq_len=256, engram=False):
        self.max_seq_len = max_seq_len
        self.offset = 0
        args = ModelArgs(window_size=WINDOW, compress_ratios=(RATIO,) * NLAYERS,
                         kv_source_layers=(0,), index_source_layers=(0, 1),
                         engram_layer_ids=(1,) if engram else ())
        self.layers = [LayerCache(1, args, i, max_seq_len) for i in range(NLAYERS)]
        self.engram_ids = (np.zeros((1, max_seq_len), dtype=np.int64) if engram
                           else None)


class StubModel:
    def __init__(self, engram=False):
        self.args = ModelArgs()
        self.engram = engram

    def make_cache(self, bsz=1, max_seq_len=None, dtype=None):
        return StubCache(max_seq_len=max_seq_len or 256, engram=self.engram)


def stub_prefill(model, ids, cache, *, argmax=False, return_taps=False, **kw):
    """Write deterministic, position-derived values into every buffer.

    Values depend only on the absolute position / group index, never on the
    chunk boundaries -- the real prefill has that property, and the replay
    oracle here relies on it.
    """
    n = len(ids)
    pos = cache.offset
    g0, g1 = pos // RATIO, (pos + n) // RATIO
    for lc in cache.layers:
        if lc.comp_kv is not None:                      # kv source only
            lc.comp_kv[0, g0:g1] = 1.0 + np.arange(g0, g1)[:, None]
            if lc.index_k is not None:
                lc.index_k[0, g0:g1] = 2.0 + np.arange(g0, g1)[:, None]
            cs = lc.comp_state
            for i in range(n):                          # per-token carry evolution
                slot = (pos + i) % RATIO
                cs.kv_state[0, slot] = 3.0 + pos + i
                cs.score_state[0, slot] = 4.0 + pos + i
        for i in range(pos, pos + n):
            lc.win_kv[0, i % WINDOW] = 5.0 + i
    if cache.engram_ids is not None:
        cache.engram_ids[0, pos:pos + n] = np.asarray(ids) + 7
    cache.offset += n
    return mx.array([[float(n)]])


def state_diff(a, b):
    """Difference of every buffer a forward at this offset can READ.

    The window ring is compared over all slots (a fresh replay writes the same
    positions), the carry only over its live rows ``[:offset % ratio]`` -- that
    is the only part the compressor reads; the tail is canonicalized on rewind
    but a stub prefill may have left arbitrary values there.
    """
    bad = []
    assert a.offset == b.offset
    m = a.offset % RATIO
    ng = a.offset // RATIO                       # complete groups readable at this offset
    for i, (x, y) in enumerate(zip(a.layers, b.layers)):
        if not bool(mx.array_equal(x.win_kv, y.win_kv)):
            bad.append(f"L{i}.win_kv")
        for nm in ("comp_kv", "index_k"):        # rows above ng are never read
            u, v = getattr(x, nm), getattr(y, nm)
            if u is None or v is None:
                continue
            if not bool(mx.array_equal(u[:, :ng], v[:, :ng])):
                bad.append(f"L{i}.{nm}[:{ng}]")
        if x.comp_state is not None and m:
            for nm in ("kv_state", "score_state"):
                if not bool(mx.array_equal(getattr(x.comp_state, nm)[:, :m],
                                           getattr(y.comp_state, nm)[:, :m])):
                    bad.append(f"L{i}.{nm}[:{m}]")
    return bad


def feed(model, seqs):
    """Cold cache fed ``seqs`` in order (the oracle)."""
    c = model.make_cache(1, max_seq_len=256)
    for s in seqs:
        stub_prefill(model, np.asarray(s), c)
    return c


def main() -> int:
    model = StubModel()
    ids = np.arange(40, dtype=np.int64)

    # delta prefill + exact state
    s = SC.SessionCache(model, max_seq_len=256, prefill_fn=stub_prefill)
    r1 = s.append_turn(ids[:20])
    r2 = s.append_turn(ids[:32])
    check("turn1_cold", r1.tokens_prefilled == 20 and r1.tokens_reused == 0, f"{r1}")
    check("turn2_delta", r2.tokens_prefilled == 12 and r2.tokens_reused == 20, f"{r2}")
    check("state_matches_fresh", not state_diff(s.cache, feed(model, [ids[:20],
                                                                     ids[20:32]])))

    # pure hit
    r3 = s.append_turn(ids[:32])
    check("hit_no_work", r3.tokens_prefilled == 0 and r3.hit, f"{r3}")

    # cancel + redo
    s.append_tokens(ids[32:35])
    dropped = s.cancel()
    check("cancel", dropped == 3 and s.offset == 32 and len(s.tokens) == 32,
          f"dropped={dropped} off={s.offset}")
    s.append_tokens(ids[32:35])
    s.append_turn(ids[:40])
    check("cancel_redo_state", not state_diff(s.cache, feed(model, [ids[:20],
                                                                    ids[20:40]])))
    check("cancel_redo_offsets", s.offset == 40 and len(s.tokens) == 40)

    # prefix mismatch: rewind to the newest boundary below the divergence
    s = SC.SessionCache(model, max_seq_len=256, prefill_fn=stub_prefill)
    s.append_turn(ids[:24])
    mut = ids.copy()
    mut[10] = 999
    r = s.append_turn(mut[:32])
    check("mismatch_rewind", r.rolled_back_from == 24 and r.turn_start == 0,
          f"rolled={r.rolled_back_from} start={r.turn_start}")
    check("mismatch_state", not state_diff(s.cache, feed(model, [mut[:32]])))

    # twin: rewound history == append-only history, bitwise
    s_a = SC.SessionCache(model, max_seq_len=256, prefill_fn=stub_prefill)
    s_b = SC.SessionCache(model, max_seq_len=256, prefill_fn=stub_prefill)
    s_a.append_turn(ids[:20]); s_b.append_turn(ids[:20])
    s_a.append_turn(ids[:32]); s_a.append_turn(ids[:23])
    s_b.append_turn(ids[:23])
    s_a.append_turn(ids[:40]); s_b.append_turn(ids[:40])
    check("twin_state", not state_diff(s_a.cache, s_b.cache))
    check("twin_output", bool(mx.array_equal(s_a.last_output, s_b.last_output)))

    # pure truncation (no delta, a checkpoint exists at the target): the state
    # rewinds and the output is None -- not the stale output of dropped rows
    s = SC.SessionCache(model, max_seq_len=256, prefill_fn=stub_prefill)
    s.append_turn(ids[:20])                      # checkpoint at 20
    s.append_turn(ids[:32])                      # extends it; checkpoint at 32
    before_out = s.last_output
    r = s.append_turn(ids[:20])
    check("truncation_rewinds", r.tokens_prefilled == 0 and r.rolled_back_from == 32
          and r.turn_start == 20 and r.logits is None, f"{r}")
    check("truncation_state", not state_diff(s.cache, feed(model, [ids[:20]])))
    check("truncation_output_not_stale", r.logits is not before_out)
    # with no checkpoint at the target it rewinds to the boundary below and refills
    s2 = SC.SessionCache(model, max_seq_len=256, prefill_fn=stub_prefill)
    s2.append_turn(ids[:32])
    r2 = s2.append_turn(ids[:20])
    check("truncation_refills_to_boundary",
          r2.tokens_prefilled == 20 and r2.turn_start == 0 and r2.rolled_back_from == 32,
          f"{r2}")

    # snapshot cap keeps memory bounded
    s = SC.SessionCache(model, max_seq_len=256, prefill_fn=stub_prefill,
                        max_snapshots=3)
    for n in (8, 16, 24, 32, 40):
        s.append_turn(ids[:n])
    check("snapshot_cap", len(s.boundaries) <= 3, f"{s.boundaries}")

    # snapshot round-trip: ring + carry restored exactly, tail rows canonical
    s = SC.SessionCache(model, max_seq_len=256, prefill_fn=stub_prefill)
    s.append_turn(ids[:21])                       # odd length -> carry m = 1
    snap = s.snapshot()
    s.append_tokens(ids[21:27])
    s.rewind(21)
    check("rewind_ring", all(bool(mx.array_equal(lc.win_kv, r))
                             for lc, r in zip(s.cache.layers, snap.rings)))
    bad = []
    for lc, car in zip(s.cache.layers, snap.carries):
        if lc.comp_state is None or car is None or car[2] == 0:
            continue
        m = car[2]
        if not bool(mx.array_equal(lc.comp_state.kv_state[:, :m], car[0])):
            bad.append("kv")
        if not bool(mx.array_equal(lc.comp_state.score_state[:, :m], car[1])):
            bad.append("sr")
    check("rewind_carry", not bad, f"{bad}")

    # engram id history is carried through delta prefills
    em = StubModel(engram=True)
    s = SC.SessionCache(em, max_seq_len=256, prefill_fn=stub_prefill)
    s.append_turn(ids[:20]); s.append_turn(ids[:32])
    ref = feed(em, [ids[:20], ids[20:32]])
    check("engram_ids", bool(mx.array_equal(
        mx.array(s.cache.engram_ids[:, :32]), mx.array(ref.engram_ids[:, :32]))))

    # store: two conversations, rekey, eviction
    store = SC.SessionStore(model, max_sessions=2, prefill_fn=stub_prefill)
    sa, _ = store.get(ids[:16])
    sb, _ = store.get(np.arange(60, 92))
    check("store_two", sa is not sb and len(store.keys()) == 2)
    sa2, ra = store.get(ids[:24])
    check("store_rekey", sa is sa2 and ra.tokens_prefilled == 8, f"{ra}")
    store.get(np.arange(200, 240))
    check("store_evict", len(store.keys()) == 2 and store.stats["evicted"] > 0,
          f"{dict(store.stats)}")

    # helpers
    check("prefix_hash_stable", SC.prefix_hash(ids) ==
          SC.prefix_hash(mx.array(ids[None].astype(np.int32))))
    check("common_prefix_len", SC.common_prefix_len(ids, mut[:32]) == 10
          and SC.common_prefix_len(ids[:5], ids) == 5
          and SC.common_prefix_len([], ids) == 0)
    check("plan_step", SC.plan_step(100, 50, chunk=32, long_chunk=8,
                                    long_threshold=64) == 8
          and SC.plan_step(10, 50, chunk=32, long_chunk=8, long_threshold=64) == 32
          and SC.plan_step(0, 5, chunk=32, long_chunk=8, long_threshold=64) == 5)

    # errors
    s = SC.SessionCache(model, max_seq_len=16, prefill_fn=stub_prefill)
    try:
        s.append_turn(np.arange(40))
        check("capacity_error", False, "no CapacityError")
    except SC.CapacityError:
        check("capacity_error", True)
    try:
        s2 = SC.SessionCache(model, max_seq_len=64, prefill_fn=stub_prefill)
        s2.rewind(7)
        check("rollback_error", False, "no RollbackError")
    except SC.RollbackError:
        check("rollback_error", True)

    print(f"[session] {'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
