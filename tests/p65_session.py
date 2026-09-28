#!/usr/bin/env python3
"""p65 -- single-node (m4-1/m4-2) session/prefix KV reuse correctness + speed.

Builds a DSv4.1 layer SUBSET covering every cache kind (window-only, ratio-2
kv+index source, ratio-2 consumer, ratio-1 candidate source, index source
without keys, DSpark taps) and checks that a session-continuation prefill
(delta rows only) equals a fresh prefill.

Oracles -- three, all needed
----------------------------

* **replay (bitwise)** -- every ``model(...)`` chunk a session feeds is recorded
  (offset, rows, tokens); a cold cache is then fed the recorded chunks. Same
  ops, same order, so this must be bitwise identical in the cache buffers, the
  logits and the hidden taps. Arms that rewound replay *without* the discarded
  segment, so bitwise equality proves the rewind left no trace.
* **append-only twin (bitwise)** -- two sessions reach the same history, one via
  rewinds, one by pure appends. Bitwise equality is the direct statement
  "reuse == no reuse".
* **cold full prefill (cos)** -- the session vs a plain cold prefill at the
  driver's chunking. The body is chunk-shape sensitive (measured in exo phase
  3, independent of session reuse: a position's residual depends on the chunk
  it was computed in), so this is reported as cos / max|d| / top-1, and arms
  whose chunk plan matches a cold prefill assert bitwise separately.

Note on comparing caches: the oracle must be a cache that never ran PAST the
offset being compared. A forward at offset ``u`` reads window-ring slots of
positions ``u-window..u``; a cache that ran further has already overwritten
some of those with later KV, and a rewind does not restore them (the session
snapshot does). So every state comparison here is against a replay.

Arms (P65_TEST=all):
  fresh      cold reference prefill + its plan
  turn2      turn 1 = prompt; turn 2 = prompt + reply + new text (reuse case)
  turn2big   same with a multi-chunk delta aligned to cold-prefill chunks
  decode     reply as 1-row appends, half cancelled and redone
  mismatch   interior token rewrite -> boundary rewind + refill
  cancel     partial generation, cancel(), redo the turn
  twin       same final history reached with and without rewinds
  store      SessionStore, two conversations + eviction
  speed      cold vs delta vs pure hit wall time and tokens prefilled
  capacity   oversized turn raises CapacityError

Env: P65_LAYERS=a,b,c P65_LEN=N P65_CHUNK=64 P65_NEW=32 P65_PKG=<worktree>
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("P65_PKG", HOME + "/dsv41-ws/E"))
import mlx.core as mx  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get(
    "P65_LAYERS", "0,1,2,3,20,21,24,25,37,38,39").split(",")]
TEST = os.environ.get("P65_TEST", "all")
CHUNK = int(os.environ.get("P65_CHUNK", "64"))
LEN = int(os.environ.get("P65_LEN", "320"))
NEW = int(os.environ.get("P65_NEW", "32"))
REPLY = int(os.environ.get("P65_REPLY", "24"))

from mlx_lm.models.deepseek_v41 import exl3_build as eb            # noqa: E402
from mlx_lm.models.deepseek_v41 import session_cache as SC         # noqa: E402
try:                                                              # stream C's driver
    from mlx_lm.models.deepseek_v41 import prefill as _PF          # noqa: F401
    DRIVER = "prefill.prefill (fenced)"
except ImportError:
    DRIVER = "session_cache.chunked_prefill (plain)"

FAIL = []


def log(*a):
    print("[p65]", *a, flush=True)


def check(name, ok, detail=""):
    print(f"[p65] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAIL.append(name)


if os.environ.get("P65_LOGIC") == "1":
    # bookkeeping-only mode: no model build, no GPU (the block below is skipped)
    LOGIC = True
else:
    LOGIC = False


# --- record every chunk the body is fed (driver-agnostic) -------------------
FED = []                       # [(offset, tokens np [n])]
_ORIG_CALL = None
TAPS = []

if not LOGIC:
    t0 = time.time()
    model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0,
                              world=2, group=None)
    model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
    TAPS = [L for L in model.args.dspark_target_layer_ids if L in LAYERS]
    log(f"built layers={LAYERS} taps={TAPS} in {time.time()-t0:.0f}s "
        f"active={mx.get_active_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")
    _ORIG_CALL = type(model).__call__

    def _rec_call(self, input_ids, cache, **kw):
        FED.append((int(cache.offset),
                    np.array(input_ids)[0].astype(np.int64).copy()))
        return _ORIG_CALL(self, input_ids, cache, **kw)

    type(model).__call__ = _rec_call


def mark():
    return len(FED)


def fed_since(m):
    return list(FED[m:])


def replay(fed, taps=False, max_seq_len=4096):
    """Cold cache fed exactly the recorded chunks -> (output, cache)."""
    c = model.make_cache(1, max_seq_len=max_seq_len)
    out = None
    n_total = max(pos + len(t) for pos, t in fed)
    for pos, toks in fed:
        assert c.offset == pos, f"replay out of order: {pos} != {c.offset}"
        last = pos + len(toks) == n_total
        r = _ORIG_CALL(model, mx.array(toks[None].astype(np.int32)), c,
                       last_logit_only=True, return_taps=taps and last, argmax=False)
        if last:
            out = r
        mx.eval(*(r if isinstance(r, tuple) else (r,)))
    return out, c


def replay_final(fed, taps=False, max_seq_len=4096):
    """Replay only the chunks that survive to the final history (drops what a
    later rewind superseded), on a cold cache."""
    keep, need = [], max(pos + len(t) for pos, t in fed)
    for pos, toks in reversed(fed):
        if pos + len(toks) == need:
            keep.append((pos, toks))
            need = pos
    assert need == 0, f"final-history replay does not reach 0 (stopped at {need})"
    return replay(list(reversed(keep)), taps=taps, max_seq_len=max_seq_len)


PROMPT = json.load(open(HOME + "/p30_prompt_ids.json"))
CORPUS = (PROMPT * 16)[:len(PROMPT) * 16]
assert LEN + REPLY + NEW + 256 <= len(CORPUS)
IDS = np.asarray(CORPUS[:LEN + REPLY + NEW], dtype=np.int64)


def make_session(**kw):
    kw.setdefault("chunk", CHUNK)
    kw.setdefault("long_chunk", CHUNK)
    return SC.SessionCache(model, max_seq_len=kw.pop("max_seq_len", 4096), **kw)


def cold_full(ids, taps=False):
    """Cold prefill at the driver's default chunking (the cos oracle)."""
    c = model.make_cache(1, max_seq_len=4096)
    return SC.chunked_prefill(model, ids, c, chunk=CHUNK, long_chunk=CHUNK,
                              clear_cache_every=0, return_taps=taps), c


# ---------------------------------------------------------------- compare

def state_of(cache, upto):
    """{name: array} of every row/slot a forward at ``upto`` can read."""
    out = {}
    for i, lc in enumerate(cache.layers):
        if i not in LAYERS:
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
    return out


def diff_state(a, b):
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append(f"{k}:missing({k in a}/{k in b})")
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            bad.append(f"{k}:shape {x.shape}!={y.shape}")
            continue
        if k == "engram_ids":
            if not bool(mx.array_equal(x, y)):
                bad.append(f"{k}:ids differ")
            continue
        if bool(mx.array_equal(x, y)):
            continue
        xf, yf = x.astype(mx.float32), y.astype(mx.float32)
        d = mx.max(mx.abs(xf - yf)).item() if xf.size else 0.0
        nx, ny = mx.linalg.norm(xf.reshape(-1)), mx.linalg.norm(yf.reshape(-1))
        c = ((xf.reshape(-1) * yf.reshape(-1)).sum() / (nx * ny + 1e-30)).item() \
            if xf.size else 1.0
        bad.append(f"{k}:max|d|={d:.3g} cos={c:.9f}")
    return bad


def state_check(name, cache_a, cache_b):
    bad = diff_state(state_of(cache_a, cache_a.offset), state_of(cache_b, cache_b.offset))
    check(name, not bad, f"bad={bad[:8]}")


def _f64(x):
    if isinstance(x, mx.array):
        return np.array(x.astype(mx.float32), dtype=np.float32).astype(np.float64)
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def cos_of(ref, cur):
    """(cos, max|d|, top1 agree) in float64: identical arrays give exactly 1.0."""
    r, c = _f64(ref).reshape(-1), _f64(cur).reshape(-1)
    if r.shape != c.shape:
        return None, None, None
    d = float(np.max(np.abs(r - c)))
    n = float(np.linalg.norm(r) * np.linalg.norm(c))
    return (float((r * c).sum() / n) if n > 0 else 1.0), d, \
        float(np.argmax(r) == np.argmax(c))


def unwrap(out):
    return out if isinstance(out, tuple) else (out, None)


def compare_exact(name, ref, cur, note=""):
    cos, d, agree = cos_of(unwrap(ref)[0], unwrap(cur)[0])
    if cos is None:
        check(name, False, "shape mismatch")
        return
    check(name, d == 0.0,
          f"exact={d == 0.0} max|d|={d:.3g} cos={cos:.12f} top1_agree={agree:.0f} {note}")


def compare_cos(name, ref, cur, bar=0.9999, note=""):
    cos, d, agree = cos_of(unwrap(ref)[0], unwrap(cur)[0])
    if cos is None:
        check(name, False, "shape mismatch")
        return
    check(name, cos >= bar,
          f"exact={d == 0.0} max|d|={d:.3g} cos={cos:.9f} top1_agree={agree:.0f} {note}")


def compare_taps(name, ref, cur, bar=0.999):
    worst, bad = 1.0, []
    ctap, rtap = unwrap(cur)[1] or {}, unwrap(ref)[1] or {}
    for L in sorted(set(rtap) | set(ctap)):
        cos, d, _ = cos_of(rtap.get(L), ctap.get(L))
        if cos is None:
            bad.append(f"L{L}:missing")
            continue
        worst = min(worst, cos)
        if d != 0.0 and cos < bar:
            bad.append(f"L{L}:cos={cos:.9f} max|d|={d:.3g}")
    check(name, not bad, f"min_cos={worst:.9f} bad={bad}")


def tap_check(name, ref, cur):
    """Taps from a bitwise-equivalent prefill -> exact."""
    rtap, ctap = unwrap(ref)[1] or {}, unwrap(cur)[1] or {}
    worst, bad = 1.0, []
    for L in sorted(set(rtap) | set(ctap)):
        cos, d, _ = cos_of(rtap.get(L), ctap.get(L))
        if cos is None:
            bad.append(f"L{L}:missing")
            continue
        worst = min(worst, cos)
        if d != 0.0:
            bad.append(f"L{L}:max|d|={d:.3g} cos={cos:.9f}")
    check(name, not bad, f"min_cos={worst:.9f} bad={bad}")


# ---------------------------------------------------------------- arms

def run():
    log(f"driver={DRIVER}")
    log(f"ids={IDS.shape} chunk={CHUNK} reply={REPLY} new={NEW} test={TEST}")
    m0 = mark()
    t0 = time.perf_counter()
    REF, REFC = cold_full(IDS, taps=bool(TAPS))
    REF_PLAN = fed_since(m0)
    log(f"reference: cold full prefill {len(IDS)} tok {time.perf_counter()-t0:.2f}s "
        f"plan={[(p, len(t)) for p, t in REF_PLAN]} peak={mx.get_peak_memory()/1e9:.1f}GB")
    last = None

    if TEST in ("all", "turn2"):
        s = make_session(); last = s
        m = mark(); s.append_turn(IDS[:LEN]); fed1 = fed_since(m)
        m = mark(); r2 = s.append_turn(IDS, return_taps=bool(TAPS)); fed2 = fed_since(m)
        log(f"turn2: {r2}  chunks(turn2)={[(p, len(t)) for p, t in fed2]}")
        check("turn2.delta", r2.tokens_prefilled == REPLY + NEW,
              f"prefilled={r2.tokens_prefilled} want={REPLY+NEW}")
        check("turn2.reuse", r2.tokens_reused == LEN and r2.rolled_back_from is None,
              f"reused={r2.tokens_reused} rollback={r2.rolled_back_from}")
        rep, repc = replay(fed1 + fed2, taps=bool(TAPS))
        compare_exact("turn2.replay_bitwise", rep, r2.logits,
                      note="(delta prefill vs cold cache fed the same chunks)")
        state_check("turn2.state_bitwise", s.cache, repc)
        compare_cos("turn2.vs_cold_full", REF, r2.logits)
        if TAPS:
            tap_check("turn2.taps_bitwise", rep, r2.logits)
        r3 = s.append_turn(np.concatenate([IDS, [11, 12, 13]]))
        check("turn3.delta3", r3.tokens_prefilled == 3, f"{r3}")

    if TEST in ("all", "turn2big"):
        nbig = 128
        ids_big = np.asarray(CORPUS[:LEN + nbig])
        s = make_session(); last = s
        m = mark(); s.append_turn(ids_big[:LEN]); fed1 = fed_since(m)
        m = mark(); r2 = s.append_turn(ids_big, return_taps=bool(TAPS)); fed2 = fed_since(m)
        log(f"turn2big: {r2} chunks={[(p, len(t)) for p, t in fed2]}")
        check("turn2big.delta", r2.tokens_prefilled == nbig,
              f"prefilled={r2.tokens_prefilled}")
        rep, repc = replay(fed1 + fed2, taps=bool(TAPS))
        compare_exact("turn2big.replay_bitwise", rep, r2.logits)
        state_check("turn2big.state_bitwise", s.cache, repc)
        coldb, coldbc = cold_full(ids_big, taps=bool(TAPS))
        compare_exact("turn2big.vs_cold_full", coldb, r2.logits)
        state_check("turn2big.state_vs_cold_full", s.cache, coldbc)

    if TEST in ("all", "decode"):
        s = make_session(); last = s
        m = mark(); s.append_turn(IDS[:LEN]); fed1 = fed_since(m)
        m = mark()
        for i in range(REPLY):
            s.append_tokens(IDS[LEN + i:LEN + i + 1])
        fed_reply = fed_since(m)
        s.snapshot()                                   # checkpoint at the reply end
        s.append_tokens(IDS[LEN + REPLY:LEN + REPLY + REPLY // 2])
        dropped = s.cancel()
        check("decode.cancel", s.offset == LEN + REPLY and dropped == REPLY // 2,
              f"offset={s.offset} dropped={dropped} bnd={s.boundaries}")
        rep_ok, repc_ok = replay(fed1 + fed_reply)
        state_check("decode.cancel_state_bitwise", s.cache, repc_ok)
        m = mark()
        for i in range(REPLY // 2):
            s.append_tokens(IDS[LEN + REPLY + i:LEN + REPLY + i + 1])
        fed_redo = fed_since(m)
        m = mark(); r = s.append_turn(IDS); fed2 = fed_since(m)
        log(f"decode-turn2: {r} chunks={[(p, len(t)) for p, t in fed2]}")
        check("decode.delta", r.tokens_prefilled == NEW - REPLY // 2,
              f"prefilled={r.tokens_prefilled} want={NEW - REPLY//2}")
        rep, repc = replay(fed1 + fed_reply + fed_redo + fed2)
        compare_exact("decode.replay_bitwise", rep, r.logits)
        state_check("decode.state_bitwise", s.cache, repc)
        cos, d, agree = cos_of(unwrap(REF)[0], unwrap(r.logits)[0])
        log(f"decode vs cold full prefill: cos={cos:.6f} max|d|={d:.3g} top1={agree:.0f} "
            f"(different chunk plan: informational)")

    if TEST in ("all", "mismatch"):
        s = make_session(); last = s
        m = mark(); s.append_turn(IDS[:LEN]); fed1 = fed_since(m)
        s.append_turn(IDS[:LEN + REPLY])               # this segment gets discarded
        mut = IDS.copy()
        mut[LEN + 5] = (int(mut[LEN + 5]) + 997) % 129280            # interior rewrite
        m = mark(); r = s.append_turn(mut, return_taps=bool(TAPS)); fed_refill = fed_since(m)
        log(f"mismatch: {r} chunks={[(p, len(t)) for p, t in fed_refill]}")
        check("mismatch.rewound",
              r.rolled_back_from == LEN + REPLY and r.turn_start == LEN,
              f"rolled={r.rolled_back_from} turn_start={r.turn_start}")
        check("mismatch.refill", r.tokens_prefilled == len(mut) - LEN,
              f"prefilled={r.tokens_prefilled} want={len(mut)-LEN}")
        # replay WITHOUT the discarded segment: bitwise => the rewind is clean
        rep, repc = replay(fed1 + fed_refill, taps=bool(TAPS))
        compare_exact("mismatch.replay_bitwise", rep, r.logits,
                      note="(discarded segment omitted from the replay)")
        state_check("mismatch.state_bitwise", s.cache, repc)
        coldm, coldmc = cold_full(mut, taps=bool(TAPS))
        compare_cos("mismatch.vs_cold_full", coldm, r.logits)
        cont = np.concatenate([mut, [7, 8, 9, 10, 11]])
        m = mark(); r3 = s.append_turn(cont); fed3 = fed_since(m)
        check("mismatch.continue_delta", r3.tokens_prefilled == 5, f"{r3}")
        rep3, repc3 = replay(fed1 + fed_refill + fed3)
        compare_exact("mismatch.continue_replay_bitwise", rep3, r3.logits)
        state_check("mismatch.continue_state_bitwise", s.cache, repc3)
        cos, d, agree = cos_of(unwrap(cold_full(cont)[0])[0], unwrap(r3.logits)[0])
        log(f"mismatch.continue vs cold full prefill: cos={cos:.6f} max|d|={d:.3g} "
            f"top1={agree:.0f}  (5-row tail vs 64-row chunks: chunk-shape only)")

    if TEST in ("all", "cancel"):
        s = make_session(); last = s
        m = mark(); s.append_turn(IDS[:LEN]); fed1 = fed_since(m)
        s.snapshot()
        before = s.offset
        s.append_tokens(IDS[LEN:LEN + 5])                      # a partial turn
        dropped = s.cancel()
        check("cancel.rewind", s.offset == before and dropped == 5,
              f"offset={s.offset} was={before} dropped={dropped} bnd={s.boundaries}")
        rep, repc = replay(fed1)
        state_check("cancel.state_restored_bitwise", s.cache, repc)
        m = mark(); r = s.append_turn(IDS, return_taps=bool(TAPS)); fed2 = fed_since(m)
        check("cancel.redo_delta", r.tokens_prefilled == REPLY + NEW, f"{r}")
        rep2, repc2 = replay(fed1 + fed2, taps=bool(TAPS))
        compare_exact("cancel.replay_bitwise", rep2, r.logits,
                      note="(discarded partial turn omitted from the replay)")
        state_check("cancel.redo_state_bitwise", s.cache, repc2)
        compare_exact("cancel.redo_vs_cold_full", REF, r.logits)
        state_check("cancel.redo_state_vs_cold_full", s.cache, REFC)
        if TAPS:
            tap_check("cancel.redo.taps", rep2, r.logits)

    if TEST in ("all", "twin"):
        # same final history; A reached it with rewinds, B by pure appends
        s_a = make_session(); last = s_a
        s_b = make_session()
        m = mark()
        s_a.append_turn(IDS[:LEN]); s_b.append_turn(IDS[:LEN])
        s_a.append_turn(IDS[:LEN + REPLY])          # A overshoots
        s_a.append_turn(IDS[:LEN + 3])              # A rewinds back to LEN
        s_a.append_turn(IDS[:LEN + 5])
        s_b.append_turn(IDS[:LEN + 3])              # B walks forward
        s_b.append_turn(IDS[:LEN + 5])
        ra = s_a.append_turn(IDS)
        rb = s_b.append_turn(IDS)
        fed_all_a = fed_since(m)
        log(f"twin: a={ra} b={rb} plan={[(p, len(t)) for p, t in fed_all_a]}")
        compare_exact("twin.a_vs_b_bitwise", rb.logits, ra.logits,
                      note="(rewound session vs append-only session)")
        state_check("twin.a_b_state_bitwise", s_a.cache, s_b.cache)
        rep, repc = replay_final(fed_all_a)
        compare_exact("twin.a_replay_bitwise", rep, ra.logits,
                      note="(surviving chunks only: the rewind left no trace)")
        state_check("twin.a_state_bitwise", s_a.cache, repc)

    if TEST in ("all", "store"):
        # sizes aligned to CHUNK so a cold full prefill uses the same plan
        store = SC.SessionStore(model, max_sessions=2, chunk=CHUNK, long_chunk=CHUNK)
        a1, a2 = np.asarray(CORPUS[:192]), np.asarray(CORPUS[:208])
        b1, b2 = np.asarray(CORPUS[300:684]), np.asarray(CORPUS[300:700])
        m = mark(); sa, ra = store.get(a1); fed_a1 = fed_since(m)
        m = mark(); sb, rb = store.get(b1); fed_b1 = fed_since(m)
        check("store.two_sessions", sa is not sb and len(store.keys()) == 2, f"{store}")
        m = mark(); sa2, ra2 = store.get(a2); fed_a2 = fed_since(m)
        m = mark(); sb2, rb2 = store.get(b2); fed_b2 = fed_since(m)
        check("store.same_session", sa is sa2 and sb is sb2, f"{store}")
        check("store.delta16", ra2.tokens_prefilled == 16 and rb2.tokens_prefilled == 16,
              f"a={ra2.tokens_prefilled} b={rb2.tokens_prefilled}")
        rep, repc = replay(fed_a1 + fed_a2)
        compare_exact("store.a_replay_bitwise", rep, ra2.logits)
        state_check("store.a_state_bitwise", sa2.cache, repc)
        rep, repc = replay(fed_b1 + fed_b2)
        compare_exact("store.b_replay_bitwise", rep, rb2.logits)
        compare_exact("store.a_vs_cold_full", cold_full(a2)[0], ra2.logits)
        state_check("store.a_state_vs_cold_full", sa2.cache, cold_full(a2)[1])
        compare_exact("store.b_vs_cold_full", cold_full(b2)[0], rb2.logits)
        state_check("store.b_state_vs_cold_full", sb2.cache, cold_full(b2)[1])
        store.get(np.asarray(CORPUS[500:600]))
        check("store.evict", len(store.keys()) == 2 and store.stats["evicted"] > 0,
              f"keys={[k[:8] for k in store.keys()]} stats={dict(store.stats)}")

    if TEST in ("all", "speed"):
        big = np.asarray(CORPUS[:512])
        big2 = np.asarray(CORPUS[:512 + NEW])
        s = make_session(); last = s
        t = time.perf_counter(); r1 = s.append_turn(big); cold = time.perf_counter() - t
        t = time.perf_counter(); r2 = s.append_turn(big2); warm = time.perf_counter() - t
        t = time.perf_counter(); r2b = s.append_turn(big2); hit = time.perf_counter() - t
        log(f"speed: cold  {r1.tokens_prefilled:4d} tok {cold:6.2f}s "
            f"= {r1.tokens_prefilled/cold:6.1f} tok/s")
        log(f"speed: delta {r2.tokens_prefilled:4d} tok {warm:6.2f}s "
            f"= {r2.tokens_prefilled/max(warm,1e-9):6.1f} tok/s")
        log(f"speed: hit   {r2b.tokens_prefilled:4d} tok {hit:6.3f}s")
        check("speed.delta_is_new", r2.tokens_prefilled == NEW, f"{r2.tokens_prefilled}")
        check("speed.hit_no_work", r2b.tokens_prefilled == 0, f"{r2b.tokens_prefilled}")
        check("speed.wall_speedup", warm < cold,
              f"cold={cold:.2f}s delta={warm:.2f}s ratio={cold/max(warm,1e-9):.1f}x")

    if TEST in ("all", "trunc"):
        # pure truncation (prefix shorter than cache, checkpoint exists at it):
        # no prefill, exact rewind, and the returned output must not be stale
        T1, T2 = 48, 96
        s = make_session(); last = s
        m = mark(); s.append_turn(IDS[:T1]); fed1 = fed_since(m)
        m = mark(); s.append_turn(IDS[:T2]); fed2 = fed_since(m)
        before = s.offset
        r = s.append_turn(IDS[:T1])
        check("trunc.delta0", r.tokens_prefilled == 0 and r.turn_start == T1, f"{r}")
        check("trunc.rewound", r.rolled_back_from == before, f"{r}")
        check("trunc.output_not_stale", r.logits is None, f"logits={r.logits}")
        rep, repc = replay(fed1)
        state_check("trunc.state_bitwise", s.cache, repc)
        m = mark(); r2 = s.append_turn(IDS[:T1 + 16]); fed3 = fed_since(m)
        check("trunc.continue_delta", r2.tokens_prefilled == 16, f"{r2}")
        rep2, repc2 = replay(fed1 + fed3)
        compare_exact("trunc.continue_replay_bitwise", rep2, r2.logits)
        state_check("trunc.continue_state_bitwise", s.cache, repc2)

    if TEST in ("all", "capacity"):
        s = make_session(max_seq_len=64); last = s
        try:
            s.append_turn(np.asarray(CORPUS[:100]))
            check("capacity.raises", False, "no CapacityError")
        except SC.CapacityError as e:
            check("capacity.raises", True, str(e)[:70])

    if last is not None:
        log(f"session: {last.summary()}")
    log(f"peak={mx.get_peak_memory()/1e9:.1f}GB active={mx.get_active_memory()/1e9:.1f}GB")
    log("P65_FAILED=" + (",".join(FAIL) if FAIL else "none"))
    log("P65_DONE")


# ---------------------------------------------------------------- logic (no GPU)

def logic_tests():
    """Bookkeeping checks on the real classes with stub weights (no model needed)."""
    from mlx_lm.models.deepseek_v41.config import ModelArgs
    from mlx_lm.models.deepseek_v41.engram import EngramHasher

    class StubLayers:
        """Duck-typed ModelCache: real LayerCache/CompressorState semantics."""

        def __init__(self, n=4, window=8, ratio=2, max_seq_len=256):
            self.max_seq_len = max_seq_len
            self.offset = 0
            from mlx_lm.models.deepseek_v41.cache import LayerCache
            a = ModelArgs(window_size=window, compress_ratios=(ratio,) * n,
                          kv_source_layers=(0,), index_source_layers=(0, 1),
                          engram_layer_ids=())
            self.layers = [LayerCache(1, a, i, max_seq_len) for i in range(n)]
            self.engram_ids = None

        def advance(self, n):
            self.offset += n

    class StubModel:
        def __init__(self, **kw):
            self.args = ModelArgs()
            self.kw = kw
            self.made = []

        def make_cache(self, bsz=1, max_seq_len=None, dtype=None):
            c = StubLayers(max_seq_len=max_seq_len or 256)
            self.made.append(c)
            return c

    def stub_prefill(model, ids, cache, *, argmax=False, return_taps=False, **kw):
        n = len(ids)                                    # pretend each row writes
        for lc in cache.layers:
            if lc.comp_kv is not None:                  # kv source only
                pos = cache.offset
                lc.comp_kv[0, pos // lc.ratio: (pos + n) // lc.ratio] = 1.0 + pos
                if lc.index_k is not None:
                    lc.index_k[0, pos // lc.ratio: (pos + n) // lc.ratio] = 2.0 + pos
                cs = lc.comp_state
                for i in range(n):                      # per-token carry evolution
                    slot = (pos + i) % lc.ratio
                    cs.kv_state[0, slot] = 3.0 + pos + i
                    cs.score_state[0, slot] = 4.0 + pos + i
            w = lc.window
            for i in range(cache.offset, cache.offset + n):
                lc.win_kv[0, i % w] = 5.0 + i           # ring: slot = pos % window
        cache.offset += n
        return mx.array([[float(n)]])

    model = StubModel()
    ids = np.arange(40, dtype=np.int64)

    # 1) fresh vs reused: same state, same offsets, delta-only prefill
    s = SC.SessionCache(model, max_seq_len=256, chunk=8, long_chunk=8,
                        prefill_fn=stub_prefill)
    r1 = s.append_turn(ids[:20])
    r2 = s.append_turn(ids[:32])
    check("logic.turn1", r1.tokens_prefilled == 20 and r1.tokens_reused == 0, f"{r1}")
    check("logic.delta", r2.tokens_prefilled == 12 and r2.tokens_reused == 20, f"{r2}")
    check("logic.offset", s.offset == 32 and len(s.tokens) == 32, f"off={s.offset}")

    # 2) replay oracle: a cold cache fed the same rows must match exactly
    fresh = model.make_cache(1, max_seq_len=256)
    stub_prefill(model, ids[:20], fresh)
    stub_prefill(model, ids[20:32], fresh)
    s_state = SC.state_of if hasattr(SC, "state_of") else None
    bad = []
    for i, (a, b) in enumerate(zip(s.cache.layers, fresh.layers)):
        for nm in ("win_kv", "comp_kv", "index_k"):
            x, y = getattr(a, nm), getattr(b, nm)
            if x is None or y is None:
                continue
            if not bool(mx.array_equal(x, y)):
                bad.append(f"L{i}.{nm}")
        if a.comp_state is not None:
            for nm in ("kv_state", "score_state"):
                if not bool(mx.array_equal(getattr(a.comp_state, nm),
                                           getattr(b.comp_state, nm))):
                    bad.append(f"L{i}.{nm}")
    check("logic.state_matches_fresh", not bad, f"bad={bad}")

    # 3) rewind: cancel + redo must be exact
    before = s.offset
    s.append_tokens(ids[32:35])
    dropped = s.cancel()
    check("logic.cancel", s.offset == before and dropped == 3 and len(s.tokens) == 32,
          f"off={s.offset} dropped={dropped}")
    s.append_tokens(ids[32:35])
    r3 = s.append_turn(ids[:40])
    check("logic.after_rewind", r3.tokens_prefilled == 5 and s.offset == 40, f"{r3}")
    fresh2 = model.make_cache(1, max_seq_len=256)
    stub_prefill(model, ids[:20], fresh2)
    stub_prefill(model, ids[20:40], fresh2)
    bad = []
    for i, (a, b) in enumerate(zip(s.cache.layers, fresh2.layers)):
        if not bool(mx.array_equal(a.win_kv, b.win_kv)):
            bad.append(f"L{i}.win_kv")
        if a.comp_state is not None:
            for nm in ("kv_state", "score_state"):
                if not bool(mx.array_equal(getattr(a.comp_state, nm),
                                           getattr(b.comp_state, nm))):
                    bad.append(f"L{i}.{nm}")
    check("logic.rewind_state_matches_fresh", not bad, f"bad={bad}")

    # 4) prefix mismatch: interior rewrite -> boundary rewind + refill
    s = SC.SessionCache(model, max_seq_len=256, chunk=8, long_chunk=8,
                        prefill_fn=stub_prefill)
    s.append_turn(ids[:24])
    mut = ids.copy()
    mut[10] = 999
    r = s.append_turn(mut[:32])
    check("logic.mismatch_rewind", r.rolled_back_from == 24 and r.turn_start == 0,
          f"rolled={r.rolled_back_from} start={r.turn_start}")

    # 5) snapshot memory + boundaries bounded by max_snapshots
    s = SC.SessionCache(model, max_seq_len=256, chunk=8, long_chunk=8,
                        prefill_fn=stub_prefill, max_snapshots=3)
    for n in (8, 16, 24, 32, 40):
        s.append_turn(ids[:n])
    check("logic.snapshot_cap", len(s.boundaries) <= 3, f"bnd={s.boundaries}")

    # 6) exact prefix hit reuses last_output, prefills nothing
    r = s.append_turn(ids[:40])
    check("logic.hit", r.tokens_prefilled == 0 and r.hit and r.logits is not None, f"{r}")

    # 7) SessionStore: two conversations interleaved, eviction, rekey
    store = SC.SessionStore(model, max_sessions=2, chunk=8, long_chunk=8,
                            prefill_fn=stub_prefill)
    a1, a2 = ids[:16].copy(), ids[:24].copy()
    b1, b2 = np.arange(60, 92), np.arange(60, 84)
    sa, ra = store.get(a1)
    sb, rb_ = store.get(b1)
    check("logic.store_two", sa is not sb and len(store.keys()) == 2, f"{store}")
    sa2, ra2 = store.get(a2)
    check("logic.store_same", sa is sa2 and ra2.tokens_prefilled == 8, f"{ra2}")
    # a third conversation evicts the oldest
    store.get(np.arange(200, 240))
    check("logic.store_evict", len(store.keys()) == 2 and store.stats["evicted"] > 0,
          f"stats={dict(store.stats)}")

    # 8) prefix_hash stability + common_prefix_len
    h1 = SC.prefix_hash(ids)
    h2 = SC.prefix_hash(mx.array(ids[None].astype(np.int32)))
    check("logic.hash_stable", h1 == h2 and len(h1) == 32, f"{h1[:12]}")
    check("logic.lcp", SC.common_prefix_len(ids, mut[:32]) == 10
          and SC.common_prefix_len(ids[:5], ids) == 5
          and SC.common_prefix_len([], ids) == 0)
    check("logic.plan_step", SC.plan_step(100, 50, chunk=32, long_chunk=8,
                                          long_threshold=64) == 8
          and SC.plan_step(10, 50, chunk=32, long_chunk=8, long_threshold=64) == 32
          and SC.plan_step(0, 5, chunk=32, long_chunk=8, long_threshold=64) == 5)

    # 9) CapacityError
    s = SC.SessionCache(model, max_seq_len=16, chunk=8, long_chunk=8,
                        prefill_fn=stub_prefill)
    try:
        s.append_turn(np.arange(40))
        check("logic.capacity", False, "no CapacityError")
    except SC.CapacityError:
        check("logic.capacity", True)

    # 10) snapshot/restore round-trip on the carry + ring, exactly as designed
    s = SC.SessionCache(model, max_seq_len=256, chunk=8, long_chunk=8,
                        prefill_fn=stub_prefill)
    s.append_turn(ids[:21])                       # odd length: carry m = 1
    snap = s.snapshot()
    s.append_tokens(ids[21:27])
    s.rewind(21)
    ok = all(bool(mx.array_equal(lc.win_kv, r)) for lc, r in
             zip(s.cache.layers, snap.rings))
    bad = []
    for lc, car in zip(s.cache.layers, snap.carries):
        if lc.comp_state is None or car is None:
            continue
        m = car[2]
        if m and not bool(mx.array_equal(lc.comp_state.kv_state[:, :m], car[0])):
            bad.append("kv")
        if m and not bool(mx.array_equal(lc.comp_state.score_state[:, :m], car[1])):
            bad.append("sc")
    check("logic.rewind_restores_ring", ok)
    check("logic.rewind_restores_carry", not bad, f"bad={bad}")
    check("logic.carry_canonical", all(
        lc.comp_state is None or bool(mx.array_equal(
            lc.comp_state.kv_state[:, 21 % lc.ratio:],
            mx.zeros((1, lc.ratio - 21 % lc.ratio, lc.comp_state.kv_state.shape[-1]))))
        for lc in s.cache.layers), "rows [m, ratio) zeroed on restore")


if __name__ == "__main__":
    if os.environ.get("P65_LOGIC") == "1":
        logic_tests()
        log("P65_FAILED=" + (",".join(FAIL) if FAIL else "none"))
        log("P65_LOGIC_DONE")
        sys.exit(1 if FAIL else 0)
    run()
