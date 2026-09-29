#!/usr/bin/env python3
"""pU7 -- 2-rank (one node): instrumentation identity + draft collectives. (U)

Runs TWO ranks of the ring backend on ONE node (the single-node rule: two
processes, one job slot, one lockf) at world 2 with layers 20,37,38,39 -- enough
for the DSpark taps (37,38,39) and their kv source (20), with the draft head
sharded exactly as ``build_mtp`` does at world > 1. At this size the model is
too shallow for real text, which is fine: what is under test is the ROUND
MACHINERY and the COLLECTIVE STRUCTURE, neither of which needs acceptance > 0.

Runs
  P  production call (spec.generate, no log)
  L  same call, log installed            -> must give P's tokens
  I  same call, log + the extra draft force (iso=1) -> must give P's tokens
  L2 L again                              -> determinism
Reported
  per-round terms + unattributed for L
  every all_sum call, tagged by phase of the round, plus payload shapes
  all_sum calls made inside head.draft(), and calls per draft token
  per-round dd / tg / committed ids, hashed and summed over ranks (identity)

Env: PU7_RANK, PU7_HOSTFILE, PU7_LAYERS, PU7_GAMMA, PU7_ROUNDS,
     PU7_BODY_EXPERTS, PU7_HEAD_EXPERTS, PU7_PKG.
NOTE: ``mx.eval`` is MLX's graph evaluation, not python's builtin.
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
PKG = os.environ.get("PU7_PKG", HOME + "/dsv41-ws2/U")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "bench"))

import mlx.core as mx  # noqa: E402

import pU_cap  # noqa: E402

_b = os.environ.get("PU7_BODY_EXPERTS")
_h = os.environ.get("PU7_HEAD_EXPERTS")
pU_cap.apply_caps(int(_b) if _b else None, int(_h) if _h else None)

HF = os.environ.get("PU7_HOSTFILE", "")
group = mx.distributed.init(strict=True, backend="ring") if HF else None
RANK = group.rank() if group else 0
WORLD = group.size() if group else 1
log = lambda *a: print(f"[pU7 r{RANK}]", *a, flush=True)

# ---- collective counter (installed before the model is built) ---------------
PHASE = ["build"]
COUNTS: dict[str, int] = {}
SHAPES: dict[str, list] = {}
_real_all_sum = mx.distributed.all_sum


def counted(x, group=None, **kw):
    p = PHASE[0]
    COUNTS[p] = COUNTS.get(p, 0) + 1
    try:
        SHAPES[p] = list(x.shape)
    except Exception:
        pass
    return _real_all_sum(x, group=group, **kw)


mx.distributed.all_sum = counted

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PU7_LAYERS", "20,37,38,39").split(",")]
GAMMA = int(os.environ.get("PU7_GAMMA", "3"))
ROUNDS = int(os.environ.get("PU7_ROUNDS", "4"))

t0 = time.time()
model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=RANK,
                          world=WORLD, group=group)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ck = Exl3Checkpoint(MODEL)
head = eb.build_mtp(ck, model.args, rank=RANK, world=WORLD, group=group)
mx.eval(mx.ones(1))
log(f"built layers={LAYERS} world={WORLD} in {time.time() - t0:.0f}s "
    f"active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")
log(f"head.vocab_sharded={getattr(head, 'vocab_sharded', False)} "
    f"markov_width={head.markov_head.weight.shape[0]} of {model.args.vocab_size} "
    f"body_head_sharded={hasattr(model.head, 'combine_argmax')} "
    f"stage0_experts_group={head.stages[0].ffn.group is not None}")

prompt = json.load(open(HOME + "/p30_prompt_ids.json"))[:24]


def call(rows=None, iso=0, budget=ROUNDS * (GAMMA + 1)):
    """One spec.generate run, optionally with the round log installed."""
    SP.set_round_log(rows if rows is not None else None)
    SP._ISOLATE = iso
    PHASE[0] = "generate"
    t = time.perf_counter()
    toks, st = SP.generate(model, head, prompt, budget, gamma=GAMMA, adaptive=False)
    dt = time.perf_counter() - t
    SP.set_round_log(None)
    SP._ISOLATE = 0
    return toks, st, dt


def h(rows):
    b = json.dumps(rows, sort_keys=True)
    return float(sum(b.encode())), len(b)


def rank_check(tag, blob: str):
    """Hash-and-sum over ranks: identical payloads iff sum == hash * world."""
    hv = float(sum(blob.encode()))
    arr = mx.array([hv, float(len(blob))])
    if group is None:
        return True
    tot = _real_all_sum(arr, group=group)
    mx.eval(tot)
    s = float(tot[0].item())
    ok = abs(s - hv * WORLD) < 1
    if RANK == 0:
        log(f"rank-identity {tag}: hash={hv:.0f} len={len(blob)} "
            f"x{WORLD}={hv * WORLD:.0f} summed={s:.0f} -> "
            f"{'IDENTICAL' if ok else 'DIFFERENT'}")
    return ok


toks_p, st_p, dt_p = call(None, 0)
log(f"P prod    : {len(toks_p) - 1} tok, {st_p['rounds']} rounds, "
    f"{dt_p * 1e3 / max(st_p['rounds'], 1):.2f} ms/round, {st_p['tok_s']:.2f} tok/s")

rows_l = []
toks_l, st_l, dt_l = call(rows_l, 0)
log(f"L logged  : {len(toks_l) - 1} tok, {len(rows_l)} rounds, "
    f"{dt_l * 1e3 / max(len(rows_l), 1):.2f} ms/round, "
    f"acc {np.mean([r['accept'] for r in rows_l]):.2f}")
log(f"CHECK P == L  (round log is inert): {toks_p == toks_l}")

rows_i = []
toks_i, st_i, dt_i = call(rows_i, 1)
log(f"I iso=1   : {len(toks_i) - 1} tok, {len(rows_i)} rounds, "
    f"{dt_i * 1e3 / max(len(rows_i), 1):.2f} ms/round, "
    f"acc {np.mean([r['accept'] for r in rows_i]):.2f}")
log(f"CHECK P == I  (eval placement is inert): {toks_p == toks_i}")

toks_l2, st_l2, dt_l2 = call(None, 0)
log(f"CHECK L == L2 (determinism): {toks_l == toks_l2}")

TERMS = ["draft_build", "draft_gpu", "body_build", "wait", "readback",
         "rollback_ctx", "emit"]


def show(tag, rows):
    rows = rows[3:]                    # drop warmup (first-ever-compile) rounds
    if not rows:
        log(f"--- {tag}: too few rounds")
        return
    tot = lambda k: sum(r.get(k, 0.0) for r in rows) / len(rows)
    wall = tot("total")
    log(f"--- {tag}: {len(rows)} rounds (warmup dropped), {wall:.2f} ms/round, "
        f"acc {np.mean([r['accept'] for r in rows]):.2f}")
    s = 0.0
    for k in TERMS:
        v = tot(k)
        s += v
        if abs(v) >= 0.005:
            log(f"    {k:12s} mean {v:8.2f} ms/round")
    log(f"    {'sum':12s} mean {s:8.2f} ms/round  => unattributed "
        f"{s - wall:+.2f} ms/round")


show("L (log, iso=0)", rows_l)
show("I (log, iso=1)", rows_i)

# ---- collectives: rerun a few rounds with phase tags ------------------------
PHASE[0] = "prefill_manual"
cache = model.make_cache(1, max_seq_len=len(prompt) + 128 + ROUNDS * 4)
am, taps = model(mx.array([prompt]), cache, last_logit_only=True,
                 return_taps=True, argmax=True)
mx.eval(am)
TAPS = list(model.args.dspark_target_layer_ids)
PHASE[0] = "append_ctx_manual"
dsc = head.make_cache(1)
head.append_ctx(mx.concatenate([taps[L] for L in TAPS], axis=-1), dsc)
mx.eval([c.win_kv for c in dsc])
nxt = am[:, -1].astype(mx.int32)
mx.eval(nxt)
PHASE[0] = "idle"
pos = cache.offset
per_round_draft_calls = []
per_round = []
for i in range(ROUNDS):
    PHASE[0] = "draft"
    c0 = COUNTS.get("draft", 0)
    d, _ = head.draft(nxt, model.embed, model.head, dsc, width=GAMMA)
    d = d.astype(mx.int32)
    per_round_draft_calls.append(COUNTS.get("draft", 0) - c0)
    vin = mx.concatenate([nxt.reshape(1, 1), d], axis=1)
    PHASE[0] = "verify"
    am, taps = model(vin, cache, return_taps=True, argmax=True)
    mx.eval(am, d)
    tg, dd = np.array(am[0]), np.array(d[0])
    n = 0
    while n < GAMMA and tg[n] == dd[n]:
        n += 1
    new = [int(v) for v in dd[:n]] + [int(tg[n])]
    PHASE[0] = "snap_rollback"
    sn = SP.snap(cache, pos)
    SP.rollback(cache, sn, pos + n + 1, SP.stashes(cache))
    PHASE[0] = "append_ctx"
    head.append_ctx(mx.concatenate([taps[L] for L in TAPS], axis=-1)[:, :n + 1], dsc)
    pos = pos + n + 1
    nxt = mx.array([new[-1]], dtype=mx.int32)
    PHASE[0] = "emit"
    mx.eval(nxt)
    per_round.append([dd.tolist(), tg.tolist(), n])
PHASE[0] = "idle"
census = dict(sorted(COUNTS.items()))
if RANK == 0:
    log(f"all_sum calls by phase: {census}")
    log(f"payload shape last seen: {dict(sorted(SHAPES.items()))}")
    log(f"all_sum inside head.draft(): {per_round_draft_calls} per round; "
        f"gamma={GAMMA} markov steps -> {[c / GAMMA for c in per_round_draft_calls]} "
        f"per draft token")
    log(f"per-round draft/target/acc: {per_round}")

rank_check("draft+verify round payloads", json.dumps(per_round, sort_keys=True))
rank_check("generate tokens", json.dumps({"p": toks_p, "l": toks_l, "i": toks_i}))

log(f"peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU7_DONE")
