#!/usr/bin/env python3
"""pU5 -- bit-identity + the final per-round breakdown, self-consistent geometry.

world=1 (no TP slicing, no collectives needed) so the model is numerically
self-consistent and acceptance is real, unlike a rank-0 slice of world=2.

Arms
  P   production call: spec.generate with NO round log installed
  L   same, log installed (instrumentation active)
  I   same, log installed + the added draft force (iso=1)
  Q   same as P but with a mx.synchronize() before each round (quiesce test)
Checks
  P == L   the instrumentation is inert  (bit-identical tokens)
  P == I   the alternative eval placement is inert
  P == Q   quiescing changes no token
Breakdown
  per-round terms for L, summed against the loop's own wall clock.

Env: PU5_LAYERS, PU5_BODY_EXPERTS, PU5_HEAD_EXPERTS, PU5_GAMMA, PU5_STEPS,
     PU5_PKG.
NOTE: ``mx.eval`` is MLX's graph evaluation, not python's builtin.
"""
import json
import os
import statistics as S
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
PKG = os.environ.get("PU5_PKG", HOME + "/dsv41-ws2/U")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "bench"))

import mlx.core as mx  # noqa: E402

import pU_cap  # noqa: E402

_b = os.environ.get("PU5_BODY_EXPERTS")
_h = os.environ.get("PU5_HEAD_EXPERTS")
pU_cap.apply_caps(int(_b) if _b else None, int(_h) if _h else None)

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PU5_LAYERS", "20,21,37,38,39").split(",")]
GAMMA = int(os.environ.get("PU5_GAMMA", "3"))
STEPS = int(os.environ.get("PU5_STEPS", "20"))

mx.random.seed(0)
log = lambda *a: print("[pU5]", *a, flush=True)

t0 = time.time()
model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=1)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ck = Exl3Checkpoint(MODEL)
head = eb.build_mtp(ck, model.args, rank=0, world=1)
mx.eval(mx.ones(1))
log(f"built layers={LAYERS} world=1 body_cap={_b} head_cap={_h} in {time.time() - t0:.0f}s "
    f"active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")
log(f"vocab_sharded={getattr(head, 'vocab_sharded', False)} "
    f"markov_width={head.markov_head.weight.shape[0]}")

prompt = json.load(open(HOME + "/p30_prompt_ids.json"))[:48]
log(f"prompt {len(prompt)} ids, gamma={GAMMA}, {STEPS} rounds")


def call(rows, iso):
    SP.set_round_log(rows)
    SP._ISOLATE = iso
    t = time.perf_counter()
    toks, st = SP.generate(model, head, prompt, STEPS * (GAMMA + 1) + 8,
                           gamma=GAMMA, adaptive=False)
    dt = time.perf_counter() - t
    SP.set_round_log(None)
    SP._ISOLATE = 0
    return toks, st, dt


# P: no instrumentation at all
t = time.perf_counter()
toks_p, st_p = SP.generate(model, head, prompt, STEPS * (GAMMA + 1) + 8,
                           gamma=GAMMA, adaptive=False)
dt_p = time.perf_counter() - t
log(f"P prod   : {len(toks_p) - 1} tok, {st_p['rounds']} rounds, "
    f"{dt_p * 1e3 / max(st_p['rounds'], 1):.2f} ms/round, {st_p['tok_s']:.2f} tok/s, "
    f"acc {st_p['mean_acc']:.2f}")

rows_l = []
toks_l, st_l, dt_l = call(rows_l, 0)
log(f"L logged : {len(toks_l) - 1} tok, {len(rows_l)} rounds, "
    f"{dt_l * 1e3 / max(len(rows_l), 1):.2f} ms/round, acc "
    f"{np.mean([r['accept'] for r in rows_l]):.2f}")
log(f"CHECK P == L (instrumentation inert): {toks_p == toks_l}")

rows_i = []
toks_i, st_i, dt_i = call(rows_i, 1)
log(f"I iso=1  : {len(toks_i) - 1} tok, {len(rows_i)} rounds, "
    f"{dt_i * 1e3 / max(len(rows_i), 1):.2f} ms/round, acc "
    f"{np.mean([r['accept'] for r in rows_i]):.2f}")
log(f"CHECK P == I (eval placement inert): {toks_p == toks_i}")

TERMS = ["draft_build", "draft_gpu", "body_build", "wait", "readback",
         "rollback_ctx", "emit"]
tot = lambda k, rows: sum(r.get(k, 0.0) for r in rows) / len(rows)


def show(tag, rows):
    wall = tot("total", rows)
    s = 0.0
    log(f"--- {tag}: {len(rows)} rounds, mean {wall:.2f} ms/round")
    for k in TERMS:
        v = tot(k, rows)
        s += v
        log(f"    {k:12s} mean {v:8.2f} ms/round")
    log(f"    {'sum':12s} mean {s:8.2f} ms/round  => unattributed "
        f"{s - wall:+.2f} ms/round")
    log(f"    gamma={rows[0]['g']} mean_acc={np.mean([r['accept'] for r in rows]):.2f} "
        f"mean_committed={np.mean([r['committed'] for r in rows]):.2f}")


show("L logged", rows_l)
show("I iso=1", rows_i)

# Q: production call with a per-round quiesce (does a drain change tokens/time?)
cache = model.make_cache(1, max_seq_len=len(prompt) + STEPS * 8 + 64)
am, taps = model(mx.array([prompt]), cache, last_logit_only=True,
                 return_taps=True, argmax=True)
mx.eval(am)
TAPS = list(model.args.dspark_target_layer_ids)
head2 = head
dsc = head2.make_cache(1)
head2.append_ctx(mx.concatenate([taps[L] for L in TAPS], axis=-1), dsc)
mx.eval([c.win_kv for c in dsc])
nxt = am[:, -1].astype(mx.int32)
mx.eval(nxt)
pos = cache.offset
out = [int(nxt.item())]
ts = []
while len(out) < STEPS * (GAMMA + 1):
    mx.synchronize()                         # quiesce before the round
    t0r = time.perf_counter()
    d, _ = head2.draft(nxt, model.embed, model.head, dsc, width=GAMMA)
    d = d.astype(mx.int32)
    vin = mx.concatenate([nxt.reshape(1, 1), d], axis=1)
    sn = SP.snap(cache, pos)
    am, taps = model(vin, cache, return_taps=True, argmax=True)
    mx.eval(am, d)
    tg, dd = np.array(am[0]), np.array(d[0])
    n = 0
    while n < GAMMA and tg[n] == dd[n]:
        n += 1
    new = [int(v) for v in dd[:n]] + [int(tg[n])]
    SP.rollback(cache, sn, pos + n + 1, SP.stashes(cache))
    head2.append_ctx(mx.concatenate([taps[L] for L in TAPS], axis=-1)[:, :n + 1], dsc)
    pos = pos + n + 1
    nxt = mx.array([new[-1]], dtype=mx.int32)
    mx.eval(nxt)
    ts.append(time.perf_counter() - t0r)
    out.extend(new)
log(f"Q quiesced: {len(out) - 1} tok, median {S.median(ts[3:]) * 1e3:.2f} ms/round")
log(f"CHECK P == Q (a per-round drain changes no token): {toks_p == out}")

log(f"peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU5_DONE")
