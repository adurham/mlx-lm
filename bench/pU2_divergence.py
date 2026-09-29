#!/usr/bin/env python3
"""pU2 -- why does an extra mx.eval change the tokens? (stream U)

The round instrumentation found that forcing the draft tokens at their own sync
(``DSV41_SPEC_ISOLATE=1``) collapses acceptance 3.00 -> 0.03, i.e. the loop's
output depends on WHERE the graph is forced. That is either (a) an ordering
hazard on a mutated buffer (a real correctness bug), or (b) an eval point that
changes a legitimately lazy input. This probe localises it:

 1. determinism: the same mode run twice must give identical tokens;
 2. first-divergence: with per-round (nxt, dd, tg) dumps at both eval points,
    find the first round where dd or tg differs;
 3. body-only control: the SAME verify input at the SAME cache state, evaluated
    with and without a preceding unrelated eval, must give identical tg
    (isolates the body from the draft/cache path);
 4. draft-only control: the same draft call, forced at two different points,
    must give identical dd (isolates the draft from the body).

Env: PU2_LAYERS, PU2_BODY_EXPERTS, PU2_HEAD_EXPERTS, PU2_WORLD, PU2_ROUNDS,
     PU2_KGAMMA, PU2_PKG.
NOTE: ``mx.eval`` is MLX's graph evaluation, not python's builtin.
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
PKG = os.environ.get("PU2_PKG", HOME + "/dsv41-ws2/U")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "bench"))

import mlx.core as mx  # noqa: E402

import pU_cap  # noqa: E402

BODY_CAP = os.environ.get("PU2_BODY_EXPERTS")
HEAD_CAP = os.environ.get("PU2_HEAD_EXPERTS")
pU_cap.apply_caps(int(BODY_CAP) if BODY_CAP else None,
                  int(HEAD_CAP) if HEAD_CAP else None)

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PU2_LAYERS", "20,37,38,39").split(",")]
WORLD = int(os.environ.get("PU2_WORLD", "2"))
KG = int(os.environ.get("PU2_KGAMMA", "3"))
ROUNDS = int(os.environ.get("PU2_ROUNDS", "8"))

mx.random.seed(0)


def log(*a):
    print("[pU2]", *a, flush=True)


t0 = time.time()
model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0,
                          world=WORLD)
model.set_token_map(json.load(open(HOME + "/dsv41_test/engram_token_map.json"))
                    if os.path.exists(HOME + "/dsv41_test/engram_token_map.json")
                    else json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ck = Exl3Checkpoint(MODEL)
head = eb.build_mtp(ck, model.args, rank=0, world=WORLD)
mx.eval(mx.ones(1))
log(f"built layers={LAYERS} world={WORLD} in {time.time() - t0:.0f}s "
    f"active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")
TAPS = list(model.args.dspark_target_layer_ids)
prompt = json.load(open(HOME + "/p30_prompt_ids.json"))[:48]


def tapcat(taps):
    return mx.concatenate([taps[L] for L in TAPS], axis=-1)


def fresh():
    cache = model.make_cache(1, max_seq_len=len(prompt) + 256)
    am, taps = model(mx.array([prompt]), cache, last_logit_only=True,
                     return_taps=True, argmax=True)
    mx.eval(am)
    dsc = head.make_cache(1)
    head.append_ctx(tapcat(taps), dsc)
    mx.eval([c.win_kv for c in dsc])
    nxt = am[:, -1].astype(mx.int32)
    mx.eval(nxt)
    return cache, dsc, nxt


def loop(iso, n=ROUNDS):
    """Replica of spec.generate's round body with dd/tg dumps."""
    cache, dsc, nxt = fresh()
    pos = cache.offset
    rows = []
    for i in range(n):
        d, _ = head.draft(nxt, model.embed, model.head, dsc, width=KG)
        d = d.astype(mx.int32)
        if iso >= 1:
            mx.eval(d)                      # the extra sync under test
        vin = mx.concatenate([nxt.reshape(1, 1), d], axis=1)
        sn = SP.snap(cache, pos)
        am, taps = model(vin, cache, return_taps=True, argmax=True)
        mx.eval(am, d)
        tg, dd = np.array(am[0]), np.array(d[0])
        n = 0
        while n < KG and tg[n] == dd[n]:
            n += 1
        new = [int(v) for v in dd[:n]] + [int(tg[n])]
        target = pos + n + 1
        SP.rollback(cache, sn, target, SP.stashes(cache))
        head.append_ctx(tapcat(taps)[:, :n + 1], dsc)
        pos = target
        nxt = mx.array([new[-1]], dtype=mx.int32)
        rows.append((i, int(nxt.item()), dd.tolist(), tg.tolist(), n))
    return rows


# 1. determinism + first divergence
a1 = loop(0)
a2 = loop(0)
b1 = loop(1)
b2 = loop(1)
log(f"iso=0 deterministic: {[r[1:] for r in a1] == [r[1:] for r in a2]}")
log(f"iso=1 deterministic: {[r[1:] for r in b1] == [r[1:] for r in b2]}")
first = None
for ra, rb in zip(a1, b1):
    if ra[1:] != rb[1:]:
        first = (ra, rb)
        break
log(f"first diverging round: {first[0][0] if first else None}")
if first:
    ra, rb = first
    log(f"  iso=0: nxt={ra[1]} dd={ra[2]} tg={ra[3]} acc={ra[4]}")
    log(f"  iso=1: nxt={rb[1]} dd={rb[2]} tg={rb[3]} acc={rb[4]}")
    # dd vs tg: which side moved?
    d_same = ra[2] == rb[2]
    t_same = ra[3] == rb[3]
    log(f"  draft tokens same: {d_same}; target tokens same: {t_same}")
for tag, rows in (("iso=0", a1), ("iso=1", b1)):
    log(f"  {tag} rounds: " + " ".join(
        f"[{r[0]}]acc={r[4]}" for r in rows[:6]))

# 3. body-only control: same input, same cache state, eval placement varied
cache, dsc, nxt = fresh()
pos = cache.offset
vin = mx.array([[int(nxt.item())] * (KG + 1)], dtype=mx.int32)
outs = []
for trial in range(4):
    sn = SP.snap(cache, pos)
    am = model(vin, cache, argmax=True)
    if trial % 2 == 1:
        mx.eval(mx.ones(1))              # unrelated eval before the force
    mx.eval(am)
    outs.append(np.array(am[0]).tolist())
    SP.rollback(cache, sn, pos + 1, SP.stashes(cache))
log(f"body-only control (same input, eval placement varied): "
    f"{'identical' if len({tuple(o) for o in outs}) == 1 else 'DIFFERENT'} {outs}")

# 4. draft-only control: same anchor + same window, forced at two points
cache, dsc, nxt = fresh()
outs = []
for trial in range(4):
    if trial % 2 == 0:
        d, c = head.draft(nxt, model.embed, model.head, dsc, width=KG)
        mx.eval(d, c)
    else:
        d, c = head.draft(nxt, model.embed, model.head, dsc, width=KG)
        h = d + 0                        # unrelated op on the same graph
        mx.eval(h, c)
    outs.append(np.array(d[0]).tolist())
log(f"draft-only control (same state, forced at two points): "
    f"{'identical' if len({tuple(o) for o in outs}) == 1 else 'DIFFERENT'} {outs}")

log(f"peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU2_DONE")
