#!/usr/bin/env python3
"""pU8 -- exact all_sum census per call site, one forward at a time. (stream U)

Answers two things with numbers rather than reasoning:

 1. how many collectives each kind of step costs (prefill / draft / verify /
    append_ctx / rollback), and of those, how many are per-draft-token inside
    the draft head;
 2. which call sites they come from (so a per-token collective can be named, not
    just counted).

Two processes, one node, ring backend (the single-node rule). Layers 20,37,38,39
at world 2: the DSpark taps (37,38,39) plus their kv source (20), draft head
sharded as build_mtp does.

Env: PU8_RANK, PU8_HOSTFILE, PU8_LAYERS, PU8_GAMMA, PU8_PKG.
NOTE: ``mx.eval`` is MLX's graph evaluation, not python's builtin.
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
PKG = os.environ.get("PU8_PKG", HOME + "/dsv41-ws2/U")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "bench"))

import mlx.core as mx  # noqa: E402

import pU_cap  # noqa: E402

_b, _h = os.environ.get("PU8_BODY_EXPERTS"), os.environ.get("PU8_HEAD_EXPERTS")
pU_cap.apply_caps(int(_b) if _b else None, int(_h) if _h else None)

HF = os.environ.get("PU8_HOSTFILE", "")
group = mx.distributed.init(strict=True, backend="ring") if HF else None
RANK = group.rank() if group else 0
WORLD = group.size() if group else 1
log = lambda *a: print(f"[pU8 r{RANK}]", *a, flush=True)

# caller-tagged counter: record (tag, caller function name)
TAG = ["build"]
HITS: list[tuple[str, str]] = []
_real = mx.distributed.all_sum


def counted(x, group=None, **kw):
    import inspect
    fr = inspect.currentframe().f_back
    who = "?"
    depth = 0
    while fr is not None and depth < 12:
        name = fr.f_code.co_qualname if hasattr(fr.f_code, "co_qualname") else fr.f_code.co_name
        if name not in ("counted", "<module>"):
            who = name
            break
        fr = fr.f_back
        depth += 1
    HITS.append((TAG[0], who))
    return _real(x, group=group, **kw)


mx.distributed.all_sum = counted

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PU8_LAYERS", "20,37,38,39").split(",")]
GAMMA = int(os.environ.get("PU8_GAMMA", "3"))


def snap(tag):
    """Return a function that reports the hits recorded while it ran."""
    def deco(fn):
        def w(*a, **k):
            TAG[0] = tag
            n0 = len(HITS)
            r = fn(*a, **k)
            sub = HITS[n0:]
            sites = {}
            for _, who in sub:
                sites[who] = sites.get(who, 0) + 1
            if RANK == 0:
                log(f"{tag:22s} {len(sub):4d} all_sum  sites={dict(sorted(sites.items()))}")
            return r
        return w
    return deco


t0 = time.time()
model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=RANK,
                          world=WORLD, group=group)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ck = Exl3Checkpoint(MODEL)
head = eb.build_mtp(ck, model.args, rank=RANK, world=WORLD, group=group)
mx.eval(mx.ones(1))
log(f"built layers={LAYERS} world={WORLD} in {time.time() - t0:.0f}s "
    f"active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")
log(f"sharded: head={getattr(head, 'vocab_sharded', False)} "
    f"markov={head.markov_head.weight.shape[0]}/{model.args.vocab_size} "
    f"bodyhead={hasattr(model.head, 'combine_argmax')} "
    f"moe_group={head.stages[0].ffn.group is not None}")

prompt = json.load(open(HOME + "/p30_prompt_ids.json"))[:24]
TAPS = list(model.args.dspark_target_layer_ids)


def tcf(taps):
    return mx.concatenate([taps[L] for L in TAPS], axis=-1)


cache = model.make_cache(1, max_seq_len=len(prompt) + 128)
TAG[0] = "prefill"
am, taps = model(mx.array([prompt]), cache, last_logit_only=True,
                 return_taps=True, argmax=True)
mx.eval(am)
TAG[0] = "append_ctx"
dsc = head.make_cache(1)
head.append_ctx(tcf(taps), dsc)
mx.eval([c.win_kv for c in dsc])
nxt = am[:, -1].astype(mx.int32)
mx.eval(nxt)
pos = cache.offset
N0 = len(HITS)

# one draft call
TAG[0] = "draft"
c0 = len(HITS)
d, _ = head.draft(nxt, model.embed, model.head, dsc, width=GAMMA)
d = d.astype(mx.int32)
mx.eval(d)
sd = [w for _, w in HITS[c0:]]
if RANK == 0:
    log(f"draft(width={GAMMA})          {len(sd):4d} all_sum  "
        f"sites={ {k: sd.count(k) for k in sorted(set(sd))} }  "
        f"-> {len(sd) / GAMMA:.1f} per draft token")

# one verify forward (R = gamma+1 rows)
TAG[0] = "verify"
c1 = len(HITS)
vin = mx.concatenate([nxt.reshape(1, 1), d], axis=1)
sn = SP.snap(cache, pos)
am, taps = model(vin, cache, return_taps=True, argmax=True)
mx.eval(am)
sv = [w for _, w in HITS[c1:]]
if RANK == 0:
    log(f"verify(R={GAMMA + 1})            {len(sv):4d} all_sum  "
        f"sites={ {k: sv.count(k) for k in sorted(set(sv))} }")

TAG[0] = "rollback"
c2 = len(HITS)
tg, dd = np.array(am[0]), np.array(d[0])
n = 0
while n < GAMMA and tg[n] == dd[n]:
    n += 1
SP.rollback(cache, sn, pos + n + 1, SP.stashes(cache))
sr = [w for _, w in HITS[c2:]]
if RANK == 0:
    log(f"rollback                  {len(sr):4d} all_sum")

TAG[0] = "append_ctx"
c3 = len(HITS)
head.append_ctx(tcf(taps)[:, :n + 1], dsc)
mx.eval([c.win_kv for c in dsc])
sa = [w for _, w in HITS[c3:]]
if RANK == 0:
    log(f"append_ctx                {len(sa):4d} all_sum")

if RANK == 0:
    tot = len(HITS) - N0
    log(f"ONE FULL ROUND (draft+verify+rollback+append_ctx): {tot} all_sum  "
        f"= draft {len(sd)} + verify {len(sv)} + rollback {len(sr)} + append {len(sa)}")
    log(f"per forward (verify): {len(sv)} for {len(LAYERS)} layers + head -> "
        f"attn {len(LAYERS)}, moe {len(LAYERS)}, head 1")

log(f"peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU8_DONE")
