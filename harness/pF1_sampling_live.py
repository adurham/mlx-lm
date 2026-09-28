#!/usr/bin/env python3
"""pF1 -- live single-node sanity run of the DSv4.1 sampling spec loop.

Layer-subset body (m4-1, ONE GPU job, wrap in lockf) with the real EXL3 weights,
the real DSpark draft head and the real cache/rollback path. Exercises what the
fixture tests cannot: the DSparkHead draft loop with the DraftProbe, the filtered
distribution over the real 129280-token vocab, cache snap/rollback under
rejections, and the adaptive gamma policy -- plus the two contract checks:

  1. greedy ``spec.generate`` (temperature 0) is untouched: same tokens before
     and after, and the sampled path's stats keep the same shape plus extras;
  2. sampled ``spec.generate`` (temperature > 0) runs, and the same seed gives
     the same token stream while a different seed does not.

The vocab-sharded head cannot be exercised on one process (its gather needs a
real all_sum); ``tests/test_dsv41_sampling.py`` covers that geometry exactly
against an explicitly padded reference (pad + all_sum == full row).

Usage (m4-1):
    EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
      lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python pF1_sampling_live.py
Env: PF1_N (tokens, default 48), PF1_LAYERS (default 20,37,38,39 -- layer 20 is
the kv source the tap layers 37-39 read, so it must be in the subset; the same
subset measured 30 GB peak, keep the box's production run in mind), PF1_TEMP,
PF1_TOPP, PF1_TOPK.
"""
import json
import os
import sys
import time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PF1_PKG", HOME + "/dsv41-ws/F"))
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
N = int(os.environ.get("PF1_N", "48"))
TEMP = float(os.environ.get("PF1_TEMP", "0.8"))
TOPP = float(os.environ.get("PF1_TOPP", "0.95"))
TOPK = int(os.environ.get("PF1_TOPK", "0"))
TAPS = (37, 38, 39)

from mlx_lm.models.deepseek_v41 import exl3_build as eb         # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP               # noqa: E402
from mlx_lm.models.deepseek_v41 import sampling as SAM          # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint            # noqa: E402

LAYERS = [int(x) for x in os.environ.get(
    "PF1_LAYERS", "20,37,38,39").split(",")]
missing = [t for t in TAPS if t not in LAYERS]
if missing:
    print(f"[pF1] tap layers {missing} must be in PF1_LAYERS", flush=True)
    sys.exit(2)

t0 = time.time()
# world=1: the full (unsharded) head. The sharded-head gather is unit-tested
# against its exact geometry instead (see the module docstring).
model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=1)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ck = Exl3Checkpoint(MODEL)
head = eb.build_mtp(ck, model.args, rank=0, world=1)
mx.eval(mx.ones(1))
print(f"[pF1] built layers={LAYERS} in {time.time()-t0:.0f}s "
      f"active={mx.get_active_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB",
      flush=True)

allids = json.load(open(HOME + "/p30_prompt_ids.json"))
prompt = allids[:64]


def null_unbuilt(cache):
    """Layers outside the subset keep a LayerCache but no block to write to it.

    Same guard p48_bench.py uses: those comp_states never receive a stash, and
    spec's rollback would trip over the empty one.
    """
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None


_orig_make_cache = model.make_cache


def make_cache(*a, **k):
    c = _orig_make_cache(*a, **k)
    null_unbuilt(c)
    return c


model.make_cache = make_cache

from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(MODEL + "/tokenizer.json")


def show(name, out, stats):
    text = tok.decode(out[:80]).replace("\n", " / ")
    extra = ""
    if "rejects" in stats:
        extra = (f" rejects={stats['rejects']}/{stats['rounds']}"
                 f" drafted={stats['drafted']}")
    print(f"[pF1] {name}: {len(out)-1} tok in {stats['ms_round']:.0f} ms/round"
          f" acc={stats['mean_acc']:.2f}{extra}\n        {text}", flush=True)


# 1. greedy path (temperature 0 -> the untouched argmax loop)
g1, s1 = SP.generate(model, head, prompt, N, gamma=3, adaptive=False)
g2, s2 = SP.generate(model, head, prompt, N, gamma=3, adaptive=False)
show("greedy", g1, s1)
print(f"[pF1] greedy determinism: identical={g1 == g2} "
      f"stats_keys={sorted(s1.keys())}", flush=True)

# 2. sampled path, same seed twice and a different seed
a, sa = SP.generate(model, head, prompt, N, gamma=3, adaptive=False,
                    temperature=TEMP, top_p=TOPP, top_k=TOPK, seed=1234)
b, sb = SP.generate(model, head, prompt, N, gamma=3, adaptive=False,
                    temperature=TEMP, top_p=TOPP, top_k=TOPK, seed=1234)
c, sc = SP.generate(model, head, prompt, N, gamma=3, adaptive=False,
                    temperature=TEMP, top_p=TOPP, top_k=TOPK, seed=99)
show("sampled seed=1234", a, sa)
show("sampled seed=99", c, sc)
print(f"[pF1] sampled determinism: same_seed_identical={a == b} "
      f"diff_seed_differs={a != c} rejects={sa['rejects']} "
      f"accept_rate={sa['accept_rate']:.3f} drafted={sa['drafted']}", flush=True)

# 3. adaptive gamma with sampling (policy path inside the sampling loop)
d, sd = SP.generate(model, head, prompt, N, gamma=3, adaptive=True,
                    temperature=TEMP, top_p=TOPP, top_k=TOPK, seed=7)
show("sampled adaptive", d, sd)
print(f"[pF1] adaptive gammas={sorted(set(sd['gammas']))} "
      f"acc={sd['mean_acc']:.2f} rejects={sd['rejects']}", flush=True)

# 4. top_k constraint honoured on the real vocab: no token outside the union
#    of per-position supports is emitted; check the emitted set is small.
e, se = SP.generate(model, head, prompt, N, gamma=3, adaptive=False,
                    temperature=1.0, top_p=1.0, top_k=64, seed=5)
print(f"[pF1] top_k=64 sampled distinct tokens={len(set(e))}, "
      f"accept_rate={se['accept_rate']:.3f}", flush=True)
print(f"[pF1] peak={mx.get_peak_memory()/1e9:.2f}GB PF1_DONE", flush=True)
