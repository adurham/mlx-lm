#!/usr/bin/env python3
"""pV_route -- expert-sharing statistics from the saved per-layer routing traces.

CPU ONLY (reads p30-records-clamp/L*.json). No GPU, safe next to production.

For a verify window of R rows each holding k routed experts, this prints the
fraction of the R*k slots that are DISTINCT experts. That fraction is the
upper bound on what per-window expert dedup could ever remove -- and pV5/pV9
measured that the MoE A2 kernel costs the same whether those slots point at
one expert or many, i.e. the dedup bound is not addressable.

Run:  python3 bench/pV_route_stats.py
"""
import json, os, sys
import numpy as np

HOME = os.path.expanduser("~")
REC = os.environ.get("PVR_REC", HOME + "/p30-records-clamp")
LAYERS = [int(x) for x in os.environ.get("PVR_LAYERS", "2,3,20,21,24,25").split(",")]
ROWS = [int(x) for x in os.environ.get("PVR_ROWS", "1,2,3,4,5,6,8").split(",")]


def load(layer):
    p = f"{REC}/L{layer:02d}.json"
    if not os.path.exists(p):
        p = f"{REC}/L{layer}.json"
    d = json.load(open(p))
    return np.array([e[1] for e in d])


print(f"[pV_route] traces in {REC}")
for L in LAYERS:
    if not os.path.exists(f"{REC}/L{L:02d}.json") and not os.path.exists(f"{REC}/L{L}.json"):
        continue
    I = load(L)
    n_tok, k = I.shape
    print(f"  layer {L:3d}: {n_tok} tokens x top-{k}; distinct experts over the "
          f"whole trace = {len(np.unique(I))}")
    for R in ROWS:
        fr = []
        for s in range(0, n_tok - R + 1):
            w = I[s:s + R]
            fr.append(len(np.unique(w)) / (R * k))
        print(f"     R={R}: distinct/(R*k) = {np.mean(fr):.3f}  "
              f"(so {1-np.mean(fr):.1%} of slots are repeats of an expert already "
              f"in the same window)")
