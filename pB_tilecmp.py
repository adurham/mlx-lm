#!/usr/bin/env python3
"""pB_tilecmp -- tiled path vs ORIGINAL-module result, one shape, one process.

Runs the NEW module's tiled path (tile forced) and the ORIGINAL module's untiled
path on identical inputs and compares the returned index tensors and the layer-20
candidate mask. Used as the "above the gate" arm of the attic check: the original
module has no tiling code at all, so this is a true old-vs-new comparison.

Env: PB_LAYER, PB_N, PB_NB, PB_SP, PB_TILE, PB_ROWS (rows to compare),
     P48_PKG (new root), PB_ORIG_PKG (original root), PB_OUT.
"""
import os
import subprocess
import sys
import tempfile

HOME = os.path.expanduser("~")
PY = sys.executable
NEW = os.environ.get("P48_PKG", HOME + "/dsv41-ws/B")
ORIG = os.environ.get("PB_ORIG_PKG", HOME + "/dsv41-ws/B_orig")
L = os.environ.get("PB_LAYER", "20")
N = os.environ.get("PB_N", "32")
NB = os.environ.get("PB_NB", "16384")
SP = os.environ.get("PB_SP", "0")
TILE = os.environ.get("PB_TILE", "512")

tmp = tempfile.mkdtemp()
env = dict(os.environ)
ok = 0
fails = 0
for root, tile in ((NEW, TILE), (ORIG, -1)):
    e = dict(env)
    # ORIG has no _TILE knob at all; the new root gets TILE forced by the script
    e.update(P48_PKG=root, PB_LAYER=L, PB_N=N, PB_NB=NB, PB_SP=SP,
             PB_TILE=str(tile), PB_OUT=os.path.join(tmp, "out_" + os.path.basename(root)))
    r = subprocess.run([PY, os.path.join(NEW, "pB_attic_one.py")], env=e,
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-400:], flush=True)
    if r.returncode:
        fails += 1

import numpy as np  # noqa: E402

a = np.load(os.path.join(tmp, "out_" + os.path.basename(NEW) + ".idx.npy"))
b = np.load(os.path.join(tmp, "out_" + os.path.basename(ORIG) + ".idx.npy"))
ar = [sorted(int(c) for c in r if c >= 0) for r in a[0]]
br = [sorted(int(c) for c in r if c >= 0) for r in b[0]]
d = sum(1 for x, y in zip(ar, br) if x != y)
exact = a.tobytes() == b.tobytes()
cand = "n/a"
pa = os.path.join(tmp, "out_" + os.path.basename(NEW) + ".cand.npy")
pb = os.path.join(tmp, "out_" + os.path.basename(ORIG) + ".cand.npy")
if os.path.exists(pa) and os.path.exists(pb):
    cand = "EXACT" if np.load(pa).tobytes() == np.load(pb).tobytes() else "DIFF"
    if cand == "DIFF":
        fails += 1
print(f"[tilecmp] L={L} n={N} nb={NB} sp={SP} tile={TILE}: idx_exact={exact} "
      f"differing_rows={d}/{len(ar)} cand={cand} "
      f"{'PASS' if (d == 0 and cand != 'DIFF') else 'FAIL'}", flush=True)
sys.exit(0 if (d == 0 and cand != "DIFF" and fails == 0) else 1)
