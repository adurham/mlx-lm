#!/usr/bin/env python3
"""pV19 -- why do the attention EXL3 chains grow with R if each module is flat?

pV8 measured every layer-20 dense EXL3 module at slope 0.000 ms/row in
isolation, yet pV18 shows qpath (wq_a+norms+wq_b) and outpath (wo_a stack +
wo_b) growing +0.25 ms each from R=1 to R=6. Two candidate mechanisms:

  A. LAUNCH/SCHEDULING INTERACTION: the modules are flat alone but the real
     graph serializes them with neighbours. Test: time the real qpath and
     outpath sub-graphs at R=1..8 exactly as Attention does.
  B. FUSED-GROUP SHAPE CHANGE: Exl3Member/Exl3FusedGroup re-runs the STACKED
     gemm per member at a new `rows` shape, and `Exl3GroupedStack` reshapes
     (rows, n_groups, in). Test: time run_stacked alone at the group's real
     input shapes for R=1..8.

If B dominates, the fix is to hold one stacked result per (shape) rather than
recomputing per member -- but note the group's members are called with the
SAME x in one pass, so it already caches on `self._x_key`; the check is
identity (`is`), which a new tensor each verify step defeats.

Run (production down, 1 layer, ~7 GB):
  PV19_PKG=~/dsv41-ws2/V PV19_LAYERS=20 EXL3_MM_MAX_ROWS=100000 \
   MTL_DISABLE_TIMEOUT=1 lockf -k ~/dsv41-gpu.lock \
   ~/repos/exo/.venv/bin/python bench/pV19_groups.py
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV19_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYER = int(os.environ.get("PV19_LAYER", "20"))
REPS = int(os.environ.get("PV19_REPS", "15"))


def log(*a):
    print("[pV19]", *a, flush=True)


def timed(fn, reps=REPS, warm=4):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=[LAYER], rank=0,
                          world=2, group=None)
mx.eval(model.parameters())
blk = model.layers[0]
log(f"layer {LAYER} active={mx.get_active_memory()/1e9:.2f}GB")

DIM = model.args.dim


def find(mod, types, path=""):
    out = []
    for name, child in mod.children().items():
        p = f"{path}.{name}" if path else name
        if isinstance(child, types):
            out.append((p, child))
        out += find(child, types, p)
    return out


groups = find(blk, (eb.Exl3FusedGroup,))
log(f"fused groups: {[p for p, _ in groups]}")
for p, g in groups:
    log(f"  {p}: {len(g.lins)} members, in={g.in_f}, outs={g.outs}, "
        f"trellis={tuple(g.trellis.shape)}")

rng = np.random.RandomState(81)
log("--- run_stacked on each real group, rows = 1..8 (the group's own timing) ---")
for p, g in groups:
    row = []
    for R in (1, 2, 3, 4, 6, 8):
        xs = mx.array(rng.randn(R, len(g.lins), g.in_f).astype(np.float16))
        mx.eval(xs)
        t = timed(lambda xs=xs, g=g: g.run_stacked(xs))
        row.append(f"R{R}:{t:6.3f}")
    log(f"  {p:26s} " + " ".join(row))

log("--- grouped stack (wo_a) at rows 1..8 ---")
for p, gs in find(blk, (eb.Exl3GroupedStack,)):
    row = []
    ng = len(gs._g.lins)
    for R in (1, 2, 3, 4, 6, 8):
        xs = mx.array(rng.randn(R, ng, gs._g.in_f).astype(np.float16))
        mx.eval(xs)
        t = timed(lambda xs=xs, gs=gs: gs(xs))
        row.append(f"R{R}:{t:6.3f}")
    log(f"  {p:26s} " + " ".join(row))

log("--- the real qpath / outpath chains, rows 1..8 ---")
attn = blk.attn
from mlx_lm.models.deepseek_v41.layers import rope_tail


def qpath(x):
    qr = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr).reshape(x.shape[0], x.shape[1], attn.n_heads, attn.head_dim)
    return q


def outpath(o, x):
    oo = o.reshape(x.shape[0], x.shape[1], attn.n_groups, -1)
    oo = attn.wo_a(oo)
    return attn.wo_b(oo.reshape(x.shape[0], x.shape[1], -1).astype(x.dtype))


for R in (1, 2, 3, 4, 6, 8):
    x = mx.array(rng.randn(1, R, DIM).astype(np.float16))
    o = mx.array(rng.randn(1, R, attn.n_groups, attn.o_lora_rank).astype(np.float16))
    mx.eval(x, o)
    tq = timed(lambda x=x: qpath(x))
    # outpath input must be the grouped layout the module expects
    og = mx.array(rng.randn(1, R, attn.n_groups * attn.o_lora_rank * (attn.n_heads * attn.head_dim // attn.n_groups) // attn.o_lora_rank).astype(np.float16)) if False else None
    log(f"  R={R}: qpath {tq:7.3f} ms")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV19_DONE")
