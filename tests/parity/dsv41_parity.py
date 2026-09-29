#!/usr/bin/env python3
"""DSv4.1 greedy parity harness (stream P, round 2) -- standalone.

One script, four modes, all using the same build path as production
(``exl3_build.build_model`` / ``build_block``):

  cos      per-layer cosine vs the p30 clamped trace, layer-isolated (gated at
           0.9998) and optionally chained (report-only), plus end NLL/top-1
           when layers 0..39 are all swept.
  record   run N prompts greedy on the plain path, save a reference file
           (prompt ids, generated tokens, fingerprint, timings, memory, NLL).
  check    run the same prompts and compare token-by-token against a saved
           reference; also runs the cos gate, and the NLL gate when the
           reference carries NLL.
  negctl   negative control: (1) a clean re-run must compare equal, (2) an
           injected one-token flip at generated index k must be caught with
           first divergence exactly k, (3) embedding noise must change tokens,
           (4) a component stub must fail the cos gate.

Runs on a layer subset (single node, half-width experts, no collective,
mem-guarded to <=8 GB weights) or on the full two-node model
(``--dist jaccl``; parent only, production stopped).

Output: human lines + exactly one ``PARITY_JSON {...}`` line. Exit 0 = every
gate passed, 1 = a gate failed, 2 = error / unusable reference.

GPU rules: never run bare -- wrap in ``lockf -k ~/dsv41-gpu.lock``; one job
per node; set EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 (the script sets
both via setdefault).

Env defaults (all overridable by flags):
  P_PKG=~/dsv41-ws2/P   P_MODEL=~/.exo/models/dealignai--...-2.9bpw
  P_NATIVE=~/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram
  P_TOKEN_MAP=~/dsv41-test/engram_token_map.json
  P_TRACE=~/p30-records-clamp   P_IDS=~/p30_prompt_ids.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("P_PKG", os.path.join(HOME, "dsv41-ws2", "P")))
sys.path.insert(0, HERE)

# the two env knobs every DSv4.1 GPU run needs (BRIEF); set before MLX import
os.environ.setdefault("EXL3_MM_MAX_ROWS", "100000")
os.environ.setdefault("MTL_DISABLE_TIMEOUT", "1")

import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402

import parity_core as pc  # noqa: E402

DEF_MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
DEF_NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
DEF_TOKEN_MAP = HOME + "/dsv41-test/engram_token_map.json"
DEF_TRACE = HOME + "/p30-records-clamp"
DEF_IDS = HOME + "/p30_prompt_ids.json"
DEF_REF = HOME + "/dsv41-ws2/parity/ref_subset.json"

# cos-mode default: one layer of every compressed kind (ratio-2 source+indexer
# at 2, ratio-1 source at 20, consumer 24 -- 20 gets auto-added). Measured on
# m4-1: sweep peak 6.43 GB.
COS_LAYERS_DEF = "2,20,24"
# record/check/negctl default: ONE layer. Measured on m4-1 (world 2 = rank-0
# half-width experts, no collective): layers [2,20] need 9.9 GB on the FIRST
# forward (compile transient) and [2,24] 11.0 GB -- over the 8 GB rule. Layer
# 2 alone peaks at 7.43 GB on the first forward, 4.87 GB steady, and it is the
# ratio-2 kv+index source: the layer kind whose compressor/indexer path is
# easiest to get wrong.
RUN_LAYERS_DEF = "2"

# First-forward compile transient + activations, measured on m4-1 (2026-09-29):
# a 1-layer run peaks 1.5 GB above its 5.9 GB resident set. Pre-flight adds this
# to the header-derived weights estimate so the budget decision happens BEFORE
# any allocation; the measured-peak check still runs afterwards.
COMPILE_RESERVE_GB = 2.0

STUB_CHOICES = ("hc", "shared", "experts", "attn", "gate", "lin", "rope",
                "fq", "sattn", "idx", "rms", "engram")

CHAT_QUESTIONS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python one-liner that reads a JSON file and prints its 'name' key.",
    "List three signs that a sourdough starter is healthy.",
]


# --------------------------------------------------------------------------
# small utils
# --------------------------------------------------------------------------

def log(*a):
    print("[parity]", *a, flush=True)


def gb(x):
    return x / 1e9


def rss_peak():
    try:
        return mx.get_peak_memory()
    except Exception:  # pragma: no cover - old mlx
        return 0


def peak_gb():
    return gb(rss_peak())


def weight_estimate_gb(model_dir: str, layers, world: int, single: bool = False) -> float:
    """Checkpoint bytes the run will hold, from the safetensors headers only.

    Only tensors ``exl3_build.build_model`` actually loads are counted: the
    built layers plus embed/norm/head. The ``mtp.*`` (DSpark) and ``vision.*``
    tensors are skipped by the builder and must not count against the budget.

    ``single`` = the cos sweep shape: one block resident at a time, so the
    estimate is the largest built layer (plus a margin), not the sum.
    """
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        wmap = json.load(f)["weight_map"]
    TOP_OK = ("embed.weight", "norm.weight", "head.")
    hdrs, per_layer, top = {}, {}, 0.0
    want = set(layers) if layers is not None else None
    for name, shard in wmap.items():
        if name.startswith(("mtp.", "vision", "aligner.", "image_")):
            continue
        ent = hdrs.get(shard)
        if ent is None:
            with open(os.path.join(model_dir, shard), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                ent = hdrs[shard] = json.loads(fh.read(n))
        e = ent[name]
        nbytes = e["data_offsets"][1] - e["data_offsets"][0]
        if name.startswith("layers."):
            L = int(name.split(".")[1])
            if want is not None and L not in want:
                continue
            if ".ffn.experts." in name and world > 1:
                nbytes //= world
            per_layer[L] = per_layer.get(L, 0) + nbytes
        elif name.startswith(TOP_OK):
            top += nbytes
    if single:
        # one block + embed rows + transients
        return (max(per_layer.values(), default=0) + 1.0) / 1e9
    return (sum(per_layer.values()) + top) / 1e9


def check_cap(args, world, where):
    """Hard abort if the measured peak passed the single-node budget.

    The 8 GB rule is for single-node runs; a full two-node run is the parent's
    business (~105 GB per rank) and is exempt.
    """
    if args.dist != "none" or args.force_oversize:
        return
    if mx.get_peak_memory() / 1e9 > args.max_gb:
        die(f"measured peak {gb(mx.get_peak_memory()):.2f} GB > "
            f"--max-gb {args.max_gb} at {where} (shrink --layers or use "
            f"--force-oversize with a reason)")


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

def free_mem_check(args):
    """Warn (or fail with --require-free-gb) when the node looks busy.

    ``lockf`` serializes GPU jobs but reserves no memory: if production is
    still resident (~92 GB) a "small" run can still push the node into swap.
    This is a pre-flight check, before any allocation.
    """
    pages = os.sysconf("SC_PAGE_SIZE")
    try:
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=10).stdout
        m = re.search(r"Pages free:\s+(\d+)", out)
        free_gb = (int(m.group(1)) * pages / 1e9) if m else None
    except Exception:
        free_gb = None
    if free_gb is None:
        return None
    need = args.require_free_gb or (args.max_gb + 4.0)
    log(f"free memory: {free_gb:.1f} GB (want >= {need:.1f} GB free)")
    if args.require_free_gb and free_gb < args.require_free_gb:
        die(f"only {free_gb:.1f} GB free < --require-free-gb "
            f"{args.require_free_gb}; is production still running?")
    elif free_gb < need:
        log(f"WARNING: only {free_gb:.1f} GB free; a peer job or production may "
            f"still be resident")
    return free_gb


def init_dist(args):
    """(world, rank, group). Single node = simulated TP: ``--world``/``--rank``
    select the rank slice WITHOUT a collective (the p48/p66 convention: rank-0
    half-width experts, no all_sum) -- same build path, half the weight memory.
    """
    if args.dist == "none":
        return args.world, args.rank, None
    group = mx.distributed.init(backend=args.dist, strict=True)
    return group.size(), group.rank(), group


def build(args, layers):
    """Full or layer-subset model on the production build path."""
    world, rank, group = init_dist(args)
    from mlx_lm.models.deepseek_v41 import exl3_build as eb
    est = weight_estimate_gb(args.model, set(layers), world)
    # Pre-flight is on a CONSERVATIVE bound, before allocating: weights plus a
    # reserve for the first-forward compile transient + activations (measured
    # 1.5 GB on a 1-layer run). The post-hoc peak check below catches anything
    # the reserve missed.
    need = est + COMPILE_RESERVE_GB
    log(f"layers={list(layers)} world={world} rank={rank} dist={args.dist} "
        f"weights_est={est:.2f}GB +reserve={COMPILE_RESERVE_GB:.1f}GB "
        f"= {need:.2f}GB budget active={gb(mx.get_active_memory()):.2f}GB")
    if args.dist == "none" and not args.force_oversize and need > args.max_gb:
        die(f"estimated {need:.2f} GB (weights {est:.2f} + reserve "
            f"{COMPILE_RESERVE_GB:.1f}) > --max-gb {args.max_gb}; shrink "
            f"--layers or pass --force-oversize with a reason")
    t0 = time.time()
    model, report = eb.build_model(args.model, native_dir=args.native,
                                   layers=list(layers), rank=rank, world=world,
                                   group=group)
    model.set_token_map(json.load(open(args.token_map)))
    if group is not None:
        mx.eval(mx.distributed.all_sum(mx.ones(1), group=group))
    build_s = time.time() - t0
    log(f"built in {build_s:.1f}s active={gb(mx.get_active_memory()):.2f}GB "
        f"peak={gb(rss_peak()):.2f}GB")
    check_cap(args, world, "build")
    return model, {"world": world, "rank": rank, "build_s": build_s,
                   "weights_est_gb": est, "report": report}


def apply_stub(name):
    """p48-style component stubs (negative-control injections)."""
    from mlx_lm.models.deepseek_v41 import exl3_build as eb
    from mlx_lm.models.deepseek_v41 import model as M, moe as MO, attention as A
    if name == "hc":
        M.hc_mixes = lambda x, *a: (
            mx.full(x.shape[:2] + (4,), 0.25), mx.full(x.shape[:2] + (4,), 0.5),
            mx.full(x.shape[:2] + (4, 4), 0.25))
        M.hc_post = lambda h, r, p, c: r + h[:, :, None, :]
        M.hc_pre = lambda x, pm: x[:, :, 0]
    elif name == "shared":
        MO.SharedExpert.__call__ = lambda self, x: x * 0
    elif name == "experts":
        eb.Exl3Experts.__call__ = lambda self, x, idx: mx.broadcast_to(
            x[:, None, :] * 0, idx.shape + (x.shape[-1],))
    elif name == "attn":
        A.Attention.__call__ = lambda self, x, sp, c, s: x
    elif name == "gate":
        MO.Gate.__call__ = lambda self, x: (
            x[:, :6] * 0 + 0.25, mx.broadcast_to(mx.arange(6), x.shape[:-1] + (6,)))
    elif name == "lin":
        def _z(x, n):
            return mx.broadcast_to(x[..., :1] * 0, x.shape[:-1] + (n,)).astype(x.dtype)
        eb.Exl3Proj.__call__ = lambda self, x: _z(x, self._lin.out_features)
        eb.Exl3Member.__call__ = lambda self, x: _z(x, self.out_features)
        eb.Exl3GroupedStack.__call__ = lambda self, x: _z(x, self._g.outs[0])
    elif name == "rope":
        from mlx_lm.models.deepseek_v41 import indexer as IX
        A.rope_tail = lambda x, rd, c, s, inverse=False: x
        IX.rope_tail = A.rope_tail
    elif name == "fq":
        from mlx_lm.models.deepseek_v41 import indexer as IX
        A.fake_quant_fp8_ue8m0 = lambda x, b=32: x
        A.fake_quant_fp4_e4m3 = lambda x, b=16: x
        IX.fake_quant_fp4_ue8m0 = lambda x, b=32: x
    elif name == "sattn":
        A.sparse_attn = lambda q, kv, sink, idx, sc, chunk=256: q
    elif name == "idx":
        from mlx_lm.models.deepseek_v41 import indexer as IX
        IX.Indexer.__call__ = lambda self, x, qr, sp, off, c, s, ik, sh: (
            mx.zeros((x.shape[0], x.shape[1], min(self.index_topk, ik.shape[1])),
                     mx.int32) + x[..., :1].astype(mx.int32) * 0)
    elif name == "rms":
        from mlx_lm.models.deepseek_v41 import layers as LY
        LY.RMSNorm.__call__ = lambda self, x: x
    elif name == "engram":
        from mlx_lm.models.deepseek_v41 import engram as EN
        EN.Engram.__call__ = lambda self, x, hid: x
    else:
        die(f"unknown stub {name!r} (choices: {', '.join(STUB_CHOICES)})")
    log(f"STUB applied: {name}")


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

def load_prompts(args):
    if args.prompt_file:
        doc = json.load(open(args.prompt_file))
        raw = doc["prompts"] if isinstance(doc, dict) else doc
        out = []
        for i, p in enumerate(raw):
            if isinstance(p, dict):
                out.append({"name": p.get("name", f"file{i}"),
                            "ids": [int(t) for t in p["ids"]]})
            else:
                out.append({"name": f"file{i}", "ids": [int(t) for t in p]})
        return out[:args.prompt_count] if args.prompt_count else out
    if args.prompts == "chat":
        try:
            from tokenizers import Tokenizer
            import jinja2
        except ImportError as e:
            die(f"--prompts chat needs tokenizers + jinja2: {e}")
        tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
        tmpl = jinja2.Environment().from_string(
            open(os.path.join(args.model, "chat_template.jinja")).read())
        out = []
        qs = CHAT_QUESTIONS[:args.prompt_count or len(CHAT_QUESTIONS)]
        for i, q in enumerate(qs):
            text = tmpl.render(messages=[{"role": "user", "content": q}],
                               add_generation_prompt=True, enable_thinking=False)
            out.append({"name": f"chat{i}", "ids": tok.encode(
                text, add_special_tokens=False).ids, "text": q})
        return out
    ids = json.load(open(args.ids))[:531]
    lens = [64, 128, 256, 531][:args.prompt_count or 4]
    return [{"name": f"p30[:{L}]", "ids": list(ids[:L])} for L in lens]


# --------------------------------------------------------------------------
# greedy plain path
# --------------------------------------------------------------------------

class _NoiseOnce:
    """Wraps an Embedding: one-shot noise of relative size ``rel`` (seeded)."""

    def __init__(self, orig, rel, seed):
        self.orig, self.rel, self.seed, self.done = orig, rel, seed, False

    def __call__(self, x):
        out = self.orig(x)
        if not self.done:
            self.done = True
            rng = np.random.default_rng(self.seed)
            noise = mx.array(rng.standard_normal(out.shape).astype(np.float32))
            scale = self.rel * float(mx.sqrt(mx.mean(
                mx.square(out.astype(mx.float32)))))
            out = (out.astype(mx.float32) + noise * scale).astype(out.dtype)
        return out


def greedy_plain(model, ids, max_new, *, perturb=None, chunk=0, margins=False):
    """Greedy decode on the plain path. Returns (tokens, stats).

    tokens[0] is produced by the prompt forward; tokens[k] for k >= 1 is the
    k-th generated token. ``perturb``:
      {"kind":"argmax","step":k}  flip generated index k (tokens[k] differs)
      {"kind":"noise","rel":r}    one-shot embedding noise on the prompt forward
    ``margins`` additionally records the fp32 top-1 minus top-2 logit gap at
    every step (returned under "margins"): a divergence with a near-zero gap is
    a float tie, not a regression -- on TP=2 the reduction order can flip an
    argmax that is decided by ~1e-6. Costs one full-vocab row per step.
    """
    ids = [int(t) for t in ids]
    cache = model.make_cache(1, max_seq_len=len(ids) + max_new + 16)
    try:
        mx.reset_peak_memory()      # per-prompt peak, not per-process
    except Exception:
        pass
    swap = None
    if perturb and perturb["kind"] == "noise":
        swap = model.embed
        model.embed = _NoiseOnce(swap, perturb["rel"], perturb.get("seed", 7))

    def step_logits(arr):
        """argmax token + top-2 gap from a forward's output.

        MLX's ``mx.topk`` returns VALUES in ASCENDING order (verified on
        mlx 0.32.3), so the max is the last element and the runner-up the one
        before it. The token comes from ``argmax`` separately. Cross-checked:
        max(topk) must equal max(row), else the topk contract changed.
        """
        row = arr.astype(mx.float32).reshape(-1)
        vals = mx.topk(row, 2)                   # ascending values
        tok = mx.argmax(row).astype(mx.int32)
        mx.eval(vals, tok)
        v = np.array(vals)
        max_row = float(mx.max(row).item())
        if abs(float(v[-1]) - max_row) > 1e-6:
            die(f"mx.topk contract changed: topk[0]={float(v[-1])} != "
                f"max(row)={max_row}; the margin diagnostic is unusable")
        gap = float(v[-1]) - float(v[-2])
        return int(tok.item()), gap

    t0 = time.perf_counter()
    if chunk and chunk < len(ids):
        from mlx_lm.models.deepseek_v41 import prefill as PF
        nxt = PF.prefill(model, ids, cache, chunk=chunk, argmax=not margins)
    else:
        nxt = model(mx.array([ids]), cache, last_logit_only=True,
                    argmax=not margins)
    gaps = []
    if margins:
        tok, gap = step_logits(nxt)
        nxt = mx.array([tok], dtype=mx.int32)
        gaps.append(gap)
    else:
        nxt = nxt.reshape(-1)[-1:]
    mx.eval(nxt)
    t_prompt = time.perf_counter() - t0
    out = [int(nxt.item())]
    steps = []
    for k in range(max_new):
        s = time.perf_counter()
        arr = model(nxt[None], cache, last_logit_only=True, argmax=not margins)
        if margins:
            tok, gap = step_logits(arr)
            nxt = mx.array([tok], dtype=mx.int32)
            gaps.append(gap)
        else:
            nxt = arr.reshape(-1)[-1:]
            mx.eval(nxt)
            tok = int(nxt.item())
        if perturb and perturb["kind"] == "argmax" and k + 1 == perturb["step"]:
            tok = (tok + 1) % model.args.vocab_size
            nxt = mx.array([tok], dtype=mx.int32)
        steps.append(time.perf_counter() - s)
        out.append(tok)
    if swap is not None:
        model.embed = swap
    st = np.array(steps[2:]) if len(steps) > 2 else np.array(steps)
    stats = {"prompt_s": t_prompt,
             "ms_step": float(np.median(st) * 1e3) if len(st) else 0.0,
             "tok_s": len(steps) / sum(steps) if sum(steps) else 0.0,
             "peak_gb": peak_gb()}
    if margins:
        stats["margins"] = gaps
    return out, stats


def run_prompts(model, prompts, max_new, *, perturb=None, chunk=0, margins=False):
    """Run every prompt; ``perturb`` applies to prompt 0 only."""
    results, t0 = [], time.perf_counter()
    for i, p in enumerate(prompts):
        toks, st = greedy_plain(model, p["ids"], max_new, chunk=chunk,
                                perturb=(perturb if i == 0 else None),
                                margins=margins)
        results.append({"name": p["name"], "ids": list(p["ids"]),
                        "tokens": toks, "stats": st})
        m = st.get("margins")
        log(f"  {p['name']}: {len(toks)} tokens, {st['ms_step']:.1f} ms/step, "
            f"{st['tok_s']:.1f} tok/s, peak={st['peak_gb']:.2f}GB, "
            f"first={toks[:8]}" + (f", min_margin={min(m):.3g}" if m else ""))
    return results, time.perf_counter() - t0


def compute_nll(model, ids):
    """Teacher-forced NLL / top-1 over one full-prompt forward (p46's method)."""
    cache = model.make_cache(1, max_seq_len=len(ids) + 8)
    logits = model(mx.array([[int(t) for t in ids]]), cache, last_logit_only=False)
    logits = logits.astype(mx.float32)[0]
    mx.eval(logits)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = mx.array([int(t) for t in ids[1:]])
    nll = -mx.take_along_axis(lp[:-1], tgt[:, None], axis=-1)[:, 0]
    top1 = mx.argmax(logits[:-1], axis=-1) == tgt
    mx.eval(nll, top1)
    nl = np.array(nll)
    return {"mean": float(nl.mean()), "median": float(np.median(nl)),
            "top1": float(top1.astype(mx.float32).mean()) * 100.0,
            "n": int(nl.size)}


# --------------------------------------------------------------------------
# cos mode (p46 method, layer-isolated)
# --------------------------------------------------------------------------

def complete_layers(layers, cargs=None):
    """Add the source layers a consumer layer needs in the same sweep.

    A compressing layer that is NOT a kv/index source reads the most recent
    source above it (``shared.kv_src_cache`` / ``shared.index_src_cache``), so
    sweeping e.g. only layer 24 crashes with a fresh ``SharedState``. The sweep
    therefore auto-inserts the required sources (highest source <= L).
    """
    out = set(int(x) for x in layers)
    if cargs is None:
        return sorted(out)
    changed = True
    while changed:
        changed = False
        for L in sorted(out):
            if cargs.compress_ratio(L) == 0:
                continue
            if L not in cargs.kv_source_layers:
                src = cargs.kv_source_for(L)
                if src not in out:
                    out.add(src)
                    changed = True
            if L not in cargs.index_source_layers:
                src = cargs.index_source_for(L)
                if src not in out:
                    out.add(src)
                    changed = True
    return sorted(out)


def embed_rows(ck, name, row_ids):
    """Read only the requested rows of a big embedding table (fp32 result).

    ``ck.np(name)`` materialises the whole table (2.6 GB fp32 for the 129280 x
    5120 embed) and a gather adds a transient copy; the cos sweep has to stay
    inside the 8 GB single-node cap, so read the 531 rows we need straight off
    the shard file (the LazyEngramTable trick, generalised).
    """
    sh = ck._shard(name)
    ent = sh.header[name]
    o0, _ = ent["data_offsets"]
    rows, dim = ent["shape"][0], ent["shape"][1]
    dtype = ent["dtype"]
    per = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
    stride = dim * per
    fd = sh._open()
    buf = b"".join(os.pread(fd, stride, sh.base + o0 + int(r) * stride)
                   for r in row_ids)
    if dtype == "BF16":
        u = np.frombuffer(buf, np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(len(row_ids), dim)
    return np.frombuffer(buf, {"F16": np.float16, "F32": np.float32}[dtype]
                         ).astype(np.float32).reshape(len(row_ids), dim)


def cos_sweep(args, layers):
    """Per-layer cosine vs the p30 trace; one block resident at a time.

    Always builds FULL-width layers on a single node (``--cos-world`` default 1):
    the p30 trace was recorded world-1, so a simulated-TP slice (half-width
    experts, no collective) is not comparable to it. With ``--dist jaccl`` the
    real collective is present and the layer output is the full-model output,
    so the trace comparison is valid again. Token runs may use ``--world``
    freely; this gate may not.
    """
    from mlx_lm.models.deepseek_v41 import exl3_build as eb
    from mlx_lm.models.deepseek_v41.config import ModelArgs
    from mlx_lm.models.deepseek_v41.model import SharedState
    from mlx_lm.models.deepseek_v41.cache import ModelCache
    from mlx_lm.models.deepseek_v41.engram import EngramHasher
    from mlx_lm.models.deepseek_v41.hyper_connections import (
        make_identity_pre_mix, hc_pre)
    from mlx_lm.models.deepseek_v41.layers import RMSNorm
    from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_linear

    if args.dist == "none":
        world, rank, group = args.cos_world, args.cos_rank, None
        if world != 1:
            log(f"WARNING: --cos-world {world} on a single node builds a sliced "
                f"layer with no collective; the p30 trace is world-1 full-width, "
                f"so the cos values will NOT be comparable")
    else:
        # real two-node run: the collective is present, so the layer output IS
        # the full-model output and the p30 trace stays comparable
        world, rank, group = init_dist(args)
        log(f"cos sweep over jaccl: world={world} rank={rank}")
    layers = complete_layers_read(args, list(layers))   # consumers need their sources
    ck = Exl3Checkpoint(args.model)
    nat = Exl3Checkpoint(args.native)
    cargs = ModelArgs.from_dict(ck.config)
    ids = json.load(open(args.ids))[:531]
    n = len(ids)
    if world == 1:
        est = weight_estimate_gb(args.model, set(layers), world, single=True)
        if est + COMPILE_RESERVE_GB > args.max_gb:
            die(f"cos: largest block {est:.2f} GB + reserve "
                f"{COMPILE_RESERVE_GB:.1f} GB > --max-gb {args.max_gb} "
                f"(a single-node cos sweep holds one block at a time)")
    # fp32 embed rows for the 531 prompt ids only (p30 used the full fp32 embed)
    emb = mx.array(embed_rows(ck, "embed.weight", ids))
    h = mx.contiguous(mx.broadcast_to(emb[:, None, :], (1, n, cargs.hc_mult, cargs.dim)))
    del emb
    mx.eval(h)
    cache = ModelCache(cargs, 1, n + 64, dtype=mx.float32)
    pre_mix = make_identity_pre_mix(1, n, cargs.hc_mult)
    shared = SharedState()
    hasher = EngramHasher(cargs, json.load(open(args.token_map)))
    hashes = mx.array(hasher(np.array([ids], dtype=np.int64), 0, cache.engram_ids))

    def cos(x, y):
        x = x.astype(mx.float32).reshape(-1)
        y = y.astype(mx.float32).reshape(-1)
        return float((x * y).sum() / (mx.sqrt((x * x).sum()) * mx.sqrt((y * y).sum())))

    items, t_all = [], time.time()
    log(f"cos sweep layers={layers} isolate={args.isolate}")
    for L in layers:
        t0 = time.time()
        try:
            mx.reset_peak_memory()  # per-layer peak
        except Exception:
            pass
        blk, rep = eb.build_block(ck, cargs, L, native=nat, rank=rank, world=world)
        if args.isolate and L > 0:
            h = mx.load(f"{args.trace}/h_{L-1:02d}.npy")
            pre_mix = mx.load(f"{args.trace}/pm_{L-1:02d}.npy")
        if blk.engram is not None:
            h = blk.engram(h, hashes[:, :, blk.engram.layer_hash_index])
        h, pre_mix = blk(h, pre_mix, 0, cache, shared)
        mx.eval(h, pre_mix)
        ref = mx.load(f"{args.trace}/h_{L:02d}.npy")
        refpm = mx.load(f"{args.trace}/pm_{L:02d}.npy")
        c, cp = cos(h, ref), cos(pre_mix, refpm)
        rel = float(mx.max(mx.abs(h.astype(mx.float32) - ref.astype(mx.float32)))) / \
            float(mx.max(mx.abs(ref.astype(mx.float32))))
        items.append({"layer": L, "cos": c, "cos_pm": cp, "maxrel": rel,
                      "build_s": time.time() - t0, "peak_gb": peak_gb(),
                      "plain": rep["plain"], "dense_groups": rep["dense_groups"]})
        log(f"  L{L:02d} cos_h={c:.6f} cos_pm={cp:.6f} maxrel={rel:.3e} "
            f"build={time.time() - t0:.1f}s peak={peak_gb():.2f}GB")
        del blk
        gc.collect()
        mx.clear_cache()
        check_cap(args, world, f"cos L{L}")
    out = {"layers": items, "gate": pc.cos_gate(items),
           "sweep_s": time.time() - t_all,
           "mode": "isolated" if args.isolate else "chained"}
    # end NLL: only meaningful for the full stack, and only when the sweep was
    # isolated (chained over a subset is not the reference computation)
    if list(layers) == list(range(cargs.n_layers)) and args.isolate:
        nrm = RMSNorm(cargs.dim, cargs.norm_eps)
        nrm.load_weights([("weight", mx.array(ck.np("norm.weight")).astype(mx.float32))])
        x = nrm(hc_pre(h, pre_mix))
        head = eb.Exl3Proj(load_dense_linear(ck, "head"))
        logits = head(x.astype(mx.float16)).astype(mx.float32)[0]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = mx.array([int(t) for t in ids[1:]])
        nll = -mx.take_along_axis(lp[:-1], tgt[:, None], axis=-1)[:, 0]
        top1 = mx.argmax(logits[:-1], axis=-1) == tgt
        mx.eval(nll, top1)
        nl = np.array(nll)
        nllv = {"mean": float(nl.mean()), "median": float(np.median(nl)),
                "top1": float(top1.astype(mx.float32).mean()) * 100.0}
        out["nll"] = nllv
        out["nll_gate"] = pc.nll_gate(nllv["mean"], nllv["median"], nllv["top1"])
        log(f"  NLL mean={nllv['mean']:.4f} median={nllv['median']:.4f} "
            f"top1={nllv['top1']:.1f}%")
    return out


# --------------------------------------------------------------------------
# fingerprint helpers
# --------------------------------------------------------------------------

PKG_FILES = [f"mlx_lm/models/deepseek_v41/{f}" for f in (
    "__init__.py", "attention.py", "cache.py", "compressor.py", "config.py",
    "dequant.py", "engram.py", "exl3_build.py", "fakequant.py", "hc_fused.py",
    "hyper_connections.py", "indexer.py", "layers.py", "model.py", "moe.py",
    "mtp.py", "prefill.py", "sampling.py", "session_cache.py",
    "sparse_attention.py", "spec.py")] + [
    f"mlx_lm/models/exl3/{f}" for f in (
        "__init__.py", "decode.py", "exl3_linear.py", "exl3_moe.py", "loader.py",
        "metal_kernels.py", "ops.py", "reconstruct.py")] + [
    "tests/parity/parity_core.py", "tests/parity/dsv41_parity.py"]


def fingerprint(args, layers, prompts, world, rank):
    cfg = {"layers": list(layers) if layers is not None else None,
           "dist": args.dist, "world": world,
           "max_new": args.max_new, "chunk": args.chunk,
           "model_dir": args.model, "native_dir": args.native,
           "token_map_digest": pc.file_digest(args.token_map),
           "trace_digest": pc.dir_manifest_digest(args.trace),
           "pkg_digest": pc.pkg_digest(args.pkg_root, PKG_FILES)}
    fp = pc.make_fingerprint(cfg, prompts)
    fp["rank"] = rank
    return fp


def die(msg, code=2):
    print(f"[parity] ERROR: {msg}", flush=True)
    print("PARITY_JSON " + json.dumps({"mode": "error", "ok": False,
                                       "error": msg}), flush=True)
    sys.exit(code)


def emit(mode, ok, checks, payload, extra=None):
    doc = {"mode": mode, "ok": bool(ok), "checks": checks, "payload": payload}
    if extra:
        doc.update(extra)
    print("PARITY_JSON " + json.dumps(doc, default=str), flush=True)
    return 0 if ok else 1


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def complete_layers_read(args, layers):
    """complete_layers, reading the config headers (no weights touched)."""
    want = sorted(set(int(x) for x in layers))
    if args.dist != "none":
        return want
    from mlx_lm.models.deepseek_v41.config import ModelArgs
    cfgp = os.path.join(args.model, "config.json")
    if not os.path.exists(cfgp):
        return want
    with open(cfgp) as f:
        cargs = ModelArgs.from_dict(json.load(f))
    return complete_layers(want, cargs=cargs)


def mode_cos(args):
    layers = pc.parse_layers(args.layers, 40)
    res = cos_sweep(args, layers)
    if args.isolate:
        checks = {"cos_gate": res["gate"]["ok"]}
        if "nll_gate" in res:
            checks["nll_gate"] = res["nll_gate"]["ok"]
        log(f"worst_cos={res['gate']['worst_cos']:.6f} at L{res['gate']['worst_layer']} "
            f"bar={res['gate']['bar']} ok={checks['cos_gate']}")
    else:
        # chained over a subset is not the same computation as the trace: report
        # only. (The chained variant is for spotting WHERE a change first moves
        # h, not for a pass/fail bar.)
        checks = {"cos_report_only": True}
        log(f"chained (report-only): worst_cos={res['gate']['worst_cos']:.6f} "
            f"at L{res['gate']['worst_layer']}")
    return emit("cos", all(checks.values()), checks, res,
                {"worst_cos": res["gate"]["worst_cos"], "nll": res.get("nll"),
                 "layers": [i["layer"] for i in res["layers"]],
                 "gate": res["gate"], "peak_gb": peak_gb()})


def mode_record(args):
    layers = complete_layers_read(args, pc.parse_layers(args.layers, 40))
    prompts = load_prompts(args)
    model, binf = build(args, layers)
    results, total = run_prompts(model, prompts, args.max_new,
                                 chunk=args.chunk, perturb=None,
                                 margins=args.margins)
    payload = {"prompts": results, "built": {k: v for k, v in binf.items()
                                             if k != "report"},
               "total_s": total, "peak_gb": peak_gb()}
    nll = None
    if len(layers) == 40:
        nll = compute_nll(model, prompts[0]["ids"])
        payload["nll"] = nll
        log(f"NLL mean={nll['mean']:.4f} median={nll['median']:.4f} top1={nll['top1']:.1f}%")
    fp = fingerprint(args, layers, prompts, binf["world"], binf["rank"])
    cfg = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "timings": {"total_s": total}, "memory": {"peak_gb": peak_gb()},
           "nll": nll}
    try:
        pc.save_reference(args.ref, cfg, fp, results, note=args.note,
                          overwrite=args.overwrite_ref)
    except FileExistsError as e:
        die(str(e))
    log(f"reference saved: {args.ref} (pkg_digest={fp['pkg_digest']})")
    return emit("record", True, {"saved": True}, payload,
                {"ref": args.ref, "fingerprint": fp, "nll": nll})


def mode_check(args):
    layers = complete_layers_read(args, pc.parse_layers(args.layers, 40))
    prompts = load_prompts(args)
    if not os.path.exists(args.ref):
        die(f"reference not found: {args.ref} (run record first)")
    ref = pc.load_reference(args.ref)
    model, binf = build(args, layers)
    perturb = pc.parse_perturb(args.perturb) if args.perturb else {"kind": "none"}
    if perturb["kind"] == "stub":
        die("check --perturb stub:* is not supported (a stub needs its own build; "
            "use `negctl --stub ...` for the cos-gate control)")
    if perturb["kind"] != "none":
        log(f"INJECTING perturbation {perturb} on prompt 0 (expect FAIL)")
    results, total = run_prompts(model, prompts, args.max_new, chunk=args.chunk,
                                 perturb=perturb if perturb["kind"] != "none" else None)
    cur = pc.compare_runs(ref["prompts"], results)
    for p in cur["prompts"]:
        log(f"  {p['name']}: match {p.get('n_match')}/{p.get('n_ref')} "
            f"first_div={p.get('first_div')} ok={p['ok']}")
    fp = fingerprint(args, layers, prompts, binf["world"], binf["rank"])
    struct, digest_diff = pc.fingerprint_mismatch(ref["fingerprint"], fp)
    drift_ok = (not digest_diff) or args.allow_code_drift
    checks = {"tokens": cur["ok"], "fingerprint_structural": not struct,
              "code_digest": drift_ok}
    payload = {"compare": cur, "fingerprint": fp,
               "ref_fingerprint": ref["fingerprint"], "structural_diff": struct,
               "pkg_digest_diff": digest_diff,
               "code_drift_allowed": bool(args.allow_code_drift),
               "total_s": total, "peak_gb": peak_gb(), "perturb": perturb}
    if digest_diff:
        log(f"{'WARNING' if args.allow_code_drift else 'FAIL'}: code digest "
            f"differs from the reference "
            f"({ref['fingerprint'].get('pkg_digest')} -> {fp['pkg_digest']})"
            f"{' (allowed by --allow-code-drift)' if args.allow_code_drift else ''}")
    nll = None
    if len(layers) == 40:
        nll = compute_nll(model, prompts[0]["ids"])
        payload["nll"] = nll
        payload["nll_ref"] = ref.get("nll")
        g = pc.nll_gate(nll["mean"], nll["median"], nll["top1"])
        checks["nll_gate"] = g["ok"]
        payload["nll_gate"] = g
        log(f"  NLL mean={nll['mean']:.4f} median={nll['median']:.4f} "
            f"top1={nll['top1']:.1f}% gate={'PASS' if g['ok'] else 'FAIL'}")
    if struct:
        log(f"STRUCTURAL fingerprint mismatch: {struct} -> token comparison invalid")
    return emit("check", all(checks.values()), checks, payload,
                {"ref": args.ref, "nll": nll})


def mode_negctl(args):
    layers = complete_layers_read(args, pc.parse_layers(args.layers, 40))
    prompts = load_prompts(args)
    if len(prompts) > 1:
        prompts = prompts[:1]                      # negative control: 1 prompt
    events = []

    # ---- (1)(2)(3) token-level controls on the greedy path -----------------
    model, binf = build(args, layers)
    base, _ = run_prompts(model, prompts, args.max_new, chunk=args.chunk)
    base = base[0]
    check_cap(args, binf["world"], "token run")
    determinism = {"seed": None, "temperature": 0.0, "decoding": "greedy argmax",
                   "runs_compared": 1}

    clean, _ = run_prompts(model, prompts, args.max_new, chunk=args.chunk)
    c = pc.compare_tokens(base["tokens"], clean[0]["tokens"])
    determinism["runs_compared"] = 2
    determinism["bit_identical"] = bool(c["ok"])
    events.append({"name": "clean_rerun_identical", "expect_detected": False,
                   "detected": not c["ok"], "first_div": c["first_div"],
                   "n_match": c["n_match"]})
    log(f"negctl (1) clean re-run: identical={c['ok']} (must be True)")

    flip, _ = run_prompts(model, prompts, args.max_new, chunk=args.chunk,
                          perturb={"kind": "argmax", "step": args.flip_step})
    c = pc.compare_tokens(base["tokens"], flip[0]["tokens"])
    detected, exact = not c["ok"], c["first_div"] == args.flip_step
    events.append({"name": f"argmax_flip@{args.flip_step}", "expect_detected": True,
                   "detected": detected, "first_div": c["first_div"],
                   "exact_index": exact, "n_match": c["n_match"]})
    log(f"negctl (2) flip @{args.flip_step}: detected={detected} "
        f"first_div={c['first_div']} exact={exact}")

    noise, rel_used, ladder = None, None, []
    for rel in args.noise_ladder:
        r, _ = run_prompts(model, prompts, args.max_new, chunk=args.chunk,
                           perturb={"kind": "noise", "rel": rel})
        c = pc.compare_tokens(base["tokens"], r[0]["tokens"])
        ladder.append({"rel": rel, "detected": not c["ok"], "first_div": c["first_div"],
                       "n_match": c["n_match"]})
        log(f"negctl (3) embed noise rel={rel:g}: detected={not c['ok']} "
            f"first_div={c['first_div']}")
        if not c["ok"] and rel_used is None:
            rel_used = rel
    events.append({"name": "embed_noise", "expect_detected": True,
                   "detected": any(e["detected"] for e in ladder),
                   "threshold_rel": rel_used, "ladder": ladder,
                   "policy": {"low_rung_must_pass": pc.NEGCTL_LOW_MUST_PASS,
                              "high_rung_must_detect": pc.NEGCTL_HIGH_MUST_DETECT}})
    low, high = ladder[0], ladder[-1]
    if rel_used is None:
        log("negctl (3) WARNING: no ladder level changed the tokens: this subset "
            "is insensitive to embedding noise at <= "
            f"{max(e['rel'] for e in ladder):g}")
    else:
        log(f"negctl (3) detection threshold: rel={rel_used:g} "
            f"(ladder {[round(e['rel'], 3) for e in ladder]})")
    log(f"negctl (3) policy: low rung {low['rel']:g} quiet={not low['detected']}, "
        f"high rung {high['rel']:g} detected={high['detected']}")
    del model
    mx.clear_cache()

    # ---- (4) the isolated cos gate must fail on a stubbed component --------
    cos_layers = complete_layers_read(args, pc.parse_layers(args.cos_layers, 40))
    clean_cos = cos_sweep(args, cos_layers)
    apply_stub(args.stub)
    stub_cos = cos_sweep(args, cos_layers)
    per_layer = [{"layer": a["layer"], "clean": a["cos"], "stub": b["cos"]}
                 for a, b in zip(clean_cos["layers"], stub_cos["layers"])]
    events.append({"name": f"cos_stub:{args.stub}", "expect_detected": True,
                   "detected": not stub_cos["gate"]["ok"],
                   "clean_ok": clean_cos["gate"]["ok"],
                   "worst_cos_clean": clean_cos["gate"]["worst_cos"],
                   "worst_cos_stub": stub_cos["gate"]["worst_cos"],
                   "per_layer": per_layer})
    log(f"negctl (4) stub={args.stub}: clean cos "
        f"ok={clean_cos['gate']['ok']} ({clean_cos['gate']['worst_cos']:.6f}) -> "
        f"stub ok={stub_cos['gate']['ok']} ({stub_cos['gate']['worst_cos']:.6f})")

    verdict = pc.negctl_verdict([{k: e[k] for k in
                                  ("name", "expect_detected", "detected")}
                                 for e in events], low_rung=low, high_rung=high)
    flip_ok = next(e["exact_index"] for e in events if e["name"].startswith("argmax"))
    checks = {"negctl_all": verdict["ok"], "flip_exact_index": flip_ok,
              "ladder_low_rung_quiet": not low["detected"],
              "ladder_high_rung_detected": high["detected"]}
    payload = {"events": events, "base_tokens": base["tokens"],
               "clean_cos": clean_cos["gate"], "stub_cos": stub_cos["gate"],
               "built": {k: v for k, v in binf.items() if k != "report"},
               "peak_gb": peak_gb(), "determinism": determinism}
    log(f"negctl verdict: all={verdict['ok']} flip_exact={flip_ok} "
        f"low_quiet={not low['detected']} high_detected={high['detected']}")
    return emit("negctl", all(checks.values()), checks, payload)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DSv4.1 greedy parity harness")
    ap.add_argument("mode", choices=("cos", "record", "check", "negctl"))
    ap.add_argument("--layers", default=None,
                    help=f"layer spec for the run (default: cos {COS_LAYERS_DEF!r}, "
                         f"else {RUN_LAYERS_DEF!r}); 'all' = 0..39")
    ap.add_argument("--cos-layers", default=COS_LAYERS_DEF,
                    help="layers for the isolated cos sweep inside negctl/check")
    ap.add_argument("--isolate", type=int, default=1,
                    help="cos mode: 1 = feed the reference input per layer (gated); "
                         "0 = chain h through the built subset (report-only)")
    ap.add_argument("--dist", default="none", choices=("none", "jaccl"))
    ap.add_argument("--world", type=int, default=None,
                    help="TP world for the build; single node simulates TP without "
                         "a collective (default: 2 for token runs, 1 for cos)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--cos-world", type=int, default=None,
                    help="TP world for the cos sweep's block builds (default 1: the "
                         "p30 trace is world-1 full-width; a sliced build is not "
                         "comparable to it)")
    ap.add_argument("--cos-rank", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("P_MODEL", DEF_MODEL))
    ap.add_argument("--native", default=os.environ.get("P_NATIVE", DEF_NATIVE))
    ap.add_argument("--token-map", default=os.environ.get("P_TOKEN_MAP", DEF_TOKEN_MAP))
    ap.add_argument("--trace", default=os.environ.get("P_TRACE", DEF_TRACE))
    ap.add_argument("--ids", default=os.environ.get("P_IDS", DEF_IDS))
    ap.add_argument("--pkg-root", default=os.environ.get("P_PKG",
                                                         os.path.join(HOME, "dsv41-ws2", "P")))
    ap.add_argument("--prompts", default="p30", choices=("p30", "chat"),
                    help="p30 (prefixes of p30_prompt_ids.json) or chat (real "
                         "chat-template prompts from the checkpoint)")
    ap.add_argument("--prompt-file", default=None,
                    help="json of prompts: [[ids...], ...] or [{name,ids}, ...]")
    ap.add_argument("--prompt-count", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=0,
                    help="chunked prompt prefill via prefill.prefill (0 = one forward)")
    ap.add_argument("--ref", default=os.environ.get("P_REF", DEF_REF))
    ap.add_argument("--note", default="")
    ap.add_argument("--overwrite-ref", action="store_true",
                    help="record mode: allow replacing an existing reference "
                         "(refs are baselines; overwriting one is deliberate)")
    ap.add_argument("--allow-code-drift", action="store_true",
                    help="check mode: comparing across code revisions is a FAIL by "
                         "default; this downgrades it to a reported warning")
    ap.add_argument("--margins", action="store_true",
                    help="record mode: also store the fp32 top-1/top-2 logit gap "
                         "per step (tells a float tie from a real divergence when "
                         "a TP reduction order changes an argmax)")
    ap.add_argument("--perturb", default=None,
                    help="check mode: inject into prompt 0 (argmax:k|noise:r) so the "
                         "comparison must FAIL")
    ap.add_argument("--flip-step", type=int, default=2,
                    help="negctl: generated index to flip (1-based into the token "
                         "stream; divergence must start exactly there)")
    ap.add_argument("--noise", type=float, default=0.01,
                    help="negctl: starting relative embedding noise (the control "
                         "escalates the ladder until the tokens change)")
    ap.add_argument("--noise-ladder", type=float, nargs="*", default=None,
                    help="negctl: explicit relative noise ladder; default 8x from "
                         "--noise (0.01 0.08 0.64)")
    ap.add_argument("--stub", default="experts", choices=STUB_CHOICES,
                    help="negctl: component stub for the cos control")
    ap.add_argument("--max-gb", type=float, default=8.0,
                    help="single-node weight/peak budget; refuse or abort above it")
    ap.add_argument("--require-free-gb", type=float, default=None,
                    help="pre-flight: fail unless this much memory is free "
                         "(production must be stopped for two-node runs)")
    ap.add_argument("--force-oversize", action="store_true")
    args = ap.parse_args()

    if args.layers is None:
        args.layers = COS_LAYERS_DEF if args.mode == "cos" else RUN_LAYERS_DEF
    if args.world is None:
        args.world = 1 if args.mode == "cos" else 2
    if args.cos_world is None:
        args.cos_world = 1
    if args.noise_ladder is None:
        args.noise_ladder = [args.noise * (8.0 ** i) for i in range(3)]
    if args.force_oversize:
        args.max_gb = float("inf")
    if args.cos_world != 1 and args.isolate and args.mode in ("cos", "negctl"):
        log(f"WARNING: --cos-world {args.cos_world} != 1: the p30 trace is a "
            f"world-1 full-width run, so a sliced build will NOT match it")
    log(f"mode={args.mode} argv={' '.join(sys.argv[1:])}")
    log(f"pkg={args.pkg_root} model={args.model}")
    log("REMINDER: run under lockf -k ~/dsv41-gpu.lock, one job per node, "
        "production untouched")
    free_mem_check(args)
    try:
        code = {"cos": mode_cos, "record": mode_record,
                "check": mode_check, "negctl": mode_negctl}[args.mode](args)
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        die(f"{type(e).__name__}: {e}")
    sys.exit(code)


if __name__ == "__main__":
    main()
