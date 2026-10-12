#!/usr/bin/env python3
"""indexer_split.py -- DSv4.1 indexer ABSOLUTE-COST bench (workstream plan Phase 1A + 1B).

Offline, single-node (studio2), idle-guarded, NO node-side edits, NO RDMA (all_sum
stubbed identity -- see below), NO generation, no cluster launch.

Modes
-----
  parity     bitwise/set sanity across the THREE paths (untiled / tiled-buffer / HIER)
             on synthetic tensors at small shapes.  Trust gate before any timing.
  roofline   MEASURED peaks on THIS node: stream bandwidth (copy of a 1 GB array)
             and fp16 matmul FLOP/s.  No spec-sheet numbers.
  split      1A: per-sub-op barriered time split vs unbarriered total, BOTH paths,
             at per-rank post-qsplit shapes (rows n_local = 1024), L in the P1_CTXS
             list; roofline class per sub-op; validity check (sum of parts vs total).
  crossover  1B: REF vs HIER map over rows x nb, unbarriered, ABAB, median+IQR,
             peak-memory delta per config; speedup table + S-threshold fit.
  all        parity, roofline, split, crossover in one process (not recommended;
             separate processes keep the memory/reset story clean).

Everything it calls is the REAL production function (mlx_lm/models/deepseek_v41:
indexer._tiled_scores_buffer / ._tiled_scores_merge / indexer.topk_from_row /
indexer.select_candidate_blocks / indexer.select... / indexer.tiled /
indexer_hierarchical.coarse_block_scores / .top_blocks / .exact_rescore_streaming /
.candidate_mask_from_block_maxima / .hierarchical_topk_prod).  Nothing is
reimplemented; the only harness-side code is (a) the *loop* that feeds real
per-tile calls (the same loop the production caller runs), and (b) the timing
fences.  Weight loading uses exl3_build._dense/_plain (real loaders), NOT
build_block, so the layer's routed experts are never loaded (memory).

Run (on a Mac, from this worktree, via the driver run1.sh):
    env -u PYTHONPATH <venv>/bin/python bench/indexer_split.py split
Env: see P1_* below + the production-relevant MLX/DSV41 env the driver sets.

LABELS: every number this script prints is MEASURED (timed) or DERIVED (arithmetic
from shapes) -- the JSON says which.  Memory-gated skips are recorded as
"not run (mem)".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as stats
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# env BEFORE mlx import (mirrors /private/tmp/p20_idxsplit_rebench/run.sh and the
# live runner env captured 2026-10-12 via `ps eww <runner pid>`)
# ---------------------------------------------------------------------------
os.environ.setdefault("MTL_DISABLE_TIMEOUT", "1")
os.environ.setdefault("MTL_COMMAND_BUFFER_TIMEOUT", "0")
os.environ.setdefault("MLX_GPU_TIME", "1")
os.environ.setdefault("DSV41_DENSE", "affine6")
os.environ.setdefault("DSV41_DENSE_POLICY",
                      "layers.*.ffn.shared_experts.*=q8g64")
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "200")
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")   # EXO_FAST_SYNCH=1 live
# NOTE: DSV41_INDEXER_* knobs are read from the driver env (parity of arms).

_repo = os.environ.get("P1_MLXLN") or os.environ.get("P20_PMP_MLXLN")
if _repo:
    sys.path.insert(0, _repo)

import numpy as np                                  # noqa: E402
import mlx.core as mx                               # noqa: E402

from mlx_lm.models.deepseek_v41 import indexer as IX             # noqa: E402
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as HC  # noqa: E402
from mlx_lm.models.deepseek_v41 import exl3_build as EB          # noqa: E402
from mlx_lm.models.deepseek_v41.config import ModelArgs          # noqa: E402
from mlx_lm.models.deepseek_v41.layers import rope_freqs         # noqa: E402

HAVE_GPU_TIME = hasattr(mx.metal, "gpu_time_ns")
HOME = os.path.expanduser("~")
MODEL_DIR = os.environ.get(
    "P1_MODEL", HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
NATIVE_DIR = os.environ.get(
    "P1_NATIVE", HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram")
OUT = os.environ.get("P1_OUT", "/private/tmp/idxprof1/raw")
LOG = os.environ.get("P1_LOG", HOME + "/exo.log")

CHUNK = int(os.environ.get("P1_CHUNK", "2048"))
WARM = int(os.environ.get("P1_WARM", "3"))
REPS = int(os.environ.get("P1_REPS", "7"))
CACHE_GB = float(os.environ.get("P1_CACHE_GB", "3"))
SCRATCH_GB = float(os.environ.get("P1_SCRATCH_GB", "10"))
IDLE_S = float(os.environ.get("P1_IDLE_S", "60"))
IDLE_POLL = float(os.environ.get("P1_IDLE_POLL", "1.0"))
GPU_BUSY_PCT = int(os.environ.get("P1_GPU_BUSY_PCT", "20"))
CAP_FRACTION = float(os.environ.get("P1_CAP_FRACTION", "0.5"))
PAGE = 16384

H_IDX = 32
D_IDX = 128
TOPK = 512
BLOCK = 8
CAND_TOPK_BLOCKS = 2048


# ---------------------------------------------------------------------------
# small utils
# ---------------------------------------------------------------------------
def log(*a):
    print("[p1]", *a, flush=True)


def now():
    return time.strftime("%H:%M:%S")


def provenance():
    """What code + checkpoint this run used (md5 of the two files under test)."""
    import hashlib
    prov = dict(model_dir=MODEL_DIR, mlxln=_repo)
    try:
        prov["git_head"] = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
             "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=30).stdout.strip()
        prov["git_dirty"] = bool(subprocess.run(
            ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
             "status", "--short"], capture_output=True, text=True,
            timeout=30).stdout.strip())
    except Exception as e:                                   # pragma: no cover
        prov["git_head"] = f"unavailable: {e}"
    for tail in ("mlx_lm/models/deepseek_v41/indexer.py",
                 "mlx_lm/models/deepseek_v41/indexer_hierarchical.py",
                 "mlx_lm/models/deepseek_v41/_gates.py"):
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), tail)
        try:
            prov[f"md5:{os.path.basename(tail)}"] = hashlib.md5(
                open(p, "rb").read()).hexdigest()
        except OSError:
            pass
    try:
        for m in ("config.json",):
            p = os.path.join(MODEL_DIR, m)
            if os.path.exists(p):
                prov[f"md5:{m}"] = hashlib.md5(open(p, "rb").read()).hexdigest()
    except OSError:
        pass
    return prov


class Res:
    """Collector: everything printed is also written to OUT/<tag>.json."""

    def __init__(self, tag):
        self.tag = tag
        self.blobs = []
        os.makedirs(OUT, exist_ok=True)
        self.prov = provenance()
        self._resumed = set()
        prev = os.path.join(OUT, f"{tag}.json")
        if os.environ.get("P1_RESUME", "1") == "1" and os.path.exists(prev):
            try:
                with open(prev) as fh:
                    old = json.load(fh)
                keep = [b for b in old.get("blobs", [])
                        if b.get("_name") not in (None, "_final")
                        and "crossover_partial" not in b.get("_name", "")]
                self.blobs = keep
                self._resumed = {b["_name"] for b in keep}
            except (OSError, ValueError):
                pass

    def have(self, name):
        return name in self._resumed

    def add(self, name, obj):
        obj = dict(obj)
        obj["_name"] = name
        self.blobs.append(obj)
        with open(os.path.join(OUT, f"{self.tag}.json"), "w") as fh:
            json.dump(dict(tag=self.tag, pid=os.getpid(), host=os.uname().nodename,
                           mlx=mx.__version__, provenance=self.prov,
                           env={k: os.environ.get(k) for k in (
                               "DSV41_INDEXER_TILE", "DSV41_INDEXER_TILE_MIN_NB",
                               "DSV41_INDEXER_TILE_MB", "DSV41_INDEXER_HIER",
                               "DSV41_INDEXER_HIER_BLOCK", "DSV41_INDEXER_HIER_STRIP",
                               "DSV41_INDEXER_HIER_OVERFETCH", "DSV41_INDEXER_HIER_EXACT_MB",
                               "DSV41_INDEXER_HIER_EXACT_STRIP", "DSV41_INDEXER_CONSUMER_SKIP",
                               "DSV41_INDEXER_ROW_BF16", "DSV41_INDEXER_COMPILE",
                               "DSV41_INDEXER_QSPLIT", "DSV41_SPARSE_FENCE_MIN_ROWS",
                               "DSV41_DENSE", "DSV41_DENSE_POLICY",
                               "MLX_METAL_FAST_SYNCH", "MLX_GPU_TIME",
                               "MLX_MAX_OPS_PER_BUFFER")},
                           blobs=self.blobs), fh, indent=1, default=float)

    def write(self):
        self.add("_final", {})


# ---------------------------------------------------------------------------
# memory accounting (DERIVED)
# ---------------------------------------------------------------------------
def vm_pages():
    """(free, inactive, speculative, purgeable, wired, compressor_pages, page_size) or None."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:
        return None
    d = {}
    for line in out.splitlines():
        m = re.match(r'"?(.+?)"?:\s+(\d+)', line.strip())
        if m:
            d[m.group(1)] = int(m.group(2))
    ps = 16384
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        ps = int(m.group(1))
    return dict(free=d.get("Pages free", 0), inactive=d.get("Pages inactive", 0),
                speculative=d.get("Pages speculative", 0),
                purgeable=d.get("Pages purgeable", 0),
                wired=d.get("Pages wired down", 0),
                compressor=d.get("Pages occupied by compressor", 0),
                page=ps)


def mem_report():
    vm = vm_pages()
    tot = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 0
    if vm is None:
        return dict(available_bytes=None, cap_bytes=None, note="vm_stat unavailable")
    avail = (vm["free"] + vm["inactive"] + vm["speculative"] + vm["purgeable"]) * vm["page"]
    return dict(available_bytes=avail, total_bytes=tot,
                cap_bytes=avail * CAP_FRACTION,
                free_pages=vm["free"], inactive_pages=vm["inactive"],
                wired_pages=vm["wired"], compressor_pages=vm["compressor"],
                page_size=vm["page"], method="vm_stat free+inactive+speculative+purgeable",
                cap_fraction=CAP_FRACTION)


def predict_peak_bytes(n, nb, path, tile=None, strip=4096, estrip=64, row_bf16=True):
    """DERIVED worst-case working set for one indexer call (bytes).

    count only what the path allocates ON TOP of index_k (which the harness
    pre-allocates and the production engine already holds):
      ref:   pre-collapse [b,n,h,tile] fp32 + row [b,n,nb] (row dtype) + argpartition
             int32 [b,n,nb] (argpartition returns a full-width index array) +
             k32 tile buffer if we split the cast out (not in the timed path)
      hier:  coarse [b,n,strip] bf16 + maxima [b,n,nb/8] fp32 + strip score/gather
             [b,n,c] fp32 + gathered keys [b,n,c,d] bf16 + merge [b,n,k*2] fp32
    """
    b = 1
    row = 2 if row_bf16 else 4
    if path == "ref":
        t = tile or nb
        return (b * n * H_IDX * t * 4          # pre-collapse transient (fp32)
                + b * n * nb * row            # stored row
                + b * n * nb * 4              # argpartition int32 indices
                + b * n * TOPK * 8)           # (v,i) k-pair
    st = max(strip, estrip)
    return (b * n * st * 2                # coarse strip score bf16
            + b * n * st * H_IDX * 2      # coarse pre-collapse bf16
            + b * n * ((nb + BLOCK - 1) // BLOCK) * 4        # block maxima fp32
            + b * n * st * 4             # exact strip score fp32
            + b * n * st * D_IDX * 2     # gathered keys bf16
            + b * n * st * D_IDX * 4     # gathered keys fp32 cast
            + b * n * TOPK * 8)          # merge


# ---------------------------------------------------------------------------
# idle guard -- studio2 production-traffic detector
#
# IDLE SIGNAL (documented in the report):
#   * ~/exo.log tail: the runner writes "received chat request" / "runner running"
#     on every generation request and "[DSV41] prefill controls" per prefill chunk.
#     We keep an incremental file offset and flag ANY new line matching
#     (received chat request|runner running|prefill controls|runner idle|
#      TextGeneration|ERROR|CRITICAL|Traceback).
#   * runner CPU-time delta: `ps -o time= -p <pid>` for the multiprocessing.spawn
#     runner child (auto-detected); any advance means the runner is doing work.
#   * ioreg "Device Utilization %" > 0 in three consecutive samples (only used
#     while waiting for idle; during timed runs the two cheap signals above run).
# 60 s of clean observation before a timed run; polled every 1-2 s DURING runs;
# any hit => the sample is DISCARDED and the discard is logged.
# ---------------------------------------------------------------------------
class IdleGuard:
    # Traffic/lifecycle lines the exo runner writes, plus genuine failure shapes.
    # The download coordinator polls HF every ~75 s and logs WARNING + traceback
    # for a model that 404s (pre-existing production noise, not load) -- excluded.
    PAT = re.compile(r"received chat request|runner running|prefill controls|"
                     r"RunnerReady|handle_generation_tasks|"
                     r"Caused GPU Timeout|SIGKILL|out of memory|\| ERROR +\||"
                     r"\| CRITICAL +\|")
    NOISE = re.compile(r"exo\.download\.|download_utils|ResumableShardDownloader|"
                       r"fetch_file_list")

    def __init__(self, path=LOG, poll=IDLE_POLL, logf=None):
        self.path = path
        self.poll = poll
        self.logf = logf
        self.events = []
        self.discards = 0
        self.samples = 0
        try:
            self.off = os.path.getsize(path)
        except OSError:
            self.off = 0
        self.runner_pid = self._find_runner()
        self._last_cpu = self._runner_cpu()
        self.own_work_ts = 0.0        # last time THIS process put work on the GPU

    def own_work(self):
        """Call after a timed group: the GPU-util reading is unattributable for
        ~30 s afterwards (it cannot tell our own kernels from production's), so
        during that window the guard relies on the log tail + runner CPU only."""
        self.own_work_ts = time.time()

    # -- helpers -----------------------------------------------------------
    def _find_runner(self):
        try:
            out = subprocess.run(["ps", "-Ao", "pid,rss,command"], capture_output=True,
                                 text=True, timeout=20).stdout
        except Exception:
            return None
        best = None
        for line in out.splitlines():
            if "multiprocessing.spawn" in line and "python" in line:
                try:
                    pid, rss = int(line.split()[0]), int(line.split()[1])
                except (ValueError, IndexError):
                    continue
                if best is None or rss > best[1]:
                    best = (pid, rss)
        return best[0] if best else None

    def _runner_cpu(self):
        if self.runner_pid is None:
            return None
        try:
            out = subprocess.run(["ps", "-o", "time=", "-p", str(self.runner_pid)],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
        if not out:
            self.runner_pid = self._find_runner()      # runner restarted?
            return None
        parts = out.split(":")
        try:
            if len(parts) == 3:
                h, m, s = parts
            elif len(parts) == 2:
                h, (m, s) = 0, parts
            else:
                return None
            return int(h) * 3600 + int(m) * 60 + float(s)
        except ValueError:
            return None

    def _gpu_busy(self):
        """Max 'Device Utilization %' across accelerators; None if unreadable."""
        try:
            out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                                 capture_output=True, text=True, timeout=20).stdout
        except Exception:
            return None
        vals = [int(m) for m in re.findall(r'"Device Utilization %"=(\d+)', out)]
        return max(vals) if vals else None

    def _tail_hits(self):
        hits = []
        try:
            sz = os.path.getsize(self.path)
        except OSError:
            return hits
        if sz < self.off:                     # rotated/truncated
            self.off = 0
        if sz == self.off:
            return hits
        with open(self.path, "rb") as fh:
            fh.seek(self.off)
            data = fh.read(sz - self.off).decode("utf-8", "replace")
            self.off = sz
        for line in data.splitlines():
            if self.PAT.search(line) and not self.NOISE.search(line):
                hits.append((now(), line[:160]))
        return hits

    def _note(self, msg):
        self.events.append((now(), msg))
        if self.logf:
            self.logf.write(f"{now()} {msg}\n")
            self.logf.flush()

    def check(self):
        """One contamination poll -> list of evidence strings (empty = clean)."""
        ev = [f"log: {l}" for _, l in self._tail_hits()]
        cpu = self._runner_cpu()
        if cpu is not None and self._last_cpu is not None and cpu > self._last_cpu + 1.0:
            ev.append(f"runner cpu +{cpu - self._last_cpu:.1f}s")
        self._last_cpu = cpu
        return ev

    def wait_idle(self, window=None, deadline_s=1800):
        """Block until the node looks idle continuously for `window` seconds.

        Busy = a request/lifecycle log line, the runner's CPU time advancing, or
        GPU 'Device Utilization %' >= GPU_BUSY_PCT (a 1-2 % single-sample blip
        from the desktop daemons is not activity; real kernels read 40-100 %).
        """
        window = window or IDLE_S
        t0 = time.time()
        clean_since = None
        while time.time() - t0 < deadline_s:
            ev = self.check()
            gpu = self._gpu_busy()
            own_recent = (time.time() - self.own_work_ts) < 30.0
            busy = bool(ev) or (not own_recent and gpu is not None
                                and gpu >= GPU_BUSY_PCT)
            if busy:
                clean_since = None
                self._note("idle-wait: " + ("; ".join(ev) if ev else
                                           f"gpu util {gpu}% >= {GPU_BUSY_PCT}"))
            elif clean_since is None:
                clean_since = time.time()
            elif time.time() - clean_since >= window:
                self._note(f"idle OK ({window:.0f}s clean)")
                return True
            time.sleep(self.poll)
        self._note(f"idle-wait TIMEOUT after {deadline_s}s")
        return False
        # note: the deadline is wall-clock, polled every IDLE_POLL s (no dumb sleeps)

    def clean(self):
        ev = self.check()
        return (not ev), ev


# ---------------------------------------------------------------------------
# weights + inputs
# ---------------------------------------------------------------------------
CK = None
ARGS = None


def load_args():
    global ARGS
    if ARGS is None:
        from mlx_lm.models.exl3.loader import Exl3Checkpoint
        global CK
        CK = Exl3Checkpoint(MODEL_DIR)
        ARGS = ModelArgs.from_dict(CK.config)
        log(f"args: n_layers={ARGS.n_layers} dim={ARGS.dim} h_idx={ARGS.index_n_heads} "
            f"d={ARGS.index_head_dim} topk={ARGS.index_topk} "
            f"kv_src={ARGS.kv_source_layers} idx_src={ARGS.index_source_layers}")
    return ARGS


_FP32_TAILS = ("norm.weight", "attn_sink", "hc_", "gate.bias", "q_weight", "k_weight")


def build_indexer(layer_id, rank=0, world=2):
    """Indexer with REAL checkpoint weights (exl3_build loaders; no experts loaded).

    Names are loaded RELATIVE to the Indexer module (``wq_b``, ``wk``,
    ``weights_proj.weight``, ``k_norm.weight``) -- the same tensors build_block
    loads for the ``layers.N.attn.indexer.`` prefix; the dense groups go through
    exl3_build._dense (Exl3Proj / AffineProj per DSV41_DENSE, as production),
    the plain ones through exl3_build._plain (fp32 for the norm).
    """
    args = load_args()
    ix = IX.Indexer(args, layer_id)
    pre = f"layers.{layer_id}.attn.indexer."
    groups = [g for g in EB._groups(CK, pre)]
    items = []
    for g in groups:
        items.append((g[len(pre):], EB._dense(CK, g)))
    for k in CK.index:
        if not k.startswith(pre):
            continue
        tail = k[len(pre):]
        if tail.rsplit(".", 1)[-1] in ("trellis", "suh", "svh", "mul1"):
            continue
        if f"{pre}{tail}" in groups:
            continue
        dt = mx.float32 if any(s in tail for s in _FP32_TAILS) else None
        items.append((tail, EB._plain(CK, k, dt)))
    ix.load_weights([(t, m) for t, m in items], strict=False)
    mx.eval([v for _, v in items if isinstance(v, mx.array)])
    got = {t for t, _ in items}
    tgt = f"layers.{layer_id}.attn."
    want = {"weights_proj.weight", "k_norm.weight"} if layer_id in (2, 8, 14, 20) else \
           {"weights_proj.weight"}
    if ix.owns_k:
        want |= {"wq_b", "wk"}
    missing = want - got
    if missing:
        raise RuntimeError(f"L{layer_id} indexer weights missing: {sorted(missing)}")
    # the dense groups must have been substituted by real exl3/affine modules
    for t in ("wq_b", "wk"):
        if t in got:
            mod = dict((a, b) for a, b in items)[t]
            if isinstance(mod, mx.array):
                raise RuntimeError(f"L{layer_id} {t} loaded as a plain array, not a proj")
    kinds = {t: type(m).__name__ for t, m in items}
    log(f"L{layer_id} indexer loaded: {kinds}")
    return ix


def _tree(params, prefix=""):
    if isinstance(params, dict):
        for k, v in params.items():
            yield from _tree(v, f"{prefix}{k}.")
    elif isinstance(params, list):
        for i, v in enumerate(params):
            yield from _tree(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], params


def freqvec_for(layer_id, ratio):
    args = load_args()
    if ratio:
        orig_len, theta = args.original_seq_len, args.compress_rope_theta
    else:
        orig_len, theta = 0, args.rope_theta
    return rope_freqs(args.rope_head_dim, orig_len, theta, args.rope_factor,
                      args.beta_fast, args.beta_slow)


class Shared:
    def __init__(self, cand=None):
        self.kv_src_cache = None
        self.index_src_cache = None
        self.topk_idxs = None
        self.candidates = cand


def synth_inputs(n, nb, seed=1, dtype_k=mx.bfloat16, lens_scale=1.0):
    """Random x/qr/index_k.  TIMING ONLY (never for tie analysis)."""
    mx.random.seed(seed)
    args = load_args()
    x = (mx.random.normal((1, n, args.dim)) * 0.3).astype(mx.bfloat16)
    qr = (mx.random.normal((1, n, args.q_lora_rank)) * 0.3).astype(mx.bfloat16)
    ik = (mx.random.normal((1, nb, args.index_head_dim)) * 0.5).astype(dtype_k)
    mx.eval(x, qr, ik)
    return x, qr, ik


def band_qw(ix, x, qr, start_pos, lo, hi, fv):
    """The band's (q32, w32, lens) exactly as Indexer.__call__ derives them.

    Uses the indexer's own projections on the FULL chunk then the band slice
    (production D2 fix: projections always run on the full chunk).  Harness-side
    duplicate of 4 lines of __call__ -- used only to FEED the part-level calls.
    """
    q_full = ix.wq_b(qr)
    w_full = ix.weights_proj(x)
    q_b, w_b = q_full[:, lo:hi, :], w_full[:, lo:hi, :]
    qpos = mx.arange(start_pos + lo, start_pos + hi)
    n = qpos.shape[0]
    q = q_b.reshape(1, n, ix.n_heads, ix.head_dim)
    from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0
    from mlx_lm.models.deepseek_v41.layers import cos_sin_at, rope_tail
    q = rope_tail(q, ix.rope_head_dim, *cos_sin_at(fv, qpos))
    q = fake_quant_fp4_ue8m0(q, 32)
    w = w_b * (ix.softmax_scale * ix.n_heads ** -0.5)
    lens = ((qpos + 1) // ix.ratio)[:, None]
    mx.eval(q, w, lens)
    return q, w, lens, qpos


def synth_cand_mask(n, nb, start_pos, ratio, qpos, seed=3):
    """Block-aligned SYNTHETIC candidate mask with production-like density.

    candidate_topk_blocks=2048 blocks of 8 = 16384 columns of nb, per row.
    Labelled synthetic; used only for CONSUMER-layer timing (24/28/32/36).
    """
    rng = np.random.RandomState(seed)
    nb_blocks = (nb + BLOCK - 1) // BLOCK
    keep_blocks = min(CAND_TOPK_BLOCKS, nb_blocks)
    m = np.zeros((1, n, nb), dtype=bool)
    for r in range(n):
        vis = int((int(qpos[r]) + 1) // ratio)          # visible columns for this row
        ok = min(nb_blocks, (vis + BLOCK - 1) // BLOCK)
        if ok <= 0:
            continue
        sel = rng.choice(ok, size=min(keep_blocks, ok), replace=False)
        for b in sel:
            c0, c1 = b * BLOCK, min((b + 1) * BLOCK, nb)
            m[0, r, c0:c1] = True
    return mx.array(m)


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------
def drain(out):
    if isinstance(out, (tuple, list)):
        mx.eval(*[o for o in out if isinstance(o, mx.array)])
    elif isinstance(out, mx.array):
        mx.eval(out)


def timed(fn, reps, warm, guard=None, gpu=False, label="", reset_peak=False):
    """Median + IQR of `reps` unbarriered runs, `warm` warmups, contamination-policed."""
    res = dict(label=label, reps_used=0, discards=0, wall_all=[], gpu_all=[],
               peak_all=[], warm=warm)
    if reset_peak:
        mx.reset_peak_memory()
    base = mx.get_active_memory()
    for i in range(warm + reps):
        if guard is not None and i >= warm:
            ok, ev = guard.clean()
            if not ok:
                res["discards"] += 1
                guard.discards += 1
                guard._note(f"DISCARD {label} rep {i}: " + "; ".join(ev))
                continue
        mx.synchronize()
        if gpu and HAVE_GPU_TIME:
            mx.metal.reset_gpu_time()
        if reset_peak:
            mx.reset_peak_memory()
        t0 = time.perf_counter()
        out = fn()
        drain(out)
        mx.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        if i >= warm:
            res["wall_all"].append(dt)
            if gpu and HAVE_GPU_TIME:
                res["gpu_all"].append(mx.metal.gpu_time_ns() / 1e6)
            if reset_peak:
                res["peak_all"].append((mx.get_peak_memory() - base) / 1e6)
        del out
    w = res["wall_all"]
    res["reps_used"] = len(w)
    res["ms"] = stats.median(w) if w else float("nan")
    if len(w) >= 4:
        q = np.percentile(w, [25, 75])
        res["iqr_ms"] = float(q[1] - q[0])
    else:
        res["iqr_ms"] = float("nan")
    res["min_ms"] = min(w) if w else float("nan")
    if res["gpu_all"]:
        res["gpu_ms"] = stats.median(res["gpu_all"])
    if res["peak_all"]:
        res["peak_mb"] = max(res["peak_all"])
    return res


class Barrier:
    """mx.eval+mx.synchronize fence for barriered sub-op timing."""

    @staticmethod
    def fence(*arrs):
        mx.eval(*[a for a in arrs if isinstance(a, mx.array)])
        mx.synchronize()


# ---------------------------------------------------------------------------
# PARITY -- trust gate (three paths, small shapes)
# ---------------------------------------------------------------------------
def cmd_parity(res: Res):
    args = load_args()
    shapes = [tuple(int(v) for v in s.split(":")) for s in
              os.environ.get("P1_PARITY_SHAPES", "32:1024:8,78:4096:20,512:16384:20,"
                                              "32:4096:24,78:16384:24,512:2048:8").split(",")]
    out = []
    for (n, nb, layer) in shapes:
        ix = build_indexer(layer)
        ratio = ix.ratio
        sp = nb * ratio - n
        fv = freqvec_for(layer, ratio)
        x, qr, ik = synth_inputs(n, nb, seed=5)
        lens_np = np.array(((np.arange(n) + sp + 1) // ratio)[:, None])
        # fp32 stored row so all three paths must agree BITWISE on the row
        saved = (IX._ROW_DTYPE, IX._L2_FULL, IX._SMALLN_ROW_BF16)
        IX._ROW_DTYPE, IX._L2_FULL, IX._SMALLN_ROW_BF16 = mx.float32, True, True
        try:
            cand = None
            if layer == 24 or ix.uses_candidates:
                cand = synth_cand_mask(n, nb, sp, ratio, np.arange(n) + sp)
            # --- path A: untiled
            IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 0, 0, False
            IX._HIER = False
            sh_a = Shared(cand)
            ia = ix(x, qr, sp, 0, fv, ik, sh_a)
            # --- path B: tiled buffer
            IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 512, 0, True
            IX._TILE_FORCE = True
            IX._HIER = False
            sh_b = Shared(cand)
            ib = ix(x, qr, sp, 0, fv, ik, sh_b)
            IX._TILE_FORCE = False
            # --- path C: HIER
            IX._TILE, IX._TILE_MIN_NB = 512, 0
            IX._HIER = True
            sh_c = Shared(cand)
            ic = ix(x, qr, sp, 0, fv, ik, sh_c)
            # --- path D: HIER forced at small n (diagnostic, not production)
            IX._FENCE_MIN_ROWS = -1
            sh_d = Shared(cand)
            id_ = ix(x, qr, sp, 0, fv, ik, sh_d)
            IX._FENCE_MIN_ROWS = int(os.environ.get("DSV41_SPARSE_FENCE_MIN_ROWS", "16"))
            mx.eval(ia, ib, ic, id_)
        finally:
            IX._ROW_DTYPE, IX._L2_FULL, IX._SMALLN_ROW_BF16 = saved
            IX._HIER = True

        def rowsets(t):
            a = np.array(t)[0]
            return [set(int(c) for c in a[r] if c >= 0) for r in range(a.shape[0])]

        sa, sb, sc, sd = rowsets(ia), rowsets(ib), rowsets(ic), rowsets(id_)
        eq_ab = sum(1 for u, v in zip(sa, sb) if u == v)
        eq_ac = sum(1 for u, v in zip(sa, sc) if u == v)
        eq_ad = sum(1 for u, v in zip(sa, sd) if u == v)
        # values: the hoisted k-th value must be the true top-k value multiset
        ka = np.sort(np.array(ia)[0], axis=-1)
        kc = np.sort(np.array(ic)[0], axis=-1)
        rec = dict(shape=f"n={n},nb={nb},L{layer}", ratio=ratio, nb=nb, n=n,
                   index_set_equal_untiled_tiled=f"{eq_ab}/{n}",
                   index_set_equal_untiled_hier=f"{eq_ac}/{n}",
                   index_set_equal_untiled_hierFORCED=f"{eq_ad}/{n}",
                   same_selected_indices_all3=bool((ka == kc).all()),
                   path=f"row_dtype=fp32(L2full) tile=512 force")
        log(f"parity {rec['shape']}: A=untiled B=tiled C=hier D=hier-forced | "
            f"A==B {eq_ab}/{n}  A==C {eq_ac}/{n}  A==D {eq_ad}/{n}  "
            f"same_indices(A,C)={rec['same_selected_indices_all3']}")
        out.append(rec)
        mx.clear_cache()
    res.add("parity", dict(rows=out))
    log("PARITY_DONE")


# ---------------------------------------------------------------------------
# ROOFLINE -- measured peaks on THIS node
# ---------------------------------------------------------------------------
def cmd_roofline(res: Res, gb=1.0, matmul=4096):
    n_el = int(gb * (1 << 30) // 4)
    x = mx.random.normal((n_el,)).astype(mx.float32)
    y = mx.random.normal((n_el,)).astype(mx.float32)
    mx.eval(x, y)
    for _ in range(2):
        z = mx.array(x); mx.eval(z); mx.synchronize()
    ts = []
    for _ in range(7):
        mx.synchronize(); t0 = time.perf_counter()
        z = mx.array(x)
        mx.eval(z); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)          # ms
    copy_all = list(ts)
    copy_ms = stats.median(ts)
    copy_bw = 2 * n_el * 4 / (copy_ms / 1e3) / 1e9           # read+write GB/s
    ts = []
    for _ in range(7):
        mx.synchronize(); t0 = time.perf_counter()
        z = x + y
        mx.eval(z); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    add_ms = stats.median(ts)
    add_bw = 3 * n_el * 4 / (add_ms / 1e3) / 1e9
    del x, y, z
    mx.clear_cache()
    # fp16 matmul
    a = (mx.random.normal((matmul, matmul)) * 0.05).astype(mx.float16)
    b = (mx.random.normal((matmul, matmul)) * 0.05).astype(mx.float16)
    mx.eval(a, b)
    for _ in range(2):
        c = mx.matmul(a, b); mx.eval(c); mx.synchronize()
    ts = []
    for _ in range(5):
        mx.synchronize(); t0 = time.perf_counter()
        c = mx.matmul(a, b)
        mx.eval(c); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    mm_ms = stats.median(ts)
    mm_all = list(ts)
    flops = 2 * matmul ** 3
    # shape-specific ceiling: the indexer's OWN score expression at production
    # widths (n=1024 rows, h=32, d=128, tile=504/512) -- the relevant upper bound.
    qq = (mx.random.normal((1, 1024, 32, 128)) * 0.3).astype(mx.float32)
    kk = (mx.random.normal((1, 504, 128)) * 0.5).astype(mx.float32)
    mx.eval(qq, kk)
    body = IX._score_body_eager
    for _ in range(2):
        ss = body(qq, kk, mx.ones((1, 1024, 32))); mx.eval(ss)
    ts = []
    for _ in range(7):
        mx.synchronize(); t0 = time.perf_counter()
        ss = body(qq, kk, mx.ones((1, 1024, 32)))
        mx.eval(ss); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    idx_ms = stats.median(ts)
    idx_flops = 2 * 1024 * 504 * 32 * 128
    rec = dict(copy_gb=gb, copy_ms=copy_ms, stream_bw_gbs=copy_bw, copy_all_ms=copy_all,
               add_ms=add_ms, add_bw_gbs=add_bw,
               matmul_n=matmul, matmul_ms=mm_ms,
               tflops=flops / (mm_ms / 1e3) / 1e12,
               matmul_all_ms=mm_all,
               idx_shape=dict(n=1024, t=504, h=32, d=128),
               idx_shape_ms=idx_ms,
               idx_shape_tflops=idx_flops / (idx_ms / 1e3) / 1e12,
               idx_shape_all_ms=list(ts),
               dtype="fp32 copy/add; fp16 matmul", reps=7, label="MEASURED",
               note="wall over mx.eval+mx.synchronize of one op; copy = mx.array(x) "
                    "1 GB (2 GB moved: 1 read + 1 write)")
    log(f"roofline: copy {gb:.2f}GB {copy_ms:.2f} ms -> {copy_bw:.1f} GB/s (rw) | "
        f"add 3-flop/elt {add_bw:.1f} GB/s | fp16 {matmul}^3 {mm_ms:.2f} ms -> "
        f"{rec['tflops']:.1f} TFLOP/s")
    res.add("roofline", rec)
    return rec


# ---------------------------------------------------------------------------
# 1A -- barriered sub-op split, both paths
# ---------------------------------------------------------------------------
def split_ref(ix, q32, w32, lens, ik, nb, tile, k, offset, band, guard):
    """Barriered parts of the row-materializing (tiled buffer) path.

    A PARTITION of the production work (each part fenced once with
    mx.eval+mx.synchronize at its boundary, so the SUM is comparable to the
    unbarriered total):
      (iii) K-cache read: ``index_k.astype(fp32)`` in ONE op over the whole
            visible cache (what the production per-tile loop does inline on
            each tile; here materialized once so it can be timed apart).
      (i)   score GEMM + head-sum + row write: production's per-tile loop over
            ``indexer._score_body`` (the same loop ``_tiled_scores_buffer``
            runs, including its per-tile ``mx.eval(s)``), reading the fp32 keys
            materialized by (iii).
      (ii)  select: ``indexer.topk_from_row`` over the buffered row.
      (iv)  scatter: mask + (+offset / -1) + ``_finish_rows`` (the padded all_sum
            is STUBBED to identity on this single node -- the pad+copy is timed,
            the collective itself is excluded and bounded in the report).
    """
    b, n = q32.shape[0], q32.shape[1]
    row_dtype = IX._row_dtype(n)
    parts = {}
    # (iii) K-cache read (one op)
    t0 = time.perf_counter()
    k32 = ik.astype(mx.float32)
    Barrier.fence(k32)
    parts["iii_kcache_read"] = (time.perf_counter() - t0) * 1e3
    # (i) score gemm + head sum + row write (production loop shape)
    row = mx.zeros((b, n, nb), dtype=row_dtype)
    body = IX._score_body(b, n, H_IDX, D_IDX, tile)
    t0 = time.perf_counter()
    for c0 in range(0, nb, tile):
        c1 = min(c0 + tile, nb)
        s = body(q32, k32[:, c0:c1], w32)
        cols = mx.arange(c0, c1, dtype=mx.int32)
        s = mx.where(cols[None, :] < lens, s, IX.NEG_INF)
        mx.eval(s)                                   # production's own fence
        row[:, :, c0:c1] = s
    Barrier.fence(row)
    parts["i_score_gemm"] = (time.perf_counter() - t0) * 1e3
    del k32
    # (ii) select
    t0 = time.perf_counter()
    v, i = IX.topk_from_row(row, k)
    Barrier.fence(v, i)
    parts["ii_select_topk"] = (time.perf_counter() - t0) * 1e3
    del row
    # (iv) scatter
    t0 = time.perf_counter()
    valid = mx.isfinite(v) & (i < lens.astype(mx.int32)[None])
    out = IX._finish_rows(mx.where(valid, i + offset, mx.array(-1, mx.int32)),
                          band, CHUNK, None)
    Barrier.fence(out)
    parts["iv_scatter"] = (time.perf_counter() - t0) * 1e3
    return parts, out


def split_hier(ix, q32, w32, lens, ik, nb, estrip, n_full, is_cand_src, cand_mask,
               offset, band, guard):
    """Barriered parts of the HIER path (real functions, production args).

    Mirrors what hierarchical_topk_prod itself calls, including the consumer
    coarse pass (``coarse_block_scores_candidates`` when a candidate mask is
    present and consumer_skip is on) -- otherwise the part sum is inflated on
    the consumer layers 24..36.
    """
    parts = {}
    skip = cand_mask is not None and IX._HIER_CONSUMER_SKIP
    t0 = time.perf_counter()
    if skip:
        bm = HC.coarse_block_scores_candidates(
            q32, ik, w32, lens, cand_mask, block=IX._HIER_BLOCK,
            strip=HC.coarse_gather_strip(IX._HIER_STRIP, H_IDX, D_IDX, IX._HIER_BLOCK),
            dtype=mx.bfloat16)
    else:
        bm = HC.coarse_block_scores(q32, ik, w32, lens, block=IX._HIER_BLOCK,
                                    strip=IX._HIER_STRIP, dtype=mx.bfloat16,
                                    col_mask=cand_mask)
    Barrier.fence(bm)
    parts["i_coarse_sweep"] = (time.perf_counter() - t0) * 1e3
    blk = None
    t0 = time.perf_counter()
    if is_cand_src:
        blk = HC.candidate_mask_from_block_maxima(bm, lens, CAND_TOPK_BLOCKS, IX._HIER_BLOCK)
    k = min(TOPK, nb)
    k_blocks = min(-(-nb // IX._HIER_BLOCK), k + IX._HIER_OVERFETCH)
    blocks = HC.top_blocks(bm, k_blocks, block_size=IX._HIER_BLOCK)
    Barrier.fence(blocks)
    parts["ii_select_blocks"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    v, i = HC.exact_rescore_streaming(q32, ik, w32, lens, blocks, strip=estrip,
                                     k=k, block=IX._HIER_BLOCK, col_mask=cand_mask)
    Barrier.fence(v, i)
    parts["iii_exact_rescore"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    valid = mx.isfinite(v) & (i < lens.astype(mx.int32)[None])
    out = IX._finish_rows(mx.where(valid, i + offset, mx.array(-1, mx.int32)),
                          band, n_full, None)
    Barrier.fence(out)
    parts["iv_scatter"] = (time.perf_counter() - t0) * 1e3
    return parts, out, blk, bm


def classify_part(part, bytes_moved, flops, peaks):
    if bytes_moved <= 0 or part <= 0:
        return "n/a"
    bw_frac = (bytes_moved / (part / 1e3)) / (peaks["stream_bw_gbs"] * 1e9)
    fl_frac = (flops / (part / 1e3)) / (peaks["tflops"] * 1e12)
    if fl_frac >= 0.6 and fl_frac >= bw_frac:
        return f"compute-bound ({fl_frac:.0%} of peak)"
    if bw_frac >= 0.6:
        return f"bandwidth-bound ({bw_frac:.0%} of peak)"
    return f"other (bw {bw_frac:.0%}, flop {fl_frac:.0%})"


def cmd_split(res: Res, peaks):
    args = load_args()
    ctxs = [int(v) for v in os.environ.get("P1_CTXS", "50000,75000,100000,130000").split(",")]
    layers = [int(v) for v in os.environ.get("P1_LAYERS", "8,20,24").split(",")]
    n_full = CHUNK
    n_local = int(os.environ.get("P1_ROWS", str(CHUNK // 2)))
    mem = mem_report()
    log(f"mem: avail={mem['available_bytes']/1e9:.1f} GB cap(50%)={mem['cap_bytes']/1e9:.1f} GB")
    res.add("mem_headroom", mem)
    guard = IdleGuard(logf=open(os.path.join(OUT, "idle.log"), "a"))
    if guard.runner_pid:
        log(f"idle guard: runner pid={guard.runner_pid} log={LOG} window={IDLE_S}s")
    for layer in layers:
        ix = build_indexer(layer)
        ratio = ix.ratio
        fv = freqvec_for(layer, ratio)
        for L in ctxs:
            nb = (L) // ratio if ratio else 0
            if nb < 64:
                continue
            tag = f"L{layer}_L{L}_split"
            if res.have(tag):
                log(f"{tag}: already measured (resume) -- skipping")
                continue
            is_cand = ix.layer_id == args.candidate_source_layer
            is_cons = ix.uses_candidates and not is_cand
            # memory gate (DERIVED, 50% of available)
            align = BLOCK if is_cand else 1
            tile = min(nb, IX.tile_width(1, n_full, H_IDX, nb, align))
            route_tiled = IX.tiled(1, n_full, H_IDX, nb, align) < nb
            estrip = (IX._HIER_EXACT_STRIP if IX._HIER_EXACT_STRIP > 0 else
                       IX.hier_strip_for_budget(1, n_local, D_IDX,
                                                int(IX._HIER_EXACT_MB * (1 << 20))))
            pk_ref = predict_peak_bytes(n_local, nb, "ref", tile=tile)
            pk_hier = predict_peak_bytes(n_local, nb, "hier", strip=IX._HIER_STRIP,
                                         estrip=estrip)
            base = nb * D_IDX * 2 + n_full * (H_IDX * D_IDX * 4 * 2)
            if max(pk_ref, pk_hier) + base > mem["cap_bytes"]:
                res.add(tag, dict(L=L, layer=layer, nb=nb, status="not run (mem)",
                                  predicted_ref_mb=pk_ref / 1e6,
                                  predicted_hier_mb=pk_hier / 1e6,
                                  base_mb=base / 1e6, cap_mb=mem["cap_bytes"] / 1e6))
                log(f"{tag}: not run (mem) pred_ref={pk_ref/1e9:.2f}GB "
                    f"pred_hier={pk_hier/1e9:.2f}GB cap={mem['cap_bytes']/1e9:.1f}GB")
                continue
            start_pos = L - n_full
            lo, hi = 0, n_local
            x, qr, ik = synth_inputs(n_full, nb, seed=11)
            q32, w32, lens, qpos = band_qw(ix, x, qr, start_pos, lo, hi, fv)
            cand = None
            if is_cons:
                cand = synth_cand_mask(n_local, nb, start_pos, ratio, qpos)
                mx.eval(cand)
            shared = Shared(cand)
            offset = min(start_pos, args.window_size) + n_full
            band = (lo, hi)

            IX._HIER = False
            IX._TILE, IX._TILE_MIN_NB = int(os.environ.get("DSV41_INDEXER_TILE", "512")), 0

            # --- unbarriered totals (real __call__, banded, per-rank rows)
            def run_ref():
                return ix(x, qr, start_pos, offset, fv, ik, Shared(cand), band=band)

            def run_hier():
                IX._HIER = True
                try:
                    return ix(x, qr, start_pos, offset, fv, ik, Shared(cand), band=band)
                finally:
                    IX._HIER = False

            guard.wait_idle()
            tot_ref = timed(run_ref, REPS, WARM, guard=guard, gpu=True,
                            label=f"{tag} REF total (tiled buffer)",
                            reset_peak=True)
            tot_hier = timed(run_hier, REPS, WARM, guard=guard, gpu=True,
                             label=f"{tag} HIER total", reset_peak=True)
            # pure untiled reference (diagnostic; production default is tiled)
            IX._TILE = 0
            tot_unt = timed(run_ref, max(3, REPS // 2), max(2, WARM - 1), guard=guard,
                            gpu=True, label=f"{tag} REF pure-untiled", reset_peak=True)
            IX._TILE = int(os.environ.get("DSV41_INDEXER_TILE", "512"))

            # --- barriered part splits (median of >=3, one fence per part)
            def parts_ref():
                return split_ref(ix, q32, w32, lens, ik, nb, tile, min(TOPK, nb),
                                 offset, band, guard)[0]

            # NB: split_hier returns (parts, out, blk, bm)
            def parts_hier_parts():
                return split_hier(ix, q32, w32, lens, ik, nb, estrip, n_full,
                                  is_cand, cand, offset, band, guard)[0]

            seq_ref, seq_hier = [], []
            for _ in range(max(3, REPS // 2)):
                ok, ev = guard.clean()
                if not ok:
                    guard.discards += 1
                    guard._note("DISCARD split_ref: " + "; ".join(ev))
                    continue
                seq_ref.append(parts_ref())
                seq_hier.append(parts_hier_parts())
            agg = lambda seq: {k: stats.median([s[k] for s in seq]) for k in seq[0]} if seq else {}
            pr, ph = agg(seq_ref), agg(seq_hier)

            # --- roofline (DERIVED bytes/flops per part)
            kbytes = nb * D_IDX * 2
            score_bytes = n_local * nb * 4 * 2      # fp32 tile transient write+read
            row_bytes = n_local * nb * 2 * 2
            gemm_flops = 2 * n_local * nb * H_IDX * D_IDX
            roofline = dict(
                stream_bw_gbs=peaks["stream_bw_gbs"], tflops=peaks["tflops"],
                iii_kcache_read=dict(bytes=kbytes,
                                     gbs=kbytes / (pr.get("iii_kcache_read", 1) or 1) / 1e6),
                i_score_gemm=dict(bytes=score_bytes + row_bytes, flops=gemm_flops,
                                  tflops=gemm_flops / (pr.get("i_score_gemm", 1) or 1) / 1e9),
                ii_select_topk=dict(bytes=n_local * nb * 2 + n_local * TOPK * 8,
                                    flops=float(n_local * nb)),
                iv_scatter=dict(bytes=n_local * TOPK * 4 * 3),
            )
            cls = {k: classify_part(pr.get(k, 0), roofline[k]["bytes"],
                                    roofline[k].get("flops", 0), peaks)
                   for k in pr}
            # HIER parts carry their own derived roofline entries
            hbytes = dict(
                i_coarse_sweep=nb * D_IDX * 2 + n_local * nb * 2,
                ii_select_blocks=n_local * ((nb + BLOCK - 1) // BLOCK) * 4,
                iii_exact_rescore=(512 + IX._HIER_OVERFETCH) * BLOCK
                                  * (2 * n_local * D_IDX * 2),
                iv_scatter=n_local * TOPK * 4 * 3)
            hflops = dict(
                i_coarse_sweep=2 * n_local * nb * H_IDX * D_IDX,
                ii_select_blocks=float(n_local * ((nb + BLOCK - 1) // BLOCK)),
                iii_exact_rescore=2 * n_local * (512 + IX._HIER_OVERFETCH) * BLOCK
                                 * H_IDX * D_IDX,
                iv_scatter=0.0)
            hcls = {k: classify_part(ph.get(k, 0), hbytes.get(k, 0),
                                     hflops.get(k, 0.0), peaks) for k in ph}
            sum_parts = sum(pr.values())
            validity = (abs(sum_parts - tot_ref["ms"]) / tot_ref["ms"] <= 0.15
                        if tot_ref["ms"] else False)
            hsum = sum(ph.values())
            hvalid = (abs(hsum - tot_hier["ms"]) / tot_hier["ms"] <= 0.15
                      if tot_hier["ms"] else False)
            rec = dict(L=L, layer=layer, ratio=ratio, nb=nb, n_local=n_local,
                       tile=tile, estrip=estrip, candidate=is_cand, consumer=is_cons,
                       ref_total_ms=tot_ref, ref_untiled_ms=tot_unt,
                       hier_total_ms=tot_hier,
                       ref_parts_ms=pr, ref_parts_sum_ms=sum_parts,
                       ref_validity_within_15pct=validity,
                       ref_label="MEASURED" if validity else "BOUND",
                       hier_parts_ms=ph, hier_parts_sum_ms=hsum,
                       hier_validity_within_15pct=hvalid,
                       hier_label="MEASURED" if hvalid else "BOUND",
                       ref_classes=cls, roofline=roofline)
            log(f"{tag}: ref_total={tot_ref['ms']:.2f} ms (iqr {tot_ref['iqr_ms']:.2f}) "
                f"untiled={tot_unt['ms']:.2f} hier={tot_hier['ms']:.2f} | "
                f"partsΣ ref={sum_parts:.2f} ({'MEASURED' if validity else 'BOUND'}) "
                f"hier={hsum:.2f} ({'MEASURED' if hvalid else 'BOUND'})")
            for k, v in sorted(pr.items(), key=lambda kv: -kv[1]):
                log(f"   ref {k:16s} {v:8.3f} ms  {100*v/max(sum_parts,1e-9):5.1f}%  {cls[k]}")
            for k, v in sorted(ph.items(), key=lambda kv: -kv[1]):
                log(f"   hier {k:16s} {v:8.3f} ms  {100*v/max(hsum,1e-9):5.1f}%")
            res.add(tag, rec)
            guard.own_work()
            del x, qr, ik, q32, w32
            mx.clear_cache()
    res.add("discards", dict(count=guard.discards, events=guard.events[-40:]))
    log("SPLIT_DONE")


# ---------------------------------------------------------------------------
# 1B -- crossover map
# ---------------------------------------------------------------------------
def cmd_crossover(res: Res, peaks):
    args = load_args()
    layer = int(os.environ.get("P1_CROSS_LAYER", "20"))
    rows_grid = [int(v) for v in os.environ.get(
        "P1_CROSS_ROWS", "1,4,8,78,512,1024,2048").split(",")]
    nb_grid = [int(v) for v in os.environ.get(
        "P1_CROSS_NB", "2048,4096,8192,16384,32768,50000,75000,100000,130000").split(",")]
    mem = mem_report()
    cap_b = mem["cap_bytes"] or 0
    guard = IdleGuard(logf=open(os.path.join(OUT, "idle.log"), "a"))
    res.add("mem_headroom", mem)
    ix = build_indexer(layer)
    ratio = ix.ratio
    fv = freqvec_for(layer, ratio)
    is_cand = layer == args.candidate_source_layer
    cells = []
    done = {}
    prev = os.path.join(OUT, f"{res.tag}.json")
    if os.environ.get("P1_RESUME", "1") == "1" and os.path.exists(prev):
        try:
            with open(prev) as fh:
                for b in json.load(fh).get("blobs", []):
                    if b.get("_name") == "crossover_partial":
                        for c in b.get("cells", []):
                            done[(c["n"], c["nb"])] = c
            cells = [done[k] for k in sorted(done)]
            log(f"resume: {len(cells)} crossover cells already measured")
        except (OSError, ValueError, KeyError):
            done, cells = {}, []
    dirty = True                                   # wait for a clean window first
    unopt = os.environ.get("P1_CROSS_UNTILED", "1") == "1"
    for n in rows_grid:
        for nb in nb_grid:
            tag = f"cross_n{n}_nb{nb}"
            if (n, nb) in done:
                log(f"{tag}: already measured (resume) -- skipping")
                continue
            align = BLOCK if is_cand else 1
            # production routes the tiled/untiled decision on the FORWARD's full
            # row count (D3 audit): a banded prefill chunk of n_local=1024 rows
            # rides a 2048-row forward; decode/verify/tail rows ride their own n.
            route_n = CHUNK if n >= 1024 else n
            tiled_now = IX.tiled(1, route_n, H_IDX, nb, align)
            tile = min(nb, IX.tile_width(1, route_n, H_IDX, nb, align))
            estrip = (IX._HIER_EXACT_STRIP if IX._HIER_EXACT_STRIP > 0 else
                       IX.hier_strip_for_budget(1, n, D_IDX,
                                                int(IX._HIER_EXACT_MB * (1 << 20))))
            pk_ref = predict_peak_bytes(n, nb, "ref", tile=tiled_now)
            pk_hier = predict_peak_bytes(n, nb, "hier", strip=IX._HIER_STRIP,
                                         estrip=estrip)
            pk_unt = predict_peak_bytes(n, nb, "ref", tile=nb)
            # the untimed per-cell base: index_k + banded q32/w32 at n rows
            base_b = nb * D_IDX * 2 + n * H_IDX * D_IDX * 4 + n * 4 * (H_IDX + D_IDX)
            if max(pk_ref, pk_hier) + base_b > cap_b:
                cells.append(dict(n=n, nb=nb, status="not run (mem)",
                                  predicted_ref_mb=pk_ref / 1e6,
                                  predicted_hier_mb=pk_hier / 1e6,
                                  predicted_untiled_mb=pk_unt / 1e6,
                                  cap_mb=cap_b / 1e6))
                log(f"{tag}: not run (mem) pred_ref={pk_ref/1e9:.2f}GB "
                    f"pred_hier={pk_hier/1e9:.2f}GB pred_untiled={pk_unt/1e9:.2f}GB "
                    f"cap={cap_b/1e9:.1f}GB")
                continue
            x, qr, ik = synth_inputs(n, nb, seed=13)
            sp = max(0, nb * max(ratio, 1) - CHUNK)
            q32, w32, lens, qpos = band_qw(ix, x, qr, sp, 0, n, fv)
            k = min(TOPK, nb)
            # the production row-materializing route: tiled buffer when the
            # real routing picks a tile < nb, else the whole-row variant
            use_tiled = tiled_now < nb
            fence_saved = IX._FENCE_MIN_ROWS
            do_untiled = unopt and nb <= int(os.environ.get("P1_CROSS_UNTILED_MAX_NB",
                                                            "131072")) \
                and pk_unt + base_b <= cap_b

            def run_ref():
                # PRODUCTION'S OWN ROUTE for this row count: the tiled buffer when
                # tiled() picks a tile < nb, else the whole-row variant (small n /
                # small transient -- what decode, verify and short tails run).
                return IX._tiled_scores_buffer(q32, ik, w32, lens, nb, k,
                                               tile if use_tiled else nb)[:2]

            def run_untiled():
                return IX._tiled_scores_buffer(q32, ik, w32, lens, nb, k, nb)[:2]

            def run_hier():
                return HC.hierarchical_topk_prod(
                    q32, ik, w32, lens, k, IX._HIER_BLOCK, IX._HIER_STRIP, estrip,
                    IX._HIER_OVERFETCH, cand_mask=None,
                    cand_src=(CAND_TOPK_BLOCKS, BLOCK) if is_cand else None,
                    consumer_skip=IX._HIER_CONSUMER_SKIP)[:2]

            def run_hier_forced():
                # production DISABLES hier for n <= _FENCE_MIN_ROWS (=16) -- this
                # arm forces it to map the small-row crossover explicitly.
                IX._FENCE_MIN_ROWS = -1
                try:
                    return run_hier()
                finally:
                    IX._FENCE_MIN_ROWS = fence_saved

            if dirty:
                guard.wait_idle()
                dirty = False
            # ABAB interleave: REF / HIER / (UNTILED when cheap) x 2 passes
            ref_runs, hier_runs, unt_runs, hfor_runs = [], [], [], []
            for _ in range(2):
                ref_runs.append(timed(run_ref, max(4, REPS // 2), max(2, WARM - 1),
                                      guard=guard, gpu=False,
                                      label=f"{tag} REF", reset_peak=True))
                hier_runs.append(timed(run_hier, max(4, REPS // 2), max(2, WARM - 1),
                                       guard=guard, gpu=False,
                                       label=f"{tag} HIER", reset_peak=True))
                if do_untiled:
                    unt_runs.append(timed(run_untiled, max(4, REPS // 2),
                                          max(2, WARM - 1), guard=guard, gpu=False,
                                          label=f"{tag} UNTILED", reset_peak=True))
                if n <= 16:
                    hfor_runs.append(timed(run_hier_forced, max(4, REPS // 2),
                                           max(2, WARM - 1), guard=guard, gpu=False,
                                           label=f"{tag} HIER-forced", reset_peak=True))
            ref = merge_runs(ref_runs)
            hier = merge_runs(hier_runs)
            sp_ratio = hier["ms"] / ref["ms"]
            S = n * H_IDX * nb
            rec = dict(n=n, nb=nb, S=S, L_ratio1=nb, L_ratio2=2 * nb,
                       ref_ms=ref, hier_ms=hier, speedup_hier_over_ref=sp_ratio,
                       ref_peak_mb=ref.get("peak_mb"), hier_peak_mb=hier.get("peak_mb"),
                       route="tiled" if use_tiled else "row-whole", route_n=route_n,
                       tile=tile, estrip=estrip, label="MEASURED")
            if n <= 16:
                rec["production_gate"] = ("HIER disabled at n <= 16 (row path is "
                                          "what production runs at this row count); "
                                          "the hier arm here is gate-forced")
            if unt_runs:
                unt = merge_runs(unt_runs)
                rec["untiled_ms"] = unt
                rec["speedup_hier_over_untiled"] = hier["ms"] / unt["ms"]
                rec["untiled_peak_mb"] = unt.get("peak_mb")
            if hfor_runs:
                hf = merge_runs(hfor_runs)
                rec["hier_forced_ms"] = hf
                rec["speedup_hier_forced_over_ref"] = hf["ms"] / ref["ms"]
                rec["hier_gate"] = ("production DISABLES hier at n <= "
                                    f"{fence_saved} (this row takes the row path)")
            if ref["discards"] + hier["discards"]:
                dirty = True                        # re-arm the idle wait
            cells.append(rec)
            log(f"{tag}: S={S:.3g} ref={ref['ms']:8.2f} (iqr {ref['iqr_ms']:6.2f}, "
                f"n={ref['reps_used']}) hier={hier['ms']:8.2f} "
                f"(iqr {hier['iqr_ms']:6.2f}, n={hier['reps_used']}) "
                f"speedup={sp_ratio:5.3f}"
                + (f" | untiled={rec['untiled_ms']['ms']:8.2f} "
                   f"sp_unt={rec['speedup_hier_over_untiled']:5.3f}" if unt_runs else "")
                + (f" | hierFORCED={rec['hier_forced_ms']['ms']:8.2f} "
                   f"sp_f={rec['speedup_hier_forced_over_ref']:5.3f}" if hfor_runs else "")
                + f" | peaks {ref.get('peak_mb')}/{hier.get('peak_mb')} MB")
            guard.own_work()
            del x, qr, ik, q32, w32
            mx.clear_cache()
            res.add("crossover_partial", dict(cells=cells, fit=fit_threshold(cells)))
    # ---- fit: pure-S threshold?  (a cell is misclassified if |speedup-1|>10%
    #      and the threshold puts it on the wrong side)
    fit = fit_threshold(cells)
    res.add("crossover", dict(rows=rows_grid, nb=nb_grid, cells=cells, fit=fit,
                              layer=layer, ratio=ratio,
                              mem=mem, discards=guard.discards))
    log("CROSSOVER_DONE")


def merge_runs(runs):
    walls = [w for r in runs for w in r["wall_all"]]
    peaks = [p for r in runs for p in r.get("peak_all", [])]
    out = dict(reps_used=len(walls), ms=stats.median(walls) if walls else float("nan"),
               discards=sum(r["discards"] for r in runs), wall_all=walls,
               peak_mb=max(peaks) if peaks else None)
    if len(walls) >= 4:
        q = np.percentile(walls, [25, 75])
        out["iqr_ms"] = float(q[1] - q[0])
    out["min_ms"] = min(walls) if walls else float("nan")
    return out


def fit_threshold(cells):
    """1-param S threshold (REF faster iff S <= S*) and the 2-param form."""
    good = [c for c in cells if "speedup_hier_over_ref" in c]
    def miscls(Sstar):
        bad = []
        for c in good:
            if abs(c["speedup_hier_over_ref"] - 1.0) <= 0.10:
                continue
            ref_faster = c["speedup_hier_over_ref"] > 1.0
            pred_ref_faster = c["S"] <= Sstar
            if ref_faster != pred_ref_faster:
                bad.append(c)
        return bad
    Slist = sorted({c["S"] for c in good})
    cand = [0] + [(Slist[i] + Slist[i + 1]) / 2 for i in range(len(Slist) - 1)] + [Slist[-1] * 1.01]
    best = min(cand, key=lambda s: (len(miscls(s)), s))
    bad = miscls(best)
    out = dict(one_param_S_star=best, one_param_misclassified=len(bad),
               one_param_fits=len(bad) == 0,
               bad=[dict(n=c["n"], nb=c["nb"], S=c["S"],
                         speedup=c["speedup_hier_over_ref"]) for c in bad])
    # 2-param: S <= S* AND L <= L*  (L here = nb * ratio as the ratio-1 equivalent)
    def miscls2(Sstar, Lstar):
        bad2 = []
        for c in good:
            if abs(c["speedup_hier_over_ref"] - 1.0) <= 0.10:
                continue
            ref_faster = c["speedup_hier_over_ref"] > 1.0
            pred = (c["S"] <= Sstar) and (c["nb"] <= Lstar)
            if ref_faster != pred:
                bad2.append(c)
        return bad2
    Ls = sorted({c["nb"] for c in good})
    best2 = None
    for Ss in cand:
        for Ls_ in Ls:
            bad2 = miscls2(Ss, Ls_)
            key = (len(bad2), Ss, Ls_)
            if best2 is None or key < best2[0]:
                best2 = (key, Ss, Ls_, bad2)
    out["two_param"] = dict(S_star=best2[1], L_star=best2[2],
                            misclassified=len(best2[3]), fits=len(best2[3]) == 0,
                            bad=[dict(n=c["n"], nb=c["nb"], S=c["S"],
                                      speedup=c["speedup_hier_over_ref"]) for c in best2[3]])
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="parity",
                    choices=["parity", "roofline", "split", "crossover", "mem", "all"])
    ap.add_argument("--out-tag", default=None)
    a = ap.parse_args()
    tag = a.out_tag or os.environ.get("P1_TAG") or a.mode
    res = Res(tag)
    IX._coll.all_sum = _stub_all_sum        # single node: no RDMA on the live link
    log(f"mode={a.mode} tag={tag} pid={os.getpid()} host={os.uname().nodename} "
        f"MLXLN={_repo} GPU_TIME={HAVE_GPU_TIME}")
    peaks = None
    if a.mode in ("roofline", "split", "crossover", "all"):
        if os.environ.get("P1_ROOFLINE", "1") == "1":
            peaks = cmd_roofline(res)
        else:
            peaks = dict(stream_bw_gbs=float(os.environ.get("P1_BW_GBS", "100")),
                         tflops=float(os.environ.get("P1_TFLOPS", "10")),
                         label="ASSUMED (P1_ROOFLINE=0)")
    if a.mode in ("parity", "all"):
        cmd_parity(res)
    if a.mode in ("split", "all"):
        cmd_split(res, peaks)
    if a.mode in ("crossover", "all"):
        cmd_crossover(res, peaks)
    if a.mode == "mem":
        res.add("mem_headroom", mem_report())
    res.write()
    log("ALL_DONE")


_ALLSUM = {"n": 0, "bytes": 0}


def _stub_all_sum(x, group=None):
    """Single-node stand-in for mlx_lm...collective.all_sum (no RDMA).

    The padded row exchange (_reconstruct_rows) is a zeros+slice-assign+all_sum;
    the assign is timed, the all_sum payload is counted but its RDMA cost is
    EXCLUDED and bounded separately in the report.
    """
    _stub_allsum_count_impl(x)
    return x


def _stub_allsum_count_impl(x):
    _ALLSUM["n"] += 1
    _ALLSUM["bytes"] += int(x.size * x.dtype.size)


if __name__ == "__main__":
    main()
