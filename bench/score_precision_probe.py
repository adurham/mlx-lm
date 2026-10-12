#!/usr/bin/env python3
"""score_precision_probe.py -- DSv4.1 indexer Phase-3 first experiment (<=2 h microbench).

Question: does any reduced-precision (int8 / fp8 / fp16 / bf16) matmul path in THIS
MLX build beat the indexer's current REF score sub-op by >= 1.5x on the production
shapes (rows=1024 per rank, tile=504, h=32, d=128, nb in {50K, 100K, 130K})?

Score sub-op semantics (indexer._score_body_eager + the _tiled_scores_buffer loop):
    s = einsum("bshd,btd->bsht", q, k)        # [b, n, h, t]
    s = relu(s) * w[..., None]
    s = sum(s, axis=2)                       # [b, n, t]
    s = where(col < lens, s, -inf); eval; row[:, :, c0:c1] = s   (row bf16)

Modes
  smoke     tiny shapes: which variants exist on this build + max|d| vs fp32 ref (no timing)
  tile      per-tile body (n=1024, t=504) fenced per call, round-robin ABAB
  loop      the full tile loop at rows=1024 x nb in P4_NBS, round-robin ABAB, + prep ops
  numerics  dScore stats + top-k agreement: synthetic (n=1024, nb=50000) and the real
            L20 rank-0 capture at L=50K (P4_CAP)

Reuses bench/indexer_split.py's IdleGuard (exo.log lifecycle tail + runner CPU delta +
thresholded ioreg) and its provenance(). Labels: every timing MEASURED; TFLOP/s,
GB/s, speedups DERIVED; with-prep sums DERIVED.
"""
from __future__ import annotations

import json
import os
import statistics as stats
import sys
import time

os.environ.setdefault("MTL_DISABLE_TIMEOUT", "1")
os.environ.setdefault("MTL_COMMAND_BUFFER_TIMEOUT", "0")
os.environ.setdefault("MLX_GPU_TIME", "1")
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "200")
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
os.environ.setdefault("P1_OUT", "/private/tmp/idxprof4/raw")

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)

import numpy as np                                    # noqa: E402
import mlx.core as mx                                 # noqa: E402

import indexer_split as S                             # noqa: E402  (guard, provenance)
from mlx_lm.models.deepseek_v41 import indexer as IX  # noqa: E402

OUT = os.environ.get("P4_OUT", "/private/tmp/idxprof4/raw")
LOG = os.environ.get("P4_LOG", os.path.expanduser("~/exo.log"))
WARM = int(os.environ.get("P4_WARM", "3"))
REPS = int(os.environ.get("P4_REPS", "7"))
IDLE_S = float(os.environ.get("P4_IDLE_S", "60"))
NBS = [int(v) for v in os.environ.get("P4_NBS", "50000,100000,130000").split(",")]
N = int(os.environ.get("P4_ROWS", "1024"))
TILE = int(os.environ.get("P4_TILE", "504"))
H, D, K = 32, 128, 512
NO_GUARD = os.environ.get("P4_NO_GUARD", "0") == "1"
NEG_INF = IX.NEG_INF
HAVE_GPU = hasattr(mx.metal, "gpu_time_ns")
# clear the MLX buffer cache before every timed rep (1st tile run did; it adds a fresh
# allocation to every ~1 ms per-tile call -> per-tile baseline 2.8 ms, not reproducing
# the 1.10 ms roofline number; P4_CLEAR=0 is the one allowed fix for the tile mode).
CLEAR = os.environ.get("P4_CLEAR", "1") == "1"


def log(*a):
    print("[p4]", time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------
# variants: each = (prep_k(ik_bf16) -> kprep, prep_q(q32) -> qprep, body(qprep, ktile, w32))
# ktile is produced by slice_k(kprep, c0, c1).
# ---------------------------------------------------------------------------
def _relu_w_sum(s, w):
    s = mx.maximum(s, 0.0) * w[..., None]
    return mx.sum(s, axis=2)


def make_variants():
    V = {}

    def dense(name, dt, epi32=False, pure=False):
        def prep_k(ik):
            return ik.astype(dt)

        def prep_q(q32, w32):
            return q32.astype(dt), w32.astype(mx.float32 if epi32 else dt)

        def slice_k(kp, c0, c1):
            return kp[:, c0:c1]

        def body(q, kt, w):
            s = mx.einsum("bshd,btd->bsht", q, kt)
            if pure:
                return s
            if epi32:
                s = s.astype(mx.float32)
            return _relu_w_sum(s, w)
        V[name] = dict(prep_k=prep_k, prep_q=prep_q, slice_k=slice_k, body=body,
                       kind="pure_gemm" if pure else "score", desc=f"einsum {dt}"
                       + (" -> fp32 epilogue" if epi32 else "") + (" (no head-sum)" if pure else ""))

    def qmm(name, bits, gs, adt, mode="affine", epi32=False, pure=False):
        def prep_k(ik):
            k = ik.astype(adt)[0]                     # [nb, d]
            r = mx.quantize(k, group_size=gs, bits=bits, mode=mode)
            return r if len(r) == 3 else (r[0], r[1], None)

        def prep_q(q32, w32):
            b, n, h, d = q32.shape
            return q32.astype(adt).reshape(b, n * h, d), w32.astype(mx.float32 if epi32 else adt)

        def slice_k(kp, c0, c1):
            wq, sc, bi = kp
            return (wq[c0:c1], sc[c0:c1], None if bi is None else bi[c0:c1])

        def body(q, kt, w):
            wq, sc, bi = kt
            b, nh, d = q.shape
            n = w.shape[1]
            s = mx.quantized_matmul(q, wq, sc, bi, transpose=True, group_size=gs,
                                    bits=bits, mode=mode)          # [b, n*h, t]
            s = s.reshape(b, n, H, -1)
            if pure:
                return s
            if epi32:
                s = s.astype(mx.float32)
            return _relu_w_sum(s, w)
        V[name] = dict(prep_k=prep_k, prep_q=prep_q, slice_k=slice_k, body=body,
                       kind="pure_gemm" if pure else "score",
                       desc=f"quantized_matmul {mode} {bits}b g{gs} K, act {adt}"
                       + (" -> fp32 epilogue" if epi32 else "") + (" (no head-sum)" if pure else ""))

    def qqmm(name, mode, gs, bits):
        def prep_k(ik):
            k = ik.astype(mx.bfloat16)[0]
            return mx.quantize(k, group_size=gs, bits=bits, mode=mode)

        def prep_q(q32, w32):
            b, n, h, d = q32.shape
            return q32.astype(mx.bfloat16).reshape(b * n * h, d), w32.astype(mx.float32)

        def slice_k(kp, c0, c1):
            return tuple(x[c0:c1] for x in kp[:2])

        def body(q, kt, w):
            wq, sc = kt
            s = mx.qqmm(q, wq, sc, group_size=gs, bits=bits, mode=mode)
            s = s.reshape(1, w.shape[1], H, -1).astype(mx.float32)
            return _relu_w_sum(s, w)
        V[name] = dict(prep_k=prep_k, prep_q=prep_q, slice_k=slice_k, body=body,
                       kind="score", desc=f"qqmm {mode} (q AND k quantized, on-the-fly x)")

    def int8mm(name):
        def prep_k(ik):
            k = ik.astype(mx.float32)
            sc = mx.max(mx.abs(k), axis=-1, keepdims=True) / 127.0
            return (mx.round(k / sc).astype(mx.int8), sc)

        def prep_q(q32, w32):
            sc = mx.max(mx.abs(q32), axis=-1, keepdims=True) / 127.0
            return (mx.round(q32 / sc).astype(mx.int8), sc), w32

        def slice_k(kp, c0, c1):
            return (kp[0][:, c0:c1], kp[1][:, c0:c1])

        def body(q, kt, w):
            (qi, qs), (ki, ks) = q, kt
            s = mx.matmul(qi, ki.swapaxes(-1, -2)[:, None])          # int8 x int8
            s = s.astype(mx.float32) * qs * ks.swapaxes(-1, -2)[:, None]
            return _relu_w_sum(s, w)
        V[name] = dict(prep_k=prep_k, prep_q=prep_q, slice_k=slice_k, body=body,
                       kind="score", desc="mx.matmul int8 x int8 (symmetric per-row)")

    dense("ref_fp32", mx.float32)
    dense("bf16_full", mx.bfloat16)
    dense("fp16_full", mx.float16)
    dense("fp16_gemm_f32epi", mx.float16, epi32=True)
    qmm("q8g64_fp16", 8, 64, mx.float16)
    qmm("q8g128_fp16", 8, 128, mx.float16)
    qmm("q8g64_bf16", 8, 64, mx.bfloat16)
    qmm("q8g64_fp16_f32epi", 8, 64, mx.float16, epi32=True)
    qmm("q4g64_fp16", 4, 64, mx.float16)
    qmm("mxfp8_fp16", 8, 32, mx.float16, mode="mxfp8")
    qqmm("qqmm_mxfp8", "mxfp8", 32, 8)
    int8mm("int8_mm")
    dense("GEMM_fp32", mx.float32, pure=True)
    dense("GEMM_fp16", mx.float16, pure=True)
    dense("GEMM_bf16", mx.bfloat16, pure=True)
    qmm("GEMM_q8g64_fp16", 8, 64, mx.float16, pure=True)
    return V


VARIANTS = make_variants()
_COMPILED = {}


def compiled(name, t):
    """Per-(variant, tile width) non-shapeless compile (the production convention)."""
    key = (name, t)
    fn = _COMPILED.get(key)
    if fn is None:
        if name == "ref_fp32":
            fn = IX._score_body(1, N, H, D, t)          # THE production body (compiled)
        else:
            fn = mx.compile(VARIANTS[name]["body"])
        _COMPILED[key] = fn
    return fn


def synth(n, nb, seed=11):
    mx.random.seed(seed)
    q32 = (mx.random.normal((1, n, H, D)) * 0.3).astype(mx.float32)
    ik = (mx.random.normal((1, nb, D)) * 0.5).astype(mx.bfloat16)
    w32 = (mx.random.normal((1, n, H)) * 0.05).astype(mx.float32)
    lens = mx.full((n, 1), nb, dtype=mx.int32)
    mx.eval(q32, ik, w32, lens)
    return q32, ik, w32, lens


def prepare(name, q32, ik, w32):
    v = VARIANTS[name]
    kp = v["prep_k"](ik)
    qp, wp = v["prep_q"](q32, w32)
    mx.eval(kp, qp, wp)
    return kp, qp, wp


def run_loop(name, kp, qp, wp, lens, nb, tile, row_dtype=mx.bfloat16, write_row=True):
    """The _tiled_scores_buffer loop with this variant's body (split_ref part (i) shape)."""
    v = VARIANTS[name]
    pure = v["kind"] == "pure_gemm"
    row = mx.zeros((1, N if qp is None else lens.shape[0], nb), dtype=row_dtype) if (write_row and not pure) else None
    last = None
    for c0 in range(0, nb, tile):
        c1 = min(c0 + tile, nb)
        s = compiled(name, c1 - c0)(qp, v["slice_k"](kp, c0, c1), wp)
        if pure:
            mx.eval(s)                                  # materialise the GEMM tile, nothing else
            last = s
            continue
        cols = mx.arange(c0, c1, dtype=mx.int32)
        s = mx.where(cols[None, :] < lens, s, NEG_INF)
        mx.eval(s)
        if row is not None:
            row[:, :, c0:c1] = s
        last = s
    if row is not None:
        mx.eval(row)
        return row
    return last


# ---------------------------------------------------------------------------
# round-robin ABAB timing, contamination-policed
# ---------------------------------------------------------------------------
def rr_time(fns: dict, guard, warm=WARM, reps=REPS, label=""):
    res = {k: dict(wall=[], gpu=[], peak=[], discards=0) for k in fns}
    for _ in range(warm):
        for k, f in fns.items():
            out = f(); mx.eval(out) if isinstance(out, mx.array) else None; del out
    mx.synchronize()
    for r in range(reps):
        order = list(fns) if r % 2 == 0 else list(reversed(list(fns)))   # ABAB / BABA
        for k in order:
            if guard is not None:
                ok, ev = guard.clean()
                if not ok:
                    res[k]["discards"] += 1
                    guard.discards += 1
                    guard._note(f"DISCARD {label} {k} rep {r}: " + "; ".join(ev))
                    continue
            mx.synchronize()
            if CLEAR:
                mx.clear_cache()
            base = mx.get_active_memory()
            mx.reset_peak_memory()
            if HAVE_GPU:
                mx.metal.reset_gpu_time()
            t0 = time.perf_counter()
            out = fns[k]()
            if isinstance(out, mx.array):
                mx.eval(out)
            mx.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            res[k]["wall"].append(dt)
            if HAVE_GPU:
                res[k]["gpu"].append(mx.metal.gpu_time_ns() / 1e6)
            res[k]["peak"].append((mx.get_peak_memory() - base) / 1e6)
            del out
    for k, r in res.items():
        w = r["wall"]
        r["ms"] = stats.median(w) if w else float("nan")
        r["iqr_ms"] = float(np.subtract(*np.percentile(w, [75, 25]))) if len(w) >= 4 else float("nan")
        r["gpu_ms"] = stats.median(r["gpu"]) if r["gpu"] else None
        r["peak_mb"] = max(r["peak"]) if r["peak"] else None
        r["reps_used"] = len(w)
        r["label"] = "MEASURED"
    return res


def writeout(tag, obj):
    os.makedirs(OUT, exist_ok=True)
    obj = dict(tag=tag, host=os.uname().nodename, mlx=mx.__version__,
               provenance=S.provenance(), N=N, TILE=TILE, H=H, D=D, warm=WARM, reps=REPS,
               env={k: os.environ.get(k) for k in ("MLX_GPU_TIME", "MLX_MAX_OPS_PER_BUFFER",
                                                   "MLX_METAL_FAST_SYNCH", "DSV41_INDEXER_COMPILE",
                                                   "MTL_DISABLE_TIMEOUT")},
               **obj)
    with open(os.path.join(OUT, f"{tag}.json"), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    log(f"wrote {OUT}/{tag}.json")


def make_guard():
    if NO_GUARD:
        return None
    os.makedirs(OUT, exist_ok=True)
    g = S.IdleGuard(path=LOG, logf=open(os.path.join(OUT, "idle.log"), "a"))
    log(f"idle guard: runner pid={g.runner_pid} log={LOG} window={IDLE_S}s")
    return g


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def cmd_smoke():
    n, nb, t = 64, 1536, 504
    q32, ik, w32, lens = synth(n, nb, seed=3)
    global N
    N = n
    ref = None
    out = {}
    for name, v in VARIANTS.items():
        try:
            kp, qp, wp = prepare(name, q32, ik, w32)
            r = run_loop(name, kp, qp, wp, lens, nb, t, row_dtype=mx.float32)
            mx.eval(r)
            if name == "ref_fp32":
                ref = np.array(r)
            rec = dict(status="ok", desc=v["desc"], kind=v["kind"], out_shape=list(r.shape))
            if v["kind"] == "score" and ref is not None:
                a = np.array(r.astype(mx.float32))
                rec["max_abs_d"] = float(np.max(np.abs(a - ref)))
                rec["ref_scale"] = float(np.max(np.abs(ref)))
            out[name] = rec
        except Exception as e:                              # record, do not crash
            out[name] = dict(status="unsupported/error", desc=v["desc"],
                             error=f"{type(e).__name__}: {str(e)[:300]}")
        log(name, json.dumps(out[name])[:300])
    writeout("smoke", dict(mode="smoke", shape=dict(n=n, nb=nb, tile=t), variants=out))
    log("SMOKE_DONE")


def ok_variants():
    p = os.path.join(OUT, "smoke.json")
    sm = json.load(open(p))["variants"]
    sel = os.environ.get("P4_VARIANTS")
    names = [k for k, v in sm.items() if v["status"] == "ok"]
    if sel:
        names = [k for k in names if k in sel.split(",")]
    return names


def cmd_tile(guard):
    names = ok_variants()
    q32, ik, w32, lens = synth(N, TILE, seed=11)
    preps = {k: prepare(k, q32, ik, w32) for k in names}
    fns = {}
    for k in names:
        kp, qp, wp = preps[k]
        kt = VARIANTS[k]["slice_k"](kp, 0, TILE)
        mx.eval(kt)
        fns[k] = (lambda k=k, kt=kt, qp=qp, wp=wp: compiled(k, TILE)(qp, kt, wp))
    # the exact roofline-mode expression (eager production body, fp32 key tile)
    k32 = ik.astype(mx.float32)
    mx.eval(k32)
    fns["ref_fp32_eager"] = lambda: IX._score_body_eager(q32, k32, w32)
    if guard:
        guard.wait_idle(IDLE_S)
    res = rr_time(fns, guard, warm=max(WARM, 5), reps=max(REPS, 15), label="tile")
    flops = 2 * N * TILE * H * D
    for k, r in res.items():
        r["tflops"] = flops / (r["ms"] / 1e3) / 1e12
        r["tflops_label"] = "DERIVED"
        r["speedup_vs_ref"] = res["ref_fp32"]["ms"] / r["ms"]
        log(f"tile {k:22s} {r['ms']:.3f} ms (IQR {r['iqr_ms']:.3f}) {r['tflops']:.2f} TF/s "
            f"x{r['speedup_vs_ref']:.2f} peak {r['peak_mb']:.0f} MB disc {r['discards']}")
    if guard:
        guard.own_work()
    writeout(os.environ.get("P4_TAG", "tile"), dict(mode="tile", clear_cache_per_rep=CLEAR,
                          shape=dict(n=N, t=TILE), flops=flops, results=res,
                          discards=0 if guard is None else guard.discards))
    log("TILE_DONE")


def cmd_loop(guard):
    names = ok_variants()
    allres = {}
    for nb in NBS:
        q32, ik, w32, lens = synth(N, nb, seed=11)
        preps = {k: prepare(k, q32, ik, w32) for k in names}
        fns = {k: (lambda k=k: run_loop(k, *preps[k], lens, nb, TILE)) for k in names}
        # prep ops (K-side whole-cache conversion + q-side cast), timed in the same rotation
        prep_fns = {}
        for k in names:
            v = VARIANTS[k]
            prep_fns["PREP_" + k] = (lambda v=v: [*_flat(v["prep_k"](ik)), *_flat(v["prep_q"](q32, w32))])
        all_fns = dict(fns)
        for pk, pf in prep_fns.items():
            all_fns[pk] = (lambda pf=pf: _evalall(pf()))
        if guard:
            guard.wait_idle(IDLE_S)
        res = rr_time(all_fns, guard, label=f"loop nb={nb}")
        if guard:
            guard.own_work()
        flops = 2 * N * nb * H * D
        ref = res["ref_fp32"]["ms"]
        for k in names:
            r = res[k]
            r["tflops"] = flops / (r["ms"] / 1e3) / 1e12
            r["speedup_vs_ref"] = ref / r["ms"]
            r["prep_ms"] = res["PREP_" + k]["ms"]
            r["with_prep_ms"] = r["ms"] + r["prep_ms"]
            r["speedup_with_prep_vs_ref_with_prep"] = (ref + res["PREP_ref_fp32"]["ms"]) / r["with_prep_ms"]
            r["derived_labels"] = "tflops/speedup/with_prep DERIVED"
            log(f"nb={nb} {k:22s} {r['ms']:8.2f} ms (IQR {r['iqr_ms']:.2f}) {r['tflops']:.2f} TF/s "
                f"x{r['speedup_vs_ref']:.2f} | prep {r['prep_ms']:.2f} ms, w/prep x"
                f"{r['speedup_with_prep_vs_ref_with_prep']:.2f} | peak {r['peak_mb']:.0f} MB "
                f"disc {r['discards']}")
        allres[str(nb)] = dict(flops=flops, results=res)
        del preps, fns, all_fns
        mx.clear_cache()
        writeout("loop", dict(mode="loop", nbs=NBS, cells=allres,
                              discards=0 if guard is None else guard.discards))
    log("LOOP_DONE")


def _flat(x):
    out = []
    if isinstance(x, (tuple, list)):
        for e in x:
            out += _flat(e)
    elif isinstance(x, mx.array):
        out.append(x)
    return out


def _evalall(xs):
    mx.eval(*xs)
    return None


def _topk_sets(row, k=K):
    a = np.array(row.astype(mx.float32))[0]                 # [n, nb]
    idx = np.argpartition(-a, k - 1, axis=-1)[:, :k]
    return [set(r.tolist()) for r in idx], a


def _numerics_case(names, q32, ik, w32, lens, nb, tag):
    out = {}
    # fp32 ref with fp32 row = the exact REF row (HIER is index-identical to it at L=50K)
    kp, qp, wp = prepare("ref_fp32", q32, ik, w32)
    ref_row = run_loop("ref_fp32", kp, qp, wp, lens, nb, TILE, row_dtype=mx.float32)
    ref_sets, ref = _topk_sets(ref_row)
    fin = np.isfinite(ref)
    scale = np.max(np.abs(np.where(fin, ref, 0)), axis=-1, keepdims=True) + 1e-30
    # production bf16 row as the noise reference
    cands = [("ref_fp32_bf16row(production)", "ref_fp32", mx.bfloat16)]
    cands += [(k, k, mx.float32) for k in names
              if VARIANTS[k]["kind"] == "score" and k != "ref_fp32"]
    for label, k, rdt in cands:
        kp, qp, wp = prepare(k, q32, ik, w32)
        r = run_loop(k, kp, qp, wp, lens, nb, TILE, row_dtype=rdt)
        sets, a = _topk_sets(r)
        d = np.abs(np.where(fin, a - ref, 0))
        rel = d / scale
        same = sum(1 for u, v in zip(sets, ref_sets) if u == v)
        rec_ = [len(u & v) / K for u, v in zip(sets, ref_sets)]
        out[label] = dict(max_abs_dscore=float(d.max()), mean_abs_dscore=float(d[fin].mean()),
                          max_rel_dscore_rowscale=float(rel.max()),
                          p99_rel_dscore_rowscale=float(np.percentile(rel[fin], 99)),
                          rows_topk_identical=f"{same}/{len(sets)}",
                          topk_recall_mean=float(np.mean(rec_)),
                          topk_recall_p1=float(np.percentile(rec_, 1)),
                          topk_recall_min=float(np.min(rec_)), label="MEASURED")
        log(f"numerics[{tag}] {label:30s} max|d| {out[label]['max_abs_dscore']:.3e} "
            f"rel {out[label]['max_rel_dscore_rowscale']:.2e} topk-identical "
            f"{out[label]['rows_topk_identical']} recall mean {out[label]['topk_recall_mean']:.5f} "
            f"min {out[label]['topk_recall_min']:.4f}")
        del r
        mx.clear_cache()
    return dict(ref_row_scale_median=float(np.median(scale)), nb=nb,
                rows=int(q32.shape[1]), variants=out)


def cmd_numerics():
    names = ok_variants()
    res = {}
    global N
    q32, ik, w32, lens = synth(N, 50000, seed=11)
    res["synthetic_n1024_nb50000"] = _numerics_case(names, q32, ik, w32, lens, 50000, "synth")
    cap = os.environ.get("P4_CAP", "/private/tmp/idxprof4/cap/cap_L50000.npz")
    if os.path.exists(cap):
        d = np.load(cap)
        q = mx.array(d["L20_q_f32_r0"])
        w = mx.array(d["L20_w_r0"])
        ln = mx.array(d["L20_lens_r0"].astype(np.int32))
        ikr = mx.array(d["L20_ik16"]).astype(mx.bfloat16)     # cache is bf16 in production
        nb = int(ikr.shape[1])
        N = int(q.shape[1])
        res["real_L20_r0_L50000"] = _numerics_case(names, q, ikr, w, ln, nb, "real L20 r0")
        res["real_L20_r0_L50000"]["note"] = ("real captured q/w/lens (rank-0 band, 424 rows) "
                                             "and index_k (fp16-stored capture -> bf16 like the cache)")
    writeout("numerics", dict(mode="numerics", k=K, cases=res))
    log("NUMERICS_DONE")


def main():
    mode = sys.argv[1]
    log(f"mode={mode} mlx={mx.__version__} N={N} TILE={TILE} NBS={NBS} warm={WARM} reps={REPS}")
    if mode == "smoke":
        cmd_smoke()
        return
    guard = make_guard()
    if mode == "tile":
        cmd_tile(guard)
    elif mode == "loop":
        cmd_loop(guard)
    elif mode == "numerics":
        cmd_numerics()
    if guard:
        log(f"guard discards={guard.discards} events={len(guard.events)}")


if __name__ == "__main__":
    main()
