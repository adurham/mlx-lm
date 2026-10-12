#!/usr/bin/env python3
"""route_probe.py -- DSv4.1 indexer route/budget R0 (idxprof5): tiled vs untiled at the
fallback-class shapes (n <= 16 under HIER=1), MEASUREMENT ONLY.

Reuses bench/indexer_split.py (idle guard, timed(), mem_report, real-weight indexer
loader, synthetic inputs).  Every timed call is the REAL production function
``Indexer.__call__`` (mlx-lm 6a5c78f4); the ONLY harness intervention is the route:
the module-global ``indexer.tiled`` is swapped for a stub returning a fixed width, so
the arms differ exactly in which branch __call__ takes and with which tile width:

  CUR    the production rule, unmodified (route recorded, then timed as one of below)
  T512   tiled branch at production's tile_width() (512, or the align multiple)
  WIDE   tiled branch at the budget-derived cap width (candidate rule R-B)
  UNT    untiled reference branch (candidate rule R-A)

Modes:
  fence    host fence microbench (per mx.eval round-trip cost)
  grid     n x nb grid, layer 20 (cand source) for all n, + consumer 24 and owner 8
           at n=4 (the verify shape); wall + gpu + peak, ABAB, median+IQR
  anchors  idxprof1 continuity cells via the idxprof1 method (_tiled_scores_buffer)
  parity   index-set + candidate-mask bit-identity across UNT / T512 / WIDE,
           random keys AND tie-injected (periodic) keys

LABELS: wall/gpu/peak = MEASURED; fences, transient bytes, rule decisions = DERIVED.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indexer_split as S                     # noqa: E402  (sets env, imports mlx)

mx, np, IX = S.mx, S.np, S.IX
OUT = os.environ.get("P5_OUT", "/private/tmp/idxprof5/raw")
os.makedirs(OUT, exist_ok=True)
REPS_HALF = int(os.environ.get("P5_REPS_HALF", "4"))   # x2 ABAB passes = 8 reps
WARM = int(os.environ.get("P5_WARM", "3"))
H = 32
ORIG_TILED = IX.tiled


def log(*a):
    print("[p5]", *a, flush=True)


def rule_route(n, nb, align):
    """Simulate the CURRENT production rule exactly (calls the real function)."""
    t = ORIG_TILED(1, n, H, nb, align)
    return ("tiled", t) if t < nb else ("untiled", nb)


def wide_width(n, nb, align):
    row = IX._row_dtype(n).size
    cap = max(align, IX._TILE_BUDGET // max(1, n * (H * 4 + row)))
    return min(nb, max(align, (cap // align) * align))


def force(width):
    IX.tiled = lambda bsz, n, nh, nb, align=1: min(width, nb)


def restore():
    IX.tiled = ORIG_TILED


def untiled_bytes(n, nb):
    """DERIVED: the budget formula's quantity (what tiled() compares to 128 MiB)."""
    return n * nb * (H * 4 + IX._row_dtype(n).size)


def pred_untiled_peak(n, nb, cons):
    return (n * H * nb * 4 + nb * 128 * 4 + n * nb * 4 * 2 + (n * nb if cons else 0)
            + n * 512 * 8)


class Rig:
    """One indexer layer + inputs at (n, nb), callable with a forced route."""

    def __init__(self, ix, layer, n, nb, cand_src_ix=None, ik=None, seed=13):
        self.ix, self.layer, self.n, self.nb = ix, layer, n, nb
        ratio = max(ix.ratio, 1)
        x, qr, ik0 = S.synth_inputs(n, nb, seed=seed)
        self.x, self.qr = x, qr
        self.ik = ik0 if ik is None else ik
        self.start = nb * ratio - n                # last row sees exactly nb groups
        self.fv = S.freqvec_for(layer, ix.ratio)
        self.cand = None
        if ix.uses_candidates:
            qpos = np.arange(self.start, self.start + n)
            self.cand = S.synth_cand_mask(n, nb, self.start, ratio, qpos)
            mx.eval(self.cand)
        self.align = ix.candidate_block_size if ix.is_candidate_source else 1
        self.shared = S.Shared(self.cand)

    def call(self):
        self.shared.candidates = self.cand
        out = self.ix(self.x, self.qr, self.start, 0, self.fv, self.ik, self.shared)
        if self.ix.is_candidate_source:
            return out, self.shared.candidates
        return out


def time_arms(rig, arms, guard, tag):
    """ABAB over arms (2 passes x REPS_HALF), per-arm median/IQR/gpu/peak."""
    runs = {a: [] for a in arms}
    for _ in range(2):
        for a, width in arms.items():
            force(width)
            try:
                runs[a].append(S.timed(rig.call, REPS_HALF, WARM, guard=guard, gpu=True,
                                       label=f"{tag} {a}", reset_peak=True))
            finally:
                restore()
    out = {}
    for a, rs in runs.items():
        m = S.merge_runs(rs)
        g = [v for r in rs for v in r.get("gpu_all", [])]
        m["gpu_ms"] = float(np.median(g)) if g else None
        if len(g) >= 4:
            q = np.percentile(g, [25, 75])
            m["gpu_iqr_ms"] = float(q[1] - q[0])
        m["width"] = arms[a]
        m["fences"] = (math.ceil(rig.nb / arms[a]) if arms[a] < rig.nb else 0)
        out[a] = m
    return out


# ---------------------------------------------------------------------------
def cmd_fence(res):
    """Per-fence host round trip: tiny op + mx.eval, back to back (MEASURED)."""
    a = mx.zeros((4, 512), dtype=mx.float32)
    mx.eval(a)
    rows = []
    for size in (512, 4 * 512, 4 * 32 * 512):
        b = mx.zeros((size,), dtype=mx.float32)
        mx.eval(b)
        for _ in range(3):
            t0 = time.perf_counter()
            for i in range(2000):
                b = b + 1.0
                mx.eval(b)
            mx.synchronize()
            dt = (time.perf_counter() - t0) / 2000 * 1e6
        rows.append(dict(elems=size, us_per_fence=dt, label="MEASURED"))
        log(f"fence elems={size}: {dt:.1f} us/fence")
    res.add("fence", dict(rows=rows))


def cmd_grid(res):
    guard = S.IdleGuard(logf=open(os.path.join(OUT, "idle.log"), "a"))
    mem = S.mem_report()
    cap = mem["cap_bytes"] or 0
    res.add("mem_headroom", mem)
    ns = [int(v) for v in os.environ.get("P5_N", "1,4,8,12,16").split(",")]
    nbs = [int(v) for v in os.environ.get(
        "P5_NB", "63000,85000,100000,127000,130000,200000,254000,300000,400000,500000"
    ).split(",")]
    deep = [int(v) for v in os.environ.get("P5_DEEP", "700000,1000000").split(",") if v]
    deep_n = [int(v) for v in os.environ.get("P5_DEEP_N", "4,1").split(",")]
    layers = [int(v) for v in os.environ.get("P5_LAYERS", "20").split(",")]
    extra = os.environ.get("P5_EXTRA", "24:4,8:4")      # layer:n pairs on full nb list
    jobs = []
    for L in layers:
        for n in ns:
            for nb in nbs:
                jobs.append((L, n, nb))
        for n in deep_n:
            for nb in deep:
                jobs.append((L, n, nb))
    for pair in [p for p in extra.split(",") if p]:
        L, n = (int(v) for v in pair.split(":"))
        for nb in nbs + deep:
            jobs.append((L, n, nb))
    done = {(c["layer"], c["n"], c["nb"]) for c in res_cells(res)}
    ixs = {}
    cells = res_cells(res)
    dirty = True
    for (L, n, nb) in jobs:
        if (L, n, nb) in done:
            continue
        if L not in ixs:
            ixs[L] = S.build_indexer(L)
        ix = ixs[L]
        align = ix.candidate_block_size if ix.is_candidate_source else 1
        route, rw = rule_route(n, nb, align)
        t512 = min(nb, IX.tile_width(1, n, H, nb, align))
        wide = wide_width(n, nb, align)
        base = nb * 128 * 2 + n * 8192 * 4
        pk = pred_untiled_peak(n, nb, ix.uses_candidates)
        tag = f"L{L}_n{n}_nb{nb}"
        rec = dict(layer=L, n=n, nb=nb, ratio=ix.ratio, L_ctx=nb * max(ix.ratio, 1),
                   rule_route=route, rule_width=rw, t512=t512, wide=wide,
                   untiled_budget_bytes=untiled_bytes(n, nb),
                   budget_bytes=IX._TILE_BUDGET, pred_untiled_peak_mb=pk / 1e6,
                   cap_mb=cap / 1e6, label="MEASURED")
        if pk + base > cap:
            rec["status"] = "not run (mem)"
            cells.append(rec)
            log(f"{tag}: not run (mem) pred={pk/1e9:.2f}GB cap={cap/1e9:.2f}GB")
            res.add("grid_partial", dict(cells=cells))
            continue
        rig = Rig(ix, L, n, nb)
        arms = {"UNT": nb, "T512": t512}
        if wide < nb and wide != t512:
            arms["WIDE"] = wide
        if dirty:
            guard.wait_idle()
            dirty = False
        r = time_arms(rig, arms, guard, tag)
        rec["arms"] = r
        cur = "UNT" if route == "untiled" else "T512"
        best = min(("UNT", "T512"), key=lambda a: r[a]["ms"])
        rec["cur_arm"], rec["faster_tiled_vs_untiled"] = cur, best
        rec["t512_over_unt"] = r["T512"]["ms"] / r["UNT"]["ms"]
        rec["per_fence_us"] = ((r["T512"]["ms"] - r["UNT"]["ms"]) * 1e3
                               / max(1, r["T512"]["fences"]))
        if "WIDE" in r:
            rec["wide_over_unt"] = r["WIDE"]["ms"] / r["UNT"]["ms"]
        if any(v["discards"] for v in r.values()):
            dirty = True
        cells.append(rec)
        log(f"{tag}: rule={route:7s} UNT={r['UNT']['ms']:7.2f} (g {r['UNT']['gpu_ms']:.2f}, "
            f"iqr {r['UNT'].get('iqr_ms', 0):.2f}, pk {r['UNT']['peak_mb']:.0f}MB) "
            f"T512={r['T512']['ms']:7.2f} (g {r['T512']['gpu_ms']:.2f}, f={r['T512']['fences']}, "
            f"pk {r['T512']['peak_mb']:.0f}MB)"
            + (f" WIDE[{wide}]={r['WIDE']['ms']:7.2f} (g {r['WIDE']['gpu_ms']:.2f}, "
               f"f={r['WIDE']['fences']}, pk {r['WIDE']['peak_mb']:.0f}MB)" if "WIDE" in r else "")
            + f" ratio={rec['t512_over_unt']:.2f} fence={rec['per_fence_us']:.0f}us")
        guard.own_work()
        del rig
        mx.clear_cache()
        res.add("grid_partial", dict(cells=cells))
    res.add("grid", dict(cells=cells, discards=guard.discards, mem=mem))
    log("GRID_DONE")


def res_cells(res):
    for b in reversed(res.blobs):
        if b.get("_name") in ("grid_partial", "grid"):
            return list(b["cells"])
    return []


def cmd_anchors(res):
    """idxprof1 method verbatim (cmd_crossover's run_ref / run_untiled)."""
    guard = S.IdleGuard(logf=open(os.path.join(OUT, "idle.log"), "a"))
    mem = S.mem_report()
    cap = mem["cap_bytes"] or 0
    ix = S.build_indexer(20)
    fv = S.freqvec_for(20, ix.ratio)
    cells = []
    guard.wait_idle()
    for n, nb in ((8, 130000), (78, 130000), (512, 130000), (1024, 130000),
                  (4, 130000)):
        route_n = S.CHUNK if n >= 1024 else n
        tile = min(nb, IX.tile_width(1, route_n, H, nb, 8))
        x, qr, ik = S.synth_inputs(n, nb, seed=13)
        sp = max(0, nb - S.CHUNK)
        q32, w32, lens, _ = S.band_qw(ix, x, qr, sp, 0, n, fv)
        k = min(S.TOPK, nb)

        def run_t():
            return IX._tiled_scores_buffer(q32, ik, w32, lens, nb, k, tile)[:2]

        def run_u():
            return IX._tiled_scores_buffer(q32, ik, w32, lens, nb, k, nb)[:2]
        do_u = S.predict_peak_bytes(n, nb, "ref", tile=nb) < min(cap, 4e9)
        rt, ru = [], []
        for _ in range(2):
            rt.append(S.timed(run_t, REPS_HALF, WARM, guard=guard, gpu=True,
                              label=f"anchor{n} T", reset_peak=True))
            if do_u:
                ru.append(S.timed(run_u, REPS_HALF, WARM, guard=guard, gpu=True,
                                  label=f"anchor{n} U", reset_peak=True))
        t = S.merge_runs(rt)
        t["gpu_ms"] = float(np.median([g for r in rt for g in r.get("gpu_all", [])]))
        rec = dict(n=n, nb=nb, tile=tile, route_n=route_n,
                   rule=rule_route(route_n, nb, 8)[0], tiled=t, label="MEASURED")
        if ru:
            u = S.merge_runs(ru)
            u["gpu_ms"] = float(np.median([g for r in ru for g in r.get("gpu_all", [])]))
            rec["untiled"] = u
        cells.append(rec)
        log(f"anchor ({n},{nb}) tile={tile} rule={rec['rule']}: tiled={t['ms']:.2f} "
            f"(gpu {t['gpu_ms']:.2f})" + (f" untiled={rec['untiled']['ms']:.2f} "
                                         f"(gpu {rec['untiled']['gpu_ms']:.2f})" if ru else ""))
        guard.own_work()
        del x, qr, ik, q32, w32
        mx.clear_cache()
        res.add("anchors_partial", dict(cells=cells))
    res.add("anchors", dict(cells=cells, discards=guard.discards))
    log("ANCHORS_DONE")


def tie_keys(nb, period, seed=7):
    """Periodic keys: column j == column j % period -> every score repeats nb/period
    times, i.e. exact ties at (almost) every k-th boundary."""
    mx.random.seed(seed)
    base = (mx.random.normal((1, period, 128)) * 0.5).astype(mx.bfloat16)
    reps = -(-nb // period)
    ik = mx.concatenate([base] * reps, axis=1)[:, :nb]
    mx.eval(ik)
    return ik


def cmd_parity(res):
    """Bit-identity of the returned index sets (and L20's candidate mask) across the
    route arms at the shapes the rule change would move."""
    shapes = [tuple(int(v) for v in s.split(":")) for s in os.environ.get(
        "P5_PAR", "4:254000,4:300000,4:500000,4:1000000,1:300000,8:130000,8:300000,"
                  "12:100000,16:63000,16:200000").split(",")]
    ixs = {L: S.build_indexer(L) for L in (20, 24, 8)}
    rows = []
    for (n, nb) in shapes:
        for keys in ("random", "ties"):
            for L in (20, 24, 8):
                ix = ixs[L]
                if L == 8 and nb > 500000:
                    continue
                ik = tie_keys(nb, 977) if keys == "ties" else None
                rig = Rig(ix, L, n, nb, ik=ik, seed=21)
                align = rig.align
                arms = {"UNT": nb, "T512": min(nb, IX.tile_width(1, n, H, nb, align)),
                        "WIDE": wide_width(n, nb, align)}
                outs = {}
                for a, w in arms.items():
                    force(w)
                    try:
                        o = rig.call()
                        mx.eval(o)
                        outs[a] = o
                    finally:
                        restore()
                ref = outs["UNT"]
                ref_i = ref[0] if isinstance(ref, tuple) else ref
                rec = dict(layer=L, n=n, nb=nb, keys=keys, widths=arms)
                for a in ("T512", "WIDE"):
                    o = outs[a]
                    oi = o[0] if isinstance(o, tuple) else o
                    rec[f"{a}_idx_identical"] = bool(mx.array_equal(oi, ref_i).item())
                    rec[f"{a}_rows_differing"] = int(
                        mx.sum(mx.any(oi != ref_i, axis=-1)).item())
                    if isinstance(o, tuple):
                        rec[f"{a}_cand_identical"] = bool(mx.array_equal(o[1], ref[1]).item())
                # tie census on the reference row (fp32 at n<=16): k-th vs (k+1)-th
                q = ref_i
                rec["valid_slots"] = int(mx.sum(q >= 0).item())
                rows.append(rec)
                log(f"parity L{L} n={n} nb={nb} {keys}: "
                    + " ".join(f"{a}={rec[a + '_idx_identical']}" for a in ("T512", "WIDE"))
                    + (f" cand T512={rec.get('T512_cand_identical')} "
                       f"WIDE={rec.get('WIDE_cand_identical')}" if L == 20 else ""))
                del rig, outs
                mx.clear_cache()
                res.add("parity_partial", dict(rows=rows))
    # tie census, done once explicitly on the score row (fp32) the topk ranks
    res.add("parity", dict(rows=rows, all_identical=all(
        r["T512_idx_identical"] and r["WIDE_idx_identical"]
        and r.get("T512_cand_identical", True) and r.get("WIDE_cand_identical", True)
        for r in rows)))
    log("PARITY_DONE")


def cmd_ties(res):
    """Prove the tie-injected inputs actually put exact ties ON the k boundary:
    score the fp32 row the way production does (score body over the full width)
    and count rows where the k-th and (k+1)-th largest are equal."""
    ix = S.build_indexer(20)
    out = []
    for (n, nb) in ((4, 300000), (16, 63000), (4, 1000000)):
        for keys in ("random", "ties"):
            ik = tie_keys(nb, 977) if keys == "ties" else S.synth_inputs(n, nb, seed=21)[2]
            x, qr, _ = S.synth_inputs(n, nb, seed=21)
            fv = S.freqvec_for(20, ix.ratio)
            q32, w32, lens, _ = S.band_qw(ix, x, qr, nb - n, 0, n, fv)
            s = IX._score_body_eager(q32, ik.astype(mx.float32), w32)   # [1,n,nb] fp32
            s = mx.where(mx.arange(nb)[None, :] < lens, s, IX.NEG_INF)
            srt = mx.sort(s, axis=-1)[..., ::-1]
            kth, k1 = srt[..., 511], srt[..., 512]
            mx.eval(kth, k1)
            ties = int(mx.sum(kth == k1).item())
            out.append(dict(n=n, nb=nb, keys=keys, rows=n, rows_tied_at_k=ties))
            log(f"ties n={n} nb={nb} {keys}: rows with kth==(k+1)th: {ties}/{n}")
            mx.clear_cache()
    res.add("ties", dict(rows=out))


def cmd_rounds(res):
    """Decode-realistic sequence: nb grows by ~2 per call (2.08 tok/round, ratio 1), so
    every call sees a NEW nb -> the ragged last tile has a new width each round.  Exposes
    per-round recompile cost of the compiled tile body (keyed per nominal width; MLX
    retraces per input shape).  Arms run as separate sequences (fresh compile cache
    state each: _COMPILED_BODIES cleared), each 3 warm rounds + 24 timed rounds."""
    guard = S.IdleGuard(logf=open(os.path.join(OUT, "idle.log"), "a"))
    ix = S.build_indexer(int(os.environ.get("P5_ROUNDS_LAYER", "20")))
    fv = S.freqvec_for(ix.layer_id, ix.ratio)
    align = ix.candidate_block_size if ix.is_candidate_source else 1
    n = 4
    rows = []
    guard.wait_idle()
    for nb0 in [int(v) for v in os.environ.get("P5_ROUNDS_NB",
                                                "130000,300000,600000,1000000").split(",")]:
        nbmax = nb0 + 2 * 40
        x, qr, ik = S.synth_inputs(n, nbmax, seed=31)
        shared = S.Shared(None)
        arms = ["UNT", "T512", "WIDE", "CUR"]
        for rep in range(2):                                      # ABAB
            for a in arms:
                IX._COMPILED_BODIES.clear()
                walls, gpus, nbs = [], [], []
                for r in range(27):
                    nb = nb0 + 2 * (r + rep * 0)
                    w = {"UNT": nb, "T512": min(nb, IX.tile_width(1, n, H, nb, align)),
                         "WIDE": wide_width(n, nb, align)}.get(a)
                    if a != "CUR":
                        force(w)
                    ok, ev = guard.clean()
                    try:
                        mx.synchronize()
                        mx.metal.reset_gpu_time()
                        t0 = time.perf_counter()
                        out = ix(x, qr, nb - n, 0, fv, ik[:, :nb], shared)
                        mx.eval(out, shared.candidates) if shared.candidates is not None \
                            else mx.eval(out)
                        mx.synchronize()
                        dt = (time.perf_counter() - t0) * 1e3
                    finally:
                        restore()
                    if r >= 3 and ok:
                        walls.append(dt)
                        gpus.append(mx.metal.gpu_time_ns() / 1e6)
                        nbs.append(nb)
                    elif not ok:
                        guard._note(f"DISCARD rounds {a} nb={nb}: " + "; ".join(ev))
                rec = dict(n=n, nb0=nb0, arm=a, rep=rep, rule=rule_route(n, nb0, align)[0],
                           ms=float(np.median(walls)), mean_ms=float(np.mean(walls)),
                           max_ms=float(np.max(walls)), gpu_ms=float(np.median(gpus)),
                           walls=walls, gpus=gpus, compiled_bodies=len(IX._COMPILED_BODIES),
                           label="MEASURED")
                rows.append(rec)
                log(f"rounds nb0={nb0} {a} rep{rep}: median={rec['ms']:.2f} mean={rec['mean_ms']:.2f} "
                    f"max={rec['max_ms']:.2f} gpu={rec['gpu_ms']:.2f}")
                guard.own_work()
        del x, qr, ik
        mx.clear_cache()
        res.add("rounds_partial", dict(rows=rows))
    res.add("rounds", dict(rows=rows, discards=guard.discards))
    log("ROUNDS_DONE")


def cmd_rowbits(res):
    """Bitwise fp32 score-row parity across tile widths (the production per-tile loop of
    _tiled_scores_buffer, compiled body, fp32 row at n<=16), at remainder edge cases, plus a
    targeted tile-boundary tie (key at cap-1 copied to cap and boosted into the top-k), with
    the full Indexer.__call__ index-set compared across UNT / T512 / WIDE."""
    ix = S.build_indexer(20)
    fv = S.freqvec_for(20, ix.ratio)
    rows = []
    cases = [(4, 254201), (4, 2 * 254200 + 1), (4, 254200 * 3 - 1), (4, 1000003),
             (16, 63545), (16, 300001), (8, 127097), (12, 254200), (1, 1100000)]
    for (n, nb) in cases:
        align = 8
        cap = wide_width(n, nb, align)
        x, qr, ik = S.synth_inputs(n, nb, seed=41)
        # targeted boundary tie: duplicate key at cap-1 into cap (inside nb) and boost both
        if cap < nb:
            ikn = np.array(ik.astype(mx.float32))
            ikn[0, cap] = ikn[0, cap - 1]
            ik = mx.array(ikn).astype(mx.bfloat16)
            mx.eval(ik)
        q32, w32, lens, _ = S.band_qw(ix, x, qr, nb - n, 0, n, fv)
        def row_for(tile):
            body = IX._score_body(1, n, H, 128, tile)
            row = mx.zeros((1, n, nb), dtype=IX._row_dtype(n))
            for c0 in range(0, nb, tile):
                c1 = min(c0 + tile, nb)
                s = body(q32, ik[:, c0:c1].astype(mx.float32), w32)
                cols = mx.arange(c0, c1, dtype=mx.int32)
                row[:, :, c0:c1] = mx.where(cols[None, :] < lens, s, IX.NEG_INF)
                mx.eval(row)
            return row
        ref = row_for(nb)
        rec = dict(n=n, nb=nb, cap=cap, row_dtype=str(IX._row_dtype(n)))
        for nm, t in (("T512", min(nb, IX.tile_width(1, n, H, nb, align))), ("WIDE", cap)):
            r = row_for(t)
            rec[f"{nm}_width"] = t
            rec[f"{nm}_row_bitwise"] = bool(mx.array_equal(r, ref).item())
            rec[f"{nm}_tail"] = nb % t
        # tie at the boundary: equal scores at cap-1 / cap?
        if cap < nb:
            rec["boundary_cols_equal_score"] = bool(mx.all(ref[..., cap - 1] == ref[..., cap]).item())
        # full __call__ index sets (forced routes) on the same tied keys
        rig = Rig(ix, 20, n, nb, ik=ik, seed=41)
        outs = {}
        for a, w in (("UNT", nb), ("T512", rec["T512_width"]), ("WIDE", cap)):
            force(w)
            try:
                o = rig.call(); mx.eval(o); outs[a] = o
            finally:
                restore()
        for a in ("T512", "WIDE"):
            rec[f"{a}_idx_identical"] = bool(mx.array_equal(outs[a][0], outs["UNT"][0]).item())
            rec[f"{a}_cand_identical"] = bool(mx.array_equal(outs[a][1], outs["UNT"][1]).item())
        rows.append(rec)
        log(f"rowbits n={n} nb={nb} cap={cap}: " + " ".join(
            f"{k}={v}" for k, v in rec.items() if k.endswith(("bitwise", "identical", "equal_score"))))
        del x, qr, ik, q32, w32, ref, rig, outs
        mx.clear_cache()
        res.add("rowbits_partial", dict(rows=rows))
    res.add("rowbits", dict(rows=rows, all_ok=all(
        v for r in rows for k, v in r.items() if k.endswith(("bitwise", "identical")))))
    log("ROWBITS_DONE")


def cmd_rowdiag(res):
    """Is the fp32 score row run-to-run deterministic at a FIXED width?  Recompute each
    width 3x; report self-consistency, cross-width diff count and max |diff|, and whether
    the score body is compiled (DSV41_INDEXER_COMPILE) -- isolates kernel nondeterminism
    from a width effect."""
    ix = S.build_indexer(20)
    fv = S.freqvec_for(20, ix.ratio)
    rows = []
    for (n, nb) in [(4, 254201), (4, 508401), (4, 1000003), (8, 127097)]:
        x, qr, ik = S.synth_inputs(n, nb, seed=41)
        q32, w32, lens, _ = S.band_qw(ix, x, qr, nb - n, 0, n, fv)
        def row_for(tile, compiled=True):
            body = IX._score_body(1, n, H, 128, tile) if compiled else IX._score_body_eager
            row = mx.zeros((1, n, nb), dtype=mx.float32)
            for c0 in range(0, nb, tile):
                c1 = min(c0 + tile, nb)
                s = body(q32, ik[:, c0:c1].astype(mx.float32), w32)
                row[:, :, c0:c1] = s
                mx.eval(row)
            return row
        widths = dict(UNT=nb, T512=512, WIDE=wide_width(n, nb, 8))
        for comp in (True, False):
            reps = {k: [row_for(w, comp) for _ in range(3)] for k, w in widths.items()}
            rec = dict(n=n, nb=nb, compiled=comp)
            for k, rs in reps.items():
                rec[f"{k}_selfconsistent"] = all(bool(mx.array_equal(rs[0], r).item()) for r in rs[1:])
                rec[f"{k}_selfdiff_elems"] = max(int(mx.sum(rs[0] != r).item()) for r in rs[1:])
            for k in ("T512", "WIDE"):
                d = reps[k][0] != reps["UNT"][0]
                rec[f"{k}_vs_UNT_elems"] = int(mx.sum(d).item())
                rec[f"{k}_vs_UNT_maxabs"] = float(mx.max(mx.abs(reps[k][0] - reps["UNT"][0])).item())
            rows.append(rec)
            log("rowdiag " + " ".join(f"{a}={b}" for a, b in rec.items()))
            del reps
            mx.clear_cache()
        res.add("rowdiag_partial", dict(rows=rows))
    res.add("rowdiag", dict(rows=rows))
    log("ROWDIAG_DONE")


class _NoEvalMx:
    """Proxy for indexer.py's module-level ``mx`` with eval a no-op: removes the per-tile
    fence of _tiled_scores_buffer WITHOUT changing a single kernel (same per-tile ops)."""

    def __getattr__(self, k):
        return getattr(mx, k)

    @staticmethod
    def eval(*a, **k):
        return None


def cmd_nofence(res):
    """Candidate R-C: keep today's width-512 tiles (today's exact bits) but drop the
    per-tile host fence.  Wall/gpu/peak vs T512/UNT/WIDE + bitwise idx/cand vs T512."""
    guard = S.IdleGuard(logf=open(os.path.join(OUT, "idle.log"), "a"))
    ix = S.build_indexer(20)
    cells = []
    guard.wait_idle()
    for (n, nb) in [(4, 300000), (4, 500000), (4, 1000000), (16, 300000)]:
        rig = Rig(ix, 20, n, nb)
        t512 = min(nb, IX.tile_width(1, n, H, nb, 8))

        def nofence():
            force(t512)
            real = IX.mx
            IX.mx = _NoEvalMx()
            try:
                return rig.call()
            finally:
                IX.mx = real
                restore()
        # bit check vs T512 (fenced)
        force(t512)
        try:
            ref = rig.call(); mx.eval(ref)
        finally:
            restore()
        o = nofence(); mx.eval(o)
        same = bool(mx.array_equal(o[0], ref[0]).item()) and bool(mx.array_equal(o[1], ref[1]).item())
        r = time_arms(rig, {"UNT": nb, "T512": t512, "WIDE": wide_width(n, nb, 8)}, guard, f"nf{n}_{nb}")
        nf = []
        for _ in range(2):
            nf.append(S.timed(nofence, REPS_HALF, WARM, guard=guard, gpu=True,
                              label=f"nf{n}_{nb} NOFENCE", reset_peak=True))
        m = S.merge_runs(nf)
        m["gpu_ms"] = float(np.median([g for x in nf for g in x.get("gpu_all", [])]))
        r["NOFENCE512"] = m
        cells.append(dict(n=n, nb=nb, arms=r, nofence_bitwise_vs_t512=same, label="MEASURED"))
        log(f"nofence n={n} nb={nb}: bitwise_vs_T512={same} " + " ".join(
            f"{a}={v['ms']:.2f}(g{v['gpu_ms']:.2f},pk{v['peak_mb']:.0f})" for a, v in r.items()))
        guard.own_work()
        del rig
        mx.clear_cache()
        res.add("nofence_partial", dict(cells=cells))
    res.add("nofence", dict(cells=cells, discards=guard.discards))
    log("NOFENCE_DONE")


def main():
    mode = sys.argv[1]
    tag = (sys.argv[2] if len(sys.argv) > 2
           else os.environ.get("P5_TAG", f"p5_{mode}"))
    S.OUT = OUT
    res = S.Res(tag)
    IX._coll.all_sum = S._stub_all_sum
    log(f"mode={mode} tag={tag} pid={os.getpid()} TILE={IX._TILE} BUDGET={IX._TILE_BUDGET} "
        f"MIN_NB={IX._TILE_MIN_NB} HIER={IX._HIER} L2FULL={IX._L2_FULL} "
        f"FENCE_MIN_ROWS={IX._FENCE_MIN_ROWS} GPU_TIME={S.HAVE_GPU_TIME}")
    {"fence": cmd_fence, "grid": cmd_grid, "anchors": cmd_anchors,
     "parity": cmd_parity, "ties": cmd_ties, "rounds": cmd_rounds,
     "rowbits": cmd_rowbits, "rowdiag": cmd_rowdiag,
     "nofence": cmd_nofence}[mode](res)
    res.add("peak_process_mb", dict(peak_mb=mx.get_peak_memory() / 1e6))
    res.write()
    log(f"ALL_DONE peak_process={mx.get_peak_memory()/1e9:.2f}GB")


if __name__ == "__main__":
    main()
