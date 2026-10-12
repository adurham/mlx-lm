#!/usr/bin/env python3
"""idx_report.py -- fold the idxprof1 raw JSONs into the report's markdown tables.

Reads /private/tmp/idxprof1/raw/*.json (or P1_OUT) and prints markdown to stdout.
No measurements are made here.  Every measured number comes from a recorded JSON
blob; every derived number is recomputed here from the recorded (shape, ms) pair
so the roofline classification is reproducible from the artifacts alone.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

OUT = os.environ.get("P1_OUT", "/private/tmp/idxprof1/raw")
H, D, TOPK, BLOCK, CAND_BLKS = 32, 128, 512, 8, 2048


def load(tag):
    p = os.path.join(OUT, f"{tag}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def blobs(d, name):
    return [b for b in (d or {}).get("blobs", []) if b.get("_name") == name]


def fmt(x, n=2):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    return f"{x:.{n}f}"


def cls(bw_frac, fl_frac):
    """The plan's rule: >=60 % of a MEASURED peak => bound by it."""
    if fl_frac >= 0.6 and fl_frac >= bw_frac:
        return f"compute-bound ({fl_frac:.0%} peak)"
    if bw_frac >= 0.6:
        return f"bandwidth-bound ({bw_frac:.0%} peak)"
    return f"other (bw {bw_frac:.0%}, flop {fl_frac:.0%})"


def ref_parts_roofline(n, nb, tile, row_b=2):
    """DERIVED bytes/flops per REFERENCE part (see split_ref docstring)."""
    of = CAND_BLKS
    return {
        "iii_kcache_read": dict(bytes=nb * D * 2, flops=0.0),
        "i_score_gemm": dict(bytes=n * nb * (H * 4 * 2) + n * nb * (row_b * 2),
                             flops=2.0 * n * nb * H * D),
        "ii_select_topk": dict(bytes=n * nb * row_b + n * TOPK * 8, flops=float(n * nb)),
        "iv_scatter": dict(bytes=n * TOPK * 4 * 3, flops=0.0),
    }


def hier_parts_roofline(n, nb, estrip, of=16):
    c = min(nb, (TOPK + of) * BLOCK)
    return {
        "i_coarse_sweep": dict(bytes=nb * D * 2 + n * nb * 2
                                     + n * ((nb + BLOCK - 1) // BLOCK) * 4,
                               flops=2.0 * n * nb * H * D),
        "ii_select_blocks": dict(bytes=n * ((nb + BLOCK - 1) // BLOCK) * 4,
                                 flops=float(n * ((nb + BLOCK - 1) // BLOCK))),
        "iii_exact_rescore": dict(bytes=n * c * D * 2 + n * c * D * 4 + n * c * 4,
                                  flops=2.0 * n * c * H * D),
        "iv_scatter": dict(bytes=n * TOPK * 4 * 3, flops=0.0),
    }


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    p1a = load("p1a_split")
    p1a2 = load("p1a_tile3840")
    p1b = load("p1b_cross")
    prov = (p1b or p1a or {}).get("provenance", {})

    print("### provenance\n")
    for k, v in prov.items():
        print(f"- `{k}` = `{v}`")
    print()

    if which in ("all", "1a") and p1a:
        roo = blobs(p1a, "roofline")
        mem = blobs(p1a, "mem_headroom")
        if mem:
            m = mem[0]
            print(f"**mem at 1A start:** avail {m['available_bytes']/1e9:.1f} GB, "
                  f"cap(50 %) {m['cap_bytes']/1e9:.1f} GB ({m.get('method')})\n")
        peak_bw = None
        peak_tf = None
        if roo:
            r = roo[0]
            peak_bw, peak_tf = r["stream_bw_gbs"], r["tflops"]
            print(f"**roofline (MEASURED, studio2):** copy 1 GB {r['copy_ms']:.2f} ms → "
                  f"**{r['stream_bw_gbs']:.1f} GB/s**; add {r['add_bw_gbs']:.1f} GB/s; "
                  f"fp16 {r['matmul_n']}³ {r['matmul_ms']:.2f} ms → "
                  f"**{r['tflops']:.1f} TFLOP/s**"
                  + (f"; indexer-shape score body (n=1024,t=504,h=32,d=128) "
                     f"{r['idx_shape_ms']:.2f} ms → {r['idx_shape_tflops']:.1f} TFLOP/s"
                     if r.get("idx_shape_ms") else "") + "\n")
        rows = [b for b in p1a["blobs"] if re.match(r"^L\d+_L\d+_split$", b.get("_name", ""))]
        rows.sort(key=lambda r: (r["layer"], r["L"]))
        print("#### 1A — sub-op split per (layer, L), per-rank post-qsplit rows\n")
        for r in rows:
            n, nb, tile, estrip = r["n_local"], r["nb"], r["tile"], r["estrip"]
            print(f"**layer {r['layer']} (ratio {r['ratio']}"
                  f"{', candidate source' if r['candidate'] else ''}"
                  f"{', consumer' if r['consumer'] else ''}) L={r['L']} → nb={nb}, "
                  f"rows={n}, tile={tile}, estrip={estrip}**\n")
            print("| sub-op | ms | % of Σ | DERIVED bytes | GB/s | DERIVED FLOPs | "
                  "TFLOP/s | class vs MEASURED peak |")
            print("|---|---:|---:|---:|---:|---:|---:|---|")
            pr, rf = r["ref_parts_ms"], ref_parts_roofline(n, nb, tile)
            s = r["ref_parts_sum_ms"]
            for k, v in sorted(pr.items(), key=lambda kv: -kv[1]):
                b, fl = rf[k]["bytes"], rf[k]["flops"]
                gbs = b / (v / 1e3) / 1e9 if v else float("nan")
                tf = fl / (v / 1e3) / 1e12 if v else float("nan")
                print(f"| {k} | {v:.3f} | {100*v/s:.1f} % | {b/1e6:.1f} MB | "
                      f"{gbs:.0f} | {fl:.3g} | {tf:.2f} | "
                      f"{cls(b/(v/1e3)/(peak_bw*1e9), fl/(v/1e3)/(peak_tf*1e12))} |")
            print(f"| **Σ parts** | **{s:.2f}** | | | | | | total "
                  f"{r['ref_total_ms']['ms']:.2f} ms (IQR {fmt(r['ref_total_ms'].get('iqr_ms'))}, "
                  f"n={r['ref_total_ms']['reps_used']}) → "
                  f"**{'MEASURED' if r['ref_validity_within_15pct'] else 'BOUND'}** "
                  f"({100*(s-r['ref_total_ms']['ms'])/r['ref_total_ms']['ms']:+.1f} %) |")
            if r.get("ref_untiled_ms"):
                print(f"| (pure untiled `TILE=0`, diagnostic) | "
                      f"{r['ref_untiled_ms']['ms']:.2f} | | | | | | |")
            ph, hf = r["hier_parts_ms"], hier_parts_roofline(n, nb, estrip)
            hs = r["hier_parts_sum_ms"]
            for k, v in sorted(ph.items(), key=lambda kv: -kv[1]):
                b, fl = hf[k]["bytes"], hf[k]["flops"]
                gbs = b / (v / 1e3) / 1e9 if v else float("nan")
                tf = fl / (v / 1e3) / 1e12 if v else float("nan")
                print(f"| HIER {k} | {v:.3f} | {100*v/hs:.1f} % | {b/1e6:.1f} MB | "
                      f"{gbs:.0f} | {fl:.3g} | {tf:.2f} | "
                      f"{cls(b/(v/1e3)/(peak_bw*1e9), fl/(v/1e3)/(peak_tf*1e12))} |")
            print(f"| **Σ parts** | **{hs:.2f}** | | | | | | total "
                  f"{r['hier_total_ms']['ms']:.2f} ms (IQR {fmt(r['hier_total_ms'].get('iqr_ms'))}, "
                  f"n={r['hier_total_ms']['reps_used']}) → "
                  f"**{'MEASURED' if r['hier_validity_within_15pct'] else 'BOUND'}** "
                  f"({100*(hs-r['hier_total_ms']['ms'])/r['hier_total_ms']['ms']:+.1f} %) |")
            print(f"\n- wall: REF {r['ref_total_ms']['ms']:.2f} ms (gpu-busy "
                  f"{fmt(r['ref_total_ms'].get('gpu_ms'))} ms) vs HIER "
                  f"{r['hier_total_ms']['ms']:.2f} ms (gpu-busy "
                  f"{fmt(r['hier_total_ms'].get('gpu_ms'))} ms) → "
                  f"wall ratio **{r['hier_total_ms']['ms']/r['ref_total_ms']['ms']:.3f}**, "
                  f"gpu-busy ratio "
                  f"**{(r['hier_total_ms'].get('gpu_ms') or float('nan'))/(r['ref_total_ms'].get('gpu_ms') or float('nan')):.3f}**")
            print(f"- peak delta per call: REF {fmt(r['ref_total_ms'].get('peak_mb'),0)} MB, "
                  f"HIER {fmt(r['hier_total_ms'].get('peak_mb'),0)} MB\n")
        for b in [x for x in p1a["blobs"] if "predicted_ref_mb" in x and "not run" in str(x.get("status", ""))]:
            print(f"- **not run (mem)**: {b['_name']}")
        dis = blobs(p1a, "discards")
        if dis:
            print(f"\n**discards (1A):** {dis[0]['count']}")
        print()
        if p1a2:
            rows2 = [b for b in p1a2["blobs"] if re.match(r"^L\d+_L\d+_split$", b.get("_name", ""))]
            for r in rows2:
                print(f"- **diagnostic (tile={r['tile']}, TILE_MB=1024)**: "
                      f"{r['_name']} REF {r['ref_total_ms']['ms']:.2f} ms "
                      f"(gpu {fmt(r['ref_total_ms'].get('gpu_ms'))}) vs HIER "
                      f"{r['hier_total_ms']['ms']:.2f} ms → wall ratio "
                      f"{r['hier_total_ms']['ms']/r['ref_total_ms']['ms']:.3f} "
                      f"(production tile=504 ratio for comparison)")
            print()

    if which in ("all", "1b"):
        if not p1b:
            print("### 1B — crossover: not available yet\n")
        else:
            c = blobs(p1b, "crossover")
            c = c[0] if c else None
            if c is None:
                print("### 1B — crossover (in progress, partial cells)\n")
                part = blobs(p1b, "crossover_partial")
                if part:
                    c = part[-1]
            if c:
                cells, nbs = c["cells"], c["nb"]
                rows = c["rows"]
                print(f"### 1B — REF vs HIER map (layer {c['layer']}, ratio {c['ratio']}; "
                      f"value = t_HIER / t_REF; **< 1 = HIER faster**)\n")
                print("| rows → / nb ↓ | " + " | ".join(str(x) for x in rows) + " |")
                print("|---|" + "---:|" * len(rows))
                for nb in nbs:
                    line = [f"| {nb} "]
                    for n in rows:
                        cell = next((x for x in cells if x["n"] == n and x["nb"] == nb), None)
                        if cell is None:
                            line.append("| — ")
                        elif "status" in cell:
                            line.append("| mem ")
                        else:
                            line.append(f"| {cell['speedup_hier_over_ref']:.3f} ")
                    print("".join(line) + "|")
                print("\n| rows | nb | S = rows·32·nb | t_ref ms (IQR, n) | t_hier ms (IQR, n) | "
                      "t_untiled ms | hier/ref | hier/untiled | peak ref/hier MB |")
                print("|---|---|---:|---:|---:|---:|---:|---:|---|")
                for x in cells:
                    if "status" in x:
                        print(f"| {x['n']} | {x['nb']} | | not run (mem) pred "
                              f"{x['predicted_ref_mb']:.0f}/{x['predicted_hier_mb']:.0f} MB "
                              f"vs cap {x['cap_mb']:.0f} MB | | | | |")
                        continue
                    u = x.get("untiled_ms")
                    u_ms = f"{u['ms']:.3f}" if u else ""
                    u_rat = f"{x['speedup_hier_over_untiled']:.3f}" if u else ""
                    print(f"| {x['n']} | {x['nb']} | {x['S']:.3g} | "
                          f"{x['ref_ms']['ms']:.3f} ({fmt(x['ref_ms'].get('iqr_ms'))}, "
                          f"{x['ref_ms']['reps_used']}) | "
                          f"{x['hier_ms']['ms']:.3f} ({fmt(x['hier_ms'].get('iqr_ms'))}, "
                          f"{x['hier_ms']['reps_used']}) | "
                          f"{u_ms} | "
                          f"{x['speedup_hier_over_ref']:.3f} | "
                          f"{u_rat} | "
                          f"{fmt(x.get('ref_peak_mb'), 0)}/"
                          f"{fmt(x.get('hier_peak_mb'), 0)} |")
                f = c.get("fit")
                if f:
                    print(f"\n**fit:** one-param S threshold S* = {f['one_param_S_star']:.4g} "
                          f"→ fits = **{f['one_param_fits']}** "
                          f"(misclassified {f['one_param_misclassified']} of the "
                          f"|speedup−1|>10 % cells)")
                    for b in f["bad"]:
                        print(f"  - bad: rows={b['n']} nb={b['nb']} S={b['S']:.4g} "
                              f"speedup={b['speedup']:.3f}")
                    t = f["two_param"]
                    print(f"- two-param form: S ≤ {t['S_star']:.4g} AND nb ≤ {t['L_star']} "
                          f"→ fits = **{t['fits']}** (misclassified {t['misclassified']})")
                    for b in t["bad"]:
                        print(f"  - bad: rows={b['n']} nb={b['nb']} S={b['S']:.4g} "
                              f"speedup={b['speedup']:.3f}")
                print("\n**reproduction checks:**")
                for n in (512, 1024):
                    x = next((x for x in cells if x["n"] == n and x["nb"] == 16384), None)
                    if x and "status" not in x:
                        print(f"- (a) nb=16384, rows={n}: hier/ref = "
                              f"{x['speedup_hier_over_ref']:.3f}"
                              + (f"; hier/untiled-row = {x['speedup_hier_over_untiled']:.3f}"
                                 if x.get("untiled_ms") else "")
                              + "   [target 1.71–2.05; the pB provenance compared HIER "
                                "against the untiled row path]")
                x = next((x for x in cells if x["n"] == 1024 and x["nb"] == 100000), None)
                if x and "status" not in x:
                    print(f"- (b) nb=100000, rows=1024 (production full chunk): "
                          f"hier/ref = {x['speedup_hier_over_ref']:.3f}   [target 0.9–1.1]")
                print()


if __name__ == "__main__":
    main()
