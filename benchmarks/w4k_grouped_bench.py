#!/usr/bin/env python3
"""W4K Mac harness: expert-grouped multi-row MoE decode A/B + bit-exactness.

Runs ONLY on a Mac with a Metal backend. Everything the LXC could prove is
already proven there (tests/test_dsv41_moe_grouped.py, 14 tests); this
harness closes the two things the LXC cannot:

  1. BIT-EXACTNESS of the REAL kernels: _decode_grouped vs _decode_fused2 on
     random trellis data must give byte-identical ygu and y (the Metal
     compiler's fma/contraction decisions are the residual risk; the source
     statement order per slot is identical, so any mismatch here is a Metal
     codegen effect and must be reported, not silenced).
  2. PERF: fused2 vs grouped at R=1..5 on (a) the measured real routing
     histograms from w3gpu (per-layer distinct-expert counts), (b) ident
     (all rows share row 0's experts), (c) dist (all slots distinct), and
     (d) captured-routing replay if a capture file is provided.

Also records a Metal GPU trace split (ALU vs load) procedure for R=1 and
R=4 via `xcrun` Metal capture - see the CAPTURE section at the bottom.

Usage (single node, production rules: lockf, EXL3_MM_MAX_ROWS, <= 8 GB):
  # arm A (control): flag off
  lockf -k ~/dsv41-gpu.lock env DSV41_MOE_GROUPED=0 EXL3_MM_MAX_ROWS=100000 \
    MTL_DISABLE_TIMEOUT=1 ~/repos/exo/.venv/bin/python w4k_grouped_bench.py --tag base
  # arm B: flag on
  lockf -k ~/dsv41-gpu.lock env DSV41_MOE_GROUPED=1 EXL3_MM_MAX_ROWS=100000 \
    MTL_DISABLE_TIMEOUT=1 ~/repos/exo/.venv/bin/python w4k_grouped_bench.py \
    --tag grouped

Kill criterion (write into the report): the grouped arm must save >= 4 ms
per verify round at gamma 3 (R=4); if the saving is below that, keep the
flag OFF and record the numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import mlx.core as mx

# Production geometry per rank (w3gpu A: E=384, D=5120, H=1152 slice of 2304)
E = int(os.environ.get("W4K_E", "384"))
D = int(os.environ.get("W4K_D", "5120"))
H = int(os.environ.get("W4K_H", "1152"))
K = 6
KK = 6  # top-k routed experts (shared expert separate)

# w3gpu run w3a_1 rank 0: per-layer distinct experts at R=1..5 (mean and the
# 40-layer vectors). Used to build real-routing-like index patterns without
# a capture file.
W3GPU_DISTINCT = {
    1: 6.00,
    2: 10.22,
    3: 13.80,
    4: 16.88,
    5: 20.75,
}


def log(*a):
    print(*a, flush=True)


def build_module(seed=1234):
    from mlx_lm.models.exl3.exl3_moe import EXL3SwitchGLU
    from mlx_lm.models.exl3.ref.codebook import CodebookMode

    rng = np.random.default_rng(seed)
    gu_tr = mx.array(
        rng.integers(0, 65536, (D // 16, E * 2 * (H // 16), 16 * K)).astype(np.uint16)
    )
    dn_tr = mx.array(
        rng.integers(0, 65536, (H // 16, E * (D // 16), 16 * K)).astype(np.uint16)
    )
    # signs +-1 like unpacked suh/svh
    gu_suh = mx.array(
        rng.choice(np.array([-1, 1], np.float16), (E, 2, D)).astype(np.float16)
    )
    gu_svh = mx.array(
        rng.choice(np.array([-1, 1], np.float16), (E, 2 * H)).astype(np.float16)
    )
    dn_suh = mx.array(
        rng.choice(np.array([-1, 1], np.float16), (E, H)).astype(np.float16)
    )
    dn_svh = mx.array(
        rng.choice(np.array([-1, 1], np.float16), (E, D)).astype(np.float16)
    )
    sg = EXL3SwitchGLU(
        gu_trellis=gu_tr, gu_suh=gu_suh, gu_svh=gu_svh,
        dn_trellis=dn_tr, dn_suh=dn_suh, dn_svh=dn_svh,
        k=K, cb=CodebookMode.MUL1, activation="silu_clamp",
    )
    mx.eval(sg._gu_trellis, sg._dn_trellis, sg._gu_suh, sg._gu_svh,
            sg._dn_suh, sg._dn_svh)
    return sg


def routing(kind: str, R: int, rng):
    """Index patterns matching w3gpu's ident / real / dist conditions."""
    if kind == "ident":
        row0 = rng.choice(E, KK, replace=False)
        return np.tile(row0, (R, 1))
    if kind == "dist":
        return np.array(
            [rng.choice(E, KK, replace=False) for _ in range(R)]
        )
    if kind == "real":
        # match the measured DISTINCT-expert count: draw rows so that the
        # union of experts hits the w3gpu mean for this R
        target = int(round(W3GPU_DISTINCT[R]))
        rows = [rng.choice(E, KK, replace=False)]
        while len(np.unique(np.concatenate(rows))) < min(target, R * KK):
            rows.append(rng.choice(E, KK, replace=False))
            if len(rows) > R:
                break
        while len(rows) < R:
            rows.append(rng.choice(E, KK, replace=False))
        return np.stack(rows[:R])
    raise ValueError(kind)


def timed(fn, reps=25, warmup=6):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--bitexact-only", action="store_true")
    ap.add_argument("--capture", default=None,
                    help="npz with keys x_r{R}, idx_r{R} (captured routing replay)")
    args = ap.parse_args()

    from mlx_lm.models.exl3 import exl3_moe as M
    grouped = M._MOE_GROUPED
    log(f"[w4k] tag={args.tag} DSV41_MOE_GROUPED={'1' if grouped else '0'} "
        f"device={mx.default_device()}")
    if str(mx.default_device().type) != "gpu":
        log("[w4k] FATAL: this harness needs the Metal backend (a Mac); "
            "the LXC path is tests/test_dsv41_moe_grouped.py")
        raise SystemExit(3)

    sg = build_module()
    rng = np.random.default_rng(9)

    # ---- 1. bit-exactness vs the fused2 kernel (run in BOTH arms; the flag
    # only picks the __call__ path, the methods are both present) ----------
    log("[w4k] bit-exactness: _decode_grouped vs _decode_fused2 (random trellis)")
    for R in (1, 4, 5, 8):
        x = mx.array(rng.standard_normal((R, D)).astype(np.float16))
        idx = mx.array(routing("real", R, rng).astype(np.int32))
        y0 = sg._decode_fused2(x, idx)
        y1 = sg._decode_grouped(x, idx)
        mx.eval(y0, y1)
        same = mx.array_equal(y0, y1)
        dmax = float(
            mx.abs(y0.astype(mx.float32) - y1.astype(mx.float32)).max()
        )
        log(f"  R={R}: array_equal={bool(same)} max|d|={dmax:.3e}")
        if not bool(same):
            log("  !! NOT bit-exact -- Metal codegen effect; report and keep flag OFF")

    if args.bitexact_only:
        return

    # ---- 2. perf: this arm's path over R=1..5 x {ident, real, dist} ------
    results = {"tag": args.tag, "grouped": grouped, "moe_ms": {}, "distinct": {}}
    for R in (1, 2, 3, 4, 5):
        for kind in ("ident", "real", "dist"):
            idx_np = routing(kind, R, rng)
            x = mx.array(rng.standard_normal((R, D)).astype(np.float16))
            idx = mx.array(idx_np.astype(np.int32))
            def fn():
                return sg(x[None], idx[None])

            med, ts = timed(fn, reps=args.reps)
            results["moe_ms"][f"{R}:{kind}"] = med
            if kind == "real":
                results["distinct"][R] = int(len(np.unique(idx_np.reshape(-1))))
            log(f"  R={R} {kind:5s} distinct={len(np.unique(idx_np.reshape(-1))):2d} "
                f"median {med:.3f} ms (min {ts[0]:.3f} max {ts[-1]:.3f})")

    # captured-routing replay if provided
    if args.capture and os.path.exists(args.capture):
        cap = np.load(args.capture)
        for R in (1, 2, 3, 4, 5):
            kx, ki = f"x_r{R}", f"idx_r{R}"
            if kx not in cap:
                continue
            x = mx.array(cap[kx].astype(np.float16))
            idx = mx.array(cap[ki].astype(np.int32))
            med, ts = timed(lambda: sg(x[None], idx[None]), reps=args.reps)
            results["moe_ms"][f"{R}:capture"] = med
            log(f"  R={R} capture median {med:.3f} ms")

    peak = mx.get_peak_memory()
    results["peak_gb"] = peak / 1e9
    log(f"[w4k] peak memory {peak/1e9:.2f} GB")
    out = f"w4k_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    log(f"[w4k] wrote {out}")

    # ---- 3. ALU-vs-load split procedure (manual, R=1 and R=4) -------------
    log("""
CAPTURE PROCEDURE (run once per arm, R=1 and R=4):
  1. xcrun -o capture.gputrace --set 'capture-mode=all' \\
       env DSV41_MOE_GROUPED=<0|1> python w4k_grouped_bench.py --bitexact-only
     (or use Xcode Metal Capturer with the 'Capture GPU Work' template)
  2. In the trace, pick ONE representative exl3_moe_gateup2* / gateup2g launch:
     - Instruction/ALU utilization histogram (SIMD integer + fp pipes)
     - Memory/Load-Store unit utilization + bytes
  3. Compute decode-ALU share = ALU-busy cycles / total slot time.
     If decode ALU > 70% of the slot time at R=1, then R=1's remaining cost
     is per-slot decode work and GROUPING CANNOT help R=1 (state that in
     the report: R=1 gains need per-slot decode work, not grouping).
""")


if __name__ == "__main__":
    main()