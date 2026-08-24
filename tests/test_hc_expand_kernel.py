# Copyright © 2026 Apple Inc.

"""Standalone correctness + microbench for the fused hc_expand Metal kernel.

Runs offline against the current compiled ``_hc_expand_op`` reference to
validate numerical equivalence (mean rel err ~ bf16 rounding class) and
report per-shape speedup. Kept as a standalone ``__main__`` script — the
main ``tests/test_models.py`` module has an unrelated collection-time
import failure in this fork that blocks running scoped ``unittest`` cases
through ``pytest`` here.

Usage::

    cd mlx-lm && uv run python tests/test_hc_expand_kernel.py

Exit code 0 iff every correctness case passes.
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
from mlx_lm.models.hyper_connection import (
    _hc_expand_op,
    _make_hc_expand_kernel,
)


def _rel_err(a: mx.array, b: mx.array) -> float:
    a_f = a.astype(mx.float32)
    b_f = b.astype(mx.float32)
    denom = mx.maximum(mx.abs(b_f), mx.array(1e-6))
    return float((mx.abs(a_f - b_f) / denom).mean())


def _max_abs(a: mx.array, b: mx.array) -> float:
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


def _has_nan_inf(a: mx.array) -> bool:
    return bool(mx.any(mx.isnan(a) | mx.isinf(a)))


def _make_kernel_call():
    kernel = _make_hc_expand_kernel()
    if kernel is None:
        raise RuntimeError("Metal GPU not available; kernel unbuildable.")

    def call(x, residual, post, comb):
        B, L, HC, D = residual.shape
        (out,) = kernel(
            inputs=[x, residual, post, comb],
            template=[
                ("T", x.dtype),
                ("U", x.dtype),
                ("HC", HC),
                ("D", D),
            ],
            grid=(B * L * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, HC, D)],
            output_dtypes=[x.dtype],
        )
        return out

    return call


def main() -> int:
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        print("SKIP: Metal GPU not available")
        return 0

    kernel_call = _make_kernel_call()
    fails = 0

    # ---- Case 1: realistic magnitudes at production shape ----
    B, L, HC, D = 1, 2048, 4, 4096
    scale = 2.2
    mx.random.seed(0)
    x = (mx.random.normal(shape=(B, L, D)) * scale).astype(mx.bfloat16)
    residual = (mx.random.normal(shape=(B, L, HC, D)) * scale).astype(
        mx.bfloat16
    )
    post = mx.random.uniform(-1, 1, shape=(B, L, HC)).astype(mx.float32)
    comb = mx.random.uniform(-1, 1, shape=(B, L, HC, HC)).astype(mx.float32)
    ref = _hc_expand_op(x, residual, post, comb)
    got = kernel_call(x, residual, post, comb)
    mx.eval(ref, got)
    rel = _rel_err(got, ref)
    mabs = _max_abs(got, ref)
    ok = rel <= 1e-3 and not _has_nan_inf(got)
    fails += int(not ok)
    print(
        f"[case1 realistic {B}x{L}x{HC}x{D}]  max_abs={mabs:.3e}  "
        f"mean_rel={rel:.3e}  {'PASS' if ok else 'FAIL'}"
    )

    # ---- Case 2: asymmetric deterministic (catches transpose bugs) ----
    Bs, Ls, HCs, Ds = 1, 3, 4, 8
    xs = (
        mx.arange(Bs * Ls * Ds, dtype=mx.float32).reshape(Bs, Ls, Ds) * 0.01
    ).astype(mx.bfloat16)
    rs = (
        mx.arange(Bs * Ls * HCs * Ds, dtype=mx.float32).reshape(
            Bs, Ls, HCs, Ds
        )
        * 0.005
    ).astype(mx.bfloat16)
    ps = mx.arange(Bs * Ls * HCs, dtype=mx.float32).reshape(Bs, Ls, HCs) * 0.1
    cs = (
        mx.arange(Bs * Ls * HCs * HCs, dtype=mx.float32).reshape(
            Bs, Ls, HCs, HCs
        )
        * 0.05
    )
    ref2 = _hc_expand_op(xs, rs, ps, cs)
    got2 = kernel_call(xs, rs, ps, cs)
    mx.eval(ref2, got2)
    rel2 = _rel_err(got2, ref2)
    mabs2 = _max_abs(got2, ref2)
    ok2 = rel2 <= 1e-3 and mabs2 < 1e-1
    fails += int(not ok2)
    print(
        f"[case2 asymmetric]                     max_abs={mabs2:.3e}  "
        f"mean_rel={rel2:.3e}  {'PASS' if ok2 else 'FAIL'}"
    )

    # ---- Case 3: HC=2 (template generality) ----
    xh = (mx.random.normal(shape=(1, 64, D)) * scale).astype(mx.bfloat16)
    rh = (mx.random.normal(shape=(1, 64, 2, D)) * scale).astype(mx.bfloat16)
    ph = mx.random.uniform(-1, 1, shape=(1, 64, 2)).astype(mx.float32)
    ch = mx.random.uniform(-1, 1, shape=(1, 64, 2, 2)).astype(mx.float32)
    refh = _hc_expand_op(xh, rh, ph, ch)
    goth = kernel_call(xh, rh, ph, ch)
    mx.eval(refh, goth)
    relh = _rel_err(goth, refh)
    okh = relh <= 1e-3
    fails += int(not okh)
    print(
        f"[case3 HC=2]                            max_abs={_max_abs(goth, refh):.3e}  "
        f"mean_rel={relh:.3e}  {'PASS' if okh else 'FAIL'}"
    )

    # ---- Case 4: decode shape L=1 (small-input build path) ----
    xd = (mx.random.normal(shape=(1, 1, D)) * scale).astype(mx.bfloat16)
    rd = (mx.random.normal(shape=(1, 1, HC, D)) * scale).astype(mx.bfloat16)
    pd_ = mx.random.uniform(-1, 1, shape=(1, 1, HC)).astype(mx.float32)
    cd_ = mx.random.uniform(-1, 1, shape=(1, 1, HC, HC)).astype(mx.float32)
    refd = _hc_expand_op(xd, rd, pd_, cd_)
    gotd = kernel_call(xd, rd, pd_, cd_)
    mx.eval(refd, gotd)
    reld = _rel_err(gotd, refd)
    okd = reld <= 1e-3
    fails += int(not okd)
    print(
        f"[case4 decode L=1]                      max_abs={_max_abs(gotd, refd):.3e}  "
        f"mean_rel={reld:.3e}  {'PASS' if okd else 'FAIL'}"
    )

    # ---- Microbench: prefill + decode ----
    print()
    print("--- microbench (pipelined, mx.eval batched, mx.synchronize framed) ---")
    for _ in range(4):
        y1 = _hc_expand_op(x, residual, post, comb)
        y2 = kernel_call(x, residual, post, comb)
        mx.eval(y1, y2)

    def bench(fn, xx, rr, pp, cc, N=40):
        mx.synchronize()
        t0 = time.perf_counter()
        ys = []
        for _ in range(N):
            ys.append(fn(xx, rr, pp, cc))
        mx.eval(ys)
        mx.synchronize()
        return (time.perf_counter() - t0) / N * 1e6

    us_ref = bench(_hc_expand_op, x, residual, post, comb)
    us_ker = bench(kernel_call, x, residual, post, comb)
    print(
        f"[perf L=2048]  current={us_ref:.1f} us  kernel={us_ker:.1f} us  "
        f"speedup={us_ref / us_ker:.2f}x"
    )

    for Ld in (1, 8, 128, 512, 1024):
        xd2 = (mx.random.normal(shape=(1, Ld, D)) * scale).astype(mx.bfloat16)
        rd2 = (mx.random.normal(shape=(1, Ld, HC, D)) * scale).astype(
            mx.bfloat16
        )
        pd2 = mx.random.uniform(-1, 1, shape=(1, Ld, HC)).astype(mx.float32)
        cd2 = mx.random.uniform(-1, 1, shape=(1, Ld, HC, HC)).astype(
            mx.float32
        )
        for _ in range(3):
            y = _hc_expand_op(xd2, rd2, pd2, cd2)
            mx.eval(y)
            y = kernel_call(xd2, rd2, pd2, cd2)
            mx.eval(y)
        a = bench(_hc_expand_op, xd2, rd2, pd2, cd2, N=40)
        b = bench(kernel_call, xd2, rd2, pd2, cd2, N=40)
        marker = "" if a >= b else "  ** kernel SLOWER **"
        print(
            f"[perf L={Ld:>4}]  current={a:7.1f} us  kernel={b:7.1f} us  "
            f"speedup={a / b:.2f}x{marker}"
        )

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
