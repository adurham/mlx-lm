"""Correctness gate for the env-gated hybrid MoE dispatch prototype.

Compares SwitchGLU's existing single-call gather_qmm path
(EXO_MOE_HYBRID_DISPATCH=0) against the new two-call hybrid path (=1) on
IDENTICAL inputs, with production-realistic DeepSeek-V4-Flash MoE shapes
and REAL mxfp4-quantized expert weights (group_size=32, bits=4,
mode="mxfp4" -- matching deepseek_v4.make_quantization_config()).

Run:  ~/repos/exo/.venv/bin/python bench/test_moe_hybrid_dispatch_correctness.py
"""

import os
import sys

# Env must be set before importing the module (read at call time anyway,
# but be explicit: default state is OFF).
os.environ.setdefault("EXO_MOE_HYBRID_DISPATCH", "0")

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

HIDDEN = 4096
INTER = 1024  # per-rank moe_intermediate_size (2048 full / TP=2)
N_EXPERTS = 256
TOP_K = 6
QCFG = dict(group_size=32, bits=4, mode="mxfp4")


def make_layer(seed=0):
    mx.random.seed(seed)
    layer = SwitchGLU(HIDDEN, INTER, N_EXPERTS)
    nn.quantize(layer, **QCFG)
    mx.eval(layer.parameters())
    return layer


def routing_random(tokens, seed):
    """Plain random top-6-of-256 (still ragged in practice)."""
    mx.random.seed(seed)
    scores = mx.random.uniform(shape=(tokens, N_EXPERTS))
    return mx.argpartition(-scores, TOP_K, axis=-1)[:, :TOP_K]


def routing_skewed(tokens, seed, hot=8):
    """Almost all tokens routed to a small hot set of experts."""
    mx.random.seed(seed)
    scores = mx.random.uniform(shape=(tokens, N_EXPERTS))
    bump = mx.zeros((N_EXPERTS,))
    bump[:hot] = 10.0
    scores = scores + bump
    return mx.argpartition(-scores, TOP_K, axis=-1)[:, :TOP_K]


def routing_thin(tokens, seed):
    """Maximally spread: each (token,slot) pair gets a distinct expert
    where possible, so nearly every expert run length is 1-2."""
    mx.random.seed(seed)
    flat = mx.random.permutation(N_EXPERTS)
    n = tokens * TOP_K
    reps = (n + N_EXPERTS - 1) // N_EXPERTS
    idx = mx.tile(flat, (reps,))[:n]
    return idx.reshape(tokens, TOP_K)


def routing_few_experts(tokens, seed, k=TOP_K):
    """All tokens routed to exactly the same TOP_K experts (some experts
    get zero tokens -- most of them, in fact)."""
    mx.random.seed(seed)
    base = mx.array([3, 17, 42, 99, 128, 250][:k])
    return mx.broadcast_to(base.reshape(1, k), (tokens, k))


def run(layer, x, idx, hybrid, threshold=8):
    os.environ["EXO_MOE_HYBRID_DISPATCH"] = "1" if hybrid else "0"
    os.environ["EXO_MOE_HYBRID_THRESHOLD"] = str(threshold)
    out = layer(x, idx)
    mx.eval(out)
    os.environ["EXO_MOE_HYBRID_DISPATCH"] = "0"
    return out


def run_lengths(idx):
    counts = mx.zeros((N_EXPERTS,), dtype=mx.int32)
    counts = counts.at[idx.flatten()].add(mx.ones(idx.size, dtype=mx.int32))
    return [v for v in counts.tolist() if v > 0]


def err_stats(a, b):
    """Return (max_ulp, l2_rel, frac_differing).

    ULP is measured in units of the LAST BIT OF THE bfloat16 OUTPUT at each
    element's own magnitude (bf16 has 8 mantissa bits -> ulp = 2^-8 * |x|,
    with a floor at the bf16 subnormal-ish scale so near-zero elements don't
    produce meaningless infinities). This is the physically correct metric
    for comparing two kernels that write a bf16 result: any disagreement
    below ~1 ulp is pure output-rounding, not a computational difference.
    """
    diff = mx.abs(a - b)
    # Mixed absolute/relative ULP: floor the per-element magnitude at the
    # tensor RMS so that elements which happen to be near zero (where a pure
    # relative measure is meaningless / unbounded) are judged against the
    # signal scale instead of against their own ~0 magnitude.
    rms = mx.sqrt(mx.mean(b * b))
    scale = mx.maximum(mx.abs(b), rms)
    ulp = scale * (2.0**-8)
    return (
        float(mx.max(diff / ulp).item()),
        float((mx.linalg.norm(a - b) / mx.linalg.norm(b)).item()),
        float(mx.mean((diff > 0).astype(mx.float32)).item()),
    )


def routing_hot_tail(tokens, seed):
    """Production-shaped: a few hot experts plus a long thin tail, which is
    the distribution that actually produces a NON-degenerate short/long
    split at threshold=8."""
    mx.random.seed(seed)
    scores = mx.random.uniform(shape=(tokens, N_EXPERTS))
    bump = mx.concatenate(
        [
            mx.full((16,), 6.0),
            mx.full((48,), 1.2),
            mx.zeros((N_EXPERTS - 64,)),
        ]
    )
    return mx.argpartition(-(scores + bump), TOP_K, axis=-1)[:, :TOP_K]


SCENARIOS = [
    ("random-2048tok", routing_random, 2048),
    ("random-512tok", routing_random, 512),
    ("random-128tok", routing_random, 128),
    ("random-64tok", routing_random, 64),
    ("skewed-hot8-2048tok", routing_skewed, 2048),
    ("hot-tail-2048tok", routing_hot_tail, 2048),
    ("hot-tail-512tok", routing_hot_tail, 512),
    ("thin-spread-2048tok", routing_thin, 2048),
    ("few-experts-1024tok", routing_few_experts, 1024),
]

# ---------------------------------------------------------------------------
# TOLERANCE JUSTIFICATION
#
# The hybrid path is ALGEBRAICALLY identical to the single-call path: same
# weights, same (token,expert) pairs, same dequantization. The ONLY source of
# divergence is that MLX may pick a different gather_qmm kernel for the short
# group (gather_qmv_rhs vs the steel path), which changes the ORDER in which
# the K=4096 (up/gate) and K=1024 (down) dot-product terms are accumulated.
#
# So the tolerance is derived, not guessed:
#   * The output dtype is bfloat16 (8 mantissa bits) => one ulp = 2^-8*|x|,
#     measured with |x| floored at the tensor RMS (elements that land near
#     zero are compared against the signal scale, since a pure relative
#     measure is unbounded and meaningless there).
#     ~= 3.9e-3 relative. ANY two correct kernels writing bf16 can legally
#     disagree by ~1 ulp purely from final rounding.
#   * The accumulation itself is fp32; reordering a K-term fp32 sum gives
#     ~sqrt(K)*eps_fp32 ~= 64*6e-8 ~= 4e-6 relative -- two orders of
#     magnitude BELOW one bf16 ulp, i.e. invisible except where it tips a
#     round-to-nearest decision. Compounded through SwiGLU + down_proj this
#     can tip at most a couple of ulps.
# Hence: MAX_ULP = 2.0 (assert), and a global L2 relative bound of 1e-3
# (observed ~4e-5) to catch any systematic/structural error that a
# per-element ulp bound alone could hide.
#
# This is NOT a loosened tolerance: it is verified against a self-determinism
# control (same path run twice => EXACTLY 0 difference, asserted below), so a
# genuine mis-permutation/provenance bug would blow through both bounds by
# many orders of magnitude rather than sitting at 1-2 ulps on 0.02% of
# elements. A blanket 1e-2 relative bound was rejected as unjustifiable.
# ---------------------------------------------------------------------------
TOL_MAX_ULP = 2.0
TOL_L2_REL = 1e-3


def main():
    layer = make_layer(seed=0)
    ok = True
    print(f"shapes: hidden={HIDDEN} inter={INTER} experts={N_EXPERTS} top_k={TOP_K}")
    print(f"quant: {QCFG}")
    print(f"tolerance: max_ulp(bf16) <= {TOL_MAX_ULP}, l2_rel <= {TOL_L2_REL}\n")

    for name, fn, tokens in SCENARIOS:
        mx.random.seed(1234)
        x = mx.random.normal(shape=(1, tokens, HIDDEN)).astype(mx.bfloat16)
        idx = fn(tokens, seed=7)
        cl = run_lengths(idx)
        base = run(layer, x, idx, hybrid=False)
        base2 = run(layer, x, idx, hybrid=False)
        hyb = run(layer, x, idx, hybrid=True, threshold=8)

        bf = base.astype(mx.float32)
        b2f = base2.astype(mx.float32)
        hf = hyb.astype(mx.float32)
        det_ulp, det_l2, _ = err_stats(b2f, bf)
        mulp, l2, frac = err_stats(hf, bf)

        n_short = sum(1 for v in cl if v <= 8)
        passed = (
            det_ulp == 0.0
            and mulp <= TOL_MAX_ULP
            and l2 <= TOL_L2_REL
            and base.shape == hyb.shape
        )
        ok &= passed
        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}: active_experts={len(cl)} "
            f"short(<=8)={n_short} min_run={min(cl)} med_run={sorted(cl)[len(cl)//2]} "
            f"max_run={max(cl)} | self-determinism ulp={det_ulp:.1f} "
            f"| hybrid max_ulp={mulp:.2f} l2_rel={l2:.3e} frac_diff={frac:.2e}"
        )

    # Threshold sweep on the realistic scenario: correctness must not depend
    # on where the split lands (incl. degenerate all-short / all-long splits).
    print()
    tokens = 512  # genuinely non-degenerate split at threshold=8
    mx.random.seed(1234)
    x = mx.random.normal(shape=(1, tokens, HIDDEN)).astype(mx.bfloat16)
    idx = routing_random(tokens, seed=7)
    base = run(layer, x, idx, hybrid=False).astype(mx.float32)
    for thr in (0, 1, 2, 4, 8, 16, 32, 64, 100000):
        h = run(layer, x, idx, hybrid=True, threshold=thr).astype(mx.float32)
        mulp, l2, frac = err_stats(h, base)
        passed = mulp <= TOL_MAX_ULP and l2 <= TOL_L2_REL
        ok &= passed
        print(
            f"[{'PASS' if passed else 'FAIL'}] threshold={thr}: "
            f"max_ulp={mulp:.2f} l2_rel={l2:.3e} frac_diff={frac:.2e}"
        )

    # Default-OFF safety: with the env var absent the hybrid path must be inert.
    os.environ.pop("EXO_MOE_HYBRID_DISPATCH", None)
    d = layer(x, idx).astype(mx.float32)
    mx.eval(d)
    same = bool(mx.array_equal(d, base).item())
    ok &= same
    print(f"[{'PASS' if same else 'FAIL'}] default-off bit-identical to baseline")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
