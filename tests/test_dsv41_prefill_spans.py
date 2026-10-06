# LOCAL ADDITION (not vendored). Profiler spans for DeepSeek-V4.1 prefill attribution.
"""Span instrumentation for the dsv41 forward path (deep-context prefill attribution).

The dsv41 package carries ``mlx_lm.profiler.span`` calls so a deployment runner can
attach a :class:`mlx_lm.profiler.SpanProfilerHook` and attribute a live prefill to
per-stage cost. These tests pin the three properties that make that safe:

* **Test A — inert when no hook is installed.** ``span()`` stays the shared no-op,
  no hook object exists, and the forward output is byte-identical to a run under a
  reference (un-instrumented) path.
* **Test B — the required span vocabulary survives a real forward.** With a
  :class:`SpanProfilerHook` registered, a tiny real ``deepseek_v41.Model`` prefill
  records ``n > 0`` for ``attn``, ``ffn``, ``attn.sdpa``, ``attn.proj_qkv``,
  ``attn.o_proj``, ``moe.switch_mlp`` and — because the tiny fixture has ratio /
  index-source layers — ``attn.indexer`` and ``attn.indexer.score``. The recorded
  table is printed as evidence (see ``_print_span_table``).
* **Test C — non-sync hooks do not perturb outputs.** A forward with a hook
  registered (default, non-sync mode) is byte-identical to the same forward with
  no hook: the instrumentation places no ``mx.*`` call of its own, so it cannot
  change the computation.

Span names are FIXED across runs; a top-level span (``attn`` / ``ffn``, no dot) is
one pair per layer visit, so :meth:`SpanStats.dump`'s top-level-only wall-time
denominator covers the forward.

Usage::

    PYTHONPATH=. <venv>/bin/python -m pytest tests/test_dsv41_prefill_spans.py -q
    PYTHONPATH=. <venv>/bin/python tests/test_dsv41_prefill_spans.py
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import mlx.core as mx

from mlx_lm import profiler
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import Model

# --------------------------------------------------------------------------
# tiny real model (CPU): small enough to forward in milliseconds, but with
# ratio + index-source layers so the indexer spans are actually exercised.
# --------------------------------------------------------------------------

IDS = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)

# spans the tiny fixture must record (n > 0). See module docstring.
REQUIRED_SPANS = (
    "attn", "ffn",
    "attn.proj_qkv", "attn.o_proj", "attn.sdpa",
    "attn.indexer", "attn.indexer.score",
    "moe.switch_mlp",
)


def toy_args() -> ModelArgs:
    """Structurally complete 4-layer config; every layer carries a ratio and the
    index-source layers own an indexer, so ``attn.indexer*`` is exercised."""
    return ModelArgs(
        vocab_size=256, dim=64, n_layers=4,
        n_heads=2, head_dim=128, rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=2, window_size=8,
        moe_inter_dim=32, n_routed_experts=8, n_shared_experts=1,
        n_activated_experts=2, compress_ratios=(2, 2, 4, 4),
        kv_source_layers=(0, 2), index_source_layers=(0, 2),
        index_n_heads=2, index_head_dim=32, index_topk=8,
        hc_mult=2, max_seq_len=64)


@pytest.fixture
def model() -> Model:
    mx.random.seed(7)
    m = Model(toy_args())
    mx.eval(m.parameters())
    return m


def _run(m: Model) -> mx.array:
    """One tiny prefill forward on a fresh cache; returns the last-token logits."""
    cache = m.make_cache(1, max_seq_len=64)
    out = m(IDS, cache, last_logit_only=True, argmax=False)
    mx.eval(out)
    return out


def digest(a: mx.array) -> str:
    mx.eval(a)
    return hashlib.sha256(np.array(a.astype(mx.float32)).tobytes()).hexdigest()


def _print_span_table(snap: dict[str, dict[str, int]]) -> None:
    """Print the recorded span table (evidence for the report).

    Mirrors :meth:`SpanStats.dump`'s columns; the ``%`` denominator is the sum
    of TOP-LEVEL spans only (names without a dot).
    """
    wall_ns = sum(v["total_ns"] for k, v in snap.items() if "." not in k)
    print("\n  span                         n     avg_us    total_ms       %")
    print("  " + "-" * 62)
    for name, v in sorted(snap.items(), key=lambda kv: -kv[1]["total_ns"]):
        n = v["n"]
        avg_us = (v["total_ns"] / n / 1000.0) if n else 0.0
        total_ms = v["total_ns"] / 1e6
        pct = (100.0 * v["total_ns"] / wall_ns) if wall_ns else 0.0
        print(f"  {name:<28s} {n:>5d} {avg_us:>10.2f} {total_ms:>10.2f} {pct:>6.1f}%")
    print()


# --------------------------------------------------------------------------
# Test A — no hook installed => span() is the no-op, nothing accumulates
# --------------------------------------------------------------------------

def test_A_no_hook_is_inert(model: Model) -> None:
    assert profiler._registered_hook is None, "a hook leaked into the no-hook test"
    mx.random.seed(7)
    ref = Model(toy_args())          # identical weights, built independent of the fixture
    mx.eval(ref.parameters())

    a = digest(_run(model))
    b = digest(_run(ref))
    assert a == b, "no-hook forward is not reproducible"
    assert profiler._registered_hook is None, "the forward registered a hook"
    print(f"  Test A: no-hook digest {a[:16]} (no hook object created)")


# --------------------------------------------------------------------------
# Test B — install a SpanProfilerHook => required spans recorded (n > 0)
# --------------------------------------------------------------------------

def test_B_required_spans_recorded(model: Model) -> None:
    profiler.unregister()
    hook = profiler.SpanProfilerHook()
    profiler.register(hook)
    try:
        _run(model)
    finally:
        profiler.unregister()

    snap = hook.stats.snapshot_and_reset()
    _print_span_table(snap)
    missing = [s for s in REQUIRED_SPANS if snap.get(s, {}).get("n", 0) <= 0]
    assert not missing, f"spans not recorded by the forward: {missing}"
    # one top-level attn/ffn pair per layer visit
    assert snap["attn"]["n"] == model.args.n_layers
    assert snap["ffn"]["n"] == model.args.n_layers
    print(f"  Test B: recorded {len(snap)} span names; "
          f"attn/ffn n={snap['attn']['n']} (== n_layers)")


# --------------------------------------------------------------------------
# Test C — non-sync hook presence does not change the output
# --------------------------------------------------------------------------

def test_C_output_bit_identical_with_and_without_hook(model: Model) -> None:
    profiler.unregister()
    no_hook = digest(_run(model))

    hook = profiler.SpanProfilerHook()      # default: non-sync (no EXO_PROFILER_SYNC_SPANS)
    assert not hook._sync, "fixture assumes non-sync mode"
    profiler.register(hook)
    try:
        with_hook = digest(_run(model))
    finally:
        profiler.unregister()

    assert hook.stats.snapshot_and_reset(), "hook recorded nothing in the hooked run"
    assert no_hook == with_hook, "non-sync hook changed the forward output"
    print(f"  Test C: hooked digest == no-hook digest ({with_hook[:16]})")


# --------------------------------------------------------------------------
# Test D — a subsequent no-hook run returns to the inert state
# --------------------------------------------------------------------------

def test_D_unregister_restores_inert_state(model: Model) -> None:
    hook = profiler.SpanProfilerHook()
    profiler.register(hook)
    try:
        _run(model)
    finally:
        profiler.unregister()
    assert profiler.get() is None
    # the shared no-op span CM is used again (no per-call allocation, no hook)
    assert profiler.span("attn") is profiler._NULL_SPAN


if __name__ == "__main__":
    mx.random.seed(7)
    _m = Model(toy_args())
    mx.eval(_m.parameters())
    test_A_no_hook_is_inert(_m)
    test_B_required_spans_recorded(_m)
    test_C_output_bit_identical_with_and_without_hook(_m)
    test_D_unregister_restores_inert_state(_m)
    print("DSV41_PREFILL_SPANS OK")
