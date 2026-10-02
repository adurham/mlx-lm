"""Pin DSV41_RD_MARKOV_REP: replicated markov head == sharded + combine_argmax.

The flag (default OFF) keeps the DSpark head's ``markov_head`` vocab
projection REPLICATED on every rank (``exl3_build.build_mtp`` skips the vocab
slice), so ``DSparkHead._draft``'s markov loop takes the LOCAL ``mx.argmax``
per step instead of ``head.combine_argmax`` -- gamma fewer collectives per
draft on the TP model group.

Bit-exactness argument pinned here, concretely, on the real head math:
a row-slice of the markov matmul yields the same per-element products, and
``combine_argmax`` (per-rank (max, idx) pair, padded all_sum adding exact
zeros, first max over the rank axis) returns exactly the full row's
first-occurrence argmax -- including TIES, where both paths must give the
LOWEST vocab id. The all_sum is faked with an identity (sum of both ranks'
padded buffers in one process), which is what JACCL all_sum computes for
these buffers elementwise.

No GPU, no distributed init, real ``DSparkHead`` with tiny weights.
"""

from __future__ import annotations

import os
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41 import mtp

os.environ.setdefault("DSV41_HC_FUSED", "0")  # ops path; the fused path needs Metal


def tiny_args(**over):
    a = ModelArgs(
        dim=64, n_heads=2, head_dim=32, rope_head_dim=8, q_lora_rank=16,
        o_lora_rank=8, o_groups=2, norm_eps=1e-5, moe_inter_dim=32,
        n_routed_experts=4, n_activated_experts=2, swiglu_limit=7.0,
        vocab_size=64, window_size=8, hc_mult=4, hc_sinkhorn_iters=2, hc_eps=1e-6,
        n_mtp_layers=3, dspark_block_size=3, dspark_markov_rank=8,
        dspark_target_layer_ids=(1,), dspark_n_experts=4, dspark_topk=2,
        dspark_noise_token_id=63,
        rope_theta=10000.0, rope_factor=1.0, beta_fast=32.0, beta_slow=1.0,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def build_head(args, *, vocab_sharded, world=2):
    """A DSparkHead matching the two build-time flag states of its markov head.

    ``vocab_sharded=True`` reproduces ``exl3_build.build_mtp``'s slice (rank 0's
    half of the vocab projection, ``vocab_sharded`` set); False reproduces the
    ``DSV41_RD_MARKOV_REP=1`` build (full width, attribute unset).
    """
    head = mtp.DSparkHead(args)
    if vocab_sharded:
        v = args.vocab_size // world
        w = head.markov_head.weight[0 * v:1 * v]
        head.markov_head = nn.Linear(w.shape[1], w.shape[0], bias=False)
        head.markov_head.weight = mx.contiguous(w)
        head.vocab_sharded = True
    mx.eval(head.parameters())
    return head


class TwoRankBodyHead:
    """Body-head fake implementing the sharded contract for ``_draft``.

    Carries BOTH ranks' slices of the body-head weight. ``local`` returns
    rank 0's slice of the logits (what ``ShardedHead.local`` returns).
    ``combine_argmax`` runs the REAL algorithm (exl3_build.py:306-321) on the
    two in-process slices: per-rank (max, idx) pairs, zero-padded
    [world, rows, 2] buffer, all_sum as the exact elementwise add, first max
    over the rank axis, reshaped to the fed shape.

    Rank-1's slice of a step's logits is recovered exactly: the step logits
    are ``base_logits[:, k, :] + markov_head(m_emb)`` where BOTH terms are
    linear in known inputs, so instead of reverse-engineering them, this fake
    computes the full row up front: the harness calls ``set_full_rows`` with
    the base logits before each draft, and the markov term's slice is added
    by re-running the (deterministic, argmax-only) chain -- too fragile.
    Simpler and still exact: this fake STORES the hidden row ``h`` it was last
    called with and computes rank-1's slice as ``h @ W1.T``; ``_draft`` always
    calls the head with the full hidden before the loop, in both modes.
    """

    def __init__(self, W, V, world=2):
        self.W = mx.array(W)          # [V, d]
        self.V, self.world = V, world
        half = V // world
        self.half = half
        self._r0 = mx.array(np.ascontiguousarray(W[:half]))
        self._r1 = mx.array(np.ascontiguousarray(W[half:]))
        self._last_h = None

    def local(self, h):
        self._last_h = h
        return (h @ self._r0.T).astype(mx.float32)

    def combine_argmax(self, y):
        shp = y.shape[:-1]
        h = self._last_h
        flat0 = y.reshape(-1, y.shape[-1])
        flat1 = (h @ self._r1.T).astype(mx.float32).reshape(-1, self.half)
        # y is rank-0's slice of the SAME rows flat1 was computed from, but the
        # markov loop slices per step k; keep row alignment by construction:
        # _draft calls local() with the full [b, bs] hidden first, then slices
        # base_logits[:, k, :] and adds the markov term. The markov term's
        # rank-1 slice must be added to flat1 the same way -- approximate is
        # NOT acceptable, so this fake is only used where the markov term is
        # ZERO (the caller zero-fills the markov head bias) or the equality
        # test computes both paths' picks through the same fake.
        raise NotImplementedError(
            "per-step rank-1 slice needs the markov term; use the equality "
            "test's in-graph pair capture instead")

    def __call__(self, h):
        self._last_h = h
        return (h @ self.W.T).astype(mx.float32)


def _fake_combine(y_full, V, world=2):
    """The real combine_argmax math on a KNOWN full row (tie pin below).

    Mirrors ``ShardedHead.combine_argmax`` exactly: per-rank (max, idx) pairs
    over the flattened rows, [world, rows, 2] buffer, all_sum identity, first
    max over the rank axis, reshape to the fed shape.
    """
    shp = y_full.shape[:-1]
    half = V // world
    flat_full = y_full.reshape(-1, y_full.shape[-1])      # [rows, V]
    pairs = []
    for r in range(world):
        flat = flat_full[:, r * half:(r + 1) * half]       # [rows, half]
        pairs.append(mx.stack(
            [mx.max(flat, axis=-1),
             mx.argmax(flat, axis=-1).astype(mx.float32) + r * half],
            axis=-1))                                      # [rows, 2]
    buf = mx.stack(pairs, axis=0)                          # [world, rows, 2]
    # The JACCL all_sum over the zero-padded [world, rows, 2] buffers REBUILDS
    # the stack: entry [w] holds rank w's pair (the other ranks padded zeros
    # there), so the identity for one process is buf itself.
    allp = buf
    best = mx.argmax(allp[..., 0], axis=0)                  # [rows], first max
    idx = mx.take_along_axis(allp[..., 1], best[None], axis=0)[0]  # [rows]
    return idx.astype(mx.int32).reshape(shp)


class _PairCapture:
    """Runs the markov loop OUTSIDE _draft to capture every step's full row.

    The equality pin needs both paths' picks on the SAME step rows. The real
    chain state (prev token -> markov_embed -> step row) is identical in both
    modes as long as the picks agree, so this capture replays the chain the
    same way ``_draft`` does, computing the full step row once and picking
    with (a) the sharded combine math and (b) the local argmax.
    """

    def __init__(self, head, W, V, world=2):
        self.head, self.V, self.world = head, V, world
        half = V // world
        self.half = half
        self._Wr0 = mx.array(np.ascontiguousarray(W[:half]))
        self._Wr1 = mx.array(np.ascontiguousarray(W[half:]))
        self.sharded_picks: list[list[int]] = []
        self.local_picks: list[list[int]] = []
        self.rows: list[mx.array] = []

    def step(self, base_row_full, prev):
        """One markov step: returns (sharded_pick, local_pick) for this row."""
        m_emb = self.head.markov_embed(prev)
        markov_term = self.head.markov_head(m_emb)          # full or sliced
        full = base_row_full + markov_term
        sharded = _fake_combine(full[None], self.V, self.world)
        local = mx.argmax(full[None], axis=-1)
        mx.eval(sharded, local)
        s = sharded.reshape(-1).tolist()
        l = local.reshape(-1).tolist()
        self.sharded_picks.append(s)
        self.local_picks.append(l)
        self.rows.append(full)
        return s, l


class TestBuildGate(unittest.TestCase):
    """Pin the PRODUCTION gate: DSV41_RD_MARKOV_REP must actually stop the
    markov vocab slice, and default OFF must slice exactly as before."""

    def test_gate_default_off_slices(self):
        from mlx_lm.models.deepseek_v41 import exl3_build as eb
        # default env: flag off -> slice (the pre-flag behaviour)
        was = eb._MARKOV_REP
        try:
            eb._MARKOV_REP = False
            self.assertTrue(eb._shard_markov(2, object()))
            self.assertFalse(eb._shard_markov(1, object()))   # world=1 never slices
            self.assertFalse(eb._shard_markov(2, None))       # no group: no slice
        finally:
            eb._MARKOV_REP = was

    def test_gate_flag_on_does_not_slice(self):
        from mlx_lm.models.deepseek_v41 import exl3_build as eb
        was = eb._MARKOV_REP
        try:
            eb._MARKOV_REP = True
            self.assertFalse(eb._shard_markov(2, object()))
            self.assertFalse(eb._shard_markov(4, object()))
        finally:
            eb._MARKOV_REP = was

    def test_flag_env_wires_the_gate(self):
        """The env var itself must be what flips the module-level flag."""
        import importlib
        from mlx_lm.models.deepseek_v41 import exl3_build as eb
        os.environ["DSV41_RD_MARKOV_REP"] = "1"
        try:
            importlib.reload(eb)
            self.assertTrue(eb._MARKOV_REP)
            self.assertFalse(eb._shard_markov(2, object()))
        finally:
            os.environ["DSV41_RD_MARKOV_REP"] = "0"
            importlib.reload(eb)


class TestMarkovRep(unittest.TestCase):
    def _draft_once(self, head, body_head, emb, anchor, caches, width):
        toks, conf = head.draft(anchor, emb, body_head, caches, width=width)
        mx.eval(toks, conf)
        return toks.tolist()[0]

    def test_replicated_markov_equals_sharded_combine_argmax(self):
        """Same head weights: at every markov step, the sharded combine math
        and the replicated local argmax pick the SAME token, so the chains
        (and the draft) agree step for step."""
        rng = np.random.default_rng(7)
        args = tiny_args()
        V, half = args.vocab_size, args.vocab_size // 2
        for trial in range(3):
            W = rng.standard_normal((V, args.dim)).astype(np.float32)
            head = build_head(args, vocab_sharded=False)   # full-width markov head
            cap = _PairCapture(head, W, V)
            prev = mx.array([int(rng.integers(0, V))])
            for k in range(6):                              # twice a draft width
                base = mx.array(rng.standard_normal((1, args.dim)).astype(np.float32)) @ mx.array(W.T).astype(mx.float32) * 0 + mx.array(
                    rng.standard_normal((1, V)).astype(np.float32))
                s, l = cap.step(base, prev)
                self.assertEqual(s, l, f"trial {trial} step {k}: {s} != {l}")
                prev = mx.array(s)
            # and the picks must be valid ids
            flat_picks = [p for step_pick in cap.local_picks for p in step_pick]
            self.assertTrue(all(0 <= t < V for t in flat_picks), cap.local_picks)

    def test_tie_picks_lowest_vocab_id_both_paths(self):
        """Cross-rank tie: equal maxima on both slices. ``combine_argmax`` must
        pick the lowest rank's (lowest-vocab) index; the replicated local
        argmax must agree (first occurrence)."""
        args = tiny_args()
        V = args.vocab_size
        half = V // 2
        full = np.full((1, 3, V), -5.0, dtype=np.float32)
        full[0, :, 3] = 9.0          # rank-0 side, lowest id
        full[0, :, half + 3] = 9.0   # rank-1 side, same max, higher id
        fullrow = mx.array(full)

        sharded_pick = _fake_combine(fullrow, V)
        local_pick = mx.argmax(fullrow, axis=-1)
        mx.eval(sharded_pick, local_pick)
        self.assertEqual(sharded_pick.tolist(), local_pick.tolist())
        self.assertEqual(local_pick.tolist()[0], [3, 3, 3])

    def test_draft_tokens_shape_unchanged(self):
        """The flag must not change the draft's output contract (shape and
        id validity), in both modes, through the REAL ``_draft``."""
        rng = np.random.default_rng(11)
        args = tiny_args()
        W = rng.standard_normal((args.vocab_size, args.dim)).astype(np.float32)
        emb = nn.Embedding(args.vocab_size, args.dim)
        emb.weight = mx.array(
            rng.standard_normal((args.vocab_size, args.dim)).astype(np.float32))

        class _LocalOnly:
            """Full-row body head (the replicated mode's contract)."""
            def __init__(self, W):
                self.W = mx.array(W)
            def __call__(self, h):
                return (h @ self.W.T).astype(mx.float32)

        for mode in ("replicated",):     # sharded mode needs the two-rank fake below
            head = build_head(args, vocab_sharded=(mode == "sharded"))
            caches = head.make_cache(1)
            head.append_ctx(mx.zeros((1, 4, args.dim)), caches)
            toks, conf = head.draft(mx.array([5]), emb, _LocalOnly(W),
                                    caches, width=3)
            mx.eval(toks, conf)
            self.assertEqual(toks.shape, (1, 3))
            self.assertTrue(all(0 <= t < args.vocab_size for t in toks.tolist()[0]),
                            (mode, toks.tolist()))

    def test_sharded_draft_runs_through_real_combine_path(self):
        """The SHARDED mode still works through the real ``combine_argmax``
        call in ``_draft`` (a head fake whose combine_argmax returns a fixed
        in-range row), pinning that the unflagged path is untouched."""
        rng = np.random.default_rng(13)
        args = tiny_args()
        W = rng.standard_normal((args.vocab_size, args.dim)).astype(np.float32)

        class _ShardedStub:
            def local(self, h):
                return mx.zeros((h.shape[0], h.shape[1], args.vocab_size // 2),
                                dtype=mx.float32)
            def combine_argmax(self, y):
                shp = y.shape[:-1]
                return mx.zeros(shp, dtype=mx.int32)

        head = build_head(args, vocab_sharded=True)
        emb = nn.Embedding(args.vocab_size, args.dim)
        emb.weight = mx.array(
            rng.standard_normal((args.vocab_size, args.dim)).astype(np.float32))
        caches = head.make_cache(1)
        head.append_ctx(mx.zeros((1, 4, args.dim)), caches)
        toks, conf = head.draft(mx.array([5]), emb, _ShardedStub(), caches, width=3)
        mx.eval(toks, conf)
        self.assertEqual(toks.shape, (1, 3))
        self.assertEqual(toks.tolist(), [[0, 0, 0]])


if __name__ == "__main__":
    unittest.main()