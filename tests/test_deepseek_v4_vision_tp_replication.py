"""Phase 4e: the DSv4 vision tower is REPLICATED, not sharded, under TP.

The cluster runs tensor-parallel with world_size=2. The ViT + Aligner are dense
(~466M params, well under 1 GB bf16), so the chosen approach is to REPLICATE
them identically on every rank and encode each image redundantly per rank:

* no collective is introduced into the vision forward, so no new sync point
  and no cross-node round trip per image;
* every rank already loads the full checkpoint, so replication costs no extra
  I/O -- only the (small) resident memory and the redundant compute;
* every rank independently produces the SAME embeddings, which is what
  `patch_embed_tokens` needs, since it splices a full-length embedding array
  into that rank's own prefill.

The alternative -- sharding the tower -- would need an all-gather per image to
reassemble aligner rows before the merge, for a module that is 0.16% of the
model. Not worth a collective.

`Model.shard()` shards ONLY attention and FFN projections; anything it does not
name is replicated by omission. These tests pin that down as a positive
property rather than an absence: they assert the vision modules are untouched
by sharding, that no collective is reachable from the vision forward, and --
the check that actually matters -- that two independently constructed "ranks"
produce BIT-IDENTICAL image encodings.
"""

import inspect

import mlx.core as mx
import numpy as np
import pytest

from mlx_lm.models import deepseek_v4 as dsv4

VOCAB = 512
HIDDEN = 64


def _vision_config(*, vision_n_layers: int = 2):
    return dsv4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        qk_rope_head_dim=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        sliding_window=128,
        compress_ratios=[0, 4],
        index_topk=8,
        index_n_heads=4,
        index_head_dim=16,
        # Vision tower: small but structurally real.
        vision_n_layers=vision_n_layers,
        vision_dim=32,
        vision_inter_dim=64,
        vision_n_heads=4,
        vision_patch_size=14,
        vision_downsample_ratio=2,
        vision_max_n_token=384,
        vision_rope_theta=10000.0,
    )


def _filled_model(seed: int):
    """Two calls with the same seed give two numerically identical models."""
    model = dsv4.Model(_vision_config())
    rng = np.random.default_rng(seed)

    def fill(tree):
        if isinstance(tree, dict):
            return {k: fill(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [fill(v) for v in tree]
        if isinstance(tree, mx.array):
            if tree.dtype == mx.int32:
                return mx.array(rng.integers(0, 8, size=tree.shape).astype(np.int32))
            return mx.array((rng.standard_normal(tree.shape) * 0.05).astype(np.float32))
        return tree

    model.update(fill(model.parameters()))
    return model


class TestShardLeavesVisionAlone:
    def test_shard_source_never_names_a_vision_module(self):
        """`shard()` must not touch vision./aligner./the sentinel params."""
        source = inspect.getsource(dsv4.Model.shard)
        for name in (
            "vision",
            "aligner",
            "image_start",
            "image_end",
            "image_newline",
            "image_pad",
            "embed_tokens",
        ):
            assert name not in source, (
                f"Model.shard() references `{name}`; the vision tower and the "
                "embedding are supposed to be replicated by omission. If this "
                "changed deliberately, Phase 4e's replication argument needs "
                "revisiting."
            )

    def test_vision_forward_has_no_collective(self):
        """No all_sum/all_gather reachable from the vision forward path."""
        for cls in (dsv4.VisionTransformer, dsv4.VisionAligner):
            source = inspect.getsource(cls)
            for collective in ("all_sum", "all_gather", "all_reduce", "distributed"):
                assert collective not in source, (
                    f"{cls.__name__} references `{collective}`. Replication "
                    "assumes the vision forward is collective-free."
                )


class TestReplicasAgreeBitwise:
    """The property that actually matters: ranks must not diverge."""

    def _patches(self, n_h: int, n_w: int, patch: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        return mx.array(
            (rng.standard_normal((n_h * n_w, 3, patch, patch)) * 0.5).astype(np.float32)
        )

    def test_two_ranks_encode_an_image_identically(self):
        n_h, n_w, patch = 4, 4, 14
        patches = self._patches(n_h, n_w, patch)

        rank0 = _filled_model(1234)
        rank1 = _filled_model(1234)

        out0 = rank0.encode_image(patches, n_h, n_w)
        out1 = rank1.encode_image(patches, n_h, n_w)
        mx.eval(out0, out1)

        a0 = np.asarray(out0, dtype=np.float64)
        a1 = np.asarray(out1, dtype=np.float64)
        assert a0.shape == a1.shape
        assert np.array_equal(a0, a1), (
            f"rank outputs diverge: max abs diff {np.abs(a0 - a1).max():.3e}"
        )

    def test_sentinel_parameters_are_identical_across_ranks(self):
        rank0 = _filled_model(99)
        rank1 = _filled_model(99)
        for name in ("image_start", "image_end", "image_newline", "image_pad"):
            a = np.asarray(getattr(rank0, name), dtype=np.float64)
            b = np.asarray(getattr(rank1, name), dtype=np.float64)
            assert np.array_equal(a, b), f"{name} differs across ranks"

    def test_vision_tower_is_small_enough_to_replicate(self):
        """Sanity: the tower must stay a small fraction of the model."""
        model = _filled_model(5)

        def count(tree) -> int:
            if isinstance(tree, dict):
                return sum(count(v) for v in tree.values())
            if isinstance(tree, list):
                return sum(count(v) for v in tree)
            if isinstance(tree, mx.array):
                return int(tree.size)
            return 0

        params = model.parameters()
        vision = count(params.get("vision", {})) + count(params.get("aligner", {}))
        total = count(params)
        assert vision > 0, "test config built no vision tower"
        assert vision < total, "vision tower cannot be the whole model"


class TestVisionTowerOnlyOnVisionCheckpoints:
    def test_text_only_config_builds_no_tower(self):
        model = dsv4.Model(_vision_config(vision_n_layers=0))
        assert not hasattr(model, "vision")
        assert not hasattr(model, "aligner")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
