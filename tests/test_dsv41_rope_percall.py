# LOCAL ADDITION. Per-call RoPE bit-exactness for DeepSeek-V4.1.
"""The per-call RoPE (``rope_freqs`` / ``cos_sin_at``) must reproduce the rows
of the old dense ``precompute_freqs_cis`` table BITWISE, for every position --
including the long-context regime (positions > 131072 and near 1048576) and the
compressor's non-contiguous latent positions ``(g0 + arange(g)) * ratio``.

Why bitwise is the correct bar: the dense table and the per-call path evaluate
``t * freq`` in fp32 and then cos/sin elementwise, so a value depends only on
its position. Any deviation would change the model's numbers at long context,
where the tables used to be silently extended by ``max(upto * 2, 4096)``.

Pure MLX (CPU), no checkpoint, no GPU; pytest-collectable.

    PYTHONPATH=. python -m pytest tests/test_dsv41_rope_percall.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from mlx_lm.models.deepseek_v41.layers import (
    cos_sin_at,
    precompute_freqs_cis,
    rope_freqs,
)

# (rope_head_dim, original_seq_len, theta, factor, beta_fast, beta_slow)
# Compress layers: YaRN on, compress_rope_theta. Window/MTP layers: YaRN off.
COMPRESS_ROPE = (64, 65536, 160000.0, 16.0, 32, 1)
WINDOW_ROPE = (64, 0, 10000.0, 16.0, 32, 1)

#: A broad position sample: the head, the old 4096 headroom edge, the 131072
#: boundary, mid/long context and both ends of the 1M window.
POSITIONS = np.array(
    [0, 1, 2, 63, 127, 128, 129, 255, 256, 1023, 4095, 4096, 4097, 8192,
     16384, 32768, 65535, 65536, 131071, 131072, 131073, 262144, 524288,
     1048575, 1048576],
    dtype=np.int64,
)


@pytest.mark.parametrize("rope", [COMPRESS_ROPE, WINDOW_ROPE],
                         ids=["compress_yarn", "window_base"])
def test_cos_sin_at_matches_dense_table_bitwise(rope) -> None:
    """Every sampled position equals the dense table row BITWISE."""
    rd, orig_len, theta, factor, bf, bs = rope
    # A dense table large enough to cover the largest sample (the old code used
    # max(upto*2, 4096) to grow; here we compare against an explicit table).
    table_cos, table_sin = precompute_freqs_cis(
        rd, int(POSITIONS.max()) + 1, orig_len, theta, factor, bf, bs)
    freqvec = rope_freqs(rd, orig_len, theta, factor, bf, bs)

    cos, sin = cos_sin_at(freqvec, mx.array(POSITIONS))
    mx.eval(cos, sin)

    # Gather the same rows via the table's own indexing (advanced int indexing
    # yields the identical rows the old code read).
    ref_cos = table_cos[mx.array(POSITIONS)]
    ref_sin = table_sin[mx.array(POSITIONS)]
    mx.eval(ref_cos, ref_sin)

    assert bool(mx.array_equal(cos, ref_cos)), "cos not bitwise-equal"
    assert bool(mx.array_equal(sin, ref_sin)), "sin not bitwise-equal"


@pytest.mark.parametrize("rope", [COMPRESS_ROPE, WINDOW_ROPE],
                         ids=["compress_yarn", "window_base"])
def test_contiguous_arange_rows_bitwise(rope) -> None:
    """A contiguous per-forward arange block matches table slicing bitwise.

    This is the query-row path in attention/indexer: ``arange(start, end)``
    (exactly the n rows of one forward) against ``table[start:end]``.
    """
    rd, orig_len, theta, factor, bf, bs = rope
    freqvec = rope_freqs(rd, orig_len, theta, factor, bf, bs)
    for start, n in [(0, 1), (0, 512), (127, 1), (1024, 128), (131072, 8),
                     (1048570, 6)]:
        table_cos, table_sin = precompute_freqs_cis(
            rd, start + n, orig_len, theta, factor, bf, bs)
        cos, sin = cos_sin_at(freqvec, mx.arange(start, start + n))
        mx.eval(cos, sin)
        assert bool(mx.array_equal(cos, table_cos[start:start + n])), (start, n)
        assert bool(mx.array_equal(sin, table_sin[start:start + n])), (start, n)


@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_compressor_latent_positions_bitwise(ratio) -> None:
    """The compressor's non-contiguous ``(g0 + arange(g)) * ratio`` positions.

    Attention ropes latents at one position per completed group. The old code
    indexed ``cos[pos]`` with that int array; the per-call path must reproduce
    exactly those rows.
    """
    rd, orig_len, theta, factor, bf, bs = COMPRESS_ROPE
    freqvec = rope_freqs(rd, orig_len, theta, factor, bf, bs)
    for g0, g in [(0, 1), (0, 17), (3, 5), (1024, 64), (262144, 9)]:
        pos = (g0 + mx.arange(g)) * ratio
        mx.eval(pos)
        table_cos, table_sin = precompute_freqs_cis(
            rd, int(pos.max()) + 1, orig_len, theta, factor, bf, bs)
        cos, sin = cos_sin_at(freqvec, pos)
        mx.eval(cos, sin)
        assert bool(mx.array_equal(cos, table_cos[pos])), (g0, g, ratio)
        assert bool(mx.array_equal(sin, table_sin[pos])), (g0, g, ratio)


def test_freqvec_is_a_tiny_vector_not_a_table() -> None:
    """The whole point: [dim//2] (32) fp32 per layer, not [seqlen, dim//2]."""
    rd = 64
    freqvec = rope_freqs(rd, 65536, 160000.0, 16.0, 32, 1)
    assert freqvec.shape == (rd // 2,)
    assert freqvec.dtype == mx.float32
    assert freqvec.nbytes == 4 * (rd // 2)   # 128 B, vs ~21 GiB at 1M as a table


def test_precompute_freqs_cis_still_works() -> None:
    """The dense builder is kept for callers/tests; it is a thin wrapper now."""
    cos, sin = precompute_freqs_cis(64, 16, 65536, 160000.0, 16.0, 32, 1)
    assert cos.shape == (16, 32) and sin.shape == (16, 32)
    freqvec = rope_freqs(64, 65536, 160000.0, 16.0, 32, 1)
    pc, ps = cos_sin_at(freqvec, mx.arange(0, 16))
    mx.eval(pc, ps)
    assert bool(mx.array_equal(cos, pc)) and bool(mx.array_equal(sin, ps))
