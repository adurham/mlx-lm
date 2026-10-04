# LOCAL ADDITION. Bit-exactness + replay tests for the DSv4.1 cache changes.
"""bf16 storage exactness and grow-on-demand capacity for the DSv4.1 cache.

No checkpoint, no GPU: the REAL ``LayerCache`` / ``CompressorState`` /
``ModelCache`` classes over a tiny layer subset. Four properties are pinned:

* **grid exactness** -- the real fake-quant functions land on grids whose
  products are exactly representable in bf16, so ``Q(x).astype(bf16).astype(
  fp32) == Q(x)`` for every buffer class (test 2);
* **bf16 storage is lossless** -- values written through the real writer paths
  read back widened bitwise equal to the fp32 values (test 3);
* **grow == never-grow** -- a cache that grows through >= 2 capacity
  boundaries, with rollback / snapshot-restore / cancel interleaved, is bitwise
  identical to one preallocated to the cap, and a draft-boundary write landing
  exactly on a capacity edge is safe (test 4);
* **rollback leaves no stale tail** -- after a rollback near a growth boundary
  the newly allocated region beyond the live rows is zeros, and sabotaging
  growth to copy the full buffer makes that test fail (test 5);
* **capacity refusal** -- a request past ``max_seq_len`` raises ``CapacityError``
  (test 6).

    PYTHONPATH=. python -m pytest tests/test_dsv41_cache_growth.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from mlx_lm.models.deepseek_v41.cache import (
    CapacityError,
    LayerCache,
    ModelCache,
)
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41 import session_cache as SC
from mlx_lm.models.deepseek_v41.fakequant import (
    fake_quant_fp4_e4m3,
    fake_quant_fp4_ue8m0,
    fake_quant_fp8_ue8m0,
)

WINDOW, RATIO, NLAYERS = 8, 2, 4
MAX_SEQ = 512


def stub_args(engram: bool = False) -> ModelArgs:
    return ModelArgs(window_size=WINDOW, compress_ratios=(RATIO,) * NLAYERS,
                     kv_source_layers=(0,), index_source_layers=(0, 1),
                     engram_layer_ids=(1,) if engram else ())


def make_cache(max_seq_len: int = MAX_SEQ, engram: bool = False,
               initial_capacity: int | None = None) -> ModelCache:
    return ModelCache(stub_args(engram), 1, max_seq_len,
                      initial_capacity=initial_capacity)


# --------------------------------------------------------------------------
# deterministic stub prefill (position-derived; the real driver is too)
# --------------------------------------------------------------------------

def stub_prefill(model, ids, cache, *, argmax=False, return_taps=False, **kw):
    n = len(ids)
    pos = int(cache.offset)
    # The real driver's invariant: grow (eval-clean) before any write.
    cache.ensure_capacity(pos + n)
    g0, g1 = pos // RATIO, (pos + n) // RATIO
    for lc in cache.layers:
        if lc.comp_kv is not None:
            lc.comp_kv[0, g0:g1] = (1.0 + np.arange(g0, g1))[:, None]
            if lc.index_k is not None:
                lc.index_k[0, g0:g1] = (2.0 + np.arange(g0, g1))[:, None]
            cs = lc.comp_state
            for i in range(n):
                slot = (pos + i) % RATIO
                cs.kv_state[0, slot] = 3.0 + pos + i
                cs.score_state[0, slot] = 4.0 + pos + i
        for i in range(pos, pos + n):
            lc.win_kv[0, i % WINDOW] = 5.0 + i
    if cache.engram_ids is not None:
        cache.engram_ids[0, pos:pos + n] = np.asarray(ids) + 7
    cache.offset += n
    return mx.array([[float(n)]])


def widen(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def capture(cache) -> dict:
    """Every buffer a forward at this offset can READ, widened to fp32."""
    ng = int(cache.offset) // RATIO
    out: dict = {}
    for i, lc in enumerate(cache.layers):
        out[f"win{i}"] = widen(lc.win_kv)                # all slots (rings alias)
        for nm in ("comp_kv", "index_k"):
            buf = getattr(lc, nm)
            if buf is not None:
                out[f"{nm}{i}"] = widen(buf[:, :ng])
        m = int(cache.offset) % RATIO
        if lc.comp_state is not None and m:
            out[f"kv{i}"] = widen(lc.comp_state.kv_state[:, :m])
            out[f"sc{i}"] = widen(lc.comp_state.score_state[:, :m])
    if cache.engram_ids is not None:
        out["engram"] = np.array(cache.engram_ids[:, :int(cache.offset)])
    return out


def assert_same_state(a: dict, b: dict, where: str = "") -> None:
    assert set(a) == set(b), f"state keys differ {where}: {set(a) ^ set(b)}"
    for k in a:
        if not np.array_equal(a[k], b[k]):
            n_diff = int(np.sum(a[k] != b[k]))
            raise AssertionError(f"state[{k}] differs ({n_diff} elems) {where}: {a[k]} != {b[k]}")


# --------------------------------------------------------------------------
# test 2 -- grid exactness of the real fake-quant functions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fn,block", [
    (fake_quant_fp8_ue8m0, 32), (fake_quant_fp4_e4m3, 16),
    (fake_quant_fp4_ue8m0, 32),
], ids=["fp8_ue8m0", "fp4_e4m3", "fp4_ue8m0"])
def test_fakequant_grid_is_bf16_exact(fn, block) -> None:
    """Q(x).astype(bf16).astype(fp32) == Q(x), over many random inputs."""
    rng = np.random.default_rng(0)
    for scale in (0.01, 0.3, 1.0, 4.0):
        x = mx.array((rng.standard_normal((8, 512)) * scale).astype(np.float32))
        q = fn(x, block)
        mx.eval(q)
        round_trip = q.astype(mx.bfloat16).astype(mx.float32)
        assert bool(mx.array_equal(q, round_trip)), (scale, int(mx.sum(q != round_trip)))


# --------------------------------------------------------------------------
# test 3 -- bf16 storage is lossless: writers store exactly what they're given
# --------------------------------------------------------------------------

def test_buffer_storage_round_trips_grid_values_bitwise() -> None:
    """Values stored through the REAL writers read back widened bitwise-equal.

    The fp32 escape: the value fed to each writer is already the fp32
    fake-quant output (the exact fp32 a reader used to see when the buffers
    were fp32). bf16 storage must reproduce it, so the comparison is a genuine
    fp32-vs-bf16 result comparison, only done by construction.
    """
    rng = np.random.default_rng(1)
    lc = LayerCache(1, stub_args(), 0, MAX_SEQ)          # a kv+index source
    assert lc.dtype == mx.bfloat16                       # storage is bf16 always

    # win_kv via write_window (fp8 grid)
    kv = mx.array(rng.standard_normal((1, 5, 512)).astype(np.float32))
    kv_q = fake_quant_fp8_ue8m0(kv, 32)
    mx.eval(kv_q)
    lc.write_window(10, kv_q)
    mx.eval(lc.win_kv)
    got = widen(lc.win_kv[0, mx.arange(10, 15) % WINDOW])
    assert np.array_equal(got, np.array(kv_q[0].astype(mx.float32)))

    # comp_kv store (fp4/e4m3 grid), exactly the attention.py store expression
    lat = mx.array(rng.standard_normal((1, 3, 512)).astype(np.float32))
    lat_q = fake_quant_fp4_e4m3(lat, 16)
    mx.eval(lat_q)
    lc.comp_kv[0, 4:7] = lat_q[0].astype(lc.dtype)
    mx.eval(lc.comp_kv)
    assert np.array_equal(widen(lc.comp_kv[0, 4:7]),
                          np.array(lat_q[0].astype(mx.float32)))

    # index_k store (fp4/ue8m0 grid), exactly the indexer.publish_keys store
    idx = mx.array(rng.standard_normal((1, 3, 128)).astype(np.float32))
    idx_q = fake_quant_fp4_ue8m0(idx, 32)
    mx.eval(idx_q)
    lc.index_k[0, 4:7] = idx_q[0].astype(lc.dtype)
    mx.eval(lc.index_k)
    assert np.array_equal(widen(lc.index_k[0, 4:7]),
                          np.array(idx_q[0].astype(mx.float32)))


def test_compressor_state_stays_fp32_and_engram_int64() -> None:
    """The two buffers NOT on a coarse grid keep their wide dtypes."""
    lc = LayerCache(1, stub_args(engram=False), 0, MAX_SEQ)
    # layer 0 is a kv source with ratio 2 -> has a CompressorState
    assert lc.comp_state is not None
    assert lc.comp_state.kv_state.dtype == mx.float32
    assert lc.comp_state.score_state.dtype == mx.float32
    c = make_cache(engram=True)
    assert c.engram_ids.dtype == np.int64


# --------------------------------------------------------------------------
# test 4 -- grow-on-demand replay is bitwise a never-grow replay
# --------------------------------------------------------------------------

def run_sequence(initial_capacity: int, engram: bool = True) -> list[dict]:
    """A deterministic op sequence crossing >= 2 growth boundaries.

    Interleaves append_turn / append_tokens / prefix-mismatch rollback /
    snapshot-then-restore / cancel, capturing the full readable state after
    every step.
    """
    model = _StubModel(engram=engram, initial_capacity=initial_capacity)
    s = SC.SessionCache(model, max_seq_len=MAX_SEQ, prefill_fn=stub_prefill,
                        max_snapshots=8)
    ids = np.arange(80, dtype=np.int64)
    steps = [capture(s.cache)]

    s.append_turn(ids[:20]);  steps.append(capture(s.cache))
    s.append_turn(ids[:40]);  steps.append(capture(s.cache))      # 1st boundary
    s.append_tokens(ids[40:70]); steps.append(capture(s.cache))   # 2nd boundary
    # rollback via a prefix mismatch
    mut = ids.copy(); mut[5] = 999
    s.append_turn(mut[:50]);  steps.append(capture(s.cache))
    # snapshot, advance, rewind to the snapshot
    snap = s.snapshot()
    s.append_tokens(ids[50:64]); steps.append(capture(s.cache))
    s.rewind(snap.pos);       steps.append(capture(s.cache))
    # cancel (rewind to the newest checkpoint below the offset)
    s.append_tokens(ids[50:58]); steps.append(capture(s.cache))
    s.cancel();               steps.append(capture(s.cache))
    s.append_turn(ids[:72]);  steps.append(capture(s.cache))
    return steps


@pytest.mark.parametrize("engram", [False, True], ids=["no_engram", "engram"])
def test_grow_matches_never_grow_bitwise(engram) -> None:
    """Growing through several boundaries == preallocated-to-cap, every step."""
    grew = run_sequence(initial_capacity=8, engram=engram)     # grows 8->16->...
    fixed = run_sequence(initial_capacity=MAX_SEQ, engram=engram)  # never grows
    assert len(grew) == len(fixed)
    for i, (a, b) in enumerate(zip(grew, fixed)):
        assert_same_state(a, b, f"step {i}")


def test_growth_actually_happened() -> None:
    """Guard: the growing run crossed >= 2 boundaries (else the test is vacuous)."""
    model = _StubModel(engram=False, initial_capacity=8)
    s = SC.SessionCache(model, max_seq_len=MAX_SEQ, prefill_fn=stub_prefill)
    caps = [s.cache.capacity]
    for n in (12, 30, 60):
        s.append_turn(np.arange(n, dtype=np.int64))
        caps.append(s.cache.capacity)
    assert caps[0] == 8
    assert sum(1 for a, b in zip(caps, caps[1:]) if b > a) >= 2, caps


def test_draft_boundary_writes_land_on_capacity_edge() -> None:
    """A speculative round's 1+gamma rows straddling the edge grows first.

    Without the ensure-before-graph invariant the rows at positions
    [cap-1, cap-1+1+gamma) would write past the allocated comp_kv/index_k; the
    growth makes them land correctly and the window ring is fixed-size anyway.
    """
    cache = make_cache(initial_capacity=8)                      # 4 latent rows
    assert cache.capacity == 8
    # advance to the edge (offset 7 = last allocated token position)
    cache.offset = 7
    gamma = 3
    need = cache.offset + 1 + gamma                            # 11 > 8
    cache.ensure_capacity(need)
    assert cache.capacity >= need
    # now the round's 1+gamma rows can be written at positions [7, 11)
    for lc in cache.layers:
        if lc.comp_kv is not None:
            g0 = 7 // RATIO
            g1 = (7 + 1 + gamma) // RATIO
            lc.comp_kv[0, g0:g1] = (7.0 + np.arange(g0, g1))[:, None]
            lc.index_k[0, g0:g1] = (8.0 + np.arange(g0, g1))[:, None]
    cache.offset = need
    mx.eval(*[lc.comp_kv for lc in cache.layers if lc.comp_kv is not None])
    g0 = 7 // RATIO
    g1 = (7 + 1 + gamma) // RATIO
    for lc in cache.layers:
        if lc.comp_kv is not None:
            # the rows the round wrote (g0 .. g1) carry the values, at the new
            # width -- proving the write landed correctly past the old edge.
            want = (7.0 + np.arange(g0, g1)).astype(np.float32)[:, None]
            assert np.array_equal(widen(lc.comp_kv[0, g0:g1]),
                                  np.broadcast_to(want, (g1 - g0, 512)))


# --------------------------------------------------------------------------
# test 5 -- rollback leaves no stale tail (poison test + sabotage)
# --------------------------------------------------------------------------

def test_rollback_then_grow_zeroes_region_beyond_used() -> None:
    """After a rollback near a boundary, everything past the live rows is 0."""
    cache = make_cache(initial_capacity=8)                      # 4 latent rows
    # fill the whole initial capacity with non-zero data
    cache.offset = 8
    for lc in cache.layers:
        if lc.comp_kv is not None:
            lc.comp_kv[0, :4] = 9.0
            lc.index_k[0, :4] = 9.0
    if cache.engram_ids is not None:
        cache.engram_ids[0, :8] = 9
    mx.eval(*[lc.comp_kv for lc in cache.layers if lc.comp_kv is not None])

    # roll the offset back, then grow well past the old width
    cache.offset = 5
    cache.ensure_capacity(40)                                  # grows 8 -> 16 -> 32
    assert cache.capacity >= 40

    used = -(-5 // RATIO)                                      # ceil(offset/ratio) = 3
    for lc in cache.layers:
        if lc.comp_kv is None:
            continue
        # live rows preserved
        assert np.array_equal(widen(lc.comp_kv[0, :used]), np.full((used, 512), 9.0))
        # EVERYTHING beyond the live rows is zero, including the old-but-now-
        # stale [used:old_width] region
        assert float(mx.sum(lc.comp_kv[0, used:])) == 0.0
        assert float(mx.sum(lc.index_k[0, used:])) == 0.0
    if cache.engram_ids is not None:
        assert (cache.engram_ids[0, 5:] == 0).all()
        assert (cache.engram_ids[0, :5] == 9).all()


def test_poison_check_sabotage_catches_full_buffer_copy(monkeypatch) -> None:
    """Sabotage: copying the FULL old buffer makes the poison assertion FAIL.

    Proves the poison test above actually exercises the copy-used-rows-only
    design (a naive grow that copies the whole old buffer would leave stale
    non-zero rows beyond ``used``).
    """
    def bad_grow(self, new_rows, used_rows):
        old = self.comp_kv
        if old is None or new_rows <= old.shape[1]:
            return
        self.comp_kv = mx.zeros((old.shape[0], new_rows, old.shape[2]),
                                dtype=old.dtype)
        self.comp_kv[:, :old.shape[1]] = old          # SABOTAGE: full copy
        if self.index_k is not None:
            idx = self.index_k
            self.index_k = mx.zeros((idx.shape[0], new_rows, idx.shape[2]),
                                    dtype=idx.dtype)
            self.index_k[:, :idx.shape[1]] = idx

    monkeypatch.setattr(LayerCache, "grow_latent", bad_grow)

    cache = make_cache(initial_capacity=8)
    cache.offset = 8
    for lc in cache.layers:
        if lc.comp_kv is not None:
            lc.comp_kv[0, :4] = 9.0
            lc.index_k[0, :4] = 9.0
    mx.eval(*[lc.comp_kv for lc in cache.layers if lc.comp_kv is not None])
    cache.offset = 5
    cache.ensure_capacity(40)

    used = -(-5 // RATIO)
    # With the sabotage the stale rows [used:old_width] survive -> sum != 0.
    lc0 = cache.layers[0]
    assert lc0.comp_kv is not None
    assert float(mx.sum(lc0.comp_kv[0, used:])) != 0.0, "sabotage was not caught"


# --------------------------------------------------------------------------
# test 6 -- CapacityError
# --------------------------------------------------------------------------

def test_capacity_error_past_max_seq_len() -> None:
    cache = make_cache(max_seq_len=64, initial_capacity=8)
    with pytest.raises(CapacityError):
        cache.ensure_capacity(65)
    # exactly at the cap is fine
    cache.ensure_capacity(64)
    assert cache.capacity == 64


def test_capacity_error_via_session_is_not_swallowed() -> None:
    """SessionCache.append_turn propagates the engine's CapacityError type."""
    model = _StubModel(engram=False)
    s = SC.SessionCache(model, max_seq_len=40, prefill_fn=stub_prefill)
    with pytest.raises(SC.CapacityError):
        s.append_turn(np.arange(64, dtype=np.int64))
    # The mlx-lm cache's own CapacityError is what SessionCache uses.
    from mlx_lm.models.deepseek_v41.cache import CapacityError as _CE
    assert SC.CapacityError is _CE


def test_allocation_failure_maps_to_capacity_error(monkeypatch) -> None:
    """A reallocation failure (OOM) is re-raised as CapacityError, naming it."""
    def boom(self, new_rows, used_rows):
        raise MemoryError("simulated OOM")

    monkeypatch.setattr(LayerCache, "grow_latent", boom)
    cache = make_cache(max_seq_len=MAX_SEQ, initial_capacity=8)
    with pytest.raises(CapacityError) as ei:
        cache.ensure_capacity(256)          # within max_seq_len -> hits the alloc
    assert "failed" in str(ei.value)


# --------------------------------------------------------------------------
# a stub model whose make_cache yields the real ModelCache
# --------------------------------------------------------------------------

class _StubModel:
    def __init__(self, engram: bool = False,
                 initial_capacity: int | None = None) -> None:
        self.args = ModelArgs()
        self.engram = engram
        self.initial_capacity = initial_capacity

    def make_cache(self, bsz: int = 1, max_seq_len: int | None = None,
                   dtype=None, **kw):
        return ModelCache(stub_args(self.engram), bsz, max_seq_len or MAX_SEQ,
                          initial_capacity=kw.get(
                              "initial_capacity", self.initial_capacity))
