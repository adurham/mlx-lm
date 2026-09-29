# Stream V — the DSv4.1 verify marginal: measured, and where it actually is

Branch `ws2/V`, commits `f2ad503`, `7c67543`, `178c49f`, `0b00921`, `87c27c9`.
All numbers below are from runs on `macstudio-m4-1`, single node, one job at a
time under `lockf`, every run printing `mx.get_peak_memory()`.

## Verdict on PLAN 2b — both named mechanisms fail on this tree

| PLAN 2b | measurement | verdict |
|---|---|---|
| **(b)** "dense EXL3 small-batch GEMM: stop re-streaming the trellis per row" | **pV8**: every real layer-20 dense module at R=1..8, slope **0.000 ms/row** (wq_a 0.096→0.094, wkv 0.092 flat, wo_b 0.436→0.586 plateau, shared_experts.w2 0.193→0.199...) | **does not hold** — the MT=8-blocked small-batch GEMM (`_gemm_simd_source` / `_gemm_devx_source`) already decodes each tile once and applies it to all 8 rows via the transposed x staging buffer |
| **(a)** "MoE at R=2..6: read each UNIQUE expert once per verify window" | **pV15**: per-expert dedup (one mt-blocked dense GEMM per unique expert, same math, **min cos 1.000000**) is **1.60/1.79/2.13/1.92× SLOWER** than today's A2 at R=2/4/6/8 | **fails** — A2's per-slot cost (0.025–0.031 ms/slot) is already below the launch overhead a grouped schedule pays to save bytes |

Supporting: **pV11** (28-slot R=4 window, unique experts swept 6→24, 3 sweeps ×
31 reps, interleaved) — end-to-end **1.157 → 1.163 ms**, a 0.5% move. A2 reads
one tile per (slot, tile) pair, so it does not dedup at all, and its flat cost
across a 4× swing in distinct experts says expert bytes are not its bottleneck.
**pV12** — grouping the same slots into per-expert launches is not cheaper than
one launch once measured correctly (the first run's `mx.eval(mx.zeros(1))` did
not order the outputs it was timing; fixed, then 0.798 ms vs 0.742 ms).
**pV10** — every existing knob in the verify band is flat or worse:
`EXL3_MOE_A2_TILES` 1/2/4 → R4−R1 2.894/2.954/3.002 ms; `EXL3_MOE_CHUNK`
0/128/256 → 2.958/2.954/~same; `EXL3_GEMM_DEVX=0` → worse at R6 (6.313 vs
4.442); `EXL3_MOE_V2=0` (v1 kernels) → much worse (4.296 vs 2.894).

## The marginal budget, layer 20, single node

pV7 phase split (corrected units; the parent caught the seconds-as-ms bug):

| phase | R=1 | R=4 | Δ |
|---|---:|---:|---:|
| hc (4 segments) | 0.644 | 0.791 | +0.147 |
| attn | 1.382 | 1.651 | +0.269 |
| ffn (MoE) | 0.728 | 1.370 | **+0.642** |
| outside the block (embed, collapse, **head**) | 2.372 | 3.285 | **+0.913** |
| **step total** | **5.124** | **7.107** | **+1.983** |

Window totals on the 2-layer probe (pV4, `PV4_LAYERS=20,21`, peak 8.42 GB):
**R1 5.656 · R2 7.523 · R3 8.057 · R4 8.545 · R5 9.655 · R6 10.069 ms**, i.e.
R4−R1 **2.889 ms (0.963 ms/row)**, R6−R1 4.413 ms.

## Attention's per-row growth (pV18) — decomposed, and it is not bytes

| segment | R=1 | R=4 | R=6 | Δ 1→6 |
|---|---:|---:|---:|---:|
| qpath (wq_a→q_norm→wq_b→rope) | 0.542 | 0.697 | 0.789 | **+0.247** |
| kvpath (wkv→norm→rope→fakequant) | 0.236 | 0.225 | 0.226 | −0.010 |
| idx (window idx + cidx concat) | 0.458 | 0.432 | 0.426 | −0.032 |
| sparse (the attention kernel) | 0.283 | 0.337 | 0.360 | +0.077 |
| outpath (rope inv, wo_a, wo_b) | 0.478 | 0.625 | 0.737 | **+0.259** |
| attention total | 1.996 | 2.316 | 2.538 | **+0.542** |

But **pV19** priced those same chains in isolation at rows 1..8: qpath
**0.509 → 0.441–0.448** (flat), `attn.wo_a` grouped stack 0.428 → 0.510–0.584
(+0.08, then flat). Layer 20 has **no `Exl3FusedGroup`** — the fused-group path
only engages above 16 rows. So the attention growth is a **scheduling/launch
interaction in the assembled graph**, not module byte traffic, and not fused-
group restacking.

## The LM head is the largest single term (pV16)

| R | head ms (one node, single-node build) |
|---:|---:|
| 1 | **2.253** |
| 2 | 3.126 |
| 4 | 3.210 |
| 6 | 3.615 |
| 8 | 3.607 |

A one-off **+0.95 ms** step from R=1 to R≥2, then ~0.10 ms/row. On the 2-layer
window (5.12–7.11 ms) this is a large share of what pV7 reports as unattributed.
Head trellis is 0.496 GB (320×8080×96 int16); the two-node build shards vocab,
so per-node bytes are ~half. pV17 (bandwidth test) had a units bug, fixed in
`178c49f`, and was not re-run — the bandwidth question is still open.

## What is dead

- **PLAN 2b(b)** (dense per-row trellis re-streaming): does not exist here.
- **PLAN 2b(a)** (per-window unique-expert dedup): measured slower, 4 ways.
- **Existing knobs**: `EXL3_MOE_A2_TILES`, `EXL3_MOE_CHUNK`, `EXL3_GEMM_DEVX`,
  `EXL3_MOE_V2` — none help in the verify band.

## What to fund next (in order of measured value)

1. **The head.** +0.91 ms of the +1.98 ms block-external marginal, and it is
   evaluated for every verify row. Check whether the acceptance loop in
   `spec.py` really needs logits for all R rows, or only the row that can be
   committed (production's exo already runs `EXO_DSV4_LMHEAD_LASTROW=1`).
2. **The qpath/outpath scheduling interaction** (+0.51 ms of the +0.54 ms
   attention growth, with each module provably flat alone). This needs a
   timeline (Metal capture), not more microbenchmarks.
3. **Tile-level dedup inside A2** — keep A2's grid and parallelism, have
   threadgroups for a repeated expert tile reuse a decoded tile instead of
   re-decoding. The only dedup form that does not pay pV15's launch tax.

## Corrections logged during the stream

- pV6: percent/unattributed math treated seconds as milliseconds (parent-caught).
- pV12: `mx.eval(mx.zeros(1))` does not order the kernel outputs being timed;
  the first grouped-arm figure (0.150 ms) was graph construction only.
- pV11: an initial write-up claimed a "4× traffic swing" A2 could exploit; the
  bytes are constant per slot, so the claim was withdrawn.
- pV17: double-scaled its ms timings by 1e3; fixed, not re-run.

## Exact GPU commands for an end-to-end before/after

Production down, one job per node, all under `lockf`. Rsync once:
`rsync -a --exclude __pycache__ <worktree>/mlx_lm macstudio-m4-1:dsv41-ws2/V/`
and same for `bench/`.

```bash
# 1. window baseline + acceptance (1-2 layers, ~8.4 GB) -- THE key command
cd ~/dsv41-ws2/V && PV4_PKG=$HOME/dsv41-ws2/V PV4_LAYERS=20,21 \
  PV4_ROWS=1,2,3,4,5,6 PV4_REPS=12 EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
  lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV4_verify.py
#    expect (measured): R1 5.656 R4 8.545 R6 10.069; R4-R1 2.889 ms.
#    rerun identically after any change; R=1 must stay bit-identical.

# 2. phase attribution (1 layer, ~6.4 GB)
cd ~/dsv41-ws2/V && PV7_PKG=$HOME/dsv41-ws2/V PV7_LAYERS=20 PV7_REPS=10 \
  PV7_ROWS=4 EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
  lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV7_attrib.py

# 3. attention decomposition (1 layer, ~6.4 GB), one R per process
cd ~/dsv41-ws2/V && PV18_PKG=$HOME/dsv41-ws2/V PV18_LAYERS=20 PV18_ROWS=4 \
  PV18_REPS=10 EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
  lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV18_attn.py

# 4. the head (1 layer, ~2 GB)
cd ~/dsv41-ws2/V && PV16_PKG=$HOME/dsv41-ws2/V EXL3_MM_MAX_ROWS=100000 \
  MTL_DISABLE_TIMEOUT=1 lockf -k ~/dsv41-gpu.lock \
  ~/repos/exo/.venv/bin/python bench/pV16_head.py

# 5. re-confirm dedup is dead before funding a kernel (1 layer, ~2 GB)
cd ~/dsv41-ws2/V && PV15_PKG=$HOME/dsv41-ws2/V EXL3_MM_MAX_ROWS=100000 \
  MTL_DISABLE_TIMEOUT=1 lockf -k ~/dsv41-gpu.lock \
  ~/repos/exo/.venv/bin/python bench/pV15_a2vsdense.py

# 6. R=1 bit-identity gate (2 layers, ~7 GB) -- run once to save, again after
cd ~/dsv41-ws2/V && PV13_PKG=$HOME/dsv41-ws2/V PV13_TAG=before \
  PV13_LAYERS=20,21 EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
  lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV13_parity.py
#    then re-run with PV13_TAG=after PV13_CHECK=$HOME/pV13_before.npz
#    expect R1 exact=True; R>=2 cos >= 0.9998.
```

Two-node and full-model runs stay with the parent. `bench/pV3_matrix.sh` drives
the 8-layer p48 harness (~23 GB) — its numbers are already in commit `f2ad503`;
do not re-run it without a scheduled window.
