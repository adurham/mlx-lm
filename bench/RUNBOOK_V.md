# Stream V — GPU runbook (run ONLY in a production-down window)

Rationale: GPU contention alone stalls production's runner (two SIGKILLs on
2026-09-28/29). Nothing in this stream may run while production is up.
Every command below prints `mx.get_peak_memory()`; all are 1-layer probes.
Rsync the tree first (once per change):

```bash
rsync -a --exclude __pycache__ /home/hermes/work/dsv41-ws2/V/mlx_lm \
    macstudio-m4-1:dsv41-ws2/V/
rsync -a --exclude __pycache__ /home/hermes/work/dsv41-ws2/V/bench \
    macstudio-m4-1:dsv41-ws2/V/
```

Common env: `EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1`
and every GPU command wrapped in `lockf -k ~/dsv41-gpu.lock`.
`P=` = `~/repos/exo/.venv/bin/python`.

---

## 0. CPU-only, safe any time

```bash
$P bench/pV_route_stats.py
```
Shows, per layer, the fraction of an R-row window's slots that are distinct
experts. Expected (already measured): R=2 0.78, R=4 0.61, R=6 0.53 at layer 20
— i.e. 39% of R=4 slots are repeats. This is the *upper bound* on dedup, and it
is the number the PLAN quotes as "59-87% unique".

## 1. Reproduce the baseline marginal (1 layer, 3.6-6.4 GB)

```bash
cd ~/dsv41-ws2/V && PV7_PKG=$HOME/dsv41-ws2/V PV7_LAYERS=20 PV7_REPS=10 \
  PV7_ROWS=1 $P bench/pV7_attrib.py 2>&1 | grep -E "pV7|rror"
# repeat with PV7_ROWS=4
```
Shows step ms and per-phase ms/layer. Already measured: R1 5.12 ms → R4 7.11 ms
for layer 20 alone; ffn (MoE) 0.73 → 1.37 ms, attn 1.38 → 1.65 ms.
**Accept/expected after the fix:** ffn at R4 must fall; R1 must be unchanged.

## 2. Kernel-level attribution (1 layer, 1.7 GB)

```bash
cd ~/dsv41-ws2/V && PV5_PKG=$HOME/dsv41-ws2/V $P bench/pV5_moe.py
```
Already measured, and this is the decisive result of the stream:
- A2 (gate+up) 7 slots 0.505 ms → 56 slots 1.384 ms, ~0.028 ms/slot
- A2 at 28 slots with **one single expert for all 28 slots**: 0.761 ms vs
  0.819 ms with 28 distinct experts — the same within noise
- A2 at 28 slots with a **broadcast x row**: 0.819 ms vs 0.820 ms distinct
**Meaning:** the A2 grid is slot-count-bound, NOT expert-bandwidth-bound. The
kernel pays per (row, expert) slot whether or not that expert was already read.

## 3. The dense-GEMM premise, tested and closed (1 layer, 3.6 GB)

```bash
cd ~/dsv41-ws2/V && PV8_PKG=$HOME/dsv41-ws2/V PV8_LAYER=20 \
  PV8_ROWS=1,2,3,4,5,6,8 $P bench/pV8_modules.py
```
Every real layer-20 dense EXL3 module measured at 1..8 rows. Already measured:
**all slopes = 0.000 ms/row** (wq_a 0.096→0.094, wo_b 0.436→0.586 plateau,
shared_experts.w2 0.193→0.199). The premise in PLAN 2b(b) — "dense small-batch
GEMM re-streams the trellis per row" — **does not hold on this tree**: the
MT=8 blocked small-batch GEMM (`_gemm_simd_source` / `_gemm_devx_source`)
already decodes each weight tile once and applies it to all MT rows via the
transposed x staging buffer. This arm is therefore *closed with evidence*, not
implemented.

## 4. Where the win has to come from (1 layer, 1.7 GB)

```bash
cd ~/dsv41-ws2/V && PV9_PKG=$HOME/dsv41-ws2/V $P bench/pV9_dedup.py
```
Same geometry through both kernels (one real expert). Already measured:
dense_gu 0.208/0.207/0.285/0.295 ms at R=1/2/4/8 vs A2 0.190/0.282/0.273/0.356
— the dense path's per-row cost is ~0 and A2's is not. a2 per-slot cost falls
0.0511 (7 slots) → 0.0278 (42 slots).

## 5. Full-subset end-to-end (still 1-2 layers, NEVER the 8-layer harness)

```bash
cd ~/dsv41-ws2/V && PV4_PKG=$HOME/dsv41-ws2/V PV4_LAYERS=20,21 \
  PV4_ROWS=1,2,3,4,5,6 PV4_REPS=12 $P bench/pV4_verify.py
```
**This is the before/after acceptance command.** Baseline already measured
(peak 8.42 GB): R1 5.656 / R2 7.523 / R3 8.057 / R4 8.545 / R5 9.655 /
R6 10.069 ms → R4−R1 2.889 ms (0.963 ms/row), R6−R1 4.413 ms.
Target from PLAN 2b: R4−R1 and R5−R1 cut toward 8 ms/row-equivalent. With the
1-row-per-slot grid unchanged this cannot be met by dedup alone — see the
report. Rerun this identical command after the kernel change and paste both
runs.

## 6. Parity (must pass before any claim)

```bash
cd ~/dsv41-ws2/V && PV4_PKG=$HOME/dsv41-ws2/V PV4_LAYERS=20,21 \
  PV4_ROWS=1 PV4_REPS=6 $P bench/pV4_verify.py
```
R=1 must be **bit-identical** to the pre-change run. Save logits to compare:
the harness prints per-step ms only; for logits use `P48_SAVE`/`P48_CHECK`
(1-2 layer subset) or diff `PV8_JSON` dumps. Bit-identity is the accept bar for
R=1; R≥2 accepts cos ≥ 0.9998 (chunk-shape dependence is inherent, see spec.py).

---

## 7. The head, which is the biggest single term (1 layer, ~2 GB)

```bash
cd ~/dsv41-ws2/V && PV16_PKG=$HOME/dsv41-ws2/V $P bench/pV16_head.py
```
Already measured: 2.253 ms at R=1 -> 3.126 at R=2 -> 3.210 at R=4 -> 3.61 at
R=6/8 on ONE node. A one-off +0.95 ms step from R=1 to R>=2, then ~0.10 ms/row.
On the 2-layer window (5.12-7.11 ms total) this is a large share of what pV7
reported as unattributed. This, not the MoE, is where the next win is.

## 8. Dedup vs today's kernel (1 layer, ~2 GB) -- the verdict arm

```bash
cd ~/dsv41-ws2/V && PV15_PKG=$HOME/dsv41-ws2/V $P bench/pV15_a2vsdense.py
```
Already measured: per-expert dedup is 1.60/1.79/2.13/1.92x SLOWER than A2 at
R=2/4/6/8 with min cos 1.000000. Re-run this before funding a dedup kernel.

## What is NOT to be run

- `bench/pV3_matrix.sh` — it drives **p48 with 8 layers at 23.2 GB active**.
  It was that job size that stalled production's runner twice on 2026-09-28/29.
  Its numbers are reproduced in the commit message; there is no need to rerun
  it. If an 8-layer figure is needed again, take it from a scheduled window.
- Any two-node command (parent only, per BRIEF).
