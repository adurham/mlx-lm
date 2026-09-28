# Stream D benchmarks (EXL3 prefill at R=512, DSv4.1)

All numbers from single-node runs on m4-1/m4-2 under `lockf ~/dsv41-gpu.lock`
with `EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1`, python
`~/repos/exo/.venv/bin/python`, tree at `~/dsv41-ws/D`.

| script | what it measures |
|---|---|
| `pD1_dense.py` | dense fullW vs striped vs fused at 8 rank-0 projection shapes, R=512 |
| `pD2_moe.py` | MoE `_prefill` geometry + segment stats (block occupancy) |
| `pD3_probe.py` | isolates the `_prefill` overflow (kernels vs fallback) |
| `pD4_moe.py` | v19c timing + proof the gather fallback overflows |
| `pD5_guard.py` | first row-guard prototype (v19d) |
| `pD6_bm.py` | bm sweep 8..64 proving block size is not the lever |
| `pD7_decode.py` | NODECODE + LUT ablations (decode = 3% of the cost) |
| `pD8_cross.py` | dense path sweep R=8..768 (the threshold evidence) |
| `pD9_variants.py` | v19c/d/e/f/g kernel variants, parity + speedup |
| `pDA_cross_band.py` | fine sweep of the 16..64 row band (threshold pin) |
| `pDB_prefill_ab.py` | end-to-end `_prefill` A/B + decode-floor check |
| `pDC_model_ab.py` | 531-token model forward A/B, logit bit-compare |
| `pDD_segment.py` | model-level MoE segment attribution (stub ablation) |
| `pDE_segment.py` | `_prefill` at the exact 531x6 shape, uniform vs skewed |
| `pDF_band.py` | dense band 17..64 logits A/B (old vs new threshold) |
| `pDG_transients.py` | transients + branch attribution |
| `pDH_head.py` | lm_head R=512 fullW vs striped (cap check) |
| `pDI_invariance.py` | decode/verify bit-invariance under both env switches |
| `pDJ_shape_cross.py` | crossover vs projection geometry (non-DSv4.1 shapes) |
| `pDK_fallback.py` | gather fallback int32 proof + seg scale to N=18432 |
| `v19d.py`, `v19e.py`, `v19f.py` | kernel-variant prototypes (source generators) |
