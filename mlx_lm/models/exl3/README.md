# EXL3 trellis kernels (vendored)

Vendored from [PonyExl3](https://github.com/beamivalice/PonyExl3) (Apache-2.0;
`LICENSE` and `NOTICE` copied alongside this file) at upstream commit
`8e7fa6b` — specifically from the **patched working tree** on
`macstudio-m4-1` (`~/repos/ref/PonyExl3`), which carries the kernel work of
2026-09-27/28 (md5-verified at vendoring time).

## Import surface

```python
from mlx_lm.models.exl3 import EXL3Linear, EXL3SwitchGLU
```

Only these two classes are exported. The rest of the upstream package
(`exl3_qmv`, `exl3_qmm`, `exl3_fused`, `weights`, `native`, `mtp`,
`generate`, model plumbing) is deliberately **not** vendored — it is not in
the import closure of the two classes. The vendored set is the full minimal
closure: 12 modules under `mlx/` plus 4 numpy reference modules under `ref/`
(kept because the Metal code builds its codebooks through them).

## Local changes vs upstream

1. Absolute `ponyexl3.*` imports rewritten to relative imports (including
   the four dynamic `__import__("ponyexl3.mlx.gemv_metal", ...)` sites in
   `exl3_moe.py`, now `__import__(f"{__package__}.gemv_metal", ...)`).
2. An 8-line provenance header prepended to each file.
3. `__init__.py` reduced to the two-class import surface (upstream's pulled
   in `forward`/`linear`/`generate` modules that are not vendored).
4. `ref/__init__.py` reduced correspondingly.
5. **`loader.py` is a LOCAL ADDITION, not vendored** — a checkpoint loader
   for exllamav3-style EXL3 safetensors directories that builds the stacked
   buffers the kernels consume, including the tensor-parallel intermediate
   slice (`load_experts(..., rank=r, world=2)`) proven in the exo repo's
   phase-12 doc. See its module docstring for the format contract.

**Numerical code is unchanged** except as noted in the headers, i.e. the
three changes already present in the node's patched tree:

- `exl3_moe.py` — v3 chunked-prologue retune; v4 ALU optimizations
  (defaulted ON via `EXL3_DECODE_SWAR`, `EXL3_XDIRECT`); additive
  `"silu_clamp"` MoE activation (`EXL3_MOE_CLAMP`, default 10.0) implementing
  DeepSeek V4.1's clamped SwiGLU semantics (gate upper-only, up two-sided).
- `gemv_metal.py` — v4 ALU optimizations (SWAR codeword decode, direct-read
  shuffle broadcast) defaulted ON.

## Equivalence gate (PASSED)

Ran 2026-09-28 on `macstudio-m4-1` (Apple M4 Max), python 3.14.2, mlx
0.32.2, against real `DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`
checkpoint tensors (layer 1 experts, the k=6 quantized head group):

| path | shapes | result |
|---|---|---|
| `EXL3SwitchGLU`, `silu` | R=1, R=4, R=8 | bit-identical (md5 match, `mx.array_equal`) |
| `EXL3SwitchGLU`, `silu_clamp` | R=1, R=4, R=8 | bit-identical |
| `EXL3Linear` (head group, k=6) | R=1, R=4 | bit-identical |
| `reconstruct_public_mlx` (expert w1) | full matrix | bit-identical |

Verdict: the vendored tree is bit-identical to upstream `ponyexl3` on every
tested path. Test script: `exl3_vendor_equiv.py` (kept in the phase-2
scratch area, not in this tree).

## Loader gate (PASSED)

`loader.py` (local addition) was gated on the same checkpoint tensors:

| check | result |
|---|---|
| loader-built module vs hand-built reference, R=1/4/8 | bit-identical |
| `rank=0/1, world=2` slice partials summed vs full | cos 1.0000000 (R=1) / 0.9999999 (R=4) |
| dense head group via `EXL3Linear` vs reconstruct+matmul | cos 0.9999996 |

Test script: `p43_loader_gate.py` (phase-12 scratch, not in this tree).

## Env knobs carried over

| var | default | meaning |
|---|---|---|
| `EXL3_DECODE_SWAR` | `1` | SWAR codeword decode (v4) |
| `EXL3_XDIRECT` | `1` | direct-read shuffle broadcast (v4) |
| `EXL3_MOE_CLAMP` | `10.0` | clamp limit for `silu_clamp` activation |
| `EXL3_MOE_A2_TILES`, `EXL3_MOE_CHUNK`, `EXL3_MOE_MM`, `EXL3_MM_MAX_ROWS` | — | tile/prologue geometry; see headers |

`EXL3_MM_MAX_ROWS` must be set (e.g. `100000`) **before import** to avoid the
upstream prefill-row crash at scale; `EXL3_MOE_MM=0` and the default
`EXL3_MM_MAX_ROWS` still crash at V4.1 scale (documented in the operations
skill).

## Provenance notes

- Source-of-record for these files is the *patched* node tree; upstream
  `8e7fa6b` alone does not contain the v3/v4/silu_clamp changes. The
  vendored copy is therefore newer than upstream `main`.
- Timings for these kernels on M4 Max (EXL3 vs production MXFP4 ratios
  1.91–2.05 across decode/verify shapes) live in the exo repo's
  `docs/benchmarks/phase9-exl3-spike-2026-09-28/`.
