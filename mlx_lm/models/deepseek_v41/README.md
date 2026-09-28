# deepseek_v41 -- DeepSeek-V4.1-Flash text stack

Model code vendored from [PipeNetwork/deepseek-v41-mlx](https://github.com/PipeNetwork/deepseek-v41-mlx)
(Apache-2.0, `LICENSE` here) at upstream `a9567ae`, plus the node-local
MTP/DSpark port edits (exo `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27`).
Imports rewritten to relative. Not vendored: `convert.py`, `load.py`,
`generate.py`, `stream.py`, `mtp_decode.py`, `fakequant` test hooks' users.

## Local changes

1. `moe.py`: `MoE.group` -- when set, the routed-expert partial sum is
   `all_sum`ed over the TP group before the (replicated) shared expert is added.
2. `attention.py`: `wo_a` may be a grouped module (EXL3 stores one tensor
   group per output-LoRA slice); the `nn.Linear` einsum path is unchanged.
3. `compressor.py`: `wkv`/`wgate` may be quantized modules; the `nn.Linear`
   fp32 path is unchanged.
4. `exl3_build.py` (new): builds the model from an EXL3 checkpoint --
   `EXL3Linear` for every trellis group, stacked `EXL3SwitchGLU` experts
   (optional rank slice), row-on-demand engram tables from the native
   release, strict per-layer accounting.

## Gate (exo phase 13)

Layer-by-layer against the p30 clamped trace (reconstructed-bf16 reference),
531-token prompt, real checkpoint: isolated per-layer cos >= 0.99983 (40/40);
teacher-forced NLL 1.0030 / top-1 78.3% (reference 1.003 / 78.3%).
