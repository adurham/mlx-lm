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
5. `vision.py` + `image_processor.py` (new, workstream H): the DSv4.1 vision
   tower -- 32-layer ViT (dim 1024, 16 heads, inter 2816, patch 14, 2D RoPE),
   3x3 aligner, and the three learned sentinel embeddings -- loaded as bf16
   from the checkpoint's `vision.*` / `aligner.*` / `image_*` keys (266
   tensors, all BF16; the tower is NOT part of the EXL3 quantization).
   `image_processor.py` is a torch-free port of the release's preprocessing
   (4-type sentinel layout, NO compress padding: this is the DSv4.1 revision,
   not the older V4-Flash 5-type one). `VisionTower.merge_image_embeddings`
   is the weight-layer merge helper.

   Verified (fp32, vs the torch reference, 10 images incl. 2048x2048 and
   512x4096): worst aligner-output cos 0.999999994, worst block cos
   0.999999994 -- patches and token layouts bit-identical, rope tables
   within 1 fp32 ulp. At bf16 the tower's residual stream grows ~900x
   over 32 blocks, so bf16-vs-bf16 cos is 0.992-0.9997 for ANY
   implementation; the control is torch-bf16 vs torch-bf16 across kernels
   (CPU vs MPS) at 0.9944, and MLX-bf16 is no worse than torch-bf16 against
   the fp32 answer (error ratio 0.83-1.16). See `benchmarks/p97_*.py`.

   Per image (bf16, M4 Max): 148 ms / +160 MB transient for a 200-token
   image, 1.37 s / +873 MB for a 1024-token maximum image; tower resident
   971 MB; the ViT forward fences every block by default
   (`DSV41_VIT_FENCE=0` disables), which is free in time and cuts the peak
   up to 8x.

## Gate (exo phase 13)

Layer-by-layer against the p30 clamped trace (reconstructed-bf16 reference),
531-token prompt, real checkpoint: isolated per-layer cos >= 0.99983 (40/40);
teacher-forced NLL 1.0030 / top-1 78.3% (reference 1.003 / 78.3%).
