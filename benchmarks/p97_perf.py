"""Per-image time and memory for the MLX DSv4.1 vision tower.

Reports, per fixture:
  * vit grid, llm grid (token count), patch count,
  * encode_image wall time (median of P97_REPS, warm), and
  * transient memory: (peak - active_before) across one forward, plus the
    tower's resident cost after load.

Memory is measured as a DELTA around the forward, with ``mx.reset_peak_memory``
called immediately before it, so the tower's own weight footprint (measured
separately) is not counted twice.

  EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python p97_perf.py

PRODUCTION IS RUNNING: hold the lock, ONE job at a time. P97_LAYERS=0-3 runs a
4-block subset (peak < 1 GB); the full 32-block run peaks at 1.1-1.9 GB MLX
total and is safe to run alone but must not be stacked against another job.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dsv41-ws/H"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
from p97_vision_parity import build_fixtures  # noqa: E402
from mlx_lm.models.deepseek_v41 import image_processor as mip  # noqa: E402
from mlx_lm.models.deepseek_v41 import vision as mv  # noqa: E402

CKPT = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)


def main() -> int:
    reps = int(os.environ.get("P97_REPS", "3"))
    dtype_name = os.environ.get("P97_DTYPE", "bf16")
    mlx_dtype = mx.float32 if dtype_name == "f32" else mx.bfloat16

    # resident cost of the tower alone
    base = mx.get_active_memory()
    t0 = time.perf_counter()
    tower, cfg = mv.load_vision_tower(CKPT, dtype=mlx_dtype)
    layers = os.environ.get("P97_LAYERS")
    if layers:  # cheap subset run (production-memory rules)
        lo, _, hi = layers.partition("-")
        tower.vision.blocks = [tower.vision.blocks[i] for i in range(int(lo), int(hi) + 1)]
        print(f"[setup] LAYER SUBSET {lo}-{hi} ({len(tower.vision.blocks)} blocks)")
    mx.eval(tower.parameters())
    load_s = time.perf_counter() - t0
    resident_mb = (mx.get_active_memory() - base) / 1e6
    print(f"[load] {resident_mb:.1f} MB resident ({dtype_name}), {load_s:.2f} s")

    rows = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for i, rec in enumerate(build_fixtures()):
        patches, nh, nw, nlh, nlw = mip.load_image(rec, cfg.preprocess)
        x = mip.patches_to_mlx(patches).astype(mlx_dtype)
        n_tok = mip.num_image_tokens(nlh, nlw)
        # warm
        out = tower.encode_image(x, nh, nw)
        mx.eval(out)
        times, peaks = [], []
        for _ in range(reps):
            before = mx.get_active_memory()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            out = tower.encode_image(x, nh, nw)
            mx.eval(out)
            times.append(time.perf_counter() - t0)
            peaks.append((mx.get_peak_memory() - before) / 1e6)
        ms = sorted(times)[len(times) // 2] * 1e3
        peak = max(peaks)
        rows.append(dict(name=rec["name"], vit=[nh, nw], llm=[nlh, nlw], tokens=n_tok,
                         patches=int(patches.shape[0]), ms=ms, peak_delta_mb=peak))
        print(f"[{i}] {rec['name']:>14} vit {nh:>3}x{nw:<3} {n_tok:>4} tok  "
              f"{ms:7.1f} ms (med of {reps})  peak+{peak:7.1f} MB")

    agg = dict(
        dtype=dtype_name,
        tower_resident_mb=resident_mb,
        load_seconds=load_s,
        median_ms=float(np.median([r["ms"] for r in rows])),
        max_peak_delta_mb=float(max(r["peak_delta_mb"] for r in rows)),
        rows=rows,
    )
    print(f"[peak] MLX total peak over the whole run: {mx.get_peak_memory() / 1e9:.2f} GB")
    print(json.dumps({k: v for k, v in agg.items() if k != "rows"}, indent=2))
    if os.environ.get("P97_JSON"):
        with open(os.path.expanduser(os.environ["P97_JSON"]), "w") as f:
            json.dump(agg, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
