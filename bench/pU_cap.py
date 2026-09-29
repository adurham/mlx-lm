#!/usr/bin/env python3
"""Shared expert-cap helper for the stream-U harnesses.

A body layer subset plus the draft head does not fit the 8 GB rule with the
full 384/128-expert tables, so the harnesses build with an expert count cap.
The cap is applied consistently at three places -- the router width
(``args.n_routed_experts`` / ``args.dspark_n_experts``) and the loaded tables
(``exl3_build.load_experts``) -- so the model is internally consistent: same
classes, same geometry rules, just fewer expert streams. The round structure
(draft, verify, sync points) is unchanged; the absolute ms differ from the
full model and the harness says so.

``apply_caps(body=None, head=None)`` patches the loader on import of
``mlx_lm.models.deepseek_v41.exl3_build``; call before building.
"""
from __future__ import annotations

import os

_cap_body = None
_cap_head = None


def apply_caps(body=None, head=None):
    global _cap_body, _cap_head
    _cap_body, _cap_head = body, head
    from mlx_lm.models.deepseek_v41 import exl3_build as eb

    if getattr(eb, "_pU_capped", False):
        return
    real = eb.load_experts

    def patched(ckpt, layer_id, *, n_experts=None, prefix=None, **kw):
        is_head = bool(prefix and prefix.startswith("mtp."))
        if n_experts is None:
            cap = _cap_head if is_head else _cap_body
            if cap:
                n_experts = cap
        return real(ckpt, layer_id, n_experts=n_experts, prefix=prefix, **kw)

    eb.load_experts = patched
    eb._pU_capped = True


def caps_from_env():
    """PU_BODY_EXPERTS / PU_HEAD_EXPERTS -> apply_caps()."""
    b = os.environ.get("PU_BODY_EXPERTS")
    h = os.environ.get("PU_HEAD_EXPERTS")
    apply_caps(int(b) if b else None, int(h) if h else None)
    return (int(b) if b else None, int(h) if h else None)


def body_args(model_dir: str, body_cap: int | None):
    """ModelArgs from the checkpoint config with the body router width capped."""
    from mlx_lm.models.deepseek_v41.config import ModelArgs
    from mlx_lm.models.exl3.loader import Exl3Checkpoint

    ck = Exl3Checkpoint(model_dir)
    args = ModelArgs.from_dict(ck.config)
    if body_cap:
        args.n_routed_experts = body_cap
    args.n_mtp_layers = 0          # body build ignores mtp; keep the arg honest
    return args
