"""Pure-python core of the DSv4.1 greedy parity harness (stream P, round 2).

Nothing here imports MLX or touches a GPU: it is the comparison / verdict /
reference-file layer, so it can be unit-tested anywhere (see
``test_parity_core.py``) and reused by the GPU runner (``dsv41_parity.py``).

Terms
-----
reference      a saved greedy run: prompt ids + the token stream the model
               produced (a JSON file, see :func:`save_reference`).
fingerprint    the fields that must be identical for two runs to be
               comparable at all (inputs, built layers, world/rank,
               max_new, checkpoint + token map identity).
cos gate       per-layer cosine between a built block's output and the p30
               reference trace, isolated layer by layer (bar 0.9998).
"""

from __future__ import annotations

import hashlib
import json
import os

SCHEMA = "dsv41-parity/1"

# The accepted full-40-layer gate (p46 / phase 13, p35 reference numbers).
COS_BAR = 0.9998
# Deviation tolerance for a reference-calibrated cos gate: a clean run must
# land within this of the recorded clean value, not merely above COS_BAR.
COS_DEV_TOL = 0.00005
NLL_BARS = {"mean_max": 1.010, "median_max": 0.040, "top1_min": 78.0}
REF_NLL = {"mean": 1.003, "median": 0.029, "top1": 78.3}
# Negative-control ladder policy: the lowest rung must NOT disturb parity, the
# highest rung must. (The rungs between are informational -- the measured
# threshold is not something to assert on.)
NEGCTL_LOW_MUST_PASS = "low rung must not disturb parity"
NEGCTL_HIGH_MUST_DETECT = "high rung must be detected"

# Structural fingerprint: mismatch here means the two runs are not the same
# experiment, so a token comparison is meaningless (hard fail). ``dist``
# separates a single-node simulated-TP build (no collective) from the real
# two-node jaccl build -- same world, different arithmetic. ``chunk`` is
# structural because the body's output depends on the prefill chunk boundaries
# (a property of the model, measured in prefill.py's docstring). ``trace_digest``
# pins WHICH recorded trace the cos gate compares against. ``rank`` is
# deliberately NOT here: ranks of one run are the same experiment.
FP_STRUCT_FIELDS = (
    "layers",
    "dist",
    "world",
    "max_new",
    "chunk",
    "model_dir",
    "native_dir",
    "token_map_digest",
    "trace_digest",
    "prompts_digest",
)

# The code-under-test digest is a gate, not a note: comparing tokens generated
# by different code to a reference is the failure mode a parity harness exists
# to prevent. Override deliberately with --allow-code-drift, which switches it
# to a reported warning (and says so in the JSON).
FP_CODE_FIELD = "pkg_digest"


# --------------------------------------------------------------------------
# hashing / digests
# --------------------------------------------------------------------------

def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_digest(path: str) -> str:
    """sha256 of a file's bytes (chunked; used for the token map)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_digest(root: str, rels) -> str:
    """Digest of the code under test: hash of (relpath, file hash) pairs.

    ``pkg_digest`` mismatch is a WARN, not a fail: comparing a run against a
    reference recorded from a different revision is exactly how you count how
    many tokens a change moved.
    """
    h = hashlib.sha256()
    missing = []
    for rel in sorted(rels):
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            missing.append(rel)
            continue
        h.update(rel.encode())
        h.update(file_digest(p).encode())
    return h.hexdigest()[:32] + (f"-missing{len(missing)}" if missing else "")


def digest_str_list(items) -> str:
    return sha256_text(canonical(list(items)))


def dir_manifest_digest(path: str, names=None) -> str:
    """Identity of a recorded trace: (name, size) for every file, sorted.

    Cheap on purpose (the p30 trace is 1.6 GB of .npy; hashing the bytes would
    cost a minute per run). The point is to catch comparing against a
    DIFFERENT trace, not to be a cryptographic commitment to its contents.
    """
    if not os.path.isdir(path):
        return "missing:" + path
    entries = []
    for name in sorted(names or os.listdir(path)):
        p = os.path.join(path, name)
        if os.path.exists(p):
            entries.append((name, os.path.getsize(p)))
    return sha256_text(canonical(entries))[:32]


# --------------------------------------------------------------------------
# config parsing helpers
# --------------------------------------------------------------------------

def parse_layers(spec, n_layers: int = 40):
    """``"20,21"`` / ``"2-5"`` / ``"all"`` -> sorted unique layer ids."""
    if spec is None or str(spec).strip().lower() in ("", "all", "*"):
        return list(range(n_layers))
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = sorted(x for x in out if x < 0 or x >= n_layers)
    if bad:
        raise ValueError(f"layers out of range 0..{n_layers - 1}: {bad}")
    return sorted(out)


def parse_perturb(spec):
    """``none`` | ``argmax:<step>`` | ``noise:<rel>`` | ``stub:<component>``."""
    s = (spec or "none").strip()
    if s in ("", "none"):
        return {"kind": "none"}
    if ":" not in s:
        raise ValueError(f"perturbation spec must be kind:value, got {s!r}")
    kind, val = s.split(":", 1)
    if kind == "argmax":
        return {"kind": "argmax", "step": int(val)}
    if kind == "noise":
        return {"kind": "noise", "rel": float(val)}
    if kind == "stub":
        return {"kind": "stub", "component": val}
    raise ValueError(f"unknown perturbation kind {kind!r}")


def parse_prompts_spec(spec):
    """``p30`` | ``chat`` | ``file:<path>`` (see dsv41_parity.load_prompts)."""
    s = (spec or "p30").strip()
    if s in ("p30", "chat") or s.startswith("file:"):
        return {"source": s}
    raise ValueError(f"unknown prompts spec {spec!r}")


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def compare_tokens(ref_tokens, cur_tokens):
    """Token-stream comparison of one prompt (greedy top-1 stream)."""
    ref = [int(t) for t in ref_tokens]
    cur = [int(t) for t in cur_tokens]
    n = min(len(ref), len(cur))
    first_div = next((i for i in range(n) if ref[i] != cur[i]), None)
    n_match = sum(1 for i in range(n) if ref[i] == cur[i])
    length_diff = len(ref) != len(cur)
    ok = first_div is None and not length_diff
    return {
        "ok": ok,
        "n_ref": len(ref),
        "n_cur": len(cur),
        "n_match": n_match,
        "n_common": n,
        "first_div": first_div,
        "length_diff": length_diff,
        "cur_tokens": cur if not ok else [],
    }


def compare_runs(ref_prompts, cur_prompts):
    """Per-prompt comparison of two recorded runs. Inputs must line up."""
    if len(ref_prompts) != len(cur_prompts):
        return {"ok": False, "error": f"prompt count {len(ref_prompts)} vs {len(cur_prompts)}",
                "prompts": [], "mismatch_total": None}
    out, bad = [], 0
    for r, c in zip(ref_prompts, cur_prompts):
        if list(r["ids"]) != list(c["ids"]):
            out.append({"name": c.get("name", r.get("name", "?")), "ok": False,
                        "error": "prompt ids differ", "first_div": None,
                        "n_match": 0, "n_ref": len(r["tokens"]),
                        "n_cur": len(c["tokens"]), "length_diff": None})
            bad += 1
            continue
        rep = compare_tokens(r["tokens"], c["tokens"])
        rep["name"] = c.get("name", r.get("name", "?"))
        rep["ids"] = list(c["ids"])
        if not rep["ok"]:
            bad += 1
        out.append(rep)
    return {"ok": bad == 0, "prompts": out, "mismatch_total": bad,
            "n_prompts": len(out)}


def cos_gate(items, bar: float = COS_BAR, ref_cos=None):
    """``items``: [{"layer": int, "cos": float, ...}] -> pass/fail vs ``bar``.

    ``ref_cos`` (optional {layer: observed-clean cos}) turns the gate into a
    DEVIATION gate: the bar becomes max(bar, ref - tolerance) per layer, which
    is what a rollout gate wants (a clean run must land where the recorded
    clean run landed, not merely above an absolute floor).
    """
    vals = [(int(i["layer"]), float(i["cos"])) for i in items]
    worst = min((c for _, c in vals), default=None)
    wl = min((l for l, _ in vals), key=lambda l: dict(vals)[l], default=None)
    if ref_cos:
        dev = {str(l): c - ref_cos[l] for l, c in vals if l in ref_cos}
        worst_dev = min(dev.values(), default=None)
        ok = worst is not None and worst >= bar and (
            worst_dev is None or worst_dev >= -COS_DEV_TOL)
        return {"ok": ok, "worst_cos": worst, "worst_layer": wl, "bar": bar,
                "n": len(vals), "failed": sorted(l for l, c in vals if c < bar),
                "deviation": dev, "worst_deviation": worst_dev,
                "dev_tol": COS_DEV_TOL}
    return {"ok": worst is not None and worst >= bar, "worst_cos": worst,
            "worst_layer": wl, "bar": bar, "n": len(vals),
            "failed": sorted(l for l, c in vals if c < bar)}


def nll_gate(mean: float, median: float, top1: float, bars=None, ref=None):
    bars = bars or NLL_BARS
    ref = ref or REF_NLL
    ok = (mean <= bars["mean_max"] and median <= bars["median_max"]
          and top1 >= bars["top1_min"])
    return {"ok": ok, "mean": mean, "median": median, "top1": top1,
            "bars": bars, "ref": ref}


def negctl_verdict(events, low_rung=None, high_rung=None):
    """``events``: [{"name", "expect_detected": bool, "detected": bool}].

    ``low_rung``/``high_rung`` are ladder-policy assertions: the quietest
    level must leave parity intact and the loudest must break it. Without the
    low assertion a harness that always reports a mismatch would "pass" the
    negative control.
    """
    ok = all(bool(e["detected"]) == bool(e["expect_detected"]) for e in events)
    extra = {}
    if low_rung is not None:
        ok = ok and (low_rung["detected"] is False)
        extra["low_rung_quiet"] = low_rung["detected"] is False
    if high_rung is not None:
        ok = ok and (high_rung["detected"] is True)
        extra["high_rung_detected"] = high_rung["detected"] is True
    out = {"ok": ok, "events": events,
           "n_detected": sum(1 for e in events if e["detected"])}
    out.update(extra)
    return out


def overall(checks: dict):
    """``checks``: {name: bool}. Returns (ok, one-line summary)."""
    ok = all(bool(v) for v in checks.values())
    return ok, " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in checks.items())


# --------------------------------------------------------------------------
# reference files
# --------------------------------------------------------------------------

def make_fingerprint(cfg: dict, prompts) -> dict:
    """Structural identity + code digest of one run."""
    fp = {k: cfg.get(k) for k in FP_STRUCT_FIELDS}
    fp["prompts_digest"] = digest_str_list([p["ids"] for p in prompts])
    fp["structural"] = sha256_text(canonical(fp))[:32]
    fp["pkg_digest"] = cfg.get("pkg_digest", "")
    return fp


def fingerprint_mismatch(ref_fp: dict, cur_fp: dict):
    """(structural_diff, digest_diff) -- structural is a hard fail."""
    struct = {k: (ref_fp.get(k), cur_fp.get(k))
              for k in FP_STRUCT_FIELDS if ref_fp.get(k) != cur_fp.get(k)}
    return struct, ref_fp.get("pkg_digest") != cur_fp.get("pkg_digest")


def save_reference(path: str, cfg: dict, fingerprint: dict, prompts,
                   note: str = "", overwrite: bool = False) -> None:
    """Write a reference JSON. Refuses to replace an existing file.

    References are comparison baselines: silently overwriting one makes every
    future check vacuous. Pass ``overwrite=True`` (the runner's
    ``--overwrite-ref``) when that is genuinely intended.
    """
    doc = {
        "schema": SCHEMA,
        "created": cfg.get("created", ""),
        "note": note,
        "config": {k: v for k, v in cfg.items() if k != "created"},
        "fingerprint": fingerprint,
        "prompts": prompts,
        "timings": cfg.get("timings", {}),
        "memory": cfg.get("memory", {}),
        "nll": cfg.get("nll"),
    }
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"{path} exists; references are baselines and are not overwritten "
            f"silently (pass --overwrite-ref to replace it, or pick another --ref)")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def load_reference(path: str) -> dict:
    with open(path) as f:
        doc = json.load(f)
    got = doc.get("schema")
    if got != SCHEMA:
        raise ValueError(f"{path}: schema {got!r} != {SCHEMA!r}")
    return doc
