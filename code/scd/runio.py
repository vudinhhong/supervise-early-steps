"""Result files.

Every experiment writes exactly one JSON blob, and every number in the paper is
generated from those blobs.  Nothing in the manuscript is typed by hand, so a
rerun cannot silently disagree with the text.

The output directory is chosen per stage by the ``SCD_RESULTS`` environment
variable, which ``run.sh`` sets.  Stages whose file names cannot collide share
``results/``; stages that write one file per grid point get their own
``results-<x>/``.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent


def results_dir() -> pathlib.Path:
    d = ROOT / os.environ.get("SCD_RESULTS", "results")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"


def _code_sha() -> str:
    """A hash over the sources that produce the numbers.

    The cluster runs outside git, where ``_git()`` returns "n/a", so nothing in a
    blob said which revision produced it.  That question came up and could not be
    answered: a probe blob recorded ``recovery = nan`` and no one could show
    whether it had run with the adapter seeded, because the file had changed since.
    A hash in every blob settles it, and it is cheap -- a few dozen sources.
    """
    import hashlib
    h = hashlib.sha256()
    for d in ("scd", "experiments"):
        for f in sorted((ROOT / d).glob("*.py")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


_SHA_CACHE: str | None = None


def code_sha() -> str:
    """``_code_sha()`` with a cache: it is called once per ledger record."""
    global _SHA_CACHE
    if _SHA_CACHE is None:
        try:
            _SHA_CACHE = _code_sha()
        except Exception:
            _SHA_CACHE = "n/a"
    return _SHA_CACHE


def _env() -> dict:
    out = {"platform": platform.platform(), "machine": platform.machine(),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": _git()}
    try:
        out["code_sha"] = _code_sha()
    except Exception as e:          # must never be what kills a run
        out["code_sha"] = "n/a (%s)" % type(e).__name__
    # Any deliberately bypassed prerequisite gate travels with the result.  run.sh
    # prints a banner, but a banner lives in a log file and the blob is what the
    # tables and the paper read; a deviation from a pre-registered stop condition
    # has to be visible wherever the numbers are.
    ov = {k: v for k, v in os.environ.items() if k.startswith("SCD_OVERRIDE_")}
    if ov:
        out["gate_overrides"] = ov
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            # Memory high-water marks go in every blob: the resource accounting the
            # protocol requires, and the evidence for or against the next VRAM
            # request. A stated budget that was never measured is how a hand-set
            # ceiling came to kill a run with two gigabytes spare.
            from . import hf
            out.update({k: round(v, 2) for k, v in hf.peak_report().items()})
    except Exception:
        pass
    return out


def save(name: str, payload: dict) -> pathlib.Path:
    path = results_dir() / f"{name}.json"
    payload = dict(payload)
    payload["_env"] = _env()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    tmp.replace(path)

    # Read the file back before returning.  One job exited zero after five hours,
    # its ledger written at the moment it finished, while the blob in the same
    # directory was eight days old and no newer copy existed anywhere on the
    # storage.  The cause is still unknown; what is certain is that it happened
    # without anyone noticing.  Reading back makes the next occurrence fail here,
    # with the absolute path.
    try:
        back = json.loads(path.read_text())
        if not isinstance(back, dict):
            raise ValueError("re-read did not return a dict")
    except Exception as e:
        raise RuntimeError(
            "wrote %s but could not read it back (%s). SCD_RESULTS=%r, "
            "directory exists=%s, writable=%s"
            % (path, e, os.environ.get("SCD_RESULTS"),
               path.parent.is_dir(), os.access(str(path.parent), os.W_OK)))
    # ``SCD_RESULTS`` may be an absolute path outside the tree -- a scheduler
    # handing each job its own output directory does exactly that -- and
    # ``relative_to`` raises rather than falling back, so ask first.
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"[runio] wrote {shown}  ({path.stat().st_size} B, {path})", flush=True)
    return path


def load(name: str, where: str | None = None) -> dict | None:
    """Load a result blob.  ``where`` overrides the stage directory."""
    base = ROOT / where if where else results_dir()
    path = base / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def exists(name: str) -> bool:
    return (results_dir() / f"{name}.json").exists()


def thresholds() -> dict:
    """Pre-registered gate thresholds.  Committed before any stage runs."""
    return json.loads((ROOT / "thresholds.json").read_text())
