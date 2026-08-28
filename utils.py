"""Small infrastructure helpers: hashing, atomic writes, metadata, and fitting."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_output_tree(root: Path) -> None:
    for rel in (
        "exp_a/raw",
        "exp_a/diagnostics",
        "exp_b/raw",
        "exp_b/diagnostics",
        "exp_c/raw",
        "figure2",
        "supplementary",
        "cache",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_metadata(config: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    return {
        "config": config,
        "config_hash": stable_hash(config)[:16],
        "created_unix": time.time(),
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "git_commit": git_commit(workspace),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }


def loglog_fit(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if valid.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n_points": int(valid.sum())}
    lx = np.log(x[valid])
    ly = np.log(y[valid])
    slope, intercept = np.polyfit(lx, ly, 1)
    fitted = intercept + slope * lx
    ss_res = float(np.sum((ly - fitted) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n_points": int(valid.sum())}


def log_checkpoints(n_max: int, dense_early: int = 16) -> np.ndarray:
    values = set(range(1, min(dense_early, n_max) + 1))
    power = 1
    while power <= n_max:
        values.add(power)
        power *= 2
    values.add(n_max)
    return np.array(sorted(values), dtype=int)


def inverse_cdf_rows(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(probabilities, axis=-1)
    cdf[..., -1] = 1.0
    return np.sum(uniforms[..., None] > cdf, axis=-1).astype(np.int16)

