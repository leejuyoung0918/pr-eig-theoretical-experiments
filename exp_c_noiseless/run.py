"""Vectorized, checkpointed noiseless box-localization Monte Carlo."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from figure2_experiments.config import ExperimentConfig, get_config
from figure2_experiments.utils import atomic_json, atomic_npz, ensure_output_tree, stable_hash


POLICIES = ("RC-MLQ", "Random Coordinate", "Entropy-EIG Uniform Tie")


def rc_theory(t: np.ndarray | int, d: int, m: int, length: float = 1.0) -> np.ndarray:
    t_arr = np.asarray(t, dtype=int)
    k, remainder = np.divmod(t_arr, d)
    return length / np.sqrt(12.0) * np.sqrt(
        (d - remainder) * m ** (-2.0 * k) + remainder * m ** (-2.0 * (k + 1))
    )


def random_theory(t: np.ndarray | int, d: int, m: int, length: float = 1.0) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    return length * np.sqrt(d / 12.0) * (
        1.0 - (1.0 - m ** -2.0) / d
    ) ** (t_arr / 2.0)


def lower_bound(t: np.ndarray | int, d: int, m: int, length: float = 1.0) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    return length * np.sqrt(d / (2.0 * np.pi * np.e)) * m ** (-t_arr / d)


def _update_coordinate(
    target: np.ndarray,
    lower: np.ndarray,
    width: np.ndarray,
    coordinate_squared_error: np.ndarray,
    coordinate: int | np.ndarray,
    m: int,
) -> None:
    rows = np.arange(len(target))
    if np.isscalar(coordinate):
        columns = np.full(len(target), int(coordinate), dtype=int)
    else:
        columns = np.asarray(coordinate, dtype=int)
    old_width = width[rows, columns]
    old_lower = lower[rows, columns]
    relative = (target[rows, columns] - old_lower) / old_width
    cell = np.floor(relative * m).astype(np.int64)
    np.clip(cell, 0, m - 1, out=cell)
    new_width = old_width / m
    new_lower = old_lower + cell * new_width
    new_center = new_lower + 0.5 * new_width
    coordinate_squared_error[rows, columns] = (new_center - target[rows, columns]) ** 2
    lower[rows, columns] = new_lower
    width[rows, columns] = new_width


def simulate_macro(seed: int, d: int, m: int, n_targets: int, t_max: int) -> np.ndarray:
    """Return per-policy sums of squared Euclidean error at every T."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 303, d, m]))
    target = rng.uniform(-0.5, 0.5, size=(n_targets, d))
    sums = np.empty((len(POLICIES), t_max + 1), dtype=float)
    states = []
    initial_coordinate_error = target * target
    for policy_index in range(len(POLICIES)):
        states.append(
            [
                np.full((n_targets, d), -0.5, dtype=float),
                np.ones((n_targets, d), dtype=float),
                initial_coordinate_error.copy(),
            ]
        )
        sums[policy_index, 0] = float(initial_coordinate_error.sum(dtype=np.float64))
    for t in range(1, t_max + 1):
        rc_coordinate = (t - 1) % d
        _update_coordinate(target, *states[0], rc_coordinate, m)
        random_coordinate = rng.integers(0, d, size=n_targets)
        _update_coordinate(target, *states[1], random_coordinate, m)
        entropy_coordinate = rng.integers(0, d, size=n_targets)
        _update_coordinate(target, *states[2], entropy_coordinate, m)
        for policy_index in range(len(POLICIES)):
            # Re-summing nonnegative coordinate errors avoids catastrophic
            # cancellation after many M-way refinements.
            sums[policy_index, t] = float(states[policy_index][2].sum(dtype=np.float64))
    return sums


def _macro_task(
    seed: int,
    d: int,
    m: int,
    macro: int,
    n_targets: int,
    t_max: int,
    path: str,
) -> str:
    output = Path(path)
    if output.exists():
        return str(output)
    sums = simulate_macro(seed + macro, d, m, n_targets, t_max)
    atomic_npz(
        output,
        d=np.array(d),
        m=np.array(m),
        macro=np.array(macro),
        seed=np.array(seed + macro),
        n_targets=np.array(n_targets),
        sum_squared_error=sums,
        policies=np.array(POLICIES),
    )
    return str(output)


def run(config: ExperimentConfig, workers: int | None = None, force: bool = False) -> dict[str, Path]:
    ensure_output_tree(config.output_root)
    raw_dir = config.output_root / "exp_c" / "raw" / "macros"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tasks = []
    for d in config.c_dimensions:
        for m in config.c_m_values:
            t_max = config.c_rounds_per_dimension * d
            for macro in range(config.profile.c_macro_seeds):
                key = stable_hash(
                    {
                        "mode": config.mode,
                        "d": d,
                        "m": m,
                        "macro": macro,
                        "targets": config.profile.c_targets_per_macro,
                        "seed": config.seed,
                    }
                )[:10]
                path = raw_dir / f"d{d}_m{m}_macro{macro:03d}_{key}.npz"
                if force and path.exists():
                    path.unlink()
                tasks.append((d, m, macro, t_max, path))
    workers = workers or min(max(1, (os.cpu_count() or 2) - 1), 8)
    def run_sequentially() -> None:
        for index, (d, m, macro, t_max, path) in enumerate(tasks, 1):
            _macro_task(
                config.seed, d, m, macro, config.profile.c_targets_per_macro, t_max, str(path)
            )
            print(f"[C macros] {index}/{len(tasks)}", flush=True)
    if workers == 1:
        run_sequentially()
    else:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _macro_task,
                        config.seed,
                        d,
                        m,
                        macro,
                        config.profile.c_targets_per_macro,
                        t_max,
                        str(path),
                    )
                    for d, m, macro, t_max, path in tasks
                ]
                for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    future.result()
                    print(f"[C macros] {index}/{len(tasks)}", flush=True)
        except (PermissionError, NotImplementedError, OSError) as exc:
            print(f"[C] multiprocessing unavailable ({exc!r}); using deterministic sequential fallback", flush=True)
            workers = 1
            run_sequentially()

    records = []
    for d, m, macro, t_max, path in tasks:
        with np.load(path) as data:
            sums = data["sum_squared_error"]
            n_targets = int(data["n_targets"])
            for policy_index, policy in enumerate(POLICIES):
                for t in range(t_max + 1):
                    records.append(
                        {
                            "d": d,
                            "M": m,
                            "macro": macro,
                            "seed": int(data["seed"]),
                            "n_targets": n_targets,
                            "policy": policy,
                            "T": t,
                            "mean_squared_error": sums[policy_index, t] / n_targets,
                        }
                    )
    raw = pd.DataFrame.from_records(records)
    raw_path = config.output_root / "exp_c" / "raw" / "macro_summary.parquet"
    raw.to_parquet(raw_path, index=False)
    atomic_json(
        config.output_root / "exp_c" / "metadata.json",
        {
            "experiment": "C",
            "mode": config.mode,
            "dimensions": config.c_dimensions,
            "M_values": config.c_m_values,
            "T_range": "0..8d",
            "macro_seeds": config.profile.c_macro_seeds,
            "targets_per_macro": config.profile.c_targets_per_macro,
            "targets_per_setting": config.profile.c_macro_seeds * config.profile.c_targets_per_macro,
            "workers": workers,
            "elapsed_seconds": time.monotonic() - started,
            "config_hash": config.config_hash,
        },
    )
    return {"raw": raw_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "paper", "exhaustive"), default="paper")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run(get_config(args.mode), workers=args.workers, force=args.force)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
