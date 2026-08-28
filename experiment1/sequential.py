"""Long-horizon scalar Gaussian-ADF experiment for Figure 2(b)."""

from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from figure2_experiments.config import get_config
from figure2_experiments.exp_b_sequential.design_cache import (
    METHODS,
    DesignCache,
    build_cache,
)
from figure2_experiments.exp_b_sequential.run import _method_key, _trajectory_randomness
from figure2_experiments.mnl import mnl_probs
from figure2_experiments.templates import quantile_template
from figure2_experiments.utils import atomic_json, atomic_npz, inverse_cdf_rows, stable_hash

from .config import Experiment1Config
from .statistics import (
    bootstrap_rms_from_variances,
    fixed_exponent_intercept,
    log_linear_regression,
)


def sequential_checkpoints(n_max: int, count: int) -> np.ndarray:
    values = np.unique(np.rint(np.logspace(0.0, np.log10(n_max), count)).astype(int))
    return np.unique(np.concatenate((np.array([1, 2, 4, 8, n_max]), values)))


def _batch_file(config: Experiment1Config, start: int, stop: int) -> Path:
    key = stable_hash(
        {
            "config": config.config_hash,
            "start": start,
            "stop": stop,
            "n_max": config.profile.b_n_max,
            "checkpoints": sequential_checkpoints(
                config.profile.b_n_max, config.profile.b_checkpoint_count
            ).tolist(),
        }
    )[:16]
    return config.output_root / "sequential" / "raw" / "batches" / f"batch_{start:04d}_{stop:04d}_{key}.npz"


def _simulate_batch(
    config: Experiment1Config,
    cache_file: str,
    start: int,
    stop: int,
    output_file: str,
) -> str:
    output = Path(output_file)
    if output.exists():
        return str(output)
    cache = DesignCache.load(Path(cache_file))
    trajectory_ids = np.arange(start, stop, dtype=np.int32)
    targets, uniforms = _trajectory_randomness(
        config.seed, trajectory_ids, config.profile.b_n_max
    )
    checkpoints = sequential_checkpoints(
        config.profile.b_n_max, config.profile.b_checkpoint_count
    )
    checkpoint_index = {int(value): index for index, value in enumerate(checkpoints)}
    template = quantile_template(config.b_m)
    arrays: dict[str, np.ndarray] = {
        "trajectory": trajectory_ids,
        "true_target": targets,
        "checkpoints": checkpoints,
    }
    started = time.perf_counter()
    for method in METHODS:
        key = _method_key(method)
        arrays[f"{key}__posterior_variance"] = np.full(
            (len(trajectory_ids), len(checkpoints)), np.nan
        )
        arrays[f"{key}__posterior_mean"] = np.full(
            (len(trajectory_ids), len(checkpoints)), np.nan
        )
        arrays[f"{key}__selected_r"] = np.full(
            (len(trajectory_ids), len(checkpoints)), np.nan
        )
        mean = np.zeros(len(trajectory_ids), dtype=float)
        sigma = np.ones(len(trajectory_ids), dtype=float)
        for n in range(1, config.profile.b_n_max + 1):
            transition = cache.lookup(method, sigma)
            radius = transition["r"]
            probabilities = mnl_probs(targets - mean, radius, template, config.b_beta)
            response = inverse_cdf_rows(probabilities, uniforms[:, n - 1])
            rows = np.arange(len(trajectory_ids))
            old_sigma = sigma.copy()
            mean += old_sigma * transition["mean_shift_sigma"][rows, response]
            sigma = old_sigma * np.sqrt(
                np.maximum(transition["variance_ratio"][rows, response], 1e-15)
            )
            if n in checkpoint_index:
                column = checkpoint_index[n]
                arrays[f"{key}__posterior_variance"][:, column] = sigma * sigma
                arrays[f"{key}__posterior_mean"][:, column] = mean
                arrays[f"{key}__selected_r"][:, column] = radius
        print(
            f"[Experiment1 B {start}:{stop}] {method} n={config.profile.b_n_max}",
            flush=True,
        )
    atomic_npz(output, elapsed_seconds=np.array(time.perf_counter() - started), **arrays)
    return str(output)


def run_sequential(
    config: Experiment1Config,
    *,
    workers: int | None = None,
    force: bool = False,
) -> dict[str, Path]:
    root = config.output_root / "sequential"
    batch_dir = root / "raw" / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    # Reuse the already validated paper cache.  It contains only model/design
    # integrals and does not depend on the old n=8192 horizon.
    source_config = get_config("paper")
    cache_path = build_cache(source_config, force=False)
    tasks = []
    for start in range(0, config.profile.b_repetitions, config.profile.b_batch_size):
        stop = min(start + config.profile.b_batch_size, config.profile.b_repetitions)
        path = _batch_file(config, start, stop)
        if force and path.exists():
            path.unlink()
        tasks.append((start, stop, path))
    workers = workers or min(max((os.cpu_count() or 2) - 1, 1), 8)
    started = time.perf_counter()

    def sequential_fallback() -> None:
        for start, stop, path in tasks:
            _simulate_batch(config, str(cache_path), start, stop, str(path))

    if workers == 1:
        sequential_fallback()
    else:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_simulate_batch, config, str(cache_path), start, stop, str(path))
                    for start, stop, path in tasks
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        except (PermissionError, NotImplementedError, OSError) as exc:
            print(f"[Experiment1 B] multiprocessing unavailable ({exc!r}); sequential fallback", flush=True)
            workers = 1
            sequential_fallback()

    records = []
    for _, _, path in tasks:
        with np.load(path) as data:
            checkpoints = data["checkpoints"]
            for method in METHODS:
                key = _method_key(method)
                variance = data[f"{key}__posterior_variance"]
                mean = data[f"{key}__posterior_mean"]
                radius = data[f"{key}__selected_r"]
                for row, trajectory in enumerate(data["trajectory"]):
                    for column, n in enumerate(checkpoints):
                        records.append(
                            {
                                "trajectory": int(trajectory),
                                "method": method,
                                "n": int(n),
                                "true_target": float(data["true_target"][row]),
                                "posterior_mean": float(mean[row, column]),
                                "posterior_variance": float(variance[row, column]),
                                "directional_rms": float(np.sqrt(variance[row, column])),
                                "selected_r": float(radius[row, column]),
                            }
                        )
    raw = pd.DataFrame.from_records(records)
    raw_path = root / "raw" / "trials.parquet"
    raw.to_parquet(raw_path, index=False)
    metadata_path = root / "run_metadata.json"
    atomic_json(
        metadata_path,
        {
            "mode": config.mode,
            "repetitions": config.profile.b_repetitions,
            "N_max": config.profile.b_n_max,
            "checkpoints": sequential_checkpoints(
                config.profile.b_n_max, config.profile.b_checkpoint_count
            ),
            "posterior_method": "Gaussian ADF with exact MNL Gauss-Hermite moment matching",
            "quadrature_order": source_config.profile.b_gh_order,
            "common_random_numbers": True,
            "cache_path": str(cache_path),
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "config_hash": config.config_hash,
        },
    )
    return {"trials": raw_path, "metadata": metadata_path, "cache": cache_path}


def _bootstrap_slopes(
    n: np.ndarray,
    bootstrap_rms: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    x = np.log(n[mask])
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    return np.sum(centered[None, :] * np.log(bootstrap_rms[:, mask]), axis=1) / denominator


def analyze_sequential(config: Experiment1Config) -> dict[str, Path]:
    root = config.output_root / "sequential"
    raw = pd.read_parquet(root / "raw" / "trials.parquet")
    curve_records = []
    slope_records = []
    guide_records = []
    bootstrap_arrays: dict[str, np.ndarray] = {}
    for method_index, method in enumerate(METHODS):
        group = raw[raw["method"] == method]
        pivot = group.pivot(index="trajectory", columns="n", values="posterior_variance").sort_index(axis=1)
        n = pivot.columns.to_numpy(float)
        variance = pivot.to_numpy(float)
        rms, ci_low, ci_high = bootstrap_rms_from_variances(
            variance,
            config.profile.b_bootstrap_samples,
            seed=config.seed + 5_003 + method_index,
        )
        rng = np.random.default_rng(config.seed + 6_007 + method_index)
        counts = rng.multinomial(
            len(variance), np.full(len(variance), 1.0 / len(variance)),
            size=config.profile.b_bootstrap_samples,
        )
        bootstrap_rms = np.sqrt(counts @ variance / len(variance))
        for index, query_count in enumerate(n.astype(int)):
            curve_records.append(
                {
                    "method": method,
                    "n": query_count,
                    "epsilon": rms[index],
                    "ci_low": ci_low[index],
                    "ci_high": ci_high[index],
                    "mean_variance": variance[:, index].mean(),
                    "repetitions": len(variance),
                }
            )
        for lower in config.b_fit_starts:
            mask = (n >= lower) & (n <= config.profile.b_n_max)
            if mask.sum() < 3:
                continue
            fit = log_linear_regression(n[mask], rms[mask])
            samples = _bootstrap_slopes(n, bootstrap_rms, mask)
            bootstrap_arrays[f"{method}__{lower}"] = samples
            slope_ci = np.quantile(samples, (0.025, 0.975))
            slope_records.append(
                {
                    "method": method,
                    "fit_window_start": lower,
                    "fit_window_end": config.profile.b_n_max,
                    "primary": lower == config.b_primary_fit_start,
                    "slope": fit["slope"],
                    "standard_error": fit["slope_se"],
                    "regression_ci_low": fit["ci_low"],
                    "regression_ci_high": fit["ci_high"],
                    "bootstrap_slope_ci_low": slope_ci[0],
                    "bootstrap_slope_ci_high": slope_ci[1],
                    "R2": fit["r2"],
                    "n_points": fit["n_points"],
                    "final_RMS": rms[-1],
                    "final_bootstrap_ci_low": ci_low[-1],
                    "final_bootstrap_ci_high": ci_high[-1],
                    "theory_slope": -0.25 if method == "PR-EIG" else -0.5,
                }
            )
        exponent = -0.25 if method == "PR-EIG" else -0.5
        primary_mask = (n >= min(config.b_primary_fit_start, max(1, config.profile.b_n_max // 3)))
        intercept = fixed_exponent_intercept(n[primary_mask], rms[primary_mask], exponent)
        for query_count in n:
            guide_records.append(
                {
                    "method": method,
                    "n": int(query_count),
                    "exponent": exponent,
                    "intercept": intercept,
                    "epsilon": float(np.exp(intercept) * query_count**exponent),
                }
            )
    curve = pd.DataFrame.from_records(curve_records)
    slopes = pd.DataFrame.from_records(slope_records)
    guides = pd.DataFrame.from_records(guide_records)
    curve_path = root / "curve.csv"
    slope_path = root / "slopes.csv"
    guide_path = root / "guides.csv"
    curve.to_csv(curve_path, index=False)
    slopes.to_csv(slope_path, index=False)
    (root / "slopes.tex").write_text(
        slopes.to_latex(index=False, float_format="%.6g"), encoding="utf-8"
    )
    guides.to_csv(guide_path, index=False)
    np.savez_compressed(root / "bootstrap_slope_samples.npz", **bootstrap_arrays)

    local_records = []
    for method, group in curve.groupby("method"):
        group = group.sort_values("n")
        local = np.gradient(np.log(group["epsilon"]), np.log(group["n"]))
        for n_value, slope in zip(group["n"], local):
            local_records.append({"method": method, "n": int(n_value), "local_slope": slope})
    local_path = root / "local_slopes.csv"
    pd.DataFrame.from_records(local_records).to_csv(local_path, index=False)
    return {"curve": curve_path, "slopes": slope_path, "guides": guide_path, "local": local_path}
