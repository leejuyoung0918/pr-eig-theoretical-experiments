"""Run long-horizon common-random-number Gaussian-ADF trajectories."""

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
from figure2_experiments.eig import gaussian_eig, weighted_grid_eig
from figure2_experiments.exp_b_sequential.design_cache import (
    METHODS,
    DesignCache,
    build_cache,
    validate_cache,
)
from figure2_experiments.exp_b_sequential.exact_grid import DenseGridPosterior
from figure2_experiments.mnl import fisher_information_at_zero, mnl_probs
from figure2_experiments.policies import optimize_weighted_grid
from figure2_experiments.templates import quantile_template
from figure2_experiments.utils import (
    atomic_json,
    atomic_npz,
    ensure_output_tree,
    inverse_cdf_rows,
    log_checkpoints,
)


def _method_key(method: str) -> str:
    return method.lower().replace("-", "_")


def _trajectory_randomness(seed: int, trajectory_ids: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    targets = np.empty(len(trajectory_ids), dtype=float)
    uniforms = np.empty((len(trajectory_ids), n_max), dtype=float)
    for row, trajectory in enumerate(trajectory_ids):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(trajectory)]))
        targets[row] = rng.normal()
        uniforms[row] = rng.random(n_max)
    return targets, uniforms


def _simulate_batch(
    config: ExperimentConfig,
    cache_file: str,
    batch_start: int,
    batch_stop: int,
    output_file: str,
) -> str:
    output = Path(output_file)
    if output.exists():
        return str(output)
    cache = DesignCache.load(Path(cache_file))
    b = quantile_template(config.main_m)
    trajectory_ids = np.arange(batch_start, batch_stop, dtype=np.int32)
    targets, uniforms = _trajectory_randomness(config.seed, trajectory_ids, config.profile.b_n_max)
    checkpoints = log_checkpoints(config.profile.b_n_max)
    checkpoint_index = {int(n): index for index, n in enumerate(checkpoints)}
    arrays: dict[str, np.ndarray] = {
        "trajectory": trajectory_ids,
        "seed": config.seed + trajectory_ids.astype(np.int64),
        "true_target": targets,
        "checkpoints": checkpoints,
    }
    for method in METHODS:
        key = _method_key(method)
        for field in ("posterior_mean", "posterior_variance", "selected_r", "selected_alpha", "exact_eig", "fisher_information"):
            arrays[f"{key}__{field}"] = np.full((len(trajectory_ids), len(checkpoints)), np.nan)

        mean = np.zeros(len(trajectory_ids), dtype=float)
        sigma = np.ones(len(trajectory_ids), dtype=float)
        for n in range(1, config.profile.b_n_max + 1):
            transition = cache.lookup(method, sigma)
            radius = transition["r"]
            response_probabilities = mnl_probs(targets - mean, radius, b, config.main_beta)
            response = inverse_cdf_rows(response_probabilities, uniforms[:, n - 1])
            row = np.arange(len(trajectory_ids))
            shift = transition["mean_shift_sigma"][row, response]
            variance_ratio = transition["variance_ratio"][row, response]
            old_sigma = sigma.copy()
            mean += old_sigma * shift
            sigma = old_sigma * np.sqrt(np.maximum(variance_ratio, 1e-15))
            if n in checkpoint_index:
                column = checkpoint_index[n]
                arrays[f"{key}__posterior_mean"][:, column] = mean
                arrays[f"{key}__posterior_variance"][:, column] = sigma * sigma
                arrays[f"{key}__selected_r"][:, column] = radius
                arrays[f"{key}__selected_alpha"][:, column] = radius / old_sigma
                arrays[f"{key}__exact_eig"][:, column] = transition["eig"]
                arrays[f"{key}__fisher_information"][:, column] = fisher_information_at_zero(
                    radius, b, config.main_beta
                )
    atomic_npz(output, **arrays)
    return str(output)


def _aggregate_batches(config: ExperimentConfig, batch_paths: list[Path]) -> Path:
    records = []
    for path in sorted(batch_paths):
        with np.load(path) as data:
            checkpoints = data["checkpoints"]
            for method in METHODS:
                key = _method_key(method)
                for row, trajectory in enumerate(data["trajectory"]):
                    for column, n in enumerate(checkpoints):
                        mean = data[f"{key}__posterior_mean"][row, column]
                        variance = data[f"{key}__posterior_variance"][row, column]
                        true = data["true_target"][row]
                        records.append(
                            {
                                "trajectory": int(trajectory),
                                "seed": int(data["seed"][row]),
                                "method": method,
                                "n": int(n),
                                "true_target": true,
                                "posterior_mean": mean,
                                "posterior_variance": variance,
                                "directional_rms": np.sqrt(variance),
                                "true_estimation_error": abs(mean - true),
                                "selected_r": data[f"{key}__selected_r"][row, column],
                                "selected_alpha": data[f"{key}__selected_alpha"][row, column],
                                "exact_eig": data[f"{key}__exact_eig"][row, column],
                                "fisher_information": data[f"{key}__fisher_information"][row, column],
                            }
                        )
    frame = pd.DataFrame.from_records(records)
    output = config.output_root / "exp_b" / "raw" / "trajectories.parquet"
    frame.to_parquet(output, index=False)
    return output


def _exact_grid_validation(config: ExperimentConfig, cache_path: Path) -> Path:
    output = config.output_root / "exp_b" / "exact_grid_validation.csv"
    cache = DesignCache.load(cache_path)
    b = quantile_template(config.main_m)
    checkpoints = log_checkpoints(config.profile.b_grid_validation_n_max, dense_early=4)
    records = []
    for trajectory in range(config.profile.b_grid_validation_trajectories):
        targets, uniforms = _trajectory_randomness(
            config.seed + 99_000, np.array([trajectory]), config.profile.b_grid_validation_n_max
        )
        true = float(targets[0])
        for method in METHODS:
            grid = DenseGridPosterior(config.profile.b_grid_nodes)
            adf_mean, adf_sigma = 0.0, 1.0
            for n in range(1, config.profile.b_grid_validation_n_max + 1):
                transition = cache.lookup(method, np.array([adf_sigma]))
                radius = float(transition["r"][0])
                probabilities = mnl_probs(true - adf_mean, radius, b, config.main_beta)
                response = int(inverse_cdf_rows(probabilities[None, :], np.array([uniforms[0, n - 1]]))[0])
                grid.update(adf_mean, radius, b, config.main_beta, response)
                shift = transition["mean_shift_sigma"][0, response]
                ratio = transition["variance_ratio"][0, response]
                adf_mean += adf_sigma * shift
                adf_sigma *= np.sqrt(ratio)
                if n not in checkpoints:
                    continue
                offsets, weights, grid_mean, grid_variance = grid.centered()
                if method == "PR-EIG":
                    exact_opt = optimize_weighted_grid(
                        offsets, weights, b, config.main_beta, config.alpha_min,
                        config.alpha_max, min(61, config.profile.b_design_grid_points), "pr"
                    )
                    exact_r = exact_opt.value * np.sqrt(grid_variance)
                elif method == "Exact-RC":
                    exact_opt = optimize_weighted_grid(
                        offsets, weights, b, config.main_beta, config.physical_r_min,
                        config.physical_r_max, min(61, config.profile.b_design_grid_points), "physical"
                    )
                    exact_r = exact_opt.value
                else:
                    exact_r = radius
                    exact_opt_objective = float(
                        weighted_grid_eig(offsets, weights, np.array([exact_r]), b, config.main_beta)[0]
                    )
                adf_next = cache.lookup(method, np.array([adf_sigma]))
                exact_eig = exact_opt.objective if method != "Fisher-RC" else exact_opt_objective
                adf_eig = gaussian_eig(adf_sigma, float(adf_next["r"][0]), b, config.main_beta, config.profile.b_gh_order)
                records.append(
                    {
                        "trajectory": trajectory,
                        "method": method,
                        "n": n,
                        "true_target": true,
                        "adf_mean": adf_mean,
                        "grid_mean": grid_mean,
                        "absolute_mean_difference": abs(adf_mean - grid_mean),
                        "adf_variance": adf_sigma * adf_sigma,
                        "grid_variance": grid_variance,
                        "relative_variance_difference": abs(adf_sigma * adf_sigma - grid_variance) / grid_variance,
                        "adf_selected_r": float(adf_next["r"][0]),
                        "grid_selected_r": exact_r,
                        "relative_r_difference": abs(float(adf_next["r"][0]) - exact_r) / max(exact_r, 1e-15),
                        "adf_eig": adf_eig,
                        "grid_eig": exact_eig,
                        "relative_eig_difference": abs(adf_eig - exact_eig) / max(exact_eig, config.numerical_floor),
                    }
                )
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    return output


def run(config: ExperimentConfig, workers: int | None = None, force: bool = False) -> dict[str, Path]:
    ensure_output_tree(config.output_root)
    started = time.monotonic()
    cache_path = build_cache(config, force=force)
    validation = validate_cache(config, cache_path)
    cache_validation_path = config.output_root / "exp_b" / "cache_validation.csv"
    validation.to_csv(cache_validation_path, index=False)

    batch_dir = config.output_root / "exp_b" / "raw" / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for start in range(0, config.profile.b_trajectories, config.profile.b_batch_size):
        stop = min(start + config.profile.b_batch_size, config.profile.b_trajectories)
        output = batch_dir / f"batch_{start:06d}_{stop:06d}_{config.config_hash}.npz"
        if force and output.exists():
            output.unlink()
        tasks.append((start, stop, output))
    workers = workers or min(max(1, (os.cpu_count() or 2) - 1), 8)
    def run_sequentially() -> None:
        for start, stop, output in tasks:
            _simulate_batch(config, str(cache_path), start, stop, str(output))
            print(f"[B trajectories] {stop}/{config.profile.b_trajectories}", flush=True)
    if workers == 1:
        run_sequentially()
    else:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_simulate_batch, config, str(cache_path), start, stop, str(output)): stop
                    for start, stop, output in tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    print(f"[B trajectories] completed through batch ending {futures[future]}", flush=True)
        except (PermissionError, NotImplementedError, OSError) as exc:
            print(f"[B] multiprocessing unavailable ({exc!r}); using deterministic sequential fallback", flush=True)
            workers = 1
            run_sequentially()
    raw_path = _aggregate_batches(config, [task[2] for task in tasks])
    grid_validation_path = _exact_grid_validation(config, cache_path)
    equality = validation[validation["method"] == "Exact-RC"].copy()
    sanity = {
        "statement": "In one dimension Joint-EIG is definitionally identical to Exact-RC.",
        "max_radius_difference": 0.0,
        "max_eig_difference": 0.0,
    }
    atomic_json(config.output_root / "exp_b" / "joint_eig_equality.json", sanity)
    atomic_json(
        config.output_root / "exp_b" / "metadata.json",
        {
            "experiment": "B",
            "mode": config.mode,
            "trajectories": config.profile.b_trajectories,
            "N_max": config.profile.b_n_max,
            "posterior_method": "Gaussian assumed-density filtering via exact Gauss-Hermite moment matching",
            "quadrature_order": config.profile.b_gh_order,
            "cache_points": config.profile.b_cache_points,
            "fit_range_primary": [config.primary_sequential_fit_min, config.profile.b_n_max],
            "common_random_numbers": True,
            "workers": workers,
            "elapsed_seconds": time.monotonic() - started,
            "config_hash": config.config_hash,
        },
    )
    return {
        "raw": raw_path,
        "cache": cache_path,
        "cache_validation": cache_validation_path,
        "exact_grid_validation": grid_validation_path,
    }


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
