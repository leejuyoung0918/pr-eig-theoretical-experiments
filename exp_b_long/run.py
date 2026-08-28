"""Resume the unchanged paper Experiment B from n=8192 through n=65536."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from figure2_experiments.config import ExperimentConfig, get_config
from figure2_experiments.eig import weighted_grid_eig
from figure2_experiments.exp_b_sequential.design_cache import METHODS, DesignCache, cache_path
from figure2_experiments.exp_b_sequential.exact_grid import DenseGridPosterior
from figure2_experiments.exp_b_sequential.run import (
    _method_key,
    _simulate_batch,
    _trajectory_randomness,
)
from figure2_experiments.mnl import fisher_information_at_zero, mnl_probs
from figure2_experiments.policies import optimize_weighted_grid
from figure2_experiments.templates import quantile_template
from figure2_experiments.utils import (
    atomic_json,
    atomic_npz,
    inverse_cdf_rows,
    loglog_fit,
    run_metadata,
    stable_hash,
)


SOURCE_N_MAX = 8_192
LONG_N_MAX = 65_536
STAGE_LIMITS = {1: 20, 2: 100, 3: 1_000}
REQUIRED_LONG_CHECKPOINTS = (8_192, 12_288, 16_384, 24_576, 32_768, 49_152, 65_536)
BOUNDARY_INTERVALS = ((8_193, 16_384), (16_385, 32_768), (32_769, 65_536))
NUMERICAL_VARIANCE_FLOOR = 1e-24
STAGE_SANITY_ATOL = 1e-12


def output_root(config: ExperimentConfig) -> Path:
    return config.output_root / "exp_b_long"


def long_checkpoints() -> np.ndarray:
    dense = np.rint(np.geomspace(SOURCE_N_MAX, LONG_N_MAX, 41)).astype(int)
    return np.unique(np.concatenate([dense, np.asarray(REQUIRED_LONG_CHECKPOINTS, dtype=int)]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths(config: ExperimentConfig) -> tuple[Path, Path]:
    raw = config.output_root / "exp_b" / "raw" / "trajectories.parquet"
    cache = cache_path(config)
    if not raw.exists():
        raise FileNotFoundError(f"source Experiment B raw output is missing: {raw}")
    if not cache.exists():
        raise FileNotFoundError(f"source Experiment B design cache is missing: {cache}")
    return raw, cache


def _original_point_slopes(raw: pd.DataFrame, lower: int = 512, upper: int = 8_192) -> dict[str, float]:
    results = {}
    for method in METHODS:
        group = raw[(raw["method"] == method) & (raw["n"].between(lower, upper))]
        curve = group.groupby("n", sort=True)["posterior_variance"].mean()
        fit = loglog_fit(curve.index.to_numpy(float), np.sqrt(curve.to_numpy(float)))
        results[method] = float(fit["slope"])
    return results


def reproduce_source(config: ExperimentConfig, force: bool = False) -> dict[str, Path]:
    """Gate the extension on an exact 20-trajectory replay through n=8192."""
    root = output_root(config)
    reproduction = root / "reproduction"
    reproduction.mkdir(parents=True, exist_ok=True)
    source_raw_path, source_cache_path = _source_paths(config)
    replay_path = reproduction / "first20_replay.npz"
    if force and replay_path.exists():
        replay_path.unlink()
    _simulate_batch(config, str(source_cache_path), 0, 20, str(replay_path))
    source = pd.read_parquet(source_raw_path)
    source = source[source["trajectory"] < 20].copy()
    comparisons: list[dict[str, float | int | str | bool]] = []
    with np.load(replay_path) as replay:
        checkpoints = replay["checkpoints"]
        for method in METHODS:
            key = _method_key(method)
            method_source = source[source["method"] == method]
            for field, source_field in (
                ("posterior_mean", "posterior_mean"),
                ("posterior_variance", "posterior_variance"),
                ("selected_r", "selected_r"),
                ("selected_alpha", "selected_alpha"),
                ("exact_eig", "exact_eig"),
                ("fisher_information", "fisher_information"),
            ):
                expected = (
                    method_source.pivot(index="trajectory", columns="n", values=source_field)
                    .loc[np.arange(20), checkpoints]
                    .to_numpy(float)
                )
                observed = replay[f"{key}__{field}"]
                difference = np.abs(observed - expected)
                scale = np.maximum(np.abs(expected), np.finfo(float).tiny)
                comparisons.append(
                    {
                        "method": method,
                        "field": field,
                        "max_absolute_difference": float(difference.max()),
                        "max_relative_difference": float((difference / scale).max()),
                        "bitwise_equal": bool(np.array_equal(observed, expected)),
                        "values_compared": int(observed.size),
                    }
                )
    comparison = pd.DataFrame.from_records(comparisons)
    comparison_path = reproduction / "checkpoint_reproduction.csv"
    comparison.to_csv(comparison_path, index=False)

    recomputed = _original_point_slopes(pd.read_parquet(source_raw_path))
    stored = pd.read_csv(config.output_root / "exp_b" / "slopes.csv")
    slope_rows = []
    for method in METHODS:
        stored_row = stored[(stored["method"] == method) & (stored["fit_min_n"] == 512)].iloc[0]
        slope_rows.append(
            {
                "method": method,
                "fit_range": "512-8192",
                "recomputed_point_slope": recomputed[method],
                "stored_point_slope": float(stored_row["point_slope"]),
                "absolute_difference": abs(recomputed[method] - float(stored_row["point_slope"])),
            }
        )
    slopes = pd.DataFrame.from_records(slope_rows)
    slopes_path = reproduction / "slope_reproduction.csv"
    slopes.to_csv(slopes_path, index=False)
    if comparison["max_absolute_difference"].max() > 1e-13 or slopes["absolute_difference"].max() > 1e-13:
        raise RuntimeError("n=8192 reproduction failed; refusing to start the extension")
    atomic_json(
        reproduction / "reproduction_status.json",
        {
            "passed": True,
            "trajectories_replayed": 20,
            "N_max": SOURCE_N_MAX,
            "all_fields_bitwise_equal": bool(comparison["bitwise_equal"].all()),
            "max_absolute_checkpoint_difference": float(comparison["max_absolute_difference"].max()),
            "max_slope_difference": float(slopes["absolute_difference"].max()),
        },
    )
    return {"checkpoints": comparison_path, "slopes": slopes_path}


def _record_checkpoint(
    arrays: dict[str, np.ndarray],
    key: str,
    column: int,
    mean: np.ndarray,
    sigma: np.ndarray,
    radius: np.ndarray,
    prequery_sigma: np.ndarray,
    eig: np.ndarray,
    b: np.ndarray,
    beta: float,
) -> None:
    arrays[f"{key}__posterior_mean"][:, column] = mean
    arrays[f"{key}__posterior_variance"][:, column] = sigma * sigma
    arrays[f"{key}__selected_r"][:, column] = radius
    arrays[f"{key}__selected_alpha"][:, column] = radius / prequery_sigma
    arrays[f"{key}__exact_eig"][:, column] = eig
    arrays[f"{key}__fisher_information"][:, column] = fisher_information_at_zero(radius, b, beta)


def _simulate_extension_batch(
    config: ExperimentConfig,
    source_raw_path: str,
    cache_file: str,
    batch_start: int,
    batch_stop: int,
    output_file: str,
) -> str:
    output = Path(output_file)
    if output.exists():
        return str(output)
    cache = DesignCache.load(Path(cache_file))
    source = pd.read_parquet(source_raw_path)
    source = source[(source["n"] == SOURCE_N_MAX) & source["trajectory"].between(batch_start, batch_stop - 1)]
    ids = np.arange(batch_start, batch_stop, dtype=np.int32)
    targets, uniforms = _trajectory_randomness(config.seed, ids, LONG_N_MAX)
    source_targets = (
        source[source["method"] == METHODS[0]].set_index("trajectory").loc[ids, "true_target"].to_numpy(float)
    )
    if not np.array_equal(targets, source_targets):
        raise RuntimeError("target replay differs from the source n=8192 experiment")

    checkpoints = long_checkpoints()
    checkpoint_index = {int(n): i for i, n in enumerate(checkpoints)}
    b = quantile_template(config.main_m)
    arrays: dict[str, np.ndarray] = {
        "trajectory": ids,
        "seed": config.seed + ids.astype(np.int64),
        "true_target": targets,
        "checkpoints": checkpoints,
        "pr_boundary_hits": np.zeros((len(ids), len(BOUNDARY_INTERVALS)), dtype=np.int32),
        "pr_interval_rounds": np.zeros(len(BOUNDARY_INTERVALS), dtype=np.int32),
    }
    numerical_counts: dict[str, int] = {}
    for method in METHODS:
        key = _method_key(method)
        for field in (
            "posterior_mean",
            "posterior_variance",
            "selected_r",
            "selected_alpha",
            "exact_eig",
            "fisher_information",
        ):
            arrays[f"{key}__{field}"] = np.full((len(ids), len(checkpoints)), np.nan)
        initial = source[source["method"] == method].set_index("trajectory").loc[ids]
        mean = initial["posterior_mean"].to_numpy(float)
        sigma = np.sqrt(initial["posterior_variance"].to_numpy(float))
        initial_radius = initial["selected_r"].to_numpy(float)
        arrays[f"{key}__posterior_mean"][:, 0] = mean
        arrays[f"{key}__posterior_variance"][:, 0] = sigma * sigma
        arrays[f"{key}__selected_r"][:, 0] = initial_radius
        arrays[f"{key}__selected_alpha"][:, 0] = initial["selected_alpha"].to_numpy(float)
        arrays[f"{key}__exact_eig"][:, 0] = initial["exact_eig"].to_numpy(float)
        arrays[f"{key}__fisher_information"][:, 0] = initial["fisher_information"].to_numpy(float)
        floor_count = 0
        cache_clip_count = 0
        probability_zero_count = 0
        probability_normalization_failure_count = 0
        eig_nonpositive_count = 0
        for n in range(SOURCE_N_MAX + 1, LONG_N_MAX + 1):
            prequery_sigma = sigma.copy()
            cache_clip_count += int(np.count_nonzero((prequery_sigma < cache.sigma[0]) | (prequery_sigma > cache.sigma[-1])))
            transition = cache.lookup(method, prequery_sigma)
            radius = transition["r"]
            probabilities = mnl_probs(targets - mean, radius, b, config.main_beta)
            if np.any(~np.isfinite(probabilities)):
                raise FloatingPointError(f"non-finite response probability in {method} at n={n}")
            probability_zero_count += int(np.count_nonzero(probabilities == 0.0))
            probability_normalization_failure_count += int(
                np.count_nonzero(np.abs(probabilities.sum(axis=1) - 1.0) > STAGE_SANITY_ATOL)
            )
            eig_nonpositive_count += int(np.count_nonzero(transition["eig"] <= 0.0))
            response = inverse_cdf_rows(probabilities, uniforms[:, n - 1])
            row = np.arange(len(ids))
            mean += prequery_sigma * transition["mean_shift_sigma"][row, response]
            sigma = prequery_sigma * np.sqrt(np.maximum(transition["variance_ratio"][row, response], 1e-15))
            if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
                raise FloatingPointError(f"non-finite/non-positive ADF state in {method} at n={n}")
            floor_count += int(np.count_nonzero(sigma * sigma <= NUMERICAL_VARIANCE_FLOOR))
            if method == "PR-EIG":
                alpha = radius / prequery_sigma
                boundary = alpha >= config.alpha_max * (1.0 - 1e-6)
                for interval_index, (lower, upper) in enumerate(BOUNDARY_INTERVALS):
                    if lower <= n <= upper:
                        arrays["pr_boundary_hits"][:, interval_index] += boundary.astype(np.int32)
            if n in checkpoint_index:
                _record_checkpoint(
                    arrays,
                    key,
                    checkpoint_index[n],
                    mean,
                    sigma,
                    radius,
                    prequery_sigma,
                    transition["eig"],
                    b,
                    config.main_beta,
                )
            if n in REQUIRED_LONG_CHECKPOINTS[1:]:
                print(f"[B-long {batch_start}:{batch_stop} {method}] n={n}", flush=True)
        numerical_counts[f"{key}__variance_floor_count"] = floor_count
        numerical_counts[f"{key}__cache_clip_count"] = cache_clip_count
        numerical_counts[f"{key}__probability_zero_count"] = probability_zero_count
        numerical_counts[
            f"{key}__probability_normalization_failure_count"
        ] = probability_normalization_failure_count
        numerical_counts[f"{key}__eig_nonpositive_count"] = eig_nonpositive_count
    arrays["pr_interval_rounds"] = np.array([upper - lower + 1 for lower, upper in BOUNDARY_INTERVALS], dtype=np.int32)
    atomic_npz(output, **arrays, **{name: np.array(value) for name, value in numerical_counts.items()})
    return str(output)


def _batch_tasks(root: Path, config: ExperimentConfig) -> list[tuple[int, int, int, Path]]:
    key = stable_hash(
        {
            "source_config": config.config_hash,
            "source_n": SOURCE_N_MAX,
            "long_n": LONG_N_MAX,
            "checkpoints": long_checkpoints().tolist(),
        }
    )[:16]
    tasks = [(1, 0, 20), (2, 20, 100)]
    for start in range(100, 1_000, 50):
        tasks.append((3, start, min(start + 50, 1_000)))
    return [
        (stage, start, stop, root / "raw" / "batches" / f"batch_{start:04d}_{stop:04d}_{key}.npz")
        for stage, start, stop in tasks
    ]


def _summarize_stage(root: Path, tasks: list[tuple[int, int, int, Path]], stage: int) -> Path:
    selected = [task for task in tasks if task[0] <= stage]
    curves = {}
    counts = {}
    for method in METHODS:
        key = _method_key(method)
        variance = []
        for _, _, _, path in selected:
            with np.load(path) as data:
                variance.append(data[f"{key}__posterior_variance"])
        matrix = np.concatenate(variance, axis=0)
        epsilon = np.sqrt(matrix.mean(axis=0))
        slope = np.gradient(np.log(epsilon), np.log(long_checkpoints()))
        curves[method] = {
            "epsilon_65536": float(epsilon[-1]),
            "local_slope_65536": float(slope[-1]),
            "finite": bool(np.all(np.isfinite(matrix))),
            "positive_variance": bool(np.all(matrix > 0.0)),
        }
        floor = clip = zeros = normalization = eig_nonpositive = 0
        for _, _, _, path in selected:
            with np.load(path) as data:
                floor += int(data[f"{key}__variance_floor_count"])
                clip += int(data[f"{key}__cache_clip_count"])
                zeros += int(data[f"{key}__probability_zero_count"])
                normalization += int(data[f"{key}__probability_normalization_failure_count"])
                eig_nonpositive += int(data[f"{key}__eig_nonpositive_count"])
        counts[method] = {
            "variance_floor_count": floor,
            "cache_clip_count": clip,
            "probability_zero_count": zeros,
            "probability_normalization_failure_count": normalization,
            "eig_nonpositive_count": eig_nonpositive,
        }
    output = root / f"stage_{stage}_summary.json"
    atomic_json(
        output,
        {
            "stage": stage,
            "trajectories": STAGE_LIMITS[stage],
            "N_max": LONG_N_MAX,
            "curves": curves,
            "numerical_counts": counts,
        },
    )
    return output


def _gate_stage(summary_path: Path) -> None:
    """Refuse to advance after a numerically invalid staged extension."""
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    failures: list[str] = []
    for method, curve in summary["curves"].items():
        if not curve["finite"]:
            failures.append(f"{method}: non-finite checkpoint")
        if not curve["positive_variance"]:
            failures.append(f"{method}: non-positive variance")
        if not np.isfinite(curve["epsilon_65536"]) or curve["epsilon_65536"] <= 0.0:
            failures.append(f"{method}: invalid epsilon at n=65536")
        counts = summary["numerical_counts"][method]
        if counts["variance_floor_count"]:
            failures.append(f"{method}: variance floor hit {counts['variance_floor_count']} times")
        if counts["cache_clip_count"]:
            failures.append(f"{method}: cache clipped {counts['cache_clip_count']} states")
        if counts["probability_normalization_failure_count"]:
            failures.append(
                f"{method}: probability normalization failed "
                f"{counts['probability_normalization_failure_count']} times"
            )
        if counts["eig_nonpositive_count"]:
            failures.append(f"{method}: non-positive EIG {counts['eig_nonpositive_count']} times")
        if counts["probability_zero_count"]:
            failures.append(
                f"{method}: response-probability underflow produced zero "
                f"{counts['probability_zero_count']} times"
            )
    if failures:
        raise RuntimeError("stage sanity gate failed: " + "; ".join(failures))


def _aggregate(config: ExperimentConfig, tasks: list[tuple[int, int, int, Path]]) -> tuple[Path, Path]:
    root = output_root(config)
    source_path, _ = _source_paths(config)
    source = pd.read_parquet(source_path)
    records = []
    boundary_records = []
    for _, _, _, path in tasks:
        with np.load(path) as data:
            checkpoints = data["checkpoints"]
            keep_columns = np.where(checkpoints > SOURCE_N_MAX)[0]
            for row, trajectory in enumerate(data["trajectory"]):
                for method in METHODS:
                    key = _method_key(method)
                    for column in keep_columns:
                        n = int(checkpoints[column])
                        mean = float(data[f"{key}__posterior_mean"][row, column])
                        variance = float(data[f"{key}__posterior_variance"][row, column])
                        true = float(data["true_target"][row])
                        records.append(
                            {
                                "trajectory": int(trajectory),
                                "seed": int(data["seed"][row]),
                                "method": method,
                                "n": n,
                                "true_target": true,
                                "posterior_mean": mean,
                                "posterior_variance": variance,
                                "directional_rms": np.sqrt(variance),
                                "true_estimation_error": abs(mean - true),
                                "selected_r": float(data[f"{key}__selected_r"][row, column]),
                                "selected_alpha": float(data[f"{key}__selected_alpha"][row, column]),
                                "exact_eig": float(data[f"{key}__exact_eig"][row, column]),
                                "fisher_information": float(data[f"{key}__fisher_information"][row, column]),
                            }
                        )
                for interval_index, (lower, upper) in enumerate(BOUNDARY_INTERVALS):
                    rounds = int(data["pr_interval_rounds"][interval_index])
                    hits = int(data["pr_boundary_hits"][row, interval_index])
                    boundary_records.append(
                        {
                            "trajectory": int(trajectory),
                            "range_start": lower - 1,
                            "range_end": upper,
                            "rounds": rounds,
                            "boundary_hits": hits,
                            "boundary_hit_rate": hits / rounds,
                        }
                    )
    long_frame = pd.DataFrame.from_records(records)
    combined = pd.concat([source, long_frame], ignore_index=True).sort_values(["method", "trajectory", "n"])
    output = root / "raw" / "checkpoints.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    boundary = root / "raw" / "pr_boundary_hits.parquet"
    pd.DataFrame.from_records(boundary_records).to_parquet(boundary, index=False)
    return output, boundary


def _exact_grid_validation(config: ExperimentConfig, cache_file: Path, force: bool = False) -> Path:
    root = output_root(config)
    output = root / "diagnostics" / "adf_exact_validation.csv"
    if output.exists() and not force:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = DesignCache.load(cache_file)
    b = quantile_template(config.main_m)
    checkpoints = np.array([512, 1024, 2048, 4096, 8192], dtype=int)
    records = []
    trajectories = 10
    nodes = 4097
    for trajectory in range(trajectories):
        targets, uniforms = _trajectory_randomness(config.seed + 99_000, np.array([trajectory]), SOURCE_N_MAX)
        true = float(targets[0])
        for method in METHODS:
            grid = DenseGridPosterior(nodes)
            adf_mean, adf_sigma = 0.0, 1.0
            for n in range(1, SOURCE_N_MAX + 1):
                transition = cache.lookup(method, np.array([adf_sigma]))
                radius = float(transition["r"][0])
                probabilities = mnl_probs(true - adf_mean, radius, b, config.main_beta)
                response = int(inverse_cdf_rows(probabilities[None, :], np.array([uniforms[0, n - 1]]))[0])
                grid.update(adf_mean, radius, b, config.main_beta, response)
                adf_mean += adf_sigma * transition["mean_shift_sigma"][0, response]
                adf_sigma *= np.sqrt(transition["variance_ratio"][0, response])
                if n not in checkpoints:
                    continue
                offsets, weights, grid_mean, grid_variance = grid.centered()
                if method == "PR-EIG":
                    optimum = optimize_weighted_grid(
                        offsets,
                        weights,
                        b,
                        config.main_beta,
                        config.alpha_min,
                        config.alpha_max,
                        61,
                        "pr",
                    )
                    grid_r = optimum.value * np.sqrt(grid_variance)
                elif method == "Exact-RC":
                    optimum = optimize_weighted_grid(
                        offsets,
                        weights,
                        b,
                        config.main_beta,
                        config.physical_r_min,
                        config.physical_r_max,
                        61,
                        "physical",
                    )
                    grid_r = optimum.value
                else:
                    grid_r = radius
                    grid_eig = float(
                        weighted_grid_eig(
                            offsets, weights, np.array([grid_r]), b, config.main_beta
                        )[0]
                    )
                adf_next_r = float(cache.lookup(method, np.array([adf_sigma]))["r"][0])
                adf_eig = float(cache.lookup(method, np.array([adf_sigma]))["eig"][0])
                if method != "Fisher-RC":
                    grid_eig = float(optimum.objective)
                records.append(
                    {
                        "trajectory": trajectory,
                        "method": method,
                        "n": n,
                        "grid_nodes": nodes,
                        "adf_mean": adf_mean,
                        "grid_mean": grid_mean,
                        "absolute_mean_difference": abs(adf_mean - grid_mean),
                        "adf_variance": adf_sigma * adf_sigma,
                        "grid_variance": grid_variance,
                        "relative_variance_difference": abs(adf_sigma * adf_sigma - grid_variance) / grid_variance,
                        "adf_epsilon": adf_sigma,
                        "grid_epsilon": np.sqrt(grid_variance),
                        "adf_selected_r": adf_next_r,
                        "grid_selected_r": grid_r,
                        "relative_r_difference": abs(adf_next_r - grid_r) / max(grid_r, 1e-15),
                        "adf_eig": adf_eig,
                        "grid_eig": grid_eig,
                        "relative_eig_difference": abs(adf_eig - grid_eig)
                        / max(grid_eig, config.numerical_floor),
                    }
                )
        print(f"[B-long exact-grid] trajectory {trajectory + 1}/{trajectories}", flush=True)
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    return output


def run(config: ExperimentConfig, force: bool = False, skip_exact_grid: bool = False) -> dict[str, Path]:
    if config.mode != "paper":
        raise ValueError("the long extension must use the unchanged paper Experiment B configuration")
    root = output_root(config)
    for rel in ("raw/batches", "summaries", "diagnostics", "figures", "reproduction"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    source_raw, source_cache = _source_paths(config)
    config_record = {
        "purpose": "Long-horizon extension only; no algorithmic/statistical definition changed.",
        "source_experiment": "outputs/figure2/exp_b",
        "source_config_hash": config.config_hash,
        "source_N_max": SOURCE_N_MAX,
        "N_max": LONG_N_MAX,
        "trajectories": 1_000,
        "stages": STAGE_LIMITS,
        "M": config.main_m,
        "beta": config.main_beta,
        "seed": config.seed,
        "prior": "N(0,1)",
        "posterior": "Gaussian ADF with Gauss-Hermite response-conditioned moment matching",
        "quadrature_order": config.profile.b_gh_order,
        "target_sampling": "One N(0,1) draw per trajectory from SeedSequence([seed, trajectory])",
        "response_randomness": "The subsequent 65536 Uniform(0,1) draws from the same generator; inverse categorical CDF",
        "template": "Centered, unit-RMS equal-mass standard-normal quantile centers",
        "pr_policy": "Exact Gaussian EIG over alpha in the unchanged fixed interval, via the source design cache",
        "exact_rc_policy": "Exact Gaussian EIG over physical r in the unchanged fixed interval, via the source design cache",
        "fisher_rc_policy": "Source Fisher-information optimum with exact MNL EIG and ADF transitions",
        "alpha_range": [config.alpha_min, config.alpha_max],
        "physical_r_initial_range": [config.physical_r_min, config.physical_r_max],
        "cache_sigma_range": [config.cache_sigma_min, config.cache_sigma_max],
        "common_random_numbers": True,
        "long_checkpoints": long_checkpoints().tolist(),
        "required_checkpoints": REQUIRED_LONG_CHECKPOINTS,
        "exact_grid_validation": {
            "trajectories": 10,
            "N_max": SOURCE_N_MAX,
            "nodes": 4_097,
            "seed_offset": 99_000,
        },
        "source_raw_sha256": _sha256(source_raw),
        "source_cache_sha256": _sha256(source_cache),
    }
    atomic_json(root / "config.json", config_record)
    metadata = run_metadata(config_record, Path(__file__).resolve().parents[2])
    metadata.update({"status": "running", "resume_from": SOURCE_N_MAX})
    atomic_json(root / "run_metadata.json", metadata)
    started = time.monotonic()
    results: dict[str, Path] = {}
    try:
        reproduction = reproduce_source(config, force=force)
        results.update({f"reproduction_{key}": value for key, value in reproduction.items()})
        tasks = _batch_tasks(root, config)
        for stage in (1, 2, 3):
            stage_started = time.monotonic()
            for task_stage, start, stop, path in tasks:
                if task_stage != stage:
                    continue
                if force and path.exists():
                    path.unlink()
                _simulate_extension_batch(config, str(source_raw), str(source_cache), start, stop, str(path))
                print(f"[B-long stage {stage}] trajectories {stop}/{STAGE_LIMITS[stage]}", flush=True)
            summary = _summarize_stage(root, tasks, stage)
            _gate_stage(summary)
            metadata[f"stage_{stage}_elapsed_seconds"] = time.monotonic() - stage_started
            results[f"stage_{stage}_summary"] = summary
            atomic_json(root / "run_metadata.json", metadata)
        checkpoints, boundary = _aggregate(config, tasks)
        results["checkpoints"] = checkpoints
        results["boundary_hits"] = boundary
        if not skip_exact_grid:
            results["exact_grid_validation"] = _exact_grid_validation(config, source_cache, force=force)
        metadata["status"] = "complete"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["results"] = {key: str(value) for key, value in results.items()}
        atomic_json(root / "run_metadata.json", metadata)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-exact-grid", action="store_true")
    args = parser.parse_args()
    result = run(get_config("paper"), force=args.force, skip_exact_grid=args.skip_exact_grid)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
