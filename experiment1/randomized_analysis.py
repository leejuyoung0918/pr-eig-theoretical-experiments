"""Environment-level analysis for the randomized robustness benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from figure2_experiments.policies import optimize_fisher, optimize_physical, optimize_pr
from figure2_experiments.templates import quantile_template
from figure2_experiments.utils import atomic_json

from .config import Experiment1Config
from .randomized import METHODS, ResolutionLibrary, resolution_library_path
from .statistics import environment_bootstrap_indices


def _environment_aggregate(trials: pd.DataFrame) -> pd.DataFrame:
    records = []
    identity = [
        "environment_id", "seed", "d", "M", "beta", "s0", "kappa",
        "initial_trace", "true_target_norm", "lapse", "method", "particle_count", "t_max",
    ]
    for keys, group in trials.groupby(identity, sort=True, dropna=False):
        row = dict(zip(identity, keys))
        t_max = int(row["t_max"])
        for column in ("T_05", "T_025", "T_01"):
            values = group[column].fillna(t_max + 1).to_numpy(float)
            row[f"median_{column}_plot"] = float(np.median(values))
            row[f"success_rate_{column}"] = float(group[column].notna().mean())
            row[f"median_{column}"] = (
                float(np.median(group[column].dropna())) if group[column].notna().any() else np.nan
            )
        row["success_025"] = bool(row["median_T_025_plot"] <= t_max)
        for column in (
            "final_rho",
            "final_normalized_target_error",
            "mean_query_design_seconds",
            "mean_resolution_seconds",
            "mean_posterior_update_seconds",
            "eig_evaluation_cost",
            "minimum_ess_fraction",
            "minimum_unique_ancestor_fraction",
            "resampling_count",
        ):
            row[f"median_{column}"] = float(group[column].median())
            row[f"mean_{column}"] = float(group[column].mean())
        row["response_repetitions"] = int(group["response_replication"].nunique())
        row["censored_response_repetitions"] = int((~group["success_025"].astype(bool)).sum())
        records.append(row)
    return pd.DataFrame.from_records(records)


def _bootstrap_statistic(
    frame: pd.DataFrame,
    samples: int,
    seed: int,
    statistic: Callable[[pd.DataFrame], float],
) -> tuple[float, float, float]:
    ids = np.sort(frame["environment_id"].unique())
    indexed = frame.set_index("environment_id").loc[ids]
    indices = environment_bootstrap_indices(len(ids), samples, seed)
    values = np.empty(samples)
    for sample_index, draw in enumerate(indices):
        sample = indexed.iloc[draw].reset_index()
        values[sample_index] = statistic(sample)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan, np.nan, np.nan
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(np.mean(finite)), float(low), float(high)


def _paired_frame(aggregate: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    columns = [
        "environment_id", "median_T_025_plot", "success_025", "median_final_rho",
        "median_final_normalized_target_error", "median_mean_query_design_seconds",
    ]
    first = aggregate[aggregate["method"] == left][columns].set_index("environment_id")
    second = aggregate[aggregate["method"] == right][columns].set_index("environment_id")
    paired = first.join(second, lsuffix="_left", rsuffix="_right", how="inner").reset_index()
    paired["left_method"] = left
    paired["right_method"] = right
    return paired


def _paired_summary(
    paired: pd.DataFrame,
    config: Experiment1Config,
    seed_offset: int,
) -> dict[str, float | int | str]:
    left_t = paired["median_T_025_plot_left"].to_numpy(float)
    right_t = paired["median_T_025_plot_right"].to_numpy(float)
    both_success = paired["success_025_left"].astype(bool) & paired["success_025_right"].astype(bool)
    wins = int(np.count_nonzero(right_t < left_t))
    ties = int(np.count_nonzero(right_t == left_t))
    losses = int(np.count_nonzero(right_t > left_t))
    ratios = right_t[both_success] / left_t[both_success]
    savings = left_t[both_success] - right_t[both_success]

    def win_stat(sample: pd.DataFrame) -> float:
        return float(np.mean(sample["median_T_025_plot_right"] < sample["median_T_025_plot_left"]))

    def ratio_stat(sample: pd.DataFrame) -> float:
        mask = sample["success_025_left"].astype(bool) & sample["success_025_right"].astype(bool)
        if not mask.any():
            return np.nan
        return float(np.median(
            sample.loc[mask, "median_T_025_plot_right"]
            / sample.loc[mask, "median_T_025_plot_left"]
        ))

    def saving_stat(sample: pd.DataFrame) -> float:
        mask = sample["success_025_left"].astype(bool) & sample["success_025_right"].astype(bool)
        if not mask.any():
            return np.nan
        return float(np.median(
            sample.loc[mask, "median_T_025_plot_left"]
            - sample.loc[mask, "median_T_025_plot_right"]
        ))

    def log_rho_stat(sample: pd.DataFrame) -> float:
        return float(np.median(
            np.log(sample["median_final_rho_right"])
            - np.log(sample["median_final_rho_left"])
        ))

    def log_error_stat(sample: pd.DataFrame) -> float:
        return float(np.median(
            np.log(np.maximum(sample["median_final_normalized_target_error_right"], 1e-300))
            - np.log(np.maximum(sample["median_final_normalized_target_error_left"], 1e-300))
        ))

    win_mean, win_low, win_high = _bootstrap_statistic(
        paired, config.profile.c_bootstrap_samples, config.seed + seed_offset, win_stat
    )
    ratio_mean, ratio_low, ratio_high = _bootstrap_statistic(
        paired, config.profile.c_bootstrap_samples, config.seed + seed_offset + 1, ratio_stat
    )
    saving_mean, saving_low, saving_high = _bootstrap_statistic(
        paired, config.profile.c_bootstrap_samples, config.seed + seed_offset + 2, saving_stat
    )
    log_rho_mean, log_rho_low, log_rho_high = _bootstrap_statistic(
        paired, config.profile.c_bootstrap_samples, config.seed + seed_offset + 3, log_rho_stat
    )
    log_error_mean, log_error_low, log_error_high = _bootstrap_statistic(
        paired, config.profile.c_bootstrap_samples, config.seed + seed_offset + 4, log_error_stat
    )
    return {
        "left_method": str(paired["left_method"].iloc[0]),
        "right_method": str(paired["right_method"].iloc[0]),
        "environments": len(paired),
        "right_wins": wins,
        "ties": ties,
        "left_wins": losses,
        "win_rate": wins / len(paired),
        "win_rate_bootstrap_mean": win_mean,
        "win_rate_ci_low": win_low,
        "win_rate_ci_high": win_high,
        "both_success_environments": int(both_success.sum()),
        "median_query_ratio": float(np.median(ratios)) if len(ratios) else np.nan,
        "query_ratio_iqr_low": float(np.quantile(ratios, 0.25)) if len(ratios) else np.nan,
        "query_ratio_iqr_high": float(np.quantile(ratios, 0.75)) if len(ratios) else np.nan,
        "query_ratio_bootstrap_mean": ratio_mean,
        "query_ratio_ci_low": ratio_low,
        "query_ratio_ci_high": ratio_high,
        "median_query_saving": float(np.median(savings)) if len(savings) else np.nan,
        "query_saving_bootstrap_mean": saving_mean,
        "query_saving_ci_low": saving_low,
        "query_saving_ci_high": saving_high,
        "left_success_rate": float(paired["success_025_left"].mean()),
        "right_success_rate": float(paired["success_025_right"].mean()),
        "median_log_final_rho_difference_right_minus_left": float(np.median(
            np.log(paired["median_final_rho_right"]) - np.log(paired["median_final_rho_left"])
        )),
        "log_final_rho_difference_ci_low": log_rho_low,
        "log_final_rho_difference_ci_high": log_rho_high,
        "median_log_final_target_error_difference_right_minus_left": float(np.median(
            np.log(np.maximum(paired["median_final_normalized_target_error_right"], 1e-300))
            - np.log(np.maximum(paired["median_final_normalized_target_error_left"], 1e-300))
        )),
        "log_final_target_error_difference_ci_low": log_error_low,
        "log_final_target_error_difference_ci_high": log_error_high,
        "median_design_time_ratio_right_over_left": float(np.median(
            paired["median_mean_query_design_seconds_right"]
            / paired["median_mean_query_design_seconds_left"]
        )),
    }


def _group_ratio(group: pd.DataFrame, label: str, value: str) -> dict[str, object]:
    paired = _paired_frame(group, "PR-EIG", "Fisher-RC")
    successful = paired[paired["success_025_left"].astype(bool) & paired["success_025_right"].astype(bool)]
    ratio = successful["median_T_025_plot_right"] / successful["median_T_025_plot_left"]
    return {
        "stratification": label,
        "level": value,
        "environments": int(paired["environment_id"].nunique()),
        "both_success_environments": len(successful),
        "median_fisher_pr_query_ratio": float(ratio.median()) if len(ratio) else np.nan,
        "q25": float(ratio.quantile(0.25)) if len(ratio) else np.nan,
        "q75": float(ratio.quantile(0.75)) if len(ratio) else np.nan,
        "fisher_win_rate": float(
            (paired["median_T_025_plot_right"] < paired["median_T_025_plot_left"]).mean()
        ),
    }


def _sensitivity(aggregate: pd.DataFrame) -> pd.DataFrame:
    nominal = aggregate[np.isclose(aggregate["lapse"], 0.0)].copy()
    records = []
    for d, group in nominal.groupby("d"):
        records.append(_group_ratio(group, "dimension", str(int(d))))
    for m, group in nominal.groupby("M"):
        records.append(_group_ratio(group, "arity", str(int(m))))
    kappa_bins = pd.cut(
        nominal["kappa"], [1.0, 3.0, 10.0, 30.0, 100.0000001],
        labels=("1-3", "3-10", "10-30", "30-100"), include_lowest=True, right=False,
    )
    for level in kappa_bins.cat.categories:
        group = nominal[kappa_bins == level]
        if len(group):
            records.append(_group_ratio(group, "anisotropy", str(level)))
    beta_edges = np.array([0.25, np.sqrt(0.25 * 0.5), np.sqrt(0.5), np.sqrt(2.0), np.sqrt(8.0), 4.0000001])
    beta_labels = ("0.25", "0.5", "1", "2", "4")
    beta_bins = pd.cut(
        nominal["beta"], beta_edges, labels=beta_labels, include_lowest=True, right=False
    )
    for level in beta_bins.cat.categories:
        group = nominal[beta_bins == level]
        if len(group):
            records.append(_group_ratio(group, "beta", str(level)))
    return pd.DataFrame.from_records(records)


def _threshold_sensitivity(aggregate: pd.DataFrame) -> pd.DataFrame:
    nominal = aggregate[np.isclose(aggregate["lapse"], 0.0)].copy()
    records = []
    for label, column in (("0.5", "T_05"), ("0.25", "T_025"), ("0.1", "T_01")):
        success_column = f"success_rate_{column}"
        median_plot_column = f"median_{column}_plot"
        for method, group in nominal.groupby("method"):
            records.append(
                {
                    "threshold": float(label),
                    "method": method,
                    "environments": int(group["environment_id"].nunique()),
                    "environment_success_rate": float((group[median_plot_column] <= group["t_max"]).mean()),
                    "mean_response_success_rate": float(group[success_column].mean()),
                    "median_query_count_plot": float(group[median_plot_column].median()),
                }
            )
        pr = nominal[nominal["method"] == "PR-EIG"].set_index("environment_id")
        fisher = nominal[nominal["method"] == "Fisher-RC"].set_index("environment_id")
        mask = (pr[median_plot_column] <= pr["t_max"]) & (fisher[median_plot_column] <= fisher["t_max"])
        ratio = fisher.loc[mask, median_plot_column] / pr.loc[mask, median_plot_column]
        records.append(
            {
                "threshold": float(label),
                "method": "Fisher/PR ratio",
                "environments": int(mask.sum()),
                "environment_success_rate": np.nan,
                "mean_response_success_rate": np.nan,
                "median_query_count_plot": float(ratio.median()) if len(ratio) else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def _runtime_table(trials: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (lapse, method), group in trials.groupby(["lapse", "method"], sort=True):
        design_values = np.concatenate(
            [np.asarray(json.loads(value), dtype=float) for value in group["query_design_times_json"]]
        )
        resolution_values = np.concatenate(
            [np.asarray(json.loads(value), dtype=float) for value in group["resolution_times_json"]]
        )
        update_values = np.concatenate(
            [np.asarray(json.loads(value), dtype=float) for value in group["posterior_update_times_json"]]
        )
        records.append(
            {
                "lapse": lapse,
                "method": method,
                "queries_timed": len(design_values),
                "mean_query_design_seconds": float(design_values.mean()),
                "median_query_design_seconds": float(np.median(design_values)),
                "mean_resolution_optimization_seconds": float(resolution_values.mean()),
                "median_resolution_optimization_seconds": float(np.median(resolution_values)),
                "mean_posterior_update_seconds": float(update_values.mean()),
                "median_posterior_update_seconds": float(np.median(update_values)),
                "total_algorithm_seconds": float(group["total_algorithm_seconds"].sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def _uncached_design_benchmark(
    trials: pd.DataFrame,
    config: Experiment1Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time direct optimizers on saved query states, without altering the run."""
    nominal = trials[np.isclose(trials["lapse"], 0.0)].copy()
    states_per_method = 12 if config.mode == "quick" else 60
    records = []
    for method_index, method in enumerate(METHODS):
        group = nominal[nominal["method"] == method].sort_values(
            ["environment_id", "response_replication"]
        )
        selected = group.iloc[
            np.unique(np.linspace(0, len(group) - 1, min(states_per_method, len(group)), dtype=int))
        ]
        for sample_index, (_, row) in enumerate(selected.iterrows()):
            sigma_path = np.asarray(json.loads(row["directional_sigma_path_json"]), dtype=float)
            radius_path = np.asarray(json.loads(row["selected_radius_path_json"]), dtype=float)
            query_index = int(
                np.floor((sample_index + 0.5) * len(sigma_path) / max(len(selected), 1))
            )
            query_index = min(query_index, len(sigma_path) - 1)
            sigma = float(sigma_path[query_index])
            template = quantile_template(int(row["M"]))
            started = time.perf_counter()
            if method == "PR-EIG":
                optimum = optimize_pr(
                    sigma, template, float(row["beta"]), config.alpha_min, config.alpha_max,
                    config.profile.c_library_grid_points, config.profile.c_library_gh_order,
                )
                radius = sigma * optimum.value
            elif method == "Exact-RC":
                optimum = optimize_physical(
                    sigma, template, float(row["beta"]), config.physical_r_min,
                    config.physical_r_max, config.profile.c_library_grid_points,
                    config.profile.c_library_gh_order,
                )
                radius = optimum.value
            else:
                optimum = optimize_fisher(
                    template, float(row["beta"]), config.physical_r_min,
                    config.physical_r_max, config.profile.c_library_grid_points,
                )
                radius = optimum.value
            elapsed = time.perf_counter() - started
            cached_radius = float(radius_path[query_index])
            records.append(
                {
                    "method": method,
                    "environment_id": int(row["environment_id"]),
                    "response_replication": int(row["response_replication"]),
                    "query_index": query_index + 1,
                    "d": int(row["d"]),
                    "M": int(row["M"]),
                    "beta": float(row["beta"]),
                    "directional_sigma": sigma,
                    "direct_radius": radius,
                    "cached_radius": cached_radius,
                    "relative_cached_radius_error": abs(radius - cached_radius) / max(radius, 1e-300),
                    "direct_optimizer_seconds": elapsed,
                    "objective_evaluations": optimum.evaluations,
                    "boundary": optimum.boundary,
                }
            )
    detailed = pd.DataFrame.from_records(records)
    summary = (
        detailed.groupby("method", as_index=False)
        .agg(
            states=("direct_optimizer_seconds", "size"),
            mean_uncached_design_seconds=("direct_optimizer_seconds", "mean"),
            median_uncached_design_seconds=("direct_optimizer_seconds", "median"),
            mean_objective_evaluations=("objective_evaluations", "mean"),
            median_relative_cached_radius_error=("relative_cached_radius_error", "median"),
            max_relative_cached_radius_error=("relative_cached_radius_error", "max"),
            boundary_fraction=("boundary", "mean"),
        )
    )
    return detailed, summary


def _diagnostics(
    trials: pd.DataFrame,
    aggregate: pd.DataFrame,
    config: Experiment1Config,
) -> dict[str, object]:
    paired_failures = 0
    parameter_columns = ["d", "M", "beta", "s0", "kappa", "initial_trace", "true_target_norm"]
    for (_, _), group in trials.groupby(["lapse", "environment_id"]):
        paired_failures += int(any(group[column].nunique(dropna=False) != 1 for column in parameter_columns))
    library = ResolutionLibrary.load(resolution_library_path(config))
    allowed_censor_columns = {"T_05", "T_025", "T_01"}
    numeric = trials.select_dtypes(include=[np.number])
    checked_numeric = numeric[[column for column in numeric.columns if column not in allowed_censor_columns]]
    return {
        "rows": len(trials),
        "environments": int(trials["environment_id"].nunique()),
        "lapse_values": sorted(trials["lapse"].unique().tolist()),
        "unexpected_nonfinite_numeric_values": int(
            np.count_nonzero(~np.isfinite(checked_numeric.to_numpy(float)))
        ),
        "maximum_weight_sum_error": float(trials["maximum_weight_sum_error"].max()),
        "minimum_covariance_eigenvalue": float(trials["minimum_covariance_eigenvalue"].min()),
        "probability_normalization_failures": int(trials["probability_normalization_failures"].sum()),
        "boundary_hits": int(trials["boundary_hits"].sum()),
        "tau_clip_count": int(trials["tau_clip_count"].sum()),
        "minimum_ess_fraction": float(trials["minimum_ess_fraction"].min()),
        "minimum_unique_ancestor_fraction": float(trials["minimum_unique_ancestor_fraction"].min()),
        "total_resampling_events": int(trials["resampling_count"].sum()),
        "censored_response_runs": int((~trials["success_025"].astype(bool)).sum()),
        "censored_environment_method_rows": int((~aggregate["success_025"].astype(bool)).sum()),
        "paired_environment_parameter_failures": paired_failures,
        "library_negative_eig_count": int(np.count_nonzero(library.eig < 0.0)),
        "library_boundary_fraction_by_method": {
            method: float(library.boundary[:, method_index].mean())
            for method_index, method in enumerate(METHODS)
        },
        "dimension_counts": trials.drop_duplicates("environment_id")["d"].value_counts().sort_index().to_dict(),
        "arity_counts": trials.drop_duplicates("environment_id")["M"].value_counts().sort_index().to_dict(),
        "beta_range": [
            float(trials["beta"].min()), float(trials["beta"].max())
        ],
        "kappa_range": [
            float(trials["kappa"].min()), float(trials["kappa"].max())
        ],
    }


def analyze_particle_convergence(config: Experiment1Config) -> Path | None:
    path = config.output_root / "randomized" / "particle_convergence_trials.parquet"
    if not path.exists():
        return None
    trials = pd.read_parquet(path)
    records = []
    for (count, method), group in trials.groupby(["particle_count", "method"], sort=True):
        t_plot = group["T_025"].fillna(group["t_max"] + 1)
        records.append(
            {
                "particle_count": int(count),
                "method": method,
                "runs": len(group),
                "success_rate": float(group["success_025"].mean()),
                "median_T_025_plot": float(t_plot.median()),
                "median_final_rho": float(group["final_rho"].median()),
                "median_final_target_error": float(group["final_normalized_target_error"].median()),
                "median_minimum_ess_fraction": float(group["minimum_ess_fraction"].median()),
            }
        )
    output = config.output_root / "randomized" / "particle_convergence.csv"
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    return output


def analyze_randomized(config: Experiment1Config) -> dict[str, Path]:
    root = config.output_root / "randomized"
    trials_path = root / "main_trials.parquet"
    trials = pd.read_parquet(trials_path)
    aggregate = _environment_aggregate(trials)
    aggregate_path = root / "environment_aggregated.parquet"
    aggregate.to_parquet(aggregate_path, index=False)

    paired_records = []
    pair_paths: dict[str, Path] = {}
    for lapse_index, lapse in enumerate(sorted(aggregate["lapse"].unique())):
        subset = aggregate[np.isclose(aggregate["lapse"], lapse)]
        for pair_index, (left, right) in enumerate(
            (("PR-EIG", "Fisher-RC"), ("PR-EIG", "Exact-RC"), ("Exact-RC", "Fisher-RC"))
        ):
            paired = _paired_frame(subset, left, right)
            paired["lapse"] = lapse
            name = f"paired_{left.lower().replace('-', '_')}_vs_{right.lower().replace('-', '_')}_lapse_{lapse:.2f}.csv"
            path = root / name
            paired.to_csv(path, index=False)
            pair_paths[name] = path
            summary = _paired_summary(
                paired, config, 10_000 + lapse_index * 100 + pair_index * 10
            )
            summary["lapse"] = lapse
            paired_records.append(summary)
    summary_table = pd.DataFrame.from_records(paired_records)
    summary_path = root / "paired_summary.csv"
    summary_table.to_csv(summary_path, index=False)
    (root / "paired_summary.tex").write_text(
        summary_table.to_latex(index=False, float_format="%.4g"), encoding="utf-8"
    )

    sensitivity = _sensitivity(aggregate)
    sensitivity_path = root / "sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False)
    threshold_sensitivity = _threshold_sensitivity(aggregate)
    threshold_sensitivity_path = root / "threshold_sensitivity.csv"
    threshold_sensitivity.to_csv(threshold_sensitivity_path, index=False)
    runtime = _runtime_table(trials)
    runtime_path = root / "runtime.csv"
    runtime.to_csv(runtime_path, index=False)
    (root / "runtime.tex").write_text(
        runtime.to_latex(index=False, float_format="%.4g"), encoding="utf-8"
    )
    uncached_detail, uncached_summary = _uncached_design_benchmark(trials, config)
    uncached_detail_path = root / "uncached_design_benchmark_trials.csv"
    uncached_summary_path = root / "uncached_design_benchmark.csv"
    uncached_detail.to_csv(uncached_detail_path, index=False)
    uncached_summary.to_csv(uncached_summary_path, index=False)
    diagnostics = _diagnostics(trials, aggregate, config)
    diagnostics_path = root / "diagnostics.json"
    atomic_json(diagnostics_path, diagnostics)
    convergence_path = analyze_particle_convergence(config)
    result = {
        "trials": trials_path,
        "aggregate": aggregate_path,
        "summary": summary_path,
        "sensitivity": sensitivity_path,
        "threshold_sensitivity": threshold_sensitivity_path,
        "runtime": runtime_path,
        "uncached_design_benchmark": uncached_summary_path,
        "uncached_design_benchmark_trials": uncached_detail_path,
        "diagnostics": diagnostics_path,
    }
    if convergence_path is not None:
        result["particle_convergence"] = convergence_path
    result.update(pair_paths)
    return result
