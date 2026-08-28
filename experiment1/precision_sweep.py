"""Precision-sweep analysis for the redesigned randomized Figure 2(c).

All statistics are extracted from saved nominal-response ``rho_path_json``
trajectories. The environment, not an environment-response row, is the top-level
statistical unit.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from figure2_experiments.experiment1.config import Experiment1Config, get_experiment1_config
from figure2_experiments.utils import atomic_json


METHODS = ("PR-EIG", "Exact-RC", "Fisher-RC")
COMPARISONS = (("Fisher/PR", "Fisher-RC"), ("Exact/PR", "Exact-RC"))


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(contents)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def crossing_matrix(paths: np.ndarray, thresholds: Iterable[float]) -> np.ndarray:
    """Return first crossing times; non-reached thresholds are ``np.inf``."""
    values = np.asarray(paths, dtype=float)
    threshold_values = np.asarray(tuple(thresholds), dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("paths must be a runs-by-time matrix including t=0")
    if np.any(~np.isfinite(values)):
        raise ValueError("rho paths must be finite")
    if np.any((threshold_values <= 0.0) | (threshold_values >= 1.0)):
        raise ValueError("precision thresholds must lie strictly between zero and one")
    result = np.full((len(values), len(threshold_values)), np.inf)
    for index, threshold in enumerate(threshold_values):
        reached = values <= threshold
        any_reached = reached.any(axis=1)
        result[any_reached, index] = reached[any_reached].argmax(axis=1)
    return result


def _strict_response_median(values: pd.Series) -> float:
    """Median with censoring treated as infinity.

    With four response replicates this is finite only when at least three reach
    the threshold, matching the mathematical median of censored crossing times.
    """
    return float(np.median(values.to_numpy(float)))


def environment_threshold_table(
    trials: pd.DataFrame,
    thresholds: Iterable[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create run-level crossings and required environment-level pairwise rows."""
    required = {
        "environment_id", "method", "response_replication", "rho_path_json",
        "d", "M", "beta", "s0", "kappa", "t_max", "lapse",
    }
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f"trials missing columns: {sorted(missing)}")
    nominal = trials[np.isclose(trials["lapse"], 0.0)].copy()
    if nominal.empty:
        raise ValueError("precision sweep requires nominal lapse=0 trajectories")
    if nominal.groupby("environment_id")["t_max"].nunique().max() != 1:
        raise ValueError("each environment must use one common horizon")
    parsed = [np.asarray(json.loads(value), dtype=float) for value in nominal["rho_path_json"]]
    lengths = {len(value) for value in parsed}
    if len(lengths) != 1:
        raise ValueError("all rho paths must have the same length")
    paths = np.vstack(parsed)
    threshold_values = tuple(float(value) for value in thresholds)
    crossings = crossing_matrix(paths, threshold_values)

    repeated = nominal[
        [
            "environment_id", "response_replication", "method", "d", "M", "beta",
            "s0", "kappa", "t_max",
        ]
    ].loc[nominal.index.repeat(len(threshold_values))].reset_index(drop=True)
    repeated["threshold"] = np.tile(threshold_values, len(nominal))
    repeated["crossing_time"] = crossings.reshape(-1)
    repeated["response_success"] = np.isfinite(repeated["crossing_time"])

    identity = ["threshold", "environment_id", "method"]
    aggregate = (
        repeated.groupby(identity, sort=True, as_index=False)
        .agg(
            median_crossing=("crossing_time", _strict_response_median),
            response_success_rate=("response_success", "mean"),
            response_repetitions=("response_replication", "nunique"),
            d=("d", "first"),
            M=("M", "first"),
            beta=("beta", "first"),
            s0=("s0", "first"),
            kappa=("kappa", "first"),
            t_max=("t_max", "first"),
        )
    )
    aggregate["success"] = np.isfinite(aggregate["median_crossing"])

    metadata = (
        aggregate.groupby(["threshold", "environment_id"], as_index=False)
        .agg(
            d=("d", "first"), M=("M", "first"), beta=("beta", "first"),
            s0=("s0", "first"), kappa=("kappa", "first"), t_max=("t_max", "first"),
            response_repetitions=("response_repetitions", "first"),
        )
    )
    crossing_wide = aggregate.pivot(
        index=["threshold", "environment_id"], columns="method", values="median_crossing"
    ).reset_index()
    success_wide = aggregate.pivot(
        index=["threshold", "environment_id"], columns="method", values="success"
    ).reset_index()
    response_wide = aggregate.pivot(
        index=["threshold", "environment_id"],
        columns="method",
        values="response_success_rate",
    ).reset_index()
    base = metadata.merge(crossing_wide, on=["threshold", "environment_id"], validate="1:1")
    success_wide = success_wide.rename(
        columns={method: f"{method}_success" for method in METHODS}
    )
    response_wide = response_wide.rename(
        columns={method: f"{method}_response_success_rate" for method in METHODS}
    )
    base = base.merge(success_wide, on=["threshold", "environment_id"], validate="1:1")
    base = base.merge(response_wide, on=["threshold", "environment_id"], validate="1:1")
    base = base.rename(
        columns={
            "PR-EIG": "median_T_PR",
            "Fisher-RC": "median_T_Fisher",
            "Exact-RC": "median_T_Exact",
            "PR-EIG_success": "PR_success",
            "Fisher-RC_success": "Fisher_success",
            "Exact-RC_success": "Exact_success",
            "PR-EIG_response_success_rate": "PR_response_success_rate",
            "Fisher-RC_response_success_rate": "Fisher_response_success_rate",
            "Exact-RC_response_success_rate": "Exact_response_success_rate",
        }
    )

    records: list[pd.DataFrame] = []
    for comparison, calibrated in COMPARISONS:
        method_label = calibrated.removesuffix("-RC")
        calibrated_t = f"median_T_{method_label}"
        calibrated_success = f"{method_label}_success"
        frame = base.copy()
        frame["comparison"] = comparison
        frame["comparable"] = frame["PR_success"] & frame[calibrated_success]
        frame["Fisher_PR_ratio"] = np.where(
            frame["PR_success"] & frame["Fisher_success"],
            frame["median_T_Fisher"] / frame["median_T_PR"],
            np.nan,
        )
        frame["Exact_PR_ratio"] = np.where(
            frame["PR_success"] & frame["Exact_success"],
            frame["median_T_Exact"] / frame["median_T_PR"],
            np.nan,
        )
        frame["Fisher_PR_difference"] = np.where(
            frame["PR_success"] & frame["Fisher_success"],
            frame["median_T_Fisher"] - frame["median_T_PR"],
            np.nan,
        )
        frame["Exact_PR_difference"] = np.where(
            frame["PR_success"] & frame["Exact_success"],
            frame["median_T_Exact"] - frame["median_T_PR"],
            np.nan,
        )
        frame["comparison_ratio"] = np.where(
            frame["comparable"], frame[calibrated_t] / frame["median_T_PR"], np.nan
        )
        frame["comparison_difference"] = np.where(
            frame["comparable"], frame[calibrated_t] - frame["median_T_PR"], np.nan
        )
        records.append(frame)
    environment = pd.concat(records, ignore_index=True)
    environment = environment.sort_values(
        ["threshold", "environment_id", "comparison"], ascending=[False, True, True]
    ).reset_index(drop=True)
    repeated["crossing_time"] = repeated["crossing_time"].replace(np.inf, np.nan)
    for column in ("median_T_PR", "median_T_Fisher", "median_T_Exact"):
        environment[column] = environment[column].replace(np.inf, np.nan)
    return repeated, environment


def _bootstrap_median(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)), endpoint=False)
    medians = np.median(values[indices], axis=1)
    low, high = np.quantile(medians, (0.025, 0.975))
    return float(low), float(high)


def summarize_thresholds(
    environment: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Aggregate ratios, differences, coverage, and environment-bootstrap CIs."""
    records: list[dict[str, Any]] = []
    thresholds = sorted(environment["threshold"].unique(), reverse=True)
    for threshold_index, threshold in enumerate(thresholds):
        threshold_rows = environment[np.isclose(environment["threshold"], threshold)]
        one_per_environment = threshold_rows[threshold_rows["comparison"] == "Fisher/PR"]
        record: dict[str, Any] = {
            "threshold": float(threshold),
            "environments": int(one_per_environment["environment_id"].nunique()),
            "response_repetitions": int(one_per_environment["response_repetitions"].iloc[0]),
            "PR_success_rate": float(one_per_environment["PR_success"].mean()),
            "Fisher_success_rate": float(one_per_environment["Fisher_success"].mean()),
            "Exact_success_rate": float(one_per_environment["Exact_success"].mean()),
            "PR_mean_response_success_rate": float(
                one_per_environment["PR_response_success_rate"].mean()
            ),
            "Fisher_mean_response_success_rate": float(
                one_per_environment["Fisher_response_success_rate"].mean()
            ),
            "Exact_mean_response_success_rate": float(
                one_per_environment["Exact_response_success_rate"].mean()
            ),
        }
        for comparison_index, (comparison, _) in enumerate(COMPARISONS):
            group = threshold_rows[
                (threshold_rows["comparison"] == comparison)
                & threshold_rows["comparable"].astype(bool)
            ]
            ratios = group["comparison_ratio"].to_numpy(float)
            differences = group["comparison_difference"].to_numpy(float)
            prefix = "Fisher_PR" if comparison == "Fisher/PR" else "Exact_PR"
            low, high = _bootstrap_median(
                ratios,
                bootstrap_samples,
                seed + 10_007 * threshold_index + 101 * comparison_index,
            )
            record.update(
                {
                    f"{prefix}_comparable_environments": int(len(group)),
                    f"{prefix}_median_ratio": float(np.median(ratios)) if len(ratios) else np.nan,
                    f"{prefix}_ratio_iqr_low": float(np.quantile(ratios, 0.25)) if len(ratios) else np.nan,
                    f"{prefix}_ratio_iqr_high": float(np.quantile(ratios, 0.75)) if len(ratios) else np.nan,
                    f"{prefix}_ratio_p10": float(np.quantile(ratios, 0.10)) if len(ratios) else np.nan,
                    f"{prefix}_ratio_p90": float(np.quantile(ratios, 0.90)) if len(ratios) else np.nan,
                    f"{prefix}_median_ratio_bootstrap_ci_low": low,
                    f"{prefix}_median_ratio_bootstrap_ci_high": high,
                    f"{prefix}_median_difference": float(np.median(differences)) if len(differences) else np.nan,
                    f"{prefix}_difference_iqr_low": float(np.quantile(differences, 0.25)) if len(differences) else np.nan,
                    f"{prefix}_difference_iqr_high": float(np.quantile(differences, 0.75)) if len(differences) else np.nan,
                }
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _crossover(summary: pd.DataFrame, prefix: str) -> float | None:
    ordered = summary.sort_values("threshold", ascending=False)
    below = ordered[ordered[f"{prefix}_median_ratio"] < 1.0]
    return float(below.iloc[0]["threshold"]) if len(below) else None


def _strict_crossover(summary: pd.DataFrame, prefix: str, suffix: str) -> float | None:
    ordered = summary.sort_values("threshold", ascending=False)
    below = ordered[ordered[f"{prefix}_{suffix}"] < 1.0]
    return float(below.iloc[0]["threshold"]) if len(below) else None


def _prefix_maximum_difference(extended: pd.DataFrame, baseline: pd.DataFrame) -> float:
    keys = ["environment_id", "response_replication", "method"]
    left = baseline[np.isclose(baseline["lapse"], 0.0)].sort_values(keys)
    right = extended[np.isclose(extended["lapse"], 0.0)].sort_values(keys)
    if not left[keys].reset_index(drop=True).equals(right[keys].reset_index(drop=True)):
        raise ValueError("extended and baseline trajectories do not share identical run keys")
    maximum = 0.0
    for baseline_json, extended_json in zip(left["rho_path_json"], right["rho_path_json"]):
        old = np.asarray(json.loads(baseline_json), dtype=float)
        new = np.asarray(json.loads(extended_json), dtype=float)
        if len(new) < len(old):
            raise ValueError("extended trajectory is shorter than its baseline")
        maximum = max(maximum, float(np.max(np.abs(old - new[: len(old)]))))
    return maximum


def _report_text(
    config: Experiment1Config,
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    metadata: dict[str, Any],
    prefix_maximum_difference: float,
) -> tuple[dict[str, Any], str]:
    at_025 = summary[np.isclose(summary["threshold"], 0.25)].iloc[0]
    fisher_cross = _crossover(summary, "Fisher_PR")
    exact_cross = _crossover(summary, "Exact_PR")
    fisher_ci_cross = _strict_crossover(
        summary, "Fisher_PR", "median_ratio_bootstrap_ci_high"
    )
    exact_ci_cross = _strict_crossover(
        summary, "Exact_PR", "median_ratio_bootstrap_ci_high"
    )
    fisher_iqr_cross = _strict_crossover(summary, "Fisher_PR", "ratio_iqr_high")
    exact_iqr_cross = _strict_crossover(summary, "Exact_PR", "ratio_iqr_high")
    finest = summary.sort_values("threshold").iloc[0]
    matching = summary[
        ["Fisher_PR_median_ratio", "Exact_PR_median_ratio"]
    ].dropna()
    maximum_gap = float(
        np.max(np.abs(matching["Fisher_PR_median_ratio"] - matching["Exact_PR_median_ratio"]))
    )
    baseline_finest = baseline.sort_values("threshold").iloc[0]
    if fisher_cross is None and exact_cross is None:
        conclusion = "RC still does not produce a practical query-count gain in this randomized benchmark."
    elif fisher_cross is not None and exact_cross is not None:
        conclusion = "RC becomes useful only at fine precision in this randomized benchmark."
    else:
        conclusion = "Exact and Fisher calibration diverge in the fine-precision regime."
    structured = {
        "config_hash": config.config_hash,
        "horizon": int(metadata["T_max"]),
        "baseline_horizon": 400,
        "thresholds": list(config.c_precision_thresholds),
        "rho_0.25": at_025.to_dict(),
        "Fisher_PR_first_median_below_one": fisher_cross,
        "Exact_PR_first_median_below_one": exact_cross,
        "Fisher_PR_first_bootstrap_CI_below_one": fisher_ci_cross,
        "Exact_PR_first_bootstrap_CI_below_one": exact_ci_cross,
        "Fisher_PR_first_IQR_below_one": fisher_iqr_cross,
        "Exact_PR_first_IQR_below_one": exact_iqr_cross,
        "maximum_absolute_median_ratio_gap_Fisher_vs_Exact": maximum_gap,
        "maximum_absolute_rho_prefix_difference_old400_vs_extended": prefix_maximum_difference,
        "finest_threshold": finest.to_dict(),
        "baseline_finest_threshold": baseline_finest.to_dict(),
        "conclusion": conclusion,
    }
    fisher_cross_text = "none" if fisher_cross is None else f"rho*={fisher_cross:g}"
    exact_cross_text = "none" if exact_cross is None else f"rho*={exact_cross:g}"
    fisher_ci_text = "none" if fisher_ci_cross is None else f"rho*={fisher_ci_cross:g}"
    exact_ci_text = "none" if exact_ci_cross is None else f"rho*={exact_ci_cross:g}"
    fisher_iqr_text = "none" if fisher_iqr_cross is None else f"rho*={fisher_iqr_cross:g}"
    exact_iqr_text = "none" if exact_iqr_cross is None else f"rho*={exact_iqr_cross:g}"
    required_reaches = int(metadata["response_repetitions"]) // 2 + 1
    markdown = f"""# Figure 2(c) precision-sweep report

This report is computed mechanically from saved nominal-response trajectories. The
environment is the bootstrap unit. Within each environment, a threshold time is the
median over {int(metadata['response_repetitions'])} response replicates with censoring
treated as infinity; at least {required_reaches} replicates must reach the threshold.

## Configuration

- Environments: {int(metadata['environment_count'])}
- Response replicates per environment: {int(metadata['response_repetitions'])}
- Old horizon: 400 queries
- Extended horizon: {int(metadata['T_max'])} queries
- Thresholds: {', '.join(f'{value:g}' for value in config.c_precision_thresholds)}
- Posterior approximation: projected particle ADF (unchanged)
- Ratio inclusion: PR-EIG and the calibrated method must both have a finite
  environment-level median threshold time.
- Main shadow: across-environment IQR. Bootstrap intervals are saved separately.

## Results

1. At rho*=0.25, the median Fisher/PR ratio is
   {at_025['Fisher_PR_median_ratio']:.6f} (IQR
   [{at_025['Fisher_PR_ratio_iqr_low']:.6f}, {at_025['Fisher_PR_ratio_iqr_high']:.6f}]);
   the median Exact/PR ratio is {at_025['Exact_PR_median_ratio']:.6f} (IQR
   [{at_025['Exact_PR_ratio_iqr_low']:.6f}, {at_025['Exact_PR_ratio_iqr_high']:.6f}]).
2. The first median ratio below one is Fisher/PR: {fisher_cross_text};
   Exact/PR: {exact_cross_text}. A median below one means fewer queries than PR-EIG.
3. The first median-bootstrap CI wholly below one is Fisher/PR: {fisher_ci_text};
   Exact/PR: {exact_ci_text}. The first full environment IQR below one is Fisher/PR:
   {fisher_iqr_text}; Exact/PR: {exact_iqr_text}.
4. At the finest rho*={finest['threshold']:g}, Fisher/PR is
   {finest['Fisher_PR_median_ratio']:.6f} from
   {int(finest['Fisher_PR_comparable_environments'])} comparable environments;
   Exact/PR is {finest['Exact_PR_median_ratio']:.6f} from
   {int(finest['Exact_PR_comparable_environments'])} environments.
5. Fisher qualitatively tracks Exact: the maximum absolute gap between their threshold-wise
   median ratios is {maximum_gap:.6f}.
6. At rho*={finest['threshold']:g}, environment success rates are PR
   {finest['PR_success_rate']:.3f}, Exact {finest['Exact_success_rate']:.3f}, and Fisher
   {finest['Fisher_success_rate']:.3f}. Coverage declines as precision becomes stricter;
   every plotted threshold reports its comparable-environment count.
7. At the old 400-query horizon, the finest-threshold Fisher/PR comparison had only
   {int(baseline_finest['Fisher_PR_comparable_environments'])} comparable environments.
   The extension raises this to {int(finest['Fisher_PR_comparable_environments'])}; it was
   required to make the fine-precision sweep interpretable.
8. The maximum absolute difference between every old 400-query rho path and the prefix
   of its extended trajectory is {prefix_maximum_difference:.3g}.

## Interpretation

{conclusion}

This conclusion is descriptive for the randomized projected-ADF benchmark. The retained
rho*=0.25 scatter remains an appendix diagnostic and is not overwritten or hidden.
"""
    return structured, markdown


def analyze_precision_sweep(
    config: Experiment1Config,
    *,
    trials_path: Path | None = None,
    baseline_path: Path | None = None,
    output_directory: Path | None = None,
) -> dict[str, Path]:
    randomized_root = config.output_root / "randomized"
    horizon = config.profile.c_precision_t_max
    trials_path = trials_path or randomized_root / f"precision_sweep_t{horizon}_trials.parquet"
    baseline_path = baseline_path or randomized_root / "main_trials.parquet"
    output = output_directory or randomized_root / "precision_sweep"
    output.mkdir(parents=True, exist_ok=True)
    trials = pd.read_parquet(trials_path)
    baseline_trials = pd.read_parquet(baseline_path)
    run_crossings, environment = environment_threshold_table(
        trials, config.c_precision_thresholds
    )
    _, baseline_environment = environment_threshold_table(
        baseline_trials, config.c_precision_thresholds
    )
    summary = summarize_thresholds(
        environment,
        bootstrap_samples=config.profile.c_bootstrap_samples,
        seed=config.seed + 8_021,
    )
    baseline_summary = summarize_thresholds(
        baseline_environment,
        bootstrap_samples=config.profile.c_bootstrap_samples,
        seed=config.seed + 8_023,
    )
    prefix_maximum_difference = _prefix_maximum_difference(trials, baseline_trials)
    metadata_path = trials_path.with_name(trials_path.stem.replace("_trials", "_metadata") + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    structured, markdown = _report_text(
        config, summary, baseline_summary, metadata, prefix_maximum_difference
    )

    paths = {
        "run_crossings": output / "run_threshold_crossings.parquet",
        "environment_parquet": output / "environment_thresholds.parquet",
        "environment_csv": output / "environment_thresholds.csv",
        "summary": output / "threshold_summary.csv",
        "summary_parquet": output / "threshold_summary.parquet",
        "baseline_summary": output / "threshold_summary_t400.csv",
        "report": output / "FIGURE2C_PRECISION_SWEEP_REPORT.md",
        "report_json": output / "FIGURE2C_PRECISION_SWEEP_REPORT.json",
        "metadata": output / "analysis_metadata.json",
    }
    run_crossings.to_parquet(paths["run_crossings"], index=False)
    environment.to_parquet(paths["environment_parquet"], index=False)
    environment.to_csv(paths["environment_csv"], index=False)
    summary.to_csv(paths["summary"], index=False)
    summary.to_parquet(paths["summary_parquet"], index=False)
    baseline_summary.to_csv(paths["baseline_summary"], index=False)
    _atomic_text(paths["report"], markdown)
    atomic_json(paths["report_json"], structured)
    atomic_json(
        paths["metadata"],
        {
            "source_trials": str(trials_path),
            "baseline_trials": str(baseline_path),
            "source_metadata": str(metadata_path),
            "thresholds": config.c_precision_thresholds,
            "bootstrap_samples": config.profile.c_bootstrap_samples,
            "environment_success_definition": (
                "finite mathematical median over response crossing times with censored values as infinity"
            ),
            "comparison_inclusion": "both PR-EIG and calibrated method environment medians finite",
            "difference_sign": "calibrated method minus PR-EIG; negative favors calibration",
            "config_hash": config.config_hash,
            "maximum_absolute_rho_prefix_difference_old400_vs_extended": (
                prefix_maximum_difference
            ),
        },
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--trials", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    outputs = analyze_precision_sweep(
        get_experiment1_config(args.mode),
        trials_path=args.trials,
        baseline_path=args.baseline,
        output_directory=args.output,
    )
    print(outputs["report"].read_text(encoding="utf-8"))
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
