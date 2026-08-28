"""Trajectory bootstrap, fixed-window slopes, and local-slope diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure2_experiments.config import ExperimentConfig, get_config
from figure2_experiments.exp_b_sequential.design_cache import METHODS
from figure2_experiments.utils import loglog_fit


def _bootstrap_method(
    group: pd.DataFrame,
    method: str,
    config: ExperimentConfig,
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame, dict[str, np.ndarray]]:
    pivot = group.pivot(index="trajectory", columns="n", values="posterior_variance").sort_index(axis=1)
    n = pivot.columns.to_numpy(float)
    variance = pivot.to_numpy(float)
    epsilon = np.sqrt(np.mean(variance, axis=0))
    numerical_floor = epsilon <= 1e-12
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 771, METHODS.index(method)]))
    bootstrap_count = config.profile.b_bootstrap_samples
    counts = rng.multinomial(len(variance), np.full(len(variance), 1.0 / len(variance)), size=bootstrap_count)
    bootstrap_variance = counts @ variance / len(variance)
    bootstrap_epsilon = np.sqrt(bootstrap_variance)
    ci_low, ci_high = np.quantile(bootstrap_epsilon, [0.025, 0.975], axis=0)
    curve = pd.DataFrame(
        {"method": method, "n": n.astype(int), "epsilon": epsilon, "ci_low": ci_low, "ci_high": ci_high, "numerical_floor": numerical_floor}
    )

    records = []
    raw_slopes: dict[str, np.ndarray] = {}
    theory = -0.25 if method == "PR-EIG" else -0.5
    for lower in config.sequential_fit_windows:
        mask = (n >= lower) & ~numerical_floor
        fit = loglog_fit(n, epsilon, mask)
        if mask.sum() >= 3:
            x = np.log(n[mask])
            x_centered = x - x.mean()
            denominator = np.sum(x_centered * x_centered)
            bootstrap_slopes = np.sum(
                x_centered[None, :] * np.log(bootstrap_epsilon[:, mask]), axis=1
            ) / denominator
            ci = np.quantile(bootstrap_slopes, [0.025, 0.975])
            raw_slopes[str(lower)] = bootstrap_slopes
            mean_slope = float(np.mean(bootstrap_slopes))
            median_slope = float(np.median(bootstrap_slopes))
        else:
            ci = (np.nan, np.nan)
            mean_slope = median_slope = np.nan
            raw_slopes[str(lower)] = np.full(bootstrap_count, np.nan)
        records.append(
            {
                "method": method,
                "fit_min_n": lower,
                "fit_max_n": int(n.max()),
                "primary": lower == config.primary_sequential_fit_min,
                "point_slope": fit["slope"],
                "bootstrap_mean": mean_slope,
                "bootstrap_median": median_slope,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "theory_slope": theory,
                "difference_from_theory": mean_slope - theory,
                "n_points": fit["n_points"],
            }
        )
    return records, curve, raw_slopes


def analyze(config: ExperimentConfig) -> dict[str, Path]:
    root = config.output_root / "exp_b"
    raw = pd.read_parquet(root / "raw" / "trajectories.parquet")
    slope_records = []
    curves = []
    raw_bootstrap = {}
    for method in METHODS:
        records, curve, samples = _bootstrap_method(raw[raw["method"] == method], method, config)
        slope_records.extend(records)
        curves.append(curve)
        for lower, values in samples.items():
            raw_bootstrap[f"{method}__{lower}"] = values
    slopes = pd.DataFrame.from_records(slope_records)
    slopes_path = root / "slopes.csv"
    bootstrap_path = root / "bootstrap.csv"
    slopes.to_csv(slopes_path, index=False)
    slopes.to_csv(bootstrap_path, index=False)
    np.savez_compressed(root / "raw" / "bootstrap_slope_samples.npz", **raw_bootstrap)

    curve = pd.concat(curves, ignore_index=True)
    curve_path = root / "curve.csv"
    curve.to_csv(curve_path, index=False)
    local_records = []
    for method, group in curve.groupby("method"):
        group = group.sort_values("n")
        local = np.gradient(np.log(group["epsilon"]), np.log(group["n"]))
        for n, value in zip(group["n"], local):
            local_records.append({"method": method, "n": int(n), "local_slope": value})
    local_frame = pd.DataFrame.from_records(local_records)
    local_path = root / "diagnostics" / "local_slopes.csv"
    local_frame.to_csv(local_path, index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    colors = {"PR-EIG": "#3b5b92", "Exact-RC": "#a33d3d", "Fisher-RC": "#2f7652"}
    for method, group in local_frame.groupby("method"):
        ax.semilogx(group["n"], group["local_slope"], label=method, color=colors[method])
    ax.axhline(-0.25, color="0.35", ls="--", lw=1, label=r"$-1/4$ guide")
    ax.axhline(-0.5, color="0.60", ls=":", lw=1.2, label=r"$-1/2$ guide")
    ax.set(xlabel="Number of queries n", ylabel="Sliding log-log slope")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "diagnostics" / "B_local_slopes.png", dpi=220)
    plt.close(fig)

    validation = pd.read_csv(root / "exact_grid_validation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for method, group in validation.groupby("method"):
        summary = group.groupby("n", as_index=False).median(numeric_only=True)
        axes[0].loglog(summary["n"], summary["relative_variance_difference"], label=method)
        axes[1].loglog(summary["n"], summary["relative_eig_difference"], label=method)
    axes[0].set(xlabel="n", ylabel="ADF/grid relative variance difference")
    axes[1].set(xlabel="n", ylabel="ADF/grid relative EIG difference")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "diagnostics" / "B_adf_exact_grid.png", dpi=220)
    plt.close(fig)
    return {"slopes": slopes_path, "bootstrap": bootstrap_path, "curve": curve_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "paper", "exhaustive"), default="paper")
    args = parser.parse_args()
    for key, path in analyze(get_config(args.mode)).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
