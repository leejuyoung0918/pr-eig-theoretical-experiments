"""Analyze Experiment A with fixed fit windows and save diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure2_experiments.config import ExperimentConfig, get_config
from figure2_experiments.utils import loglog_fit


COLORS = {"PR-EIG": "#3b5b92", "Exact-RC": "#a33d3d", "Fisher-RC": "#2f7652"}
MARKERS = {"PR-EIG": "o", "Exact-RC": "s", "Fisher-RC": "^"}


def _slopes(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    records = []
    for (beta, m, method), group in frame.groupby(["beta", "M", "method"], sort=True):
        x = group["sigma"].to_numpy(float)
        y = group["eig"].to_numpy(float)
        floor = group["numerical_floor"].to_numpy(bool)
        for upper in config.sigma_fit_windows:
            result = loglog_fit(x, y, (x <= upper) & ~floor)
            records.append(
                {
                    "beta": beta,
                    "M": m,
                    "method": method,
                    "sigma_fit_max": upper,
                    "primary": upper == config.primary_sigma_fit_max,
                    "theory_slope": 4.0 if method == "PR-EIG" else 2.0,
                    "difference_from_theory": result["slope"]
                    - (4.0 if method == "PR-EIG" else 2.0),
                    **result,
                }
            )
    return pd.DataFrame.from_records(records)


def _diagnostics(main: pd.DataFrame, robustness: pd.DataFrame, config: ExperimentConfig) -> None:
    output = config.output_root / "exp_a" / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for method, group in main.groupby("method"):
        axes[0].loglog(group["sigma"], group["r"], label=method, color=COLORS[method], marker=MARKERS[method], markevery=10, ms=3)
        axes[1].loglog(group["sigma"], group["alpha"], label=method, color=COLORS[method], marker=MARKERS[method], markevery=10, ms=3)
    axes[0].set(xlabel=r"Posterior width $\sigma$", ylabel=r"Selected physical radius $r^\star$")
    axes[1].set(xlabel=r"Posterior width $\sigma$", ylabel=r"$\alpha^\star=r^\star/\sigma$")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "A_resolution_and_alpha.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    for method, group in main.groupby("method"):
        power = 4 if method == "PR-EIG" else 2
        ax.loglog(group["sigma"], group["eig"] / group["sigma"] ** power, label=method, color=COLORS[method], marker=MARKERS[method], markevery=10, ms=3)
    ax.set(xlabel=r"Posterior width $\sigma$", ylabel=r"Scaled information $I/\sigma^k$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "A_information_ratios.png", dpi=220)
    plt.close(fig)

    late = robustness[(robustness["method"] == "Exact-RC")].copy()
    late = late.loc[late.groupby(["beta", "M"])["sigma"].idxmin()]
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    for m, group in late.groupby("M"):
        ax.plot(group["beta"], group["r_sqrt_beta"], marker="o", label=f"M={m}")
    ax.set_xscale("log", base=2)
    ax.set(xlabel=r"Inverse temperature $\beta$", ylabel=r"$r^\star\sqrt{\beta}$")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "A_beta_calibration.png", dpi=220)
    plt.close(fig)

    exact = main[main["method"] == "Exact-RC"].sort_values("sigma")
    fisher = main[main["method"] == "Fisher-RC"].sort_values("sigma")
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.loglog(exact["sigma"], exact["r"], label="Exact-RC", color=COLORS["Exact-RC"])
    ax.loglog(fisher["sigma"], fisher["r"], label="Fisher-RC", color=COLORS["Fisher-RC"], ls="--")
    ax.set(xlabel=r"Posterior width $\sigma$", ylabel="Selected radius")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "A_exact_vs_fisher_resolution.png", dpi=220)
    plt.close(fig)

    slope_summary = _slopes(robustness, config)
    slope_summary = slope_summary[np.isclose(slope_summary["sigma_fit_max"], config.primary_sigma_fit_max)]
    fig, axes = plt.subplots(1, len(config.m_grid), figsize=(12.0, 3.0), sharey=True)
    for ax, m in zip(axes, config.m_grid):
        subset = slope_summary[slope_summary["M"] == m]
        for method, group in subset.groupby("method"):
            ax.semilogx(group["beta"], group["slope"], base=2, marker=MARKERS[method], color=COLORS[method], label=method)
        ax.axhline(4.0, color="0.25", ls="--", lw=0.8)
        ax.axhline(2.0, color="0.55", ls=":", lw=0.9)
        ax.set_title(f"M={m}")
        ax.set_xlabel(r"$\beta$")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pre-registered information slope")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(config.output_root / "supplementary" / "A_robustness_slopes.png", dpi=220)
    plt.close(fig)


def analyze(config: ExperimentConfig) -> dict[str, Path]:
    root = config.output_root / "exp_a"
    main = pd.read_parquet(root / "raw" / "main.parquet")
    robustness = pd.read_parquet(root / "raw" / "robustness.parquet")
    slopes = _slopes(robustness, config)
    slopes_path = root / "slopes.csv"
    slopes.to_csv(slopes_path, index=False)

    summary = slopes.copy()
    summary["optimal_r_smallest_sigma"] = np.nan
    for (beta, m, method), group in robustness.groupby(["beta", "M", "method"]):
        idx = summary.index[(summary["beta"] == beta) & (summary["M"] == m) & (summary["method"] == method)]
        smallest = group.loc[group["sigma"].idxmin()]
        summary.loc[idx, "optimal_r_smallest_sigma"] = smallest["r"]
        summary.loc[idx, "boundary_at_smallest_sigma"] = bool(smallest["boundary"])
        summary.loc[idx, "boundary_fraction"] = float(group["boundary"].mean())
    summary["r_sqrt_beta_smallest_sigma"] = summary["optimal_r_smallest_sigma"] * np.sqrt(summary["beta"])
    robustness_path = root / "robustness.csv"
    summary.to_csv(robustness_path, index=False)
    _diagnostics(main, robustness, config)
    return {"slopes": slopes_path, "robustness": robustness_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "paper", "exhaustive"), default="paper")
    args = parser.parse_args()
    for key, path in analyze(get_config(args.mode)).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
