"""Publication figures for the redesigned Experiment 1, from saved results only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure2_experiments.experiment1.config import Experiment1Config, get_experiment1_config
from figure2_experiments.utils import atomic_json


METHODS = ("PR-EIG", "Exact-RC", "Fisher-RC")
COLORS = {"PR-EIG": "#355C9A", "Exact-RC": "#A23B3B", "Fisher-RC": "#2E7550"}
MARKERS = {"PR-EIG": "o", "Exact-RC": "s", "Fisher-RC": "^"}
LINESTYLES = {"PR-EIG": "-", "Exact-RC": "--", "Fisher-RC": "-."}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.2,
            "axes.labelsize": 11.0,
            "axes.titlesize": 10.6,
            "xtick.labelsize": 9.1,
            "ytick.labelsize": 9.1,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _clean(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3.0, width=0.7)
    ax.grid(False)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12.0, fontweight="bold", clip_on=False,
    )


def _fmt_sig(value: float, digits: int = 3) -> str:
    """Format a finite scalar with exactly ``digits`` significant digits."""
    if not np.isfinite(value):
        return str(value)
    if value == 0:
        return "0." + "0" * (digits - 1)
    exponent = int(np.floor(np.log10(abs(value))))
    decimal_places = digits - 1 - exponent
    if -3 <= exponent < digits:
        return f"{value:.{max(0, decimal_places)}f}"
    return f"{value:.{digits - 1}e}"


def draw_panel_a(ax: plt.Axes, root: Path) -> None:
    curve = pd.read_parquet(root / "controlled" / "curve.parquet")
    slopes = pd.read_csv(root / "controlled" / "slopes.csv").set_index("method")
    guides = pd.read_csv(root / "controlled" / "guides.csv")
    for method in METHODS:
        group = curve[curve["method"] == method].sort_values("sigma")
        if method == "Exact-RC":
            markevery = (0, 6)
            zorder = 4
        elif method == "Fisher-RC":
            markevery = (3, 6)
            zorder = 5
        else:
            markevery = (0, 9)
            zorder = 3
        label = "Baseline" if method == "PR-EIG" else method
        ax.loglog(
            group["sigma"], group["eig"], color=COLORS[method], ls=LINESTYLES[method],
            marker=MARKERS[method], markevery=markevery, ms=3.3, mfc="white",
            mew=0.8, label=label, zorder=zorder,
        )
    guide_styles = {"sigma^4": ("0.28", (0, (4, 2))), "sigma^2": ("0.52", (0, (2, 2)))}
    for guide, group in guides.groupby("guide", sort=False):
        color, linestyle = guide_styles[guide]
        label = r"$\sigma^4$ guide" if guide == "sigma^4" else r"$\sigma^2$ guide"
        ax.loglog(group["sigma"], group["eig"], color=color, ls=linestyle, lw=1.0, label=label, zorder=1)
    annotation = "\n".join(
        [
            f"Baseline: {_fmt_sig(float(slopes.loc['PR-EIG', 'fitted_slope']))}",
            f"Exact: {_fmt_sig(float(slopes.loc['Exact-RC', 'fitted_slope']))}",
            f"Fisher: {_fmt_sig(float(slopes.loc['Fisher-RC', 'fitted_slope']))}",
        ]
    )
    ax.text(
        0.04, 0.05, annotation, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8.3, bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.82", "lw": 0.6},
    )
    ax.set_xlabel(r"Posterior width $\sigma$")
    ax.set_ylabel("One-step mutual information")
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.8, handlelength=2.0)
    _clean(ax)


def draw_panel_b(ax: plt.Axes, root: Path) -> None:
    curve = pd.read_csv(root / "sequential" / "curve.csv")
    slopes = pd.read_csv(root / "sequential" / "slopes.csv")
    primary = slopes[slopes["primary"].astype(bool)].set_index("method")
    guides = pd.read_csv(root / "sequential" / "guides.csv")
    for method in METHODS:
        group = curve[curve["method"] == method].sort_values("n")
        ax.fill_between(
            group["n"].to_numpy(float), group["ci_low"].to_numpy(float),
            group["ci_high"].to_numpy(float), color=COLORS[method], alpha=0.20, lw=0,
        )
        label = "Baseline" if method == "PR-EIG" else method
        ax.loglog(
            group["n"], group["epsilon"], color=COLORS[method], ls=LINESTYLES[method],
            marker=MARKERS[method], markevery=max(1, len(group) // 10), ms=3.0,
            mfc="white", mew=0.75, label=label,
        )
    for method, label, color, linestyle in (
        ("PR-EIG", r"$n^{-1/4}$ guide", "0.28", (0, (4, 2))),
        ("Fisher-RC", r"$n^{-1/2}$ guide", "0.52", (0, (2, 2))),
    ):
        group = guides[guides["method"] == method].sort_values("n")
        ax.loglog(group["n"], group["epsilon"], color=color, ls=linestyle, lw=1.0, label=label)
    annotation = "\n".join(
        [
            f"Baseline: {_fmt_sig(float(primary.loc['PR-EIG', 'slope']))}",
            f"Exact: {_fmt_sig(float(primary.loc['Exact-RC', 'slope']))}",
            f"Fisher: {_fmt_sig(float(primary.loc['Fisher-RC', 'slope']))}",
        ]
    )
    ax.text(
        0.04, 0.05, annotation, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8.3, bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.82", "lw": 0.6},
    )
    ax.set_xlabel(r"Number of queries $n$")
    ax.set_ylabel(r"Directional posterior RMS $\epsilon_n$")
    ax.legend(
        frameon=False, loc="upper right", ncol=1, handlelength=2.0,
        fontsize=7.7, labelspacing=0.25,
    )
    _clean(ax)


def draw_panel_c(ax: plt.Axes, root: Path) -> None:
    summary = pd.read_csv(root / "randomized" / "precision_sweep" / "threshold_summary.csv")
    summary = summary.sort_values("threshold", ascending=False)
    x = summary["threshold"].to_numpy(float)
    series = (
        (
            "Fisher_PR", "Fisher-RC / Baseline", COLORS["Fisher-RC"], "-", "^",
        ),
        (
            "Exact_PR", "Exact-RC / Baseline", COLORS["Exact-RC"], "--", "s",
        ),
    )
    for prefix, label, color, linestyle, marker in series:
        median = summary[f"{prefix}_median_ratio"].to_numpy(float)
        low = summary[f"{prefix}_ratio_iqr_low"].to_numpy(float)
        high = summary[f"{prefix}_ratio_iqr_high"].to_numpy(float)
        ax.fill_between(x, low, high, color=color, alpha=0.16, lw=0, zorder=1)
        ax.plot(
            x, median, color=color, ls=linestyle, marker=marker, ms=3.4,
            mfc="white", mew=0.8, label=label, zorder=3,
        )
    ax.axhline(1.0, color="0.28", ls=(0, (4, 2)), lw=0.95, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(float(x.max()) * 1.08, float(x.min()) / 1.08)
    key_ticks = np.array([0.50, 0.10, 0.02])
    ax.set_xticks(key_ticks, [f"{value:g}" for value in key_ticks])
    ax.minorticks_off()
    ax.set_xlabel(r"Target normalized precision $\rho^\star$")
    ax.set_ylabel("Query ratio to Baseline")
    annotation = (
        f"{int(summary['environments'].max())} environments\n"
        f"{int(summary['response_repetitions'].max())} response reps\n"
        "Band: environment IQR"
    )
    ax.text(
        0.04, 0.05, annotation, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.82", "lw": 0.6},
    )
    ax.legend(frameon=False, loc="upper left", fontsize=7.7, handletextpad=0.4)
    _clean(ax)


def _save_panel(draw, root: Path, path: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(3.70, 2.95), layout="constrained")
    draw(ax, root)
    _panel_label(ax, label)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_sensitivity(sensitivity: pd.DataFrame, stratification: str, path: Path, xlabel: str) -> None:
    group = sensitivity[sensitivity["stratification"] == stratification].copy()
    if stratification in ("beta", "dimension", "arity"):
        group["order"] = pd.to_numeric(group["level"])
        group = group.sort_values("order")
        x = group["order"].to_numpy(float)
    else:
        order = {"1-3": 1, "3-10": 2, "10-30": 3, "30-100": 4}
        group["order"] = group["level"].map(order)
        group = group.sort_values("order")
        x = group["order"].to_numpy(float)
    y = group["median_fisher_pr_query_ratio"].to_numpy(float)
    low = group["q25"].to_numpy(float)
    high = group["q75"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.3, 3.1), layout="constrained")
    ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), color=COLORS["Fisher-RC"], marker="o", capsize=2.5)
    ax.axhline(1.0, color="0.35", ls="--", lw=0.9)
    if stratification == "beta":
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, group["level"])
    elif stratification == "anisotropy":
        ax.set_xticks(x, group["level"])
    else:
        ax.set_xticks(x, group["level"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Median $T_{0.25}^{Fisher}/T_{0.25}^{PR}$")
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_noiseless(source_root: Path, path: Path) -> None:
    data = pd.read_csv(source_root / "outputs" / "figure2" / "exp_c" / "full_grid.csv")
    dimensions = sorted(data["d"].unique())
    arities = sorted(data["M"].unique())
    fig, axes = plt.subplots(len(dimensions), len(arities), figsize=(10.2, 6.7), sharex=False, layout="constrained")
    for row_index, d in enumerate(dimensions):
        for column_index, m in enumerate(arities):
            ax = axes[row_index, column_index]
            subset = data[(data["d"] == d) & (data["M"] == m)]
            for policy, color in (("RC-MLQ", COLORS["Exact-RC"]), ("Random Coordinate", COLORS["PR-EIG"])):
                group = subset[subset["policy"] == policy].sort_values("T")
                ax.fill_between(group["T"], group["ci_low"], group["ci_high"], color=color, alpha=0.12, lw=0)
                ax.semilogy(group["T"], group["empirical_rms"], color=color, lw=1.0)
                ax.semilogy(group["T"], group["matching_theory"], color=color, ls="--", lw=0.8)
            rc = subset[subset["policy"] == "RC-MLQ"].sort_values("T")
            ax.semilogy(rc["T"], rc["universal_lower_bound"], color="0.35", ls=":", lw=0.8)
            ax.set_title(f"d={d}, M={m}", fontsize=8.0)
            if row_index == len(dimensions) - 1:
                ax.set_xlabel("T")
            if column_index == 0:
                ax.set_ylabel(r"$E_T$")
            _clean(ax)
    handles = [
        plt.Line2D([], [], color=COLORS["Exact-RC"], label="RC empirical"),
        plt.Line2D([], [], color=COLORS["Exact-RC"], ls="--", label="RC theory"),
        plt.Line2D([], [], color=COLORS["PR-EIG"], label="Random empirical"),
        plt.Line2D([], [], color=COLORS["PR-EIG"], ls="--", label="Random theory"),
        plt.Line2D([], [], color="0.35", ls=":", label="Universal lower bound"),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=5,
        frameon=False, fontsize=7.2,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _noiseless_representative_ci(
    source_root: Path,
    path: Path,
    *,
    dimension: int = 8,
    arity: int = 4,
) -> None:
    """Main-text noiseless diagnostic with explicit Monte Carlo 95% CIs."""
    data = pd.read_csv(source_root / "outputs" / "figure2" / "exp_c" / "full_grid.csv")
    subset = data[(data["d"] == dimension) & (data["M"] == arity)].copy()
    if subset.empty:
        raise ValueError(f"missing noiseless results for d={dimension}, M={arity}")

    policies = (
        ("RC-MLQ", COLORS["Exact-RC"], "o", "RC-MLQ empirical"),
        ("Random Coordinate", COLORS["PR-EIG"], "s", "Random-coordinate empirical"),
        ("Entropy-EIG Uniform Tie", COLORS["Fisher-RC"], "^", "Entropy-tie empirical"),
    )
    fig, ax = plt.subplots(figsize=(6.9, 3.35), layout="constrained")
    for policy, color, marker, label in policies:
        group = subset[subset["policy"] == policy].sort_values("T")
        x = group["T"].to_numpy(float)
        y = group["empirical_rms"].to_numpy(float)
        low = group["ci_low"].to_numpy(float)
        high = group["ci_high"].to_numpy(float)
        ax.fill_between(x, low, high, color=color, alpha=0.24, lw=0, zorder=1)
        ax.semilogy(
            x, y, color=color, marker=marker, markevery=dimension, ms=4.1,
            mfc="white", mew=0.9, lw=1.45, label=label, zorder=3,
        )
        selected = np.arange(0, len(group), max(1, dimension))
        ax.errorbar(
            x[selected], y[selected],
            yerr=np.vstack((y[selected] - low[selected], high[selected] - y[selected])),
            fmt="none", ecolor=color, elinewidth=0.9, capsize=2.0, capthick=0.8,
            alpha=0.95, zorder=4,
        )

    rc = subset[subset["policy"] == "RC-MLQ"].sort_values("T")
    random = subset[subset["policy"] == "Random Coordinate"].sort_values("T")
    ax.semilogy(
        rc["T"], rc["matching_theory"], color=COLORS["Exact-RC"],
        ls="--", lw=1.05, label="RC-MLQ exact theory", zorder=2,
    )
    ax.semilogy(
        random["T"], random["matching_theory"], color=COLORS["PR-EIG"],
        ls="--", lw=1.05, label="Random-coordinate theory", zorder=2,
    )
    ax.semilogy(
        rc["T"], rc["universal_lower_bound"], color="0.30",
        ls=":", lw=1.15, label="Universal lower bound", zorder=2,
    )
    targets = int(subset["targets"].max())
    ax.text(
        0.75, 0.95,
        rf"$d={dimension},\ M={arity}$; {targets:,} targets per policy" + "\n"
        "95% bootstrap CI: shading + caps\n"
        "20 macro-seeds x 5,000 resamples",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.2,
        color="0.28",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.82",
            "linewidth": 0.6,
            "alpha": 0.92,
        },
        zorder=8,
    )
    ax.set_xlabel(r"Number of noiseless queries $T$")
    ax.set_ylabel(r"Euclidean RMS $\mathcal{E}_T$")
    ax.set_xlim(float(subset["T"].min()), float(subset["T"].max()))
    ax.legend(
        frameon=False, loc="lower left", ncol=2, fontsize=7.6,
        columnspacing=0.9, handlelength=2.2, labelspacing=0.35,
    )
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_sequential_windows(root: Path, path: Path) -> None:
    slopes = pd.read_csv(root / "sequential" / "slopes.csv")
    fig, ax = plt.subplots(figsize=(4.7, 3.2), layout="constrained")
    for method in METHODS:
        group = slopes[slopes["method"] == method].sort_values("fit_window_start")
        ax.semilogx(group["fit_window_start"], group["slope"], color=COLORS[method], marker=MARKERS[method], ls=LINESTYLES[method], label=method)
    ax.axhline(-0.25, color="0.35", ls="--", lw=0.9)
    ax.axhline(-0.5, color="0.6", ls=":", lw=0.9)
    ax.set_xlabel("Fit-window start (end fixed at 100,000)")
    ax.set_ylabel("Fitted log-log slope")
    ax.legend(frameon=False)
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_scatter_rho025(root: Path, path: Path) -> None:
    """Retain the historical single-threshold PR/Fisher view unchanged in meaning."""
    paired = pd.read_csv(
        root / "randomized" / "paired_pr_eig_vs_fisher_rc_lapse_0.00.csv"
    )
    success = paired["success_025_left"].astype(bool) & paired["success_025_right"].astype(bool)
    fig, ax = plt.subplots(figsize=(4.2, 3.6), layout="constrained")
    ax.scatter(
        paired.loc[success, "median_T_025_plot_left"],
        paired.loc[success, "median_T_025_plot_right"],
        s=17, marker="o", facecolor="#4C78A8", edgecolor="white", linewidth=0.35,
        alpha=0.48, label="Both reached target", zorder=3,
    )
    ax.scatter(
        paired.loc[~success, "median_T_025_plot_left"],
        paired.loc[~success, "median_T_025_plot_right"],
        s=25, marker="x", color="#B04A4A", linewidth=0.8, alpha=0.8,
        label="Censored case", zorder=4,
    )
    maximum = int(
        max(
            paired["median_T_025_plot_left"].max(),
            paired["median_T_025_plot_right"].max(),
        )
    )
    ax.plot([0, maximum], [0, maximum], color="0.3", ls="--", lw=1.0, label=r"Equality $y=x$")
    ax.set(
        xlabel=r"PR-EIG queries to $\rho\leq0.25$",
        ylabel=r"Fisher-RC queries to $\rho\leq0.25$",
        xlim=(0, maximum + 4),
        ylim=(0, maximum + 4),
        title=r"Single-threshold diagnostic ($\rho^\star=0.25$)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="lower right", fontsize=7.0)
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _precision_summary(root: Path) -> pd.DataFrame:
    return pd.read_csv(
        root / "randomized" / "precision_sweep" / "threshold_summary.csv"
    ).sort_values("threshold", ascending=False)


def _precision_axis(ax: plt.Axes, values: np.ndarray) -> None:
    ax.set_xscale("log")
    ax.set_xlim(float(values.max()) * 1.08, float(values.min()) / 1.08)
    ticks = np.array([0.50, 0.10, 0.02])
    ax.set_xticks(ticks, [f"{value:g}" for value in ticks])
    ax.set_xlabel(r"Target normalized precision $\rho^\star$")


def _appendix_success_vs_threshold(root: Path, path: Path) -> None:
    data = _precision_summary(root)
    x = data["threshold"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.7, 3.3), layout="constrained")
    for method_index, method in enumerate(METHODS):
        prefix = method.split("-")[0]
        ax.plot(
            x, data[f"{prefix}_success_rate"], color=COLORS[method],
            ls=LINESTYLES[method], marker=MARKERS[method], markevery=(method_index, 3),
            mfc="white", mew=0.8, label=method,
        )
    _precision_axis(ax, x)
    ax.set_ylabel("Environment success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_counts_vs_threshold(root: Path, path: Path) -> None:
    data = _precision_summary(root)
    x = data["threshold"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.7, 3.3), layout="constrained")
    ax.plot(
        x, data["Fisher_PR_comparable_environments"], color=COLORS["Fisher-RC"],
        marker="^", markevery=(0, 2), mfc="white", mew=0.8, label="Fisher/PR",
    )
    ax.plot(
        x, data["Exact_PR_comparable_environments"], color=COLORS["Exact-RC"],
        ls="--", marker="s", markevery=(1, 2), mfc="white", mew=0.8,
        label="Exact/PR",
    )
    _precision_axis(ax, x)
    ax.set_ylabel("Comparable environments")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_diff_vs_threshold(root: Path, path: Path) -> None:
    data = _precision_summary(root)
    x = data["threshold"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.7, 3.3), layout="constrained")
    for prefix, label, method, linestyle, marker, markevery in (
        ("Fisher_PR", "Fisher-RC - PR-EIG", "Fisher-RC", "-", "^", (0, 2)),
        ("Exact_PR", "Exact-RC - PR-EIG", "Exact-RC", "--", "s", (1, 2)),
    ):
        median = data[f"{prefix}_median_difference"].to_numpy(float)
        low = data[f"{prefix}_difference_iqr_low"].to_numpy(float)
        high = data[f"{prefix}_difference_iqr_high"].to_numpy(float)
        ax.fill_between(x, low, high, color=COLORS[method], alpha=0.16, lw=0)
        ax.plot(
            x, median, color=COLORS[method], ls=linestyle, marker=marker,
            markevery=markevery, mfc="white", mew=0.8, label=label,
        )
    ax.axhline(0.0, color="0.3", ls="--", lw=0.9)
    _precision_axis(ax, x)
    ax.set_ylabel("Query difference (calibrated - PR)")
    ax.legend(frameon=False, fontsize=7.2)
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_exact_fisher_scatter(root: Path, path: Path) -> None:
    paired = pd.read_csv(root / "randomized" / "paired_exact_rc_vs_fisher_rc_lapse_0.00.csv")
    fig, ax = plt.subplots(figsize=(4.0, 3.6), layout="constrained")
    success = paired["success_025_left"].astype(bool) & paired["success_025_right"].astype(bool)
    ax.scatter(paired.loc[success, "median_T_025_plot_left"], paired.loc[success, "median_T_025_plot_right"], s=16, alpha=0.45, color=COLORS["Fisher-RC"])
    ax.scatter(paired.loc[~success, "median_T_025_plot_left"], paired.loc[~success, "median_T_025_plot_right"], s=24, marker="x", color="#B04A4A")
    maximum = max(paired["median_T_025_plot_left"].max(), paired["median_T_025_plot_right"].max())
    ax.plot([0, maximum], [0, maximum], color="0.35", ls="--")
    ax.set(xlabel="Exact-RC queries", ylabel="Fisher-RC queries", xlim=(0, maximum + 4), ylim=(0, maximum + 4))
    ax.set_aspect("equal", adjustable="box")
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_entropy_sanity(source_root: Path, path: Path) -> None:
    slopes = pd.read_csv(source_root / "outputs" / "figure2" / "exp_c" / "slopes.csv")
    labels = [f"d={int(row.d)},M={int(row.M)}" for row in slopes.itertuples()]
    values = 100.0 * slopes["entropy_random_max_relative_discrepancy"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
    ax.bar(np.arange(len(values)), values, color="0.45")
    ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
    ax.set_ylabel("Max entropy/random discrepancy (%)")
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_particle_convergence(root: Path, path: Path) -> None:
    data = pd.read_csv(root / "randomized" / "particle_convergence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    for method in METHODS:
        group = data[data["method"] == method].sort_values("particle_count")
        axes[0].semilogx(group["particle_count"], group["median_final_rho"], color=COLORS[method], marker=MARKERS[method], label=method)
        axes[1].semilogx(group["particle_count"], group["median_T_025_plot"], color=COLORS[method], marker=MARKERS[method])
    axes[0].set(xlabel="Directional particles", ylabel=r"Median final $\rho$")
    axes[1].set(xlabel="Directional particles", ylabel=r"Median $T_{0.25}^{plot}$")
    axes[0].legend(frameon=False)
    for ax in axes:
        _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_mismatch(root: Path, path: Path) -> None:
    summary = pd.read_csv(root / "randomized" / "paired_summary.csv")
    data = summary[(summary["left_method"] == "PR-EIG") & (summary["right_method"] == "Fisher-RC")].sort_values("lapse")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), layout="constrained")
    y = data["median_query_ratio"].to_numpy(float)
    axes[0].errorbar(
        100 * data["lapse"], y,
        yerr=np.vstack((y - data["query_ratio_ci_low"], data["query_ratio_ci_high"] - y)),
        color=COLORS["Fisher-RC"], marker="o", capsize=2.5,
    )
    axes[0].axhline(1.0, color="0.35", ls="--", lw=0.9)
    axes[0].set(xlabel="Lapse probability (%)", ylabel="Median Fisher/PR query ratio")
    axes[1].plot(100 * data["lapse"], data["left_success_rate"], color=COLORS["PR-EIG"], marker="o", label="PR-EIG")
    axes[1].plot(100 * data["lapse"], data["right_success_rate"], color=COLORS["Fisher-RC"], marker="^", label="Fisher-RC")
    axes[1].set(xlabel="Lapse probability (%)", ylabel="Environment success rate", ylim=(0.85, 1.0))
    axes[1].legend(frameon=False)
    for ax in axes:
        _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_a_relative(root: Path, path: Path) -> None:
    data = pd.read_csv(root / "controlled" / "exact_fisher_relative.csv")
    fig, ax = plt.subplots(figsize=(4.3, 3.1), layout="constrained")
    ax.semilogx(data["sigma"], data["fisher_over_exact"], color=COLORS["Fisher-RC"])
    ax.axhline(1.0, color="0.35", ls="--", lw=0.9)
    ax.set(xlabel=r"Posterior width $\sigma$", ylabel=r"$I_{Fisher}/I_{Exact}$")
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_thresholds(root: Path, path: Path) -> None:
    data = pd.read_csv(root / "randomized" / "threshold_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), layout="constrained")
    for method in METHODS:
        group = data[data["method"] == method].sort_values("threshold", ascending=False)
        axes[0].plot(group["threshold"], group["environment_success_rate"], color=COLORS[method], marker=MARKERS[method], label=method)
    ratio = data[data["method"] == "Fisher/PR ratio"].sort_values("threshold", ascending=False)
    axes[1].plot(ratio["threshold"], ratio["median_query_count_plot"], color=COLORS["Fisher-RC"], marker="o")
    axes[1].axhline(1.0, color="0.35", ls="--", lw=0.9)
    axes[0].set(xlabel=r"Threshold $\rho$", ylabel="Environment success rate")
    axes[1].set(xlabel=r"Threshold $\rho$", ylabel="Median Fisher/PR query ratio")
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.invert_xaxis()
        _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _appendix_runtime(root: Path, path: Path) -> None:
    data = pd.read_csv(root / "randomized" / "uncached_design_benchmark.csv").set_index("method").loc[list(METHODS)]
    fig, ax = plt.subplots(figsize=(4.5, 3.1), layout="constrained")
    x = np.arange(len(data))
    ax.bar(x, 1e3 * data["median_uncached_design_seconds"], color=[COLORS[m] for m in data.index])
    ax.set_xticks(x, data.index, rotation=15)
    ax.set_ylabel("Median uncached design time (ms)")
    _clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_precision_sweep(config: Experiment1Config) -> dict[str, Path]:
    """Update Figure 2(c), combined Figure 2, and the four requested diagnostics.

    Individual panels (a) and (b) are intentionally not rewritten.
    """
    if config.mode != "full":
        raise ValueError("publication plotting requires the saved full profile")
    _style()
    root = config.output_root
    source_root = Path(__file__).resolve().parents[1]
    figure_dir = source_root / "outputs" / "figure2" / "figures"
    appendix_dir = (
        source_root / "outputs" / "figure2" / "supplementary" / "experiment1_redesign"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "figure2c": figure_dir / "figure2c_final.pdf",
        "combined": figure_dir / "figure2_combined_final.pdf",
        "preview": figure_dir / "figure2_combined_final.png",
        "scatter_rho025": appendix_dir / "appendix_figure_c_scatter_rho025.pdf",
        "success": appendix_dir / "appendix_figure_c_success_vs_threshold.pdf",
        "counts": appendix_dir / "appendix_figure_c_counts_vs_threshold.pdf",
        "differences": appendix_dir / "appendix_figure_c_diff_vs_threshold.pdf",
    }
    _save_panel(draw_panel_c, root, outputs["figure2c"], "(c)")
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.72), layout="constrained")
    for ax, draw, label in zip(
        axes, (draw_panel_a, draw_panel_b, draw_panel_c), ("(a)", "(b)", "(c)")
    ):
        draw(ax, root)
        _panel_label(ax, label)
    fig.savefig(outputs["combined"], bbox_inches="tight")
    fig.savefig(outputs["preview"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    _appendix_scatter_rho025(root, outputs["scatter_rho025"])
    _appendix_success_vs_threshold(root, outputs["success"])
    _appendix_counts_vs_threshold(root, outputs["counts"])
    _appendix_diff_vs_threshold(root, outputs["differences"])

    inputs = (
        root / "randomized" / "precision_sweep" / "threshold_summary.csv",
        root / "randomized" / "precision_sweep" / "environment_thresholds.parquet",
        root / "randomized" / "paired_pr_eig_vs_fisher_rc_lapse_0.00.csv",
        root / "controlled" / "curve.parquet",
        root / "sequential" / "curve.csv",
    )
    metadata = {
        "statement": (
            "Figure 2(c) and its diagnostics were generated from saved full trajectories; "
            "no simulation, smoothing, or post-hoc threshold selection occurred in plotting."
        ),
        "inputs": {
            str(path.relative_to(source_root)): _sha256(path) for path in inputs
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
        "config_hash": config.config_hash,
        "panels_a_b_rewritten": False,
    }
    metadata_path = figure_dir / "figure2c_precision_sweep_metadata.json"
    atomic_json(metadata_path, metadata)
    outputs["metadata"] = metadata_path
    return outputs


def plot_all(config: Experiment1Config) -> dict[str, Path]:
    if config.mode != "full":
        raise ValueError("publication plotting requires the saved full profile")
    _style()
    root = config.output_root
    source_root = Path(__file__).resolve().parents[1]
    figure_dir = source_root / "outputs" / "figure2" / "figures"
    appendix_dir = source_root / "outputs" / "figure2" / "supplementary" / "experiment1_redesign"
    figure_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {
        "figure2a": figure_dir / "figure2a_final.pdf",
        "figure2b": figure_dir / "figure2b_final.pdf",
        "figure2c": figure_dir / "figure2c_final.pdf",
        "combined": figure_dir / "figure2_combined_final.pdf",
        "preview": figure_dir / "figure2_combined_final.png",
        "noiseless_representative_ci": figure_dir / "noiseless_representative_ci.pdf",
    }
    _save_panel(draw_panel_a, root, outputs["figure2a"], "(a)")
    _save_panel(draw_panel_b, root, outputs["figure2b"], "(b)")
    _save_panel(draw_panel_c, root, outputs["figure2c"], "(c)")
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.05), layout="constrained")
    for ax, draw, label in zip(axes, (draw_panel_a, draw_panel_b, draw_panel_c), ("(a)", "(b)", "(c)")):
        draw(ax, root)
        _panel_label(ax, label)
    fig.savefig(outputs["combined"], bbox_inches="tight")
    fig.savefig(outputs["preview"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    _noiseless_representative_ci(
        source_root,
        outputs["noiseless_representative_ci"],
    )

    appendix = {
        "noiseless": appendix_dir / "noiseless_theorem_verification.pdf",
        "sequential_windows": appendix_dir / "sequential_slope_window_sensitivity.pdf",
        "beta": appendix_dir / "randomized_performance_vs_beta.pdf",
        "arity": appendix_dir / "randomized_performance_vs_arity.pdf",
        "dimension": appendix_dir / "randomized_performance_vs_dimension.pdf",
        "anisotropy": appendix_dir / "randomized_performance_vs_anisotropy.pdf",
        "exact_fisher": appendix_dir / "randomized_exact_vs_fisher_scatter.pdf",
        "entropy": appendix_dir / "noiseless_random_vs_entropy_tie_sanity.pdf",
        "particles": appendix_dir / "particle_count_convergence.pdf",
        "mismatch": appendix_dir / "model_mismatch_lapse_robustness.pdf",
        "a_relative": appendix_dir / "controlled_fisher_vs_exact_relative_information.pdf",
        "thresholds": appendix_dir / "randomized_secondary_thresholds.pdf",
        "runtime": appendix_dir / "randomized_uncached_runtime_comparison.pdf",
        "c_scatter_rho025": appendix_dir / "appendix_figure_c_scatter_rho025.pdf",
        "c_success": appendix_dir / "appendix_figure_c_success_vs_threshold.pdf",
        "c_counts": appendix_dir / "appendix_figure_c_counts_vs_threshold.pdf",
        "c_differences": appendix_dir / "appendix_figure_c_diff_vs_threshold.pdf",
    }
    _appendix_noiseless(source_root, appendix["noiseless"])
    _appendix_sequential_windows(root, appendix["sequential_windows"])
    sensitivity = pd.read_csv(root / "randomized" / "sensitivity.csv")
    _save_sensitivity(sensitivity, "beta", appendix["beta"], r"Inverse temperature $\beta$")
    _save_sensitivity(sensitivity, "arity", appendix["arity"], "Slate arity M")
    _save_sensitivity(sensitivity, "dimension", appendix["dimension"], "Dimension d")
    _save_sensitivity(sensitivity, "anisotropy", appendix["anisotropy"], r"Condition number $\kappa$")
    _appendix_exact_fisher_scatter(root, appendix["exact_fisher"])
    _appendix_entropy_sanity(source_root, appendix["entropy"])
    _appendix_particle_convergence(root, appendix["particles"])
    _appendix_mismatch(root, appendix["mismatch"])
    _appendix_a_relative(root, appendix["a_relative"])
    _appendix_thresholds(root, appendix["thresholds"])
    _appendix_runtime(root, appendix["runtime"])
    _appendix_scatter_rho025(root, appendix["c_scatter_rho025"])
    _appendix_success_vs_threshold(root, appendix["c_success"])
    _appendix_counts_vs_threshold(root, appendix["c_counts"])
    _appendix_diff_vs_threshold(root, appendix["c_differences"])
    outputs.update({f"appendix_{key}": value for key, value in appendix.items()})
    metadata = {
        "statement": "All panels were generated from saved full result tables; no simulation or random draw occurred in plotting.",
        "inputs": {
            str(path.relative_to(source_root)): _sha256(path)
            for path in (
                root / "controlled" / "curve.parquet",
                root / "controlled" / "slopes.csv",
                root / "sequential" / "curve.csv",
                root / "sequential" / "slopes.csv",
                root / "randomized" / "environment_aggregated.parquet",
                root / "randomized" / "paired_summary.csv",
            )
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
        "config_hash": config.config_hash,
    }
    metadata_path = figure_dir / "figure2_final_metadata.json"
    atomic_json(metadata_path, metadata)
    outputs["metadata"] = metadata_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full",), default="full")
    parser.add_argument(
        "--figure-c-only",
        action="store_true",
        help="rewrite only panel (c), the combined figure, and panel-(c) diagnostics",
    )
    args = parser.parse_args()
    config = get_experiment1_config(args.mode)
    outputs = plot_precision_sweep(config) if args.figure_c_only else plot_all(config)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
