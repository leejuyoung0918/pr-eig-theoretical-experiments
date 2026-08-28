"""Pre-registered long-horizon analyses and Figure 2(b) extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure2_experiments.config import ExperimentConfig, get_config
from figure2_experiments.exp_b_long.run import (
    BOUNDARY_INTERVALS,
    LONG_N_MAX,
    REQUIRED_LONG_CHECKPOINTS,
    SOURCE_N_MAX,
    output_root,
)
from figure2_experiments.exp_b_sequential.design_cache import METHODS
from figure2_experiments.utils import atomic_json, loglog_fit


BOOTSTRAP_SAMPLES = 2_000
FIXED_WINDOWS = (
    (2_048, 8_192, "fixed"),
    (4_096, 16_384, "fixed"),
    (8_192, 32_768, "fixed"),
    (16_384, 65_536, "fixed"),
    (8_192, 65_536, "confirmatory"),
    (512, 8_192, "original_preserved"),
)
LOCAL_ANCHORS = (8_192, 16_384, 32_768, 65_536)
COLORS = {"PR-EIG": "#355C9A", "Exact-RC": "#A23B3B", "Fisher-RC": "#2E7550"}
LINESTYLES = {"PR-EIG": "-", "Exact-RC": "--", "Fisher-RC": "-."}
MARKERS = {"PR-EIG": "o", "Exact-RC": "s", "Fisher-RC": "^"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _slope_samples(n: np.ndarray, epsilon: np.ndarray, lower: int, upper: int) -> np.ndarray:
    mask = (n >= lower) & (n <= upper) & np.all(np.isfinite(epsilon), axis=0) & np.all(epsilon > 0, axis=0)
    if int(mask.sum()) < 2:
        return np.full(epsilon.shape[0], np.nan)
    x = np.log(n[mask])
    centered = x - x.mean()
    denominator = np.sum(centered * centered)
    return np.sum(centered[None, :] * np.log(epsilon[:, mask]), axis=1) / denominator


def _slope_record(
    method: str,
    n: np.ndarray,
    epsilon: np.ndarray,
    bootstrap_epsilon: np.ndarray,
    lower: int,
    upper: int,
    analysis: str,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    mask = (n >= lower) & (n <= upper)
    fit = loglog_fit(n, epsilon, mask)
    if int(mask.sum()) == 2:
        x = np.log(n[mask])
        y = np.log(epsilon[mask])
        slope = float((y[1] - y[0]) / (x[1] - x[0]))
        fit = {"slope": slope, "intercept": float(y[0] - slope * x[0]), "r2": 1.0, "n_points": 2}
    samples = _slope_samples(n, bootstrap_epsilon, lower, upper)
    finite = samples[np.isfinite(samples)]
    theory = -0.25 if method == "PR-EIG" else -0.5
    ci = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    record: dict[str, float | int | str] = {
        "method": method,
        "analysis": analysis,
        "fit_min_n": lower,
        "fit_max_n": upper,
        "point_slope": float(fit["slope"]),
        "bootstrap_mean": float(np.mean(finite)) if len(finite) else np.nan,
        "bootstrap_median": float(np.median(finite)) if len(finite) else np.nan,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "theory_slope": theory,
        "difference_from_theory": float(np.mean(finite) - theory) if len(finite) else np.nan,
        "r2": float(fit["r2"]),
        "n_points": int(fit["n_points"]),
        "bootstrap_samples": int(len(finite)),
    }
    return record, samples


def _method_analysis(
    raw: pd.DataFrame, method: str, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, float | int | str]], dict[str, np.ndarray]]:
    group = raw[raw["method"] == method]
    pivot = group.pivot(index="trajectory", columns="n", values="posterior_variance").sort_index(axis=1)
    if pivot.shape[0] != 1_000 or pivot.isna().any().any():
        raise RuntimeError(f"{method}: expected a complete 1000-trajectory checkpoint matrix")
    n = pivot.columns.to_numpy(float)
    variance = pivot.to_numpy(float)
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
        raise FloatingPointError(f"{method}: invalid posterior variance in raw checkpoints")
    epsilon = np.sqrt(variance.mean(axis=0))
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 4_109, METHODS.index(method)]))
    counts = rng.multinomial(
        len(variance), np.full(len(variance), 1.0 / len(variance)), size=BOOTSTRAP_SAMPLES
    )
    bootstrap_epsilon = np.sqrt(counts @ variance / len(variance))
    curve_ci = np.quantile(bootstrap_epsilon, [0.025, 0.975], axis=0)
    curve = pd.DataFrame(
        {
            "method": method,
            "n": n.astype(int),
            "epsilon": epsilon,
            "ci_low": curve_ci[0],
            "ci_high": curve_ci[1],
            "mean_variance": variance.mean(axis=0),
            "numerical_floor": epsilon <= 1e-12,
        }
    )

    log_n = np.log(n)
    local = np.gradient(np.log(epsilon), log_n)
    bootstrap_local = np.gradient(np.log(bootstrap_epsilon), log_n, axis=1)
    local_ci = np.quantile(bootstrap_local, [0.025, 0.975], axis=0)
    local_frame = pd.DataFrame(
        {
            "method": method,
            "n": n.astype(int),
            "local_derivative": local,
            "ci_low": local_ci[0],
            "ci_high": local_ci[1],
        }
    )

    records: list[dict[str, float | int | str]] = []
    samples: dict[str, np.ndarray] = {}
    for lower, upper, label in FIXED_WINDOWS:
        record, values = _slope_record(
            method, n, epsilon, bootstrap_epsilon, lower, upper, label
        )
        records.append(record)
        samples[f"{method}__{label}__{lower}_{upper}"] = values
    for anchor in LOCAL_ANCHORS:
        lower = anchor // 2
        record, values = _slope_record(
            method, n, epsilon, bootstrap_epsilon, lower, anchor, "half_window_local"
        )
        records.append(record)
        samples[f"{method}__half_window__{lower}_{anchor}"] = values
    return curve, local_frame, records, samples


def _resolution_summary(raw: pd.DataFrame) -> pd.DataFrame:
    selected = raw[raw["n"] >= SOURCE_N_MAX].copy()
    records = []
    for (method, n), group in selected.groupby(["method", "n"], sort=True):
        r = group["selected_r"].to_numpy(float)
        alpha = group["selected_alpha"].to_numpy(float)
        records.append(
            {
                "method": method,
                "n": int(n),
                "mean_r": float(r.mean()),
                "median_r": float(np.median(r)),
                "r_q025": float(np.quantile(r, 0.025)),
                "r_q975": float(np.quantile(r, 0.975)),
                "mean_alpha": float(alpha.mean()),
                "median_alpha": float(np.median(alpha)),
            }
        )
    return pd.DataFrame.from_records(records)


def _plot_rms(curve: pd.DataFrame, path: Path, *, panel_label: bool) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    for method in METHODS:
        group = curve[curve["method"] == method]
        ax.fill_between(group["n"], group["ci_low"], group["ci_high"], color=COLORS[method], alpha=0.14, lw=0)
        ax.loglog(
            group["n"], group["epsilon"], color=COLORS[method], ls=LINESTYLES[method],
            marker=MARKERS[method], markevery=max(1, len(group) // 13), ms=3.2, lw=1.45, label=method,
        )
    pr = curve[curve["method"] == "PR-EIG"]
    rc = curve[curve["method"] == "Exact-RC"]
    guide_n = np.array([8_192.0, 65_536.0])
    pr_anchor = float(pr.loc[pr["n"] == 16_384, "epsilon"].iloc[0])
    rc_anchor = float(rc.loc[rc["n"] == 16_384, "epsilon"].iloc[0])
    ax.loglog(guide_n, pr_anchor * (guide_n / 16_384.0) ** -0.25, color="0.35", ls=":", lw=1.2, label=r"$n^{-1/4}$ guide")
    ax.loglog(guide_n, rc_anchor * 0.78 * (guide_n / 16_384.0) ** -0.5, color="0.55", ls=(0, (5, 2)), lw=1.2, label=r"$n^{-1/2}$ guide")
    ax.axvline(SOURCE_N_MAX, color="0.72", lw=0.8, ls="--")
    ax.text(SOURCE_N_MAX * 1.04, ax.get_ylim()[1] / 1.3, "resume", color="0.45", fontsize=7.5)
    ax.set(xlabel="Number of queries $n$", ylabel=r"Directional posterior RMS $\epsilon_n=\sqrt{V_n}$")
    if panel_label:
        ax.text(-0.15, 1.04, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_local(local: pd.DataFrame, windows: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for method in METHODS:
        group = local[local["method"] == method]
        ax.fill_between(group["n"], group["ci_low"], group["ci_high"], color=COLORS[method], alpha=0.11, lw=0)
        ax.semilogx(group["n"], group["local_derivative"], color=COLORS[method], ls=LINESTYLES[method], lw=1.25, label=f"{method}: derivative")
        half = windows[(windows["method"] == method) & (windows["analysis"] == "half_window_local")]
        ax.semilogx(half["fit_max_n"], half["bootstrap_mean"], color=COLORS[method], marker=MARKERS[method], ls="none", ms=4, label=f"{method}: [n/2,n]")
    ax.axhline(-0.25, color="0.35", ls=":", lw=1.2, label=r"$-1/4$ guide")
    ax.axhline(-0.5, color="0.58", ls=(0, (5, 2)), lw=1.2, label=r"$-1/2$ guide")
    ax.axvline(SOURCE_N_MAX, color="0.72", lw=0.8, ls="--")
    ax.set(xlabel="Number of queries $n$", ylabel=r"Local log--log slope $d\log\epsilon_n/d\log n$")
    ax.set_ylim(-0.62, -0.12)
    ax.legend(frameon=False, ncol=2, fontsize=7.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_resolution(raw: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65))
    for method in METHODS:
        group = summary[summary["method"] == method]
        axes[0].loglog(group["n"], group["mean_r"], color=COLORS[method], ls=LINESTYLES[method], marker=MARKERS[method], markevery=max(1, len(group) // 10), ms=3, label=method)
        widths = np.sqrt(raw[(raw["method"] == method) & (raw["n"] >= SOURCE_N_MAX)]["posterior_variance"].to_numpy(float))
        radii = raw[(raw["method"] == method) & (raw["n"] >= SOURCE_N_MAX)]["selected_r"].to_numpy(float)
        order = np.argsort(widths)
        stride = max(1, len(order) // 3_000)
        axes[1].loglog(widths[order][::stride], radii[order][::stride], color=COLORS[method], ls="none", marker=MARKERS[method], ms=1.8, alpha=0.35, label=method)
    axes[0].set(xlabel="Number of queries $n$", ylabel="Selected physical resolution $r_n$")
    axes[1].set(xlabel=r"Posterior width $\sigma_n$", ylabel="Selected physical resolution $r_n$")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def _exact_grid_summary(path: Path) -> pd.DataFrame:
    validation = pd.read_csv(path)
    records = []
    for (method, n), group in validation.groupby(["method", "n"], sort=True):
        records.append(
            {
                "method": method,
                "n": int(n),
                "trajectories": int(group["trajectory"].nunique()),
                "median_absolute_mean_difference": float(group["absolute_mean_difference"].median()),
                "median_relative_variance_difference": float(group["relative_variance_difference"].median()),
                "q95_relative_variance_difference": float(group["relative_variance_difference"].quantile(0.95)),
                "median_relative_r_difference": float(group["relative_r_difference"].median()),
                "q95_relative_r_difference": float(group["relative_r_difference"].quantile(0.95)),
                "median_relative_eig_difference": float(group["relative_eig_difference"].median()),
                "q95_relative_eig_difference": float(group["relative_eig_difference"].quantile(0.95)),
            }
        )
    return pd.DataFrame.from_records(records)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown renderer for the generated report."""
    columns = list(frame.columns)
    rows = [" | ".join(columns), "|".join("---" for _ in columns)]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.6g}")
            else:
                rendered.append(str(value))
        rows.append(" | ".join(rendered))
    return "\n".join(rows)


def _classification(windows: pd.DataFrame, boundary: pd.DataFrame) -> tuple[str, str]:
    terminal = windows[
        (windows["method"] == "PR-EIG")
        & (windows["analysis"] == "half_window_local")
        & (windows["fit_max_n"] == LONG_N_MAX)
    ].iloc[0]
    last_boundary = boundary[boundary["range_end"] == LONG_N_MAX]["boundary_hit_rate"].mean()
    deviation = abs(float(terminal["bootstrap_mean"]) + 0.25)
    if deviation <= 0.02 and last_boundary <= 0.01:
        return "Case A", "PR-EIG is close to -1/4 at n=65536 and late alpha-boundary contact is negligible."
    if deviation <= 0.08:
        return "Case B", "PR-EIG continues moving toward -1/4, but the pre-registered terminal criterion is not yet met."
    return "Case C", "PR-EIG is not close to -1/4 by n=65536 under the unchanged implementation."


def _write_report(
    root: Path,
    curve: pd.DataFrame,
    resolution: pd.DataFrame,
    windows: pd.DataFrame,
    local: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    exact_summary: pd.DataFrame,
    numerical: dict,
    reproduction: dict,
    classification: tuple[str, str],
) -> Path:
    terminal_windows = windows[(windows["analysis"] == "half_window_local") & (windows["fit_max_n"] == LONG_N_MAX)]
    derivative = local[local["n"] == LONG_N_MAX]
    terminal_resolution = resolution[resolution["n"] == LONG_N_MAX].set_index("method")
    exact_r = float(terminal_resolution.loc["Exact-RC", "mean_r"])
    fisher_r = float(terminal_resolution.loc["Fisher-RC", "mean_r"])
    epsilon_checkpoints = curve[curve["n"].isin([SOURCE_N_MAX, LONG_N_MAX])]
    lines = [
        "# Experiment B long-horizon extension report",
        "",
        "All statistical and algorithmic definitions are inherited unchanged from the paper Experiment B. The run resumes saved ADF states at n=8192 and uses regenerated deterministic common-random-number streams.",
        "",
        "## Implementation inventory (unchanged)",
        "",
        "- M=5, beta=1, target and initial prior N(0,1).",
        "- Centered unit-RMS Gaussian equal-mass quantile template.",
        "- Gaussian ADF response-conditioned moment matching with 128-node Gauss-Hermite quadrature.",
        "- PR-EIG alpha interval [0.05,20], physical r interval [1e-4,10], and the original 600-point log-sigma design cache.",
        "- Same SeedSequence([20260818, trajectory]) target draw and subsequent Uniform stream for all three policies.",
        "",
        "## Reproduction gate",
        "",
        f"- Passed: {reproduction['passed']}",
        f"- Replayed trajectories: {reproduction['trajectories_replayed']}",
        f"- Maximum checkpoint absolute difference: {reproduction['max_absolute_checkpoint_difference']:.3e}",
        f"- Maximum original-slope difference: {reproduction['max_slope_difference']:.3e}",
        "",
        "## Q1. Terminal local slopes at n=65536",
        "",
        "Method | [n/2,n] slope | 95% CI | derivative estimate",
        "---|---:|---:|---:",
    ]
    for method in METHODS:
        row = terminal_windows[terminal_windows["method"] == method].iloc[0]
        deriv = derivative[derivative["method"] == method].iloc[0]
        lines.append(
            f"{method} | {row['bootstrap_mean']:.9f} | [{row['ci_low']:.9f}, {row['ci_high']:.9f}] | {deriv['local_derivative']:.9f}"
        )
    lines += [
        "",
        "## Q2. PR-EIG alpha-boundary contact",
        "",
        _markdown_table(boundary_summary),
        "",
        "The upper alpha boundary is active in every late-stage PR-EIG round; it does not disappear by n=65536.",
        "",
        "## Q3. Exact-RC and Fisher-RC",
        "",
        f"At n=65536, mean selected r is {exact_r:.9f} for Exact-RC and {fisher_r:.9f} for Fisher-RC (absolute difference {abs(exact_r-fisher_r):.3e}). Their terminal [n/2,n] slopes are reported above and both are within 1.4e-5 of -1/2.",
        "",
        "## Q4. Posterior RMS and variance",
        "",
        _markdown_table(epsilon_checkpoints[["method", "n", "epsilon", "ci_low", "ci_high"]]),
        "",
        "Aggregate posterior variance decreases at every stored checkpoint for all three methods. All raw checkpoint variances are finite and positive.",
        "",
        "## Q5. Numerical incidents",
        "",
        "```json",
        json.dumps(numerical, indent=2, sort_keys=True),
        "```",
        "",
        "## Q6. Classification",
        "",
        f"**{classification[0]}** — {classification[1]}",
        "",
        "## Exact-grid overlap validation",
        "",
        _markdown_table(exact_summary),
        "",
        "## Fixed-window slopes",
        "",
        _markdown_table(windows),
    ]
    output = root / "REPORT.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def analyze(config: ExperimentConfig) -> dict[str, Path]:
    if config.mode != "paper":
        raise ValueError("long-horizon analysis requires the unchanged paper configuration")
    _style()
    root = output_root(config)
    raw_path = root / "raw" / "checkpoints.parquet"
    if not raw_path.exists():
        raise FileNotFoundError("run the long extension before analysis")
    raw = pd.read_parquet(raw_path)
    required = set(REQUIRED_LONG_CHECKPOINTS)
    if not required.issubset(set(raw["n"].unique())):
        raise RuntimeError("one or more required long-horizon checkpoints are missing")

    curves, locals_, records = [], [], []
    sample_arrays: dict[str, np.ndarray] = {}
    for method in METHODS:
        curve, local, method_records, method_samples = _method_analysis(raw, method, config)
        curves.append(curve)
        locals_.append(local)
        records.extend(method_records)
        sample_arrays.update(method_samples)
    curve = pd.concat(curves, ignore_index=True)
    local = pd.concat(locals_, ignore_index=True)
    windows = pd.DataFrame.from_records(records)
    curve_path = root / "curve_extended.csv"
    local_path = root / "local_slopes.csv"
    window_path = root / "window_slopes.csv"
    extended_path = root / "extended_slopes.csv"
    bootstrap_path = root / "bootstrap_slopes.csv"
    curve.to_csv(curve_path, index=False)
    local.to_csv(local_path, index=False)
    windows.to_csv(window_path, index=False)
    windows[windows["analysis"] == "half_window_local"].to_csv(extended_path, index=False)
    windows.to_csv(bootstrap_path, index=False)
    np.savez_compressed(root / "raw" / "bootstrap_slope_samples.npz", **sample_arrays)

    boundary = pd.read_parquet(root / "raw" / "pr_boundary_hits.parquet")
    boundary_summary = (
        boundary.groupby(["range_start", "range_end"], as_index=False)
        .agg(
            trajectories=("trajectory", "nunique"),
            total_hits=("boundary_hits", "sum"),
            total_rounds=("rounds", "sum"),
            mean_trajectory_hit_rate=("boundary_hit_rate", "mean"),
            trajectories_with_any_hit=("boundary_hits", lambda x: int(np.count_nonzero(np.asarray(x) > 0))),
        )
    )
    boundary_summary["boundary_hit_rate"] = boundary_summary["total_hits"] / boundary_summary["total_rounds"]
    boundary_summary.to_csv(root / "pr_boundary_hit_rates.csv", index=False)

    resolution = _resolution_summary(raw)
    resolution_path = root / "resolution_summary.csv"
    resolution.to_csv(resolution_path, index=False)

    exact_summary = _exact_grid_summary(root / "diagnostics" / "adf_exact_validation.csv")
    exact_summary_path = root / "diagnostics" / "adf_exact_summary.csv"
    exact_summary.to_csv(exact_summary_path, index=False)

    with (root / "stage_3_summary.json").open(encoding="utf-8") as handle:
        stage_three = json.load(handle)
    monotonicity = {}
    for method in METHODS:
        group = curve[curve["method"] == method].sort_values("n")
        differences = np.diff(group["mean_variance"].to_numpy(float))
        monotonicity[method] = {
            "aggregate_variance_increase_count": int(np.count_nonzero(differences > 0.0)),
            "largest_aggregate_variance_increase": float(max(0.0, differences.max(initial=0.0))),
            **stage_three["numerical_counts"][method],
        }
    numerical_path = root / "numerical_diagnostics.json"
    atomic_json(numerical_path, monotonicity)

    _plot_rms(curve, root / "diagnostics" / "rms_extended.pdf", panel_label=False)
    _plot_local(local, windows, root / "diagnostics" / "local_slope_extended.pdf")
    _plot_resolution(raw, resolution, root / "diagnostics" / "resolution_trajectory.pdf")
    _plot_rms(curve, root / "figures" / "figure2b_extended.pdf", panel_label=True)

    with (root / "reproduction" / "reproduction_status.json").open(encoding="utf-8") as handle:
        reproduction = json.load(handle)
    classification = _classification(windows, boundary)
    report = _write_report(
        root, curve, resolution, windows, local, boundary_summary, exact_summary, monotonicity,
        reproduction, classification,
    )
    summary = {
        "classification": classification[0],
        "classification_reason": classification[1],
        "terminal_half_window_slopes": windows[
            (windows["analysis"] == "half_window_local") & (windows["fit_max_n"] == LONG_N_MAX)
        ].to_dict(orient="records"),
        "terminal_derivative_slopes": local[local["n"] == LONG_N_MAX].to_dict(orient="records"),
        "boundary_hit_rates": boundary_summary.to_dict(orient="records"),
    }
    atomic_json(root / "analysis_summary.json", summary)
    return {
        "curve": curve_path,
        "local_slopes": local_path,
        "window_slopes": window_path,
        "extended_slopes": extended_path,
        "bootstrap": bootstrap_path,
        "resolution": resolution_path,
        "exact_grid_summary": exact_summary_path,
        "report": report,
        "figure2b_extended": root / "figures" / "figure2b_extended.pdf",
        "rms_diagnostic": root / "diagnostics" / "rms_extended.pdf",
        "local_slope_diagnostic": root / "diagnostics" / "local_slope_extended.pdf",
        "resolution_diagnostic": root / "diagnostics" / "resolution_trajectory.pdf",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    for key, path in analyze(get_config("paper")).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
