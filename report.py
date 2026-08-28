"""Generate a compact machine-derived Markdown run report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from figure2_experiments.config import ExperimentConfig
from figure2_experiments.exp_b_sequential.design_cache import METHODS


def _fmt(value: float, digits: int = 4) -> str:
    return "not estimable at this horizon" if not np.isfinite(value) else f"{value:.{digits}f}"


def generate_report(config: ExperimentConfig) -> Path:
    root = config.output_root
    a_meta = json.loads((root / "exp_a" / "metadata.json").read_text())
    b_meta = json.loads((root / "exp_b" / "metadata.json").read_text())
    c_meta = json.loads((root / "exp_c" / "metadata.json").read_text())
    a = pd.read_csv(root / "exp_a" / "slopes.csv")
    a = a[
        (a["beta"] == config.main_beta)
        & (a["M"] == config.main_m)
        & np.isclose(a["sigma_fit_max"], config.primary_sigma_fit_max)
    ]
    b = pd.read_csv(root / "exp_b" / "slopes.csv")
    b = b[b["fit_min_n"] == config.primary_sequential_fit_min]
    c = pd.read_csv(root / "exp_c" / "slopes.csv")
    robustness = pd.read_csv(root / "exp_a" / "robustness.csv")
    lines = [
        "# Figure 2 simulation report",
        "",
        f"Execution profile: `{config.mode}`; config hash: `{config.config_hash}`.",
        "",
        "## Experiment A - finite-beta noisy MNL information collapse",
        "",
        f"- beta = {a_meta['beta_main']}",
        f"- M = {a_meta['M_main']}",
        f"- sigma range = {a_meta['sigma_range']}",
        f"- quadrature order = {a_meta['quadrature_order']}",
        f"- alpha range = {a_meta['alpha_range']}",
        f"- initial physical r range = {a_meta['physical_r_initial_range']} (automatically expanded only on boundary contact)",
        f"- bounded PR optimum touched alpha=20 at {a_meta['pr_boundary_count']} of {a_meta['sigma_points']} main-grid points; expanded-alpha sensitivity is retained in raw output",
        "",
    ]
    for method in METHODS:
        row = a[a["method"] == method]
        slope = float(row.iloc[0]["slope"]) if not row.empty else np.nan
        theory = 4.0 if method == "PR-EIG" else 2.0
        lines.append(f"- {method} slope = {_fmt(slope)} (theory {theory:g})")
    even_boundary = robustness[
        (robustness["method"] == "Exact-RC")
        & (robustness["M"].isin([2, 8]))
        & np.isclose(robustness["sigma_fit_max"], config.primary_sigma_fit_max)
        & (robustness["boundary_at_smallest_sigma"] == True)
    ]
    odd_calibration = robustness[
        (robustness["method"] == "Exact-RC")
        & (robustness["M"].isin([3, 5]))
        & np.isclose(robustness["sigma_fit_max"], config.primary_sigma_fit_max)
    ]
    lines.extend(["", "### Resolution robustness"])
    if not odd_calibration.empty:
        for m, group in odd_calibration.groupby("M"):
            values = group["r_sqrt_beta_smallest_sigma"].to_numpy(float)
            lines.append(
                f"- M={m}: r*sqrt(beta) mean {_fmt(values.mean(), 6)}, range "
                f"[{_fmt(values.min(), 6)}, {_fmt(values.max(), 6)}]."
            )
    if not even_boundary.empty:
        lines.append(
            "- Important failure mode: for the symmetric even-M templates M=2 and M=8, "
            "the physical/Fisher optimum remained on the repeatedly expanded upper boundary. "
            "The central equidistant pair makes local Fisher information grow with radius, so "
            "a finite nonzero calibration optimum and the sigma^2 robustness claim do not apply "
            "to these even-M settings. Their raw slopes and boundary flags are preserved."
        )
    lines.extend(
        [
            "",
            "## Experiment B - finite-beta noisy MNL sequential localization",
            "",
            f"- trajectories = {b_meta['trajectories']}",
            f"- N_max = {b_meta['N_max']}",
            f"- posterior method = {b_meta['posterior_method']}",
            f"- quadrature order = {b_meta['quadrature_order']}",
            f"- primary fit range = {b_meta['fit_range_primary']}",
            "",
        ]
    )
    for method in METHODS:
        row = b[b["method"] == method]
        if row.empty:
            lines.append(f"- {method} slope = not estimable at this horizon")
        else:
            item = row.iloc[0]
            lines.append(
                f"- {method} slope = {_fmt(item['bootstrap_mean'], 6)}; "
                f"95% CI [{_fmt(item['ci_low'], 6)}, {_fmt(item['ci_high'], 6)}]"
            )
    if not b.empty:
        pr_row = b[b["method"] == "PR-EIG"]
        if not pr_row.empty and abs(float(pr_row.iloc[0]["bootstrap_mean"]) + 0.25) > 0.03:
            lines.append(
                "- Pre-asymptotic disclosure: the pre-registered PR fit has not reached -1/4. "
                "The sliding slope should be inspected at the final checkpoint; no fit window was changed after seeing this result."
            )
    lines.extend(
        [
            "",
            "Sliding slopes are stored in `exp_b/diagnostics/local_slopes.csv`; ADF/exact-grid discrepancies are stored in `exp_b/exact_grid_validation.csv`.",
            "",
            "## Experiment C - separate noiseless M-way allocation benchmark",
            "",
            f"Targets per (d,M) = {c_meta['targets_per_setting']}",
            "",
            "| d | M | RC empirical slope | theory slope | difference | 95% CI | random max rel. discrepancy | lower-bound violations |",
            "|---:|---:|---:|---:|---:|:---:|---:|---:|",
        ]
    )
    for row in c.itertuples(index=False):
        lines.append(
            f"| {row.d} | {row.M} | {row.bootstrap_mean_slope:.5f} | {row.theory_slope:.5f} | "
            f"{row.difference:.3g} | [{row.ci_low:.5f}, {row.ci_high:.5f}] | "
            f"{row.random_max_relative_theory_discrepancy:.3g} | {row.lower_bound_violations} |"
        )
    lines.extend(
        [
            "",
            "The noisy finite-beta experiments (A/B) and the noiseless allocation benchmark (C) are intentionally analyzed as different theorems. Dashed/dotted plot curves are analytic guides or theory, never generated empirical observations.",
            "",
        ]
    )
    output = root / "REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
