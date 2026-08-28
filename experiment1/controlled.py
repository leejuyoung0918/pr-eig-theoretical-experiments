"""Controlled deterministic information-scaling experiment for Figure 2(a)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from figure2_experiments.eig import gaussian_eig
from figure2_experiments.policies import optimize_fisher, optimize_physical, optimize_pr
from figure2_experiments.templates import quantile_template
from figure2_experiments.utils import atomic_json, atomic_npz

from .config import Experiment1Config
from .statistics import fixed_exponent_intercept, log_linear_regression


METHODS = ("PR-EIG", "Exact-RC", "Fisher-RC")


def run_controlled(config: Experiment1Config, force: bool = False) -> dict[str, Path]:
    root = config.output_root / "controlled"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "raw_checkpoint.npz"
    if force and checkpoint.exists():
        checkpoint.unlink()
    sigma = np.geomspace(
        config.a_sigma_min, config.a_sigma_max, config.profile.a_sigma_points
    )
    shape = (len(METHODS), len(sigma))
    radius = np.full(shape, np.nan)
    eig = np.full(shape, np.nan)
    boundary = np.zeros(shape, dtype=bool)
    evaluations = np.zeros(shape, dtype=np.int32)
    search_min = np.full(shape, np.nan)
    search_max = np.full(shape, np.nan)
    done = np.zeros(shape, dtype=bool)
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if np.array_equal(saved["sigma"], sigma):
                for name, target in (
                    ("radius", radius), ("eig", eig), ("boundary", boundary),
                    ("evaluations", evaluations), ("search_min", search_min),
                    ("search_max", search_max), ("done", done),
                ):
                    if name in saved:
                        target[...] = saved[name]
    template = quantile_template(config.a_m)
    fisher = optimize_fisher(
        template, config.a_beta, config.physical_r_min, config.physical_r_max,
        config.profile.a_design_grid_points,
    )
    started = time.perf_counter()
    for index, width in enumerate(sigma):
        if not done[0, index]:
            optimum = optimize_pr(
                float(width), template, config.a_beta, config.alpha_min, config.alpha_max,
                config.profile.a_design_grid_points, config.profile.a_gh_order,
            )
            radius[0, index] = width * optimum.value
            eig[0, index] = optimum.objective
            boundary[0, index] = optimum.boundary
            evaluations[0, index] = optimum.evaluations
            search_min[0, index] = optimum.search_min
            search_max[0, index] = optimum.search_max
            done[0, index] = True
        if not done[1, index]:
            optimum = optimize_physical(
                float(width), template, config.a_beta,
                config.physical_r_min, config.physical_r_max,
                config.profile.a_design_grid_points, config.profile.a_gh_order,
            )
            radius[1, index] = optimum.value
            eig[1, index] = optimum.objective
            boundary[1, index] = optimum.boundary
            evaluations[1, index] = optimum.evaluations
            search_min[1, index] = optimum.search_min
            search_max[1, index] = optimum.search_max
            done[1, index] = True
        if not done[2, index]:
            radius[2, index] = fisher.value
            eig[2, index] = gaussian_eig(
                float(width), fisher.value, template, config.a_beta, config.profile.a_gh_order
            )
            boundary[2, index] = fisher.boundary
            evaluations[2, index] = fisher.evaluations
            search_min[2, index] = fisher.search_min
            search_max[2, index] = fisher.search_max
            done[2, index] = True
        atomic_npz(
            checkpoint,
            sigma=sigma,
            radius=radius,
            eig=eig,
            boundary=boundary,
            evaluations=evaluations,
            search_min=search_min,
            search_max=search_max,
            done=done,
        )
        if (index + 1) % max(1, len(sigma) // 10) == 0:
            print(f"[Experiment1 A] {index + 1}/{len(sigma)}", flush=True)

    records = []
    for method_index, method in enumerate(METHODS):
        for index, width in enumerate(sigma):
            records.append(
                {
                    "method": method,
                    "sigma": float(width),
                    "beta": config.a_beta,
                    "M": config.a_m,
                    "radius": float(radius[method_index, index]),
                    "alpha": float(radius[method_index, index] / width),
                    "eig": float(eig[method_index, index]),
                    "boundary": bool(boundary[method_index, index]),
                    "optimizer_evaluations": int(evaluations[method_index, index]),
                    "search_min": float(search_min[method_index, index]),
                    "search_max": float(search_max[method_index, index]),
                }
            )
    curve = pd.DataFrame.from_records(records)
    curve_path = root / "curve.parquet"
    curve.to_parquet(curve_path, index=False)

    validation_records = []
    for width in np.geomspace(config.a_sigma_min, config.a_sigma_max, 9):
        for method in METHODS:
            group = curve[curve["method"] == method]
            row = group.iloc[int(np.argmin(np.abs(np.log(group["sigma"]) - np.log(width))))]
            values = {
                order: gaussian_eig(
                    float(row["sigma"]), float(row["radius"]), template, config.a_beta, order
                )
                for order in (64, 128, 256)
            }
            validation_records.append(
                {
                    "method": method,
                    "sigma": float(row["sigma"]),
                    "radius": float(row["radius"]),
                    "eig_64": values[64],
                    "eig_128": values[128],
                    "eig_256": values[256],
                    "relative_128_256": abs(values[128] - values[256]) / max(values[256], 1e-300),
                }
            )
    validation_path = root / "quadrature_validation.csv"
    pd.DataFrame.from_records(validation_records).to_csv(validation_path, index=False)
    atomic_json(
        root / "run_metadata.json",
        {
            "mode": config.mode,
            "M": config.a_m,
            "beta": config.a_beta,
            "sigma_range": [config.a_sigma_min, config.a_sigma_max],
            "sigma_points": len(sigma),
            "quadrature_order": config.profile.a_gh_order,
            "alpha_range": [config.alpha_min, config.alpha_max],
            "initial_physical_range": [config.physical_r_min, config.physical_r_max],
            "elapsed_seconds": time.perf_counter() - started,
            "config_hash": config.config_hash,
        },
    )
    return {"curve": curve_path, "validation": validation_path}


def analyze_controlled(config: Experiment1Config) -> dict[str, Path]:
    root = config.output_root / "controlled"
    curve_path = root / "curve.parquet"
    curve = pd.read_parquet(curve_path)
    mask_range = curve["sigma"].between(config.a_fit_min, config.a_fit_max)
    slope_records = []
    for method in METHODS:
        group = curve[(curve["method"] == method) & mask_range].sort_values("sigma")
        fit = log_linear_regression(group["sigma"].to_numpy(), group["eig"].to_numpy())
        slope_records.append(
            {
                "method": method,
                "fitted_slope": fit["slope"],
                "standard_error": fit["slope_se"],
                "ci_low": fit["ci_low"],
                "ci_high": fit["ci_high"],
                "R2": fit["r2"],
                "n_points": fit["n_points"],
                "fit_sigma_min": config.a_fit_min,
                "fit_sigma_max": config.a_fit_max,
                "theory_slope": 4.0 if method == "PR-EIG" else 2.0,
                "difference_from_theory": fit["slope"] - (4.0 if method == "PR-EIG" else 2.0),
            }
        )
    slopes = pd.DataFrame.from_records(slope_records)
    slopes_path = root / "slopes.csv"
    slopes.to_csv(slopes_path, index=False)
    (root / "slopes.tex").write_text(
        slopes.to_latex(index=False, float_format="%.6g"), encoding="utf-8"
    )

    pr = curve[(curve["method"] == "PR-EIG") & mask_range]
    physical = curve[(curve["method"].isin(("Exact-RC", "Fisher-RC"))) & mask_range]
    guide4_intercept = fixed_exponent_intercept(pr["sigma"], pr["eig"], 4.0)
    guide2_intercept = fixed_exponent_intercept(physical["sigma"], physical["eig"], 2.0)
    x = np.sort(curve["sigma"].unique())
    guides = pd.DataFrame(
        {
            "sigma": np.concatenate((x, x)),
            "guide": ["sigma^4"] * len(x) + ["sigma^2"] * len(x),
            "fixed_exponent": [4.0] * len(x) + [2.0] * len(x),
            "intercept": [guide4_intercept] * len(x) + [guide2_intercept] * len(x),
            "eig": np.concatenate(
                (np.exp(guide4_intercept) * x**4, np.exp(guide2_intercept) * x**2)
            ),
        }
    )
    guide_path = root / "guides.csv"
    guides.to_csv(guide_path, index=False)

    exact = curve[curve["method"] == "Exact-RC"].sort_values("sigma")
    fisher = curve[curve["method"] == "Fisher-RC"].sort_values("sigma")
    relative = pd.DataFrame(
        {
            "sigma": exact["sigma"].to_numpy(),
            "fisher_over_exact": fisher["eig"].to_numpy() / exact["eig"].to_numpy(),
            "relative_difference": (fisher["eig"].to_numpy() - exact["eig"].to_numpy())
            / exact["eig"].to_numpy(),
        }
    )
    relative_path = root / "exact_fisher_relative.csv"
    relative.to_csv(relative_path, index=False)
    return {"slopes": slopes_path, "guides": guide_path, "relative": relative_path}
