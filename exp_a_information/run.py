"""Run deterministic Gauss-Hermite information-collapse experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from figure2_experiments.config import ExperimentConfig, get_config
from figure2_experiments.eig import gaussian_eig
from figure2_experiments.mnl import fisher_information_at_zero
from figure2_experiments.policies import optimize_fisher, optimize_physical, optimize_pr
from figure2_experiments.templates import quantile_template
from figure2_experiments.utils import atomic_json, atomic_npz, ensure_output_tree, stable_hash


METHODS = ("PR-EIG", "Exact-RC", "Fisher-RC")


def _setting_path(config: ExperimentConfig, beta: float, m: int, points: int) -> Path:
    key = stable_hash(
        {
            "experiment": "A",
            "mode": config.mode,
            "beta": beta,
            "m": m,
            "points": points,
            "profile": config.profile.__dict__,
            "alpha": [config.alpha_min, config.alpha_max],
            "r": [config.physical_r_min, config.physical_r_max],
        }
    )[:16]
    return config.output_root / "exp_a" / "raw" / f"setting_b{beta:g}_m{m}_{key}.npz"


def _run_setting(
    config: ExperimentConfig,
    beta: float,
    m: int,
    sigma_grid: np.ndarray,
    *,
    expanded_sensitivity: bool,
) -> pd.DataFrame:
    profile = config.profile
    checkpoint = _setting_path(config, beta, m, len(sigma_grid))
    b = quantile_template(m)
    n = len(sigma_grid)
    arrays = {
        "pr_alpha": np.full(n, np.nan),
        "pr_r": np.full(n, np.nan),
        "pr_eig": np.full(n, np.nan),
        "pr_boundary": np.zeros(n, dtype=bool),
        "pr_expanded_alpha": np.full(n, np.nan),
        "pr_expanded_eig": np.full(n, np.nan),
        "exact_r": np.full(n, np.nan),
        "exact_eig": np.full(n, np.nan),
        "exact_boundary": np.zeros(n, dtype=bool),
        "fisher_eig": np.full(n, np.nan),
        "done": np.zeros(n, dtype=bool),
    }
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if np.array_equal(saved["sigma"], sigma_grid):
                for key in arrays:
                    if key in saved:
                        arrays[key][...] = saved[key]

    fisher = optimize_fisher(
        b,
        beta,
        config.physical_r_min,
        config.physical_r_max,
        profile.a_r_grid_points,
    )
    fisher_r = fisher.value
    started = time.monotonic()
    for index, sigma in enumerate(sigma_grid):
        if arrays["done"][index]:
            continue
        pr = optimize_pr(
            float(sigma),
            b,
            beta,
            config.alpha_min,
            config.alpha_max,
            profile.a_alpha_grid_points,
            profile.a_gh_order,
        )
        exact = optimize_physical(
            float(sigma),
            b,
            beta,
            config.physical_r_min,
            config.physical_r_max,
            profile.a_r_grid_points,
            profile.a_gh_order,
        )
        arrays["pr_alpha"][index] = pr.value
        arrays["pr_r"][index] = pr.value * sigma
        arrays["pr_eig"][index] = pr.objective
        arrays["pr_boundary"][index] = pr.boundary
        arrays["exact_r"][index] = exact.value
        arrays["exact_eig"][index] = exact.objective
        arrays["exact_boundary"][index] = exact.boundary
        arrays["fisher_eig"][index] = gaussian_eig(
            float(sigma), fisher_r, b, beta, profile.a_gh_order
        )
        if expanded_sensitivity and pr.boundary:
            expanded = optimize_pr(
                float(sigma),
                b,
                beta,
                config.expanded_alpha_min,
                config.expanded_alpha_max,
                profile.a_alpha_grid_points,
                profile.a_gh_order,
            )
            arrays["pr_expanded_alpha"][index] = expanded.value
            arrays["pr_expanded_eig"][index] = expanded.objective
        arrays["done"][index] = True
        atomic_npz(
            checkpoint,
            sigma=sigma_grid,
            beta=np.array(beta),
            m=np.array(m),
            fisher_r=np.array(fisher_r),
            fisher_j=np.array(fisher.objective),
            elapsed_seconds=np.array(time.monotonic() - started),
            **arrays,
        )
        if (index + 1) % max(1, n // 10) == 0:
            print(f"[A beta={beta:g} M={m}] {index + 1}/{n}", flush=True)

    records: list[dict[str, float | int | str | bool]] = []
    for i, sigma in enumerate(sigma_grid):
        shared = {"beta": beta, "M": m, "sigma": float(sigma), "fisher_r": fisher_r}
        records.extend(
            [
                {
                    **shared,
                    "method": "PR-EIG",
                    "r": arrays["pr_r"][i],
                    "alpha": arrays["pr_alpha"][i],
                    "eig": arrays["pr_eig"][i],
                    "expanded_alpha": arrays["pr_expanded_alpha"][i],
                    "expanded_eig": arrays["pr_expanded_eig"][i],
                    "boundary": bool(arrays["pr_boundary"][i]),
                },
                {
                    **shared,
                    "method": "Exact-RC",
                    "r": arrays["exact_r"][i],
                    "alpha": arrays["exact_r"][i] / sigma,
                    "eig": arrays["exact_eig"][i],
                    "expanded_alpha": np.nan,
                    "expanded_eig": np.nan,
                    "boundary": bool(arrays["exact_boundary"][i]),
                },
                {
                    **shared,
                    "method": "Fisher-RC",
                    "r": fisher_r,
                    "alpha": fisher_r / sigma,
                    "eig": arrays["fisher_eig"][i],
                    "expanded_alpha": np.nan,
                    "expanded_eig": np.nan,
                    "boundary": bool(fisher.boundary),
                },
            ]
        )
    frame = pd.DataFrame.from_records(records)
    frame["r_sqrt_beta"] = frame["r"] * np.sqrt(beta)
    frame["numerical_floor"] = frame["eig"] <= config.numerical_floor
    return frame


def _quadrature_validation(config: ExperimentConfig, main: pd.DataFrame) -> pd.DataFrame:
    b = quantile_template(config.main_m)
    unique_sigma = np.geomspace(max(config.profile.a_sigma_min, 1e-4), 1.0, 9)
    records = []
    for sigma in unique_sigma:
        closest = main.iloc[(main["sigma"] - sigma).abs().argsort()[:3]]
        for method in METHODS:
            row = closest[closest["method"] == method].iloc[0]
            values = {
                order: gaussian_eig(
                    float(sigma), float(row["r"]), b, config.main_beta, order
                )
                for order in (64, 128, 256)
            }
            scale = max(values[256], config.numerical_floor)
            records.append(
                {
                    "sigma": sigma,
                    "method": method,
                    "r": row["r"],
                    "eig_64": values[64],
                    "eig_128": values[128],
                    "eig_256": values[256],
                    "relative_64_128": abs(values[64] - values[128]) / scale,
                    "relative_128_256": abs(values[128] - values[256]) / scale,
                    "passes_rtol": abs(values[128] - values[256]) / scale <= config.quadrature_rtol,
                }
            )
    return pd.DataFrame.from_records(records)


def run(config: ExperimentConfig, force: bool = False) -> dict[str, Path]:
    ensure_output_tree(config.output_root)
    sigma_grid = np.geomspace(
        config.profile.a_sigma_min, config.sigma_max, config.profile.a_sigma_points
    )
    main_path = config.output_root / "exp_a" / "raw" / "main.parquet"
    robustness_path = config.output_root / "exp_a" / "raw" / "robustness.parquet"
    if force:
        for path in config.output_root.glob("exp_a/raw/setting_*.npz"):
            path.unlink()

    main = _run_setting(
        config,
        config.main_beta,
        config.main_m,
        sigma_grid,
        expanded_sensitivity=True,
    )
    main.to_parquet(main_path, index=False)
    validation = _quadrature_validation(config, main)
    validation_path = config.output_root / "exp_a" / "quadrature_validation.csv"
    validation.to_csv(validation_path, index=False)

    robustness_frames = []
    if config.profile.a_run_robustness:
        robust_points = max(61, config.profile.a_sigma_points // 2)
        robust_sigma = np.geomspace(config.profile.a_sigma_min, config.sigma_max, robust_points)
        for beta in config.beta_grid:
            for m in config.m_grid:
                if beta == config.main_beta and m == config.main_m:
                    robustness_frames.append(main.copy())
                else:
                    robustness_frames.append(
                        _run_setting(
                            config,
                            beta,
                            m,
                            robust_sigma,
                            expanded_sensitivity=False,
                        )
                    )
    else:
        robustness_frames.append(main.copy())
    robustness = pd.concat(robustness_frames, ignore_index=True)
    robustness.to_parquet(robustness_path, index=False)

    metadata = {
        "experiment": "A",
        "mode": config.mode,
        "beta_main": config.main_beta,
        "M_main": config.main_m,
        "sigma_range": [float(sigma_grid.min()), float(sigma_grid.max())],
        "sigma_points": len(sigma_grid),
        "quadrature_order": config.profile.a_gh_order,
        "alpha_range": [config.alpha_min, config.alpha_max],
        "expanded_alpha_range": [config.expanded_alpha_min, config.expanded_alpha_max],
        "physical_r_initial_range": [config.physical_r_min, config.physical_r_max],
        "pr_boundary_count": int(main.query("method == 'PR-EIG'")["boundary"].sum()),
        "physical_boundary_count": int(main.query("method == 'Exact-RC'")["boundary"].sum()),
        "quadrature_failures": int((~validation["passes_rtol"]).sum()),
        "config_hash": config.config_hash,
    }
    atomic_json(config.output_root / "exp_a" / "metadata.json", metadata)
    return {
        "main": main_path,
        "robustness": robustness_path,
        "validation": validation_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "paper", "exhaustive"), default="paper")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run(get_config(args.mode), force=args.force)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
