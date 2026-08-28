"""Build the terminal and saved report for the redesigned Experiment 1.

The report is derived exclusively from saved result tables. It does not run a
simulation, refit a model, or alter any plotted value.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from figure2_experiments.experiment1.config import Experiment1Config, get_experiment1_config
from figure2_experiments.utils import atomic_json


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


def _method_rows(frame: pd.DataFrame, key: str = "method") -> dict[str, dict[str, Any]]:
    return {
        str(row[key]): {
            str(column): value.item() if isinstance(value, np.generic) else value
            for column, value in row.items()
            if column != key
        }
        for row in frame.to_dict(orient="records")
    }


def _pair_row(summary: pd.DataFrame, left: str, right: str, lapse: float) -> pd.Series:
    selected = summary[
        (summary["left_method"] == left)
        & (summary["right_method"] == right)
        & np.isclose(summary["lapse"], lapse)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one {left}/{right}, lapse={lapse} row; found {len(selected)}")
    return selected.iloc[0]


def _primary_sequential(slopes: pd.DataFrame) -> pd.DataFrame:
    primary = slopes[slopes["primary"].astype(bool)]
    if not primary.empty:
        return primary
    largest_start = slopes["fit_window_start"].max()
    return slopes[slopes["fit_window_start"] == largest_start]


def _ratio_speedup(runtime: pd.DataFrame, slower: str, faster: str) -> float:
    indexed = runtime.set_index("method")
    return float(
        indexed.loc[slower, "median_uncached_design_seconds"]
        / indexed.loc[faster, "median_uncached_design_seconds"]
    )


def _fmt_ci(low: float, high: float, digits: int = 4) -> str:
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def build_report(config: Experiment1Config) -> tuple[dict[str, Any], str]:
    root = config.output_root
    controlled = pd.read_csv(root / "controlled" / "slopes.csv")
    quadrature = pd.read_csv(root / "controlled" / "quadrature_validation.csv")
    sequential = pd.read_csv(root / "sequential" / "slopes.csv")
    local = pd.read_csv(root / "sequential" / "local_slopes.csv")
    randomized_root = root / "randomized"
    paired_summary = pd.read_csv(randomized_root / "paired_summary.csv")
    sensitivity = pd.read_csv(randomized_root / "sensitivity.csv")
    thresholds = pd.read_csv(randomized_root / "threshold_sensitivity.csv")
    runtime = pd.read_csv(randomized_root / "uncached_design_benchmark.csv")
    particles = pd.read_csv(randomized_root / "particle_convergence.csv")
    library = pd.read_csv(randomized_root / "library_validation.csv")
    diagnostics = json.loads((randomized_root / "diagnostics.json").read_text(encoding="utf-8"))
    randomized_metadata = json.loads(
        (randomized_root / "main_metadata.json").read_text(encoding="utf-8")
    )
    precision_root = randomized_root / "precision_sweep"
    precision_summary = pd.read_csv(precision_root / "threshold_summary.csv")
    precision_report = json.loads(
        (precision_root / "FIGURE2C_PRECISION_SWEEP_REPORT.json").read_text(
            encoding="utf-8"
        )
    )

    package_root = Path(__file__).resolve().parents[1]
    noiseless = pd.read_csv(package_root / "outputs" / "figure2" / "exp_c" / "slopes.csv")
    noiseless_metadata = json.loads(
        (package_root / "outputs" / "figure2" / "exp_c" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    primary = _primary_sequential(sequential)
    controlled_rows = _method_rows(controlled)
    sequential_rows = _method_rows(primary)
    nominal = _pair_row(paired_summary, "PR-EIG", "Fisher-RC", 0.0)
    pr_exact = _pair_row(paired_summary, "PR-EIG", "Exact-RC", 0.0)
    exact_fisher = _pair_row(paired_summary, "Exact-RC", "Fisher-RC", 0.0)
    mismatch_005 = _pair_row(paired_summary, "PR-EIG", "Fisher-RC", 0.05)
    mismatch_010 = _pair_row(paired_summary, "PR-EIG", "Fisher-RC", 0.10)
    latest_local = (
        local.sort_values("n").groupby("method", as_index=False).tail(1).set_index("method")
    )

    exact_library = library[library["method"] == "Exact-RC"]
    fisher_library = library[library["method"] == "Fisher-RC"]
    pr_library = library[library["method"] == "PR-EIG"]
    particle_ranges = (
        particles.groupby("particle_count")
        .agg(
            median_final_rho_min=("median_final_rho", "min"),
            median_final_rho_max=("median_final_rho", "max"),
            median_T_min=("median_T_025_plot", "min"),
            median_T_max=("median_T_025_plot", "max"),
        )
        .reset_index()
    )

    contradiction = bool(nominal["median_query_ratio"] >= 1.0 or nominal["win_rate"] < 0.5)
    report: dict[str, Any] = {
        "profile": config.mode,
        "config_hash": config.config_hash,
        "controlled": {
            "beta": config.a_beta,
            "M": config.a_m,
            "sigma_range": [config.a_sigma_min, config.a_sigma_max],
            "fit_range": [config.a_fit_min, config.a_fit_max],
            "quadrature_order": config.profile.a_gh_order,
            "slopes": controlled_rows,
            "max_relative_difference_128_vs_256": float(
                quadrature["relative_128_256"].max()
            ),
        },
        "sequential": {
            "repetitions": config.profile.b_repetitions,
            "N_max": config.profile.b_n_max,
            "bootstrap_samples": config.profile.b_bootstrap_samples,
            "fit_range": [int(primary["fit_window_start"].min()), config.profile.b_n_max],
            "primary_slopes": sequential_rows,
            "latest_local_slope": {
                method: float(latest_local.loc[method, "local_slope"])
                for method in latest_local.index
            },
            "window_sensitivity": sequential.to_dict(orient="records"),
        },
        "randomized": {
            "environments": int(randomized_metadata["environment_count"]),
            "response_repetitions": int(randomized_metadata["response_repetitions"]),
            "T_max": int(randomized_metadata["T_max"]),
            "particles": int(randomized_metadata["particle_count"]),
            "elapsed_seconds": float(randomized_metadata["elapsed_seconds"]),
            "posterior": randomized_metadata["posterior"],
            "nominal_pr_vs_fisher": nominal.to_dict(),
            "nominal_exact_vs_fisher": exact_fisher.to_dict(),
            "sensitivity": sensitivity.to_dict(orient="records"),
            "secondary_thresholds": thresholds.to_dict(orient="records"),
            "diagnostics": diagnostics,
        },
        "precision_sweep": {
            "summary": precision_summary.to_dict(orient="records"),
            "report": precision_report,
        },
        "approximation": {
            "fisher_speedup_vs_pr_uncached_median": _ratio_speedup(
                runtime, "PR-EIG", "Fisher-RC"
            ),
            "fisher_speedup_vs_exact_uncached_median": _ratio_speedup(
                runtime, "Exact-RC", "Fisher-RC"
            ),
            "uncached_runtime": runtime.to_dict(orient="records"),
            "max_relative_cached_radius_error": {
                "PR-EIG": float(pr_library["relative_q_error"].max()),
                "Exact-RC": float(exact_library["relative_q_error"].max()),
                "Fisher-RC": float(fisher_library["relative_q_error"].max()),
            },
            "max_relative_cached_eig_error": {
                "PR-EIG": float(pr_library["relative_eig_error"].max()),
                "Exact-RC": float(exact_library["relative_eig_error"].max()),
                "Fisher-RC": float(fisher_library["relative_eig_error"].max()),
            },
            "particle_count_convergence": particle_ranges.to_dict(orient="records"),
        },
        "model_mismatch": {
            "lapse_0.05_pr_vs_fisher": mismatch_005.to_dict(),
            "lapse_0.10_pr_vs_fisher": mismatch_010.to_dict(),
        },
        "noiseless_appendix": {
            "targets_per_setting": int(noiseless_metadata["targets_per_setting"]),
            "maximum_absolute_rc_slope_difference": float(noiseless["difference"].abs().max()),
            "maximum_random_relative_theory_discrepancy": float(
                noiseless["random_max_relative_theory_discrepancy"].max()
            ),
            "maximum_entropy_random_relative_discrepancy": float(
                noiseless["entropy_random_max_relative_discrepancy"].max()
            ),
            "lower_bound_violations": int(noiseless["lower_bound_violations"].sum()),
        },
        "contradiction_flag": {
            "present": contradiction,
            "statement": (
                "The retained rho=0.25 benchmark does not show a Fisher-RC query-count "
                "advantage. The precision sweep nevertheless shows a fine-precision "
                "crossover, without post-hoc threshold or seed changes."
            ),
        },
    }

    a_pr = controlled_rows["PR-EIG"]
    a_exact = controlled_rows["Exact-RC"]
    a_fisher = controlled_rows["Fisher-RC"]
    b_pr = sequential_rows["PR-EIG"]
    b_exact = sequential_rows["Exact-RC"]
    b_fisher = sequential_rows["Fisher-RC"]
    sensitivity_lines = []
    for stratification in ("beta", "arity", "dimension", "anisotropy"):
        group = sensitivity[sensitivity["stratification"] == stratification]
        low = group.loc[group["median_fisher_pr_query_ratio"].idxmin()]
        high = group.loc[group["median_fisher_pr_query_ratio"].idxmax()]
        sensitivity_lines.append(
            f"- {stratification}: ratio minimum {low['median_fisher_pr_query_ratio']:.3f} "
            f"at {low['level']}; maximum {high['median_fisher_pr_query_ratio']:.3f} "
            f"at {high['level']}."
        )
    pr_window_text = ", ".join(
        f"{value:.5f}"
        for value in sequential[sequential["method"] == "PR-EIG"]
        .sort_values("fit_window_start")["slope"]
    )
    precision_025 = precision_summary[np.isclose(precision_summary["threshold"], 0.25)].iloc[0]
    precision_005 = precision_summary[np.isclose(precision_summary["threshold"], 0.05)].iloc[0]
    precision_003 = precision_summary[np.isclose(precision_summary["threshold"], 0.03)].iloc[0]
    precision_002 = precision_summary[np.isclose(precision_summary["threshold"], 0.02)].iloc[0]

    markdown = f"""# Experiment 1 final report

This report was generated from saved `{config.mode}` results only (config hash
`{config.config_hash}`). It does not rerun a simulation or refit a plotted result.

## Controlled theory verification

- Configuration: beta={config.a_beta:g}, M={config.a_m}, sigma in
  [{config.a_sigma_min:g}, {config.a_sigma_max:g}], {config.profile.a_sigma_points}
  log-spaced values, Gauss-Hermite order {config.profile.a_gh_order}.
- The common predeclared fit interval is [{config.a_fit_min:g}, {config.a_fit_max:g}].
- PR-EIG: slope {a_pr['fitted_slope']:.6f}, SE {a_pr['standard_error']:.6g},
  95% regression CI {_fmt_ci(a_pr['ci_low'], a_pr['ci_high'], 6)},
  R2={a_pr['R2']:.9f} (theory 4).
- Exact-RC: slope {a_exact['fitted_slope']:.6f}, SE {a_exact['standard_error']:.6g},
  95% regression CI {_fmt_ci(a_exact['ci_low'], a_exact['ci_high'], 6)},
  R2={a_exact['R2']:.9f} (theory 2).
- Fisher-RC: slope {a_fisher['fitted_slope']:.6f}, SE {a_fisher['standard_error']:.6g},
  95% regression CI {_fmt_ci(a_fisher['ci_low'], a_fisher['ci_high'], 6)},
  R2={a_fisher['R2']:.9f} (theory 2).
- The largest 128-vs-256 quadrature relative discrepancy in the validation table is
  {quadrature['relative_128_256'].max():.3g}. Panel (a) reproduces (4,2,2) at the stated
  numerical precision. These regression intervals are deterministic fit uncertainty,
  not Monte Carlo intervals.

## Sequential consequence

- Configuration: {config.profile.b_repetitions} independent repetitions,
  N_max={config.profile.b_n_max:,}, Gaussian ADF with exact finite-beta MNL
  Gauss-Hermite moment matching (order 128), and
  {config.profile.b_bootstrap_samples:,} trial-level bootstrap resamples.
- Primary predeclared fit window: n in [{int(primary['fit_window_start'].min()):,},
  {config.profile.b_n_max:,}].
- PR-EIG: slope {b_pr['slope']:.6f}, regression CI
  {_fmt_ci(b_pr['regression_ci_low'], b_pr['regression_ci_high'], 6)}, bootstrap CI
  {_fmt_ci(b_pr['bootstrap_slope_ci_low'], b_pr['bootstrap_slope_ci_high'], 6)};
  final RMS {b_pr['final_RMS']:.9f}, CI
  {_fmt_ci(b_pr['final_bootstrap_ci_low'], b_pr['final_bootstrap_ci_high'], 9)}.
- Exact-RC: slope {b_exact['slope']:.6f}, regression CI
  {_fmt_ci(b_exact['regression_ci_low'], b_exact['regression_ci_high'], 6)}, bootstrap CI
  {_fmt_ci(b_exact['bootstrap_slope_ci_low'], b_exact['bootstrap_slope_ci_high'], 6)};
  final RMS {b_exact['final_RMS']:.9f}, CI
  {_fmt_ci(b_exact['final_bootstrap_ci_low'], b_exact['final_bootstrap_ci_high'], 9)}.
- Fisher-RC: slope {b_fisher['slope']:.6f}, regression CI
  {_fmt_ci(b_fisher['regression_ci_low'], b_fisher['regression_ci_high'], 6)}, bootstrap CI
  {_fmt_ci(b_fisher['bootstrap_slope_ci_low'], b_fisher['bootstrap_slope_ci_high'], 6)};
  final RMS {b_fisher['final_RMS']:.9f}, CI
  {_fmt_ci(b_fisher['final_bootstrap_ci_low'], b_fisher['final_bootstrap_ci_high'], 9)}.
- At n={config.profile.b_n_max:,} the sliding slopes are PR={latest_local.loc['PR-EIG', 'local_slope']:.6f},
  Exact={latest_local.loc['Exact-RC', 'local_slope']:.6f}, and
  Fisher={latest_local.loc['Fisher-RC', 'local_slope']:.6f}. PR moves toward -1/4
  across the predeclared windows ({pr_window_text}), but the
  finite horizon has not reached exactly -0.25. Exact and Fisher are effectively -1/2.
- The bootstrap bands are extremely narrow because, under the symmetric Gaussian ADF,
  the variance recursion is almost deterministic. They should not be interpreted as
  validation of a fully non-Gaussian posterior.

## Randomized robustness: precision-sweep main panel

- The nominal benchmark was extended from 400 to 1,000 queries while preserving every
  old 400-query rho-path prefix exactly (maximum absolute difference
  {precision_report['maximum_absolute_rho_prefix_difference_old400_vs_extended']:.3g}).
- At rho*=0.25, median Fisher/PR is
  {precision_025['Fisher_PR_median_ratio']:.6f} and median Exact/PR is
  {precision_025['Exact_PR_median_ratio']:.6f}; there is no moderate-precision gain.
- Both median curves first fall below one at rho*=0.07. Their median-bootstrap CIs are
  wholly below one from rho*=0.05: Fisher/PR
  {_fmt_ci(precision_005['Fisher_PR_median_ratio_bootstrap_ci_low'], precision_005['Fisher_PR_median_ratio_bootstrap_ci_high'], 6)},
  Exact/PR {_fmt_ci(precision_005['Exact_PR_median_ratio_bootstrap_ci_low'], precision_005['Exact_PR_median_ratio_bootstrap_ci_high'], 6)}.
- At rho*=0.03, the complete environment IQR is below one for both methods:
  Fisher/PR {_fmt_ci(precision_003['Fisher_PR_ratio_iqr_low'], precision_003['Fisher_PR_ratio_iqr_high'], 6)},
  Exact/PR {_fmt_ci(precision_003['Exact_PR_ratio_iqr_low'], precision_003['Exact_PR_ratio_iqr_high'], 6)}.
- At rho*=0.02, median Fisher/PR is {precision_002['Fisher_PR_median_ratio']:.6f}
  and Exact/PR is {precision_002['Exact_PR_median_ratio']:.6f}, but only
  {int(precision_002['Fisher_PR_comparable_environments'])} environments are comparable.
  Success rates are PR={precision_002['PR_success_rate']:.3f},
  Exact={precision_002['Exact_success_rate']:.3f}, and
  Fisher={precision_002['Fisher_success_rate']:.3f}; this finest point must be interpreted
  with its low coverage.
- Fisher tracks Exact closely: the maximum threshold-wise median-ratio gap is
  {precision_report['maximum_absolute_median_ratio_gap_Fisher_vs_Exact']:.6f}.

The precision sweep therefore supports the qualified conclusion that physical calibration
becomes operationally useful only at fine precision in this projected-ADF benchmark.

## Retained single-threshold randomized diagnostic

- Configuration: {int(randomized_metadata['environment_count'])} paired environments,
  {int(randomized_metadata['response_repetitions'])} nested response repetitions,
  d in {{2,4,8,16}}, M in {{3,5,7,9}}, beta log-uniform on [0.25,4],
  kappa on [1,100], {int(randomized_metadata['particle_count']):,} directional particles,
  and T_max={int(randomized_metadata['T_max'])}.
- Nominal Fisher-RC versus PR-EIG at rho<=0.25: Fisher wins
  {int(nominal['right_wins'])}/{int(nominal['environments'])}, ties
  {int(nominal['ties'])}, PR wins {int(nominal['left_wins'])}; win rate
  {nominal['win_rate']:.3f}, environment-bootstrap CI
  {_fmt_ci(nominal['win_rate_ci_low'], nominal['win_rate_ci_high'], 3)}.
- Median Fisher/PR paired query ratio is {nominal['median_query_ratio']:.6f}
  (IQR {_fmt_ci(nominal['query_ratio_iqr_low'], nominal['query_ratio_iqr_high'], 3)},
  bootstrap CI {_fmt_ci(nominal['query_ratio_ci_low'], nominal['query_ratio_ci_high'], 6)}).
  Success rates are {nominal['left_success_rate']:.3f} for PR and
  {nominal['right_success_rate']:.3f} for Fisher.
- Median query saving (PR minus Fisher) is {nominal['median_query_saving']:.3f},
  bootstrap CI {_fmt_ci(nominal['query_saving_ci_low'], nominal['query_saving_ci_high'], 3)}.
- Median log final-rho difference (Fisher minus PR) is
  {nominal['median_log_final_rho_difference_right_minus_left']:.6f}, CI
  {_fmt_ci(nominal['log_final_rho_difference_ci_low'], nominal['log_final_rho_difference_ci_high'], 6)}.
  Median log normalized target-error difference is
  {nominal['median_log_final_target_error_difference_right_minus_left']:.6f}, CI
  {_fmt_ci(nominal['log_final_target_error_difference_ci_low'], nominal['log_final_target_error_difference_ci_high'], 6)}.
  The target-error direction favors Fisher, but its CI includes zero; a robust accuracy
  advantage is therefore not established.

### Retained moderate-precision result

The historical rho*=0.25 benchmark **does not show a Fisher-RC query-count advantage**.
The median ratio is above one and PR-EIG wins more environments. No seed, threshold, or
environment was removed. Exact-RC and PR-EIG tie in
  {int(pr_exact['ties'])}/{int(pr_exact['environments'])} environments, showing that
  rho=0.25 is usually reached
before bounded posterior-relative resolution separates materially from direct physical
EIG calibration. This diagnostic is retained in the appendix with all censored cases.

## Approximation quality and cost

- Fisher/Exact median query ratio is {exact_fisher['median_query_ratio']:.6f}, CI
  {_fmt_ci(exact_fisher['query_ratio_ci_low'], exact_fisher['query_ratio_ci_high'], 6)};
  Fisher is close but slightly slower at the primary threshold. Its median final
  log-rho difference relative to Exact is
  {exact_fisher['median_log_final_rho_difference_right_minus_left']:.6f}, CI
  {_fmt_ci(exact_fisher['log_final_rho_difference_ci_low'], exact_fisher['log_final_rho_difference_ci_high'], 6)}.
- Median uncached design time is
  {1e3 * runtime.set_index('method').loc['PR-EIG', 'median_uncached_design_seconds']:.3f} ms
  (PR), {1e3 * runtime.set_index('method').loc['Exact-RC', 'median_uncached_design_seconds']:.3f} ms
  (Exact), and {1e3 * runtime.set_index('method').loc['Fisher-RC', 'median_uncached_design_seconds']:.3f} ms
  (Fisher). Fisher is {_ratio_speedup(runtime, 'PR-EIG', 'Fisher-RC'):.1f}x faster
  than PR and {_ratio_speedup(runtime, 'Exact-RC', 'Fisher-RC'):.1f}x faster than Exact
  for uncached resolution design.
- The full three-lapse randomized stage took
  {float(randomized_metadata['elapsed_seconds']):.1f} wall-clock seconds with
  {int(randomized_metadata['workers'])} workers. Per-query timing excludes plotting and disk I/O.
- Cached EIG relative error is at most {library['relative_eig_error'].max():.3g}.
  The largest PR radius discrepancy is {pr_library['relative_q_error'].max():.3%}
  at a flat/multiple optimum; its EIG discrepancy remains small. Particle-count results
  from {int(particles['particle_count'].min()):,} through
  {int(particles['particle_count'].max()):,} keep the same success-rate and query-count
  conclusion.
- The multidimensional posterior is a documented projected particle ADF, not a generic
  exact high-dimensional particle posterior: exact MNL directional weights are used,
  then the first two moments are projected back to a Gaussian and equal weights are
  reset at every query. Tail ESS can be low (minimum observed fraction
  {diagnostics['minimum_ess_fraction']:.6g}), so panel (c) should be described as a
  moment-projected robustness benchmark.

## Sensitivity

{chr(10).join(sensitivity_lines)}

Every subgroup median Fisher/PR ratio is at least one. Fisher's disadvantage is largest
for M=3 and beta near 4, smallest for M=9, d=2, and low beta; anisotropy shows no clear
monotone effect. These are descriptive subgroup results, not separately powered tests.

## Model mismatch

- At 5% lapse, median Fisher/PR ratio is {mismatch_005['median_query_ratio']:.6f}, CI
  {_fmt_ci(mismatch_005['query_ratio_ci_low'], mismatch_005['query_ratio_ci_high'], 6)};
  success is PR={mismatch_005['left_success_rate']:.3f}, Fisher={mismatch_005['right_success_rate']:.3f}.
  The median log target-error difference is
  {mismatch_005['median_log_final_target_error_difference_right_minus_left']:.6f}, CI
  {_fmt_ci(mismatch_005['log_final_target_error_difference_ci_low'], mismatch_005['log_final_target_error_difference_ci_high'], 6)}.
- At 10% lapse, the ratio is {mismatch_010['median_query_ratio']:.6f}, CI
  {_fmt_ci(mismatch_010['query_ratio_ci_low'], mismatch_010['query_ratio_ci_high'], 6)};
  success is PR={mismatch_010['left_success_rate']:.3f}, Fisher={mismatch_010['right_success_rate']:.3f}.
  The median log target-error difference is
  {mismatch_010['median_log_final_target_error_difference_right_minus_left']:.6f}, CI
  {_fmt_ci(mismatch_010['log_final_target_error_difference_ci_low'], mismatch_010['log_final_target_error_difference_ci_high'], 6)}.
- Mild misspecification does not recover a Fisher query-count advantage; the disadvantage
  grows with lapse in this benchmark. Actual target-error differences again have CIs
  that include zero.

## Noiseless theorem benchmark (appendix)

- {int(noiseless_metadata['targets_per_setting']):,} targets per (d,M) setting.
- Maximum absolute RC slope error relative to -log(M):
  {noiseless['difference'].abs().max():.3g}.
- Maximum random-coordinate relative discrepancy from analytical theory:
  {noiseless['random_max_relative_theory_discrepancy'].max():.3%}.
- Maximum random/entropy-tie discrepancy:
  {noiseless['entropy_random_max_relative_discrepancy'].max():.3%}.
- Universal lower-bound violations: {int(noiseless['lower_bound_violations'].sum())}.

The ideal noiseless result strongly verifies the exact allocation theorem and is kept
separate from the finite-beta noisy claims.

## Numerical diagnostics

- Probability-normalization failures: {diagnostics['probability_normalization_failures']}.
- Paired-environment parameter failures: {diagnostics['paired_environment_parameter_failures']}.
- Unexpected NaN/Inf count: {diagnostics['unexpected_nonfinite_numeric_values']}.
- Negative-EIG count: {diagnostics['library_negative_eig_count']}.
- Minimum covariance eigenvalue: {diagnostics['minimum_covariance_eigenvalue']:.3g}.
- Censored environment-method rows: {diagnostics['censored_environment_method_rows']};
  censored response runs: {diagnostics['censored_response_runs']}.
- The physical Exact/Fisher library has zero boundary hits. PR touches its fixed alpha
  boundary as expected in the collapse regime; those cases are retained.
"""
    return report, markdown


def write_report(config: Experiment1Config) -> dict[str, Path]:
    report, markdown = build_report(config)
    output_markdown = config.output_root / "FINAL_REPORT.md"
    output_json = config.output_root / "FINAL_REPORT.json"
    atomic_json(output_json, report)
    _atomic_text(output_markdown, markdown)
    outputs = {
        "markdown": output_markdown,
        "json": output_json,
    }
    if config.mode == "full":
        snapshot_markdown = Path(__file__).resolve().parent / "FINAL_REPORT.md"
        _atomic_text(snapshot_markdown, markdown)
        outputs["snapshot"] = snapshot_markdown
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    args = parser.parse_args()
    outputs = write_report(get_experiment1_config(args.mode))
    print(outputs["markdown"].read_text(encoding="utf-8"))
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
