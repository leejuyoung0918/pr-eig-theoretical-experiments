"""Aggregate macro-seed Monte Carlo, CIs, theory comparisons, and M slopes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure2_experiments.config import ExperimentConfig, get_config
from figure2_experiments.exp_c_noiseless.run import (
    POLICIES,
    lower_bound,
    random_theory,
    rc_theory,
)
from figure2_experiments.utils import loglog_fit


def analyze(config: ExperimentConfig) -> dict[str, Path]:
    root = config.output_root / "exp_c"
    raw = pd.read_parquet(root / "raw" / "macro_summary.parquet")
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 991]))
    full_records = []
    slope_records = []
    for (d, m), setting in raw.groupby(["d", "M"], sort=True):
        macro_ids = np.sort(setting["macro"].unique())
        bootstrap_counts = rng.multinomial(
            len(macro_ids),
            np.full(len(macro_ids), 1.0 / len(macro_ids)),
            size=config.profile.c_bootstrap_samples,
        )
        policy_bootstrap: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for policy in POLICIES:
            pivot = setting[setting["policy"] == policy].pivot(
                index="macro", columns="T", values="mean_squared_error"
            ).sort_index(axis=1)
            t = pivot.columns.to_numpy(int)
            mse = pivot.to_numpy(float)
            empirical = np.sqrt(mse.mean(axis=0))
            bootstrap_rms = np.sqrt(bootstrap_counts @ mse / len(macro_ids))
            ci_low, ci_high = np.quantile(bootstrap_rms, [0.025, 0.975], axis=0)
            policy_bootstrap[policy] = (t, empirical, bootstrap_rms)
            rc_values = rc_theory(t, d, m, config.c_length)
            random_values = random_theory(t, d, m, config.c_length)
            lb_values = lower_bound(t, d, m, config.c_length)
            for index, round_index in enumerate(t):
                theory = rc_values[index] if policy == "RC-MLQ" else random_values[index]
                full_records.append(
                    {
                        "d": d,
                        "M": m,
                        "T": round_index,
                        "T_over_d": round_index / d,
                        "policy": policy,
                        "targets": config.profile.c_macro_seeds * config.profile.c_targets_per_macro,
                        "empirical_rms": empirical[index],
                        "ci_low": ci_low[index],
                        "ci_high": ci_high[index],
                        "matching_theory": theory,
                        "relative_theory_discrepancy": (empirical[index] - theory) / theory,
                        "universal_lower_bound": lb_values[index],
                        "above_lower_bound": empirical[index] + 1e-15 >= lb_values[index],
                    }
                )

        t, empirical, bootstrap_rms = policy_bootstrap["RC-MLQ"]
        endpoint_mask = (t % d == 0) & (t / d >= 2)
        x = t[endpoint_mask] / d
        point = loglog_fit(np.exp(x), empirical[endpoint_mask])
        x_centered = x - x.mean()
        denominator = np.sum(x_centered * x_centered)
        bootstrap_slopes = np.sum(
            x_centered[None, :] * np.log(bootstrap_rms[:, endpoint_mask]), axis=1
        ) / denominator
        theory_slope = -np.log(m)
        ci = np.quantile(bootstrap_slopes, [0.025, 0.975])
        random_rows = [
            row for row in full_records
            if row["d"] == d and row["M"] == m and row["policy"] == "Random Coordinate"
        ]
        entropy_rows = [
            row for row in full_records
            if row["d"] == d and row["M"] == m and row["policy"] == "Entropy-EIG Uniform Tie"
        ]
        random_discrepancy = max(abs(row["relative_theory_discrepancy"]) for row in random_rows)
        entropy_random_discrepancy = max(
            abs(e["empirical_rms"] - r["empirical_rms"]) / r["empirical_rms"]
            for e, r in zip(entropy_rows, random_rows)
        )
        slope_records.append(
            {
                "d": d,
                "M": m,
                "targets": config.profile.c_macro_seeds * config.profile.c_targets_per_macro,
                "empirical_slope": point["slope"],
                "bootstrap_mean_slope": float(bootstrap_slopes.mean()),
                "theory_slope": theory_slope,
                "difference": float(bootstrap_slopes.mean() - theory_slope),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "random_max_relative_theory_discrepancy": random_discrepancy,
                "entropy_random_max_relative_discrepancy": entropy_random_discrepancy,
                "lower_bound_violations": sum(
                    not row["above_lower_bound"]
                    for row in full_records
                    if row["d"] == d and row["M"] == m and row["policy"] == "RC-MLQ"
                ),
            }
        )
    full_grid = pd.DataFrame.from_records(full_records)
    slopes = pd.DataFrame.from_records(slope_records)
    full_path = root / "full_grid.csv"
    slopes_path = root / "slopes.csv"
    full_grid.to_csv(full_path, index=False)
    slopes.to_csv(slopes_path, index=False)

    fig, axes = plt.subplots(
        len(config.c_dimensions), len(config.c_m_values), figsize=(12.0, 8.0), sharex="row"
    )
    for row_index, d in enumerate(config.c_dimensions):
        for column_index, m in enumerate(config.c_m_values):
            ax = axes[row_index, column_index]
            subset = full_grid[(full_grid["d"] == d) & (full_grid["M"] == m)]
            rc = subset[subset["policy"] == "RC-MLQ"].sort_values("T")
            random = subset[subset["policy"] == "Random Coordinate"].sort_values("T")
            entropy = subset[subset["policy"] == "Entropy-EIG Uniform Tie"].sort_values("T")
            ax.semilogy(rc["T"], rc["empirical_rms"], color="#a33d3d", lw=1.0, label="RC empirical")
            ax.semilogy(rc["T"], rc["matching_theory"], color="#6f1d1d", ls="--", lw=0.8, label="RC theory")
            ax.semilogy(random["T"], random["empirical_rms"], color="#3b5b92", lw=0.9, label="Random empirical")
            ax.semilogy(random["T"], random["matching_theory"], color="#18345d", ls="--", lw=0.8, label="Random theory")
            ax.semilogy(entropy["T"], entropy["empirical_rms"], color="#2f7652", ls="-.", lw=0.8, label="Entropy tie")
            ax.semilogy(rc["T"], rc["universal_lower_bound"], color="0.35", ls=":", lw=0.8, label="Lower bound")
            ax.set_title(f"d={d}, M={m}", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == len(config.c_dimensions) - 1:
                ax.set_xlabel("T")
            if column_index == 0:
                ax.set_ylabel(r"$E_T$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(config.output_root / "supplementary" / "C_full_grid.png", dpi=220)
    plt.close(fig)
    return {"full_grid": full_path, "slopes": slopes_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "paper", "exhaustive"), default="paper")
    args = parser.parse_args()
    for key, path in analyze(get_config(args.mode)).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
