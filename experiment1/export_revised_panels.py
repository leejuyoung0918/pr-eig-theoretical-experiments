"""Export the five reviewer-requested figure PDFs as standalone assets."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure2_experiments.experiment1.config import get_experiment1_config
from figure2_experiments.experiment1.plot import (
    COLORS,
    _clean,
    _fmt_sig,
    _noiseless_representative_ci,
    _panel_label,
    _style,
    draw_panel_a,
    draw_panel_b,
    draw_panel_c,
)


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "pdf" / "revised_figures"


def _save_figure2_panel(draw, root: Path, path: Path, label: str, note: str | None) -> None:
    fig, ax = plt.subplots(figsize=(3.75, 3.05), layout="constrained")
    draw(ax, root)
    _panel_label(ax, label)
    if note:
        ax.text(
            0.98,
            0.045,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.9,
            linespacing=1.12,
            color="0.25",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.82",
                "linewidth": 0.6,
                "alpha": 0.94,
            },
            zorder=8,
        )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_figure8_right(source_root: Path, path: Path) -> None:
    slopes = pd.read_csv(source_root / "outputs" / "figure2" / "exp_c" / "slopes.csv")
    dimensions = sorted(int(value) for value in slopes["d"].unique())
    arities = sorted(int(value) for value in slopes["M"].unique())
    group_colors = {4: "#4C78A8", 8: "#59A14F", 16: "#E17C05"}

    positions: list[float] = []
    values: list[float] = []
    colors: list[str] = []
    tick_labels: list[str] = []
    group_centers: dict[int, float] = {}
    for group_index, dimension in enumerate(dimensions):
        group_positions: list[float] = []
        for arity_index, arity in enumerate(arities):
            position = group_index * (len(arities) + 1) + arity_index
            row = slopes[(slopes["d"] == dimension) & (slopes["M"] == arity)].iloc[0]
            positions.append(float(position))
            values.append(100.0 * float(row["entropy_random_max_relative_discrepancy"]))
            colors.append(group_colors[dimension])
            tick_labels.append(f"M={arity}")
            group_positions.append(float(position))
        group_centers[dimension] = float(np.mean(group_positions))

    fig, ax = plt.subplots(figsize=(7.2, 3.65), layout="constrained")
    bars = ax.bar(positions, values, color=colors, width=0.76, edgecolor="white", linewidth=0.5)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.055,
            _fmt_sig(value),
            ha="center",
            va="bottom",
            fontsize=8.0,
        )
    ax.set_xticks(positions, tick_labels)
    ax.set_ylabel("Max entropy-tie/random-coordinate discrepancy (%)")
    ax.set_ylim(0.0, max(values) * 1.18)
    ax.yaxis.grid(True, color="0.90", linewidth=0.65)
    ax.set_axisbelow(True)
    for group_index in range(1, len(dimensions)):
        ax.axvline(group_index * (len(arities) + 1) - 1.0, color="0.86", lw=0.8)
    for dimension, center in group_centers.items():
        ax.text(
            center,
            1.01,
            f"d={dimension}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9.3,
            fontweight="bold",
        )
    ax.text(
        0.01,
        0.98,
        "Bar labels: 3 significant digits",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="0.30",
    )
    _clean(ax)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    _style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = get_experiment1_config("full")
    root = config.output_root
    source_root = Path(__file__).resolve().parents[1]

    _save_figure2_panel(
        draw_panel_a,
        root,
        OUTPUT_DIR / "figure2a_revised.pdf",
        "(a)",
        "Log-log axes\nslopes fit on log I vs log sigma",
    )
    _save_figure2_panel(
        draw_panel_b,
        root,
        OUTPUT_DIR / "figure2b_revised.pdf",
        "(b)",
        "Shading: pointwise 95% bootstrap CI\n2,000 resamples; 200 repetitions",
    )
    _save_figure2_panel(
        draw_panel_c,
        root,
        OUTPUT_DIR / "figure2c_revised.pdf",
        "(c)",
        None,
    )
    _noiseless_representative_ci(
        source_root,
        OUTPUT_DIR / "figure8_left_noiseless_representative_ci_revised.pdf",
    )
    _save_figure8_right(
        source_root,
        OUTPUT_DIR / "figure8_right_entropy_tie_sanity_revised.pdf",
    )


if __name__ == "__main__":
    main()
