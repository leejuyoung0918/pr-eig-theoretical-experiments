"""Publication-ready Figure 2 from existing A/B/C result files only.

The loaders, panel renderers, and figure builder are intentionally independent so
that Experiment B can be replaced without touching or recomputing A or C.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure2_experiments.config import ExperimentConfig, get_config
from figure2_experiments.utils import atomic_json


METHODS = ("PR-EIG", "Exact-RC", "Fisher-RC")
COLORS = {"PR-EIG": "#355C9A", "Exact-RC": "#A23B3B", "Fisher-RC": "#2E7550"}
MARKERS = {"PR-EIG": "o", "Exact-RC": "s", "Fisher-RC": "^"}
LINESTYLES = {"PR-EIG": "-", "Exact-RC": "--", "Fisher-RC": "-."}
SHORT_NAMES = {"PR-EIG": "Baseline", "Exact-RC": "Exact", "Fisher-RC": "Fisher"}
COMBINED_SIZE = (7.05, 2.55)
PANEL_SIZE = (3.35, 2.55)


@dataclass(frozen=True)
class ExpAData:
    root: Path
    curve_path: Path
    slope_path: Path
    curve: pd.DataFrame
    slopes: pd.DataFrame
    beta: float
    m: int
    fit_max_sigma: float


@dataclass(frozen=True)
class ExpBData:
    root: Path
    curve_path: Path
    slope_path: Path
    curve: pd.DataFrame
    slopes: pd.DataFrame
    selected_slopes: pd.DataFrame
    annotation_title: str


@dataclass(frozen=True)
class ExpCData:
    root: Path
    curve_path: Path
    slope_path: Path
    curve: pd.DataFrame
    slopes: pd.DataFrame
    d: int
    m: int


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def load_exp_a(
    root: str | Path,
    *,
    beta: float = 1.0,
    m: int = 5,
    fit_max_sigma: float = 0.03,
) -> ExpAData:
    root = Path(root).expanduser().resolve()
    curve_path = root / "raw" / "main.parquet"
    slope_path = root / "slopes.csv"
    curve = pd.read_parquet(curve_path)
    slopes = pd.read_csv(slope_path)
    _require_columns(curve, {"beta", "M", "sigma", "method", "eig"}, curve_path)
    _require_columns(slopes, {"beta", "M", "method", "sigma_fit_max", "slope"}, slope_path)
    curve = curve[(curve["beta"] == beta) & (curve["M"] == m)].copy()
    selected = slopes[
        (slopes["beta"] == beta)
        & (slopes["M"] == m)
        & np.isclose(slopes["sigma_fit_max"], fit_max_sigma)
    ].copy()
    if set(curve["method"].unique()) < set(METHODS) or set(selected["method"].unique()) < set(METHODS):
        raise ValueError("Experiment A does not contain all three main methods for the requested setting")
    return ExpAData(root, curve_path, slope_path, curve, selected, beta, m, fit_max_sigma)


def _select_b_slopes(slopes: pd.DataFrame, n_max: int) -> tuple[pd.DataFrame, str]:
    """Select a pre-existing fit definition without fitting or changing data."""
    if "analysis" in slopes.columns:
        half_window = slopes[
            (slopes["analysis"] == "half_window_local") & (slopes["fit_max_n"] == n_max)
        ].copy()
        if set(half_window["method"].unique()) >= set(METHODS):
            return half_window, "Extended slopes"
        confirmatory = slopes[slopes["analysis"] == "confirmatory"].copy()
        if set(confirmatory["method"].unique()) >= set(METHODS):
            return confirmatory, "Extended slopes"
    if "primary" in slopes.columns:
        primary = slopes[slopes["primary"].astype(bool)].copy()
        if set(primary["method"].unique()) >= set(METHODS):
            return primary, "Late-stage slopes"
    candidates = slopes[slopes["fit_max_n"] == n_max].copy()
    if candidates.empty:
        candidates = slopes.copy()
    fit_min = candidates["fit_min_n"].max()
    selected = candidates[candidates["fit_min_n"] == fit_min].copy()
    if set(selected["method"].unique()) < set(METHODS):
        raise ValueError("Experiment B slope summary does not contain all three main methods")
    return selected, "Late-stage slopes"


def load_exp_b(root: str | Path, *, annotation_title: str | None = None) -> ExpBData:
    root = Path(root).expanduser().resolve()
    curve_candidates = (root / "curve_extended.csv", root / "curve.csv")
    slope_candidates = (root / "window_slopes.csv", root / "slopes.csv", root / "extended_slopes.csv")
    curve_path = next((path for path in curve_candidates if path.exists()), None)
    slope_path = next((path for path in slope_candidates if path.exists()), None)
    if curve_path is None or slope_path is None:
        raise FileNotFoundError(
            f"could not find a supported Experiment B curve/slope pair under {root}"
        )
    curve = pd.read_csv(curve_path)
    slopes = pd.read_csv(slope_path)
    _require_columns(curve, {"method", "n", "epsilon", "ci_low", "ci_high"}, curve_path)
    _require_columns(
        slopes,
        {"method", "fit_min_n", "fit_max_n", "bootstrap_mean", "ci_low", "ci_high"},
        slope_path,
    )
    selected, automatic_title = _select_b_slopes(slopes, int(curve["n"].max()))
    if set(curve["method"].unique()) < set(METHODS):
        raise ValueError("Experiment B curve does not contain all three main methods")
    return ExpBData(
        root,
        curve_path,
        slope_path,
        curve,
        slopes,
        selected,
        annotation_title or automatic_title,
    )


def load_exp_c(root: str | Path, *, d: int = 8, m: int = 4) -> ExpCData:
    root = Path(root).expanduser().resolve()
    curve_path = root / "full_grid.csv"
    slope_path = root / "slopes.csv"
    curve = pd.read_csv(curve_path)
    slopes = pd.read_csv(slope_path)
    _require_columns(
        curve,
        {
            "d", "M", "T", "policy", "empirical_rms", "ci_low", "ci_high",
            "matching_theory", "universal_lower_bound",
        },
        curve_path,
    )
    _require_columns(slopes, {"d", "M", "empirical_slope", "theory_slope"}, slope_path)
    curve = curve[(curve["d"] == d) & (curve["M"] == m)].copy()
    selected_slopes = slopes[(slopes["d"] == d) & (slopes["M"] == m)].copy()
    expected = {"RC-MLQ", "Random Coordinate", "Entropy-EIG Uniform Tie"}
    if set(curve["policy"].unique()) < expected:
        raise ValueError("Experiment C does not contain all representative policies")
    return ExpCData(root, curve_path, slope_path, curve, selected_slopes, d, m)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.8,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.3,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="major", direction="out", length=2.6, pad=1.8)
    ax.tick_params(which="minor", direction="out", length=1.6)
    ax.grid(False)
    ax.set_facecolor("white")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_panel_a(ax: plt.Axes, data: ExpAData) -> None:
    handles: list = []
    labels: list[str] = []
    for method in METHODS:
        group = data.curve[data.curve["method"] == method].sort_values("sigma")
        handle, = ax.loglog(
            group["sigma"],
            group["eig"],
            color=COLORS[method],
            ls=LINESTYLES[method],
            marker=MARKERS[method],
            ms=2.45,
            markevery=max(1, len(group) // 9),
            lw=1.15,
            mec="white",
            mew=0.25,
            zorder=3,
        )
        handles.append(handle)
        labels.append("Baseline" if method == "PR-EIG" else method)
    x_guide = np.geomspace(2e-4, 8e-3, 32)
    pr = data.curve[data.curve["method"] == "PR-EIG"].sort_values("sigma")
    rc = data.curve[data.curve["method"] == "Exact-RC"].sort_values("sigma")
    anchor = 2e-3
    pr_anchor = np.exp(np.interp(np.log(anchor), np.log(pr["sigma"]), np.log(pr["eig"])))
    rc_anchor = np.exp(np.interp(np.log(anchor), np.log(rc["sigma"]), np.log(rc["eig"])))
    guide4, = ax.loglog(
        x_guide, pr_anchor * (x_guide / anchor) ** 4,
        color="0.32", ls=(0, (3.2, 2.0)), lw=0.8, zorder=1,
    )
    guide2, = ax.loglog(
        x_guide, rc_anchor * 0.35 * (x_guide / anchor) ** 2,
        color="0.58", ls=":", lw=0.9, zorder=1,
    )
    handles.extend((guide4, guide2))
    labels.extend((r"$\sigma^4$ guide", r"$\sigma^2$ guide"))

    rows = data.slopes.set_index("method")
    slope_text = "Slopes\n" + "\n".join(
        f"{SHORT_NAMES[method]}  {float(rows.loc[method, 'slope']):.2f}" for method in METHODS
    )
    ax.text(
        0.98,
        0.035,
        slope_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        linespacing=1.05,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.8},
        zorder=5,
    )
    ax.set_xlabel(r"Posterior width $\sigma$")
    ax.set_ylabel("One-step mutual information (nats)")
    ax.set_xlim(float(data.curve["sigma"].min()), float(data.curve["sigma"].max()))
    ax.legend(
        handles, labels, frameon=False, loc="upper left", ncol=2,
        handlelength=1.75, handletextpad=0.45, columnspacing=0.75,
        borderaxespad=0.25, labelspacing=0.35,
    )
    _clean_axis(ax)


def plot_panel_b(ax: plt.Axes, data: ExpBData) -> None:
    handles: list = []
    labels: list[str] = []
    for method in METHODS:
        group = data.curve[data.curve["method"] == method].sort_values("n")
        x = group["n"].to_numpy(float)
        y = group["epsilon"].to_numpy(float)
        low = group["ci_low"].to_numpy(float)
        high = group["ci_high"].to_numpy(float)
        ax.fill_between(x, low, high, color=COLORS[method], alpha=0.11, lw=0, zorder=1)
        handle, = ax.loglog(
            x,
            y,
            color=COLORS[method],
            ls=LINESTYLES[method],
            marker=MARKERS[method],
            ms=2.45,
            markevery=max(1, len(group) // 8),
            lw=1.15,
            mec="white",
            mew=0.25,
            zorder=3,
        )
        handles.append(handle)
        labels.append("Baseline" if method == "PR-EIG" else method)
    n_min = max(4, int(data.curve["n"].min()))
    n_max = int(data.curve["n"].max())
    guide_start = max(n_min, n_max // 16)
    guide_n = np.geomspace(guide_start, n_max, 40)
    pr = data.curve[data.curve["method"] == "PR-EIG"].sort_values("n")
    rc = data.curve[data.curve["method"] == "Exact-RC"].sort_values("n")
    pr_anchor = np.exp(
        np.interp(np.log(guide_start), np.log(pr["n"]), np.log(pr["epsilon"]))
    )
    rc_anchor = np.exp(
        np.interp(np.log(guide_start), np.log(rc["n"]), np.log(rc["epsilon"]))
    )
    guide_pr, = ax.loglog(
        guide_n,
        pr_anchor * 0.74 * (guide_n / guide_start) ** -0.25,
        color="0.32",
        ls=(0, (3.2, 2.0)),
        lw=0.8,
        zorder=1,
    )
    guide_rc, = ax.loglog(
        guide_n,
        rc_anchor * 0.53 * (guide_n / guide_start) ** -0.5,
        color="0.58",
        ls=":",
        lw=0.9,
        zorder=1,
    )
    handles.extend((guide_pr, guide_rc))
    labels.extend((r"$n^{-1/4}$ guide", r"$n^{-1/2}$ guide"))

    rows = data.selected_slopes.set_index("method")
    slope_text = data.annotation_title + "\n" + "\n".join(
        f"{SHORT_NAMES[method]}  {float(rows.loc[method, 'bootstrap_mean']):.2f}"
        for method in METHODS
    )
    ax.text(
        0.025,
        0.035,
        slope_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        linespacing=1.05,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.8},
        zorder=5,
    )
    ax.set_xlabel(r"Number of queries $n$")
    ax.set_ylabel(r"Directional posterior RMS $\epsilon_n$")
    ax.set_xlim(float(data.curve["n"].min()), float(data.curve["n"].max()))
    ax.legend(
        handles, labels, frameon=False, loc="upper right", ncol=2,
        handlelength=1.75, handletextpad=0.45, columnspacing=0.75,
        borderaxespad=0.25, labelspacing=0.35,
    )
    _clean_axis(ax)


def plot_panel_c(ax: plt.Axes, data: ExpCData) -> None:
    group = data.curve
    rc = group[group["policy"] == "RC-MLQ"].sort_values("T")
    random = group[group["policy"] == "Random Coordinate"].sort_values("T")
    entropy = group[group["policy"] == "Entropy-EIG Uniform Tie"].sort_values("T")

    rc_theory, = ax.semilogy(
        rc["T"], rc["matching_theory"], color="#6F1D1D", ls="--", lw=0.85, zorder=4
    )
    random_theory, = ax.semilogy(
        random["T"], random["matching_theory"], color="#18345D", ls="--", lw=0.85, zorder=4
    )
    lower, = ax.semilogy(
        rc["T"], rc["universal_lower_bound"], color="0.28", ls=":", lw=0.95, zorder=1
    )

    empirical_specs = (
        (rc, "#A23B3B", "o"),
        (random, "#355C9A", "s"),
        (entropy, "#2E7550", "^"),
    )
    empirical_handles = []
    for policy_group, color, marker in empirical_specs:
        x = policy_group["T"].to_numpy(float)
        y = policy_group["empirical_rms"].to_numpy(float)
        ax.fill_between(
            x,
            policy_group["ci_low"].to_numpy(float),
            policy_group["ci_high"].to_numpy(float),
            color=color,
            alpha=0.09,
            lw=0,
            zorder=1,
        )
        handle, = ax.semilogy(
            x,
            y,
            color=color,
            ls="-",
            marker=marker,
            markevery=data.d,
            ms=2.45,
            lw=1.05,
            mec="white",
            mew=0.25,
            zorder=3,
        )
        empirical_handles.append(handle)

    handles = (
        empirical_handles[0], rc_theory,
        empirical_handles[1], random_theory,
        empirical_handles[2], lower,
    )
    labels = (
        "RC empirical", "RC exact theory",
        "Random empirical", "Random theory",
        "Entropy-tie", "Universal lower bound",
    )
    ax.set_xlabel(r"Number of noiseless queries $T$")
    ax.set_ylabel(r"Euclidean RMS $\mathcal{E}_T$")
    ax.set_xlim(float(group["T"].min()), float(group["T"].max()))
    ax.text(
        0.98,
        0.96,
        rf"$d={data.d},\ M={data.m}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
    )
    ax.legend(
        handles, labels, frameon=False, loc="lower left", ncol=1,
        handlelength=1.75, handletextpad=0.45, columnspacing=0.65,
        borderaxespad=0.25, labelspacing=0.35, fontsize=5.7,
    )
    _clean_axis(ax)


# Backwards-compatible wrappers for callers of the earlier plotting module.
def draw_a(ax: plt.Axes, config: ExperimentConfig) -> None:
    plot_panel_a(
        ax,
        load_exp_a(
            config.output_root / "exp_a",
            beta=config.main_beta,
            m=config.main_m,
            fit_max_sigma=config.primary_sigma_fit_max,
        ),
    )


def draw_b(ax: plt.Axes, config: ExperimentConfig) -> None:
    plot_panel_b(ax, load_exp_b(config.output_root / "exp_b"))


def draw_c(ax: plt.Axes, config: ExperimentConfig) -> None:
    plot_panel_c(
        ax,
        load_exp_c(config.output_root / "exp_c", d=config.c_main_d, m=config.c_main_m),
    )


def _save_panel(draw, data, path: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
    draw(ax, data)
    _panel_label(ax, label)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_figure2(
    exp_a: ExpAData,
    exp_b: ExpBData,
    exp_c: ExpCData,
    *,
    output: str | Path,
    png_dpi: int = 600,
) -> dict[str, Path]:
    """Build final panels and combined figure without changing any source data."""
    _style()
    combined = Path(output).expanduser().resolve()
    if combined.suffix.lower() != ".pdf":
        raise ValueError("--output must be a PDF path")
    combined.parent.mkdir(parents=True, exist_ok=True)
    suffix = "_final" if combined.stem.endswith("_final") else ""
    paths = {
        "a": combined.parent / f"figure2a{suffix}.pdf",
        "b": combined.parent / f"figure2b{suffix}.pdf",
        "c": combined.parent / f"figure2c{suffix}.pdf",
        "combined": combined,
        "preview": combined.with_suffix(".png"),
        "metadata": combined.parent / f"figure2{suffix}_metadata.json",
    }
    _save_panel(plot_panel_a, exp_a, paths["a"], "(a)")
    _save_panel(plot_panel_b, exp_b, paths["b"], "(b)")
    _save_panel(plot_panel_c, exp_c, paths["c"], "(c)")

    fig, axes = plt.subplots(1, 3, figsize=COMBINED_SIZE, layout="constrained")
    for ax, draw, data, label in zip(
        axes,
        (plot_panel_a, plot_panel_b, plot_panel_c),
        (exp_a, exp_b, exp_c),
        ("(a)", "(b)", "(c)"),
    ):
        draw(ax, data)
        _panel_label(ax, label)
    fig.savefig(paths["combined"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["preview"], dpi=png_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    input_files = (
        exp_a.curve_path, exp_a.slope_path,
        exp_b.curve_path, exp_b.slope_path,
        exp_c.curve_path, exp_c.slope_path,
    )
    atomic_json(
        paths["metadata"],
        {
            "statement": "Existing result files were read-only inputs; no experiment or fit was rerun.",
            "figure_size_inches": list(COMBINED_SIZE),
            "png_dpi": png_dpi,
            "sources": {str(path): _sha256(path) for path in input_files},
            "experiment_b_annotation": exp_b.annotation_title,
            "experiment_b_selected_slopes": exp_b.selected_slopes.to_dict(orient="records"),
            "outputs": {key: str(path) for key, path in paths.items() if key != "metadata"},
        },
    )
    return paths


def plot(config: ExperimentConfig) -> dict[str, Path]:
    """Preserve the original run_all API and original output filenames."""
    root = config.output_root
    exp_a = load_exp_a(
        root / "exp_a",
        beta=config.main_beta,
        m=config.main_m,
        fit_max_sigma=config.primary_sigma_fit_max,
    )
    exp_b = load_exp_b(root / "exp_b")
    exp_c = load_exp_c(root / "exp_c", d=config.c_main_d, m=config.c_main_m)
    return build_figure2(exp_a, exp_b, exp_c, output=root / "figure2" / "figure2_combined.pdf", png_dpi=600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "paper", "exhaustive"), default="paper")
    parser.add_argument("--exp-a", type=Path)
    parser.add_argument("--exp-b", type=Path)
    parser.add_argument("--exp-c", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--b-slope-title", type=str, default=None)
    parser.add_argument("--png-dpi", type=int, default=600)
    args = parser.parse_args()

    config = get_config(args.mode)
    root = config.output_root
    exp_a = load_exp_a(
        args.exp_a or root / "exp_a",
        beta=config.main_beta,
        m=config.main_m,
        fit_max_sigma=config.primary_sigma_fit_max,
    )
    exp_b = load_exp_b(args.exp_b or root / "exp_b", annotation_title=args.b_slope_title)
    exp_c = load_exp_c(args.exp_c or root / "exp_c", d=config.c_main_d, m=config.c_main_m)
    output = args.output or root / "figures" / "figure2_combined_final.pdf"
    for key, path in build_figure2(
        exp_a, exp_b, exp_c, output=output, png_dpi=args.png_dpi
    ).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
