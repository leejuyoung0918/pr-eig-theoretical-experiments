"""Single entry point for smoke, paper, and exhaustive profiles."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Support both `python figure2_experiments/run_all.py` and `cd ...; python run_all.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figure2_experiments.config import get_config
from figure2_experiments.exp_a_information.analyze import analyze as analyze_a
from figure2_experiments.exp_a_information.run import run as run_a
from figure2_experiments.exp_b_sequential.analyze import analyze as analyze_b
from figure2_experiments.exp_b_sequential.run import run as run_b
from figure2_experiments.exp_c_noiseless.analyze import analyze as analyze_c
from figure2_experiments.exp_c_noiseless.run import run as run_c
from figure2_experiments.plotting.plot_figure2 import plot
from figure2_experiments.report import generate_report
from figure2_experiments.utils import atomic_json, ensure_output_tree, run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "paper", "exhaustive"), default="paper")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="recompute completed checkpoints")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("a", "b", "c"),
        default=("a", "b", "c"),
    )
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    config = get_config(args.mode)
    ensure_output_tree(config.output_root)
    metadata = run_metadata(config.as_dict(), Path(__file__).resolve().parent.parent)
    metadata["status"] = "running"
    metadata["selected_experiments"] = args.experiments
    metadata["started_monotonic"] = time.monotonic()
    atomic_json(config.output_root / "run_metadata.json", metadata)
    results: dict[str, dict[str, str]] = {}
    try:
        if "a" in args.experiments:
            results["experiment_a_run"] = {key: str(value) for key, value in run_a(config, force=args.force).items()}
            results["experiment_a_analysis"] = {key: str(value) for key, value in analyze_a(config).items()}
        if "b" in args.experiments:
            results["experiment_b_run"] = {key: str(value) for key, value in run_b(config, workers=args.workers, force=args.force).items()}
            results["experiment_b_analysis"] = {key: str(value) for key, value in analyze_b(config).items()}
        if "c" in args.experiments:
            results["experiment_c_run"] = {key: str(value) for key, value in run_c(config, workers=args.workers, force=args.force).items()}
            results["experiment_c_analysis"] = {key: str(value) for key, value in analyze_c(config).items()}
        if not args.skip_plot and set(args.experiments) == {"a", "b", "c"}:
            results["plotting"] = {key: str(value) for key, value in plot(config).items()}
            results["report"] = {"path": str(generate_report(config))}
        metadata["status"] = "complete"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - metadata["started_monotonic"]
        metadata.pop("started_monotonic", None)
        metadata["results"] = results
        atomic_json(config.output_root / "run_metadata.json", metadata)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
