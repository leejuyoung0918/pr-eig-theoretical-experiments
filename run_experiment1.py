"""Staged entry point for the redesigned Experiment 1."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figure2_experiments.experiment1.config import get_experiment1_config
from figure2_experiments.experiment1.controlled import analyze_controlled, run_controlled
from figure2_experiments.experiment1.randomized import (
    run_particle_convergence,
    run_precision_sweep_extension,
    run_randomized,
)
from figure2_experiments.experiment1.randomized_analysis import analyze_randomized
from figure2_experiments.experiment1.precision_sweep import analyze_precision_sweep
from figure2_experiments.experiment1.report import write_report
from figure2_experiments.experiment1.sequential import analyze_sequential, run_sequential
from figure2_experiments.utils import atomic_json, run_metadata


COMPONENTS = (
    "controlled",
    "sequential",
    "particle-convergence",
    "randomized",
    "precision-sweep",
    "analyze",
    "precision-analyze",
    "report",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--precision-environments",
        type=int,
        default=None,
        help="optional precision-sweep subset size for a pilot",
    )
    parser.add_argument(
        "--precision-t-max",
        type=int,
        default=None,
        help="optional precision-sweep horizon override",
    )
    parser.add_argument(
        "--precision-label",
        default=None,
        help="optional checkpoint label for a precision-sweep pilot",
    )
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=COMPONENTS)
    parser.add_argument(
        "--nominal-only",
        action="store_true",
        help="run lapse=0 only (intended for profiling; not a complete full run)",
    )
    args = parser.parse_args()
    config = get_experiment1_config(args.mode)
    root = config.output_root
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "config.json", config.as_dict())
    metadata = run_metadata(config.as_dict(), Path(__file__).resolve().parent.parent)
    metadata.update(
        {
            "status": "running",
            "components": args.components,
            "nominal_only": args.nominal_only,
        }
    )
    started = time.perf_counter()
    metadata_path = root / "run_metadata.json"
    atomic_json(metadata_path, metadata)
    outputs: dict[str, dict[str, str] | str] = {}
    try:
        if "controlled" in args.components:
            outputs["controlled_run"] = {
                key: str(value) for key, value in run_controlled(config, force=args.force).items()
            }
        if "sequential" in args.components:
            outputs["sequential_run"] = {
                key: str(value)
                for key, value in run_sequential(
                    config, workers=args.workers, force=args.force
                ).items()
            }
        if "particle-convergence" in args.components:
            outputs["particle_convergence"] = str(
                run_particle_convergence(config, workers=args.workers, force=args.force)
            )
        if "randomized" in args.components:
            lapse_values = (0.0,) if args.nominal_only else None
            outputs["randomized_run"] = {
                key: str(value)
                for key, value in run_randomized(
                    config,
                    workers=args.workers,
                    force=args.force,
                    force_library=args.force and "particle-convergence" not in args.components,
                    lapse_values=lapse_values,
                ).items()
            }
        if "precision-sweep" in args.components:
            outputs["precision_sweep_run"] = {
                key: str(value)
                for key, value in run_precision_sweep_extension(
                    config,
                    workers=args.workers,
                    force=args.force,
                    environment_count=args.precision_environments,
                    t_max=args.precision_t_max,
                    label=args.precision_label,
                ).items()
            }
        if "analyze" in args.components:
            if (root / "controlled" / "curve.parquet").exists():
                outputs["controlled_analysis"] = {
                    key: str(value) for key, value in analyze_controlled(config).items()
                }
            if (root / "sequential" / "raw" / "trials.parquet").exists():
                outputs["sequential_analysis"] = {
                    key: str(value) for key, value in analyze_sequential(config).items()
                }
            if (root / "randomized" / "main_trials.parquet").exists():
                outputs["randomized_analysis"] = {
                    key: str(value) for key, value in analyze_randomized(config).items()
                }
        if "precision-analyze" in args.components:
            precision_trials = (
                root
                / "randomized"
                / f"precision_sweep_t{config.profile.c_precision_t_max}_trials.parquet"
            )
            if precision_trials.exists():
                outputs["precision_sweep_analysis"] = {
                    key: str(value) for key, value in analyze_precision_sweep(config).items()
                }
        if "report" in args.components:
            outputs["report"] = {
                key: str(value) for key, value in write_report(config).items()
            }
        metadata["status"] = "complete"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["outputs"] = outputs
        atomic_json(metadata_path, metadata)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
