"""Run and analyze the unchanged paper Experiment B through n=65536."""

from __future__ import annotations

import argparse

from figure2_experiments.config import get_config
from figure2_experiments.exp_b_long.analyze import analyze
from figure2_experiments.exp_b_long.run import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-exact-grid", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()
    config = get_config("paper")
    run(config, force=args.force, skip_exact_grid=args.skip_exact_grid)
    if not args.run_only:
        analyze(config)


if __name__ == "__main__":
    main()
