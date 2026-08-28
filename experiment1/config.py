"""Predeclared quick and full configurations for the redesigned Experiment 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from figure2_experiments.utils import stable_hash


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment1Profile:
    name: str
    # Controlled panel.
    a_sigma_points: int
    a_gh_order: int
    a_design_grid_points: int
    # Sequential panel.
    b_repetitions: int
    b_n_max: int
    b_batch_size: int
    b_checkpoint_count: int
    b_bootstrap_samples: int
    # Randomized panel.
    c_environments: int
    c_response_repetitions: int
    c_t_max: int
    c_precision_t_max: int
    c_particles: int
    c_environment_batch_size: int
    c_bootstrap_samples: int
    c_library_tau_points: int
    c_library_gh_order: int
    c_library_grid_points: int
    c_convergence_environments: int
    c_convergence_response_repetitions: int
    c_convergence_t_max: int
    c_run_mismatch: bool


PROFILES: dict[str, Experiment1Profile] = {
    "quick": Experiment1Profile(
        name="quick",
        a_sigma_points=41,
        a_gh_order=128,
        a_design_grid_points=81,
        b_repetitions=20,
        b_n_max=3_000,
        b_batch_size=10,
        b_checkpoint_count=55,
        b_bootstrap_samples=200,
        c_environments=8,
        c_response_repetitions=2,
        c_t_max=40,
        c_precision_t_max=120,
        c_particles=500,
        c_environment_batch_size=1,
        c_bootstrap_samples=200,
        c_library_tau_points=120,
        c_library_gh_order=64,
        c_library_grid_points=61,
        c_convergence_environments=2,
        c_convergence_response_repetitions=1,
        c_convergence_t_max=20,
        c_run_mismatch=True,
    ),
    "full": Experiment1Profile(
        name="full",
        a_sigma_points=121,
        a_gh_order=128,
        a_design_grid_points=241,
        b_repetitions=200,
        b_n_max=100_000,
        b_batch_size=25,
        b_checkpoint_count=120,
        b_bootstrap_samples=2_000,
        c_environments=500,
        c_response_repetitions=4,
        # A 20-environment nominal pilot at T=200 left 15% of environments
        # censored for every method, so the predeclared escalation rule selects 400.
        c_t_max=400,
        c_precision_t_max=1_000,
        c_particles=2_000,
        c_environment_batch_size=1,
        c_bootstrap_samples=2_000,
        c_library_tau_points=800,
        c_library_gh_order=128,
        c_library_grid_points=161,
        c_convergence_environments=6,
        c_convergence_response_repetitions=2,
        c_convergence_t_max=60,
        c_run_mismatch=True,
    ),
}


@dataclass(frozen=True)
class Experiment1Config:
    mode: str = "full"
    seed: int = 20_260_820
    methods: tuple[str, ...] = ("PR-EIG", "Exact-RC", "Fisher-RC")
    # Shared resolution definitions.
    alpha_min: float = 0.05
    alpha_max: float = 20.0
    physical_r_min: float = 1e-4
    physical_r_max: float = 10.0
    # Panel A.
    a_beta: float = 1.0
    a_m: int = 5
    a_sigma_min: float = 1e-4
    a_sigma_max: float = 1.0
    a_fit_min: float = 1e-4
    a_fit_max: float = 1e-2
    # Panel B.
    b_beta: float = 1.0
    b_m: int = 5
    b_fit_starts: tuple[int, ...] = (1_000, 3_000, 10_000, 30_000)
    b_primary_fit_start: int = 10_000
    # Panel C.
    c_dimensions: tuple[int, ...] = (2, 4, 8, 16)
    c_arities: tuple[int, ...] = (3, 5, 7, 9)
    c_beta_min: float = 0.25
    c_beta_max: float = 4.0
    c_kappa_min: float = 1.0
    c_kappa_max: float = 100.0
    c_s0_min: float = 0.5
    c_s0_max: float = 2.0
    c_thresholds: tuple[float, ...] = (0.5, 0.25, 0.1)
    c_primary_threshold: float = 0.25
    c_precision_thresholds: tuple[float, ...] = (
        0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02,
    )
    c_lapse_values: tuple[float, ...] = (0.0, 0.05, 0.10)
    c_tau_min: float = 1e-4
    c_tau_max: float = 16.0
    c_particle_counts: tuple[int, ...] = (2_000, 5_000, 10_000, 20_000)

    @property
    def profile(self) -> Experiment1Profile:
        try:
            return PROFILES[self.mode]
        except KeyError as exc:
            raise ValueError(f"unknown Experiment 1 mode: {self.mode}") from exc

    @property
    def output_root(self) -> Path:
        return PACKAGE_ROOT / "outputs" / "experiment1" / self.mode

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["profile"] = asdict(self.profile)
        return result

    @property
    def config_hash(self) -> str:
        return stable_hash(self.as_dict())[:16]


def get_experiment1_config(mode: str) -> Experiment1Config:
    if mode not in PROFILES:
        raise ValueError(f"mode must be one of {sorted(PROFILES)}")
    return Experiment1Config(mode=mode)
