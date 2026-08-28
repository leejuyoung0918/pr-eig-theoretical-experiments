"""Fixed-before-seeing-results configuration for all three experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .utils import stable_hash


PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Profile:
    name: str
    # Experiment A
    a_sigma_min: float
    a_sigma_points: int
    a_gh_order: int
    a_alpha_grid_points: int
    a_r_grid_points: int
    a_run_robustness: bool
    # Experiment B
    b_trajectories: int
    b_n_max: int
    b_batch_size: int
    b_cache_points: int
    b_gh_order: int
    b_design_grid_points: int
    b_bootstrap_samples: int
    b_grid_validation_trajectories: int
    b_grid_validation_n_max: int
    b_grid_nodes: int
    # Experiment C
    c_macro_seeds: int
    c_targets_per_macro: int
    c_bootstrap_samples: int


PROFILES: dict[str, Profile] = {
    "smoke": Profile(
        name="smoke",
        a_sigma_min=1e-4,
        a_sigma_points=41,
        a_gh_order=128,
        a_alpha_grid_points=81,
        a_r_grid_points=101,
        a_run_robustness=False,
        b_trajectories=20,
        b_n_max=256,
        b_batch_size=10,
        b_cache_points=120,
        b_gh_order=128,
        b_design_grid_points=81,
        b_bootstrap_samples=200,
        b_grid_validation_trajectories=2,
        b_grid_validation_n_max=64,
        b_grid_nodes=4097,
        c_macro_seeds=4,
        c_targets_per_macro=1_000,
        c_bootstrap_samples=200,
    ),
    "paper": Profile(
        name="paper",
        a_sigma_min=1e-4,
        a_sigma_points=121,
        a_gh_order=128,
        a_alpha_grid_points=241,
        a_r_grid_points=281,
        a_run_robustness=True,
        b_trajectories=1_000,
        b_n_max=8_192,
        b_batch_size=50,
        b_cache_points=600,
        b_gh_order=128,
        b_design_grid_points=201,
        b_bootstrap_samples=2_000,
        b_grid_validation_trajectories=8,
        b_grid_validation_n_max=512,
        b_grid_nodes=16_385,
        c_macro_seeds=20,
        c_targets_per_macro=5_000,
        c_bootstrap_samples=2_000,
    ),
    "exhaustive": Profile(
        name="exhaustive",
        a_sigma_min=1e-5,
        a_sigma_points=181,
        a_gh_order=256,
        a_alpha_grid_points=321,
        a_r_grid_points=361,
        a_run_robustness=True,
        b_trajectories=2_000,
        b_n_max=16_384,
        b_batch_size=50,
        b_cache_points=1_000,
        b_gh_order=256,
        b_design_grid_points=241,
        b_bootstrap_samples=5_000,
        b_grid_validation_trajectories=16,
        b_grid_validation_n_max=1_024,
        b_grid_nodes=32_769,
        c_macro_seeds=20,
        c_targets_per_macro=50_000,
        c_bootstrap_samples=5_000,
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "paper"
    seed: int = 20_260_818
    main_beta: float = 1.0
    main_m: int = 5
    beta_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    m_grid: tuple[int, ...] = (2, 3, 5, 8)
    alpha_min: float = 0.05
    alpha_max: float = 20.0
    expanded_alpha_min: float = 0.005
    expanded_alpha_max: float = 200.0
    physical_r_min: float = 1e-4
    physical_r_max: float = 10.0
    sigma_max: float = 1.0
    numerical_floor: float = 1e-22
    quadrature_rtol: float = 2e-5
    primary_sigma_fit_max: float = 0.03
    sigma_fit_windows: tuple[float, ...] = (0.1, 0.03, 0.01)
    sequential_fit_windows: tuple[int, ...] = (256, 512, 1024, 2048)
    primary_sequential_fit_min: int = 512
    cache_sigma_min: float = 1e-5
    cache_sigma_max: float = 1.5
    c_dimensions: tuple[int, ...] = (4, 8, 16)
    c_m_values: tuple[int, ...] = (2, 4, 8, 16)
    c_rounds_per_dimension: int = 8
    c_length: float = 1.0
    c_main_d: int = 8
    c_main_m: int = 4

    @property
    def profile(self) -> Profile:
        try:
            return PROFILES[self.mode]
        except KeyError as exc:
            raise ValueError(f"unknown execution profile: {self.mode}") from exc

    @property
    def output_root(self) -> Path:
        return PACKAGE_ROOT / "outputs" / "figure2"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["profile"] = asdict(self.profile)
        return data

    @property
    def config_hash(self) -> str:
        return stable_hash(self.as_dict())[:16]


def get_config(mode: str) -> ExperimentConfig:
    if mode not in PROFILES:
        raise ValueError(f"mode must be one of {sorted(PROFILES)}")
    return ExperimentConfig(mode=mode)

