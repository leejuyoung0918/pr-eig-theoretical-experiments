"""Paired randomized Bayesian localization benchmark for Figure 2(c).

The original Experiment 1 implementation is scalar-only. This module therefore
uses a documented projected particle ADF for the randomized multidimensional
benchmark. Query geometry, MNL likelihood, direction rule, and resolution
definitions are shared by all methods; only the resolution policy differs. This
is a moment-projected approximation, not a generic exact particle posterior.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from figure2_experiments.eig import gaussian_eig
from figure2_experiments.mnl import mnl_log_probs, mnl_probs
from figure2_experiments.policies import optimize_fisher, optimize_physical, optimize_pr
from figure2_experiments.templates import quantile_template
from figure2_experiments.utils import atomic_json, atomic_npz, stable_hash

from .config import Experiment1Config
from .statistics import normalized_posterior_rms


METHODS = ("PR-EIG", "Exact-RC", "Fisher-RC")


@dataclass(frozen=True)
class RandomEnvironment:
    environment_id: int
    seed: int
    d: int
    m: int
    beta: float
    s0: float
    kappa: float
    covariance: np.ndarray
    eigenvalues: np.ndarray
    target: np.ndarray

    @property
    def initial_trace(self) -> float:
        return float(np.trace(self.covariance))

    def record(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "seed": self.seed,
            "d": self.d,
            "M": self.m,
            "beta": self.beta,
            "s0": self.s0,
            "kappa": self.kappa,
            "initial_trace": self.initial_trace,
            "true_target_norm": float(np.linalg.norm(self.target)),
            "covariance_json": json.dumps(self.covariance.tolist(), separators=(",", ":")),
            "eigenvalues_json": json.dumps(self.eigenvalues.tolist(), separators=(",", ":")),
            "target_json": json.dumps(self.target.tolist(), separators=(",", ":")),
        }


def generate_environment(config: Experiment1Config, environment_id: int) -> RandomEnvironment:
    """Generate one environment using an ID-addressable deterministic seed."""
    sequence = np.random.SeedSequence([config.seed, 1_003, int(environment_id)])
    rng = np.random.default_rng(sequence)
    d = int(rng.choice(config.c_dimensions))
    m = int(rng.choice(config.c_arities))
    beta = float(np.exp(rng.uniform(np.log(config.c_beta_min), np.log(config.c_beta_max))))
    kappa = float(np.exp(rng.uniform(np.log(config.c_kappa_min), np.log(config.c_kappa_max))))
    s0 = float(np.exp(rng.uniform(np.log(config.c_s0_min), np.log(config.c_s0_max))))

    gaussian = rng.normal(size=(d, d))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q *= signs[None, :]
    raw_eigenvalues = np.geomspace(1.0, kappa, d)
    raw_eigenvalues = raw_eigenvalues[rng.permutation(d)]
    eigenvalues = d * raw_eigenvalues / raw_eigenvalues.sum()
    covariance = s0 * s0 * (q @ np.diag(eigenvalues) @ q.T)
    covariance = 0.5 * (covariance + covariance.T)
    target = np.linalg.cholesky(covariance) @ rng.normal(size=d)
    return RandomEnvironment(
        environment_id=int(environment_id),
        seed=int(sequence.generate_state(1, dtype=np.uint64)[0]),
        d=d,
        m=m,
        beta=beta,
        s0=s0,
        kappa=kappa,
        covariance=covariance,
        eigenvalues=eigenvalues,
        target=target,
    )


def _principal_direction(covariance: np.ndarray) -> tuple[np.ndarray, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    min_eigenvalue = float(eigenvalues[0])
    index = int(np.argmax(eigenvalues))
    direction = eigenvectors[:, index].copy()
    anchor = int(np.argmax(np.abs(direction)))
    if direction[anchor] < 0.0:
        direction *= -1.0
    return direction, float(max(eigenvalues[index], 1e-15)), min_eigenvalue


class ParticlePosterior:
    """Projected particle ADF with exact weighted one-dimensional moments.

    The MNL likelihood depends only on the queried principal direction.  Each
    update therefore computes normalized log weights and the exact particle
    approximation to the tilted mean/variance in that direction.  The full
    Gaussian state receives the corresponding rank-one moment update, after
    which the standardized directional cloud is reset to equal weights.  This
    is the multidimensional extension of the repository's
    existing Gaussian ADF; it avoids high-dimensional importance-weight
    degeneracy without pretending that a static cloud is an exact posterior.
    """

    def __init__(
        self,
        standard_particles: np.ndarray,
        *,
        seed: np.random.SeedSequence,
    ) -> None:
        self.standard_particles = np.asarray(standard_particles, dtype=float).copy()
        if self.standard_particles.ndim != 1:
            raise ValueError("projected particles must be one-dimensional")
        self.log_weights = np.full(
            len(self.standard_particles), -np.log(len(self.standard_particles))
        )
        self.rng = np.random.default_rng(seed)
        self.resampling_count = 0
        self.min_unique_fraction = 1.0
        self.max_weight_sum_error = 0.0
        self.min_ess_fraction = 1.0
        self.min_covariance_eigenvalue = math.inf

    def weights(self) -> np.ndarray:
        weights = np.exp(self.log_weights)
        total = float(weights.sum())
        self.max_weight_sum_error = max(self.max_weight_sum_error, abs(total - 1.0))
        return weights / total

    def update(
        self,
        center: np.ndarray,
        prior_covariance: np.ndarray,
        direction: np.ndarray,
        radius: float,
        template: np.ndarray,
        beta: float,
        response: int,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        prior_directional_variance = float(direction @ prior_covariance @ direction)
        offsets = math.sqrt(max(prior_directional_variance, 1e-15)) * self.standard_particles
        log_likelihood = mnl_log_probs(offsets, radius, template, beta)[:, response]
        self.log_weights = log_likelihood
        maximum = float(np.max(self.log_weights))
        normalizer = maximum + float(np.log(np.exp(self.log_weights - maximum).sum()))
        self.log_weights -= normalizer
        weights = self.weights()
        ess = float(1.0 / np.dot(weights, weights))
        self.min_ess_fraction = min(self.min_ess_fraction, ess / len(weights))
        tilted_mean = float(np.dot(weights, offsets))
        tilted_variance = float(
            np.dot(weights, (offsets - tilted_mean) * (offsets - tilted_mean))
        )
        mean = center + tilted_mean * direction
        covariance = prior_covariance + (
            max(tilted_variance, 1e-15) - prior_directional_variance
        ) * np.outer(direction, direction)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalue_floor = 1e-14 * max(float(np.trace(prior_covariance)), 1.0)
        eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
        covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        self.min_covariance_eigenvalue = min(
            self.min_covariance_eigenvalue, float(eigenvalues[0])
        )
        self.log_weights.fill(-np.log(len(self.standard_particles)))
        self.resampling_count += 1
        resampled = True
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(covariance)):
            raise FloatingPointError("non-finite particle posterior moment")
        return mean, covariance, ess, resampled


@dataclass
class ResolutionLibrary:
    tau: np.ndarray
    q: np.ndarray
    eig: np.ndarray
    boundary: np.ndarray
    evaluations: np.ndarray
    arities: np.ndarray
    build_seconds: float

    @classmethod
    def load(cls, path: Path) -> "ResolutionLibrary":
        with np.load(path) as saved:
            return cls(
                tau=saved["tau"].copy(),
                q=saved["q"].copy(),
                eig=saved["eig"].copy(),
                boundary=saved["boundary"].copy(),
                evaluations=saved["evaluations"].copy(),
                arities=saved["arities"].copy(),
                build_seconds=float(saved["build_seconds"]),
            )

    def lookup(self, method: str, m: int, sigma: float, beta: float) -> dict[str, float | bool]:
        method_index = METHODS.index(method)
        arity_index = int(np.where(self.arities == m)[0][0])
        tau_value = float(sigma * np.sqrt(beta))
        log_tau = np.log(np.clip(tau_value, self.tau[0], self.tau[-1]))
        q_source = self.q[arity_index, method_index]
        eig_source = self.eig[arity_index, method_index]
        q_value = float(np.exp(np.interp(log_tau, np.log(self.tau), np.log(q_source))))
        eig_value = float(np.exp(np.interp(log_tau, np.log(self.tau), np.log(np.maximum(eig_source, 1e-300)))))
        nearest = int(np.argmin(np.abs(np.log(self.tau) - log_tau)))
        return {
            "r": q_value / np.sqrt(beta),
            "eig": eig_value,
            "boundary": bool(self.boundary[arity_index, method_index, nearest]),
            "evaluations": float(self.evaluations[arity_index, method_index, nearest]),
            "tau_clipped": bool(tau_value < self.tau[0] or tau_value > self.tau[-1]),
        }


def resolution_library_path(config: Experiment1Config) -> Path:
    key = stable_hash(
        {
            "arities": config.c_arities,
            "tau": [config.c_tau_min, config.c_tau_max, config.profile.c_library_tau_points],
            "order": config.profile.c_library_gh_order,
            "grid": config.profile.c_library_grid_points,
            "alpha": [config.alpha_min, config.alpha_max],
        }
    )[:16]
    return config.output_root / "randomized" / "cache" / f"resolution_library_{key}.npz"


def build_resolution_library(config: Experiment1Config, force: bool = False) -> Path:
    """Build standardized beta=1 tables; beta scaling is algebraically exact."""
    path = resolution_library_path(config)
    if force:
        validation = config.output_root / "randomized" / "library_validation.csv"
        if validation.exists():
            validation.unlink()
    if path.exists() and not force:
        with np.load(path) as saved:
            if "q" in saved and np.all(np.isfinite(saved["q"])):
                return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tau = np.geomspace(config.c_tau_min, config.c_tau_max, config.profile.c_library_tau_points)
    shape = (len(config.c_arities), len(METHODS), len(tau))
    q = np.full(shape, np.nan)
    eig = np.full(shape, np.nan)
    boundary = np.zeros(shape, dtype=bool)
    evaluations = np.zeros(shape, dtype=np.int32)
    started = time.perf_counter()
    # This broad standardized domain contains the mapped physical domain for
    # beta in [0.25,4]; optima are checked for boundary contact below.
    q_min = config.physical_r_min * np.sqrt(config.c_beta_min)
    q_max = config.physical_r_max * np.sqrt(config.c_beta_max)
    for arity_index, m in enumerate(config.c_arities):
        template = quantile_template(m)
        fisher = optimize_fisher(template, 1.0, q_min, q_max, config.profile.c_library_grid_points)
        for tau_index, width in enumerate(tau):
            pr = optimize_pr(
                float(width), template, 1.0, config.alpha_min, config.alpha_max,
                config.profile.c_library_grid_points, config.profile.c_library_gh_order,
            )
            exact = optimize_physical(
                float(width), template, 1.0, q_min, q_max,
                config.profile.c_library_grid_points, config.profile.c_library_gh_order,
            )
            values = (
                (width * pr.value, pr.objective, pr.boundary, pr.evaluations),
                (exact.value, exact.objective, exact.boundary, exact.evaluations),
                (
                    fisher.value,
                    gaussian_eig(
                        float(width), fisher.value, template, 1.0,
                        config.profile.c_library_gh_order,
                    ),
                    fisher.boundary,
                    fisher.evaluations,
                ),
            )
            for method_index, value in enumerate(values):
                q[arity_index, method_index, tau_index] = value[0]
                eig[arity_index, method_index, tau_index] = max(value[1], 1e-300)
                boundary[arity_index, method_index, tau_index] = value[2]
                evaluations[arity_index, method_index, tau_index] = value[3]
        atomic_npz(
            path,
            tau=tau,
            q=q,
            eig=eig,
            boundary=boundary,
            evaluations=evaluations,
            arities=np.asarray(config.c_arities),
            build_seconds=np.array(time.perf_counter() - started),
        )
        print(f"[Experiment1 C library] M={m} complete", flush=True)
    return path


def validate_resolution_library(config: Experiment1Config, path: Path) -> Path:
    """Compare interpolation at log-grid midpoints with direct optimization."""
    output = config.output_root / "randomized" / "library_validation.csv"
    if output.exists():
        return output
    library = ResolutionLibrary.load(path)
    midpoint = np.sqrt(library.tau[:-1] * library.tau[1:])
    indices = np.unique(np.linspace(0, len(midpoint) - 1, 7, dtype=int))
    q_min = config.physical_r_min * np.sqrt(config.c_beta_min)
    q_max = config.physical_r_max * np.sqrt(config.c_beta_max)
    records = []
    for m in config.c_arities:
        template = quantile_template(m)
        fisher = optimize_fisher(
            template, 1.0, q_min, q_max, config.profile.c_library_grid_points
        )
        for tau in midpoint[indices]:
            for method in METHODS:
                cached = library.lookup(method, m, float(tau), 1.0)
                if method == "PR-EIG":
                    optimum = optimize_pr(
                        float(tau), template, 1.0, config.alpha_min, config.alpha_max,
                        config.profile.c_library_grid_points, config.profile.c_library_gh_order,
                    )
                    direct_q = float(tau * optimum.value)
                    direct_eig = float(optimum.objective)
                elif method == "Exact-RC":
                    optimum = optimize_physical(
                        float(tau), template, 1.0, q_min, q_max,
                        config.profile.c_library_grid_points, config.profile.c_library_gh_order,
                    )
                    direct_q = float(optimum.value)
                    direct_eig = float(optimum.objective)
                else:
                    direct_q = float(fisher.value)
                    direct_eig = gaussian_eig(
                        float(tau), direct_q, template, 1.0,
                        config.profile.c_library_gh_order,
                    )
                records.append(
                    {
                        "M": m,
                        "tau": tau,
                        "method": method,
                        "cached_q": cached["r"],
                        "direct_q": direct_q,
                        "relative_q_error": abs(float(cached["r"]) - direct_q) / direct_q,
                        "cached_eig": cached["eig"],
                        "direct_eig": direct_eig,
                        "relative_eig_error": abs(float(cached["eig"]) - direct_eig)
                        / max(direct_eig, 1e-300),
                    }
                )
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    return output


def _sample_standard_particles(environment: RandomEnvironment, count: int, config: Experiment1Config) -> np.ndarray:
    rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, 2_011, environment.environment_id])
    )
    standard = rng.normal(size=count)
    standard -= standard.mean()
    standard /= np.sqrt(np.mean(standard * standard))
    return standard


def _response_uniforms(
    config: Experiment1Config,
    environment_id: int,
    response_repetitions: int,
    t_max: int,
    *,
    preserved_horizon: int = 400,
) -> np.ndarray:
    """Generate CRNs while preserving the original full-run 400-query prefix.

    The historical full run drew one ``(response_repetitions, 400)`` matrix.
    Extended runs retain that matrix exactly, then append an independently seeded
    continuation for each response replicate. This makes old/new censoring
    comparisons meaningful without modifying any existing checkpoint.
    """
    base_rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, 3_017, environment_id])
    )
    if t_max <= preserved_horizon:
        return base_rng.random((response_repetitions, t_max))
    base = base_rng.random((response_repetitions, preserved_horizon))
    continuation = np.empty((response_repetitions, t_max - preserved_horizon))
    for response_replication in range(response_repetitions):
        continuation_rng = np.random.default_rng(
            np.random.SeedSequence(
                [config.seed, 3_017, environment_id, 7_019, response_replication]
            )
        )
        continuation[response_replication] = continuation_rng.random(
            t_max - preserved_horizon
        )
    return np.concatenate((base, continuation), axis=1)


def _simulate_one_environment(
    config: Experiment1Config,
    environment_id: int,
    library_file: str,
    output_file: str,
    lapse: float,
    particle_count: int,
    response_repetitions: int,
    t_max: int,
) -> str:
    output = Path(output_file)
    if output.exists():
        return str(output)
    environment = generate_environment(config, environment_id)
    template = quantile_template(environment.m)
    library = ResolutionLibrary.load(Path(library_file))
    standard_particles = _sample_standard_particles(environment, particle_count, config)
    initial_particle_mean = np.zeros(environment.d)
    initial_particle_covariance = environment.covariance.copy()
    initial_particle_trace = environment.initial_trace
    uniforms = _response_uniforms(
        config, environment_id, response_repetitions, t_max
    )
    rows: list[dict[str, Any]] = []
    environment_started = time.perf_counter()
    for response_replication in range(response_repetitions):
        for method_index, method in enumerate(METHODS):
            posterior = ParticlePosterior(
                standard_particles,
                seed=np.random.SeedSequence(
                    [config.seed, 4_021, environment_id, response_replication, method_index,
                     int(round(lapse * 1000)), particle_count]
                ),
            )
            mean = initial_particle_mean.copy()
            covariance = initial_particle_covariance.copy()
            rho_path = np.empty(t_max + 1)
            rho_path[0] = normalized_posterior_rms(covariance, initial_particle_trace)
            query_design_times = np.empty(t_max, dtype=float)
            resolution_times = np.empty(t_max, dtype=float)
            posterior_update_times = np.empty(t_max, dtype=float)
            directional_sigma_path = np.empty(t_max, dtype=float)
            selected_radius_path = np.empty(t_max, dtype=float)
            design_eig_path = np.empty(t_max, dtype=float)
            design_seconds = resolution_seconds = update_seconds = 0.0
            eig_evaluations = 0.0
            boundary_hits = tau_clips = probability_failures = 0
            for t in range(1, t_max + 1):
                design_start = time.perf_counter()
                direction, directional_variance, min_eigenvalue = _principal_direction(covariance)
                posterior.min_covariance_eigenvalue = min(
                    posterior.min_covariance_eigenvalue, min_eigenvalue
                )
                resolution_start = time.perf_counter()
                design = library.lookup(
                    method, environment.m, math.sqrt(directional_variance), environment.beta
                )
                resolution_elapsed = time.perf_counter() - resolution_start
                resolution_seconds += resolution_elapsed
                resolution_times[t - 1] = resolution_elapsed
                radius = float(design["r"])
                directional_sigma_path[t - 1] = math.sqrt(directional_variance)
                selected_radius_path[t - 1] = radius
                design_eig_path[t - 1] = float(design["eig"])
                eig_evaluations += float(design["evaluations"])
                boundary_hits += int(bool(design["boundary"]))
                tau_clips += int(bool(design["tau_clipped"]))
                true_offset = float((environment.target - mean) @ direction)
                nominal_probabilities = mnl_probs(
                    true_offset, radius, template, environment.beta
                )
                true_probabilities = (1.0 - lapse) * nominal_probabilities + lapse / environment.m
                if abs(float(true_probabilities.sum()) - 1.0) > 1e-12:
                    probability_failures += 1
                cdf = np.cumsum(true_probabilities)
                cdf[-1] = 1.0
                response = int(np.searchsorted(cdf, uniforms[response_replication, t - 1], side="right"))
                design_elapsed = time.perf_counter() - design_start
                design_seconds += design_elapsed
                query_design_times[t - 1] = design_elapsed
                update_start = time.perf_counter()
                mean, covariance, _, _ = posterior.update(
                    mean, covariance, direction, radius, template, environment.beta, response
                )
                update_elapsed = time.perf_counter() - update_start
                update_seconds += update_elapsed
                posterior_update_times[t - 1] = update_elapsed
                rho_path[t] = normalized_posterior_rms(covariance, initial_particle_trace)
            crossings: dict[float, int | None] = {}
            for threshold in config.c_thresholds:
                hits = np.flatnonzero(rho_path <= threshold)
                crossings[threshold] = int(hits[0]) if len(hits) else None
            final_error = float(np.linalg.norm(mean - environment.target) / np.sqrt(initial_particle_trace))
            primary_t = crossings[config.c_primary_threshold]
            rows.append(
                {
                    **environment.record(),
                    "method": method,
                    "response_replication": response_replication,
                    "lapse": lapse,
                    "particle_count": particle_count,
                    "t_max": t_max,
                    "success_025": primary_t is not None,
                    "T_05": crossings[0.5],
                    "T_025": primary_t,
                    "T_01": crossings[0.1],
                    "final_rho": float(rho_path[-1]),
                    "final_normalized_target_error": final_error,
                    "query_design_seconds": design_seconds,
                    "resolution_optimization_seconds": resolution_seconds,
                    "posterior_update_seconds": update_seconds,
                    "total_algorithm_seconds": design_seconds + update_seconds,
                    "mean_query_design_seconds": design_seconds / t_max,
                    "mean_resolution_seconds": resolution_seconds / t_max,
                    "mean_posterior_update_seconds": update_seconds / t_max,
                    "eig_evaluation_cost": eig_evaluations,
                    "online_eig_evaluation_cost": 0,
                    "boundary_hits": boundary_hits,
                    "tau_clip_count": tau_clips,
                    "probability_normalization_failures": probability_failures,
                    "resampling_count": posterior.resampling_count,
                    "minimum_ess_fraction": posterior.min_ess_fraction,
                    "minimum_unique_ancestor_fraction": posterior.min_unique_fraction,
                    "maximum_weight_sum_error": posterior.max_weight_sum_error,
                    "minimum_covariance_eigenvalue": posterior.min_covariance_eigenvalue,
                    "rho_path_json": json.dumps(rho_path.tolist(), separators=(",", ":")),
                    "query_design_times_json": json.dumps(query_design_times.tolist(), separators=(",", ":")),
                    "resolution_times_json": json.dumps(resolution_times.tolist(), separators=(",", ":")),
                    "posterior_update_times_json": json.dumps(posterior_update_times.tolist(), separators=(",", ":")),
                    "directional_sigma_path_json": json.dumps(directional_sigma_path.tolist(), separators=(",", ":")),
                    "selected_radius_path_json": json.dumps(selected_radius_path.tolist(), separators=(",", ":")),
                    "design_eig_path_json": json.dumps(design_eig_path.tolist(), separators=(",", ":")),
                }
            )
    payload = {
        "environment_id": environment_id,
        "lapse": lapse,
        "particle_count": particle_count,
        "response_repetitions": response_repetitions,
        "t_max": t_max,
        "elapsed_seconds": time.perf_counter() - environment_started,
        "rows": rows,
    }
    atomic_json(output, payload)
    return str(output)


def _raw_directory(config: Experiment1Config, lapse: float, label: str = "main") -> Path:
    return config.output_root / "randomized" / "raw" / label / f"lapse_{lapse:.2f}"


def run_randomized(
    config: Experiment1Config,
    *,
    workers: int | None = None,
    force: bool = False,
    force_library: bool | None = None,
    lapse_values: tuple[float, ...] | None = None,
    environment_count: int | None = None,
    particle_count: int | None = None,
    response_repetitions: int | None = None,
    t_max: int | None = None,
    label: str = "main",
) -> dict[str, Path]:
    """Run checkpointed environment tasks and aggregate raw trial rows."""
    if force_library is None:
        force_library = force
    library_path = build_resolution_library(config, force=force_library)
    library_validation = validate_resolution_library(config, library_path)
    environment_count = environment_count or config.profile.c_environments
    particle_count = particle_count or config.profile.c_particles
    response_repetitions = response_repetitions or config.profile.c_response_repetitions
    t_max = t_max or config.profile.c_t_max
    lapse_values = lapse_values or ((0.0,) if not config.profile.c_run_mismatch else config.c_lapse_values)
    started = time.perf_counter()
    tasks: list[tuple[float, int, Path]] = []
    for lapse in lapse_values:
        directory = _raw_directory(config, lapse, label)
        directory.mkdir(parents=True, exist_ok=True)
        for environment_id in range(environment_count):
            path = directory / f"environment_{environment_id:05d}.json"
            if force and path.exists():
                path.unlink()
            tasks.append((float(lapse), environment_id, path))
    workers = workers or min(max((os.cpu_count() or 2) - 1, 1), 8)

    def submit_sequentially() -> None:
        for index, (lapse, environment_id, path) in enumerate(tasks, 1):
            _simulate_one_environment(
                config, environment_id, str(library_path), str(path), lapse,
                particle_count, response_repetitions, t_max,
            )
            print(f"[Experiment1 C {label}] {index}/{len(tasks)}", flush=True)

    if workers == 1:
        submit_sequentially()
    else:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(
                        _simulate_one_environment,
                        config,
                        environment_id,
                        str(library_path),
                        str(path),
                        lapse,
                        particle_count,
                        response_repetitions,
                        t_max,
                    ): (lapse, environment_id)
                    for lapse, environment_id, path in tasks
                }
                for index, future in enumerate(concurrent.futures.as_completed(future_map), 1):
                    future.result()
                    print(f"[Experiment1 C {label}] {index}/{len(tasks)}", flush=True)
        except (PermissionError, NotImplementedError, OSError) as exc:
            print(f"[Experiment1 C] multiprocessing unavailable ({exc!r}); sequential fallback", flush=True)
            workers = 1
            submit_sequentially()

    records: list[dict[str, Any]] = []
    for _, _, path in tasks:
        with path.open(encoding="utf-8") as handle:
            records.extend(json.load(handle)["rows"])
    raw = pd.DataFrame.from_records(records)
    output = config.output_root / "randomized" / f"{label}_trials.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(output, index=False)
    metadata = {
        "profile": config.mode,
        "label": label,
        "environment_count": environment_count,
        "response_repetitions": response_repetitions,
        "particle_count": particle_count,
        "T_max": t_max,
        "lapse_values": lapse_values,
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
        "posterior": "projected particle ADF: normalized log weights on a standardized directional particle cloud, weighted directional moments, rank-one Gaussian moment projection, equal-weight reset every query",
        "resolution_design": "exact Gaussian directional EIG/Fisher library with algebraically exact beta scaling",
        "eig_evaluation_cost_definition": "equivalent uncached objective evaluations represented by library lookups; online_eig_evaluation_cost is zero after table construction",
        "library_path": str(library_path),
        "config_hash": config.config_hash,
    }
    metadata_path = config.output_root / "randomized" / f"{label}_metadata.json"
    atomic_json(metadata_path, metadata)
    return {
        "trials": output,
        "metadata": metadata_path,
        "library": library_path,
        "library_validation": library_validation,
    }


def run_particle_convergence(
    config: Experiment1Config,
    *,
    workers: int | None = None,
    force: bool = False,
) -> Path:
    """Run the predeclared particle-count sensitivity suite on fixed environments."""
    frames = []
    # Build (or rebuild) the shared design library once; environment checkpoints
    # remain independently force-recomputable below.
    build_resolution_library(config, force=force)
    counts = config.c_particle_counts
    if config.mode == "quick":
        counts = (200, 500, 1_000, 2_000)
    for count in counts:
        result = run_randomized(
            config,
            workers=workers,
            force=force,
            force_library=False,
            lapse_values=(0.0,),
            environment_count=config.profile.c_convergence_environments,
            particle_count=count,
            response_repetitions=config.profile.c_convergence_response_repetitions,
            t_max=config.profile.c_convergence_t_max,
            label=f"particle_convergence_{count}",
        )
        frame = pd.read_parquet(result["trials"])
        frames.append(frame)
    output = config.output_root / "randomized" / "particle_convergence_trials.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(output, index=False)
    return output


def run_precision_sweep_extension(
    config: Experiment1Config,
    *,
    workers: int | None = None,
    force: bool = False,
    environment_count: int | None = None,
    t_max: int | None = None,
    label: str | None = None,
) -> dict[str, Path]:
    """Run the nominal-response horizon extension used by Figure 2(c)."""
    horizon = int(t_max or config.profile.c_precision_t_max)
    run_label = label or f"precision_sweep_t{horizon}"
    return run_randomized(
        config,
        workers=workers,
        force=force,
        force_library=False,
        lapse_values=(0.0,),
        environment_count=environment_count or config.profile.c_environments,
        particle_count=config.profile.c_particles,
        response_repetitions=config.profile.c_response_repetitions,
        t_max=horizon,
        label=run_label,
    )
