"""High-precision policy/ADF transition cache on a log-sigma grid."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from figure2_experiments.config import ExperimentConfig
from figure2_experiments.eig import gaussian_adf_table
from figure2_experiments.mnl import fisher_information_at_zero
from figure2_experiments.policies import optimize_fisher, optimize_physical, optimize_pr
from figure2_experiments.templates import quantile_template
from figure2_experiments.utils import atomic_npz, stable_hash


METHODS = ("PR-EIG", "Exact-RC", "Fisher-RC")


def cache_path(config: ExperimentConfig) -> Path:
    key = stable_hash(
        {
            "experiment": "B-cache",
            "mode": config.mode,
            "beta": config.main_beta,
            "M": config.main_m,
            "sigma": [config.cache_sigma_min, config.cache_sigma_max],
            "points": config.profile.b_cache_points,
            "order": config.profile.b_gh_order,
            "design_grid": config.profile.b_design_grid_points,
            "alpha": [config.alpha_min, config.alpha_max],
            "r": [config.physical_r_min, config.physical_r_max],
        }
    )[:16]
    return config.output_root / "cache" / f"sequential_design_{key}.npz"


@dataclass
class DesignCache:
    sigma: np.ndarray
    r: np.ndarray
    eig: np.ndarray
    fisher: np.ndarray
    predictive: np.ndarray
    mean_shift_sigma: np.ndarray
    variance_ratio: np.ndarray
    boundary: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "DesignCache":
        with np.load(path) as data:
            return cls(**{name: data[name].copy() for name in cls.__annotations__})

    def lookup(self, method: str, sigma: np.ndarray | float) -> dict[str, np.ndarray]:
        method_index = METHODS.index(method)
        query = np.asarray(sigma, dtype=float)
        flat = query.ravel()
        log_grid = np.log(self.sigma)
        log_query = np.log(np.clip(flat, self.sigma[0], self.sigma[-1]))

        def interp(values: np.ndarray, *, log_value: bool = False) -> np.ndarray:
            source = values[method_index]
            if source.ndim == 1:
                transformed = np.log(np.maximum(source, np.finfo(float).tiny)) if log_value else source
                result = np.interp(log_query, log_grid, transformed)
                if log_value:
                    result = np.exp(result)
                return result.reshape(query.shape)
            columns = []
            for column in range(source.shape[1]):
                transformed = source[:, column]
                if log_value:
                    transformed = np.log(np.maximum(transformed, np.finfo(float).tiny))
                values_column = np.interp(log_query, log_grid, transformed)
                if log_value:
                    values_column = np.exp(values_column)
                columns.append(values_column)
            return np.stack(columns, axis=-1).reshape(query.shape + (source.shape[1],))

        predictive = interp(self.predictive)
        predictive /= predictive.sum(axis=-1, keepdims=True)
        return {
            "r": interp(self.r, log_value=True),
            "eig": interp(self.eig, log_value=True),
            "fisher": interp(self.fisher, log_value=True),
            "predictive": predictive,
            "mean_shift_sigma": interp(self.mean_shift_sigma),
            "variance_ratio": interp(self.variance_ratio, log_value=True),
        }


def build_cache(config: ExperimentConfig, force: bool = False) -> Path:
    path = cache_path(config)
    if path.exists() and not force:
        with np.load(path) as data:
            if "done" in data and bool(np.all(data["done"])):
                return path
    sigma = np.geomspace(
        config.cache_sigma_min, config.cache_sigma_max, config.profile.b_cache_points
    )
    shape = (len(METHODS), len(sigma))
    m = config.main_m
    b = quantile_template(m)
    arrays = {
        "r": np.full(shape, np.nan),
        "eig": np.full(shape, np.nan),
        "fisher": np.full(shape, np.nan),
        "predictive": np.full(shape + (m,), np.nan),
        "mean_shift_sigma": np.full(shape + (m,), np.nan),
        "variance_ratio": np.full(shape + (m,), np.nan),
        "boundary": np.zeros(shape, dtype=bool),
        "done": np.zeros(shape, dtype=bool),
    }
    if path.exists() and not force:
        with np.load(path) as saved:
            if np.array_equal(saved["sigma"], sigma):
                for name in arrays:
                    if name in saved:
                        arrays[name][...] = saved[name]

    fisher_optimum = optimize_fisher(
        b,
        config.main_beta,
        config.physical_r_min,
        config.physical_r_max,
        config.profile.b_design_grid_points,
    )
    save_every = max(1, len(sigma) // 50)
    started = time.monotonic()
    for method_index, method in enumerate(METHODS):
        for index, width in enumerate(sigma):
            if arrays["done"][method_index, index]:
                continue
            if method == "PR-EIG":
                optimum = optimize_pr(
                    float(width),
                    b,
                    config.main_beta,
                    config.alpha_min,
                    config.alpha_max,
                    config.profile.b_design_grid_points,
                    config.profile.b_gh_order,
                )
                radius = width * optimum.value
            elif method == "Exact-RC":
                optimum = optimize_physical(
                    float(width),
                    b,
                    config.main_beta,
                    config.physical_r_min,
                    config.physical_r_max,
                    config.profile.b_design_grid_points,
                    config.profile.b_gh_order,
                )
                radius = optimum.value
            else:
                optimum = fisher_optimum
                radius = optimum.value
            transition = gaussian_adf_table(
                float(width), float(radius), b, config.main_beta, config.profile.b_gh_order
            )
            arrays["r"][method_index, index] = radius
            arrays["eig"][method_index, index] = transition["eig"]
            arrays["fisher"][method_index, index] = fisher_information_at_zero(
                np.array(radius), b, config.main_beta
            )
            arrays["predictive"][method_index, index] = transition["predictive"]
            arrays["mean_shift_sigma"][method_index, index] = transition["mean_shift_sigma"]
            arrays["variance_ratio"][method_index, index] = transition["variance_ratio"]
            arrays["boundary"][method_index, index] = optimum.boundary
            arrays["done"][method_index, index] = True
            if (index + 1) % save_every == 0 or index + 1 == len(sigma):
                atomic_npz(
                    path,
                    sigma=sigma,
                    methods=np.array(METHODS),
                    elapsed_seconds=np.array(time.monotonic() - started),
                    **arrays,
                )
                print(f"[B cache {method}] {index + 1}/{len(sigma)}", flush=True)
    return path


def validate_cache(config: ExperimentConfig, path: Path) -> pd.DataFrame:
    cache = DesignCache.load(path)
    b = quantile_template(config.main_m)
    # Geometric midpoints are deliberately the hardest points for log interpolation.
    candidates = np.sqrt(cache.sigma[:-1] * cache.sigma[1:])
    indices = np.unique(np.linspace(0, len(candidates) - 1, min(25, len(candidates)), dtype=int))
    records = []
    fisher_optimum = optimize_fisher(
        b,
        config.main_beta,
        config.physical_r_min,
        config.physical_r_max,
        config.profile.b_design_grid_points,
    )
    for sigma in candidates[indices]:
        for method in METHODS:
            cached = cache.lookup(method, np.array([sigma]))
            if method == "PR-EIG":
                optimum = optimize_pr(
                    float(sigma), b, config.main_beta, config.alpha_min, config.alpha_max,
                    config.profile.b_design_grid_points, config.profile.b_gh_order
                )
                direct_r = sigma * optimum.value
            elif method == "Exact-RC":
                optimum = optimize_physical(
                    float(sigma), b, config.main_beta, config.physical_r_min,
                    config.physical_r_max, config.profile.b_design_grid_points,
                    config.profile.b_gh_order
                )
                direct_r = optimum.value
            else:
                direct_r = fisher_optimum.value
            direct = gaussian_adf_table(
                float(sigma), float(direct_r), b, config.main_beta, config.profile.b_gh_order
            )
            records.append(
                {
                    "sigma": sigma,
                    "method": method,
                    "cached_r": cached["r"][0],
                    "direct_r": direct_r,
                    "relative_r_error": abs(cached["r"][0] - direct_r) / direct_r,
                    "relative_eig_error": abs(cached["eig"][0] - direct["eig"]) / max(direct["eig"], config.numerical_floor),
                    "max_predictive_error": float(np.max(np.abs(cached["predictive"][0] - direct["predictive"]))),
                    "max_mean_shift_error": float(np.max(np.abs(cached["mean_shift_sigma"][0] - direct["mean_shift_sigma"]))),
                    "max_variance_ratio_error": float(np.max(np.abs(cached["variance_ratio"][0] - direct["variance_ratio"]))),
                }
            )
    return pd.DataFrame.from_records(records)

