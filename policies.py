"""Equal-budget coarse-to-continuous design optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from .eig import gaussian_eig, gaussian_eig_many, weighted_grid_eig
from .mnl import fisher_information_at_zero


@dataclass(frozen=True)
class Optimum:
    value: float
    objective: float
    boundary: bool
    search_min: float
    search_max: float
    evaluations: int


def _refine_log_grid(
    grid: np.ndarray,
    objectives: np.ndarray,
    scalar_objective: Callable[[float], float],
) -> Optimum:
    index = int(np.nanargmax(objectives))
    boundary = index == 0 or index == len(grid) - 1
    if boundary:
        return Optimum(
            value=float(grid[index]),
            objective=float(objectives[index]),
            boundary=True,
            search_min=float(grid[0]),
            search_max=float(grid[-1]),
            evaluations=len(grid),
        )
    lower = float(np.log(grid[index - 1]))
    upper = float(np.log(grid[index + 1]))
    result = minimize_scalar(
        lambda log_x: -scalar_objective(float(np.exp(log_x))),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 2e-10, "maxiter": 100},
    )
    value = float(np.exp(result.x))
    objective = float(-result.fun)
    if objective < float(objectives[index]):
        value, objective = float(grid[index]), float(objectives[index])
    return Optimum(
        value=value,
        objective=objective,
        boundary=False,
        search_min=float(grid[0]),
        search_max=float(grid[-1]),
        evaluations=len(grid) + int(result.nfev),
    )


def optimize_pr(
    sigma: float,
    b: np.ndarray,
    beta: float,
    alpha_min: float,
    alpha_max: float,
    grid_points: int,
    order: int,
) -> Optimum:
    grid = np.geomspace(alpha_min, alpha_max, grid_points)
    objectives = gaussian_eig_many(sigma, sigma * grid, b, beta, order)
    return _refine_log_grid(
        grid,
        objectives,
        lambda alpha: gaussian_eig(sigma, sigma * alpha, b, beta, order),
    )


def optimize_physical(
    sigma: float,
    b: np.ndarray,
    beta: float,
    r_min: float,
    r_max: float,
    grid_points: int,
    order: int,
    max_expansions: int = 4,
) -> Optimum:
    current_min, current_max = r_min, r_max
    for _ in range(max_expansions + 1):
        grid = np.geomspace(current_min, current_max, grid_points)
        objectives = gaussian_eig_many(sigma, grid, b, beta, order)
        optimum = _refine_log_grid(
            grid, objectives, lambda r: gaussian_eig(sigma, r, b, beta, order)
        )
        if not optimum.boundary:
            return optimum
        best_index = int(np.nanargmax(objectives))
        if best_index == 0:
            current_min /= 10.0
        else:
            current_max *= 10.0
    return optimum


def optimize_fisher(
    b: np.ndarray,
    beta: float,
    r_min: float,
    r_max: float,
    grid_points: int,
    max_expansions: int = 4,
) -> Optimum:
    current_min, current_max = r_min, r_max
    for _ in range(max_expansions + 1):
        grid = np.geomspace(current_min, current_max, grid_points)
        objectives = fisher_information_at_zero(grid, b, beta)
        optimum = _refine_log_grid(
            grid,
            objectives,
            lambda r: float(fisher_information_at_zero(np.array(r), b, beta)),
        )
        if not optimum.boundary:
            return optimum
        best_index = int(np.nanargmax(objectives))
        if best_index == 0:
            current_min /= 10.0
        else:
            current_max *= 10.0
    return optimum


def optimize_weighted_grid(
    offsets: np.ndarray,
    weights: np.ndarray,
    b: np.ndarray,
    beta: float,
    lower: float,
    upper: float,
    grid_points: int,
    mode: str,
) -> Optimum:
    grid = np.geomspace(lower, upper, grid_points)
    if mode == "pr":
        sigma = float(np.sqrt(np.sum(weights * offsets * offsets)))
        radii = sigma * grid
        values = weighted_grid_eig(offsets, weights, radii, b, beta)
        return _refine_log_grid(
            grid,
            values,
            lambda alpha: float(weighted_grid_eig(offsets, weights, np.array([sigma * alpha]), b, beta)[0]),
        )
    values = weighted_grid_eig(offsets, weights, grid, b, beta)
    return _refine_log_grid(
        grid,
        values,
        lambda radius: float(weighted_grid_eig(offsets, weights, np.array([radius]), b, beta)[0]),
    )

