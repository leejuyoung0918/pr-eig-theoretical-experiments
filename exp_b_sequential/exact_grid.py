"""Dense adaptive-grid posterior used only to validate Gaussian ADF."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from figure2_experiments.mnl import mnl_probs


class DenseGridPosterior:
    def __init__(self, nodes: int, lower: float = -8.0, upper: float = 8.0):
        if nodes < 1025 or nodes % 2 == 0:
            raise ValueError("dense grid requires an odd node count >= 1025")
        self.nodes = nodes
        self.grid = np.linspace(lower, upper, nodes)
        dx = self.grid[1] - self.grid[0]
        density = np.exp(-0.5 * self.grid * self.grid) / np.sqrt(2.0 * np.pi)
        self.weights = density * dx
        self.weights /= self.weights.sum()

    def moments(self) -> tuple[float, float]:
        mean = float(np.dot(self.weights, self.grid))
        variance = float(np.dot(self.weights, (self.grid - mean) ** 2))
        return mean, max(variance, 1e-18)

    def update(self, center: float, radius: float, b: np.ndarray, beta: float, response: int) -> None:
        likelihood = mnl_probs(self.grid - center, radius, b, beta)[:, response]
        self.weights *= likelihood
        total = self.weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("exact-grid posterior normalization failed")
        self.weights /= total
        self._maybe_regrid()

    def _maybe_regrid(self) -> None:
        mean, variance = self.moments()
        sigma = np.sqrt(variance)
        dx = self.grid[1] - self.grid[0]
        if sigma / dx >= 100.0:
            return
        lower, upper = mean - 12.0 * sigma, mean + 12.0 * sigma
        new_grid = np.linspace(lower, upper, self.nodes)
        old_density = self.weights / dx
        new_density = np.interp(new_grid, self.grid, old_density, left=0.0, right=0.0)
        new_dx = new_grid[1] - new_grid[0]
        new_weights = new_density * new_dx
        total = new_weights.sum()
        if total > 0.999999999:
            self.grid = new_grid
            self.weights = new_weights / total

    def centered(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        mean, variance = self.moments()
        return self.grid - mean, self.weights.copy(), mean, variance

