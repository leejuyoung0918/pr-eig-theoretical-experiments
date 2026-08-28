"""Regression, threshold, and environment-bootstrap helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def log_linear_regression(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """OLS for log(y)=intercept+slope*log(x), including classical SE/CI."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    lx = np.log(x[valid])
    ly = np.log(y[valid])
    n = len(lx)
    if n < 3:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "slope_se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2": np.nan,
            "n_points": n,
        }
    x_centered = lx - lx.mean()
    denominator = float(np.dot(x_centered, x_centered))
    slope = float(np.dot(x_centered, ly - ly.mean()) / denominator)
    intercept = float(ly.mean() - slope * lx.mean())
    residual = ly - (intercept + slope * lx)
    rss = float(np.dot(residual, residual))
    tss = float(np.dot(ly - ly.mean(), ly - ly.mean()))
    residual_variance = rss / (n - 2)
    slope_se = float(np.sqrt(residual_variance / denominator))
    critical = float(student_t.ppf(0.975, n - 2))
    return {
        "slope": slope,
        "intercept": intercept,
        "slope_se": slope_se,
        "ci_low": slope - critical * slope_se,
        "ci_high": slope + critical * slope_se,
        "r2": 1.0 - rss / tss if tss > 0.0 else 1.0,
        "n_points": n,
    }


def fixed_exponent_intercept(x: np.ndarray, y: np.ndarray, exponent: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if not np.any(valid):
        raise ValueError("no finite positive observations for fixed-exponent fit")
    return float(np.mean(np.log(y[valid]) - exponent * np.log(x[valid])))


def first_threshold_crossings(values: np.ndarray, thresholds: tuple[float, ...]) -> dict[float, int | None]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("threshold values must be a finite one-dimensional array")
    result: dict[float, int | None] = {}
    for threshold in thresholds:
        hits = np.flatnonzero(values <= threshold)
        result[float(threshold)] = int(hits[0]) if len(hits) else None
    return result


def normalized_posterior_rms(covariance: np.ndarray, initial_trace: float) -> float:
    covariance = np.asarray(covariance, dtype=float)
    if initial_trace <= 0.0:
        raise ValueError("initial covariance trace must be positive")
    trace = float(np.trace(covariance))
    if trace < -1e-12:
        raise ValueError("posterior covariance has negative trace")
    return float(np.sqrt(max(trace, 0.0) / initial_trace))


def bootstrap_rms_from_variances(
    variances: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trial bootstrap of sqrt(mean(V)); rows are independent trials."""
    variances = np.asarray(variances, dtype=float)
    if variances.ndim != 2 or len(variances) == 0:
        raise ValueError("variances must be a nonempty trials-by-checkpoints matrix")
    if np.any(~np.isfinite(variances)) or np.any(variances < 0.0):
        raise ValueError("variances must be finite and nonnegative")
    rms = np.sqrt(variances.mean(axis=0))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(variances), np.full(len(variances), 1.0 / len(variances)), size=samples
    )
    bootstrap = np.sqrt(counts @ variances / len(variances))
    low, high = np.quantile(bootstrap, (0.025, 0.975), axis=0)
    return rms, low, high


def environment_bootstrap_indices(n_environments: int, samples: int, seed: int) -> np.ndarray:
    if n_environments <= 0 or samples <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_environments, size=(samples, n_environments), endpoint=False)
