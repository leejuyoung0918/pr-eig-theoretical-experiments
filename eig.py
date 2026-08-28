"""High-precision deterministic EIG and Gaussian ADF transition integrals."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import roots_hermite

from .mnl import mnl_probs


@lru_cache(maxsize=None)
def gaussian_hermite_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 8:
        raise ValueError("Gauss-Hermite order must be at least 8")
    nodes, weights = roots_hermite(order)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def _stable_discrete_kl(probabilities: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """KL(p||q) without subtracting nearly equal entropies.

    q * ((1+x) log(1+x) - x), x=(p-q)/q, removes the analytically
    cancelling first-order term and retains O(x^2) information in float64.
    """
    x = np.maximum((probabilities - marginal) / marginal, -1.0)
    kernel = np.empty_like(x)
    zero_probability = x <= -1.0 + 4.0 * np.finfo(float).eps
    kernel[zero_probability] = 1.0
    regular = ~zero_probability
    kernel[regular] = (1.0 + x[regular]) * np.log1p(x[regular]) - x[regular]
    return np.sum(marginal * kernel, axis=-1)


def gaussian_eig(
    sigma: float,
    r: float,
    b: np.ndarray,
    beta: float,
    order: int = 128,
) -> float:
    return float(gaussian_eig_many(sigma, np.array([r]), b, beta, order)[0])


def gaussian_eig_many(
    sigma: float,
    r_values: np.ndarray,
    b: np.ndarray,
    beta: float,
    order: int = 128,
    chunk_size: int = 512,
) -> np.ndarray:
    """Evaluate EIG for many physical radii using deterministic quadrature."""
    z, weights = gaussian_hermite_rule(order)
    radii = np.atleast_1d(np.asarray(r_values, dtype=float))
    output = np.empty(radii.size, dtype=float)
    s = sigma * z
    for start in range(0, radii.size, chunk_size):
        stop = min(start + chunk_size, radii.size)
        rr = radii[start:stop]
        probs = mnl_probs(s[None, :], rr[:, None], b, beta)
        marginal = np.einsum("q,rqm->rm", weights, probs)
        marginal = np.maximum(marginal, np.finfo(float).tiny)
        kl = _stable_discrete_kl(probs, marginal[:, None, :])
        values = np.einsum("q,rq->r", weights, kl)
        output[start:stop] = np.maximum(values, 0.0)
    return output


def gaussian_adf_table(
    sigma: float,
    r: float,
    b: np.ndarray,
    beta: float,
    order: int = 128,
) -> dict[str, np.ndarray | float]:
    """Exact quadrature moments for every categorical response.

    The returned mean shift is in units of sigma and the variance ratio is
    Var(S|Y=m)/sigma**2, making translation-invariant table interpolation easy.
    """
    z, weights = gaussian_hermite_rule(order)
    probs = mnl_probs(sigma * z, r, b, beta)
    predictive = np.einsum("q,qm->m", weights, probs)
    predictive = np.maximum(predictive, np.finfo(float).tiny)
    normalized = weights[:, None] * probs / predictive[None, :]
    mean_z = np.einsum("qm,q->m", normalized, z)
    second_z = np.einsum("qm,q->m", normalized, z * z)
    variance_ratio = np.maximum(second_z - mean_z * mean_z, 1e-15)
    # Use the cancellation-safe calculation for EIG.
    eig = float(np.dot(weights, _stable_discrete_kl(probs, predictive[None, :])))
    return {
        "predictive": predictive,
        "mean_shift_sigma": mean_z,
        "variance_ratio": variance_ratio,
        "eig": max(eig, 0.0),
    }


def weighted_grid_eig(
    offsets: np.ndarray,
    weights: np.ndarray,
    r_values: np.ndarray,
    b: np.ndarray,
    beta: float,
    chunk_size: int = 32,
) -> np.ndarray:
    """Exact EIG for an arbitrary normalized 1D posterior on a dense grid."""
    offsets = np.asarray(offsets, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    radii = np.atleast_1d(np.asarray(r_values, dtype=float))
    output = np.empty(radii.size, dtype=float)
    for start in range(0, radii.size, chunk_size):
        stop = min(start + chunk_size, radii.size)
        rr = radii[start:stop]
        probs = mnl_probs(offsets[None, :], rr[:, None], b, beta)
        marginal = np.einsum("q,rqm->rm", weights, probs)
        kl = _stable_discrete_kl(probs, marginal[:, None, :])
        output[start:stop] = np.einsum("q,rq->r", weights, kl)
    return np.maximum(output, 0.0)
