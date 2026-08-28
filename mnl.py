"""Numerically stable squared-distance multinomial-logit likelihood."""

from __future__ import annotations

import numpy as np


def logsumexp(values: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = values - maximum
    result = maximum + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)


def mnl_log_probs(s: np.ndarray | float, r: np.ndarray | float, b: np.ndarray, beta: float) -> np.ndarray:
    """Log P(Y=m | s,r) with arbitrary broadcastable s and r."""
    s_arr = np.asarray(s, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    logits = -float(beta) * (s_arr[..., None] - r_arr[..., None] * b) ** 2
    return logits - logsumexp(logits, axis=-1, keepdims=True)


def mnl_probs(s: np.ndarray | float, r: np.ndarray | float, b: np.ndarray, beta: float) -> np.ndarray:
    return np.exp(mnl_log_probs(s, r, b, beta))


def fisher_information_at_zero(r: np.ndarray | float, b: np.ndarray, beta: float) -> np.ndarray:
    """Scalar Fisher information in s at s=0 for the exact MNL model."""
    r_arr = np.asarray(r, dtype=float)
    probabilities = mnl_probs(np.zeros_like(r_arr), r_arr, b, beta)
    locations = r_arr[..., None] * b
    mean_location = np.sum(probabilities * locations, axis=-1, keepdims=True)
    score = 2.0 * beta * (locations - mean_location)
    return np.sum(probabilities * score * score, axis=-1)

