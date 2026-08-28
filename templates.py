"""Symmetric posterior-quantile query templates."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri


def quantile_template(m: int) -> np.ndarray:
    """Return centered, unit-RMS equal-mass standard-normal quantile centers.

    Raw centers are Phi^{-1}((j+1/2)/M).  Floating-point centering enforces
    exact zero mean; division by sqrt(mean(b**2)) fixes the template RMS to 1.
    """
    if m < 2:
        raise ValueError("M must be at least 2")
    probabilities = (np.arange(m, dtype=float) + 0.5) / m
    b = ndtri(probabilities)
    b -= b.mean()
    b /= np.sqrt(np.mean(b * b))
    return b

