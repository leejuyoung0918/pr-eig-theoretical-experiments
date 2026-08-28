import numpy as np

from figure2_experiments.eig import gaussian_eig
from figure2_experiments.policies import optimize_fisher, optimize_physical
from figure2_experiments.templates import quantile_template


def test_a1_bounded_pr_radius_goes_to_zero():
    alpha = 20.0
    sigma = np.array([1e-4, 2e-4, 4e-4])
    radius = alpha * sigma
    assert np.all(np.diff(radius) > 0)
    assert radius[0] < radius[-1] / 3.9


def test_a2_physical_optimum_does_not_track_sigma_to_zero():
    b = quantile_template(5)
    radii = np.array(
        [optimize_physical(sigma, b, 1.0, 1e-4, 10.0, 101, 128).value for sigma in (1e-4, 2e-4, 4e-4)]
    )
    assert radii.min() / radii.max() > 0.99


def test_a3_a4_scaled_information_approaches_a_constant():
    b = quantile_template(5)
    sigma = np.array([1e-4, 2e-4, 4e-4])
    pr = np.array([gaussian_eig(s, 20.0 * s, b, 1.0, 128) / s**4 for s in sigma])
    fisher_radius = optimize_fisher(b, 1.0, 1e-4, 10.0, 101).value
    physical = np.array([gaussian_eig(s, fisher_radius, b, 1.0, 128) / s**2 for s in sigma])
    assert pr.max() / pr.min() < 1.01
    assert physical.max() / physical.min() < 1.001

