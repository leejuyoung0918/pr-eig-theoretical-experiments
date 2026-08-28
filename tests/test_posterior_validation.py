import numpy as np

from figure2_experiments.eig import gaussian_adf_table
from figure2_experiments.exp_b_sequential.exact_grid import DenseGridPosterior
from figure2_experiments.templates import quantile_template


def test_b1_one_step_adf_matches_dense_grid_moments():
    b = quantile_template(5)
    radius = 1.8
    response = 1
    transition = gaussian_adf_table(1.0, radius, b, 1.0, 256)
    grid = DenseGridPosterior(32_769)
    grid.update(0.0, radius, b, 1.0, response)
    mean, variance = grid.moments()
    assert abs(mean - transition["mean_shift_sigma"][response]) < 2e-5
    assert abs(variance - transition["variance_ratio"][response]) < 2e-5

