import numpy as np

from figure2_experiments.exp_c_noiseless.run import random_theory, rc_theory, simulate_macro


def test_c1_c2_c3_empirical_box_simulation_matches_analytic_expectations():
    d, m, t_max, n = 4, 4, 16, 100_000
    sums = simulate_macro(1234, d, m, n, t_max)
    empirical = np.sqrt(sums / n)
    t = np.arange(t_max + 1)
    rc = rc_theory(t, d, m)
    random = random_theory(t, d, m)
    assert np.max(np.abs(empirical[0] / rc - 1.0)) < 0.012
    assert np.max(np.abs(empirical[1] / random - 1.0)) < 0.02
    assert np.max(np.abs(empirical[2] / random - 1.0)) < 0.02
    assert np.max(np.abs(empirical[2] / empirical[1] - 1.0)) < 0.025

