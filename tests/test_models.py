import numpy as np

from figure2_experiments.eig import gaussian_eig
from figure2_experiments.mnl import mnl_log_probs, mnl_probs
from figure2_experiments.templates import quantile_template


def test_quantile_template_is_symmetric_centered_and_unit_rms():
    for m in (2, 3, 5, 8):
        b = quantile_template(m)
        assert np.allclose(b, -b[::-1], atol=1e-14)
        assert abs(b.mean()) < 1e-14
        assert np.isclose(np.mean(b * b), 1.0)


def test_mnl_is_normalized_and_log_stable():
    b = quantile_template(8)
    s = np.array([-1e3, -1.0, 0.0, 1.0, 1e3])
    probabilities = mnl_probs(s, 7.0, b, beta=8.0)
    log_probabilities = mnl_log_probs(s, 7.0, b, beta=8.0)
    assert np.all(np.isfinite(log_probabilities))
    assert np.allclose(probabilities.sum(axis=-1), 1.0)


def test_gauss_hermite_orders_agree_away_from_floor():
    b = quantile_template(5)
    for sigma, radius in ((0.3, 1.8), (0.03, 1.99), (0.03, 0.6)):
        values = np.array([gaussian_eig(sigma, radius, b, 1.0, order) for order in (64, 128, 256)])
        assert abs(values[1] - values[2]) / values[2] < 2e-5

