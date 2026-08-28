import json

import numpy as np
import pandas as pd

from figure2_experiments.eig import gaussian_eig
from figure2_experiments.experiment1.config import get_experiment1_config
from figure2_experiments.experiment1.randomized import (
    ParticlePosterior,
    _response_uniforms,
    generate_environment,
)
from figure2_experiments.experiment1.precision_sweep import (
    crossing_matrix,
    environment_threshold_table,
    summarize_thresholds,
)
from figure2_experiments.experiment1.statistics import (
    bootstrap_rms_from_variances,
    first_threshold_crossings,
    log_linear_regression,
    normalized_posterior_rms,
)
from figure2_experiments.mnl import fisher_information_at_zero, mnl_probs
from figure2_experiments.policies import optimize_fisher
from figure2_experiments.templates import quantile_template


def test_redesign_mnl_likelihood_is_finite_and_normalized():
    template = quantile_template(9)
    offsets = np.linspace(-20.0, 20.0, 101)
    probabilities = mnl_probs(offsets, 3.0, template, beta=4.0)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=2e-15)


def test_redesign_exact_eig_is_nonnegative_and_quadrature_stable():
    template = quantile_template(5)
    for sigma, radius in ((1e-3, 0.02), (0.1, 1.7), (1.0, 1.2)):
        eig_128 = gaussian_eig(sigma, radius, template, beta=1.0, order=128)
        eig_256 = gaussian_eig(sigma, radius, template, beta=1.0, order=256)
        assert eig_128 >= 0.0
        assert abs(eig_128 - eig_256) / max(eig_256, 1e-300) < 2e-5


def test_redesign_fisher_optimizer_maximizes_the_declared_objective():
    template = quantile_template(5)
    optimum = optimize_fisher(template, 1.0, 1e-4, 10.0, 161)
    center = fisher_information_at_zero(optimum.value, template, 1.0)
    neighbors = fisher_information_at_zero(
        np.array([optimum.value * 0.98, optimum.value * 1.02]), template, 1.0
    )
    assert not optimum.boundary
    assert np.all(center >= neighbors - 1e-12)


def test_random_environment_covariance_has_declared_trace_condition_and_is_reproducible():
    config = get_experiment1_config("quick")
    first = generate_environment(config, 7)
    second = generate_environment(config, 7)
    np.testing.assert_array_equal(first.covariance, second.covariance)
    np.testing.assert_array_equal(first.target, second.target)
    eigenvalues = np.linalg.eigvalsh(first.covariance)
    assert np.all(eigenvalues > 0.0)
    assert np.isclose(np.trace(first.covariance), first.d * first.s0**2, rtol=1e-12)
    assert np.isclose(eigenvalues[-1] / eigenvalues[0], first.kappa, rtol=1e-10)


def test_normalized_rms_and_threshold_detection_use_predeclared_definitions():
    covariance = np.diag([0.25, 0.75])
    assert np.isclose(normalized_posterior_rms(covariance, initial_trace=4.0), 0.5)
    crossings = first_threshold_crossings(
        np.array([1.0, 0.7, 0.49, 0.3, 0.24, 0.11]), (0.5, 0.25, 0.1)
    )
    assert crossings == {0.5: 2, 0.25: 4, 0.1: None}


def test_trial_bootstrap_aggregates_sqrt_of_mean_variance_not_mean_rms():
    variance = np.array([[1.0, 4.0], [9.0, 16.0]])
    rms, low, high = bootstrap_rms_from_variances(variance, samples=500, seed=123)
    np.testing.assert_allclose(rms, np.sqrt([5.0, 10.0]))
    assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))
    assert np.all(low <= rms) and np.all(rms <= high)


def test_log_regression_recovers_exact_power_and_reports_finite_statistics():
    x = np.geomspace(1e-4, 1e-2, 31)
    y = 2.7 * x**4 * np.exp(1e-5 * np.sin(np.arange(len(x))))
    fit = log_linear_regression(x, y)
    assert abs(fit["slope"] - 4.0) < 1e-4
    assert fit["slope_se"] > 0.0
    assert fit["ci_low"] < fit["slope"] < fit["ci_high"]
    assert fit["r2"] > 0.999999


def test_particle_update_normalizes_weights_and_returns_psd_covariance():
    config = get_experiment1_config("quick")
    environment = generate_environment(config, 2)
    rng = np.random.default_rng(91)
    particles = rng.normal(size=800)
    particles = (particles - particles.mean()) / particles.std()
    posterior = ParticlePosterior(
        particles,
        seed=np.random.SeedSequence(92),
    )
    mean = np.zeros(environment.d)
    covariance = environment.covariance.copy()
    direction = np.linalg.eigh(covariance)[1][:, -1]
    mean, covariance, _, _ = posterior.update(
        mean,
        covariance,
        direction,
        radius=1.0,
        template=quantile_template(environment.m),
        beta=environment.beta,
        response=0,
    )
    assert abs(posterior.weights().sum() - 1.0) < 1e-14
    assert np.linalg.eigvalsh(covariance)[0] >= -1e-12
    assert np.all(np.isfinite(mean))


def test_precision_sweep_crossings_use_first_hit_and_preserve_censoring():
    paths = np.array(
        [
            [1.0, 0.8, 0.49, 0.24, 0.10],
            [1.0, 0.9, 0.7, 0.6, 0.55],
        ]
    )
    result = crossing_matrix(paths, (0.5, 0.25, 0.02))
    np.testing.assert_array_equal(result[0, :2], [2.0, 3.0])
    assert np.isinf(result[0, 2])
    assert np.all(np.isinf(result[1]))


def test_precision_sweep_environment_median_requires_strict_response_majority():
    records = []
    crossing_times = {
        "PR-EIG": (2, 2, 2, None),
        "Fisher-RC": (1, 2, 2, None),
        "Exact-RC": (1, 1, None, None),
    }
    for method, times in crossing_times.items():
        for response_replication, crossing in enumerate(times):
            path = np.ones(5)
            if crossing is not None:
                path[crossing:] = 0.1
            records.append(
                {
                    "environment_id": 0,
                    "method": method,
                    "response_replication": response_replication,
                    "rho_path_json": json.dumps(path.tolist()),
                    "d": 2,
                    "M": 3,
                    "beta": 1.0,
                    "s0": 1.0,
                    "kappa": 2.0,
                    "t_max": 4,
                    "lapse": 0.0,
                }
            )
    _, environment = environment_threshold_table(pd.DataFrame(records), (0.25,))
    fisher = environment[environment["comparison"] == "Fisher/PR"].iloc[0]
    exact = environment[environment["comparison"] == "Exact/PR"].iloc[0]
    assert fisher["PR_success"] and fisher["Fisher_success"]
    assert fisher["comparable"] and np.isclose(fisher["comparison_ratio"], 1.0)
    assert not exact["Exact_success"] and not exact["comparable"]
    summary = summarize_thresholds(environment, bootstrap_samples=50, seed=9)
    assert summary.loc[0, "Fisher_PR_comparable_environments"] == 1
    assert summary.loc[0, "Exact_PR_comparable_environments"] == 0


def test_precision_extension_preserves_original_full_response_prefix():
    config = get_experiment1_config("quick")
    expected = np.random.default_rng(
        np.random.SeedSequence([config.seed, 3_017, 11])
    ).random((4, 400))
    extended = _response_uniforms(config, 11, 4, 1_000)
    np.testing.assert_array_equal(extended[:, :400], expected)
    assert extended.shape == (4, 1_000)
