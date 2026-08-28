# Experiment 1 pre-change inspection

This inventory was completed before implementing the redesigned Experiment 1.

## Existing algorithms retained unchanged

- **Likelihood:** `mnl.py` evaluates the squared-distance multinomial-logit model
  using a max-shifted log-sum-exp implementation.
- **Template:** `templates.py` uses centered, unit-RMS standard-normal equal-mass
  quantile centers for every slate arity.
- **PR-EIG:** `policies.optimize_pr` optimizes exact Gaussian EIG over the fixed
  bounded interval `alpha in [0.05, 20]`, followed by the same log-radius scalar
  refinement used for physical optimization.
- **Exact-RC:** `policies.optimize_physical` optimizes exact Gaussian EIG over a
  fixed physical radius domain and expands the numerical domain only after a
  detected boundary optimum.
- **Fisher-RC:** `policies.optimize_fisher` maximizes the exact scalar Fisher
  information at posterior offset zero.  Exact EIG, rather than the Fisher
  approximation, remains the plotted information metric.
- **Posterior update:** the existing noisy sequential experiment uses Gaussian
  assumed-density filtering.  Every response-conditioned normalization, mean,
  and variance is computed from the actual MNL likelihood by Gauss-Hermite
  quadrature before Gaussian moment projection.
- **Response sampling:** targets and uniform streams are trajectory-addressable;
  methods apply the same uniform variate to their own categorical inverse CDF.
- **Direction/covariance:** the existing noisy code is scalar-only and therefore
  contains neither a multidimensional direction rule nor a multidimensional
  covariance update.

## Existing outputs and plotting path

`plotting/plot_figure2.py` reads saved outputs under `outputs/figure2/exp_a`,
`exp_b` (or `exp_b_long`), and `exp_c`, then writes
`figure2a_final.pdf`, `figure2b_final.pdf`, and `figure2c_final.pdf`.  The old
panel (c) is the separate noiseless box-localization benchmark.

The inspection found two plotting-only issues relevant to the redesign:

1. panel (a) theory guides were short, manually scaled segments;
2. the main panel (c) still showed the noiseless theorem benchmark instead of a
   randomized noisy robustness comparison.

## Reuse decisions

- Panel (a) reuses the exact quadrature and all three resolution optimizers.
- Panel (b) reuses the validated Gaussian-ADF transition cache and common-random-
  number construction, but runs a separate predeclared 100,000-query experiment.
- Panel (c) uses a separate multidimensional particle SMC posterior because no
  exact multidimensional posterior exists in the inspected code.  All methods
  share the same particles, environment, target, direction rule, template, and
  MNL response stream; only resolution selection differs.  Each exact weighted
  directional update is followed by an explicit Gaussian moment projection and
  equal-weight standardized directional particle reset, matching the existing
  Gaussian-ADF semantics while retaining particle-count sensitivity checks.
- The existing noiseless benchmark and all of its raw results remain unchanged
  and are routed to an appendix figure.
