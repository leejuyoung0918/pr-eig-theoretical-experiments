# Experiment 1 final report

This report was generated from saved `full` results only (config hash
`03c08bf0c5be1668`). It does not rerun a simulation or refit a plotted result.

## Controlled theory verification

- Configuration: beta=1, M=5, sigma in
  [0.0001, 1], 121
  log-spaced values, Gauss-Hermite order 128.
- The common predeclared fit interval is [0.0001, 0.01].
- PR-EIG: slope 3.995858, SE 0.000535835,
  95% regression CI [3.994786, 3.996931],
  R2=0.999998939 (theory 4).
- Exact-RC: slope 1.999991, SE 1.18243e-06,
  95% regression CI [1.999988, 1.999993],
  R2=1.000000000 (theory 2).
- Fisher-RC: slope 1.999991, SE 1.18277e-06,
  95% regression CI [1.999988, 1.999993],
  R2=1.000000000 (theory 2).
- The largest 128-vs-256 quadrature relative discrepancy in the validation table is
  1.15e-07. Panel (a) reproduces (4,2,2) at the stated
  numerical precision. These regression intervals are deterministic fit uncertainty,
  not Monte Carlo intervals.

## Sequential consequence

- Configuration: 200 independent repetitions,
  N_max=100,000, Gaussian ADF with exact finite-beta MNL
  Gauss-Hermite moment matching (order 128), and
  2,000 trial-level bootstrap resamples.
- Primary predeclared fit window: n in [10,000,
  100,000].
- PR-EIG: slope -0.261509, regression CI
  [-0.262401, -0.260617], bootstrap CI
  [-0.261509, -0.261509];
  final RMS 0.007557060, CI
  [0.007557060, 0.007557060].
- Exact-RC: slope -0.499980, regression CI
  [-0.499982, -0.499978], bootstrap CI
  [-0.499980, -0.499980];
  final RMS 0.002318489, CI
  [0.002318489, 0.002318489].
- Fisher-RC: slope -0.499985, regression CI
  [-0.499986, -0.499983], bootstrap CI
  [-0.499985, -0.499984];
  final RMS 0.002318492, CI
  [0.002318492, 0.002318492].
- At n=100,000 the sliding slopes are PR=-0.255950,
  Exact=-0.499989, and
  Fisher=-0.499990. PR moves toward -1/4
  across the predeclared windows (-0.27775, -0.26810, -0.26151, -0.25815), but the
  finite horizon has not reached exactly -0.25. Exact and Fisher are effectively -1/2.
- The bootstrap bands are extremely narrow because, under the symmetric Gaussian ADF,
  the variance recursion is almost deterministic. They should not be interpreted as
  validation of a fully non-Gaussian posterior.

## Randomized robustness: precision-sweep main panel

- The nominal benchmark was extended from 400 to 1,000 queries while preserving every
  old 400-query rho-path prefix exactly (maximum absolute difference
  0).
- At rho*=0.25, median Fisher/PR is
  1.008989 and median Exact/PR is
  1.000000; there is no moderate-precision gain.
- Both median curves first fall below one at rho*=0.07. Their median-bootstrap CIs are
  wholly below one from rho*=0.05: Fisher/PR
  [0.979602, 0.995325],
  Exact/PR [0.979434, 0.995690].
- At rho*=0.03, the complete environment IQR is below one for both methods:
  Fisher/PR [0.904293, 0.997214],
  Exact/PR [0.900634, 0.998092].
- At rho*=0.02, median Fisher/PR is 0.923592
  and Exact/PR is 0.922159, but only
  31 environments are comparable.
  Success rates are PR=0.062,
  Exact=0.074, and
  Fisher=0.074; this finest point must be interpreted
  with its low coverage.
- Fisher tracks Exact closely: the maximum threshold-wise median-ratio gap is
  0.011765.

The precision sweep therefore supports the qualified conclusion that physical calibration
becomes operationally useful only at fine precision in this projected-ADF benchmark.

## Retained single-threshold randomized diagnostic

- Configuration: 500 paired environments,
  4 nested response repetitions,
  d in {2,4,8,16}, M in {3,5,7,9}, beta log-uniform on [0.25,4],
  kappa on [1,100], 2,000 directional particles,
  and T_max=400.
- Nominal Fisher-RC versus PR-EIG at rho<=0.25: Fisher wins
  49/500, ties
  155, PR wins 296; win rate
  0.098, environment-bootstrap CI
  [0.072, 0.124].
- Median Fisher/PR paired query ratio is 1.011472
  (IQR [1.000, 1.050],
  bootstrap CI [1.008000, 1.015957]).
  Success rates are 0.936 for PR and
  0.936 for Fisher.
- Median query saving (PR minus Fisher) is -0.500,
  bootstrap CI [-1.000, -0.500].
- Median log final-rho difference (Fisher minus PR) is
  -0.003226, CI
  [-0.005583, -0.001454].
  Median log normalized target-error difference is
  -0.015881, CI
  [-0.041041, 0.004252].
  The target-error direction favors Fisher, but its CI includes zero; a robust accuracy
  advantage is therefore not established.

### Retained moderate-precision result

The historical rho*=0.25 benchmark **does not show a Fisher-RC query-count advantage**.
The median ratio is above one and PR-EIG wins more environments. No seed, threshold, or
environment was removed. Exact-RC and PR-EIG tie in
  467/500 environments, showing that
  rho=0.25 is usually reached
before bounded posterior-relative resolution separates materially from direct physical
EIG calibration. This diagnostic is retained in the appendix with all censored cases.

## Approximation quality and cost

- Fisher/Exact median query ratio is 1.011664, CI
  [1.008380, 1.015957];
  Fisher is close but slightly slower at the primary threshold. Its median final
  log-rho difference relative to Exact is
  0.001430, CI
  [0.001233, 0.001795].
- Median uncached design time is
  3.958 ms
  (PR), 4.789 ms
  (Exact), and 0.291 ms
  (Fisher). Fisher is 13.6x faster
  than PR and 16.5x faster than Exact
  for uncached resolution design.
- The full three-lapse randomized stage took
  420.8 wall-clock seconds with
  8 workers. Per-query timing excludes plotting and disk I/O.
- Cached EIG relative error is at most 8.6e-05.
  The largest PR radius discrepancy is 3.358%
  at a flat/multiple optimum; its EIG discrepancy remains small. Particle-count results
  from 2,000 through
  20,000 keep the same success-rate and query-count
  conclusion.
- The multidimensional posterior is a documented projected particle ADF, not a generic
  exact high-dimensional particle posterior: exact MNL directional weights are used,
  then the first two moments are projected back to a Gaussian and equal weights are
  reset at every query. Tail ESS can be low (minimum observed fraction
  0.000500007), so panel (c) should be described as a
  moment-projected robustness benchmark.

## Sensitivity

- beta: ratio minimum 1.005 at 0.25; maximum 1.049 at 4.
- arity: ratio minimum 1.000 at 9; maximum 1.087 at 3.
- dimension: ratio minimum 1.000 at 2; maximum 1.017 at 16.
- anisotropy: ratio minimum 1.007 at 30-100; maximum 1.017 at 3-10.

Every subgroup median Fisher/PR ratio is at least one. Fisher's disadvantage is largest
for M=3 and beta near 4, smallest for M=9, d=2, and low beta; anisotropy shows no clear
monotone effect. These are descriptive subgroup results, not separately powered tests.

## Model mismatch

- At 5% lapse, median Fisher/PR ratio is 1.025641, CI
  [1.021277, 1.031735];
  success is PR=0.940, Fisher=0.938.
  The median log target-error difference is
  -0.017943, CI
  [-0.042992, 0.004778].
- At 10% lapse, the ratio is 1.042553, CI
  [1.036585, 1.046512];
  success is PR=0.944, Fisher=0.940.
  The median log target-error difference is
  -0.028913, CI
  [-0.055796, 0.004330].
- Mild misspecification does not recover a Fisher query-count advantage; the disadvantage
  grows with lapse in this benchmark. Actual target-error differences again have CIs
  that include zero.

## Noiseless theorem benchmark (appendix)

- 1,000,000 targets per (d,M) setting.
- Maximum absolute RC slope error relative to -log(M):
  0.00011.
- Maximum random-coordinate relative discrepancy from analytical theory:
  2.483%.
- Maximum random/entropy-tie discrepancy:
  2.990%.
- Universal lower-bound violations: 0.

The ideal noiseless result strongly verifies the exact allocation theorem and is kept
separate from the finite-beta noisy claims.

## Numerical diagnostics

- Probability-normalization failures: 0.
- Paired-environment parameter failures: 0.
- Unexpected NaN/Inf count: 0.
- Negative-EIG count: 0.
- Minimum covariance eigenvalue: 1.44e-05.
- Censored environment-method rows: 275;
  censored response runs: 1109.
- The physical Exact/Fisher library has zero boundary hits. PR touches its fixed alpha
  boundary as expected in the collapse regime; those cases are retained.
