# Experiment B long-horizon extension report

All statistical and algorithmic definitions are inherited unchanged from the paper Experiment B. The run resumes saved ADF states at n=8192 and uses regenerated deterministic common-random-number streams.

## Implementation inventory (unchanged)

- M=5, beta=1, target and initial prior N(0,1).
- Centered unit-RMS Gaussian equal-mass quantile template.
- Gaussian ADF response-conditioned moment matching with 128-node Gauss-Hermite quadrature.
- PR-EIG alpha interval [0.05,20], physical r interval [1e-4,10], and the original 600-point log-sigma design cache.
- Same SeedSequence([20260818, trajectory]) target draw and subsequent Uniform stream for all three policies.

## Reproduction gate

- Passed: True
- Replayed trajectories: 20
- Maximum checkpoint absolute difference: 0.000e+00
- Maximum original-slope difference: 5.551e-17

## Q1. Terminal local slopes at n=65536

Method | [n/2,n] slope | 95% CI | derivative estimate
---|---:|---:|---:
PR-EIG | -0.259072822 | [-0.259072846, -0.259072796] | -0.257435720
Exact-RC | -0.499986801 | [-0.499986845, -0.499986754] | -0.499997406
Fisher-RC | -0.499989663 | [-0.499989697, -0.499989626] | -0.499999430

## Q2. PR-EIG alpha-boundary contact

range_start | range_end | trajectories | total_hits | total_rounds | mean_trajectory_hit_rate | trajectories_with_any_hit | boundary_hit_rate
---|---|---|---|---|---|---|---
8192 | 16384 | 1000 | 8192000 | 8192000 | 1 | 1000 | 1
16384 | 32768 | 1000 | 16384000 | 16384000 | 1 | 1000 | 1
32768 | 65536 | 1000 | 32768000 | 32768000 | 1 | 1000 | 1

The upper alpha boundary is active in every late-stage PR-EIG round; it does not disappear by n=65536.

## Q3. Exact-RC and Fisher-RC

At n=65536, mean selected r is 1.997478175 for Exact-RC and 1.997461748 for Fisher-RC (absolute difference 1.643e-05). Their terminal [n/2,n] slopes are reported above and both are within 1.4e-5 of -1/2.

## Q4. Posterior RMS and variance

method | n | epsilon | ci_low | ci_high
---|---|---|---|---
PR-EIG | 8192 | 0.0145961 | 0.0145961 | 0.0145961
PR-EIG | 65536 | 0.00842234 | 0.00842234 | 0.00842234
Exact-RC | 8192 | 0.00809996 | 0.00809995 | 0.00809996
Exact-RC | 65536 | 0.00286394 | 0.00286394 | 0.00286394
Fisher-RC | 8192 | 0.00810008 | 0.00810008 | 0.00810008
Fisher-RC | 65536 | 0.00286394 | 0.00286394 | 0.00286394

Aggregate posterior variance decreases at every stored checkpoint for all three methods. All raw checkpoint variances are finite and positive.

## Q5. Numerical incidents

```json
{
  "Exact-RC": {
    "aggregate_variance_increase_count": 0,
    "cache_clip_count": 0,
    "eig_nonpositive_count": 0,
    "largest_aggregate_variance_increase": 0.0,
    "probability_normalization_failure_count": 0,
    "probability_zero_count": 0,
    "variance_floor_count": 0
  },
  "Fisher-RC": {
    "aggregate_variance_increase_count": 0,
    "cache_clip_count": 0,
    "eig_nonpositive_count": 0,
    "largest_aggregate_variance_increase": 0.0,
    "probability_normalization_failure_count": 0,
    "probability_zero_count": 0,
    "variance_floor_count": 0
  },
  "PR-EIG": {
    "aggregate_variance_increase_count": 0,
    "cache_clip_count": 0,
    "eig_nonpositive_count": 0,
    "largest_aggregate_variance_increase": 0.0,
    "probability_normalization_failure_count": 0,
    "probability_zero_count": 0,
    "variance_floor_count": 0
  }
}
```

## Q6. Classification

**Case B** — PR-EIG continues moving toward -1/4, but the pre-registered terminal criterion is not yet met.

## Exact-grid overlap validation

method | n | trajectories | median_absolute_mean_difference | median_relative_variance_difference | q95_relative_variance_difference | median_relative_r_difference | q95_relative_r_difference | median_relative_eig_difference | q95_relative_eig_difference
---|---|---|---|---|---|---|---|---|---
Exact-RC | 512 | 10 | 0.000270254 | 0.00111647 | 0.0018635 | 1.22043e-06 | 1.8901e-06 | 0.00111555 | 0.00186197
Exact-RC | 1024 | 10 | 0.000131562 | 0.00057719 | 0.00109762 | 3.17719e-07 | 5.86091e-07 | 0.000576907 | 0.00109716
Exact-RC | 2048 | 10 | 6.80399e-05 | 0.000263568 | 0.000556654 | 7.88506e-08 | 1.64525e-07 | 0.000263475 | 0.000556577
Exact-RC | 4096 | 10 | 3.26983e-05 | 0.000128668 | 0.000361333 | 2.2553e-08 | 4.13321e-08 | 0.000128655 | 0.000361295
Exact-RC | 8192 | 10 | 1.71046e-05 | 0.000153145 | 0.000264871 | 3.55669e-09 | 9.11224e-09 | 0.000153147 | 0.000264868
Fisher-RC | 512 | 10 | 4.59234e-05 | 0.000259954 | 0.000889694 | 0 | 0 | 0.000259837 | 0.000889051
Fisher-RC | 1024 | 10 | 2.22055e-05 | 0.000191179 | 0.000577913 | 0 | 0 | 0.000191143 | 0.000577716
Fisher-RC | 2048 | 10 | 1.08583e-05 | 0.000187936 | 0.000362644 | 0 | 0 | 0.000187936 | 0.000362607
Fisher-RC | 4096 | 10 | 5.69145e-06 | 0.000181414 | 0.000276689 | 0 | 0 | 0.000181395 | 0.00027666
Fisher-RC | 8192 | 10 | 3.01082e-06 | 0.000159246 | 0.000214023 | 0 | 0 | 0.000159247 | 0.000214022
PR-EIG | 512 | 10 | 0.000346702 | 0.00136984 | 0.0022981 | 0.000684879 | 0.00114932 | 0.00203114 | 0.00341341
PR-EIG | 1024 | 10 | 0.000194857 | 0.000847966 | 0.00153332 | 0.000424073 | 0.000766788 | 0.00144718 | 0.00260468
PR-EIG | 2048 | 10 | 0.000125244 | 0.000440362 | 0.000968485 | 0.000220157 | 0.000484368 | 0.000769486 | 0.00179596
PR-EIG | 4096 | 10 | 7.78647e-05 | 0.000260547 | 0.000851584 | 0.000130278 | 0.000425886 | 0.000489958 | 0.00160199
PR-EIG | 8192 | 10 | 5.50023e-05 | 0.000308551 | 0.00067971 | 0.000154288 | 0.000339914 | 0.000602279 | 0.00131497

## Fixed-window slopes

method | analysis | fit_min_n | fit_max_n | point_slope | bootstrap_mean | bootstrap_median | ci_low | ci_high | theory_slope | difference_from_theory | r2 | n_points | bootstrap_samples
---|---|---|---|---|---|---|---|---|---|---|---|---|---
PR-EIG | fixed | 2048 | 8192 | -0.291422 | -0.291422 | -0.291422 | -0.291422 | -0.291421 | -0.25 | -0.0414215 | 0.99966 | 3 | 2000
PR-EIG | fixed | 4096 | 16384 | -0.275296 | -0.275296 | -0.275296 | -0.275297 | -0.275296 | -0.25 | -0.0252964 | 0.999783 | 17 | 2000
PR-EIG | fixed | 8192 | 32768 | -0.266845 | -0.266845 | -0.266845 | -0.266846 | -0.266845 | -0.25 | -0.0168455 | 0.999949 | 31 | 2000
PR-EIG | fixed | 16384 | 65536 | -0.261205 | -0.261205 | -0.261205 | -0.261205 | -0.261205 | -0.25 | -0.011205 | 0.999978 | 31 | 2000
PR-EIG | confirmatory | 8192 | 65536 | -0.264025 | -0.264025 | -0.264025 | -0.264025 | -0.264025 | -0.25 | -0.0140253 | 0.999925 | 46 | 2000
PR-EIG | original_preserved | 512 | 8192 | -0.321927 | -0.321927 | -0.321927 | -0.321928 | -0.321926 | -0.25 | -0.0719268 | 0.996438 | 5 | 2000
PR-EIG | half_window_local | 4096 | 8192 | -0.282106 | -0.282106 | -0.282106 | -0.282106 | -0.282106 | -0.25 | -0.0321062 | 1 | 2 | 2000
PR-EIG | half_window_local | 8192 | 16384 | -0.270533 | -0.270533 | -0.270533 | -0.270533 | -0.270533 | -0.25 | -0.020533 | 0.999978 | 16 | 2000
PR-EIG | half_window_local | 16384 | 32768 | -0.263552 | -0.263552 | -0.263552 | -0.263552 | -0.263552 | -0.25 | -0.0135522 | 0.999991 | 16 | 2000
PR-EIG | half_window_local | 32768 | 65536 | -0.259073 | -0.259073 | -0.259073 | -0.259073 | -0.259073 | -0.25 | -0.00907282 | 0.999996 | 16 | 2000
Exact-RC | fixed | 2048 | 8192 | -0.499867 | -0.499867 | -0.499867 | -0.499868 | -0.499866 | -0.5 | 0.000133001 | 1 | 3 | 2000
Exact-RC | fixed | 4096 | 16384 | -0.499934 | -0.499934 | -0.499934 | -0.499935 | -0.499934 | -0.5 | 6.55442e-05 | 1 | 17 | 2000
Exact-RC | fixed | 8192 | 32768 | -0.499964 | -0.499964 | -0.499964 | -0.499965 | -0.499964 | -0.5 | 3.55503e-05 | 1 | 31 | 2000
Exact-RC | fixed | 16384 | 65536 | -0.499981 | -0.499981 | -0.499981 | -0.499981 | -0.499981 | -0.5 | 1.88406e-05 | 1 | 31 | 2000
Exact-RC | confirmatory | 8192 | 65536 | -0.499973 | -0.499973 | -0.499973 | -0.499973 | -0.499973 | -0.5 | 2.71635e-05 | 1 | 46 | 2000
Exact-RC | original_preserved | 512 | 8192 | -0.499706 | -0.499706 | -0.499706 | -0.499707 | -0.499705 | -0.5 | 0.000294015 | 1 | 5 | 2000
Exact-RC | half_window_local | 4096 | 8192 | -0.499908 | -0.499908 | -0.499908 | -0.499909 | -0.499908 | -0.5 | 9.17901e-05 | 1 | 2 | 2000
Exact-RC | half_window_local | 8192 | 16384 | -0.499953 | -0.499953 | -0.499953 | -0.499953 | -0.499952 | -0.5 | 4.74421e-05 | 1 | 16 | 2000
Exact-RC | half_window_local | 16384 | 32768 | -0.499975 | -0.499975 | -0.499975 | -0.499975 | -0.499974 | -0.5 | 2.54835e-05 | 1 | 16 | 2000
Exact-RC | half_window_local | 32768 | 65536 | -0.499987 | -0.499987 | -0.499987 | -0.499987 | -0.499987 | -0.5 | 1.31985e-05 | 1 | 16 | 2000
Fisher-RC | fixed | 2048 | 8192 | -0.499901 | -0.499901 | -0.499901 | -0.499901 | -0.499901 | -0.5 | 9.89441e-05 | 1 | 3 | 2000
Fisher-RC | fixed | 4096 | 16384 | -0.49995 | -0.49995 | -0.49995 | -0.499951 | -0.49995 | -0.5 | 4.96293e-05 | 1 | 17 | 2000
Fisher-RC | fixed | 8192 | 32768 | -0.499973 | -0.499973 | -0.499973 | -0.499973 | -0.499973 | -0.5 | 2.73301e-05 | 1 | 31 | 2000
Fisher-RC | fixed | 16384 | 65536 | -0.499985 | -0.499985 | -0.499985 | -0.499985 | -0.499985 | -0.5 | 1.46775e-05 | 1 | 31 | 2000
Fisher-RC | confirmatory | 8192 | 65536 | -0.499979 | -0.499979 | -0.499979 | -0.499979 | -0.499979 | -0.5 | 2.09693e-05 | 1 | 46 | 2000
Fisher-RC | original_preserved | 512 | 8192 | -0.499787 | -0.499787 | -0.499787 | -0.499788 | -0.499786 | -0.5 | 0.000212649 | 1 | 5 | 2000
Fisher-RC | half_window_local | 4096 | 8192 | -0.499931 | -0.499931 | -0.499931 | -0.499931 | -0.499931 | -0.5 | 6.90739e-05 | 1 | 2 | 2000
Fisher-RC | half_window_local | 8192 | 16384 | -0.499964 | -0.499964 | -0.499964 | -0.499964 | -0.499964 | -0.5 | 3.62358e-05 | 1 | 16 | 2000
Fisher-RC | half_window_local | 16384 | 32768 | -0.49998 | -0.49998 | -0.49998 | -0.49998 | -0.49998 | -0.5 | 1.98212e-05 | 1 | 16 | 2000
Fisher-RC | half_window_local | 32768 | 65536 | -0.49999 | -0.49999 | -0.49999 | -0.49999 | -0.49999 | -0.5 | 1.03375e-05 | 1 | 16 | 2000
