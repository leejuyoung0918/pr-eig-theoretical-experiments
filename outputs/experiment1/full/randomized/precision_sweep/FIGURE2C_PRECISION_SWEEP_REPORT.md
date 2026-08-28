# Figure 2(c) precision-sweep report

This report is computed mechanically from saved nominal-response trajectories. The
environment is the bootstrap unit. Within each environment, a threshold time is the
median over 4 response replicates with censoring
treated as infinity; at least 3 replicates must reach the threshold.

## Configuration

- Environments: 500
- Response replicates per environment: 4
- Old horizon: 400 queries
- Extended horizon: 1000 queries
- Thresholds: 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02
- Posterior approximation: projected particle ADF (unchanged)
- Ratio inclusion: PR-EIG and the calibrated method must both have a finite
  environment-level median threshold time.
- Main shadow: across-environment IQR. Bootstrap intervals are saved separately.

## Results

1. At rho*=0.25, the median Fisher/PR ratio is
   1.008989 (IQR
   [1.000000, 1.047619]);
   the median Exact/PR ratio is 1.000000 (IQR
   [1.000000, 1.000000]).
2. The first median ratio below one is Fisher/PR: rho*=0.07;
   Exact/PR: rho*=0.07. A median below one means fewer queries than PR-EIG.
3. The first median-bootstrap CI wholly below one is Fisher/PR: rho*=0.05;
   Exact/PR: rho*=0.05. The first full environment IQR below one is Fisher/PR:
   rho*=0.03; Exact/PR: rho*=0.03.
4. At the finest rho*=0.02, Fisher/PR is
   0.923592 from
   31 comparable environments;
   Exact/PR is 0.922159 from
   31 environments.
5. Fisher qualitatively tracks Exact: the maximum absolute gap between their threshold-wise
   median ratios is 0.011765.
6. At rho*=0.02, environment success rates are PR
   0.062, Exact 0.074, and Fisher
   0.074. Coverage declines as precision becomes stricter;
   every plotted threshold reports its comparable-environment count.
7. At the old 400-query horizon, the finest-threshold Fisher/PR comparison had only
   4 comparable environments.
   The extension raises this to 31; it was
   required to make the fine-precision sweep interpretable.
8. The maximum absolute difference between every old 400-query rho path and the prefix
   of its extended trajectory is 0.

## Interpretation

RC becomes useful only at fine precision in this randomized benchmark.

This conclusion is descriptive for the randomized projected-ADF benchmark. The retained
rho*=0.25 scatter remains an appendix diagnostic and is not overwritten or hidden.
