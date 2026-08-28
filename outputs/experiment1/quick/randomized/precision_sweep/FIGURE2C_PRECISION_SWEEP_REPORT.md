# Figure 2(c) precision-sweep report

This report is computed mechanically from saved nominal-response trajectories. The
environment is the bootstrap unit. Within each environment, a threshold time is the
median over 2 response replicates with censoring
treated as infinity; with four replicates, at least three must reach the threshold.

## Configuration

- Environments: 8
- Response replicates per environment: 2
- Old horizon: 400 queries
- Extended horizon: 120 queries
- Thresholds: 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02
- Posterior approximation: projected particle ADF (unchanged)
- Ratio inclusion: PR-EIG and the calibrated method must both have a finite
  environment-level median threshold time.
- Main shadow: across-environment IQR. Bootstrap intervals are saved separately.

## Results

1. At rho*=0.25, the median Fisher/PR ratio is
   1.084746 (IQR
   [1.003311, 1.144801]);
   the median Exact/PR ratio is 1.000000 (IQR
   [1.000000, 1.000000]).
2. The first median ratio below one is Fisher/PR: none;
   Exact/PR: rho*=0.07. A median below one means fewer queries than PR-EIG.
3. At the finest rho*=0.02, Fisher/PR is
   nan from
   0 comparable environments;
   Exact/PR is nan from
   0 environments.
4. Fisher qualitatively tracks Exact: the maximum absolute gap between their threshold-wise
   median ratios is 0.185185.
5. At rho*=0.02, environment success rates are PR
   0.000, Exact 0.000, and Fisher
   0.000. Coverage declines as precision becomes stricter;
   every plotted threshold reports its comparable-environment count.
6. At the old 400-query horizon, the finest-threshold Fisher/PR comparison had only
   0 comparable environments.
   The extension raises this to 0; it was
   required to make the fine-precision sweep interpretable.

## Interpretation

Exact and Fisher calibration diverge in the fine-precision regime.

This conclusion is descriptive for the randomized projected-ADF benchmark. The retained
rho*=0.25 scatter remains an appendix diagnostic and is not overwritten or hidden.
