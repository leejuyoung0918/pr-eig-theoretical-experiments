# Figure 2 simulation report

Execution profile: `paper`; config hash: `88f8def3ac47d044`.

## Experiment A - finite-beta noisy MNL information collapse

- beta = 1.0
- M = 5
- sigma range = [0.0001, 1.0]
- quadrature order = 128
- alpha range = [0.05, 20.0]
- initial physical r range = [0.0001, 10.0] (automatically expanded only on boundary contact)
- bounded PR optimum touched alpha=20 at 91 of 121 main-grid points; expanded-alpha sensitivity is retained in raw output

- PR-EIG slope = 3.9743 (theory 4)
- Exact-RC slope = 1.9999 (theory 2)
- Fisher-RC slope = 1.9999 (theory 2)

### Resolution robustness
- M=3: r*sqrt(beta) mean 0.987608, range [0.987608, 0.987609].
- M=5: r*sqrt(beta) mean 1.997462, range [1.997461, 1.997462].
- Important failure mode: for the symmetric even-M templates M=2 and M=8, the physical/Fisher optimum remained on the repeatedly expanded upper boundary. The central equidistant pair makes local Fisher information grow with radius, so a finite nonzero calibration optimum and the sigma^2 robustness claim do not apply to these even-M settings. Their raw slopes and boundary flags are preserved.

## Experiment B - finite-beta noisy MNL sequential localization

- trajectories = 1000
- N_max = 8192
- posterior method = Gaussian assumed-density filtering via exact Gauss-Hermite moment matching
- quadrature order = 128
- primary fit range = [512, 8192]

- PR-EIG slope = -0.321927; 95% CI [-0.321928, -0.321926]
- Exact-RC slope = -0.499706; 95% CI [-0.499707, -0.499705]
- Fisher-RC slope = -0.499787; 95% CI [-0.499788, -0.499786]
- Pre-asymptotic disclosure: the pre-registered PR fit has not reached -1/4. The sliding slope should be inspected at the final checkpoint; no fit window was changed after seeing this result.

Sliding slopes are stored in `exp_b/diagnostics/local_slopes.csv`; ADF/exact-grid discrepancies are stored in `exp_b/exact_grid_validation.csv`.

## Experiment C - separate noiseless M-way allocation benchmark

Targets per (d,M) = 1000000

| d | M | RC empirical slope | theory slope | difference | 95% CI | random max rel. discrepancy | lower-bound violations |
|---:|---:|---:|---:|---:|:---:|---:|---:|
| 4 | 2 | -0.69311 | -0.69315 | 4.21e-05 | [-0.69319, -0.69301] | 0.00174 | 0 |
| 4 | 4 | -1.38618 | -1.38629 | 0.00011 | [-1.38625, -1.38612] | 0.00884 | 0 |
| 4 | 8 | -2.07941 | -2.07944 | 3.21e-05 | [-2.07950, -2.07931] | 0.00868 | 0 |
| 4 | 16 | -2.77257 | -2.77259 | 1.68e-05 | [-2.77267, -2.77248] | 0.0173 | 0 |
| 8 | 2 | -0.69315 | -0.69315 | -6.22e-06 | [-0.69323, -0.69308] | 0.00165 | 0 |
| 8 | 4 | -1.38630 | -1.38629 | -5.95e-06 | [-1.38637, -1.38624] | 0.00816 | 0 |
| 8 | 8 | -2.07942 | -2.07944 | 2.59e-05 | [-2.07948, -2.07936] | 0.00992 | 0 |
| 8 | 16 | -2.77256 | -2.77259 | 2.52e-05 | [-2.77260, -2.77253] | 0.0099 | 0 |
| 16 | 2 | -0.69313 | -0.69315 | 1.79e-05 | [-0.69317, -0.69309] | 0.00162 | 0 |
| 16 | 4 | -1.38631 | -1.38629 | -1.82e-05 | [-1.38637, -1.38626] | 0.00894 | 0 |
| 16 | 8 | -2.07945 | -2.07944 | -6.53e-06 | [-2.07949, -2.07940] | 0.00835 | 0 |
| 16 | 16 | -2.77260 | -2.77259 | -1.56e-05 | [-2.77266, -2.77256] | 0.0248 | 0 |

The noisy finite-beta experiments (A/B) and the noiseless allocation benchmark (C) are intentionally analyzed as different theorems. Dashed/dotted plot curves are analytic guides or theory, never generated empirical observations.
