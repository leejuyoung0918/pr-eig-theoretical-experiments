# Figure 2 experiments

This directory is an independent simulation package for three distinct claims:

1. finite-`beta` noisy MNL one-step information collapse;
2. finite-`beta` noisy MNL long-horizon sequential convergence;
3. a separate noiseless M-way coordinate-allocation benchmark.

## Redesigned Experiment 1 (final Figure 2)

The paper-facing Experiment 1 now has a separate, staged entry point. It replaces
the main noiseless panel with a paired randomized finite-beta benchmark while
preserving the original noiseless theorem check in the appendix:

1. controlled exact information scaling;
2. long-horizon scalar sequential consequence;
3. randomized multidimensional robustness.

The main randomized panel is now a predeclared precision sweep over
`rho* in {0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02}`.
It plots the environment-median Fisher/PR and Exact/PR query ratios with
across-environment IQR bands. The historical `rho*=0.25` scatter remains an
appendix diagnostic.

Run the lightweight validation before the paper-quality profile:

```bash
python3 -m figure2_experiments.run_experiment1 --mode quick --workers 4
python3 -m figure2_experiments.run_experiment1 --mode full --workers 8
python3 -m figure2_experiments.experiment1.plot --mode full --figure-c-only
python3 -m figure2_experiments.experiment1.report --mode full
```

All stages resume from deterministic batch/environment checkpoints unless
`--force` is supplied. Components may be selected independently, for example:

```bash
python3 -m figure2_experiments.run_experiment1 --mode full \
  --components controlled sequential analyze report --workers 8
```

The plotting command reads saved `full` results only and contains no random-data
generation. The final PDFs and 600-dpi preview are written to
`outputs/figure2/figures/`; appendix figures are written to
`outputs/figure2/supplementary/experiment1_redesign/`. The machine-readable and
human-readable terminal reports are `outputs/experiment1/full/FINAL_REPORT.json`
and `outputs/experiment1/full/FINAL_REPORT.md`.

The `full` profile uses 121 controlled sigma values; 200 sequential repetitions
through 100,000 queries with 2,000 bootstrap samples; and 500 randomized
environments, four nested response replicates, and 2,000 directional particles.
The original randomized benchmark remains at `T_max=400`. Its fine-threshold
coverage was insufficient (only four comparable Fisher/PR environments at
`rho*=0.02`), so the nominal-response trajectories were extended to `T_max=1000`
without changing environment distributions, methods, or approximation. Every old
400-query rho-path prefix matches its extension exactly. The randomized posterior is explicitly a
projected particle ADF: exact MNL likelihood weights provide directional moments,
followed by rank-one Gaussian moment projection and an equal-weight reset. It is
not claimed to be a generic exact high-dimensional particle posterior.

On the tested 8-worker Apple Silicon host, the saved full controlled and
sequential stages took about 1.5 s and 46 s; the three-lapse randomized stage took
about 421 s after its resolution library and particle-count checks were available.
Timing is hardware-dependent. The report retains the lack of an RC advantage at
`rho*=0.25`, while the mechanically evaluated sweep shows both calibrated medians
crossing below one at `rho*=0.07` and median-bootstrap CIs below one from
`rho*=0.05`. Coverage falls to 31 comparable environments at `rho*=0.02`, which is
reported rather than hidden.

Experiment C is not used to justify the noisy MNL rates in Experiments A/B. Analytic expressions in C and dashed slope guides in A/B are plotted only as theory; no empirical observation is generated from a theoretical error formula.

## Model and template normalization

For a posterior-centered scalar coordinate `s`, physical radius `r`, inverse temperature `beta`, and template `b`, the likelihood is

```text
P(Y=m | s,r) = exp[-beta (s-r b_m)^2] / sum_j exp[-beta (s-r b_j)^2].
```

The implementation evaluates log probabilities with max-shifted log-sum-exp. The default `M=5` template starts from equal-mass standard-normal quantile centers

```text
b_m(raw) = Phi^{-1}((m+1/2)/M),  m=0,...,M-1.
```

It then subtracts the floating-point mean and divides by `sqrt(mean(b**2))`. Thus the final template has exactly zero numerical mean and unit RMS. This same normalization is used for all `M in {2,3,5,8}` robustness settings.

## Numerical estimators

Experiment A uses deterministic Gauss-Hermite quadrature, not particles. The primary order is 128 and the exhaustive order is 256. A 64/128/256 comparison is written before rate interpretation. The KL integrand is evaluated as

```text
q * [(1+x) log(1+x) - x],  x=(p-q)/q,
```

which analytically removes the cancelling first-order term when `p` and `q` are nearly equal. The pre-registered numerical floor is `I <= 1e-22`; flagged values remain in raw data but are excluded from slope fits.

Both PR-EIG and Exact-RC use a logarithmic coarse grid followed by the same bounded scalar refinement in log radius. PR-EIG keeps the pre-registered bounded interval `alpha in [0.05,20]`; contact with that boundary is recorded, and the main setting also evaluates `[0.005,200]` as a sensitivity diagnostic. The primary PR curve is never silently replaced by the expanded interval. Physical search starts at `r in [1e-4,10]` and expands by decades only if the optimum touches a boundary. Fisher-RC maximizes the exact Fisher information at `s=0`, but its plotted y value is exact quadrature EIG at the selected Fisher radius.

Experiment B uses Gaussian assumed-density filtering. Every response-conditioned normalization, mean, and variance is obtained by integrating the actual MNL likelihood with Gauss-Hermite quadrature, then projecting only its first two moments to a Gaussian. No target rate is present in the update. A dense adaptive-grid posterior validates posterior moments, selected resolution, and EIG over an overlapping horizon. For speed, design and response-conditioned transitions are cached on a dense `log(sigma)` grid. Geometric midpoints are checked against direct optimization and direct quadrature; all interpolation errors are saved.

The three sequential policies for a given trajectory share the same prior draw and the same round-by-round uniform variates. Each policy maps that common uniform through its own categorical inverse CDF.

Experiment C samples actual targets uniformly inside the initial box, performs actual M-way cell updates, and computes Euclidean squared errors from the final cell midpoints. Its analytic curves are computed only during analysis.

## Fixed fit rules

- A primary: `sigma <= 0.03`; sensitivities: `sigma <= 0.1, 0.03, 0.01`.
- B primary: `n >= 512`; sensitivities: `n >= 256, 512, 1024, 2048`.
- C: RC slopes use pre-registered balanced-cycle endpoints `T % d == 0` and `T/d >= 2`, which removes the known arbitrary-T within-cycle oscillation without selecting points from the observed values.

The smoke profile ends at `n=256`, so it intentionally does not report a primary B slope. It verifies plumbing and numerical sanity only.

## Installation and execution

From the workspace parent directory:

```bash
python3 -m pip install -r figure2_experiments/requirements.txt
python3 figure2_experiments/run_all.py --mode smoke
python3 figure2_experiments/run_all.py --mode paper
python3 figure2_experiments/run_all.py --mode exhaustive
```

From inside this directory, the requested shorthand is also supported:

```bash
python run_all.py --mode smoke
python run_all.py --mode paper
python run_all.py --mode exhaustive
```

Useful controls:

```bash
python run_all.py --mode paper --workers 8
python run_all.py --mode paper --experiments a c --skip-plot
python run_all.py --mode smoke --force
```

### Experiment B long-horizon extension

The publication B run remains unchanged at 1,000 trajectories x 8,192 queries. Its
saved ADF states and deterministic common-random-number streams can be extended to
65,536 queries without modifying or overwriting the original output:

```bash
python3 -m figure2_experiments.run_exp_b_long
```

The extension first replays 20 complete trajectories through query 8,192 and
requires checkpoint fields plus the original `[512,8192]` slopes to agree within
`1e-13`. It then runs gated stages of 20, 100, and 1,000 trajectories. Completed
batch files are immutable resume points. New results live only under
`outputs/figure2/exp_b_long/`. The pre-registered slope windows are
`[2048,8192]`, `[4096,16384]`, `[8192,32768]`, `[16384,65536]`, and confirmatory
`[8192,65536]`; `[512,8192]` is retained as the original-window comparison.

`--force` deliberately discards matching completed checkpoints for the selected profile. Without it, every A setting, B trajectory batch, B design cache, and C macro-seed resumes from disk.

## Publication-ready Figure 2

The final plotting command reads existing result files only. It does not run an
experiment, refit a slope, smooth a curve, or alter a confidence interval:

```bash
python3 -m figure2_experiments.plotting.plot_figure2 \
  --exp-a figure2_experiments/outputs/figure2/exp_a \
  --exp-b figure2_experiments/outputs/figure2/exp_b \
  --exp-c figure2_experiments/outputs/figure2/exp_c \
  --output figure2_experiments/outputs/figure2/figures/figure2_combined_final.pdf
```

The loaders and panel renderers are independent (`load_exp_a`, `load_exp_b`,
`load_exp_c`, `plot_panel_a`, `plot_panel_b`, `plot_panel_c`). To replace only
the sequential panel with the completed long-horizon result, change one argument:

```bash
python3 -m figure2_experiments.plotting.plot_figure2 \
  --exp-a figure2_experiments/outputs/figure2/exp_a \
  --exp-b figure2_experiments/outputs/figure2/exp_b_long \
  --exp-c figure2_experiments/outputs/figure2/exp_c \
  --output figure2_experiments/outputs/figure2/figures/figure2_combined_long.pdf
```

The standard final combined figure is authored at 7.05 inches wide for a
two-column `figure*` environment, and its PNG preview is exported at 600 dpi.

Run tests from the workspace parent:

```bash
python3 -m pytest figure2_experiments/tests
```

## Execution profiles and expected cost

| profile | A sigma/order | B trajectories x horizon | B bootstrap | C targets per `(d,M)` | intended use |
|---|---:|---:|---:|---:|---|
| smoke | 41 / 128 | 20 x 256 | 200 | 4,000 | plumbing and unit checks |
| paper | 121 / 128 | 1,000 x 8,192 | 2,000 | 100,000 | manuscript result |
| exhaustive | 181 / 256, down to `1e-5` | 2,000 x 16,384 | 5,000 | 1,000,000 | precision/stress run |

Wall time is hardware-dependent. The transition cache makes the long B trajectories inexpensive after cache construction; A's full 20-setting robustness grid and B's exact-grid validation dominate paper mode. Typical expectations are minutes for smoke, tens of minutes to several hours for paper, and many CPU-hours plus substantial disk/memory bandwidth for exhaustive. Metadata records elapsed time, CPU count, platform, Python/NumPy versions, config hash, and a Git commit when available.

## Checkpoints and outputs

All outputs live under `outputs/figure2/`:

```text
exp_a/
  raw/main.parquet
  raw/robustness.parquet
  slopes.csv
  robustness.csv
  quadrature_validation.csv
  diagnostics/*.png
exp_b/
  raw/trajectories.parquet
  raw/batches/*.npz
  slopes.csv
  bootstrap.csv
  curve.csv
  cache_validation.csv
  exact_grid_validation.csv
  diagnostics/*.png
exp_c/
  raw/macros/*.npz
  raw/macro_summary.parquet
  slopes.csv
  full_grid.csv
figure2/
  figure2a.pdf
  figure2b.pdf
  figure2c.pdf
  figure2_combined.pdf
  figure2_combined.png
REPORT.md
run_metadata.json
```

B trajectory rows include trajectory/seed/method/query count, true target, posterior mean and variance, directional RMS, true estimation error, selected radius and alpha, exact EIG, and Fisher information at logarithmic checkpoints including every power of two. C raw files contain macro-level sums rather than theoretical samples, enabling exact resume and macro-seed confidence intervals.

## Interpretation safeguards

- PR boundary contact is disclosed, not optimized away.
- Exact-RC and Fisher-RC share the exact-EIG y metric.
- Joint-EIG equals Exact-RC in one dimension and is recorded as an equality sanity check rather than duplicated in the figure.
- Fit windows are fixed in configuration before results are loaded.
- Numerical-floor flags remain visible in raw outputs.
- Random-coordinate and entropy-tie use independent uniform coordinate draws and are compared statistically.
- The universal lower bound is labeled as a bound, never as a policy.
