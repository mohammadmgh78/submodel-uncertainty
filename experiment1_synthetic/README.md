# Experiment 1: synthetic example (Section 5.2)

*Y = p(X1) + q(X2)* with *X1 ~ N(1, 0.5²)*, *X2 ~ N(1, 2²)*, *p(x) = x⁷ + x⁴ + 3x³ + sin(x) + 4* and
*q(x) = x³ + x² + 4x*; E[Y] = 49.13. The fitted submodels are normal distributions for *X1* and *X2*, a quadratic
polynomial for *p*, and a linear function for *q*.

| File | Content |
|---|---|
| `synthetic_model.py` | True submodels, fitting, bootstrap |
| `coverage.py` | Table 1 and Figure 1 |
| `factorial.py` | Figures 2 and 3 and the ANOVA |
| `importance.py` | Pilot, n\*, and importance scores: Figure 4, and Figure 5 with `--n-x2 1000` |
| `bagging.py` | Table 2 |
| `data/` | Saved runs of Table 1 (`coverage_*.json`) and of Figures 2 and 3 (`factorial_outputs.pkl`) |
| `results/` | Figures and tables; `table2_runs.npz` holds the runs of Table 2 |

## Settings

| | |
|---|---|
| Table 1 | 100 macro-replications, training size 50, B = 100, n = 100, 90% percentile intervals. The reported ± is the half-width of a 90% *t*-interval for the mean width |
| Figures 2 and 3 | 16 combinations of true and estimated submodels, 500 training datasets of size 50, n = 500 |
| Figures 4 and 5 | Training size 25 (X2: 1,000 in Figure 5), B = 8, S = 64, n0 = 10, τ = 0.1, 50 bagged trees, seed 1 |
| Table 2 | 20 training datasets, 20 repetitions each, n = 345, seeds 1000 to 1019 |

## Rerun the simulations

`importance.py` simulates every time it is run and gives identical results. `bagging.py --simulate` reruns Table 2 with
identical results. `coverage.py --simulate` and `factorial.py --simulate` run the
experiments again with a seed; the runs used in the paper were not seeded, so the new results are equivalent but not
identical.

`anova.csv` has two columns: the p-values of the ANOVA of the KPI estimates, and those of the variance of the estimator
(one value per combination).
