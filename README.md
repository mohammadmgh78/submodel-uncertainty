# Submodel uncertainty in stochastic simulation and digital twins

Code and data for the numerical experiments of the paper *Quantifying and Attributing Submodel Uncertainty in Stochastic
Simulation Models and Digital Twins*.

| Folder | Experiment | Section |
|---|---|---|
| [`experiment1_synthetic`](experiment1_synthetic) | Synthetic example, *Y = p(X1) + q(X2)* | 5.2 |
| [`experiment2_contact_center`](experiment2_contact_center) | Digital twin of a contact center | 6.3 |

## Setup

Python 3.12. The results were produced with the package versions in `requirements.txt`.

```
pip install -r requirements.txt
```

`torch` and `torchbnn` are needed only for the contact center (the Bayesian routing submodels).

## Reproduce the figures and tables

The outputs of all simulation runs are included, so every figure and table can be regenerated without rerunning a
simulation of the contact center. Each command writes to the `results/` folder of its experiment.

| Result | Command | Output (in `results/`) | Time |
|---|---|---|---|
| | `cd experiment1_synthetic` | | |
| Table 1, Figure 1 | `python coverage.py` | `table1.csv`, `figure1.png` | seconds |
| Figures 2 and 3, ANOVA | `python factorial.py` | `figure2.png`, `figure3a.png`, `figure3b.png`, `bias_variance.csv`, `anova.csv` | seconds |
| Figure 4 | `python importance.py` | `figure4.png`, `figure4_shares.csv`, `figure4_summary.csv` | 1 min |
| Figure 5 | `python importance.py --n-x2 1000` | `figure5.png`, `figure5_shares.csv`, `figure5_summary.csv` | 1 min |
| Table 2 | `python bagging.py` | `table2.csv` | seconds |
| | `cd experiment2_contact_center` | | |
| Figures 6, 7 and 8 | `python analyze.py` | `figure6_*`, `figure7_*`, `figure8_*`, `summary.csv` | 2 min |
| LH vs. random designs (response letter) | `python analyze_lh_vs_random.py` | `lh_vs_random_summary.csv` | 1 min |

The README of each folder describes the method, the data files, and how to rerun the simulations.

## Seeds

All steps are seeded except the ones below, which were run without a seed. Their saved outputs are included and are used
by the scripts, and the code that generates them is included with a seed for new runs.

| Step | Saved output |
|---|---|
| Runs of Table 1 and of Figures 2 and 3 | `experiment1_synthetic/data/` |
| Contacts of the real contact center (1,000 days) | `experiment2_contact_center/inputs/true_contacts.csv.gz` |
| Contacts of the point-estimate submodels (100 days) | `experiment2_contact_center/inputs/point_estimate_contacts.csv.gz` |
| Observed system states and recorded routing decisions | `experiment2_contact_center/inputs/` |
| Bootstrap instances of the patience-and-handle-time submodels | `experiment2_contact_center/inputs/copula_instances.pkl` |
| Instances of the frequentist routing submodels and of the expert-side Bayesian routing submodel | `experiment2_contact_center/models/` |

All data are simulated; no real customer data are used.

## License

No license has been chosen yet.
