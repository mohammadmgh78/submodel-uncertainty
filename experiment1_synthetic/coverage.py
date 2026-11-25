"""
Table 1 and Figure 1: coverage and width of 90% intervals with no epistemic uncertainty, input uncertainty only,
and submodel uncertainty.

    python coverage.py               analyze the saved outputs used in the paper (data/coverage_*.json)
    python coverage.py --simulate    rerun the experiment with a seed and analyze the new outputs

The outputs used in the paper were generated without a seed, so they are provided in data/. A new run gives
statistically equivalent, not identical, results (saved as data/rerun_coverage_*.json).

Each file holds 100 macro-replications x B = 100 estimates of E[Y] (n = 100 replications each).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import synthetic_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = {'no_epistemic': 'No epistemic uncertainty',
         'input_uncertainty': 'Input uncertainty only',
         'submodel_uncertainty': 'Submodel uncertainty'}
MACRO_REPS, TRAIN_SIZE, B, N = 100, 50, 100, 100
SEED = 1


def simulate():
    rng = np.random.default_rng(SEED)
    out = {c: [] for c in CASES}
    for _ in range(MACRO_REPS):
        data = M.training_data(TRAIN_SIZE, rng)
        point = M.fit_submodels(data)
        out['no_epistemic'].append([M.estimate_kpi(point, N, rng) for _ in range(B)])
        out['input_uncertainty'].append([M.estimate_kpi(M.fit_submodels(data, rng, responses=False), N, rng)
                                         for _ in range(B)])
        out['submodel_uncertainty'].append([M.estimate_kpi(M.fit_submodels(data, rng), N, rng) for _ in range(B)])
    for c, v in out.items():
        with open(os.path.join(HERE, 'data', f'rerun_coverage_{c}.json'), 'w') as f:
            json.dump(v, f)


def analyze(prefix):
    rows, ci = [], {}
    for c, label in CASES.items():
        with open(os.path.join(HERE, 'data', f'{prefix}coverage_{c}.json')) as f:
            kpi = np.array(json.load(f))
        ci[c] = np.percentile(kpi, [5, 95], axis=1).T
        width = ci[c][:, 1] - ci[c][:, 0]
        m = len(width)
        half = stats.t.ppf(0.95, m - 1) * width.std(ddof=1) / np.sqrt(m)      # 90% half-width of the mean width
        cover = 100 * np.mean((ci[c][:, 0] <= M.TRUE_KPI) & (M.TRUE_KPI <= ci[c][:, 1]))
        rows.append({'Scenario': label, 'Coverage (%)': cover, 'Avg. CI width': width.mean(), '90% half-width': half})
    table = pd.DataFrame(rows).set_index('Scenario')
    table.to_csv(os.path.join(HERE, 'results', f'{prefix}table1.csv'))
    print(table.round(2).to_string())

    gap, group = 3, 7
    base = np.arange(MACRO_REPS) * (2 * gap + group)
    plt.figure(figsize=(16, 5))
    for k, (c, color, label) in enumerate(zip(CASES, ('tab:blue', 'tab:orange', 'tab:green'),
                                              ('No epistemic uncertainty', 'IU only', 'SU'))):
        plt.vlines(base + k * gap, ci[c][:, 0], ci[c][:, 1], color=color, linewidth=2, label=label, alpha=0.8)
    plt.axhline(M.TRUE_KPI, color='red', linestyle='--', linewidth=1.5, label='True KPI')
    plt.xticks([])
    plt.yticks(fontsize=14)
    plt.ylabel('KPI', fontsize=18)
    plt.legend(fontsize=14, frameon=False)
    plt.gca().spines[['right', 'top', 'bottom']].set_visible(False)
    plt.savefig(os.path.join(HERE, 'results', f'{prefix}figure1.png'))
    plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--simulate', action='store_true')
    a = ap.parse_args()
    if a.simulate:
        simulate()
    analyze('rerun_' if a.simulate else '')
