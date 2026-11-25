"""
Figures 2 and 3 and the ANOVA of Section 5.2: all 2^4 combinations of true and estimated submodels.

    python factorial.py               analyze the saved outputs used in the paper (data/factorial_outputs.pkl)
    python factorial.py --simulate    rerun the experiment with a seed and analyze the new outputs

The outputs used in the paper were generated without a seed, so they are provided in data/. A new run gives
statistically equivalent, not identical, results (saved as data/rerun_factorial_outputs.pkl).

factorial_outputs.pkl: {(X1, X2, p, q): list of 500 estimates of E[Y]}, where 1 = true submodel and 0 = estimated
submodel. Each estimate uses n = 500 replications and one of 500 training datasets of size 50. As in the runs of the
paper, the estimated submodels are fitted to a bootstrap resample of the training dataset.
"""
import argparse
import itertools
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, t
from statsmodels.formula.api import ols

import synthetic_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
MACRO_REPS, TRAIN_SIZE, N = 500, 50, 500
SEED = 1
ALPHA = 0.05


def simulate():
    rng = np.random.default_rng(SEED)
    train = [M.training_data(TRAIN_SIZE, rng) for _ in range(MACRO_REPS)]
    true = ((M.MU_1, M.SIGMA_1), (M.MU_2, M.SIGMA_2), M.p_true, M.q_true)
    out = {}
    for key in itertools.product((0, 1), repeat=4):
        out[key] = []
        for data in train:
            est = M.fit_submodels(data, rng)
            out[key].append(M.estimate_kpi([tr if k else es for k, tr, es in zip(key, true, est)], N, rng))
    with open(os.path.join(HERE, 'data', 'rerun_factorial_outputs.pkl'), 'wb') as f:
        pickle.dump(out, f)


def anova(out):
    """Main-effects ANOVA (type II) of the KPI estimates and of their variance (one value per combination)."""
    cols = ['X1', 'X2', 'p', 'q']
    df = pd.DataFrame([(*k, y) for k, v in out.items() for y in v], columns=cols + ['kpi'])
    df_var = pd.DataFrame([(*k, np.var(v, ddof=1)) for k, v in out.items()], columns=cols + ['variance'])
    mean = sm.stats.anova_lm(ols('kpi ~ X1 + X2 + p + q', df).fit(), typ=2)
    var = sm.stats.anova_lm(ols('variance ~ X1 + X2 + p + q', df_var).fit(), typ=2)
    return pd.DataFrame({'p-value (KPI)': mean['PR(>F)'], 'p-value (variance)': var['PR(>F)']}).dropna()


def response_grid_plot(values, filename, color='#7A0019'):
    inner, gap, rscale = 0.7, 2.0, 0.2
    vmax = max(abs(v) for v in values.values())
    fig, ax = plt.subplots(figsize=(7, 7))
    for xs, ys in (([inner, gap], [inner / 2] * 2), ([inner, gap], [gap + inner / 2] * 2),
                   ([inner / 2] * 2, [inner, gap]), ([gap + inner / 2] * 2, [inner, gap])):
        ax.plot(xs, ys, lw=2, color=color)
    for (x1, x2, p, q), v in values.items():
        ox, oy = x1 * gap, x2 * gap
        ax.plot([ox, ox + inner, ox + inner, ox, ox], [oy, oy, oy + inner, oy + inner, oy], lw=2, color=color)
        ax.add_patch(plt.Circle((ox + p * inner, oy + q * inner), rscale * np.sqrt(abs(v) / vmax),
                                color='tab:blue' if v > 0 else 'tab:red', alpha=0.8))
    ax.text(gap / 2 + inner / 2 - 0.1, -0.6, r'$X_1$', ha='center', fontsize=12)
    ax.text(-0.6, gap / 2 + inner / 2 - 0.1, r'$X_2$', va='center', rotation=90, fontsize=12)
    ax.text(gap + inner / 2, -0.2, r'$p$', ha='center', fontsize=12)
    ax.text(gap + inner + 0.2, inner / 2, r'$q$', va='center', rotation=90, fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-0.8, gap + inner + 0.8)
    ax.set_ylim(-0.8, gap + inner + 0.8)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def analyze(prefix):
    with open(os.path.join(HERE, 'data', f'{prefix}factorial_outputs.pkl'), 'rb') as f:
        out = pickle.load(f)
    rows = {}
    for key, v in out.items():
        v = np.array(v)
        m = len(v)
        s2, bias = v.var(ddof=1), v.mean() - M.TRUE_KPI
        h = t.ppf(1 - ALPHA / 2, m - 1) * np.sqrt(s2 / m)
        rows[key] = dict(variance=s2, variance_low=(m - 1) * s2 / chi2.ppf(1 - ALPHA / 2, m - 1),
                         variance_high=(m - 1) * s2 / chi2.ppf(ALPHA / 2, m - 1),
                         bias=bias, bias_low=bias - h, bias_high=bias + h, mse=np.mean((v - M.TRUE_KPI) ** 2))
    tab = pd.DataFrame.from_dict(rows, orient='index')
    tab.index = pd.MultiIndex.from_tuples(rows, names=['X1', 'X2', 'p', 'q'])
    tab.to_csv(os.path.join(HERE, 'results', f'{prefix}bias_variance.csv'))
    pvals = anova(out)
    pvals.to_csv(os.path.join(HERE, 'results', f'{prefix}anova.csv'))
    print(tab[['variance', 'bias', 'mse']].round(2).to_string())
    print(pvals.round(4).to_string())

    keys = sorted(rows, key=lambda k: (sum(k), k))            # from all estimated to all true
    x = np.arange(len(keys))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, m in zip(axes, ('variance', 'bias')):
        v, lo, hi = (np.array([rows[k][c] for k in keys]) for c in (m, f'{m}_low', f'{m}_high'))
        ax.bar(x, v, yerr=np.vstack([v - lo, hi - v]), capsize=5, error_kw={'linewidth': 2})
        ax.set_ylabel(m.capitalize(), fontsize=14)
        ax.tick_params(axis='both', labelsize=13)
        ax.spines[['right', 'top']].set_visible(False)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(k) for k in keys], rotation=45, ha='right', fontsize=12)
    axes[1].set_xlabel('Model Combination (True / Estimated)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'results', f'{prefix}figure2.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    for m, panel in (('variance', 'a'), ('bias', 'b')):
        response_grid_plot({k: r[m] for k, r in rows.items()},
                           os.path.join(HERE, 'results', f'{prefix}figure3{panel}.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--simulate', action='store_true')
    a = ap.parse_args()
    if a.simulate:
        simulate()
    analyze('rerun_' if a.simulate else '')
