"""
Section 6.3: pilot and n*, search over B and S (Figure 6), intervals with and without submodel uncertainty (Figure 7),
and importance scores (Figure 8). Uses the saved simulation outputs in data/; no simulation is run.

    python analyze.py

Output: results/ (figures and tables).
"""
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

import common as C

PILOT = (4, 2, 10)                                 # (B, S, n0)
TAU = 0.1
FOREST_SEED = {'freq': 225, 'bayesian': 23135}
OUT = C.path('results')


def pilot_n_star(df, n0):
    """n* from the variance components of the pilot, averaged over the states."""
    parts = []
    for s in range(C.NUM_STATES):
        parts.append(C.variance_components(*C.config_matrix(df[df['state'] == s])))
    su, mc = np.mean([p[0] for p in parts]), np.mean([p[1] for p in parts])
    return math.ceil(n0 * mc / su * (1 - TAU) / TAU), mc / (mc + su)


def plot_search(hist, filename):
    h = hist[hist['iteration'] > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h['iteration'], h['actual_change'], marker='o')
    for _, r in h.iterrows():
        ax.annotate(f'({int(r.B)}, {int(r.S)}, {int(r.n)})', (r.iteration, r.actual_change),
                    textcoords='offset points', xytext=(0, 8), ha='center', fontsize=12)
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Change in 90% CI Width', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.1, y=0.15)
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)


def interval_table(data, base, truth, n):
    """Per state: quantile interval with SU, and t-interval without SU from the first n baseline replications."""
    su = C.state_intervals(data)
    b = base[base['rep'] <= n]
    no = np.array([C.t_interval(b[b['state'] == s]['wait'].to_numpy()) for s in range(C.NUM_STATES)])
    d = pd.DataFrame({'state': np.arange(1, C.NUM_STATES + 1), 'true_kpi': truth, 'nosu_low': no[:, 0],
                      'nosu_high': no[:, 1], 'su_low': su['low'].to_numpy(), 'su_high': su['high'].to_numpy()})
    for k in ('nosu', 'su'):
        d[f'{k}_width'] = d[f'{k}_high'] - d[f'{k}_low']
        d[f'{k}_covers'] = (d[f'{k}_low'] <= d['true_kpi']) & (d['true_kpi'] <= d[f'{k}_high'])
    return d


def plot_intervals(d, filename):
    x = np.arange(C.NUM_STATES)
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, tv in enumerate(d['true_kpi']):
        ax.hlines(tv, k - 0.3, k + 0.2, linewidth=2, color='red', alpha=0.6, label='True value' if k == 0 else None)
    for kind, off, lab, col in (('nosu', -0.2, 'No SU', 'orange'), ('su', 0.0, 'SU', 'blue')):
        ax.vlines(x + off, d[f'{kind}_low'], d[f'{kind}_high'], linewidth=2, color=col, alpha=0.9, label=lab)
    ax.set_xlabel('Interval', fontsize=16)
    ax.set_xticks(x, [f'{k + 1}' for k in x], fontsize=14)
    ax.set_ylabel('Mean Waiting Time (min)', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_importance(imp, chosen, filename):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ax, tag in zip(axes, C.SETTINGS):
        s = imp[tag].sort_values()
        ax.barh([C.PRETTY[f] for f in s.index], s.to_numpy(), color=[C.PALETTE[f] for f in s.index])
        for k, v in enumerate(s.to_numpy()):
            ax.text(v + 0.005, k, f'{v:.4f}', va='center', fontsize=12)
        ax.set_title('{}, (B, S, n) = ({}, {}, {})'.format(C.TITLE[tag], *chosen[tag]), fontsize=13)
        ax.tick_params(axis='y', labelsize=12)
        ax.xaxis.set_visible(False)
        ax.spines[['bottom', 'right', 'top']].set_visible(False)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    out = lambda name: os.path.join(OUT, name)
    truth = C.true_kpi()
    rows, chosen, imp = [], {}, {}
    for tag in C.SETTINGS:
        data = pd.read_csv(C.path('data', f'design_{tag}.csv'))
        base = pd.read_csv(C.path('data', f'baseline_{tag}.csv'))
        n_star, noise_share = pilot_n_star(C.select(data, *PILOT), PILOT[2])
        n = max(n_star, PILOT[2])
        assert n <= data['rep'].max(), 'n* exceeds the simulated replications'

        (B, S), hist = C.search_bs(data, PILOT[0], PILOT[1], n, max_S=data['stack'].max())
        chosen[tag] = (B, S, n)
        hist.to_csv(out(f'figure6_search_{tag}.csv'), index=False)
        plot_search(hist, out(f'figure6_search_{tag}.png'))

        final = C.select(data, B, S, n)
        d = interval_table(final, base, truth, n)
        d.to_csv(out(f'figure7_intervals_{tag}.csv'), index=False)
        plot_intervals(d, out(f'figure7_intervals_{tag}.png'))

        imp[tag] = C.importance(final, FOREST_SEED[tag])
        rows.append(dict(setting=tag, pilot_noise_share=noise_share, n_star=n_star, B=B, S=S, n=n,
                         last_change=hist['actual_change'].iloc[-1], width_nosu=d['nosu_width'].mean(),
                         width_su=d['su_width'].mean(), epochs_covered_nosu=int(d['nosu_covers'].sum()),
                         epochs_covered_su=int(d['su_covers'].sum()),
                         epochs_missed_su=' '.join(str(s) for s in d.loc[~d['su_covers'], 'state']),
                         noise_share=imp[tag]['Error']))
    tab = pd.DataFrame(imp).rename(index=C.PRETTY)
    tab.to_csv(out('figure8_importance.csv'))
    plot_importance(imp, chosen, out('figure8_importance.png'))
    summary = pd.DataFrame(rows)
    summary.to_csv(out('summary.csv'), index=False)
    print(summary.set_index('setting').T.to_string())
    print(tab.round(4).to_string())


if __name__ == '__main__':
    main()
