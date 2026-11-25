"""
Figures 4 and 5: importance scores of the synthetic example (Sections 5.1 and 5.2).

    python importance.py              Figure 4: 25 training observations for every submodel
    python importance.py --n-x2 1000  Figure 5: 1,000 training observations for X2

Steps
  1. Fit B = 8 bootstrap instances of each submodel to one training dataset.
  2. Stacked Latin hypercube design with S = 64 stacks (B' = 512 configurations).
  3. Pilot with n0 = 10 replications per configuration -> n* for the tolerance tau = 0.1.
  4. Final simulation with n = max(n0, n*) and attribution with bagged regression trees.

Attribution (a fully grown tree; N = number of outputs, L = number of leaves, i.e., of distinct configurations)
  sigma2_MC = RSS / (n (N - L))
  sigma2_SU = max(0, (TSS - RSS) / (n (L - 1)) - sigma2_MC)
  I_l       = mean TSS reduction of the splits on submodel l; the I_l are normalized and averaged over bagged trees
  shares    = (sigma2_SU I_l / sum_k I_k, sigma2_MC) / (sigma2_SU + sigma2_MC)

The exact shares use the Shapley effects over all B^4 configurations; the fitted submodels of p and q are polynomials
and those of X1 and X2 are normal, so Gauss-Hermite quadrature is exact.
"""
import argparse
import math
import os
from itertools import combinations
from math import factorial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from synthetic_model import DEG_P, DEG_Q, MU_1, MU_2, SIGMA_1, SIGMA_2, p_true, q_true

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1
N_TRAIN = 25
B, S = 8, 64
N_PILOT = 10
TAU = 0.1
N_TREES = 50
FEATURES = ['X_1', 'X_2', 'p', 'q']
COLUMNS = ['Input 1', 'Input 2', 'Surrogate 1', 'Surrogate 2']


def boot_idx(n):
    return np.random.choice(n, size=n, replace=True)


def train_instances(n_x2=N_TRAIN):
    """B bootstrap instances of each submodel; `n_x2` observations for X2 and N_TRAIN for the others."""
    x1 = np.random.normal(MU_1, SIGMA_1, N_TRAIN)
    x2 = np.random.normal(MU_2, SIGMA_2, max(n_x2, N_TRAIN))
    x2_train = x2[:n_x2]
    p_in, p_out = x1, p_true(x1)
    q_in, q_out = x2[:N_TRAIN], q_true(x2[:N_TRAIN])
    x1_inst, x2_inst, p_inst, q_inst = [], [], [], []
    for _ in range(B):
        x1_b = x1[boot_idx(N_TRAIN)]
        x2_b = x2_train[boot_idx(n_x2)]
        x1_inst.append((x1_b.mean(), x1_b.std()))
        x2_inst.append((x2_b.mean(), x2_b.std()))
        ip, iq = boot_idx(N_TRAIN), boot_idx(N_TRAIN)
        p_inst.append(np.poly1d(np.polyfit(p_in[ip], p_out[ip], DEG_P)))
        q_inst.append(np.poly1d(np.polyfit(q_in[iq], q_out[iq], DEG_Q)))
    return x1_inst, x2_inst, p_inst, q_inst


def stacked_lh_design(k, stacks, levels, seed):
    rng = np.random.default_rng(seed)
    return np.vstack([np.column_stack([rng.permutation(levels) for _ in range(k)]) for _ in range(stacks)])


design = stacked_lh_design(4, S, B, seed=42)


def simulate(inst, n):
    """Outputs of n replications of every configuration (B' x n)."""
    x1_inst, x2_inst, p_inst, q_inst = inst
    out = []
    for c in design:
        mu1, s1 = x1_inst[c[0]]
        mu2, s2 = x2_inst[c[1]]
        out.append(p_inst[c[2]](np.random.normal(mu1, s1, n)) + q_inst[c[3]](np.random.normal(mu2, s2, n)))
    return np.array(out)


def tree_quantities(configs, outputs):
    """TSS, RSS, number of leaves, and the mean TSS reduction of the splits on each submodel (I_l, normalized)."""
    X = pd.get_dummies(pd.DataFrame(np.repeat(configs, outputs.shape[1], axis=0), columns=COLUMNS).astype(str))
    y = outputs.ravel()
    tree = DecisionTreeRegressor(random_state=42).fit(X, y)
    owner = [next(j for j, s in enumerate(COLUMNS) if col.startswith(s)) for col in X.columns]
    t = tree.tree_
    total, count = np.zeros(4), np.zeros(4)
    for i in range(t.node_count):
        if t.feature[i] < 0:
            continue
        left, right = t.children_left[i], t.children_right[i]
        k = owner[t.feature[i]]
        total[k] += (t.weighted_n_node_samples[i] * t.impurity[i]
                     - t.weighted_n_node_samples[left] * t.impurity[left]
                     - t.weighted_n_node_samples[right] * t.impurity[right])
        count[k] += 1
    importance = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    tss = ((y - y.mean()) ** 2).sum()
    rss = ((y - tree.predict(X)) ** 2).sum()
    return tss, rss, tree.get_n_leaves(), importance / importance.sum()


def variance_components(configs, outputs):
    """sigma2_SU, sigma2_MC and the normalized I_l of a single tree."""
    n = outputs.shape[1]
    tss, rss, leaves, w = tree_quantities(configs, outputs)
    mc = rss / (outputs.size - leaves) / n
    su = max(0.0, (tss - rss) / (n * (leaves - 1)) - mc)
    return su, mc, w


def bagged_importance(outputs, n_trees=N_TREES):
    """Normalized I_l averaged over trees; configurations and then replications are resampled with replacement."""
    n = outputs.shape[1]
    w = []
    for _ in range(n_trees):
        idx = np.random.choice(len(design), size=len(design), replace=True)
        boot = outputs[idx[:, None], np.random.randint(n, size=(len(idx), n))]
        w.append(tree_quantities(design[idx], boot)[3])
    return np.mean(w, axis=0)


GH_NODES, GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(40)
GH_WEIGHTS = GH_WEIGHTS / GH_WEIGHTS.sum()


def normal_mean_var(f, m, s):
    v = f(m + s * GH_NODES)
    mean = GH_WEIGHTS @ v
    return mean, GH_WEIGHTS @ (v - mean) ** 2


def exact_contributions(inst, n):
    """Shapley effects of X1, X2, p, q on Var(E[Y | configuration]) over all B^4 configurations, and sigma^2 / n."""
    x1_inst, x2_inst, p_inst, q_inst = inst
    A, VA = np.array([[normal_mean_var(p_inst[k], *x1_inst[i]) for k in range(B)] for i in range(B)]).transpose(2, 0, 1)
    C, VC = np.array([[normal_mean_var(q_inst[k], *x2_inst[i]) for k in range(B)] for i in range(B)]).transpose(2, 0, 1)
    mu_grid = A[:, None, :, None] + C[None, :, None, :]

    def closed_var(T):
        others = tuple(a for a in range(4) if a not in T)
        return np.var(mu_grid.mean(axis=others)) if T else 0.0

    shapley = np.array([sum(factorial(len(T)) * factorial(3 - len(T)) / factorial(4)
                            * (closed_var(T + (l,)) - closed_var(T))
                            for r in range(4) for T in combinations([a for a in range(4) if a != l], r))
                        for l in range(4)])
    sigma2 = (VA[design[:, 0], design[:, 2]] + VC[design[:, 1], design[:, 3]]).mean()
    return np.append(shapley, sigma2 / n)


def plot_shares(values, filename):
    features = np.array(FEATURES + ['Error'])
    colors = np.array(['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'])
    order = np.argsort(values)
    plt.figure(figsize=(8, 4))
    plt.barh(features[order], values[order], color=colors[order])
    for i, v in enumerate(values[order]):
        plt.text(v + 0.005, i, f'{v:.4f}', va='center')
    ax = plt.gca()
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.xaxis.set_visible(False)
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main(n_x2):
    figure = 'figure4' if n_x2 == N_TRAIN else 'figure5'
    out_path = lambda name: os.path.join(HERE, 'results', name)
    np.random.seed(SEED)
    inst = train_instances(n_x2)

    pilot = simulate(inst, N_PILOT)
    su0, mc0, _ = variance_components(design, pilot)
    n_star = math.ceil(N_PILOT * mc0 / su0 * (1 - TAU) / TAU)
    n = max(N_PILOT, n_star)

    outputs = simulate(inst, n)
    su, mc, w_single = variance_components(design, outputs)
    w_bagged = bagged_importance(outputs)
    exact = exact_contributions(inst, n)
    shares = pd.DataFrame({'Bagged': np.append(w_bagged * su, mc) / (su + mc),
                           'Single tree': np.append(w_single * su, mc) / (su + mc),
                           'Exact': exact / exact.sum()}, index=FEATURES + ['Error'])
    summary = pd.DataFrame([dict(n_x2=n_x2, tau=TAU, n0=N_PILOT, pilot_su=su0, pilot_mc=mc0,
                                 pilot_noise_share=mc0 / (mc0 + su0), n_star=n_star, n=n, su=su, mc=mc,
                                 noise_share=mc / (mc + su), exact_su=exact[:4].sum(), exact_mc=exact[4])])
    shares.to_csv(out_path(f'{figure}_shares.csv'))
    summary.to_csv(out_path(f'{figure}_summary.csv'), index=False)
    plot_shares(shares['Bagged'].to_numpy(), out_path(f'{figure}.png'))
    print(summary.round(3).T.to_string(header=False))
    print(shares.round(4).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-x2', type=int, default=N_TRAIN, help='training observations for X2')
    main(ap.parse_args().n_x2)
