"""
Table 2: variance reduction factor (VRF) of bagging.

    python bagging.py             recompute the table from the saved runs (results/table2_runs.npz)
    python bagging.py --simulate  rerun all simulations (20 datasets x 20 repetitions, n = 345)

For each of 20 training datasets (submodel instances held fixed), the simulation is repeated 20 times with n = 345.
Each repetition gives the single-tree and the bagged (50 trees) shares. VRF = (variance of the single-tree share across
repetitions, averaged over the datasets) / (the same for the bagged share).
"""
import argparse
import os

import numpy as np
import pandas as pd

import importance as E

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'results', 'table2_runs.npz')
N_DATA, N_RESIM, N_REPS = 20, 20, 345


def one_dataset(d):
    np.random.seed(1000 + d)
    inst = E.train_instances()
    exact = E.exact_contributions(inst, N_REPS)
    single, bagged = [], []
    for _ in range(N_RESIM):
        out = E.simulate(inst, N_REPS)
        su, mc, w = E.variance_components(E.design, out)
        wb = E.bagged_importance(out)
        single.append(np.append(w * su, mc) / (su + mc))
        bagged.append(np.append(wb * su, mc) / (su + mc))
    return np.array(single), np.array(bagged), exact / exact.sum()


def simulate(jobs):
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=jobs, backend='loky')(delayed(one_dataset)(d) for d in range(N_DATA))
    np.savez(RUNS, single=np.array([r[0] for r in res]), bagged=np.array([r[1] for r in res]),
             exact=np.array([r[2] for r in res]))


def analyze():
    z = np.load(RUNS)
    single, bagged, exact = z['single'], z['bagged'], z['exact']      # (dataset, repetition, share), (dataset, share)

    def rmse(shares):
        return np.sqrt(((shares - exact[:, None, :]) ** 2).mean(axis=(0, 1)))

    var_s, var_b = (a.var(axis=1, ddof=1).mean(axis=0) for a in (single, bagged))
    tab = pd.DataFrame({'VRF': var_s / var_b, 'RMSE single tree': rmse(single), 'RMSE bagged': rmse(bagged),
                        'Mean share single': single.mean(axis=(0, 1)), 'Mean share bagged': bagged.mean(axis=(0, 1)),
                        'Mean exact share': exact.mean(axis=0)}, index=E.FEATURES + ['Error'])
    tab.to_csv(os.path.join(HERE, 'results', 'table2.csv'))
    print(tab.round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--simulate', action='store_true')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    if a.simulate:
        simulate(a.jobs)
    analyze()
