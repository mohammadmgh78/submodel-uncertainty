"""
Simulation runs of the contact-center experiment. The outputs are provided in data/; this script regenerates them.

    python run_simulations.py truth       true KPI of every state: 1,000 replications of the real system
    python run_simulations.py baseline    no submodel uncertainty: point-estimate submodels, 100 replications per state
    python run_simulations.py design      stacked LH design (S = 16 stacks) x n = 69 replications; the pilot, the search
                                          and the final designs of the paper are subsets of this run
    python run_simulations.py comparison  LH vs. random designs (reported in the response letter): 10 designs of each
                                          type, 16 configurations per state, n = 10, and the reference design (S = 32,
                                          128 configurations per state, n = 40)

Options: --jobs N (parallel workers), --out DIR (default data_rerun/, so that data/ is not overwritten).
Every run is seeded per design point; the design runs can be interrupted and resumed. With 10 workers on a laptop, the
design takes about 40-60 minutes per setting and the comparison about three hours.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common as C
from simulator import run_replications, kpi

TRUTH_REPS, CHUNK = 1000, 100
BASELINE_REPS = 100


def truth_chunk(state, reps):
    seed = (C.SEED * 1000 + state * 100 + reps[0] // CHUNK) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    contacts = C.contacts_by_rep('true_contacts.csv.gz', 'Day')
    return [(state, r, w) for r, w in C.kpi_by_rep({j: contacts[j] for j in reps}, state, models=None)]


def truth(jobs, out):
    tasks = [(s, list(range(r, r + CHUNK))) for s in range(C.NUM_STATES) for r in range(1, TRUTH_REPS + 1, CHUNK)]
    res = Parallel(n_jobs=jobs, backend='loky', batch_size=1)(delayed(truth_chunk)(*t) for t in tasks)
    pd.DataFrame([x for r in res for x in r], columns=['state', 'rep', 'wait']).to_csv(
        os.path.join(out, 'truth.csv'), index=False)


def baseline_state(state, bayesian):
    import zlib
    s = zlib.crc32(f"baseline{C.SEED}|{'bayes' if bayesian else 'freq'}|{[state]}".encode())
    random.seed(s)
    np.random.seed(s)
    contacts = C.contacts_by_rep('point_estimate_contacts.csv.gz', 'Rep')
    models = C.routing_models(bayesian, 'point', 'point')
    return [(state, r, w) for r, w in C.kpi_by_rep({j: contacts[j] for j in range(1, BASELINE_REPS + 1)}, state, models)]


def baseline(jobs, out):
    for name, bayesian in C.SETTINGS.items():
        res = Parallel(n_jobs=jobs, backend='loky', batch_size=1)(
            delayed(baseline_state)(s, bayesian) for s in range(C.NUM_STATES))
        pd.DataFrame([x for r in res for x in r], columns=['state', 'rep', 'wait']).to_csv(
            os.path.join(out, f'baseline_{name}.csv'), index=False)


def design_tasks(bayesian, S=16, n=69):
    rows = [r for r in C.design_lh(32, C.SEED + 1) if r[7] <= S]
    return C.design_tasks(rows, f'large{n}', bayesian, n)


def design(jobs, out):
    for name, bayesian in C.SETTINGS.items():
        C.run_tasks(design_tasks(bayesian), os.path.join(out, f'design_{name}.csv'), jobs, C.DESIGN_COLS)


def comparison_tasks(bayesian, n=10, n_designs=10):
    tasks = []
    for r in range(n_designs):
        for dname, d in ((f'lh{r}', C.design_lh(4, C.SEED + 100 + r)), (f'random{r}', C.design_random(16, C.SEED + 200 + r))):
            tasks += [dict(t, design=dname) for t in C.design_tasks(d, dname, bayesian, n)]
    return tasks


def comparison(jobs, out):
    for name, bayesian in C.SETTINGS.items():
        C.run_tasks(C.design_tasks(C.design_lh(32, C.SEED + 1), 'large', bayesian, 40),
                    os.path.join(out, f'reference_{name}.csv'), jobs, C.DESIGN_COLS)
        C.run_tasks(comparison_tasks(bayesian), os.path.join(out, f'lh_vs_random_{name}.csv'), jobs,
                    ('design', *C.DESIGN_COLS))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', choices=['truth', 'baseline', 'design', 'comparison'])
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--out', default=C.path('data_rerun'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {'truth': truth, 'baseline': baseline, 'design': design, 'comparison': comparison}[a.run](a.jobs, a.out)
