"""
Stacked Latin hypercube vs. independent random configurations at the same budget (table in the response letter).
Uses the saved simulation outputs in data/; no simulation is run.

    python analyze_lh_vs_random.py

Ten designs of each type, 16 configurations per state and n = 10 replications. Reference: the LH design with S = 32
stacks (128 configurations per state), of which the first 10 replications are used. For each design, the CI Jaccard
distance 1 - |A and A_ref| / |A or A_ref| between its 90% interval and the reference interval is averaged over states.
"""
import numpy as np
import pandas as pd

import common as C

N = 10


def jaccard_distance(iv, ref):
    inter = np.clip(np.minimum(iv['high'], ref['high']) - np.maximum(iv['low'], ref['low']), 0, None)
    union = np.maximum(iv['high'], ref['high']) - np.minimum(iv['low'], ref['low'])
    return 1 - inter / union


def main():
    rows = []
    for tag in C.SETTINGS:
        ref = C.state_intervals(pd.read_csv(C.path('data', f'reference_{tag}.csv')).query('rep <= @N'))
        print(f"{C.TITLE[tag]}: average width of the reference intervals = {ref['width'].mean():.2f}")
        for name, d in pd.read_csv(C.path('data', f'lh_vs_random_{tag}.csv')).groupby('design'):
            iv = C.state_intervals(d)
            rows.append(dict(setting=tag, design=name, type='LH' if name.startswith('lh') else 'Random',
                             avg_width=iv['width'].mean(), jaccard=jaccard_distance(iv, ref).mean()))
    res = pd.DataFrame(rows)
    res.to_csv(C.path('results', 'lh_vs_random_per_design.csv'), index=False)
    summary = res.groupby(['setting', 'type'])[['avg_width', 'jaccard']].agg(['mean', 'std', 'max'])
    summary.to_csv(C.path('results', 'lh_vs_random_summary.csv'))
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
