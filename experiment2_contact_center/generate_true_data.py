"""
Data of the real system.

    python generate_true_data.py training_day    simulate day 1 with the true routing rules; checks that the result
                                                 matches inputs/training_day.csv
    python generate_true_data.py contacts        generate contacts of 1,000 days from the true input processes
    python generate_true_data.py routing         simulate 10 days and record the routing decisions

The contacts (inputs/true_contacts.csv.gz) and the routing decisions (inputs/routing_decisions_*.csv) used in the paper
were generated without a seed, so they are provided. `contacts` and `routing` write new, seeded data to data_rerun/.
"""
import os
import sys

import numpy as np
import pandas as pd

import common as C
import simulator

# true input processes: mean interarrival time in each hour, Gamma patience and handle times, copula correlation
PARAMS = {1: dict(shape_p=5, scale_p=0.9, shape_s=4, scale_s=1, rho=0.7,
                  interarrival=[2, 1.5, 1, 1.8, 1.7, 1.9, 1.2, 2.8, 2.5]),
          2: dict(shape_p=2, scale_p=5, shape_s=4, scale_s=1.5, rho=0.7,
                  interarrival=[1, 0.5, 0.6, 0.8, 0.7, 0.9, 1.2, 1.8, 1.5])}


def simulate_days(days):
    contacts = C.contacts_by_rep('true_contacts.csv.gz', 'Day')
    return simulator.run_replications({d: contacts[d] for d in days}, {d: [[], []] for d in days}, 0)


def training_day():
    new = simulator.customers_frame(simulate_days([1])[1], 1)
    old = pd.read_csv(C.path('inputs', 'training_day.csv'), index_col=0)
    cols = ['Contact Group', 'Expert Group', 'Arrival Time', 'Handle Time', 'Patience Time', 'Abandon', 'Waiting Time']
    print('training day reproduced:', len(new) == len(old) and bool(np.allclose(new[cols], old[cols], atol=1e-9, rtol=0)))


def contacts():
    rng = np.random.default_rng(C.SEED)
    parts = []
    for day in range(1, 1001):
        for g, p in PARAMS.items():
            A = C.generate_arrivals(rng, p['interarrival'])
            P, H = C.sample_copula(rng, len(A), [p[k] for k in ('shape_p', 'scale_p', 'shape_s', 'scale_s', 'rho')])
            parts.append(pd.DataFrame({'Day': day, 'Contact Group': g, 'Arrival Time': A, 'Patience Time': P,
                                       'Handle Time': H}).round(6))
    pd.concat(parts).to_csv(C.path('data_rerun', 'true_contacts.csv.gz'), index=False)


def routing():
    sims = simulate_days(range(1, 11))
    for side in ('contact', 'expert'):
        pd.concat([pd.DataFrame(s.routing_obj.routing_dict[f'routing {side}']) for s in sims.values()]).to_csv(
            C.path('data_rerun', f'routing_decisions_{side}.csv'), index=False)


if __name__ == '__main__':
    os.makedirs(C.path('data_rerun'), exist_ok=True)
    {'training_day': training_day, 'contacts': contacts, 'routing': routing}[sys.argv[1]]()
