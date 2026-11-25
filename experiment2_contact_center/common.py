"""
Shared code of the contact-center experiment: designs, simulation inputs, initial states, the simulation of one design
point, and the analysis functions (intervals, search over B and S, importance scores).

Settings: 18 observed states (one per half hour), six submodels with B = 8 instances each (index 0..7):
A1, PH1, A2, PH2 = arrival and patience-and-handle-time submodels of contact groups 1 and 2;
RC, RE = contact-side and expert-side routing submodels. 'freq' = frequentist, 'bayesian' = Bayesian submodels.
"""
import functools
import io
import os
import random
import time
import zlib

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from scipy.stats import t as student_t

import simulator

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20251013
B_MAX = 8                # instances of each submodel
NUM_STATES = 18
STACK_SIZE = 72          # design points per stack (18 states x 4 configurations), see design_lh
ALPHA = 0.10             # 90% intervals
TOL = 0.05               # tolerance of the search over B and S (minutes)
N_TREES = 40             # bagged trees per state
T_MAX, HOUR = 540, 60    # minutes

CFG = ['A1', 'PH1', 'A2', 'PH2', 'RC', 'RE']
PRETTY = {'A1': 'Contact Group 1 Arrival', 'PH1': 'Contact Group 1 Patience and Handle Time',
          'A2': 'Contact Group 2 Arrival', 'PH2': 'Contact Group 2 Patience and Handle Time',
          'RC': 'Contact-Side Routing Model', 'RE': 'Expert-Side Routing Model', 'Error': 'Error'}
PALETTE = {'PH2': '#5ec962', 'A2': '#21918c', 'PH1': '#3b528b', 'RC': '#fde725', 'RE': '#d9ef8b',
           'A1': '#440154', 'Error': '#fee08b'}
SETTINGS = {'freq': False, 'bayesian': True}
TITLE = {'freq': 'Frequentist', 'bayesian': 'Bayesian'}
DESIGN_COLS = (*CFG, 'state', 'stack', 'dp')


def path(*p):
    return os.path.join(HERE, *p)


# ----------------------------------------------------------------------------------------------------------------------
# Designs
# ----------------------------------------------------------------------------------------------------------------------
def design_lh(S, seed):
    """Stacked Latin hypercube design with S stacks per state. Two-level LH designs over the six submodels give base
    rows; each row is assigned to a state (cycling through the 18 states without replacement) and expanded to the
    instances x, x+2, x+4, x+6. Every block of 72 design points gives each state one base row, i.e., one stack of four
    configurations. The first S' stacks do not depend on S. Rows [A1, PH1, A2, PH2, RC, RE, state, stack]."""
    rng = np.random.default_rng(seed)
    rows, states_left = [], []
    for _ in range(9 * S):                               # 9 LH designs x 2 rows = 18 base rows per stack
        lhs = np.column_stack([rng.permutation(2) for _ in CFG])
        for x in lhs:
            if not states_left:
                states_left = list(range(NUM_STATES))
            st = states_left.pop(int(rng.integers(len(states_left))))
            for shift in (0, 2, 4, 6):
                rows.append([int(v) for v in x + shift] + [st])
    return [r + [k // STACK_SIZE + 1] for k, r in enumerate(rows)]


def design_random(n_cfg_per_state, seed):
    """Independent random configurations: every instance drawn uniformly from 0..7 (stack = 1)."""
    rng = np.random.default_rng(seed)
    return [[int(v) for v in rng.integers(B_MAX, size=len(CFG))] + [st, 1]
            for st in range(NUM_STATES) for _ in range(n_cfg_per_state)]


# ----------------------------------------------------------------------------------------------------------------------
# Simulation inputs
# ----------------------------------------------------------------------------------------------------------------------
def generate_arrivals(rng, scale_list, T_max=T_MAX, interval=HOUR):
    """Arrival times with exponential interarrival times whose mean depends on the hour."""
    arrivals, t = [], 0.0
    while t < T_max:
        t += rng.exponential(scale=scale_list[int(t // interval)])
        if t <= T_max:
            arrivals.append(t)
    return np.array(arrivals)


def sample_copula(rng, n, params):
    """n (patience, handle) pairs: Gamma marginals joined by a Gaussian copula."""
    shape_p, scale_p, shape_s, scale_s, rho = params
    z = rng.standard_normal((n, 2))
    u1 = np.clip(norm.cdf(z[:, 0]), 1e-12, 1 - 1e-12)
    u2 = np.clip(norm.cdf(rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]), 1e-12, 1 - 1e-12)
    return gamma.ppf(u1, a=shape_p, scale=scale_p), gamma.ppf(u2, a=shape_s, scale=scale_s)


@functools.lru_cache(None)
def input_instances():
    return (pd.read_pickle(path('inputs', 'arrival_instances.pkl')), pd.read_pickle(path('inputs', 'copula_instances.pkl')))


def make_contacts(cfg, rep, key):
    """One day of contacts for the configuration cfg; `key` identifies the random stream (design point)."""
    arr, cop = input_instances()
    parts = []
    for g in (1, 2):
        a, c = cfg[2 * g - 2], cfg[2 * g - 1]
        rng = np.random.default_rng([SEED, zlib.crc32(key.encode()), g, rep])
        A = generate_arrivals(rng, arr['boot'][g][a])
        P, H = sample_copula(rng, len(A), cop['boot'][g][c])
        parts.append(pd.DataFrame({'Contact Group': g, 'Arrival Time': A, 'Patience Time': P, 'Handle Time': H}))
    return pd.concat(parts, ignore_index=True)


@functools.lru_cache(None)
def contacts_by_rep(filename, rep_col):
    """Saved contacts ({rep: DataFrame})."""
    d = pd.read_csv(path('inputs', filename), float_precision='round_trip')
    cols = ['Contact Group', 'Arrival Time', 'Patience Time', 'Handle Time']
    return {int(r): g[cols].reset_index(drop=True) for r, g in d.groupby(rep_col)}


# ----------------------------------------------------------------------------------------------------------------------
# Initial states
# ----------------------------------------------------------------------------------------------------------------------
def sample_conditional(rng, params, lower, given):
    """(patience, handle) pairs from the fitted copula given patience > lower (given='P') or handle > lower ('S')."""
    shape_p, scale_p, shape_s, scale_s, rho = params
    lower = np.asarray(lower, dtype=float)
    a1, s1, a2, s2 = (shape_p, scale_p, shape_s, scale_s) if given == 'P' else (shape_s, scale_s, shape_p, scale_p)
    v = rng.uniform(0.0, 1.0, size=lower.shape) * gamma.sf(np.maximum(lower, 0.0), a=a1, scale=s1)
    v = np.clip(v, 1e-300, 1.0)
    x1, z1 = gamma.isf(v, a=a1, scale=s1), norm.isf(v)
    z2 = rho * z1 + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(size=lower.shape)
    x2 = gamma.isf(norm.sf(z2), a=a2, scale=s2)
    return (x1, x2) if given == 'P' else (x2, x1)


@functools.lru_cache(None)
def observed_state(state):
    """Customers waiting and in service at the observed state (after state x 30 minutes of the training day)."""
    return tuple(pd.read_csv(path('inputs', 'observed_states', f'{n}_{state * 0.5}_Day_1.csv')) for n in ('Queue', 'Service'))


def initial_state(state, rep):
    """[waiting, in service] customers of one replication. Their unknown patience and handle times are sampled given
    the time already waited or served. State 0 is the empty system at the start of the day."""
    if state == 0:
        return [[], []]
    params = pd.read_pickle(path('inputs', 'copula_point_estimate.pkl'))
    rng = np.random.default_rng([SEED, state - 1, rep])
    out = []
    for d, known, given in zip(observed_state(state), ('Waiting Time', 'Time Elapsed'), ('P', 'S')):
        d = d.copy()
        d['Patience Time'] = np.nan
        d['Handle Time'] = np.nan
        for g in (1, 2):
            m = (d['Contact Group'] == g).to_numpy()
            if m.any():
                d.loc[m, 'Patience Time'], d.loc[m, 'Handle Time'] = sample_conditional(
                    rng, params[g], d.loc[m, known].to_numpy(), given)
        out.append(pd.read_csv(io.StringIO(d.to_csv(index=False))))      # same rounding as when stored as CSV files
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------------------------------------------------
@functools.lru_cache(None)
def routing_models(bayesian, rc, re):
    p = 'bayes' if bayesian else 'freq'
    return simulator.load_model(f'{p}_contact_{rc}'), simulator.load_model(f'{p}_expert_{re}')


def kpi_by_rep(contacts, state, models):
    """[(rep, KPI)]: mean waiting time of contact group 2 over the epoch that starts at the state."""
    states = {r: initial_state(state, r) for r in contacts}
    sims = simulator.run_replications(contacts, states, int(state * 30), models)
    out = [(r, simulator.kpi(s, state + 1)) for r, s in sims.items()]
    return [(r, w) for r, w in out if w is not None]


def simulate(task):
    """Simulate one design point. task: key (seeds the random streams), state, cfg (six instances), bayesian, n."""
    s = zlib.crc32(f"{SEED}|{task['key']}|{'bayes' if task['bayesian'] else 'freq'}".encode())
    random.seed(s)
    np.random.seed(s)
    cfg = task['cfg']
    contacts = {j: make_contacts(cfg, j, task['key']) for j in range(1, task['n'] + 1)}
    return kpi_by_rep(contacts, task['state'], routing_models(task['bayesian'], cfg[4], cfg[5]))


def design_tasks(design, name, bayesian, n):
    """One task per design point; dp = index in the design."""
    tasks = []
    for k, row in enumerate(design):
        t = dict(zip(CFG, row[:6]), state=row[6], stack=row[7], dp=k)
        t.update(key=f'{name}|{k}', cfg=row[:6], bayesian=bayesian, n=n)
        tasks.append(t)
    return tasks


def run_tasks(tasks, out_csv, jobs, extra_cols=()):
    """Run `simulate` over the tasks in parallel and append the KPIs to out_csv (columns: key, extra_cols, rep, wait).
    Tasks already in out_csv are skipped, so an interrupted run can be resumed."""
    from joblib import Parallel, delayed
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = '1'
    done = set(pd.read_csv(out_csv, usecols=['key'])['key']) if os.path.exists(out_csv) else set()
    todo = [t for t in tasks if t['key'] not in done]
    chunk, t0 = max(jobs * 20, 1), time.time()
    for c0 in range(0, len(todo), chunk):
        part = todo[c0:c0 + chunk]
        res = Parallel(n_jobs=jobs, backend='loky', batch_size=1)(delayed(simulate)(t) for t in part)
        rows = [[t['key']] + [t[c] for c in extra_cols] + [r, w] for t, rr in zip(part, res) for r, w in rr]
        pd.DataFrame(rows, columns=['key', *extra_cols, 'rep', 'wait']).to_csv(
            out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
        print(f'  {os.path.basename(out_csv)}: {c0 + len(part)}/{len(todo)} design points, '
              f'{(time.time() - t0) / 60:.1f} min', flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# Intervals and the search over B and S
# ----------------------------------------------------------------------------------------------------------------------
def true_kpi():
    """True KPI of each state (mean of 1,000 replications of the real system)."""
    return pd.read_csv(path('data', 'truth.csv')).groupby('state')['wait'].mean().to_numpy()


def select(df, B, S, n):
    """First B instances, S stacks and n replications of the simulated design."""
    return df[(df[CFG] < B).all(axis=1) & (df['stack'] <= S) & (df['rep'] <= n)]


def state_intervals(df, alpha=ALPHA):
    """Quantile interval of the configuration means, per state."""
    m = df.groupby(['state', 'dp'])['wait'].mean().reset_index()
    q = m.groupby('state')['wait'].quantile([alpha / 2, 1 - alpha / 2]).unstack()
    q.columns = ['low', 'high']
    q['width'] = q['high'] - q['low']
    q['n_cfg'] = m.groupby('state').size()
    return q.reindex(range(NUM_STATES))


def avg_width(df):
    return state_intervals(df)['width'].mean()


def t_interval(y, alpha=ALPHA):
    h = student_t.ppf(1 - alpha / 2, len(y) - 1) * np.std(y, ddof=1) / np.sqrt(len(y))
    return np.mean(y) - h, np.mean(y) + h


def search_bs(df, B, S, n, max_S, tol=TOL):
    """Search over B and S with n fixed (Section 5.1). At each iteration, one instance (of all submodels) or one stack is
    left out at a time; the parameter with the smaller average leave-one-out width (the larger predicted change) is
    increased: B is doubled, S grows by one. B is limited to the B_MAX = 8 available instances. Stops when the actual
    change of the average width falls below tol. Returns (B, S) and the history."""
    p = [B, S]
    cur = avg_width(select(df, *p, n))
    hist = [dict(iteration=0, B=B, S=S, n=n, avg_ci_width=cur)]
    while True:
        loo = {}
        for m, limit in enumerate((B_MAX, max_S)):
            if p[m] >= limit or p[m] <= 1:
                continue
            widths = []
            for k in range(p[m]):
                units = [list(range(p[0])), list(range(1, p[1] + 1))]
                units[m].pop(k)
                d = df[df[CFG].isin(units[0]).all(axis=1) & df['stack'].isin(units[1]) & (df['rep'] <= n)]
                if not d.empty:
                    widths.append(avg_width(d))
            loo[m] = np.mean(widths) if widths else np.inf
        if not loo:
            hist[-1]['stop'] = 'B and S at their limits'
            break
        m = min(loo, key=loo.get)
        p[m] = 2 * p[m] if m == 0 else p[m] + 1
        new = avg_width(select(df, *p, n))
        hist.append(dict(iteration=len(hist), increased='BS'[m], B=p[0], S=p[1], n=n, avg_ci_width=new,
                         actual_change=new - cur, **{f'loo_width_{"BS"[k]}': v for k, v in loo.items()}))
        if abs(new - cur) < tol:
            hist[-1]['stop'] = 'change below tolerance'
            break
        cur = new
    return tuple(p), pd.DataFrame(hist)


# ----------------------------------------------------------------------------------------------------------------------
# Importance scores (Section 5.1), per state
# ----------------------------------------------------------------------------------------------------------------------
def config_matrix(d):
    """One state's rows -> configurations (m x 6) and outputs (m x n)."""
    w = d.pivot_table(index='dp', columns='rep', values='wait', aggfunc='first').sort_index()
    if w.isna().any().any():
        raise ValueError('unequal numbers of replications across configurations')
    return d.groupby('dp')[CFG].first().loc[w.index].to_numpy(dtype=int), w.to_numpy(dtype=float)


def tree_quantities(configs, outputs):
    """Fully grown regression tree on the one-hot encoded instances: TSS, RSS, number of leaves L, and the mean TSS
    reduction of the splits on each submodel (I_l, normalized)."""
    from sklearn.tree import DecisionTreeRegressor
    X = pd.get_dummies(pd.DataFrame(np.repeat(configs, outputs.shape[1], axis=0), columns=CFG).astype(str), drop_first=True)
    y = outputs.ravel()
    tree = DecisionTreeRegressor(random_state=42).fit(X, y)
    owner = [next(j for j, s in enumerate(CFG) if col.startswith(s + '_')) for col in X.columns]
    t = tree.tree_
    total, count = np.zeros(len(CFG)), np.zeros(len(CFG))
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
    """sigma2_SU (not truncated at zero) and sigma2_MC = RSS / (n (N - L))."""
    n = outputs.shape[1]
    tss, rss, leaves, _ = tree_quantities(configs, outputs)
    mc = rss / (outputs.size - leaves) / n
    return (tss - rss) / (n * (leaves - 1)) - mc, mc


def importance_state(d, rng, n_trees=N_TREES):
    """Shares of the six submodels and of simulation noise for one state, with bagging."""
    configs, out = config_matrix(d)
    n = out.shape[1]
    su, mc = variance_components(configs, out)
    su = max(0.0, su)
    w = []
    for _ in range(n_trees):
        idx = rng.integers(len(configs), size=len(configs))
        if len(np.unique(configs[idx], axis=0)) == 1:
            continue
        boot = out[idx[:, None], rng.integers(n, size=(len(idx), n))]
        w.append(tree_quantities(configs[idx], boot)[3])
    return np.append(np.mean(w, axis=0) * su, mc) / (su + mc)


def importance(df, seed):
    """Shares averaged over the states."""
    rng = np.random.default_rng(seed)
    return pd.Series(np.mean([importance_state(df[df['state'] == s], rng) for s in range(NUM_STATES)], axis=0),
                     index=CFG + ['Error'])
