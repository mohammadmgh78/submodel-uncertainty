"""
Input submodels: arrivals (exponential interarrival times by hour) and patience and handle times (Gamma marginals with a
Gaussian copula), fitted to inputs/training_day.csv by maximum likelihood.

    python fit_input_submodels.py

Recomputes and compares with the files in inputs/:
  * arrival_instances.pkl      point estimates and B = 8 bootstrap instances (seeded);
  * copula_point_estimate.pkl  point estimates;
  * copula_instances.pkl       B = 8 bootstrap instances. The original bootstrap was not seeded and its parameters were
                               not stored, so they were recovered by maximum likelihood from the records drawn from
                               each instance (copula_instance_draws.csv.gz). The two independent sets of records,
                               'single' and 'design', are first compared, then pooled.

As noted in the paper, the Bayesian setting uses the same input submodels. `bootstrap_copula` shows how new instances
would be generated.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import expon, gamma, norm

import common as C


def fit_arrivals(df, rng=None):
    """Mean interarrival time in each hour. With rng, the interarrival times within each hour are bootstrapped."""
    df = df.sort_values(['Arrival Time']).copy()
    gap = df['Arrival Time'].diff()
    df['gap'] = np.where(gap > 0, gap, 0.1)
    df['hour'] = (df['Arrival Time'] // C.HOUR).astype(int)
    scales = []
    for h in range(C.T_MAX // C.HOUR):
        x = df.loc[df['hour'] == h, 'gap'].to_numpy()
        if len(x) > 1:
            if rng is not None:
                x = rng.choice(x, size=len(x), replace=True)
            scales.append(expon.fit(x, floc=0)[1])
        else:
            scales.append(df['gap'].mean())
    return scales


def fit_copula(df):
    """MLE of (shape_p, scale_p, shape_s, scale_s, rho) from censored records: abandoned contacts reveal the patience
    time; served contacts reveal the handle time and a lower bound of the patience time."""
    w, s, abandon = (df[c].to_numpy() for c in ('Waiting Time', 'Handle Time', 'Abandon'))
    eps = 1e-12

    def neg_log_likelihood(theta):
        shape_s, scale_s, shape_p, scale_p, rho = theta
        A, S = abandon == 1, (abandon == 0) & np.isfinite(s)
        ll = 0.0
        if np.any(A):
            ll += gamma.logpdf(np.maximum(w[A], eps), a=shape_p, scale=scale_p).sum()
        if np.any(S):
            sS, wS = np.maximum(s[S], eps), np.maximum(w[S], eps)
            ll += gamma.logpdf(sS, a=shape_s, scale=scale_s).sum()
            z1 = norm.ppf(np.clip(gamma.cdf(wS, a=shape_p, scale=scale_p), eps, 1 - eps))
            z2 = norm.ppf(np.clip(gamma.cdf(sS, a=shape_s, scale=scale_s), eps, 1 - eps))
            ll += norm.logsf((z1 - rho * z2) / np.sqrt(max(1.0 - rho ** 2, eps))).sum()
        return -ll

    def gamma_moments(x):
        x = x[np.isfinite(x) & (x > 0)]
        if x.size == 0:
            return 1.0, 1.0
        m, v = x.mean(), x.var(ddof=1) if x.size > 1 else x.mean() ** 2
        return max((m * m) / max(v, 1e-8), 1e-3), max(v / max(m, 1e-8), 1e-3)

    x0 = np.array([*gamma_moments(s[abandon == 0]), *gamma_moments(w[abandon == 1]), 0.1], dtype=float)
    res = minimize(neg_log_likelihood, x0=x0, method='L-BFGS-B', bounds=[(1e-4, None)] * 4 + [(-0.999, 0.999)])
    shape_s, scale_s, shape_p, scale_p, rho = res.x
    return shape_p, scale_p, shape_s, scale_s, rho


def bootstrap_copula(df, B, rng):
    """B bootstrap instances for one contact group."""
    return [fit_copula(df.iloc[rng.integers(len(df), size=len(df))].reset_index(drop=True)) for _ in range(B)]


def fit_complete(P, H):
    """MLE from complete (uncensored) pairs."""
    a_p, _, s_p = gamma.fit(P, floc=0)
    a_s, _, s_s = gamma.fit(H, floc=0)
    z1 = norm.ppf(np.clip(gamma.cdf(P, a_p, scale=s_p), 1e-12, 1 - 1e-12))
    z2 = norm.ppf(np.clip(gamma.cdf(H, a_s, scale=s_s), 1e-12, 1 - 1e-12))
    return a_p, s_p, a_s, s_s, float(np.corrcoef(z1, z2)[0, 1])


def recover_copula_instances():
    d = pd.read_csv(C.path('inputs', 'copula_instance_draws.csv.gz'), float_precision='round_trip')
    boot = {1: [], 2: []}
    for (g, c), x in d.groupby(['Group', 'Instance']):
        fits = {}
        for src, part in (('single', x[x['Source'] == 'single']), ('design', x[x['Source'] == 'design'])):
            f = fit_complete(part['Patience Time'].to_numpy(), part['Handle Time'].to_numpy())
            fits[src] = np.array([f[0] * f[1], f[2] * f[3], f[4]])              # mean patience, mean handle, rho
            fits[src + '_se'] = np.array([part['Patience Time'].std(), part['Handle Time'].std(), 1 - f[4] ** 2]) / np.sqrt(len(part))
        z = (fits['single'] - fits['design']) / np.hypot(fits['single_se'], fits['design_se'])
        assert np.abs(z).max() < 3.5, 'the two sets of records of an instance disagree'
        x = pd.concat([x[x['Source'] == 'single'], x[x['Source'] == 'design']])
        boot[g].append(fit_complete(x['Patience Time'].to_numpy(), x['Handle Time'].to_numpy()))
    return boot


def check(name, new, old):
    err = max(np.abs(np.asarray(a, float) - np.asarray(b, float)).max() for a, b in zip(new, old))
    print(f'{name}: max difference from the provided file = {err:.1e}')


if __name__ == '__main__':
    train = pd.read_csv(C.path('inputs', 'training_day.csv'))
    by_group = {g: train[train['Contact Group'] == g] for g in (1, 2)}
    arr, cop = C.input_instances()
    rng = np.random.default_rng(C.SEED)
    for g, d in by_group.items():
        check(f'arrivals, group {g}, point estimate', [fit_arrivals(d)], [arr['mle'][g]])
        check(f'arrivals, group {g}, instances', [fit_arrivals(d, rng) for _ in range(C.B_MAX)], arr['boot'][g])
    point = pd.read_pickle(C.path('inputs', 'copula_point_estimate.pkl'))
    for g, d in by_group.items():
        check(f'patience and handle, group {g}, point estimate', [fit_copula(d)], [point[g]])
    rec = recover_copula_instances()
    for g in (1, 2):
        check(f'patience and handle, group {g}, instances', rec[g], cop['boot'][g])
