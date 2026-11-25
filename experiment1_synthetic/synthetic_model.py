"""The synthetic simulation model of Section 5.2: Y = p(X1) + q(X2)."""
import numpy as np
from scipy.stats import norm

MU_1, SIGMA_1 = 1, 0.5
MU_2, SIGMA_2 = 1, 2
TRUE_KPI = 49.13            # E[Y]
DEG_P, DEG_Q = 2, 1         # degrees of the polynomials fitted to p and q


def p_true(x):
    return x**7 + x**4 + 3*x**3 + 4 + np.sin(x)


def q_true(x):
    return x**3 + x**2 + 4*x


def training_data(size, rng):
    """Observations of X1, p(X1), X2 and q(X2) from `size` replications with the true submodels."""
    x1 = rng.normal(MU_1, SIGMA_1, size)
    x2 = rng.normal(MU_2, SIGMA_2, size)
    return x1, p_true(x1), x2, q_true(x2)


def fit_submodels(data, rng=None, inputs=True, responses=True):
    """Fit the four submodels. With `rng`, the input data and/or the input-output pairs are bootstrapped first:
    X1 and X2 are resampled independently, and the pairs of p and of q are resampled jointly."""
    x1, p_out, x2, q_out = data
    size = len(x1)

    def idx(resample):
        return rng.choice(size, size=size, replace=True) if rng is not None and resample else np.arange(size)

    m1, s1 = norm.fit(x1[idx(inputs)])
    m2, s2 = norm.fit(x2[idx(inputs)])
    ip, iq = idx(responses), idx(responses)
    p_hat = np.poly1d(np.polyfit(x1[ip], p_out[ip], DEG_P))
    q_hat = np.poly1d(np.polyfit(x2[iq], q_out[iq], DEG_Q))
    return (m1, s1), (m2, s2), p_hat, q_hat


def estimate_kpi(submodels, n, rng):
    """Average output of n replications."""
    (m1, s1), (m2, s2), p, q = submodels
    return float(np.mean(p(rng.normal(m1, s1, n)) + q(rng.normal(m2, s2, n))))
