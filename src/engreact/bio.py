"""Enzyme kinetics and bioreactors: Michaelis-Menten with competitive, uncompetitive and non-competitive
inhibition, the Lineweaver-Burk and Hanes plots and nonlinear fitting, Monod growth with substrate and
product terms, the chemostat (steady state, washout, maximum productivity), batch growth and fed-batch.

>>> from engreact import bio
>>> round(bio.chemostat_washout(mu_max=0.5, K_s=0.1, S_in=10.0), 4)       # dilution rate at washout (1/h)
0.495
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize


# ------------------------------------------------------------------ enzyme kinetics
def michaelis_menten(
    S, V_max: float, K_m: float, inhibitor=0.0, K_I: float = np.inf, inhibition: str | None = None
) -> float:
    """Rate V_max S / (K_m + S) with optional inhibition: 'competitive' (K_m (1 + I/K_I)), 'uncompetitive'
    (S (1 + I/K_I) in the denominator), 'noncompetitive' (V_max / (1 + I/K_I))."""
    S = np.asarray(S, float)
    f = 1 + inhibitor / K_I
    if inhibition == "competitive":
        return V_max * S / (K_m * f + S)
    if inhibition == "uncompetitive":
        return V_max * S / (K_m + S * f)
    if inhibition == "noncompetitive":
        return V_max / f * S / (K_m + S)
    return V_max * S / (K_m + S)


def lineweaver_burk(S, rates) -> dict:
    """1/v = (K_m/V_max)(1/S) + 1/V_max; returns V_max, K_m from the linear fit."""
    x, y = 1 / np.asarray(S, float), 1 / np.asarray(rates, float)
    slope, intercept = np.polyfit(x, y, 1)
    return {"V_max": float(1 / intercept), "K_m": float(slope / intercept)}


def hanes_woolf(S, rates) -> dict:
    """S/v = S/V_max + K_m/V_max."""
    S, rates = np.asarray(S, float), np.asarray(rates, float)
    slope, intercept = np.polyfit(S, S / rates, 1)
    return {"V_max": float(1 / slope), "K_m": float(intercept / slope)}


def fit_michaelis_menten(S, rates) -> dict:
    """Nonlinear least squares with standard errors (the recommended method)."""
    S, rates = np.asarray(S, float), np.asarray(rates, float)
    guess = hanes_woolf(S, rates)
    res = optimize.least_squares(lambda q: michaelis_menten(S, q[0], q[1]) - rates, [guess["V_max"], guess["K_m"]])
    n, p = len(S), 2
    s2 = np.sum(res.fun**2) / max(n - p, 1)
    cov = s2 * np.linalg.inv(res.jac.T @ res.jac)
    se = np.sqrt(np.diag(cov))
    return {
        "V_max": float(res.x[0]),
        "K_m": float(res.x[1]),
        "se_V_max": float(se[0]),
        "se_K_m": float(se[1]),
        "rss": float(np.sum(res.fun**2)),
    }


def batch_enzyme(S0: float, V_max: float, K_m: float, t) -> np.ndarray:
    """Substrate in a batch enzyme reactor (integrated Michaelis-Menten solved for S at each t)."""
    t = np.asarray(t, float)

    def S_of(tt):
        f = lambda S: K_m * np.log(S0 / S) + (S0 - S) - V_max * tt  # noqa: E731
        return optimize.brentq(f, 1e-15 * S0, S0) if tt > 0 else S0

    return np.array([S_of(tt) for tt in t])


# ------------------------------------------------------------------ growth and bioreactors
def monod(S, mu_max: float, K_s: float, K_i: float | None = None) -> float:
    """Specific growth rate mu = mu_max S/(K_s + S), with optional substrate inhibition (Andrews): /(1 + S/K_i)."""
    S = np.asarray(S, float)
    mu = mu_max * S / (K_s + S)
    return mu / (1 + S / K_i) if K_i else mu


def batch_growth(
    X0: float, S0: float, mu_max: float, K_s: float, Y_xs: float, t, k_d: float = 0.0, m_s: float = 0.0
) -> dict:
    """Batch: dX/dt = (mu - k_d) X, dS/dt = -(mu/Y_xs + m_s) X."""

    def rhs(_, y):
        X, S = y
        mu = monod(max(S, 0.0), mu_max, K_s)
        return [(mu - k_d) * X, -(mu / Y_xs + m_s) * X if S > 0 else 0.0]

    t = np.asarray(t, float)
    sol = integrate.solve_ivp(rhs, (t[0], t[-1]), [X0, S0], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10)
    return {"t": sol.t, "X": sol.y[0], "S": np.maximum(sol.y[1], 0.0)}


def chemostat_steady(D: float, S_in: float, mu_max: float, K_s: float, Y_xs: float, k_d: float = 0.0) -> dict:
    """Steady state: mu(S) = D + k_d -> S = K_s (D + k_d)/(mu_max - D - k_d) and X = Y_xs (S_in - S) D/(D + k_d)
    (X = Y_xs (S_in - S) for k_d = 0). Washout (X = 0) when S would exceed S_in."""
    if D + k_d >= mu_max:
        return {"S": S_in, "X": 0.0, "washout": True}
    S = K_s * (D + k_d) / (mu_max - D - k_d)
    if S >= S_in:
        return {"S": S_in, "X": 0.0, "washout": True}
    X = Y_xs * (S_in - S) * D / (D + k_d)
    return {"S": float(S), "X": float(X), "washout": False, "productivity": float(D * X)}


def chemostat_washout(mu_max: float, K_s: float, S_in: float) -> float:
    """Critical dilution rate D_crit = mu_max S_in/(K_s + S_in)."""
    return mu_max * S_in / (K_s + S_in)


def chemostat_optimum(mu_max: float, K_s: float, S_in: float) -> float:
    """Dilution rate maximising the biomass productivity D X: D_opt = mu_max (1 - sqrt(K_s/(K_s + S_in)))."""
    return mu_max * (1 - np.sqrt(K_s / (K_s + S_in)))


def fed_batch(
    X0: float, S0: float, V0: float, S_feed: float, F: float, mu_max: float, K_s: float, Y_xs: float, t
) -> dict:
    """Fed-batch with constant feed rate F (volume/time):
    d(VX)/dt = mu V X, d(VS)/dt = F S_feed - (mu/Y) V X, dV/dt = F."""

    def rhs(_, y):
        VX, VS, V = y
        S = max(VS / V, 0.0)
        mu = monod(S, mu_max, K_s)
        return [mu * VX, F * S_feed - mu / Y_xs * VX, F]

    t = np.asarray(t, float)
    sol = integrate.solve_ivp(
        rhs, (t[0], t[-1]), [X0 * V0, S0 * V0, V0], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10
    )
    V = sol.y[2]
    return {"t": sol.t, "V": V, "X": sol.y[0] / V, "S": np.maximum(sol.y[1] / V, 0.0)}
