"""Analysis of rate data: the differential method (numerical differentiation of concentration-time data and
the log-log plot), the integral method (test an assumed order), the half-life and initial-rate methods, and
nonlinear regression of integrated rate laws with standard errors and model comparison by AIC.

>>> from engreact import rate_data
>>> import numpy as np
>>> t = np.linspace(0, 100, 11); C = 2.0 / (1 + 2.0 * 0.05 * t)               # second order, k = 0.05
>>> fit = rate_data.nonlinear_fit(t, C, order=2)
>>> round(fit["k"], 4), round(fit["C0"], 4)
(0.05, 2.0)
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


def integrated_concentration(t, C0: float, k: float, order: float):
    """C(t) for -dC/dt = k C^n (n != 1: C^(1-n) = C0^(1-n) + (n-1) k t)."""
    t = np.asarray(t, float)
    if abs(order - 1) < 1e-12:
        return C0 * np.exp(-k * t)
    base = C0 ** (1 - order) + (order - 1) * k * t
    return np.where(base > 0, np.abs(base) ** (1 / (1 - order)), 0.0)


def half_life(C0: float, k: float, order: float) -> float:
    if abs(order - 1) < 1e-12:
        return float(np.log(2) / k)
    return float((2 ** (order - 1) - 1) / ((order - 1) * k * C0 ** (order - 1)))


def differential_method(t, C, smoothing: int = 3) -> dict:
    """Estimate -dC/dt at each point (central differences on a local polynomial of given degree), then
    regress ln(-dC/dt) on ln C: order = slope, k = exp(intercept). Returns rates and the fit."""
    t, C = np.asarray(t, float), np.asarray(C, float)
    n = len(t)
    rates = np.zeros(n)
    half = max(smoothing, 2)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        coef = np.polyfit(t[lo:hi], C[lo:hi], min(2, hi - lo - 1))
        rates[i] = -np.polyval(np.polyder(coef), t[i])
    keep = (rates > 0) & (C > 0)
    x, y = np.log(C[keep]), np.log(rates[keep])
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    r2 = 1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"rates": rates, "order": float(slope), "k": float(np.exp(intercept)), "r2": float(r2)}


def integral_method(t, C, order: float) -> dict:
    """Linearise the integrated rate law for the assumed order and fit k by least squares (through the known C0
    intercept): first order: ln(C0/C) = k t; nth order: C^(1-n) - C0^(1-n) = (n-1) k t. Returns k and r^2."""
    t, C = np.asarray(t, float), np.asarray(C, float)
    C0 = C[0]
    if abs(order - 1) < 1e-12:
        y = np.log(C0 / C)
    else:
        y = (C ** (1 - order) - C0 ** (1 - order)) / (order - 1)
    k = float(np.sum(t * y) / np.sum(t * t))
    fit = k * t
    r2 = 1 - np.sum((y - fit) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300)
    return {"k": k, "r2": float(r2), "y": y}


def half_life_method(C0s, half_lives) -> dict:
    """Order from the dependence of the half-life on the initial concentration: ln t1/2 = ln a + (1 - n) ln C0."""
    x, y = np.log(np.asarray(C0s, float)), np.log(np.asarray(half_lives, float))
    slope, intercept = np.polyfit(x, y, 1)
    order = 1 - slope
    return {"order": float(order), "intercept": float(intercept)}


def initial_rates_method(C0s, rates0) -> dict:
    """Order and k from initial rates: ln r0 = ln k + n ln C0."""
    x, y = np.log(np.asarray(C0s, float)), np.log(np.asarray(rates0, float))
    slope, intercept = np.polyfit(x, y, 1)
    return {"order": float(slope), "k": float(np.exp(intercept))}


def nonlinear_fit(t, C, order: float | None = None, fit_C0: bool = True) -> dict:
    """Least-squares fit of the integrated rate law to concentration-time data. If order is None it is fitted
    too. Returns k, order, C0, their standard errors, the residual sum of squares and the AIC."""
    t, C = np.asarray(t, float), np.asarray(C, float)

    def model(params):
        if order is None:
            k, n = params[0], params[1]
            C0 = params[2] if fit_C0 else C[0]
        else:
            k, n = params[0], order
            C0 = params[1] if fit_C0 else C[0]
        return integrated_concentration(t, C0, k, n)

    p0 = [max(abs(np.gradient(C, t)[0]) / max(C[0], 1e-12), 1e-6)]
    if order is None:
        p0.append(1.0)
    if fit_C0:
        p0.append(C[0])
    res = optimize.least_squares(lambda p: model(p) - C, p0, x_scale="jac")
    n, p = len(C), len(res.x)
    rss = float(np.sum(res.fun**2))
    s2 = rss / max(n - p, 1)
    try:
        cov = s2 * np.linalg.inv(res.jac.T @ res.jac)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    out = {
        "k": float(res.x[0]),
        "se_k": float(se[0]),
        "order": float(res.x[1]) if order is None else float(order),
        "rss": rss,
        "aic": float(n * np.log(max(rss, 1e-300) / n) + 2 * p),
        "C0": float(res.x[-1]) if fit_C0 else float(C[0]),
        "predict": lambda tt: integrated_concentration(
            tt, res.x[-1] if fit_C0 else C[0], res.x[0], res.x[1] if order is None else order
        ),
    }
    if order is None:
        out["se_order"] = float(se[1])
    return out


def compare_orders(t, C, orders=(0, 0.5, 1, 1.5, 2)) -> list[dict]:
    """Fit each candidate order and rank by AIC."""
    fits = [{**nonlinear_fit(t, C, order=n), "order": n} for n in orders]
    return sorted(fits, key=lambda f: f["aic"])
