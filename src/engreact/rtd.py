"""Residence-time distributions: E(t) and F(t) from pulse and step tracer data, moments (mean residence
time, variance, skewness), ideal-reactor RTDs, the tanks-in-series and axial-dispersion models and their
fitting, and conversion for a first-order reaction under the segregation and maximum-mixedness models.

>>> from engreact import rtd
>>> import numpy as np
>>> t = np.linspace(0, 50, 501); E = rtd.tanks_in_series_E(t, tau=10.0, n=4)
>>> round(rtd.moments(t, E)["variance"] / 10.0**2, 3)                # sigma^2/tau^2 = 1/n
0.25
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, special

from ._common import trapezoid


def E_from_pulse(t, C) -> np.ndarray:
    """E(t) = C(t) / int C dt from a pulse experiment."""
    t, C = np.asarray(t, float), np.asarray(C, float)
    return C / trapezoid(C, t)


def F_from_step(C, C_max: float) -> np.ndarray:
    """F(t) = C(t)/C_max from a positive step."""
    return np.asarray(C, float) / C_max


def E_from_F(t, F) -> np.ndarray:
    return np.gradient(np.asarray(F, float), np.asarray(t, float))


def F_from_E(t, E) -> np.ndarray:
    t, E = np.asarray(t, float), np.asarray(E, float)
    return np.concatenate([[0.0], integrate.cumulative_trapezoid(E, t)])


def moments(t, E) -> dict:
    t, E = np.asarray(t, float), np.asarray(E, float)
    area = trapezoid(E, t)
    tm = trapezoid(t * E, t) / area
    var = trapezoid((t - tm) ** 2 * E, t) / area
    skew = trapezoid((t - tm) ** 3 * E, t) / area / var**1.5 if var > 0 else 0.0
    return {"area": float(area), "mean": float(tm), "variance": float(var), "skewness": float(skew)}


# ------------------------------------------------------------------ ideal and model RTDs
def cstr_E(t, tau: float) -> np.ndarray:
    t = np.asarray(t, float)
    return np.exp(-t / tau) / tau


def pfr_E(t, tau: float, width: float | None = None) -> np.ndarray:
    """A narrow Gaussian approximation to the delta function at tau (for plotting and numerics)."""
    t = np.asarray(t, float)
    w = width or max(tau * 0.01, 1e-9)
    return np.exp(-0.5 * ((t - tau) / w) ** 2) / (w * np.sqrt(2 * np.pi))


def tanks_in_series_E(t, tau: float, n: float) -> np.ndarray:
    """E(t) = t^(n-1) exp(-n t/tau) n^n / (tau^n Gamma(n)) with tau the total mean residence time."""
    t = np.asarray(t, float)
    return t ** (n - 1) * np.exp(-n * t / tau) * n**n / (tau**n * special.gamma(n))


def dispersion_step_response(Pe: float, theta_max: float = 3.0, n_nodes: int = 200, n_out: int = 300) -> dict:
    """Closed-closed axial dispersion model solved numerically (method of lines) for a step in inlet tracer:
    dC/dtheta = (1/Pe) d2C/dz2 - dC/dz with Danckwerts conditions C - (1/Pe) dC/dz = C_in at z = 0 and
    dC/dz = 0 at z = 1. Returns theta and F(theta) = C(z = 1, theta)."""
    dz = 1.0 / n_nodes

    def rhs(_, C):
        # Danckwerts inlet via a ghost node C_g: (C0 + C_g)/2 - (C0 - C_g)/(Pe dz) = 1; outlet: zero gradient
        C_g = (1 - C[0] / 2 + C[0] / (Pe * dz)) / (0.5 + 1 / (Pe * dz))
        C_ext = np.concatenate([[C_g], C, [C[-1]]])
        flux = 0.5 * (C_ext[:-1] + C_ext[1:]) - (C_ext[1:] - C_ext[:-1]) / (Pe * dz)  # central convection + diffusion
        return -(flux[1:] - flux[:-1]) / dz

    theta = np.linspace(0, theta_max, n_out)
    sol = integrate.solve_ivp(
        rhs, (0, theta_max), np.zeros(n_nodes), t_eval=theta, method="LSODA", rtol=1e-8, atol=1e-10
    )
    return {"theta": theta, "F": sol.y[-1]}


def dispersion_E(theta, Pe: float, vessel: str = "closed") -> np.ndarray:
    """Dimensionless E(theta) of the axial dispersion model. 'open' (or any Pe >= 100): the analytic
    open-vessel form sqrt(Pe/(4 pi theta)) exp(-Pe (1 - theta)^2 / (4 theta)); 'closed': numerical solution of
    the closed-closed model differentiated from its step response."""
    theta = np.asarray(theta, float)
    if vessel == "open" or Pe >= 100:
        with np.errstate(divide="ignore", invalid="ignore"):
            E = np.sqrt(Pe / (4 * np.pi * theta)) * np.exp(-Pe * (1 - theta) ** 2 / (4 * theta))
        return np.nan_to_num(E)
    step = dispersion_step_response(Pe, theta_max=max(float(np.max(theta)), 1e-6), n_nodes=400, n_out=800)
    E_grid = np.gradient(step["F"], step["theta"])
    return np.maximum(np.interp(theta, step["theta"], E_grid), 0.0)


def dispersion_variance(Pe: float, vessel: str = "closed") -> float:
    """sigma^2/tau^2: closed vessel 2/Pe - 2/Pe^2 (1 - exp(-Pe)); open vessel 2/Pe + 8/Pe^2."""
    if vessel == "closed":
        return 2 / Pe - 2 / Pe**2 * (1 - np.exp(-Pe))
    return 2 / Pe + 8 / Pe**2


def fit_tanks_in_series(t, E) -> dict:
    """n from the variance: n = tau^2 / sigma^2 (method of moments), plus a least-squares refinement."""
    m = moments(t, E)
    n_mom = m["mean"] ** 2 / m["variance"]
    t = np.asarray(t, float)
    res = optimize.least_squares(
        lambda q: tanks_in_series_E(t, q[0], q[1]) - E, [m["mean"], n_mom], bounds=([1e-6, 0.5], [np.inf, 1e4])
    )
    return {
        "n_moments": float(n_mom),
        "tau": float(res.x[0]),
        "n": float(res.x[1]),
        "mean": m["mean"],
        "variance": m["variance"],
    }


def peclet_from_variance(sigma2_over_tau2: float, vessel: str = "closed") -> float:
    return float(optimize.brentq(lambda Pe: dispersion_variance(Pe, vessel) - sigma2_over_tau2, 1e-4, 1e6))


# ------------------------------------------------------------------ conversion with an RTD
def segregation_conversion(t, E, k: float, order: float = 1, C_A0: float = 1.0) -> float:
    """Segregation model: X = int (X_batch(t)) E(t) dt, with the batch conversion for the given order."""
    t, E = np.asarray(t, float), np.asarray(E, float)
    if abs(order - 1) < 1e-12:
        Xb = 1 - np.exp(-k * t)
    else:
        Xb = 1 - (1 + (order - 1) * k * C_A0 ** (order - 1) * t) ** (-1 / (order - 1))
    return float(trapezoid(Xb * E, t) / trapezoid(E, t))


def maximum_mixedness_conversion(t, E, k: float, order: float = 1, C_A0: float = 1.0) -> float:
    """Maximum-mixedness model (Zwietering): integrate dX/dlambda = -r(X)/C_A0 + E(lambda)/(1 - F(lambda)) X from
    lambda = t_max back to 0."""
    t, E = np.asarray(t, float), np.asarray(E, float)
    F = F_from_E(t, E)
    F = F / F[-1]

    def rhs(lam, X):
        i = np.searchsorted(t, lam, side="right") - 1
        i = min(max(i, 0), len(t) - 1)
        Ei, Fi = np.interp(lam, t, E), np.interp(lam, t, F)
        rate = k * C_A0 ** (order - 1) * (1 - X[0]) ** order
        return [-rate + Ei / max(1 - Fi, 1e-9) * X[0]]

    lam_max = t[np.nonzero(F < 0.999)[0][-1]] if np.any(F < 0.999) else t[-1]
    sol = integrate.solve_ivp(rhs, (lam_max, 0.0), [0.0], method="LSODA", rtol=1e-8, atol=1e-10, max_step=(t[1] - t[0]))
    return float(sol.y[0][-1])


def tanks_in_series_conversion(k: float, tau: float, n: int) -> float:
    return 1 - (1 + k * tau / n) ** (-n)


def dispersion_conversion(k: float, tau: float, Pe: float) -> float:
    """First-order closed-vessel dispersion model (Danckwerts 1953):
    X = 1 - 4 q exp(Pe/2) / ((1+q)^2 exp(q Pe/2) - (1-q)^2 exp(-q Pe/2)),  q = sqrt(1 + 4 k tau / Pe)."""
    q = np.sqrt(1 + 4 * k * tau / Pe)
    # written with (q - 1) Pe/2 -> k tau as Pe -> infinity, so the exponents stay finite at any Peclet number
    return float(1 - 4 * q / ((1 + q) ** 2 * np.exp((q - 1) * Pe / 2) - (1 - q) ** 2 * np.exp(-(q + 1) * Pe / 2)))
