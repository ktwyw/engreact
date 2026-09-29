"""The ideal reactors: batch, CSTR, PFR and PBR mole balances - analytic solutions for power-law kinetics and
general numerical solutions for any rate law - plus Levenspiel plots, reactors in series, space time and
Damkohler numbers, recycle, and gas-phase packed beds with Ergun pressure drop.

Rate laws are functions of conversion, -r_A(X) (per unit volume, or per unit catalyst mass for a PBR), or of
concentrations where stated.

>>> from engreact import ideal
>>> round(ideal.cstr_conversion_first_order(k=0.5, tau=2.0), 3)
0.5
>>> round(ideal.pfr_conversion_first_order(k=0.5, tau=2.0), 3)
0.632
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize


# ------------------------------------------------------------------ analytic special cases
def batch_conversion_first_order(k: float, t) -> float:
    return 1 - np.exp(-k * np.asarray(t, float))


def batch_time_nth_order(k: float, C_A0: float, X: float, order: float) -> float:
    """Time to reach conversion X in a constant-volume batch reactor for -r = k C^n."""
    if abs(order - 1) < 1e-12:
        return float(-np.log(1 - X) / k)
    return float(((1 - X) ** (1 - order) - 1) / ((order - 1) * k * C_A0 ** (order - 1)))


def cstr_conversion_first_order(k: float, tau: float) -> float:
    Da = k * tau
    return Da / (1 + Da)


def pfr_conversion_first_order(k: float, tau: float) -> float:
    return float(1 - np.exp(-k * tau))


def cstr_conversion_second_order(k: float, C_A0: float, tau: float) -> float:
    Da = k * C_A0 * tau
    return float((1 + 2 * Da - np.sqrt(1 + 4 * Da)) / (2 * Da))


def pfr_conversion_second_order(k: float, C_A0: float, tau: float) -> float:
    Da = k * C_A0 * tau
    return Da / (1 + Da)


def damkohler(k: float, tau: float, C_A0: float = 1.0, order: float = 1) -> float:
    return k * C_A0 ** (order - 1) * tau


# ------------------------------------------------------------------ general design equations, rate given as -r_A(X)
def cstr_volume(F_A0: float, X: float, rate_fn) -> float:
    """V = F_A0 X / (-r_A(X))."""
    return F_A0 * X / rate_fn(X)


def pfr_volume(F_A0: float, X: float, rate_fn) -> float:
    """V = F_A0 int_0^X dX / (-r_A)."""
    val, _ = integrate.quad(lambda x: 1 / rate_fn(x), 0, X)
    return F_A0 * val


def cstr_conversion(F_A0: float, V: float, rate_fn, X_max: float = 0.999999) -> float:
    return float(optimize.brentq(lambda X: F_A0 * X / rate_fn(X) - V, 1e-12, X_max))


def pfr_conversion(F_A0: float, V: float, rate_fn, X_max: float = 0.999999) -> float:
    return float(optimize.brentq(lambda X: pfr_volume(F_A0, X, rate_fn) - V, 1e-12, X_max))


def pfr_profile(F_A0: float, V: float, rate_fn, n: int = 200) -> dict:
    """Conversion along a PFR: dX/dV = -r_A(X) / F_A0."""
    sol = integrate.solve_ivp(
        lambda v, X: [rate_fn(X[0]) / F_A0],
        (0, V),
        [0.0],
        t_eval=np.linspace(0, V, n),
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )
    return {"V": sol.t, "X": sol.y[0]}


def levenspiel(X, rate, F_A0: float = 1.0) -> dict:
    """Levenspiel plot data: F_A0/(-r_A) against X, with CSTR (rectangle) and PFR (trapezoidal area) volumes to
    each X from tabulated rates."""
    X, rate = np.asarray(X, float), np.asarray(rate, float)
    y = F_A0 / rate
    pfr = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(X))])
    return {"X": X, "F_over_r": y, "V_cstr": X * y, "V_pfr": pfr}


def _stage_out(kind: str, V: float, X_in: float, F_A0: float, rate_fn) -> float:
    if kind == "cstr":
        return float(optimize.brentq(lambda x: F_A0 * (x - X_in) / rate_fn(x) - V, X_in + 1e-12, 0.999999))
    return float(
        optimize.brentq(
            lambda x: F_A0 * integrate.quad(lambda xx: 1 / rate_fn(xx), X_in, x)[0] - V, X_in + 1e-12, 0.999999
        )
    )


def reactors_in_series(F_A0: float, stages, rate_fn) -> list[dict]:
    """A sequence of ('cstr' or 'pfr', V) stages; returns the conversion entering and leaving each."""
    out, X = [], 0.0
    for kind, V in stages:
        X_out = _stage_out(kind, V, X, F_A0, rate_fn)
        out.append({"type": kind, "V": V, "X_in": X, "X_out": X_out})
        X = X_out
    return out


def cstrs_in_series_first_order(k: float, tau_each: float, n: int) -> float:
    """Conversion after n equal CSTRs: 1 - 1/(1 + k tau)^n."""
    return 1 - (1 + k * tau_each) ** (-n)


def space_time(V: float, v0: float) -> float:
    return V / v0


def recycle_reactor(F_A0: float, V: float, rate_fn, R: float) -> dict:
    """PFR with recycle ratio R (recycle flow / product flow): the reactor inlet conversion is X_in = R X_out/(1 + R)
    and the balance is V = (R + 1) F_A0 int_{X_in}^{X_out} dX/(-r_A)."""

    def residual(X_out):
        X_in = R * X_out / (1 + R)
        return (R + 1) * F_A0 * integrate.quad(lambda x: 1 / rate_fn(x), X_in, X_out)[0] - V

    X_out = float(optimize.brentq(residual, 1e-12, 0.999999))
    return {"X_out": X_out, "X_in": R * X_out / (1 + R)}


# ------------------------------------------------------------------ numerical reactors with rate laws in concentrations
def batch(rate_fns, C0: dict, t, volume_fn=None) -> dict:
    """Constant-volume batch reactor for any set of species: rate_fns[species](C: dict) = net production rate
    (mol/(m3 s)). Returns concentration profiles."""
    names = list(C0)

    def rhs(_, y):
        C = dict(zip(names, y))
        return [rate_fns[s](C) if s in rate_fns else 0.0 for s in names]

    t = np.asarray(t, float)
    sol = integrate.solve_ivp(
        rhs, (t[0], t[-1]), [C0[s] for s in names], t_eval=t, method="LSODA", rtol=1e-9, atol=1e-12
    )
    return {"t": sol.t, **{s: sol.y[i] for i, s in enumerate(names)}}


def cstr_steady(rate_fns, C_in: dict, tau: float, guess: dict | None = None) -> dict:
    """Steady-state CSTR: 0 = (C_in - C)/tau + r(C)."""
    names = list(C_in)
    x0 = [(guess or C_in)[s] for s in names]

    def residual(y):
        C = dict(zip(names, y))
        return [(C_in[s] - C[s]) / tau + (rate_fns[s](C) if s in rate_fns else 0.0) for s in names]

    settled = integrate.solve_ivp(
        lambda _, y: residual(np.maximum(y, 0.0)), (0, 50 * tau), x0, method="LSODA", rtol=1e-9, atol=1e-12
    )
    sol = optimize.fsolve(residual, np.maximum(settled.y[:, -1], 0.0), xtol=1e-12)
    return dict(zip(names, np.maximum(sol, 0)))


def pfr(rate_fns, C_in: dict, tau, n: int = 200) -> dict:
    """PFR with liquid-phase (constant density) kinetics: dC/dtau = r(C) along the residence time."""
    out = batch(rate_fns, C_in, np.linspace(0, tau, n))
    out["tau"] = out.pop("t")
    return out


# ------------------------------------------------------------------ packed beds and pressure drop
def ergun(G: float, rho: float, mu: float, D_p: float, phi: float) -> float:
    """Ergun pressure gradient -dP/dz (Pa/m) for superficial mass flux G (kg/(m2 s)), fluid density rho, viscosity
    mu, particle diameter D_p and bed void fraction phi."""
    return G / (rho * D_p) * (1 - phi) / phi**3 * (150 * (1 - phi) * mu / D_p + 1.75 * G)


def ergun_alpha(beta0: float, P0: float, A_c: float, rho_c: float, phi: float) -> float:
    """alpha = 2 beta0 / (A_c rho_c (1 - phi) P0) in 1/kg of catalyst, with beta0 the inlet Ergun gradient (Pa/m)."""
    return 2 * beta0 / (A_c * rho_c * (1 - phi) * P0)


def pbr_with_pressure_drop(F_A0: float, W: float, rate_fn, alpha: float, eps: float = 0.0, n: int = 200) -> dict:
    """Isothermal packed bed with pressure drop: dX/dW = -r_A'(X, y)/F_A0, dy/dW = -alpha (1 + eps X) / (2 y),
    y = P/P0. rate_fn(X, y) is the rate per kg of catalyst including the pressure and expansion effects on
    concentration (use kinetics.concentration_from_conversion with P_ratio = y)."""

    def rhs(_, s):
        X, y = s
        return [rate_fn(X, y) / F_A0, -alpha * (1 + eps * X) / (2 * y)]

    sol = integrate.solve_ivp(rhs, (0, W), [0.0, 1.0], t_eval=np.linspace(0, W, n), rtol=1e-9, atol=1e-12)
    return {"W": sol.t, "X": sol.y[0], "y": sol.y[1]}


def pressure_ratio_no_reaction(alpha: float, W) -> float:
    """y = sqrt(1 - alpha W) for eps = 0 (the analytic solution without volume change)."""
    return np.sqrt(1 - alpha * np.asarray(W, float))
