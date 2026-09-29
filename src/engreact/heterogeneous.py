"""Non-catalytic gas-solid reactions: the shrinking-core model for a spherical particle with film diffusion,
ash-layer diffusion or chemical reaction controlling (Levenspiel, Ch. 25), the time for complete conversion
in each regime, the conversion-time relations, and the combined-resistance model.

>>> from engreact import heterogeneous
>>> round(heterogeneous.shrinking_core_time_fraction(0.5, "reaction"), 4)     # t/tau for X = 0.5, reaction control
0.2063
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


def shrinking_core_time_fraction(X, control: str) -> float:
    """t/tau_complete as a function of solid conversion X for a shrinking-core sphere:
    'film': X; 'ash': 1 - 3(1-X)^(2/3) + 2(1-X); 'reaction': 1 - (1-X)^(1/3)."""
    scalar = np.ndim(X) == 0
    X = np.asarray(X, float)
    if control == "film":
        out = X
    elif control == "ash":
        out = 1 - 3 * (1 - X) ** (2 / 3) + 2 * (1 - X)
    else:
        out = 1 - (1 - X) ** (1 / 3)
    return float(out) if scalar else out


def time_for_complete_conversion(
    control: str, rho_B: float, R: float, b: float, C_Ag: float, k_g: float = 1.0, D_e: float = 1.0, k_s: float = 1.0
) -> float:
    """tau for a sphere: film: rho_B R/(3 b k_g C_Ag); ash: rho_B R^2/(6 b D_e C_Ag); reaction: rho_B R/(b k_s C_Ag)
    (rho_B moles of solid per volume, b the stoichiometric coefficient of the solid per mole of gas A)."""
    if control == "film":
        return rho_B * R / (3 * b * k_g * C_Ag)
    if control == "ash":
        return rho_B * R**2 / (6 * b * D_e * C_Ag)
    return rho_B * R / (b * k_s * C_Ag)


def _invert_ash(f: float) -> float:
    return float(optimize.brentq(lambda X: shrinking_core_time_fraction(X, "ash") - f, 0.0, 1.0))


def conversion_at_time(t, tau: float, control: str) -> float:
    """Invert the conversion-time relation."""
    t = np.atleast_1d(np.asarray(t, float))
    out = []
    for tt in t:
        f = tt / tau
        if f >= 1:
            out.append(1.0)
        elif control == "film":
            out.append(f)
        elif control == "reaction":
            out.append(1 - (1 - f) ** 3)
        else:
            out.append(_invert_ash(f))
    return np.array(out) if len(out) > 1 else float(out[0])


def combined_resistances(X, tau_film: float, tau_ash: float, tau_reaction: float) -> float:
    """Time to conversion X when the three resistances act in series: t = sum tau_i f_i(X)."""
    return (
        tau_film * shrinking_core_time_fraction(X, "film")
        + tau_ash * shrinking_core_time_fraction(X, "ash")
        + tau_reaction * shrinking_core_time_fraction(X, "reaction")
    )


def identify_regime(t, X) -> dict:
    """Which single control regime fits conversion-time data best (least squares on t/tau with tau fitted)."""
    t, X = np.asarray(t, float), np.asarray(X, float)
    out = {}
    for control in ("film", "ash", "reaction"):
        f = shrinking_core_time_fraction(X, control)
        tau = float(np.sum(t * f) / np.sum(f * f))
        out[control] = {"tau": tau, "rss": float(np.sum((t - tau * f) ** 2))}
    best = min(out, key=lambda c: out[c]["rss"])
    return {"best": best, "fits": out}


def regime_size_dependence(control: str) -> str:
    """How tau scales with particle radius: film R (R^1.5-2 for small, Stokes-regime particles), ash R^2, reaction R."""
    return {"film": "R^1 (large particles) to R^2 (small, Stokes)", "ash": "R^2", "reaction": "R^1"}[control]
