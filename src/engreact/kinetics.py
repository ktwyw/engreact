"""Rate laws and stoichiometry: Arrhenius temperature dependence and its fitting, power-law and reversible rate
laws, the stoichiometric table (concentrations as functions of conversion for liquids and for gases with
volume change, pressure and temperature), the expansion factor epsilon, and equilibrium conversion.

>>> from engreact import kinetics
>>> kinetics.epsilon(y_A0=0.5, delta=1)           # A -> 2B in a 50 % inert feed: volume grows by 50 % at X = 1
0.5
>>> round(kinetics.rate_ratio(80e3, 300, 350), 1)          # how much faster at 350 K than at 300 K
97.7
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .constants import R


# ------------------------------------------------------------------ temperature dependence
def arrhenius(k0: float, Ea: float, T) -> float:
    """k = k0 exp(-Ea / R T); Ea in J/mol."""
    return k0 * np.exp(-Ea / (R * np.asarray(T, float)))


def arrhenius_from_two(k1: float, T1: float, k2: float, T2: float) -> dict:
    Ea = R * np.log(k2 / k1) / (1 / T1 - 1 / T2)
    return {"Ea": float(Ea), "k0": float(k1 * np.exp(Ea / (R * T1)))}


def fit_arrhenius(T, k) -> dict:
    """Least squares of ln k against 1/T: k0, Ea (J/mol) with standard errors, and r^2."""
    T, k = np.asarray(T, float), np.asarray(k, float)
    x, y = 1 / T, np.log(k)
    A = np.vstack([np.ones_like(x), x]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ coef
    n, p = len(y), 2
    s2 = np.sum((y - fit) ** 2) / max(n - p, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2)
    return {
        "k0": float(np.exp(coef[0])),
        "Ea": float(-R * coef[1]),
        "se_Ea": float(R * se[1]),
        "se_lnk0": float(se[0]),
        "r2": float(r2),
    }


def rate_ratio(Ea: float, T1: float, T2: float) -> float:
    """k(T2)/k(T1)."""
    return float(np.exp(-Ea / R * (1 / T2 - 1 / T1)))


def temperature_for_rate_ratio(Ea: float, T1: float, ratio: float) -> float:
    """T2 at which k(T2)/k(T1) = ratio."""
    return 1 / (1 / T1 - R * np.log(ratio) / Ea)


# ------------------------------------------------------------------ rate laws
def power_law(k: float, concentrations, orders) -> float:
    """-r = k prod C_i^n_i."""
    out = k
    for C, n in zip(concentrations, orders):
        out = out * np.asarray(C, float) ** n
    return out


def reversible_rate(kf: float, Kc: float, C_react, C_prod, nu_react=None, nu_prod=None) -> float:
    """Net rate of an elementary reversible reaction: r = kf (prod C_react^|nu| - prod C_prod^nu / Kc)."""
    nu_react = [1] * len(C_react) if nu_react is None else nu_react
    nu_prod = [1] * len(C_prod) if nu_prod is None else nu_prod
    fwd = np.prod([np.asarray(C, float) ** n for C, n in zip(C_react, nu_react)], axis=0)
    rev = np.prod([np.asarray(C, float) ** n for C, n in zip(C_prod, nu_prod)], axis=0)
    return kf * (fwd - rev / Kc)


# ------------------------------------------------------------------ stoichiometry
@dataclass
class Reaction:
    """stoich: {species: nu} with nu < 0 for reactants; key: the limiting reactant on which conversion is based."""

    stoich: dict
    key: str

    def nu(self, species: str) -> float:
        return self.stoich.get(species, 0.0)

    @property
    def delta(self) -> float:
        """Change in total moles per mole of key reacted: sum(nu_i)/|nu_key|."""
        return sum(self.stoich.values()) / abs(self.stoich[self.key])

    def __repr__(self):
        lhs = " + ".join(f"{-v:g} {s}" if v != -1 else s for s, v in self.stoich.items() if v < 0)
        rhs = " + ".join(f"{v:g} {s}" if v != 1 else s for s, v in self.stoich.items() if v > 0)
        return f"{lhs} -> {rhs}"


def epsilon(y_A0: float, delta: float) -> float:
    """Expansion factor: fractional change in total moles at complete conversion of A (feed mole fraction y_A0)."""
    return y_A0 * delta


def stoichiometric_table(
    rxn: Reaction, feed: dict, X, phase: str = "liquid", C_A0: float = 1.0, P_ratio: float = 1.0, T_ratio: float = 1.0
) -> dict:
    """Concentrations of every species at conversion X of the key reactant.
    feed: {species: moles or molar flow} (Theta_i = feed_i/feed_key); inerts may be included.
    Liquid: C_i = C_A0 (Theta_i + nu_i X / |nu_A|).
    Gas:    C_i = C_A0 (Theta_i + nu_i X / |nu_A|) / (1 + eps X) * (P/P0) (T0/T)."""
    X = np.asarray(X, float)
    key = rxn.key
    nuA = abs(rxn.stoich[key])
    F0 = feed[key]
    theta = {s: feed.get(s, 0.0) / F0 for s in set(feed) | set(rxn.stoich)}
    total = sum(feed.values())
    eps = epsilon(F0 / total, rxn.delta) if phase == "gas" else 0.0
    out = {}
    for s in theta:
        moles = theta[s] + rxn.nu(s) * X / nuA
        out[s] = C_A0 * moles / (1 + eps * X) * (P_ratio / T_ratio if phase == "gas" else 1.0)
    return {"C": out, "epsilon": eps, "theta": theta}


def concentration_from_conversion(
    C_A0: float, X, eps: float = 0.0, P_ratio: float = 1.0, T_ratio: float = 1.0
) -> float:
    """C_A = C_A0 (1 - X)/(1 + eps X) (P/P0)(T0/T)."""
    X = np.asarray(X, float)
    return C_A0 * (1 - X) / (1 + eps * X) * P_ratio / T_ratio


def conversion_from_concentration(C_A0: float, C_A, eps: float = 0.0) -> float:
    C_A = np.asarray(C_A, float)
    return (C_A0 - C_A) / (C_A0 + eps * C_A)


def limiting_reactant(rxn: Reaction, feed: dict) -> str:
    return min((s for s, nu in rxn.stoich.items() if nu < 0), key=lambda s: feed[s] / abs(rxn.stoich[s]))


# ------------------------------------------------------------------ equilibrium conversion
def equilibrium_conversion(rxn: Reaction, feed: dict, Kc: float, C_A0: float, phase: str = "liquid") -> float:
    """Solve prod C_i^nu_i = Kc for the conversion of the key reactant (Kc in the concentration units of C_A0)."""
    if Kc <= 0:
        raise ValueError("Kc must be positive.")

    def residual(X):
        C = stoichiometric_table(rxn, feed, X, phase, C_A0)["C"]
        q = 1.0
        for s, nu in rxn.stoich.items():
            q *= max(C[s], 1e-300) ** nu
        return np.log(q) - np.log(Kc)

    X_max = min(
        1.0, min(feed[s] / abs(nu) for s, nu in rxn.stoich.items() if nu < 0) / feed[rxn.key] * abs(rxn.stoich[rxn.key])
    )
    return float(optimize.brentq(residual, 1e-12, X_max * (1 - 1e-12)))
