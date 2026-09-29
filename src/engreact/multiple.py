"""Multiple reactions: series (A -> B -> C) and parallel schemes with analytic solutions (Bateman), selectivity
and yield definitions, the optimum time / space time for an intermediate, and general reaction networks
integrated in batch, PFR, CSTR and semibatch reactors from a list of reactions with rate laws.

>>> from engreact import multiple
>>> round(multiple.series_optimum_time(k1=1.0, k2=0.5), 4)          # t_max for B in A -> B -> C
1.3863
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize


# ------------------------------------------------------------------ series reactions A -> B -> C (first order)
def series_batch(k1: float, k2: float, C_A0: float, t) -> dict:
    """Bateman solution for A -> B -> C with first-order steps in a batch reactor / PFR."""
    t = np.asarray(t, float)
    A = C_A0 * np.exp(-k1 * t)
    if abs(k1 - k2) < 1e-12:
        B = C_A0 * k1 * t * np.exp(-k1 * t)
    else:
        B = C_A0 * k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    return {"t": t, "A": A, "B": B, "C": C_A0 - A - B}


def series_optimum_time(k1: float, k2: float) -> float:
    """Time (or PFR space time) maximising the intermediate B."""
    if abs(k1 - k2) < 1e-12:
        return 1 / k1
    return float(np.log(k2 / k1) / (k2 - k1))


def series_max_B(k1: float, k2: float, C_A0: float = 1.0) -> float:
    if abs(k1 - k2) < 1e-12:
        return C_A0 / np.e
    return float(C_A0 * (k1 / k2) ** (k2 / (k2 - k1)))


def series_cstr(k1: float, k2: float, C_A0: float, tau) -> dict:
    """Steady CSTR for A -> B -> C (first order)."""
    tau = np.asarray(tau, float)
    A = C_A0 / (1 + k1 * tau)
    B = C_A0 * k1 * tau / ((1 + k1 * tau) * (1 + k2 * tau))
    return {"tau": tau, "A": A, "B": B, "C": C_A0 - A - B}


def series_cstr_optimum_tau(k1: float, k2: float) -> float:
    return float(1 / np.sqrt(k1 * k2))


# ------------------------------------------------------------------ selectivity and yield
def instantaneous_selectivity(r_desired, r_undesired) -> float:
    return np.asarray(r_desired, float) / np.asarray(r_undesired, float)


def overall_selectivity(F_desired, F_undesired) -> float:
    return np.asarray(F_desired, float) / np.asarray(F_undesired, float)


def overall_yield(F_desired, F_A0: float, F_A) -> float:
    """Moles of desired product per mole of A consumed."""
    return np.asarray(F_desired, float) / (F_A0 - np.asarray(F_A, float))


def parallel_selectivity_power_law(kD: float, kU: float, aD: float, aU: float, C_A) -> float:
    """S_DU = kD C_A^(aD - aU) / kU for parallel A -> D (order aD) and A -> U (order aU)."""
    return kD / kU * np.asarray(C_A, float) ** (aD - aU)


# ------------------------------------------------------------------ general reaction networks
class Network:
    """A set of reactions: each entry is (stoich dict, rate function of the concentration dict) with the rate in
    mol/(m3 s) per unit of stoichiometric coefficient; species production rates are sum_j nu_ij r_j."""

    def __init__(self, reactions, species=None):
        self.reactions = list(reactions)
        self.species = species or sorted({s for st, _ in self.reactions for s in st})

    def production_rates(self, C: dict) -> dict:
        out = {s: 0.0 for s in self.species}
        for stoich, rate in self.reactions:
            r = rate(C)
            for s, nu in stoich.items():
                out[s] += nu * r
        return out

    def batch(self, C0: dict, t) -> dict:
        names = self.species

        def rhs(_, y):
            C = dict(zip(names, np.maximum(y, 0.0)))
            r = self.production_rates(C)
            return [r[s] for s in names]

        t = np.asarray(t, float)
        sol = integrate.solve_ivp(
            rhs, (t[0], t[-1]), [C0.get(s, 0.0) for s in names], t_eval=t, method="LSODA", rtol=1e-9, atol=1e-12
        )
        return {"t": sol.t, **{s: sol.y[i] for i, s in enumerate(names)}}

    def pfr(self, C_in: dict, tau, n: int = 200) -> dict:
        out = self.batch(C_in, np.linspace(0, tau, n))
        out["tau"] = out.pop("t")
        return out

    def cstr(self, C_in: dict, tau: float, guess: dict | None = None, settle: float = 50.0) -> dict:
        """Steady state found by integrating the transient CSTR for `settle` residence times from the guess (or the
        feed), then polishing with Newton's method on the unclipped residual."""
        names = self.species

        def residual(y):
            C = dict(zip(names, y))
            r = self.production_rates(C)
            return [(C_in.get(s, 0.0) - C[s]) / tau + r[s] for s in names]

        x0 = [(guess or C_in).get(s, 0.0) for s in names]
        sol_t = integrate.solve_ivp(
            lambda _, y: residual(np.maximum(y, 0.0)), (0, settle * tau), x0, method="LSODA", rtol=1e-9, atol=1e-12
        )
        polished = optimize.fsolve(residual, np.maximum(sol_t.y[:, -1], 0.0), xtol=1e-12)
        return dict(zip(names, np.maximum(polished, 0.0)))

    def semibatch(self, C0: dict, V0: float, feed_C: dict, v_in: float, t) -> dict:
        """Semibatch with a constant volumetric feed v_in of composition feed_C: d(V C_i)/dt = v_in C_i,in + r_i V."""
        names = self.species

        def rhs(_, y):
            V = y[-1]
            N = y[:-1]
            C = dict(zip(names, np.maximum(N / V, 0.0)))
            r = self.production_rates(C)
            return [v_in * feed_C.get(s, 0.0) + r[s] * V for s in names] + [v_in]

        t = np.asarray(t, float)
        sol = integrate.solve_ivp(
            rhs,
            (t[0], t[-1]),
            [C0.get(s, 0.0) * V0 for s in names] + [V0],
            t_eval=t,
            method="LSODA",
            rtol=1e-9,
            atol=1e-12,
        )
        V = sol.y[-1]
        return {"t": sol.t, "V": V, **{s: sol.y[i] / V for i, s in enumerate(names)}}
