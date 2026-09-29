"""Energy balances for reactors: adiabatic and heat-exchange PFR/PBR (conversion and temperature profiles),
the CSTR heat-generation and heat-removal curves with multiple steady states and their stability, the
unsteady CSTR (start-up, runaway, extinction), a batch reactor with heat exchange, and Semenov's criterion
for thermal explosion.

Rates are given as functions r(X, T) (mol/(m3 s), positive for consumption of A) unless stated otherwise;
heats of reaction dH are per mole of A consumed (J/mol, negative for exothermic).

>>> from engreact import energy
>>> ss = energy.cstr_steady_states(k0=1e10, Ea=75e3, dH=-250e3, C_A0=2000.0, T0=300.0, tau=10.0, cp_vol=4e6)
>>> [round(T) for T in ss["T"]], ss["stable"]                   # extinguished, unstable middle, ignited
([301, 351, 423], [True, False, True])
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

from .constants import R
from .kinetics import arrhenius


# ------------------------------------------------------------------ plug flow
def adiabatic_pfr(
    rate_XT, F_A0: float, V: float, T0: float, dH: float, sum_theta_cp: float, delta_cp: float = 0.0, n: int = 300
) -> dict:
    """Adiabatic PFR: dX/dV = -r_A(X, T)/F_A0 with T = T0 + X (-dH)/(sum_theta_cp + X delta_cp) (energy balance,
    dH per mole of A and cp sums per mole of A fed)."""

    def T_of(X):
        return T0 + X * (-dH) / (sum_theta_cp + X * delta_cp)

    sol = integrate.solve_ivp(
        lambda v, s: [rate_XT(s[0], T_of(s[0])) / F_A0],
        (0, V),
        [0.0],
        t_eval=np.linspace(0, V, n),
        method="LSODA",
        rtol=1e-9,
        atol=1e-12,
    )
    return {"V": sol.t, "X": sol.y[0], "T": T_of(sol.y[0])}


def pfr_heat_exchange(
    rate_XT,
    F_A0: float,
    V: float,
    T0: float,
    dH: float,
    sum_F_cp: float,
    Ua: float,
    T_a: float,
    delta_cp: float = 0.0,
    n: int = 300,
) -> dict:
    """PFR with heat exchange (constant coolant T_a): dX/dV = -r/F_A0,
    dT/dV = [Ua (T_a - T) + (-r)(-dH(T))] / (F_A0 (sum_theta_cp + X delta_cp)),
    with sum_F_cp = F_A0 sum theta_i cp_i."""

    def rhs(_, s):
        X, T = s
        r = rate_XT(X, T)
        dHT = dH + delta_cp * (T - T0)
        return [r / F_A0, (Ua * (T_a - T) + r * (-dHT)) / (sum_F_cp + F_A0 * X * delta_cp)]

    sol = integrate.solve_ivp(
        rhs, (0, V), [0.0, T0], t_eval=np.linspace(0, V, n), method="LSODA", rtol=1e-9, atol=1e-12
    )
    return {"V": sol.t, "X": sol.y[0], "T": sol.y[1]}


def equilibrium_line(K_fn, T, C_A0: float = 1.0) -> np.ndarray:
    """Equilibrium conversion X_e(T) for A <-> B with Kc(T) given by K_fn: X_e = K/(1 + K)."""
    K = np.asarray([K_fn(t) for t in np.atleast_1d(T)], float)
    return K / (1 + K)


# ------------------------------------------------------------------ CSTR steady states
def cstr_heat_curves(
    k0: float,
    Ea: float,
    dH: float,
    C_A0: float,
    T0: float,
    tau: float,
    cp_vol: float,
    T,
    UA_over_v0cp: float = 0.0,
    T_a: float = 300.0,
) -> dict:
    """First-order exothermic reaction in a CSTR: at each T, conversion X = k tau/(1 + k tau); heat generated
    G(T) = (-dH) X (per unit volume feed, J/m3); heat removed R(T) = cp_vol (1 + kappa)(T - T_c), with
    kappa = UA/(v0 cp_vol) and T_c = (T0 + kappa T_a)/(1 + kappa). Steady states are where G = R."""
    T = np.asarray(T, float)
    k = arrhenius(k0, Ea, T)
    X = k * tau / (1 + k * tau)
    G = (-dH) * C_A0 * X
    kappa = UA_over_v0cp
    T_c = (T0 + kappa * T_a) / (1 + kappa)
    Rm = cp_vol * (1 + kappa) * (T - T_c)
    return {"T": T, "X": X, "G": G, "R": Rm, "T_c": T_c}


def cstr_steady_states(
    k0: float,
    Ea: float,
    dH: float,
    C_A0: float,
    T0: float,
    tau: float,
    cp_vol: float,
    UA_over_v0cp: float = 0.0,
    T_a: float = 300.0,
    T_range=(250.0, 900.0),
) -> dict:
    """All steady states (G = R) with their conversions and a stability verdict from the slope condition
    dG/dT < dR/dT (necessary for stability of the first-order CSTR)."""
    T = np.linspace(T_range[0], T_range[1], 4000)
    c = cstr_heat_curves(k0, Ea, dH, C_A0, T0, tau, cp_vol, T, UA_over_v0cp, T_a)
    f = c["G"] - c["R"]
    roots = []
    for i in np.nonzero(np.sign(f[:-1]) != np.sign(f[1:]))[0]:
        fn = lambda t: float(
            cstr_heat_curves(k0, Ea, dH, C_A0, T0, tau, cp_vol, t, UA_over_v0cp, T_a)["G"]
            - cstr_heat_curves(k0, Ea, dH, C_A0, T0, tau, cp_vol, t, UA_over_v0cp, T_a)["R"]
        )  # noqa: E731
        roots.append(float(optimize.brentq(fn, T[i], T[i + 1])))
    X = [float(cstr_heat_curves(k0, Ea, dH, C_A0, T0, tau, cp_vol, t, UA_over_v0cp, T_a)["X"]) for t in roots]
    dG = [
        float(
            (
                cstr_heat_curves(k0, Ea, dH, C_A0, T0, tau, cp_vol, t + 0.01, UA_over_v0cp, T_a)["G"]
                - cstr_heat_curves(k0, Ea, dH, C_A0, T0, tau, cp_vol, t - 0.01, UA_over_v0cp, T_a)["G"]
            )
            / 0.02
        )
        for t in roots
    ]
    dR = cp_vol * (1 + UA_over_v0cp)
    return {"T": roots, "X": X, "stable": [g < dR for g in dG]}


def ignition_extinction(
    k0: float,
    Ea: float,
    dH: float,
    C_A0: float,
    tau: float,
    cp_vol: float,
    T0_range,
    UA_over_v0cp: float = 0.0,
    T_a: float = 300.0,
) -> dict:
    """Steady-state temperatures against feed temperature: the hysteresis (ignition-extinction) diagram."""
    rows = []
    for T0 in np.atleast_1d(T0_range):
        ss = cstr_steady_states(k0, Ea, dH, C_A0, T0, tau, cp_vol, UA_over_v0cp, T_a)
        rows.append((float(T0), ss["T"], ss["stable"]))
    return {"T0": [r[0] for r in rows], "T_ss": [r[1] for r in rows], "stable": [r[2] for r in rows]}


# ------------------------------------------------------------------ unsteady CSTR and batch
def cstr_unsteady(
    k0: float,
    Ea: float,
    dH: float,
    C_A0: float,
    T0: float,
    tau: float,
    cp_vol: float,
    t,
    C_A_init: float,
    T_init: float,
    UA_over_Vcp: float = 0.0,
    T_a: float = 300.0,
    order: float = 1.0,
) -> dict:
    """Transient CSTR (constant volume, constant properties): dC_A/dt = (C_A0 - C_A)/tau - k C_A^n,
    dT/dt = (T0 - T)/tau + (-dH) k C_A^n / cp_vol - UA/(V cp_vol) (T - T_a)."""

    def rhs(_, s):
        C, T = s
        r = arrhenius(k0, Ea, T) * max(C, 0.0) ** order
        return [(C_A0 - C) / tau - r, (T0 - T) / tau + (-dH) * r / cp_vol - UA_over_Vcp * (T - T_a)]

    t = np.asarray(t, float)
    sol = integrate.solve_ivp(rhs, (t[0], t[-1]), [C_A_init, T_init], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10)
    return {"t": sol.t, "C_A": sol.y[0], "T": sol.y[1]}


def batch_with_heat_exchange(
    k0: float,
    Ea: float,
    dH: float,
    C_A0: float,
    T_init: float,
    cp_vol: float,
    UA_over_V: float,
    T_a: float,
    t,
    order: float = 1.0,
) -> dict:
    """Batch: dC/dt = -k C^n, cp_vol dT/dt = (-dH) k C^n - (UA/V)(T - T_a). Returns profiles and the maximum T."""

    def rhs(_, s):
        C, T = s
        r = arrhenius(k0, Ea, T) * max(C, 0.0) ** order
        return [-r, ((-dH) * r - UA_over_V * (T - T_a)) / cp_vol]

    t = np.asarray(t, float)
    sol = integrate.solve_ivp(rhs, (t[0], t[-1]), [C_A0, T_init], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10)
    return {"t": sol.t, "C_A": sol.y[0], "T": sol.y[1], "T_max": float(sol.y[1].max())}


def adiabatic_temperature_rise(dH: float, C_A0: float, cp_vol: float) -> float:
    """Delta T_ad = (-dH) C_A0 / cp_vol."""
    return (-dH) * C_A0 / cp_vol


def semenov_critical(dH: float, C_A0: float, k0: float, Ea: float, T_a: float, cp_vol: float = 1.0) -> dict:
    """Semenov's criterion for a zero-order-like reaction in a well-stirred vessel: runaway if
    (-dH) C_A0 k(T_a) Ea / (R T_a^2) > (UA/V) / e. Returns the critical UA/V and the critical temperature rise
    R T_a^2 / Ea."""
    q = (-dH) * C_A0 * arrhenius(k0, Ea, T_a)
    UA_V_crit = q * Ea / (R * T_a**2) * np.e
    return {"UA_over_V_critical": float(UA_V_crit), "dT_critical": float(R * T_a**2 / Ea)}
