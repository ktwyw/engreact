"""Heterogeneous catalysis: Langmuir adsorption and LHHW rate laws, internal diffusion (Thiele modulus and
effectiveness factor for slab, cylinder and sphere; the Weisz-Prater criterion), external mass transfer
(Mears criterion, overall effectiveness), the falsified kinetics of the strong-diffusion regime, and catalyst
deactivation laws.

>>> from engreact import catalysis
>>> round(catalysis.effectiveness_factor(3.0, "sphere"), 4)       # eta = (3/phi^2)(phi coth phi - 1)
0.6716
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

from ._common import trapezoid


# ------------------------------------------------------------------ adsorption and LHHW
def langmuir_coverage(K: float, p) -> float:
    """theta = K p / (1 + K p) for single-site adsorption."""
    p = np.asarray(p, float)
    return K * p / (1 + K * p)


def langmuir_dissociative(K: float, p) -> float:
    """theta = sqrt(K p)/(1 + sqrt(K p)) for dissociative adsorption of a diatomic."""
    x = np.sqrt(K * np.asarray(p, float))
    return x / (1 + x)


def langmuir_competitive(Ks: dict, ps: dict) -> dict:
    denom = 1 + sum(Ks[s] * ps[s] for s in Ks)
    return {s: Ks[s] * ps[s] / denom for s in Ks}


def lhhw_rate(
    k: float,
    p_A,
    p_B=None,
    K_A: float = 0.0,
    K_B: float = 0.0,
    K_P: float = 0.0,
    p_P=0.0,
    mode: str = "surface",
    Kp: float | None = None,
) -> float:
    """Langmuir-Hinshelwood-Hougen-Watson rates.
    mode 'surface' (dual-site surface reaction limiting, A + B -> P):
        r = k K_A K_B p_A p_B (1 - p_P/(Kp p_A p_B)) / (1 + K_A p_A + K_B p_B + K_P p_P)^2
    mode 'adsorption' (adsorption of A limiting, A -> P):
        r = k p_A (1 - p_P/(Kp p_A)) / (1 + K_P p_P + K_A p_P/Kp)
    mode 'single' (single-site, A -> P): r = k K_A p_A / (1 + K_A p_A + K_P p_P)."""
    p_A = np.asarray(p_A, float)
    if mode == "surface":
        rev = 1 - (p_P / (Kp * p_A * p_B) if Kp else 0.0)
        return k * K_A * K_B * p_A * p_B * rev / (1 + K_A * p_A + K_B * p_B + K_P * p_P) ** 2
    if mode == "adsorption":
        rev = 1 - (p_P / (Kp * p_A) if Kp else 0.0)
        return k * p_A * rev / (1 + K_P * p_P + (K_A * p_P / Kp if Kp else 0.0))
    return k * K_A * p_A / (1 + K_A * p_A + K_P * p_P)


def fit_lhhw_single(p_A, rates) -> dict:
    """Fit r = a p_A/(1 + b p_A) by nonlinear least squares (initial guess from the linearised 1/r vs 1/p plot)."""
    p_A, rates = np.asarray(p_A, float), np.asarray(rates, float)
    slope, intercept = np.polyfit(1 / p_A, 1 / rates, 1)
    a0, b0 = 1 / slope, intercept / slope
    res = optimize.least_squares(lambda q: q[0] * p_A / (1 + q[1] * p_A) - rates, [a0, b0])
    return {"k_K": float(res.x[0]), "K": float(res.x[1]), "rss": float(np.sum(res.fun**2))}


# ------------------------------------------------------------------ internal diffusion
def thiele_modulus(k: float, D_e: float, L: float, C_s: float = 1.0, order: float = 1) -> float:
    """Generalised Thiele modulus phi = L sqrt((n+1)/2 k C_s^(n-1) / D_e) with L the characteristic length
    (volume/external area: half-thickness for a slab, R/2 for a cylinder, R/3 for a sphere)."""
    return float(L * np.sqrt((order + 1) / 2 * k * C_s ** (order - 1) / D_e))


def effectiveness_factor(phi, geometry: str = "sphere") -> float:
    """First-order effectiveness factor. Sphere: phi is defined with R (eta = 3/phi^2 (phi coth phi - 1));
    slab: phi with the half-thickness (eta = tanh(phi)/phi); cylinder: phi with R (eta = 2 I1(phi)/(phi I0(phi)))."""
    scalar = np.ndim(phi) == 0
    phi = np.atleast_1d(np.asarray(phi, float))
    if geometry == "slab":
        out = np.where(phi < 1e-8, 1.0, np.tanh(phi) / np.where(phi < 1e-8, 1.0, phi))
    elif geometry == "cylinder":
        from scipy.special import i0, i1

        out = np.where(phi < 1e-8, 1.0, 2 * i1(phi) / (np.where(phi < 1e-8, 1.0, phi) * i0(phi)))
    else:
        small = phi < 1e-3
        safe = np.where(small, 1.0, phi)
        out = np.where(small, 1 - phi**2 / 15, 3 / safe**2 * (safe / np.tanh(safe) - 1))
    return float(out[0]) if scalar else out


def concentration_profile_sphere(phi: float, r_over_R) -> float:
    """C/C_s inside a spherical catalyst pellet, first order: (R/r) sinh(phi r/R)/sinh(phi)."""
    x = np.asarray(r_over_R, float)
    return np.where(x < 1e-9, phi / np.sinh(phi), np.sinh(phi * x) / (np.where(x < 1e-9, 1.0, x) * np.sinh(phi)))


def weisz_prater(r_obs: float, rho_c: float, R: float, D_e: float, C_s: float) -> float:
    """C_WP = -r_obs rho_c R^2 / (D_e C_s): internal diffusion limitations are negligible when C_WP << 1."""
    return r_obs * rho_c * R**2 / (D_e * C_s)


def falsified_kinetics(true_order: float, true_Ea: float) -> dict:
    """In the strong-diffusion regime (eta ~ 1/phi) the apparent order is (n + 1)/2 and the apparent activation
    energy (Ea + E_diffusion)/2 ~ Ea/2."""
    return {"apparent_order": (true_order + 1) / 2, "apparent_Ea": true_Ea / 2}


def effectiveness_numerical(phi: float, order: float = 1.0, geometry: str = "sphere") -> float:
    """Effectiveness factor for an nth-order reaction by solving the pellet BVP (dimensionless, phi with R):
    (1/x^s) d/dx (x^s dy/dx) = phi^2 y^n, y(1) = 1, dy/dx(0) = 0, with s = 0, 1, 2 for slab, cylinder, sphere."""
    s = {"slab": 0, "cylinder": 1, "sphere": 2}[geometry]

    def rhs(x, y):
        y0 = np.maximum(y[0], 0.0)
        return np.vstack(
            [
                y[1],
                phi**2 * y0**order
                - np.where(x > 1e-9, s * y[1] / np.where(x > 1e-9, x, 1.0), s * phi**2 * y0**order / (s + 1)),
            ]
        )

    x = np.linspace(0, 1, 400)
    guess = np.vstack([np.cosh(phi * x) / np.cosh(phi), phi * np.sinh(phi * x) / np.cosh(phi)])
    sol = integrate.solve_bvp(rhs, lambda ya, yb: [ya[1], yb[0] - 1], x, guess, tol=1e-8, max_nodes=100000)
    xx = np.linspace(0, 1, 2000)
    y = np.maximum(sol.sol(xx)[0], 0.0)
    return float((s + 1) * trapezoid(y**order * xx**s, xx))


# ------------------------------------------------------------------ external mass transfer
def sherwood_frossling(Re: float, Sc: float) -> float:
    """Sh = 2 + 0.6 Re^1/2 Sc^1/3 for flow past a sphere."""
    return 2 + 0.6 * Re**0.5 * Sc ** (1 / 3)


def mears_criterion(r_obs: float, rho_b: float, R: float, order: float, k_c: float, C_b: float) -> float:
    """Mears: external mass transfer is negligible when -r_obs rho_b R n / (k_c C_b) < 0.15."""
    return r_obs * rho_b * R * order / (k_c * C_b)


def overall_effectiveness(eta: float, k: float, k_c: float, a_c: float) -> float:
    """Omega = eta / (1 + eta k / (k_c a_c)) for a first-order reaction with external resistance."""
    return eta / (1 + eta * k / (k_c * a_c))


def external_limited_conversion(k_c: float, a_c: float, U: float, L) -> float:
    """Mass-transfer-limited packed bed: X = 1 - exp(-k_c a_c L / U)."""
    return 1 - np.exp(-k_c * a_c * np.asarray(L, float) / U)


# ------------------------------------------------------------------ deactivation
def activity(t, k_d: float, law: str = "first", n: float = 2.0) -> float:
    """Activity a(t): 'first' a = exp(-k_d t); 'second' a = 1/(1 + k_d t); 'nth' a = (1 + (n-1) k_d t)^(-1/(n-1));
    'linear' a = 1 - k_d t (clipped at 0)."""
    t = np.asarray(t, float)
    if law == "first":
        return np.exp(-k_d * t)
    if law == "second":
        return 1 / (1 + k_d * t)
    if law == "nth":
        return (1 + (n - 1) * k_d * t) ** (-1 / (n - 1))
    return np.maximum(1 - k_d * t, 0.0)


def fit_deactivation(t, a, law: str = "first") -> dict:
    t, a = np.asarray(t, float), np.asarray(a, float)
    res = optimize.least_squares(lambda q: activity(t, q[0], law) - a, [0.01], bounds=([0], [np.inf]))
    return {"k_d": float(res.x[0]), "rss": float(np.sum(res.fun**2))}


def temperature_time_trajectory(
    k_d0: float, E_d: float, k0: float, Ea: float, T0: float, t_max: float, T_max: float = 1e4, n: int = 400
) -> dict:
    """Raise the temperature to keep k(T) a(t) = k(T0) as the catalyst decays (first-order decay with an Arrhenius
    decay constant). The required temperature is explicit, 1/T = 1/T0 + (R/Ea) ln a, and the activity follows
    da/dt = -k_d(T(a)) a, integrated until t_max or until T reaches T_max (the trajectory then stops: the catalyst
    must be regenerated)."""
    from .constants import R
    from .kinetics import arrhenius

    def T_of(a):
        return 1.0 / (1.0 / T0 + R * np.log(max(a, 1e-300)) / Ea)

    def rhs(_, y):
        return [-arrhenius(k_d0, E_d, T_of(y[0])) * y[0]]

    def hit_Tmax(_, y):
        return T_of(y[0]) - T_max

    hit_Tmax.terminal = True
    sol = integrate.solve_ivp(
        rhs, (0, t_max), [1.0], t_eval=np.linspace(0, t_max, n), events=hit_Tmax, method="LSODA", rtol=1e-9, atol=1e-12
    )
    t_out, a = sol.t, sol.y[0]
    hit = bool(sol.t_events[0].size)
    if hit:  # include the stopping point itself
        t_out = np.append(t_out, sol.t_events[0][0])
        a = np.append(a, sol.y_events[0][0][0])
    return {"t": t_out, "a": a, "T": np.array([T_of(x) for x in a]), "reached_T_max": hit}
