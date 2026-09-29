"""A small self-contained thermochemistry: standard enthalpies and Gibbs energies of formation and heat-capacity
polynomials for common species, heats of reaction as a function of temperature, equilibrium constants by
van 't Hoff (with the full Cp integration), and adiabatic temperature rise.

Species data (298.15 K, ideal gas unless noted; Cp = a + b T + c T^2 + d T^3 in J/(mol K), T in K, valid
roughly 298-1500 K) are from the NIST Chemistry WebBook and Perry's Chemical Engineers' Handbook (Table 2-153
style polynomials, refitted); the Cp fits reproduce the tabulated values within about 1 % over the range.

>>> from engreact import thermo
>>> round(thermo.heat_of_reaction({"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}) / 1e3, 1)   # water-gas shift, 298 K
-41.2
"""

from __future__ import annotations

import math

import numpy as np

from .constants import T_STD, R

# name: (dHf298 J/mol, dGf298 J/mol, (a, b, c, d) for Cp J/(mol K), phase)
SPECIES: dict = {
    "H2": (0.0, 0.0, (29.11, -0.1916e-2, 0.4003e-5, -0.8704e-9), "g"),
    "O2": (0.0, 0.0, (25.48, 1.520e-2, -0.7155e-5, 1.312e-9), "g"),
    "N2": (0.0, 0.0, (28.90, -0.1571e-2, 0.8081e-5, -2.873e-9), "g"),
    "CO": (-110530.0, -137170.0, (28.16, 0.1675e-2, 0.5372e-5, -2.222e-9), "g"),
    "CO2": (-393520.0, -394360.0, (22.26, 5.981e-2, -3.501e-5, 7.469e-9), "g"),
    "H2O": (-241820.0, -228570.0, (32.24, 0.1923e-2, 1.055e-5, -3.595e-9), "g"),
    "H2O(l)": (-285830.0, -237130.0, (75.4, 0.0, 0.0, 0.0), "l"),
    "CH4": (-74850.0, -50790.0, (19.89, 5.024e-2, 1.269e-5, -11.01e-9), "g"),
    "C2H4": (52280.0, 68120.0, (3.95, 15.64e-2, -8.344e-5, 17.67e-9), "g"),
    "C2H6": (-84680.0, -32890.0, (6.900, 17.27e-2, -6.406e-5, 7.285e-9), "g"),
    "C3H8": (-103850.0, -23490.0, (-4.04, 30.48e-2, -15.72e-5, 31.74e-9), "g"),
    "NH3": (-46110.0, -16450.0, (27.31, 2.383e-2, 1.707e-5, -11.85e-9), "g"),
    "NO": (90250.0, 86550.0, (29.34, -0.09395e-2, 0.9747e-5, -4.187e-9), "g"),
    "NO2": (33180.0, 51310.0, (22.93, 5.711e-2, -3.520e-5, 7.870e-9), "g"),
    "SO2": (-296830.0, -300190.0, (25.78, 5.795e-2, -3.812e-5, 8.612e-9), "g"),
    "SO3": (-395720.0, -371060.0, (16.40, 14.58e-2, -11.20e-5, 32.42e-9), "g"),
    "CH3OH": (-200660.0, -161960.0, (21.15, 7.092e-2, 2.587e-5, -28.52e-9), "g"),
    "C2H5OH": (-235100.0, -168490.0, (9.014, 21.41e-2, -8.390e-5, 1.373e-9), "g"),
    "CH3CHO": (-166190.0, -128860.0, (7.716, 18.22e-2, -10.07e-5, 23.80e-9), "g"),
    "C2H4O": (-52630.0, -13010.0, (-7.519, 22.22e-2, -12.56e-5, 25.92e-9), "g"),  # ethylene oxide
    "C4H8": (-0.13e3, 71300.0, (-2.994, 35.32e-2, -19.89e-5, 44.63e-9), "g"),  # 1-butene
    "C4H6": (110160.0, 150670.0, (-1.687, 34.19e-2, -23.40e-5, 63.35e-9), "g"),  # 1,3-butadiene
    "C6H6": (82930.0, 129650.0, (-36.22, 48.48e-2, -31.57e-5, 77.62e-9), "g"),
    "Cl2": (0.0, 0.0, (26.93, 3.384e-2, -3.869e-5, 15.47e-9), "g"),
    "HCl": (-92310.0, -95300.0, (30.67, -0.7201e-2, 1.246e-5, -3.898e-9), "g"),
}


def cp(species: str, T) -> float:
    """Heat capacity J/(mol K) at T (K)."""
    a, b, c, d = SPECIES[species][2]
    T = np.asarray(T, float)
    return a + b * T + c * T**2 + d * T**3


def enthalpy_change(species: str, T1: float, T2: float) -> float:
    """Integral of Cp from T1 to T2 (J/mol)."""
    a, b, c, d = SPECIES[species][2]
    f = lambda T: a * T + b * T**2 / 2 + c * T**3 / 3 + d * T**4 / 4  # noqa: E731
    return f(T2) - f(T1)


def entropy_change(species: str, T1: float, T2: float) -> float:
    a, b, c, d = SPECIES[species][2]
    f = lambda T: a * math.log(T) + b * T + c * T**2 / 2 + d * T**3 / 3  # noqa: E731
    return f(T2) - f(T1)


def heat_of_reaction(stoich: dict, T: float = T_STD) -> float:
    """Delta H_rxn(T) in J per mole of reaction: stoich = {species: nu}, nu < 0 for reactants."""
    dH = sum(nu * SPECIES[s][0] for s, nu in stoich.items())
    return dH + sum(nu * enthalpy_change(s, T_STD, T) for s, nu in stoich.items())


def delta_cp(stoich: dict, T) -> float:
    return sum(nu * cp(s, T) for s, nu in stoich.items())


def gibbs_of_reaction(stoich: dict, T: float = T_STD) -> float:
    """Delta G_rxn(T) with the full temperature dependence (Delta H and Delta S integrated with Cp)."""
    dH298 = sum(nu * SPECIES[s][0] for s, nu in stoich.items())
    dG298 = sum(nu * SPECIES[s][1] for s, nu in stoich.items())
    dS298 = (dH298 - dG298) / T_STD
    dH = dH298 + sum(nu * enthalpy_change(s, T_STD, T) for s, nu in stoich.items())
    dS = dS298 + sum(nu * entropy_change(s, T_STD, T) for s, nu in stoich.items())
    return dH - T * dS


def equilibrium_constant(stoich: dict, T: float = T_STD) -> float:
    """Thermodynamic K (activities; for gases in pressure units of bar at the standard state)."""
    return math.exp(-gibbs_of_reaction(stoich, T) / (R * T))


def equilibrium_constant_vanthoff(K1: float, T1: float, dH: float, T2: float) -> float:
    """K at T2 from K at T1 with a constant heat of reaction."""
    return K1 * math.exp(-dH / R * (1 / T2 - 1 / T1))


def kc_from_kp(Kp: float, delta_nu: float, T: float, p_std: float = 1e5) -> float:
    """K in concentration units (mol/m3) from the pressure-based K (standard state p_std Pa): Kc = Kp (p0/RT)^dnu."""
    return Kp * (p_std / (R * T)) ** delta_nu


def adiabatic_temperature_simple(T0: float, X, dH_key: float, sum_theta_cp: float, delta_cp: float = 0.0) -> float:
    """Constant-Cp energy balance (Fogler Eq. 11-29): T = T0 + X (-Delta H_rxn(T0)) / (sum Theta_i Cp_i + X Delta Cp),
    with dH_key the heat of reaction per mole of the key reactant consumed and the Cp sums per mole of key fed."""
    X = np.asarray(X, float)
    return T0 + X * (-dH_key) / (sum_theta_cp + X * delta_cp)


def adiabatic_temperature(T0: float, X: float, stoich: dict, key: str, feed: dict) -> float:
    """Adiabatic outlet temperature after conversion X of the key reactant, from the full energy balance with the
    species Cp(T) polynomials: sum_i F_i(X) [H_i(T) - H_i(T0)] = -X F_key0 Delta H_rxn(T0)/|nu_key|.
    feed = {species: moles} including inerts (species must be in the table)."""
    from scipy import optimize

    nu_key = abs(stoich[key])
    F = {s: feed.get(s, 0.0) + stoich.get(s, 0.0) * X * feed[key] / nu_key for s in set(feed) | set(stoich)}
    heat = -X * feed[key] / nu_key * heat_of_reaction(stoich, T0)

    def residual(T):
        return sum(F[s] * enthalpy_change(s, T0, T) for s in F) - heat

    # scan upward (or downward for an endothermic reaction) to the first sign change: the Cp polynomials are
    # extrapolated above about 1500 K and must not be trusted beyond the first root
    grid = np.arange(T0, 4500.0, 25.0) if heat >= 0 else np.arange(T0, 150.0, -25.0)
    vals = [residual(T) for T in grid]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if fa == 0:
            return float(a)
        if fa * fb < 0:
            return float(optimize.brentq(residual, a, b))
    raise RuntimeError("No adiabatic temperature found in the tabulated range.")
