"""Validation of engreact against analytic solutions, independent numerical integration and published values.

Run:  python docs/validate.py      (writes docs/VALIDATION.md)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import integrate

from engreact import ideal, kinetics, rate_data, thermo
from engreact.constants import R
from engreact.kinetics import Reaction

RESULTS: list[tuple] = []
rng = np.random.default_rng(2026)


def check(section, name, reference, value, expected, rtol=0.0, atol=0.0):
    ok = bool(np.isclose(float(value), float(expected), rtol=rtol, atol=atol))
    RESULTS.append((section, name, reference, float(value), float(expected), ok))
    if not ok:
        print(f"FAIL: {section} | {name}: computed {value!r}, expected {expected!r}")


# ------------------------------------------------------------------ 1. thermochemistry
S = "Thermochemistry"
wgs = {"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}
check(
    S, "water-gas shift Delta H at 298 K", "NIST: -41.2 kJ/mol", thermo.heat_of_reaction(wgs) / 1e3, -41.17, rtol=2e-3
)
check(
    S,
    "methane combustion Delta H at 298 K (to H2O vapour)",
    "NIST: -802.3 kJ/mol",
    thermo.heat_of_reaction({"CH4": -1, "O2": -2, "CO2": 1, "H2O": 2}) / 1e3,
    -802.3,
    rtol=2e-3,
)
check(
    S,
    "ammonia synthesis Delta H at 298 K (per mol N2)",
    "NIST: -92.2 kJ/mol",
    thermo.heat_of_reaction({"N2": -1, "H2": -3, "NH3": 2}) / 1e3,
    -92.22,
    rtol=2e-3,
)
check(S, "Cp of N2 at 298 K", "NIST 29.1 J/(mol K)", float(thermo.cp("N2", 298.15)), 29.1, rtol=1e-2)
check(S, "Cp of CO2 at 1000 K", "NIST 54.3 J/(mol K)", float(thermo.cp("CO2", 1000.0)), 54.3, rtol=2e-2)
check(S, "Cp of H2O vapour at 1000 K", "NIST 41.3 J/(mol K)", float(thermo.cp("H2O", 1000.0)), 41.3, rtol=2e-2)
check(
    S,
    "water-gas shift K at 298 K",
    "exp(28.6 kJ / RT) = 1.05e5 (from Delta G = -28.6 kJ)",
    thermo.equilibrium_constant(wgs),
    math.exp(28620 / (R * 298.15)),
    rtol=2e-2,
)
check(
    S,
    "water-gas shift K at 1000 K",
    "Smith, Van Ness & Abbott Ex. 13.4-type value ~1.4 (0.7-1.6 across sources)",
    thermo.equilibrium_constant(wgs, 1000.0),
    1.4,
    rtol=0.3,
)
check(
    S,
    "van 't Hoff with constant Delta H agrees with the full Cp integration within 2 % over 298 -> 400 K (WGS)",
    "consistency",
    thermo.equilibrium_constant_vanthoff(thermo.equilibrium_constant(wgs), 298.15, thermo.heat_of_reaction(wgs), 400.0)
    / thermo.equilibrium_constant(wgs, 400.0),
    1.0,
    rtol=0.02,
)
check(
    S,
    "... but is 25 % off by 800 K (Delta Cp matters over a wide range)",
    "consistency",
    thermo.equilibrium_constant_vanthoff(thermo.equilibrium_constant(wgs), 298.15, thermo.heat_of_reaction(wgs), 800.0)
    / thermo.equilibrium_constant(wgs, 800.0),
    0.76,
    atol=0.05,
)
check(
    S,
    "ammonia synthesis (1/2 N2 + 3/2 H2 -> NH3): log10 K at 700 K",
    "Smith, Van Ness & Abbott: K = 7.6e-3 (log = -2.12)",
    float(np.log10(thermo.equilibrium_constant({"N2": -0.5, "H2": -1.5, "NH3": 1}, 700.0))),
    -2.12,
    atol=0.15,
)
check(
    S,
    "Kc from Kp: A -> 2B at 500 K, Kp = 1 bar",
    "Kp (p0/RT)^dnu",
    thermo.kc_from_kp(1.0, 1, 500.0),
    1e5 / (R * 500.0),
    rtol=1e-12,
)
T_ad = thermo.adiabatic_temperature(
    298.15, 1.0, {"H2": -2, "O2": -1, "H2O": 2}, "H2", {"H2": 2.0, "O2": 1.0, "N2": 3.76}
)
check(
    S,
    "adiabatic flame temperature of stoichiometric H2/air (no dissociation)",
    "Turns, An Introduction to Combustion: ~2500 K without dissociation (2380 K with)",
    T_ad,
    2500.0,
    rtol=0.05,
)
T_ch4 = thermo.adiabatic_temperature(
    298.15, 1.0, {"CH4": -1, "O2": -2, "CO2": 1, "H2O": 2}, "CH4", {"CH4": 1.0, "O2": 2.0, "N2": 7.52}
)
check(
    S,
    "adiabatic flame temperature of stoichiometric methane/air (no dissociation)",
    "Turns: ~2330 K",
    T_ch4,
    2330.0,
    rtol=0.05,
)
T_simple = thermo.adiabatic_temperature_simple(
    298.15,
    1.0,
    thermo.heat_of_reaction({"H2": -2, "O2": -1, "H2O": 2}) / 2,
    sum_theta_cp=(2 * 28.8 + 29.4 + 3.76 * 29.1) / 2,
    delta_cp=(2 * 33.6 - 2 * 28.8 - 29.4) / 2,
)
check(
    S,
    "the constant-Cp formula overestimates the flame temperature (Cp grows with T)",
    "physics",
    float(T_simple > T_ad),
    1.0,
)

# ------------------------------------------------------------------ 2. kinetics and stoichiometry
S = "Kinetics and stoichiometry"
check(
    S,
    "Arrhenius: doubling rule of thumb - Ea = 50 kJ/mol gives k(308)/k(298) = 1.92",
    "exact",
    kinetics.rate_ratio(50e3, 298.15, 308.15),
    math.exp(-50e3 / R * (1 / 308.15 - 1 / 298.15)),
    rtol=1e-12,
)
Ts = np.array([300, 320, 340, 360, 380.0])
ks = kinetics.arrhenius(2e8, 65e3, Ts) * np.exp(rng.normal(0, 0.01, 5))
fit = kinetics.fit_arrhenius(Ts, ks)
check(S, "fit_arrhenius recovers Ea from 1 % noisy data within 3 %", "recovery", fit["Ea"], 65e3, rtol=0.03)
check(
    S,
    "... and k0 within 40 % (k0 is poorly determined by extrapolation to 1/T = 0)",
    "recovery",
    fit["k0"],
    2e8,
    rtol=0.4,
)
check(
    S,
    "arrhenius_from_two round trip",
    "identity",
    kinetics.arrhenius_from_two(kinetics.arrhenius(1e5, 40e3, 300), 300, kinetics.arrhenius(1e5, 40e3, 400), 400)["Ea"],
    40e3,
    rtol=1e-10,
)
check(
    S,
    "temperature for a tenfold rate increase, Ea = 100 kJ/mol from 300 K",
    "exact 318.5 K",
    kinetics.temperature_for_rate_ratio(100e3, 300.0, 10.0),
    1 / (1 / 300 - R * math.log(10) / 100e3),
    rtol=1e-12,
)
rxn = Reaction({"A": -1, "B": -2, "C": 1}, "A")
tab = kinetics.stoichiometric_table(rxn, {"A": 1.0, "B": 3.0, "I": 1.0}, 0.5, "liquid", C_A0=2.0)
check(
    S,
    "liquid stoichiometric table: A + 2B -> C with Theta_B = 3 at X = 0.5: C_B = C_A0 (3 - 2 x 0.5) = 4",
    "Fogler Table 4-1",
    tab["C"]["B"],
    4.0,
)
check(S, "... C_C = C_A0 X = 1, inert unchanged", "exact", tab["C"]["C"] + tab["C"]["I"], 1.0 + 2.0)
gas = kinetics.stoichiometric_table(Reaction({"A": -1, "B": 2}, "A"), {"A": 1.0, "I": 1.0}, 0.5, "gas", C_A0=1.0)
check(
    S,
    "gas-phase A -> 2B, 50 % inert: epsilon = 0.5 and C_A(0.5) = 0.5/1.25 = 0.4",
    "Fogler Ex. 4-2-type",
    gas["C"]["A"] + gas["epsilon"],
    0.4 + 0.5,
)
check(S, "... C_B(0.5) = 1.0/1.25 = 0.8", "exact", gas["C"]["B"], 0.8)
check(
    S,
    "concentration/conversion round trip with epsilon",
    "identity",
    kinetics.conversion_from_concentration(2.0, kinetics.concentration_from_conversion(2.0, 0.37, eps=0.5), eps=0.5),
    0.37,
    rtol=1e-12,
)
X_eq = kinetics.equilibrium_conversion(Reaction({"A": -1, "B": 1}, "A"), {"A": 1.0}, Kc=3.0, C_A0=1.0)
check(S, "equilibrium conversion of A <-> B with Kc = 3 is 0.75", "exact Kc/(1+Kc)", X_eq, 0.75, rtol=1e-9)
X_eq2 = kinetics.equilibrium_conversion(Reaction({"A": -1, "B": 2}, "A"), {"A": 1.0}, Kc=2.0, C_A0=1.0, phase="gas")
C_ = kinetics.stoichiometric_table(Reaction({"A": -1, "B": 2}, "A"), {"A": 1.0}, X_eq2, "gas", 1.0)["C"]
check(
    S, "gas-phase A <-> 2B equilibrium satisfies C_B^2 / C_A = Kc", "identity", C_["B"] ** 2 / C_["A"], 2.0, rtol=1e-8
)
check(
    S,
    "limiting reactant of 2A + B -> C with feed A:B = 1:1 is A",
    "exact",
    float(kinetics.limiting_reactant(Reaction({"A": -2, "B": -1, "C": 1}, "A"), {"A": 1.0, "B": 1.0}) == "A"),
    1.0,
)
check(
    S,
    "reversible rate vanishes at equilibrium",
    "identity",
    float(kinetics.reversible_rate(2.0, 4.0, [1.0], [4.0])),
    0.0,
    atol=1e-15,
)

# ------------------------------------------------------------------ 3. rate data
S = "Analysis of rate data"
t = np.linspace(0, 200, 21)
for order, k in ((1, 0.02), (2, 0.01), (0.5, 0.02), (0, 0.004)):
    t = np.linspace(0, 80, 21) if order == 0.5 else np.linspace(0, 200, 21)  # half-order data hit zero at t = 100
    C = rate_data.integrated_concentration(t, 1.0, k, order)
    fit_ = rate_data.nonlinear_fit(t, C * (1 + rng.normal(0, 0.005, t.size)))
    check(
        S,
        f"nonlinear fit recovers order {order} from 0.5 % noisy data within 0.05",
        "recovery",
        fit_["order"],
        order,
        atol=0.05,
    )
    ok_int = rate_data.integral_method(t, C, order)["k"]
    check(S, f"integral method with the correct order recovers k = {k} exactly", "exact", ok_int, k, rtol=1e-8)
C2 = rate_data.integrated_concentration(np.linspace(0, 100, 41), 2.0, 0.02, 2)
dm = rate_data.differential_method(np.linspace(0, 100, 41), C2)
check(
    S,
    "differential method on second-order data: order within 0.1 and k within 10 %",
    "recovery",
    float(abs(dm["order"] - 2) < 0.1 and abs(dm["k"] / 0.02 - 1) < 0.1),
    1.0,
)
check(
    S, "half-life of a second-order reaction is 1/(k C0)", "exact", rate_data.half_life(2.0, 0.05, 2), 10.0, rtol=1e-12
)
check(
    S,
    "half-life of a first-order reaction is ln 2 / k",
    "exact",
    rate_data.half_life(2.0, 0.05, 1),
    math.log(2) / 0.05,
    rtol=1e-12,
)
hl = rate_data.half_life_method([0.5, 1.0, 2.0, 4.0], [rate_data.half_life(c, 0.05, 2) for c in (0.5, 1.0, 2.0, 4.0)])
check(S, "half-life method recovers order 2", "exact", hl["order"], 2.0, rtol=1e-10)
ir = rate_data.initial_rates_method([0.5, 1, 2, 4], [0.03 * c**1.5 for c in (0.5, 1, 2, 4)])
check(S, "initial-rates method recovers order 1.5 and k", "exact", ir["order"] + ir["k"], 1.5 + 0.03, rtol=1e-10)
t = np.linspace(0, 200, 21)
best = rate_data.compare_orders(
    t, rate_data.integrated_concentration(t, 1.0, 0.01, 2) * (1 + rng.normal(0, 0.01, t.size))
)[0]
check(S, "AIC model comparison picks order 2 for second-order data with 1 % noise", "exact", best["order"], 2)
check(
    S,
    "integrated rate law for n = 1 is the exponential",
    "exact",
    float(rate_data.integrated_concentration(50.0, 3.0, 0.02, 1)),
    3.0 * math.exp(-1.0),
    rtol=1e-12,
)

# ------------------------------------------------------------------ 4. ideal reactors
S = "Ideal reactors"
k, tau, C0 = 0.3, 5.0, 2.0
check(
    S,
    "first order: CSTR X = k tau/(1 + k tau)",
    "exact",
    ideal.cstr_conversion_first_order(k, tau),
    1.5 / 2.5,
    rtol=1e-12,
)
check(
    S,
    "first order: PFR X = 1 - exp(-k tau) equals the batch result at t = tau",
    "exact",
    ideal.pfr_conversion_first_order(k, tau),
    float(ideal.batch_conversion_first_order(k, tau)),
    rtol=1e-12,
)
check(
    S,
    "second order: CSTR conversion satisfies the design equation",
    "identity",
    ideal.cstr_conversion_second_order(k, C0, tau)
    / (k * C0 * tau * (1 - ideal.cstr_conversion_second_order(k, C0, tau)) ** 2),
    1.0,
    rtol=1e-10,
)
check(
    S,
    "second order: PFR X = Da/(1 + Da)",
    "exact",
    ideal.pfr_conversion_second_order(k, C0, tau),
    3.0 / 4.0,
    rtol=1e-12,
)
rate = lambda X: k * C0 * (1 - X)  # noqa: E731
check(
    S,
    "general CSTR volume for first order equals F_A0 X / (k C_A0 (1 - X))",
    "exact",
    ideal.cstr_volume(1.0, 0.6, rate),
    0.6 / (k * C0 * 0.4),
    rtol=1e-12,
)
check(
    S,
    "general PFR volume for first order equals -ln(1 - X)/(k C_A0) x F_A0",
    "exact",
    ideal.pfr_volume(1.0, 0.6, rate),
    -math.log(0.4) / (k * C0),
    rtol=1e-10,
)
check(
    S,
    "pfr_conversion inverts pfr_volume",
    "identity",
    ideal.pfr_conversion(1.0, ideal.pfr_volume(1.0, 0.6, rate), rate),
    0.6,
    rtol=1e-8,
)
check(
    S,
    "cstr_conversion inverts cstr_volume",
    "identity",
    ideal.cstr_conversion(1.0, ideal.cstr_volume(1.0, 0.6, rate), rate),
    0.6,
    rtol=1e-8,
)
prof = ideal.pfr_profile(1.0, ideal.pfr_volume(1.0, 0.6, rate), rate)
check(S, "pfr_profile ends at the design conversion", "identity", prof["X"][-1], 0.6, rtol=1e-6)
Xs = np.linspace(0, 0.8, 9)
lev = ideal.levenspiel(Xs, k * C0 * (1 - Xs), F_A0=1.0)
check(
    S,
    "Levenspiel trapezoids vs exact PFR integral at X = 0.8 (9 points, within 2 %: the quadrature error of a coarse table)",
    "quadrature",
    lev["V_pfr"][-1],
    -math.log(0.2) / (k * C0),
    rtol=0.02,
)
check(
    S,
    "Levenspiel CSTR rectangle exceeds the PFR area for a rate that falls with X",
    "Levenspiel",
    float(lev["V_cstr"][-1] > lev["V_pfr"][-1]),
    1.0,
)
check(
    S,
    "for first order the PFR/CSTR volume ratio at X = 0.9 is (0.1 ln 10)/0.9 = 0.256",
    "Levenspiel Fig. 6.6",
    ideal.pfr_volume(1.0, 0.9, rate) / ideal.cstr_volume(1.0, 0.9, rate),
    0.1 * math.log(10) / 0.9,
    rtol=1e-8,
)
series = ideal.reactors_in_series(1.0, [("cstr", 1.0), ("cstr", 1.0), ("cstr", 1.0)], rate)
check(
    S,
    "three equal CSTRs in series (first order) match 1 - 1/(1 + k tau)^3",
    "exact",
    series[-1]["X_out"],
    ideal.cstrs_in_series_first_order(k * C0, 1.0, 3),
    rtol=1e-8,
)
n_eq = ideal.cstrs_in_series_first_order(k * C0, tau / 50, 50)
check(S, "50 CSTRs in series approach the PFR", "limit", n_eq, ideal.pfr_conversion_first_order(k * C0, tau), rtol=0.03)
rec0 = ideal.recycle_reactor(1.0, 2.0, rate, R=1e-9)["X_out"]
recinf = ideal.recycle_reactor(1.0, 2.0, rate, R=1e4)["X_out"]
check(
    S,
    "recycle reactor: R -> 0 gives the PFR",
    "Levenspiel Ch. 6",
    rec0,
    ideal.pfr_conversion(1.0, 2.0, rate),
    rtol=1e-6,
)
check(
    S, "... R -> infinity gives the CSTR", "Levenspiel Ch. 6", recinf, ideal.cstr_conversion(1.0, 2.0, rate), rtol=1e-3
)
rates = {"A": lambda C: -k * C["A"], "B": lambda C: k * C["A"]}
b = ideal.batch(rates, {"A": 2.0, "B": 0.0}, np.linspace(0, 10, 11))
check(S, "numerical batch reactor vs exponential decay", "analytic", b["A"][-1], 2.0 * math.exp(-k * 10), rtol=1e-7)
check(S, "... conserves mass", "identity", b["A"][-1] + b["B"][-1], 2.0, rtol=1e-8)
cs = ideal.cstr_steady(rates, {"A": 2.0, "B": 0.0}, tau)
check(S, "numerical steady CSTR vs C_A0/(1 + k tau)", "analytic", cs["A"], 2.0 / (1 + k * tau), rtol=1e-9)
p = ideal.pfr(rates, {"A": 2.0, "B": 0.0}, tau)
check(S, "numerical PFR vs exponential", "analytic", p["A"][-1], 2.0 * math.exp(-k * tau), rtol=1e-7)
G, rho, mu, Dp, phi = 1.0, 1.2, 1.8e-5, 3e-3, 0.4
check(
    S,
    "Ergun: laminar term dominates at low flux (Blake-Kozeny limit within 5 %)",
    "Ergun 1952",
    ideal.ergun(0.01, rho, mu, Dp, phi),
    150 * mu * 0.01 / (rho * Dp**2) * (1 - phi) ** 2 / phi**3,
    rtol=0.05,
)
check(
    S,
    "Ergun: turbulent term dominates at high flux (Burke-Plummer limit within 5 %)",
    "Ergun 1952",
    ideal.ergun(50.0, rho, mu, Dp, phi),
    1.75 * 50.0**2 / (rho * Dp) * (1 - phi) / phi**3,
    rtol=0.05,
)
alpha = 0.01
pb = ideal.pbr_with_pressure_drop(1.0, 50.0, lambda X, y: 0.0, alpha)
check(
    S,
    "pressure drop without reaction: y = sqrt(1 - alpha W)",
    "Fogler Eq. 5-33",
    pb["y"][-1],
    float(ideal.pressure_ratio_no_reaction(alpha, 50.0)),
    rtol=1e-7,
)
k_prime, CA0 = 0.05, 1.0
pb2 = ideal.pbr_with_pressure_drop(
    1.0, 10.0, lambda X, y: k_prime * kinetics.concentration_from_conversion(CA0, X, 0.0, y), 0.0
)
check(
    S,
    "packed bed without pressure drop equals the PFR",
    "analytic",
    pb2["X"][-1],
    1 - math.exp(-k_prime * CA0 * 10.0),
    rtol=1e-7,
)
pb3 = ideal.pbr_with_pressure_drop(
    1.0, 10.0, lambda X, y: k_prime * kinetics.concentration_from_conversion(CA0, X, 0.0, y), 0.02
)
check(
    S,
    "pressure drop lowers conversion for a positive-order gas reaction",
    "physics",
    float(pb3["X"][-1] < pb2["X"][-1]),
    1.0,
)
sol = integrate.solve_ivp(
    lambda w, s: [k_prime * CA0 * (1 - s[0]) * s[1], -0.02 / (2 * s[1])], (0, 10), [0, 1], rtol=1e-10, atol=1e-12
)
check(S, "... and agrees with an independent scipy integration", "scipy", pb3["X"][-1], sol.y[0][-1], rtol=1e-6)
check(
    S,
    "batch time for second order to X = 0.9: X/(k C0 (1-X)) = 9/(k C0)",
    "exact",
    ideal.batch_time_nth_order(0.5, 2.0, 0.9, 2),
    9 / (0.5 * 2.0),
    rtol=1e-12,
)

# ------------------------------------------------------------------ 5. multiple reactions
from engreact import bio, catalysis, energy, heterogeneous, multiple, rtd  # noqa: E402

S = "Multiple reactions"
k1, k2 = 1.0, 0.4
t = np.linspace(0, 10, 201)
bat = multiple.series_batch(k1, k2, 1.0, t)
net = multiple.Network([({"A": -1, "B": 1}, lambda C: k1 * C["A"]), ({"B": -1, "C": 1}, lambda C: k2 * C["B"])])
num = net.batch({"A": 1.0}, t)
check(
    S,
    "Bateman A -> B -> C vs the numerical network (max |dB|)",
    "analytic",
    float(np.max(np.abs(bat["B"] - num["B"]))),
    0.0,
    atol=1e-6,
)
check(
    S,
    "... mass balance A + B + C = C_A0 at every time",
    "identity",
    float(np.max(np.abs(num["A"] + num["B"] + num["C"] - 1.0))),
    0.0,
    atol=1e-8,
)
t_opt = multiple.series_optimum_time(k1, k2)
check(
    S,
    "optimum time for B: dB/dt = 0 there (numerical derivative)",
    "identity",
    float(
        (multiple.series_batch(k1, k2, 1.0, t_opt + 1e-5)["B"] - multiple.series_batch(k1, k2, 1.0, t_opt - 1e-5)["B"])
        / 2e-5
    ),
    0.0,
    atol=1e-6,
)
check(
    S,
    "maximum B equals (k1/k2)^(k2/(k2-k1)) x C_A0",
    "analytic",
    float(multiple.series_batch(k1, k2, 1.0, t_opt)["B"]),
    multiple.series_max_B(k1, k2),
    rtol=1e-10,
)
check(
    S,
    "equal rate constants: B_max = C_A0/e at t = 1/k",
    "analytic",
    float(multiple.series_batch(1.0, 1.0, 1.0, 1.0)["B"]),
    1 / math.e,
    rtol=1e-12,
)
cstr = multiple.series_cstr(k1, k2, 1.0, 2.0)
cstr_num = net.cstr({"A": 1.0}, 2.0)
check(S, "series CSTR analytic vs numerical", "analytic", float(cstr["B"]), cstr_num["B"], rtol=1e-8)
check(
    S,
    "CSTR optimum tau for B: 1/sqrt(k1 k2)",
    "Levenspiel Eq. 8.15",
    multiple.series_cstr_optimum_tau(k1, k2),
    1 / math.sqrt(k1 * k2),
    rtol=1e-12,
)
taus = np.linspace(0.1, 10, 500)
check(
    S,
    "... maximises B in the CSTR (numerically)",
    "identity",
    float(taus[np.argmax(multiple.series_cstr(k1, k2, 1.0, taus)["B"])]),
    1 / math.sqrt(k1 * k2),
    rtol=0.02,
)
check(
    S,
    "PFR beats the CSTR for the intermediate (B_max PFR > B_max CSTR)",
    "Levenspiel",
    float(
        multiple.series_max_B(k1, k2)
        > float(multiple.series_cstr(k1, k2, 1.0, multiple.series_cstr_optimum_tau(k1, k2))["B"])
    ),
    1.0,
)
sb = net.semibatch({"A": 0.0}, 1.0, {"A": 1.0}, 0.1, np.linspace(0, 10, 101))
check(
    S,
    "semibatch volume grows linearly and the A balance closes",
    "identity",
    float(sb["V"][-1] + (sb["A"][-1] + sb["B"][-1] + sb["C"][-1]) * sb["V"][-1]),
    2.0 + 1.0,
    rtol=1e-6,
)
check(
    S,
    "parallel selectivity for orders 2 (desired) and 1: S = kD C_A / kU",
    "Fogler Ch. 8",
    float(multiple.parallel_selectivity_power_law(2.0, 0.5, 2, 1, 3.0)),
    12.0,
    rtol=1e-12,
)
check(
    S,
    "overall yield of B at t_opt",
    "identity",
    float(
        multiple.overall_yield(
            multiple.series_batch(k1, k2, 1.0, t_opt)["B"], 1.0, multiple.series_batch(k1, k2, 1.0, t_opt)["A"]
        )
    ),
    multiple.series_max_B(k1, k2) / (1 - math.exp(-k1 * t_opt)),
    rtol=1e-10,
)

# ------------------------------------------------------------------ 6. energy balances
S = "Energy balances"
k0, Ea, dH, C_A0, cp_vol = 1e10, 75e3, -250e3, 2000.0, 4e6
ss = energy.cstr_steady_states(k0, Ea, dH, C_A0, 300.0, 10.0, cp_vol)
check(
    S,
    "three steady states: G = R at each (residual)",
    "identity",
    float(
        max(
            abs(
                float(
                    energy.cstr_heat_curves(k0, Ea, dH, C_A0, 300.0, 10.0, cp_vol, T)["G"]
                    - energy.cstr_heat_curves(k0, Ea, dH, C_A0, 300.0, 10.0, cp_vol, T)["R"]
                )
            )
            for T in ss["T"]
        )
    ),
    0.0,
    atol=1e-3,
)
check(
    S,
    "... each steady state satisfies the adiabatic line T = T0 + X Delta T_ad",
    "energy balance",
    float(
        max(
            abs(T - (300.0 + X * energy.adiabatic_temperature_rise(dH, C_A0, cp_vol))) for T, X in zip(ss["T"], ss["X"])
        )
    ),
    0.0,
    atol=1e-6,
)
check(
    S,
    "... stability pattern stable / unstable / stable",
    "slope condition",
    float(ss["stable"] == [True, False, True]),
    1.0,
)
un = energy.cstr_unsteady(k0, Ea, dH, C_A0, 300.0, 10.0, cp_vol, np.linspace(0, 400, 401), C_A_init=C_A0, T_init=300.0)
check(S, "unsteady CSTR started cold settles on the low steady state", "dynamics", un["T"][-1], ss["T"][0], rtol=1e-3)
un2 = energy.cstr_unsteady(k0, Ea, dH, C_A0, 300.0, 10.0, cp_vol, np.linspace(0, 400, 401), C_A_init=0.0, T_init=450.0)
check(S, "... started hot settles on the high steady state", "dynamics", un2["T"][-1], ss["T"][2], rtol=1e-3)
un3 = energy.cstr_unsteady(
    k0,
    Ea,
    dH,
    C_A0,
    300.0,
    10.0,
    cp_vol,
    np.linspace(0, 400, 401),
    C_A_init=C_A0 * (1 - ss["X"][1]),
    T_init=ss["T"][1] + 0.5,
)
check(
    S,
    "... a perturbation above the middle state runs away to the ignited branch",
    "instability",
    un3["T"][-1],
    ss["T"][2],
    rtol=1e-3,
)
ie = energy.ignition_extinction(k0, Ea, dH, C_A0, 10.0, cp_vol, np.linspace(260, 360, 11))
n_ss = [len(t_) for t_ in ie["T_ss"]]
check(
    S,
    "ignition-extinction diagram: multiplicity appears and disappears as T0 rises",
    "hysteresis",
    float(n_ss[0] == 1 and 3 in n_ss and n_ss[-1] == 1),
    1.0,
)
rate_XT = lambda X, T: kinetics.arrhenius(k0, Ea, T) * C_A0 * (1 - X)  # noqa: E731
ad = energy.adiabatic_pfr(rate_XT, F_A0=C_A0 * 1e-3, V=0.02, T0=300.0, dH=dH, sum_theta_cp=cp_vol / C_A0)
check(
    S,
    "adiabatic PFR: T = T0 + X Delta T_ad along the reactor",
    "energy balance",
    float(np.max(np.abs(ad["T"] - (300.0 + ad["X"] * energy.adiabatic_temperature_rise(dH, C_A0, cp_vol))))),
    0.0,
    atol=1e-8,
)
hx = energy.pfr_heat_exchange(
    rate_XT, F_A0=C_A0 * 1e-3, V=0.02, T0=300.0, dH=dH, sum_F_cp=C_A0 * 1e-3 * cp_vol / C_A0, Ua=0.0, T_a=300.0
)
check(S, "PFR with Ua = 0 reproduces the adiabatic PFR", "consistency", hx["X"][-1], ad["X"][-1], rtol=1e-6)
hx2 = energy.pfr_heat_exchange(
    rate_XT, F_A0=C_A0 * 1e-3, V=0.02, T0=300.0, dH=dH, sum_F_cp=C_A0 * 1e-3 * cp_vol / C_A0, Ua=1e6, T_a=300.0
)
check(S, "... with strong cooling it stays near isothermal", "physics", float(hx2["T"].max() - 300.0 < 2.0), 1.0)
bh = energy.batch_with_heat_exchange(
    k0, Ea, dH, C_A0, 300.0, cp_vol, UA_over_V=0.0, T_a=300.0, t=np.linspace(0, 2000, 2001)
)
check(
    S,
    "adiabatic batch reaches T0 + Delta T_ad at complete conversion",
    "energy balance",
    bh["T_max"],
    300.0 + energy.adiabatic_temperature_rise(dH, C_A0, cp_vol),
    rtol=1e-4,
)
sem = energy.semenov_critical(dH, C_A0, k0, Ea, 350.0)
check(
    S,
    "Semenov critical temperature rise R T_a^2/Ea for T_a = 350 K, Ea = 75 kJ/mol",
    "Semenov",
    sem["dT_critical"],
    R * 350.0**2 / 75e3,
    rtol=1e-12,
)
check(
    S,
    "equilibrium line X_e = K/(1 + K) for A <-> B",
    "identity",
    float(energy.equilibrium_line(lambda T: 3.0, 400.0)[0]),
    0.75,
    rtol=1e-12,
)

# ------------------------------------------------------------------ 7. catalysis
S = "Catalysis"
check(
    S,
    "Langmuir: coverage 0.5 at K p = 1; dissociative 0.5 at K p = 1",
    "exact",
    float(catalysis.langmuir_coverage(2.0, 0.5) + catalysis.langmuir_dissociative(4.0, 0.25)),
    1.0,
    rtol=1e-12,
)
comp = catalysis.langmuir_competitive({"A": 1.0, "B": 3.0}, {"A": 1.0, "B": 1.0})
check(
    S, "competitive adsorption: coverages sum to K p sum / (1 + sum)", "exact", comp["A"] + comp["B"], 4 / 5, rtol=1e-12
)
pA = np.array([0.2, 0.5, 1, 2, 5.0])
rates_l = 2.0 * 1.5 * pA / (1 + 1.5 * pA)
fl = catalysis.fit_lhhw_single(pA, rates_l)
check(S, "LHHW single-site fit recovers kK and K", "exact", fl["k_K"] + fl["K"], 3.0 + 1.5, rtol=1e-8)
for phi_ in (0.5, 2.0, 5.0, 20.0):
    check(
        S,
        f"sphere effectiveness factor phi = {phi_}: closed form vs BVP",
        "scipy solve_bvp",
        catalysis.effectiveness_factor(phi_, "sphere"),
        catalysis.effectiveness_numerical(phi_, 1.0, "sphere"),
        rtol=2e-3,
    )
check(
    S,
    "slab effectiveness factor tanh(phi)/phi vs BVP (phi = 3)",
    "scipy solve_bvp",
    catalysis.effectiveness_factor(3.0, "slab"),
    catalysis.effectiveness_numerical(3.0, 1.0, "slab"),
    rtol=2e-3,
)
check(
    S,
    "cylinder effectiveness factor 2 I1/(phi I0) vs BVP (phi = 3)",
    "scipy solve_bvp",
    catalysis.effectiveness_factor(3.0, "cylinder"),
    catalysis.effectiveness_numerical(3.0, 1.0, "cylinder"),
    rtol=2e-3,
)
check(
    S,
    "large-phi asymptote: sphere eta -> 3/phi",
    "asymptotic",
    catalysis.effectiveness_factor(100.0, "sphere") * 100 / 3,
    1.0,
    rtol=0.02,
)
check(
    S,
    "second-order sphere at phi = 3: eta from the BVP lies between the first-order value and 1... between 0.5 and 0.9",
    "physics",
    float(0.5 < catalysis.effectiveness_numerical(3.0, 2.0, "sphere") < 0.9),
    1.0,
)
check(
    S,
    "pellet concentration profile: C/C_s = 1 at the surface and phi/sinh(phi) at the centre",
    "analytic",
    float(catalysis.concentration_profile_sphere(2.0, 1.0) + catalysis.concentration_profile_sphere(2.0, 0.0)),
    1 + 2 / math.sinh(2.0),
    rtol=1e-10,
)
check(
    S,
    "generalised Thiele modulus for first order is L sqrt(k/D_e)",
    "exact",
    catalysis.thiele_modulus(4.0, 1.0, 0.5),
    1.0,
    rtol=1e-12,
)
check(
    S,
    "Weisz-Prater: eta phi^2 identity for a sphere (C_WP = eta phi^2 = 3 (phi coth phi - 1))",
    "Weisz-Prater",
    catalysis.weisz_prater(catalysis.effectiveness_factor(2.0, "sphere") * 1.0 * 1.0, 1.0, 1.0, 1.0 / 4.0, 1.0),
    catalysis.effectiveness_factor(2.0, "sphere") * 4.0,
    rtol=1e-12,
)
check(
    S,
    "falsified kinetics: second order appears as 1.5, Ea halves",
    "Weisz",
    catalysis.falsified_kinetics(2, 80e3)["apparent_order"]
    + catalysis.falsified_kinetics(2, 80e3)["apparent_Ea"] / 40e3,
    2.5,
    rtol=1e-12,
)
check(
    S,
    "overall effectiveness -> eta when k_c a_c >> eta k",
    "limit",
    catalysis.overall_effectiveness(0.6, 1.0, 1e6, 1.0),
    0.6,
    rtol=1e-5,
)
check(S, "Frossling: Sh = 2 at rest", "exact", catalysis.sherwood_frossling(0.0, 1.0), 2.0)
check(
    S,
    "mass-transfer-limited bed conversion 1 - exp(-k_c a_c L/U)",
    "exact",
    float(catalysis.external_limited_conversion(0.01, 100.0, 1.0, 2.0)),
    1 - math.exp(-2.0),
    rtol=1e-12,
)
td = np.linspace(0, 100, 51)
check(
    S,
    "deactivation fit recovers k_d (second order law)",
    "exact",
    catalysis.fit_deactivation(td, catalysis.activity(td, 0.03, "second"), "second")["k_d"],
    0.03,
    rtol=1e-8,
)
check(
    S,
    "nth-order decay with n = 2 equals the second-order law",
    "identity",
    float(np.max(np.abs(catalysis.activity(td, 0.03, "nth", 2.0) - catalysis.activity(td, 0.03, "second")))),
    0.0,
    atol=1e-12,
)
traj = catalysis.temperature_time_trajectory(2e3, 60e3, 1e8, 50e3, 500.0, 2000.0, T_max=700.0)
check(
    S,
    "temperature-time trajectory keeps k(T) a(t) constant (checked at the end)",
    "identity",
    float(kinetics.arrhenius(1e8, 50e3, traj["T"][-1]) * traj["a"][-1] / kinetics.arrhenius(1e8, 50e3, 500.0)),
    1.0,
    rtol=1e-6,
)
check(S, "... and the temperature rises monotonically", "physics", float(np.all(np.diff(traj["T"]) >= -1e-9)), 1.0)

# ------------------------------------------------------------------ 8. residence-time distributions
S = "Residence-time distributions"
t = np.linspace(0, 300, 12001)
E_c = rtd.cstr_E(t, 10.0)
m = rtd.moments(t, E_c)
check(
    S,
    "CSTR E(t): area 1, mean tau, variance tau^2",
    "exact",
    float(m["area"] + m["mean"] / 10 + m["variance"] / 100),
    3.0,
    rtol=1e-4,
)
for n_ in (1, 3, 10):
    mt = rtd.moments(t, rtd.tanks_in_series_E(t, 10.0, n_))
    check(
        S,
        f"tanks-in-series n = {n_}: mean tau and sigma^2/tau^2 = 1/n",
        "exact",
        float(mt["mean"] / 10 + mt["variance"] / 100 * n_),
        2.0,
        rtol=2e-3,
    )
fit_t = rtd.fit_tanks_in_series(t, rtd.tanks_in_series_E(t, 10.0, 4))
check(S, "tanks-in-series fit recovers n = 4 and tau = 10", "exact", fit_t["n"] + fit_t["tau"], 14.0, rtol=1e-3)
th = np.linspace(0, 12, 2401)
for Pe in (2.0, 5.0, 20.0):
    md = rtd.moments(th, rtd.dispersion_E(th, Pe))
    check(
        S,
        f"closed-vessel dispersion model (numerical) Pe = {Pe}: variance vs Levenspiel's formula",
        "Levenspiel Eq. 13.13",
        md["variance"] / md["mean"] ** 2,
        rtd.dispersion_variance(Pe),
        rtol=5e-3,
    )
    check(S, f"... area 1 and mean 1 (Pe = {Pe})", "exact", md["area"] + md["mean"], 2.0, rtol=1e-3)
check(
    S,
    "peclet_from_variance inverts dispersion_variance",
    "identity",
    rtd.peclet_from_variance(rtd.dispersion_variance(7.0)),
    7.0,
    rtol=1e-8,
)
check(
    S,
    "open-vessel E(theta) at large Pe has mean ~1 + 2/Pe",
    "Levenspiel",
    rtd.moments(th, rtd.dispersion_E(th, 200.0))["mean"],
    1 + 2 / 200.0,
    rtol=2e-3,
)
check(
    S,
    "F from E and back",
    "identity",
    float(np.max(np.abs(rtd.E_from_F(t, rtd.F_from_E(t, E_c))[10:-10] - E_c[10:-10]))),
    0.0,
    atol=1e-4,
)
k_ = 0.2
check(
    S,
    "segregation model with the CSTR RTD equals the CSTR result for first order",
    "Levenspiel",
    rtd.segregation_conversion(t, E_c, k_),
    k_ * 10 / (1 + k_ * 10),
    rtol=1e-3,
)
check(
    S,
    "maximum mixedness with the CSTR RTD also equals the CSTR result (first order: RTD suffices)",
    "Zwietering",
    rtd.maximum_mixedness_conversion(t, E_c, k_),
    k_ * 10 / (1 + k_ * 10),
    rtol=5e-3,
)
E4 = rtd.tanks_in_series_E(t, 10.0, 4)
check(
    S,
    "segregation model with the 4-tank RTD (first order) equals the tanks-in-series formula",
    "exact",
    rtd.segregation_conversion(t, E4, k_),
    rtd.tanks_in_series_conversion(k_, 10.0, 4),
    rtol=1e-3,
)
check(
    S,
    "segregation > maximum mixedness for a second-order reaction (CSTR RTD)",
    "Zwietering",
    float(rtd.segregation_conversion(t, E_c, 0.05, 2, 1.0) > rtd.maximum_mixedness_conversion(t, E_c, 0.05, 2, 1.0)),
    1.0,
)
check(
    S,
    "dispersion-model conversion -> PFR as Pe -> infinity",
    "Danckwerts",
    rtd.dispersion_conversion(k_, 10.0, 1e4),
    1 - math.exp(-2.0),
    rtol=1e-3,
)
check(
    S,
    "dispersion-model conversion -> CSTR as Pe -> 0",
    "Danckwerts",
    rtd.dispersion_conversion(k_, 10.0, 1e-3),
    2.0 / 3.0,
    rtol=1e-3,
)
pulse = rtd.E_from_pulse(t, 5.0 * rtd.tanks_in_series_E(t, 10.0, 3))
check(S, "E from a pulse is normalised", "exact", rtd.moments(t, pulse)["area"], 1.0, rtol=1e-6)

# ------------------------------------------------------------------ 9. bioreactors and gas-solid reactions
S = "Bioreactors and gas-solid reactions"
S_ = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10.0])
v = bio.michaelis_menten(S_, 2.0, 0.5)
check(
    S,
    "Lineweaver-Burk and Hanes recover V_max, K_m from exact data",
    "exact",
    bio.lineweaver_burk(S_, v)["K_m"] + bio.hanes_woolf(S_, v)["V_max"],
    2.5,
    rtol=1e-10,
)
fit_mm = bio.fit_michaelis_menten(S_, v * (1 + rng.normal(0, 0.02, S_.size)))
check(
    S,
    "nonlinear Michaelis-Menten fit within 10 % from 2 % noisy data",
    "recovery",
    fit_mm["V_max"] / 2.0 + fit_mm["K_m"] / 0.5,
    2.0,
    rtol=0.1,
)
check(
    S,
    "competitive inhibition raises the apparent K_m, not V_max",
    "exact",
    float(bio.michaelis_menten(100.0, 2.0, 0.5, inhibitor=1.0, K_I=0.5, inhibition="competitive")),
    2.0 * 100 / (0.5 * 3 + 100),
    rtol=1e-12,
)
Sb = bio.batch_enzyme(2.0, 1.0, 0.5, np.array([0.0, 1.0]))
check(
    S,
    "integrated Michaelis-Menten satisfies K_m ln(S0/S) + (S0 - S) = V_max t",
    "identity",
    0.5 * math.log(2.0 / Sb[1]) + (2.0 - Sb[1]),
    1.0,
    rtol=1e-8,
)
check(
    S,
    "chemostat washout D_crit = mu_max S_in/(K_s + S_in)",
    "exact",
    bio.chemostat_washout(0.5, 0.1, 10.0),
    0.5 * 10 / 10.1,
    rtol=1e-12,
)
css = bio.chemostat_steady(0.3, 10.0, 0.5, 0.1, 0.5)
check(S, "chemostat steady state: mu(S) = D", "identity", float(bio.monod(css["S"], 0.5, 0.1)), 0.3, rtol=1e-10)
check(S, "... washout above D_crit", "exact", float(bio.chemostat_steady(0.499, 10.0, 0.5, 0.1, 0.5)["washout"]), 1.0)
Ds = np.linspace(0.01, 0.49, 400)
prod = [bio.chemostat_steady(D, 10.0, 0.5, 0.1, 0.5).get("productivity", 0.0) for D in Ds]
check(
    S,
    "optimum dilution rate maximises D X",
    "exact",
    float(Ds[int(np.argmax(prod))]),
    bio.chemostat_optimum(0.5, 0.1, 10.0),
    rtol=0.01,
)
bg = bio.batch_growth(0.1, 10.0, 0.5, 0.1, 0.5, np.linspace(0, 30, 301))
check(
    S,
    "batch growth: final biomass = X0 + Y (S0 - S) (yield balance)",
    "identity",
    bg["X"][-1],
    0.1 + 0.5 * (10.0 - bg["S"][-1]),
    rtol=1e-6,
)
fb = bio.fed_batch(1.0, 0.0, 1.0, 10.0, 0.1, 0.5, 0.1, 0.5, np.linspace(0, 10, 101))
check(S, "fed-batch: volume grows by F t", "identity", fb["V"][-1], 2.0, rtol=1e-10)
check(
    S,
    "shrinking core: t/tau at X = 1 is 1 in every regime",
    "exact",
    float(sum(heterogeneous.shrinking_core_time_fraction(1.0, c) for c in ("film", "ash", "reaction"))),
    3.0,
    rtol=1e-12,
)
check(
    S,
    "... ash-diffusion relation at X = 0.5 (Levenspiel Ch. 25)",
    "exact 1 - 3(0.5)^(2/3) + 1 = 0.1101",
    heterogeneous.shrinking_core_time_fraction(0.5, "ash"),
    1 - 3 * 0.5 ** (2 / 3) + 1,
    rtol=1e-12,
)
check(
    S,
    "conversion_at_time inverts the ash relation",
    "identity",
    float(heterogeneous.conversion_at_time(0.1101 * 50.0, 50.0, "ash")),
    0.5,
    rtol=1e-3,
)
tt = np.array([2, 5, 10, 20, 40.0])
Xd = heterogeneous.conversion_at_time(tt, 50.0, "ash")
check(
    S,
    "identify_regime picks ash diffusion for ash-controlled data",
    "exact",
    float(heterogeneous.identify_regime(tt, Xd)["best"] == "ash"),
    1.0,
)
check(
    S,
    "tau for complete conversion: ash scales with R^2, reaction with R",
    "Levenspiel",
    heterogeneous.time_for_complete_conversion("ash", 1.0, 2.0, 1, 1.0, D_e=1.0)
    / heterogeneous.time_for_complete_conversion("ash", 1.0, 1.0, 1, 1.0, D_e=1.0)
    + heterogeneous.time_for_complete_conversion("reaction", 1.0, 2.0, 1, 1.0, k_s=1.0)
    / heterogeneous.time_for_complete_conversion("reaction", 1.0, 1.0, 1, 1.0, k_s=1.0),
    6.0,
    rtol=1e-12,
)
check(
    S,
    "combined resistances reduce to the single regime when the others vanish",
    "identity",
    float(heterogeneous.combined_resistances(0.5, 0.0, 30.0, 0.0)),
    30.0 * heterogeneous.shrinking_core_time_fraction(0.5, "ash"),
    rtol=1e-12,
)

# ------------------------------------------------------------------ report
passed = sum(r[5] for r in RESULTS)
lines = [
    "# Validation report",
    "",
    "Generated by `python docs/validate.py`; rerun in CI on every push. Checks compare engreact with analytic",
    "solutions, independent numerical integration (scipy), published thermochemical data and textbook values.",
    "",
    f"**{passed} / {len(RESULTS)} checks pass.**",
    "",
]
for section in dict.fromkeys(r[0] for r in RESULTS):
    lines += ["", f"## {section}", "", "| Check | Reference | engreact | Expected | Result |", "|---|---|---|---|:-:|"]
    for s, name, ref, v, ex, ok in RESULTS:
        if s == section:
            lines.append(f"| {name} | {ref} | {v:.10g} | {ex:.10g} | {'pass' if ok else 'FAIL'} |")
Path(__file__).with_name("VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{passed}/{len(RESULTS)} checks pass")
