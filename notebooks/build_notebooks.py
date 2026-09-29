"""Generate the course notebooks (single source of truth) and execute them.

    python notebooks/build_notebooks.py            # write and execute all notebooks
    python notebooks/build_notebooks.py 03 15      # only notebooks whose names start with 03 or 15
    python notebooks/build_notebooks.py --no-run   # write without executing
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
NOTEBOOKS: dict[str, list] = {}
EXTRAS: dict[str, dict] = {}

COLAB = '''try:                      # on Google Colab (or anywhere engreact is missing): install it from GitHub
    import engreact
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/ktwyw/engreact"], check=True)'''

STYLE = '''plt.rcParams.update({"figure.figsize": (7, 4), "figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

import inspect
from IPython.display import Code

def show_source(obj):
    """Display the source code of a library function or class."""
    return Code(inspect.getsource(obj), language="python")

def heading(text):
    print(f"\\n{text}\\n" + "=" * len(text))'''


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


def setup_cell(extra_imports=()):
    lines = ["import matplotlib.pyplot as plt", "import numpy as np", "import pandas as pd"]
    for imp in extra_imports:
        if imp not in lines:
            lines.append(imp)
    return code(COLAB + "\n" + "\n".join(lines) + "\n" + STYLE)


def read_data(name):
    """Code snippet reading a bundled CSV file (comment lines start with #)."""
    return f'pd.read_csv(datasets.path("{name}"), comment="#")'



# =====================================================================================================




NOTEBOOKS["00_python_for_reaction_engineering"] = [
    md(r"""
# 00 · Python for reaction engineering

A reactor is a mole balance, a rate law and an energy balance solved together, and most of the time the
solution is an ordinary differential equation or a root of an algebraic equation. This notebook sets up the
tools - units, exponentials that span ten orders of magnitude, `scipy` for integrals, roots and ODEs, and the
`engreact` library and its data - with the reaction that will recur throughout the course: an exothermic
first-order reaction A -> B.
"""),
    setup_cell(["from scipy import integrate, optimize", "from engreact import datasets, kinetics, thermo"]),
    md("## Units: SI throughout, and why"),
    code(r"""k = kinetics.arrhenius(5.0e7, 68e3, 320.0)           # 1/s, Ea in J/mol, T in K
print(f"k at 320 K = {k:.3e} 1/s = {k*3600:.3f} 1/h; half-life {np.log(2)/k/60:.1f} min")
C_A0 = 2500.0                                          # mol/m3 = 2.5 mol/L
print(f"C_A0 = {C_A0} mol/m3 = {C_A0/1e3} mol/L; rate at t = 0: {k*C_A0:.3f} mol/(m3 s)")"""),
    md(r"""
Every quantity in `engreact` is in SI: moles, metres, seconds, joules, kelvin, pascals. Textbook problems
come in litres, minutes, atmospheres and calories; converting at the boundary (input and output) and computing
in SI inside is the discipline that prevents the factor-of-1000 errors that reactor design is famous for.

## Exponentials and the Arrhenius law
"""),
    code(r"""T = np.linspace(280, 420, 8)
k = kinetics.arrhenius(5.0e7, 68e3, T)
print(pd.DataFrame({"T (K)": T, "k (1/s)": k, "k / k(300 K)": k / kinetics.arrhenius(5.0e7, 68e3, 300.0)}).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
Tf = np.linspace(280, 420, 200)
ax[0].plot(Tf, kinetics.arrhenius(5.0e7, 68e3, Tf)); ax[0].set(xlabel="T (K)", ylabel="k (1/s)", title="linear axes: an explosion")
ax[1].semilogy(1000 / Tf, kinetics.arrhenius(5.0e7, 68e3, Tf)); ax[1].set(xlabel="1000/T (1/K)", ylabel="k (1/s)", title="Arrhenius plot: a straight line"); plt.show()"""),
    md(r"""
A 140 K rise multiplies the rate by ten thousand. On linear axes that is a wall; against 1/T on a log axis it
is a straight line with slope $-E_a/R$ - the Arrhenius plot that every kineticist draws first, and the reason
temperature is the most powerful and the most dangerous control variable in a reactor.

## Integrals, roots and ODEs
"""),
    code(r"""k300 = kinetics.arrhenius(5.0e7, 68e3, 300.0)
# integral: PFR volume for 80 % conversion of a first-order reaction, F_A0 = 1 mol/s, C_A0 = 2500 mol/m3
V_pfr, _ = integrate.quad(lambda X: 1 / (k300 * C_A0 * (1 - X)), 0, 0.8)
print(f"PFR volume for X = 0.8: {V_pfr:.2f} m3 (analytic {-np.log(0.2)/(k300*C_A0):.2f})")
# root: the CSTR conversion for the same volume
X_cstr = optimize.brentq(lambda X: X / (k300 * C_A0 * (1 - X)) - V_pfr, 0, 0.999)
print(f"a CSTR of the same volume gives X = {X_cstr:.3f}")
# ODE: batch reactor
sol = integrate.solve_ivp(lambda t, C: [-k300 * C[0]], (0, 7200), [C_A0], t_eval=np.linspace(0, 7200, 5), rtol=1e-8)
print("batch concentration at 0, 30, 60, 90, 120 min:", np.round(sol.y[0], 1), "(analytic:", np.round(C_A0 * np.exp(-k300 * sol.t), 1), ")")"""),
    md(r"""
Three tools cover most of the course: `quad` for design integrals, `brentq` for the algebraic balances of
a CSTR, `solve_ivp` for batch and plug-flow reactors. Each is checked here against the closed form it
reproduces; the library uses the same tools inside, with tighter tolerances and checks of its own.

## Thermochemistry from a species table
"""),
    code(r"""wgs = {"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}
for T in (298.15, 500.0, 800.0, 1000.0):
    print(f"water-gas shift at {T:6.1f} K: Delta H = {thermo.heat_of_reaction(wgs, T)/1e3:6.1f} kJ/mol, K = {thermo.equilibrium_constant(wgs, T):9.3g}")"""),
    md(r"""
The library carries a small table of formation enthalpies and heat-capacity polynomials, enough for the
course's reactions: heats of reaction at any temperature and equilibrium constants with the full Cp
integration. The water-gas shift, exothermic and equilibrium-limited, favours low temperature - which is why
industry runs it in two stages.

## The course data
"""),
    code(r"""for name in datasets.available():
    print(f"{name:<32} {datasets.describe(name)[:90]}")
d = pd.read_csv(datasets.path("pilot_cstr_runs.csv"), comment="#")
print("\npilot CSTR runs:", d.shape[0], "rows; columns", list(d.columns))"""),
    md(r"""
Two kinds of data: a literature table with its sources, and pseudo-experimental files generated from the
library's own models with stated noise and deliberate faults (a mis-read sample, a temperature in the wrong
unit, a run that ran away). The header of each file says exactly how it was made - the honesty a real
laboratory notebook owes its reader - and the project of notebook 17 works through the pilot-plant files
to a design.
"""),
]

NOTEBOOKS["01_mole_balances_and_ideal_reactors"] = [
    md(r"""
# 01 · Mole balances and the ideal reactors

Everything starts with a balance on a species: in minus out plus generation equals accumulation. Applied
to a well-mixed tank with no flow it gives the batch reactor; with flow, the CSTR; to a thin slice of a tube,
the plug-flow reactor; to a slice of catalyst, the packed bed. This notebook derives the four design equations,
solves them for the course's first-order reaction, and compares the reactors on the same job.
"""),
    setup_cell(["from engreact import ideal, kinetics"]),
    md(r"""
## The general mole balance
$$F_{A0} - F_A + \int^V r_A\,dV = \frac{dN_A}{dt}$$
Batch (no flow): $dN_A/dt = r_A V$. CSTR (steady, well mixed): $V = (F_{A0} - F_A)/(-r_A)$. PFR (steady,
no axial mixing): $dF_A/dV = r_A$. PBR: $dF_A/dW = r_A'$ per mass of catalyst. Conversion $X = (F_{A0} - F_A)/F_{A0}$
turns each into an equation in $X$.
"""),
    code(r"""k = kinetics.arrhenius(5.0e7, 68e3, 320.0)
C_A0 = 2500.0
tau = np.linspace(0, 3600, 200)
fig, ax = plt.subplots()
ax.plot(tau / 60, ideal.batch_conversion_first_order(k, tau), label="batch (time t)")
ax.plot(tau / 60, [ideal.pfr_conversion_first_order(k, t) for t in tau], "--", label="PFR (space time tau)")
ax.plot(tau / 60, [ideal.cstr_conversion_first_order(k, t) for t in tau], label="CSTR (space time tau)")
ax.set(xlabel="t or tau (min)", ylabel="conversion X", title=f"first order, k = {k:.2e} 1/s at 320 K"); ax.legend(); plt.show()
for X in (0.5, 0.9, 0.99):
    t_b = ideal.batch_time_nth_order(k, C_A0, X, 1)
    print(f"X = {X}: batch/PFR needs {t_b/60:6.1f} min; a CSTR needs tau = {X/(k*(1-X))/60:7.1f} min ({X/(k*(1-X))/t_b:.1f} x longer)")"""),
    md(r"""
The batch reactor and the PFR obey the same equation - time in one is space time in the other - and both
beat the CSTR for a positive-order reaction, because the CSTR runs entirely at the low outlet concentration.
At 99 % conversion the CSTR needs twenty times the residence time: the penalty grows without bound as X
approaches 1.

## Design equations for any rate law
"""),
    code(r"""rate = lambda X: k * C_A0 * (1 - X)                    # -r_A(X) in mol/(m3 s)
F_A0 = 2.0                                              # mol/s (v0 = 0.8 L/s of 2.5 M feed)
for X in (0.6, 0.9):
    print(f"X = {X}: V_CSTR = {ideal.cstr_volume(F_A0, X, rate):.3f} m3, V_PFR = {ideal.pfr_volume(F_A0, X, rate):.3f} m3")
V = 2.0
print(f"\na 2 m3 reactor: CSTR reaches X = {ideal.cstr_conversion(F_A0, V, rate):.3f}, PFR X = {ideal.pfr_conversion(F_A0, V, rate):.3f}")
prof = ideal.pfr_profile(F_A0, V, rate)
print("PFR conversion at 25 %, 50 %, 75 % of the length:", np.round(np.interp([0.5, 1.0, 1.5], prof["V"], prof["X"]), 3))"""),
    md(r"""
With the rate given as a function of conversion, the library sizes either reactor by quadrature or root
finding, and gives the profile along a PFR - which shows the first quarter of the reactor doing more than the
last: rates fall as reactant is consumed.

## A numerical reactor for any set of species
"""),
    code(r"""rates = {"A": lambda C: -k * C["A"], "B": lambda C: k * C["A"]}
b = ideal.batch(rates, {"A": C_A0, "B": 0.0}, np.linspace(0, 3600, 7))
print(pd.DataFrame({"t (min)": b["t"] / 60, "C_A": b["A"], "C_B": b["B"], "C_A + C_B": b["A"] + b["B"]}).round(1).to_string(index=False))
c = ideal.cstr_steady(rates, {"A": C_A0, "B": 0.0}, tau=1200.0)
print(f"\nCSTR with tau = 20 min: C_A = {c['A']:.1f}, C_B = {c['B']:.1f} mol/m3 (X = {1 - c['A']/C_A0:.3f}; analytic {ideal.cstr_conversion_first_order(k, 1200.0):.3f})")"""),
    md(r"""
When rate laws are given in concentrations and there are several species, the reactors are integrated
numerically; the analytic cases above are then checks, and the atom balance (A + B constant) is the
sanity test that costs nothing and catches most mistakes.

## The four reactors in one table
"""),
    code(r"""rows = [("batch", "dN_A/dt = r_A V", "unsteady, no flow", "small volumes, pharmaceuticals, polymers, trials"),
        ("CSTR", "V = F_A0 X / (-r_A)", "steady, perfectly mixed", "liquid phase, fermentation, easy temperature control"),
        ("PFR", "dX/dV = -r_A / F_A0", "steady, no axial mixing", "gas phase, high conversion, large throughput"),
        ("PBR", "dX/dW = -r_A' / F_A0", "steady, catalyst mass W", "heterogeneous catalysis: most of the chemical industry")]
print(pd.DataFrame(rows, columns=["reactor", "design equation", "assumptions", "typical use"]).to_string(index=False))"""),
    md(r"""
Four idealisations, and the rest of the course: how the rate depends on concentration and temperature
(notebooks 03, 05), how to choose and combine reactors (02, 04), what happens with several reactions (06),
with heat (07-09), with catalysts (10-13), and when the mixing is not ideal (14-15).
"""),
]

NOTEBOOKS["02_conversion_and_reactor_sizing"] = [
    md(r"""
# 02 · Conversion and reactor sizing: Levenspiel plots

Given rate data as a function of conversion - from a laboratory reactor, not a rate law - how large must the
reactor be, and which kind? Levenspiel's plot of $F_{A0}/(-r_A)$ against $X$ answers both by geometry: the
CSTR volume is a rectangle, the PFR volume the area under the curve. This notebook draws the plot, sizes
reactors, arranges them in series, and shows when a CSTR before a PFR is the right order.
"""),
    setup_cell(["from engreact import ideal, kinetics"]),
    md("## Rate data as a table"),
    code(r"""X = np.array([0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8])
rate = np.array([0.45, 0.37, 0.30, 0.195, 0.113, 0.079, 0.05])     # -r_A in mol/(m3 s), a laboratory table
F_A0 = 0.4                                                          # mol/s
lev = ideal.levenspiel(X, rate, F_A0)
print(pd.DataFrame({"X": X, "-r_A": rate, "F_A0/(-r_A) (m3)": lev["F_over_r"], "V_CSTR (m3)": lev["V_cstr"], "V_PFR (m3)": lev["V_pfr"]}).round(3).to_string(index=False))
fig, ax = plt.subplots()
Xf = np.linspace(0, 0.8, 200)
ax.plot(X, lev["F_over_r"], "ko-", label="F_A0 / (-r_A)")
ax.fill_between(Xf, np.interp(Xf, X, lev["F_over_r"]), alpha=0.25, label=f"PFR area to X = 0.8: {lev['V_pfr'][-1]:.2f} m3")
ax.bar([0.4], [lev["F_over_r"][-1]], width=0.8, alpha=0.2, color="C3", align="center", label=f"CSTR rectangle: {lev['V_cstr'][-1]:.2f} m3")
ax.set(xlabel="conversion X", ylabel="F_A0 / (-r_A) (m3)", xlim=(0, 0.85), ylim=(0, 9)); ax.legend(); plt.show()"""),
    md(r"""
The rate falls as conversion rises, so $F_{A0}/(-r_A)$ climbs, and the CSTR - which pays the *final* (lowest)
rate for every mole converted - needs almost three times the PFR volume for 80 % conversion. The plot needs no
rate law: it is the design method for rate data that come as a table.

## Reactors in series
"""),
    code(r"""rate_fn = lambda x: np.interp(x, X, rate)
for name, stages in (("CSTR 0.8 m3 then PFR 1.5 m3", [("cstr", 0.8), ("pfr", 1.5)]), ("PFR 1.5 m3 then CSTR 0.8 m3", [("pfr", 1.5), ("cstr", 0.8)]),
                     ("two CSTRs of 1.15 m3", [("cstr", 1.15), ("cstr", 1.15)]), ("one CSTR of 2.3 m3", [("cstr", 2.3)])):
    out = ideal.reactors_in_series(F_A0, stages, rate_fn)
    print(f"{name:<32}: final X = {out[-1]['X_out']:.3f}   (" + ", ".join(f"{s['type']} {s['X_in']:.2f}->{s['X_out']:.2f}" for s in out) + ")")"""),
    md(r"""
For a rate that only falls with conversion, the PFR-first order wins by a hair and two CSTRs beat one of the
same total volume - each stage runs at a higher concentration than a single big tank. Many CSTRs in series
approach a PFR, which is why a cascade of stirred tanks is the industrial way to get plug flow with the
temperature control of a tank.

## When a CSTR should come first
"""),
    code(r"""X2 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
rate2 = np.array([0.02, 0.05, 0.09, 0.12, 0.13, 0.12, 0.09, 0.055, 0.03])      # autocatalytic: fastest in the middle
lev2 = ideal.levenspiel(X2, rate2, F_A0)
fig, ax = plt.subplots(); ax.plot(X2, lev2["F_over_r"], "ko-"); ax.set(xlabel="X", ylabel="F_A0/(-r_A) (m3)", title="an autocatalytic rate"); plt.show()
from scipy import integrate
rf2 = lambda x: np.interp(x, X2, rate2)
best = None
for V1 in np.linspace(0.5, 6, 23):
    X1 = ideal.reactors_in_series(F_A0, [("cstr", V1)], rf2)[0]["X_out"]
    kinks = [x for x in X2 if X1 < x < 0.8]                                           # the table's breakpoints
    V2 = F_A0 * integrate.quad(lambda x: 1 / rf2(x), X1, 0.8, points=kinks)[0]     # PFR from X1 to 0.8
    if best is None or V1 + V2 < best[0]:
        best = (V1 + V2, V1, X1, V2)
print(f"PFR alone to X = 0.8: {lev2['V_pfr'][-1]:.2f} m3; CSTR alone: {lev2['V_cstr'][-1]:.2f} m3")
print(f"best CSTR ({best[1]:.2f} m3, to X = {best[2]:.2f}) followed by a PFR ({best[3]:.2f} m3): total {best[0]:.2f} m3")"""),
    md(r"""
When the rate rises before it falls - autocatalysis, fermentation, some polymerisations - the CSTR should
operate at the maximum rate and a PFR should finish the job: the rectangle sits in the valley of the curve
and the area covers the climb out of it. Levenspiel's geometry makes the choice visible and the optimum
computable.

## From the plot to a rate law
"""),
    code(r"""from engreact import rate_data
C_A = 2500.0 * (1 - X)
fit = rate_data.initial_rates_method(C_A[1:], rate[1:])
print(f"the first table looks like a power law of order {fit['order']:.2f} with k = {fit['k']:.2e}; a first-order fit gives k = {np.mean(rate[:-1]/C_A[:-1]):.2e} 1/s")"""),
    md(r"""
Levenspiel plots do not need a rate law, but a rate law lets you extrapolate, change temperature and design
for conditions not in the table. The next notebook builds rate laws; notebook 05 extracts them from data.
"""),
]

NOTEBOOKS["03_rate_laws_and_stoichiometry"] = [
    md(r"""
# 03 · Rate laws and stoichiometry

A rate law says how fast; stoichiometry says what everything else is doing while A reacts. Between them
they turn a design equation in $X$ into something solvable: the rate as a function of conversion. This
notebook builds stoichiometric tables for liquids and for gases (where the volume changes), handles the
expansion factor, reversible reactions and the equilibrium limit, and shows how temperature enters through
Arrhenius and van 't Hoff.
"""),
    setup_cell(["from engreact import kinetics, thermo", "from engreact.kinetics import Reaction"]),
    md("## The stoichiometric table"),
    code(r"""rxn = Reaction({"A": -1, "B": -2, "C": 1}, key="A")
print("reaction:", rxn, "| moles change per mole of A:", rxn.delta)
feed = {"A": 1.0, "B": 3.0, "I": 0.5}                                  # Theta_B = 3, inert
X = np.array([0, 0.25, 0.5, 0.75, 1.0])
tab = kinetics.stoichiometric_table(rxn, feed, X, phase="liquid", C_A0=2.0)
print("\nliquid phase (C_A0 = 2 mol/L):")
print(pd.DataFrame({"X": X, **{s: tab["C"][s] for s in ("A", "B", "C", "I")}}).round(3).to_string(index=False))"""),
    md(r"""
Every concentration is $C_{A0}(\Theta_i + \nu_i X/|\nu_A|)$: B with $\Theta_B = 3$ falls twice as fast as A
because two are consumed per A, C rises, the inert stays. The table is the bookkeeping that lets a rate law in
concentrations become a function of $X$ alone.

## Gas phase: the volume changes
"""),
    code(r"""gas = Reaction({"A": -1, "B": 2}, key="A")                       # A -> 2B: moles double
for inert in (0.0, 1.0, 4.0):
    feed = {"A": 1.0, "I": inert}
    tab = kinetics.stoichiometric_table(gas, feed, X, phase="gas", C_A0=1.0)
    print(f"y_A0 = {1/(1+inert):.2f}: epsilon = {tab['epsilon']:.2f};  C_A(X) = {np.round(tab['C']['A'], 3)}")
print("\ncompare the liquid-like case (no expansion): C_A(X) =", np.round(1.0 * (1 - X), 3))"""),
    md(r"""
When moles increase, the gas expands, and every concentration is divided by $1 + \varepsilon X$: at complete
conversion of a pure A -> 2B feed the volume has doubled, so $C_A$ falls faster than $(1 - X)$ and the reaction
slows itself. Diluting with inert reduces $\varepsilon$ - one reason gas-phase reactors run with inerts or in
excess of one reactant.

## Pressure, temperature and the rate as a function of X
"""),
    code(r"""k = kinetics.arrhenius(5.0e7, 68e3, 320.0)
C_A0 = 40.0                                                            # mol/m3, a gas at 1 bar and 300 K-ish
for eps, label in ((0.0, "no expansion"), (1.0, "A -> 2B, pure feed")):
    C_A = kinetics.concentration_from_conversion(C_A0, X, eps=eps)
    print(f"{label:<20}: -r_A(X) = k C_A = {np.round(k * C_A, 5)} mol/(m3 s)")
print("\nwith 20 % pressure drop and the gas heated from T0 to 1.1 T0 at X = 0.5:",
      f"C_A = {kinetics.concentration_from_conversion(C_A0, 0.5, eps=1.0, P_ratio=0.8, T_ratio=1.1):.2f} mol/m3 (vs {kinetics.concentration_from_conversion(C_A0, 0.5, eps=1.0):.2f} at P0, T0)")"""),
    md(r"""
The full gas-phase expression $C_A = C_{A0}\frac{1 - X}{1 + \varepsilon X}\frac{P}{P_0}\frac{T_0}{T}$ collects
three effects: consumption, expansion and the state of the gas. Pressure drop (notebook 04) and heating
(notebook 07) both enter the rate through it.

## Reversible reactions and equilibrium conversion
"""),
    code(r"""rev = Reaction({"A": -1, "B": 1}, key="A")
for Kc in (0.5, 2.0, 10.0, 100.0):
    print(f"A <-> B with Kc = {Kc:5.1f}: X_e = {kinetics.equilibrium_conversion(rev, {'A': 1.0}, Kc, C_A0=1.0):.3f}")
diss = Reaction({"A": -1, "B": 2}, key="A")
for Kc in (0.1, 1.0, 10.0):
    Xe_l = kinetics.equilibrium_conversion(diss, {"A": 1.0}, Kc, C_A0=1.0, phase="liquid")
    Xe_g = kinetics.equilibrium_conversion(diss, {"A": 1.0}, Kc, C_A0=1.0, phase="gas")
    print(f"A <-> 2B with Kc = {Kc:5.1f} (C_A0 = 1): X_e = {Xe_l:.3f} at constant volume, {Xe_g:.3f} with gas expansion")
C = kinetics.stoichiometric_table(rev, {"A": 1.0}, 0.3, "liquid", 1.0)["C"]
print(f"\nnet rate of A <-> B (kf = 0.1, Kc = 2) at X = 0.3: {kinetics.reversible_rate(0.1, 2.0, [C['A']], [C['B']]):.4f}; at X = X_e: {kinetics.reversible_rate(0.1, 2.0, [1/3], [2/3]):.2e}")"""),
    md(r"""
A reversible reaction stops at the equilibrium conversion, where the forward and reverse rates cancel; the
net rate law $k_f(C_A - C_B/K_c)$ builds that in. Expansion helps a dissociation (Le Chatelier: more moles,
so lower pressure favours products), which the gas-phase table captures automatically.

## Temperature: Arrhenius and van 't Hoff together
"""),
    code(r"""stoich = {"C2H4": -1, "H2O": -1, "C2H5OH": 1}                       # ethylene hydration, exothermic
for T in (400.0, 500.0, 600.0):
    K = thermo.equilibrium_constant(stoich, T)
    Kc = thermo.kc_from_kp(K, -1, T)
    print(f"{T:.0f} K: Delta H = {thermo.heat_of_reaction(stoich, T)/1e3:6.1f} kJ/mol, Kp = {K:9.3g} 1/bar, Kc = {Kc:9.3g} m3/mol")
print("\nrate constants rise with T (Arrhenius) but the equilibrium falls (exothermic): the optimum is a compromise - notebook 07")"""),
    md(r"""
For an exothermic reversible reaction, heating speeds the approach to equilibrium and lowers the
equilibrium itself. Ethylene hydration, ammonia synthesis and SO$_2$ oxidation all live on this trade-off,
and the reactor's temperature profile is designed around it.
"""),
]

NOTEBOOKS["04_isothermal_reactor_design"] = [
    md(r"""
# 04 · Isothermal reactor design

Put the pieces together: a rate law, the stoichiometric table and a design equation give the volume of a
reactor - or its conversion, or the time a batch needs. This notebook designs a batch process and scales it
up, sizes a CSTR and a PFR for a second-order liquid reaction, designs a gas-phase packed bed with volume
change and pressure drop (the Ergun equation), and looks at recycle and at a membrane reactor.
"""),
    setup_cell(["from engreact import ideal, kinetics, datasets"]),
    md("## A batch process and its cycle time"),
    code(r"""k2 = 0.11e-3                                                        # m3/(mol s): saponification at 25 degC (0.11 L/(mol s))
C_A0 = 200.0                                                        # mol/m3 (0.2 M, equimolar)
t_react = ideal.batch_time_nth_order(k2, C_A0, 0.9, 2)
t_cycle = t_react + 30 * 60                                         # 30 min to fill, heat, empty, clean
V = 2.0
moles_per_day = C_A0 * V * 0.9 * 86400 / t_cycle
print(f"time to 90 % conversion: {t_react/60:.1f} min; cycle {t_cycle/60:.1f} min; a 2 m3 batch reactor makes {moles_per_day:.0f} mol/day of product")
Xs = np.linspace(0.3, 0.99, 70)
daily = [C_A0 * V * X * 86400 / (ideal.batch_time_nth_order(k2, C_A0, X, 2) + 1800) for X in Xs]
X_best = Xs[int(np.argmax(daily))]
print(f"daily output is maximised by stopping each batch at X = {X_best:.2f} ({max(daily):.0f} mol/day); at X = 0.99 it falls to {daily[-1]:.0f} mol/day")
plt.plot(Xs, daily); plt.xlabel("conversion at which the batch is stopped"); plt.ylabel("product (mol/day)"); plt.show()"""),
    md(r"""
A second-order reaction takes as long to go from 75 % to 90 % as from 0 to 75 %: the batch time is
$X/(kC_{A0}(1-X))$, unbounded as $X \to 1$. With a fixed dead time per batch, the daily output has a maximum at
a conversion below complete - here in the high eighties, because the dead time is short compared with the
reaction; with a longer dead time the optimum moves to higher conversion, and unreacted material is recycled
either way. Batch scheduling is an optimisation, not a chemistry question.

## CSTR versus PFR for a second-order reaction
"""),
    code(r"""F_A0 = 0.5                                                          # mol/s
rate = lambda X: k2 * C_A0**2 * (1 - X) ** 2
for X in (0.5, 0.8, 0.9, 0.95):
    print(f"X = {X}: CSTR {ideal.cstr_volume(F_A0, X, rate):7.2f} m3, PFR {ideal.pfr_volume(F_A0, X, rate):6.2f} m3, ratio {ideal.cstr_volume(F_A0, X, rate)/ideal.pfr_volume(F_A0, X, rate):5.1f}")
print("\nanalytic check at X = 0.8: PFR X/(1-X) = Da ->", ideal.pfr_conversion_second_order(k2, C_A0, ideal.pfr_volume(F_A0, 0.8, rate) / (F_A0 / C_A0)))"""),
    md(r"""
For second order the CSTR penalty is worse than for first: the rate falls with the square of the
concentration. At 95 % conversion the CSTR is nineteen times larger. When mixing (for heat transfer, for
slurries, for gas absorption) demands a tank, the answer is tanks in series or a PFR after the tank.

## A gas-phase packed bed with pressure drop
"""),
    code(r"""d = pd.read_csv(datasets.path("packed_bed_pressure_drop.csv"), comment="#")
rho, mu, Dp, phi, L = 1.15, 1.9e-5, 4e-3, 0.42, 1.5
pred = [ideal.ergun(rho * u, rho, mu, Dp, phi) * L for u in d.u_m_s]
print(pd.DataFrame({"u (m/s)": d.u_m_s, "measured dP (Pa)": d.dP_Pa, "Ergun (Pa)": np.round(pred, 0)}).to_string(index=False))
plt.loglog(d.u_m_s, d.dP_Pa, "o", label="measured"); plt.loglog(d.u_m_s, pred, label="Ergun"); plt.xlabel("superficial velocity (m/s)"); plt.ylabel("pressure drop (Pa)"); plt.legend(); plt.show()"""),
    md(r"""
The Ergun equation - a viscous term linear in velocity plus an inertial term quadratic in it - reproduces the
bed measurements within their scatter. Pressure drop matters for a gas-phase reaction because it lowers the
concentration downstream, and for the compressor bill.

## Conversion and pressure along the bed
"""),
    code(r"""k_prime = 2.0e-5                                                     # m3/(kg s), first order per kg catalyst
C_A0g, eps = 40.0, 0.5                                               # A -> 1.5 products... eps = 0.5 (y_A0 = 1, delta = 0.5)
F_A0 = 0.02
alpha = ideal.ergun_alpha(beta0=ideal.ergun(rho * 0.8, rho, mu, Dp, phi), P0=2e5, A_c=0.015, rho_c=1200.0, phi=phi)
print(f"alpha = {alpha:.2e} 1/kg -> pressure ratio at W = 100 kg without reaction: {ideal.pressure_ratio_no_reaction(alpha, 100.0):.3f}")
rate_pbr = lambda X, y: k_prime * kinetics.concentration_from_conversion(C_A0g, X, eps, P_ratio=y)
pb = ideal.pbr_with_pressure_drop(F_A0, 100.0, rate_pbr, alpha, eps)
pb0 = ideal.pbr_with_pressure_drop(F_A0, 100.0, rate_pbr, 0.0, eps)
fig, ax = plt.subplots(); ax.plot(pb["W"], pb["X"], label="X with pressure drop"); ax.plot(pb0["W"], pb0["X"], "--", label="X without"); ax.plot(pb["W"], pb["y"], "C3", label="P/P0")
ax.set(xlabel="catalyst mass W (kg)", ylabel="X, P/P0"); ax.legend(); plt.show()
print(f"outlet: X = {pb['X'][-1]:.3f} with pressure drop vs {pb0['X'][-1]:.3f} without; outlet pressure {pb['y'][-1]*2:.2f} bar")"""),
    md(r"""
Two coupled equations along the bed: conversion rises, pressure falls (faster as the gas expands), and the
falling pressure dilutes the reactant and costs conversion. Doubling the pellet size or the tube diameter
would cut $\alpha$ - the trade-off between pressure drop and catalyst effectiveness (notebook 11) is a
central packed-bed design decision.

## Recycle
"""),
    code(r"""rate1 = lambda X: kinetics.arrhenius(5.0e7, 68e3, 320.0) * 2500.0 * (1 - X)
V = 1.0
for R in (0.0, 0.5, 2.0, 10.0, 1000.0):
    out = ideal.recycle_reactor(2.0, V, rate1, R)
    print(f"recycle ratio R = {R:6.1f}: X_out = {out['X_out']:.3f} (reactor inlet X = {out['X_in']:.3f})")
print(f"limits: PFR {ideal.pfr_conversion(2.0, V, rate1):.3f}, CSTR {ideal.cstr_conversion(2.0, V, rate1):.3f}")"""),
    md(r"""
Recycling part of the product to the inlet spans the whole range between plug flow (no recycle) and a CSTR
(infinite recycle). It costs conversion for a positive-order reaction, but it is how a tube is given the
temperature stability and the dilution of a tank - and for autocatalytic kinetics, a moderate recycle is
optimal.
"""),
]

NOTEBOOKS["05_analysis_of_rate_data"] = [
    md(r"""
# 05 · Analysis of rate data

A rate law is measured, not derived: from a batch reactor's concentration-time curve, from initial rates at
several concentrations, from half-lives, or from a differential reactor. This notebook applies the classical
methods to the course data - the decomposition of N$_2$O$_5$, a saponification, an Arrhenius series - and
argues for nonlinear regression with honest uncertainties, catching a mis-read sample and a temperature in
the wrong unit on the way.
"""),
    setup_cell(["from engreact import datasets, kinetics, rate_data"]),
    md("## Look at the data"),
    code(r"""d = pd.read_csv(datasets.path("n2o5_decomposition.csv"), comment="#")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(d.t_s / 60, d.C_mmol_L, "o-"); ax[0].set(xlabel="t (min)", ylabel="C (mmol/L)", title="N2O5 in CCl4, 45 degC")
ax[1].semilogy(d.t_s / 60, d.C_mmol_L, "o-"); ax[1].set(xlabel="t (min)", ylabel="C (mmol/L)", title="log scale: a straight line except one point"); plt.show()
print(datasets.describe("n2o5_decomposition.csv"))"""),
    md(r"""
On a log axis first-order decay is a straight line, and one sample sits well above it. The file header
admits it was mis-read; a real notebook would not, and the plot is what finds it.

## The integral method: assume an order, test the linearity
"""),
    code(r"""t, C = d.t_s.to_numpy(), d.C_mmol_L.to_numpy()
mask = np.ones_like(t, bool); mask[7] = False                   # exclude the bad sample
for order in (0, 1, 2):
    r = rate_data.integral_method(t[mask], C[mask], order)
    print(f"order {order}: k = {r['k']:.3e}, r^2 = {r['r2']:.5f}")
r1 = rate_data.integral_method(t, C, 1)
print(f"\nfirst order with the bad point included: k = {r1['k']:.3e} (r^2 {r1['r2']:.4f}) - an 8 % error from one sample")"""),
    md(r"""
The integral method tests a hypothesis: for first order $\ln(C_0/C)$ is linear in $t$, and the data agree
($r^2$ = 0.9995) whereas orders 0 and 2 curve. Its weakness is that it needs the order guessed - and that a
single outlier bends the line.

## The differential method
"""),
    code(r"""dm = rate_data.differential_method(t[mask], C[mask])
print(f"differential method: order {dm['order']:.2f}, k = {dm['k']:.2e}")
s = pd.read_csv(datasets.path("saponification.csv"), comment="#")
dm2 = rate_data.differential_method(s.t_s, s.C_mol_L)
print(f"saponification: order {dm2['order']:.2f}, k = {dm2['k']:.3f} L/(mol s) (literature 0.10-0.12)")
plt.loglog(s.C_mol_L, dm2["rates"], "o"); plt.xlabel("C (mol/L)"); plt.ylabel("-dC/dt (mol/(L s))"); plt.title("differential method: slope = order"); plt.show()"""),
    md(r"""
Differentiating data amplifies noise - the rates scatter far more than the concentrations - but the method
needs no assumed order: the slope of the log-log plot *is* the order, and it finds first order for N$_2$O$_5$
and second for the saponification. Use it to choose the order, then the integral method or regression to
pin down k.

## Nonlinear regression: the honest method
"""),
    code(r"""fit = rate_data.nonlinear_fit(s.t_s, s.C_mol_L)
print(f"free order: n = {fit['order']:.2f} +/- {fit['se_order']:.2f}, k = {fit['k']:.4f} +/- {fit['se_k']:.4f}")
ranked = rate_data.compare_orders(s.t_s, s.C_mol_L)
print("orders ranked by AIC:", [(f["order"], round(f["aic"], 1)) for f in ranked])
best = ranked[0]
print(f"second order: k = {best['k']:.4f} +/- {best['se_k']:.4f} L/(mol s), C0 = {best['C0']:.4f} mol/L")"""),
    md(r"""
Fitting the integrated rate law directly to the concentrations - no transformation, no differentiation -
weights every point as it was measured and returns standard errors. The free-order fit lands on 2 with a
tight uncertainty, and the AIC comparison ranks the candidate orders. This is the method to report; the
classical ones are for looking.

## Temperature dependence: the Arrhenius series
"""),
    code(r"""a = pd.read_csv(datasets.path("arrhenius_runs.csv"), comment="#")
print(a.to_string(index=False))
raw = kinetics.fit_arrhenius(a.T_K, a.k_1_s)
fixed = a.copy(); fixed.loc[fixed.T_K < 100, "T_K"] += 273.15
fix = kinetics.fit_arrhenius(fixed.T_K, fixed.k_1_s)
print(f"\nas recorded: Ea = {raw['Ea']/1e3:.1f} kJ/mol (r^2 {raw['r2']:.3f});  57 read as degC -> 330 K: Ea = {fix['Ea']/1e3:.1f} +/- {fix['se_Ea']/1e3:.1f} kJ/mol (r^2 {fix['r2']:.4f}); true 72")
plt.semilogy(1000 / fixed.T_K, fixed.k_1_s, "o", label="data (corrected)"); Tf = np.linspace(295, 355, 100); plt.semilogy(1000 / Tf, kinetics.arrhenius(fix["k0"], fix["Ea"], Tf), label="fit"); plt.xlabel("1000/T (1/K)"); plt.ylabel("k (1/s)"); plt.legend(); plt.show()"""),
    md(r"""
One temperature written in Celsius wrecks the Arrhenius fit (a "temperature" of 57 K); the plot shows the
point far off the line, and the corrected series gives the activation energy within its standard error.
Units are the commonest error in kinetics, and the Arrhenius plot is the check.

## Half-lives and initial rates
"""),
    code(r"""C0s = np.array([0.5, 1.0, 2.0, 4.0])
half = np.array([rate_data.half_life(c, 0.05, 2) for c in C0s]) * (1 + np.random.default_rng(1).normal(0, 0.03, 4))
print(f"half-lives {np.round(half, 1)} at C0 = {C0s}: order = {rate_data.half_life_method(C0s, half)['order']:.2f} (half-life falls with C0 -> order above 1)")
r0 = 0.03 * C0s**1.5
print(f"initial rates: order = {rate_data.initial_rates_method(C0s, r0)['order']:.2f}")"""),
    md(r"""
Half-life and initial-rate methods use several runs at different starting concentrations: the way the
half-life scales with $C_0$ ($t_{1/2} \propto C_0^{1-n}$) gives the order without any curve fitting, and initial
rates avoid complications from products. Every method here gives the same answer on clean data; on messy data,
using two of them is how you find out your data are messy.
"""),
]

NOTEBOOKS["06_multiple_reactions_and_selectivity"] = [
    md(r"""
# 06 · Multiple reactions and selectivity

Real feeds do more than one thing: the product reacts further, a side reaction consumes reactant, two
paths compete. Then the question is not conversion but *selectivity* - how much of the right product per
unit of the wrong one - and the reactor type, the temperature and the concentration all become levers. This
notebook works through series and parallel reactions, the optimum time for an intermediate, the choice of
reactor, and a general reaction network.
"""),
    setup_cell(["from engreact import multiple"]),
    md("## Series reactions: A -> B -> C"),
    code(r"""k1, k2 = 0.5, 0.2                                             # 1/min
t = np.linspace(0, 20, 400)
b = multiple.series_batch(k1, k2, 1.0, t)
t_opt, B_max = multiple.series_optimum_time(k1, k2), multiple.series_max_B(k1, k2, 1.0)
fig, ax = plt.subplots()
for s in "ABC":
    ax.plot(t, b[s], label=s)
ax.axvline(t_opt, color="k", ls=":", label=f"t_opt = {t_opt:.2f} min, B_max = {B_max:.3f}")
ax.set(xlabel="t (min)", ylabel="C / C_A0", title="series reactions in a batch reactor / PFR"); ax.legend(); plt.show()
c = multiple.series_cstr(k1, k2, 1.0, t[1:])
tau_opt = multiple.series_cstr_optimum_tau(k1, k2)
print(f"batch/PFR: B_max = {B_max:.3f} at t = {t_opt:.2f} min | CSTR: B_max = {float(multiple.series_cstr(k1, k2, 1.0, tau_opt)['B']):.3f} at tau = {tau_opt:.2f} min")"""),
    md(r"""
The intermediate B rises, peaks and decays; the peak is at $t = \ln(k_2/k_1)/(k_2 - k_1)$ and its height
depends only on the ratio $k_1/k_2$. A PFR (or batch) always beats a CSTR for the intermediate: the CSTR
holds B at its outlet concentration, where it is exposed to the second reaction for the whole residence
time. Overshoot the optimum time and you convert your product into waste.

## Parallel reactions: the selectivity lever
"""),
    code(r"""kD, kU = 1.0, 0.5                                             # A -> D (desired, order 2), A -> U (undesired, order 1)
for C_A in (0.1, 1.0, 10.0):
    print(f"C_A = {C_A:5.1f}: instantaneous selectivity S_DU = {float(multiple.parallel_selectivity_power_law(kD, kU, 2, 1, C_A)):5.1f}")
net = multiple.Network([({"A": -1, "D": 1}, lambda C: kD * C["A"] ** 2), ({"A": -1, "U": 1}, lambda C: kU * C["A"])])
for name, out in (("PFR", net.pfr({"A": 2.0}, 5.0)), ("CSTR", net.cstr({"A": 2.0}, 5.0))):
    D, U, A = (out["D"][-1], out["U"][-1], out["A"][-1]) if name == "PFR" else (out["D"], out["U"], out["A"])
    print(f"{name}: X = {1 - A/2:.3f}, overall selectivity D/U = {D/U:.2f}, yield of D = {D/(2 - A):.3f}")"""),
    md(r"""
When the desired reaction has the higher order, high concentration favours it: keep the reactant
concentrated - a PFR, no inerts, high pressure for a gas - and the PFR's selectivity beats the CSTR's. If
the undesired reaction had the higher order, everything reverses: a CSTR, dilution, and slow addition. The
orders, not the reactor, decide.

## Temperature as a selectivity lever
"""),
    code(r"""from engreact import kinetics
ED, EU = 60e3, 90e3                                            # activation energies, J/mol
for T in (300, 350, 400, 450):
    ratio = kinetics.arrhenius(1.0, ED, T) / kinetics.arrhenius(1.0, EU, T)
    print(f"T = {T} K: kD/kU relative to 300 K = {ratio / (kinetics.arrhenius(1.0, ED, 300.0) / kinetics.arrhenius(1.0, EU, 300.0)):.3f}")"""),
    md(r"""
The reaction with the larger activation energy speeds up more with temperature. If the *undesired* path has
the larger $E_a$, run cold; if the desired one does, run hot. Selectivity is often the reason a reactor runs
at a temperature that looks slow.

## A reaction network in a semibatch reactor
"""),
    code(r"""# A + B -> D (desired, k = 1), B + D -> U (undesired, k = 0.5): add B slowly to keep it low
netw = multiple.Network([({"A": -1, "B": -1, "D": 1}, lambda C: 1.0 * C["A"] * C["B"]), ({"B": -1, "D": -1, "U": 1}, lambda C: 0.5 * C["B"] * C["D"])])
t = np.linspace(0, 20, 200)
allatonce = netw.batch({"A": 1.0, "B": 1.0}, t)
semi = netw.semibatch({"A": 2.0}, V0=0.5, feed_C={"B": 2.0}, v_in=0.5 / 10.0, t=np.linspace(0, 10, 200))   # B fed over 10 min into the same total volume
print(f"all at once (batch, 20 min): D = {allatonce['D'][-1]:.3f}, U = {allatonce['U'][-1]:.3f}, D/U = {allatonce['D'][-1]/allatonce['U'][-1]:.2f}")
print(f"semibatch (B fed over 10 min): D = {semi['D'][-1]:.3f}, U = {semi['U'][-1]:.3f}, D/U = {semi['D'][-1]/semi['U'][-1]:.2f} (final volume {semi['V'][-1]:.2f})")"""),
    md(r"""
Adding B slowly keeps its concentration low, which starves the undesired reaction (first order in B and in
D) more than the desired one: selectivity doubles. Semibatch operation is the standard trick for exothermic
or selectivity-sensitive chemistry - and the `Network` class handles any scheme, in any of the ideal
reactors, from a list of reactions and rate laws.

## Yield, selectivity, conversion: three different numbers
"""),
    code(r"""out = net.pfr({"A": 2.0}, 5.0)
A, D, U = out["A"][-1], out["D"][-1], out["U"][-1]
print(f"conversion of A: {1 - A/2:.3f} | overall selectivity S_DU = D/U = {D/U:.2f} | overall yield Y_D = D/(A0 - A) = {D/(2-A):.3f} | D per mole fed = {D/2:.3f}")"""),
    md(r"""
Conversion says how much reactant is gone, selectivity how it split, yield how much of it became product.
A plant is judged on yield and on the cost of separating U; a high conversion with poor selectivity is
usually worse than a low conversion with recycle. Keep the three numbers apart - and check which definition
a paper uses, because two conventions for selectivity are in circulation.
"""),
]

NOTEBOOKS["07_energy_balances_and_adiabatic_reactors"] = [
    md(r"""
# 07 · Energy balances and adiabatic reactors

Heat is where reactor engineering earns its keep: an exothermic reaction heats itself, speeds up, and, if
nothing removes the heat, runs away; an endothermic one cools and stalls. This notebook adds the energy
balance to the mole balance, follows an adiabatic PFR along its temperature and conversion profile, meets
the equilibrium limit of an exothermic reversible reaction, and shows why the industrial answer is
interstage cooling.
"""),
    setup_cell(["from engreact import energy, kinetics, thermo"]),
    md("## The adiabatic energy balance"),
    code(r"""k0, Ea, dH = 5.0e7, 68e3, -250e3                                 # the course reaction A -> B
C_A0, cp_vol = 2500.0, 3.6e6                                    # mol/m3, J/(m3 K)
dT_ad = energy.adiabatic_temperature_rise(dH, C_A0, cp_vol)
print(f"adiabatic temperature rise: {dT_ad:.0f} K - every 10 % of conversion adds {dT_ad/10:.1f} K")
rate = lambda X, T: kinetics.arrhenius(k0, Ea, T) * C_A0 * (1 - X)
F_A0 = C_A0 * 1e-3                                              # 1 L/s feed
for T0 in (290.0, 300.0, 310.0):
    ad = energy.adiabatic_pfr(rate, F_A0, 2.0, T0, dH, sum_theta_cp=cp_vol / C_A0)
    print(f"T0 = {T0:.0f} K: outlet X = {ad['X'][-1]:.3f}, T = {ad['T'][-1]:.1f} K")
ad = energy.adiabatic_pfr(rate, F_A0, 2.0, 300.0, dH, sum_theta_cp=cp_vol / C_A0)
fig, ax = plt.subplots(); ax.plot(ad["V"], ad["X"], label="X"); ax2 = ax.twinx(); ax2.plot(ad["V"], ad["T"], "C3", label="T"); ax.set(xlabel="V (m3)", ylabel="X"); ax2.set_ylabel("T (K)", color="C3"); ax.set_title("adiabatic PFR, T0 = 300 K"); plt.show()"""),
    md(r"""
In an adiabatic reactor temperature and conversion are locked together, $T = T_0 + X\,\Delta T_{ad}$: the
energy balance is a straight line in the $X$-$T$ plane. A 10 K change in feed temperature moves the outlet
from about a tenth converted to complete - the exponential sensitivity that makes exothermic reactors both
efficient and hazardous. The profile shows an induction length where little happens, then a
front where everything does.

## Exothermic and reversible: the equilibrium ceiling
"""),
    code(r"""stoich = {"SO2": -1, "O2": -0.5, "SO3": 1}                            # SO2 oxidation, the contact process
T = np.linspace(600, 1000, 200)
K = np.array([thermo.equilibrium_constant(stoich, t) for t in T])
p = 1.0                                                                # bar; feed 10 % SO2, 11 % O2 (air), balance N2
def X_eq(Kp, T):
    from scipy import optimize
    f = lambda X: (0.1 * X / (1 - 0.05 * X)) / ((0.1 * (1 - X) / (1 - 0.05 * X)) * np.sqrt((0.11 - 0.05 * X) / (1 - 0.05 * X) * p)) - Kp
    return optimize.brentq(f, 1e-9, 1 - 1e-9)
Xe = np.array([X_eq(k_, t) for k_, t in zip(K, T)])
print(f"equilibrium SO2 conversion: {Xe[T >= 650][0]:.3f} at 650 K, {Xe[T >= 800][0]:.3f} at 800 K, {Xe[T >= 900][0]:.3f} at 900 K")
fig, ax = plt.subplots()
ax.plot(T, Xe, "k", label="equilibrium X_e(T)")
for T0 in (650.0, 700.0):
    dT = 0.1 * 98.9e3 / 32.0                                            # adiabatic rise per unit conversion for 10 % SO2 in a ~32 J/(mol K) gas... per mole of feed
    Xline = np.linspace(0, 1, 50)
    ax.plot(T0 + Xline * dT, Xline, "--", label=f"adiabatic line from {T0:.0f} K")
ax.set(xlabel="T (K)", ylabel="conversion of SO2", ylim=(0, 1.02)); ax.legend(); plt.show()"""),
    md(r"""
SO$_2$ oxidation is exothermic and reversible: the equilibrium conversion falls from near 1 at 650 K to
below 0.5 at 900 K, while the catalyst needs at least 700 K to work. An adiabatic bed climbs its straight
line until it hits the equilibrium curve - at perhaps 70 % - and stops. Hence the industrial design: four
adiabatic beds with cooling between them, each restarting at low temperature, staircasing up the
equilibrium curve to 99.7 %.

## Interstage cooling
"""),
    code(r"""k_cat = lambda T: kinetics.arrhenius(2.0e6, 80e3, T)                # a made-up catalyst: rate proportional to the distance from equilibrium
def bed(X_in, T_in, V):
    from scipy import integrate
    def rhs(v, s):
        X, T = s
        Kp = thermo.equilibrium_constant(stoich, T)
        Xe_ = X_eq(Kp, T)
        r = k_cat(T) * max(Xe_ - X, 0.0)                                # rate proportional to the distance from equilibrium
        return [r, r * dT]
    sol = integrate.solve_ivp(rhs, (0, V), [X_in, T_in], rtol=1e-8)
    return sol.y[0][-1], sol.y[1][-1]
X, T_ = 0.0, 690.0
for stage in range(1, 5):
    X, T_ = bed(X, T_, 1.0)
    print(f"bed {stage}: X = {X:.3f}, T = {T_:.0f} K -> cooled back to 690 K")
    T_ = 690.0"""),
    md(r"""
Each bed converts until it nears the equilibrium curve; cooling the gas back to the inlet temperature lets
the next bed continue. Four stages reach what one bed cannot at any temperature. The same logic - staged
adiabatic beds with interstage cooling, or cold-shot injection of fresh feed - runs ammonia synthesis,
methanol synthesis and the contact process.

## Endothermic reactions: the opposite problem
"""),
    code(r"""dH_endo = +80e3
rate_e = lambda X, T: kinetics.arrhenius(1e9, 90e3, T) * C_A0 * (1 - X)
for T0 in (600.0, 650.0, 700.0):
    ad_e = energy.adiabatic_pfr(rate_e, F_A0, 2.0, T0, dH_endo, sum_theta_cp=cp_vol / C_A0)
    print(f"endothermic, T0 = {T0:.0f} K: outlet X = {ad_e['X'][-1]:.3f}, T = {ad_e['T'][-1]:.0f} K (cooled by {T0 - ad_e['T'][-1]:.0f} K)")"""),
    md(r"""
An endothermic reaction cools its own reactor and slows itself: adiabatic operation stalls, and the design
must supply heat - furnace tubes (steam cracking, reforming), or a fluidised bed with a hot solid circulating
in (catalytic cracking). Exothermic reactors fight to remove heat; endothermic ones fight to add it. Both
are energy-balance problems first.
"""),
]

NOTEBOOKS["08_cstr_multiple_steady_states"] = [
    md(r"""
# 08 · The exothermic CSTR: multiple steady states and stability

A stirred tank with an exothermic reaction can sit at more than one steady state for the same feed: a cold,
low-conversion state, a hot, high-conversion state, and an unstable one between them. Which one the reactor
finds depends on its history; a small change in feed temperature can make it jump ("ignite") or collapse
("extinguish"), with hysteresis. This notebook builds the heat-generation and heat-removal curves, finds the
steady states, tests their stability, and maps the ignition-extinction diagram of the course reaction.
"""),
    setup_cell(["from engreact import energy"]),
    md("## Heat generated and heat removed"),
    code(r"""k0, Ea, dH, C_A0, cp_vol = 5.0e7, 68e3, -250e3, 2500.0, 3.6e6
tau, kappa, T_a = 240.0, 0.8, 295.0
T = np.linspace(280, 480, 400)
fig, ax = plt.subplots(figsize=(7, 4.5))
for T0, col in ((300.0, "C0"), (315.0, "C1"), (330.0, "C3")):
    c = energy.cstr_heat_curves(k0, Ea, dH, C_A0, T0, tau, cp_vol, T, kappa, T_a)
    ax.plot(T, c["R"] / 1e6, color=col, ls="--", label=f"heat removed, T0 = {T0:.0f} K")
ax.plot(T, c["G"] / 1e6, "k", lw=2, label="heat generated G(T)")
ax.set(xlabel="reactor temperature T (K)", ylabel="heat per m3 of feed (MJ/m3)", ylim=(0, 700), title="the S-shaped generation curve and the straight removal lines"); ax.legend(); plt.show()
for T0 in (300.0, 315.0, 330.0):
    ss = energy.cstr_steady_states(k0, Ea, dH, C_A0, T0, tau, cp_vol, kappa, T_a, T_range=(250.0, 800.0))
    print(f"T0 = {T0:.0f} K: steady states at T = {[round(t) for t in ss['T']]} K, X = {[round(x, 3) for x in ss['X']]}, stable {ss['stable']}")"""),
    md(r"""
Heat generation follows the conversion, an S-curve in temperature (Arrhenius below, complete conversion
above); heat removal is a straight line through the feed and coolant temperatures. Their intersections are
the steady states: three at 300 K (the two upper ones close together, just inside the fold), three at 315 K
well separated, and a single hot state at 330 K, where the removal line has moved past the S-curve's knee.

## Stability
"""),
    code(r"""ss = energy.cstr_steady_states(k0, Ea, dH, C_A0, 315.0, tau, cp_vol, kappa, T_a, T_range=(250.0, 800.0))
t = np.linspace(0, 3000, 600)
fig, ax = plt.subplots()
for T_init, lbl in ((ss["T"][1] - 1, "1 K below the middle state"), (ss["T"][1] + 1, "1 K above the middle state"), (300.0, "cold start"), (450.0, "hot start")):
    C_init = C_A0 * (1 - ss["X"][1]) if "middle" in lbl else (C_A0 if "cold" in lbl else 0.0)
    un = energy.cstr_unsteady(k0, Ea, dH, C_A0, 315.0, tau, cp_vol, t, C_init, T_init, UA_over_Vcp=kappa / tau, T_a=T_a)
    ax.plot(t / 60, un["T"], label=lbl)
for T_ss, st in zip(ss["T"], ss["stable"]):
    ax.axhline(T_ss, color="k", ls=":" if st else "--", lw=0.8)
ax.set(xlabel="t (min)", ylabel="T (K)", title="transients at T0 = 315 K (dotted: stable states, dashed: unstable)"); ax.legend(); plt.show()"""),
    md(r"""
Start one degree below the middle state and the reactor cools to the extinguished branch; one degree above
and it ignites to 400 K. The middle state is a knife edge: at it, heat generation rises faster with
temperature than heat removal, so any deviation grows. The slope test $dG/dT < dR/dT$ is the necessary
condition for stability, and the transient integration is the proof.

## Ignition and extinction: hysteresis
"""),
    code(r"""T0s = np.linspace(285, 345, 61)
ie = energy.ignition_extinction(k0, Ea, dH, C_A0, tau, cp_vol, T0s, kappa, T_a)
fig, ax = plt.subplots()
for T0, Ts, sts in zip(ie["T0"], ie["T_ss"], ie["stable"]):
    for T_, st in zip(Ts, sts):
        ax.plot(T0, T_, "o" if st else "x", color="C0" if st else "C3", ms=4)
ax.set(xlabel="feed temperature T0 (K)", ylabel="steady-state reactor temperature (K)", title="ignition-extinction diagram (o stable, x unstable)"); plt.show()
n_ss = np.array([len(Ts) for Ts in ie["T_ss"]])
multi = T0s[n_ss == 3]
print(f"three steady states for T0 between {multi.min():.0f} and {multi.max():.0f} K: ignition on heating at about {multi.max():.0f} K, extinction on cooling at about {multi.min():.0f} K")"""),
    md(r"""
Raise the feed temperature slowly and the reactor stays cold until the upper limit of the multiplicity
region, then ignites; cool it back and it stays hot until the lower limit, then extinguishes. Between the
two, the state depends on history. Plant operators know this as "the reactor lit off" and "we lost the
reaction"; the diagram tells them where the edges are.

## Design implications
"""),
    code(r"""for kappa_ in (0.0, 0.8, 3.0):
    T0s = np.linspace(280, 400, 121)
    ie = energy.ignition_extinction(k0, Ea, dH, C_A0, tau, cp_vol, T0s, kappa_, T_a)
    n = np.array([len(Ts) for Ts in ie["T_ss"]])
    band = T0s[n == 3]
    print(f"kappa = UA/(v0 rho cp) = {kappa_}: " + (f"multiplicity for T0 in [{band.min():.0f}, {band.max():.0f}] K" if band.size else "no multiplicity in this range"))"""),
    md(r"""
More cooling (a larger $\kappa$) shrinks or removes the multiplicity region; so does a smaller residence time
or a more dilute feed. A reactor designed to run on the ignited branch must be started deliberately (heated
feed, then cooled) and protected against extinction; one designed for the low branch must be protected
against ignition. Either way, the operating point should sit far from the fold - which is what the
ignition-extinction diagram is for.
"""),
]

NOTEBOOKS["09_unsteady_reactors_and_runaway"] = [
    md(r"""
# 09 · Unsteady reactors, batch heat exchange and runaway

Start-up, shutdown, a coolant failure, a batch that is charged too fast: transients are where reactors are
lost. This notebook follows an exothermic batch reactor with jacket cooling to its temperature peak, finds the
cooling below which it runs away, applies Semenov's criterion for thermal explosion, and simulates a CSTR
coolant failure - the calculations behind a reactor's safety case.
"""),
    setup_cell(["from engreact import energy, kinetics"]),
    md("## A batch reactor with a cooling jacket"),
    code(r"""k0, Ea, dH, C_A0, cp_vol = 5.0e7, 68e3, -250e3, 2500.0, 3.6e6
t = np.linspace(0, 7200, 1441)
fig, ax = plt.subplots()
for UA_V in (0.0, 500.0, 1000.0, 2000.0, 4000.0):
    b = energy.batch_with_heat_exchange(k0, Ea, dH, C_A0, 320.0, cp_vol, UA_V, T_a=300.0, t=t)
    ax.plot(t / 60, b["T"], label=f"UA/V = {UA_V:.0f} W/(m3 K): T_max = {b['T_max']:.0f} K")
ax.set(xlabel="t (min)", ylabel="T (K)", title="batch reactor started at 320 K, jacket at 300 K"); ax.legend(fontsize=8); plt.show()"""),
    md(r"""
Without cooling the batch reaches the full adiabatic rise (174 K); with modest cooling it still peaks near
450 K because the heat release outruns the jacket once the reaction accelerates; only strong cooling keeps
the peak small. The peak temperature, not the average, is what decomposes the product, boils the solvent or
bursts the vessel.

## The critical cooling: a sharp transition
"""),
    code(r"""UAs = np.linspace(500, 6000, 45)
peaks = [energy.batch_with_heat_exchange(k0, Ea, dH, C_A0, 320.0, cp_vol, ua, 300.0, t)["T_max"] for ua in UAs]
plt.plot(UAs, peaks, "o-"); plt.xlabel("UA/V (W/(m3 K))"); plt.ylabel("peak temperature (K)"); plt.title("runaway boundary of the batch"); plt.show()
i = int(np.argmax(np.abs(np.diff(peaks))))
print(f"the peak temperature drops most sharply between UA/V = {UAs[i]:.0f} and {UAs[i+1]:.0f} W/(m3 K): the practical runaway boundary")
sem = energy.semenov_critical(dH, C_A0, k0, Ea, 320.0)
print(f"Semenov (zero-order estimate at 320 K): critical UA/V = {sem['UA_over_V_critical']:.0f} W/(m3 K), critical self-heating {sem['dT_critical']:.1f} K")"""),
    md(r"""
The peak temperature falls by hundreds of kelvin across a narrow band of cooling capacity: runaway is a
threshold phenomenon. Semenov's criterion - heat generation at the ambient temperature times $E_a/RT_a^2$,
compared with the cooling coefficient over $e$ - gives the same order of magnitude analytically, and its
critical self-heating of $RT_a^2/E_a$ (about 12 K here) is the temperature rise above which the balance
tips. A reactor that is "only" 15 K above its jacket may already be past the point of no return.

## The adiabatic time to maximum rate
"""),
    code(r"""for T_start in (310.0, 320.0, 330.0, 340.0):
    b = energy.batch_with_heat_exchange(k0, Ea, dH, C_A0, T_start, cp_vol, 0.0, 300.0, np.linspace(0, 36000, 7201))
    i = int(np.argmax(np.gradient(b["T"])))
    print(f"start at {T_start:.0f} K: time to maximum rate {b['t'][i]/60:6.1f} min")"""),
    md(r"""
The time to maximum rate under adiabatic conditions (TMR$_{ad}$) is a standard safety metric from reaction
calorimetry: how long after a cooling failure the operators have before the reaction runs. It shortens
exponentially with the starting temperature - halving for every ten kelvin here - which is why the process
temperature, not just the jacket, is a safety parameter.

## CSTR coolant failure
"""),
    code(r"""tau, kappa, T_a = 240.0, 0.8, 295.0
ss = energy.cstr_steady_states(k0, Ea, dH, C_A0, 305.0, tau, cp_vol, kappa, T_a, T_range=(250.0, 800.0))
T_ss, X_ss = ss["T"][0], ss["X"][0]
t = np.linspace(0, 20 * 3600, 2401)
normal = energy.cstr_unsteady(k0, Ea, dH, C_A0, 305.0, tau, cp_vol, t, C_A0 * (1 - X_ss), T_ss, UA_over_Vcp=kappa / tau, T_a=T_a)
failed = energy.cstr_unsteady(k0, Ea, dH, C_A0, 305.0, tau, cp_vol, t, C_A0 * (1 - X_ss), T_ss, UA_over_Vcp=0.0, T_a=T_a)
plt.plot(t / 3600, normal["T"], label="jacket working"); plt.plot(t / 3600, failed["T"], label="coolant fails at t = 0"); plt.xlabel("t (h)"); plt.ylabel("T (K)"); plt.legend(); plt.show()
crossed = np.nonzero(failed["T"] > T_ss + 50)[0]
print(f"operating at {T_ss:.1f} K (X = {X_ss:.3f}); after the coolant failure the reactor" + (f" passes +50 K after {t[crossed[0]]/3600:.1f} h and reaches {failed['T'].max():.0f} K" if crossed.size else " never ignites"))
print("the adiabatic reactor at this feed temperature has a single steady state, the ignited one (notebook 08, kappa = 0): the low state no longer exists")"""),
    md(r"""
On the low branch the reactor looks docile: 2 % conversion, a few kelvin above the feed. Lose the jacket and
for hours almost nothing seems to happen - the temperature creeps up by a degree an hour - and then, once
the self-heating passes Semenov's critical rise, it ignites to the adiabatic state within minutes. The long,
quiet induction is the dangerous part: it looks like nothing is wrong. The safety case must ask "what if the cooling stops",
and the answer is a number of minutes to a trip - a relief valve, an emergency quench, a shutdown of the
feed - that this simulation sizes.
"""),
]

NOTEBOOKS["10_catalysis_and_heterogeneous_kinetics"] = [
    md(r"""
# 10 · Catalysis and heterogeneous kinetics

Most of the chemical industry runs on solid catalysts: a reactant adsorbs on a surface, reacts there, and the
product desorbs. Each step can limit the rate, and the resulting rate laws - Langmuir-Hinshelwood-Hougen-
Watson - have a form that power laws cannot mimic: they saturate, they pass through maxima, they are inhibited
by products. This notebook builds adsorption isotherms, derives LHHW rate laws, discriminates between
mechanisms with data, and follows a catalyst as it deactivates.
"""),
    setup_cell(["from engreact import catalysis, datasets"]),
    md("## Adsorption: the Langmuir isotherm"),
    code(r"""p = np.linspace(0, 5, 200)
fig, ax = plt.subplots()
for K in (0.5, 2.0, 10.0):
    ax.plot(p, catalysis.langmuir_coverage(K, p), label=f"molecular, K = {K}")
ax.plot(p, catalysis.langmuir_dissociative(2.0, p), "k--", label="dissociative, K = 2")
ax.set(xlabel="partial pressure (bar)", ylabel="fractional coverage"); ax.legend(); plt.show()
comp = catalysis.langmuir_competitive({"A": 2.0, "B": 8.0}, {"A": 1.0, "B": 1.0})
print("competitive adsorption of A and B at 1 bar each (K_B = 4 K_A):", {k: round(v, 3) for k, v in comp.items()}, "- B crowds A off the surface")"""),
    md(r"""
Langmuir's isotherm - rate of adsorption proportional to the empty sites, of desorption to the covered ones
- gives a coverage that saturates: beyond a point, more pressure adds nothing because there are no sites
left. Competitive adsorption is the origin of inhibition: a strongly adsorbing product or poison occupies
the sites the reactant needs.

## LHHW rate laws
"""),
    code(r"""pA = np.linspace(0.01, 10, 300)
fig, ax = plt.subplots()
ax.plot(pA, catalysis.lhhw_rate(1.0, pA, K_A=1.0, mode="single"), label="single site: r = k K_A p_A / (1 + K_A p_A)")
ax.plot(pA, catalysis.lhhw_rate(1.0, pA, 1.0, K_A=1.0, K_B=1.0, mode="surface"), label="dual site, p_B = 1: r ~ p_A / (1 + K_A p_A + K_B p_B)^2")
ax.plot(pA, catalysis.lhhw_rate(1.0, pA, K_A=1.0, K_P=2.0, p_P=1.0, mode="single"), "--", label="single site with product inhibition (p_P = 1)")
ax.set(xlabel="p_A (bar)", ylabel="rate"); ax.legend(fontsize=8); plt.show()"""),
    md(r"""
A single-site mechanism gives a rate that saturates (first order at low pressure, zero order at high); a
dual-site surface reaction passes through a maximum, because the reactant crowds its partner off the
surface; product inhibition lowers everything. A power law fitted over a narrow range can mimic any of
these locally, which is why the *shape* over a wide range is what identifies a mechanism.

## Discriminating mechanisms with data
"""),
    code(r"""p_obs = np.array([0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
r_obs = catalysis.lhhw_rate(1.0, p_obs, K_A=1.5, mode="single") * (1 + np.random.default_rng(3).normal(0, 0.03, 6))
fit = catalysis.fit_lhhw_single(p_obs, r_obs)
print(f"single-site fit: k K_A = {fit['k_K']:.3f}, K_A = {fit['K']:.3f} (true 1.5, 1.5), rss = {fit['rss']:.2e}")
n_fit = np.polyfit(np.log(p_obs), np.log(r_obs), 1)
print(f"a power law fitted to the same data: order {n_fit[0]:.2f} - neither 0 nor 1, a sign that the mechanism saturates")
plt.plot(1 / p_obs, 1 / r_obs, "o"); plt.xlabel("1/p_A"); plt.ylabel("1/r"); plt.title("linearised single-site plot: straight line = single site"); plt.show()"""),
    md(r"""
The reciprocal plot linearises the single-site law, as Lineweaver-Burk does for enzymes (notebook 16); a
straight line supports the mechanism and its slope and intercept give the constants, refined by nonlinear
regression. Testing several mechanisms against data over a wide range of pressures is the heart of
heterogeneous kinetics - and the reason a "fractional order of 0.4" should make you suspect a saturating
mechanism.

## Catalyst deactivation
"""),
    code(r"""d = pd.read_csv(datasets.path("catalyst_deactivation.csv"), comment="#")
fits = {law: catalysis.fit_deactivation(d.t_h, d.activity, law) for law in ("first", "second", "linear")}
for law, f in fits.items():
    print(f"{law:<7} order decay: k_d = {f['k_d']:.4f} 1/h, rss = {f['rss']:.4f}")
best = min(fits, key=lambda l: fits[l]["rss"])
tt = np.linspace(0, 240, 200)
plt.plot(d.t_h, d.activity, "o", label="measured"); plt.plot(tt, catalysis.activity(tt, fits[best]["k_d"], best), label=f"{best}-order decay"); plt.xlabel("time on stream (h)"); plt.ylabel("activity"); plt.legend(); plt.show()"""),
    md(r"""
Catalysts die - by coking, sintering or poisoning - and the activity's time dependence tells which: coking
often follows second-order-like decay, sintering first order, poisoning by a feed impurity roughly linear.
The data pick second order clearly. Knowing the law lets the engineer schedule regeneration and, in the
meantime, compensate.

## Compensating by raising the temperature
"""),
    code(r"""traj = catalysis.temperature_time_trajectory(k_d0=2.0, E_d=60e3, k0=1e8, Ea=50e3, T0=500.0, t_max=400 * 86400, T_max=560.0)
plt.plot(traj["t"] / 86400, traj["T"] - 273.15); plt.xlabel("time on stream (days)"); plt.ylabel("reactor temperature (degC)"); plt.title("temperature-time trajectory for constant conversion"); plt.show()
print(f"the temperature must rise {traj['T'][-1]-500:.0f} K over {traj['t'][-1]/86400:.0f} days to hold conversion; it reaches the 560 K limit: {traj['reached_T_max']} - time to regenerate")"""),
    md(r"""
Raising the temperature offsets the loss of activity - until the maximum allowable temperature is reached,
and the accelerating deactivation (its own activation energy is higher than the reaction's) makes the
climb steepen. Refinery reformers and hydrotreaters run exactly this trajectory over a cycle of months
between regenerations.
"""),
]

NOTEBOOKS["11_internal_diffusion_and_effectiveness_factors"] = [
    md(r"""
# 11 · Internal diffusion and effectiveness factors

A porous catalyst pellet is a reactor inside a reactor: reactant must diffuse in through the pores while
reacting on the walls, and if reaction is fast the interior sees little of it. The Thiele modulus compares
reaction to diffusion and the effectiveness factor says what fraction of the pellet is used. This notebook
computes both for the standard geometries, checks the closed forms against the pellet's differential
equation, shows the falsified kinetics of the diffusion-limited regime, and applies the Weisz-Prater test.
"""),
    setup_cell(["from engreact import catalysis"]),
    md("## The Thiele modulus and the effectiveness factor"),
    code(r"""phi = np.logspace(-1, 2, 200)
fig, ax = plt.subplots()
for geom in ("slab", "cylinder", "sphere"):
    ax.loglog(phi, catalysis.effectiveness_factor(phi, geom), label=geom)
ax.loglog(phi, np.minimum(1, 3 / phi), "k:", label="asymptote 3/phi (sphere)")
ax.set(xlabel="Thiele modulus phi", ylabel="effectiveness factor eta"); ax.legend(); plt.show()
for phi_ in (0.3, 1.0, 3.0, 10.0, 30.0):
    print(f"phi = {phi_:5.1f}: eta (sphere) = {catalysis.effectiveness_factor(phi_, 'sphere'):.3f}")"""),
    md(r"""
Below $\phi \approx 1$ the pellet is fully used; above about 3 the effectiveness factor falls as $3/\phi$ for a
sphere and the reaction happens in an outer shell. Doubling the pellet radius doubles $\phi$; at large $\phi$
that halves the rate per pellet volume. The three geometries agree closely when $\phi$ is defined with the
volume-to-surface ratio - the generalised modulus - which is why "the" curve is quoted.

## Inside the pellet
"""),
    code(r"""r = np.linspace(0, 1, 200)
fig, ax = plt.subplots()
for phi_ in (0.5, 2.0, 5.0, 20.0):
    ax.plot(r, catalysis.concentration_profile_sphere(phi_, r), label=f"phi = {phi_}")
ax.set(xlabel="r / R", ylabel="C / C_s", title="reactant concentration inside a spherical pellet"); ax.legend(); plt.show()"""),
    md(r"""
At $\phi$ = 20 the reactant penetrates a tenth of the radius: the core is dead weight. That is why industrial
catalysts are shaped as rings, trilobes and thin shells (eggshell catalysts) - to put the active material
where the reactant can reach it - and why crushing a catalyst to a fine powder in the laboratory gives a
different rate than the pellets in the plant.

## Checking the closed forms against the pellet equation
"""),
    code(r"""for geom in ("slab", "cylinder", "sphere"):
    for phi_ in (1.0, 5.0):
        print(f"{geom:<8} phi = {phi_}: closed form {catalysis.effectiveness_factor(phi_, geom):.4f}, BVP {catalysis.effectiveness_numerical(phi_, 1.0, geom):.4f}")
print("\nsecond-order reaction, sphere, phi = 5: eta =", round(catalysis.effectiveness_numerical(5.0, 2.0, "sphere"), 4), "(first order:", round(catalysis.effectiveness_factor(5.0, "sphere"), 4), ")")"""),
    md(r"""
The closed forms are first-order results; for any other order the pellet's boundary-value problem must be
solved numerically, and the library's solver reproduces the closed forms to four digits before being trusted
on the second-order case. A validated numerical tool is how one handles the kinetics that textbooks leave
as "it can be shown".

## Falsified kinetics
"""),
    code(r"""from engreact import kinetics
k_true, Ea_true, D_e, R_p = 1.0, 80e3, 1e-8, 2e-3
print("in the strong-diffusion regime:", catalysis.falsified_kinetics(2, Ea_true))
T = np.array([500, 525, 550, 575, 600.0])
k_T = kinetics.arrhenius(1e8, Ea_true, T)
phi_T = np.array([catalysis.thiele_modulus(k_, D_e, R_p / 3) * 3 for k_ in k_T])       # sphere modulus with R
r_obs = k_T * catalysis.effectiveness_factor(phi_T, "sphere")
Ea_app = kinetics.fit_arrhenius(T, r_obs)["Ea"]
print(f"phi from {phi_T[0]:.0f} to {phi_T[-1]:.0f}: the apparent activation energy is {Ea_app/1e3:.0f} kJ/mol against a true {Ea_true/1e3:.0f}")"""),
    md(r"""
When diffusion limits, the observed rate goes as $\sqrt{k}$: the apparent activation energy halves and the
apparent order moves towards one. An Arrhenius plot that bends to a lower slope at high temperature is the
classic signature of the onset of diffusion control. Kinetic parameters measured on large pellets are
falsified; the true ones need small particles or a correction.

## The Weisz-Prater criterion: is my measurement diffusion-free?
"""),
    code(r"""for R_p in (0.1e-3, 0.5e-3, 2e-3, 5e-3):
    C_wp = catalysis.weisz_prater(r_obs=1e-3, rho_c=1500.0, R=R_p, D_e=1e-7, C_s=20.0)      # 1 mmol/(kg s) observed
    print(f"pellet radius {R_p*1e3:4.1f} mm: C_WP = {C_wp:7.2f} -> " + ("negligible internal diffusion" if C_wp < 0.3 else "diffusion limited"))"""),
    md(r"""
Weisz and Prater's criterion uses only observable quantities - the measured rate, the pellet size, the
effective diffusivity and the surface concentration - to decide whether the measurement is
diffusion-free: $C_{WP} = \eta\phi^2 \ll 1$. It is the first thing a referee asks about a reported rate law
for a solid catalyst, and the calculation takes one line.
"""),
]

NOTEBOOKS["12_external_mass_transfer_and_packed_beds"] = [
    md(r"""
# 12 · External mass transfer and packed-bed reactors

Before a reactant can diffuse into a pellet it must cross the boundary layer around it, and at high
temperature or on very active catalysts that film is the bottleneck: the reaction becomes "mass-transfer
limited", indifferent to temperature and to the catalyst itself. This notebook estimates mass-transfer
coefficients from correlations, combines external and internal resistances into an overall effectiveness
factor, tests for external limitations with the Mears criterion, and designs a mass-transfer-limited bed.
"""),
    setup_cell(["from engreact import catalysis, kinetics"]),
    md("## Mass-transfer coefficients from the Frossling correlation"),
    code(r"""D_AB, nu, d_p = 1.5e-5, 1.6e-5, 3e-3                              # gas at ~400 K: diffusivity, kinematic viscosity, pellet diameter
Sc = nu / D_AB
print(f"Schmidt number {Sc:.2f}")
rows = []
for U in (0.05, 0.2, 0.5, 1.0, 2.0):
    Re = U * d_p / nu
    Sh = catalysis.sherwood_frossling(Re, Sc)
    rows.append((U, Re, Sh, Sh * D_AB / d_p))
print(pd.DataFrame(rows, columns=["U (m/s)", "Re", "Sh", "k_c (m/s)"]).round(3).to_string(index=False))"""),
    md(r"""
The mass-transfer coefficient grows with the square root of velocity (Frossling: Sh = 2 + 0.6 Re$^{1/2}$Sc$^{1/3}$)
and never falls below the stagnant limit Sh = 2. Doubling the gas velocity raises $k_c$ by 40 % - the reason
a mass-transfer-limited reactor responds to flow rate and not to temperature.

## Combining the resistances
"""),
    code(r"""k_c, a_c = 0.05, 1200.0                                             # m/s, m2 per m3 of bed (3 mm spheres, phi = 0.4)
for k_int, label in ((0.1, "slow kinetics"), (10.0, "moderate"), (1000.0, "very fast")):
    for eta in (1.0, 0.3):
        Omega = catalysis.overall_effectiveness(eta, k_int, k_c, a_c)
        print(f"{label:<14} k = {k_int:7.1f} 1/s, eta = {eta}: overall effectiveness Omega = {Omega:.4f}; observed rate = {Omega * k_int:8.3f} x C_bulk")"""),
    md(r"""
The observed rate is $\Omega k C_b$ with $\Omega = \eta/(1 + \eta k/(k_c a_c))$: for slow kinetics the film does
not matter and $\Omega = \eta$; for very fast kinetics the observed rate saturates at $k_c a_c C_b$ regardless
of $k$ or $\eta$. Three resistances in series - film, pore, surface - and the largest one rules.

## The Mears criterion
"""),
    code(r"""for r_obs in (1e-3, 1e-2, 1e-1, 1.0):
    M = catalysis.mears_criterion(r_obs, rho_b=900.0, R=1.5e-3, order=1, k_c=0.05, C_b=20.0)
    print(f"observed rate {r_obs:.0e} mol/(kg s): Mears number {M:.3f} -> " + ("external mass transfer negligible" if M < 0.15 else "external mass transfer matters"))"""),
    md(r"""
Mears' test, like Weisz-Prater's for internal diffusion, uses only observables: if $-r_{obs}\rho_b R n/(k_c C_b) < 0.15$,
the film drop in concentration is below 5 % and the measurement reflects the catalyst. Both criteria belong
in every laboratory report on a solid catalyst.

## A mass-transfer-limited packed bed
"""),
    code(r"""L = np.linspace(0, 0.25, 100)
fig, ax = plt.subplots()
for U in (0.2, 0.5, 1.0):
    Re = U * d_p / nu
    k_c_ = catalysis.sherwood_frossling(Re, Sc) * D_AB / d_p
    ax.plot(L, catalysis.external_limited_conversion(k_c_, a_c, U, L), label=f"U = {U} m/s")
ax.set(xlabel="bed length (m)", ylabel="conversion", title="mass-transfer-limited bed: X = 1 - exp(-k_c a_c L / U)"); ax.legend(); plt.show()
U = 0.5
k_c_ = catalysis.sherwood_frossling(U * d_p / nu, Sc) * D_AB / d_p
L99 = -np.log(0.01) * U / (k_c_ * a_c)
print(f"at U = 0.5 m/s the bed length for 99 % conversion is {L99:.2f} m; doubling the velocity needs {(-np.log(0.01) * 1.0 / (catalysis.sherwood_frossling(1.0 * d_p / nu, Sc) * D_AB / d_p * a_c)):.2f} m")"""),
    md(r"""
When the film controls, the bed behaves like an absorber: conversion depends on $k_c a_c L/U$, and since
$k_c \propto U^{1/2}$ the required length grows as $U^{1/2}$. Catalytic converters and combustion catalysts run
here - and the apparent activation energy of such a reactor is a few kJ/mol, the temperature dependence of
diffusion, which is how one recognises the regime in the plant.
"""),
]

NOTEBOOKS["13_catalyst_deactivation_and_reactor_operation"] = [
    md(r"""
# 13 · Catalyst deactivation and reactor operation

Catalysts lose activity by sintering, coking and poisoning, and a reactor designed for fresh catalyst falls
short within days. This notebook models the decay laws and their mechanisms, follows a packed bed as its
catalyst dies (the conversion falls, then the temperature is raised to compensate), sizes the catalyst
inventory for a cycle, and looks at the moving-bed and fluidised-bed designs that handle catalysts too
short-lived for fixed beds.
"""),
    setup_cell(["from engreact import catalysis, datasets, kinetics"]),
    md("## Decay laws and mechanisms"),
    code(r"""t = np.linspace(0, 300, 300)
fig, ax = plt.subplots()
for law, lbl in (("first", "first order (sintering-like)"), ("second", "second order (coking-like)"), ("linear", "linear (poisoning by a feed impurity)")):
    ax.plot(t, catalysis.activity(t, 0.01, law), label=lbl)
ax.set(xlabel="time on stream (h)", ylabel="activity a(t)", ylim=(0, 1.02)); ax.legend(); plt.show()
d = pd.read_csv(datasets.path("catalyst_deactivation.csv"), comment="#")
fit = catalysis.fit_deactivation(d.t_h, d.activity, "second")
print(f"the course data: second-order decay with k_d = {fit['k_d']:.4f} 1/h -> activity 0.5 after {1/fit['k_d']:.0f} h, 0.2 after {4/fit['k_d']:.0f} h")"""),
    md(r"""
The three laws have different tails: first-order decay is exponential, second-order slows itself (the
remaining activity decays more slowly as coke covers the surface), linear poisoning marches to zero. The
fitted second-order constant gives the cycle length directly - and for a coking catalyst the cycle ends with
a regeneration by burning the coke off.

## A packed bed with a dying catalyst
"""),
    code(r"""k0, Ea = 1e8, 50e3
W_over_F = 5.0                                                       # kg s/mol, the bed's catalyst-to-feed ratio... in "tau" units k' W/F
def conversion(T, a):
    return 1 - np.exp(-kinetics.arrhenius(k0, Ea, T) * a * W_over_F * 1e-3)   # first order, per kg of catalyst
hours = np.linspace(0, 200, 201)
a = catalysis.activity(hours, fit["k_d"], "second")
fig, ax = plt.subplots()
ax.plot(hours, conversion(500.0, a), label="constant T = 500 K: conversion decays")
traj = catalysis.temperature_time_trajectory(k_d0=fit["k_d"] / 3600 / np.exp(-60e3 / (8.314 * 500.0)), E_d=60e3, k0=k0, Ea=Ea, T0=500.0, t_max=200 * 3600, T_max=545.0)
ax.plot(traj["t"] / 3600, [conversion(T_, a_) for T_, a_ in zip(traj["T"], traj["a"])], label="temperature raised to hold conversion")
ax.set(xlabel="time on stream (h)", ylabel="conversion"); ax.legend(); plt.show()
print(f"holding conversion needs the temperature to reach 545 K after {traj['t'][-1]/3600:.0f} h; raising T also accelerates the decay (its own activation energy) - the climb steepens")"""),
    md(r"""
At constant temperature the conversion of a first-order reaction sags as the activity does. Raising the
temperature restores it - until the maximum allowable temperature, and with an accelerating climb because
the deactivation reaction is itself thermally activated. Reformers, hydrotreaters and ammonia converters run
this schedule over months; the end-of-run temperature defines the cycle.

## Sizing the catalyst inventory for a cycle
"""),
    code(r"""target, cycle_h = 0.9, 720.0                                        # 90 % conversion for a 30-day cycle at constant T
a_end = catalysis.activity(cycle_h, fit["k_d"], "second")
k500 = kinetics.arrhenius(k0, Ea, 500.0)
W_over_F_needed = -np.log(1 - target) / (k500 * a_end * 1e-3)
W_over_F_fresh = -np.log(1 - target) / (k500 * 1e-3)
print(f"activity at the end of the cycle: {a_end:.3f}; catalyst needed: {W_over_F_needed:.1f} kg s/mol against {W_over_F_fresh:.1f} for fresh catalyst - {W_over_F_needed/W_over_F_fresh:.1f} x")"""),
    md(r"""
A bed sized for fresh catalyst meets specification on day one and fails by the end of the month. Sizing for
the end-of-cycle activity means installing several times the "kinetic" inventory - the true cost of
deactivation, and the incentive for the alternatives below.

## Moving beds and fluidised beds
"""),
    code(r"""# moving bed: catalyst and gas flow together (co-current); the activity at the outlet is a(t_residence)
U_s = np.array([0.5, 1.0, 2.0, 5.0])                                 # catalyst mass flow (kg/h) through a 100 kg bed
for u in U_s:
    t_res = 100.0 / u
    a_out = catalysis.activity(t_res, fit["k_d"], "second")
    a_mean = np.mean(catalysis.activity(np.linspace(0, t_res, 200), fit["k_d"], "second"))
    print(f"catalyst flow {u:4.1f} kg/h: residence {t_res:6.0f} h, outlet activity {a_out:.3f}, mean activity in the bed {a_mean:.3f}")
print("\nfluidised bed with continuous regeneration: the catalyst is well mixed, so its activity is a single steady value set by the regeneration rate (FCC: minutes of residence, coke burned off in the regenerator)")"""),
    md(r"""
When the catalyst dies in hours rather than months, the fixed bed gives way to a moving bed (catalyst
slowly circulated and regenerated, as in continuous catalytic reforming) or a fluidised bed with a
regenerator (fluid catalytic cracking, where the catalyst lives for seconds between regenerations). The
choice of reactor follows from the deactivation time scale - one more reason to measure it early.
"""),
]

NOTEBOOKS["14_residence_time_distributions"] = [
    md(r"""
# 14 · Residence-time distributions

Real reactors are neither perfectly mixed nor perfectly plug-flow: some fluid short-circuits, some sits in
dead zones, and a tubular reactor has a velocity profile. The residence-time distribution, measured with a
tracer, characterises this without opening the vessel. This notebook extracts E(t) and F(t) from pulse and
step experiments, computes their moments, compares them with the ideal reactors, and diagnoses bypassing
and dead volume - starting from tracer data with the baseline drift real detectors have.
"""),
    setup_cell(["from engreact import datasets, rtd"]),
    md("## A pulse experiment"),
    code(r"""d = pd.read_csv(datasets.path("rtd_tracer_pulse.csv"), comment="#")
fig, ax = plt.subplots(); ax.plot(d.t_s, d.C_g_m3, "o-", ms=3); ax.set(xlabel="t (s)", ylabel="tracer (g/m3)", title="raw detector signal"); plt.show()
print("late-time baseline:", np.round(d.C_g_m3.to_numpy()[-5:], 3), "- the signal does not return to zero: offset and drift")"""),
    md(r"""
The tail never quite reaches zero: detectors drift, and a baseline offset integrated over a long tail adds
spurious area at large t, inflating the mean residence time and, much more, the variance. Correcting the
baseline is the first step of every RTD analysis.

## Baseline correction and E(t)
"""),
    code(r"""t, C = d.t_s.to_numpy(), d.C_g_m3.to_numpy()
tail = t > 140
base = np.polyfit(t[tail], C[tail], 1)                              # linear baseline from the tail
C_corr = np.maximum(C - np.polyval(base, t), 0.0)
E_raw, E_corr = rtd.E_from_pulse(t, C), rtd.E_from_pulse(t, C_corr)
for label, E in (("raw", E_raw), ("baseline-corrected", E_corr)):
    m = rtd.moments(t, E)
    print(f"{label:<20}: mean residence time {m['mean']:6.1f} s, variance {m['variance']:7.0f} s2, sigma^2/tau^2 = {m['variance']/m['mean']**2:.3f}")
fig, ax = plt.subplots(); ax.plot(t, E_raw, label="raw"); ax.plot(t, E_corr, label="corrected"); ax.set(xlabel="t (s)", ylabel="E(t) (1/s)"); ax.legend(); plt.show()"""),
    md(r"""
The correction moves the mean residence time by a few percent and the variance by much more, because the
variance weights the tail by $t^2$. The corrected mean of about 40 s is the vessel's volume over the flow
rate - a check that the tracer is inert and no fluid is trapped: a mean shorter than V/v$_0$ means dead
volume, longer means the tracer adsorbs.

## E(t), F(t) and the ideal reactors
"""),
    code(r"""m = rtd.moments(t, E_corr)
tau = m["mean"]
F = rtd.F_from_E(t, E_corr)
tt = np.linspace(0, 180, 400)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(t, E_corr, "o", ms=3, label="measured"); ax[0].plot(tt, rtd.cstr_E(tt, tau), label="CSTR"); ax[0].plot(tt, rtd.pfr_E(tt, tau, 2.0), label="PFR (delta at tau)")
ax[0].set(xlabel="t (s)", ylabel="E(t)", ylim=(0, 0.04)); ax[0].legend()
ax[1].plot(t, F, "o-", ms=3, label="F(t) measured"); ax[1].axvline(tau, color="k", ls=":", label="tau"); ax[1].set(xlabel="t (s)", ylabel="F(t)"); ax[1].legend(); plt.show()
print(f"fraction of fluid leaving before 0.5 tau: {np.interp(0.5 * tau, t, F):.3f}; before 2 tau: {np.interp(2 * tau, t, F):.3f} (CSTR: 0.393 and 0.865; PFR: 0 and 1)")"""),
    md(r"""
The vessel is between the ideals: no fluid leaves immediately (unlike a CSTR) but the distribution is
broad (unlike a PFR). A CSTR's E(t) is an exponential and a PFR's a spike at $\tau$; the measured curve's
delayed peak and skewed tail are what the models of the next notebook - tanks in series, dispersion - will
quantify.

## Diagnosing faults from the RTD
"""),
    code(r"""tt = np.linspace(0, 200, 2001)
tau_design = 40.0
def bypass_E(frac, tau_eff):
    E = (1 - frac) * rtd.cstr_E(tt, tau_eff)
    E[:3] += frac / (tt[1] - tt[0]) / 3
    return E
E_dead = rtd.cstr_E(tt, 0.6 * tau_design)
fig, ax = plt.subplots()
ax.plot(tt, rtd.cstr_E(tt, tau_design), label="ideal CSTR, tau = 40 s")
ax.plot(tt, E_dead, "--", label="40 % dead volume: mean 24 s")
ax.plot(tt, bypass_E(0.3, tau_design / 0.7), ":", label="30 % bypass: spike at t = 0, longer tail")
ax.set(xlabel="t (s)", ylabel="E(t)", xlim=(0, 150), ylim=(0, 0.05)); ax.legend(); plt.show()
print(f"mean residence time with 40 % dead volume: {rtd.moments(tt, E_dead)['mean']:.1f} s against V/v0 = 40 s")"""),
    md(r"""
Two classic faults have signatures: dead volume shortens the mean residence time below $V/v_0$ (part of the
vessel does not participate); bypassing puts a spike of tracer at the outlet almost immediately and
lengthens the tail. Both waste reactor volume and both are found by a tracer test before anyone lifts a
manhole cover.
"""),
]

NOTEBOOKS["15_models_for_nonideal_flow"] = [
    md(r"""
# 15 · Models for nonideal flow

Given an RTD, what conversion will the reactor achieve? For a first-order reaction the RTD alone answers;
for any other order it depends also on *how* fluid elements mix, and two limiting models bracket the answer.
This notebook fits the tanks-in-series and axial-dispersion models to the tracer data, predicts conversion
with the segregation and maximum-mixedness models, and shows when the difference matters.
"""),
    setup_cell(["from engreact import datasets, rtd"]),
    md("## Tanks in series"),
    code(r"""d = pd.read_csv(datasets.path("rtd_tracer_pulse.csv"), comment="#")
t, C = d.t_s.to_numpy(), d.C_g_m3.to_numpy()
base = np.polyfit(t[t > 140], C[t > 140], 1)
E = rtd.E_from_pulse(t, np.maximum(C - np.polyval(base, t), 0.0))
fit = rtd.fit_tanks_in_series(t, E)
print(f"tanks-in-series: n = {fit['n_moments']:.2f} from the moments, n = {fit['n']:.2f} and tau = {fit['tau']:.1f} s from the least-squares fit (the file was generated with n = 3.7)")
tt = np.linspace(0, 180, 400)
plt.plot(t, E, "o", ms=3, label="measured"); plt.plot(tt, rtd.tanks_in_series_E(tt, fit["tau"], fit["n"]), label=f"tanks in series, n = {fit['n']:.1f}"); plt.xlabel("t (s)"); plt.ylabel("E(t)"); plt.legend(); plt.show()"""),
    md(r"""
The tanks-in-series model has one parameter: $n$ tanks, with $\sigma^2/\tau^2 = 1/n$. The moments give $n$
directly; a least-squares fit refines it, and a non-integer $n$ is fine for the fit (the gamma function
interpolates). $n$ = 1 is a CSTR, $n \to \infty$ a PFR; this vessel is "about four tanks' worth" of mixing.

## The axial-dispersion model
"""),
    code(r"""m = rtd.moments(t, E)
s2 = m["variance"] / m["mean"] ** 2
Pe = rtd.peclet_from_variance(s2, "closed")
print(f"sigma^2/tau^2 = {s2:.3f} -> Peclet number Pe = uL/D = {Pe:.1f} (closed vessel); tanks-in-series equivalent n ~ Pe/2 = {Pe/2:.1f}")
theta = np.linspace(0, 4, 400)
fig, ax = plt.subplots()
for Pe_ in (1.0, 5.0, Pe, 50.0):
    ax.plot(theta, rtd.dispersion_E(theta, Pe_), label=f"Pe = {Pe_:.0f}")
ax.plot(t / m["mean"], E * m["mean"], "ko", ms=3, label="measured")
ax.set(xlabel="theta = t/tau", ylabel="E(theta)", xlim=(0, 3)); ax.legend(); plt.show()"""),
    md(r"""
The dispersion model treats the reactor as plug flow with an axial diffusion-like spreading; its parameter is
the Peclet number, large for near-plug flow. Its E(θ) has the same skewed shape and the same variance
relation ($\sigma^2/\tau^2 = 2/Pe - 2/Pe^2(1 - e^{-Pe})$ for a closed vessel) - the two models fit tracer data
almost equally well, and for a first-order reaction predict almost the same conversion. Choose by
physics: tanks for stirred stages, dispersion for tubes and packed beds.

## Conversion from the RTD: first order
"""),
    code(r"""k = 0.05                                                            # 1/s
tau = m["mean"]
rows = [("PFR (ideal)", 1 - np.exp(-k * tau)), ("CSTR (ideal)", k * tau / (1 + k * tau)),
        (f"tanks in series, n = {fit['n']:.1f}", 1 - (1 + k * fit["tau"] / fit["n"]) ** (-fit["n"])),
        (f"dispersion, Pe = {Pe:.1f}", rtd.dispersion_conversion(k, tau, Pe)),
        ("segregation model on E(t)", rtd.segregation_conversion(t, E, k)), ("maximum mixedness on E(t)", rtd.maximum_mixedness_conversion(t, E, k))]
print(pd.DataFrame(rows, columns=["model", "conversion (first order)"]).round(4).to_string(index=False))"""),
    md(r"""
For first order every model that reproduces the RTD gives the same conversion, because the reaction
rate is linear in concentration and the order of mixing does not matter: the RTD is a complete description.
The real vessel loses a few percent against the ideal PFR and gains a lot against a CSTR - the practical
meaning of "four tanks in series".

## Second order: mixing earliness matters
"""),
    code(r"""k2, C_A0 = 0.002, 25.0                                              # m3/(mol s), mol/m3: k2 C_A0 tau ~ 2
rows = []
for name, E_model in (("CSTR E(t)", rtd.cstr_E(t, tau)), ("measured E(t)", E), (f"{fit['n']:.1f} tanks", rtd.tanks_in_series_E(t, fit["tau"], fit["n"]))):
    seg = rtd.segregation_conversion(t, E_model, k2, 2, C_A0)
    mm = rtd.maximum_mixedness_conversion(t, E_model, k2, 2, C_A0)
    rows.append((name, seg, mm, seg - mm))
print(pd.DataFrame(rows, columns=["RTD", "segregation (late mixing)", "maximum mixedness (early)", "difference"]).round(4).to_string(index=False))
print(f"ideal CSTR, second order: {(1 + 2*k2*C_A0*tau - np.sqrt(1 + 4*k2*C_A0*tau))/(2*k2*C_A0*tau):.4f} (= maximum mixedness with the CSTR RTD)")"""),
    md(r"""
For a second-order reaction, fluid that reacts as separate clumps (segregation, mixing at the last moment)
converts more than fluid mixed as early as possible, because the rate is convex in concentration. The two
models bracket the truth; the difference is largest for a CSTR-like RTD and shrinks as the vessel approaches
plug flow - for the measured RTD it is a percent. Whether a real reactor is nearer one limit is a question
of micromixing, decided by physics (viscosity, feed injection) rather than by the tracer curve.
"""),
]

NOTEBOOKS["16_bioreactors_and_enzyme_kinetics"] = [
    md(r"""
# 16 · Bioreactors and enzyme kinetics

Enzymes are catalysts with a rate law of their own - saturating, inhibitable - and living cells are catalysts
that make more of themselves. This notebook fits Michaelis-Menten kinetics to initial-rate data (and shows
why the classic linearised plots mislead), models growth with Monod kinetics in batch, chemostat and
fed-batch reactors, and finds the dilution rate at which a chemostat washes out and the one at which it
produces most.
"""),
    setup_cell(["from engreact import bio, datasets"]),
    md("## Michaelis-Menten kinetics from initial rates"),
    code(r"""d = pd.read_csv(datasets.path("enzyme_initial_rates.csv"), comment="#")
S, v = d.S_mmol_L.to_numpy(), d.v_mmol_L_min.to_numpy()
lb, hw, nl = bio.lineweaver_burk(S, v), bio.hanes_woolf(S, v), bio.fit_michaelis_menten(S, v)
print(f"Lineweaver-Burk: V_max = {lb['V_max']:.3f}, K_m = {lb['K_m']:.3f}")
print(f"Hanes-Woolf:     V_max = {hw['V_max']:.3f}, K_m = {hw['K_m']:.3f}")
print(f"nonlinear:       V_max = {nl['V_max']:.3f} +/- {nl['se_V_max']:.3f}, K_m = {nl['K_m']:.3f} +/- {nl['se_K_m']:.3f}   (true 1.8, 0.35)")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
Sf = np.linspace(0.01, 7, 200)
ax[0].plot(S, v, "o", label="data"); ax[0].plot(Sf, bio.michaelis_menten(Sf, nl["V_max"], nl["K_m"]), label="nonlinear fit"); ax[0].set(xlabel="S (mmol/L)", ylabel="v (mmol/(L min))"); ax[0].legend()
ax[1].plot(1 / S, 1 / v, "o"); ax[1].plot(1 / Sf[Sf > 0.04], 1 / bio.michaelis_menten(Sf[Sf > 0.04], lb["V_max"], lb["K_m"])); ax[1].set(xlabel="1/S", ylabel="1/v", title="Lineweaver-Burk: the smallest rates dominate"); plt.show()"""),
    md(r"""
Lineweaver-Burk straightens the hyperbola but weights the points with the smallest rates - the least
precise - most heavily, and its $K_m$ is off by a third here; Hanes-Woolf is better, and nonlinear regression
on the untransformed data is right, with standard errors. The linearised plots are for looking, not for
reporting - the same conclusion as notebook 05 for rate laws.

## Inhibition
"""),
    code(r"""Sf = np.linspace(0.01, 10, 300)
fig, ax = plt.subplots()
ax.plot(Sf, bio.michaelis_menten(Sf, 1.8, 0.35), "k", label="no inhibitor")
for kind in ("competitive", "uncompetitive", "noncompetitive"):
    ax.plot(Sf, bio.michaelis_menten(Sf, 1.8, 0.35, inhibitor=1.0, K_I=0.5, inhibition=kind), label=kind)
ax.set(xlabel="S", ylabel="v", title="I = 1, K_I = 0.5"); ax.legend(); plt.show()
print("competitive: same V_max, higher apparent K_m (overcome by more substrate); noncompetitive: lower V_max; uncompetitive: both lower")"""),
    md(r"""
A competitive inhibitor competes for the active site and can be out-competed by substrate; a
noncompetitive one disables the enzyme wherever it binds; an uncompetitive one binds only the
enzyme-substrate complex. Drug design is largely the design of inhibitors, and their type is read from how
the apparent $V_{max}$ and $K_m$ shift.

## Monod growth in a batch culture
"""),
    code(r"""mu_max, K_s, Y = 0.4, 0.2, 0.5                                     # 1/h, g/L, g cells per g substrate
t = np.linspace(0, 30, 300)
bg = bio.batch_growth(0.05, 20.0, mu_max, K_s, Y, t)
fig, ax = plt.subplots(); ax.plot(t, bg["X"], label="biomass X"); ax.plot(t, bg["S"], label="substrate S"); ax.set(xlabel="t (h)", ylabel="g/L"); ax.legend(); plt.show()
print(f"lag-free exponential growth at mu_max: doubling time {np.log(2)/mu_max:.2f} h; final biomass {bg['X'][-1]:.2f} g/L = X0 + Y S0 = {0.05 + Y * 20:.2f}")"""),
    md(r"""
Monod's law is Michaelis-Menten for cells: growth is exponential while substrate is plentiful and stops when
it is gone, and the yield coefficient fixes how much biomass a gram of substrate makes. Real cultures add a
lag phase, death, and product inhibition (ethanol stops yeast at about 12 %), but Monod is the skeleton.

## The chemostat: steady growth, washout and the optimum
"""),
    code(r"""S_in = 20.0
Ds = np.linspace(0.005, 0.42, 300)
ss = [bio.chemostat_steady(D, S_in, mu_max, K_s, Y) for D in Ds]
X_ss, S_ss, prod = np.array([s["X"] for s in ss]), np.array([s["S"] for s in ss]), np.array([s.get("productivity", 0.0) for s in ss])
D_wash, D_opt = bio.chemostat_washout(mu_max, K_s, S_in), bio.chemostat_optimum(mu_max, K_s, S_in)
fig, ax = plt.subplots(); ax.plot(Ds, X_ss, label="biomass X"); ax.plot(Ds, S_ss, label="substrate S"); ax.plot(Ds, prod, label="productivity D X (g/(L h))")
ax.axvline(D_wash, color="k", ls=":", label=f"washout D = {D_wash:.3f}"); ax.axvline(D_opt, color="C3", ls="--", label=f"optimum D = {D_opt:.3f}"); ax.set(xlabel="dilution rate D (1/h)", ylabel="g/L"); ax.legend(fontsize=8); plt.show()
print(f"at D = 0.2 1/h: S = {bio.chemostat_steady(0.2, S_in, mu_max, K_s, Y)['S']:.3f} g/L, X = {bio.chemostat_steady(0.2, S_in, mu_max, K_s, Y)['X']:.2f} g/L; the culture washes out above D = {D_wash:.3f} 1/h")"""),
    md(r"""
In a chemostat the cells grow exactly as fast as they are washed out, $\mu(S) = D$, so the substrate
concentration is set by the dilution rate and the biomass by the yield. Too fast a dilution and the culture
cannot keep up: washout, sharp and total. The productivity has a maximum just below washout - the
operating point of a continuous fermenter, run with a margin, because the cliff is close.

## Fed-batch
"""),
    code(r"""fb = bio.fed_batch(X0=5.0, S0=0.5, V0=1.0, S_feed=200.0, F=0.02, mu_max=mu_max, K_s=K_s, Y_xs=Y, t=np.linspace(0, 40, 400))
fig, ax = plt.subplots(); ax.plot(fb["t"], fb["X"], label="X (g/L)"); ax.plot(fb["t"], fb["S"], label="S (g/L)"); ax.plot(fb["t"], fb["V"], "k:", label="V (L)"); ax.set(xlabel="t (h)"); ax.legend(); plt.show()
print(f"substrate stays low ({fb['S'].max():.3f} g/L at most) while the biomass rises to {fb['X'][-1]:.1f} g/L in a volume of {fb['V'][-1]:.1f} L")"""),
    md(r"""
Feeding concentrated substrate slowly keeps its concentration low - avoiding overflow metabolism, oxygen
limitation and substrate inhibition - while the biomass accumulates to densities a batch cannot reach. Baker's
yeast, penicillin and most therapeutic proteins are made this way; the feed profile is the process.
"""),
]

NOTEBOOKS["17_project_pilot_plant_to_reactor_design"] = [
    md(r"""
# 17 · Project: from pilot-plant data to a reactor design and safety report

**Problem.** A pilot plant has run the exothermic liquid-phase reaction A -> B in a jacketed 0.5 m$^3$ CSTR
at 25 combinations of feed temperature and space time, and the laboratory has batch kinetics at three
temperatures. The files have the usual faults, and one run ended in a trip. The task: extract the rate law and
its activation energy, reconcile the pilot runs with it, explain the trip, and design a 10 m$^3$ production
reactor with a stated safe operating window - written up so a reviewer can reproduce every number.
"""),
    setup_cell(["from engreact import datasets, energy, kinetics, rate_data"]),
    md("## 1. Kinetics from the batch runs"),
    code(r"""b = pd.read_csv(datasets.path("pilot_batch_kinetics.csv"), comment="#")
fits = {}
for T, g in b.groupby("T_K"):
    f = rate_data.nonlinear_fit(g.t_s, g.C_A_mol_m3, order=1, fit_C0=True)
    free = rate_data.nonlinear_fit(g.t_s, g.C_A_mol_m3)
    fits[T] = f
    print(f"T = {T:.0f} K: first order k = {f['k']:.3e} +/- {f['se_k']:.1e} 1/s (free-order fit: n = {free['order']:.2f} +/- {free['se_order']:.2f})")
arr = kinetics.fit_arrhenius(list(fits), [fits[T]["k"] for T in fits])
print(f"\nArrhenius: Ea = {arr['Ea']/1e3:.1f} +/- {arr['se_Ea']/1e3:.1f} kJ/mol, k0 = {arr['k0']:.2e} 1/s (the generator used 68 kJ/mol, 5.0e7 1/s)")"""),
    md(r"""
Three isothermal batch runs give first-order kinetics: at 315 and 330 K the free-order fits agree with order
one, while at 300 K the run reaches only 40 % conversion in two hours and the order is poorly determined
(0.4 +/- 0.4) - a slow run cannot tell orders apart, and the hotter runs settle it. The Arrhenius fit gives an
activation energy near 69 kJ/mol; three points is the minimum for an activation energy, and the standard
error says how much to trust it.

## 2. Look at the pilot runs
"""),
    code(r"""p = pd.read_csv(datasets.path("pilot_cstr_runs.csv"), comment="#")
print(p.describe().round(3).to_string())
odd = p[(p.X > 1.05) | (p.T_out_K < 200)]
print("\nrows that cannot be right:\n", odd.to_string(index=False))
print("\nrun with a note:\n", p[p.note != "ok"].to_string(index=False))"""),
    md("## 3. Clean, with a log"),
    code(r"""log = []
q = p.copy()
i = q.index[q.X > 1.05]; q.loc[i, "X"] = q.loc[i, "X"] / 100; log.append(f"run {int(q.loc[i, 'run'].iloc[0])}: conversion was entered in percent, divided by 100")
i = q.index[q.T_out_K < 200]; q.loc[i, "T_out_K"] = q.loc[i, "T_out_K"] + 273.15; log.append(f"run {int(q.loc[i, 'run'].iloc[0])}: outlet temperature was in degC, converted to K")
runaway = q[q.note != "ok"]; q = q[q.note == "ok"]; log.append(f"run {int(runaway.run.iloc[0])} (runaway) set aside for the safety analysis")
print("\n".join(log))
print(f"\n{len(q)} steady-state runs; conversions {q.X.min():.3f}-{q.X.max():.3f}, outlet temperatures {q.T_out_K.min():.0f}-{q.T_out_K.max():.0f} K")"""),
    md("## 4. Do the pilot runs agree with the kinetics? Extracting the heat of reaction and the jacket"),
    code(r"""k0, Ea = arr["k0"], arr["Ea"]
q = q.copy()
q["X_pred_isothermal"] = kinetics.arrhenius(k0, Ea, q.T_out_K) * q.tau_s / (1 + kinetics.arrhenius(k0, Ea, q.T_out_K) * q.tau_s)
print("conversion predicted at the measured outlet T vs measured X (should agree if the kinetics transfer):")
print(q[["run", "T_feed_K", "tau_s", "T_out_K", "X", "X_pred_isothermal"]].round(3).to_string(index=False))
# energy balance: rho cp (1 + kappa)(T_out - T_c) = (-dH) C_A0 X  ->  regress T_out - T_feed... use the low-conversion runs: T_out = T_c + X (-dH) C_A0/(rho cp (1+kappa))
low = q[q.X < 0.2]
A = np.vstack([np.ones(len(low)), low.X, low.T_feed_K]).T
coef, *_ = np.linalg.lstsq(A, low.T_out_K, rcond=None)
print(f"\nenergy balance regression on the low-conversion runs: T_out = {coef[0]:.1f} + {coef[1]:.1f} X + {coef[2]:.3f} T_feed")
kappa_est = 1 / coef[2] - 1
dTad_over = coef[1]
print(f"-> kappa = UA/(v0 rho cp) = {kappa_est:.2f} (generator: 0.8); (-dH) C_A0 / (rho cp) = {dTad_over * (1 + kappa_est):.0f} K adiabatic rise (generator: 174 K)")"""),
    md(r"""
At the measured outlet temperature the batch kinetics predict the pilot conversions within their scatter -
including the runs that ignited - so the rate law transfers from flask to pilot reactor. The energy balance
of a cooled CSTR is linear in X and $T_{feed}$; regressing the low-conversion runs recovers the jacket
parameter and the adiabatic temperature rise, the two numbers the design needs and the laboratory could not
provide.

## 5. Explaining the trip
"""),
    code(r"""C_A0, cp_vol = 2500.0, 3.6e6
dH = -dTad_over * (1 + kappa_est) * cp_vol / C_A0
T0s = np.linspace(285, 345, 61)
fig, ax = plt.subplots()
for tau, col in ((120.0, "C0"), (240.0, "C1"), (480.0, "C3")):
    ie = energy.ignition_extinction(k0, Ea, dH, C_A0, tau, cp_vol, T0s, kappa_est, 295.0)
    for T0, Ts, sts in zip(ie["T0"], ie["T_ss"], ie["stable"]):
        for T_, st in zip(Ts, sts):
            ax.plot(T0, T_, "o" if st else "x", color=col, ms=3)
    band = T0s[np.array([len(Ts) for Ts in ie["T_ss"]]) == 3]
    print(f"tau = {tau:4.0f} s: multiplicity for T_feed in [{band.min():.0f}, {band.max():.0f}] K -> ignition on heating at about {band.max():.0f} K")
sub = q[q.note == "ok"]
ax.plot(sub.T_feed_K, sub.T_out_K, "k^", ms=6, label="pilot runs")
ax.plot(runaway.T_feed_K, runaway.T_out_K, "r*", ms=12, label="runaway run")
ax.set(xlabel="feed temperature (K)", ylabel="steady-state reactor temperature (K)", title="ignition-extinction diagram from the fitted model (blue 120 s, orange 240 s, red 480 s)"); ax.legend(); plt.show()"""),
    md(r"""
The fitted model reproduces the pilot plant: the runs sit on the extinguished branch except those at the
longest space time and highest feed temperatures, which ignited - and the trip at 330 K and 480 s is exactly
where the model says the low state ceases to exist. The operators did not do anything wrong; the reactor had
no cold steady state to go to.

## 6. Designing the production reactor
"""),
    code(r"""V_prod, v0 = 10.0, 10.0 / 3600                                        # m3, 10 m3/h feed -> tau = 3600 s
tau_prod = V_prod / v0
T0s_ = np.linspace(260, 340, 41)
rows = []
for kappa_new in (kappa_est, 1.5, 2.0, 3.0, 4.0):
    ss = energy.cstr_steady_states(k0, Ea, dH, C_A0, 300.0, tau_prod, cp_vol, kappa_new, 295.0, T_range=(250.0, 900.0))
    n = np.array([len(energy.cstr_steady_states(k0, Ea, dH, C_A0, T0, tau_prod, cp_vol, kappa_new, 295.0, T_range=(250.0, 900.0))["T"]) for T0 in T0s_])
    band = T0s_[n == 3]
    rows.append((round(kappa_new, 2), [round(t) for t in ss["T"]], [round(x, 3) for x in ss["X"]], f"{band.min():.0f}-{band.max():.0f} K" if band.size else "none"))
print("production reactor, tau = 1 h, T_feed = 300 K:")
print(pd.DataFrame(rows, columns=["jacket kappa", "steady-state T (K)", "X", "multiplicity for T_feed in"]).to_string(index=False))"""),
    md(r"""
At the production space time of an hour the pilot's jacket ratio leaves only the ignited state for any feed
warmer than 270 K: the reactor will light off on start-up, run at 400 K and 99.6 % conversion, and cannot be
extinguished by a cold feed - robust, but hot. More jacket capacity lowers the operating temperature, and
too much of it collapses the conversion (38 % at kappa = 4, where the reactor is cooled below its kinetics).
A jacket ratio of about 2 gives 95 % conversion at 355 K with a unique steady state for every feed
temperature above about 272 K: the design point, with 45 K of margin below the pilot's operating temperature
and no fold to fall off.

## 7. The report
"""),
    code(r"""kappa_design = 2.0
ss_op = energy.cstr_steady_states(k0, Ea, dH, C_A0, 300.0, tau_prod, cp_vol, kappa_design, 295.0, T_range=(250.0, 900.0))
report = f'''# Production reactor for A -> B: design basis

Kinetics (3 isothermal batch runs, nonlinear regression): first order; Ea = {Ea/1e3:.1f} +/- {arr['se_Ea']/1e3:.1f} kJ/mol, k0 = {k0:.2e} 1/s.
Pilot data: {len(p)} runs; {'; '.join(log)}. Kinetics reproduce the pilot conversions at the measured outlet temperatures.
Energy balance (regression on {len(low)} low-conversion pilot runs): adiabatic rise {dTad_over*(1+kappa_est):.0f} K (Delta H = {dH/1e3:.0f} kJ/mol), jacket kappa = {kappa_est:.2f}.
The pilot trip (run {int(runaway.run.iloc[0])}, T_feed 330 K, tau 480 s) is the loss of the extinguished steady state predicted by the model.

Design: V = 10 m3, feed 10 m3/h (tau = 1 h), T_feed = 300 K, jacket sized for kappa = UA/(v0 rho cp) = {kappa_design}: unique steady state at {ss_op['T'][-1]:.0f} K, X = {ss_op['X'][-1]:.3f}, for every feed temperature above 272 K.
Safety: with the pilot's jacket ratio ({kappa_est:.2f}) the reactor would run at 400 K; with kappa = 4 conversion collapses. The design point sits between, with no fold within the operating range. High-temperature trip at {ss_op['T'][-1]+30:.0f} K; the coolant-failure case (adiabatic time to maximum rate, notebook 09) is to be run before commissioning.
'''
print(report)"""),
    md(r"""
Every number in the report traces to a file and a method, the pilot trip is explained rather than blamed,
and the safety recommendation is quantitative: the jacket ratio that removes the multiplicity, a trip
setting and the failure case still to be run. A design that cannot explain the pilot plant should not be
built; one that can, and states its window, is engineering.
"""),
]

# =====================================================================================================
# Teaching layer
def _x(objectives, prereq, time, inside, exercises=(), implement=""):
    return dict(objectives=objectives, prereq=prereq, time=time, inside=inside, exercises=list(exercises),
                implement=implement)





EXTRAS["00_python_for_reaction_engineering"] = _x(
    ["work in SI units and convert at the boundaries", "compute Arrhenius rate constants and read an Arrhenius plot",
     "use quad, brentq and solve_ivp for design integrals, algebraic balances and batch reactors", "read the course data and their provenance"],
    "basic Python", "45 min",
    [md(r"""
## Inside the algorithm: Arrhenius in one line
"""), code(r"""from engreact.constants import R
print("by hand:", 5.0e7 * np.exp(-68e3 / (R * 320.0)), " library:", float(kinetics.arrhenius(5.0e7, 68e3, 320.0)))""")],
    ["A reaction has Ea = 100 kJ/mol. By what factor does the rate rise between 25 and 35 degC, and between 300 and 310 degC? "
     "Why is the 'rate doubles per 10 degrees' rule only a rule of thumb?",
     "Integrate the batch reactor for a second-order reaction (-r = k C^2, k = 0.11 L/(mol s), C0 = 0.02 mol/L) with solve_ivp "
     "and compare with the closed form C = C0/(1 + k C0 t) at 5, 10 and 20 min."],
    "**Implement it yourself:** write a function that returns the temperature at which a rate constant reaches a given value "
    "(invert Arrhenius) and check it against `kinetics.temperature_for_rate_ratio`.")

EXTRAS["01_mole_balances_and_ideal_reactors"] = _x(
    ["derive the batch, CSTR, PFR and PBR design equations from the general balance", "solve them for first-order kinetics",
     "size reactors for any rate law given as a function of conversion", "integrate a multi-species batch reactor numerically"],
    "notebook 00", "60 min",
    [md(r"""
## Inside the algorithm: the PFR design integral
"""), code(r"""from scipy import integrate
val, _ = integrate.quad(lambda x: 1 / (k * C_A0 * (1 - x)), 0, 0.9)
print("by hand: V =", F_A0 * val, "m3; library:", ideal.pfr_volume(F_A0, 0.9, rate))""")],
    ["A liquid-phase reaction with -r_A = k C_A^0.5 (k = 0.05 (mol/m3)^0.5/s, C_A0 = 400 mol/m3, F_A0 = 1 mol/s) is to reach 90 % "
     "conversion. Size a CSTR and a PFR, and explain why the ratio is smaller than for first order.",
     "For a zero-order reaction, show numerically that the CSTR and PFR volumes are equal for any conversion, and explain why "
     "with the Levenspiel picture."],
    "**Implement it yourself:** write the CSTR balance for a first-order reaction as a residual function and solve it with "
    "brentq for X at tau = 20 min; compare with `ideal.cstr_conversion_first_order`.")

EXTRAS["02_conversion_and_reactor_sizing"] = _x(
    ["construct Levenspiel plots from rate data", "read CSTR and PFR volumes from them",
     "size reactors in series and choose their order", "recognise when a CSTR should precede a PFR"],
    "notebook 01", "60 min",
    [md(r"""
## Inside the algorithm: trapezoidal area under the Levenspiel curve
"""), code(r"""y = F_A0 / rate
area = sum(0.5 * (y[i] + y[i + 1]) * (X[i + 1] - X[i]) for i in range(len(X) - 1))
print("by hand: PFR volume", round(area, 3), "m3; library:", round(lev["V_pfr"][-1], 3))""")],
    ["Using the first rate table, find the minimum total volume of two CSTRs in series reaching X = 0.8 by scanning the "
     "intermediate conversion, and compare with a single CSTR and with a PFR.",
     "Fit the first rate table to -r_A = k C_A (C_A = C_A0 (1 - X)) and recompute the PFR and CSTR volumes from the rate law "
     "instead of the table. How much do they differ from the trapezoidal areas, and why?"],
    "**Implement it yourself:** write `cstr_series(F_A0, volumes, rate_fn)` that solves each stage's balance with brentq "
    "and reproduces `ideal.reactors_in_series` for two CSTRs.")

EXTRAS["03_rate_laws_and_stoichiometry"] = _x(
    ["build stoichiometric tables for liquids and gases", "use the expansion factor with pressure and temperature",
     "compute equilibrium conversions for reversible reactions", "see how Arrhenius and van 't Hoff combine"],
    "notebook 01", "60 min",
    [md(r"""
## Inside the algorithm: the gas-phase concentration
"""), code(r"""X_, eps_, C0_ = 0.5, 1.0, 40.0
print("by hand:", C0_ * (1 - X_) / (1 + eps_ * X_), " library:", float(kinetics.concentration_from_conversion(C0_, X_, eps_)))""")],
    ["Ammonia synthesis N2 + 3H2 <-> 2NH3 with a stoichiometric feed at 450 degC: compute Kp from `thermo`, convert to Kc at "
     "100 bar, and find the equilibrium conversion (gas phase, epsilon = -0.5). Then repeat at 300 bar and at 400 degC to see "
     "why the process runs at high pressure.",
     "For A + B -> C in the liquid phase with C_A0 = 1 mol/L, compare the rate law -r = k C_A C_B as a function of X for "
     "Theta_B = 1, 2 and 5. Explain why excess B makes the reaction look first order."],
    "**Implement it yourself:** write the stoichiometric table for 2A + B -> 3C in the gas phase with a 2:1:1 feed of A:B:inert "
    "by hand (Theta, nu, epsilon) and check every column against `kinetics.stoichiometric_table` at X = 0.4.")

EXTRAS["04_isothermal_reactor_design"] = _x(
    ["design a batch process and its cycle", "size CSTRs and PFRs for nonlinear kinetics",
     "include volume change and Ergun pressure drop in a packed bed", "understand recycle reactors"],
    "notebooks 01-03", "75 min",
    [md(r"""
## Inside the algorithm: the coupled bed equations
"""), code(r"""from scipy import integrate
def rhs(W, s):
    X, y = s
    return [rate_pbr(X, y) / F_A0, -alpha * (1 + eps * X) / (2 * y)]
sol = integrate.solve_ivp(rhs, (0, 100.0), [0.0, 1.0], rtol=1e-9)
print("by hand: X =", round(sol.y[0][-1], 4), "y =", round(sol.y[1][-1], 4), "; library:", round(pb["X"][-1], 4), round(pb["y"][-1], 4))""")],
    ["The packed bed's pellets are replaced by 8 mm ones (halving the Ergun gradient at the same flow... compute it). Recompute "
     "alpha and the outlet conversion. If the reaction is diffusion-limited inside the pellets (notebook 11), what else changes?",
     "A membrane reactor removes product B from A <-> B + C (K_c small) as it forms. Model it with `multiple.Network` in a PFR "
     "with an extra removal term for B (k_m C_B) and show the conversion exceeding the equilibrium value."],
    "**Implement it yourself:** compute the batch cycle-time optimum: for a fixed dead time of 30 min, scan the conversion at "
    "which the batch is stopped and find the one maximising daily production for the saponification.")

EXTRAS["05_analysis_of_rate_data"] = _x(
    ["apply the integral, differential, half-life and initial-rate methods", "fit rate laws by nonlinear regression with uncertainties",
     "determine activation energies from Arrhenius series", "detect faulty samples and unit errors from the plots"],
    "notebooks 00 and 03", "75 min",
    [md(r"""
## Inside the algorithm: the integral method for first order
"""), code(r"""y = np.log(C[mask][0] / C[mask])
k_hand = np.sum(t[mask] * y) / np.sum(t[mask] ** 2)          # least squares through the origin
print("by hand: k =", k_hand, " library:", rate_data.integral_method(t[mask], C[mask], 1)["k"])""")],
    ["Fit the N2O5 data with the mis-read sample included, using nonlinear regression, and compare the fitted k and its standard "
     "error with the fit excluding it. Does the standard error 'notice' the outlier? What would a robust loss do?",
     "The Arrhenius parameters table lists A and Ea for cyclopropane isomerisation. Compute k at 450, 500 and 550 degC, the "
     "half-life at each, and the temperature at which the half-life is one minute."],
    "**Implement it yourself:** write the half-life method (a straight line of ln t_1/2 against ln C0) and apply it to "
    "half-lives you compute from the saponification fit at four starting concentrations.")

EXTRAS["06_multiple_reactions_and_selectivity"] = _x(
    ["analyse series and parallel reactions and their optimum operating points", "compute selectivity and yield and distinguish them",
     "choose reactor type, concentration and temperature for selectivity", "integrate reaction networks in batch, PFR, CSTR and semibatch"],
    "notebooks 01 and 03", "75 min",
    [md(r"""
## Inside the algorithm: Bateman's solution
"""), code(r"""tt = 3.0
B_hand = 1.0 * k1 / (k2 - k1) * (np.exp(-k1 * tt) - np.exp(-k2 * tt))
print("by hand:", B_hand, " library:", float(multiple.series_batch(k1, k2, 1.0, tt)["B"]))""")],
    ["For A -> B -> C with k1 = 0.5 and k2 = 0.2 1/min, find the CSTR residence time and the PFR space time that each give "
     "90 % conversion of A, and compare the yields of B. Which reactor would you choose if B is the product, and if C is?",
     "Two parallel reactions A -> D (first order, Ea = 40 kJ/mol) and A -> U (second order, Ea = 80 kJ/mol). Using `Network`, "
     "find the combination of temperature (300-400 K) and feed concentration (0.1-10 mol/L) that maximises the selectivity "
     "D/U in a PFR at 90 % conversion."],
    "**Implement it yourself:** solve A -> B -> C in a batch reactor with solve_ivp and locate the time of maximum B "
    "numerically; compare with `multiple.series_optimum_time`.")

EXTRAS["07_energy_balances_and_adiabatic_reactors"] = _x(
    ["apply the adiabatic energy balance and the X-T line", "compute equilibrium conversion curves for exothermic reactions",
     "explain and simulate interstage cooling", "contrast exothermic and endothermic adiabatic operation"],
    "notebooks 03 and 04", "75 min",
    [md(r"""
## Inside the algorithm: the adiabatic line
"""), code(r"""X_ = 0.6
print("by hand: T =", 300.0 + X_ * dT_ad, "K; library profile at X = 0.6:", float(np.interp(X_, ad["X"], ad["T"])))""")],
    ["Ammonia synthesis with a stoichiometric feed at 200 bar: compute the equilibrium conversion curve X_e(T) from `thermo` "
     "between 600 and 900 K, draw the adiabatic line from 650 K (Delta T_ad about 200 K per unit conversion), and read the "
     "maximum single-bed conversion. How many beds with cooling to 650 K reach 25 % conversion?",
     "An endothermic first-order reaction (Ea = 90 kJ/mol, Delta H = +80 kJ/mol) is run in a PFR heated by a jacket at 750 K "
     "with Ua = 500 W/(m3 K). Using `energy.pfr_heat_exchange`, find the reactor volume for 80 % conversion and compare with "
     "the adiabatic case."],
    "**Implement it yourself:** integrate the adiabatic PFR yourself with solve_ivp, substituting the energy balance for T(X) "
    "inside the rate, and reproduce `energy.adiabatic_pfr` for T0 = 300 K.")

EXTRAS["08_cstr_multiple_steady_states"] = _x(
    ["construct heat-generation and heat-removal curves", "find all steady states and classify their stability",
     "simulate transients to see ignition and extinction", "map the hysteresis and draw design conclusions"],
    "notebooks 04 and 07", "75 min",
    [md(r"""
## Inside the algorithm: the heat-generation curve
"""), code(r"""from engreact import kinetics
T_ = 350.0
kk = kinetics.arrhenius(k0, Ea, T_)
X_ = kk * tau / (1 + kk * tau)
print("by hand: G =", (-dH) * C_A0 * X_ / 1e6, "MJ/m3; library:", float(energy.cstr_heat_curves(k0, Ea, dH, C_A0, 315.0, tau, cp_vol, T_, kappa, T_a)["G"]) / 1e6)""")],
    ["Halve the residence time to 120 s and repeat the ignition-extinction map. Does multiplicity survive? Now double the "
     "feed concentration instead. Explain both results with the adiabatic temperature rise and the Damkohler number.",
     "Find the feed temperature range over which the reactor can be operated on the *ignited* branch at 240 s with kappa = 0.8, "
     "and the start-up procedure (feed temperature schedule) that gets it there from a cold start without exceeding 420 K."],
    "**Implement it yourself:** write the slope stability test dG/dT < dR/dT with a finite difference and apply it to the three "
    "steady states at T0 = 315 K; compare with the library's verdicts.")

EXTRAS["09_unsteady_reactors_and_runaway"] = _x(
    ["simulate batch reactors with jacket cooling and locate the runaway boundary", "apply Semenov's criterion",
     "compute the adiabatic time to maximum rate", "analyse a CSTR coolant failure"],
    "notebook 08", "60-75 min",
    [md(r"""
## Inside the algorithm: Semenov's balance
"""), code(r"""from engreact.constants import R
q = (-dH) * C_A0 * kinetics.arrhenius(k0, Ea, 320.0)
print("by hand: critical UA/V =", q * Ea / (R * 320.0**2) * np.e, "; library:", sem["UA_over_V_critical"])""")],
    ["A semibatch alternative: feed A over 60 min into the reactor at 320 K with UA/V = 1000 W/(m3 K) (use `multiple.Network` "
     "with an energy balance added by hand, or approximate with the batch model by feeding in ten portions). How much lower "
     "is the peak temperature than the all-at-once batch?",
     "Relief sizing: assume the reactor is vented when T exceeds 400 K. For the adiabatic batch started at 320 K, compute the "
     "rate of heat release (W/m3) at the moment of venting and the fraction of A still unreacted."],
    "**Implement it yourself:** write the batch energy balance yourself (two ODEs) and reproduce the peak temperature for "
    "UA/V = 1000 W/(m3 K); then add a term for evaporative cooling that caps T at the solvent's boiling point.")

EXTRAS["10_catalysis_and_heterogeneous_kinetics"] = _x(
    ["use Langmuir isotherms including dissociative and competitive adsorption", "derive and recognise LHHW rate laws",
     "discriminate mechanisms with rate data", "model catalyst deactivation and the temperature-time trajectory"],
    "notebooks 03 and 05", "75 min",
    [md(r"""
## Inside the algorithm: the single-site rate law from site balances
"""), code(r"""K_A_, p_ = 1.5, 2.0
theta_A = K_A_ * p_ / (1 + K_A_ * p_)            # coverage from the site balance
print("by hand: r = k theta_A =", 1.0 * theta_A, "; library:", float(catalysis.lhhw_rate(1.0, p_, K_A=K_A_, mode="single")))""")],
    ["Derive the rate law for A + B -> C when A adsorbs molecularly, B dissociatively, and the surface reaction is rate "
     "limiting. Plot it against p_A at fixed p_B and find the pressure of maximum rate.",
     "The deactivation data are refitted with the nth-order law with n free. Fit n and k_d by nonlinear least squares and "
     "report whether n = 2 is within the uncertainty. How long until the activity falls to 0.3?"],
    "**Implement it yourself:** integrate da/dt = -k_d a^2 with solve_ivp and confirm it reproduces `catalysis.activity` with "
    "the 'second' law.")

EXTRAS["11_internal_diffusion_and_effectiveness_factors"] = _x(
    ["compute Thiele moduli and effectiveness factors for the standard geometries", "interpret concentration profiles in pellets",
     "recognise falsified kinetics", "apply the Weisz-Prater criterion"],
    "notebook 10", "60 min",
    [md(r"""
## Inside the algorithm: the sphere effectiveness factor
"""), code(r"""phi_ = 3.0
print("by hand: eta =", 3 / phi_**2 * (phi_ / np.tanh(phi_) - 1), "; library:", catalysis.effectiveness_factor(phi_, "sphere"))""")],
    ["A first-order reaction (k = 5 1/s per pellet volume, D_e = 2e-9 m2/s) runs on spherical pellets. Find the pellet radius "
     "for eta = 0.9 and for eta = 0.5, and the rate per unit reactor volume in each case if the bed void fraction is 0.4. "
     "Which would you choose given the pressure-drop result of notebook 04?",
     "An eggshell catalyst puts the active phase in a 0.2 mm shell on 3 mm pellets. Treating the shell as a slab of "
     "half-thickness 0.1 mm, compare its effectiveness factor with that of a uniformly impregnated 3 mm sphere for the "
     "kinetics above."],
    "**Implement it yourself:** solve the slab pellet equation y'' = phi^2 y with solve_bvp and compute eta from the flux at "
    "the surface; compare with tanh(phi)/phi for phi = 2.")

EXTRAS["12_external_mass_transfer_and_packed_beds"] = _x(
    ["estimate mass-transfer coefficients from correlations", "combine film, pore and surface resistances",
     "apply the Mears criterion", "design a mass-transfer-limited bed and recognise the regime"],
    "notebook 11", "60 min",
    [md(r"""
## Inside the algorithm: resistances in series
"""), code(r"""eta_, k_, kc_, ac_ = 0.3, 10.0, 0.05, 1200.0
print("by hand: Omega =", 1 / (1 / eta_ + k_ / (kc_ * ac_)), "; library:", catalysis.overall_effectiveness(eta_, k_, kc_, ac_))""")],
    ["A catalytic converter oxidises CO at 800 K on a monolith (k_c a_c = 60 1/s at U = 5 m/s). Find the monolith length for "
     "99 % conversion, and the apparent activation energy you would measure if the film controls (D_AB ~ T^1.75).",
     "For the packed bed of the notebook, compute the pressure drop (Ergun, notebook 04) for the bed lengths needed at U = 0.2, "
     "0.5 and 1.0 m/s. Which velocity minimises the compressor work per unit conversion?"],
    "**Implement it yourself:** write the Frossling correlation and compute k_c for pellets of 1, 3 and 10 mm at U = 0.5 m/s; "
    "explain why small pellets have larger k_c but also a larger pressure drop.")

EXTRAS["13_catalyst_deactivation_and_reactor_operation"] = _x(
    ["fit decay laws and relate them to mechanisms", "predict the conversion of a bed as its catalyst deactivates",
     "size catalyst inventory for a cycle and plan a temperature schedule", "choose between fixed, moving and fluidised beds"],
    "notebook 10", "60 min",
    [md(r"""
## Inside the algorithm: second-order decay
"""), code(r"""t_ = 100.0
print("by hand: a =", 1 / (1 + fit["k_d"] * t_), "; library:", float(catalysis.activity(t_, fit["k_d"], "second")))""")],
    ["Regeneration takes 24 h and restores full activity. For a bed at constant 500 K, find the cycle length that maximises the "
     "time-averaged conversion (production during the cycle divided by cycle + regeneration time).",
     "Poisoning by a feed impurity gives linear decay with k_d proportional to the impurity concentration. If a guard bed "
     "removes 90 % of the impurity, how much longer is the cycle? Compare with the cost of 10 x more catalyst."],
    "**Implement it yourself:** integrate the activity ODE da/dt = -k_d(T) a^2 along the library's temperature-time "
    "trajectory and confirm that the product k(T) a stays constant.")

EXTRAS["14_residence_time_distributions"] = _x(
    ["extract E(t) and F(t) from pulse and step data", "correct baselines and compute moments",
     "compare measured RTDs with the ideal reactors", "diagnose dead volume and bypassing"],
    "notebook 01", "60 min",
    [md(r"""
## Inside the algorithm: the mean residence time from E(t)
"""), code(r"""area = np.trapezoid(E_corr, t) if hasattr(np, "trapezoid") else np.trapz(E_corr, t)
tm = (np.trapezoid(t * E_corr, t) if hasattr(np, "trapezoid") else np.trapz(t * E_corr, t)) / area
print("by hand: mean residence time", round(tm, 2), "s; library:", round(rtd.moments(t, E_corr)["mean"], 2))""")],
    ["A step experiment gives F(t). Simulate one from the corrected E(t) of the notebook (cumulative integral), add 2 % noise, "
     "and recover E(t) by differentiation. Why is the pulse experiment preferred for the variance and the step for the mean?",
     "The vessel's design volume is 4.0 m3 and the flow 0.1 m3/s. Compare with the measured mean residence time. Is there dead "
     "volume, and how much? What would adsorption of the tracer on the walls do to the estimate?"],
    "**Implement it yourself:** write `moments(t, E)` returning mean, variance and skewness with trapezoidal integration and "
    "check it on the CSTR E(t) (skewness 2).")

EXTRAS["15_models_for_nonideal_flow"] = _x(
    ["fit the tanks-in-series and dispersion models to tracer data", "convert between n and Pe",
     "predict first-order conversion from an RTD", "bracket higher-order conversion with segregation and maximum mixedness"],
    "notebook 14", "60-75 min",
    [md(r"""
## Inside the algorithm: the segregation model
"""), code(r"""Xb = 1 - np.exp(-k * t)
seg = (np.trapezoid(Xb * E, t) if hasattr(np, "trapezoid") else np.trapz(Xb * E, t)) / (np.trapezoid(E, t) if hasattr(np, "trapezoid") else np.trapz(E, t))
print("by hand: segregation conversion", round(seg, 4), "; library:", round(rtd.segregation_conversion(t, E, k), 4))""")],
    ["A 20 m packed bed with U = 0.5 m/s and D_ax = 0.02 m2/s: compute Pe, sigma^2/tau^2, the equivalent n, and the first-order "
     "conversion loss against plug flow for k tau = 3. At what Pe is the loss below 1 %?",
     "For a half-order reaction (concave rate), which of segregation and maximum mixedness gives the higher conversion? Test "
     "with the CSTR RTD and explain with the shape of the rate law."],
    "**Implement it yourself:** compute the conversion for n tanks in series analytically for first order, and by feeding the "
    "tanks-in-series E(t) into the segregation model; show they agree for n = 1, 2, 5.")

EXTRAS["16_bioreactors_and_enzyme_kinetics"] = _x(
    ["fit Michaelis-Menten kinetics correctly", "recognise inhibition types", "model batch, chemostat and fed-batch growth",
     "find washout and the optimum dilution rate"],
    "notebooks 05 and 10", "60-75 min",
    [md(r"""
## Inside the algorithm: the chemostat steady state
"""), code(r"""D_ = 0.2
S_ss_hand = K_s * D_ / (mu_max - D_)
print("by hand: S =", S_ss_hand, "X =", Y * (S_in - S_ss_hand), "; library:", bio.chemostat_steady(D_, S_in, mu_max, K_s, Y))""")],
    ["Include a maintenance term m_s = 0.02 g substrate/(g cells h) and a death rate k_d = 0.01 1/h in the batch growth model, "
     "and show how they change the final biomass and the stationary phase.",
     "Two chemostats in series (each D = 0.3 1/h relative to its own volume): compute the outlet substrate and biomass, and "
     "compare the total residence time for 99 % substrate conversion with a single chemostat."],
    "**Implement it yourself:** integrate the batch enzyme reactor dS/dt = -V_max S/(K_m + S) with solve_ivp and compare with "
    "`bio.batch_enzyme` at 5 time points.")

EXTRAS["17_project_pilot_plant_to_reactor_design"] = _x(
    ["extract kinetics and an activation energy from batch data with uncertainties", "clean pilot-plant data and log every repair",
     "reconcile pilot runs with kinetics and an energy balance to extract the heat of reaction and the jacket parameter",
     "explain a runaway with the multiplicity diagram and write a design with a quantified safety window"],
    "notebooks 05, 07, 08 and 09", "90 min",
    [md(r"""
## Inside the algorithm: the CSTR energy balance as a regression
"""), code(r"""x = low.X.to_numpy(); Tf = low.T_feed_K.to_numpy(); To = low.T_out_K.to_numpy()
A_ = np.vstack([np.ones_like(x), x, Tf]).T
c_hand = np.linalg.solve(A_.T @ A_, A_.T @ To)
print("by hand:", np.round(c_hand, 3), "; notebook:", np.round(coef, 3))""")],
    ["Run the coolant-failure scenario (notebook 09) for the production reactor operating on the ignited branch: how far does "
     "the temperature rise, and how fast? Then for the kappa = 3 design. Which needs a relief system?",
     "The batch kinetics have 1.5 % noise. Bootstrap the Arrhenius fit (resample residuals, refit 200 times) to get a "
     "distribution of Ea, and propagate it to the ignition temperature at tau = 480 s. How uncertain is the safe window?"],
    "**Implement it yourself:** write `clean_pilot_runs(df)` that applies the two unit repairs by rule (X > 1.05 -> percent; "
    "T < 200 -> degC) and returns the cleaned table with a log; verify it reproduces the notebook's table.")


def apply_extras(name, cells):
    ex = EXTRAS[name]
    header = md("**What you will learn**\n" + "\n".join(f"- {o}" for o in ex["objectives"])
                + f"\n\n**Before you start:** {ex['prereq']}  ·  **Time:** about {ex['time']}")
    cells = [cells[0], header] + list(cells[1:])
    idx = [i for i, c in enumerate(cells) if c.cell_type == "markdown" and "## Exercises" in c.source]
    if idx:
        i = idx[0]
        n_items = len(re.findall(r"^\d+\. ", cells[i].source, flags=re.MULTILINE))
        cells[i] = md(cells[i].source.rstrip() + f"\n{n_items + 1}. {ex['implement']}")
        return cells[:i] + list(ex["inside"]) + cells[i:]
    items = ex["exercises"] + [ex["implement"]]
    exercises = md("## Exercises\n" + "\n".join(f"{k}. {t}" for k, t in enumerate(items, 1)))
    return cells + list(ex["inside"]) + [exercises]


def write_all(run: bool = True, only=()):
    import nbclient

    for name in sorted(NOTEBOOKS):
        if only and not name.startswith(tuple(only)):
            continue
        cells = apply_extras(name, NOTEBOOKS[name])
        nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                "language": "python"},
                                                 "language_info": {"name": "python"}})
        if run:
            nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                    resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, HERE / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    write_all(run="--no-run" not in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("--")])
