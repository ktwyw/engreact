"""Build the worked solutions to every exercise of the course, as executed notebooks.

The exercise texts are taken from ../notebooks/build_notebooks.py, so questions and solutions cannot
drift apart; the build fails if a notebook's number of solutions differs from its number of exercises.

    python solutions/build_solutions.py            # build and execute all solution notebooks
    python solutions/build_solutions.py 05 12      # only those whose names start with 05 or 12
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("course", HERE.parent / "notebooks" / "build_notebooks.py")
course = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(course)
md, code = course.md, course.code
SOLUTIONS: dict[str, dict] = {}


def solutions(name, setup="", imports=()):
    """Register the solutions of one course notebook: a function returning one list of cells per exercise."""

    def deco(fn):
        SOLUTIONS[name] = {"setup": setup, "imports": list(imports), "answers": fn()}
        return fn

    return deco


def exercises(name) -> list[str]:
    cells = course.apply_extras(name, list(course.NOTEBOOKS[name]))
    text = [c.source for c in cells if c.cell_type == "markdown" and "## Exercises" in c.source][0]
    items = re.split(r"\n(?=\d+\. )", text.split("## Exercises")[1].strip())
    return [re.sub(r"^\d+\.\s*", "", it).strip() for it in items]


def notebook_cells(name) -> list:
    title = re.sub(r"^#\s*", "", course.NOTEBOOKS[name][0].source.splitlines()[0])
    entry = SOLUTIONS[name]
    ex = exercises(name)
    if len(ex) != len(entry["answers"]):
        raise ValueError(f"{name}: {len(ex)} exercises but {len(entry['answers'])} solutions")
    cells = [
        md(f"""
# Solutions · {title}

Worked solutions to the exercises of [notebook {name[:2]}](../notebooks/{name}.ipynb). Try each exercise
yourself before reading its solution - the learning happens in the attempt. Solutions are one way to
answer each question; other correct approaches exist.
"""),
        course.setup_cell(entry["imports"]),
    ] + ([code(entry["setup"])] if entry["setup"] else [])
    for i, (text, answer) in enumerate(zip(ex, entry["answers"]), 1):
        cells.append(md(f"## Exercise {i}\n\n{text}"))
        cells.extend(answer)
    return cells


def write_all(only=()):
    import nbclient

    for name in SOLUTIONS:
        if only and not name.startswith(tuple(only)):
            continue
        nb = new_notebook(
            cells=notebook_cells(name),
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"},
            },
        )
        nbclient.NotebookClient(
            nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}
        ).execute()
        path = HERE / f"{name[:2]}_solutions.ipynb"
        nbformat.write(nb, path)
        print("wrote", path.name)


# =====================================================================================================
@solutions(
    "00_python_for_reaction_engineering",
    imports=[
        "from scipy import integrate, optimize",
        "from engreact import kinetics",
        "from engreact.constants import R",
    ],
)
def _():
    return [
        [
            code(r"""Ea = 100e3
for T1, T2 in ((298.15, 308.15), (573.15, 583.15)):
    print(f"{T1-273.15:.0f} -> {T2-273.15:.0f} degC: rate x {kinetics.rate_ratio(Ea, T1, T2):.2f}")
print("the 'doubling per 10 K' rule holds for Ea ~ 50 kJ/mol near room temperature only: the factor is exp(Ea/R x 10/T^2), which falls as T rises")"""),
            md(r"""
With $E_a$ = 100 kJ/mol the rate rises 3.7-fold per 10 K at room temperature but only 1.4-fold at 300 degC:
the Arrhenius factor for a fixed step depends on $E_a/RT^2$. The rule of thumb encodes one activation
energy at one temperature.
"""),
        ],
        [
            code(r"""k2, C0 = 0.11, 0.02                                          # L/(mol s), mol/L
sol = integrate.solve_ivp(lambda t, C: [-k2 * C[0] ** 2], (0, 1200), [C0], t_eval=[300, 600, 1200], rtol=1e-10, atol=1e-14)
exact = C0 / (1 + k2 * C0 * sol.t)
print(pd.DataFrame({"t (min)": sol.t / 60, "solve_ivp": sol.y[0], "closed form": exact, "rel. error": sol.y[0] / exact - 1}).to_string(index=False, float_format=lambda v: f"{v:.6g}"))"""),
            md(r"""
The integrator reproduces the closed form to eight digits with tight tolerances - and the closed form is
what tells you the tolerances were tight enough. Always keep an analytic case in reach.
"""),
        ],
        [
            code(r"""def temperature_for_k(k0, Ea, k_target):
    return -Ea / (R * np.log(k_target / k0))
k_target = kinetics.arrhenius(5.0e7, 68e3, 300.0) * 10
T_hand = temperature_for_k(5.0e7, 68e3, k_target)
print(f"by hand: {T_hand:.2f} K; library: {kinetics.temperature_for_rate_ratio(68e3, 300.0, 10.0):.2f} K")"""),
            md(r"""
Inverting $k = k_0 e^{-E_a/RT}$ is one logarithm; both routes give the temperature for a tenfold rate.
"""),
        ],
    ]


# =====================================================================================================
@solutions("01_mole_balances_and_ideal_reactors", imports=["from scipy import optimize", "from engreact import ideal"])
def _():
    return [
        [
            code(r"""k, C_A0, F_A0 = 0.05, 400.0, 1.0
rate_half = lambda X: k * (C_A0 * (1 - X)) ** 0.5
for X in (0.5, 0.9):
    Vc, Vp = ideal.cstr_volume(F_A0, X, rate_half), ideal.pfr_volume(F_A0, X, rate_half)
    print(f"X = {X}: CSTR {Vc:.3f} m3, PFR {Vp:.3f} m3, ratio {Vc/Vp:.2f}")
rate_first = lambda X: 0.05 * 400.0 ** 0.5 / 400.0 * C_A0 * (1 - X)      # a first-order law with the same initial rate
print(f"for comparison, first order at X = 0.9: ratio {ideal.cstr_volume(F_A0, 0.9, rate_first)/ideal.pfr_volume(F_A0, 0.9, rate_first):.2f}")"""),
            md(r"""
A half-order rate falls more slowly with conversion than a first-order one, so the CSTR's penalty for
running at the outlet concentration is smaller: the volume ratio at 90 % is about 2 against 3.9 for first
order. The lower the order, the less the reactor type matters; at zero order it does not matter at all.
"""),
        ],
        [
            code(r"""rate0 = lambda X: 0.1 + 0 * X                                          # zero order: constant rate
for X in (0.3, 0.6, 0.9):
    print(f"X = {X}: CSTR {ideal.cstr_volume(1.0, X, rate0):.3f} m3, PFR {ideal.pfr_volume(1.0, X, rate0):.3f} m3")"""),
            md(r"""
With a constant rate the Levenspiel curve is flat, so the rectangle and the area under the curve coincide:
every mole is converted at the same rate wherever it is, and the reactor's mixing pattern is irrelevant.
"""),
        ],
        [
            code(r"""k, tau = 0.05 / 60, 1200.0
X_hand = optimize.brentq(lambda X: X - k * tau * (1 - X), 0, 1)
print(f"by hand: X = {X_hand:.4f}; library: {ideal.cstr_conversion_first_order(k, tau):.4f}")"""),
            md(r"""
The CSTR balance $X = k\tau(1 - X)$ is linear here and hardly needs a root finder; the same three lines
solve it for any rate law where it is not.
"""),
        ],
    ]


# =====================================================================================================
@solutions(
    "02_conversion_and_reactor_sizing",
    imports=["from scipy import optimize, integrate", "from engreact import ideal, rate_data"],
    setup=r"""X = np.array([0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8])
rate = np.array([0.45, 0.37, 0.30, 0.195, 0.113, 0.079, 0.05])
F_A0 = 0.4
rate_fn = lambda x: np.interp(x, X, rate)""",
)
def _():
    return [
        [
            code(r"""best = None
for X1 in np.linspace(0.05, 0.75, 141):
    V1 = F_A0 * X1 / rate_fn(X1)
    V2 = F_A0 * (0.8 - X1) / rate_fn(0.8)
    if best is None or V1 + V2 < best[0]:
        best = (V1 + V2, X1, V1, V2)
print(f"two CSTRs: minimum total {best[0]:.2f} m3 with the first to X = {best[1]:.2f} ({best[2]:.2f} + {best[3]:.2f} m3)")
print(f"single CSTR: {F_A0 * 0.8 / rate_fn(0.8):.2f} m3; PFR: {ideal.levenspiel(X, rate, F_A0)['V_pfr'][-1]:.2f} m3")"""),
            md(r"""
Splitting the CSTR duty into two stages nearly halves the total volume; the optimum intermediate
conversion is where the two rectangles have equal "height gain" - Levenspiel's maximisation of the
rectangle area. Two stages capture most of the benefit; the PFR is the limit of infinitely many.
"""),
        ],
        [
            code(r"""C_A = 2500.0 * (1 - X)
fit = rate_data.initial_rates_method(C_A[1:], rate[1:])
n, k = fit["order"], fit["k"]
rate_law = lambda x: k * (2500.0 * (1 - x)) ** n
Vp, Vc = ideal.pfr_volume(F_A0, 0.8, rate_law), ideal.cstr_volume(F_A0, 0.8, rate_law)
lev = ideal.levenspiel(X, rate, F_A0)
print(f"power-law fit: order {n:.2f}; PFR {Vp:.2f} m3 (table: {lev['V_pfr'][-1]:.2f}), CSTR {Vc:.2f} m3 (table: {lev['V_cstr'][-1]:.2f})")"""),
            md(r"""
The rate law and the table agree within a few percent for the PFR (the fitted curve smooths the trapezoids)
and exactly-ish for the CSTR, which uses only the last point. The differences measure the interpolation
error of a 7-point table - and the fit lets one extrapolate beyond X = 0.8, which the table cannot.
"""),
        ],
        [
            code(r"""def cstr_series(F_A0, volumes, rate_fn):
    X_in, out = 0.0, []
    for V in volumes:
        X_out = optimize.brentq(lambda x: F_A0 * (x - X_in) / rate_fn(x) - V, X_in + 1e-9, 0.999)
        out.append(X_out); X_in = X_out
    return out
print("by hand:", np.round(cstr_series(F_A0, [1.15, 1.15], rate_fn), 4), "; library:", [round(s["X_out"], 4) for s in ideal.reactors_in_series(F_A0, [("cstr", 1.15), ("cstr", 1.15)], rate_fn)])"""),
            md(r"""
Each stage's balance uses the conversion leaving the previous one as its inlet; a root find per stage is all
there is to a cascade.
"""),
        ],
    ]


# =====================================================================================================
@solutions(
    "03_rate_laws_and_stoichiometry",
    imports=["from engreact import kinetics, thermo", "from engreact.kinetics import Reaction"],
)
def _():
    return [
        [
            code(r"""nh3 = Reaction({"N2": -1, "H2": -3, "NH3": 2}, key="N2")
feed = {"N2": 1.0, "H2": 3.0}
for T_C, p_bar in ((450, 100), (450, 300), (400, 300)):
    T = T_C + 273.15
    Kp = thermo.equilibrium_constant({"N2": -1, "H2": -3, "NH3": 2}, T)          # (bar)^-2
    Kc = thermo.kc_from_kp(Kp, -2, T)                                            # (m3/mol)^2
    C_A0 = 0.25 * p_bar * 1e5 / (8.314 * T)                                      # N2 concentration in the feed, mol/m3
    Xe = kinetics.equilibrium_conversion(nh3, feed, Kc, C_A0, phase="gas")
    print(f"{T_C} degC, {p_bar} bar: Kp = {Kp:.2e} 1/bar2, equilibrium N2 conversion = {Xe:.3f}")"""),
            md(r"""
At 450 degC the equilibrium conversion of nitrogen is about 28 % at 100 bar and 48 % at 300 bar (fewer moles
on the right: pressure helps); lowering the temperature to 400 degC raises it to 60 %, but the catalyst slows.
Haber-Bosch runs at 400-500 degC and 150-300 bar with recycle of the unconverted gas - the whole process
is written in these three numbers.
"""),
        ],
        [
            code(r"""rxn = Reaction({"A": -1, "B": -1, "C": 1}, key="A")
Xs = np.linspace(0, 0.95, 6)
for theta in (1, 2, 5):
    tab = kinetics.stoichiometric_table(rxn, {"A": 1.0, "B": theta}, Xs, "liquid", C_A0=1.0)
    r = tab["C"]["A"] * tab["C"]["B"]
    print(f"Theta_B = {theta}: -r/k = C_A C_B = {np.round(r, 3)}; ratio to first order C_A: {np.round(r / tab['C']['A'], 3)}")"""),
            md(r"""
With a large excess of B its concentration barely changes, and $k C_A C_B \approx (k C_{B0}) C_A$: a
pseudo-first-order rate law with an apparent constant $kC_{B0}$. Excess reagent is how second-order kinetics
are measured one reactant at a time - and why the "order" reported depends on the conditions.
"""),
        ],
        [
            code(r"""rxn = Reaction({"A": -2, "B": -1, "C": 3}, key="A")
feed = {"A": 2.0, "B": 1.0, "I": 1.0}
X_ = 0.4
theta_B, theta_C, theta_I = 0.5, 0.0, 0.5
delta = (3 - 2 - 1) / 2                              # 0: no change in moles per mole of A
eps = 0.5 * delta
C_A0 = 1.0
by_hand = {"A": C_A0 * (1 - X_) / (1 + eps * X_), "B": C_A0 * (theta_B - X_ / 2) / (1 + eps * X_), "C": C_A0 * (theta_C + 1.5 * X_) / (1 + eps * X_), "I": C_A0 * theta_I / (1 + eps * X_)}
lib = kinetics.stoichiometric_table(rxn, feed, X_, "gas", C_A0)
print("epsilon:", eps, "library:", lib["epsilon"])
print({s: (round(by_hand[s], 4), round(float(lib["C"][s]), 4)) for s in by_hand})"""),
            md(r"""
For 2A + B -> 3C the total moles do not change ($\delta$ = 0), so $\varepsilon$ = 0 and the gas-phase table
reduces to the liquid one; every column matches.
"""),
        ],
    ]


# =====================================================================================================
@solutions(
    "04_isothermal_reactor_design",
    imports=["from scipy import integrate", "from engreact import ideal, kinetics, multiple"],
)
def _():
    return [
        [
            code(r"""rho, mu, phi, F_A0, C_A0g, eps, k_prime = 1.15, 1.9e-5, 0.42, 0.02, 40.0, 0.5, 2.0e-5
rate_pbr = lambda X, y: k_prime * kinetics.concentration_from_conversion(C_A0g, X, eps, P_ratio=y)
for Dp in (4e-3, 8e-3):
    beta0 = ideal.ergun(rho * 0.8, rho, mu, Dp, phi)
    alpha = ideal.ergun_alpha(beta0, P0=2e5, A_c=0.015, rho_c=1200.0, phi=phi)
    pb = ideal.pbr_with_pressure_drop(F_A0, 100.0, rate_pbr, alpha, eps)
    print(f"D_p = {Dp*1e3:.0f} mm: Ergun gradient {beta0:.0f} Pa/m, alpha = {alpha:.2e} 1/kg, outlet X = {pb['X'][-1]:.3f}, P/P0 = {pb['y'][-1]:.3f}")
print("larger pellets: less pressure drop, but a larger Thiele modulus - if the reaction is diffusion-limited, k' falls with 1/D_p and conversion drops (notebook 11)")"""),
            md(r"""
Doubling the pellet diameter roughly halves the inertial Ergun term (and quarters the viscous one), so the
pressure holds up and the conversion rises. The catch is inside the pellet: if internal diffusion limits, the
effective rate constant per kilogram falls in proportion to the diameter, and the larger pellets convert
less - the classic packed-bed trade-off.
"""),
        ],
        [
            code(r"""kf, Kc = 0.5, 0.2                                    # A <-> B + C, equilibrium-limited
net = multiple.Network([({"A": -1, "B": 1, "C": 1}, lambda C: kf * (C["A"] - C["B"] * C["C"] / Kc))])
pfr = net.pfr({"A": 1.0}, 40.0)
for k_m in (0.0, 0.5, 2.0):
    mem = multiple.Network([({"A": -1, "B": 1, "C": 1}, lambda C: kf * (C["A"] - C["B"] * C["C"] / Kc)), ({"B": -1}, lambda C, k_m=k_m: k_m * C["B"])])
    out = mem.pfr({"A": 1.0}, 40.0)
    print(f"membrane removal k_m = {k_m}: conversion of A = {1 - out['A'][-1]:.3f}" + ("  (equilibrium limit)" if k_m == 0 else ""))"""),
            md(r"""
Removing B through a membrane keeps the reverse reaction starved, and the conversion climbs past the
equilibrium value - not a violation of thermodynamics but a change of the system: the products never
accumulate. Membrane reactors do this for dehydrogenations (removing hydrogen through palladium).
"""),
        ],
        [
            code(r"""k2, C_A0, V, dead = 0.11e-3, 200.0, 2.0, 1800.0
Xs = np.linspace(0.3, 0.995, 200)
daily = [C_A0 * V * X * 86400 / (ideal.batch_time_nth_order(k2, C_A0, X, 2) + dead) for X in Xs]
print(f"optimum conversion per batch: {Xs[int(np.argmax(daily))]:.3f} -> {max(daily):.0f} mol/day")"""),
            md(r"""
The daily output rises with conversion while the reaction is fast and falls when the batch time balloons;
the optimum is where the marginal product per extra minute equals the average - the same argument as for
choosing a harvest time.
"""),
        ],
    ]


# =====================================================================================================
@solutions("05_analysis_of_rate_data", imports=["from engreact import datasets, kinetics, rate_data"])
def _():
    return [
        [
            code(r"""d = pd.read_csv(datasets.path("n2o5_decomposition.csv"), comment="#")
t, C = d.t_s.to_numpy(), d.C_mmol_L.to_numpy()
with_bad = rate_data.nonlinear_fit(t, C, order=1)
mask = np.ones_like(t, bool); mask[7] = False
without = rate_data.nonlinear_fit(t[mask], C[mask], order=1)
print(f"with the bad sample:    k = {with_bad['k']:.3e} +/- {with_bad['se_k']:.1e} 1/s")
print(f"without:                k = {without['k']:.3e} +/- {without['se_k']:.1e} 1/s")
print(f"the standard error with the outlier is {with_bad['se_k']/without['se_k']:.0f} x larger: the residual it leaves inflates the error estimate, which is how least squares 'notices'; a robust (soft-L1) loss would instead down-weight it")"""),
            md(r"""
Least squares cannot exclude the bad point but its standard error grows several-fold, which is the signal to
look at the residuals. A robust loss would reduce the outlier's pull automatically - useful when you cannot
inspect every sample, dangerous when the "outlier" is real chemistry.
"""),
        ],
        [
            code(r"""tab = pd.read_csv(datasets.path("arrhenius_parameters.csv"), comment="#")
row = tab[tab.reaction.str.startswith("cyclopropane")].iloc[0]
A, Ea = float(row.A), row.Ea_kJ_mol * 1e3
for T_C in (450, 500, 550):
    k = kinetics.arrhenius(A, Ea, T_C + 273.15)
    print(f"{T_C} degC: k = {k:.3e} 1/s, half-life {np.log(2)/k:.1f} s")
T_1min = kinetics.temperature_for_rate_ratio(Ea, 450 + 273.15, np.log(2) / 60 / kinetics.arrhenius(A, Ea, 450 + 273.15))
print(f"half-life of one minute at {T_1min - 273.15:.0f} degC")"""),
            md(r"""
The isomerisation of cyclopropane, the textbook unimolecular reaction, has a half-life of five hours at
450 degC and under two minutes at 550 degC, its activation energy of 272 kJ/mol making it exquisitely
temperature-sensitive; a one-minute half-life is reached near 556 degC.
"""),
        ],
        [
            code(r"""s = pd.read_csv(datasets.path("saponification.csv"), comment="#")
fit = rate_data.nonlinear_fit(s.t_s, s.C_mol_L, order=2)
C0s = np.array([0.01, 0.02, 0.04, 0.08])
halves = [rate_data.half_life(c, fit["k"], 2) for c in C0s]
x, y = np.log(C0s), np.log(halves)
slope = np.polyfit(x, y, 1)[0]
print(f"half-lives {np.round(halves, 1)} s; slope of ln t1/2 vs ln C0 = {slope:.3f} -> order {1 - slope:.2f}; library: {rate_data.half_life_method(C0s, halves)['order']:.2f}")"""),
            md(r"""
For second order the half-life is inversely proportional to the initial concentration - slope -1, order 2 -
which is why a single half-life means nothing for a reaction of unknown order.
"""),
        ],
    ]


# =====================================================================================================
@solutions(
    "06_multiple_reactions_and_selectivity",
    imports=["from scipy import integrate, optimize", "from engreact import kinetics, multiple"],
)
def _():
    return [
        [
            code(r"""k1, k2 = 0.5, 0.2
tau_c = optimize.brentq(lambda tau: k1 * tau / (1 + k1 * tau) - 0.9, 0, 1e3)
tau_p = -np.log(0.1) / k1
Bc = float(multiple.series_cstr(k1, k2, 1.0, tau_c)["B"])
Bp = float(multiple.series_batch(k1, k2, 1.0, tau_p)["B"])
print(f"90 % conversion of A: CSTR tau = {tau_c:.1f} min, yield of B = {Bc/0.9:.3f}; PFR tau = {tau_p:.2f} min, yield of B = {Bp/0.9:.3f}")
print(f"C formed: CSTR {0.9 - Bc:.3f}, PFR {0.9 - Bp:.3f}")"""),
            md(r"""
At the same 90 % conversion the PFR keeps 45 % of the reacted A as B, the CSTR only 20 %, because the CSTR
holds B at the outlet concentration for eighteen minutes. If B is the product, PFR; if C is, the CSTR gets
there with less volume than a PFR run long enough to consume B.
"""),
        ],
        [
            code(r"""best = None
for T in np.linspace(300, 400, 11):
    for C0 in np.logspace(-1, 1, 9):
        kD, kU = kinetics.arrhenius(1e5, 40e3, T), kinetics.arrhenius(1e10, 80e3, T)
        net = multiple.Network([({"A": -1, "D": 1}, lambda C, kD=kD: kD * C["A"]), ({"A": -1, "U": 1}, lambda C, kU=kU: kU * C["A"] ** 2)])
        tau = optimize.brentq(lambda tau: 1 - net.pfr({"A": C0}, tau, n=50)["A"][-1] / C0 - 0.9, 1e-3, 1e5)
        out = net.pfr({"A": C0}, tau, n=50)
        S = out["D"][-1] / max(out["U"][-1], 1e-12)
        if best is None or S > best[0]:
            best = (S, T, C0, tau)
print(f"best selectivity D/U = {best[0]:.1f} at T = {best[1]:.0f} K, C_A0 = {best[2]:.2f} mol/L (tau = {best[3]:.1f})")"""),
            md(r"""
The desired reaction is first order with the lower activation energy, the undesired second order with the
higher: low concentration and low temperature both favour D, so the optimum sits at the corner of the
box (300 K, 0.1 mol/L). The price is a long residence time - selectivity and throughput pull opposite
ways, and the economics decide.
"""),
        ],
        [
            code(r"""sol = integrate.solve_ivp(lambda t, y: [-k1 * y[0], k1 * y[0] - k2 * y[1]], (0, 20), [1.0, 0.0], dense_output=True, rtol=1e-10)
t_max = optimize.minimize_scalar(lambda t: -sol.sol(t)[1], bounds=(0, 20), method="bounded").x
print(f"numerical t_max = {t_max:.4f} min; analytic ln(k2/k1)/(k2 - k1) = {multiple.series_optimum_time(k1, k2):.4f}")"""),
            md(r"""
Two ODEs, a dense solution and a one-dimensional maximisation reproduce the analytic optimum to four
decimals - the numerical route that works when the kinetics are not first order and no formula exists.
"""),
        ],
    ]


# =====================================================================================================
@solutions(
    "07_energy_balances_and_adiabatic_reactors",
    imports=["from scipy import optimize", "from engreact import energy, kinetics, thermo"],
)
def _():
    return [
        [
            code(r"""stoich = {"N2": -1, "H2": -3, "NH3": 2}
p = 200.0
def X_eq(T):
    Kp = thermo.equilibrium_constant(stoich, T)
    def f(X):
        n = 4 - 2 * X                                                        # total moles per mole of N2 fed
        y_N2, y_H2, y_NH3 = (1 - X) / n, 3 * (1 - X) / n, 2 * X / n
        return (y_NH3 * p) ** 2 / ((y_N2 * p) * (y_H2 * p) ** 3) - Kp
    return optimize.brentq(f, 1e-9, 1 - 1e-9)
T = np.linspace(600, 900, 61)
Xe = np.array([X_eq(t) for t in T])
dT_ad = 200.0
T_line = 650.0 + np.linspace(0, 1, 100) * dT_ad
X_line = np.linspace(0, 1, 100)
cross = optimize.brentq(lambda X: X_eq(650.0 + X * dT_ad) - X, 1e-6, 0.999)
print(f"equilibrium conversion: {X_eq(650):.3f} at 650 K, {X_eq(750):.3f} at 750 K, {X_eq(850):.3f} at 850 K")
print(f"an adiabatic bed from 650 K reaches the equilibrium curve at X = {cross:.3f} (T = {650 + cross*dT_ad:.0f} K)")
X, beds = 0.0, 0
while X < 0.55 and beds < 10:
    X = optimize.brentq(lambda Xo: X_eq(650.0 + (Xo - X) * dT_ad) - Xo, X + 1e-6, 0.999); beds += 1
    print(f"  bed {beds}: X = {X:.3f}")
print(f"{beds} beds with interstage cooling to 650 K reach X = {X:.3f}")
plt.plot(T, Xe, "k", label="X_e(T) at 200 bar"); plt.plot(T_line, X_line, "--", label="adiabatic line from 650 K"); plt.xlabel("T (K)"); plt.ylabel("N2 conversion"); plt.legend(); plt.show()"""),
            md(r"""
At 200 bar the equilibrium conversion falls from 59 % at 650 K to 18 % at 850 K; the adiabatic line from
650 K meets the curve at 39 % and 728 K, and each further bed adds less as the operating point climbs the
curve: 55 % needs three beds. Industrial converters use three or four beds and a recycle, because the last
few percent cost the most.
"""),
        ],
        [
            code(r"""k0e, Eae, dHe, C_A0, cp_vol = 2e4, 90e3, 80e3, 2500.0, 3.6e6
F_A0 = 2.5
rate_e = lambda X, T: kinetics.arrhenius(k0e, Eae, T) * C_A0 * (1 - X)
def outlet(V, Ua):
    if Ua == 0:
        return energy.adiabatic_pfr(rate_e, F_A0, V, 750.0, dHe, sum_theta_cp=cp_vol / C_A0)["X"][-1]
    return energy.pfr_heat_exchange(rate_e, F_A0, V, 750.0, dHe, sum_F_cp=F_A0 * cp_vol / C_A0, Ua=Ua, T_a=750.0)["X"][-1]
for Ua, label in ((0.0, "adiabatic"), (5e3, "jacket 750 K, Ua = 5e3"), (5e4, "jacket 750 K, Ua = 5e4"), (5e5, "jacket 750 K, Ua = 5e5")):
    V80 = optimize.brentq(lambda V: outlet(V, Ua) - 0.8, 1e-3, 200.0)
    print(f"{label:<26}: volume for 80 % conversion = {V80:.3f} m3")"""),
            md(r"""
Adiabatically the endothermic reaction cools itself by 55 K and slows; a jacket at the feed temperature helps
only when its capacity is comparable with the reaction's heat demand of several hundred kW per cubic metre -
an ordinary jacket (Ua of a few thousand W/(m3 K)) changes almost nothing, furnace-tube heat transfer
(10^5 W/(m3 K)) keeps the reactor nearly isothermal and cuts the volume. Endothermic reactors are heat-transfer
equipment with catalyst in it.
"""),
        ],
        [
            code(r"""from scipy import integrate
k0, Ea, dH = 5.0e7, 68e3, -250e3
dT_ad_ = energy.adiabatic_temperature_rise(dH, C_A0, cp_vol)
sol = integrate.solve_ivp(lambda v, X: [kinetics.arrhenius(k0, Ea, 300.0 + X[0] * dT_ad_) * C_A0 * (1 - X[0]) / F_A0], (0, 0.3), [0.0], method="LSODA", rtol=1e-9)
lib = energy.adiabatic_pfr(lambda X, T: kinetics.arrhenius(k0, Ea, T) * C_A0 * (1 - X), F_A0, 0.3, 300.0, dH, sum_theta_cp=cp_vol / C_A0)
print(f"by hand: X = {sol.y[0][-1]:.5f}; library: {lib['X'][-1]:.5f}")"""),
            md(r"""
Substituting $T(X)$ from the energy balance into the rate turns the adiabatic PFR into a single ODE in X;
the library does the same and the two agree to five digits.
"""),
        ],
    ]


# =====================================================================================================
@solutions(
    "08_cstr_multiple_steady_states",
    imports=["from engreact import energy"],
    setup=r"""k0, Ea, dH, C_A0, cp_vol, kappa, T_a = 5.0e7, 68e3, -250e3, 2500.0, 3.6e6, 0.8, 295.0""",
)
def _():
    return [
        [
            code(r"""for tau, C0 in ((120.0, C_A0), (240.0, C_A0), (240.0, 2 * C_A0)):
    T0s = np.linspace(260, 360, 51)
    n = np.array([len(energy.cstr_steady_states(k0, Ea, dH, C0, T0, tau, cp_vol, kappa, T_a, T_range=(250.0, 900.0))["T"]) for T0 in T0s])
    band = T0s[n == 3]
    print(f"tau = {tau:.0f} s, C_A0 = {C0:.0f} mol/m3 (Delta T_ad = {energy.adiabatic_temperature_rise(dH, C0, cp_vol):.0f} K): " + (f"multiplicity for T0 in [{band.min():.0f}, {band.max():.0f}] K" if band.size else "none"))"""),
            md(r"""
Halving the residence time (halving the Damkohler number) moves the multiplicity to higher feed
temperatures but does not remove it; doubling the feed concentration doubles the adiabatic rise and widens
the band. Multiplicity needs a large heat release relative to the cooling and a rate that can outrun it; both
levers appear in the dimensionless groups $\Delta T_{ad} E_a/RT^2$ and $k\tau$.
"""),
        ],
        [
            code(r"""T0s = np.linspace(270, 360, 91)
ignited = [(T0, ss["T"][-1], ss["X"][-1]) for T0 in T0s for ss in [energy.cstr_steady_states(k0, Ea, dH, C_A0, T0, 240.0, cp_vol, kappa, T_a, T_range=(250.0, 900.0))] if ss["X"][-1] > 0.5 and ss["stable"][-1]]
lo, hi = ignited[0][0], ignited[-1][0]
print(f"the ignited branch exists and is stable for T0 from {lo:.0f} K upward (below, the reactor extinguishes); at T0 = 300 K it sits at {[t for t in ignited if abs(t[0]-300) < 0.6][0][1]:.0f} K")
T_max_allowed = 420.0
ok = [t for t in ignited if t[1] <= T_max_allowed]
print(f"staying below {T_max_allowed:.0f} K on the ignited branch requires T0 <= {ok[-1][0]:.0f} K -> operate with T0 between {lo:.0f} and {ok[-1][0]:.0f} K")
print("start-up: feed at >= 330 K (above the ignition point) until the reactor is hot, then lower the feed to the operating value: the hot state persists down to the extinction limit")"""),
            md(r"""
The ignited branch is stable from about 300 K of feed temperature upward, and stays below 420 K only up to a
certain feed temperature: the operating window. Because the ignited state persists once reached, the start-up
recipe is "heat to ignite, then cool to operate" - the hysteresis working for the operator instead of
against them.
"""),
        ],
        [
            code(r"""ss = energy.cstr_steady_states(k0, Ea, dH, C_A0, 315.0, 240.0, cp_vol, kappa, T_a, T_range=(250.0, 900.0))
dR = cp_vol * (1 + kappa)
for T_ in ss["T"]:
    G = lambda T: float(energy.cstr_heat_curves(k0, Ea, dH, C_A0, 315.0, 240.0, cp_vol, T, kappa, T_a)["G"])
    dG = (G(T_ + 0.01) - G(T_ - 0.01)) / 0.02
    print(f"T = {T_:.0f} K: dG/dT = {dG/1e3:.1f} kJ/(m3 K), dR/dT = {dR/1e3:.1f} kJ/(m3 K) -> {'stable' if dG < dR else 'UNSTABLE'}")
print("library:", ss["stable"])"""),
            md(r"""
The slope test is a finite difference of the generation curve against the constant slope of the removal
line; the middle state fails it. It is a necessary condition (the full test involves the concentration
dynamics too), but for this first-order system it is also sufficient.
"""),
        ],
    ]


# =====================================================================================================
@solutions(
    "09_unsteady_reactors_and_runaway",
    imports=["from scipy import integrate", "from engreact import energy, kinetics"],
    setup=r"""k0, Ea, dH, C_A0, cp_vol = 5.0e7, 68e3, -250e3, 2500.0, 3.6e6""",
)
def _():
    return [
        [
            code(r"""t = np.linspace(0, 7200, 7201)
batch = energy.batch_with_heat_exchange(k0, Ea, dH, C_A0, 320.0, cp_vol, 1000.0, 300.0, t)
# semibatch approximated as ten portions of A added every 6 min into a reactor that already holds the solvent
def semibatch_portions(n_portions=10, UA_V=1000.0):
    T, C, out = 320.0, 0.0, []
    for i in range(n_portions):
        C += C_A0 / n_portions                                              # add a portion (mixing dilutes only the temperature slightly; volume assumed constant)
        seg = energy.batch_with_heat_exchange(k0, Ea, dH, C, T, cp_vol, UA_V, 300.0, np.linspace(0, 360, 361))
        T, C = seg["T"][-1], seg["C_A"][-1]
        out.append(seg["T"].max())
    return max(out)
print(f"batch, all at once: peak {batch['T_max']:.0f} K; semibatch in ten portions over an hour: peak {semibatch_portions():.0f} K; forty portions over four hours: {semibatch_portions(40):.0f} K")"""),
            md(r"""
Feeding the reactant in portions caps the amount that can react at any moment: ten portions cut the peak by
60 K, forty portions over four hours by far more, because the jacket then keeps up with the heat release.
Semibatch dosing controlled by the temperature (feed only while the reactor is cool enough) is the standard
way to run a fast exothermic reaction safely; the dosing rate is the safety parameter.
"""),
        ],
        [
            code(r"""t = np.linspace(0, 36000, 36001)
ad = energy.batch_with_heat_exchange(k0, Ea, dH, C_A0, 320.0, cp_vol, 0.0, 300.0, t)
i = int(np.argmax(ad["T"] >= 400.0))
q = (-dH) * kinetics.arrhenius(k0, Ea, ad["T"][i]) * ad["C_A"][i]
print(f"T reaches 400 K after {ad['t'][i]/60:.1f} min with {ad['C_A'][i]/C_A0:.1%} of A unreacted; heat release {q/1e6:.2f} MW/m3 at that moment")"""),
            md(r"""
By the time the reactor hits the vent temperature more than half the reactant remains and the heat release
is megawatts per cubic metre: the relief system must handle the vapour generated by that power, and a vent
sized for the average rate would be far too small. This is why relief sizing for runaway reactions uses the
DIERS methods and calorimetry, not steady-state heat balances.
"""),
        ],
        [
            code(r"""def rhs(t_, y, UA_V, T_b=None):
    C, T = y
    r = kinetics.arrhenius(k0, Ea, T) * max(C, 0.0)
    dT = ((-dH) * r - UA_V * (T - 300.0)) / cp_vol
    if T_b is not None and T >= T_b and dT > 0:
        dT = 0.0                                                             # evaporative cooling holds the boiling point
    return [-r, dT]
t = np.linspace(0, 7200, 7201)
sol = integrate.solve_ivp(rhs, (0, 7200), [C_A0, 320.0], t_eval=t, args=(1000.0,), method="LSODA", rtol=1e-8)
lib = energy.batch_with_heat_exchange(k0, Ea, dH, C_A0, 320.0, cp_vol, 1000.0, 300.0, t)
sol_b = integrate.solve_ivp(rhs, (0, 7200), [C_A0, 320.0], t_eval=t, args=(1000.0, 380.0), method="LSODA", rtol=1e-8)
print(f"by hand peak {sol.y[1].max():.1f} K, library {lib['T_max']:.1f} K; with a solvent boiling at 380 K the peak is capped at {sol_b.y[1].max():.0f} K")"""),
            md(r"""
The hand-written balance matches the library; adding a boiling-point cap shows why refluxing solvents are
a safety feature: latent heat removes energy far faster than a jacket, and the temperature cannot exceed the
boiling point as long as the condenser keeps up.
"""),
        ],
    ]


# =====================================================================================================
@solutions(
    "10_catalysis_and_heterogeneous_kinetics",
    imports=["from scipy import integrate, optimize", "from engreact import catalysis, datasets"],
)
def _():
    return [
        [
            code(r"""# A adsorbs molecularly (K_A), B dissociatively (K_B), surface reaction rate limiting: r = k K_A p_A sqrt(K_B p_B) / (1 + K_A p_A + sqrt(K_B p_B))^2
k, K_A, K_B, p_B = 1.0, 2.0, 1.0, 1.0
rate = lambda p_A: k * K_A * p_A * np.sqrt(K_B * p_B) / (1 + K_A * p_A + np.sqrt(K_B * p_B)) ** 2
p = np.linspace(0.01, 10, 400)
p_max = optimize.minimize_scalar(lambda x: -rate(x), bounds=(0.01, 10), method="bounded").x
print(f"maximum rate at p_A = {p_max:.3f} bar (analytic: (1 + sqrt(K_B p_B))/K_A = {(1 + np.sqrt(K_B * p_B))/K_A:.3f})")
plt.plot(p, rate(p)); plt.xlabel("p_A (bar)"); plt.ylabel("rate"); plt.title("dual-site rate with dissociative B: a maximum"); plt.show()"""),
            md(r"""
A dual-site mechanism with one dissociating reactant has the squared site balance in its denominator and a
maximum at $p_A = (1 + \sqrt{K_B p_B})/K_A$: beyond it, A crowds B off the surface and the rate falls with
more A - a behaviour no power law can produce and a clear fingerprint of the mechanism.
"""),
        ],
        [
            code(r"""d = pd.read_csv(datasets.path("catalyst_deactivation.csv"), comment="#")
t, a = d.t_h.to_numpy(), d.activity.to_numpy()
res = optimize.least_squares(lambda q: catalysis.activity(t, q[0], "nth", q[1]) - a, [0.01, 2.0], bounds=([0, 1.01], [1, 5]))
n_, k_d = res.x[1], res.x[0]
s2 = np.sum(res.fun ** 2) / (len(t) - 2)
se = np.sqrt(np.diag(s2 * np.linalg.inv(res.jac.T @ res.jac)))
print(f"free-order decay: n = {n_:.2f} +/- {se[1]:.2f}, k_d = {k_d:.4f} +/- {se[0]:.4f} 1/h -> n = 2 lies {'within' if abs(n_ - 2) < 2 * se[1] else 'outside'} two standard errors")
t_03 = optimize.brentq(lambda tt: catalysis.activity(tt, k_d, "nth", n_) - 0.3, 0, 1e5)
print(f"activity 0.3 after {t_03:.0f} h ({t_03/24:.1f} days)")"""),
            md(r"""
With the order free the data still point to second-order-like decay, and the time to 30 % activity - the
end of a typical cycle - is about ten days. The uncertainty on n is the honest statement of what ten points
can tell.
"""),
        ],
        [
            code(r"""k_d = 0.0121
sol = integrate.solve_ivp(lambda t_, y: [-k_d * y[0] ** 2], (0, 240), [1.0], t_eval=np.linspace(0, 240, 7), rtol=1e-10)
print("ODE:", np.round(sol.y[0], 4)); print("law:", np.round(catalysis.activity(sol.t, k_d, "second"), 4))"""),
            md(r"""
The second-order decay law is the solution of $da/dt = -k_d a^2$, and the integrator confirms it to four
decimals.
"""),
        ],
    ]


# =====================================================================================================
@solutions(
    "11_internal_diffusion_and_effectiveness_factors",
    imports=["from scipy import integrate, optimize", "from engreact import catalysis, ideal"],
)
def _():
    return [
        [
            code(r"""k, D_e, phi_bed = 0.05, 2e-9, 0.4
for eta_target in (0.9, 0.5):
    R_p = optimize.brentq(lambda R: catalysis.effectiveness_factor(R * np.sqrt(k / D_e), "sphere") - eta_target, 1e-6, 1e-1)

    print(f"eta = {eta_target}: pellet radius {R_p*1e3:.2f} mm; rate per unit bed volume = eta k (1 - phi) C = {eta_target * k * (1 - phi_bed):.3f} C per second")
print("smaller pellets use the catalyst better but raise the pressure drop (Ergun ~ 1/D_p^2 in the viscous term): the trade-off is settled by the compressor cost against catalyst cost")"""),
            md(r"""
Full utilisation needs pellets a fraction of a millimetre across, which a packed bed cannot tolerate for
pressure drop; a couple of millimetres at $\eta \approx 0.5$ is the usual compromise, giving up half the
catalyst's activity to keep the compressor bill sane. The pressure-drop result of notebook 04 and this one are
the two halves of the same decision.
"""),
        ],
        [
            code(r"""shell = 0.1e-3                                                     # half-thickness of the active shell, slab geometry
eta_shell = catalysis.effectiveness_factor(shell * np.sqrt(k / D_e), "slab")
R_p = 1.5e-3
eta_sphere = catalysis.effectiveness_factor(R_p * np.sqrt(k / D_e), "sphere")
vol_fraction = 1 - ((R_p - 2 * shell) / R_p) ** 3
print(f"eggshell: eta = {eta_shell:.3f} on {vol_fraction:.0%} of the pellet volume -> {eta_shell * vol_fraction:.3f} of a fully active pellet's rate")
print(f"uniform 3 mm sphere: eta = {eta_sphere:.3f}")"""),
            md(r"""
The eggshell pellet's active layer is nearly fully effective but holds only a third of the volume; the
uniform pellet uses the whole volume at a third of its activity. Per pellet the two are comparable; per gram
of active metal the eggshell wins by a wide margin, which is why precious-metal catalysts are almost always
eggshell.
"""),
        ],
        [
            code(r"""phi = 2.0
x = np.linspace(0, 1, 200)
sol = integrate.solve_bvp(lambda x_, y: np.vstack([y[1], phi**2 * y[0]]), lambda ya, yb: [ya[1], yb[0] - 1], x, np.vstack([np.cosh(phi * x) / np.cosh(phi), phi * np.sinh(phi * x) / np.cosh(phi)]), tol=1e-8)
eta_flux = sol.sol(1.0)[1] / phi**2                                # flux at the surface / (reaction if C = C_s everywhere)
print(f"from the surface flux: eta = {eta_flux:.5f}; tanh(phi)/phi = {np.tanh(phi)/phi:.5f}; library: {catalysis.effectiveness_factor(phi, 'slab'):.5f}")"""),
            md(r"""
The effectiveness factor is the flux into the pellet divided by the rate the whole pellet would have at the
surface concentration; the BVP solution's surface gradient gives it directly, matching $\tanh\phi/\phi$.
"""),
        ],
    ]


# =====================================================================================================
@solutions("12_external_mass_transfer_and_packed_beds", imports=["from engreact import catalysis, ideal"])
def _():
    return [
        [code(r"""kc_ac, U = 60.0, 5.0                                             # 1/s, m/s
L99 = -np.log(0.01) * U / kc_ac
print(f"monolith length for 99 % CO conversion: {L99:.2f} m")
for T1, T2 in ((700.0, 800.0),):
    ratio = (T2 / T1) ** 1.75 * (T1 / T2)                             # D_AB ~ T^1.75, concentration ~ 1/T at fixed pressure... k_c a_c ~ D_AB^(2/3) roughly
    Ea_app = 8.314 * np.log((T2 / T1) ** (1.75 * 2 / 3)) / (1 / T1 - 1 / T2)
    print(f"apparent activation energy in the film-controlled regime: about {Ea_app/1e3:.1f} kJ/mol (from k_c ~ D_AB^(2/3) ~ T^1.17)")"""),
         md(r"""
A film-controlled converter needs a fraction of a metre, and its "activation energy" is the temperature
dependence of gas diffusion - a few kJ/mol against the 50-100 of surface kinetics. An Arrhenius plot that
flattens to that slope at high temperature is the signature of external mass-transfer control.
""")],
        [code(r"""D_AB, nu, d_p, a_c, rho, mu, phi = 1.5e-5, 1.6e-5, 3e-3, 1200.0, 0.9, 2.5e-5, 0.4
Sc = nu / D_AB
for U in (0.2, 0.5, 1.0):
    k_c = catalysis.sherwood_frossling(U * d_p / nu, Sc) * D_AB / d_p
    L99 = -np.log(0.01) * U / (k_c * a_c)
    dP = ideal.ergun(rho * U, rho, mu, d_p, phi) * L99
    print(f"U = {U} m/s: L = {L99:.3f} m, pressure drop {dP:.0f} Pa, compressor work per unit gas ~ dP: {dP:.0f} J/m3")"""),
         md(r"""
Both the bed length (as $U^{1/2}$) and the Ergun gradient (as $U$ to $U^2$) grow with velocity, so the pressure
drop for a fixed conversion rises steeply: the lowest velocity that still gives acceptable throughput
minimises compressor work. Mass-transfer-limited beds are run slow and wide.
""")],
        [code(r"""def frossling(Re, Sc):
    return 2 + 0.6 * Re ** 0.5 * Sc ** (1 / 3)
for d in (1e-3, 3e-3, 10e-3):
    Re = 0.5 * d / nu
    k_c = frossling(Re, Sc) * D_AB / d
    print(f"d_p = {d*1e3:4.0f} mm: Sh = {frossling(Re, Sc):.1f}, k_c = {k_c*1e3:.1f} mm/s (library {catalysis.sherwood_frossling(Re, Sc)*D_AB/d*1e3:.1f}); Ergun gradient {ideal.ergun(rho*0.5, rho, mu, d, phi):.0f} Pa/m")"""),
         md(r"""
Small pellets have thin films and large mass-transfer coefficients (Sh grows slower than $d_p$, so $k_c
\propto d_p^{-1/2}$ or so) and a large interfacial area, but the pressure gradient grows even faster; the
same trade-off as internal diffusion, with the same resolution in practice - a few millimetres.
""")],
    ]


# =====================================================================================================
@solutions("13_catalyst_deactivation_and_reactor_operation", imports=["from scipy import integrate, optimize", "from engreact import catalysis, kinetics"],
           setup=r"""k_d, k0, Ea, W_over_F = 0.0121, 1e8, 50e3, 5.0
conversion = lambda T, a: 1 - np.exp(-kinetics.arrhenius(k0, Ea, T) * a * W_over_F * 1e-3)""")
def _():
    return [
        [code(r"""def avg_conversion(cycle_h, regen_h=24.0):
    t = np.linspace(0, cycle_h, 400)
    X = conversion(500.0, catalysis.activity(t, k_d, "second"))
    produced = (np.trapezoid(X, t) if hasattr(np, "trapezoid") else np.trapz(X, t))
    return produced / (cycle_h + regen_h)
cycles = np.linspace(20, 800, 200)
avg = [avg_conversion(c) for c in cycles]
best = cycles[int(np.argmax(avg))]
print(f"time-averaged conversion is maximised by a cycle of {best:.0f} h ({best/24:.1f} days): {max(avg):.3f} against {avg_conversion(720):.3f} for a monthly cycle")"""),
         md(r"""
Short cycles waste time regenerating; long ones run a dead catalyst. The optimum here is about a week -
where the marginal conversion during the last hour of the cycle equals the average over the whole cycle plus
regeneration, the same structure as the batch-scheduling optimum of notebook 04.
""")],
        [code(r"""k_lin = 0.004                                                       # 1/h, linear poisoning at the current impurity level
end = lambda k_: 1 / k_ * (1 - 0.3)                                 # time for activity to fall to 0.3 under linear decay
print(f"cycle to activity 0.3: {end(k_lin):.0f} h now; with a guard bed removing 90 % of the poison: {end(0.1 * k_lin):.0f} h - ten times longer")
print("ten times more catalyst would hold conversion for about as long only if the poison were spread evenly; in a fixed bed it deposits at the inlet, so extra catalyst buys less than the guard bed does, and costs more")"""),
         md(r"""
Linear poisoning is proportional to the poison arriving, so cutting the poison tenfold lengthens the cycle
tenfold; a guard bed is cheap catalyst (or an adsorbent) sacrificed to protect expensive catalyst. Extra
main-bed catalyst is the worse investment, because poisons deposit as a front from the inlet rather than
uniformly.
""")],
        [code(r"""traj = catalysis.temperature_time_trajectory(k_d0=2.0, E_d=60e3, k0=1e8, Ea=50e3, T0=500.0, t_max=200 * 86400, T_max=560.0)
T_of_t = lambda t: np.interp(t, traj["t"], traj["T"])
sol = integrate.solve_ivp(lambda t, a: [-kinetics.arrhenius(2.0, 60e3, T_of_t(t)) * a[0] ** 1], (0, traj["t"][-1]), [1.0], t_eval=traj["t"], rtol=1e-9)
product = kinetics.arrhenius(1e8, 50e3, traj["T"]) * sol.y[0]
print(f"k(T) a along the trajectory: min {product.min():.4f}, max {product.max():.4f} (target k(T0) = {kinetics.arrhenius(1e8, 50e3, 500.0):.4f}) - constant, as designed")"""),
         md(r"""
Re-integrating the decay along the library's temperature schedule and multiplying by the Arrhenius factor
gives a constant product: the schedule does what it claims. (The library's decay is first order in activity;
the exercise's $a^2$ would need its own schedule.)
""")],
    ]


# =====================================================================================================
@solutions("14_residence_time_distributions", imports=["from engreact import datasets, rtd"],
           setup=r"""d = pd.read_csv(datasets.path("rtd_tracer_pulse.csv"), comment="#")
t, C = d.t_s.to_numpy(), d.C_g_m3.to_numpy()
base = np.polyfit(t[t > 140], C[t > 140], 1)
E = rtd.E_from_pulse(t, np.maximum(C - np.polyval(base, t), 0.0))""")
def _():
    return [
        [code(r"""rng = np.random.default_rng(0)
F = rtd.F_from_E(t, E)
F_noisy = F * (1 + rng.normal(0, 0.02, F.size))
E_back = rtd.E_from_F(t, F_noisy)
trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
mean_from_F = trap(1 - np.clip(F_noisy, 0, 1), t)                     # mean residence time = integral of (1 - F): no differentiation needed
m_step, m_pulse = rtd.moments(t, np.maximum(E_back, 0)), rtd.moments(t, E)
print(f"pulse: mean {m_pulse['mean']:.1f} s, variance {m_pulse['variance']:.0f} s2")
print(f"noisy step, differentiated: mean {m_step['mean']:.1f} s, variance {m_step['variance']:.0f} s2 (both wrecked by the noise)")
print(f"noisy step, mean from the integral of (1 - F): {mean_from_F:.1f} s (robust)")
plt.plot(t, E, label="E from the pulse"); plt.plot(t, E_back, ".", ms=3, label="E from a 2 % noisy step"); plt.xlabel("t (s)"); plt.ylabel("E(t)"); plt.legend(); plt.show()"""),
         md(r"""
Differentiating a noisy step amplifies the noise into a ragged E(t) whose moments are unreliable; but the
mean residence time can be taken from the step without differentiating, as the area above the F curve, and
that estimate is robust. The pulse is the better experiment for the variance, the step for the mean. In
practice both are run, and they must agree.
""")],
        [code(r"""V_design, v0 = 4.0, 0.1
tau_nominal = V_design / v0
m = rtd.moments(t, E)
print(f"nominal V/v0 = {tau_nominal:.1f} s; measured mean residence time {m['mean']:.1f} s -> effective volume {m['mean'] * v0:.2f} m3, dead volume {(1 - m['mean']/tau_nominal):.1%}")
print("adsorbing tracer: it lingers on the walls and leaves late, lengthening the mean - which would mask dead volume or invent 'extra' volume; the tracer must be inert")"""),
         md(r"""
A measured mean below $V/v_0$ means part of the vessel is stagnant; here the shortfall is small, within the
uncertainty of the flow measurement. An adsorbing tracer would bias the mean upward and could cancel a real
dead volume - a reason tracer tests use salts or dyes chosen for inertness.
""")],
        [code(r"""def moments(t, E):
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = trap(E, t); mean = trap(t * E, t) / area
    var = trap((t - mean) ** 2 * E, t) / area
    skew = trap((t - mean) ** 3 * E, t) / area / var ** 1.5
    return mean, var, skew
tt = np.linspace(0, 600, 60001)
print("CSTR E(t), tau = 10: (mean, variance, skewness) =", np.round(moments(tt, rtd.cstr_E(tt, 10.0)), 3), "- exact (10, 100, 2)")"""),
         md(r"""
The exponential distribution has mean $\tau$, variance $\tau^2$ and skewness 2; the trapezoidal moments
reproduce them once the integration window covers the tail.
""")],
    ]


# =====================================================================================================
@solutions("15_models_for_nonideal_flow", imports=["from engreact import rtd"])
def _():
    return [
        [code(r"""U, L, D_ax, k_tau = 0.5, 20.0, 0.02, 3.0
Pe = U * L / D_ax
s2 = rtd.dispersion_variance(Pe)
n_eq = 1 / s2
X_pfr = 1 - np.exp(-k_tau)
X_disp = rtd.dispersion_conversion(k_tau / 1.0, 1.0, Pe)
print(f"Pe = {Pe:.0f}, sigma^2/tau^2 = {s2:.4f}, equivalent n = {n_eq:.0f}; conversion {X_disp:.4f} vs plug flow {X_pfr:.4f} (loss {X_pfr - X_disp:.4f})")
Pe_1pct = next(p for p in np.logspace(0, 4, 400) if (X_pfr - rtd.dispersion_conversion(k_tau, 1.0, p)) / X_pfr < 0.01)
print(f"the loss falls below 1 % of the conversion for Pe above about {Pe_1pct:.0f}")"""),
         md(r"""
A long packed bed has a Peclet number in the hundreds and behaves as plug flow to within a tenth of a
percent; dispersion matters in short beds, laminar tubes and vessels with internals. For this conversion the
loss drops below 1 % above Pe of about 45; the common rule of thumb of Pe > 100 carries a margin.
""")],
        [code(r"""t = np.linspace(0, 300, 12001)
E_c = rtd.cstr_E(t, 10.0)
k_half, C_A0 = 0.05, 1.0
seg = rtd.segregation_conversion(t, E_c, k_half, 0.5, C_A0)
mm = rtd.maximum_mixedness_conversion(t, E_c, k_half, 0.5, C_A0)
print(f"half-order reaction, CSTR RTD: segregation {seg:.4f}, maximum mixedness {mm:.4f} -> early mixing gives MORE conversion for a concave rate law")
seg2, mm2 = rtd.segregation_conversion(t, E_c, 0.05, 2, 1.0), rtd.maximum_mixedness_conversion(t, E_c, 0.05, 2, 1.0)
print(f"second order for comparison: segregation {seg2:.4f}, maximum mixedness {mm2:.4f}")"""),
         md(r"""
For a rate law concave in concentration (order below one), averaging concentrations before reacting is
better than reacting separately - Jensen's inequality in reverse - so maximum mixedness beats segregation.
The convex second-order case goes the other way. The sign of the difference is the curvature of the rate law.
""")],
        [code(r"""k, tau = 0.1, 10.0
t = np.linspace(0, 400, 40001)
for n in (1, 2, 5):
    analytic = 1 - (1 + k * tau / n) ** (-n)
    numeric = rtd.segregation_conversion(t, rtd.tanks_in_series_E(t, tau, n), k)
    print(f"n = {n}: analytic {analytic:.5f}, segregation on the tanks RTD {numeric:.5f}")"""),
         md(r"""
For first order the segregation integral over the tanks-in-series E(t) reproduces the closed form exactly -
the RTD suffices, and any consistent route gives the same answer.
""")],
    ]


# =====================================================================================================
@solutions("16_bioreactors_and_enzyme_kinetics", imports=["from scipy import integrate, optimize", "from engreact import bio"],
           setup=r"""mu_max, K_s, Y, S_in = 0.4, 0.2, 0.5, 20.0""")
def _():
    return [
        [code(r"""t = np.linspace(0, 60, 600)
plain = bio.batch_growth(0.05, 20.0, mu_max, K_s, Y, t)
real = bio.batch_growth(0.05, 20.0, mu_max, K_s, Y, t, k_d=0.01, m_s=0.02)
print(f"final biomass: ideal {plain['X'][-1]:.2f} g/L; with maintenance and death {real['X'][-1]:.2f} g/L (peak {real['X'].max():.2f} at t = {t[int(np.argmax(real['X']))]:.1f} h)")
plt.plot(t, plain["X"], label="ideal Monod"); plt.plot(t, real["X"], label="with maintenance and death"); plt.xlabel("t (h)"); plt.ylabel("X (g/L)"); plt.legend(); plt.show()"""),
         md(r"""
Maintenance diverts substrate from growth, lowering the peak; death makes the stationary phase a slow
decline instead of a plateau. Real growth curves show both, and fitting them with plain Monod
overestimates the yield.
""")],
        [code(r"""D = 0.3
s1 = bio.chemostat_steady(D, S_in, mu_max, K_s, Y)
# tank 2 receives biomass, so its balances must be solved: D (X1 - X2) + mu(S2) X2 = 0, D (S1 - S2) = mu(S2) X2 / Y
def tank2(y):
    S2, X2 = y
    mu = bio.monod(S2, mu_max, K_s)
    return [D * (s1["X"] - X2) + mu * X2, D * (s1["S"] - S2) - mu * X2 / Y]
S2, X2 = optimize.fsolve(tank2, [s1["S"] / 5, s1["X"] * 1.02])
s2 = {"S": S2}
print(f"tank 1: S = {s1['S']:.3f}, X = {s1['X']:.2f}; tank 2: S = {S2:.4f}, X = {X2:.2f} g/L")
S_target = 0.01 * S_in
D1 = optimize.brentq(lambda D_: bio.chemostat_steady(D_, S_in, mu_max, K_s, Y)["S"] - S_target, 1e-4, 0.39)
print(f"99 % substrate conversion in one chemostat needs D = {D1:.3f} 1/h (residence time {1/D1:.1f} h); two tanks at D = 0.3 each (total residence {2/0.3:.1f} h) reach S = {s2['S']:.4f}, i.e. {1 - s2['S']/S_in:.2%} conversion")"""),
         md(r"""
The second tank receives a dense culture and runs at a substrate level far below the first, mopping up what
a single tank cannot without a long residence time - the same series advantage as for CSTRs in notebook 02.
Its balances must be solved with the incoming biomass: the library's single-tank formula (which fixes S by
mu(S) = D) assumes a sterile feed and would wrongly give the same S in both tanks.
""")],
        [code(r"""S0, V_max, K_m = 2.0, 1.0, 0.5
sol = integrate.solve_ivp(lambda t, S: [-V_max * S[0] / (K_m + S[0])], (0, 3), [S0], t_eval=[0.5, 1.0, 1.5, 2.0, 2.5], rtol=1e-10)
lib = bio.batch_enzyme(S0, V_max, K_m, sol.t)
print(pd.DataFrame({"t": sol.t, "solve_ivp": sol.y[0], "integrated MM": lib}).round(5).to_string(index=False))"""),
         md(r"""
The integrated Michaelis-Menten equation is implicit in S, so the library solves it by root finding; the
direct integration agrees to five decimals.
""")],
    ]


# =====================================================================================================
@solutions("17_project_pilot_plant_to_reactor_design", imports=["from engreact import datasets, energy, kinetics, rate_data"],
           setup=r"""b = pd.read_csv(datasets.path("pilot_batch_kinetics.csv"), comment="#")
ks = {T: rate_data.nonlinear_fit(g.t_s, g.C_A_mol_m3, order=1)["k"] for T, g in b.groupby("T_K")}
arr = kinetics.fit_arrhenius(list(ks), list(ks.values()))
k0, Ea = arr["k0"], arr["Ea"]
C_A0, cp_vol, dH, kappa_pilot, T_a = 2500.0, 3.6e6, -265e3, 0.79, 295.0
tau_prod = 3600.0""")
def _():
    return [
        [code(r"""t = np.linspace(0, 6 * 3600, 2161)
for kappa, label in ((kappa_pilot, "pilot jacket ratio 0.79"), (2.0, "design jacket ratio 2.0")):
    ss = energy.cstr_steady_states(k0, Ea, dH, C_A0, 300.0, tau_prod, cp_vol, kappa, T_a, T_range=(250.0, 900.0))
    T_ss, X_ss = ss["T"][-1], ss["X"][-1]
    fail = energy.cstr_unsteady(k0, Ea, dH, C_A0, 300.0, tau_prod, cp_vol, t, C_A0 * (1 - X_ss), T_ss, UA_over_Vcp=0.0, T_a=T_a)
    i = np.nonzero(fail["T"] > T_ss + 30)[0]
    print(f"{label}: operating at {T_ss:.0f} K; after coolant failure T rises to {fail['T'].max():.0f} K" + (f", passing +30 K after {t[i[0]]/60:.0f} min" if i.size else " - never passes +30 K"))"""),
         md(r"""
Without cooling both reactors settle to the adiabatic steady state near 484 K: the pilot-jacket case rises
84 K, the kappa = 2 design 130 K, and the design case does so faster because it holds more unreacted feed.
Neither is a runaway in the explosive sense - the reaction is first order and finite - but a 130 K excursion
within a quarter of an hour is a relief-system case, and the trip at operating temperature + 30 K must act
within about ten minutes.
""")],
        [code(r"""rng = np.random.default_rng(2)
Eas, T_ign = [], []
Ts = sorted(ks)
for _ in range(200):
    ks_b = {}
    for T, g in b.groupby("T_K"):
        f = rate_data.nonlinear_fit(g.t_s, g.C_A_mol_m3, order=1)
        resid = g.C_A_mol_m3.to_numpy() - f["predict"](g.t_s.to_numpy())
        C_b = f["predict"](g.t_s.to_numpy()) + rng.choice(resid, resid.size, replace=True)
        ks_b[T] = rate_data.nonlinear_fit(g.t_s, C_b, order=1)["k"]
    a = kinetics.fit_arrhenius(Ts, [ks_b[T] for T in Ts])
    Eas.append(a["Ea"])
    T0s = np.linspace(280, 340, 61)
    n = np.array([len(energy.cstr_steady_states(a["k0"], a["Ea"], dH, C_A0, T0, 480.0, cp_vol, kappa_pilot, T_a, T_range=(250.0, 900.0))["T"]) for T0 in T0s])
    band = T0s[n == 3]
    T_ign.append(band.max() if band.size else np.nan)
print(f"Ea: {np.mean(Eas)/1e3:.1f} +/- {np.std(Eas)/1e3:.1f} kJ/mol (bootstrap); ignition temperature at tau = 480 s: {np.nanmean(T_ign):.1f} +/- {np.nanstd(T_ign):.1f} K")"""),
         md(r"""
Resampling the residuals and refitting two hundred times gives the spread of the activation energy directly
from the data, and pushing each fit through the multiplicity calculation propagates it to the ignition
temperature: an uncertainty of a kelvin or two on the edge of the safe window. That number belongs in the
report next to the window itself.
""")],
        [code(r"""def clean_pilot_runs(df):
    q, log = df.copy(), []
    for i in q.index[q.X > 1.05]:
        q.loc[i, "X"] /= 100; log.append(f"run {int(q.loc[i, 'run'])}: X in percent -> fraction")
    for i in q.index[q.T_out_K < 200]:
        q.loc[i, "T_out_K"] += 273.15; log.append(f"run {int(q.loc[i, 'run'])}: T_out in degC -> K")
    return q, log
p = pd.read_csv(datasets.path("pilot_cstr_runs.csv"), comment="#")
q, log = clean_pilot_runs(p)
print(log); print("all conversions in [0, 1.05] and temperatures above 250 K:", bool((q.X <= 1.05).all() and (q.T_out_K > 250).all()))"""),
         md(r"""
Two rules, applied mechanically and logged: the cleaning is reproducible and auditable, which matters more
than the two corrections themselves.
""")],
    ]


if __name__ == "__main__":
    write_all(only=[a for a in sys.argv[1:] if not a.startswith("--")])
