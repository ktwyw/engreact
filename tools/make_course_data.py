"""Write the course data files into src/engreact/data. The literature table is transcribed with its sources; the
pseudo-experimental files are generated from the library's own models with the parameters, noise level and
deliberate faults stated in each header (seed 2026). Run: python tools/make_course_data.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engreact import bio, catalysis, energy, ideal, kinetics, rate_data, rtd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "src" / "engreact" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)


def write(name, header_lines, rows, columns):
    def cell(v):
        v = str(v)
        return f'"{v}"' if "," in v else v

    lines = [f"# {line}" for line in header_lines] + [",".join(columns)] + [",".join(cell(v) for v in r) for r in rows]
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


# ---- literature: Arrhenius parameters of classic reactions
write(
    "arrhenius_parameters.csv",
    [
        "Arrhenius parameters of classic gas- and liquid-phase reactions, as tabulated in Laidler, Chemical Kinetics (3rd ed., 1987)",
        "and Atkins' Physical Chemistry; A in the units given, Ea in kJ/mol. Indicative textbook values.",
    ],
    [
        ("N2O5 decomposition (gas)", "4.9e13", "1/s", 103.4, "first order; Daniels & Johnston 1921"),
        ("cyclopropane -> propene (gas)", "1.6e15", "1/s", 272.0, "first order, unimolecular"),
        ("H2 + I2 -> 2 HI (gas)", "1.6e11", "L/(mol s)", 165.0, "second order; Bodenstein"),
        ("2 NO2 -> 2 NO + O2 (gas)", "2.0e9", "L/(mol s)", 111.0, "second order"),
        ("ethyl acetate + NaOH (aq)", "1.4e7", "L/(mol s)", 47.0, "second order saponification"),
        ("sucrose inversion (aq, acid)", "1.5e15", "1/s", 108.0, "pseudo-first order"),
        ("CH3COOC2H5 hydrolysis (aq, OH-)", "2.9e9", "L/(mol s)", 46.0, "second order"),
        ("C2H5Br + OH- (ethanol)", "4.3e11", "L/(mol s)", 89.5, "second order, SN2"),
    ],
    ["reaction", "A", "A_units", "Ea_kJ_mol", "notes"],
)

# ---- N2O5 decomposition: first order, batch, concentration vs time at 45 degC
k_n2o5 = 6.2e-4  # 1/s, order of the value at 45 degC
t = np.arange(0, 3601, 300)
C = 2.33e-3 * np.exp(-k_n2o5 * t) * (1 + rng.normal(0, 0.01, t.size))
rows = list(zip(t, np.round(C * 1e3, 4)))
rows[7] = (rows[7][0], round(rows[7][1] * 1.3, 4))  # a mis-read sample
write(
    "n2o5_decomposition.csv",
    [
        "Concentration of N2O5 during its decomposition in CCl4 at 45 degC, sampled every 5 min for an hour. Generated as first-order",
        "decay with k = 6.2e-4 1/s (the order of Daniels & Johnston's 1921 value) from 2.33 mmol/L with 1 % noise; the sample at",
        "35 min was mis-read (30 % high). Columns: time (s), concentration (mmol/L).",
    ],
    rows,
    ["t_s", "C_mmol_L"],
)

# ---- ethyl acetate saponification: second order (equimolar), 25 degC
k_sap = 0.11  # L/(mol s)
t = np.arange(0, 1201, 60)
C0 = 0.02
C = C0 / (1 + k_sap * C0 * t) * (1 + rng.normal(0, 0.015, t.size))
write(
    "saponification.csv",
    [
        "Equimolar saponification of ethyl acetate with NaOH at 25 degC, followed by conductivity, converted to NaOH concentration.",
        "Generated as second-order decay with k = 0.11 L/(mol s) (literature: 0.10-0.12) from 0.020 mol/L with 1.5 % noise.",
        "Columns: time (s), concentration (mol/L).",
    ],
    list(zip(t, np.round(C, 6))),
    ["t_s", "C_mol_L"],
)

# ---- Arrhenius series with one temperature recorded in degC instead of K
T = np.array([300, 310, 320, 330, 340, 350.0])
k = kinetics.arrhenius(3.0e9, 72e3, T) * (1 + rng.normal(0, 0.03, T.size))
rows = list(zip(T, np.round(k, 8)))
rows[3] = (57.0, rows[3][1])  # 330 K written as 57 degC
write(
    "arrhenius_runs.csv",
    [
        "Rate constants of a first-order liquid-phase reaction measured at six temperatures. Generated from A = 3.0e9 1/s,",
        "Ea = 72 kJ/mol with 3 % noise; the fourth temperature was recorded in degC (57) instead of K (330).",
        "Columns: temperature (K), k (1/s).",
    ],
    rows,
    ["T_K", "k_1_s"],
)

# ---- enzyme kinetics (initial rates vs substrate) with product inhibition absent
S = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
v = bio.michaelis_menten(S, 1.8, 0.35) * (1 + rng.normal(0, 0.04, S.size))
write(
    "enzyme_initial_rates.csv",
    [
        "Initial rates of an enzyme-catalysed reaction against substrate concentration (25 degC, pH 7). Generated from",
        "Michaelis-Menten with V_max = 1.8 mmol/(L min) and K_m = 0.35 mmol/L with 4 % noise. Columns: S (mmol/L), v (mmol/(L min)).",
    ],
    list(zip(S, np.round(v, 5))),
    ["S_mmol_L", "v_mmol_L_min"],
)

# ---- RTD tracer pulse from a real-ish stirred vessel: tanks-in-series n = 3.7, baseline drift
t = np.arange(0, 181, 3.0)
E = rtd.tanks_in_series_E(t, 40.0, 3.7)
conc = 12.0 * E * 40.0 / 1.0
conc = conc * (1 + rng.normal(0, 0.03, t.size)) + 0.05 + 0.0008 * t  # baseline offset and slow drift
write(
    "rtd_tracer_pulse.csv",
    [
        "Outlet tracer concentration after a pulse injection into a stirred tubular vessel (mean residence time about 40 s).",
        "Generated from the tanks-in-series model with n = 3.7, tau = 40 s, 3 % noise, a baseline offset of 0.05 g/m3 and a",
        "slow baseline drift (0.0008 g/m3 per s) as a real detector shows. Columns: time (s), concentration (g/m3).",
    ],
    list(zip(t, np.round(conc, 4))),
    ["t_s", "C_g_m3"],
)

# ---- catalyst deactivation: activity vs time on stream
t = np.array([0, 5, 10, 20, 40, 60, 90, 120, 180, 240.0])
a = catalysis.activity(t, 0.012, "second") * (1 + rng.normal(0, 0.03, t.size))
write(
    "catalyst_deactivation.csv",
    [
        "Relative activity of a hydrocracking catalyst against time on stream (h), from the conversion at fixed conditions.",
        "Generated from second-order decay, k_d = 0.012 1/h, with 3 % noise. Columns: time (h), activity (-).",
    ],
    list(zip(t, np.round(a, 4))),
    ["t_h", "activity"],
)

# ---- packed-bed pressure drop at several superficial velocities
u = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6])
rho, mu, Dp, phi, L = 1.15, 1.9e-5, 4e-3, 0.42, 1.5
dp = np.array([ideal.ergun(rho * uu, rho, mu, Dp, phi) * L for uu in u]) * (1 + rng.normal(0, 0.02, u.size))
write(
    "packed_bed_pressure_drop.csv",
    [
        "Pressure drop across a 1.5 m bed of 4 mm catalyst spheres (void fraction 0.42) for air at 60 degC (rho = 1.15 kg/m3,",
        "mu = 1.9e-5 Pa s) at several superficial velocities. Generated from the Ergun equation with 2 % noise.",
        "Columns: superficial velocity (m/s), pressure drop (Pa).",
    ],
    list(zip(u, np.round(dp, 1))),
    ["u_m_s", "dP_Pa"],
)

# ---- pilot-plant CSTR runs for the project: exothermic first-order A -> B in a jacketed CSTR
k0, Ea, dH, C_A0, cp_vol = 5.0e7, 68e3, -250e3, 2500.0, 3.6e6  # SI; Delta T_ad = 174 K
rows = []
run = 1
for T0 in (290, 295, 300, 305, 310, 315, 320, 325):
    for tau in (120.0, 240.0, 480.0):
        ss = energy.cstr_steady_states(
            k0, Ea, dH, C_A0, T0, tau, cp_vol, UA_over_v0cp=0.8, T_a=295.0, T_range=(250.0, 700.0)
        )
        # the operators always start the reactor cold: the lowest steady state is what they observe
        T_ss, X_ss = ss["T"][0], ss["X"][0]
        rows.append(
            (run, T0, tau, round(T_ss + rng.normal(0, 0.4), 1), round(X_ss * (1 + rng.normal(0, 0.02)), 4), "ok")
        )
        run += 1
ign = energy.cstr_steady_states(
    k0, Ea, dH, C_A0, 330, 480.0, cp_vol, UA_over_v0cp=0.8, T_a=295.0, T_range=(250.0, 800.0)
)
rows.append(
    (run, 330, 480.0, round(ign["T"][-1], 1), round(ign["X"][-1], 4), "runaway: reactor tripped on high temperature")
)
rows[5] = (rows[5][0], rows[5][1], rows[5][2], rows[5][3], rows[5][4] * 100, "ok")  # conversion entered in percent
rows[12] = (rows[12][0], rows[12][1], rows[12][2], rows[12][3] - 273.15, rows[12][4], "ok")  # outlet T in degC
write(
    "pilot_cstr_runs.csv",
    [
        "Steady-state runs of a jacketed pilot CSTR (0.5 m3) for the exothermic liquid-phase reaction A -> B: feed temperature,",
        "space time, outlet temperature and conversion. Generated from a first-order rate law (A = 5.0e7 1/s, Ea = 68 kJ/mol),",
        "Delta H = -250 kJ/mol, C_A0 = 2500 mol/m3, rho cp = 3.6 MJ/(m3 K), jacket kappa = UA/(v0 rho cp) = 0.8 at T_a = 295 K,",
        "with 0.4 K and 2 % measurement noise; the operators start cold so the low steady state is the one observed. Faults:",
        "one conversion entered in percent, one outlet temperature in degC, and a final run at 330 K that ignited (ran away).",
        "Columns: run, T_feed (K), tau (s), T_out (K), X (-), note.",
    ],
    rows,
    ["run", "T_feed_K", "tau_s", "T_out_K", "X", "note"],
)

# ---- batch rate data for the project reaction at three temperatures
rows = []
for T in (300.0, 315.0, 330.0):
    kk = kinetics.arrhenius(k0, Ea, T)
    t = np.arange(0, 7201, 600)
    C = rate_data.integrated_concentration(t, C_A0, kk, 1) * (1 + rng.normal(0, 0.015, t.size))
    rows += [(T, tt, round(cc, 2)) for tt, cc in zip(t, C)]
write(
    "pilot_batch_kinetics.csv",
    [
        "Batch kinetic runs for the project reaction A -> B at three temperatures (isothermal, 1 L flask): concentration of A",
        "against time. Generated from the same rate law as pilot_cstr_runs.csv (first order, A = 5.0e7 1/s, Ea = 68 kJ/mol)",
        "with 1.5 % noise. Columns: temperature (K), time (s), C_A (mol/m3).",
    ],
    rows,
    ["T_K", "t_s", "C_A_mol_m3"],
)
