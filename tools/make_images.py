"""Generate the README images (docs/images/*.png and *.gif) from the library. Run: python tools/make_images.py"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engreact import bio, catalysis, energy, ideal, multiple, rtd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ 1. Levenspiel plot
X = np.array([0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8])
rate = np.array([0.45, 0.37, 0.30, 0.195, 0.113, 0.079, 0.05])
lev = ideal.levenspiel(X, rate, 0.4)
fig, ax = plt.subplots(figsize=(5.6, 4))
Xf = np.linspace(0, 0.8, 200)
ax.plot(X, lev["F_over_r"], "ko-", label="F_A0 / (-r_A)")
ax.fill_between(Xf, np.interp(Xf, X, lev["F_over_r"]), alpha=0.25, label=f"PFR area: {lev['V_pfr'][-1]:.1f} m3")
ax.bar(
    [0.4],
    [lev["F_over_r"][-1]],
    width=0.8,
    fill=False,
    edgecolor="C3",
    hatch="//",
    lw=1.5,
    label=f"CSTR rectangle: {lev['V_cstr'][-1]:.1f} m3",
)
ax.set(
    xlabel="conversion X",
    ylabel="F_A0 / (-r_A) (m3)",
    xlim=(0, 0.85),
    ylim=(0, 9),
    title="Levenspiel plot: sizing a PFR and a CSTR",
)
ax.legend(frameon=False, fontsize=8)
save(fig, "levenspiel.png")

# ------------------------------------------------------------------ 2. series reactions A -> B -> C
t = np.linspace(0, 20, 400)
b = multiple.series_batch(0.5, 0.2, 1.0, t)
fig, ax = plt.subplots(figsize=(5.6, 4))
for s in "ABC":
    ax.plot(t, b[s], label=s)
t_opt = multiple.series_optimum_time(0.5, 0.2)
ax.axvline(t_opt, color="k", ls=":", label=f"t_opt = {t_opt:.2f}")
ax.set(xlabel="time", ylabel="C / C_A0", title="Series reactions A -> B -> C: the intermediate peaks")
ax.legend(frameon=False)
save(fig, "series_reactions.png")

# ------------------------------------------------------------------ 3. effectiveness factor
phi = np.logspace(-1, 2, 200)
fig, ax = plt.subplots(figsize=(5.6, 4))
for geom in ("slab", "cylinder", "sphere"):
    ax.loglog(phi, catalysis.effectiveness_factor(phi, geom), label=geom)
ax.loglog(phi, np.minimum(1, 3 / phi), "k:", label="3/phi")
ax.set(xlabel="Thiele modulus", ylabel="effectiveness factor", title="Internal diffusion: effectiveness factor")
ax.legend(frameon=False)
save(fig, "effectiveness_factor.png")

# ------------------------------------------------------------------ 4. multiple steady states
k0, Ea, dH, C_A0, cp_vol = 5.0e7, 68e3, -250e3, 2500.0, 3.6e6
tau, kappa, T_a = 240.0, 0.8, 295.0
T = np.linspace(280, 480, 400)
fig, ax = plt.subplots(figsize=(5.6, 4))
for T0, col in ((300.0, "C0"), (315.0, "C1"), (330.0, "C3")):
    c = energy.cstr_heat_curves(k0, Ea, dH, C_A0, T0, tau, cp_vol, T, kappa, T_a)
    ax.plot(T, c["R"] / 1e6, color=col, ls="--", label=f"removed, T0 = {T0:.0f} K")
ax.plot(T, c["G"] / 1e6, "k", lw=2, label="generated G(T)")
ss = energy.cstr_steady_states(k0, Ea, dH, C_A0, 315.0, tau, cp_vol, kappa, T_a, T_range=(250.0, 800.0))
for T_, st in zip(ss["T"], ss["stable"]):
    ax.plot(
        T_,
        float(energy.cstr_heat_curves(k0, Ea, dH, C_A0, 315.0, tau, cp_vol, T_, kappa, T_a)["G"]) / 1e6,
        "o" if st else "X",
        color="C1",
        ms=11,
        mec="k",
        zorder=5,
    )
ax.set(
    xlabel="reactor temperature (K)",
    ylabel="heat per m3 of feed (MJ/m3)",
    ylim=(0, 700),
    title="Exothermic CSTR: three steady states",
)
ax.legend(frameon=False, fontsize=8)
save(fig, "multiple_steady_states.png")

# ------------------------------------------------------------------ 5. RTD models
theta = np.linspace(0.001, 3, 500)
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.plot(theta, rtd.cstr_E(theta, 1.0), label="CSTR")
for n in (3, 10, 50):
    ax.plot(theta, rtd.tanks_in_series_E(theta, 1.0, n), label=f"{n} tanks in series")
ax.plot(theta, rtd.dispersion_E(theta, 20.0), "k--", label="dispersion, Pe = 20")
ax.set(xlabel="theta = t / tau", ylabel="E(theta)", ylim=(0, 3.2), title="Residence-time distributions")
ax.legend(frameon=False, fontsize=8)
save(fig, "rtd_models.png")

# ------------------------------------------------------------------ 6. chemostat
mu_max, K_s, Y, S_in = 0.4, 0.2, 0.5, 20.0
Ds = np.linspace(0.005, 0.42, 300)
ss_b = [bio.chemostat_steady(D, S_in, mu_max, K_s, Y) for D in Ds]
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.plot(Ds, [s["X"] for s in ss_b], label="biomass X")
ax.plot(Ds, [s["S"] for s in ss_b], label="substrate S")
ax.plot(Ds, [s.get("productivity", 0.0) for s in ss_b], label="productivity D X")
ax.axvline(bio.chemostat_washout(mu_max, K_s, S_in), color="k", ls=":", label="washout")
ax.set(xlabel="dilution rate (1/h)", ylabel="g/L", title="Chemostat: steady state, optimum and washout")
ax.legend(frameon=False, fontsize=8)
save(fig, "chemostat.png")

# ------------------------------------------------------------------ hero: the removal line sweeps the S-curve
T0s = np.concatenate([np.linspace(290, 335, 30), np.linspace(335, 290, 30)])
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(T, c["G"] / 1e6, "k", lw=2, label="heat generated")
(line,) = ax.plot([], [], "C3--", lw=1.5, label="heat removed")
dots = ax.scatter([], [], s=90, zorder=5, edgecolor="k")
ax.set(xlabel="reactor temperature (K)", ylabel="heat per m3 of feed (MJ/m3)", xlim=(280, 480), ylim=(0, 700))
ax.legend(frameon=False, loc="upper left")
title = ax.set_title("")
fig.tight_layout()


def update(i):
    T0 = T0s[i]
    c_ = energy.cstr_heat_curves(k0, Ea, dH, C_A0, T0, tau, cp_vol, T, kappa, T_a)
    line.set_data(T, c_["R"] / 1e6)
    ss_ = energy.cstr_steady_states(k0, Ea, dH, C_A0, T0, tau, cp_vol, kappa, T_a, T_range=(250.0, 800.0))
    pts = np.array(
        [
            [T_, float(energy.cstr_heat_curves(k0, Ea, dH, C_A0, T0, tau, cp_vol, T_, kappa, T_a)["G"]) / 1e6]
            for T_ in ss_["T"]
        ]
    )
    dots.set_offsets(pts)
    dots.set_color(["C0" if st else "C1" for st in ss_["stable"]])
    n = len(ss_["T"])
    state = "one steady state" if n == 1 else "three steady states (orange = unstable)"
    title.set_text(f"Feed at {T0:.0f} K, {'heating' if i < 30 else 'cooling'}: {state}")
    return line, dots, title


anim = FuncAnimation(fig, update, frames=len(T0s), blit=False)
anim.save(OUT / "cstr_ignition.gif", writer=PillowWriter(fps=6), dpi=80)
plt.close(fig)
print("wrote cstr_ignition.gif")
