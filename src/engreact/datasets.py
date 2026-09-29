"""Bundled data files: a literature table with sources and pseudo-experimental course data whose generating
model, noise and deliberate faults are stated in each file's header.

>>> from engreact import datasets
>>> "pilot_cstr_runs.csv" in datasets.available()
True
"""

from __future__ import annotations

from importlib import resources

INFO = {
    "arrhenius_parameters.csv": "Arrhenius parameters (A, Ea) of eight classic reactions, from Laidler and Atkins.",
    "n2o5_decomposition.csv": "N2O5 decomposition in CCl4 at 45 degC: first-order batch data, 1 % noise, one mis-read sample.",
    "saponification.csv": "Equimolar ethyl acetate saponification at 25 degC: second-order batch data, 1.5 % noise.",
    "arrhenius_runs.csv": "Rate constants at six temperatures (3 % noise); one temperature recorded in degC instead of K.",
    "enzyme_initial_rates.csv": "Enzyme initial rates vs substrate concentration (Michaelis-Menten, 4 % noise).",
    "rtd_tracer_pulse.csv": "Tracer pulse response of a stirred tubular vessel (tanks-in-series n = 3.7) with a baseline offset and drift.",
    "catalyst_deactivation.csv": "Catalyst activity vs time on stream (second-order decay, 3 % noise).",
    "packed_bed_pressure_drop.csv": "Packed-bed pressure drop at seven superficial velocities (Ergun, 2 % noise).",
    "pilot_cstr_runs.csv": "Steady-state runs of an exothermic pilot CSTR with unit faults and a runaway run, for the project (notebook 17).",
    "pilot_batch_kinetics.csv": "Batch kinetic runs of the project reaction at three temperatures.",
}


def path(name: str):
    if name not in INFO:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(sorted(INFO))}")
    return resources.files("engreact.data") / name


def available() -> list[str]:
    return sorted(INFO)


def describe(name: str) -> str:
    return INFO[name]
