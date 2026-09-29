"""engreact - chemical reaction engineering: readable, validated Python with a notebook course.

Modules
-------
constants      physical constants
thermo         species table, heats of reaction, equilibrium constants, adiabatic temperature
kinetics       Arrhenius, rate laws, stoichiometric tables, expansion factor, equilibrium conversion
rate_data      differential, integral, half-life and initial-rate methods; nonlinear regression; model comparison
ideal          batch, CSTR, PFR and PBR design equations; Levenspiel plots; reactors in series; recycle; Ergun
multiple       series and parallel reactions (Bateman), selectivity and yield, general reaction networks, semibatch
energy         adiabatic and heat-exchange PFR, CSTR heat curves and multiple steady states, unsteady CSTR, runaway
catalysis      LHHW rate laws, Thiele modulus and effectiveness factors, Weisz-Prater and Mears, deactivation
rtd            E(t) and F(t), moments, tanks-in-series and dispersion models, segregation and maximum mixedness
bio            Michaelis-Menten and inhibition, Monod, chemostat washout and optimum, batch and fed-batch growth
heterogeneous  shrinking-core model: film, ash and reaction control, combined resistances
datasets       bundled literature table and course data files
"""

__version__ = "0.1.0"

from . import (
    bio,
    catalysis,
    constants,
    datasets,
    energy,
    heterogeneous,
    ideal,
    kinetics,
    multiple,
    rate_data,
    rtd,
    thermo,
)

__all__ = [
    "__version__",
    "bio",
    "catalysis",
    "constants",
    "datasets",
    "energy",
    "heterogeneous",
    "ideal",
    "kinetics",
    "multiple",
    "rate_data",
    "rtd",
    "thermo",
]
