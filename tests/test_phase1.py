import math

import numpy as np
import pytest

from engreact import ideal, kinetics, rate_data, thermo
from engreact.kinetics import Reaction


def test_thermo():
    assert thermo.heat_of_reaction({"H2": -2, "O2": -1, "H2O": 2}) == pytest.approx(-483640.0)
    assert thermo.heat_of_reaction({"H2": -2, "O2": -1, "H2O(l)": 2}) == pytest.approx(-571660.0)
    assert thermo.gibbs_of_reaction({"N2": -1, "H2": -3, "NH3": 2}) == pytest.approx(-32900.0)
    assert thermo.enthalpy_change("N2", 300.0, 300.0) == 0.0 and thermo.entropy_change("N2", 300.0, 300.0) == 0.0
    assert thermo.delta_cp({"N2": -1, "O2": 1}, 400.0) == pytest.approx(
        float(thermo.cp("O2", 400.0) - thermo.cp("N2", 400.0))
    )
    with pytest.raises(RuntimeError):
        thermo.adiabatic_temperature(
            298.15, 1.0, {"CH4": -1, "O2": -2, "CO2": 1, "H2O": 2}, "CH4", {"CH4": 1.0, "O2": 2.0}
        )  # no inert: above the table


def test_kinetics_and_stoichiometry():
    rxn = Reaction({"A": -1, "B": -1, "C": 1}, "A")
    assert repr(rxn) == "A + B -> C" and rxn.delta == -1 and rxn.nu("Z") == 0
    assert kinetics.limiting_reactant(rxn, {"A": 2.0, "B": 1.0}) == "B"
    assert float(kinetics.power_law(2.0, [1.0, 3.0], [1, 2])) == 18.0
    assert kinetics.fit_arrhenius([300, 400], kinetics.arrhenius(1e3, 20e3, np.array([300, 400.0])))[
        "Ea"
    ] == pytest.approx(20e3)
    with pytest.raises(ValueError):
        kinetics.equilibrium_conversion(rxn, {"A": 1.0, "B": 1.0}, Kc=-1.0, C_A0=1.0)


def test_rate_data_and_ideal():
    t = np.linspace(0, 50, 11)
    C = 2.0 * np.exp(-0.05 * t)
    assert rate_data.nonlinear_fit(t, C, order=1, fit_C0=False)["k"] == pytest.approx(0.05)
    assert rate_data.integrated_concentration(1e6, 1.0, 0.1, 0.5) == 0.0  # clipped at exhaustion
    assert ideal.damkohler(0.5, 4.0) == 2.0 and ideal.space_time(2.0, 0.5) == 4.0
    assert ideal.batch_time_nth_order(0.1, 1.0, 0.5, 1) == pytest.approx(math.log(2) / 0.1)
    rate = lambda X: 0.2 * (1 - X)  # noqa: E731
    r = ideal.reactors_in_series(1.0, [("pfr", 2.0), ("cstr", 3.0)], rate)
    assert r[0]["X_out"] == pytest.approx(1 - math.exp(-0.4)) and r[1]["X_in"] == r[0]["X_out"]
    assert ideal.ergun_alpha(1000.0, 1e5, 0.01, 2000.0, 0.4) == pytest.approx(2 * 1000 / (0.01 * 2000 * 0.6 * 1e5))
