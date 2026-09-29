import numpy as np
import pytest

from engreact import bio, catalysis, energy, heterogeneous, multiple, rtd


def test_multiple_and_energy():
    assert multiple.series_optimum_time(1.0, 1.0) == 1.0 and multiple.series_max_B(1.0, 1.0) == pytest.approx(1 / np.e)
    net = multiple.Network([({"A": -1, "B": 1}, lambda C: 0.5 * C["A"])], species=["A", "B", "I"])
    out = net.pfr({"A": 1.0, "I": 2.0}, 2.0)
    assert out["I"][-1] == 2.0 and out["A"][-1] == pytest.approx(np.exp(-1.0), rel=1e-6)
    assert (
        float(multiple.instantaneous_selectivity(2.0, 0.5)) == 4.0
        and float(multiple.overall_selectivity(3.0, 1.5)) == 2.0
    )
    sem = energy.semenov_critical(-1e5, 1000.0, 1e8, 80e3, 400.0)
    assert sem["UA_over_V_critical"] > 0
    assert energy.adiabatic_temperature_rise(-100e3, 1000.0, 4e6) == pytest.approx(25.0)


def test_catalysis():
    assert catalysis.effectiveness_factor(0.0, "slab") == 1.0 and catalysis.effectiveness_factor(0.0, "cylinder") == 1.0
    assert catalysis.effectiveness_factor(1e-6, "sphere") == pytest.approx(1.0, abs=1e-6)
    arr = catalysis.effectiveness_factor(np.array([1.0, 2.0]), "sphere")
    assert arr.shape == (2,) and arr[0] > arr[1]
    assert catalysis.lhhw_rate(1.0, 2.0, 1.0, K_A=1.0, K_B=1.0, mode="surface") == pytest.approx(2.0 / 16)
    assert catalysis.lhhw_rate(1.0, 2.0, K_A=1.0, mode="single") == pytest.approx(2.0 / 3)
    assert float(catalysis.lhhw_rate(1.0, 2.0, K_A=0.5, K_P=1.0, p_P=1.0, mode="adsorption", Kp=4.0)) == pytest.approx(
        2.0 * (1 - 1 / 8) / (1 + 1 + 0.125)
    )
    assert catalysis.mears_criterion(1.0, 1.0, 0.1, 1, 10.0, 1.0) == pytest.approx(0.01)
    assert float(catalysis.activity(200.0, 0.01, "linear")) == 0.0
    tr = catalysis.temperature_time_trajectory(2e3, 60e3, 1e8, 50e3, 500.0, 5000.0, T_max=560.0)
    assert tr["reached_T_max"] and tr["T"][-1] == pytest.approx(560.0, abs=0.5)


def test_rtd_bio_heterogeneous():
    t = np.linspace(0, 60, 601)
    F = rtd.F_from_step(np.minimum(t / 30, 1.0) * 5.0, 5.0)
    assert F[-1] == 1.0
    assert rtd.pfr_E(t, 30.0).sum() * (t[1] - t[0]) == pytest.approx(1.0, rel=1e-3)
    assert rtd.tanks_in_series_conversion(0.1, 10.0, 1) == pytest.approx(0.5)
    assert rtd.dispersion_variance(1e-6, "open") > 1e6
    assert bio.michaelis_menten(1.0, 2.0, 0.5, inhibitor=1.0, K_I=1.0, inhibition="uncompetitive") == pytest.approx(
        2.0 / (0.5 + 2.0)
    )
    assert bio.michaelis_menten(1.0, 2.0, 0.5, inhibitor=1.0, K_I=1.0, inhibition="noncompetitive") == pytest.approx(
        1.0 / 1.5
    )
    assert float(bio.monod(1.0, 1.0, 1.0, K_i=1.0)) == 0.25
    assert bio.chemostat_steady(0.6, 10.0, 0.5, 0.1, 0.5)["washout"]
    assert (
        heterogeneous.conversion_at_time(100.0, 50.0, "film") == 1.0
        and heterogeneous.conversion_at_time(25.0, 50.0, "film") == 0.5
    )
    assert heterogeneous.regime_size_dependence("ash") == "R^2"
    assert heterogeneous.time_for_complete_conversion("film", 1.0, 1.0, 1, 1.0, k_g=2.0) == pytest.approx(1 / 6)


def test_package_exposes_every_module():
    import pkgutil

    import engreact

    modules = sorted(
        m.name for m in pkgutil.iter_modules(engreact.__path__) if not m.name.startswith("_") and m.name != "data"
    )
    assert all(hasattr(engreact, m) for m in modules), [m for m in modules if not hasattr(engreact, m)]
    assert set(modules) <= set(engreact.__all__)
