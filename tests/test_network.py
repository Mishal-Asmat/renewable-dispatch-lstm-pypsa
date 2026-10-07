import numpy as np
import pandas as pd
import pytest

pypsa = pytest.importorskip("pypsa")

import build_network as bn
import config as cfg


def small_network(hours=48):
    idx = pd.date_range("2019-01-14", periods=hours, freq="h")
    solar = pd.Series(np.clip(np.sin(np.pi * (idx.hour - 6) / 12), 0, 1) * 0.5, index=idx)
    wind = pd.Series(0.4, index=idx)
    return idx, bn.build_network(idx, bn.synthetic_demand(idx), solar, wind)


def test_cost_formula_matches_v1_at_zero_tax():
    assert bn.thermal_cost(cfg.GAS, 0) == pytest.approx(50.0)
    assert bn.thermal_cost(cfg.COAL, 0) == pytest.approx(30.0)


def test_breakeven_equalises_marginal_costs():
    t = bn.breakeven_tax()
    assert bn.thermal_cost(cfg.GAS, t) == pytest.approx(bn.thermal_cost(cfg.COAL, t))
    assert 30 < t < 40          # ~33.1 EUR/t with the default parameters


def test_synthetic_demand_mean():
    idx = pd.date_range("2019-01-14", periods=168, freq="h")
    d = bn.synthetic_demand(idx)
    assert d.mean() == pytest.approx(cfg.MEAN_DEMAND_MW)
    assert d.min() > 0.5 * cfg.MEAN_DEMAND_MW


def test_fuel_switch_brackets_analytic_breakeven():
    _, n = small_network()
    t = bn.breakeven_tax()
    below, _ = bn.run_scenario(n, t - 5)
    above, _ = bn.run_scenario(n, t + 5)
    assert below["gas_mwh"] < 1 and below["coal_mwh"] > 0
    assert above["coal_mwh"] < 1 and above["gas_mwh"] > 0
    assert above["emissions_tco2"] < below["emissions_tco2"]


def test_energy_balance_every_hour():
    _, n = small_network()
    _, h = bn.run_scenario(n, 50)
    supply = h[["coal", "gas", "solar", "wind", "battery_discharge"]].sum(axis=1) - h["battery_charge"]
    np.testing.assert_allclose(supply, h["demand"], atol=1e-6)


def test_battery_is_cyclic():
    _, n = small_network(hours=72)
    # abundant renewables make the battery useful
    n2 = n.copy()
    n2.generators.loc[["solar_farm", "wind_farm"], "p_nom"] *= 4
    _, h = bn.run_scenario(n2, 50)
    # cyclic state of charge: round-trip losses only, no net energy created
    assert h["battery_discharge"].sum() <= h["battery_charge"].sum() * 0.9 * 0.9 + 1e-6


def test_real_demand_is_rescaled_to_mean(tmp_path, monkeypatch):
    idx = pd.date_range("2019-01-01", periods=24 * 30, freq="h")
    fc = pd.DataFrame({"solar_fc_h24_mw": 1000.0, "wind_fc_h24_mw": 2000.0,
                       "solar_actual_mw": 1100.0, "wind_actual_mw": 2100.0,
                       "solar_cap_mw": 40000.0, "wind_cap_mw": 40000.0,
                       "load_mw": 50000.0 + 5000 * np.sin(np.arange(len(idx)) / 5)}, index=idx)
    fc.index.name = "utc_timestamp"
    path = tmp_path / "forecast.csv"
    fc.to_csv(path)
    monkeypatch.setattr(cfg, "FORECAST_CSV", path)
    idx_w, avail, load = bn.load_availability("2019-01-08")
    assert len(load) == cfg.DISPATCH_HOURS
    assert 0.8 * cfg.MEAN_DEMAND_MW < load.mean() < 1.2 * cfg.MEAN_DEMAND_MW
    assert avail["forecast"]["wind"].between(0, 1).all()
