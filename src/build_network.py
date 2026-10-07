"""
Stage 2 - least-cost dispatch with PyPSA under a range of carbon taxes.

  * Fuel-based thermal costs with efficiencies:
        marginal_cost = fuel_price/eff + emission_factor_thermal/eff * tax
    so the carbon tax acts on emissions per MWh of ELECTRICITY.
  * Renewable availability is the LSTM *day-ahead style* (24 h-ahead) forecast,
    normalised by a fixed capacity reference. Each scenario is also solved with
    perfect foresight (actual availability) as a reference.
  * Two representative out-of-sample weeks (winter and summer).
  * Cyclic battery state of charge is used.
  * Real hourly German load (ENTSO-E via OPSD), rescaled to a 500 MW annual mean.
  * Emissions, system cost, curtailment and shadow price are reported per scenario.
  * Each scenario is solved ONCE and reused for tables and figures.
  * Battery charging is plotted below zero.

Usage (repository root):  python src/build_network.py
Requires output/forecast.csv from lstm_model.py.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pypsa

import config as cfg

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pypsa").setLevel(logging.ERROR)
logging.getLogger("linopy").setLevel(logging.WARNING)


# -- Cost model --

def thermal_cost(plant: dict, tax: float) -> float:
    """Marginal cost in EUR per MWh of electricity, incl. carbon tax."""
    return (plant["fuel_price"] + plant["ef_thermal"] * tax) / plant["eff"]


def emissions_per_mwh_el(plant: dict) -> float:
    """tCO2 per MWh of electricity."""
    return plant["ef_thermal"] / plant["eff"]


def breakeven_tax(coal: dict = cfg.COAL, gas: dict = cfg.GAS) -> float:
    """Carbon tax (EUR/t) at which coal and gas have equal marginal cost."""
    return (thermal_cost(gas, 0) - thermal_cost(coal, 0)) / (
        emissions_per_mwh_el(coal) - emissions_per_mwh_el(gas))


def synthetic_demand(index: pd.DatetimeIndex, mean_mw: float = cfg.MEAN_DEMAND_MW
                     ) -> pd.Series:
    """Stylised load: smooth daily cycle (trough ~03:00, peak ~15:00 UTC) and
    8% lower on weekends, rescaled so that the mean equals `mean_mw`."""
    hour = np.asarray(index.hour)
    shape = 1 + 0.18 * np.sin(2 * np.pi * (hour - 9) / 24)
    shape = shape * np.where(np.asarray(index.dayofweek) >= 5, 0.92, 1.0)
    return pd.Series(shape / shape.mean() * mean_mw, index=index, name="load_mw")


# -- Building Network --

def build_network(index: pd.DatetimeIndex, load: pd.Series, solar_pu: pd.Series,
                  wind_pu: pd.Series, re_scale: float = 1.0) -> pypsa.Network:
    """Single-bus system. `re_scale` multiplies solar and wind capacity
    (used only for the high-renewables sensitivity analysis)."""
    n = pypsa.Network()
    n.set_snapshots(index)
    n.add("Carrier", "AC")

    n.add("Carrier", "gas", co2_emissions=cfg.GAS["ef_thermal"])
    n.add("Carrier", "coal", co2_emissions=cfg.COAL["ef_thermal"])
    n.add("Carrier", ["solar", "wind", "battery"])
    n.add("Bus", "main_grid", v_nom=110, carrier="AC")
    n.add("Load", "city_demand", bus="main_grid", p_set=load)

    n.add("Generator", "gas_plant", bus="main_grid", carrier="gas",
          p_nom=cfg.GAS["p_nom"], efficiency=cfg.GAS["eff"],
          marginal_cost=thermal_cost(cfg.GAS, 0))
    n.add("Generator", "coal_plant", bus="main_grid", carrier="coal",
          p_nom=cfg.COAL["p_nom"], efficiency=cfg.COAL["eff"],
          marginal_cost=thermal_cost(cfg.COAL, 0))
    n.add("Generator", "solar_farm", bus="main_grid", carrier="solar",
          p_nom=cfg.SOLAR_P_NOM * re_scale, marginal_cost=0.0, p_max_pu=solar_pu)
    n.add("Generator", "wind_farm", bus="main_grid", carrier="wind",
          p_nom=cfg.WIND_P_NOM * re_scale, marginal_cost=0.0, p_max_pu=wind_pu)

    b = cfg.BATTERY
    n.add("StorageUnit", "big_battery", bus="main_grid", carrier="battery",
          p_nom=b["p_nom"], max_hours=b["max_hours"],
          efficiency_store=b["eff_store"], efficiency_dispatch=b["eff_dispatch"],
          marginal_cost=b["marginal_cost"], cyclic_state_of_charge=True)
    return n


def run_scenario(n: pypsa.Network, tax: float):
    """Solve the network for one carbon tax and return (summary dict, hourly df)."""
    m = n.copy()
    m.generators.loc["gas_plant", "marginal_cost"] = thermal_cost(cfg.GAS, tax)
    m.generators.loc["coal_plant", "marginal_cost"] = thermal_cost(cfg.COAL, tax)
    status, condition = m.optimize(solver_name="highs", log_to_console=False,
                                   include_objective_constant=False)
    if status != "ok":
        raise RuntimeError(f"Optimisation failed at tax={tax}: {status}/{condition}")

    p = m.generators_t.p
    su = m.storage_units_t
    hourly = pd.DataFrame({
        "demand": m.loads_t.p_set["city_demand"],
        "coal": p["coal_plant"], "gas": p["gas_plant"],
        "solar": p["solar_farm"], "wind": p["wind_farm"],
        "battery_discharge": su.p_dispatch["big_battery"],
        "battery_charge": su.p_store["big_battery"],
        "battery_soc": su.state_of_charge["big_battery"],
    })
    avail = (m.generators_t.p_max_pu[["solar_farm", "wind_farm"]]
             * m.generators.loc[["solar_farm", "wind_farm"], "p_nom"])
    emissions = (hourly["gas"].sum() * emissions_per_mwh_el(cfg.GAS)
                 + hourly["coal"].sum() * emissions_per_mwh_el(cfg.COAL))
    summary = {
        "coal_mwh": hourly["coal"].sum(), "gas_mwh": hourly["gas"].sum(),
        "solar_mwh": hourly["solar"].sum(), "wind_mwh": hourly["wind"].sum(),
        "battery_discharge_mwh": hourly["battery_discharge"].sum(),
        "battery_charge_mwh": hourly["battery_charge"].sum(),
        "curtailed_mwh": max(0.0, float(avail.sum().sum()
                                        - hourly["solar"].sum() - hourly["wind"].sum())),
        "demand_mwh": hourly["demand"].sum(),
        "emissions_tco2": emissions,
        "carbon_cost_eur": tax * emissions,
        "objective_eur": m.objective,
        "mean_price_eur_mwh": m.buses_t.marginal_price["main_grid"].mean(),
    }
    return summary, hourly


# -- Inputs --

def load_availability(week_start: str, hours: int = cfg.DISPATCH_HOURS):
    """Per-unit availability (forecast and actual) and demand for one dispatch week.

    Returns (index, availability, load). Demand is the real German load rescaled so
    that its mean over the whole forecast period equals cfg.MEAN_DEMAND_MW.
    """
    fc = pd.read_csv(cfg.FORECAST_CSV, index_col=0, parse_dates=True)
    h = cfg.FORECAST_HORIZON_FOR_DISPATCH
    idx = pd.date_range(week_start, periods=hours, freq="h")
    week = fc.reindex(idx)
    if week.isna().any().any():
        raise ValueError(f"Forecast file does not cover the week starting {week_start}")
    out = {"forecast": {}, "actual": {}}
    for s in cfg.TARGETS:
        out["forecast"][s] = (week[f"{s}_fc_h{h}_mw"] / week[f"{s}_cap_mw"]).clip(0, 1)
        out["actual"][s] = (week[f"{s}_actual_mw"] / week[f"{s}_cap_mw"]).clip(0, 1)
    if "load_mw" in fc.columns and not week["load_mw"].isna().any():
        load = (week["load_mw"] * cfg.MEAN_DEMAND_MW / fc["load_mw"].mean()).rename("load_mw")
    else:
        load = synthetic_demand(idx)
    return idx, out, load


def run_all(taxes=cfg.CARBON_TAXES, weeks=cfg.WEEKS):
    rows, hourly_rows = [], []
    for season, start in weeks.items():
        idx, avail, load = load_availability(start)
        for source in ("forecast", "actual"):
            n = build_network(idx, load, avail[source]["solar"], avail[source]["wind"])
            for tax in taxes:
                summary, hourly = run_scenario(n, tax)
                rows.append(dict(week=season, week_start=start, availability=source,
                                 carbon_tax=tax, **summary))
                if source == "forecast" and tax in cfg.PLOT_TAXES:
                    h = hourly.copy()
                    h.insert(0, "carbon_tax", tax)
                    h.insert(0, "week", season)
                    hourly_rows.append(h)
            print(f"  solved {season:6s} / {source:8s} ({len(taxes)} taxes)")
    results = pd.DataFrame(rows)
    hourly_all = pd.concat(hourly_rows).rename_axis("snapshot").reset_index()
    return results, hourly_all


def run_sensitivity(scales=cfg.RE_SCALES, taxes=cfg.SENSITIVITY_TAXES,
                    weeks=cfg.WEEKS) -> pd.DataFrame:
    """How do curtailment, emissions and battery use change when the renewable
    fleet is scaled up? (Forecast availability, same demand.)"""
    rows = []
    for season, start in weeks.items():
        idx, avail, load = load_availability(start)
        for scale in scales:
            n = build_network(idx, load, avail["forecast"]["solar"],
                              avail["forecast"]["wind"], re_scale=scale)
            for tax in taxes:
                summary, _ = run_scenario(n, tax)
                rows.append(dict(week=season, re_scale=scale, carbon_tax=tax, **summary))
    return pd.DataFrame(rows)


# -- Figures --

def make_figures(results: pd.DataFrame, hourly: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = dict(coal="#222222", gas="#3b6fb6", solar="#f2a900", wind="#2e9e5b",
                  battery_discharge="#8e44ad")
    be = breakeven_tax()

    # 1) dispatch grids (one per season)
    for season in hourly["week"].unique():
        fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True, sharey=True)
        for ax, tax in zip(axes.flat, cfg.PLOT_TAXES):
            d = hourly[(hourly.week == season) & (hourly.carbon_tax == tax)]
            x = np.arange(len(d))
            ax.stackplot(x, d["coal"], d["gas"], d["solar"], d["wind"],
                         d["battery_discharge"], colors=list(colors.values()),
                         labels=["Coal", "Gas", "Solar", "Wind", "Battery discharge"])
            ax.fill_between(x, 0, -d["battery_charge"], color="#d7b8e6",
                            label="Battery charging")
            ax.plot(x, d["demand"], "r--", lw=1.3, label="Demand")
            ax.set_title(f"Carbon tax {tax} EUR/t", fontsize=10)
            ax.set_xlim(0, len(d) - 1)
            ax.set_xticks(range(0, 169, 48))
            ax.grid(alpha=0.3)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=7, fontsize=10, bbox_to_anchor=(0.5, 0.995))
        for ax in axes[-1]:
            ax.set_xlabel("Hour of week")
        for ax in axes[:, 0]:
            ax.set_ylabel("Power (MW)")
        fig.suptitle(f"Hourly dispatch, {season} week (forecast availability); "
                     f"coal-to-gas break-even = {be:.1f} EUR/t", y=1.03)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(cfg.FIG_DIR / f"dispatch_grid_{season}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 2) generation mix vs tax
    fc = results[results.availability == "forecast"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, (season, d) in zip(axes, fc.groupby("week")):
        d = d.set_index("carbon_tax")
        d[["coal_mwh", "gas_mwh", "solar_mwh", "wind_mwh"]].plot.bar(
            stacked=True, ax=ax, color=[colors[k] for k in ("coal", "gas", "solar", "wind")],
            width=0.85, legend=False)
        ax.set_title(f"{season.capitalize()} week")
        ax.set_xlabel("Carbon tax (EUR/t)")
    axes[0].set_ylabel("Energy over 168 h (MWh)")
    axes[1].legend(["Coal", "Gas", "Solar", "Wind"], ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "generation_mix_vs_tax.png", dpi=150)
    plt.close(fig)

    # 3) emissions, cost and price vs tax
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for (season, src), d in results.groupby(["week", "availability"]):
        ls = "-" if src == "forecast" else ":"
        lab = f"{season}, {src}"
        axes[0].plot(d.carbon_tax, d.emissions_tco2, ls, marker="o", ms=3, label=lab)
        axes[1].plot(d.carbon_tax, d.objective_eur / 1e3, ls, marker="o", ms=3, label=lab)
        axes[2].plot(d.carbon_tax, d.mean_price_eur_mwh, ls, marker="o", ms=3, label=lab)
    for ax, t in zip(axes, ["Emissions (tCO$_2$/week)", "System cost (kEUR/week)",
                            "Mean electricity price (EUR/MWh)"]):
        ax.axvline(be, color="gray", ls="--", lw=1)
        ax.set_xlabel("Carbon tax (EUR/t)")
        ax.set_title(t)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "emissions_cost_vs_tax.png", dpi=150)
    plt.close(fig)

    # 4) example week of forecast vs actual availability
    fcst = pd.read_csv(cfg.FORECAST_CSV, index_col=0, parse_dates=True)
    h = cfg.FORECAST_HORIZON_FOR_DISPATCH
    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex="col")
    for j, (season, start) in enumerate(cfg.WEEKS.items()):
        idx = pd.date_range(start, periods=cfg.DISPATCH_HOURS, freq="h")
        w = fcst.reindex(idx)
        for i, s in enumerate(cfg.TARGETS):
            ax = axes[i, j]
            ax.plot(idx, w[f"{s}_actual_mw"] / 1e3, label="Actual")
            ax.plot(idx, w[f"{s}_fc_h{h}_mw"] / 1e3, "--", label=f"LSTM {h} h-ahead")
            ax.set_ylabel(f"{s.capitalize()} (GW)")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"{season.capitalize()} week")
    axes[0, 0].legend(fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "forecast_vs_actual_weeks.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Analytical coal-to-gas break-even: {breakeven_tax():.1f} EUR/tCO2")
    results, hourly = run_all()
    results.round(3).to_csv(cfg.SCENARIO_CSV, index=False)
    hourly.round(3).to_csv(cfg.DISPATCH_CSV, index=False)
    make_figures(results, hourly)
    sens = run_sensitivity()
    sens.round(3).to_csv(cfg.SENSITIVITY_CSV, index=False)

    fc = results[results.availability == "forecast"]
    for season, d in fc.groupby("week"):
        coal_on = d[d.coal_mwh > 1].carbon_tax.max()
        gas_on = d[d.coal_mwh <= 1].carbon_tax.min()
        print(f"{season:6s}: last tax with coal = {coal_on}, first all-gas tax = {gas_on}")
    print(f"Saved {cfg.SCENARIO_CSV.name}, {cfg.DISPATCH_CSV.name}, "
          f"{cfg.SENSITIVITY_CSV.name} and figures/")


if __name__ == "__main__":
    main()
