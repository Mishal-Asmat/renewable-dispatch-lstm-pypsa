"""
Central configuration for the whole pipeline.

Every tunable number lives here so that experiments are reproducible and the
README can point to a single source of truth.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

GENERATION_CSV = DATA_DIR / "generation_data.csv"
RAW_OPSD_CSV = DATA_DIR / "time_series_60min_singleindex.csv"
# OPSD time-series release used in this project (data end 2020-09-30).
OPSD_URL = ("https://data.open-power-system-data.org/time_series/"
            "2020-10-06/time_series_60min_singleindex.csv")

FORECAST_CSV = OUTPUT_DIR / "forecast.csv"
PREDICTIONS_NPZ = OUTPUT_DIR / "predictions_test.npz"
SKILL_CSV = OUTPUT_DIR / "forecast_skill.csv"
SCENARIO_CSV = OUTPUT_DIR / "scenario_results.csv"
DISPATCH_CSV = OUTPUT_DIR / "dispatch_hourly.csv"
SENSITIVITY_CSV = OUTPUT_DIR / "sensitivity_results.csv"

# ----------------------------------------------------------------------------
# Data cleaning
# ----------------------------------------------------------------------------
# Years of hourly data used for modelling (inclusive). Five years = 43,824 rows
# (about 5 MB), so memory is not a concern as long as training uses mini-batches.
MODEL_YEARS = (2015, 2019)
MAX_YEARS = 6                      # safety guard against loading far more data
MAX_INTERP_GAP_H = 6               # linearly fill gaps up to this length
SOLAR_MIDDAY_UTC_HOURS = (10, 14)  # German solar is never exactly 0 here

# ----------------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------------
TARGETS = ("solar", "wind")
HORIZONS = (1, 3, 6, 24)           # forecast lead times in hours
WINDOW = 48                        # look-back window (hours)

# Chronological split by forecast ISSUE time (inclusive). Targets of a split never
# leak into the next one: samples need issue_time + max(HORIZONS) <= split end.
# The test year is a full calendar year, so every season is evaluated.
SPLITS = {
    "train": ("2015-01-01 00:00", "2017-12-31 23:00"),
    "val":   ("2018-01-01 00:00", "2018-12-31 23:00"),
    "test":  ("2019-01-01 00:00", "2019-12-31 23:00"),
}

LSTM = dict(
    hidden_size=32,
    num_layers=2,
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=256,
    max_epochs=40,
    patience=6,
    seeds=(0, 1, 2),
)

GBM = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, random_state=0)
GBM_LAGS = (0, 1, 2, 3, 6, 12, 24, 47)   # <= WINDOW - 1

BOOTSTRAP = dict(block_hours=24, n_resamples=1000, seed=0)

# ----------------------------------------------------------------------------
# Power-system model
# ----------------------------------------------------------------------------
# Thermal plants are parameterised by FUEL price and efficiency, so that the
# carbon price correctly enters through emissions per MWh of ELECTRICITY:
#   marginal_cost = fuel_price/eff + emission_factor_thermal/eff * carbon_tax
# Fuel prices are chosen so that the zero-tax electric marginal costs equal the
# original v1 values (gas 50, coal 30 EUR/MWh_el).
GAS = dict(p_nom=1000, eff=0.55, fuel_price=27.5, ef_thermal=0.202)   # tCO2/MWh_th
COAL = dict(p_nom=600, eff=0.35, fuel_price=10.5, ef_thermal=0.340)
SOLAR_P_NOM = 800
WIND_P_NOM = 300
BATTERY = dict(p_nom=800, max_hours=12, eff_store=0.9, eff_dispatch=0.9,
               marginal_cost=10.0)

MEAN_DEMAND_MW = 500.0
DISPATCH_HOURS = 168
# Representative weeks inside the TEST year (Mondays, 00:00 UTC), so the
# renewable forecasts used for dispatch are strictly out-of-sample.
WEEKS = {"winter": "2019-01-14", "summer": "2019-07-08"}
CARBON_TAXES = tuple(range(0, 201, 10))            # EUR/tCO2
PLOT_TAXES = (0, 20, 30, 40, 60, 80, 100, 150, 200)
FORECAST_HORIZON_FOR_DISPATCH = 24                 # day-ahead style forecast

# High-renewables sensitivity (multiplies solar and wind capacity).
RE_SCALES = (1, 3, 6)
SENSITIVITY_TAXES = (0, 50, 100)
