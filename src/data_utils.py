"""
Data loading, cleaning and supervised-sample construction.

Design choices:

* Only the configured years (cfg.MODEL_YEARS, default 2015-2019) are ever loaded.
* Series are converted to capacity factors (generation / installed capacity).
  German wind and solar capacity grew 31% (solar) and 54% (wind) over 2015-2019, so raw MW levels
  are non-stationary. The real installed capacity from OPSD is used when available; otherwise 
  the training-period maximum is the fallback.
* Targets are RESIDUALS relative to the last observed value (persistence), so a
  model starts from the persistence benchmark and only has to learn corrections.
* Splits are chronological and purged, and nothing is fitted on test data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import config as cfg

SERIES = cfg.TARGETS  # ("solar", "wind")
FEATURE_COLS = ["solar_pu", "wind_pu", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


# -- Loading and cleaning --

OPTIONAL_COLS = ["solar_capacity_mw", "wind_capacity_mw", "load_mw"]

def load_generation(path=cfg.GENERATION_CSV, report: dict | None = None) -> pd.DataFrame:
    """Load hourly solar/wind generation (MW), optional capacity and demand.

    Invalid values are marked NaN. Handles a raw-derived file (true NaN) and the
    legacy zero-filled file:
      * wind == 0 MW is physically implausible for the German fleet -> missing
      * solar == 0 MW at midday (10-14 UTC) is implausible -> missing
    Short gaps (<= MAX_INTERP_GAP_H) are linearly interpolated; longer gaps stay
    NaN and are excluded from training/evaluation windows. Capacity columns are
    forward/back-filled (installed capacity changes slowly).

    Only rows of cfg.MODEL_YEARS are kept, even if the file contains more.
    """
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df.index.name = "utc_timestamp"
    keep = ["solar_mw", "wind_mw"] + [c for c in OPTIONAL_COLS if c in df.columns]
    df = df[keep].astype(float).sort_index()
    y0, y1 = cfg.MODEL_YEARS
    if y1 - y0 + 1 > cfg.MAX_YEARS:
        raise ValueError(f"MODEL_YEARS spans more than MAX_YEARS={cfg.MAX_YEARS}")
    df = df.loc[f"{y0}-01-01":f"{y1}-12-31 23:00"]

    n_in_file = int(df[["solar_mw", "wind_mw"]].isna().sum().sum())
    df.loc[df["wind_mw"] <= 0, "wind_mw"] = np.nan
    lo, hi = cfg.SOLAR_MIDDAY_UTC_HOURS
    midday = (df.index.hour >= lo) & (df.index.hour <= hi)
    df.loc[midday & (df["solar_mw"] <= 0), "solar_mw"] = np.nan
    flagged = df[["solar_mw", "wind_mw"]].isna().sum().to_dict()

    ts_cols = [c for c in ("solar_mw", "wind_mw", "load_mw") if c in df.columns]
    df[ts_cols] = df[ts_cols].interpolate(method="time", limit=cfg.MAX_INTERP_GAP_H,
                                          limit_area="inside")
    cap_cols = [c for c in ("solar_capacity_mw", "wind_capacity_mw") if c in df.columns]
    if cap_cols:
        df[cap_cols] = df[cap_cols].ffill().bfill()
    after = df[["solar_mw", "wind_mw"]].isna().sum().to_dict()
    if report is not None:
        report.update(dict(rows=len(df), years=f"{y0}-{y1}", nan_in_file=n_in_file,
                           flagged_missing=flagged, missing_after_interpolation=after,
                           capacity_source="OPSD installed capacity" if cap_cols
                           else "training-period maximum (fallback)"))
    return df


def add_capacity(df: pd.DataFrame, train_end) -> pd.DataFrame:
    """Add `<series>_cap` (MW) and `<series>_pu` (capacity factor) columns.

    Uses the real installed capacity when the file provides it (exogenous,
    slowly varying, so no leakage); otherwise falls back to the maximum observed
    in the TRAINING period.
    """
    out = df.copy()
    for s in SERIES:
        col = f"{s}_capacity_mw"
        if col in out.columns:
            out[f"{s}_cap"] = out[col]
        else:
            out[f"{s}_cap"] = out.loc[:pd.Timestamp(train_end), f"{s}_mw"].max()
        out[f"{s}_pu"] = out[f"{s}_mw"] / out[f"{s}_cap"]
    return out


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical calendar features (23:00 and 00:00 are neighbours)."""
    hour = np.asarray(index.hour)
    doy = np.asarray(index.dayofyear)
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=index)


def build_frame(path=cfg.GENERATION_CSV, report: dict | None = None) -> pd.DataFrame:
    """Cleaned, capacity-factor frame with calendar features."""
    df = add_capacity(load_generation(path, report), cfg.SPLITS["train"][1])
    return df.join(time_features(df.index))


# -- Supervised samples --

def split_issue_range(name: str, horizons=cfg.HORIZONS):
    """Inclusive (first, last) forecast-issue timestamps of a split.

    The last issue time is pulled back by max(horizons) so that no target
    falls outside the split (purging).
    """
    start, end = (pd.Timestamp(x) for x in cfg.SPLITS[name])
    return start, end - pd.Timedelta(hours=max(horizons))


def make_samples(frame: pd.DataFrame, issue_start, issue_end,
                 window: int = cfg.WINDOW, horizons=cfg.HORIZONS) -> dict:
    """Build supervised samples for forecasts issued in [issue_start, issue_end].

    A sample issued at time t uses inputs from [t-window+1, t] (inclusive) and
    predicts the series at t+h for each horizon h.

    Returns dict with
      X          (n, window, n_features)   model inputs (per-unit + calendar)
      base_pu    (n, n_series)             last observed per-unit value (persistence)
      cap        (n, n_series)             capacity proxy at t [MW]
      y_pu       (n, n_series, n_horizons) true per-unit targets
      y_mw       (n, n_series, n_horizons) true targets in MW
      issue_time (n,)                      DatetimeIndex of t
    Samples with any NaN in inputs or targets are dropped.
    """
    horizons = tuple(horizons)
    idx = frame.index
    feats = frame[FEATURE_COLS].to_numpy(dtype=float)
    pu = frame[[f"{s}_pu" for s in SERIES]].to_numpy(dtype=float)
    mw = frame[[f"{s}_mw" for s in SERIES]].to_numpy(dtype=float)
    cap = frame[[f"{s}_cap" for s in SERIES]].to_numpy(dtype=float)

    i0 = idx.get_indexer([pd.Timestamp(issue_start)])[0]
    i1 = idx.get_indexer([pd.Timestamp(issue_end)])[0]
    if i0 < 0 or i1 < 0:
        raise ValueError("Issue range outside available data")
    i0 = max(i0, window - 1)
    pos = np.arange(i0, i1 + 1)

    wins = sliding_window_view(feats, window, axis=0)       # (T-w+1, f, w)
    X = np.transpose(wins[pos - (window - 1)], (0, 2, 1))   # (n, w, f)
    base_pu = pu[pos]
    cap_t = cap[pos]
    y_pu = np.stack([pu[pos + h] for h in horizons], axis=-1)
    y_mw = np.stack([mw[pos + h] for h in horizons], axis=-1)

    ok = (~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(base_pu).any(axis=1)
          & ~np.isnan(cap_t).any(axis=1) & ~np.isnan(y_pu).any(axis=(1, 2)))
    return dict(X=X[ok].astype(np.float32), base_pu=base_pu[ok], cap=cap_t[ok],
                y_pu=y_pu[ok], y_mw=y_mw[ok], issue_time=idx[pos[ok]])


def get_split(frame: pd.DataFrame, name: str, **kw) -> dict:
    s, e = split_issue_range(name, kw.get("horizons", cfg.HORIZONS))
    return make_samples(frame, s, e, **kw)


def residual_targets(samples: dict) -> np.ndarray:
    """Per-unit change relative to persistence: y(t+h) - y(t). Shape (n, s, h)."""
    return samples["y_pu"] - samples["base_pu"][:, :, None]


def to_mw(base_pu, resid_pu, cap, clip=True):
    """Convert (persistence + predicted residual) in per-unit back to MW."""
    pu = base_pu[:, :, None] + resid_pu
    if clip:
        pu = np.clip(pu, 0.0, None)
    return pu * cap[:, :, None]


def persistence_forecast(samples: dict) -> np.ndarray:
    """y_hat(t+h) = y(t). Shape (n, s, h) in MW."""
    n, _, n_h = samples["y_pu"].shape
    return np.repeat((samples["base_pu"] * samples["cap"])[:, :, None], n_h, axis=2)


def seasonal_naive_forecast(frame: pd.DataFrame, samples: dict,
                            horizons=cfg.HORIZONS) -> np.ndarray:
    """y_hat(t+h) = y(t+h-24): the same hour yesterday (valid for h <= 24)."""
    mw = frame[[f"{s}_mw" for s in SERIES]]
    out = []
    for h in horizons:
        if h > 24:
            raise ValueError("seasonal naive (24 h) requires horizon <= 24")
        ts = samples["issue_time"] + pd.Timedelta(hours=h - 24)
        out.append(mw.reindex(ts).to_numpy())
    return np.stack(out, axis=-1)
