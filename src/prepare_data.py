"""
Prepare data/generation_data.csv (the modelling years) from the Open Power System
Data (OPSD) 60-minute time series.

"""
from __future__ import annotations

import argparse
import urllib.request

import pandas as pd

import config as cfg

COLUMNS = {"DE_solar_generation_actual": "solar_mw",
           "DE_wind_generation_actual": "wind_mw",
           "DE_solar_capacity": "solar_capacity_mw",
           "DE_wind_capacity": "wind_capacity_mw",
           "DE_load_actual_entsoe_transparency": "load_mw"}


def prepare(years=cfg.MODEL_YEARS, raw=cfg.RAW_OPSD_CSV,
            out=cfg.GENERATION_CSV) -> pd.DataFrame:
    if not raw.exists():
        raise FileNotFoundError(
            f"{raw} not found. Please place the OPSD file at:\n  {raw}\n"
            "The file should be named 'time_series_60min_singleindex.csv'.")
    wanted = tuple(str(y) for y in range(years[0], years[1] + 1))
    keep = []
    reader = pd.read_csv(raw, usecols=["utc_timestamp", *COLUMNS], chunksize=20_000)
    for chunk in reader:
        chunk = chunk[chunk["utc_timestamp"].str.startswith(wanted)]
        if len(chunk):
            keep.append(chunk)
    df = pd.concat(keep).rename(columns=COLUMNS).set_index("utc_timestamp")
    df.to_csv(out)
    print(f"Wrote {out} ({len(df):,} rows, {df.index.min()} to {df.index.max()})")
    print("Missing values per column (kept as NaN):")
    print(df.isna().sum().to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs=2, default=list(cfg.MODEL_YEARS),
                    metavar=("FIRST", "LAST"))
    args = ap.parse_args()

    prepare(tuple(args.years))
