# **Data**

## Source
Hourly German data from the **Open Power System Data (OPSD)** 
*Time series* package, file `time_series_60min_singleindex.csv` (release 2020-10-06).

- Portal: https://data.open-power-system-data.org/time_series/
- License: CC BY 4.0.
- Resolution: 60 min, UTC timestamps

| Raw column | Name in this project | Used for |
|---|---|---|
| `DE_solar_generation_actual` | `solar_mw` | forecasting target |
| `DE_wind_generation_actual` | `wind_mw` | forecasting target |
| `DE_solar_capacity` | `solar_capacity_mw` | converting generation to capacity factors |
| `DE_wind_capacity` | `wind_capacity_mw` | converting generation to capacity factors |
| `DE_load_actual_entsoe_transparency` | `load_mw` | demand in the dispatch model |

## Years used
**2015 to 2019** (43,824 hourly rows). The raw file also covers the last hour of 2014.
`data_utils.load_generation()` drops every row outside `config.MODEL_YEARS` and refuses ranges longer than `config.MAX_YEARS`.

## Files
| File | Description |
|---|---|
| `time_series_60min_singleindex.csv` | Raw OPSD downloaded, to build the data file. |
| `generation_data.csv` | The five columns above, 2015-01-01 to 2019-12-31 UTC. |

## Building the data file
```bash
python src/prepare_data.py                # filters the years when raw file is placed in data/
```
The raw file is read in chunks of 20,000 rows with only the needed columns parsed.


## Data quality (2015 to 2019)
| Item | Count | Handling |
|---|---|---|
| Solar generation missing | 103 hours (31 in 2015, 72 in 2016) | gaps up to 6 h interpolated; 79 hours stay missing and are excluded |
| Wind generation missing | 74 hours (all in 2016) | gaps up to 6 h interpolated; 54 hours stay missing and are excluded |
| Capacity missing | 25 hours at the end of 2019 | forward-filled (capacity changes slowly) |
| Load missing | 0 | n/a |

2017, 2018 and 2019 are complete. Missing values are kept as NaN in the file. Legacy zero-filled files are handled: wind at exactly 0 MW and solar at exactly 0 MW between 
10:00 and 14:00 UTC are treated as missing.

## Capacity growth
Installed capacity at year end (MW, OPSD):

| Year | Solar | Wind |
|---|---|---|
| 2015 | 38,631 | 32,812 |
| 2016 | 40,089 | 37,149 |
| 2017 | 41,731 | 42,863 |
| 2018 | 47,480 | 48,974 |
| 2019 | 50,508 | 50,452 |

Solar grew 31% and wind 54% over the period, which is why all modelling uses capacity factors.
The OPSD capacity series is stepwise, so generation can occasionally exceed it by a few percent.