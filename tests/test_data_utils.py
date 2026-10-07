import numpy as np
import pandas as pd
import pytest

import config as cfg
import data_utils as du


def synthetic_frame(n_days=900, seed=0):
    idx = pd.date_range("2015-01-01", periods=24 * n_days, freq="h")
    rng = np.random.default_rng(seed)
    h = idx.hour.values
    solar = np.clip(np.sin(np.pi * (h - 6) / 12), 0, None) * 1000 + rng.normal(0, 10, len(idx))
    wind = 500 + 200 * np.sin(np.arange(len(idx)) / 50) + rng.normal(0, 20, len(idx))
    df = pd.DataFrame({"solar_mw": np.clip(solar, 0, None), "wind_mw": wind}, index=idx)
    df = du.add_capacity(df, "2017-12-31 23:00")
    return df.join(du.time_features(df.index))


def test_sample_alignment():
    fr = synthetic_frame()
    s = du.make_samples(fr, "2016-03-01 00:00", "2016-03-10 00:00")
    i = 40
    t = s["issue_time"][i]
    for k, h in enumerate(cfg.HORIZONS):
        assert s["y_mw"][i, 1, k] == pytest.approx(fr.loc[t + pd.Timedelta(hours=h), "wind_mw"])
    # last window row is time t, first row is t-47h
    assert s["X"][i, -1, 1] == pytest.approx(fr.loc[t, "wind_pu"])
    assert s["X"][i, 0, 1] == pytest.approx(fr.loc[t - pd.Timedelta(hours=cfg.WINDOW - 1), "wind_pu"])


def test_capacity_fallback_uses_training_data_only():
    fr = synthetic_frame()
    raw = fr[["solar_mw", "wind_mw"]].copy()
    base = du.add_capacity(raw, "2015-08-31 23:00")["wind_cap"].iloc[0]
    raw.loc["2015-09-15":, :] *= 100.0           # inflate everything after training
    again = du.add_capacity(raw, "2015-08-31 23:00")
    assert again["wind_cap"].iloc[0] == pytest.approx(base)
    assert again.loc[:"2015-08-31 23:00", "wind_pu"].max() == pytest.approx(1.0)


def test_real_installed_capacity_is_used_when_available():
    idx = pd.date_range("2015-01-01", periods=240, freq="h")
    df = pd.DataFrame({"solar_mw": 100.0, "wind_mw": 1000.0,
                       "solar_capacity_mw": np.linspace(500, 600, 240),
                       "wind_capacity_mw": np.linspace(4000, 5000, 240)}, index=idx)
    out = du.add_capacity(df, "2015-01-05")
    np.testing.assert_allclose(out["wind_cap"], df["wind_capacity_mw"])
    np.testing.assert_allclose(out["wind_pu"], 1000.0 / df["wind_capacity_mw"])


def test_gaps_are_excluded_from_samples():
    fr = synthetic_frame()
    gap_start = pd.Timestamp("2016-03-05 00:00")
    fr.loc[gap_start:gap_start + pd.Timedelta(hours=11), "wind_pu"] = np.nan
    s = du.make_samples(fr, "2016-03-01 00:00", "2016-03-10 00:00")
    assert not np.isnan(s["X"]).any() and not np.isnan(s["y_pu"]).any()
    # no sample may use an input hour or a target hour that lies inside the gap
    gap = set(pd.date_range(gap_start, periods=12, freq="h"))
    for t in s["issue_time"]:
        used = set(pd.date_range(t - pd.Timedelta(hours=cfg.WINDOW - 1), t, freq="h"))
        used |= {t + pd.Timedelta(hours=h) for h in cfg.HORIZONS}
        assert not (used & gap)
    # ... and samples that do not touch the gap must still be kept
    assert pd.Timestamp("2016-03-03 00:00") in set(s["issue_time"])


def test_splits_are_ordered_disjoint_and_purged():
    prev_end = None
    for name in ("train", "val", "test"):
        start, end = (pd.Timestamp(x) for x in cfg.SPLITS[name])
        first, last = du.split_issue_range(name)
        assert last + pd.Timedelta(hours=max(cfg.HORIZONS)) <= end   # purged
        assert first == start
        if prev_end is not None:
            assert start > prev_end
        prev_end = end


def test_load_generation_flags_legacy_zeros(tmp_path):
    idx = pd.date_range("2015-06-01", periods=96, freq="h", tz="UTC")
    solar = np.where((idx.hour >= 6) & (idx.hour <= 18), 800.0, 0.0)
    wind = np.full(96, 5000.0)
    wind[10:13] = 0.0              # 3 h gap  -> interpolated
    wind[40:60] = 0.0              # 20 h gap -> stays missing
    solar[34] = 0.0                # midday zero (hour 10) -> missing then interpolated
    df = pd.DataFrame({"solar_mw": solar, "wind_mw": wind}, index=idx)
    df.index.name = "utc_timestamp"
    path = tmp_path / "gen.csv"
    df.to_csv(path)
    rep = {}
    out = du.load_generation(path, rep)
    assert out["wind_mw"].iloc[10:13].notna().all()
    assert (out["wind_mw"].iloc[10:13] == 5000).all()
    assert out["wind_mw"].iloc[40:60].isna().sum() >= 14     # long gap not invented
    assert out["solar_mw"].iloc[34] > 0
    assert rep["rows"] == 96


def test_only_configured_years_are_loaded(tmp_path):
    idx = pd.date_range("2013-12-30", "2021-01-03", freq="h", tz="UTC")
    df = pd.DataFrame({"solar_mw": 100.0, "wind_mw": 5000.0, "load_mw": 50000.0}, index=idx)
    df.index.name = "utc_timestamp"
    path = tmp_path / "multi.csv"
    df.to_csv(path)
    out = du.load_generation(path)
    y0, y1 = cfg.MODEL_YEARS
    assert out.index.min() >= pd.Timestamp(f"{y0}-01-01")
    assert out.index.max() <= pd.Timestamp(f"{y1}-12-31 23:00")
    assert "load_mw" in out.columns


def test_too_many_years_are_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "MODEL_YEARS", (2010, 2020))
    idx = pd.date_range("2015-01-01", periods=48, freq="h", tz="UTC")
    pd.DataFrame({"solar_mw": 1.0, "wind_mw": 1.0}, index=idx).rename_axis("utc_timestamp").to_csv(tmp_path / "x.csv")
    with pytest.raises(ValueError):
        du.load_generation(tmp_path / "x.csv")


def test_to_mw_clips_and_scales():
    base = np.array([[0.5, 0.2]])
    resid = np.array([[[-0.9], [0.1]]])
    cap = np.array([[1000.0, 2000.0]])
    mw = du.to_mw(base, resid, cap)
    assert mw[0, 0, 0] == 0.0                      # clipped at zero
    assert mw[0, 1, 0] == pytest.approx(0.3 * 2000)


def test_persistence_and_seasonal_naive_shapes():
    fr = synthetic_frame()
    s = du.make_samples(fr, "2016-03-01 00:00", "2016-03-05 00:00")
    assert du.persistence_forecast(s).shape == s["y_mw"].shape
    sn = du.seasonal_naive_forecast(fr, s)
    assert sn.shape == s["y_mw"].shape
    # h = 24: same hour yesterday == value at issue time t
    k = cfg.HORIZONS.index(24)
    np.testing.assert_allclose(sn[:, :, k], s["base_pu"] * s["cap"], rtol=1e-9)
