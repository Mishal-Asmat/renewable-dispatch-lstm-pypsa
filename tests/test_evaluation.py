import numpy as np
import pandas as pd

import config as cfg
import data_utils as du
import evaluate_baselines as ev
from test_data_utils import synthetic_frame


def test_lag_features_shape():
    fr = synthetic_frame()
    s = du.make_samples(fr, "2016-03-01 00:00", "2016-03-05 00:00")
    F = ev.lag_features(s)
    assert F.shape == (len(s["X"]), 2 * len(cfg.GBM_LAGS) + 4)


def test_skill_of_persistence_is_zero_and_perfect_is_one():
    fr = synthetic_frame()
    s = du.make_samples(fr, "2016-03-01 00:00", "2016-03-20 00:00")
    pers = du.persistence_forecast(s)
    preds = {"Persistence": pers, "Perfect": s["y_mw"].copy()}
    table = ev.evaluate(preds, s["y_mw"], s["cap"])
    p = table[table.model == "Persistence"]
    q = table[table.model == "Perfect"]
    assert np.allclose(p.skill_vs_persistence, 0.0)
    assert np.allclose(q.skill_vs_persistence, 1.0)


def test_block_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 10, 24 * 200)
    mod = ref * 0.7
    lo, hi = ev.block_bootstrap_skill(mod, ref, 24, 500, 0)
    point = 1 - np.abs(mod).mean() / np.abs(ref).mean()
    assert lo <= point <= hi
