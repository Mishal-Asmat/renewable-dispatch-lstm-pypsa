"""
Benchmark the LSTM against simple and strong baselines on the held-out test period.

Models compared (identical test samples, horizons 1/3/6/24 h):
  * Persistence      y_hat(t+h) = y(t)
  * Seasonal naive   y_hat(t+h) = y(t+h-24)  (same hour yesterday)
  * Gradient boosting (scikit-learn HistGradientBoosting, residual target, lag features)
  * LSTM ensemble    (output/predictions_test.npz written by lstm_model.py)

Metrics: MAE, RMSE (MW) and skill score  1 - MAE_model / MAE_persistence
(>0 means better than persistence). A moving-block bootstrap (24 h blocks)
gives a 95% confidence interval for the skill score, which accounts for the
strong autocorrelation of hourly errors.

Usage (repository root):  python src/evaluate_baselines.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config as cfg
import data_utils as du

SERIES, HORIZONS = cfg.TARGETS, cfg.HORIZONS


def lag_features(samples: dict) -> np.ndarray:
    """Flat feature matrix for gradient boosting, built from the same window."""
    X = samples["X"]                                  # (n, w, f)
    cols = [X[:, -1 - lag, :2] for lag in cfg.GBM_LAGS]   # solar/wind pu at lags
    cols.append(X[:, -1, 2:])                         # calendar features at t
    return np.concatenate(cols, axis=1)


def fit_predict_gbm(tr: dict, te: dict) -> np.ndarray:
    """One model per (series, horizon); predicts the per-unit residual."""
    Ftr, Fte = lag_features(tr), lag_features(te)
    r_tr = du.residual_targets(tr)
    pred = np.zeros((len(Fte), len(SERIES), len(HORIZONS)))
    for s in range(len(SERIES)):
        for h in range(len(HORIZONS)):
            m = HistGradientBoostingRegressor(loss="absolute_error", **cfg.GBM)
            m.fit(Ftr, r_tr[:, s, h])
            pred[:, s, h] = m.predict(Fte)
    return du.to_mw(te["base_pu"], pred, te["cap"])


def block_bootstrap_skill(err_model, err_ref, block, n_resamples, seed):
    """95% CI of skill = 1 - mean|e_model| / mean|e_ref| using block resampling."""
    n = (len(err_model) // block) * block
    a = np.abs(err_model[:n]).reshape(-1, block).sum(axis=1)
    b = np.abs(err_ref[:n]).reshape(-1, block).sum(axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_resamples, len(a)))
    skill = 1.0 - a[idx].sum(axis=1) / b[idx].sum(axis=1)
    return np.percentile(skill, [2.5, 97.5])


def evaluate(preds: dict, y: np.ndarray, cap: np.ndarray) -> pd.DataFrame:
    rows = []
    bs = cfg.BOOTSTRAP
    for s_idx, s in enumerate(SERIES):
        for h_idx, h in enumerate(HORIZONS):
            truth = y[:, s_idx, h_idx]
            ref_err = truth - preds["Persistence"][:, s_idx, h_idx]
            mae_ref = np.abs(ref_err).mean()
            for name, p in preds.items():
                err = truth - p[:, s_idx, h_idx]
                mae = np.abs(err).mean()
                row = dict(series=s, horizon_h=h, model=name, MAE_MW=mae,
                           RMSE_MW=np.sqrt((err ** 2).mean()),
                           nMAE_pct_cap=100 * np.mean(np.abs(err) / cap[:, s_idx]),
                           skill_vs_persistence=1 - mae / mae_ref,
                           skill_ci_low=np.nan, skill_ci_high=np.nan)
                if name != "Persistence":
                    lo, hi = block_bootstrap_skill(err, ref_err, bs["block_hours"],
                                                   bs["n_resamples"], bs["seed"])
                    row["skill_ci_low"], row["skill_ci_high"] = lo, hi
                rows.append(row)
    return pd.DataFrame(rows)


def plot_skill(table: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    styles = {"LSTM": ("tab:blue", "o-"), "Gradient boosting": ("tab:orange", "s--"),
              "Seasonal naive": ("tab:gray", "^:")}
    for ax, s in zip(axes, SERIES):
        for name, (c, st) in styles.items():
            d = table[(table.series == s) & (table.model == name)].sort_values("horizon_h")
            ax.plot(d.horizon_h, d.skill_vs_persistence, st, color=c, label=name)
            if name == "LSTM":
                ax.fill_between(d.horizon_h, d.skill_ci_low, d.skill_ci_high,
                                color=c, alpha=0.15)
        ax.axhline(0, color="k", lw=1)
        ax.set_ylim(-0.35, 0.95)   # seasonal naive is far below 0 for wind (off-scale)
        ax.set_xscale("log")
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels(HORIZONS)
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_title(f"{s.capitalize()}: skill vs persistence")
        ax.grid(alpha=0.3)
    axes[1].text(0.97, 0.04, "seasonal naive off-scale (skill < -0.35)\nat 1 h, 3 h and 6 h",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8, color="gray")
    axes[0].set_ylabel("Skill = 1 - MAE / MAE$_{persistence}$")
    axes[0].legend()
    fig.suptitle("Test period 20 Oct - 31 Dec 2015 (shaded: 95% block-bootstrap CI for the LSTM)", y=1.02)
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "skill_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame = du.build_frame()
    tr = du.get_split(frame, "train")
    te = du.get_split(frame, "test")

    z = np.load(cfg.PREDICTIONS_NPZ)
    issue = pd.to_datetime(z["issue_time"])
    assert len(issue) == len(te["issue_time"]) and (issue == te["issue_time"]).all(), \
        "predictions_test.npz does not match the current test split; rerun lstm_model.py"

    print("Fitting gradient-boosting baseline ...")
    preds = {
        "Persistence": du.persistence_forecast(te),
        "Seasonal naive": du.seasonal_naive_forecast(frame, te),
        "Gradient boosting": fit_predict_gbm(tr, te),
        "LSTM": z["lstm"],
    }
    table = evaluate(preds, te["y_mw"], te["cap"])
    table.round(4).to_csv(cfg.SKILL_CSV, index=False)
    plot_skill(table)

    show = table.copy()
    show["skill_95CI"] = show.apply(
        lambda r: "" if np.isnan(r.skill_ci_low)
        else f"[{r.skill_ci_low:+.2f}, {r.skill_ci_high:+.2f}]", axis=1)
    pd.set_option("display.width", 160)
    for s in SERIES:
        print(f"\n=== {s.upper()} ({len(te['y_mw']):,} test samples) ===")
        print(show[show.series == s].pivot_table(
            index="model", columns="horizon_h", values="MAE_MW").round(0).loc[
            ["Persistence", "Seasonal naive", "Gradient boosting", "LSTM"]])
        print(show[(show.series == s) & (show.model == "LSTM")][
            ["horizon_h", "skill_vs_persistence", "skill_95CI"]].round(3).to_string(index=False))
    print(f"\nSaved {cfg.SKILL_CSV.name} and figures/skill_by_horizon.png")


if __name__ == "__main__":
    main()
