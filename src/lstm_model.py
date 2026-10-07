"""
Stage 1 - multi-horizon LSTM forecasting of German solar and wind generation.

Improvements over v1 (see CHANGELOG.md):
  * Five years of data (2015-2019) with a chronological split: train 2015-17,
    validation 2018, test 2019 (a full year, so every season is evaluated).
    Splits are purged so targets never cross split boundaries.
  * Capacity-factor inputs (generation / real installed capacity) -> handles the
    31% (solar) and 54% (wind) growth of German capacity over the period.
  * RESIDUAL targets y(t+h) - y(t): the network starts at persistence and only
    learns corrections.
  * Separate model per target (solar / wind), direct multi-horizon output
    (1, 3, 6, 24 h ahead).
  * MINI-BATCHES everywhere, including prediction and validation, so memory stays
    small however many years are used.
  * Early stopping on the validation set, L1 loss (matches the MAE metric),
    AdamW + gradient clipping, fixed seeds, 3-seed ensemble.
  * Nothing is fitted on test data.

Usage (from the repository root):
    python src/lstm_model.py                # full run
    python src/lstm_model.py --quick        # 1 seed, 2 epochs (smoke test)
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
import data_utils as du


class ResidualLSTM(nn.Module):
    """LSTM encoder -> linear head producing one residual per forecast horizon."""

    def __init__(self, n_features: int, n_horizons: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, n_horizons)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            outs.append(model(torch.from_numpy(X[i:i + batch_size])).numpy())
    return np.concatenate(outs, axis=0)


def train_one(X_tr, y_tr, X_va, y_va, seed: int, max_epochs: int, log: list,
              tag: str) -> nn.Module:
    """Train one LSTM with early stopping on validation L1 loss."""
    p = cfg.LSTM
    set_seed(seed)
    model = ResidualLSTM(X_tr.shape[-1], y_tr.shape[-1], p["hidden_size"],
                         p["num_layers"], p["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"],
                            weight_decay=p["weight_decay"])
    loss_fn = nn.L1Loss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=p["batch_size"], shuffle=True)
    X_va_t, y_va_t = torch.from_numpy(X_va), torch.from_numpy(y_va)
    va_batch = 1024

    best, best_state, bad = np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():       # validation in mini-batches too (memory-safe)
            val = sum(loss_fn(model(X_va_t[i:i + va_batch]), y_va_t[i:i + va_batch]).item()
                      * len(X_va_t[i:i + va_batch]) for i in range(0, len(X_va_t), va_batch)
                      ) / len(X_va_t)
        log.append(dict(model=tag, seed=seed, epoch=epoch,
                        train_loss=tot / nb, val_loss=val))
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= p["patience"]:
                break
    model.load_state_dict(best_state)
    print(f"    {tag} seed {seed}: best val L1 {best:.4f} after {epoch} epochs")
    return model


def main(quick: bool = False, seeds=None, epochs=None) -> None:
    t0 = time.time()
    seeds = tuple(seeds) if seeds else cfg.LSTM["seeds"]
    max_epochs = epochs or cfg.LSTM["max_epochs"]
    if quick:
        seeds, max_epochs = seeds[:1], 2
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    frame = du.build_frame()
    tr, va, te = (du.get_split(frame, s) for s in ("train", "val", "test"))
    print(f"Samples  train {len(tr['X']):,}  val {len(va['X']):,}  test {len(te['X']):,}"
          f"  | window {cfg.WINDOW} h, horizons {cfg.HORIZONS}")

    r_tr, r_va = du.residual_targets(tr), du.residual_targets(va)
    n_s, n_h = len(cfg.TARGETS), len(cfg.HORIZONS)
    pred_resid = np.zeros((len(seeds), len(te["X"]), n_s, n_h))
    log: list = []

    for s_idx, target in enumerate(cfg.TARGETS):
        # Standardise residuals per horizon using TRAIN statistics only.
        scale = r_tr[:, s_idx, :].std(axis=0)
        y_tr = (r_tr[:, s_idx, :] / scale).astype(np.float32)
        y_va = (r_va[:, s_idx, :] / scale).astype(np.float32)
        print(f"Training {target} models ...")
        for k, seed in enumerate(seeds):
            model = train_one(tr["X"], y_tr, va["X"], y_va, seed, max_epochs,
                              log, tag=target)
            pred_resid[k, :, s_idx, :] = predict(model, te["X"]) * scale
            torch.save(dict(state_dict=model.state_dict(), scale=scale,
                            horizons=cfg.HORIZONS, window=cfg.WINDOW,
                            lstm=cfg.LSTM),
                       cfg.MODEL_DIR / f"lstm_{target}_seed{seed}.pt")

    # Back to MW: persistence + predicted residual, clipped at zero.
    lstm_seeds = np.stack([du.to_mw(te["base_pu"], pred_resid[k], te["cap"])
                           for k in range(len(seeds))])      # (K, n, s, h)
    lstm_ens = lstm_seeds.mean(axis=0)

    np.savez_compressed(
        cfg.PREDICTIONS_NPZ,
        issue_time=te["issue_time"].values.astype("datetime64[ns]").astype("int64"),
        lstm=lstm_ens, lstm_seeds=lstm_seeds.astype(np.float32),
        y_mw=te["y_mw"], cap=te["cap"], base_pu=te["base_pu"],
        horizons=np.array(cfg.HORIZONS), seeds=np.array(seeds))
    pd.DataFrame(log).to_csv(cfg.OUTPUT_DIR / "training_log.csv", index=False)

    write_forecast_csv(frame, te["issue_time"], lstm_ens)
    mae = np.abs(lstm_ens - te["y_mw"]).mean(axis=0)          # (s, h)
    print("\nTest MAE of the LSTM ensemble (MW):")
    print(pd.DataFrame(mae, index=cfg.TARGETS,
                       columns=[f"h={h}" for h in cfg.HORIZONS]).round(0))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


def write_forecast_csv(frame: pd.DataFrame, issue_time: pd.DatetimeIndex,
                       lstm_ens: np.ndarray) -> None:
    """Wide forecast table indexed by the VALIDITY time of each forecast.

    Column  `<series>_fc_h<h>_mw` holds the forecast for that hour issued h hours
    earlier. This is what the dispatch model consumes (h = 24: day-ahead style).
    """
    cols = {}
    for s_idx, s in enumerate(cfg.TARGETS):
        for h_idx, h in enumerate(cfg.HORIZONS):
            idx = issue_time + pd.Timedelta(hours=h)
            cols[f"{s}_fc_h{h}_mw"] = pd.Series(lstm_ens[:, s_idx, h_idx], index=idx)
    out = pd.DataFrame(cols).sort_index()
    for s in cfg.TARGETS:
        out[f"{s}_actual_mw"] = frame[f"{s}_mw"].reindex(out.index)
        out[f"{s}_cap_mw"] = frame[f"{s}_cap"].reindex(out.index)
    if "load_mw" in frame.columns:      # real demand, used by the dispatch stage
        out["load_mw"] = frame["load_mw"].reindex(out.index)
    out.index.name = "utc_timestamp"
    out.round(2).to_csv(cfg.FORECAST_CSV)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 seed, 2 epochs")
    ap.add_argument("--seeds", type=int, nargs="+", help="override seeds")
    ap.add_argument("--epochs", type=int, help="override max epochs")
    a = ap.parse_args()
    main(quick=a.quick, seeds=a.seeds, epochs=a.epochs)
