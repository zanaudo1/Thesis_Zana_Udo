"""
Step 06: Revised RNN and LSTM next-day trajectory models.

The recurrent models use true ordered occasion-level sequences and directly
forecast all six AffWB positions of the next planned day. Sequence length,
predictor set, target representation, hidden size, and dropout are selected by
rolling-origin validation. Multiple random seeds quantify neural-model
instability. Day 7 is used only after all tuning decisions are fixed.

Run from the project root:
    python 06_train_recurrent_models.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

STEP = "06"
VALIDATION_DAYS = [4, 5, 6]
TEST_DAY = 7
SEQUENCE_SPECS = ["L03", "L06", "L12", "L18", "all_history"]
PREDICTOR_SETS = ["mood_only", "mood_social"]
TARGET_MODES = ["absolute", "residual_persistence", "residual_seasonal"]
MODEL_TYPES = ["RNN", "LSTM"]
STAGE1_SEEDS = [42]
STAGE2_SEEDS = [42, 43, 44]
FINAL_SEEDS = [42, 43, 44, 45, 46]
MAX_EPOCHS_STAGE1 = 35
MAX_EPOCHS_STAGE2 = 50
PATIENCE = 6
BATCH_SIZE = 64
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0

HYPER_GRID = [
    {"hidden_size": 8, "dropout": 0.0},
    {"hidden_size": 16, "dropout": 0.0},
    {"hidden_size": 16, "dropout": 0.2},
    {"hidden_size": 32, "dropout": 0.2},
]

EVIDENCE_GUIDED_SEQUENCE_CONFIGS = [
    ("L03", "mood_only", "residual_persistence"),
    ("L06", "mood_only", "absolute"),
    ("L12", "mood_only", "residual_seasonal"),
    ("L18", "mood_only", "residual_seasonal"),
    ("all_history", "mood_only", "absolute"),
    ("L12", "mood_social", "residual_persistence"),
    ("all_history", "mood_social", "residual_persistence"),
]

ALL_CHANNELS = [
    "AffWB", "AffWBMask", "ExpandingMean", "ExpandingMeanMask", "AffWBDeviation",
    "PositionSin", "PositionCos", "DayNorm",
    "PassiveUse", "PassiveMask", "ActiveUse", "ActiveMask",
]
MOOD_CHANNELS = ALL_CHANNELS[:8]
MOOD_SOCIAL_CHANNELS = ALL_CHANNELS
VALUE_TO_MASK = {
    "AffWB": "AffWBMask",
    "ExpandingMean": "ExpandingMeanMask",
    "AffWBDeviation": "AffWBMask",
    "PassiveUse": "PassiveMask",
    "ActiveUse": "ActiveMask",
}


@dataclass
class Record:
    origin_id: str
    participant_id: int
    target_day: int
    X: np.ndarray
    y: np.ndarray
    y_mask: np.ndarray
    persistence_raw: float
    seasonal_raw: np.ndarray
    history_mean_raw: float


@dataclass
class ScaleInfo:
    channel_means: dict[str, float]
    channel_stds: dict[str, float]
    target_mean: float
    target_std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_means": self.channel_means,
            "channel_stds": self.channel_stds,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }


class RecurrentForecaster(nn.Module):
    def __init__(self, model_type: str, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        if model_type == "RNN":
            self.recurrent = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh")
        elif model_type == "LSTM":
            self.recurrent = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(model_type)
        self.model_type = model_type
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(6)])

    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        if self.model_type == "LSTM":
            _, (hidden, _) = self.recurrent(packed)
        else:
            _, hidden = self.recurrent(packed)
        encoded = self.dropout(hidden[-1])
        return torch.cat([head(encoded) for head in self.heads], dim=1)


def make_dirs(root: Path) -> dict[str, Path]:
    dirs = {name: root / "outputs" / name for name in ["tables", "images", "reports", "models", "data"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def load_full_grid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    pieces = []
    for pid, g in df.groupby("ParticipantID", sort=True):
        g = g.sort_values("Occasion").drop_duplicates("Occasion", keep="last").set_index("Occasion")
        g = g.reindex(range(1, 43))
        g["ParticipantID"] = int(pid)
        g["Occasion"] = g.index.astype(int)
        for c in ["AffWB__observed", "OverallPassiveUse__observed", "OverallActiveUse__observed"]:
            g[c] = g[c].fillna(0).astype(int)
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def expanding_mean(values: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(len(values), np.nan, dtype=float)
    mask = np.zeros(len(values), dtype=int)
    total = 0.0
    count = 0
    for i in range(len(values)):
        if observed[i] == 1 and np.isfinite(values[i]):
            total += float(values[i])
            count += 1
        if count:
            result[i] = total / count
            mask[i] = 1
    return result, mask


def build_records(grid: pd.DataFrame) -> list[Record]:
    records: list[Record] = []
    for pid, g in grid.groupby("ParticipantID", sort=True):
        g = g.sort_values("Occasion")
        aff = g["AffWB"].to_numpy(float)
        aff_mask = g["AffWB__observed"].to_numpy(int)
        pas = g["OverallPassiveUse"].to_numpy(float)
        pas_mask = g["OverallPassiveUse__observed"].to_numpy(int)
        act = g["OverallActiveUse"].to_numpy(float)
        act_mask = g["OverallActiveUse__observed"].to_numpy(int)
        exp_mean, exp_mask = expanding_mean(aff, aff_mask)
        deviation = np.where((aff_mask == 1) & np.isfinite(aff) & np.isfinite(exp_mean), aff - exp_mean, np.nan)
        positions = np.arange(42) % 6
        pos_sin = np.sin(2 * np.pi * positions / 6.0)
        pos_cos = np.cos(2 * np.pi * positions / 6.0)
        day_norm = (np.arange(42) // 6) / 6.0
        full_X = np.column_stack([
            aff, aff_mask, exp_mean, exp_mask, deviation,
            pos_sin, pos_cos, day_norm,
            pas, pas_mask, act, act_mask,
        ]).astype(float)

        for target_day in range(2, 8):
            origin = (target_day - 1) * 6
            target_slice = slice(origin, origin + 6)
            y = aff[target_slice].astype(float)
            y_mask = ((aff_mask[target_slice] == 1) & np.isfinite(y)).astype(float)
            y = np.where(y_mask == 1, y, np.nan)
            hist_obs = aff[:origin][(aff_mask[:origin] == 1) & np.isfinite(aff[:origin])]
            history_mean = float(np.mean(hist_obs)) if len(hist_obs) else np.nan
            persistence = float(hist_obs[-1]) if len(hist_obs) else np.nan
            seasonal = np.full(6, np.nan, dtype=float)
            for h in range(6):
                idx = origin + h - 6
                if idx >= 0 and aff_mask[idx] == 1 and np.isfinite(aff[idx]):
                    seasonal[h] = aff[idx]
            records.append(Record(
                origin_id=f"P{int(pid):02d}_D{target_day}",
                participant_id=int(pid), target_day=target_day,
                X=full_X[:origin].copy(), y=y, y_mask=y_mask,
                persistence_raw=persistence, seasonal_raw=seasonal,
                history_mean_raw=history_mean,
            ))
    return records


def sequence_length(spec: str, available: int) -> int:
    if spec == "all_history":
        return available
    return min(int(spec[1:]), available)


def select_channels(predictor_set: str) -> list[str]:
    if predictor_set == "mood_only":
        return MOOD_CHANNELS
    if predictor_set == "mood_social":
        return MOOD_SOCIAL_CHANNELS
    raise ValueError(predictor_set)


def baseline_matrix(records: list[Record], target_mode: str, fallback: float) -> np.ndarray:
    result = np.zeros((len(records), 6), dtype=float)
    for i, r in enumerate(records):
        history_mean = r.history_mean_raw if np.isfinite(r.history_mean_raw) else fallback
        persistence = r.persistence_raw if np.isfinite(r.persistence_raw) else history_mean
        if not np.isfinite(persistence):
            persistence = fallback
        seasonal = np.where(np.isfinite(r.seasonal_raw), r.seasonal_raw, persistence)
        seasonal = np.where(np.isfinite(seasonal), seasonal, history_mean)
        seasonal = np.where(np.isfinite(seasonal), seasonal, fallback)
        if target_mode == "absolute":
            result[i] = 0.0
        elif target_mode == "residual_persistence":
            result[i] = persistence
        elif target_mode == "residual_seasonal":
            result[i] = seasonal
        else:
            raise ValueError(target_mode)
    return result


def fit_scaler(records: list[Record], sequence_spec: str, channels: list[str], y_residual: np.ndarray, y_mask: np.ndarray) -> ScaleInfo:
    means, stds = {}, {}
    channel_indices = [ALL_CHANNELS.index(c) for c in channels]
    for channel, source_idx in zip(channels, channel_indices):
        if channel not in VALUE_TO_MASK:
            continue
        mask_name = VALUE_TO_MASK[channel]
        mask_idx = ALL_CHANNELS.index(mask_name)
        values = []
        for r in records:
            length = sequence_length(sequence_spec, len(r.X))
            seq = r.X[-length:]
            obs = (seq[:, mask_idx] == 1) & np.isfinite(seq[:, source_idx])
            if obs.any():
                vals = seq[obs, source_idx].astype(float)
                if channel in {"PassiveUse", "ActiveUse"}:
                    vals = np.log1p(np.maximum(vals, 0.0))
                values.extend(vals.tolist())
        arr = np.asarray(values, float)
        means[channel] = float(np.mean(arr)) if len(arr) else 0.0
        sd = float(np.std(arr, ddof=0)) if len(arr) else 1.0
        stds[channel] = sd if np.isfinite(sd) and sd > 1e-8 else 1.0
    target_values = y_residual[(y_mask == 1) & np.isfinite(y_residual)]
    target_mean = float(np.mean(target_values)) if len(target_values) else 0.0
    target_std = float(np.std(target_values, ddof=0)) if len(target_values) else 1.0
    if not np.isfinite(target_std) or target_std < 1e-8:
        target_std = 1.0
    return ScaleInfo(means, stds, target_mean, target_std)


def transform_records(
    records: list[Record], sequence_spec: str, channels: list[str], scaler: ScaleInfo,
    baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.array([sequence_length(sequence_spec, len(r.X)) for r in records], dtype=np.int64)
    max_len = int(lengths.max())
    X = np.zeros((len(records), max_len, len(channels)), dtype=np.float32)
    channel_indices = [ALL_CHANNELS.index(c) for c in channels]
    for i, r in enumerate(records):
        length = lengths[i]
        seq = r.X[-length:, channel_indices].copy().astype(float)
        for j, channel in enumerate(channels):
            if channel not in VALUE_TO_MASK:
                seq[:, j] = np.nan_to_num(seq[:, j], nan=0.0)
                continue
            source_idx = ALL_CHANNELS.index(channel)
            mask_idx = ALL_CHANNELS.index(VALUE_TO_MASK[channel])
            raw_seq = r.X[-length:]
            obs = (raw_seq[:, mask_idx] == 1) & np.isfinite(raw_seq[:, source_idx])
            vals = raw_seq[:, source_idx].astype(float)
            if channel in {"PassiveUse", "ActiveUse"}:
                vals = np.log1p(np.maximum(vals, 0.0))
            transformed = np.zeros(length, dtype=float)
            transformed[obs] = (vals[obs] - scaler.channel_means[channel]) / scaler.channel_stds[channel]
            seq[:, j] = transformed
        X[i, :length] = seq.astype(np.float32)
    y = np.stack([r.y for r in records]).astype(float)
    y_mask = np.stack([r.y_mask for r in records]).astype(np.float32)
    residual = y - baseline
    y_scaled = np.zeros_like(residual, dtype=np.float32)
    obs = (y_mask == 1) & np.isfinite(residual)
    y_scaled[obs] = ((residual[obs] - scaler.target_mean) / scaler.target_std).astype(np.float32)
    return X, lengths, y_scaled, y_mask, y


def make_loader(X: np.ndarray, lengths: np.ndarray, y: np.ndarray, y_mask: np.ndarray, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(lengths, dtype=torch.int64),
        torch.tensor(y, dtype=torch.float32), torch.tensor(y_mask, dtype=torch.float32),
    )
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=min(BATCH_SIZE, len(ds)), shuffle=shuffle, generator=gen)


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (torch.abs(pred - target) * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def predict_scaled(model: nn.Module, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32), torch.tensor(lengths, dtype=torch.int64)).cpu().numpy()


def mae(actual: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    valid = (mask == 1) & np.isfinite(actual) & np.isfinite(prediction)
    return float(np.mean(np.abs(prediction[valid] - actual[valid]))) if valid.any() else np.nan


def train_with_early_stopping(
    model_type: str, hidden_size: int, dropout: float,
    train_records: list[Record], val_records: list[Record],
    sequence_spec: str, predictor_set: str, target_mode: str,
    seed: int, max_epochs: int,
) -> tuple[nn.Module, ScaleInfo, int, np.ndarray, np.ndarray, np.ndarray]:
    set_seed(seed)
    channels = select_channels(predictor_set)
    fallback = float(np.nanmedian(np.concatenate([r.y[r.y_mask == 1] for r in train_records])))
    b_train = baseline_matrix(train_records, target_mode, fallback)
    b_val = baseline_matrix(val_records, target_mode, fallback)
    y_train_raw = np.stack([r.y for r in train_records])
    y_train_mask = np.stack([r.y_mask for r in train_records])
    scaler = fit_scaler(train_records, sequence_spec, channels, y_train_raw - b_train, y_train_mask)
    Xtr, Ltr, ytr, mtr, _ = transform_records(train_records, sequence_spec, channels, scaler, b_train)
    Xv, Lv, _, mv, yv_actual = transform_records(val_records, sequence_spec, channels, scaler, b_val)
    loader = make_loader(Xtr, Ltr, ytr, mtr, True, seed)
    model = RecurrentForecaster(model_type, len(channels), hidden_size, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_state = copy.deepcopy(model.state_dict())
    best_mae = np.inf
    best_epoch = 1
    no_improvement = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, lb, yb, mb in loader:
            optimizer.zero_grad()
            pred = model(xb, lb)
            loss = masked_mae_loss(pred, yb, mb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
        val_scaled = predict_scaled(model, Xv, Lv)
        val_residual = val_scaled * scaler.target_std + scaler.target_mean
        val_prediction = np.clip(val_residual + b_val, 1.0, 7.0)
        current = mae(yv_actual, val_prediction, mv)
        if current < best_mae - 1e-5:
            best_mae = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= PATIENCE:
                break
    model.load_state_dict(best_state)
    val_scaled = predict_scaled(model, Xv, Lv)
    val_prediction = np.clip(val_scaled * scaler.target_std + scaler.target_mean + b_val, 1.0, 7.0)
    return model, scaler, best_epoch, val_prediction, yv_actual, mv


def train_fixed_epochs(
    model_type: str, hidden_size: int, dropout: float,
    train_records: list[Record], test_records: list[Record],
    sequence_spec: str, predictor_set: str, target_mode: str,
    seed: int, epochs: int,
) -> tuple[nn.Module, ScaleInfo, np.ndarray, np.ndarray, np.ndarray]:
    set_seed(seed)
    channels = select_channels(predictor_set)
    fallback = float(np.nanmedian(np.concatenate([r.y[r.y_mask == 1] for r in train_records])))
    b_train = baseline_matrix(train_records, target_mode, fallback)
    b_test = baseline_matrix(test_records, target_mode, fallback)
    y_train_raw = np.stack([r.y for r in train_records])
    y_train_mask = np.stack([r.y_mask for r in train_records])
    scaler = fit_scaler(train_records, sequence_spec, channels, y_train_raw - b_train, y_train_mask)
    Xtr, Ltr, ytr, mtr, _ = transform_records(train_records, sequence_spec, channels, scaler, b_train)
    Xt, Lt, _, mt, yt_actual = transform_records(test_records, sequence_spec, channels, scaler, b_test)
    loader = make_loader(Xtr, Ltr, ytr, mtr, True, seed)
    model = RecurrentForecaster(model_type, len(channels), hidden_size, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for _ in range(max(1, epochs)):
        model.train()
        for xb, lb, yb, mb in loader:
            optimizer.zero_grad()
            loss = masked_mae_loss(model(xb, lb), yb, mb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
    pred_scaled = predict_scaled(model, Xt, Lt)
    prediction = np.clip(pred_scaled * scaler.target_std + scaler.target_mean + b_test, 1.0, 7.0)
    return model, scaler, prediction, yt_actual, mt


def evaluate_candidate(
    records: list[Record], model_type: str, sequence_spec: str, predictor_set: str,
    target_mode: str, hidden_size: int, dropout: float, seeds: list[int], max_epochs: int,
    return_predictions: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None, list[int]]:
    fold_metrics = []
    prediction_rows = []
    all_epochs = []
    for val_day in VALIDATION_DAYS:
        train_records = [r for r in records if 2 <= r.target_day < val_day]
        val_records = [r for r in records if r.target_day == val_day]
        seed_predictions = []
        for seed in seeds:
            _, _, best_epoch, pred, actual, mask = train_with_early_stopping(
                model_type, hidden_size, dropout, train_records, val_records,
                sequence_spec, predictor_set, target_mode, seed, max_epochs,
            )
            seed_predictions.append(pred)
            all_epochs.append(best_epoch)
        ensemble = np.mean(seed_predictions, axis=0)
        fold_mae = mae(actual, ensemble, mask)
        valid = (mask == 1) & np.isfinite(actual) & np.isfinite(ensemble)
        errors = ensemble[valid] - actual[valid]
        fold_metrics.append({
            "ValidationDay": val_day, "MAE": fold_mae,
            "RMSE": float(np.sqrt(np.mean(errors ** 2))), "Bias": float(np.mean(errors)),
            "N": int(valid.sum()), "SeedPredictionSD": float(np.mean(np.std(np.stack(seed_predictions), axis=0))),
        })
        if return_predictions:
            for i, r in enumerate(val_records):
                for h in range(6):
                    prediction_rows.append({
                        "Model": model_type, "OriginID": r.origin_id, "ParticipantID": r.participant_id,
                        "TargetDay": r.target_day, "Split": "validation", "Horizon": h + 1,
                        "TargetOccasion": (r.target_day - 1) * 6 + h + 1,
                        "Actual": actual[i, h] if mask[i, h] == 1 else np.nan,
                        "ActualObserved": int(mask[i, h]), "Prediction": ensemble[i, h],
                    })
    folds = pd.DataFrame(fold_metrics)
    summary = {
        "Model": model_type, "SequenceSpec": sequence_spec, "PredictorSet": predictor_set,
        "TargetMode": target_mode, "HiddenSize": hidden_size, "Dropout": dropout,
        "Seeds": json.dumps(seeds), "ValidationMacroMAE": float(folds["MAE"].mean()),
        "ValidationWorstFoldMAE": float(folds["MAE"].max()),
        "ValidationFoldSD": float(folds["MAE"].std(ddof=0)),
        "ValidationPooledRMSE": float(np.sqrt(np.average(folds["RMSE"] ** 2, weights=folds["N"]))),
        "MeanSeedPredictionSD": float(folds["SeedPredictionSD"].mean()),
        "MedianBestEpoch": int(np.median(all_epochs)), "ValidationObservedTargets": int(folds["N"].sum()),
    }
    for _, row in folds.iterrows():
        summary[f"Day{int(row.ValidationDay)}MAE"] = row.MAE
    return summary, pd.DataFrame(prediction_rows) if prediction_rows else None, all_epochs


def prediction_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall, horizon = [], []
    for (model, split), g in predictions.groupby(["Model", "Split"]):
        valid = np.isfinite(g.Actual) & np.isfinite(g.Prediction)
        e = g.loc[valid, "Prediction"].to_numpy() - g.loc[valid, "Actual"].to_numpy()
        overall.append({"Model": model, "Split": split, "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)), "Bias": np.mean(e), "N": len(e)})
        for h, gh in g.groupby("Horizon"):
            validh = np.isfinite(gh.Actual) & np.isfinite(gh.Prediction)
            eh = gh.loc[validh, "Prediction"].to_numpy() - gh.loc[validh, "Actual"].to_numpy()
            horizon.append({"Model": model, "Split": split, "Horizon": int(h), "MAE": np.mean(np.abs(eh)), "RMSE": np.sqrt(np.mean(eh ** 2)), "Bias": np.mean(eh), "N": len(eh)})
    return pd.DataFrame(overall), pd.DataFrame(horizon)


def daily_mean_outputs(predictions: pd.DataFrame, minimum_positions: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, g in predictions.groupby(["Model", "Split", "OriginID", "ParticipantID", "TargetDay"]):
        valid = g[np.isfinite(g.Actual) & np.isfinite(g.Prediction)]
        if len(valid) < minimum_positions:
            continue
        rows.append({
            "Model": keys[0], "Split": keys[1], "OriginID": keys[2], "ParticipantID": keys[3],
            "TargetDay": keys[4], "ObservedPositions": len(valid),
            "ActualDailyMean": valid.Actual.mean(), "PredictedDailyMean": valid.Prediction.mean(),
        })
    pred = pd.DataFrame(rows)
    met = []
    for (model, split), g in pred.groupby(["Model", "Split"]):
        e = g.PredictedDailyMean.to_numpy() - g.ActualDailyMean.to_numpy()
        met.append({"Model": model, "Split": split, "MinimumObservedPositions": minimum_positions, "ParticipantDays": len(g), "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)), "Bias": np.mean(e)})
    return pred, pd.DataFrame(met)


def save_plots(tuning: pd.DataFrame, horizon_metrics: pd.DataFrame, image_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    best = tuning.sort_values("ValidationMacroMAE").groupby(["Model", "SequenceSpec"], as_index=False).first()
    for model, g in best.groupby("Model"):
        order = {s: i for i, s in enumerate(SEQUENCE_SPECS)}
        g = g.assign(Order=g.SequenceSpec.map(order)).sort_values("Order")
        ax.plot(g.SequenceSpec, g.ValidationMacroMAE, marker="o", label=model)
    ax.set_xlabel("Sequence representation")
    ax.set_ylabel("Rolling-validation macro MAE")
    ax.set_title("Best recurrent validation error by sequence length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / "06_recurrent_validation_mae_by_sequence.png", dpi=180)
    plt.close(fig)

    test = horizon_metrics[horizon_metrics.Split == "test"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for model, g in test.groupby("Model"):
        g = g.sort_values("Horizon")
        ax.plot(g.Horizon, g.MAE, marker="o", label=model)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test MAE")
    ax.set_xticks(range(1, 7))
    ax.set_title("RNN and LSTM test MAE by next-day horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / "06_recurrent_test_mae_by_horizon.png", dpi=180)
    plt.close(fig)


def write_report(selected: pd.DataFrame, metrics_df: pd.DataFrame, daily: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Step 06 — Revised RNN and LSTM Models",
        "",
        "## Methodological revision",
        "",
        "RNN and LSTM models now use three rolling-origin validation folds and directly output the six AffWB values of the next planned day. Candidate sequence lengths are 3, 6, 12, 18, and all available history. Mood-only and mood-plus-social-media inputs are compared, as are absolute targets and residual targets relative to persistence and previous-day seasonal forecasts.",
        "",
        "Each network uses one compact recurrent layer and six distinct output heads. Validation tuning averages predictions across three deterministic seeds, while the selected final configuration is refit with five seeds and evaluated as an ensemble on Day 7.",
        "",
        "## Selected configurations",
        "",
        selected.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Overall validation and test metrics",
        "",
        metrics_df.sort_values(["Split", "MAE"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Derived next-day mean metrics",
        "",
        daily.sort_values(["Split", "MAE"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "The recurrent models are evaluated against strong persistence and seasonal statistical baselines rather than assumed to outperform them. Selection considers mean rolling-validation error, the worst validation day, between-fold variability, and prediction variability across seeds. Given the short seven-day participant records, a complex recurrent model that does not improve on simple baselines is a substantive result rather than a pipeline failure.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--stage2-top", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    root = args.project_root.resolve()
    dirs = make_dirs(root)
    input_path = dirs["data"] / "03_modeling_long_data.csv"
    grid = load_full_grid(input_path)
    records = build_records(grid)

    sample_manifest = pd.DataFrame([{
        "OriginID": r.origin_id, "ParticipantID": r.participant_id, "TargetDay": r.target_day,
        "AvailableHistoryLength": len(r.X), "ObservedTargets": int(r.y_mask.sum()),
    } for r in records])
    sample_manifest.to_csv(dirs["tables"] / "06_sequence_sample_manifest.csv", index=False)

    print("Stage 1: evidence-guided sequence/predictor/target screening...", flush=True)
    stage1_tasks = [
        (model_type, sequence_spec, predictor_set, target_mode)
        for model_type in MODEL_TYPES
        for sequence_spec, predictor_set, target_mode in EVIDENCE_GUIDED_SEQUENCE_CONFIGS
    ]

    def run_stage1(task):
        model_type, sequence_spec, predictor_set, target_mode = task
        summary, _, _ = evaluate_candidate(
            records, model_type, sequence_spec, predictor_set, target_mode,
            hidden_size=16, dropout=0.1, seeds=STAGE1_SEEDS,
            max_epochs=MAX_EPOCHS_STAGE1,
        )
        return summary

    if args.n_jobs == 1:
        stage1_rows = [run_stage1(task) for task in stage1_tasks]
    else:
        stage1_rows = Parallel(n_jobs=max(1, args.n_jobs), backend="loky", verbose=5, max_nbytes=None)(
            delayed(run_stage1)(task) for task in stage1_tasks
        )
    stage1 = pd.DataFrame(stage1_rows).sort_values(["Model", "ValidationMacroMAE"])
    stage1.to_csv(dirs["tables"] / "06_recurrent_validation_screening.csv", index=False)

    print("Stage 2: architecture tuning with three seeds...", flush=True)
    stage2_tasks = []
    for model_type in MODEL_TYPES:
        top = stage1[stage1.Model == model_type].sort_values(["ValidationMacroMAE", "ValidationWorstFoldMAE"]).head(args.stage2_top)
        for row in top.itertuples(index=False):
            for hp in HYPER_GRID:
                stage2_tasks.append((model_type, row.SequenceSpec, row.PredictorSet, row.TargetMode, hp))

    def run_stage2(task):
        model_type, sequence_spec, predictor_set, target_mode, hp = task
        summary, _, _ = evaluate_candidate(
            records, model_type, sequence_spec, predictor_set, target_mode,
            hidden_size=hp["hidden_size"], dropout=hp["dropout"], seeds=STAGE2_SEEDS,
            max_epochs=MAX_EPOCHS_STAGE2,
        )
        return summary

    if args.n_jobs == 1:
        stage2_rows = [run_stage2(task) for task in stage2_tasks]
    else:
        stage2_rows = Parallel(n_jobs=max(1, args.n_jobs), backend="loky", verbose=5, max_nbytes=None)(
            delayed(run_stage2)(task) for task in stage2_tasks
        )
    tuning = pd.DataFrame(stage2_rows)
    tuning["Selected"] = 0
    selected_rows = []
    for model_type, g in tuning.groupby("Model"):
        idx = g.sort_values(["ValidationMacroMAE", "ValidationWorstFoldMAE", "ValidationFoldSD", "MeanSeedPredictionSD"]).index[0]
        tuning.loc[idx, "Selected"] = 1
        selected_rows.append(tuning.loc[idx].to_dict())
    tuning = tuning.sort_values(["Model", "ValidationMacroMAE"])
    selected = pd.DataFrame(selected_rows).sort_values("Model")
    tuning.to_csv(dirs["tables"] / "06_recurrent_validation_tuning.csv", index=False)
    selected.to_csv(dirs["tables"] / "06_selected_recurrent_configs.csv", index=False)

    print("Generating five-seed validation ensembles and final Day 7 predictions...", flush=True)
    prediction_parts = []
    config_json: dict[str, Any] = {}
    for row in selected.itertuples(index=False):
        summary, val_pred, epochs = evaluate_candidate(
            records, row.Model, row.SequenceSpec, row.PredictorSet, row.TargetMode,
            hidden_size=int(row.HiddenSize), dropout=float(row.Dropout), seeds=FINAL_SEEDS,
            max_epochs=MAX_EPOCHS_STAGE2, return_predictions=True,
        )
        prediction_parts.append(val_pred)
        final_epochs = int(np.median(epochs))
        dev_records = [r for r in records if 2 <= r.target_day <= 6]
        test_records = [r for r in records if r.target_day == TEST_DAY]
        seed_predictions = []
        for seed in FINAL_SEEDS:
            model, scaler, pred, actual, mask = train_fixed_epochs(
                row.Model, int(row.HiddenSize), float(row.Dropout), dev_records, test_records,
                row.SequenceSpec, row.PredictorSet, row.TargetMode, seed, final_epochs,
            )
            seed_predictions.append(pred)
            torch.save({
                "state_dict": model.state_dict(), "model_type": row.Model,
                "sequence_spec": row.SequenceSpec, "predictor_set": row.PredictorSet,
                "target_mode": row.TargetMode, "hidden_size": int(row.HiddenSize),
                "dropout": float(row.Dropout), "channels": select_channels(row.PredictorSet),
                "scaler": scaler.to_dict(), "epochs": final_epochs, "seed": seed,
            }, dirs["models"] / f"06_{row.Model.lower()}_seed_{seed}.pt")
        ensemble = np.mean(seed_predictions, axis=0)
        test_rows = []
        for i, r in enumerate(test_records):
            for h in range(6):
                test_rows.append({
                    "Model": row.Model, "OriginID": r.origin_id, "ParticipantID": r.participant_id,
                    "TargetDay": r.target_day, "Split": "test", "Horizon": h + 1,
                    "TargetOccasion": (r.target_day - 1) * 6 + h + 1,
                    "Actual": actual[i, h] if mask[i, h] == 1 else np.nan,
                    "ActualObserved": int(mask[i, h]), "Prediction": ensemble[i, h],
                })
        prediction_parts.append(pd.DataFrame(test_rows))
        config_json[row.Model] = {
            "sequence_spec": row.SequenceSpec, "predictor_set": row.PredictorSet,
            "target_mode": row.TargetMode, "hidden_size": int(row.HiddenSize),
            "dropout": float(row.Dropout), "validation_macro_mae_three_seed": float(row.ValidationMacroMAE),
            "validation_macro_mae_five_seed": float(summary["ValidationMacroMAE"]),
            "final_epochs": final_epochs, "final_seeds": FINAL_SEEDS,
        }

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(dirs["tables"] / "06_recurrent_model_predictions.csv", index=False)
    model_metrics, horizon_metrics = prediction_metrics(predictions)
    model_metrics.to_csv(dirs["tables"] / "06_recurrent_model_metrics.csv", index=False)
    horizon_metrics.to_csv(dirs["tables"] / "06_recurrent_model_horizon_metrics.csv", index=False)
    daily_pred, daily_metrics = daily_mean_outputs(predictions)
    daily_pred.to_csv(dirs["tables"] / "06_recurrent_daily_mean_predictions.csv", index=False)
    daily_metrics.to_csv(dirs["tables"] / "06_recurrent_daily_mean_metrics.csv", index=False)
    (dirs["models"] / "06_selected_recurrent_model_configs.json").write_text(json.dumps(config_json, indent=2), encoding="utf-8")
    save_plots(tuning, horizon_metrics, dirs["images"])
    write_report(selected, model_metrics, daily_metrics, dirs["reports"] / "06_recurrent_models_report.md")

    pd.DataFrame([{
        "Python": sys.version.split()[0], "Platform": platform.platform(),
        "Pandas": pd.__version__, "NumPy": np.__version__, "PyTorch": torch.__version__,
        "Stage2Seeds": json.dumps(STAGE2_SEEDS), "FinalSeeds": json.dumps(FINAL_SEEDS),
    }]).to_csv(dirs["tables"] / "06_runtime_environment.csv", index=False)
    manifest = []
    for folder in [dirs["tables"], dirs["images"], dirs["reports"], dirs["models"]]:
        for path in sorted(folder.glob("06_*")):
            manifest.append({"Path": str(path.relative_to(root)), "Bytes": path.stat().st_size, "SHA256": sha256(path)})
    pd.DataFrame(manifest).to_csv(dirs["tables"] / "06_output_manifest.csv", index=False)
    pd.DataFrame([{"Path": str(input_path.relative_to(root)), "Bytes": input_path.stat().st_size, "SHA256": sha256(input_path)}]).to_csv(dirs["tables"] / "06_step_input_manifest.csv", index=False)
    print("Step 06 completed.", flush=True)


if __name__ == "__main__":
    main()
