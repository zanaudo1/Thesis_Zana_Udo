"""
Step 05: Revised horizon-specific Random Forest, GBM, and XGBoost models.

The step implements rolling-origin validation (Days 4, 5, and 6), six separate
models per algorithm (one per forecast horizon), structured lag-set tuning,
optional social-media predictors, and absolute or residual targets relative to
persistence and seasonal-naive baselines. Day 7 remains untouched until the
selected configuration for every algorithm/horizon has been fixed.

Run from the project root:
    python 05_train_tree_models.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

STEP = "05"
SEED = 42
VALIDATION_DAYS = [4, 5, 6]
TEST_DAY = 7
HORIZONS = list(range(1, 7))

FEATURE_SPECS: dict[str, dict[str, list[int]]] = {
    "short_123": {"lags": [1, 2, 3], "windows": [3]},
    "day_1_6": {"lags": list(range(1, 7)), "windows": [3, 6]},
    "twoday_1_12": {"lags": list(range(1, 13)), "windows": [3, 6, 12]},
    "threeday_1_18": {"lags": list(range(1, 19)), "windows": [3, 6, 12, 18]},
    "sparse_1_6": {"lags": [1, 6], "windows": [6]},
    "sparse_1_6_12": {"lags": [1, 6, 12], "windows": [6, 12]},
    "sparse_1_6_12_18": {"lags": [1, 6, 12, 18], "windows": [6, 12, 18]},
    "mixed_123_6_12": {"lags": [1, 2, 3, 6, 12], "windows": [3, 6, 12]},
}

PREDICTOR_TARGET_COMBINATIONS = [
    (predictor_set, target_mode)
    for predictor_set in ["mood_only", "mood_social"]
    for target_mode in ["absolute", "residual_persistence", "residual_seasonal"]
] + [("social_only", "absolute")]

BASE_PARAMS: dict[str, dict[str, Any]] = {
    "RandomForest": {
        "n_estimators": 120,
        "max_depth": 6,
        "min_samples_leaf": 4,
        "max_features": 0.7,
    },
    "GBM": {
        "n_estimators": 120,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 4,
        "subsample": 0.8,
    },
    "XGBoost": {
        "n_estimators": 140,
        "learning_rate": 0.05,
        "max_depth": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "min_child_weight": 3.0,
    },
}

HYPER_GRIDS: dict[str, list[dict[str, Any]]] = {
    "RandomForest": [
        {"n_estimators": 80, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.7},
        {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 4, "max_features": 1.0},
    ],
    "GBM": [
        {"n_estimators": 80, "learning_rate": 0.04, "max_depth": 1, "min_samples_leaf": 5, "subsample": 0.8},
        {"n_estimators": 100, "learning_rate": 0.04, "max_depth": 2, "min_samples_leaf": 5, "subsample": 0.8},
    ],
    "XGBoost": [
        {"n_estimators": 80, "learning_rate": 0.04, "max_depth": 2, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "min_child_weight": 3.0},
        {"n_estimators": 100, "learning_rate": 0.04, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0, "min_child_weight": 3.0},
    ],
}

EVIDENCE_GUIDED_REPRESENTATIONS = [
    ("short_123", "mood_only", "residual_persistence"),
    ("day_1_6", "mood_only", "absolute"),
    ("sparse_1_6_12", "mood_only", "residual_seasonal"),
    ("sparse_1_6_12_18", "mood_only", "residual_seasonal"),
    ("mixed_123_6_12", "mood_social", "residual_persistence"),
]


@dataclass(frozen=True)
class Candidate:
    feature_spec: str
    predictor_set: str
    target_mode: str
    params: dict[str, Any]

    @property
    def id(self) -> str:
        return f"{self.feature_spec}|{self.predictor_set}|{self.target_mode}|{json.dumps(self.params, sort_keys=True)}"


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


def load_full_grid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "ParticipantID", "Occasion", "AffWB", "AffWB__observed",
        "OverallPassiveUse", "OverallPassiveUse__observed",
        "OverallActiveUse", "OverallActiveUse__observed",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Step 03 columns: {missing}")

    pieces = []
    for pid, g in df.groupby("ParticipantID", sort=True):
        g = g.sort_values("Occasion").drop_duplicates("Occasion", keep="last").set_index("Occasion")
        g = g.reindex(range(1, 43))
        g["ParticipantID"] = int(pid)
        g["Occasion"] = g.index.astype(int)
        for c in ["AffWB__observed", "OverallPassiveUse__observed", "OverallActiveUse__observed"]:
            g[c] = g[c].fillna(0).astype(int)
        g["PlannedDayIndex"] = ((g["Occasion"] - 1) // 6 + 1).astype(int)
        g["PlannedOccasionWithinDay"] = ((g["Occasion"] - 1) % 6 + 1).astype(int)
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def safe_slope(values: np.ndarray, observed: np.ndarray) -> float:
    idx = np.flatnonzero(observed & np.isfinite(values))
    if len(idx) < 2:
        return np.nan
    return float(np.polyfit(idx.astype(float), values[idx].astype(float), 1)[0])


def summary_features(values: np.ndarray, observed: np.ndarray, prefix: str, window: int) -> dict[str, float]:
    vals = values[-window:].astype(float)
    obs = observed[-window:].astype(bool) & np.isfinite(vals)
    selected = vals[obs]
    out = {
        f"{prefix}_w{window}_mean": float(np.mean(selected)) if len(selected) else np.nan,
        f"{prefix}_w{window}_std": float(np.std(selected, ddof=0)) if len(selected) > 1 else np.nan,
        f"{prefix}_w{window}_min": float(np.min(selected)) if len(selected) else np.nan,
        f"{prefix}_w{window}_max": float(np.max(selected)) if len(selected) else np.nan,
        f"{prefix}_w{window}_slope": safe_slope(vals, obs),
        f"{prefix}_w{window}_observed_fraction": float(np.mean(obs)) if len(obs) else 0.0,
    }
    return out


def latest_observed(values: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    idx = np.flatnonzero(observed.astype(bool) & np.isfinite(values))
    if not len(idx):
        return np.nan, np.nan
    last = int(idx[-1])
    return float(values[last]), float(len(values) - last)


def build_samples(grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pid, g in grid.groupby("ParticipantID", sort=True):
        g = g.sort_values("Occasion").set_index("Occasion")
        aff = g["AffWB"].to_numpy(float)
        aff_obs = g["AffWB__observed"].to_numpy(int)
        pas = g["OverallPassiveUse"].to_numpy(float)
        pas_obs = g["OverallPassiveUse__observed"].to_numpy(int)
        act = g["OverallActiveUse"].to_numpy(float)
        act_obs = g["OverallActiveUse__observed"].to_numpy(int)

        for target_day in range(2, 8):
            origin = (target_day - 1) * 6
            hist_aff, hist_aff_obs = aff[:origin], aff_obs[:origin]
            hist_pas, hist_pas_obs = pas[:origin], pas_obs[:origin]
            hist_act, hist_act_obs = act[:origin], act_obs[:origin]
            persistence_raw, last_obs_lag = latest_observed(hist_aff, hist_aff_obs)
            observed_hist = hist_aff[(hist_aff_obs == 1) & np.isfinite(hist_aff)]
            history_mean_raw = float(np.mean(observed_hist)) if len(observed_hist) else np.nan
            history_std_raw = float(np.std(observed_hist, ddof=0)) if len(observed_hist) > 1 else np.nan

            for horizon in HORIZONS:
                target_occ = origin + horizon
                target_idx = target_occ - 1
                actual = aff[target_idx] if target_idx < len(aff) else np.nan
                actual_observed = int(
                    target_idx < len(aff)
                    and aff_obs[target_idx] == 1
                    and np.isfinite(actual)
                )
                seasonal_idx = target_occ - 6 - 1
                seasonal_raw = (
                    float(aff[seasonal_idx])
                    if seasonal_idx >= 0 and aff_obs[seasonal_idx] == 1 and np.isfinite(aff[seasonal_idx])
                    else np.nan
                )
                base = {
                    "OriginID": f"P{int(pid):02d}_D{target_day}",
                    "ParticipantID": int(pid),
                    "TargetDay": target_day,
                    "Horizon": horizon,
                    "TargetOccasion": target_occ,
                    "Split": "test" if target_day == 7 else ("validation" if target_day in VALIDATION_DAYS else "development"),
                    "Actual": float(actual) if actual_observed else np.nan,
                    "ActualObserved": actual_observed,
                    "PersistenceRaw": persistence_raw,
                    "SeasonalRaw": seasonal_raw,
                    "HistoryMeanRaw": history_mean_raw,
                    "OriginDayIndex": target_day - 1,
                    "HistoryObservedCount": int(np.sum((hist_aff_obs == 1) & np.isfinite(hist_aff))),
                    "HistoryObservedFraction": float(np.mean((hist_aff_obs == 1) & np.isfinite(hist_aff))),
                    "HistoryMean": history_mean_raw,
                    "HistoryStd": history_std_raw,
                    "LastObservedLag": last_obs_lag,
                }
                for variable, vals, obs, prefix in [
                    ("AffWB", hist_aff, hist_aff_obs, "aff"),
                    ("Passive", hist_pas, hist_pas_obs, "passive"),
                    ("Active", hist_act, hist_act_obs, "active"),
                ]:
                    for lag in range(1, 19):
                        idx = origin - lag
                        value = vals[idx] if idx >= 0 else np.nan
                        is_obs = int(idx >= 0 and obs[idx] == 1 and np.isfinite(value))
                        base[f"{prefix}_lag_{lag}"] = float(value) if is_obs else np.nan
                        base[f"{prefix}_lag_{lag}_observed"] = is_obs
                    for window in [3, 6, 12, 18]:
                        base.update(summary_features(vals, obs, prefix, window))
                base["aff_lag1_minus_lag6"] = (
                    base["aff_lag_1"] - base["aff_lag_6"]
                    if np.isfinite(base["aff_lag_1"]) and np.isfinite(base["aff_lag_6"])
                    else np.nan
                )
                base["aff_lag6_minus_lag12"] = (
                    base["aff_lag_6"] - base["aff_lag_12"]
                    if np.isfinite(base["aff_lag_6"]) and np.isfinite(base["aff_lag_12"])
                    else np.nan
                )
                rows.append(base)
    return pd.DataFrame(rows)


def feature_columns(samples: pd.DataFrame, feature_spec: str, predictor_set: str) -> list[str]:
    spec = FEATURE_SPECS[feature_spec]
    columns = ["OriginDayIndex", "HistoryObservedCount", "HistoryObservedFraction", "LastObservedLag"]
    include_mood = predictor_set in {"mood_only", "mood_social"}
    include_social = predictor_set in {"social_only", "mood_social"}

    if include_mood:
        columns += ["HistoryMean", "HistoryStd", "aff_lag1_minus_lag6", "aff_lag6_minus_lag12"]
        for lag in spec["lags"]:
            columns += [f"aff_lag_{lag}", f"aff_lag_{lag}_observed"]
        for window in spec["windows"]:
            columns += [c for c in samples.columns if c.startswith(f"aff_w{window}_")]
    if include_social:
        for prefix in ["passive", "active"]:
            for lag in spec["lags"]:
                columns += [f"{prefix}_lag_{lag}", f"{prefix}_lag_{lag}_observed"]
            for window in spec["windows"]:
                columns += [c for c in samples.columns if c.startswith(f"{prefix}_w{window}_")]
    return list(dict.fromkeys(columns))


def build_model(name: str, params: dict[str, Any]) -> Pipeline:
    if name == "RepresentationScreen":
        estimator = ExtraTreesRegressor(random_state=SEED, n_jobs=-1, n_estimators=30, max_depth=6, min_samples_leaf=5, max_features=0.7)
    elif name == "RandomForest":
        estimator = RandomForestRegressor(random_state=SEED, n_jobs=1, **params)
    elif name == "GBM":
        estimator = GradientBoostingRegressor(random_state=SEED, loss="huber", **params)
    elif name == "XGBoost":
        estimator = XGBRegressor(
            objective="reg:absoluteerror",
            random_state=SEED,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
            **params,
        )
    else:
        raise ValueError(name)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", estimator),
    ])


def fill_baseline(df: pd.DataFrame, target_mode: str, fallback: float) -> np.ndarray:
    history = df["HistoryMeanRaw"].to_numpy(float)
    persistence = df["PersistenceRaw"].to_numpy(float)
    seasonal = df["SeasonalRaw"].to_numpy(float)
    persistence = np.where(np.isfinite(persistence), persistence, history)
    persistence = np.where(np.isfinite(persistence), persistence, fallback)
    seasonal = np.where(np.isfinite(seasonal), seasonal, persistence)
    seasonal = np.where(np.isfinite(seasonal), seasonal, history)
    seasonal = np.where(np.isfinite(seasonal), seasonal, fallback)
    if target_mode == "absolute":
        return np.zeros(len(df), dtype=float)
    if target_mode == "residual_persistence":
        return persistence.astype(float)
    if target_mode == "residual_seasonal":
        return seasonal.astype(float)
    raise ValueError(target_mode)


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "N": 0}
    error = pred[mask] - actual[mask]
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "Bias": float(np.mean(error)),
        "N": int(mask.sum()),
    }


def evaluate_candidate(
    samples: pd.DataFrame,
    model_name: str,
    horizon: int,
    candidate: Candidate,
    return_predictions: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    hdata = samples[(samples["Horizon"] == horizon) & (samples["ActualObserved"] == 1)].copy()
    features = feature_columns(samples, candidate.feature_spec, candidate.predictor_set)
    fold_rows = []
    pred_rows = []
    for val_day in VALIDATION_DAYS:
        train = hdata[(hdata["TargetDay"] >= 2) & (hdata["TargetDay"] < val_day)].copy()
        val = hdata[hdata["TargetDay"] == val_day].copy()
        if train.empty or val.empty:
            continue
        fallback = float(np.nanmedian(train["Actual"]))
        b_train = fill_baseline(train, candidate.target_mode, fallback)
        b_val = fill_baseline(val, candidate.target_mode, fallback)
        y_train = train["Actual"].to_numpy(float) - b_train
        model = build_model(model_name, candidate.params)
        model.fit(train[features], y_train)
        prediction = np.clip(model.predict(val[features]) + b_val, 1.0, 7.0)
        m = metrics(val["Actual"].to_numpy(float), prediction)
        fold_rows.append({"ValidationDay": val_day, **m})
        if return_predictions:
            out = val[["OriginID", "ParticipantID", "TargetDay", "Split", "Horizon", "TargetOccasion", "Actual", "ActualObserved"]].copy()
            out.insert(0, "Model", model_name)
            out["Prediction"] = prediction
            out["ValidationDay"] = val_day
            pred_rows.append(out)

    folds = pd.DataFrame(fold_rows)
    summary = {
        "Model": model_name,
        "Horizon": horizon,
        "FeatureSpec": candidate.feature_spec,
        "PredictorSet": candidate.predictor_set,
        "TargetMode": candidate.target_mode,
        "Parameters": json.dumps(candidate.params, sort_keys=True),
        "ValidationMacroMAE": float(folds["MAE"].mean()),
        "ValidationPooledRMSE": float(np.sqrt(np.average(folds["RMSE"] ** 2, weights=folds["N"]))) if len(folds) else np.nan,
        "ValidationWorstFoldMAE": float(folds["MAE"].max()),
        "ValidationFoldSD": float(folds["MAE"].std(ddof=0)),
        "ValidationObservedTargets": int(folds["N"].sum()),
        "ValidationFolds": int(len(folds)),
    }
    for _, row in folds.iterrows():
        summary[f"Day{int(row['ValidationDay'])}MAE"] = row["MAE"]
    return summary, pd.concat(pred_rows, ignore_index=True) if pred_rows else None


def select_top_configs(screening: pd.DataFrame, n: int = 3) -> dict[int, list[tuple[str, str, str]]]:
    result: dict[int, list[tuple[str, str, str]]] = {}
    for horizon, g in screening.groupby("Horizon"):
        top = g.sort_values(["ValidationMacroMAE", "ValidationWorstFoldMAE", "ValidationFoldSD"]).head(n)
        result[int(horizon)] = [(r.FeatureSpec, r.PredictorSet, r.TargetMode) for r in top.itertuples(index=False)]
    return result


def final_fit_predict(
    samples: pd.DataFrame,
    model_name: str,
    horizon: int,
    candidate: Candidate,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, list[str]]:
    features = feature_columns(samples, candidate.feature_spec, candidate.predictor_set)
    hdata = samples[(samples["Horizon"] == horizon) & (samples["ActualObserved"] == 1)].copy()
    dev = hdata[(hdata["TargetDay"] >= 2) & (hdata["TargetDay"] <= 6)].copy()
    test = hdata[hdata["TargetDay"] == TEST_DAY].copy()
    fallback = float(np.nanmedian(dev["Actual"]))
    b_dev = fill_baseline(dev, candidate.target_mode, fallback)
    b_test = fill_baseline(test, candidate.target_mode, fallback)
    y_dev = dev["Actual"].to_numpy(float) - b_dev
    model = build_model(model_name, candidate.params)
    model.fit(dev[features], y_dev)
    prediction = np.clip(model.predict(test[features]) + b_test, 1.0, 7.0)
    out = test[["OriginID", "ParticipantID", "TargetDay", "Split", "Horizon", "TargetOccasion", "Actual", "ActualObserved"]].copy()
    out.insert(0, "Model", model_name)
    out["Prediction"] = prediction
    return model, out, dev, features


def prediction_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows, horizon_rows = [], []
    for (model, split), g in predictions.groupby(["Model", "Split"]):
        m = metrics(g["Actual"].to_numpy(float), g["Prediction"].to_numpy(float))
        model_rows.append({"Model": model, "Split": split, **m})
        for h, gh in g.groupby("Horizon"):
            mh = metrics(gh["Actual"].to_numpy(float), gh["Prediction"].to_numpy(float))
            horizon_rows.append({"Model": model, "Split": split, "Horizon": int(h), **mh})
    return pd.DataFrame(model_rows), pd.DataFrame(horizon_rows)


def daily_mean_outputs(predictions: pd.DataFrame, minimum_positions: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, split, origin, pid, day), g in predictions.groupby(["Model", "Split", "OriginID", "ParticipantID", "TargetDay"]):
        valid = g[np.isfinite(g["Actual"]) & np.isfinite(g["Prediction"])]
        if len(valid) < minimum_positions:
            continue
        rows.append({
            "Model": model, "Split": split, "OriginID": origin, "ParticipantID": pid,
            "TargetDay": day, "ObservedPositions": len(valid),
            "ActualDailyMean": valid["Actual"].mean(), "PredictedDailyMean": valid["Prediction"].mean(),
        })
    pred = pd.DataFrame(rows)
    met = []
    for (model, split), g in pred.groupby(["Model", "Split"]):
        m = metrics(g["ActualDailyMean"].to_numpy(float), g["PredictedDailyMean"].to_numpy(float))
        met.append({"Model": model, "Split": split, "MinimumObservedPositions": minimum_positions, "ParticipantDays": len(g), **m})
    return pred, pd.DataFrame(met)


def extract_importance(model: Pipeline, features: list[str], model_name: str, horizon: int) -> pd.DataFrame:
    imputer = model.named_steps["imputer"]
    estimator = model.named_steps["model"]
    try:
        names = imputer.get_feature_names_out(features)
    except Exception:
        names = np.asarray(features)
    importance = getattr(estimator, "feature_importances_", None)
    if importance is None or len(importance) != len(names):
        return pd.DataFrame(columns=["Model", "Horizon", "Feature", "Importance"])
    return pd.DataFrame({"Model": model_name, "Horizon": horizon, "Feature": names, "Importance": importance})


def save_plots(tuning: pd.DataFrame, horizon_metrics: pd.DataFrame, importance: pd.DataFrame, image_dir: Path) -> None:
    selected = tuning[tuning["Selected"] == 1].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    for model, g in selected.groupby("Model"):
        g = g.sort_values("Horizon")
        ax.plot(g["Horizon"], g["ValidationMacroMAE"], marker="o", label=model)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Rolling-validation macro MAE")
    ax.set_title("Selected tree configuration validation error by horizon")
    ax.set_xticks(HORIZONS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / "05_tree_validation_mae_by_horizon.png", dpi=180)
    plt.close(fig)

    test = horizon_metrics[horizon_metrics["Split"] == "test"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for model, g in test.groupby("Model"):
        g = g.sort_values("Horizon")
        ax.plot(g["Horizon"], g["MAE"], marker="o", label=model)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test MAE")
    ax.set_title("Tree-model test MAE by next-day horizon")
    ax.set_xticks(HORIZONS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / "05_tree_test_mae_by_horizon.png", dpi=180)
    plt.close(fig)

    if not importance.empty:
        top = (importance.groupby(["Model", "Feature"], as_index=False)["Importance"].mean()
               .sort_values(["Model", "Importance"], ascending=[True, False])
               .groupby("Model", as_index=False).head(12))
        pivot = top.pivot_table(index="Feature", columns="Model", values="Importance", fill_value=0)
        order = pivot.max(axis=1).sort_values().index
        fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(order))))
        pivot.loc[order].plot(kind="barh", ax=ax)
        ax.set_xlabel("Mean feature importance across horizons")
        ax.set_title("Most influential features in selected tree models")
        fig.tight_layout()
        fig.savefig(image_dir / "05_tree_top_feature_importance.png", dpi=180)
        plt.close(fig)


def write_report(selected: pd.DataFrame, metrics_df: pd.DataFrame, daily: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Step 05 — Revised Tree-Based Forecasting Models",
        "",
        "## Methodological revision",
        "",
        "Random Forest, GBM, and XGBoost were trained as six horizon-specific models rather than one pooled model with a horizon indicator. Configuration selection used three rolling-origin validation folds: Days 1–3 to predict Day 4, Days 1–4 to predict Day 5, and Days 1–5 to predict Day 6. Day 7 was used only after all configurations had been selected.",
        "",
        "The tuning procedure compared short contiguous histories, one-to-three-day histories, sparse lag sets containing lags 1, 6, 12, and 18, mood-only and mood-plus-social-media predictors, and absolute versus residual targets relative to persistence and previous-day seasonal baselines.",
        "",
        "## Selected configurations",
        "",
        selected[["Model", "Horizon", "FeatureSpec", "PredictorSet", "TargetMode", "Parameters", "ValidationMacroMAE"]].sort_values(["Model", "Horizon"]).to_markdown(index=False),
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
        "The revised design tests whether nonlinear models benefit from the short-term persistence signal, the six-occasion daily structure, longer lags at 12 and 18 occasions, and social-media history. A social-media predictor set is retained only when it improves rolling validation for a particular model and horizon; it is not forced into all models. Residual-target configurations directly test whether a nonlinear model can improve on persistence or previous-day seasonal forecasts rather than merely reproducing them.",
        "",
        "Because the dataset contains only seven planned days per participant, all tree models are deliberately regularized and the validation-to-test gap remains essential for interpretation.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()

    root = args.project_root.resolve()
    dirs = make_dirs(root)
    input_path = dirs["data"] / "03_modeling_long_data.csv"
    print("Loading Step 03 data...", flush=True)
    grid = load_full_grid(input_path)
    samples = build_samples(grid)
    samples.to_csv(dirs["data"] / "05_tree_model_samples.csv", index=False)

    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
    print("Evidence-guided rolling-validation tuning for tree models...", flush=True)
    tasks = []
    for model_name in ["RandomForest", "GBM", "XGBoost"]:
        for horizon in HORIZONS:
            for feature_spec, predictor_set, target_mode in EVIDENCE_GUIDED_REPRESENTATIONS:
                for params in HYPER_GRIDS[model_name]:
                    tasks.append((model_name, horizon, Candidate(feature_spec, predictor_set, target_mode, params)))

    def run_task(task):
        model_name, horizon, candidate = task
        summary, _ = evaluate_candidate(samples, model_name, horizon, candidate)
        return summary

    tuning_rows = Parallel(n_jobs=max(1, args.n_jobs), backend="loky", verbose=5)(
        delayed(run_task)(task) for task in tasks
    )
    tuning = pd.DataFrame(tuning_rows)
    tuning["Selected"] = 0
    selected_rows = []
    for (model, horizon), g in tuning.groupby(["Model", "Horizon"]):
        idx = g.sort_values(["ValidationMacroMAE", "ValidationWorstFoldMAE", "ValidationFoldSD"]).index[0]
        tuning.loc[idx, "Selected"] = 1
        selected_rows.append(tuning.loc[idx].to_dict())
    tuning = tuning.sort_values(["Model", "Horizon", "ValidationMacroMAE"])
    selected = pd.DataFrame(selected_rows).sort_values(["Model", "Horizon"])
    tuning.to_csv(dirs["tables"] / "05_tree_validation_screening.csv", index=False)
    tuning.to_csv(dirs["tables"] / "05_tree_validation_tuning.csv", index=False)
    selected.to_csv(dirs["tables"] / "05_selected_tree_horizon_configs.csv", index=False)

    print("Generating selected rolling-validation and final Day 7 predictions...", flush=True)
    prediction_parts, importance_parts = [], []
    model_bundle: dict[str, Any] = {}
    config_bundle: dict[str, Any] = {}
    for row in selected.itertuples(index=False):
        params = json.loads(row.Parameters)
        candidate = Candidate(row.FeatureSpec, row.PredictorSet, row.TargetMode, params)
        _, val_pred = evaluate_candidate(samples, row.Model, int(row.Horizon), candidate, return_predictions=True)
        if val_pred is not None:
            prediction_parts.append(val_pred.drop(columns=["ValidationDay"]))
        model, test_pred, _, features = final_fit_predict(samples, row.Model, int(row.Horizon), candidate)
        prediction_parts.append(test_pred)
        key = f"{row.Model}_h{int(row.Horizon)}"
        model_bundle[key] = model
        config_bundle[key] = {
            "feature_spec": row.FeatureSpec,
            "predictor_set": row.PredictorSet,
            "target_mode": row.TargetMode,
            "params": params,
            "features": features,
            "validation_macro_mae": float(row.ValidationMacroMAE),
        }
        importance_parts.append(extract_importance(model, features, row.Model, int(row.Horizon)))

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(dirs["tables"] / "05_tree_model_predictions.csv", index=False)
    model_metrics, horizon_metrics = prediction_metrics(predictions)
    model_metrics.to_csv(dirs["tables"] / "05_tree_model_metrics.csv", index=False)
    horizon_metrics.to_csv(dirs["tables"] / "05_tree_model_horizon_metrics.csv", index=False)
    daily_pred, daily_metrics = daily_mean_outputs(predictions)
    daily_pred.to_csv(dirs["tables"] / "05_tree_daily_mean_predictions.csv", index=False)
    daily_metrics.to_csv(dirs["tables"] / "05_tree_daily_mean_metrics.csv", index=False)
    importance = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    importance.to_csv(dirs["tables"] / "05_tree_model_feature_importance.csv", index=False)

    joblib.dump(model_bundle, dirs["models"] / "05_selected_tree_models.joblib", compress=3)
    (dirs["models"] / "05_selected_tree_model_configs.json").write_text(json.dumps(config_bundle, indent=2), encoding="utf-8")
    save_plots(tuning, horizon_metrics, importance, dirs["images"])
    write_report(selected, model_metrics, daily_metrics, dirs["reports"] / "05_tree_models_report.md")

    runtime = pd.DataFrame([{
        "Python": sys.version.split()[0], "Platform": platform.platform(),
        "Pandas": pd.__version__, "NumPy": np.__version__,
        "ScikitLearn": __import__("sklearn").__version__, "XGBoost": __import__("xgboost").__version__,
        "RandomSeed": SEED,
    }])
    runtime.to_csv(dirs["tables"] / "05_runtime_environment.csv", index=False)
    manifest = []
    for folder in [dirs["tables"], dirs["images"], dirs["reports"], dirs["models"], dirs["data"]]:
        for path in sorted(folder.glob("05_*")):
            manifest.append({"Path": str(path.relative_to(root)), "Bytes": path.stat().st_size, "SHA256": sha256(path)})
    pd.DataFrame(manifest).to_csv(dirs["tables"] / "05_output_manifest.csv", index=False)
    pd.DataFrame([{"Path": str(input_path.relative_to(root)), "Bytes": input_path.stat().st_size, "SHA256": sha256(input_path)}]).to_csv(dirs["tables"] / "05_step_input_manifest.csv", index=False)
    print("Step 05 completed.", flush=True)


if __name__ == "__main__":
    main()
