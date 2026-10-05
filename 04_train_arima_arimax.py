"""
Step 04: Diagnose, tune, and evaluate ARIMA/SARIMA and ARIMAX/SARIMAX-X models.

This revised implementation addresses the methodological limitations of the
initial Step 04. It performs the following operations:

1. Calculates AffWB ACF/PACF diagnostics through lag 18 and lagged correlations
   between prior social-media exposure and later AffWB through lag 18.
2. Screens a broad, explicitly documented set of non-seasonal ARIMA and
   seasonal SARIMA candidates, including autoregressive orders 6 and 12 and a
   seasonal period of six ESM positions.
3. Selects the statistical response model by chronological rolling validation:
   histories ending before Days 4, 5, and 6 forecast the following six-position
   day. Day 7 remains untouched until the final test.
4. Tunes direct horizon-specific ARIMAX/SARIMAX-X models. For forecast horizon
   h, only exposure lags greater than or equal to h are eligible, because future
   target-day exposure is unknown at forecast time. Candidate lag structures
   include recent lags, lag 6, lag 12, combinations, rolling summaries, and
   correlation-guided lag sets.
5. Evaluates persistence, seasonal-naive, ARIMA/SARIMA, and ARIMAX/SARIMAX-X on
   the untouched Day 7 targets and reports six-step, horizon-specific, and
   derived next-day mean errors.

Run from the project root:
    python 04_train_arima_arimax.py

Optional runtime controls:
    python 04_train_arima_arimax.py --n-jobs 4
    python 04_train_arima_arimax.py --quick
"""

from __future__ import annotations

# Limit numerical-library thread pools before importing NumPy/statsmodels. This
# prevents each parallel worker from creating additional BLAS worker threads.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import concurrent.futures as futures
import hashlib
import json
import math
import multiprocessing as mp
import pickle
import platform
import tempfile
import signal
import subprocess
import sys
import threading
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -----------------------------------------------------------------------------
# 1. Project and validation configuration
# -----------------------------------------------------------------------------

STEP_PREFIX = "04_"
MEASUREMENTS_PER_DAY = 6
ARIMA_VALIDATION_DAYS = (4, 5, 6)
# Lag-12 ARIMAX needs at least two preceding days before the response/exogenous
# alignment can be estimated. Therefore, all ARIMAX candidates are compared on
# the same two rolling validation folds: Days 5 and 6.
ARIMAX_VALIDATION_DAYS = (5, 6)
TEST_DAY = 7
MAX_DIAGNOSTIC_LAG = 18
MAX_EXOG_LAG = 18
RANDOM_SEED = 42
MIN_DAILY_MEAN_TARGETS = 4
DEFAULT_N_JOBS = max(1, min(4, (os.cpu_count() or 2) - 1))
FIT_TIMEOUT_SECONDS = 4
ARIMA_CANDIDATE_TIMEOUT_SECONDS = 12
ARIMAX_CONFIGURATION_TIMEOUT_SECONDS = 12

EXPOSURE_SET_COLUMNS = {
    "passive": ["OverallPassiveUse"],
    "active": ["OverallActiveUse"],
    "both": ["OverallPassiveUse", "OverallActiveUse"],
}


@dataclass(frozen=True)
class StatisticalSpec:
    """One ARIMA/SARIMA response-model candidate."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def is_seasonal(self) -> bool:
        """Return True when the specification contains a seasonal component."""
        return any(self.seasonal_order[:3])

    @property
    def complexity(self) -> int:
        """Return a simple parameter-count proxy used only for tie breaking."""
        p, d, q = self.order
        P, D, Q, _ = self.seasonal_order
        return p + d + q + P + D + Q

    @property
    def candidate_id(self) -> str:
        """Create a stable, human-readable candidate identifier."""
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        if self.is_seasonal:
            return f"SARIMA({p},{d},{q})x({P},{D},{Q})[{s}]"
        return f"ARIMA({p},{d},{q})"


@dataclass(frozen=True)
class LagPattern:
    """A direct ARIMAX exogenous-lag structure."""

    name: str
    exact_lags: tuple[int, ...] = ()
    window_start: int | None = None
    window_end: int | None = None
    category: str = "other"

    @property
    def minimum_lag(self) -> int:
        """Return the smallest target-relative lag used by the pattern."""
        values = list(self.exact_lags)
        if self.window_start is not None:
            values.append(self.window_start)
        if not values:
            raise ValueError(f"Lag pattern {self.name} has no lags.")
        return min(values)

    @property
    def maximum_lag(self) -> int:
        """Return the largest target-relative lag used by the pattern."""
        values = list(self.exact_lags)
        if self.window_end is not None:
            values.append(self.window_end)
        if not values:
            raise ValueError(f"Lag pattern {self.name} has no lags.")
        return max(values)

    @property
    def complexity(self) -> int:
        """Return the number of derived numeric features per exposure variable."""
        count = len(self.exact_lags)
        if self.window_start is not None:
            count += 1
        return count


@dataclass
class FitResult:
    """Forecast values and fit-status metadata returned by one model fit."""

    forecast: np.ndarray
    fit_success: bool
    converged: bool
    aic: float
    error_type: str




@contextmanager
def fit_time_limit(seconds: int = FIT_TIMEOUT_SECONDS):
    """Interrupt pathologically slow short-series fits on POSIX systems."""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise TimeoutError("Model fit exceeded the per-fit time limit.")

    previous_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def create_directories(project_root: Path) -> dict[str, Path]:
    """Create all output directories used by this step."""
    directories = {
        "tables": project_root / "outputs" / "tables",
        "images": project_root / "outputs" / "images",
        "reports": project_root / "outputs" / "reports",
        "data": project_root / "outputs" / "data",
        "models": project_root / "outputs" / "models",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def file_sha256(path: Path) -> str:
    """Return a SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# -----------------------------------------------------------------------------
# 2. Input loading and participant-level ordered series
# -----------------------------------------------------------------------------


def load_inputs(modeling_path: Path, target_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the Step 03 occasion-level data and target manifest."""
    if not modeling_path.exists() or not target_path.exists():
        raise FileNotFoundError("Step 03 inputs are missing. Run Step 03 first.")

    modeling = pd.read_csv(modeling_path)
    targets = pd.read_csv(target_path)

    required_modeling = {
        "ParticipantID",
        "Occasion",
        "AffWB",
        "AffWB__observed",
        "OverallPassiveUse",
        "OverallPassiveUse__observed",
        "OverallActiveUse",
        "OverallActiveUse__observed",
    }
    required_targets = {
        "OriginID",
        "ParticipantID",
        "TargetDay",
        "Split",
        "Horizon",
        "TargetOccasion",
        "AffWBTarget",
        "AffWBTargetObserved",
    }
    missing_modeling = sorted(required_modeling - set(modeling.columns))
    missing_targets = sorted(required_targets - set(targets.columns))
    if missing_modeling:
        raise ValueError(f"Modeling columns missing: {missing_modeling}")
    if missing_targets:
        raise ValueError(f"Target columns missing: {missing_targets}")

    modeling = modeling.sort_values(["ParticipantID", "Occasion"]).reset_index(drop=True)
    targets = targets.sort_values(["ParticipantID", "TargetDay", "Horizon"]).reset_index(drop=True)

    if modeling.duplicated(["ParticipantID", "Occasion"]).any():
        raise ValueError("ParticipantID-Occasion keys must be unique.")
    if targets.duplicated(["ParticipantID", "TargetDay", "Horizon"]).any():
        raise ValueError("ParticipantID-TargetDay-Horizon keys must be unique.")

    return modeling, targets


def build_participant_series(modeling: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """
    Create an internal ordered-position view for each participant.

    The canonical Step 01/03 rows are not expanded or overwritten. Reindexing is
    performed only inside the model-specific view so that a missing planned
    position remains a gap rather than shifting all later lags.
    """
    result: dict[int, pd.DataFrame] = {}
    for participant_id, group in modeling.groupby("ParticipantID", sort=True):
        maximum_occasion = int(group["Occasion"].max())
        complete = pd.DataFrame({"Occasion": np.arange(1, maximum_occasion + 1, dtype=int)})
        complete = complete.merge(group, on="Occasion", how="left")
        complete["ParticipantID"] = int(participant_id)
        result[int(participant_id)] = complete
    return result


def target_lookup(targets: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    """Index six target rows by participant and target day."""
    return {
        (int(participant_id), int(target_day)): group.sort_values("Horizon").reset_index(drop=True)
        for (participant_id, target_day), group in targets.groupby(["ParticipantID", "TargetDay"])
    }


def extract_history(frame: pd.DataFrame, history_end: int, columns: Sequence[str]) -> pd.DataFrame:
    """Return positions 1..history_end with gaps preserved as missing values."""
    positions = pd.DataFrame({"Occasion": np.arange(1, history_end + 1, dtype=int)})
    available_columns = ["Occasion", *columns]
    return positions.merge(frame[available_columns], on="Occasion", how="left")


# -----------------------------------------------------------------------------
# 3. ACF/PACF and exogenous-lag diagnostics
# -----------------------------------------------------------------------------


def causal_fill_for_diagnostics(values: np.ndarray) -> np.ndarray:
    """Create a complete diagnostic series using forward fill and a level mean."""
    series = pd.Series(np.asarray(values, dtype=float))
    mean_value = series.mean(skipna=True)
    if not np.isfinite(mean_value):
        return np.full(len(series), np.nan)
    return series.ffill().fillna(mean_value).to_numpy(dtype=float)


def safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    """Calculate a correlation only when both arrays have sufficient variation."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x_valid = x[mask]
    y_valid = y[mask]
    if np.nanstd(x_valid) == 0 or np.nanstd(y_valid) == 0:
        return np.nan
    if method == "pearson":
        return float(pearsonr(x_valid, y_valid).statistic)
    return float(spearmanr(x_valid, y_valid).statistic)


def build_affwb_acf_pacf_diagnostics(
    participant_series: dict[int, pd.DataFrame],
    max_lag: int = MAX_DIAGNOSTIC_LAG,
) -> pd.DataFrame:
    """Aggregate individual ACF/PACF values and observed-pair lag correlations."""
    rows: list[dict[str, object]] = []

    for lag in range(1, max_lag + 1):
        individual_acf: list[float] = []
        individual_pacf: list[float] = []
        pooled_previous: list[float] = []
        pooled_current: list[float] = []
        pooled_previous_centered: list[float] = []
        pooled_current_centered: list[float] = []

        for participant_id, frame in participant_series.items():
            # Diagnostics are restricted to the development period (Days 1-6).
            # The untouched Day 7 test outcomes cannot influence candidate design.
            development_frame = frame.loc[
                frame["Occasion"].le((TEST_DAY - 1) * MEASUREMENTS_PER_DAY)
            ]
            y = development_frame["AffWB"].to_numpy(dtype=float)
            observed = y[np.isfinite(y)]
            if len(y) > max_lag + 3 and len(observed) >= max_lag + 3:
                filled = causal_fill_for_diagnostics(y)
                try:
                    acf_values = acf(filled, nlags=max_lag, fft=False)
                    pacf_values = pacf(filled, nlags=max_lag, method="ywmle")
                    individual_acf.append(float(acf_values[lag]))
                    individual_pacf.append(float(pacf_values[lag]))
                except Exception:
                    pass

            if len(y) > lag:
                previous = y[:-lag]
                current = y[lag:]
                mask = np.isfinite(previous) & np.isfinite(current)
                if mask.any():
                    previous_valid = previous[mask]
                    current_valid = current[mask]
                    pooled_previous.extend(previous_valid.tolist())
                    pooled_current.extend(current_valid.tolist())
                    participant_mean = float(np.nanmean(y))
                    pooled_previous_centered.extend((previous_valid - participant_mean).tolist())
                    pooled_current_centered.extend((current_valid - participant_mean).tolist())

        pooled_previous_array = np.asarray(pooled_previous, dtype=float)
        pooled_current_array = np.asarray(pooled_current, dtype=float)
        centered_previous_array = np.asarray(pooled_previous_centered, dtype=float)
        centered_current_array = np.asarray(pooled_current_centered, dtype=float)
        valid_pairs = len(pooled_current_array)
        approximate_bound = 1.96 / math.sqrt(valid_pairs) if valid_pairs else np.nan

        rows.append(
            {
                "Lag": lag,
                "ValidObservedPairs": valid_pairs,
                "PooledPearson": safe_corr(pooled_previous_array, pooled_current_array, "pearson"),
                "PooledSpearman": safe_corr(pooled_previous_array, pooled_current_array, "spearman"),
                "WithinPersonCenteredPearson": safe_corr(
                    centered_previous_array, centered_current_array, "pearson"
                ),
                "MeanIndividualACF": float(np.nanmean(individual_acf)) if individual_acf else np.nan,
                "MedianIndividualACF": float(np.nanmedian(individual_acf)) if individual_acf else np.nan,
                "MeanIndividualPACF": float(np.nanmean(individual_pacf)) if individual_pacf else np.nan,
                "MedianIndividualPACF": float(np.nanmedian(individual_pacf)) if individual_pacf else np.nan,
                "ParticipantsWithACF": len(individual_acf),
                "ParticipantsWithPACF": len(individual_pacf),
                "ApproximatePooled95Bound": approximate_bound,
            }
        )

    return pd.DataFrame(rows)


def build_exogenous_lag_diagnostics(
    participant_series: dict[int, pd.DataFrame],
    max_lag: int = MAX_EXOG_LAG,
) -> pd.DataFrame:
    """Calculate lagged exposure-to-AffWB correlations without using future exposure."""
    rows: list[dict[str, object]] = []

    for predictor in ["OverallPassiveUse", "OverallActiveUse"]:
        for lag in range(1, max_lag + 1):
            pooled_x: list[float] = []
            pooled_y: list[float] = []
            centered_x: list[float] = []
            centered_y: list[float] = []
            participants = 0

            for participant_id, frame in participant_series.items():
                # Use only Days 1-6 so lag selection cannot inspect Day 7 test data.
                development_frame = frame.loc[
                    frame["Occasion"].le((TEST_DAY - 1) * MEASUREMENTS_PER_DAY)
                ]
                x = development_frame[predictor].to_numpy(dtype=float)
                y = development_frame["AffWB"].to_numpy(dtype=float)
                if len(y) <= lag:
                    continue
                lagged_x = x[:-lag]
                current_y = y[lag:]
                mask = np.isfinite(lagged_x) & np.isfinite(current_y)
                if mask.sum() == 0:
                    continue

                participants += 1
                x_valid = lagged_x[mask]
                y_valid = current_y[mask]
                pooled_x.extend(x_valid.tolist())
                pooled_y.extend(y_valid.tolist())
                centered_x.extend((x_valid - np.mean(x_valid)).tolist())
                centered_y.extend((y_valid - np.mean(y_valid)).tolist())

            x_array = np.asarray(pooled_x, dtype=float)
            y_array = np.asarray(pooled_y, dtype=float)
            centered_x_array = np.asarray(centered_x, dtype=float)
            centered_y_array = np.asarray(centered_y, dtype=float)

            rows.append(
                {
                    "Predictor": predictor,
                    "Lag": lag,
                    "ValidPairs": len(x_array),
                    "Participants": participants,
                    "PooledPearson": safe_corr(x_array, y_array, "pearson"),
                    "PooledSpearman": safe_corr(x_array, y_array, "spearman"),
                    "WithinPersonCenteredPearson": safe_corr(
                        centered_x_array, centered_y_array, "pearson"
                    ),
                    "WithinPersonCenteredSpearman": safe_corr(
                        centered_x_array, centered_y_array, "spearman"
                    ),
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 4. ARIMA/SARIMA candidate generation and model fitting
# -----------------------------------------------------------------------------


def generate_statistical_candidates(quick: bool = False) -> list[StatisticalSpec]:
    """
    Generate a broad but finite ARIMA/SARIMA candidate set.

    The non-seasonal grid explicitly includes p=6 and p=12 as requested. The
    seasonal grid represents lag 6 through P=1 and lag 12 through P=2, while Q
    terms test seasonal moving-average structure. Double differencing d+D>1 is
    excluded because only 18-30 historical positions are available in validation.
    """
    candidates: set[StatisticalSpec] = set()

    if quick:
        pure_ar_p = [0, 1, 2, 3, 6, 12]
        pure_ma_q = [1, 2, 3, 6]
        mixed_pairs = [(1, 1), (2, 1), (1, 2), (3, 1), (6, 1)]
    else:
        # Every pure AR order from 0 through 12 is examined. Mixed models are
        # intentionally bounded so their parameter count remains defensible for
        # the short participant histories available at rolling origins.
        pure_ar_p = list(range(0, 13))
        pure_ma_q = list(range(1, 7))
        mixed_pairs = [
            (p, q)
            for p in range(1, 7)
            for q in range(1, 4)
            if p + q <= 7
        ]
        mixed_pairs.extend([(12, 1), (1, 6), (6, 6)])

    for d in [0, 1]:
        for p in pure_ar_p:
            candidates.add(StatisticalSpec(order=(p, d, 0)))
        for q in pure_ma_q:
            candidates.add(StatisticalSpec(order=(0, d, q)))
        for p, q in mixed_pairs:
            candidates.add(StatisticalSpec(order=(p, d, q)))

    # The seasonal grid is intentionally parsimonious because each participant
    # contributes only 18-30 observations at validation origins. We test the
    # scientifically relevant period-six AR, MA, and seasonal-difference terms,
    # plus a simple P=2 structure representing lag 12. High-order interactions
    # between several seasonal and non-seasonal components are excluded because
    # they are not estimable reliably in these short series.
    base_orders = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
    ]
    primary_seasonal_orders = [
        (1, 0, 0, 6),
        (0, 0, 1, 6),
        (0, 1, 0, 6),
    ]
    if quick:
        base_orders = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0)]

    for order in base_orders:
        for seasonal_order in primary_seasonal_orders:
            p, d, q = order
            P, D, Q, _ = seasonal_order
            if d + D > 1:
                continue
            if p + q + P + Q > 3:
                continue
            candidates.add(StatisticalSpec(order=order, seasonal_order=seasonal_order))

    # Lag-12 and mixed seasonal terms are tested only with a simple response
    # order, avoiding the unstable over-parameterized combinations observed in
    # short participant histories.
    candidates.add(StatisticalSpec(order=(0, 0, 0), seasonal_order=(2, 0, 0, 6)))
    candidates.add(StatisticalSpec(order=(1, 1, 0), seasonal_order=(2, 0, 0, 6)))
    candidates.add(StatisticalSpec(order=(0, 0, 0), seasonal_order=(0, 0, 2, 6)))
    candidates.add(StatisticalSpec(order=(0, 0, 0), seasonal_order=(1, 0, 1, 6)))

    return sorted(candidates, key=lambda spec: (spec.is_seasonal, spec.complexity, spec.candidate_id))


def trend_for_spec(spec: StatisticalSpec) -> str:
    """Use an intercept only when neither ordinary nor seasonal differencing is applied."""
    _, d, _ = spec.order
    _, D, _, _ = spec.seasonal_order
    return "c" if d + D == 0 else "n"


def required_observations(spec: StatisticalSpec, exog_columns: int = 0) -> int:
    """Return a conservative minimum history size for a candidate fit."""
    p, d, q = spec.order
    P, D, Q, s = spec.seasonal_order
    largest_lag = max(p, q, P * s if s else 0, Q * s if s else 0)
    parameter_proxy = p + q + P + Q + exog_columns
    return max(10, largest_lag + d + D * (s or 0) + 3, parameter_proxy + 7)


def fit_sarimax_forecast(
    y: np.ndarray,
    spec: StatisticalSpec,
    steps: int,
    exog_history: np.ndarray | None = None,
    exog_future: np.ndarray | None = None,
) -> FitResult:
    """Fit one participant-specific ARIMA/SARIMAX model and forecast."""
    y_array = np.asarray(y, dtype=float)
    observed_count = int(np.isfinite(y_array).sum())
    exog_columns = 0 if exog_history is None else int(exog_history.shape[1])
    if observed_count < required_observations(spec, exog_columns):
        return FitResult(
            forecast=np.full(steps, np.nan),
            fit_success=False,
            converged=False,
            aic=np.nan,
            error_type="insufficient_history",
        )

    try:
        with fit_time_limit():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if not spec.is_seasonal and exog_history is None:
                    # Hannan-Rissanen is substantially faster and more stable for
                    # the broad non-seasonal grid on these short participant
                    # series. Filling is restricted to the history available at
                    # the forecast origin; no validation/test target is used.
                    y_filled = causal_fill_for_diagnostics(y_array)
                    model = ARIMAModel(
                        y_filled,
                        exog=exog_history,
                        order=spec.order,
                        trend=trend_for_spec(spec),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(method="hannan_rissanen")
                    forecast = np.asarray(
                        fitted.forecast(steps=steps, exog=exog_future), dtype=float
                    )
                    converged = True
                else:
                    # Seasonal candidates and models with exogenous regressors use
                    # the ARIMA state-space wrapper. Its bounded optimizer is more
                    # stable for these short participant histories than directly
                    # optimizing a raw SARIMAX object.
                    model = ARIMAModel(
                        y_array,
                        exog=exog_history,
                        order=spec.order,
                        seasonal_order=spec.seasonal_order,
                        trend=trend_for_spec(spec),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(
                        method="statespace",
                        method_kwargs={
                            "disp": 0,
                            "maxiter": 30,
                        },
                        low_memory=True,
                    )
                    forecast = np.asarray(
                        fitted.forecast(steps=steps, exog=exog_future), dtype=float
                    )
                    converged = bool(fitted.mle_retvals.get("converged", True))

                aic_value = float(fitted.aic) if np.isfinite(fitted.aic) else np.nan

        if len(forecast) != steps or not np.isfinite(forecast).any():
            raise ValueError("Forecast did not contain finite values.")
        return FitResult(
            forecast=np.clip(forecast, 1.0, 7.0),
            fit_success=True,
            converged=converged,
            aic=aic_value,
            error_type="",
        )
    except Exception as exc:
        return FitResult(
            forecast=np.full(steps, np.nan),
            fit_success=False,
            converged=False,
            aic=np.nan,
            error_type=type(exc).__name__,
        )


# -----------------------------------------------------------------------------
# 5. Common prediction rows and metrics
# -----------------------------------------------------------------------------


def prediction_rows(
    participant_id: int,
    target_day: int,
    split: str,
    model_name: str,
    predictions: Sequence[float],
    target_group: pd.DataFrame,
    configuration_id: str,
) -> list[dict[str, object]]:
    """Convert one trajectory forecast into the common long prediction schema."""
    rows: list[dict[str, object]] = []
    for horizon, prediction in enumerate(predictions, start=1):
        target_row = target_group.iloc[horizon - 1]
        rows.append(
            {
                "Model": model_name,
                "ConfigurationID": configuration_id,
                "OriginID": target_row["OriginID"],
                "ParticipantID": participant_id,
                "TargetDay": target_day,
                "Split": split,
                "Horizon": horizon,
                "TargetOccasion": int(target_row["TargetOccasion"]),
                "Actual": target_row["AffWBTarget"],
                "ActualObserved": int(target_row["AffWBTargetObserved"]),
                "Prediction": float(prediction) if np.isfinite(prediction) else np.nan,
            }
        )
    return rows


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate masked forecast metrics and coverage."""
    eligible = predictions.loc[predictions["ActualObserved"].eq(1)].copy()
    valid = eligible.loc[
        np.isfinite(eligible["Prediction"]) & np.isfinite(eligible["Actual"])
    ].copy()
    coverage = len(valid) / len(eligible) if len(eligible) else np.nan

    if len(valid) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
            "ObservedTargets": int(len(eligible)),
            "PredictedTargets": 0,
            "Coverage": coverage,
        }

    errors = valid["Prediction"].to_numpy(dtype=float) - valid["Actual"].to_numpy(dtype=float)
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "Bias": float(np.mean(errors)),
        "ObservedTargets": int(len(eligible)),
        "PredictedTargets": int(len(valid)),
        "Coverage": float(coverage),
    }


def summarize_candidate_predictions(
    predictions: pd.DataFrame,
    fit_attempts: int,
    fit_successes: int,
    converged_fits: int,
) -> dict[str, float]:
    """Summarize pooled and fold-macro validation performance."""
    overall = calculate_metrics(predictions)
    fold_metrics = []
    for target_day, group in predictions.groupby("TargetDay"):
        metrics = calculate_metrics(group)
        fold_metrics.append(
            {
                "TargetDay": int(target_day),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "Coverage": metrics["Coverage"],
            }
        )

    fold_frame = pd.DataFrame(fold_metrics)
    return {
        "ValidationMAE": overall["MAE"],
        "ValidationRMSE": overall["RMSE"],
        "ValidationBias": overall["Bias"],
        "MacroFoldMAE": float(fold_frame["MAE"].mean()) if not fold_frame.empty else np.nan,
        "MacroFoldRMSE": float(fold_frame["RMSE"].mean()) if not fold_frame.empty else np.nan,
        "MinimumFoldCoverage": float(fold_frame["Coverage"].min()) if not fold_frame.empty else np.nan,
        "Coverage": overall["Coverage"],
        "ObservedTargets": overall["ObservedTargets"],
        "PredictedTargets": overall["PredictedTargets"],
        "FitAttempts": int(fit_attempts),
        "FitSuccesses": int(fit_successes),
        "FitSuccessRate": fit_successes / fit_attempts if fit_attempts else np.nan,
        "ConvergedFits": int(converged_fits),
        "ConvergenceRateAmongSuccessful": (
            converged_fits / fit_successes if fit_successes else np.nan
        ),
    }


def per_horizon_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate masked metrics separately for each forecast horizon."""
    rows = []
    for horizon, group in predictions.groupby("Horizon"):
        rows.append({"Horizon": int(horizon), **calculate_metrics(group)})
    return pd.DataFrame(rows)


def derived_daily_mean_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Derive participant-day means from the same observed target positions."""
    rows: list[dict[str, object]] = []
    group_columns = ["Model", "Split", "ParticipantID", "TargetDay", "OriginID"]
    for keys, group in predictions.groupby(group_columns, dropna=False):
        model_name, split, participant_id, target_day, origin_id = keys
        valid = group.loc[
            group["ActualObserved"].eq(1)
            & np.isfinite(group["Actual"])
            & np.isfinite(group["Prediction"])
        ]
        if valid.empty:
            continue
        rows.append(
            {
                "Model": model_name,
                "Split": split,
                "ParticipantID": int(participant_id),
                "TargetDay": int(target_day),
                "OriginID": origin_id,
                "ObservedPositions": int(len(valid)),
                "ActualDailyMean": float(valid["Actual"].mean()),
                "PredictedDailyMean": float(valid["Prediction"].mean()),
            }
        )
    return pd.DataFrame(rows)


def daily_mean_metrics(daily_predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived next-day mean metrics with a minimum observed-position rule."""
    rows: list[dict[str, object]] = []
    if daily_predictions.empty:
        return pd.DataFrame(rows)
    eligible = daily_predictions.loc[
        daily_predictions["ObservedPositions"] >= MIN_DAILY_MEAN_TARGETS
    ].copy()
    for (model_name, split), group in eligible.groupby(["Model", "Split"]):
        errors = group["PredictedDailyMean"] - group["ActualDailyMean"]
        rows.append(
            {
                "Model": model_name,
                "Split": split,
                "MinimumObservedPositions": MIN_DAILY_MEAN_TARGETS,
                "ParticipantDays": int(len(group)),
                "MAE": float(np.mean(np.abs(errors))),
                "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
                "Bias": float(np.mean(errors)),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 6. ARIMA/SARIMA candidate evaluation and staged tuning
# -----------------------------------------------------------------------------


def evaluate_arima_spec(
    spec: StatisticalSpec,
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate one common ARIMA/SARIMA structure across participants and days."""
    rows: list[dict[str, object]] = []
    fit_attempts = 0
    fit_successes = 0
    converged_fits = 0

    for target_day in target_days:
        history_end = (target_day - 1) * MEASUREMENTS_PER_DAY
        for participant_id, frame in participant_series.items():
            history = extract_history(frame, history_end, ["AffWB"])
            y = history["AffWB"].to_numpy(dtype=float)
            result = fit_sarimax_forecast(y, spec, steps=MEASUREMENTS_PER_DAY)
            fit_attempts += 1
            fit_successes += int(result.fit_success)
            converged_fits += int(result.fit_success and result.converged)
            rows.extend(
                prediction_rows(
                    participant_id=participant_id,
                    target_day=target_day,
                    split=split,
                    model_name="ARIMA",
                    predictions=result.forecast,
                    target_group=targets_by_key[(participant_id, target_day)],
                    configuration_id=spec.candidate_id,
                )
            )

    prediction_frame = pd.DataFrame(rows)
    summary = summarize_candidate_predictions(
        prediction_frame, fit_attempts, fit_successes, converged_fits
    )
    p, d, q = spec.order
    P, D, Q, s = spec.seasonal_order
    summary.update(
        {
            "CandidateID": spec.candidate_id,
            "ModelForm": "SARIMA" if spec.is_seasonal else "ARIMA",
            "p": p,
            "d": d,
            "q": q,
            "P": P,
            "D": D,
            "Q": Q,
            "SeasonalPeriod": s,
            "Trend": trend_for_spec(spec),
            "Complexity": spec.complexity,
        }
    )
    return summary, prediction_frame


def _arima_evaluation_worker(
    result_path: str,
    spec: StatisticalSpec,
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
) -> None:
    """Run one ARIMA/SARIMA candidate and persist its result atomically."""
    try:
        payload = (
            "ok",
            evaluate_arima_spec(
                spec, participant_series, targets_by_key, target_days, split
            ),
        )
    except BaseException as exc:
        payload = ("error", f"{type(exc).__name__}: {exc}")
    with open(result_path, "wb") as output_file:
        pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def _arimax_evaluation_worker(
    result_path: str,
    spec: StatisticalSpec,
    exogenous_set: str,
    pattern: LagPattern,
    horizon: int,
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
) -> None:
    """Run one direct ARIMAX/SARIMAX-X configuration and persist its result."""
    try:
        payload = (
            "ok",
            evaluate_arimax_configuration(
                spec,
                exogenous_set,
                pattern,
                horizon,
                participant_series,
                targets_by_key,
                target_days,
                split,
            ),
        )
    except BaseException as exc:
        payload = ("error", f"{type(exc).__name__}: {exc}")
    with open(result_path, "wb") as output_file:
        pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def failed_arima_evaluation(
    spec: StatisticalSpec,
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
    status: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Create explicit unavailable predictions when a whole candidate times out."""
    rows: list[dict[str, object]] = []
    for target_day in target_days:
        for participant_id in participant_series:
            rows.extend(
                prediction_rows(
                    participant_id,
                    target_day,
                    split,
                    "ARIMA",
                    np.full(MEASUREMENTS_PER_DAY, np.nan),
                    targets_by_key[(participant_id, target_day)],
                    spec.candidate_id,
                )
            )
    frame = pd.DataFrame(rows)
    summary = summarize_candidate_predictions(
        frame,
        fit_attempts=len(participant_series) * len(target_days),
        fit_successes=0,
        converged_fits=0,
    )
    p, d, q = spec.order
    P, D, Q, seasonal_period = spec.seasonal_order
    summary.update(
        {
            "CandidateID": spec.candidate_id,
            "ModelForm": "SARIMA" if spec.is_seasonal else "ARIMA",
            "p": p,
            "d": d,
            "q": q,
            "P": P,
            "D": D,
            "Q": Q,
            "SeasonalPeriod": seasonal_period,
            "Trend": trend_for_spec(spec),
            "Complexity": spec.complexity,
            "EvaluationStatus": status,
        }
    )
    return summary, frame


def failed_arimax_evaluation(
    spec: StatisticalSpec,
    exogenous_set: str,
    pattern: LagPattern,
    horizon: int,
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
    status: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Create explicit unavailable direct forecasts after a configuration timeout."""
    configuration_id = (
        f"{spec.candidate_id}|{exogenous_set}|h{horizon}|{pattern.name}"
    )
    rows: list[dict[str, object]] = []
    for target_day in target_days:
        for participant_id in participant_series:
            target_group = targets_by_key[(participant_id, target_day)]
            target_row = target_group.loc[target_group["Horizon"].eq(horizon)].iloc[0]
            rows.append(
                {
                    "Model": "ARIMAX",
                    "ConfigurationID": configuration_id,
                    "OriginID": target_row["OriginID"],
                    "ParticipantID": participant_id,
                    "TargetDay": target_day,
                    "Split": split,
                    "Horizon": horizon,
                    "TargetOccasion": int(target_row["TargetOccasion"]),
                    "Actual": target_row["AffWBTarget"],
                    "ActualObserved": int(target_row["AffWBTargetObserved"]),
                    "Prediction": np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    summary = summarize_candidate_predictions(
        frame,
        fit_attempts=len(participant_series) * len(target_days),
        fit_successes=0,
        converged_fits=0,
    )
    p, d, q = spec.order
    P, D, Q, seasonal_period = spec.seasonal_order
    summary.update(
        {
            "ConfigurationID": configuration_id,
            "ResponseCandidateID": spec.candidate_id,
            "ModelForm": "SARIMAX" if spec.is_seasonal else "ARIMAX",
            "p": p,
            "d": d,
            "q": q,
            "P": P,
            "D": D,
            "Q": Q,
            "SeasonalPeriod": seasonal_period,
            "Horizon": horizon,
            "ExogenousSet": exogenous_set,
            "LagPattern": pattern.name,
            "LagCategory": pattern.category,
            "ExactLags": ",".join(map(str, pattern.exact_lags)),
            "WindowStart": pattern.window_start,
            "WindowEnd": pattern.window_end,
            "MinimumLag": pattern.minimum_lag,
            "MaximumLag": pattern.maximum_lag,
            "PatternComplexity": pattern.complexity,
            "ResponseComplexity": spec.complexity,
            "MedianEffectiveExogColumns": np.nan,
            "EvaluationStatus": status,
        }
    )
    return summary, frame


def run_arima_candidate_with_timeout(
    spec: StatisticalSpec,
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate one candidate with a hard process-level timeout."""
    context = mp.get_context("spawn")
    file_descriptor, result_path = tempfile.mkstemp(prefix="mda_arima_", suffix=".pkl")
    os.close(file_descriptor)
    process = context.Process(
        target=_arima_evaluation_worker,
        args=(result_path, spec, participant_series, targets_by_key, target_days, split),
    )
    process.start()
    process.join(timeout=ARIMA_CANDIDATE_TIMEOUT_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
        Path(result_path).unlink(missing_ok=True)
        return failed_arima_evaluation(
            spec, participant_series, targets_by_key, target_days, split, "timeout"
        )

    try:
        with open(result_path, "rb") as input_file:
            status, payload = pickle.load(input_file)
    except Exception as exc:
        Path(result_path).unlink(missing_ok=True)
        return failed_arima_evaluation(
            spec,
            participant_series,
            targets_by_key,
            target_days,
            split,
            f"result_read_error:{type(exc).__name__}",
        )
    finally:
        Path(result_path).unlink(missing_ok=True)

    if status == "ok":
        summary, frame = payload
        summary.setdefault("EvaluationStatus", "completed")
        return summary, frame
    return failed_arima_evaluation(
        spec,
        participant_series,
        targets_by_key,
        target_days,
        split,
        f"worker_error:{payload}",
    )


def run_arimax_configuration_with_timeout(
    spec: StatisticalSpec,
    exogenous_set: str,
    pattern: LagPattern,
    horizon: int,
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate one ARIMAX configuration with a hard process-level timeout."""
    context = mp.get_context("spawn")
    file_descriptor, result_path = tempfile.mkstemp(prefix="mda_arimax_", suffix=".pkl")
    os.close(file_descriptor)
    process = context.Process(
        target=_arimax_evaluation_worker,
        args=(
            result_path,
            spec,
            exogenous_set,
            pattern,
            horizon,
            participant_series,
            targets_by_key,
            target_days,
            split,
        ),
    )
    process.start()
    process.join(timeout=ARIMAX_CONFIGURATION_TIMEOUT_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
        Path(result_path).unlink(missing_ok=True)
        return failed_arimax_evaluation(
            spec,
            exogenous_set,
            pattern,
            horizon,
            participant_series,
            targets_by_key,
            target_days,
            split,
            "timeout",
        )

    try:
        with open(result_path, "rb") as input_file:
            status, payload = pickle.load(input_file)
    except Exception as exc:
        Path(result_path).unlink(missing_ok=True)
        return failed_arimax_evaluation(
            spec,
            exogenous_set,
            pattern,
            horizon,
            participant_series,
            targets_by_key,
            target_days,
            split,
            f"result_read_error:{type(exc).__name__}",
        )
    finally:
        Path(result_path).unlink(missing_ok=True)

    if status == "ok":
        summary, frame = payload
        summary.setdefault("EvaluationStatus", "completed")
        return summary, frame
    return failed_arimax_evaluation(
        spec,
        exogenous_set,
        pattern,
        horizon,
        participant_series,
        targets_by_key,
        target_days,
        split,
        f"worker_error:{payload}",
    )


def choose_arima_stage2_candidates(stage1: pd.DataFrame, maximum: int = 24) -> list[str]:
    """Select top and mandatory representative candidates for rolling validation."""
    eligible = stage1.loc[
        stage1["Coverage"].ge(0.75)
        & stage1["FitSuccessRate"].ge(0.70)
        & stage1["ConvergenceRateAmongSuccessful"].ge(0.50)
    ].copy()
    if eligible.empty:
        raise RuntimeError("No ARIMA/SARIMA candidate passed the Stage 1 coverage threshold.")

    selected: list[str] = []

    def add_rows(frame: pd.DataFrame, count: int = 1) -> None:
        for candidate_id in frame.sort_values(
            ["ValidationMAE", "ValidationRMSE", "Complexity"]
        )["CandidateID"].head(count):
            if candidate_id not in selected:
                selected.append(candidate_id)

    # Reserve space for theory-driven structures before filling the shortlist.
    add_rows(eligible, count=min(8, maximum))
    # Mandatory structures ensure the user's specific hypotheses are not removed
    # by the coarse Day 6 screening stage.
    mandatory_filters = [
        eligible["p"].eq(6) & eligible["ModelForm"].eq("ARIMA"),
        eligible["p"].eq(12) & eligible["ModelForm"].eq("ARIMA"),
        eligible["P"].eq(1),
        eligible["P"].eq(2),
        eligible["Q"].ge(1),
        eligible["d"].eq(1),
        eligible["D"].eq(1),
    ]
    for mask in mandatory_filters:
        subset = eligible.loc[mask]
        if not subset.empty:
            add_rows(subset, count=1)

    # Fill any remaining slots with the best validation candidates.
    add_rows(eligible, count=maximum)
    return selected[:maximum]


def choose_arimax_response_specs(
    eligible_arima: pd.DataFrame,
    spec_by_id: dict[str, StatisticalSpec],
    maximum: int = 7,
) -> list[StatisticalSpec]:
    """
    Select a diverse response-order shortlist for ARIMAX rolling validation.

    ARIMAX order tuning is not reduced to one previously selected ARIMA order.
    The shortlist retains the strongest overall candidate plus representative
    non-seasonal d=0, non-seasonal d=1, seasonal, seasonal-AR(12), and seasonal-MA
    structures when those structures passed the ARIMA validation quality rules.
    """
    ordered = eligible_arima.sort_values(
        ["MacroFoldMAE", "ValidationMAE", "MacroFoldRMSE", "Complexity"]
    )
    selected_ids: list[str] = []

    def add_best(frame: pd.DataFrame, count: int = 1) -> None:
        for candidate_id in frame.head(count)["CandidateID"].astype(str):
            if candidate_id not in selected_ids:
                selected_ids.append(candidate_id)

    add_best(ordered, count=2)
    add_best(ordered.loc[ordered["ModelForm"].eq("ARIMA") & ordered["d"].eq(0)])
    add_best(ordered.loc[ordered["ModelForm"].eq("ARIMA") & ordered["d"].eq(1)])
    add_best(ordered.loc[ordered["ModelForm"].eq("SARIMA")])
    add_best(ordered.loc[ordered["P"].ge(2)])
    add_best(ordered.loc[ordered["Q"].ge(1)])

    # Retain a simple level-plus-exogenous reference if it survived the broad
    # tuning grid. This helps identify whether exogenous lags add value without
    # an unnecessarily complex response process.
    if "ARIMA(0,0,0)" in spec_by_id and "ARIMA(0,0,0)" not in selected_ids:
        selected_ids.append("ARIMA(0,0,0)")

    return [spec_by_id[candidate_id] for candidate_id in selected_ids[:maximum]]


def run_worker_process(
    command: list[str],
    result_path: Path,
    timeout_seconds: int,
) -> tuple[str, object]:
    """Run one isolated worker and return its pickled result or failure status."""
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, stderr_text = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        return "timeout", "worker_timeout"

    if process.returncode != 0:
        return "error", (stderr_text or "worker_failed").strip()[-1000:]
    try:
        with result_path.open("rb") as input_file:
            return pickle.load(input_file)
    except Exception as exc:
        return "error", f"result_read_error:{type(exc).__name__}:{exc}"


def external_arima_evaluation(
    spec: StatisticalSpec,
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate one rolling ARIMA/SARIMA candidate in an isolated executable."""
    file_descriptor, result_name = tempfile.mkstemp(prefix="mda_arima_worker_", suffix=".pkl")
    os.close(file_descriptor)
    result_path = Path(result_name)
    result_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "04_statistical_worker.py"),
        "--kind", "arima",
        "--order", ",".join(map(str, spec.order)),
        "--seasonal-order", ",".join(map(str, spec.seasonal_order)),
        "--target-days", ",".join(map(str, target_days)),
        "--split", split,
        "--output", str(result_path),
    ]
    status, payload = run_worker_process(command, result_path, timeout_seconds=30)
    result_path.unlink(missing_ok=True)
    if status == "ok":
        summary, frame = payload
        summary.setdefault("EvaluationStatus", "completed")
        return summary, frame
    return failed_arima_evaluation(
        spec,
        participant_series,
        targets_by_key,
        target_days,
        split,
        f"{status}:{payload}",
    )


def external_arimax_evaluation(
    spec: StatisticalSpec,
    exogenous_set: str,
    pattern: LagPattern,
    horizon: int,
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate one rolling ARIMAX configuration in an isolated executable."""
    file_descriptor, result_name = tempfile.mkstemp(prefix="mda_arimax_worker_", suffix=".pkl")
    os.close(file_descriptor)
    result_path = Path(result_name)
    result_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "04_statistical_worker.py"),
        "--kind", "arimax",
        "--order", ",".join(map(str, spec.order)),
        "--seasonal-order", ",".join(map(str, spec.seasonal_order)),
        "--target-days", ",".join(map(str, target_days)),
        "--split", split,
        "--output", str(result_path),
        "--exogenous-set", exogenous_set,
        "--lag-pattern-json", json.dumps(asdict(pattern)),
        "--horizon", str(horizon),
    ]
    status, payload = run_worker_process(command, result_path, timeout_seconds=75)
    result_path.unlink(missing_ok=True)
    if status == "ok":
        summary, frame = payload
        summary.setdefault("EvaluationStatus", "completed")
        return summary, frame
    return failed_arimax_evaluation(
        spec,
        exogenous_set,
        pattern,
        horizon,
        participant_series,
        targets_by_key,
        target_days,
        split,
        f"{status}:{payload}",
    )


def run_parallel_arima_evaluation(
    specs: Sequence[StatisticalSpec],
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate every statistical candidate in an isolated OS process."""
    with futures.ThreadPoolExecutor(max_workers=max(1, n_jobs)) as executor:
        submitted = [
            executor.submit(
                external_arima_evaluation,
                spec,
                participant_series,
                targets_by_key,
                target_days,
                split,
            )
            for spec in specs
        ]
        results = []
        for index, completed in enumerate(futures.as_completed(submitted), start=1):
            results.append(completed.result())
            if index % 10 == 0 or index == len(specs):
                print(
                    f"  completed isolated ARIMA/SARIMA candidate "
                    f"{index}/{len(specs)}"
                )
    summaries = pd.DataFrame([summary for summary, _ in results])
    predictions = {summary["CandidateID"]: frame for summary, frame in results}
    return summaries, predictions


# -----------------------------------------------------------------------------
# 7. Direct horizon-specific ARIMAX lag construction and tuning
# -----------------------------------------------------------------------------


def exposure_columns(exogenous_set: str) -> list[str]:
    """Return the overall exposure variables in an ARIMAX candidate."""
    try:
        return EXPOSURE_SET_COLUMNS[exogenous_set]
    except KeyError as exc:
        raise ValueError(f"Unknown exogenous set: {exogenous_set}") from exc


def correlation_guided_lags(
    diagnostics: pd.DataFrame,
    exogenous_set: str,
    horizon: int,
    count: int,
) -> tuple[int, ...]:
    """Select high-absolute within-person lag correlations subject to causality."""
    predictors = exposure_columns(exogenous_set)
    subset = diagnostics.loc[
        diagnostics["Predictor"].isin(predictors)
        & diagnostics["Lag"].between(horizon, MAX_EXOG_LAG)
    ].copy()
    if subset.empty:
        return (horizon,)

    aggregated = (
        subset.assign(
            AbsoluteCorrelation=subset["WithinPersonCenteredPearson"].abs()
        )
        .groupby("Lag", as_index=False)["AbsoluteCorrelation"]
        .mean()
        .sort_values(["AbsoluteCorrelation", "Lag"], ascending=[False, True])
    )
    selected = sorted(aggregated["Lag"].head(count).astype(int).tolist())
    return tuple(selected) if selected else (horizon,)


def generate_lag_patterns(
    horizon: int,
    exogenous_set: str,
    diagnostics: pd.DataFrame,
) -> list[LagPattern]:
    """
    Create a broad causal lag-pattern grid for one forecast horizon.

    A target-relative lag must be at least the forecast horizon. For example,
    horizon 1 can use exposure lags 1-18, whereas horizon 6 can use only lags
    6-12 because exposure from the future target day is not known at forecast
    origin. Individual lags are tested first; combined and summary patterns then
    test whether short-, daily-, and two-day effects add predictive information.
    """
    raw_patterns: list[LagPattern] = []

    # Test every individually available lag up to two study days. This directly
    # answers whether lag 1, lag 2, lag 6, lag 12, or another lag is most useful.
    for lag in range(horizon, MAX_EXOG_LAG + 1):
        category = "recent" if lag <= horizon + 2 else (
            "daily" if lag == 6 else ("long" if lag == 12 else "intermediate")
        )
        raw_patterns.append(
            LagPattern(f"lag_{lag}", exact_lags=(lag,), category=category)
        )

    # Explicit combinations represent the hypotheses discussed with the user:
    # immediate + same-time previous day, immediate + two-day lag, and lag 6+12.
    combined_sets = [
        tuple(range(horizon, min(horizon + 3, MAX_EXOG_LAG + 1))),
        tuple(sorted(set([horizon, 6]))),
        tuple(sorted(set([horizon, 12]))),
        tuple(sorted(set([horizon, 18]))),
        (6, 12),
        (6, 12, 18),
        tuple(sorted(set([horizon, 6, 12]))),
        tuple(sorted(set([horizon, 6, 12, 18]))),
    ]
    for lags in combined_sets:
        if not lags or min(lags) < horizon or max(lags) > MAX_EXOG_LAG:
            continue
        raw_patterns.append(
            LagPattern(
                "lags_" + "_".join(map(str, lags)),
                exact_lags=lags,
                category="combined",
            )
        )

    # Summary windows capture cumulative exposure rather than assuming that one
    # exact lag is solely responsible for later AffWB.
    raw_patterns.extend(
        [
            LagPattern(
                f"mean_lags_{horizon}_{min(horizon + 5, MAX_EXOG_LAG)}",
                window_start=horizon,
                window_end=min(horizon + 5, MAX_EXOG_LAG),
                category="summary",
            ),
            LagPattern(
                f"mean_lags_{horizon}_{min(12, MAX_EXOG_LAG)}",
                window_start=horizon,
                window_end=min(12, MAX_EXOG_LAG),
                category="summary",
            ),
            LagPattern(
                f"mean_lags_{horizon}_{MAX_EXOG_LAG}",
                window_start=horizon,
                window_end=MAX_EXOG_LAG,
                category="summary",
            ),
        ]
    )

    for count in (2, 3):
        selected = correlation_guided_lags(
            diagnostics, exogenous_set, horizon, count
        )
        raw_patterns.append(
            LagPattern(
                "corr_top" + str(count) + "_" + "_".join(map(str, selected)),
                exact_lags=selected,
                category="correlation",
            )
        )

    # Deduplicate patterns that collapse to the same lag structure at later
    # horizons. Correlations only rank candidates; validation decides selection.
    unique: dict[tuple[tuple[int, ...], int | None, int | None], LagPattern] = {}
    for pattern in raw_patterns:
        if pattern.minimum_lag < horizon or pattern.maximum_lag > MAX_EXOG_LAG:
            continue
        key = (pattern.exact_lags, pattern.window_start, pattern.window_end)
        unique.setdefault(key, pattern)
    return list(unique.values())


def build_raw_exog_features(
    frame: pd.DataFrame,
    target_times: Sequence[int],
    exogenous_set: str,
    pattern: LagPattern,
) -> tuple[np.ndarray, list[str]]:
    """Construct target-relative lag features from known historical exposure."""
    columns = exposure_columns(exogenous_set)
    indexed = frame.set_index("Occasion")
    feature_rows: list[list[float]] = []
    feature_names: list[str] = []

    for predictor in columns:
        for lag in pattern.exact_lags:
            feature_names.append(f"{predictor}_lag_{lag}")
        if pattern.window_start is not None and pattern.window_end is not None:
            feature_names.append(
                f"{predictor}_mean_lags_{pattern.window_start}_{pattern.window_end}"
            )

    for target_time in target_times:
        row: list[float] = []
        for predictor in columns:
            for lag in pattern.exact_lags:
                source_time = int(target_time) - lag
                value = indexed[predictor].get(source_time, np.nan)
                row.append(np.log1p(value) if np.isfinite(value) else np.nan)

            if pattern.window_start is not None and pattern.window_end is not None:
                values = []
                for lag in range(pattern.window_start, pattern.window_end + 1):
                    source_time = int(target_time) - lag
                    value = indexed[predictor].get(source_time, np.nan)
                    if np.isfinite(value):
                        values.append(np.log1p(value))
                row.append(float(np.mean(values)) if values else np.nan)
        feature_rows.append(row)

    return np.asarray(feature_rows, dtype=float), feature_names


def prepare_arimax_design(
    frame: pd.DataFrame,
    history_end: int,
    horizon: int,
    exogenous_set: str,
    pattern: LagPattern,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build aligned response, historical exogenous, and known future exogenous rows.

    The response is AffWB at each target time. Exogenous features are constructed
    from target-relative lags. Because every candidate lag is at least the forecast
    horizon, all exogenous rows required for the h-step forecast are known at the
    forecast origin.
    """
    maximum_lag = pattern.maximum_lag
    history_target_times = list(range(maximum_lag + 1, history_end + 1))
    future_target_times = list(range(history_end + 1, history_end + horizon + 1))
    if not history_target_times:
        raise ValueError("No aligned historical target times remain after lagging.")

    indexed = frame.set_index("Occasion")
    y_history = np.asarray(
        [indexed["AffWB"].get(time, np.nan) for time in history_target_times],
        dtype=float,
    )
    raw_history, feature_names = build_raw_exog_features(
        frame, history_target_times, exogenous_set, pattern
    )
    raw_future, _ = build_raw_exog_features(
        frame, future_target_times, exogenous_set, pattern
    )

    # The imputer and scaler are fitted only on the historical design available at
    # the forecast origin. Missingness indicators preserve whether a value was
    # unavailable rather than a true zero-minute response.
    history_missing = ~np.isfinite(raw_history)
    future_missing = ~np.isfinite(raw_future)
    medians = np.nanmedian(raw_history, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    history_filled = np.where(history_missing, medians, raw_history)
    future_filled = np.where(future_missing, medians, raw_future)

    means = np.mean(history_filled, axis=0)
    standard_deviations = np.std(history_filled, axis=0)
    variable_columns = standard_deviations > 1e-8
    if not variable_columns.any():
        raise ValueError("All derived exogenous features are constant.")

    history_scaled = (
        history_filled[:, variable_columns] - means[variable_columns]
    ) / standard_deviations[variable_columns]
    future_scaled = (
        future_filled[:, variable_columns] - means[variable_columns]
    ) / standard_deviations[variable_columns]

    missing_variable = history_missing[:, variable_columns].astype(float)
    future_missing_variable = future_missing[:, variable_columns].astype(float)
    # Constant missingness columns provide no information and can cause singular
    # designs. Retain only missing indicators that vary in the training history.
    varying_missing = np.std(missing_variable, axis=0) > 1e-8

    exog_history_parts = [history_scaled]
    exog_future_parts = [future_scaled]
    final_names = [
        feature_names[index]
        for index, keep in enumerate(variable_columns)
        if keep
    ]
    if varying_missing.any():
        exog_history_parts.append(missing_variable[:, varying_missing])
        exog_future_parts.append(future_missing_variable[:, varying_missing])
        variable_feature_names = [
            feature_names[index]
            for index, keep in enumerate(variable_columns)
            if keep
        ]
        final_names.extend(
            [
                f"{name}__missing"
                for name, keep in zip(variable_feature_names, varying_missing)
                if keep
            ]
        )

    exog_history = np.concatenate(exog_history_parts, axis=1)
    exog_future = np.concatenate(exog_future_parts, axis=1)
    return y_history, exog_history, exog_future, final_names


def evaluate_arimax_configuration(
    spec: StatisticalSpec,
    exogenous_set: str,
    pattern: LagPattern,
    horizon: int,
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate one direct horizon-specific ARIMAX/SARIMAX-X configuration."""
    rows: list[dict[str, object]] = []
    fit_attempts = 0
    fit_successes = 0
    converged_fits = 0
    effective_feature_counts: list[int] = []
    configuration_id = (
        f"{spec.candidate_id}|{exogenous_set}|h{horizon}|{pattern.name}"
    )

    for target_day in target_days:
        history_end = (target_day - 1) * MEASUREMENTS_PER_DAY
        for participant_id, frame in participant_series.items():
            fit_attempts += 1
            try:
                y_history, exog_history, exog_future, feature_names = prepare_arimax_design(
                    frame=frame,
                    history_end=history_end,
                    horizon=horizon,
                    exogenous_set=exogenous_set,
                    pattern=pattern,
                )
                result = fit_sarimax_forecast(
                    y=y_history,
                    spec=spec,
                    steps=horizon,
                    exog_history=exog_history,
                    exog_future=exog_future,
                )
                effective_feature_counts.append(len(feature_names))
            except Exception:
                result = FitResult(
                    forecast=np.full(horizon, np.nan),
                    fit_success=False,
                    converged=False,
                    aic=np.nan,
                    error_type="design_error",
                )

            fit_successes += int(result.fit_success)
            converged_fits += int(result.fit_success and result.converged)
            prediction = result.forecast[-1] if len(result.forecast) else np.nan
            target_group = targets_by_key[(participant_id, target_day)]
            target_row = target_group.loc[target_group["Horizon"].eq(horizon)].iloc[0]
            rows.append(
                {
                    "Model": "ARIMAX",
                    "ConfigurationID": configuration_id,
                    "OriginID": target_row["OriginID"],
                    "ParticipantID": participant_id,
                    "TargetDay": target_day,
                    "Split": split,
                    "Horizon": horizon,
                    "TargetOccasion": int(target_row["TargetOccasion"]),
                    "Actual": target_row["AffWBTarget"],
                    "ActualObserved": int(target_row["AffWBTargetObserved"]),
                    "Prediction": float(prediction) if np.isfinite(prediction) else np.nan,
                }
            )

    prediction_frame = pd.DataFrame(rows)
    summary = summarize_candidate_predictions(
        prediction_frame, fit_attempts, fit_successes, converged_fits
    )
    p, d, q = spec.order
    P, D, Q, s = spec.seasonal_order
    summary.update(
        {
            "ConfigurationID": configuration_id,
            "ResponseCandidateID": spec.candidate_id,
            "ModelForm": "SARIMAX" if spec.is_seasonal else "ARIMAX",
            "p": p,
            "d": d,
            "q": q,
            "P": P,
            "D": D,
            "Q": Q,
            "SeasonalPeriod": s,
            "Horizon": horizon,
            "ExogenousSet": exogenous_set,
            "LagPattern": pattern.name,
            "LagCategory": pattern.category,
            "ExactLags": ",".join(map(str, pattern.exact_lags)),
            "WindowStart": pattern.window_start,
            "WindowEnd": pattern.window_end,
            "MinimumLag": pattern.minimum_lag,
            "MaximumLag": pattern.maximum_lag,
            "PatternComplexity": pattern.complexity,
            "ResponseComplexity": spec.complexity,
            "MedianEffectiveExogColumns": (
                float(np.median(effective_feature_counts))
                if effective_feature_counts
                else np.nan
            ),
        }
    )
    return summary, prediction_frame


def choose_arimax_stage2_rows(stage1_horizon: pd.DataFrame) -> pd.DataFrame:
    """Choose a broad, diverse lag/exposure shortlist after rolling screening."""
    eligible = stage1_horizon.loc[
        stage1_horizon["Coverage"].ge(0.65)
        & stage1_horizon["FitSuccessRate"].ge(0.60)
    ].copy()
    if eligible.empty:
        return stage1_horizon.sort_values(
            ["MacroFoldMAE", "ValidationMAE"]
        ).head(4)

    selected_ids: list[str] = []

    def add_best(frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        row = frame.sort_values(
            ["MacroFoldMAE", "ValidationMAE", "PatternComplexity"]
        ).iloc[0]
        candidate_id = str(row["ConfigurationID"])
        if candidate_id not in selected_ids:
            selected_ids.append(candidate_id)

    # Best overall and best candidate for each exposure definition.
    add_best(eligible)
    for exogenous_set in EXPOSURE_SET_COLUMNS:
        add_best(eligible.loc[eligible["ExogenousSet"].eq(exogenous_set)])

    # Preserve the immediate horizon, one-day, two-day, three-day, and combined
    # lag hypotheses even if their Day 5-6 screening ranks differ slightly.
    horizon = int(eligible["Horizon"].iloc[0])
    mandatory_patterns = {
        f"lag_{horizon}",
        "lag_6",
        "lag_12",
        "lag_18",
        "lags_6_12",
        "lags_6_12_18",
    }
    for pattern_name in mandatory_patterns:
        add_best(eligible.loc[eligible["LagPattern"].eq(pattern_name)])

    # Ensure cumulative and correlation-guided histories are represented.
    add_best(eligible.loc[eligible["LagCategory"].eq("summary")])
    add_best(eligible.loc[eligible["LagCategory"].eq("correlation")])
    add_best(eligible.loc[eligible["LagCategory"].eq("combined")])

    # Cap the joint stage to a reproducible, still diverse set. The cap is based
    # on validation rank only after the mandatory hypotheses above are retained.
    selected = eligible.loc[eligible["ConfigurationID"].isin(selected_ids)].copy()
    if len(selected) > 12:
        selected = selected.sort_values(
            ["MacroFoldMAE", "ValidationMAE", "PatternComplexity"]
        ).head(12)
    return selected


# -----------------------------------------------------------------------------
# 8. Baselines and test uncertainty
# -----------------------------------------------------------------------------


def baseline_forecast(y: np.ndarray, model_name: str) -> np.ndarray:
    """Generate persistence, participant-mean, or seasonal-naive forecasts."""
    y_array = np.asarray(y, dtype=float)
    observed = y_array[np.isfinite(y_array)]
    if len(observed) == 0:
        return np.full(MEASUREMENTS_PER_DAY, np.nan)

    if model_name == "Persistence":
        return np.repeat(observed[-1], MEASUREMENTS_PER_DAY)
    if model_name == "ParticipantMean":
        return np.repeat(float(np.mean(observed)), MEASUREMENTS_PER_DAY)
    if model_name == "SeasonalNaive6":
        if len(y_array) < MEASUREMENTS_PER_DAY:
            return np.full(MEASUREMENTS_PER_DAY, np.nan)
        return y_array[-MEASUREMENTS_PER_DAY:].astype(float)
    raise ValueError(f"Unknown baseline: {model_name}")


def evaluate_baselines(
    participant_series: dict[int, pd.DataFrame],
    targets_by_key: dict[tuple[int, int], pd.DataFrame],
    target_days: Sequence[int],
    split: str,
) -> pd.DataFrame:
    """Evaluate transparent non-fitted baselines on chronological forecast origins."""
    rows: list[dict[str, object]] = []
    for target_day in target_days:
        history_end = (target_day - 1) * MEASUREMENTS_PER_DAY
        for participant_id, frame in participant_series.items():
            history = extract_history(frame, history_end, ["AffWB"])
            y = history["AffWB"].to_numpy(dtype=float)
            for model_name in ["Persistence", "SeasonalNaive6", "ParticipantMean"]:
                rows.extend(
                    prediction_rows(
                        participant_id=participant_id,
                        target_day=target_day,
                        split=split,
                        model_name=model_name,
                        predictions=baseline_forecast(y, model_name),
                        target_group=targets_by_key[(participant_id, target_day)],
                        configuration_id=model_name,
                    )
                )
    return pd.DataFrame(rows)


def participant_bootstrap_metrics(
    predictions: pd.DataFrame,
    iterations: int = 2000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Estimate participant-clustered confidence intervals for test MAE."""
    test = predictions.loc[predictions["Split"].eq("test")].copy()
    participants = sorted(test["ParticipantID"].unique().tolist())
    random = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for model_name, model_group in test.groupby("Model"):
        by_participant = {
            int(participant_id): group
            for participant_id, group in model_group.groupby("ParticipantID")
        }
        bootstrap_mae: list[float] = []
        for _ in range(iterations):
            sampled = random.choice(participants, size=len(participants), replace=True)
            sampled_frames = [by_participant[int(pid)] for pid in sampled if int(pid) in by_participant]
            if not sampled_frames:
                continue
            metrics = calculate_metrics(pd.concat(sampled_frames, ignore_index=True))
            if np.isfinite(metrics["MAE"]):
                bootstrap_mae.append(metrics["MAE"])
        rows.append(
            {
                "Model": model_name,
                "Iterations": len(bootstrap_mae),
                "BootstrapMeanMAE": float(np.mean(bootstrap_mae)) if bootstrap_mae else np.nan,
                "MAE_CI_Lower_2_5": float(np.quantile(bootstrap_mae, 0.025)) if bootstrap_mae else np.nan,
                "MAE_CI_Upper_97_5": float(np.quantile(bootstrap_mae, 0.975)) if bootstrap_mae else np.nan,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 9. Figures, reports, and manifests
# -----------------------------------------------------------------------------


def plot_acf_pacf(diagnostics: pd.DataFrame, path: Path) -> None:
    """Plot mean individual ACF/PACF and within-person lag correlation."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(diagnostics["Lag"], diagnostics["MeanIndividualACF"], marker="o", label="Mean individual ACF")
    ax.plot(diagnostics["Lag"], diagnostics["MeanIndividualPACF"], marker="o", label="Mean individual PACF")
    ax.plot(
        diagnostics["Lag"],
        diagnostics["WithinPersonCenteredPearson"],
        marker="o",
        label="Observed-pair within-person correlation",
    )
    ax.axhline(0, linewidth=1)
    ax.set_title("AffWB Temporal Diagnostics by Lag")
    ax.set_xlabel("Lag in ESM positions")
    ax.set_ylabel("Correlation")
    ax.set_xticks(range(1, MAX_DIAGNOSTIC_LAG + 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_exogenous_lags(diagnostics: pd.DataFrame, path: Path) -> None:
    """Plot within-person exposure-to-future-AffWB lag correlations."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for predictor, group in diagnostics.groupby("Predictor"):
        ax.plot(
            group["Lag"],
            group["WithinPersonCenteredPearson"],
            marker="o",
            label=predictor,
        )
    ax.axhline(0, linewidth=1)
    ax.set_title("Lagged Social-Media Exposure vs Later AffWB")
    ax.set_xlabel("Predictor lag in ESM positions")
    ax.set_ylabel("Within-person centered Pearson correlation")
    ax.set_xticks(range(1, MAX_EXOG_LAG + 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_top_arima_validation(tuning: pd.DataFrame, path: Path) -> None:
    """Plot eligible rolling-validation ARIMA/SARIMA configurations."""
    if "EligibleForSelection" in tuning.columns:
        eligible = tuning.loc[tuning["EligibleForSelection"]].copy()
    else:
        eligible = tuning.loc[
            tuning["Coverage"].ge(0.85)
            & tuning["MinimumFoldCoverage"].ge(0.75)
            & tuning["FitSuccessRate"].ge(0.75)
            & tuning["ConvergenceRateAmongSuccessful"].ge(0.25)
        ].copy()
    ordered = eligible.sort_values(["MacroFoldMAE", "ValidationMAE"]).head(24).copy()
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.barh(ordered["CandidateID"], ordered["MacroFoldMAE"])
    ax.invert_yaxis()
    ax.set_title("Top ARIMA/SARIMA Rolling-Validation Configurations")
    ax.set_xlabel("Macro validation MAE across target Days 4-6")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_arimax_selected_lags(selected: pd.DataFrame, path: Path) -> None:
    """Plot the validation MAE of the selected direct ARIMAX configuration by horizon."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(selected["Horizon"], selected["MacroFoldMAE"], marker="o")
    for _, row in selected.iterrows():
        ax.annotate(
            f"{row['LagPattern']}\n{row['ExogenousSet']}",
            (row["Horizon"], row["MacroFoldMAE"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_title("Selected Direct ARIMAX Lag Structure by Forecast Horizon")
    ax.set_xlabel("Forecast horizon within next day")
    ax.set_ylabel("Macro validation MAE across target Days 5-6")
    ax.set_xticks(range(1, 7))
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_test_horizon_metrics(horizon_metrics: pd.DataFrame, path: Path) -> None:
    """Plot Day 7 MAE by horizon for fitted models and baselines."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, group in horizon_metrics.groupby("Model"):
        ax.plot(group["Horizon"], group["MAE"], marker="o", label=model_name)
    ax.set_title("Revised Statistical Models: Day 7 Test MAE by Horizon")
    ax.set_xlabel("Forecast horizon within next day")
    ax.set_ylabel("MAE")
    ax.set_xticks(range(1, 7))
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_test_actual_predicted(predictions: pd.DataFrame, path: Path) -> None:
    """Plot actual and predicted Day 7 AffWB for ARIMA and ARIMAX."""
    subset = predictions.loc[
        predictions["Split"].eq("test")
        & predictions["Model"].isin(["ARIMA", "ARIMAX"])
        & predictions["ActualObserved"].eq(1)
    ].copy()
    fig, ax = plt.subplots(figsize=(7, 7))
    for model_name, group in subset.groupby("Model"):
        ax.scatter(group["Actual"], group["Prediction"], alpha=0.45, label=model_name)
    ax.plot([1, 7], [1, 7], linestyle="--", linewidth=1)
    ax.set_xlim(0.8, 7.2)
    ax.set_ylim(0.8, 7.2)
    ax.set_title("Day 7 Actual vs Predicted AffWB")
    ax.set_xlabel("Actual AffWB")
    ax.set_ylabel("Predicted AffWB")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_runtime_environment(path: Path, n_jobs: int, quick: bool) -> None:
    """Record software versions and execution controls."""
    import joblib
    import scipy
    import statsmodels

    pd.DataFrame(
        [
            {"Item": "Python", "Value": platform.python_version()},
            {"Item": "Platform", "Value": platform.platform()},
            {"Item": "numpy", "Value": np.__version__},
            {"Item": "pandas", "Value": pd.__version__},
            {"Item": "scipy", "Value": scipy.__version__},
            {"Item": "statsmodels", "Value": statsmodels.__version__},
            {"Item": "joblib", "Value": joblib.__version__},
            {"Item": "n_jobs", "Value": n_jobs},
            {"Item": "quick_mode", "Value": quick},
        ]
    ).to_csv(path, index=False)


def build_output_manifest(paths: Iterable[Path], project_root: Path, destination: Path) -> None:
    """Write checksums for all Step 04 outputs."""
    rows = []
    for path in sorted(set(paths)):
        if path.exists() and path.is_file():
            rows.append(
                {
                    "File": str(path.relative_to(project_root)),
                    "Bytes": path.stat().st_size,
                    "SHA256": file_sha256(path),
                }
            )
    pd.DataFrame(rows).to_csv(destination, index=False)


def write_report(
    path: Path,
    selected_arima: pd.Series,
    selected_arimax: pd.DataFrame,
    summary: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    acf_diagnostics: pd.DataFrame,
    exog_diagnostics: pd.DataFrame,
    arima_tuning: pd.DataFrame,
    arimax_tuning: pd.DataFrame,
) -> None:
    """Write an English report of the revised statistical-model methodology and results."""
    test_summary = summary.loc[summary["Split"].eq("test")].set_index("Model")
    daily_test = daily_metrics.loc[daily_metrics["Split"].eq("test")].set_index("Model")

    diagnostic_lines = []
    for lag in [1, 6, 12, 18]:
        row = acf_diagnostics.loc[acf_diagnostics["Lag"].eq(lag)].iloc[0]
        diagnostic_lines.append(
            f"- Lag {lag}: observed-pair within-person correlation="
            f"{row['WithinPersonCenteredPearson']:.4f}, mean individual ACF="
            f"{row['MeanIndividualACF']:.4f}, mean individual PACF="
            f"{row['MeanIndividualPACF']:.4f}."
        )

    exog_lines = []
    for predictor, group in exog_diagnostics.groupby("Predictor"):
        best = group.assign(
            AbsoluteCorrelation=group["WithinPersonCenteredPearson"].abs()
        ).sort_values(["AbsoluteCorrelation", "Lag"], ascending=[False, True]).head(5)
        lag_text = ", ".join(
            f"lag {int(row['Lag'])} ({row['WithinPersonCenteredPearson']:.4f})"
            for _, row in best.iterrows()
        )
        exog_lines.append(f"- {predictor}: strongest development-data lags were {lag_text}.")

    selected_lines = []
    for _, row in selected_arimax.sort_values("Horizon").iterrows():
        selected_lines.append(
            f"- h={int(row['Horizon'])}: {row['ResponseCandidateID']}; "
            f"exposure={row['ExogenousSet']}; pattern={row['LagPattern']}; "
            f"rolling-validation macro MAE={row['MacroFoldMAE']:.4f}."
        )

    report_lines = [
        "# Step 04 Revised ARIMA/ARIMAX Statistical Baseline Report",
        "",
        "## Forecasting task",
        "",
        "The models forecast the six AffWB measurements of the next planned ESM day. "
        "The six-position trajectory is the primary output, horizon 1 is the immediately next measurement, "
        "and the next-day mean is derived from the same observed forecast positions.",
        "",
        "## Methodological revision",
        "",
        f"- {len(arima_tuning)} ARIMA/SARIMA response structures were evaluated with rolling validation on target Days 4-6.",
        "- The non-seasonal search included pure AR orders through p=12, MA orders through q=6, ordinary differencing d=0/1, and bounded mixed structures.",
        "- Seasonal models used period 6 and included seasonal AR, MA, and differencing terms, including a P=2 lag-12 structure.",
        f"- {len(arimax_tuning)} joint horizon-specific ARIMAX/SARIMAX-X configurations were evaluated after the broad rolling lag screen.",
        "- Social-media lags were considered through lag 18, including individual lags, lag combinations, cumulative windows, and correlation-guided patterns.",
        "- ARIMAX lag patterns were tuned separately for horizons 1-6. Only predictor values known before the forecasted day were allowed.",
        "- Day 7 was not used for response-order, predictor, lag, preprocessing, or convergence decisions.",
        "",
        "## AffWB temporal diagnostics",
        "",
        *diagnostic_lines,
        "",
        "## Social-media lag diagnostics",
        "",
        *exog_lines,
        "",
        "These correlations are descriptive screening information. Rolling validation, not correlation magnitude alone, selected the final lag patterns.",
        "",
        "## Selected ARIMA/SARIMA baseline",
        "",
        f"- Configuration: {selected_arima['CandidateID']}.",
        f"- Rolling-validation macro MAE: {selected_arima['MacroFoldMAE']:.4f}.",
        f"- Rolling-validation pooled MAE: {selected_arima['ValidationMAE']:.4f}.",
        f"- Validation coverage: {selected_arima['Coverage']:.4f}.",
        f"- Day 7 test MAE: {test_summary.loc['ARIMA', 'MAE']:.4f}.",
        f"- Day 7 test RMSE: {test_summary.loc['ARIMA', 'RMSE']:.4f}.",
        "",
        "## Selected direct ARIMAX/SARIMAX-X configurations",
        "",
        *selected_lines,
        "",
        f"- Combined ARIMAX Day 7 test MAE: {test_summary.loc['ARIMAX', 'MAE']:.4f}.",
        f"- Combined ARIMAX Day 7 test RMSE: {test_summary.loc['ARIMAX', 'RMSE']:.4f}.",
        "",
        "## Transparent reference forecasts",
        "",
    ]
    for model_name in ["Persistence", "SeasonalNaive6", "ParticipantMean"]:
        if model_name in test_summary.index:
            report_lines.append(
                f"- {model_name}: Day 7 MAE={test_summary.loc[model_name, 'MAE']:.4f}, "
                f"RMSE={test_summary.loc[model_name, 'RMSE']:.4f}, "
                f"coverage={test_summary.loc[model_name, 'Coverage']:.4f}."
            )

    report_lines.extend(["", "## Derived next-day mean", ""])
    for model_name in ["ARIMA", "ARIMAX", "Persistence", "SeasonalNaive6", "ParticipantMean"]:
        if model_name in daily_test.index:
            report_lines.append(
                f"- {model_name}: MAE={daily_test.loc[model_name, 'MAE']:.4f}, "
                f"RMSE={daily_test.loc[model_name, 'RMSE']:.4f}, "
                f"participant-days={int(daily_test.loc[model_name, 'ParticipantDays'])}."
            )

    report_lines.extend(["", "## Participant-clustered bootstrap", ""])
    for _, row in bootstrap.iterrows():
        report_lines.append(
            f"- {row['Model']}: Day 7 MAE 95% interval "
            f"[{row['MAE_CI_Lower_2_5']:.4f}, {row['MAE_CI_Upper_97_5']:.4f}]."
        )

    report_lines.extend(
        [
            "",
            "## Interpretation limitations",
            "",
            "- The ESM positions are ordered, but exact timestamps and equal elapsed intervals are unavailable.",
            "- Period-six seasonality represents the planned six-position daily structure, not exact clock-time seasonality.",
            "- Participant histories are short; high-complexity candidates may fail or have poor coverage and are excluded from selection by prespecified quality thresholds.",
            "- Lag selection is predictive rather than causal. A selected social-media lag does not establish that exposure caused later changes in AffWB.",
        ]
    )
    path.write_text("\n".join(report_lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# 10. Main execution
# -----------------------------------------------------------------------------


def main() -> None:
    """Run revised diagnostics, staged tuning, and untouched Day 7 testing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modeling-data",
        type=Path,
        default=Path("outputs/data/03_modeling_long_data.csv"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("outputs/tables/03_next_day_target_trajectory.csv"),
    )
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced candidate set for a fast code check, not final results.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    directories = create_directories(project_root)
    modeling_path = (
        args.modeling_data
        if args.modeling_data.is_absolute()
        else project_root / args.modeling_data
    )
    target_path = args.targets if args.targets.is_absolute() else project_root / args.targets

    modeling, targets = load_inputs(modeling_path, target_path)
    participant_series = build_participant_series(modeling)
    targets_by_key = target_lookup(targets)
    output_paths: list[Path] = []

    # ------------------------------------------------------------------
    # A. Temporal and exogenous-lag diagnostics
    # ------------------------------------------------------------------
    acf_diagnostics = build_affwb_acf_pacf_diagnostics(participant_series)
    exog_diagnostics = build_exogenous_lag_diagnostics(participant_series)
    acf_path = directories["tables"] / "04_affwb_acf_pacf_diagnostics.csv"
    exog_diagnostic_path = directories["tables"] / "04_exogenous_lag_diagnostics.csv"
    acf_diagnostics.to_csv(acf_path, index=False)
    exog_diagnostics.to_csv(exog_diagnostic_path, index=False)
    output_paths.extend([acf_path, exog_diagnostic_path])

    # ------------------------------------------------------------------
    # B. Broad ARIMA/SARIMA Stage 1 screen on validation Day 6
    # ------------------------------------------------------------------
    all_specs = generate_statistical_candidates(quick=args.quick)
    candidate_grid_path = directories["tables"] / "04_statistical_candidate_grid.csv"
    pd.DataFrame(
        [
            {
                "CandidateID": spec.candidate_id,
                "ModelForm": "SARIMA" if spec.is_seasonal else "ARIMA",
                "p": spec.order[0],
                "d": spec.order[1],
                "q": spec.order[2],
                "P": spec.seasonal_order[0],
                "D": spec.seasonal_order[1],
                "Q": spec.seasonal_order[2],
                "SeasonalPeriod": spec.seasonal_order[3],
                "Order": str(spec.order),
                "SeasonalOrder": str(spec.seasonal_order),
                "Complexity": spec.complexity,
            }
            for spec in all_specs
        ]
    ).to_csv(candidate_grid_path, index=False)
    output_paths.append(candidate_grid_path)

    print(f"Stage 1 ARIMA/SARIMA screen: {len(all_specs)} candidates on Day 6")
    stage1_arima, _ = run_parallel_arima_evaluation(
        specs=all_specs,
        participant_series=participant_series,
        targets_by_key=targets_by_key,
        target_days=[6],
        split="validation_screen",
        n_jobs=max(1, args.n_jobs),
    )
    stage1_arima_path = directories["tables"] / "04_arima_stage1_screening.csv"
    stage1_arima.sort_values(["ValidationMAE", "ValidationRMSE"]).to_csv(
        stage1_arima_path, index=False
    )
    output_paths.append(stage1_arima_path)

    # ------------------------------------------------------------------
    # C. Rolling ARIMA/SARIMA validation on Days 4-6
    # ------------------------------------------------------------------
    spec_by_id = {spec.candidate_id: spec for spec in all_specs}
    if args.quick:
        stage2_ids = choose_arima_stage2_candidates(stage1_arima, maximum=14)
        stage2_specs = [spec_by_id[candidate_id] for candidate_id in stage2_ids]
    else:
        # The final analysis evaluates the complete candidate grid on all rolling
        # folds. This prevents a configuration with uneven fold performance from
        # being discarded solely because of the preliminary Day 6 screen.
        stage2_specs = all_specs
    print(
        f"Stage 2 ARIMA/SARIMA rolling validation: {len(stage2_specs)} candidates "
        f"on Days {ARIMA_VALIDATION_DAYS}"
    )
    arima_tuning, arima_validation_predictions = run_parallel_arima_evaluation(
        specs=stage2_specs,
        participant_series=participant_series,
        targets_by_key=targets_by_key,
        target_days=ARIMA_VALIDATION_DAYS,
        split="validation",
        n_jobs=max(1, args.n_jobs),
    )
    arima_tuning["EligibleForSelection"] = (
        arima_tuning["Coverage"].ge(0.85)
        & arima_tuning["MinimumFoldCoverage"].ge(0.75)
        & arima_tuning["FitSuccessRate"].ge(0.75)
        & arima_tuning["ConvergenceRateAmongSuccessful"].ge(0.25)
    )
    arima_tuning = arima_tuning.sort_values(
        ["EligibleForSelection", "MacroFoldMAE", "ValidationMAE", "MacroFoldRMSE", "Complexity"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    arima_tuning_path = directories["tables"] / "04_arima_validation_tuning.csv"
    arima_tuning.to_csv(arima_tuning_path, index=False)
    output_paths.append(arima_tuning_path)

    eligible_arima = arima_tuning.loc[arima_tuning["EligibleForSelection"]].copy()
    if eligible_arima.empty:
        raise RuntimeError("No ARIMA/SARIMA candidate passed rolling-validation quality thresholds.")
    selected_arima_row = eligible_arima.iloc[0]
    selected_arima_spec = spec_by_id[str(selected_arima_row["CandidateID"])]
    selected_arima_validation = arima_validation_predictions[
        selected_arima_spec.candidate_id
    ]

    # ARIMAX receives its own response-order tuning shortlist. The shortlist is
    # derived from broad rolling ARIMA/SARIMA validation but remains diverse in
    # differencing, non-seasonal, and seasonal structures.
    order_shortlist = choose_arimax_response_specs(
        eligible_arima, spec_by_id, maximum=5 if args.quick else 7
    )

    # ------------------------------------------------------------------
    # D. ARIMAX Stage 1 rolling lag/exposure screening on Days 5-6
    # ------------------------------------------------------------------
    # The selected univariate response structure is used for the broad lag
    # screen. Stage 2 then retunes the retained lag patterns across a diverse
    # response-order shortlist, allowing the response dynamics to change after
    # exogenous information is added.
    best_response_spec = selected_arima_spec
    arimax_stage1_jobs = []
    for horizon in range(1, MEASUREMENTS_PER_DAY + 1):
        for exogenous_set in EXPOSURE_SET_COLUMNS:
            for pattern in generate_lag_patterns(
                horizon, exogenous_set, exog_diagnostics
            ):
                arimax_stage1_jobs.append(
                    (best_response_spec, exogenous_set, pattern, horizon)
                )

    print(
        f"Stage 1 ARIMAX rolling lag screen: {len(arimax_stage1_jobs)} configurations "
        f"on Days {ARIMAX_VALIDATION_DAYS}"
    )
    # Each lag screen is isolated in its own worker process. This avoids
    # numerical state leaking between hundreds of short state-space fits.
    with futures.ThreadPoolExecutor(max_workers=max(1, args.n_jobs)) as executor:
        submitted = [
            executor.submit(
                external_arimax_evaluation,
                spec,
                exogenous_set,
                pattern,
                horizon,
                participant_series,
                targets_by_key,
                ARIMAX_VALIDATION_DAYS,
                "validation_screen",
            )
            for spec, exogenous_set, pattern, horizon in arimax_stage1_jobs
        ]
        arimax_stage1_results = []
        for index, completed in enumerate(futures.as_completed(submitted), start=1):
            arimax_stage1_results.append(completed.result())
            if index % 25 == 0 or index == len(arimax_stage1_jobs):
                print(
                    f"  completed isolated ARIMAX screening configuration "
                    f"{index}/{len(arimax_stage1_jobs)}"
                )
    arimax_stage1 = pd.DataFrame([summary for summary, _ in arimax_stage1_results])
    arimax_stage1_path = directories["tables"] / "04_arimax_stage1_screening.csv"
    arimax_stage1.sort_values(["Horizon", "ValidationMAE"]).to_csv(
        arimax_stage1_path, index=False
    )
    output_paths.append(arimax_stage1_path)

    # ------------------------------------------------------------------
    # E. Direct horizon-specific ARIMAX rolling validation on Days 5-6
    # ------------------------------------------------------------------
    stage2_arimax_jobs = []
    stage1_lookup: dict[str, tuple[str, LagPattern, int]] = {}
    for horizon in range(1, MEASUREMENTS_PER_DAY + 1):
        selected_stage1_rows = choose_arimax_stage2_rows(
            arimax_stage1.loc[arimax_stage1["Horizon"].eq(horizon)]
        )
        for _, row in selected_stage1_rows.iterrows():
            exogenous_set = str(row["ExogenousSet"])
            matching_patterns = generate_lag_patterns(
                horizon, exogenous_set, exog_diagnostics
            )
            pattern = next(
                pattern
                for pattern in matching_patterns
                if pattern.name == str(row["LagPattern"])
            )
            for response_spec in order_shortlist:
                stage2_arimax_jobs.append(
                    (response_spec, exogenous_set, pattern, horizon)
                )

    # Remove duplicate jobs that arise when the selected and opposite structure
    # are identical after candidate deduplication.
    unique_jobs = {}
    for spec, exogenous_set, pattern, horizon in stage2_arimax_jobs:
        key = (spec.candidate_id, exogenous_set, pattern.name, horizon)
        unique_jobs[key] = (spec, exogenous_set, pattern, horizon)
    stage2_arimax_jobs = list(unique_jobs.values())

    print(
        f"Stage 2 ARIMAX rolling validation: {len(stage2_arimax_jobs)} configurations "
        f"on Days {ARIMAX_VALIDATION_DAYS}"
    )
    with futures.ThreadPoolExecutor(max_workers=max(1, args.n_jobs)) as executor:
        submitted = [
            executor.submit(
                external_arimax_evaluation,
                spec,
                exogenous_set,
                pattern,
                horizon,
                participant_series,
                targets_by_key,
                ARIMAX_VALIDATION_DAYS,
                "validation",
            )
            for spec, exogenous_set, pattern, horizon in stage2_arimax_jobs
        ]
        arimax_stage2_results = []
        for index, completed in enumerate(futures.as_completed(submitted), start=1):
            arimax_stage2_results.append(completed.result())
            if index % 10 == 0 or index == len(stage2_arimax_jobs):
                print(
                    f"  completed isolated ARIMAX rolling configuration "
                    f"{index}/{len(stage2_arimax_jobs)}"
                )
    arimax_tuning = pd.DataFrame([summary for summary, _ in arimax_stage2_results])
    arimax_prediction_lookup = {
        summary["ConfigurationID"]: frame
        for summary, frame in arimax_stage2_results
    }
    arimax_tuning["EligibleForSelection"] = (
        arimax_tuning["Coverage"].ge(0.75)
        & arimax_tuning["MinimumFoldCoverage"].ge(0.60)
        & arimax_tuning["FitSuccessRate"].ge(0.65)
    )
    arimax_tuning = arimax_tuning.sort_values(
        ["Horizon", "EligibleForSelection", "MacroFoldMAE", "ValidationMAE", "PatternComplexity", "ResponseComplexity"],
        ascending=[True, False, True, True, True, True],
    ).reset_index(drop=True)
    arimax_tuning_path = directories["tables"] / "04_arimax_validation_tuning.csv"
    arimax_tuning.to_csv(arimax_tuning_path, index=False)
    output_paths.append(arimax_tuning_path)

    selected_arimax_rows: list[pd.Series] = []
    selected_arimax_validation_frames: list[pd.DataFrame] = []
    for horizon in range(1, MEASUREMENTS_PER_DAY + 1):
        horizon_rows = arimax_tuning.loc[arimax_tuning["Horizon"].eq(horizon)].copy()
        eligible = horizon_rows.loc[horizon_rows["EligibleForSelection"]]
        if eligible.empty:
            eligible = horizon_rows
        selected = eligible.sort_values(
            ["MacroFoldMAE", "ValidationMAE", "PatternComplexity", "ResponseComplexity"]
        ).iloc[0]
        selected_arimax_rows.append(selected)
        selected_arimax_validation_frames.append(
            arimax_prediction_lookup[str(selected["ConfigurationID"])]
        )

    selected_arimax = pd.DataFrame(selected_arimax_rows).sort_values("Horizon")
    selected_arimax_path = directories["tables"] / "04_selected_arimax_horizon_configs.csv"
    selected_arimax.to_csv(selected_arimax_path, index=False)
    output_paths.append(selected_arimax_path)
    selected_arimax_validation = pd.concat(
        selected_arimax_validation_frames, ignore_index=True
    )

    # ------------------------------------------------------------------
    # F. Untouched Day 7 test
    # ------------------------------------------------------------------
    _, arima_test = external_arima_evaluation(
        selected_arima_spec,
        participant_series,
        targets_by_key,
        [TEST_DAY],
        "test",
    )

    arimax_test_frames: list[pd.DataFrame] = []
    for _, row in selected_arimax.iterrows():
        spec = spec_by_id[str(row["ResponseCandidateID"])]
        exogenous_set = str(row["ExogenousSet"])
        horizon = int(row["Horizon"])
        pattern = next(
            pattern
            for pattern in generate_lag_patterns(horizon, exogenous_set, exog_diagnostics)
            if pattern.name == str(row["LagPattern"])
        )
        _, prediction_frame = external_arimax_evaluation(
            spec,
            exogenous_set,
            pattern,
            horizon,
            participant_series,
            targets_by_key,
            [TEST_DAY],
            "test",
        )
        arimax_test_frames.append(prediction_frame)
    arimax_test = pd.concat(arimax_test_frames, ignore_index=True)

    validation_baselines = evaluate_baselines(
        participant_series,
        targets_by_key,
        ARIMA_VALIDATION_DAYS,
        "validation",
    )
    test_baselines = evaluate_baselines(
        participant_series,
        targets_by_key,
        [TEST_DAY],
        "test",
    )

    all_predictions = pd.concat(
        [
            selected_arima_validation,
            arima_test,
            selected_arimax_validation,
            arimax_test,
            validation_baselines,
            test_baselines,
        ],
        ignore_index=True,
    )

    predictions_path = directories["tables"] / "04_statistical_model_predictions.csv"
    all_predictions.to_csv(predictions_path, index=False)
    output_paths.append(predictions_path)

    # ------------------------------------------------------------------
    # G. Summary, horizon, daily-mean, and bootstrap outputs
    # ------------------------------------------------------------------
    summary_rows = []
    horizon_frames = []
    for (model_name, split), group in all_predictions.groupby(["Model", "Split"]):
        summary_rows.append({"Model": model_name, "Split": split, **calculate_metrics(group)})
        horizon = per_horizon_metrics(group)
        horizon.insert(0, "Split", split)
        horizon.insert(0, "Model", model_name)
        horizon_frames.append(horizon)

    summary = pd.DataFrame(summary_rows)
    horizon_metrics = pd.concat(horizon_frames, ignore_index=True)
    summary_path = directories["tables"] / "04_statistical_model_metrics.csv"
    horizon_path = directories["tables"] / "04_statistical_model_horizon_metrics.csv"
    summary.to_csv(summary_path, index=False)
    horizon_metrics.to_csv(horizon_path, index=False)
    output_paths.extend([summary_path, horizon_path])

    daily_predictions = derived_daily_mean_predictions(all_predictions)
    daily_metrics = daily_mean_metrics(daily_predictions)
    daily_predictions_path = directories["tables"] / "04_statistical_daily_mean_predictions.csv"
    daily_metrics_path = directories["tables"] / "04_statistical_daily_mean_metrics.csv"
    daily_predictions.to_csv(daily_predictions_path, index=False)
    daily_metrics.to_csv(daily_metrics_path, index=False)
    output_paths.extend([daily_predictions_path, daily_metrics_path])

    bootstrap = participant_bootstrap_metrics(all_predictions)
    bootstrap_path = directories["tables"] / "04_statistical_test_bootstrap_mae.csv"
    bootstrap.to_csv(bootstrap_path, index=False)
    output_paths.append(bootstrap_path)

    # ------------------------------------------------------------------
    # H. Configuration, figures, report, and reproducibility files
    # ------------------------------------------------------------------
    selected_config = {
        "forecast_horizon": 6,
        "arima_validation_days": list(ARIMA_VALIDATION_DAYS),
        "arimax_validation_days": list(ARIMAX_VALIDATION_DAYS),
        "test_day": TEST_DAY,
        "ARIMA": {
            "candidate_id": selected_arima_spec.candidate_id,
            "order": list(selected_arima_spec.order),
            "seasonal_order": list(selected_arima_spec.seasonal_order),
            "trend": trend_for_spec(selected_arima_spec),
            "macro_validation_mae": float(selected_arima_row["MacroFoldMAE"]),
        },
        "ARIMAX": {
            "strategy": "six direct horizon-specific SARIMAX-X models",
            "horizon_configurations": [
                {
                    "horizon": int(row["Horizon"]),
                    "response_candidate_id": str(row["ResponseCandidateID"]),
                    "order": [int(row["p"]), int(row["d"]), int(row["q"])],
                    "seasonal_order": [
                        int(row["P"]),
                        int(row["D"]),
                        int(row["Q"]),
                        int(row["SeasonalPeriod"]),
                    ],
                    "exogenous_set": str(row["ExogenousSet"]),
                    "lag_pattern": str(row["LagPattern"]),
                    "exact_lags": str(row["ExactLags"]),
                    "window_start": (
                        None if pd.isna(row["WindowStart"]) else int(row["WindowStart"])
                    ),
                    "window_end": (
                        None if pd.isna(row["WindowEnd"]) else int(row["WindowEnd"])
                    ),
                    "macro_validation_mae": float(row["MacroFoldMAE"]),
                }
                for _, row in selected_arimax.iterrows()
            ],
        },
    }
    config_path = directories["models"] / "04_selected_statistical_model_configs.json"
    config_path.write_text(json.dumps(selected_config, indent=2), encoding="utf-8")
    output_paths.append(config_path)

    acf_plot_path = directories["images"] / "04_affwb_acf_pacf_diagnostics.png"
    exog_plot_path = directories["images"] / "04_exogenous_lag_diagnostics.png"
    arima_plot_path = directories["images"] / "04_arima_validation_tuning.png"
    arimax_plot_path = directories["images"] / "04_arimax_selected_lags.png"
    horizon_plot_path = directories["images"] / "04_statistical_test_mae_by_horizon.png"
    actual_plot_path = directories["images"] / "04_statistical_test_actual_vs_predicted.png"
    plot_acf_pacf(acf_diagnostics, acf_plot_path)
    plot_exogenous_lags(exog_diagnostics, exog_plot_path)
    plot_top_arima_validation(arima_tuning, arima_plot_path)
    plot_arimax_selected_lags(selected_arimax, arimax_plot_path)
    plot_test_horizon_metrics(
        horizon_metrics.loc[horizon_metrics["Split"].eq("test")], horizon_plot_path
    )
    plot_test_actual_predicted(all_predictions, actual_plot_path)
    output_paths.extend(
        [
            acf_plot_path,
            exog_plot_path,
            arima_plot_path,
            arimax_plot_path,
            horizon_plot_path,
            actual_plot_path,
        ]
    )

    report_path = directories["reports"] / "04_statistical_models_report.md"
    write_report(
        report_path,
        selected_arima_row,
        selected_arimax,
        summary,
        daily_metrics,
        bootstrap,
        acf_diagnostics,
        exog_diagnostics,
        arima_tuning,
        arimax_tuning,
    )
    output_paths.append(report_path)

    input_manifest_path = directories["tables"] / "04_step_input_manifest.csv"
    pd.DataFrame(
        [
            {
                "Input": str(modeling_path.relative_to(project_root)),
                "Bytes": modeling_path.stat().st_size,
                "SHA256": file_sha256(modeling_path),
            },
            {
                "Input": str(target_path.relative_to(project_root)),
                "Bytes": target_path.stat().st_size,
                "SHA256": file_sha256(target_path),
            },
        ]
    ).to_csv(input_manifest_path, index=False)
    output_paths.append(input_manifest_path)

    runtime_path = directories["tables"] / "04_runtime_environment.csv"
    write_runtime_environment(runtime_path, args.n_jobs, args.quick)
    output_paths.append(runtime_path)

    manifest_path = directories["tables"] / "04_output_manifest.csv"
    build_output_manifest(output_paths, project_root, manifest_path)

    print("Step 04 revised statistical modeling completed successfully.")
    print(f"Selected ARIMA/SARIMA: {selected_arima_spec.candidate_id}")
    print("Selected ARIMAX horizon configurations:")
    print(
        selected_arimax[
            [
                "Horizon",
                "ResponseCandidateID",
                "ExogenousSet",
                "LagPattern",
                "MacroFoldMAE",
            ]
        ].to_string(index=False)
    )
    print("\nValidation and test summary:")
    print(summary.sort_values(["Split", "MAE"]).to_string(index=False))


if __name__ == "__main__":
    main()
