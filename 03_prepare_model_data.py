"""
Step 03: Prepare model-specific data structures for next-day six-step forecasting.

The script preserves the Step 01 occasion-level data and creates three model-ready
representations of the same forecasting problem:

1. ARIMA/ARIMAX occasion-level series with causal exogenous predictors.
2. RF/GBM/XGBoost tabular lag and history-summary datasets.
3. RNN/LSTM ordered sequence tensors with value masks and target masks.

The forecasting target is the six AffWB assessments of the next planned study day.
The training targets are Days 2-5, Day 6 is validation, and Day 7 is the untouched
final test period. No target values are imputed.

Run from the project root:
    python 03_prepare_model_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Project configuration
# -----------------------------------------------------------------------------

STEP_PREFIX = "03_"
MEASUREMENTS_PER_DAY = 6
TARGET_DAYS = list(range(2, 8))
TRAIN_TARGET_DAYS = [2, 3, 4, 5]
VALIDATION_TARGET_DAY = 6
TEST_TARGET_DAY = 7
FIXED_LOOKBACKS = [6, 12, 18, 24]
SEQUENCE_MAX_LENGTH = 36

BASE_PLATFORM_COMPONENTS = {
    "Instagram": {
        "passive": ["InsPas"],
        "active": ["InsAct_1", "InsAct_2"],
    },
    "WhatsApp": {
        "passive": ["WAPas"],
        "active": ["WAAct"],
    },
    "Snapchat": {
        "passive": ["SnpPas"],
        "active": ["SnpAct"],
    },
    "Games": {
        "passive": ["GamPas"],
        "active": ["GamAct"],
    },
    "YouTube": {
        "passive": ["YTPas"],
        "active": ["YTAct"],
    },
}

SEQUENCE_VALUE_COLUMNS = ["AffWB", "OverallPassiveUse", "OverallActiveUse"]
SEQUENCE_MASK_COLUMNS = [f"{column}__observed" for column in SEQUENCE_VALUE_COLUMNS]
SEQUENCE_CHANNELS = SEQUENCE_VALUE_COLUMNS + SEQUENCE_MASK_COLUMNS


def create_directories(project_root: Path) -> dict[str, Path]:
    """Create all project output folders used by this and later modeling steps."""
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
    """Return a SHA-256 checksum for an input or output file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# -----------------------------------------------------------------------------
# 2. Source loading and full-sample social-media exposure construction
# -----------------------------------------------------------------------------

def load_step01_data(path: Path) -> pd.DataFrame:
    """Load and validate the Step 01 canonical occasion-level dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Step 01 input not found: {path}")

    data = pd.read_csv(path)
    required = {
        "ParticipantID",
        "Occasion",
        "PlannedDayIndex",
        "PlannedOccasionWithinDay",
        "AffWB",
        "AffWB__observed",
    }
    for components in BASE_PLATFORM_COMPONENTS.values():
        for column in components["passive"] + components["active"]:
            required.update({column, f"{column}__observed", f"{column}__available"})

    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Required Step 01 columns are missing: {missing}")

    data = data.sort_values(["ParticipantID", "Occasion"]).reset_index(drop=True)
    if data.duplicated(["ParticipantID", "Occasion"]).any():
        raise ValueError("ParticipantID-Occasion keys must be unique.")

    return data


def strict_component_sum(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    """Sum components only when every required component has an observed numeric response."""
    observed_columns = [f"{column}__observed" for column in columns]
    all_observed = frame[observed_columns].eq(1).all(axis=1)
    values = frame[columns].sum(axis=1, min_count=len(columns)).where(all_observed)
    return values, all_observed.astype(int)


def add_overall_social_media_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Construct overall passive and active social-media exposure across each
    participant's personalized set of three platforms.

    The construction follows the source article's logic: platform components are
    summed and totals above 60 minutes are capped at 60. A total is observed only
    when every component required by the participant's assigned platform set is
    observed at that occasion. Structural non-assignment is not treated as zero.
    """
    result = data.copy()

    platform_value_columns: dict[str, dict[str, str]] = {}
    for platform_name, components in BASE_PLATFORM_COMPONENTS.items():
        platform_value_columns[platform_name] = {}
        for use_type in ["passive", "active"]:
            columns = components[use_type]
            value_column = f"{platform_name}_{use_type}_value"
            observed_column = f"{platform_name}_{use_type}_observed"
            available_column = f"{platform_name}_{use_type}_available"

            values, observed = strict_component_sum(result, columns)
            availability = result[[f"{column}__available" for column in columns]].eq(1).all(axis=1)

            result[value_column] = values.where(availability)
            result[observed_column] = (observed.eq(1) & availability).astype(int)
            result[available_column] = availability.astype(int)
            platform_value_columns[platform_name][use_type] = value_column

    for use_type, output_name in [
        ("passive", "OverallPassiveUse"),
        ("active", "OverallActiveUse"),
    ]:
        values = []
        observed_flags = []
        availability_flags = []

        for row_index, row in result.iterrows():
            assigned_values: list[float] = []
            assigned_observed: list[int] = []
            assigned_platform_count = 0

            for platform_name in BASE_PLATFORM_COMPONENTS:
                available_column = f"{platform_name}_{use_type}_available"
                observed_column = f"{platform_name}_{use_type}_observed"
                value_column = platform_value_columns[platform_name][use_type]

                if int(row[available_column]) == 1:
                    assigned_platform_count += 1
                    assigned_observed.append(int(row[observed_column]))
                    assigned_values.append(row[value_column])

            all_assigned_observed = (
                assigned_platform_count > 0
                and len(assigned_observed) == assigned_platform_count
                and all(flag == 1 for flag in assigned_observed)
            )

            if all_assigned_observed:
                total = float(np.nansum(assigned_values))
                values.append(min(total, 60.0))
                observed_flags.append(1)
            else:
                values.append(np.nan)
                observed_flags.append(0)

            availability_flags.append(1 if assigned_platform_count > 0 else 0)

        result[output_name] = values
        result[f"{output_name}__observed"] = observed_flags
        result[f"{output_name}__available"] = availability_flags

    return result


# -----------------------------------------------------------------------------
# 3. Common forecasting origins and target trajectories
# -----------------------------------------------------------------------------

def split_from_target_day(target_day: int) -> str:
    """Map each forecast target day to train, validation, or test."""
    if target_day in TRAIN_TARGET_DAYS:
        return "train"
    if target_day == VALIDATION_TARGET_DAY:
        return "validation"
    if target_day == TEST_TARGET_DAY:
        return "test"
    raise ValueError(f"Unexpected target day: {target_day}")


def build_complete_participant_index(data: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """
    Build an internal planned-position index for model-specific window extraction.

    This does not modify the Step 01 canonical dataset. It only permits target and
    history windows to retain their intended positions when a source row is absent.
    """
    indexed: dict[int, pd.DataFrame] = {}
    for participant_id, group in data.groupby("ParticipantID", sort=True):
        complete = pd.DataFrame({"Occasion": np.arange(1, 43, dtype=int)})
        complete = complete.merge(group, on="Occasion", how="left", suffixes=("", "_source"))
        complete["ParticipantID"] = int(participant_id)
        complete["PlannedDayIndex"] = ((complete["Occasion"] - 1) // MEASUREMENTS_PER_DAY) + 1
        complete["PlannedOccasionWithinDay"] = ((complete["Occasion"] - 1) % MEASUREMENTS_PER_DAY) + 1

        for value_column in SEQUENCE_VALUE_COLUMNS:
            observed_column = f"{value_column}__observed"
            if observed_column not in complete:
                complete[observed_column] = complete[value_column].notna().astype(int)
            else:
                complete[observed_column] = complete[observed_column].fillna(0).astype(int)

        indexed[int(participant_id)] = complete

    return indexed


def build_origin_and_target_tables(
    participant_index: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one forecast origin per participant-day and six target rows per origin."""
    origin_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []

    for participant_id, frame in participant_index.items():
        for target_day in TARGET_DAYS:
            history_end = (target_day - 1) * MEASUREMENTS_PER_DAY
            target_start = history_end + 1
            origin_id = f"P{participant_id:02d}_D{target_day}"
            split = split_from_target_day(target_day)

            history = frame.loc[frame["Occasion"] <= history_end]
            target = frame.loc[
                (frame["Occasion"] >= target_start)
                & (frame["Occasion"] <= target_start + MEASUREMENTS_PER_DAY - 1)
            ]

            origin_rows.append(
                {
                    "OriginID": origin_id,
                    "ParticipantID": participant_id,
                    "TargetDay": target_day,
                    "Split": split,
                    "HistoryStartOccasion": 1,
                    "HistoryEndOccasion": history_end,
                    "PlannedHistoryLength": history_end,
                    "ObservedHistoryAffWB": int(history["AffWB__observed"].sum()),
                    "ObservedHistoryPassiveUse": int(history["OverallPassiveUse__observed"].sum()),
                    "ObservedHistoryActiveUse": int(history["OverallActiveUse__observed"].sum()),
                    "ObservedTargetAffWB": int(target["AffWB__observed"].sum()),
                }
            )

            for horizon, (_, row) in enumerate(target.iterrows(), start=1):
                target_rows.append(
                    {
                        "OriginID": origin_id,
                        "ParticipantID": participant_id,
                        "TargetDay": target_day,
                        "Split": split,
                        "Horizon": horizon,
                        "TargetOccasion": int(row["Occasion"]),
                        "AffWBTarget": row["AffWB"],
                        "AffWBTargetObserved": int(row["AffWB__observed"]),
                    }
                )

    return pd.DataFrame(origin_rows), pd.DataFrame(target_rows)


# -----------------------------------------------------------------------------
# 4. Tree-model tabular windows
# -----------------------------------------------------------------------------

def observed_slope(values: np.ndarray, mask: np.ndarray) -> float:
    """Calculate a simple linear slope using only observed positions."""
    positions = np.arange(len(values), dtype=float)
    valid = (mask == 1) & np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    x = positions[valid]
    y = values[valid]
    if np.allclose(x, x[0]):
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def history_summary(prefix: str, values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Return causal history summaries for one variable and one lookback window."""
    valid = (mask == 1) & np.isfinite(values)
    observed_values = values[valid]
    result: dict[str, float] = {
        f"{prefix}_observed_count": float(valid.sum()),
        f"{prefix}_observed_fraction": float(valid.mean()) if len(valid) else np.nan,
        f"{prefix}_mean": float(np.mean(observed_values)) if len(observed_values) else np.nan,
        f"{prefix}_median": float(np.median(observed_values)) if len(observed_values) else np.nan,
        f"{prefix}_sd": float(np.std(observed_values, ddof=1)) if len(observed_values) > 1 else np.nan,
        f"{prefix}_min": float(np.min(observed_values)) if len(observed_values) else np.nan,
        f"{prefix}_max": float(np.max(observed_values)) if len(observed_values) else np.nan,
        f"{prefix}_slope": observed_slope(values, mask),
    }

    if valid.any():
        last_valid_index = np.where(valid)[0][-1]
        result[f"{prefix}_last_observed"] = float(values[last_valid_index])
        result[f"{prefix}_gap_since_last_observed"] = float(len(values) - 1 - last_valid_index)
    else:
        result[f"{prefix}_last_observed"] = np.nan
        result[f"{prefix}_gap_since_last_observed"] = float(len(values))

    return result


def build_tree_dataset(
    participant_index: dict[int, pd.DataFrame],
    targets: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """Create a pooled direct multi-horizon table for one fixed history length."""
    rows: list[dict[str, object]] = []

    for target_row in targets.itertuples(index=False):
        history_end = (int(target_row.TargetDay) - 1) * MEASUREMENTS_PER_DAY
        if history_end < lookback:
            continue

        frame = participant_index[int(target_row.ParticipantID)]
        history = frame.loc[
            (frame["Occasion"] > history_end - lookback)
            & (frame["Occasion"] <= history_end)
        ].sort_values("Occasion")

        if len(history) != lookback:
            raise RuntimeError("Internal model window extraction did not preserve planned positions.")

        feature_row: dict[str, object] = {
            "OriginID": target_row.OriginID,
            "ParticipantID": int(target_row.ParticipantID),
            "TargetDay": int(target_row.TargetDay),
            "Split": target_row.Split,
            "Horizon": int(target_row.Horizon),
            "TargetOccasion": int(target_row.TargetOccasion),
            "Lookback": lookback,
            "AffWBTarget": target_row.AffWBTarget,
            "AffWBTargetObserved": int(target_row.AffWBTargetObserved),
        }

        for value_column in SEQUENCE_VALUE_COLUMNS:
            mask_column = f"{value_column}__observed"
            values = history[value_column].to_numpy(dtype=float)
            masks = history[mask_column].fillna(0).to_numpy(dtype=int)
            short_name = {
                "AffWB": "mood",
                "OverallPassiveUse": "passive_use",
                "OverallActiveUse": "active_use",
            }[value_column]

            # lag_1 is the last planned assessment before the forecast day.
            reversed_values = values[::-1]
            reversed_masks = masks[::-1]
            for lag in range(1, lookback + 1):
                feature_row[f"{short_name}_lag_{lag}"] = reversed_values[lag - 1]
                feature_row[f"{short_name}_observed_lag_{lag}"] = int(reversed_masks[lag - 1])

            feature_row.update(history_summary(short_name, values, masks))

        rows.append(feature_row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 5. RNN/LSTM sequence tensors
# -----------------------------------------------------------------------------

def build_sequence_dataset(
    participant_index: dict[int, pd.DataFrame],
    origins: pd.DataFrame,
    targets: pd.DataFrame,
    lookback: int | None,
) -> dict[str, np.ndarray]:
    """
    Build ordered sequence tensors.

    Fixed-lookback tensors contain exactly L recent planned positions. The
    all-history tensor is right-padded to 36 positions and includes sequence
    lengths so recurrent models can ignore technical padding.
    """
    sample_origins = origins.copy()
    if lookback is not None:
        sample_origins = sample_origins.loc[sample_origins["PlannedHistoryLength"] >= lookback]

    origin_ids = sample_origins["OriginID"].tolist()
    max_length = lookback if lookback is not None else SEQUENCE_MAX_LENGTH

    x = np.zeros((len(origin_ids), max_length, len(SEQUENCE_CHANNELS)), dtype=np.float32)
    lengths = np.zeros(len(origin_ids), dtype=np.int64)
    y = np.full((len(origin_ids), MEASUREMENTS_PER_DAY), np.nan, dtype=np.float32)
    y_mask = np.zeros((len(origin_ids), MEASUREMENTS_PER_DAY), dtype=np.float32)
    participant_ids = np.zeros(len(origin_ids), dtype=np.int64)
    target_days = np.zeros(len(origin_ids), dtype=np.int64)
    splits: list[str] = []

    targets_by_origin = {origin_id: group.sort_values("Horizon") for origin_id, group in targets.groupby("OriginID")}

    for sample_index, origin in enumerate(sample_origins.itertuples(index=False)):
        frame = participant_index[int(origin.ParticipantID)]
        history_end = int(origin.HistoryEndOccasion)
        start = 1 if lookback is None else history_end - lookback + 1
        history = frame.loc[
            (frame["Occasion"] >= start) & (frame["Occasion"] <= history_end)
        ].sort_values("Occasion")

        if lookback is None:
            history = history.tail(SEQUENCE_MAX_LENGTH)
        sequence_length = len(history)
        lengths[sample_index] = sequence_length

        values = history[SEQUENCE_VALUE_COLUMNS].to_numpy(dtype=np.float32)
        masks = history[SEQUENCE_MASK_COLUMNS].fillna(0).to_numpy(dtype=np.float32)
        x[sample_index, :sequence_length, : len(SEQUENCE_VALUE_COLUMNS)] = values
        x[sample_index, :sequence_length, len(SEQUENCE_VALUE_COLUMNS) :] = masks

        target_group = targets_by_origin[origin.OriginID]
        y[sample_index, :] = target_group["AffWBTarget"].to_numpy(dtype=np.float32)
        y_mask[sample_index, :] = target_group["AffWBTargetObserved"].to_numpy(dtype=np.float32)
        participant_ids[sample_index] = int(origin.ParticipantID)
        target_days[sample_index] = int(origin.TargetDay)
        splits.append(origin.Split)

    return {
        "X": x,
        "lengths": lengths,
        "y": y,
        "y_mask": y_mask,
        "participant_ids": participant_ids,
        "target_days": target_days,
        "origin_ids": np.asarray(origin_ids, dtype="U16"),
        "splits": np.asarray(splits, dtype="U16"),
        "channel_names": np.asarray(SEQUENCE_CHANNELS, dtype="U64"),
        "lookback": np.asarray([-1 if lookback is None else lookback], dtype=np.int64),
    }


# -----------------------------------------------------------------------------
# 6. Quality checks, summaries, and file manifests
# -----------------------------------------------------------------------------

def build_quality_checks(
    data: pd.DataFrame,
    origins: pd.DataFrame,
    targets: pd.DataFrame,
    tree_tables: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Create explicit pass/fail checks for the model-preparation outputs."""
    checks = [
        (
            "One origin per participant and target day",
            not origins.duplicated(["ParticipantID", "TargetDay"]).any(),
            f"rows={len(origins)} expected={data['ParticipantID'].nunique() * len(TARGET_DAYS)}",
        ),
        (
            "Six target positions per origin",
            targets.groupby("OriginID").size().eq(MEASUREMENTS_PER_DAY).all(),
            f"target_rows={len(targets)}",
        ),
        (
            "Test period is Day 7 only",
            set(origins.loc[origins["Split"] == "test", "TargetDay"]) == {7},
            "test_target_day=7",
        ),
        (
            "Validation period is Day 6 only",
            set(origins.loc[origins["Split"] == "validation", "TargetDay"]) == {6},
            "validation_target_day=6",
        ),
        (
            "Overall exposure does not overwrite source variables",
            all(column in data.columns for column in ["OverallPassiveUse", "OverallActiveUse"]),
            "derived_columns_present",
        ),
        (
            "Observed overall exposure is numeric",
            data.loc[data["OverallPassiveUse__observed"] == 1, "OverallPassiveUse"].notna().all()
            and data.loc[data["OverallActiveUse__observed"] == 1, "OverallActiveUse"].notna().all(),
            "observed_flags_match_values",
        ),
    ]

    for lookback, table in tree_tables.items():
        checks.append(
            (
                f"Tree table L={lookback} uses only eligible origins",
                bool((table["TargetDay"] * MEASUREMENTS_PER_DAY - MEASUREMENTS_PER_DAY >= lookback).all()),
                f"rows={len(table)}",
            )
        )

    result = pd.DataFrame(checks, columns=["Check", "Passed", "Details"])
    if not result["Passed"].all():
        failed = result.loc[~result["Passed"], "Check"].tolist()
        raise RuntimeError(f"Step 03 quality checks failed: {failed}")
    return result


def write_runtime_environment(path: Path) -> None:
    """Record the Python and package environment used for Step 03."""
    rows = [
        {"Item": "Python", "Value": platform.python_version()},
        {"Item": "Platform", "Value": platform.platform()},
        {"Item": "pandas", "Value": pd.__version__},
        {"Item": "numpy", "Value": np.__version__},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def build_output_manifest(paths: Iterable[Path], destination: Path) -> None:
    """Write a checksum manifest for all Step 03 outputs."""
    rows = []
    for path in sorted(set(paths)):
        if path.exists() and path.is_file():
            rows.append(
                {
                    "File": str(path.relative_to(destination.parents[2])),
                    "Bytes": path.stat().st_size,
                    "SHA256": file_sha256(path),
                }
            )
    pd.DataFrame(rows).to_csv(destination, index=False)


# -----------------------------------------------------------------------------
# 7. Main execution
# -----------------------------------------------------------------------------

def main() -> None:
    """Execute all Step 03 transformations and write reproducible outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/tables/01_prepared_long_data.csv"),
        help="Step 01 prepared long CSV relative to the project root.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    directories = create_directories(project_root)
    input_path = args.input if args.input.is_absolute() else project_root / args.input

    output_paths: list[Path] = []

    source = load_step01_data(input_path)
    modeling_long = add_overall_social_media_variables(source)
    participant_index = build_complete_participant_index(modeling_long)
    origins, targets = build_origin_and_target_tables(participant_index)

    # Save the common occasion-level and forecast-origin representations.
    modeling_long_path = directories["data"] / "03_modeling_long_data.csv"
    origin_path = directories["tables"] / "03_forecast_origin_manifest.csv"
    target_path = directories["tables"] / "03_next_day_target_trajectory.csv"
    modeling_long.to_csv(modeling_long_path, index=False)
    origins.to_csv(origin_path, index=False)
    targets.to_csv(target_path, index=False)
    output_paths.extend([modeling_long_path, origin_path, target_path])

    # Create tree-model tables for every candidate fixed lookback.
    tree_tables: dict[int, pd.DataFrame] = {}
    for lookback in FIXED_LOOKBACKS:
        tree_table = build_tree_dataset(participant_index, targets, lookback)
        tree_path = directories["data"] / f"03_tree_samples_L{lookback:02d}.csv"
        tree_table.to_csv(tree_path, index=False)
        tree_tables[lookback] = tree_table
        output_paths.append(tree_path)

    # Create recurrent sequence tensors for fixed and all-history configurations.
    sequence_specs: list[tuple[str, int | None]] = [
        (f"L{lookback:02d}", lookback) for lookback in FIXED_LOOKBACKS
    ] + [("all_history", None)]

    for label, lookback in sequence_specs:
        tensors = build_sequence_dataset(participant_index, origins, targets, lookback)
        sequence_path = directories["data"] / f"03_sequence_samples_{label}.npz"
        np.savez_compressed(sequence_path, **tensors)
        output_paths.append(sequence_path)

    # Write compact summaries of split sizes and model-ready sample availability.
    split_summary = (
        origins.groupby(["Split", "TargetDay"], as_index=False)
        .agg(
            Origins=("OriginID", "size"),
            Participants=("ParticipantID", "nunique"),
            ObservedTargetAffWB=("ObservedTargetAffWB", "sum"),
            MeanObservedTargetsPerOrigin=("ObservedTargetAffWB", "mean"),
        )
    )
    split_summary_path = directories["tables"] / "03_split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False)
    output_paths.append(split_summary_path)

    availability_rows = []
    for lookback, table in tree_tables.items():
        for split, group in table.groupby("Split"):
            availability_rows.append(
                {
                    "Representation": "tree_tabular",
                    "Lookback": lookback,
                    "Split": split,
                    "Rows": len(group),
                    "ObservedTargets": int(group["AffWBTargetObserved"].sum()),
                    "Participants": int(group["ParticipantID"].nunique()),
                    "Origins": int(group["OriginID"].nunique()),
                }
            )
    availability = pd.DataFrame(availability_rows)
    availability_path = directories["tables"] / "03_model_sample_availability.csv"
    availability.to_csv(availability_path, index=False)
    output_paths.append(availability_path)

    predictor_definition = pd.DataFrame(
        [
            {
                "PredictorSet": "Mood history",
                "Variables": "AffWB values and observed-value masks",
                "Use": "ARIMA baseline and all machine-learning models",
            },
            {
                "PredictorSet": "Overall social-media history",
                "Variables": "OverallPassiveUse, OverallActiveUse, and masks",
                "Use": "ARIMAX, RF, GBM, XGBoost, RNN, and LSTM",
            },
            {
                "PredictorSet": "Target",
                "Variables": "Six AffWB values on the next planned study day",
                "Use": "Six-step trajectory forecast with masked evaluation",
            },
        ]
    )
    predictor_definition_path = directories["tables"] / "03_model_input_output_definitions.csv"
    predictor_definition.to_csv(predictor_definition_path, index=False)
    output_paths.append(predictor_definition_path)

    quality_checks = build_quality_checks(modeling_long, origins, targets, tree_tables)
    quality_path = directories["tables"] / "03_data_quality_checks.csv"
    quality_checks.to_csv(quality_path, index=False)
    output_paths.append(quality_path)

    input_manifest_path = directories["tables"] / "03_step_input_manifest.csv"
    pd.DataFrame(
        [
            {
                "Input": str(input_path.relative_to(project_root)),
                "Bytes": input_path.stat().st_size,
                "SHA256": file_sha256(input_path),
            }
        ]
    ).to_csv(input_manifest_path, index=False)
    output_paths.append(input_manifest_path)

    runtime_path = directories["tables"] / "03_runtime_environment.csv"
    write_runtime_environment(runtime_path)
    output_paths.append(runtime_path)

    report_path = directories["reports"] / "03_model_data_preparation_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Step 03 Model Data Preparation Report",
                "",
                f"- Source rows retained: {len(modeling_long):,}",
                f"- Participants: {modeling_long['ParticipantID'].nunique()}",
                f"- Forecast origins: {len(origins):,}",
                f"- Six-step target rows: {len(targets):,}",
                "- Training target days: 2-5",
                "- Validation target day: 6",
                "- Final test target day: 7",
                f"- Fixed lookbacks prepared: {', '.join(map(str, FIXED_LOOKBACKS))}",
                "- All-history recurrent representation: prepared with right padding and sequence lengths",
                "- Target imputation: none",
                "- Structural platform non-assignment treated as zero: no",
                "- Overall exposure totals capped at 60 minutes as in the source study",
                "",
                "All quality checks passed.",
            ]
        ),
        encoding="utf-8",
    )
    output_paths.append(report_path)

    manifest_path = directories["tables"] / "03_output_manifest.csv"
    build_output_manifest(output_paths, manifest_path)

    print("Step 03 completed successfully.")
    print(f"Forecast origins: {len(origins):,}")
    print(f"Target rows: {len(targets):,}")
    print(f"Outputs written under: {project_root / 'outputs'}")


if __name__ == "__main__":
    main()
