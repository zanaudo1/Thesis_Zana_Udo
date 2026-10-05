"""
Step 02: Perform a comprehensive descriptive and temporal audit of the prepared ESM data.

This script reads the Step 01 canonical long-format dataset and creates the
summary tables, diagnostic figures, and generated report needed before any
model-specific feature engineering or data splitting is performed.

The script does not impute values, aggregate the modeling dataset by day, select
predictors, choose a history length, or fit forecasting models.

Run from the project root:
    python 02_descriptive_analysis.py

Optional custom input:
    python 02_descriptive_analysis.py \
        --input outputs/tables/01_prepared_long_data.csv
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats


# -----------------------------------------------------------------------------
# 1. Project configuration and variable metadata
# -----------------------------------------------------------------------------

STEP_PREFIX = "02_"
EXPECTED_ASSESSMENTS_PER_PARTICIPANT = 42
PLANNED_ASSESSMENTS_PER_DAY = 6

IDENTIFIER_COLUMNS = [
    "source_row_number",
    "ParticipantID",
    "Occasion",
    "ClassNumber",
    "PlannedDayIndex",
    "PlannedOccasionWithinDay",
]

MEASUREMENT_COLUMNS = [
    "AffWB",
    "InsPas",
    "InsAct_1",
    "InsAct_2",
    "WAPas",
    "WAAct",
    "SnpPas",
    "SnpAct",
    "GamPas",
    "GamAct",
    "YTPas",
    "YTAct",
]

PREDICTOR_COLUMNS = [column for column in MEASUREMENT_COLUMNS if column != "AffWB"]

VARIABLE_LABELS = {
    "AffWB": "Affective well-being",
    "InsPas": "Passive Instagram use",
    "InsAct_1": "Active Instagram broadcasting",
    "InsAct_2": "Active Instagram messaging",
    "WAPas": "Passive WhatsApp use",
    "WAAct": "Active WhatsApp use",
    "SnpPas": "Passive Snapchat use",
    "SnpAct": "Active Snapchat use",
    "GamPas": "Passive game-chat use",
    "GamAct": "Active game-chat use",
    "YTPas": "Passive YouTube use",
    "YTAct": "Active YouTube use",
}

PLATFORM_VARIABLES = {
    "Instagram": ["InsPas", "InsAct_1", "InsAct_2"],
    "WhatsApp": ["WAPas", "WAAct"],
    "Snapchat": ["SnpPas", "SnpAct"],
    "Games": ["GamPas", "GamAct"],
    "YouTube": ["YTPas", "YTAct"],
}

HISTORY_WINDOWS = [1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 30, 36]
AUTOCORRELATION_LAGS = list(range(1, 13))

REQUIRED_COLUMNS = IDENTIFIER_COLUMNS.copy()
for variable in MEASUREMENT_COLUMNS:
    REQUIRED_COLUMNS.extend(
        [
            variable,
            f"{variable}__status",
            f"{variable}__observed",
            f"{variable}__available",
        ]
    )


# -----------------------------------------------------------------------------
# 2. Generic file, formatting, and statistical helper functions
# -----------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments used by Step 02."""

    parser = argparse.ArgumentParser(
        description="Run descriptive and temporal diagnostics for the Step 01 ESM data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/tables/01_prepared_long_data.csv"),
        help="Path to the Step 01 prepared long-format CSV file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root output directory containing tables, images, and reports.",
    )
    return parser.parse_args()


def ensure_project_directories(output_root: Path) -> tuple[Path, Path, Path]:
    """Create the standard project output directories when they do not exist."""

    tables_dir = output_root / "tables"
    images_dir = output_root / "images"
    reports_dir = output_root / "reports"

    for directory in [Path("inputs"), output_root, tables_dir, images_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return tables_dir, images_dir, reports_dir


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 checksum for a file used by or created in this step."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_slug(text: str) -> str:
    """Convert a variable name into a stable lowercase file-name fragment."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "variable"


def save_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a CSV using a stable missing-value representation."""

    dataframe.to_csv(path, index=False, na_rep="")


def safe_pearson(x: Iterable[float], y: Iterable[float]) -> float:
    """Return a Pearson correlation or NaN when it cannot be estimated."""

    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]

    if len(x_array) < 3 or np.nanstd(x_array) == 0 or np.nanstd(y_array) == 0:
        return float("nan")
    return float(stats.pearsonr(x_array, y_array).statistic)


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    """Return a Spearman correlation or NaN when it cannot be estimated."""

    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]

    if len(x_array) < 3 or len(np.unique(x_array)) < 2 or len(np.unique(y_array)) < 2:
        return float("nan")
    return float(stats.spearmanr(x_array, y_array).statistic)


def format_number(value: float | int | None, digits: int = 3) -> str:
    """Format numeric values for the generated Markdown report."""

    if value is None or pd.isna(value):
        return "not estimable"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):,.{digits}f}"


# -----------------------------------------------------------------------------
# 3. Input loading and validation
# -----------------------------------------------------------------------------


def load_prepared_data(input_path: Path) -> pd.DataFrame:
    """Read the Step 01 canonical dataset and restore expected data types."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Step 02 input was not found: {input_path}. Run Step 01 first."
        )

    data = pd.read_csv(input_path)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "The Step 01 input is missing required columns: " + ", ".join(missing_columns)
        )

    integer_columns = [
        "source_row_number",
        "ParticipantID",
        "Occasion",
        "ClassNumber",
        "PlannedDayIndex",
        "PlannedOccasionWithinDay",
    ]
    integer_columns += [
        f"{variable}{suffix}"
        for variable in MEASUREMENT_COLUMNS
        for suffix in ["__observed", "__available"]
    ]

    for column in integer_columns:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(int)

    for variable in MEASUREMENT_COLUMNS:
        data[variable] = pd.to_numeric(data[variable], errors="coerce")
        data[f"{variable}__status"] = data[f"{variable}__status"].astype("string")

    data = data.sort_values(["ParticipantID", "Occasion"], kind="stable").reset_index(
        drop=True
    )
    return data


def build_quality_checks(data: pd.DataFrame) -> pd.DataFrame:
    """Run critical input checks before calculating descriptive outputs."""

    checks: list[dict[str, object]] = []

    def add_check(name: str, passed: bool, actual: object, expectation: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "actual": actual,
                "expectation": expectation,
            }
        )

    duplicate_keys = int(data.duplicated(["ParticipantID", "Occasion"]).sum())
    add_check(
        "Unique participant-occasion keys",
        duplicate_keys == 0,
        duplicate_keys,
        "0 duplicated ParticipantID-Occasion pairs",
    )

    sorted_copy = data.sort_values(["ParticipantID", "Occasion"], kind="stable")
    is_sorted = data[["ParticipantID", "Occasion"]].reset_index(drop=True).equals(
        sorted_copy[["ParticipantID", "Occasion"]].reset_index(drop=True)
    )
    add_check(
        "Rows sorted by participant and occasion",
        bool(is_sorted),
        bool(is_sorted),
        "True",
    )

    occasion_min = int(data["Occasion"].min())
    occasion_max = int(data["Occasion"].max())
    occasion_valid = occasion_min >= 1 and occasion_max <= EXPECTED_ASSESSMENTS_PER_PARTICIPANT
    add_check(
        "Occasion range",
        occasion_valid,
        f"{occasion_min}-{occasion_max}",
        "All occasions between 1 and 42",
    )

    derived_day = ((data["Occasion"] - 1) // PLANNED_ASSESSMENTS_PER_DAY) + 1
    derived_position = ((data["Occasion"] - 1) % PLANNED_ASSESSMENTS_PER_DAY) + 1
    helper_consistent = bool(
        (derived_day == data["PlannedDayIndex"]).all()
        and (derived_position == data["PlannedOccasionWithinDay"]).all()
    )
    add_check(
        "Derived planned indices",
        helper_consistent,
        helper_consistent,
        "Planned indices match Occasion-derived values",
    )

    flags_consistent = True
    availability_consistent = True
    ranges_consistent = True

    for variable in MEASUREMENT_COLUMNS:
        observed = data[f"{variable}__observed"].eq(1)
        available = data[f"{variable}__available"].eq(1)
        numeric_present = data[variable].notna()
        status = data[f"{variable}__status"]

        flags_consistent = flags_consistent and bool((observed == numeric_present).all())
        availability_consistent = availability_consistent and bool(
            ((status.eq("structural_not_asked")) == (~available)).all()
        )

        if variable == "AffWB":
            ranges_consistent = ranges_consistent and bool(
                data.loc[observed, variable].between(1, 7).all()
            )
        else:
            ranges_consistent = ranges_consistent and bool(
                data.loc[observed, variable].between(0, 60).all()
            )

    add_check(
        "Observed flags match numeric values",
        flags_consistent,
        flags_consistent,
        "Observed=1 exactly when a numeric value is present",
    )
    add_check(
        "Structural availability flags",
        availability_consistent,
        availability_consistent,
        "Available=0 exactly for structural_not_asked",
    )
    add_check(
        "Observed numeric ranges",
        ranges_consistent,
        ranges_consistent,
        "AffWB within 1-7 and social-media minutes within 0-60",
    )

    check_table = pd.DataFrame(checks)
    failures = check_table.loc[check_table["status"].eq("FAIL"), "check"].tolist()
    if failures:
        raise ValueError("Step 02 quality checks failed: " + "; ".join(failures))

    return check_table


# -----------------------------------------------------------------------------
# 4. Dataset, variable, status, and value-frequency summaries
# -----------------------------------------------------------------------------


def build_dataset_overview(data: pd.DataFrame) -> pd.DataFrame:
    """Create a compact table describing the dataset used by Step 02."""

    participant_counts = data.groupby("ParticipantID", sort=True).size()
    valid_affwb = int(data["AffWB__observed"].sum())

    rows = [
        ("source_rows", len(data), "Rows retained from the Step 01 canonical dataset"),
        ("participants", data["ParticipantID"].nunique(), "Distinct participants"),
        ("classes", data["ClassNumber"].nunique(), "Distinct class identifiers"),
        ("minimum_records_per_participant", int(participant_counts.min()), "Observed source rows"),
        ("median_records_per_participant", float(participant_counts.median()), "Observed source rows"),
        ("maximum_records_per_participant", int(participant_counts.max()), "Observed source rows"),
        ("maximum_planned_assessments", EXPECTED_ASSESSMENTS_PER_PARTICIPANT, "Study design"),
        ("valid_affwb_values", valid_affwb, "Observed prediction outcomes"),
        ("affwb_observation_rate", valid_affwb / len(data), "Among retained source rows"),
        ("minimum_occasion", int(data["Occasion"].min()), "Sequential occasion index"),
        ("maximum_occasion", int(data["Occasion"].max()), "Sequential occasion index"),
    ]

    return pd.DataFrame(rows, columns=["metric", "value", "interpretation"])


def build_variable_descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed numeric values and descriptive distribution shape."""

    records: list[dict[str, object]] = []
    total_rows = len(data)

    for variable in MEASUREMENT_COLUMNS:
        observed_mask = data[f"{variable}__observed"].eq(1)
        available_mask = data[f"{variable}__available"].eq(1)
        values = data.loc[observed_mask, variable].dropna().astype(float)

        response_missing_count = int((available_mask & ~observed_mask).sum())
        structural_missing_count = int((~available_mask).sum())
        zero_count = int(values.eq(0).sum()) if variable != "AffWB" else 0

        if len(values) >= 3 and values.nunique() > 1:
            skewness = float(stats.skew(values, bias=False, nan_policy="omit"))
            excess_kurtosis = float(
                stats.kurtosis(values, fisher=True, bias=False, nan_policy="omit")
            )
        else:
            skewness = float("nan")
            excess_kurtosis = float("nan")

        if 3 <= len(values) <= 5000 and values.nunique() > 1:
            shapiro_result = stats.shapiro(values)
            shapiro_w = float(shapiro_result.statistic)
            shapiro_p = float(shapiro_result.pvalue)
        else:
            shapiro_w = float("nan")
            shapiro_p = float("nan")

        if len(values) >= 8 and values.nunique() > 1:
            normaltest_result = stats.normaltest(values)
            dagostino_k2 = float(normaltest_result.statistic)
            dagostino_p = float(normaltest_result.pvalue)
        else:
            dagostino_k2 = float("nan")
            dagostino_p = float("nan")

        records.append(
            {
                "variable": variable,
                "variable_label": VARIABLE_LABELS[variable],
                "role": "target" if variable == "AffWB" else "time-varying predictor",
                "scale": "ordinal 1-7" if variable == "AffWB" else "bounded minutes 0-60",
                "total_rows": total_rows,
                "available_question_rows": int(available_mask.sum()),
                "observed_count": int(len(values)),
                "observed_rate_total_rows": len(values) / total_rows,
                "observed_rate_available_rows": (
                    len(values) / available_mask.sum() if available_mask.sum() else np.nan
                ),
                "response_missing_count": response_missing_count,
                "structural_missing_count": structural_missing_count,
                "unique_observed_values": int(values.nunique()),
                "mean": values.mean(),
                "standard_deviation": values.std(ddof=1),
                "minimum": values.min(),
                "q1": values.quantile(0.25),
                "median": values.median(),
                "q3": values.quantile(0.75),
                "maximum": values.max(),
                "iqr": values.quantile(0.75) - values.quantile(0.25),
                "zero_count": zero_count,
                "zero_rate_observed": zero_count / len(values) if len(values) else np.nan,
                "skewness": skewness,
                "excess_kurtosis": excess_kurtosis,
                "shapiro_w": shapiro_w,
                "shapiro_p_value": shapiro_p,
                "dagostino_k2": dagostino_k2,
                "dagostino_p_value": dagostino_p,
                "normality_test_caution": (
                    "Repeated observations are not independent; bounded, ordinal, and zero-inflated "
                    "variables should be evaluated visually and substantively, not by p-values alone."
                ),
            }
        )

    return pd.DataFrame(records)


def build_value_frequencies(data: pd.DataFrame) -> pd.DataFrame:
    """List every observed numeric alternative and its representation."""

    frequency_tables: list[pd.DataFrame] = []

    for variable in MEASUREMENT_COLUMNS:
        observed_values = data.loc[data[f"{variable}__observed"].eq(1), variable]
        counts = observed_values.value_counts(dropna=False).sort_index()
        table = counts.rename("count").reset_index().rename(columns={variable: "value"})
        table.insert(0, "variable_label", VARIABLE_LABELS[variable])
        table.insert(0, "variable", variable)
        table["percentage_of_observed"] = table["count"] / table["count"].sum()
        table["percentage_of_total_rows"] = table["count"] / len(data)
        frequency_tables.append(table)

    return pd.concat(frequency_tables, ignore_index=True)


def build_status_frequencies(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize every observed, response-missing, and structural status."""

    status_tables: list[pd.DataFrame] = []

    for variable in MEASUREMENT_COLUMNS:
        status_column = f"{variable}__status"
        counts = data[status_column].fillna("blank_status").value_counts(dropna=False)
        table = counts.rename("count").reset_index().rename(columns={status_column: "status"})
        table.insert(0, "variable_label", VARIABLE_LABELS[variable])
        table.insert(0, "variable", variable)
        table["percentage_of_total_rows"] = table["count"] / len(data)
        status_tables.append(table)

    return pd.concat(status_tables, ignore_index=True)


def build_class_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize participant and row representation by class identifier."""

    return (
        data.groupby("ClassNumber", as_index=False)
        .agg(
            participants=("ParticipantID", "nunique"),
            source_rows=("ParticipantID", "size"),
            observed_affwb=("AffWB__observed", "sum"),
        )
        .assign(
            participant_percentage=lambda frame: frame["participants"]
            / data["ParticipantID"].nunique(),
            row_percentage=lambda frame: frame["source_rows"] / len(data),
        )
    )


def build_platform_coverage(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize questionnaire availability and observed response coverage by platform."""

    rows: list[dict[str, object]] = []
    total_participants = data["ParticipantID"].nunique()

    for platform_name, variables in PLATFORM_VARIABLES.items():
        participant_available = (
            data.groupby("ParticipantID")[[f"{variable}__available" for variable in variables]]
            .max()
            .max(axis=1)
            .astype(bool)
        )
        available_participants = int(participant_available.sum())

        platform_available_rows = data[[f"{variable}__available" for variable in variables]].max(
            axis=1
        )
        platform_observed_rows = data[[f"{variable}__observed" for variable in variables]].max(
            axis=1
        )

        rows.append(
            {
                "platform": platform_name,
                "variables": ", ".join(variables),
                "participants_total": total_participants,
                "participants_with_questionnaire": available_participants,
                "participant_coverage_rate": available_participants / total_participants,
                "rows_with_questionnaire": int(platform_available_rows.sum()),
                "rows_with_at_least_one_observed_value": int(platform_observed_rows.sum()),
                "row_response_rate_when_available": (
                    platform_observed_rows.sum() / platform_available_rows.sum()
                    if platform_available_rows.sum()
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 5. AffWB summaries across participants and planned sequence positions
# -----------------------------------------------------------------------------


def summarize_affwb_by_group(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Calculate consistent AffWB summaries for a selected grouping variable."""

    observed = data.loc[data["AffWB__observed"].eq(1)].copy()

    summary = (
        observed.groupby(group_column, as_index=False)
        .agg(
            participants=("ParticipantID", "nunique"),
            observed_affwb=("AffWB", "size"),
            mean_affwb=("AffWB", "mean"),
            standard_deviation=("AffWB", "std"),
            median_affwb=("AffWB", "median"),
            minimum_affwb=("AffWB", "min"),
            maximum_affwb=("AffWB", "max"),
        )
        .sort_values(group_column)
    )
    summary["standard_error"] = summary["standard_deviation"] / np.sqrt(
        summary["observed_affwb"]
    )
    summary["ci95_lower"] = summary["mean_affwb"] - 1.96 * summary["standard_error"]
    summary["ci95_upper"] = summary["mean_affwb"] + 1.96 * summary["standard_error"]
    return summary


def build_affwb_by_participant(data: pd.DataFrame) -> pd.DataFrame:
    """Describe AffWB level and variability for every participant."""

    grouped = data.groupby("ParticipantID", sort=True)
    rows: list[dict[str, object]] = []

    for participant_id, participant_data in grouped:
        values = participant_data.loc[participant_data["AffWB__observed"].eq(1), "AffWB"]
        rows.append(
            {
                "ParticipantID": participant_id,
                "ClassNumber": int(participant_data["ClassNumber"].iloc[0]),
                "source_rows": len(participant_data),
                "first_occasion": int(participant_data["Occasion"].min()),
                "last_occasion": int(participant_data["Occasion"].max()),
                "observed_affwb": int(len(values)),
                "missing_affwb": int(len(participant_data) - len(values)),
                "affwb_observation_rate": len(values) / len(participant_data),
                "mean_affwb": values.mean(),
                "standard_deviation": values.std(ddof=1),
                "minimum_affwb": values.min(),
                "q1_affwb": values.quantile(0.25),
                "median_affwb": values.median(),
                "q3_affwb": values.quantile(0.75),
                "maximum_affwb": values.max(),
                "unique_affwb_values": int(values.nunique()),
            }
        )

    return pd.DataFrame(rows)


def build_affwb_variance_decomposition(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate descriptive within- and between-participant AffWB variation and ICC(1)."""

    observed = data.loc[data["AffWB__observed"].eq(1), ["ParticipantID", "AffWB"]].copy()
    participant_summary = observed.groupby("ParticipantID")["AffWB"].agg(["size", "mean"])

    n_observations = len(observed)
    n_participants = len(participant_summary)
    grand_mean = observed["AffWB"].mean()

    ss_between = float(
        (participant_summary["size"] * (participant_summary["mean"] - grand_mean) ** 2).sum()
    )
    observed_with_means = observed.join(
        participant_summary["mean"].rename("participant_mean"), on="ParticipantID"
    )
    ss_within = float(
        ((observed_with_means["AffWB"] - observed_with_means["participant_mean"]) ** 2).sum()
    )

    df_between = n_participants - 1
    df_within = n_observations - n_participants
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    n0 = (
        n_observations
        - (participant_summary["size"].pow(2).sum() / n_observations)
    ) / df_between

    between_variance_component = max((ms_between - ms_within) / n0, 0.0)
    within_variance_component = ms_within
    icc1 = between_variance_component / (
        between_variance_component + within_variance_component
    )

    metrics = [
        ("observed_affwb", n_observations, "Observed AffWB values"),
        ("participants", n_participants, "Participants with observed AffWB"),
        ("grand_mean", grand_mean, "Pooled AffWB mean"),
        ("ss_between", ss_between, "Between-participant sum of squares"),
        ("ss_within", ss_within, "Within-participant sum of squares"),
        ("ms_between", ms_between, "Between-participant mean square"),
        ("ms_within", ms_within, "Within-participant mean square"),
        ("effective_cluster_size_n0", n0, "Unequal-cluster-size correction"),
        (
            "between_variance_component",
            between_variance_component,
            "Estimated participant-level variance component",
        ),
        (
            "within_variance_component",
            within_variance_component,
            "Estimated within-participant variance component",
        ),
        (
            "icc1",
            icc1,
            "Descriptive proportion of variance attributable to participant differences",
        ),
    ]

    return pd.DataFrame(metrics, columns=["metric", "value", "interpretation"])


# -----------------------------------------------------------------------------
# 6. Missingness summaries by participant
# -----------------------------------------------------------------------------


def build_missingness_by_participant(data: pd.DataFrame) -> pd.DataFrame:
    """Count observed, response-missing, and structural values for every participant."""

    rows: list[dict[str, object]] = []

    for participant_id, participant_data in data.groupby("ParticipantID", sort=True):
        base = {
            "ParticipantID": participant_id,
            "ClassNumber": int(participant_data["ClassNumber"].iloc[0]),
            "source_rows": len(participant_data),
            "first_occasion": int(participant_data["Occasion"].min()),
            "last_occasion": int(participant_data["Occasion"].max()),
        }

        for variable in MEASUREMENT_COLUMNS:
            observed = participant_data[f"{variable}__observed"].eq(1)
            available = participant_data[f"{variable}__available"].eq(1)
            base[f"{variable}__observed_count"] = int(observed.sum())
            base[f"{variable}__response_missing_count"] = int((available & ~observed).sum())
            base[f"{variable}__structural_missing_count"] = int((~available).sum())
            base[f"{variable}__observed_rate_when_available"] = (
                observed.sum() / available.sum() if available.sum() else np.nan
            )

        rows.append(base)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 7. Temporal dependence, lagged associations, and history-window availability
# -----------------------------------------------------------------------------


def build_affwb_autocorrelation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate pooled and participant-specific sequential AffWB correlations."""

    summary_rows: list[dict[str, object]] = []
    individual_rows: list[dict[str, object]] = []

    sequence = data[["ParticipantID", "Occasion", "AffWB"]].copy()

    for lag in AUTOCORRELATION_LAGS:
        lagged = sequence.copy()
        lagged["AffWB_lagged"] = lagged.groupby("ParticipantID")["AffWB"].shift(lag)
        pairs = lagged.dropna(subset=["AffWB", "AffWB_lagged"]).copy()

        pairs["AffWB_centered"] = pairs["AffWB"] - pairs.groupby("ParticipantID")[
            "AffWB"
        ].transform("mean")
        pairs["AffWB_lagged_centered"] = pairs["AffWB_lagged"] - pairs.groupby(
            "ParticipantID"
        )["AffWB_lagged"].transform("mean")

        participant_correlations: list[float] = []
        participant_pair_counts: list[int] = []

        for participant_id, participant_pairs in pairs.groupby("ParticipantID"):
            pair_count = len(participant_pairs)
            correlation = safe_pearson(
                participant_pairs["AffWB_lagged"], participant_pairs["AffWB"]
            )
            if pair_count >= 5 and not pd.isna(correlation):
                participant_correlations.append(correlation)
                participant_pair_counts.append(pair_count)
                if lag == 1:
                    individual_rows.append(
                        {
                            "ParticipantID": participant_id,
                            "valid_pairs": pair_count,
                            "lag1_pearson_correlation": correlation,
                            "lag1_spearman_correlation": safe_spearman(
                                participant_pairs["AffWB_lagged"],
                                participant_pairs["AffWB"],
                            ),
                        }
                    )

        summary_rows.append(
            {
                "lag": lag,
                "valid_pairs": len(pairs),
                "participants_with_pairs": pairs["ParticipantID"].nunique(),
                "pooled_pearson": safe_pearson(pairs["AffWB_lagged"], pairs["AffWB"]),
                "pooled_spearman": safe_spearman(
                    pairs["AffWB_lagged"], pairs["AffWB"]
                ),
                "within_person_centered_pearson": safe_pearson(
                    pairs["AffWB_lagged_centered"], pairs["AffWB_centered"]
                ),
                "participants_with_estimable_individual_correlation": len(
                    participant_correlations
                ),
                "mean_individual_pearson": (
                    np.mean(participant_correlations) if participant_correlations else np.nan
                ),
                "median_individual_pearson": (
                    np.median(participant_correlations) if participant_correlations else np.nan
                ),
                "minimum_individual_pearson": (
                    np.min(participant_correlations) if participant_correlations else np.nan
                ),
                "maximum_individual_pearson": (
                    np.max(participant_correlations) if participant_correlations else np.nan
                ),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(individual_rows)


def build_lagged_predictor_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """Describe associations between lagged social-media variables and later AffWB."""

    rows: list[dict[str, object]] = []

    for predictor in PREDICTOR_COLUMNS:
        working = data[["ParticipantID", "Occasion", "AffWB", predictor]].copy()

        for lag in AUTOCORRELATION_LAGS:
            working[f"{predictor}_lagged"] = working.groupby("ParticipantID")[predictor].shift(
                lag
            )
            pairs = working.dropna(subset=["AffWB", f"{predictor}_lagged"]).copy()

            if len(pairs):
                pairs["AffWB_centered"] = pairs["AffWB"] - pairs.groupby("ParticipantID")[
                    "AffWB"
                ].transform("mean")
                pairs["predictor_centered"] = pairs[f"{predictor}_lagged"] - pairs.groupby(
                    "ParticipantID"
                )[f"{predictor}_lagged"].transform("mean")
            else:
                pairs["AffWB_centered"] = pd.Series(dtype=float)
                pairs["predictor_centered"] = pd.Series(dtype=float)

            rows.append(
                {
                    "predictor": predictor,
                    "predictor_label": VARIABLE_LABELS[predictor],
                    "lag": lag,
                    "valid_pairs": len(pairs),
                    "participants": pairs["ParticipantID"].nunique(),
                    "pooled_pearson": safe_pearson(
                        pairs[f"{predictor}_lagged"], pairs["AffWB"]
                    ),
                    "pooled_spearman": safe_spearman(
                        pairs[f"{predictor}_lagged"], pairs["AffWB"]
                    ),
                    "within_person_centered_pearson": safe_pearson(
                        pairs["predictor_centered"], pairs["AffWB_centered"]
                    ),
                    "within_person_centered_spearman": safe_spearman(
                        pairs["predictor_centered"], pairs["AffWB_centered"]
                    ),
                }
            )

            working = working.drop(columns=[f"{predictor}_lagged"])

    return pd.DataFrame(rows)


def build_history_window_availability(data: pd.DataFrame) -> pd.DataFrame:
    """Count usable target windows for multiple candidate history lengths and variables."""

    rows: list[dict[str, object]] = []
    target_observed = data["AffWB__observed"].eq(1)

    participant_position = data.groupby("ParticipantID").cumcount()

    for variable in MEASUREMENT_COLUMNS:
        observed_indicator = data[f"{variable}__observed"].astype(float)

        for lookback in HISTORY_WINDOWS:
            history_observed_count = (
                observed_indicator.groupby(data["ParticipantID"])
                .transform(
                    lambda series: series.shift(1)
                    .rolling(window=lookback, min_periods=1)
                    .sum()
                )
                .fillna(0)
            )

            has_required_prior_rows = participant_position >= lookback
            candidate_mask = target_observed & has_required_prior_rows
            complete_mask = candidate_mask & history_observed_count.eq(lookback)
            eighty_mask = candidate_mask & history_observed_count.ge(math.ceil(0.8 * lookback))
            half_mask = candidate_mask & history_observed_count.ge(math.ceil(0.5 * lookback))

            candidate_count = int(candidate_mask.sum())
            rows.append(
                {
                    "history_variable": variable,
                    "history_variable_label": VARIABLE_LABELS[variable],
                    "lookback_length": lookback,
                    "candidate_observed_targets": candidate_count,
                    "candidate_participants": data.loc[
                        candidate_mask, "ParticipantID"
                    ].nunique(),
                    "complete_history_windows": int(complete_mask.sum()),
                    "complete_history_participants": data.loc[
                        complete_mask, "ParticipantID"
                    ].nunique(),
                    "complete_history_rate": (
                        complete_mask.sum() / candidate_count if candidate_count else np.nan
                    ),
                    "at_least_80_percent_history_windows": int(eighty_mask.sum()),
                    "at_least_80_percent_history_rate": (
                        eighty_mask.sum() / candidate_count if candidate_count else np.nan
                    ),
                    "at_least_50_percent_history_windows": int(half_mask.sum()),
                    "at_least_50_percent_history_rate": (
                        half_mask.sum() / candidate_count if candidate_count else np.nan
                    ),
                    "mean_observed_history_fraction": (
                        (history_observed_count[candidate_mask] / lookback).mean()
                        if candidate_count
                        else np.nan
                    ),
                    "definition": (
                        "The target AffWB at the current row is observed, the participant has at "
                        "least the requested number of prior source rows, and history completeness "
                        "is calculated from numeric observations of the named history variable."
                    ),
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 8. Diagnostic figures
# -----------------------------------------------------------------------------


def save_distribution_plot(data: pd.DataFrame, variable: str, path: Path) -> None:
    """Create a histogram with a same-mean and same-SD normal reference curve."""

    values = data.loc[data[f"{variable}__observed"].eq(1), variable].dropna().astype(float)

    figure, axis = plt.subplots(figsize=(9, 5.5))

    if variable == "AffWB":
        bins = np.arange(0.5, 8.5, 1)
    else:
        bins = np.arange(-2.5, 65, 5)

    axis.hist(values, bins=bins, density=True, alpha=0.75, edgecolor="black")

    if len(values) > 1 and values.std(ddof=1) > 0:
        x_values = np.linspace(values.min(), values.max(), 300)
        normal_density = stats.norm.pdf(x_values, values.mean(), values.std(ddof=1))
        axis.plot(x_values, normal_density, linewidth=2, label="Normal reference")
        axis.legend()

    axis.set_title(f"Distribution of {VARIABLE_LABELS[variable]} ({variable})")
    axis.set_xlabel("AffWB score" if variable == "AffWB" else "Minutes in the previous hour")
    axis.set_ylabel("Density")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_qq_plot(data: pd.DataFrame, variable: str, path: Path) -> None:
    """Create a normal Q-Q plot for the observed values of one variable."""

    values = data.loc[data[f"{variable}__observed"].eq(1), variable].dropna().astype(float)

    figure, axis = plt.subplots(figsize=(6.5, 6.5))
    stats.probplot(values, dist="norm", plot=axis)
    axis.set_title(f"Normal Q-Q Plot: {VARIABLE_LABELS[variable]} ({variable})")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_affwb_sequence_mean_plot(occasion_summary: pd.DataFrame, path: Path) -> None:
    """Plot mean AffWB across the ordered 1-42 occasion index."""

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(occasion_summary["Occasion"], occasion_summary["mean_affwb"], marker="o")
    axis.fill_between(
        occasion_summary["Occasion"],
        occasion_summary["ci95_lower"],
        occasion_summary["ci95_upper"],
        alpha=0.2,
    )
    for boundary in range(6, 42, 6):
        axis.axvline(boundary + 0.5, linestyle="--", linewidth=0.8, alpha=0.5)
    axis.set_title("Mean AffWB Across the Sequential ESM Occasion Index")
    axis.set_xlabel("Occasion index (1-42)")
    axis.set_ylabel("Mean AffWB")
    axis.set_ylim(1, 7)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_affwb_group_mean_plot(
    summary: pd.DataFrame,
    group_column: str,
    x_label: str,
    title: str,
    path: Path,
) -> None:
    """Plot mean AffWB and 95% descriptive confidence intervals by a sequence helper."""

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.errorbar(
        summary[group_column],
        summary["mean_affwb"],
        yerr=1.96 * summary["standard_error"],
        marker="o",
        capsize=4,
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Mean AffWB")
    axis.set_ylim(1, 7)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_participant_affwb_plot(participant_summary: pd.DataFrame, path: Path) -> None:
    """Plot participant-specific AffWB means with one-SD error bars."""

    plot_data = participant_summary.sort_values("mean_affwb").reset_index(drop=True)
    x_values = np.arange(1, len(plot_data) + 1)

    figure, axis = plt.subplots(figsize=(13, 6))
    axis.errorbar(
        x_values,
        plot_data["mean_affwb"],
        yerr=plot_data["standard_deviation"].fillna(0),
        fmt="o",
        capsize=2,
    )
    axis.set_xticks(x_values)
    axis.set_xticklabels(plot_data["ParticipantID"].astype(str), rotation=90)
    axis.set_title("Participant-Specific AffWB Mean and Within-Participant SD")
    axis.set_xlabel("Participant ID, ordered by mean AffWB")
    axis.set_ylabel("AffWB")
    axis.set_ylim(1, 7)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_affwb_observation_heatmap(data: pd.DataFrame, path: Path) -> None:
    """Visualize observed, response-missing, and absent source positions for AffWB."""

    participants = sorted(data["ParticipantID"].unique())
    participant_to_row = {participant: index for index, participant in enumerate(participants)}
    matrix = np.zeros((len(participants), EXPECTED_ASSESSMENTS_PER_PARTICIPANT), dtype=int)

    for row in data.itertuples(index=False):
        row_index = participant_to_row[row.ParticipantID]
        column_index = row.Occasion - 1
        matrix[row_index, column_index] = 2 if row.AffWB__observed == 1 else 1

    figure, axis = plt.subplots(figsize=(13, 9))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=2)
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["No source row", "AffWB missing", "AffWB observed"])
    axis.set_title("AffWB Observation Status by Participant and Occasion")
    axis.set_xlabel("Occasion index")
    axis.set_ylabel("Participant ID")
    axis.set_xticks(np.arange(0, 42, 3))
    axis.set_xticklabels(np.arange(1, 43, 3))
    y_tick_positions = np.arange(0, len(participants), 4)
    axis.set_yticks(y_tick_positions)
    axis.set_yticklabels([participants[position] for position in y_tick_positions])
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_variable_observation_rates_plot(
    descriptive_statistics: pd.DataFrame, path: Path
) -> None:
    """Plot observed, response-missing, and structural proportions for each variable."""

    plot_data = descriptive_statistics.copy()
    plot_data["observed"] = plot_data["observed_count"] / plot_data["total_rows"]
    plot_data["response_missing"] = (
        plot_data["response_missing_count"] / plot_data["total_rows"]
    )
    plot_data["structural_missing"] = (
        plot_data["structural_missing_count"] / plot_data["total_rows"]
    )

    x_values = np.arange(len(plot_data))
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(x_values, plot_data["observed"], label="Observed")
    axis.bar(
        x_values,
        plot_data["response_missing"],
        bottom=plot_data["observed"],
        label="Response missing",
    )
    axis.bar(
        x_values,
        plot_data["structural_missing"],
        bottom=plot_data["observed"] + plot_data["response_missing"],
        label="Structural missing",
    )
    axis.set_xticks(x_values)
    axis.set_xticklabels(plot_data["variable"], rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Proportion of source rows")
    axis.set_title("Observation and Missingness Composition by Variable")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_platform_coverage_plot(platform_coverage: pd.DataFrame, path: Path) -> None:
    """Plot the percentage of participants who received each platform questionnaire."""

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.bar(platform_coverage["platform"], platform_coverage["participant_coverage_rate"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Participant questionnaire coverage")
    axis.set_title("Platform Questionnaire Coverage Across Participants")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_affwb_autocorrelation_plot(autocorrelation: pd.DataFrame, path: Path) -> None:
    """Plot pooled and within-person-centered AffWB correlations across lags."""

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        autocorrelation["lag"],
        autocorrelation["pooled_pearson"],
        marker="o",
        label="Pooled Pearson",
    )
    axis.plot(
        autocorrelation["lag"],
        autocorrelation["within_person_centered_pearson"],
        marker="o",
        label="Within-person centered Pearson",
    )
    axis.axhline(0, linewidth=0.8)
    axis.set_xticks(autocorrelation["lag"])
    axis.set_xlabel("Lag in sequential ESM occasions")
    axis.set_ylabel("Correlation")
    axis.set_title("AffWB Sequential Dependence by Lag")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_lagged_predictor_heatmap(correlations: pd.DataFrame, path: Path) -> None:
    """Visualize within-person lagged predictor-AffWB correlations."""

    pivot = correlations.pivot(
        index="predictor", columns="lag", values="within_person_centered_pearson"
    ).reindex(PREDICTOR_COLUMNS)

    figure, axis = plt.subplots(figsize=(12, 7))
    image = axis.imshow(pivot.values, aspect="auto", interpolation="nearest", cmap="coolwarm")
    figure.colorbar(image, ax=axis, label="Within-person centered Pearson correlation")
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns)
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels(pivot.index)
    axis.set_xlabel("Lag in sequential ESM occasions")
    axis.set_ylabel("Predictor")
    axis.set_title("Lagged Social-Media Predictors and Later AffWB")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_history_availability_plot(history_table: pd.DataFrame, path: Path) -> None:
    """Plot complete and partially observed AffWB history availability by lookback."""

    plot_data = history_table.loc[history_table["history_variable"].eq("AffWB")].copy()

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        plot_data["lookback_length"],
        plot_data["complete_history_rate"],
        marker="o",
        label="100% observed history",
    )
    axis.plot(
        plot_data["lookback_length"],
        plot_data["at_least_80_percent_history_rate"],
        marker="o",
        label="At least 80% observed history",
    )
    axis.plot(
        plot_data["lookback_length"],
        plot_data["at_least_50_percent_history_rate"],
        marker="o",
        label="At least 50% observed history",
    )
    axis.set_xticks(plot_data["lookback_length"])
    axis.set_ylim(0, 1)
    axis.set_xlabel("Lookback length in prior source rows")
    axis.set_ylabel("Proportion of candidate observed targets")
    axis.set_title("Availability of AffWB History for Candidate Lookback Lengths")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# -----------------------------------------------------------------------------
# 9. Generated Markdown report and manifests
# -----------------------------------------------------------------------------


def build_runtime_environment() -> pd.DataFrame:
    """Record the software environment used for the current Step 02 run."""

    rows = [
        ("python", platform.python_version()),
        ("platform", platform.platform()),
        ("pandas", pd.__version__),
        ("numpy", np.__version__),
        ("matplotlib", matplotlib.__version__),
        ("scipy", scipy.__version__),
    ]
    return pd.DataFrame(rows, columns=["component", "version"])


def build_input_manifest(input_path: Path) -> pd.DataFrame:
    """Record the exact Step 01 file used as the Step 02 analytical input."""

    return pd.DataFrame(
        [
            {
                "input_role": "prepared_long_data",
                "path": str(input_path),
                "file_name": input_path.name,
                "size_bytes": input_path.stat().st_size,
                "sha256": sha256_file(input_path),
            }
        ]
    )


def write_generated_report(
    report_path: Path,
    overview: pd.DataFrame,
    descriptive: pd.DataFrame,
    platform_coverage: pd.DataFrame,
    affwb_participant: pd.DataFrame,
    variance_decomposition: pd.DataFrame,
    autocorrelation: pd.DataFrame,
    lagged_correlations: pd.DataFrame,
    history_availability: pd.DataFrame,
    quality_checks: pd.DataFrame,
) -> None:
    """Write an English-language report populated with results from the current run."""

    overview_map = overview.set_index("metric")["value"].to_dict()
    affwb_stats = descriptive.set_index("variable").loc["AffWB"]
    variance_map = variance_decomposition.set_index("metric")["value"].to_dict()

    lag1 = autocorrelation.loc[autocorrelation["lag"].eq(1)].iloc[0]
    best_affwb_lag = autocorrelation.loc[
        autocorrelation["within_person_centered_pearson"].abs().idxmax()
    ]

    predictor_rank = (
        lagged_correlations.dropna(subset=["within_person_centered_pearson"])
        .assign(abs_correlation=lambda frame: frame["within_person_centered_pearson"].abs())
        .sort_values("abs_correlation", ascending=False)
    )
    strongest_predictor = predictor_rank.iloc[0] if len(predictor_rank) else None

    selected_history = history_availability.loc[
        history_availability["history_variable"].eq("AffWB")
        & history_availability["lookback_length"].isin([1, 3, 6, 12, 24, 36])
    ]

    platform_lines = []
    for row in platform_coverage.itertuples(index=False):
        platform_lines.append(
            f"- **{row.platform}:** {int(row.participants_with_questionnaire)} of "
            f"{int(row.participants_total)} participants "
            f"({row.participant_coverage_rate:.1%}) received the corresponding questionnaire."
        )

    history_lines = []
    for row in selected_history.itertuples(index=False):
        history_lines.append(
            f"- Lookback {row.lookback_length}: {int(row.complete_history_windows):,} complete "
            f"AffWB-history windows ({row.complete_history_rate:.1%} of candidate observed targets); "
            f"{int(row.complete_history_participants)} participants represented."
        )

    strongest_predictor_text = (
        "No estimable lagged predictor correlation was available."
        if strongest_predictor is None
        else (
            f"The largest absolute within-person-centered lagged correlation in this descriptive scan "
            f"was {strongest_predictor['within_person_centered_pearson']:.3f} for "
            f"{strongest_predictor['predictor']} at lag {int(strongest_predictor['lag'])}. "
            "This is an exploratory association, not evidence of causality or final predictor utility."
        )
    )

    report = f"""# Step 02 — Descriptive and Temporal Data Audit Report

## 1. Scope

Step 02 describes the Step 01 canonical ESM dataset before model-specific processing. It retains each source row at the sequential occasion level, does not average observations by day, does not add artificial rows, and does not impute missing values.

The analysis treats `Occasion` as an ordered sequence index. `PlannedDayIndex` and `PlannedOccasionWithinDay` are used only for descriptive summaries based on the planned six-assessments-per-day design. Exact timestamps are not available, so equal elapsed time between adjacent rows is not assumed.

## 2. Dataset Overview

- Source rows: **{int(overview_map['source_rows']):,}**
- Participants: **{int(overview_map['participants'])}**
- Classes: **{int(overview_map['classes'])}**
- Available records per participant: **{int(overview_map['minimum_records_per_participant'])} to {int(overview_map['maximum_records_per_participant'])}**
- Observed AffWB values: **{int(overview_map['valid_affwb_values']):,}** ({overview_map['affwb_observation_rate']:.1%} of retained rows)

## 3. AffWB Distribution

Observed AffWB has a mean of **{affwb_stats['mean']:.3f}**, a median of **{affwb_stats['median']:.3f}**, and an SD of **{affwb_stats['standard_deviation']:.3f}**. Values range from **{affwb_stats['minimum']:.0f} to {affwb_stats['maximum']:.0f}**.

AffWB is a bounded 1-7 response rather than a continuous unbounded Gaussian variable. The normality statistics and Q-Q plot are provided as descriptive diagnostics only. Repeated measurements within participants violate the independence assumption of ordinary pooled normality tests, and large samples can reject normality for small deviations.

## 4. Missingness and Questionnaire Coverage

Response missingness and structural questionnaire unavailability remain separate. A valid zero for a social-media variable remains an observed value and is not treated as missing.

{chr(10).join(platform_lines)}

The limited participant coverage for Games and YouTube means that these variables should not automatically be included in a single full-sample primary model. Step 02 does not remove them; it documents the coverage needed for later model-specific decisions.

## 5. Within- and Between-Participant AffWB Variation

The descriptive ICC(1) estimate is **{variance_map['icc1']:.3f}**. This indicates the approximate share of AffWB variation attributable to stable differences between participants under a one-way random-intercept decomposition. The remaining variation occurs within participants across occasions.

Participant-specific summaries and the participant mean/SD figure show whether the pooled distribution hides substantial individual differences. These differences are relevant because later validation should report both overall performance and variation in performance across participants.

## 6. Sequential Dependence

At lag 1, the pooled AffWB Pearson correlation is **{lag1['pooled_pearson']:.3f}**, while the within-person-centered correlation is **{lag1['within_person_centered_pearson']:.3f}**, based on **{int(lag1['valid_pairs']):,}** valid pairs.

Across lags 1-12, the largest absolute within-person-centered AffWB correlation occurs at lag **{int(best_affwb_lag['lag'])}** and equals **{best_affwb_lag['within_person_centered_pearson']:.3f}**. This descriptive result does not select a final history length. It shows how sequential association changes as observations become more distant.

{strongest_predictor_text}

## 7. Candidate History Lengths

The project will not impose a single history length in advance. The history-window table evaluates several candidate lookbacks while retaining the same next-occasion target definition.

{chr(10).join(history_lines)}

For ARIMA/ARIMAX, later steps can use an expanding history containing all observations available before each forecast origin. Fixed lookback comparisons are especially relevant to lag-feature models and RNN/LSTM sequence construction.

## 8. Interpretation Boundaries

Step 02 is descriptive and exploratory. It does not:

- select a final predictor set;
- choose a final lookback length;
- impute model inputs;
- create train, validation, or test partitions;
- fit ARIMA/ARIMAX, tree-based, RNN, or LSTM models;
- claim that lagged social-media use causes later changes in AffWB.

These decisions must be made in later steps using chronological validation and model-specific data preparation.

## 9. Quality Checks

All **{len(quality_checks)}** Step 02 input checks passed. Detailed results are stored in `outputs/tables/02_data_quality_checks.csv`.

## 10. Main Output Groups

- Dataset and variable summaries: `outputs/tables/02_*.csv`
- Distribution and normality figures: `outputs/images/02_*_distribution.png` and `outputs/images/02_*_qq_plot.png`
- Temporal and missingness figures: `outputs/images/02_affwb_*`, `02_lagged_*`, and `02_history_*`
- This generated report: `outputs/reports/02_descriptive_analysis_report.md`
"""

    report_path.write_text(report, encoding="utf-8")


def build_output_manifest(project_root: Path, output_path: Path) -> pd.DataFrame:
    """List all Step 02 files and checksums except the manifest itself."""

    step_files = sorted(
        path
        for path in project_root.rglob(f"{STEP_PREFIX}*")
        if path.is_file() and path.resolve() != output_path.resolve()
    )

    rows = []
    for path in step_files:
        rows.append(
            {
                "relative_path": str(path.relative_to(project_root)),
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 10. Main Step 02 workflow
# -----------------------------------------------------------------------------


def main() -> None:
    """Execute the complete Step 02 descriptive and temporal audit."""

    arguments = parse_arguments()
    project_root = Path.cwd()
    tables_dir, images_dir, reports_dir = ensure_project_directories(arguments.output_root)

    # Load the Step 01 output and stop immediately if critical structure checks fail.
    data = load_prepared_data(arguments.input)
    quality_checks = build_quality_checks(data)

    # Record reproducibility information before producing analytical results.
    input_manifest = build_input_manifest(arguments.input)
    runtime_environment = build_runtime_environment()

    # Build general descriptive, frequency, missingness, and coverage tables.
    dataset_overview = build_dataset_overview(data)
    variable_descriptive = build_variable_descriptive_statistics(data)
    value_frequencies = build_value_frequencies(data)
    status_frequencies = build_status_frequencies(data)
    class_distribution = build_class_distribution(data)
    platform_coverage = build_platform_coverage(data)
    missingness_by_participant = build_missingness_by_participant(data)

    # Build target-specific summaries without modifying the occasion-level dataset.
    affwb_by_participant = build_affwb_by_participant(data)
    affwb_by_occasion = summarize_affwb_by_group(data, "Occasion")
    affwb_by_planned_day = summarize_affwb_by_group(data, "PlannedDayIndex")
    affwb_by_within_day_position = summarize_affwb_by_group(
        data, "PlannedOccasionWithinDay"
    )
    affwb_variance = build_affwb_variance_decomposition(data)

    # Quantify sequential dependence and data availability for alternative histories.
    affwb_autocorrelation, affwb_individual_lag1 = build_affwb_autocorrelation(data)
    lagged_predictor_correlations = build_lagged_predictor_correlations(data)
    history_window_availability = build_history_window_availability(data)

    # Write all Step 02 analytical tables using the required 02_ prefix.
    table_outputs = {
        "02_step_input_manifest.csv": input_manifest,
        "02_runtime_environment.csv": runtime_environment,
        "02_data_quality_checks.csv": quality_checks,
        "02_dataset_overview.csv": dataset_overview,
        "02_variable_descriptive_statistics.csv": variable_descriptive,
        "02_value_frequencies.csv": value_frequencies,
        "02_status_frequencies.csv": status_frequencies,
        "02_class_distribution.csv": class_distribution,
        "02_platform_coverage.csv": platform_coverage,
        "02_missingness_by_participant.csv": missingness_by_participant,
        "02_affwb_by_participant.csv": affwb_by_participant,
        "02_affwb_by_occasion.csv": affwb_by_occasion,
        "02_affwb_by_planned_day.csv": affwb_by_planned_day,
        "02_affwb_by_within_day_position.csv": affwb_by_within_day_position,
        "02_affwb_variance_decomposition.csv": affwb_variance,
        "02_affwb_autocorrelation_by_lag.csv": affwb_autocorrelation,
        "02_affwb_individual_lag1_autocorrelation.csv": affwb_individual_lag1,
        "02_lagged_predictor_affwb_correlations.csv": lagged_predictor_correlations,
        "02_history_window_availability.csv": history_window_availability,
    }

    for file_name, table in table_outputs.items():
        save_csv(table, tables_dir / file_name)

    # Create one distribution figure and one Q-Q figure per numeric measurement variable.
    for variable in MEASUREMENT_COLUMNS:
        slug = safe_slug(variable)
        save_distribution_plot(
            data,
            variable,
            images_dir / f"02_{slug}_distribution.png",
        )
        save_qq_plot(
            data,
            variable,
            images_dir / f"02_{slug}_qq_plot.png",
        )

    # Create the principal target, missingness, coverage, temporal, and history figures.
    save_affwb_sequence_mean_plot(
        affwb_by_occasion, images_dir / "02_affwb_mean_by_occasion.png"
    )
    save_affwb_group_mean_plot(
        affwb_by_planned_day,
        "PlannedDayIndex",
        "Planned day index",
        "Mean AffWB by Planned Day Index",
        images_dir / "02_affwb_mean_by_planned_day.png",
    )
    save_affwb_group_mean_plot(
        affwb_by_within_day_position,
        "PlannedOccasionWithinDay",
        "Planned occasion position within day",
        "Mean AffWB by Planned Within-Day Position",
        images_dir / "02_affwb_mean_by_within_day_position.png",
    )
    save_participant_affwb_plot(
        affwb_by_participant, images_dir / "02_affwb_participant_mean_sd.png"
    )
    save_affwb_observation_heatmap(
        data, images_dir / "02_affwb_observation_status_heatmap.png"
    )
    save_variable_observation_rates_plot(
        variable_descriptive, images_dir / "02_variable_observation_rates.png"
    )
    save_platform_coverage_plot(
        platform_coverage, images_dir / "02_platform_participant_coverage.png"
    )
    save_affwb_autocorrelation_plot(
        affwb_autocorrelation, images_dir / "02_affwb_autocorrelation_by_lag.png"
    )
    save_lagged_predictor_heatmap(
        lagged_predictor_correlations,
        images_dir / "02_lagged_predictor_affwb_correlation_heatmap.png",
    )
    save_history_availability_plot(
        history_window_availability,
        images_dir / "02_affwb_history_window_availability.png",
    )

    # Generate the human-readable English report from the current analytical results.
    report_path = reports_dir / "02_descriptive_analysis_report.md"
    write_generated_report(
        report_path=report_path,
        overview=dataset_overview,
        descriptive=variable_descriptive,
        platform_coverage=platform_coverage,
        affwb_participant=affwb_by_participant,
        variance_decomposition=affwb_variance,
        autocorrelation=affwb_autocorrelation,
        lagged_correlations=lagged_predictor_correlations,
        history_availability=history_window_availability,
        quality_checks=quality_checks,
    )

    # Create the manifest after every other Step 02 output has been written.
    output_manifest_path = tables_dir / "02_output_manifest.csv"
    output_manifest = build_output_manifest(project_root, output_manifest_path)
    save_csv(output_manifest, output_manifest_path)

    print("Step 02 completed successfully.")
    print(f"Input rows: {len(data):,}")
    print(f"Participants: {data['ParticipantID'].nunique()}")
    print(f"Observed AffWB values: {int(data['AffWB__observed'].sum()):,}")
    print(f"Tables created: {len(table_outputs) + 1}")
    print(f"Images created: {len(MEASUREMENT_COLUMNS) * 2 + 10}")
    print(f"Generated report: {report_path}")


if __name__ == "__main__":
    main()
