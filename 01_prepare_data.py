"""
Step 01: Prepare the source data and create the project data-quality outputs.

This script intentionally does not build model-specific datasets. It creates a
traceable long-format dataset that preserves the original participant sequences,
keeps different missing-data mechanisms separate, and provides the quality-control
outputs required before descriptive analysis and time-series modeling.

Run from the project root:
    python 01_prepare_data.py

Optional custom input:
    python 01_prepare_data.py --input inputs/01_beyens_2020_data.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Project configuration and source metadata
# -----------------------------------------------------------------------------

STEP_PREFIX = "01_"
EXPECTED_ASSESSMENTS_PER_PARTICIPANT = 42
PLANNED_ASSESSMENTS_PER_DAY = 6

IDENTIFIER_COLUMNS = ["EthicaID", "OccNrR", "ClassNr"]
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
REQUIRED_COLUMNS = IDENTIFIER_COLUMNS + MEASUREMENT_COLUMNS

# The three Dutch labels are SPSS value labels for numeric AffWB values 1, 4, and 7.
AFFWB_LABEL_TO_VALUE = {
    "Helemaal niet": 1.0,
    "Een beetje": 4.0,
    "Helemaal wel": 7.0,
}

# Response-missing labels indicate that the question existed but no usable value was recorded.
RESPONSE_MISSING_LABELS = {
    "Blocked": "blocked",
    "Canceled": "canceled",
    "Expired": "expired",
    "Other missing": "other_missing",
}

# Structural labels indicate that the relevant platform-specific questionnaire version
# was not assigned to the participant. They must not be treated as zero minutes.
STRUCTURAL_MISSING_LABELS = {
    "No Instagram survey version",
    "No Snapchat survey version",
    "No Games survey version",
    "No YouTube survey version",
}

VARIABLE_METADATA: dict[str, dict[str, str]] = {
    "EthicaID": {
        "plain_name": "Participant ID",
        "role": "identifier",
        "description": "Anonymous participant identifier.",
        "valid_range": "1-63 in the supplied dataset",
    },
    "OccNrR": {
        "plain_name": "Occasion number",
        "role": "time index",
        "description": "Ordered ESM occasion number within the study.",
        "valid_range": "1-42",
    },
    "ClassNr": {
        "plain_name": "Class number",
        "role": "participant-level context",
        "description": "School class identifier.",
        "valid_range": "1-6 in the supplied dataset",
    },
    "AffWB": {
        "plain_name": "Momentary affective well-being",
        "role": "primary target",
        "description": "How happy the participant feels at the current ESM occasion.",
        "valid_range": "1-7",
    },
    "InsPas": {
        "plain_name": "Passive Instagram use",
        "role": "time-varying predictor",
        "description": "Minutes in the past hour spent viewing Instagram posts or stories.",
        "valid_range": "0-60 minutes",
    },
    "InsAct_1": {
        "plain_name": "Active Instagram broadcasting",
        "role": "time-varying predictor",
        "description": "Minutes in the past hour spent posting to an Instagram feed or story.",
        "valid_range": "0-60 minutes",
    },
    "InsAct_2": {
        "plain_name": "Active Instagram messaging",
        "role": "time-varying predictor",
        "description": "Minutes in the past hour spent sending direct Instagram messages or chatting.",
        "valid_range": "0-60 minutes",
    },
    "WAPas": {
        "plain_name": "Passive WhatsApp use",
        "role": "time-varying predictor",
        "description": "Minutes in the past hour spent reading WhatsApp messages.",
        "valid_range": "0-60 minutes",
    },
    "WAAct": {
        "plain_name": "Active WhatsApp use",
        "role": "time-varying predictor",
        "description": "Minutes in the past hour spent sending WhatsApp messages.",
        "valid_range": "0-60 minutes",
    },
    "SnpPas": {
        "plain_name": "Passive Snapchat use",
        "role": "time-varying predictor",
        "description": "Minutes in the past hour spent viewing Snapchat content from others.",
        "valid_range": "0-60 minutes",
    },
    "SnpAct": {
        "plain_name": "Active Snapchat use",
        "role": "time-varying predictor",
        "description": "Minutes in the past hour spent sending Snapchat content or posting a story.",
        "valid_range": "0-60 minutes",
    },
    "GamPas": {
        "plain_name": "Passive game-chat use",
        "role": "time-varying predictor",
        "description": "Minutes in the past hour spent reading messages in a game chat.",
        "valid_range": "0-60 minutes",
    },
    "GamAct": {
        "plain_name": "Active game-chat use",
        "role": "time-varying predictor",
        "description": "Minutes in the past hour spent sending messages in a game chat.",
        "valid_range": "0-60 minutes",
    },
    "YTPas": {
        "plain_name": "Passive YouTube use",
        "role": "time-varying predictor",
        "description": "Minutes in the past hour spent watching YouTube clips.",
        "valid_range": "0-60 minutes",
    },
    "YTAct": {
        "plain_name": "Active YouTube use",
        "role": "time-varying predictor",
        "description": "Minutes in the past hour spent posting YouTube clips.",
        "valid_range": "0-60 minutes",
    },
}

PLATFORM_VARIABLES = {
    "Instagram": ["InsPas", "InsAct_1", "InsAct_2"],
    "WhatsApp": ["WAPas", "WAAct"],
    "Snapchat": ["SnpPas", "SnpAct"],
    "Games": ["GamPas", "GamAct"],
    "YouTube": ["YTPas", "YTAct"],
}


# -----------------------------------------------------------------------------
# 2. Project directory creation
# -----------------------------------------------------------------------------

def create_project_directories(project_root: Path) -> dict[str, Path]:
    """Create the required project directories if they do not already exist."""
    directories = {
        "inputs": project_root / "inputs",
        "outputs": project_root / "outputs",
        "tables": project_root / "outputs" / "tables",
        "images": project_root / "outputs" / "images",
        "reports": project_root / "outputs" / "reports",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


# -----------------------------------------------------------------------------
# 3. Input loading and structural validation
# -----------------------------------------------------------------------------

def load_source_csv(input_path: Path) -> pd.DataFrame:
    """Load the label-preserving CSV and retain the original row number for traceability."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    source = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    source.columns = [column.strip() for column in source.columns]
    source.insert(0, "source_row_number", np.arange(1, len(source) + 1, dtype=int))

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(source.columns))
    unexpected_columns = sorted(set(source.columns) - set(REQUIRED_COLUMNS) - {"source_row_number"})
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")
    if unexpected_columns:
        print(f"Warning: unexpected source columns will be retained only in the input file: {unexpected_columns}")

    return source[["source_row_number"] + REQUIRED_COLUMNS].copy()


def standardize_identifier_columns(source: pd.DataFrame) -> pd.DataFrame:
    """Convert identifiers to integers and create transparent planned-time helper columns."""
    prepared = pd.DataFrame(index=source.index)
    prepared["source_row_number"] = source["source_row_number"].astype(int)
    prepared["ParticipantID"] = pd.to_numeric(source["EthicaID"], errors="raise").astype(int)
    prepared["Occasion"] = pd.to_numeric(source["OccNrR"], errors="raise").astype(int)
    prepared["ClassNumber"] = pd.to_numeric(source["ClassNr"], errors="raise").astype(int)

    # These helper columns follow the planned design of six assessments per day.
    # They do not represent exact timestamps because clock times are unavailable.
    prepared["PlannedDayIndex"] = ((prepared["Occasion"] - 1) // PLANNED_ASSESSMENTS_PER_DAY) + 1
    prepared["PlannedOccasionWithinDay"] = (
        (prepared["Occasion"] - 1) % PLANNED_ASSESSMENTS_PER_DAY
    ) + 1
    return prepared


# -----------------------------------------------------------------------------
# 4. Missing-status parsing without imputation
# -----------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    """Normalize a source cell to a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def parse_measurement_value(variable: str, raw_value: Any) -> tuple[float, str, int, int]:
    """
    Convert one source value to a numeric value plus explicit status indicators.

    Returns:
        numeric_value: observed numeric value or NaN
        status: observed, response-missing status, or structural_not_asked
        observed: 1 only when a numeric value is available
        available: 0 only when the platform question was structurally not asked
    """
    text = normalize_text(raw_value)

    if text in STRUCTURAL_MISSING_LABELS:
        return np.nan, "structural_not_asked", 0, 0

    if text in RESPONSE_MISSING_LABELS:
        return np.nan, RESPONSE_MISSING_LABELS[text], 0, 1

    if text == "":
        return np.nan, "blank_source_cell", 0, 1

    if variable == "AffWB" and text in AFFWB_LABEL_TO_VALUE:
        return AFFWB_LABEL_TO_VALUE[text], "observed", 1, 1

    try:
        numeric_value = float(text)
    except ValueError as exc:
        raise ValueError(f"Unrecognized value for {variable}: {text!r}") from exc

    if variable == "AffWB":
        if not 1.0 <= numeric_value <= 7.0:
            raise ValueError(f"AffWB value outside the valid 1-7 range: {numeric_value}")
    elif not 0.0 <= numeric_value <= 60.0:
        raise ValueError(f"{variable} value outside the valid 0-60 minute range: {numeric_value}")

    return numeric_value, "observed", 1, 1


def build_prepared_dataset(source: pd.DataFrame) -> pd.DataFrame:
    """
    Build the long-format prepared dataset.

    The function does not add rows, average measurements, or impute values. Every
    prepared row corresponds to one row that is actually present in the source file.
    """
    prepared = standardize_identifier_columns(source)

    for variable in MEASUREMENT_COLUMNS:
        parsed = source[variable].map(lambda value: parse_measurement_value(variable, value))
        parsed_frame = pd.DataFrame(
            parsed.tolist(),
            columns=[variable, f"{variable}__status", f"{variable}__observed", f"{variable}__available"],
            index=source.index,
        )
        prepared = pd.concat([prepared, parsed_frame], axis=1)

    # Sort rows into a reproducible participant-by-occasion order without altering occasion numbers.
    prepared = prepared.sort_values(["ParticipantID", "Occasion"], kind="stable").reset_index(drop=True)
    return prepared


# -----------------------------------------------------------------------------
# 5. Data-quality checks
# -----------------------------------------------------------------------------

def evaluate_quality_checks(source: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    """Evaluate critical assumptions and return a machine-readable quality table."""
    checks: list[dict[str, str]] = []

    def add_check(name: str, passed: bool, details: str, severity: str = "critical") -> None:
        checks.append(
            {
                "check_name": name,
                "status": "PASS" if passed else "FAIL",
                "severity": severity,
                "details": details,
            }
        )

    add_check(
        "Required columns are present",
        set(REQUIRED_COLUMNS).issubset(source.columns),
        f"Expected columns: {len(REQUIRED_COLUMNS)}; source columns used: {len(REQUIRED_COLUMNS)}.",
    )
    add_check(
        "Participant-occasion key is unique",
        not prepared.duplicated(["ParticipantID", "Occasion"]).any(),
        f"Duplicate keys: {int(prepared.duplicated(['ParticipantID', 'Occasion']).sum())}.",
    )
    add_check(
        "Occasion range is valid",
        prepared["Occasion"].between(1, EXPECTED_ASSESSMENTS_PER_PARTICIPANT).all(),
        f"Observed range: {prepared['Occasion'].min()}-{prepared['Occasion'].max()}.",
    )
    add_check(
        "AffWB observed values are valid",
        prepared.loc[prepared["AffWB__observed"].eq(1), "AffWB"].between(1, 7).all(),
        "All observed AffWB values must be between 1 and 7.",
    )

    time_columns = [column for column in MEASUREMENT_COLUMNS if column != "AffWB"]
    time_values_valid = all(
        prepared.loc[prepared[f"{column}__observed"].eq(1), column].between(0, 60).all()
        for column in time_columns
    )
    add_check(
        "Observed social-media minutes are valid",
        time_values_valid,
        "All observed social-media values must be between 0 and 60 minutes.",
    )

    class_counts = prepared.groupby("ParticipantID")["ClassNumber"].nunique()
    add_check(
        "Class number is constant within participant",
        class_counts.le(1).all(),
        f"Participants with multiple class values: {int(class_counts.gt(1).sum())}.",
    )

    noncontiguous_participants: list[int] = []
    for participant_id, group in prepared.groupby("ParticipantID", sort=True):
        occasions = group["Occasion"].tolist()
        expected_contiguous = list(range(1, len(occasions) + 1))
        if occasions != expected_contiguous:
            noncontiguous_participants.append(int(participant_id))

    add_check(
        "Available participant sequences are contiguous from occasion 1",
        len(noncontiguous_participants) == 0,
        (
            "All available sequences are contiguous from 1."
            if not noncontiguous_participants
            else f"Non-contiguous participant IDs: {noncontiguous_participants}."
        ),
    )

    quality = pd.DataFrame(checks)
    failed_critical = quality.query("severity == 'critical' and status == 'FAIL'")
    if not failed_critical.empty:
        failed_names = failed_critical["check_name"].tolist()
        raise ValueError(f"Critical data-quality checks failed: {failed_names}")

    return quality


# -----------------------------------------------------------------------------
# 6. Reproducibility metadata
# -----------------------------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 checksum without loading the entire file into memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_input_manifest(inputs_dir: Path) -> pd.DataFrame:
    """Create a provenance manifest for all Step 01 input files."""
    rows: list[dict[str, Any]] = []
    for file_path in sorted(path for path in inputs_dir.iterdir() if path.is_file()):
        rows.append(
            {
                "file_name": file_path.name,
                "relative_path": file_path.relative_to(inputs_dir.parent).as_posix(),
                "file_size_bytes": file_path.stat().st_size,
                "sha256": calculate_sha256(file_path),
            }
        )
    return pd.DataFrame(rows)


def create_runtime_environment() -> pd.DataFrame:
    """Record the main software versions used to execute Step 01."""
    return pd.DataFrame(
        [
            {"component": "Python", "version": platform.python_version()},
            {"component": "pandas", "version": pd.__version__},
            {"component": "NumPy", "version": np.__version__},
            {"component": "Matplotlib", "version": plt.matplotlib.__version__},
        ]
    )


# -----------------------------------------------------------------------------
# 7. Audit tables
# -----------------------------------------------------------------------------

def create_variable_dictionary() -> pd.DataFrame:
    """Create a plain-English dictionary for all source and derived columns."""
    rows: list[dict[str, str]] = []
    for variable in REQUIRED_COLUMNS:
        metadata = VARIABLE_METADATA[variable]
        rows.append(
            {
                "source_variable": variable,
                "prepared_value_column": {
                    "EthicaID": "ParticipantID",
                    "OccNrR": "Occasion",
                    "ClassNr": "ClassNumber",
                }.get(variable, variable),
                "plain_name": metadata["plain_name"],
                "role": metadata["role"],
                "description": metadata["description"],
                "valid_range": metadata["valid_range"],
                "missing_data_treatment": (
                    "Not applicable"
                    if variable in IDENTIFIER_COLUMNS
                    else "Numeric value is retained when observed; otherwise NaN plus status/observed/available indicators."
                ),
            }
        )

    derived_rows = [
        {
            "source_variable": "Derived",
            "prepared_value_column": "PlannedDayIndex",
            "plain_name": "Planned day index",
            "role": "derived time helper",
            "description": "Derived from Occasion assuming six planned assessments per day; not an exact timestamp.",
            "valid_range": "1-7",
            "missing_data_treatment": "Derived only for rows present in the source data.",
        },
        {
            "source_variable": "Derived",
            "prepared_value_column": "PlannedOccasionWithinDay",
            "plain_name": "Planned occasion within day",
            "role": "derived time helper",
            "description": "Position 1-6 within the planned day block; not an exact clock-time measure.",
            "valid_range": "1-6",
            "missing_data_treatment": "Derived only for rows present in the source data.",
        },
    ]
    return pd.DataFrame(rows + derived_rows)


def create_value_label_table() -> pd.DataFrame:
    """Document the source value labels and their prepared interpretation."""
    rows: list[dict[str, str | float]] = []

    for label, value in AFFWB_LABEL_TO_VALUE.items():
        rows.append(
            {
                "applies_to": "AffWB",
                "source_label": label,
                "prepared_numeric_value": value,
                "prepared_status": "observed",
                "available_indicator": 1,
                "interpretation": "Observed AffWB category encoded as a numeric scale value.",
            }
        )

    for label, status in RESPONSE_MISSING_LABELS.items():
        rows.append(
            {
                "applies_to": "All measurement variables",
                "source_label": label,
                "prepared_numeric_value": np.nan,
                "prepared_status": status,
                "available_indicator": 1,
                "interpretation": "Question was available, but no usable numeric response was recorded.",
            }
        )

    for label in sorted(STRUCTURAL_MISSING_LABELS):
        rows.append(
            {
                "applies_to": "Platform-specific variables",
                "source_label": label,
                "prepared_numeric_value": np.nan,
                "prepared_status": "structural_not_asked",
                "available_indicator": 0,
                "interpretation": "The participant did not receive this platform-specific questionnaire version.",
            }
        )

    return pd.DataFrame(rows)


def create_participant_completeness(prepared: pd.DataFrame) -> pd.DataFrame:
    """Summarize the length and target completeness of each participant sequence."""
    rows: list[dict[str, Any]] = []
    for participant_id, group in prepared.groupby("ParticipantID", sort=True):
        occasions = group["Occasion"].tolist()
        expected_contiguous = list(range(1, len(occasions) + 1))
        rows.append(
            {
                "ParticipantID": int(participant_id),
                "ClassNumber": int(group["ClassNumber"].iloc[0]),
                "ExpectedAssessmentsByDesign": EXPECTED_ASSESSMENTS_PER_PARTICIPANT,
                "AvailableRows": int(len(group)),
                "FirstOccasion": int(group["Occasion"].min()),
                "LastOccasion": int(group["Occasion"].max()),
                "SequenceContiguousFromOne": bool(occasions == expected_contiguous),
                "UnobservedAssessmentsRelativeToDesign": int(
                    EXPECTED_ASSESSMENTS_PER_PARTICIPANT - len(group)
                ),
                "ValidAffWB": int(group["AffWB__observed"].sum()),
                "AffWBResponseMissing": int(
                    group["AffWB__status"].isin(RESPONSE_MISSING_LABELS.values()).sum()
                ),
                "RowCompletionPercent": round(
                    100 * len(group) / EXPECTED_ASSESSMENTS_PER_PARTICIPANT, 2
                ),
                "AffWBObservedPercentOfAvailableRows": round(
                    100 * group["AffWB__observed"].mean(), 2
                ),
            }
        )
    return pd.DataFrame(rows)


def create_missingness_summary(prepared: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed, response-missing, and structural-not-asked values separately."""
    rows: list[dict[str, Any]] = []
    response_statuses = set(RESPONSE_MISSING_LABELS.values()) | {"blank_source_cell"}

    for variable in MEASUREMENT_COLUMNS:
        status = prepared[f"{variable}__status"]
        total = len(prepared)
        observed = int(status.eq("observed").sum())
        response_missing = int(status.isin(response_statuses).sum())
        structural = int(status.eq("structural_not_asked").sum())
        available = total - structural

        rows.append(
            {
                "variable": variable,
                "plain_name": VARIABLE_METADATA[variable]["plain_name"],
                "total_rows": total,
                "observed_numeric": observed,
                "response_missing": response_missing,
                "structural_not_asked": structural,
                "available_rows": available,
                "observed_percent_total": round(100 * observed / total, 2),
                "observed_percent_available": round(100 * observed / available, 2) if available else np.nan,
                "response_missing_percent_available": (
                    round(100 * response_missing / available, 2) if available else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def create_status_counts(prepared: pd.DataFrame) -> pd.DataFrame:
    """Create a long table containing every status count for every measurement variable."""
    rows: list[dict[str, Any]] = []
    for variable in MEASUREMENT_COLUMNS:
        counts = prepared[f"{variable}__status"].value_counts(dropna=False)
        for status, count in counts.items():
            rows.append(
                {
                    "variable": variable,
                    "plain_name": VARIABLE_METADATA[variable]["plain_name"],
                    "status": str(status),
                    "count": int(count),
                    "percent_of_rows": round(100 * count / len(prepared), 2),
                }
            )
    return pd.DataFrame(rows).sort_values(["variable", "status"]).reset_index(drop=True)


def create_numeric_summary(prepared: pd.DataFrame) -> pd.DataFrame:
    """Create basic numeric summaries for range checking; full EDA belongs to the next step."""
    rows: list[dict[str, Any]] = []
    for variable in MEASUREMENT_COLUMNS:
        values = prepared.loc[prepared[f"{variable}__observed"].eq(1), variable]
        rows.append(
            {
                "variable": variable,
                "plain_name": VARIABLE_METADATA[variable]["plain_name"],
                "count": int(values.count()),
                "mean": round(float(values.mean()), 4) if not values.empty else np.nan,
                "standard_deviation": round(float(values.std(ddof=1)), 4) if len(values) > 1 else np.nan,
                "minimum": float(values.min()) if not values.empty else np.nan,
                "q1": float(values.quantile(0.25)) if not values.empty else np.nan,
                "median": float(values.median()) if not values.empty else np.nan,
                "q3": float(values.quantile(0.75)) if not values.empty else np.nan,
                "maximum": float(values.max()) if not values.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def create_platform_availability(prepared: pd.DataFrame) -> pd.DataFrame:
    """Summarize which platform questionnaire versions are available for each participant."""
    rows: list[dict[str, Any]] = []
    for participant_id, group in prepared.groupby("ParticipantID", sort=True):
        row: dict[str, Any] = {"ParticipantID": int(participant_id)}
        for platform, variables in PLATFORM_VARIABLES.items():
            availability_values = pd.concat(
                [group[f"{variable}__available"] for variable in variables], axis=0
            )
            row[f"{platform}Available"] = int(availability_values.max())
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 8. Diagnostic figures
# -----------------------------------------------------------------------------

def save_participant_completion_plot(
    completeness: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the number of available source rows for every participant."""
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(completeness["ParticipantID"].astype(str), completeness["AvailableRows"])
    axis.axhline(
        EXPECTED_ASSESSMENTS_PER_PARTICIPANT,
        linestyle="--",
        linewidth=1.5,
        label="42 planned assessments",
    )
    axis.set_title("Available Source Rows by Participant")
    axis.set_xlabel("Participant ID")
    axis.set_ylabel("Available rows")
    axis.set_ylim(0, EXPECTED_ASSESSMENTS_PER_PARTICIPANT + 3)
    axis.tick_params(axis="x", labelrotation=90)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_missingness_composition_plot(
    missingness: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot observed values, response missingness, and structural missingness separately."""
    plot_frame = missingness.set_index("variable")[[
        "observed_numeric",
        "response_missing",
        "structural_not_asked",
    ]]
    axis = plot_frame.plot(kind="bar", stacked=True, figsize=(13, 7))
    axis.set_title("Measurement Status Composition by Variable")
    axis.set_xlabel("Variable")
    axis.set_ylabel("Number of source rows")
    axis.legend(["Observed numeric", "Response missing", "Structurally not asked"])
    axis.tick_params(axis="x", labelrotation=45)
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# -----------------------------------------------------------------------------
# 9. Generated step report
# -----------------------------------------------------------------------------

def build_report(
    project_root: Path,
    input_path: Path,
    prepared: pd.DataFrame,
    completeness: pd.DataFrame,
    missingness: pd.DataFrame,
    quality: pd.DataFrame,
) -> str:
    """Build a concise, data-derived Markdown report for Step 01."""
    participants = int(prepared["ParticipantID"].nunique())
    rows = int(len(prepared))
    valid_affwb = int(prepared["AffWB__observed"].sum())
    response_missing_affwb = int(prepared["AffWB__status"].ne("observed").sum())
    short_sequences = int((completeness["AvailableRows"] < EXPECTED_ASSESSMENTS_PER_PARTICIPANT).sum())

    platform_lines: list[str] = []
    availability = create_platform_availability(prepared)
    for platform in PLATFORM_VARIABLES:
        available_count = int(availability[f"{platform}Available"].sum())
        platform_lines.append(f"- {platform}: {available_count} of {participants} participants")

    failed_checks = int(quality["status"].eq("FAIL").sum())
    affwb_row = missingness.loc[missingness["variable"].eq("AffWB")].iloc[0]

    input_display = input_path.relative_to(project_root).as_posix() if input_path.is_relative_to(project_root) else input_path.as_posix()

    return f"""# Step 01 Runtime Report: Project and Data Preparation

## Input

- File: `{input_display}`
- Source rows: {rows:,}
- Participants: {participants}
- Planned assessments per participant: {EXPECTED_ASSESSMENTS_PER_PARTICIPANT}

## Sequence Structure

- The prepared dataset contains only rows that are present in the source file.
- No rows were added to force all participants to a length of 42.
- No daily aggregation was performed.
- {short_sequences} participants have fewer than 42 available source rows.
- All available participant sequences are contiguous from occasion 1 according to the quality checks.

## Missing-Data Representation

- Valid AffWB values: {valid_affwb:,}
- AffWB rows without a usable target value: {response_missing_affwb:,}
- AffWB observed percentage among available rows: {affwb_row['observed_percent_available']:.2f}%
- Missing target values were not imputed.
- Social-media response missingness and structural questionnaire unavailability were stored separately.
- Valid zero-minute responses were retained as numeric zeros.

## Platform Questionnaire Coverage

{chr(10).join(platform_lines)}

## Quality Control

- Quality checks executed: {len(quality)}
- Failed checks: {failed_checks}

## Modeling Boundary

This step does not create lagged predictors, impute model inputs, split data into train/validation/test sets, or fit models. Those actions must be performed later within model-specific pipelines using chronological information only.
"""


# -----------------------------------------------------------------------------
# 10. Output writing
# -----------------------------------------------------------------------------

def write_outputs(
    project_root: Path,
    directories: dict[str, Path],
    input_path: Path,
    prepared: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    """Create all Step 01 tables, figures, and the runtime report."""
    tables_dir = directories["tables"]
    images_dir = directories["images"]
    reports_dir = directories["reports"]

    input_manifest = create_input_manifest(directories["inputs"])
    runtime_environment = create_runtime_environment()
    variable_dictionary = create_variable_dictionary()
    value_labels = create_value_label_table()
    completeness = create_participant_completeness(prepared)
    missingness = create_missingness_summary(prepared)
    status_counts = create_status_counts(prepared)
    numeric_summary = create_numeric_summary(prepared)
    platform_availability = create_platform_availability(prepared)

    input_manifest.to_csv(
        tables_dir / "01_input_manifest.csv", index=False, encoding="utf-8"
    )
    runtime_environment.to_csv(
        tables_dir / "01_runtime_environment.csv", index=False, encoding="utf-8"
    )
    prepared.to_csv(tables_dir / "01_prepared_long_data.csv", index=False, encoding="utf-8")
    variable_dictionary.to_csv(
        tables_dir / "01_variable_dictionary.csv", index=False, encoding="utf-8"
    )
    value_labels.to_csv(tables_dir / "01_value_labels.csv", index=False, encoding="utf-8")
    completeness.to_csv(
        tables_dir / "01_participant_completeness.csv", index=False, encoding="utf-8"
    )
    missingness.to_csv(
        tables_dir / "01_missingness_summary.csv", index=False, encoding="utf-8"
    )
    status_counts.to_csv(
        tables_dir / "01_status_counts_by_variable.csv", index=False, encoding="utf-8"
    )
    numeric_summary.to_csv(
        tables_dir / "01_numeric_range_summary.csv", index=False, encoding="utf-8"
    )
    platform_availability.to_csv(
        tables_dir / "01_platform_availability_by_participant.csv",
        index=False,
        encoding="utf-8",
    )
    quality.to_csv(tables_dir / "01_data_quality_checks.csv", index=False, encoding="utf-8")

    save_participant_completion_plot(
        completeness,
        images_dir / "01_participant_record_counts.png",
    )
    save_missingness_composition_plot(
        missingness,
        images_dir / "01_missingness_composition.png",
    )

    report = build_report(project_root, input_path, prepared, completeness, missingness, quality)
    (reports_dir / "01_data_preparation_report.md").write_text(report, encoding="utf-8")


# -----------------------------------------------------------------------------
# 11. Command-line entry point
# -----------------------------------------------------------------------------

def main() -> None:
    """Run the complete Step 01 workflow."""
    parser = argparse.ArgumentParser(description="Prepare the MixedDataAnalysis source dataset.")
    parser.add_argument(
        "--input",
        default="inputs/01_beyens_2020_data.csv",
        help="CSV input path relative to the project root or as an absolute path.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    directories = create_project_directories(project_root)

    requested_path = Path(args.input)
    input_path = requested_path if requested_path.is_absolute() else project_root / requested_path

    source = load_source_csv(input_path)
    prepared = build_prepared_dataset(source)
    quality = evaluate_quality_checks(source, prepared)
    write_outputs(project_root, directories, input_path, prepared, quality)

    summary = {
        "step": "01",
        "input": str(input_path),
        "rows": int(len(prepared)),
        "participants": int(prepared["ParticipantID"].nunique()),
        "valid_affwb": int(prepared["AffWB__observed"].sum()),
        "outputs_directory": str(directories["outputs"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
