# Step 01 — Project Initialization and Data Preparation

## 1. Purpose

The purpose of Step 01 is to create a reproducible project structure and a traceable analysis-ready representation of the source ESM data before any descriptive analysis, feature engineering, data splitting, or modeling is performed.

This step establishes the rules that all later model-specific pipelines must follow. It deliberately preserves the sequential structure of the source records and keeps different missing-data mechanisms separate.

## 2. Inputs

The `inputs` directory contains the following Step 01 files:

- `inputs/01_beyens_2020_data.csv` — the executable source table with SPSS value labels rendered as text.
- `inputs/01_beyens_2020_data.sav` — the original SPSS source file retained for provenance and metadata verification.
- `inputs/01_beyens_2020_article.pdf` — the study article used to understand the ESM design and variable meaning.

The script reads the CSV file by default because it can be processed without an SPSS-specific Python dependency. The original SAV file is not modified.

## 3. Project Structure

Running `01_prepare_data.py` creates the following folders when they do not already exist:

```text
inputs/
outputs/
    tables/
    images/
    reports/
```

All Python scripts are stored in the project root. Every file created specifically for this step begins with the prefix `01_`.

## 4. Data Preparation Decisions

### 4.1 No daily aggregation

The source records are retained at the ESM-occasion level. Values are not averaged by day because later models must be able to test different history lengths using the original ordered measurements.

### 4.2 No artificial expansion to 42 rows per participant

The study design planned up to 42 assessments per participant, but the source file contains participant sequences of different available lengths. Step 01 does not add rows at the beginning or end of shorter sequences.

The reasons are:

1. Added rows would not contain new observed information.
2. The data do not justify inventing positions for records that are absent from the source table.
3. ARIMA/ARIMAX can use the full available history directly.
4. RF/GBM/XGBoost datasets will later be constructed from valid lag windows.
5. RNN/LSTM padding, when required for batching, is a model-specific tensor operation and must not be stored as a real observation in the canonical dataset.

The term *canonical dataset* therefore means a consistent variable structure, data type, ordering, and missing-status representation—not an artificially equal sequence length.

### 4.3 Sequence ordering

Rows are sorted by participant and `OccNrR`. The prepared columns are:

- `ParticipantID`
- `Occasion`
- `ClassNumber`
- `PlannedDayIndex`
- `PlannedOccasionWithinDay`

`PlannedDayIndex` and `PlannedOccasionWithinDay` are derived from the planned schedule of six assessments per day. They are helper indices only. Exact survey timestamps are not available, so they must not be interpreted as exact clock-time intervals.

### 4.4 Missing target values are not imputed

When `AffWB` is absent because the source value is `Expired`, `Blocked`, `Canceled`, `Other missing`, or blank, its numeric value is stored as `NaN`.

A row without an observed `AffWB` cannot later be used as a training, validation, or test target because no true outcome is available for evaluating a prediction.

### 4.5 Missing predictors are not imputed in Step 01

No mean, median, forward fill, interpolation, or zero replacement is applied. Model-specific input handling will be designed and tested later.

This prevents premature assumptions and prevents future values from being used to fill earlier observations.

### 4.6 Valid zero is different from missing

For social-media variables, a value of `0` means that the participant reported zero minutes of that activity in the previous hour. It remains a valid observed numeric value.

A missing response must never be replaced with zero without an explicit modeling decision and a separate missingness indicator.

### 4.7 Response missingness is different from structural missingness

The following source labels represent response missingness:

- `Expired`
- `Blocked`
- `Canceled`
- `Other missing`

They indicate that the question was available but no usable numeric response was recorded.

The following labels represent structural questionnaire unavailability:

- `No Instagram survey version`
- `No Snapchat survey version`
- `No Games survey version`
- `No YouTube survey version`

They mean that the participant did not receive the relevant platform-specific questionnaire version. These values are not zero use and are not ordinary item nonresponse.

For every measurement variable, the prepared dataset contains:

- the numeric value;
- `__status` — the reason the value is or is not available;
- `__observed` — 1 only for a valid numeric response;
- `__available` — 0 only when the question was structurally not asked.

### 4.8 No participant exclusion

All participants and all source rows are retained at this step. Later model-specific eligibility rules must be reported separately, including how many participants and prediction targets remain for each history length and model family.

## 5. Quality Checks

The script stops when a critical requirement fails. The checks include:

- required source columns are present;
- each participant-occasion key is unique;
- occasion values are within 1–42;
- observed `AffWB` values are within 1–7;
- observed social-media times are within 0–60 minutes;
- class number is constant within each participant;
- available participant sequences are contiguous from occasion 1.

The checks do not assume that every participant must have exactly 42 rows.

## 6. Outputs

### Reproducibility metadata

- `outputs/tables/01_input_manifest.csv` — input file names, sizes, and SHA-256 checksums.
- `outputs/tables/01_runtime_environment.csv` — Python and library versions used for the run.

### Prepared data

- `outputs/tables/01_prepared_long_data.csv` — ordered long-format dataset with numeric values and explicit missing-status indicators.

### Metadata and audit tables

- `outputs/tables/01_variable_dictionary.csv`
- `outputs/tables/01_value_labels.csv`
- `outputs/tables/01_participant_completeness.csv`
- `outputs/tables/01_missingness_summary.csv`
- `outputs/tables/01_status_counts_by_variable.csv`
- `outputs/tables/01_numeric_range_summary.csv`
- `outputs/tables/01_platform_availability_by_participant.csv`
- `outputs/tables/01_data_quality_checks.csv`

### Diagnostic images

- `outputs/images/01_participant_record_counts.png`
- `outputs/images/01_missingness_composition.png`

These figures are quality-control visuals, not the full descriptive analysis. Distributional and time-series exploration will be performed in the next step.

### Runtime report

- `outputs/reports/01_data_preparation_report.md` — actual row counts, participant counts, platform coverage, target availability, and quality-check results generated from the current input.

## 7. How to Run

From the project root:

```bash
pip install -r 01_requirements.txt
python 01_prepare_data.py
```

A custom CSV path can be provided:

```bash
python 01_prepare_data.py --input inputs/01_beyens_2020_data.csv
```

## 8. Boundary of Step 01

This step does **not**:

- average observations by day;
- add lagged variables;
- select a history length;
- impute model inputs;
- create train, validation, or test partitions;
- standardize predictors;
- fit or compare models.

Those decisions will be made later using the actual descriptive and temporal evidence, separately for ARIMA/ARIMAX, RF/GBM/XGBoost, and RNN/LSTM while preserving a common prediction target and chronological evaluation framework.
