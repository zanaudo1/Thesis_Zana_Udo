# Step 03 — Model-Specific Data Preparation for Next-Day Forecasting

## 1. Purpose

Step 03 converts the Step 01 occasion-level canonical dataset into three model-specific representations while preserving one common forecasting task:

> Use information available before the start of a new study day to forecast the six `AffWB` assessments of that next day.

The model predicts a six-value trajectory. The first output is the immediately next ESM assessment (`h=1`), and the remaining outputs extend the forecast through the sixth assessment (`h=6`). A derived mean of the available six predictions is evaluated later as a secondary next-day summary.

## 2. Input

- `outputs/tables/01_prepared_long_data.csv`

The input contains the original participant sequences, numeric values, response-status fields, observed-value indicators, and questionnaire-availability indicators created in Step 01. Step 03 does not read the raw SAV or CSV directly.

## 3. Common Forecasting Definition

For participant `i` and target day `d`, the forecast origin is the end of day `d-1`:

```text
all eligible history before day d
    -> AffWB at horizons h=1,2,3,4,5,6 of day d
```

The chronological split is fixed before modeling:

- target Days 2-5: training;
- target Day 6: validation and hyperparameter selection;
- target Day 7: final untouched test.

No model may use any value from the target day as an input. Social-media use reported during the target day is therefore not available to the forecast.

## 4. Derived Overall Social-Media Predictors

The source study used personalized questionnaires covering three platforms per participant. Step 03 creates full-sample overall passive and active exposure variables using only the platforms assigned to that participant.

- Instagram active use is the strict sum of `InsAct_1` and `InsAct_2` when both are observed.
- Overall passive use is the strict sum of passive-use values across the participant's assigned platforms.
- Overall active use is the strict sum of active-use values across the assigned platforms.
- Totals above 60 minutes are capped at 60, following the source article.
- A total is missing when any required component for an assigned platform is unavailable at that occasion.
- A platform that was not assigned is not interpreted as zero.

The original variables are not overwritten.

## 5. ARIMA and ARIMAX Representation

### ARIMA input

ARIMA receives the ordered `AffWB` series for one participant. At a forecast origin it uses every available historical position before the target day and produces a six-step forecast.

### ARIMAX input

ARIMAX uses the same `AffWB` history plus causal lag-6 overall passive and active social-media predictors. Lag 6 aligns each forecast position with the corresponding position of the previous day. This is necessary because future target-day social-media use is not known at forecast time.

### Output

Both model types output six predicted `AffWB` values for the next day.

## 6. RF, GBM, and XGBoost Representation

The tree-based models receive pooled tabular rows. One row represents one participant, one target day, and one forecast horizon.

Candidate history lengths are:

```text
6, 12, 18, and 24 prior planned ESM positions
```

For each of `AffWB`, overall passive use, and overall active use, the table contains:

- individual lag values;
- lag observed-value indicators;
- observed count and fraction;
- mean, median, standard deviation, minimum, maximum;
- linear history slope;
- last observed value;
- gap since the last observed value.

`Horizon` is included so one pooled direct model can produce non-recursive forecasts for all six positions. `ParticipantID` and other identifiers are retained for evaluation but excluded from model features.

## 7. RNN and LSTM Representation

The recurrent models receive three-dimensional tensors:

```text
samples × sequence length × channels
```

Channels are:

- `AffWB`;
- overall passive use;
- overall active use;
- one observed-value mask for each numeric channel.

Fixed sequence lengths of 6, 12, 18, and 24 are prepared. An additional all-history representation uses every planned historical position available before the target day, up to 36 positions for the final Day 7 forecast. Variable-length histories are right-padded for storage, and sequence lengths are saved so the recurrent model can ignore technical padding.

The recurrent target is a six-value vector with a parallel target mask. Missing target values are never imputed and do not contribute to the training loss or evaluation metrics.

## 8. Missing-Data Decisions

- No `AffWB` target is imputed.
- Missing input values remain missing in the tabular outputs and are handled inside each later model pipeline using training-only information.
- Recurrent numeric tensors retain `NaN` before scaling; Step 06 replaces scaled missing values with zero only after adding explicit observed-value masks.
- Valid social-media zero values remain zero.
- Structural non-assignment is not converted to zero.
- The temporary planned-position index used to extract model windows does not replace or expand the canonical Step 01 dataset.

## 9. Outputs

### Common data and manifests

- `outputs/data/03_modeling_long_data.csv`
- `outputs/tables/03_forecast_origin_manifest.csv`
- `outputs/tables/03_next_day_target_trajectory.csv`
- `outputs/tables/03_split_summary.csv`
- `outputs/tables/03_model_sample_availability.csv`
- `outputs/tables/03_model_input_output_definitions.csv`

### Tree-model data

- `outputs/data/03_tree_samples_L06.csv`
- `outputs/data/03_tree_samples_L12.csv`
- `outputs/data/03_tree_samples_L18.csv`
- `outputs/data/03_tree_samples_L24.csv`

### Recurrent-model data

- `outputs/data/03_sequence_samples_L06.npz`
- `outputs/data/03_sequence_samples_L12.npz`
- `outputs/data/03_sequence_samples_L18.npz`
- `outputs/data/03_sequence_samples_L24.npz`
- `outputs/data/03_sequence_samples_all_history.npz`

### Reproducibility and reporting

- `outputs/tables/03_step_input_manifest.csv`
- `outputs/tables/03_runtime_environment.csv`
- `outputs/tables/03_data_quality_checks.csv`
- `outputs/tables/03_output_manifest.csv`
- `outputs/reports/03_model_data_preparation_report.md`

## 10. Run Command

```bash
python 03_prepare_model_data.py
```
