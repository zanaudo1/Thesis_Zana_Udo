# Step 04 — ARIMA and ARIMAX Modeling

## 1. Purpose

Step 04 tunes and evaluates the two statistical time-series models:

1. ARIMA as the univariate statistical baseline;
2. ARIMAX as the extension that adds historical social-media exposure.

Both models forecast the six `AffWB` positions of the next planned day. The order and the ARIMAX predictor set are selected using Day 6 validation performance. Day 7 remains untouched until the final evaluation.

## 2. Inputs

- `outputs/data/03_modeling_long_data.csv`
- `outputs/tables/03_next_day_target_trajectory.csv`

The first file contains ordered occasion-level `AffWB` and derived overall active/passive exposure. The second defines the six targets and their observed-value masks for every participant-day forecast origin.

## 3. ARIMA Input and Output

For each participant and forecast origin, ARIMA receives all planned `AffWB` positions before the target day. Missing target values are never imputed. For model fitting only, gaps inside the historical AffWB input are filled causally: forward filling uses earlier observed values, and any leading gaps are filled with the mean estimated from that participant’s available pre-forecast training history. This transformation never uses the target day.

Example for forecasting Day 7:

```text
AffWB occasions 1-36 -> six-step forecast for occasions 37-42
```

A common low-complexity order `(p,d,q)` is selected across participants by aggregated validation MAE. The coefficients are estimated separately from each participant's own history.

Output:

```text
Predicted AffWB at horizons 1,2,3,4,5,6
```

## 4. ARIMAX Input and Output

Future target-day social-media use is unknown at forecast time. ARIMAX therefore uses a causal lag-6 design:

```text
AffWB at position t <- social-media exposure at position t-6
```

For the six-step forecast, the known six exposure values from the previous day are supplied in the matching within-day order. The candidate exogenous sets are:

- overall passive use;
- overall active use;
- both overall passive and active use.

Exposure values are transformed with `log1p`. Missing historical exogenous values are filled using medians estimated from that participant's available pre-forecast history, and explicit missingness indicators are included. No future values are used.

Output is the same six-value next-day AffWB trajectory as ARIMA.

## 5. Tuning and Chronological Evaluation

### Validation

- history through Day 5;
- forecast Day 6;
- select the lowest-MAE configuration with at least 90% prediction coverage.

### Test

- history through Day 6;
- forecast Day 7;
- apply the already selected configuration once.

The test set is not used to choose `(p,d,q)`, exogenous variables, or any preprocessing decision.

## 6. Persistence Reference

A non-tuned persistence reference predicts all six future positions as the participant's most recent observed `AffWB`. It is reported for context but is not one of the seven candidate models.

## 7. Metrics

- masked MAE;
- masked RMSE;
- mean signed bias;
- prediction coverage;
- MAE/RMSE separately for horizons 1-6.

Only observed target positions contribute to the metrics.

## 8. Outputs

### Tables

- `outputs/tables/04_arima_validation_tuning.csv`
- `outputs/tables/04_arimax_validation_tuning.csv`
- `outputs/tables/04_statistical_model_predictions.csv`
- `outputs/tables/04_statistical_model_metrics.csv`
- `outputs/tables/04_statistical_model_horizon_metrics.csv`
- `outputs/tables/04_step_input_manifest.csv`
- `outputs/tables/04_runtime_environment.csv`
- `outputs/tables/04_output_manifest.csv`

### Model configuration

- `outputs/models/04_selected_statistical_model_configs.json`

### Images

- `outputs/images/04_arima_validation_tuning.png`
- `outputs/images/04_arimax_validation_tuning.png`
- `outputs/images/04_statistical_test_mae_by_horizon.png`

### Report

- `outputs/reports/04_statistical_models_report.md`

## 9. Run Command

```bash
python 04_train_arima_arimax.py
```
