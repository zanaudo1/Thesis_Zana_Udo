# Step 06 — Revised RNN and LSTM Models

## Objective

Evaluate compact recurrent neural networks for direct six-value next-day AffWB forecasting while respecting the short participant histories and the rolling-validation design established after Step 04.

## Core design

- The output is a six-value next-day trajectory.
- Candidate sequence lengths are 3, 6, 12, 18, and all available history.
- Mood-only and mood-plus-social-media channel sets are compared.
- Absolute targets and residual targets from persistence and previous-day seasonal baselines are compared.
- One recurrent layer is followed by six separate horizon heads.
- Rolling validation uses Days 4, 5, and 6.
- Architecture tuning averages three seeds; final validation and Day 7 predictions average five seeds.
- Missing values remain represented by explicit masks and are not treated as observed zeros.

## Main outputs

- `outputs/tables/06_recurrent_validation_screening.csv`
- `outputs/tables/06_recurrent_validation_tuning.csv`
- `outputs/tables/06_selected_recurrent_configs.csv`
- `outputs/tables/06_recurrent_model_predictions.csv`
- `outputs/tables/06_recurrent_model_metrics.csv`
- `outputs/tables/06_recurrent_model_horizon_metrics.csv`
- `outputs/tables/06_recurrent_daily_mean_metrics.csv`
- `outputs/reports/06_recurrent_models_report.md`
- `outputs/models/06_selected_recurrent_model_configs.json`

## Interpretation rule

The neural networks are not considered successful merely because they are more complex. They must demonstrate stable validation performance and useful generalization relative to persistence and SARIMA on the untouched Day 7 test set.
