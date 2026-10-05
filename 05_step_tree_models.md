# Step 05 — Revised Tree-Based Models

## Objective

Train and evaluate Random Forest, Gradient Boosting Machine, and XGBoost as nonlinear next-day forecasting models after incorporating the temporal findings from Step 04.

## Core design

- Six separate horizon-specific models are trained for each algorithm.
- Rolling-origin validation uses target Days 4, 5, and 6.
- Day 7 is not accessed until configuration selection is complete.
- Candidate history structures include contiguous lags and sparse lag combinations centered on lags 1, 6, 12, and 18.
- Predictor sets are `mood_only`, `mood_social`, and `social_only`.
- Targets are modeled directly or as residuals from persistence and previous-day seasonal baselines.
- Missing predictors are handled using training-only median imputation with missingness indicators.

## Main outputs

- `outputs/tables/05_tree_validation_screening.csv`
- `outputs/tables/05_tree_validation_tuning.csv`
- `outputs/tables/05_selected_tree_horizon_configs.csv`
- `outputs/tables/05_tree_model_predictions.csv`
- `outputs/tables/05_tree_model_metrics.csv`
- `outputs/tables/05_tree_model_horizon_metrics.csv`
- `outputs/tables/05_tree_daily_mean_metrics.csv`
- `outputs/tables/05_tree_model_feature_importance.csv`
- `outputs/reports/05_tree_models_report.md`
- `outputs/models/05_selected_tree_models.joblib`

## Interpretation rule

Social-media predictors are not assumed to be useful. They are retained only when rolling validation selects them for a specific algorithm and horizon. The final test evaluates whether those validation gains generalize.
