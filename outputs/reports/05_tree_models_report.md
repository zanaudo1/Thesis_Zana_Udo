# Step 05 — Revised Tree-Based Forecasting Models

## Methodological revision

Random Forest, GBM, and XGBoost were trained as six horizon-specific models rather than one pooled model with a horizon indicator. Configuration selection used three rolling-origin validation folds: Days 1–3 to predict Day 4, Days 1–4 to predict Day 5, and Days 1–5 to predict Day 6. Day 7 was used only after all configurations had been selected.

The tuning procedure compared short contiguous histories, one-to-three-day histories, sparse lag sets containing lags 1, 6, 12, and 18, mood-only and mood-plus-social-media predictors, and absolute versus residual targets relative to persistence and previous-day seasonal baselines.

## Selected configurations

| Model        |   Horizon | FeatureSpec    | PredictorSet   | TargetMode           | Parameters                                                                                                                                          |   ValidationMacroMAE |
|:-------------|----------:|:---------------|:---------------|:---------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------:|
| GBM          |         1 | day_1_6        | mood_only      | absolute             | {"learning_rate": 0.04, "max_depth": 2, "min_samples_leaf": 5, "n_estimators": 100, "subsample": 0.8}                                               |             0.841186 |
| GBM          |         2 | day_1_6        | mood_only      | absolute             | {"learning_rate": 0.04, "max_depth": 2, "min_samples_leaf": 5, "n_estimators": 100, "subsample": 0.8}                                               |             0.790253 |
| GBM          |         3 | day_1_6        | mood_only      | absolute             | {"learning_rate": 0.04, "max_depth": 1, "min_samples_leaf": 5, "n_estimators": 80, "subsample": 0.8}                                                |             0.752225 |
| GBM          |         4 | day_1_6        | mood_only      | absolute             | {"learning_rate": 0.04, "max_depth": 2, "min_samples_leaf": 5, "n_estimators": 100, "subsample": 0.8}                                               |             0.823556 |
| GBM          |         5 | mixed_123_6_12 | mood_social    | residual_persistence | {"learning_rate": 0.04, "max_depth": 2, "min_samples_leaf": 5, "n_estimators": 100, "subsample": 0.8}                                               |             0.901045 |
| GBM          |         6 | day_1_6        | mood_only      | absolute             | {"learning_rate": 0.04, "max_depth": 2, "min_samples_leaf": 5, "n_estimators": 100, "subsample": 0.8}                                               |             0.768622 |
| RandomForest |         1 | day_1_6        | mood_only      | absolute             | {"max_depth": 4, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 80}                                                                    |             0.814121 |
| RandomForest |         2 | day_1_6        | mood_only      | absolute             | {"max_depth": 4, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 80}                                                                    |             0.820997 |
| RandomForest |         3 | day_1_6        | mood_only      | absolute             | {"max_depth": 4, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 80}                                                                    |             0.780777 |
| RandomForest |         4 | day_1_6        | mood_only      | absolute             | {"max_depth": 8, "max_features": 1.0, "min_samples_leaf": 4, "n_estimators": 100}                                                                   |             0.82827  |
| RandomForest |         5 | day_1_6        | mood_only      | absolute             | {"max_depth": 4, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 80}                                                                    |             0.874387 |
| RandomForest |         6 | day_1_6        | mood_only      | absolute             | {"max_depth": 4, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 80}                                                                    |             0.780928 |
| XGBoost      |         1 | day_1_6        | mood_only      | absolute             | {"colsample_bytree": 0.8, "learning_rate": 0.04, "max_depth": 3, "min_child_weight": 3.0, "n_estimators": 100, "reg_lambda": 3.0, "subsample": 0.8} |             0.773254 |
| XGBoost      |         2 | day_1_6        | mood_only      | absolute             | {"colsample_bytree": 0.8, "learning_rate": 0.04, "max_depth": 3, "min_child_weight": 3.0, "n_estimators": 100, "reg_lambda": 3.0, "subsample": 0.8} |             0.740183 |
| XGBoost      |         3 | day_1_6        | mood_only      | absolute             | {"colsample_bytree": 0.8, "learning_rate": 0.04, "max_depth": 2, "min_child_weight": 3.0, "n_estimators": 80, "reg_lambda": 2.0, "subsample": 0.8}  |             0.715983 |
| XGBoost      |         4 | day_1_6        | mood_only      | absolute             | {"colsample_bytree": 0.8, "learning_rate": 0.04, "max_depth": 2, "min_child_weight": 3.0, "n_estimators": 80, "reg_lambda": 2.0, "subsample": 0.8}  |             0.816634 |
| XGBoost      |         5 | day_1_6        | mood_only      | absolute             | {"colsample_bytree": 0.8, "learning_rate": 0.04, "max_depth": 2, "min_child_weight": 3.0, "n_estimators": 80, "reg_lambda": 2.0, "subsample": 0.8}  |             0.875909 |
| XGBoost      |         6 | day_1_6        | mood_only      | absolute             | {"colsample_bytree": 0.8, "learning_rate": 0.04, "max_depth": 2, "min_child_weight": 3.0, "n_estimators": 80, "reg_lambda": 2.0, "subsample": 0.8}  |             0.737614 |

## Overall validation and test metrics

| Model        | Split      |    MAE |   RMSE |    Bias |   N |
|:-------------|:-----------|-------:|-------:|--------:|----:|
| XGBoost      | test       | 0.7211 | 0.9785 | -0.0181 | 292 |
| GBM          | test       | 0.7571 | 0.9907 | -0.0550 | 292 |
| RandomForest | test       | 0.7623 | 0.9786 | -0.1800 | 292 |
| XGBoost      | validation | 0.7792 | 1.1071 |  0.0888 | 929 |
| GBM          | validation | 0.8172 | 1.1265 |  0.0248 | 929 |
| RandomForest | validation | 0.8203 | 1.1170 | -0.0669 | 929 |

## Derived next-day mean metrics

| Model        | Split      |   MinimumObservedPositions |   ParticipantDays |    MAE |   RMSE |    Bias |   N |
|:-------------|:-----------|---------------------------:|------------------:|-------:|-------:|--------:|----:|
| XGBoost      | test       |                          4 |                51 | 0.4954 | 0.6516 | -0.0738 |  51 |
| GBM          | test       |                          4 |                51 | 0.5143 | 0.6610 | -0.1063 |  51 |
| RandomForest | test       |                          4 |                51 | 0.5512 | 0.6868 | -0.2303 |  51 |
| XGBoost      | validation |                          4 |               165 | 0.4646 | 0.6236 |  0.0876 | 165 |
| GBM          | validation |                          4 |               165 | 0.4755 | 0.6255 |  0.0248 | 165 |
| RandomForest | validation |                          4 |               165 | 0.4797 | 0.6257 | -0.0693 | 165 |

## Interpretation

The revised design tests whether nonlinear models benefit from the short-term persistence signal, the six-occasion daily structure, longer lags at 12 and 18 occasions, and social-media history. A social-media predictor set is retained only when it improves rolling validation for a particular model and horizon; it is not forced into all models. Residual-target configurations directly test whether a nonlinear model can improve on persistence or previous-day seasonal forecasts rather than merely reproducing them.

Because the dataset contains only seven planned days per participant, all tree models are deliberately regularized and the validation-to-test gap remains essential for interpretation.