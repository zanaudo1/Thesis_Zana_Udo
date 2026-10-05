# Step 07 — Revised Model Comparison

## Selection rule

The final model was selected using rolling-validation macro MAE, not Day 7. Eligibility required predictions for all three validation days and at least 95% validation and test coverage. Day 7 was then used once for the independent final evaluation. Models with lower coverage remain in the descriptive comparison but were not eligible for final selection.

The validation-selected model is **LSTM** with rolling-validation macro MAE **0.7437**. Its untouched Day 7 test MAE is **0.6274** and test RMSE is **0.8730**.

The lowest descriptive test MAE was also obtained by **LSTM** (0.6274); therefore the validation-based choice and the test ranking agree in this run.

## Validation and test comparison

| Model           |   ValidationMAE |   ValidationMacroMAE |   ValidationRMSE |   ValidationCoverage |   ValidationPeriods |   TestMAE |   TestRMSE |   TestBias |   TestCoverage |   TestObservedTargets |   EligibleForFinalSelection |   ValidationToTestGap |   ValidationRank |   TestRank |   SelectedByValidation |
|:----------------|----------------:|---------------------:|-----------------:|---------------------:|--------------------:|----------:|-----------:|-----------:|---------------:|----------------------:|----------------------------:|----------------------:|-----------------:|-----------:|-----------------------:|
| LSTM            |          0.7470 |               0.7437 |           1.0834 |               1.0000 |                   3 |    0.6274 |     0.8730 |     0.0136 |         1.0000 |                   292 |                           1 |               -0.1163 |           1.0000 |     1.0000 |                      1 |
| RNN             |          0.7601 |               0.7579 |           1.0913 |               1.0000 |                   3 |    0.6548 |     0.8861 |    -0.0198 |         1.0000 |                   292 |                           1 |               -0.1032 |           2.0000 |     2.0000 |                      0 |
| XGBoost         |          0.7792 |               0.7760 |           1.1071 |               1.0000 |                   3 |    0.7211 |     0.9785 |    -0.0181 |         1.0000 |                   292 |                           1 |               -0.0549 |           3.0000 |     5.0000 |                      0 |
| SARIMA          |          0.7984 |               0.7971 |           1.1368 |               0.9914 |                   3 |    0.7126 |     0.9213 |    -0.2012 |         1.0000 |                   292 |                           1 |               -0.0844 |           4.0000 |     4.0000 |                      0 |
| GBM             |          0.8172 |               0.8134 |           1.1265 |               1.0000 |                   3 |    0.7571 |     0.9907 |    -0.0550 |         1.0000 |                   292 |                           1 |               -0.0563 |           5.0000 |     8.0000 |                      0 |
| RandomForest    |          0.8203 |               0.8174 |           1.1170 |               1.0000 |                   3 |    0.7623 |     0.9786 |    -0.1800 |         1.0000 |                   292 |                           1 |               -0.0551 |           6.0000 |     9.0000 |                      0 |
| ParticipantMean |          0.8218 |               0.8204 |           1.1371 |               1.0000 |                   3 |    0.7391 |     0.9339 |    -0.2092 |         1.0000 |                   292 |                           1 |               -0.0813 |           7.0000 |     7.0000 |                      0 |
| Persistence     |          0.9085 |               0.9055 |           1.4428 |               1.0000 |                   3 |    0.6884 |     1.1864 |     0.0445 |         1.0000 |                   292 |                           1 |               -0.2172 |           8.0000 |     3.0000 |                      0 |
| SeasonalNaive6  |          0.8449 |               0.8435 |           1.4450 |               0.8816 |                   3 |    0.7249 |     1.2363 |    -0.0349 |         0.7842 |                   292 |                           0 |               -0.1186 |         nan      |     6.0000 |                      0 |
| ARIMAX          |          0.7071 |               0.7071 |           1.1232 |               0.8918 |                   2 |    0.8038 |     1.2260 |    -0.2688 |         0.9932 |                   292 |                           0 |                0.0967 |         nan      |    10.0000 |                      0 |

## Common-target test comparison (N = 292 observed positions)

| Model           |    MAE |   RMSE |    Bias |   ObservedTargets |   PredictedTargets |   Coverage |
|:----------------|-------:|-------:|--------:|------------------:|-------------------:|-----------:|
| LSTM            | 0.6274 | 0.8730 |  0.0136 |               292 |                292 |     1.0000 |
| RNN             | 0.6548 | 0.8861 | -0.0198 |               292 |                292 |     1.0000 |
| Persistence     | 0.6884 | 1.1864 |  0.0445 |               292 |                292 |     1.0000 |
| SARIMA          | 0.7126 | 0.9213 | -0.2012 |               292 |                292 |     1.0000 |
| XGBoost         | 0.7211 | 0.9785 | -0.0181 |               292 |                292 |     1.0000 |
| ParticipantMean | 0.7391 | 0.9339 | -0.2092 |               292 |                292 |     1.0000 |
| GBM             | 0.7571 | 0.9907 | -0.0550 |               292 |                292 |     1.0000 |
| RandomForest    | 0.7623 | 0.9786 | -0.1800 |               292 |                292 |     1.0000 |

## Derived next-day mean comparison

| Model           | Split   |   MinimumObservedPositions |   ParticipantDays |    MAE |   RMSE |    Bias |
|:----------------|:--------|---------------------------:|------------------:|-------:|-------:|--------:|
| LSTM            | test    |                          4 |                51 | 0.4240 | 0.5742 | -0.0302 |
| RNN             | test    |                          4 |                51 | 0.4403 | 0.5748 | -0.0650 |
| XGBoost         | test    |                          4 |                51 | 0.4954 | 0.6516 | -0.0738 |
| SeasonalNaive6  | test    |                          4 |                36 | 0.5009 | 0.6936 | -0.0796 |
| GBM             | test    |                          4 |                51 | 0.5143 | 0.6610 | -0.1063 |
| SARIMA          | test    |                          4 |                51 | 0.5503 | 0.6703 | -0.2477 |
| RandomForest    | test    |                          4 |                51 | 0.5512 | 0.6868 | -0.2303 |
| Persistence     | test    |                          4 |                51 | 0.5709 | 0.9621 | -0.0127 |
| ParticipantMean | test    |                          4 |                51 | 0.5793 | 0.7050 | -0.2648 |
| ARIMAX          | test    |                          4 |                51 | 0.5870 | 0.8554 | -0.2919 |

## Main interpretation

The revised LSTM improved on the statistical, tree-based, and naive baselines for the complete six-value Day 7 trajectory. Its selected configuration used the most recent six mood observations only; social-media variables were not selected. This is consistent with Step 04, where adding social-media predictors through ARIMAX did not generalize to the test period.

Persistence remained a strong benchmark, particularly for the immediately next assessment, while SARIMA continued to provide a relatively stable seasonal benchmark. The comparison therefore supports a limited conclusion: the compact LSTM captured useful nonlinear information in the recent mood sequence, but the gain must be interpreted in light of the small seven-day record and bootstrap uncertainty.