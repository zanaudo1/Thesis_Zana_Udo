# Step 06 — Revised RNN and LSTM Models

## Methodological revision

RNN and LSTM models now use three rolling-origin validation folds and directly output the six AffWB values of the next planned day. Candidate sequence lengths are 3, 6, 12, 18, and all available history. Mood-only and mood-plus-social-media inputs are compared, as are absolute targets and residual targets relative to persistence and previous-day seasonal forecasts.

Each network uses one compact recurrent layer and six distinct output heads. Validation tuning averages predictions across three deterministic seeds, while the selected final configuration is refit with five seeds and evaluated as an ensemble on Day 7.

## Selected configurations

| Model   | SequenceSpec   | PredictorSet   | TargetMode   |   HiddenSize |   Dropout | Seeds        |   ValidationMacroMAE |   ValidationWorstFoldMAE |   ValidationFoldSD |   ValidationPooledRMSE |   MeanSeedPredictionSD |   MedianBestEpoch |   ValidationObservedTargets |   Day4MAE |   Day5MAE |   Day6MAE |   Selected |
|:--------|:---------------|:---------------|:-------------|-------------:|----------:|:-------------|---------------------:|-------------------------:|-------------------:|-----------------------:|-----------------------:|------------------:|----------------------------:|----------:|----------:|----------:|-----------:|
| LSTM    | L06            | mood_only      | absolute     |           32 |    0.2000 | [42, 43, 44] |               0.7418 |                   0.8086 |             0.0586 |                 1.0826 |                 0.0723 |                20 |                         929 |    0.8086 |    0.7508 |    0.6660 |          1 |
| RNN     | all_history    | mood_only      | absolute     |           32 |    0.2000 | [42, 43, 44] |               0.7608 |                   0.8208 |             0.0461 |                 1.0943 |                 0.1002 |                12 |                         929 |    0.8208 |    0.7531 |    0.7086 |          1 |

## Overall validation and test metrics

| Model   | Split      |    MAE |   RMSE |    Bias |   N |
|:--------|:-----------|-------:|-------:|--------:|----:|
| LSTM    | test       | 0.6274 | 0.8730 |  0.0136 | 292 |
| RNN     | test       | 0.6548 | 0.8861 | -0.0198 | 292 |
| LSTM    | validation | 0.7470 | 1.0834 |  0.0858 | 929 |
| RNN     | validation | 0.7601 | 1.0913 |  0.0749 | 929 |

## Derived next-day mean metrics

| Model   | Split      |   MinimumObservedPositions |   ParticipantDays |    MAE |   RMSE |    Bias |
|:--------|:-----------|---------------------------:|------------------:|-------:|-------:|--------:|
| LSTM    | test       |                          4 |                51 | 0.4240 | 0.5742 | -0.0302 |
| RNN     | test       |                          4 |                51 | 0.4403 | 0.5748 | -0.0650 |
| LSTM    | validation |                          4 |               165 | 0.4474 | 0.6098 |  0.0772 |
| RNN     | validation |                          4 |               165 | 0.4530 | 0.6214 |  0.0687 |

## Interpretation

The recurrent models are evaluated against strong persistence and seasonal statistical baselines rather than assumed to outperform them. Selection considers mean rolling-validation error, the worst validation day, between-fold variability, and prediction variability across seeds. Given the short seven-day participant records, a complex recurrent model that does not improve on simple baselines is a substantive result rather than a pipeline failure.