# Step 04 Revised ARIMA/ARIMAX Statistical Baseline Report

## Forecasting task

The models forecast the six AffWB measurements of the next planned ESM day. The six-position trajectory is the primary output, horizon 1 is the immediately next measurement, and the next-day mean is derived from the same observed forecast positions.

## Methodological revision

- 91 ARIMA/SARIMA response structures were evaluated with rolling validation on target Days 4-6.
- The non-seasonal search included pure AR orders through p=12, MA orders through q=6, ordinary differencing d=0/1, and bounded mixed structures.
- Seasonal models used period 6 and included seasonal AR, MA, and differencing terms, including a P=2 lag-12 structure.
- 318 joint horizon-specific ARIMAX/SARIMAX-X configurations were evaluated after the broad rolling lag screen.
- Social-media lags were considered through lag 18, including individual lags, lag combinations, cumulative windows, and correlation-guided patterns.
- ARIMAX lag patterns were tuned separately for horizons 1-6. Only predictor values known before the forecasted day were allowed.
- Day 7 was not used for response-order, predictor, lag, preprocessing, or convergence decisions.

## AffWB temporal diagnostics

- Lag 1: observed-pair within-person correlation=0.0485, mean individual ACF=0.1560, mean individual PACF=0.1509.
- Lag 6: observed-pair within-person correlation=0.0854, mean individual ACF=0.0529, mean individual PACF=-0.0028.
- Lag 12: observed-pair within-person correlation=0.0720, mean individual ACF=0.0166, mean individual PACF=-0.0205.
- Lag 18: observed-pair within-person correlation=0.0547, mean individual ACF=-0.0100, mean individual PACF=-0.0335.

## Social-media lag diagnostics

- OverallActiveUse: strongest development-data lags were lag 11 (-0.0635), lag 9 (-0.0620), lag 14 (-0.0615), lag 17 (-0.0519), lag 16 (-0.0501).
- OverallPassiveUse: strongest development-data lags were lag 14 (-0.0743), lag 16 (-0.0707), lag 10 (-0.0558), lag 9 (-0.0543), lag 15 (-0.0525).

These correlations are descriptive screening information. Rolling validation, not correlation magnitude alone, selected the final lag patterns.

## Selected ARIMA/SARIMA baseline

- Configuration: SARIMA(0,0,0)x(1,0,0)[6].
- Rolling-validation macro MAE: 0.7971.
- Rolling-validation pooled MAE: 0.7984.
- Validation coverage: 0.9914.
- Day 7 test MAE: 0.7126.
- Day 7 test RMSE: 0.9213.

## Selected direct ARIMAX/SARIMAX-X configurations

- h=1: SARIMA(0,0,0)x(0,0,2)[6]; exposure=active; pattern=lag_6; rolling-validation macro MAE=0.6948.
- h=2: SARIMA(0,0,0)x(0,0,2)[6]; exposure=passive; pattern=lag_6; rolling-validation macro MAE=0.6468.
- h=3: SARIMA(0,0,0)x(2,0,0)[6]; exposure=passive; pattern=lag_3; rolling-validation macro MAE=0.6923.
- h=4: SARIMA(0,0,0)x(1,0,0)[6]; exposure=passive; pattern=lag_12; rolling-validation macro MAE=0.7604.
- h=5: ARIMA(0,1,2); exposure=active; pattern=lag_6; rolling-validation macro MAE=0.7890.
- h=6: ARIMA(0,1,2); exposure=passive; pattern=lag_10; rolling-validation macro MAE=0.6547.

- Combined ARIMAX Day 7 test MAE: 0.8038.
- Combined ARIMAX Day 7 test RMSE: 1.2260.

## Transparent reference forecasts

- Persistence: Day 7 MAE=0.6884, RMSE=1.1864, coverage=1.0000.
- SeasonalNaive6: Day 7 MAE=0.7249, RMSE=1.2363, coverage=0.7842.
- ParticipantMean: Day 7 MAE=0.7391, RMSE=0.9339, coverage=1.0000.

## Derived next-day mean

- ARIMA: MAE=0.5503, RMSE=0.6703, participant-days=51.
- ARIMAX: MAE=0.5870, RMSE=0.8554, participant-days=51.
- Persistence: MAE=0.5709, RMSE=0.9621, participant-days=51.
- SeasonalNaive6: MAE=0.5009, RMSE=0.6936, participant-days=36.
- ParticipantMean: MAE=0.5793, RMSE=0.7050, participant-days=51.

## Participant-clustered bootstrap

- ARIMA: Day 7 MAE 95% interval [0.6138, 0.8078].
- ARIMAX: Day 7 MAE 95% interval [0.6573, 0.9620].
- ParticipantMean: Day 7 MAE 95% interval [0.6417, 0.8348].
- Persistence: Day 7 MAE 95% interval [0.4983, 0.8993].
- SeasonalNaive6: Day 7 MAE 95% interval [0.5648, 0.8841].

## Interpretation limitations

- The ESM positions are ordered, but exact timestamps and equal elapsed intervals are unavailable.
- Period-six seasonality represents the planned six-position daily structure, not exact clock-time seasonality.
- Participant histories are short; high-complexity candidates may fail or have poor coverage and are excluded from selection by prespecified quality thresholds.
- Lag selection is predictive rather than causal. A selected social-media lag does not establish that exposure caused later changes in AffWB.

## Practical interpretation and conclusions

### 1. Immediately next AffWB measurement (h=1)

For the first ESM measurement of the next day, the simplest persistence forecast performed best by MAE:

- Persistence: MAE=0.4565, RMSE=0.8209.
- Selected SARIMA: MAE=0.5947, RMSE=0.7806.
- Participant mean: MAE=0.6054, RMSE=0.7719.
- Selected ARIMAX: MAE=0.7461, RMSE=1.3314.

The practical interpretation is that the immediately next mood measurement is dominated by short-term mood inertia: the last observed AffWB value is often a very strong estimate of the next value. SARIMA had a slightly lower RMSE than persistence, which means that although persistence was more accurate on average by MAE, SARIMA produced fewer or smaller large errors.

### 2. Six-measurement trajectory of the next day

Across all six forecast positions of Day 7, the overall test results were:

- Persistence: MAE=0.6884, RMSE=1.1864.
- Selected SARIMA: MAE=0.7126, RMSE=0.9213.
- Participant mean: MAE=0.7391, RMSE=0.9339.
- Selected ARIMAX: MAE=0.8038, RMSE=1.2260.

Persistence therefore had the lowest average absolute error, but SARIMA had a substantially lower RMSE. This indicates a trade-off: persistence was more often very close to the observed value, while SARIMA was more stable and less likely to make very large errors.

For the selected SARIMA, the horizon-specific MAE values were:

- h=1: 0.5947.
- h=2: 0.7400.
- h=3: 0.7142.
- h=4: 0.7507.
- h=5: 0.6515.
- h=6: 0.8077.

Forecast accuracy did not deteriorate monotonically from h=1 to h=6, although the sixth measurement was the most difficult forecast position on the final test day.

### 3. Derived mean AffWB of the next day

The next-day mean was calculated from the same forecast positions for which observed AffWB values were available. For participant-days with sufficient observed measurements, the results were:

- Selected SARIMA: MAE=0.5503, RMSE=0.6703, participant-days=51.
- Persistence: MAE=0.5709, RMSE=0.9621, participant-days=51.
- Participant mean: MAE=0.5793, RMSE=0.7050, participant-days=51.
- Selected ARIMAX: MAE=0.5870, RMSE=0.8554, participant-days=51.

SARIMA was therefore the best full-coverage model for the derived average mood of the next day. SeasonalNaive6 achieved a lower MAE of 0.5009, but it covered only 36 participant-days, so it is not directly comparable with the models evaluated on all 51 eligible participant-days.

### 4. Overall substantive interpretation

The statistical baseline models support the following conclusions:

1. The immediately next AffWB value is strongly influenced by short-term persistence. A participant's most recent observed mood is difficult to outperform for h=1.
2. For the complete six-position next-day trajectory, persistence remains a strong baseline, but the seasonal SARIMA model produces more stable forecasts and fewer large errors.
3. The period-six seasonal component is useful for forecasting the average mood of the next day, suggesting that approximately corresponding positions in consecutive study days contain predictive information.
4. Adding lagged social-media-use predictors through ARIMAX did not improve independent Day 7 performance. The ARIMAX validation advantage did not generalize to the test day.
5. The selected SARIMA showed a negative test bias of approximately -0.20 for individual measurements and approximately -0.25 for the next-day mean, indicating a tendency to slightly underestimate future AffWB.
6. Differences between SARIMA and persistence are not decisive because their participant-clustered bootstrap intervals overlap substantially. Persistence is preferable by MAE for individual measurements, whereas SARIMA is preferable by RMSE and for the derived next-day mean.

The most defensible conclusion is therefore not that one statistical model is universally superior, but that different aspects of the forecasting task favor different baselines: persistence is strongest for the immediately next and average individual measurement error, while SARIMA is stronger for forecast stability and the next-day mean.

