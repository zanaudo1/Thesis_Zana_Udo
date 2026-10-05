# Step 08 — Final Results

## Final model

The final model is **LSTM**, selected before test evaluation using the lowest eligible rolling-validation macro MAE (0.7437). The selected recurrent configuration used recent mood history without social-media predictors.

## Primary and secondary Day 7 results

| Result                                         | Metric        |   Value |   N |
|:-----------------------------------------------|:--------------|--------:|----:|
| Primary six-position trajectory                | MAE           |  0.6274 | 292 |
| Primary six-position trajectory                | RMSE          |  0.8730 | 292 |
| Immediately next assessment (h=1)              | MAE           |  0.5119 |  46 |
| Derived next-day mean (>=4 observed positions) | MAE           |  0.4240 |  51 |
| Participant-clustered bootstrap                | MAE lower 95% |  0.5205 | 292 |
| Participant-clustered bootstrap                | MAE upper 95% |  0.7347 | 292 |

## Six-position interpretation

Across 292 observed Day 7 target positions, the model achieved MAE 0.6274, RMSE 0.8730, and bias 0.0136. The immediately next assessment had MAE 0.5119. The derived next-day mean had MAE 0.4240 across 51 participant-days.

## Comparison with baselines

The selected model outperformed the persistence and SARIMA baselines on the complete six-position test trajectory. The next-lowest test MAE among the remaining models was RNN at 0.6548. Participant-clustered bootstrap differences are provided in `07_bootstrap_differences_from_selected.csv` and should be used to judge whether apparent gaps are robust.

## Predictor conclusion

Neither ARIMAX nor the selected tree/recurrent configurations provided stable evidence that active or passive social-media history improved Day 7 mood prediction. The final LSTM selected mood-only input. This is a predictive conclusion and does not establish the absence of causal social-media effects.

## Scientific conclusion

The revised pipeline indicates that the recent mood sequence contains useful information for predicting the next day's six AffWB measurements. A compact LSTM generalized better than the tested statistical and tree-based alternatives, while simple persistence remained a strong benchmark. Because the dataset is short and the uncertainty intervals overlap for some comparisons, the result should be presented as evidence of improved prediction in this sample rather than proof of universal superiority.