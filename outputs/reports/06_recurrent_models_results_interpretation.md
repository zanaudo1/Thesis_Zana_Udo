# Step 06 — Practical Interpretation of Model Group 3 Results

## 1. What Model Group 3 contains

Model Group 3 consists of two recurrent neural-network forecasting methods:

- a standard Recurrent Neural Network (RNN);
- a Long Short-Term Memory network (LSTM).

Unlike the horizon-specific tree models in Step 05, each recurrent model received an ordered sequence of past observations and produced the complete six-value AffWB trajectory for the next planned day in one forecast. The network used one recurrent layer followed by six separate output heads, one for each future position.

The tuning process compared:

- sequence lengths of 3, 6, 12, and 18 occasions, plus all available history;
- mood-only inputs versus mood plus active and passive social-media-use inputs;
- direct prediction of the six future AffWB values versus prediction of residuals from persistence or previous-day seasonal baselines;
- compact architectures with different hidden sizes and dropout levels.

Configuration selection used rolling validation on Days 4–6. Day 7 remained untouched until the final test. Candidate configurations were evaluated across three random seeds, and the selected final models were refitted and ensembled across five seeds.

## 2. Main result of Model Group 3

LSTM was the strongest recurrent model.

| Model | Test MAE | Test RMSE | Test Bias |
|---|---:|---:|---:|
| LSTM | 0.627 | 0.873 | 0.014 |
| RNN | 0.655 | 0.886 | -0.020 |

The LSTM reduced trajectory MAE by approximately 0.028 AffWB points relative to the RNN. The participant-clustered bootstrap comparison estimated the RNN-minus-LSTM MAE difference at 0.028, with a 95% interval from approximately 0.003 to 0.054. In this sample, the LSTM therefore showed a small but consistent advantage over the standard RNN.

Both models had overall bias close to zero. Across the full Day 7 trajectory, neither network showed a strong general tendency to overpredict or underpredict mood.

## 3. What configurations were selected

The validation-selected LSTM used:

- the six most recent AffWB observations;
- mood history only;
- direct prediction of the six future AffWB values;
- 32 hidden units;
- dropout of 0.20;
- a five-seed ensemble for the final model.

The validation-selected RNN used:

- all available AffWB history;
- mood history only;
- direct prediction of the six future AffWB values;
- 32 hidden units;
- dropout of 0.20;
- a five-seed ensemble for the final model.

The selected LSTM therefore summarized approximately the previous day of mood observations, whereas the selected RNN used the complete available sequence. However, the LSTM configuration using all available history had almost the same validation MAE as the selected six-observation configuration: 0.743 versus 0.742. The data therefore do not demonstrate that six observations are universally superior to longer history. They show that the shorter recent sequence was sufficient and marginally preferable under the predefined validation rule.

## 4. What the models learned about useful history

The strongest recurrent configurations relied only on the participant's own AffWB sequence. This supports three practical conclusions.

First, the order of recent mood observations contains useful predictive information. The recurrent models did better than models based only on a participant mean and also improved on the tested statistical and tree-based alternatives for the complete Day 7 trajectory.

Second, a relatively short sequence was sufficient for the LSTM. Its best configuration used the previous six occasions, corresponding approximately to one planned day of measurements. This is consistent with the earlier evidence that recent mood continuity and a six-position daily structure are important.

Third, the standard RNN benefited slightly more from all available history, but it remained less accurate than the LSTM. A plausible interpretation is that the LSTM's gated memory mechanism was better able to retain useful information and suppress less useful fluctuations. This is an architectural interpretation, not evidence of a psychological mechanism.

Unlike tree models, RNNs and LSTMs do not provide a simple feature-importance ranking for individual lags. Therefore, the results do not allow a defensible statement such as “lag 1 caused the improvement” or “lag 6 was the single dominant input.” The supported conclusion comes from model ablation and configuration selection: recent ordered mood history was useful, while adding the tested social-media channels was not.

## 5. What was found about social-media predictors

Neither selected recurrent model used active or passive social-media history.

Mood-plus-social-media configurations were tested, but they did not achieve the best rolling-validation performance. For example, screened LSTM and RNN configurations using social-media channels with persistence-residual targets had validation MAEs above 0.81, whereas the selected mood-only direct models achieved approximately 0.74 and 0.76 respectively.

The recurrent results therefore reinforce the findings from ARIMAX and the tree models: the available active and passive social-media variables did not add stable out-of-sample predictive value beyond the participant's recent mood history.

This remains a predictive conclusion. It does not prove that social-media use has no psychological association with mood, and it does not rule out weak, delayed, nonlinear, or participant-specific effects that this short dataset cannot estimate reliably.

## 6. What was found about direct versus residual forecasting

The selected RNN and LSTM both predicted the six future AffWB values directly.

Configurations that predicted corrections to persistence performed worse in screening, and configurations that predicted corrections to the previous-day seasonal forecast were weaker still. The best screened persistence-residual LSTM had validation MAE of approximately 0.810, compared with approximately 0.742 for the selected direct LSTM. Seasonal-residual configurations were close to 0.879.

This indicates that the recurrent networks worked better when allowed to learn the full mapping from the mood sequence to the next-day trajectory rather than being restricted to correcting a predefined baseline.

The result does not mean that persistence or seasonal structure was irrelevant. Those patterns can still be represented internally by the network. It means only that explicitly converting the target into a baseline residual did not improve validation accuracy.

## 7. Forecasting the immediately next mood measurement

For the first forecast horizon, the Day 7 MAEs were:

- LSTM: 0.512;
- RNN: 0.546.

The LSTM was therefore the better recurrent model for the nearest next assessment. However, the simple persistence baseline remained better for this single task, with MAE 0.457.

This result indicates that immediate mood continuity is especially strong. For the very next observation, the simple rule that mood will remain close to its last observed value was difficult for the neural networks to improve upon.

The LSTM's first-horizon bias was -0.300, meaning that it tended to underpredict the first observed mood of Day 7. Consequently, its near-zero overall bias should not be interpreted as perfectly balanced predictions at every future position.

## 8. Forecasting all six measurements of the next day

For the complete Day 7 trajectory, the LSTM achieved test MAE 0.627 and RMSE 0.873. The RNN achieved MAE 0.655 and RMSE 0.886.

The LSTM's horizon-specific test errors were:

| Horizon | LSTM Test MAE | LSTM Bias |
|---:|---:|---:|
| 1 | 0.512 | -0.300 |
| 2 | 0.716 | -0.120 |
| 3 | 0.552 | 0.111 |
| 4 | 0.649 | 0.013 |
| 5 | 0.620 | 0.136 |
| 6 | 0.697 | 0.204 |

The first and third positions were the easiest to predict. The second and sixth positions were the most difficult. Accuracy therefore did not deteriorate in a simple linear pattern as the forecast horizon increased; predictability also depended on the position within the next-day sequence.

The bias pattern changed across the day. The LSTM underpredicted the first two positions and overpredicted several later positions. These opposing errors largely cancelled, producing an overall bias close to zero. This is why both overall and horizon-specific bias should be reported.

The LSTM was better than the RNN at every one of the six test horizons. This consistency supports the group-level conclusion that the gated LSTM architecture handled the short mood sequences more effectively than the standard RNN.

## 9. Forecasting the mean mood of the next day

When the six forecasts were combined into a predicted next-day mean, the results were:

| Model | Next-Day Mean Test MAE | RMSE | Bias |
|---|---:|---:|---:|
| LSTM | 0.424 | 0.574 | -0.030 |
| RNN | 0.440 | 0.575 | -0.065 |

The LSTM was again the stronger recurrent model. It also produced the lowest next-day-mean MAE among all tested models in the revised pipeline.

The daily mean was easier to estimate than each individual future observation. Averaging the six predicted positions reduces some position-specific positive and negative errors and emphasizes the participant's general mood level for the following day.

This is one of the clearest practical strengths of the recurrent approach: even when the exact within-day trajectory remains difficult, the LSTM can provide a comparatively accurate estimate of the participant's overall next-day affective well-being.

## 10. Stability across validation folds and random seeds

The selected LSTM achieved rolling-validation macro MAE 0.742 during three-seed tuning. Its fold-specific MAEs were:

- Day 4: 0.809;
- Day 5: 0.751;
- Day 6: 0.666.

The selected RNN achieved validation macro MAE 0.761, with fold-specific MAEs of 0.821, 0.753, and 0.709.

Both models performed better on later validation days, possibly because more participant history was available or because the later days were intrinsically easier to predict. These data alone cannot separate those explanations.

The LSTM also showed lower variability in predictions across tuning seeds than the RNN: mean seed prediction standard deviation was approximately 0.072 for LSTM and 0.100 for RNN. The five-seed ensemble used for final prediction reduced dependence on any single random initialization.

Day 7 test error was lower than rolling-validation error for both models. This is encouraging, but it should not be interpreted as proof that future unseen days will always be easier. Day 7 may simply have been more predictable than the average validation day.

## 11. Comparison with the earlier model groups

For the complete six-position Day 7 trajectory, the recurrent models produced the two lowest MAEs in the full comparison:

| Model | Test MAE |
|---|---:|
| LSTM | 0.627 |
| RNN | 0.655 |
| Persistence | 0.688 |
| SARIMA | 0.713 |
| XGBoost | 0.721 |

The LSTM therefore improved trajectory MAE by approximately:

- 0.028 relative to RNN;
- 0.061 relative to persistence;
- 0.085 relative to SARIMA;
- 0.094 relative to XGBoost.

Participant-clustered bootstrap comparisons supported a clear LSTM advantage over RNN and XGBoost in this sample. The interval for the LSTM-versus-persistence difference included zero, and the interval against SARIMA narrowly included zero. Therefore, the LSTM's numerical advantage over the strongest simple baselines should be described as promising rather than definitive.

For the nearest next measurement, persistence remained best. For the complete trajectory and the derived daily mean, the LSTM was best. The preferred model therefore depends partly on the forecasting objective.

## 12. Model-specific interpretation

### RNN

The RNN used all available mood history and produced competitive forecasts. It achieved the second-best trajectory MAE in the full model comparison. However, it was less accurate than the LSTM at every horizon and showed greater variability across random seeds. The standard RNN appears capable of learning useful temporal dependence, but it was less effective at controlling what information to retain from the sequence.

### LSTM

The LSTM was the strongest model in Group 3 and the validation-selected final model. It used only the most recent six mood observations, indicating that a compact representation of approximately the previous day was sufficient. Its gated memory structure captured useful nonlinear temporal patterns beyond the tested statistical, tree-based, and naive approaches. It was particularly effective for the complete next-day trajectory and the next-day average, although it did not beat persistence for the immediately next observation.

## 13. Overall conclusion for Model Group 3

The recurrent models showed that the ordered sequence of a participant's recent mood contains predictive information that is not fully captured by participant averages, statistical seasonal terms, or manually constructed tree features. The LSTM was the strongest recurrent method and the strongest overall model in the revised pipeline, with Day 7 trajectory MAE 0.627 and next-day-mean MAE 0.424.

The selected LSTM relied only on the six most recent AffWB observations. Social-media variables, longer mandatory histories, and residual targets did not produce a stable validation improvement. The results therefore point to recent mood dynamics as the dominant predictive signal.

At the same time, the conclusions must remain cautious. The dataset contains only seven planned days per participant, the difference from persistence was not statistically decisive under participant-clustered bootstrap, and neural-network internals do not provide a simple causal interpretation of individual lags. The evidence supports improved prediction in this sample, not universal superiority of LSTM models or proof of a causal mood mechanism.

## 14. Client-ready summary

Among the recurrent neural-network models, the LSTM provided the strongest results. It predicted the next day's six mood measurements using only the participant's six most recent mood observations and achieved lower error than the RNN, statistical models, tree-based models, and simple baselines for the complete next-day trajectory. It also produced the most accurate estimate of average mood on the following day. The models did not find stable added value from active or passive social-media-use variables, and direct prediction worked better than predicting corrections to persistence or previous-day forecasts. Persistence remained the strongest approach for the immediately next measurement, so the LSTM's main advantage was in modelling the full next-day pattern rather than only the next single mood value.
