# Step 05 — Practical Interpretation of Model Group 2 Results

## 1. What Model Group 2 contains

Model Group 2 consists of three nonlinear tree-based forecasting methods:

- Random Forest;
- Gradient Boosting Machine (GBM);
- XGBoost.

Each method was trained separately for the six positions of the next day. Therefore, every algorithm produced six horizon-specific forecasts rather than one generic forecast for all future measurements.

The models were allowed to use different representations of the available history, including recent AffWB values, summaries of the previous 3, 6, 12, and 18 occasions, selected seasonal lags, and active and passive social-media-use variables. They were also tested both as direct predictors of future AffWB and as models of deviations from simple persistence or previous-day baselines.

Configuration selection was performed using rolling validation on Days 4–6. Day 7 was used only for the final test.

## 2. Main result of Model Group 2

XGBoost was the strongest model in this group for forecasting the complete six-value trajectory of the next day.

| Model | Test MAE | Test RMSE | Test Bias |
|---|---:|---:|---:|
| XGBoost | 0.721 | 0.979 | -0.018 |
| GBM | 0.757 | 0.991 | -0.055 |
| Random Forest | 0.762 | 0.979 | -0.180 |

XGBoost therefore produced the smallest average absolute error among the tree models. Its overall bias was also close to zero, meaning that its overpredictions and underpredictions were approximately balanced across the complete test set.

The differences were not extremely large. XGBoost improved MAE by approximately 0.04 AffWB points relative to Random Forest and by approximately 0.04 points relative to GBM. The result supports XGBoost as the best tree-based method in this dataset, but it does not indicate a dramatic superiority over the other two tree algorithms.

## 3. What the models learned about the useful history

The clearest and most consistent finding was that the recent history of AffWB was more useful than a large and complex collection of predictors.

For all six XGBoost horizons and all six Random Forest horizons, rolling validation selected the same general input structure:

- the six most recent AffWB occasions;
- summary information calculated from the recent 3 and 6 occasions;
- no social-media predictors;
- direct prediction of the future AffWB value.

GBM selected the same mood-only, previous-six-occasion structure for five of the six horizons. Only the fifth horizon selected a configuration that also contained social-media history and predicted a residual relative to persistence.

This isolated exception is not sufficient evidence that social-media use had a stable predictive effect. Across the group as a whole, the selected models overwhelmingly preferred the participant's recent mood history. The tree results therefore support the same broad conclusion as ARIMAX: active and passive social-media-use variables did not provide a consistent improvement beyond the history of AffWB itself.

## 4. Interpretation of the most important features

Across Random Forest, GBM, and XGBoost, the most frequently important variables were:

- the participant's mean AffWB across the available history;
- the mean AffWB during the previous six occasions;
- the mean, minimum, maximum, and variability during the previous three or six occasions;
- selected recent AffWB lags, especially lags 1, 4, 5, and 6;
- the recent slope or direction of change in AffWB for some horizons.

This means that the models mainly used two kinds of information.

First, they used the participant's general mood level. A participant who had generally reported higher AffWB tended also to receive a higher next-day forecast.

Second, they used the participant's recent mood state. The average, minimum, maximum, and variability over approximately the previous day helped the models adjust the forecast above or below the participant's longer-term level.

The tree models did not identify one single lag that dominated all six future positions. Different individual lags became useful for different horizons, but the broader previous-six-occasion window was selected much more consistently than isolated lags 12 or 18. This suggests that the recent daily mood profile was more informative than relying only on a single value from two or three days earlier.

Feature importance must not be interpreted as a causal effect. It shows that a variable was useful for prediction within a fitted model, but it does not show that changing that variable would cause a change in mood. Standard tree importance also does not directly show whether a higher value increases or decreases the prediction.

## 5. Forecasting the next mood measurement

For the first forecast horizon, corresponding to the nearest mood measurement of the next day, the tree-model test MAEs were:

- GBM: 0.655;
- XGBoost: 0.663;
- Random Forest: 0.688.

GBM was therefore marginally the best tree model for the nearest next measurement. However, the simple persistence baseline remained substantially better for this specific task, with an MAE of 0.457.

The interpretation is that the nearest mood measurement is strongly influenced by immediate mood continuity. A complex tree model did not improve on the simple rule that the next mood is likely to be close to the last observed mood.

## 6. Forecasting all six measurements of the next day

For the complete six-position trajectory, XGBoost was the best tree model with a test MAE of 0.721.

Its horizon-specific errors were:

| Horizon | XGBoost Test MAE |
|---:|---:|
| 1 | 0.663 |
| 2 | 0.789 |
| 3 | 0.648 |
| 4 | 0.704 |
| 5 | 0.756 |
| 6 | 0.755 |

The third and first positions were the easiest for XGBoost, while the second, fifth, and sixth positions were more difficult. Forecast accuracy therefore did not decrease in a perfectly linear manner from horizon 1 to horizon 6. Predictability also depended on the position within the next-day sequence.

Compared with the Step 04 baselines, XGBoost was slightly worse than SARIMA for overall trajectory MAE (0.721 versus 0.713) and worse than persistence (0.721 versus 0.688). Therefore, the nonlinear tree model did not clearly outperform the strongest simpler methods for the complete six-value trajectory.

Its near-zero bias is nevertheless useful: unlike Random Forest, which tended to underpredict mood, XGBoost did not show a strong systematic tendency in either direction at the overall level.

## 7. Forecasting the mean mood of the next day

When the six forecasts were combined into a predicted next-day mean, XGBoost again produced the best result within Model Group 2:

| Model | Next-Day Mean Test MAE |
|---|---:|
| XGBoost | 0.495 |
| GBM | 0.514 |
| Random Forest | 0.551 |

This is an important positive result for the tree models. XGBoost's daily-mean MAE was lower than the corresponding values for SARIMA, persistence, participant mean, and ARIMAX.

The likely reason is that averaging the six forecasts reduces some position-specific errors. Even when individual future observations are difficult to predict exactly, the model can still estimate the participant's general mood level for the next day with greater accuracy.

## 8. What was found about social-media predictors

The models did not find robust evidence that active or passive social-media use improves next-day mood prediction.

Only one of the eighteen final horizon-specific tree configurations used social-media variables: GBM for horizon 5. All six XGBoost models and all six Random Forest models selected mood-only inputs, as did five of the six GBM models.

Social-media variables therefore did not show a repeatable contribution across algorithms or forecast horizons. The one selected GBM configuration should be interpreted as a local validation result rather than evidence of a general relationship.

This finding concerns predictive usefulness, not the absence of any psychological association. The dataset may contain weak, heterogeneous, delayed, or participant-specific relationships that are not strong or stable enough to improve out-of-sample forecasting.

## 9. What was found about residual forecasting

Most selected models predicted the future AffWB value directly. Configurations designed to predict corrections to persistence or previous-day seasonal forecasts were generally not selected.

This indicates that, for these tree algorithms and this dataset, it was more effective to learn a direct nonlinear mapping from recent mood history to the future value than to model only the error of a simple baseline.

The sole exception was the GBM model for horizon 5, which selected a persistence-residual target. Because this pattern was not repeated across the other horizons or algorithms, it does not support residual forecasting as the preferred general strategy for Model Group 2.

## 10. Model-specific interpretation

### Random Forest

Random Forest provided stable forecasts but was the weakest model in Group 2 by test MAE. Its overall bias of -0.180 indicates a noticeable tendency to predict mood below the observed value. Its feature importance was strongly concentrated on the participant's historical mean and the mean of the previous six occasions.

### GBM

GBM slightly improved on Random Forest and achieved the best tree-based result for the nearest next measurement. It also relied mainly on recent mood summaries. Its overall bias was small, although it underpredicted the first two horizons and slightly overpredicted later horizons.

### XGBoost

XGBoost was the strongest and most balanced tree model. It achieved the best overall trajectory MAE and the best next-day-mean MAE within the group. Its importance patterns were more distributed across recent means, minima, maxima, variability, and individual lags. This suggests that XGBoost captured more detailed nonlinear combinations of the recent mood profile than the other tree models.

## 11. Overall conclusion for Model Group 2

The tree-based models found that the most useful information for forecasting next-day affective well-being was the participant's own recent mood history, particularly the overall historical level and summaries of approximately the previous six measurement occasions. XGBoost was the best tree-based method, with a trajectory test MAE of 0.721 and a next-day-mean MAE of 0.495. However, it did not outperform persistence or SARIMA for the complete six-value trajectory, and it did not outperform persistence for the nearest next mood measurement.

The most valuable result of Model Group 2 is therefore not evidence that complex tree models dominate simpler approaches. Instead, the models show that nonlinear combinations of recent mood level, range, variability, and short-term history can provide a useful and relatively unbiased prediction, particularly for the average mood of the next day. Social-media predictors and longer isolated lags did not produce a stable improvement, while recent mood information remained the dominant predictive signal.

## 12. Client-ready summary

Among the nonlinear tree-based models, XGBoost provided the strongest overall performance. The models consistently showed that a participant's recent mood history—especially the average and variability across approximately the previous day—was more useful than social-media-use variables or isolated longer-term lags. Tree models did not improve on the strongest simple baselines for predicting the nearest measurement or the full six-value trajectory, but XGBoost was comparatively effective for predicting the participant's average mood on the following day. These results suggest that next-day affective well-being is driven primarily by recent mood continuity and the participant's current mood level, while the available social-media measures add little stable out-of-sample predictive value.
