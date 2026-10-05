# Step 08 — Final Comparative Interpretation: Next-Day Mood Prediction and the Findings of Beyens et al. (2020)

## 1. Purpose of this report

This report integrates the results of the revised time-series forecasting pipeline with the findings reported by Beyens et al. (2020), *The effect of social media on well-being differs from adolescent to adolescent*.

The comparison is useful because both analyses use the same intensive longitudinal experience-sampling data, but they answer different scientific questions:

- **Beyens et al. (2020)** examined whether social-media use during the previous hour was associated with **current momentary affective well-being**, and whether these associations differed between adolescents.
- **The revised pipeline** examined whether already observed mood and social-media history could **predict the six AffWB measurements of the next day**, the immediately next measurement, and the next-day mean.

The original article is therefore an explanatory association study, whereas the revised pipeline is an out-of-sample forecasting study. The two sets of findings should be interpreted as complementary rather than as direct replications or contradictions.

## 2. Main conclusion

The clearest and most stable predictor of next-day affective well-being was **the participant's own recent mood history**.

The final validation-selected model was a compact LSTM that used only the **six most recent AffWB observations**, approximately the previous planned day. It did not use active or passive social-media history. On the untouched Day 7 test set, it achieved:

| Forecasting objective | Test result |
|---|---:|
| Six-position next-day trajectory MAE | **0.627** |
| Six-position next-day trajectory RMSE | **0.873** |
| Immediately next measurement MAE | **0.512** |
| Derived next-day mean MAE | **0.424** |

For the immediately next measurement, the simple persistence rule was even stronger, with MAE **0.457**. This means that the next mood report was often close to the last observed mood report. For the complete next-day trajectory and the next-day mean, the LSTM produced the best numerical results.

The social-media variables occasionally appeared useful during ARIMAX validation at particular lags, but these patterns did not generalize to the final test day. ARIMAX test MAE was **0.804**, compared with **0.713** for SARIMA without social-media predictors and **0.627** for the mood-only LSTM. Tree and recurrent model tuning also selected mood-only configurations. Therefore, the revised analysis did not find stable evidence that aggregate active or passive social-media history improved next-day mood prediction beyond recent mood itself.

## 3. What predicts next-day mood in the revised pipeline

### 3.1 Immediately previous mood: lag 1

The strongest simple signal for the immediately next assessment was the last observed AffWB value.

The persistence model applies the rule:

> the next mood value will be approximately equal to the most recent observed mood value.

This model obtained the lowest Day 7 MAE for horizon 1: **0.457**. The result supports a positive short-term continuity pattern: relatively high mood is commonly followed by relatively high mood, and relatively low mood is commonly followed by relatively low mood.

Because the surveys were semi-random within daily time windows and exact timestamps were unavailable, lag 1 means the immediately previous planned or completed ESM occasion, not a fixed number of hours.

### 3.2 The most recent six mood observations: lags 1–6

The validation-selected LSTM used the six most recent AffWB observations and predicted all six values of the next day jointly. This sequence corresponds approximately to one previous planned day.

The selected configuration indicates that the following information was sufficient for the best overall forecast:

- the recent mood level;
- the order in which mood rose or fell across the previous six occasions;
- short-term variability across those occasions;
- nonlinear combinations of these recent values.

The LSTM does not provide a separate conventional coefficient for each lag. Therefore, the result does **not** mean that every lag from 1 to 6 had an equal effect. It means that the ordered six-value sequence, considered as a whole, contained the strongest validated predictive information.

### 3.3 Previous-day seasonal position: lag 6

The best statistical model without external predictors was:

`SARIMA(0,0,0) × (1,0,0)[6]`.

Its seasonal period of 6 means that the model tested whether mood at a given position was related to mood approximately six occasions earlier, corresponding to a similar ordinal position on the previous planned day.

This result provides moderate evidence of daily recurrence:

> mood at a particular part of the next day is partly predictable from mood in a roughly corresponding part of the previous day.

The seasonal effect was useful enough to win statistical-model validation, but it was not overwhelmingly strong. SARIMA test MAE was **0.713**, and the simpler persistence model had test MAE **0.688**. The six-occasion pattern should therefore be described as a useful structure, not as a deterministic daily cycle.

### 3.4 Longer mood lags: 12 and 18

The descriptive ACF, PACF, and within-person lag-correlation diagnostics showed weak local increases near lags 6, 12, and 18. In planned-day terms, these correspond approximately to one, two, and three days earlier.

However:

- lag 6 received the clearest support through the selected SARIMA model;
- lag 12 was weaker and did not become part of the final mood-only forecasting model;
- lag 18 remained a candidate diagnostic pattern rather than a validated final effect.

The analysis therefore does not support a strong claim that mood two or three days earlier independently predicts the next day after recent mood has been considered.

### 3.5 Recent mood summaries in the tree models

XGBoost was the strongest tree-based method, with test MAE **0.721**. Its six horizon-specific models all selected the `day_1_6` mood-only feature set.

The most useful tree features were repeatedly drawn from:

- mean mood across the last 3 or 6 observations;
- minimum and maximum mood across those windows;
- mood variability, especially the recent standard deviation;
- individual recent lags, commonly including lags 1, 4, 5, and 6;
- the participant's expanding historical mood mean.

This supports the conclusion that next-day mood depends not only on one isolated previous value but also on the recent **level, range, variability, and pattern** of mood.

Feature importance does not establish a positive or negative causal effect. For example, the importance of recent variability means that variability helped separate predictions, but it does not by itself prove that greater variability always raises or lowers next-day mood.

### 3.6 Participant's general mood level

The participant-mean baseline achieved test MAE **0.739**. It was weaker than LSTM, persistence, and SARIMA, but it was still reasonably competitive.

This indicates that adolescents differ in their typical AffWB level, and this stable personal level is useful for forecasting. However, the improvement of LSTM over the participant mean shows that recent within-person dynamics add information beyond knowing whether a participant is generally happier or less happy across the week.

## 4. What was found about social-media predictors and their lags

### 4.1 Predictors included in the forecasting pipeline

The forecasting pipeline constructed two aggregate exposure variables consistent with the overall-use logic of the article:

- overall active social-media use;
- overall passive social-media use.

Each represented reported minutes across the participant's personalized platform set. These historical predictors were tested alone and in combination with mood history in ARIMAX, tree models, RNN, and LSTM configurations.

### 4.2 Lags selected during ARIMAX validation

The horizon-specific ARIMAX tuning selected the following social-media predictor and lag combinations:

| Future horizon | Validation-selected social-media predictor | Selected lag | Approximate interpretation |
|---:|---|---:|---|
| h=1 | Active use | 6 | roughly the corresponding position one day earlier |
| h=2 | Passive use | 6 | roughly the corresponding position one day earlier |
| h=3 | Passive use | 3 | approximately three prior ESM occasions |
| h=4 | Passive use | 12 | roughly the corresponding position two days earlier |
| h=5 | Active use | 6 | roughly one day earlier |
| h=6 | Passive use | 10 | approximately one day plus four ESM occasions earlier |

These selections show that forcing one universal predictor lag would have been inappropriate. Different future positions could favor different historical offsets during validation.

However, these are **exploratory validation selections**, not stable substantive effects. The combined ARIMAX strategy deteriorated from validation to test and produced test MAE **0.804**. It also had lower validation coverage than the fully eligible models. The selected lag tells us where a predictor was useful for minimizing validation error; it does not establish the direction of an effect, statistical significance, or causality.

### 4.3 Why the social-media lag patterns were not retained as main findings

Three independent parts of the pipeline reached the same conclusion:

1. ARIMAX with lagged social-media predictors performed worse on Day 7 than SARIMA without them.
2. XGBoost selected mood-only features for all six horizons.
3. The final LSTM selected mood-only input; mood-plus-social-media configurations did not improve rolling validation.

Therefore, the robust forecasting conclusion is:

> aggregate active and passive social-media history did not add stable next-day predictive information beyond recent mood history in this sample.

This does not mean that social media has no psychological effect. It means that the measured exposure variables, at the tested lags and with only seven planned days per participant, did not improve out-of-sample next-day forecasts.

## 5. What Beyens et al. (2020) found

### 5.1 Different time relationship

At each ESM assessment, participants reported:

- how happy they felt **at that moment**;
- how much time they had spent actively and passively using social media **during the previous hour**.

The article's models therefore examined a near-contemporaneous relationship between past-hour use and current mood. They did not test whether social-media use predicted the following day, did not evaluate lags such as 3, 6, 10, or 12 occasions, and did not use train/validation/test forecasting metrics.

### 5.2 Overall social-media use

The original paper reported that:

- average categorical overall active or passive use versus no use was not significantly associated with current well-being;
- the duration of overall active use had a small positive average association with current well-being;
- the categorical association between overall passive use and well-being varied substantially between adolescents.

For the person-specific categorical passive-use association:

- **44.26%** of adolescents had no or only a very small association;
- **45.90%** had a positive association;
- **9.84%** had a negative association.

The main message was not that passive use has one universal effect. It was that adolescents differ considerably in their susceptibility.

### 5.3 Instagram

The paper found:

- a positive average categorical association between passive Instagram use and current well-being;
- no significant average categorical association for active Instagram use;
- no average dose-response association between duration of Instagram use and well-being;
- substantial person-specific heterogeneity in the association between passive Instagram duration and well-being.

For passive Instagram duration, most adolescents showed little or no association, while smaller groups showed positive or negative associations.

### 5.4 WhatsApp

The paper found:

- a positive average categorical association between passive WhatsApp use and current well-being;
- a positive average dose-response association for time spent passively reading WhatsApp messages;
- no significant average association for active WhatsApp use.

The article therefore identified some small positive immediate associations, particularly for passive WhatsApp use, while emphasizing that results depend on activity type, platform, and individual adolescent.

## 6. Side-by-side comparison

| Question | Beyens et al. (2020) | Revised forecasting pipeline |
|---|---|---|
| Primary objective | Explain within-person associations | Predict unseen next-day AffWB |
| Outcome timing | Current happiness | Six mood values on the following day |
| Exposure timing | Social-media use during previous hour | Historical mood and social-media lags before the day boundary |
| Main method | Multilevel fixed and random effects | SARIMA/ARIMAX, RF/GBM/XGB, RNN/LSTM |
| Individual heterogeneity | Explicit random slopes and person-specific effect sizes | Global predictive models using participant histories; no person-specific social-media slope estimation |
| Main mood predictor | Mood history was not the main research target | Recent mood history was the dominant predictor |
| Social-media finding | Some small positive immediate associations; strong heterogeneity for selected passive-use effects | No stable added value for next-day prediction |
| Lag conclusion | Previous-hour exposure paired with current mood; no next-day lag comparison | Lag 1 continuity and an approximate lag 6 daily structure were useful; social-media lags 3/6/10/12 were unstable |
| Evaluation | Coefficients, p-values, random-effect variance | Rolling-validation and untouched Day 7 MAE/RMSE |

## 7. How the two studies fit together

The results are compatible once the different time scales are recognized.

### 7.1 Immediate association does not guarantee next-day predictability

A teenager may feel slightly better while or shortly after reading positive messages, viewing enjoyable content, or actively communicating. This can produce a positive past-hour/current-mood association in the original multilevel models.

That short-lived association may disappear by the next day or be overwhelmed by stronger factors such as:

- the adolescent's current mood state;
- sleep, school, family, and peer events not measured in the dataset;
- regression toward the participant's typical mood level;
- different reactions to social media across adolescents.

Therefore, finding a small immediate association in the article and no stable next-day forecasting benefit in our models is not contradictory.

### 7.2 Heterogeneity can reduce pooled forecasting value

The article showed that passive social-media use was related to better mood for some adolescents, worse mood for others, and no meaningful change for many. When these opposite patterns are pooled into one global forecasting model, they can cancel each other.

Our mood-only LSTM may outperform social-media-enhanced models partly because recent mood already summarizes the net result of many unobserved and person-specific influences. Aggregate social-media exposure is a comparatively weak and heterogeneous signal.

### 7.3 Platform-specific effects may be hidden by aggregate predictors

The original article separately analyzed overall use, Instagram, and WhatsApp. The revised forecasting models primarily used aggregate active and passive exposure histories to maintain common coverage and a manageable model space.

Consequently, the forecasting analysis does not rule out platform-specific delayed effects. A positive passive WhatsApp signal could be diluted when combined with Instagram, Snapchat, YouTube, or game-chat exposure. Testing platform-specific next-day lags would require enough observations for each platform and a carefully restricted model family to avoid severe overfitting.

### 7.4 Recent mood is a summary of multiple causes

The final LSTM's reliance on the previous six mood observations does not imply that mood causes itself in a simple psychological sense. Recent AffWB is a compact summary of many influences that occurred before forecasting, including social contact, media experiences, daily events, individual baseline differences, and unmeasured contextual factors.

The predictive result is therefore:

> recent mood is the best available statistical summary for forecasting the next day.

It is not a claim that other experiences are unimportant.

## 8. Integrated interpretation of lag effects

The evidence can be summarized in three levels.

### Strongest and most reproducible evidence

- **Lag 1 mood:** strong short-term continuity, especially for the immediately next assessment.
- **Mood sequence across lags 1–6:** strongest overall input for the six-value next-day trajectory and next-day mean.
- **Recent mood summaries:** mean, minimum, maximum, and variability over the last 3–6 occasions were repeatedly useful in tree models.

### Moderate evidence

- **Lag 6 mood:** approximate previous-day positional recurrence, supported by the best SARIMA configuration and consistent with the six-observation LSTM history window.
- **Participant's historical mean:** useful baseline information, but weaker than recent dynamics.

### Weak or unstable evidence

- **Mood lags 12 and 18:** weak diagnostic recurrence but no stable independent final effect.
- **Active social-media lag 6:** selected for ARIMAX horizons 1 and 5 in validation, but not test-stable.
- **Passive social-media lags 3, 6, 10, and 12:** selected for specific ARIMAX horizons in validation, but not test-stable.
- **Social-media histories in nonlinear models:** not selected by the final XGBoost or LSTM configurations.

Because no exact timestamps are available, all lag interpretations refer to ordered ESM occasions. Lag 6 is approximately one planned day, but not exactly 24 hours.

## 9. Scientific conclusions

1. **Recent affective well-being is the primary predictor of next-day affective well-being.** The immediately previous mood is particularly informative for the next assessment, while the ordered previous six mood observations provide the strongest overall forecast of the following day.

2. **There is evidence of a weak daily structure.** The statistical model favored a seasonal lag of 6, indicating partial recurrence across corresponding positions of adjacent planned days.

3. **The recent mood distribution matters.** Tree models used recent averages, minima, maxima, variability, and selected individual lags, suggesting that next-day mood depends on both recent level and recent dynamics.

4. **Aggregate social-media history was not a robust next-day predictor.** Although ARIMAX selected several active and passive exposure lags during validation, the resulting model did not generalize to Day 7, and the strongest tree and recurrent models excluded social-media variables.

5. **The original article and the forecasting analysis answer different questions.** Beyens et al. found small and sometimes positive near-contemporaneous social-media associations, together with substantial person-specific heterogeneity. The revised pipeline found that these variables did not provide stable delayed forecasting value for the next day.

6. **The two findings are complementary.** Social-media use may be associated with current mood for some adolescents without being a reliable universal predictor of tomorrow's mood.

## 10. Limitations

The conclusions should be interpreted with the following constraints:

- only seven planned days were available per participant;
- exact timestamps and exact elapsed durations between occasions were unavailable;
- the test consisted of one held-out day;
- social-media effects may differ by participant and platform, while the forecasting predictors were mainly aggregate active/passive measures;
- LSTM predictions are nonlinear and do not yield simple causal coefficients for each lag;
- several model differences, especially LSTM versus persistence and SARIMA, had uncertainty intervals that included or narrowly included zero;
- unmeasured factors such as sleep, school events, family interactions, and offline peer experiences may explain additional next-day variation.

The results support forecasting conclusions within this dataset, not universal causal claims.

## 11. Thesis-ready final conclusion

The revised time-series analysis shows that adolescents' next-day affective well-being is predicted primarily by their own recent mood dynamics. The immediately next mood assessment is best approximated by the last observed mood, while the complete six-measurement trajectory and the next-day average are best predicted by an LSTM using the six most recent AffWB observations. A weaker daily recurrence is also present at lag 6, corresponding approximately to the same measurement position on the previous day. Recent mood level, range, and variability were more useful than aggregate active or passive social-media history. Although ARIMAX identified candidate social-media lags of 3, 6, 10, and 12 occasions for specific horizons during validation, these patterns did not generalize to the held-out test day and should not be treated as established effects.

These results complement Beyens et al. (2020). The original study found several small positive associations between social-media use during the previous hour and current well-being, especially for passive Instagram and WhatsApp use, while also showing substantial person-specific heterogeneity. Our analysis indicates that such immediate or individual-specific associations do not translate into stable population-level prediction of the following day's mood. Taken together, the findings suggest that social-media effects may be short-lived, platform-specific, and highly individual, whereas recent mood is the most reliable general predictor of next-day affective well-being in the available data.

## 12. Source files

Pipeline results used in this report:

- `04_statistical_models_report.md`
- `04_selected_arimax_horizon_configs.csv`
- `05_tree_models_results_interpretation.md`
- `05_tree_model_feature_importance.csv`
- `06_recurrent_models_results_interpretation.md`
- `07_model_comparison_report.md`
- `07_bootstrap_differences_from_selected.csv`
- `08_final_results_report.md`

Reference:

Beyens, I., Pouwels, J. L., van Driel, I. I., Keijsers, L., & Valkenburg, P. M. (2020). The effect of social media on well-being differs from adolescent to adolescent. *Scientific Reports, 10*, 10763. https://doi.org/10.1038/s41598-020-67727-7
