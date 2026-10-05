# Selected Model Card — LSTM

## Intended use

Forecast the six AffWB assessments of the next planned study day from occasion-level history available at the end of the preceding day.

## Selection

The model was selected by rolling-validation macro MAE across target Days 4–6. Day 7 was not used for tuning.

## Selected configuration

```json
{
  "sequence_spec": "L06",
  "predictor_set": "mood_only",
  "target_mode": "absolute",
  "hidden_size": 32,
  "dropout": 0.2,
  "validation_macro_mae_three_seed": 0.7418096887961477,
  "validation_macro_mae_five_seed": 0.7436837378625013,
  "final_epochs": 19,
  "final_seeds": [
    42,
    43,
    44,
    45,
    46
  ]
}
```

## Final test performance

| Result                                         | Metric        |   Value |   N |
|:-----------------------------------------------|:--------------|--------:|----:|
| Primary six-position trajectory                | MAE           |  0.6274 | 292 |
| Primary six-position trajectory                | RMSE          |  0.8730 | 292 |
| Immediately next assessment (h=1)              | MAE           |  0.5119 |  46 |
| Derived next-day mean (>=4 observed positions) | MAE           |  0.4240 |  51 |
| Participant-clustered bootstrap                | MAE lower 95% |  0.5205 | 292 |
| Participant-clustered bootstrap                | MAE upper 95% |  0.7347 | 292 |

## Limitations

The participant records cover only seven planned days, exact timestamps are unavailable, some target positions are missing, and the results are specific to this dataset. The model supports prediction, not causal claims about social-media use or mood.