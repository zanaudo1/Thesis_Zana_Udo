# Step 07 — Revised Model Comparison

## Objective

Compare statistical, nonlinear tree-based, recurrent, and naive models under one common evaluation framework without using Day 7 for hyperparameter tuning.

## Final selection rule

The deployable/final model is selected by rolling-validation macro MAE. A model must cover all three validation periods and achieve at least 95% prediction coverage in validation and test. Day 7 is then used once to estimate generalization performance.

## Outputs

- complete and common-target prediction tables;
- validation and test metrics;
- six-horizon results;
- participant-level results;
- derived next-day mean results;
- participant-clustered bootstrap intervals and paired differences;
- final validation-based ranking;
- model-comparison report and figures.
