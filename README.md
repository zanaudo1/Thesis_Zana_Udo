# MixedDataAnalysis — Revised Next-Day AffWB Forecasting Pipeline

## Objective

This project forecasts the six momentary affective well-being (`AffWB`) measurements of the next planned study day from each participant's preceding occasion-level ESM history.

The primary target is the complete six-position next-day trajectory. Secondary outcomes are:

- the immediately next assessment (`h = 1`);
- the mean of the six next-day forecasts, evaluated against the mean of observed target positions.

## Models

The pipeline evaluates:

1. statistical and naive baselines: persistence, participant mean, previous-day seasonal naive, SARIMA, and ARIMAX;
2. tree models: Random Forest, GBM, and XGBoost;
3. sequence models: RNN and LSTM.

## Revised validation design

Hyperparameters are not selected on the final test day.

Rolling validation uses:

```text
Train target Days 2–3 → validate target Day 4
Train target Days 2–4 → validate target Day 5
Train target Days 2–5 → validate target Day 6
```

After model selection, the chosen configuration is fitted using development target Days 2–6 and evaluated once on target Day 7.

The final model is selected by rolling-validation macro MAE among models that:

- cover all three validation periods;
- have at least 95% validation coverage;
- have at least 95% test coverage.

Day 7 is not used for retuning.

## Step structure

### Step 01 — Data preparation

Reads the original SPSS/CSV data, preserves occasion-level records, audits ranges and missingness, and constructs explicit value, observed, availability, and status fields.

### Step 02 — Descriptive analysis

Produces distribution, participant, missingness, autocorrelation, temporal-position, platform-coverage, and lagged-correlation diagnostics.

### Step 03 — Modeling data preparation

Creates the canonical occasion-level modeling dataset and defines the next-day six-value target and chronological forecast origins.

### Step 04 — Statistical baselines

Performs revised ARIMA/SARIMA and ARIMAX tuning using rolling validation. It evaluates short and seasonal structures, including a six-occasion cycle and predictor lags up to 18.

### Step 05 — Tree models

Trains six horizon-specific models for each algorithm. The candidate set tests:

- short-term lags 1–3;
- one-day history 1–6;
- sparse seasonal combinations involving lags 1, 6, 12, and 18;
- mood-only and mood-plus-social-media predictors;
- absolute and residual targets relative to persistence and seasonal baselines.

### Step 06 — RNN and LSTM

Uses true ordered sequences and six separate output heads. It compares sequence lengths 3, 6, 12, 18, and all available history, mood-only versus mood-plus-social-media channels, and absolute versus residual targets. Multiple random seeds are averaged.

### Step 07 — Model comparison

Combines all predictions, applies the validation-based selection rule, calculates common-target test metrics, horizon and participant results, next-day mean results, and participant-clustered bootstrap uncertainty.

### Step 08 — Final results

Produces the selected-model summary, model card, final report, primary and secondary result tables, and diagnostic figures.

## Current executed result

The validation-selected model is a compact **LSTM** with:

- sequence length: last 6 planned occasions;
- predictors: mood history only;
- target: direct six-value AffWB trajectory;
- hidden units: 32;
- dropout: 0.2;
- five-seed final ensemble.

Its Day 7 results are:

- six-position trajectory MAE: **0.6274**;
- six-position trajectory RMSE: **0.8730**;
- immediately next assessment MAE: **0.5119**;
- derived next-day mean MAE: **0.4240**;
- participant-clustered 95% bootstrap interval for trajectory MAE: **0.5205–0.7347**.

The LSTM had the lowest eligible rolling-validation macro MAE (**0.7437**) and also the lowest descriptive Day 7 test MAE. Its selected input did not include social-media predictors. ARIMAX and the nonlinear social-media candidate configurations did not provide stable test improvement.

## Main scientific interpretation

Recent mood history contains useful predictive information for the next day's AffWB trajectory. The compact LSTM improved on the tested persistence, SARIMA, tree, and participant-mean baselines for the complete Day 7 trajectory. Persistence remained a strong benchmark, and its paired bootstrap difference from LSTM included zero, so the superiority claim should remain cautious.

The analysis supports prediction, not causal inference. Failure to improve forecasting with active or passive social-media history does not prove that social-media use has no causal effect on mood.

## Running

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python run_pipeline.py
```

Run only revised modeling and reporting steps:

```bash
python run_pipeline.py --from-step 5 --to-step 8
```

Run Step 06 sequentially for maximum reproducibility:

```bash
python 06_train_recurrent_models.py --n-jobs 1
```

See `HowToRun.txt` and each numbered Markdown document for more details.
