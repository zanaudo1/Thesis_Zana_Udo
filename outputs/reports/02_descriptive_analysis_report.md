# Step 02 — Descriptive and Temporal Data Audit Report

## 1. Scope

Step 02 describes the Step 01 canonical ESM dataset before model-specific processing. It retains each source row at the sequential occasion level, does not average observations by day, does not add artificial rows, and does not impute missing values.

The analysis treats `Occasion` as an ordered sequence index. `PlannedDayIndex` and `PlannedOccasionWithinDay` are used only for descriptive summaries based on the planned six-assessments-per-day design. Exact timestamps are not available, so equal elapsed time between adjacent rows is not assumed.

## 2. Dataset Overview

- Source rows: **2,635**
- Participants: **63**
- Classes: **6**
- Available records per participant: **39 to 42**
- Observed AffWB values: **2,210** (83.9% of retained rows)

## 3. AffWB Distribution

Observed AffWB has a mean of **5.626**, a median of **6.000**, and an SD of **1.302**. Values range from **1 to 7**.

AffWB is a bounded 1-7 response rather than a continuous unbounded Gaussian variable. The normality statistics and Q-Q plot are provided as descriptive diagnostics only. Repeated measurements within participants violate the independence assumption of ordinary pooled normality tests, and large samples can reject normality for small deviations.

## 4. Missingness and Questionnaire Coverage

Response missingness and structural questionnaire unavailability remain separate. A valid zero for a social-media variable remains an observed value and is not treated as missing.

- **Instagram:** 60 of 63 participants (95.2%) received the corresponding questionnaire.
- **WhatsApp:** 63 of 63 participants (100.0%) received the corresponding questionnaire.
- **Snapchat:** 48 of 63 participants (76.2%) received the corresponding questionnaire.
- **Games:** 4 of 63 participants (6.3%) received the corresponding questionnaire.
- **YouTube:** 14 of 63 participants (22.2%) received the corresponding questionnaire.

The limited participant coverage for Games and YouTube means that these variables should not automatically be included in a single full-sample primary model. Step 02 does not remove them; it documents the coverage needed for later model-specific decisions.

## 5. Within- and Between-Participant AffWB Variation

The descriptive ICC(1) estimate is **0.309**. This indicates the approximate share of AffWB variation attributable to stable differences between participants under a one-way random-intercept decomposition. The remaining variation occurs within participants across occasions.

Participant-specific summaries and the participant mean/SD figure show whether the pooled distribution hides substantial individual differences. These differences are relevant because later validation should report both overall performance and variation in performance across participants.

## 6. Sequential Dependence

At lag 1, the pooled AffWB Pearson correlation is **0.382**, while the within-person-centered correlation is **0.090**, based on **1,862** valid pairs.

Across lags 1-12, the largest absolute within-person-centered AffWB correlation occurs at lag **6** and equals **0.107**. This descriptive result does not select a final history length. It shows how sequential association changes as observations become more distant.

The largest absolute within-person-centered lagged correlation in this descriptive scan was -0.115 for YTAct at lag 12. This is an exploratory association, not evidence of causality or final predictor utility.

## 7. Candidate History Lengths

The project will not impose a single history length in advance. The history-window table evaluates several candidate lookbacks while retaining the same next-occasion target definition.

- Lookback 1: 1,862 complete AffWB-history windows (86.1% of candidate observed targets); 63 participants represented.
- Lookback 3: 1,342 complete AffWB-history windows (65.4% of candidate observed targets); 63 participants represented.
- Lookback 6: 882 complete AffWB-history windows (46.7% of candidate observed targets); 58 participants represented.
- Lookback 12: 426 complete AffWB-history windows (27.5% of candidate observed targets); 38 participants represented.
- Lookback 24: 88 complete AffWB-history windows (9.8% of candidate observed targets); 13 participants represented.
- Lookback 36: 0 complete AffWB-history windows (0.0% of candidate observed targets); 0 participants represented.

For ARIMA/ARIMAX, later steps can use an expanding history containing all observations available before each forecast origin. Fixed lookback comparisons are especially relevant to lag-feature models and RNN/LSTM sequence construction.

## 8. Interpretation Boundaries

Step 02 is descriptive and exploratory. It does not:

- select a final predictor set;
- choose a final lookback length;
- impute model inputs;
- create train, validation, or test partitions;
- fit ARIMA/ARIMAX, tree-based, RNN, or LSTM models;
- claim that lagged social-media use causes later changes in AffWB.

These decisions must be made in later steps using chronological validation and model-specific data preparation.

## 9. Quality Checks

All **7** Step 02 input checks passed. Detailed results are stored in `outputs/tables/02_data_quality_checks.csv`.

## 10. Main Output Groups

- Dataset and variable summaries: `outputs/tables/02_*.csv`
- Distribution and normality figures: `outputs/images/02_*_distribution.png` and `outputs/images/02_*_qq_plot.png`
- Temporal and missingness figures: `outputs/images/02_affwb_*`, `02_lagged_*`, and `02_history_*`
- This generated report: `outputs/reports/02_descriptive_analysis_report.md`
