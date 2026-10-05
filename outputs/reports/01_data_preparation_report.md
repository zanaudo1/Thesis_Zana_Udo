# Step 01 Runtime Report: Project and Data Preparation

## Input

- File: `inputs/01_beyens_2020_data.csv`
- Source rows: 2,635
- Participants: 63
- Planned assessments per participant: 42

## Sequence Structure

- The prepared dataset contains only rows that are present in the source file.
- No rows were added to force all participants to a length of 42.
- No daily aggregation was performed.
- 7 participants have fewer than 42 available source rows.
- All available participant sequences are contiguous from occasion 1 according to the quality checks.

## Missing-Data Representation

- Valid AffWB values: 2,210
- AffWB rows without a usable target value: 425
- AffWB observed percentage among available rows: 83.87%
- Missing target values were not imputed.
- Social-media response missingness and structural questionnaire unavailability were stored separately.
- Valid zero-minute responses were retained as numeric zeros.

## Platform Questionnaire Coverage

- Instagram: 60 of 63 participants
- WhatsApp: 63 of 63 participants
- Snapchat: 48 of 63 participants
- Games: 4 of 63 participants
- YouTube: 14 of 63 participants

## Quality Control

- Quality checks executed: 7
- Failed checks: 0

## Modeling Boundary

This step does not create lagged predictors, impute model inputs, split data into train/validation/test sets, or fit models. Those actions must be performed later within model-specific pipelines using chronological information only.
