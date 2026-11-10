# BRFSS Diabetes Classification

> Portfolio README draft. Publish only after curating the source files and verifying the run instructions.

This project compares nine supervised machine-learning classifiers for classifying **self-reported diabetes status** among respondents in the 2024 Behavioral Risk Factor Surveillance System (BRFSS). It is a cross-sectional classification benchmark. It does not diagnose diabetes or predict who will develop diabetes in the future.

## Dataset and outcome

The local 2024 combined landline/cellular file contains 457,670 respondents and 301 columns. The binary outcome uses `DIABETE4 = 1` as positive and `DIABETE4 = 3` as negative; other responses are excluded. The analytic sample contains 441,934 respondents, including 65,809 positives (14.89%). The 67 retained predictors are described in the project feature inventory.

The CDC 2024 annual page says the linked XPT has 345 variables, while the linked file and year-by-year table indicate 301. The source audit does not resolve this difference. The analysis fields were checked against the downloaded XPT; the provenance discrepancy remains a limitation.

## Method

- Stratified 70/15/15 train, validation, and holdout split (seed 42).
- Training-only missingness screening and preprocessing fitted inside model pipelines.
- Five-fold stratified cross-validation with five sampled configurations per model.
- Validation-based artifact and threshold selection, followed by a single current holdout evaluation using the frozen choices.
- Algorithm class weighting addresses class imbalance; BRFSS survey weights were not applied.

The nine classifiers are Logistic Regression, Decision Tree, Random Forest, Extra Trees, SVM, XGBoost, LightGBM, CatBoost, and MLPClassifier. Majority-class and prior-probability references are also reported.

## Selected 2024 results

| Measure | Result |
|---|---:|
| Highest observed classifier macro F1 (CatBoost) | 0.6893 (95% respondent-bootstrap interval 0.6848–0.6941) |
| Highest classifier accuracy (XGBoost) | 0.8327 |
| Highest macro recall (LightGBM) | 0.7164 |
| Majority-baseline accuracy / positive recall | 0.8511 / 0.0000 |

No paired top-four macro-F1 comparison remained significant after Holm correction. The model with the highest observed score should not be described as a proven winner.

## Important limits

- Earlier project versions had evaluated six classifiers on this same holdout, so it is not untouched.
- Results describe an unweighted analytic respondent sample, not U.S. population performance.
- A frozen earlier-year evaluation on 2023 BRFSS is backward cross-year transfer, not prospective or independent-population validation.
- The 2024 source-page variable-count conflict remains unexplained.
- A clean end-to-end training and tuning reproduction has not been completed.

## Reproducibility and data

BRFSS data are available from the [CDC 2024 annual data page](https://www.cdc.gov/brfss/annual_data/annual_2024.html). Do not commit respondent data, model binaries, API credentials, or private thesis/submission files. The current local workspace contains more than the public research code; create a curated copy before publishing. Add tested setup commands and a small, non-sensitive results example after the public copy has been checked from a clean environment.

## Project status

This README and its selected source files are in a local staging copy. The analysis scripts, environment, and public data download procedure have not yet been verified as a self-contained clean clone. Code availability is currently unresolved; no public repository URL is designated.

## License and citation

Choose a code license only after confirming that the code and included assets may be shared. Add a citation file if the thesis or paper is approved for public release.
