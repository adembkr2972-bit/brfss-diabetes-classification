# Analysis Protocol

Status: approved analytical protocol for the current 2024 BRFSS respondent-sample benchmark. Steps 11-13 ran on 2026-10-16, and post-holdout analysis ran afterwards.

## Scope and population

- The benchmark is unweighted. Claims are limited to the analytic respondent sample and are not U.S. population estimates.
- The input used by the current artifacts is `data/raw/LLCP2024XPT/LLCP2024.csv`, identified by the SHA-256 in `outputs/reports/source_layout_audit.md`.
- The local 301-column input does not reconcile with the current CDC page's 345-variable XPT description. The source-layout audit documents the exact header comparison; resolve the source version before presenting the dataset provenance as settled.

## Target and feature handling

- `DIABETE4 == 1` maps to positive class 1; `DIABETE4 == 3` maps to negative class 0. Other target responses are excluded.
- Leakage, administrative, and module-dependent fields are removed using the lists in `scripts/config/config.py`.
- Candidate predictor missingness is screened at a 40% cutoff using only the training respondents from the seeded stratified split. The screen does not use validation or holdout values.
- The current selection retains 67 predictors. Special unknown/refused codes are recoded to missing or zero according to the documented maps in `scripts/config/config.py`. `ALCDAY4` weekly and monthly codes are converted to mean drinking days per week.
- Numerical imputation, categorical imputation, encoding, missingness indicators, and scaling are fit inside the model pipelines using training data/folds. Nominal fields are one-hot encoded; only explicitly justified ordinal and continuous fields use numeric treatment.
- Split ratios are 70% train, 15% validation, and 15% holdout, stratified with random seed 42.

## Class imbalance and model fitting

- Do not apply BRFSS survey weights. Classifier balancing weights are algorithmic training choices and are not survey weights.
- Use the current model-specific imbalance policy: balanced class weights where supported, `scale_pos_weight` for XGBoost, LightGBM class weighting, CatBoost balanced class weights, and balanced per-row sample weights for `MLPClassifier`. Do not resample validation or holdout rows.
- `MLPClassifier` supports `sample_weight` in the installed scikit-learn version; the baseline and tuning pipelines pass weights computed from the training labels.
- Compare the nine prespecified models: Logistic Regression, Decision Tree, Random Forest, Extra Trees, SVM, XGBoost, LightGBM, CatBoost, and MLPClassifier.
- SVM baseline fitting uses a 50,000-row stratified training subsample; SVM tuning uses a 10,000-row stratified subsample. Report this mismatch as a limitation and display the model as “SVM (trained on a 50,000-row stratified subsample)”.

## Tuning, artifact choice, and thresholds

- Hyperparameter selection uses five-fold stratified cross-validation on training data, scoring macro F1, with the existing five-configuration search budget. Do not retune while creating the validation comparison.
- Compare existing baseline and tuned artifacts on validation data. Primary criterion: PR-AUC computed as scikit-learn average precision. Tie-break: each artifact's maximum validation macro F1 over thresholds 0.05 through 0.95 in 0.01 increments. Treat PR-AUC differences within 1e-12 as a tie; if both criteria tie, retain baseline.
- Save all candidate scores, chosen artifact, and reason in `outputs/metrics/tuned_vs_baseline_decision.csv`.
- For each selected model, choose the validation threshold that maximizes macro F1 on the same 0.05–0.95 grid. Resolve threshold ties by choosing the value closest to 0.50, then the lower value. Save the choice in `outputs/metrics/chosen_thresholds.csv`.
- These per-model thresholds are frozen before holdout evaluation. The holdout stage must read the saved file and must not optimize, recalibrate, or modify thresholds using holdout results. `score_source` in the threshold file records whether the cutoff applies to a probability or decision score. The selected SVM baseline artifact has `SVC(probability=True)`, so its saved 0.30 cutoff applies to its probability estimate; its tuned candidate had `probability=False` but was rejected.

## Shortlist and final comparison

- The research question is a comparison of all nine prespecified classifiers. Keep all nine in the shortlist when validation metrics are available; do not eliminate a model because of lower performance or a larger train–validation gap. A model may be omitted only when a required experiment failed, and that failure must be reported.
- The final comparison includes both the most-frequent-class reference and `DummyClassifier(strategy="prior")`, fitted using training labels only, alongside the nine classifiers.
- Report macro F1, macro recall, accuracy, weighted F1, class-specific precision/recall/F1, confusion matrices, ROC-AUC, and PR-AUC where scores are available. Include uncertainty only with an explicit statement that respondent-level intervals are not BRFSS design-based intervals.

## Holdout disclosure and environment

- Earlier project outputs evaluated six classifiers on the same holdout. Disclose this prior access; do not describe this holdout as never inspected or as an external validation set.
- lock file captured after steps 8-10 from the same environment
- `requirements-lock.txt` records the installed package pins after the current steps 8–10 rerun. The completed rerun did not use that lock file; do not claim otherwise.
- When the final stage is authorized, run only the exact documented command after confirming the threshold file is present. Preserve validation outputs and record any holdout-stage failures without revising the protocol.

## Post-hoc validation-only analyses (added after the holdout run)

- **Task A:** Compare the baseline weighted and unweighted model fits using validation data only.
- **Task B:** Assess probability calibration using validation data only.
- **Task C:** Assess threshold and illustrative misclassification-cost sensitivity using validation data only.
- These analyses were added after the holdout results were seen. They do not use holdout outcomes for fitting, calibration, threshold selection, or evaluation.
- Task A's full 1,000-resample validation-only run completed on 2026-10-19. The run note records 66,290 validation rows, all nine models, seed 42, and `holdout_read: false`; results are in `outputs/imbalance_ablation/ablation_metrics.csv` and `outputs/imbalance_ablation/ablation_paired_differences.csv`.
