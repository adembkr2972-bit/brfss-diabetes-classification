import logging
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


SVM_LABEL = config.display_model_label("SVM")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("13_comparison_report")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOGS_DIR / "13_comparison_report.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for path in [config.LOGS_DIR, config.REPORTS_DIR, config.FIGURES_DIR]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting comparison report generation.")

    baseline_df = pd.read_csv(config.BASELINE_METRICS_PATH)
    tuned_df = pd.read_csv(config.TUNED_METRICS_PATH)
    # The report assigns canonical display labels from config below. Drop the
    # copied labels from merged source tables to avoid duplicate display_label
    # columns when final holdout metrics are joined.
    validation_df = pd.read_csv(config.FULL_VALIDATION_METRICS_PATH).drop(
        columns=["display_label"], errors="ignore"
    )
    overfit_df = pd.read_csv(config.OVERFITTING_ANALYSIS_PATH).drop(
        columns=["display_label"], errors="ignore"
    )
    shortlist_df = pd.read_csv(config.MODEL_SHORTLIST_PATH)
    decision_df = pd.read_csv(config.TUNED_BASELINE_DECISION_PATH)
    threshold_df = pd.read_csv(config.CHOSEN_THRESHOLDS_PATH)
    final_df = pd.read_csv(config.FINAL_TEST_METRICS_PATH) if config.FINAL_TEST_METRICS_PATH.exists() else pd.DataFrame()

    comparison_model_names = config.MODEL_NAMES + ["MajorityBaseline", "DummyClassifier_prior"]
    master = (
        pd.DataFrame({"model_name": comparison_model_names})
        .merge(baseline_df[["model_name", "f1_macro"]].rename(columns={"f1_macro": "baseline_f1_macro"}), on="model_name", how="left")
        .merge(tuned_df[["model_name", "f1_macro"]].rename(columns={"f1_macro": "tuned_f1_macro"}), on="model_name", how="left")
        .merge(validation_df, on="model_name", how="left", suffixes=("", "_validation"))
        .merge(overfit_df, on="model_name", how="left")
        .merge(shortlist_df[["model_name", "shortlisted"]], on="model_name", how="left")
        .merge(
            decision_df[
                [
                    "model_name",
                    "baseline_validation_pr_auc_average_precision",
                    "baseline_validation_macro_f1_at_optimized_threshold",
                    "tuned_validation_pr_auc_average_precision",
                    "tuned_validation_macro_f1_at_optimized_threshold",
                    "chosen_artifact",
                    "chosen_reason",
                ]
            ],
            on="model_name",
            how="left",
        )
        .merge(
            threshold_df[["model_name", "threshold", "validation_macro_f1_at_threshold"]].rename(
                columns={
                    "threshold": "locked_validation_threshold",
                    "validation_macro_f1_at_threshold": "locked_threshold_validation_macro_f1",
                }
            ),
            on="model_name",
            how="left",
        )
    )
    master["display_label"] = master["model_name"].map(config.display_model_label)

    if not final_df.empty:
        final_cols = final_df.rename(
            columns={
                "accuracy": "test_accuracy",
                "precision_macro": "test_precision_macro",
                "precision_weighted": "test_precision_weighted",
                "recall_macro": "test_recall_macro",
                "recall_weighted": "test_recall_weighted",
                "f1_macro": "test_f1_macro",
                "f1_weighted": "test_f1_weighted",
                "roc_auc": "test_roc_auc",
                "log_loss": "test_log_loss",
            }
        )
        final_cols = final_cols.drop(columns=["display_label"], errors="ignore")
        master = master.merge(final_cols, on="model_name", how="left")

    master["shortlisted"] = master["shortlisted"].fillna(0).astype(int).astype(bool)
    master["rank"] = master["f1_macro"].rank(method="min", ascending=False)
    master = master.round(4)
    master.to_csv(config.FINAL_COMPARISON_TABLE_PATH, index=False)

    classifier_master = master[master["model_name"].isin(config.MODEL_NAMES)].copy()
    heatmap_cols = [
        "baseline_validation_pr_auc_average_precision",
        "tuned_validation_pr_auc_average_precision",
        "locked_threshold_validation_macro_f1",
        "roc_auc",
        "recall_macro",
        "overfit_gap",
    ]
    heatmap_df = classifier_master.set_index("display_label")[heatmap_cols].copy()
    normalized = heatmap_df.copy()
    for column in heatmap_cols:
        col_min = normalized[column].min()
        col_max = normalized[column].max()
        if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
            normalized[column] = 0
        else:
            normalized[column] = (normalized[column] - col_min) / (col_max - col_min)
    if "overfit_gap" in normalized.columns:
        normalized["overfit_gap"] = 1 - normalized["overfit_gap"]

    mask = ~classifier_master.set_index("display_label")["shortlisted"]
    plt.figure(figsize=(10, 6))
    sns.heatmap(normalized, annot=heatmap_df, fmt=".4f", cmap="YlGnBu", cbar=True, linewidths=0.5)
    for index, model_name in enumerate(normalized.index):
        if mask.loc[model_name]:
            plt.gca().add_patch(plt.Rectangle((0, index), len(heatmap_cols), 1, fill=True, color="lightgrey", alpha=0.2))
    plt.title("Model Comparison Heatmap")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "final_comparison_heatmap.png", dpi=200)
    plt.close()

    improvement_df = classifier_master[["model_name", "baseline_f1_macro", "tuned_f1_macro", "shortlisted"]].copy()
    x = np.arange(len(improvement_df))
    width = 0.35
    plt.figure(figsize=(11, 6))
    baseline_bars = plt.bar(x - width / 2, improvement_df["baseline_f1_macro"], width=width, label="Baseline")
    tuned_bars = plt.bar(x + width / 2, improvement_df["tuned_f1_macro"], width=width, label="Tuned")
    for idx, shortlisted in enumerate(improvement_df["shortlisted"]):
        if shortlisted:
            baseline_bars[idx].set_edgecolor("black")
            tuned_bars[idx].set_edgecolor("black")
            baseline_bars[idx].set_linewidth(1.5)
            tuned_bars[idx].set_linewidth(1.5)
    plt.xticks(
        x,
        [config.display_model_label(name) for name in improvement_df["model_name"]],
        rotation=45,
        ha="right",
    )
    plt.ylabel("F1 (macro)")
    plt.title("Baseline vs Tuned Validation F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "baseline_vs_tuned_improvement.png", dpi=150)
    plt.close()

    dataset_summary = "Dataset Summary\nThe analysis input is the local 2024 BRFSS combined landline/cell file (457,670 rows and 301 fields before filtering). The downloaded CSV and XPT headers each contain 301 names; the CDC annual page describes its XPT export as 345 variables, while the linked official variable layout lists 300 variable rows. Header comparison found 296 names in common, five names only in the local input, and four layout names absent from it. The mismatch and exact SHA-256 checksums are documented in outputs/reports/source_layout_audit.md; the local 301-field input is the one used in this experiment. Tennessee is not represented in the combined public-use release."
    shortlisted_count = int(master["shortlisted"].sum())
    overfit_count = int(classifier_master["overfit_gap"].fillna(0).gt(0.10).sum())
    overfit_evaluable = int(classifier_master["overfit_gap"].notna().sum())
    preprocessing_summary = (
        "Preprocessing Summary\nDIABETE4 codes 1 and 3 were mapped to positive and negative classes; other target responses were excluded. Predictor screening for missingness used only the training split. Leakage/admin/module fields were excluded by the documented lists. ALCDAY4 weekly and monthly responses were normalized to mean drinking days per week. Imputation, encoding, missingness indicators, and scaling were fitted in model pipelines within training folds. Race/ethnicity and other nominal fields were one-hot encoded; ordered features and continuous measurements were declared explicitly. A stratified 70/15/15 split was used."
    )
    performance_summary = (
        "Model Performance Summary\nThe comparison table merges validation-based baseline-versus-tuned decisions, locked validation thresholds, and final holdout metrics for all nine classifier families. It also includes the majority-class and DummyClassifier(strategy='prior') reference baselines."
    )
    overfitting_summary = (
        f"Overfitting Analysis\nOverfit gap was measured as train macro-F1 minus validation macro-F1. {overfit_count} of {overfit_evaluable} models with comparable training scores exceeded the 0.10 warning threshold. {SVM_LABEL} was excluded because it was trained on a stratified subset and a full-training prediction would not be an in-sample score. These gaps are descriptive and were not used to remove any of the nine prespecified classifiers."
    )
    imbalance_summary = (
        "Class Imbalance Handling\nThis is an unweighted respondent-level analysis: BRFSS survey weights were not applied, so findings are limited to this analytic respondent sample and are not population estimates. To address classification imbalance, the training protocol used balanced class weights for supported estimators, XGBoost scale_pos_weight, LightGBM class_weight, CatBoost auto_class_weights, and balanced per-observation sample weights for MLP. The final comparison reports both the majority-class reference and DummyClassifier(strategy='prior')."
    )
    top_models = classifier_master.sort_values("locked_threshold_validation_macro_f1", ascending=False)["display_label"].head(5).tolist()
    findings_lines = [
        "Key Findings",
        f"- Highest validation ranking: {top_models[0] if top_models else 'N/A'}",
        f"- Shortlisted models ({shortlisted_count}): {', '.join(master.loc[master['shortlisted'], 'display_label']) if master['shortlisted'].any() else 'None'}",
        f"- Best validation F1-macro at the locked thresholds: {master['locked_threshold_validation_macro_f1'].max():.4f}" if master["locked_threshold_validation_macro_f1"].notna().any() else "- Best validation F1-macro at the locked thresholds: N/A",
        f"- Best observed validation ROC-AUC: {master['roc_auc'].max():.4f}" if master["roc_auc"].notna().any() else "- Best observed validation ROC-AUC: N/A",
        "- Tuned-versus-baseline selection used validation PR-AUC (average precision) first, with validation macro F1 at its optimal 0.05–0.95 threshold as the tie-break.",
        "- The locked per-model validation thresholds are in outputs/metrics/chosen_thresholds.csv and are read as-is during the later holdout stage.",
        "- Structural missingness from optional modules was excluded from the main benchmark.",
    ]
    limitations = (
        f"Limitations and Caveats\nThis is a self-reported cross-sectional survey classification task, not prospective clinical prediction or a diagnosis dataset. Without BRFSS survey design weights and variance estimation, the results do not estimate U.S. population performance. The CDC documents removals that can create additional missing values. Classifier-level balancing weights are not survey weights. Five configurations per classifier were searched, so tuning was intentionally limited. {SVM_LABEL} used {config.SVM_BASELINE_MAX_ROWS:,} stratified training rows for baseline and {config.SVM_TUNING_MAX_ROWS:,} for tuning, so its tuned-versus-baseline comparison uses different subsample sizes. The selected SVM baseline uses SVC(probability=True); its saved threshold is applied to its probability estimate, and scikit-learn documents that this adds internal five-fold calibration and can make predicted probabilities differ from predicted classes. A respondent-level bootstrap is conditional on the test split and is not a survey-design confidence interval. The holdout was evaluated by earlier project versions for six classifiers; it is not untouched. The repository lock file was created after the completed rerun and did not pin that rerun."
    )
    reproducibility = (
        "Reproducibility Notes\nAll scripts use config-managed paths and random state 42. Five-fold stratified cross-validation on the training split selected configurations using macro F1. Baseline-versus-tuned artifact choice and per-model classification thresholds were then fixed on validation data; neither is recomputed during the holdout stage. Earlier project versions evaluated six classifiers on the same holdout, so the current run does not make it a never-before-seen test set. See the access note, source-layout audit, and experiment manifest."
    )

    report_sections = [
        dataset_summary,
        preprocessing_summary,
        performance_summary,
        overfitting_summary,
        imbalance_summary,
        "\n".join(findings_lines),
        limitations,
        reproducibility,
    ]
    config.FINAL_REPORT_PATH.write_text("\n\n".join(report_sections), encoding="utf-8")

    raw_header = pd.read_csv(config.RAW_CSV_PATH, nrows=0).columns.tolist()
    split_summary = pd.read_csv(config.SPLIT_SUMMARY_PATH).to_dict(orient="records")
    feature_list = [
        line.strip()
        for line in config.FINAL_FEATURE_LIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected_models = pd.read_csv(config.TUNED_METRICS_PATH)
    selected_model_details = selected_models[
        [
            column
            for column in [
                "model_name",
                "display_label",
                "selected_artifact",
                "best_params",
                "training_rows_fitted",
                "search_training_rows",
                "search_time_seconds",
            ]
            if column in selected_models.columns
        ]
    ].to_dict(orient="records")
    package_names = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "pyarrow": "pyarrow",
        "joblib": "joblib",
    }
    versions = {}
    for import_name, distribution_name in package_names.items():
        try:
            versions[import_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[import_name] = None

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_scope": "Unweighted analytic respondent-sample classification; no BRFSS survey weights or population inference.",
        "dataset": {
            "local_file": str(config.RAW_CSV_PATH.relative_to(config.PROJECT_ROOT)),
            "sha256": sha256_file(config.RAW_CSV_PATH),
            "bytes": config.RAW_CSV_PATH.stat().st_size,
            "rows_before_target_filter": 457670,
            "columns_before_target_filter": len(raw_header),
            "cdc_page": "https://www.cdc.gov/brfss/annual_data/annual_2024.html",
            "official_codebook": "https://www.cdc.gov/brfss/annual_data/2024/zip/codebook24_llcp-v2-508.zip",
            "official_questionnaire": "https://www.cdc.gov/brfss/questionnaires/pdf-ques/2024-BRFSS-Questionnaire-508.pdf",
            "calculated_variables_guide": "https://www.cdc.gov/brfss/annual_data/2024/pdf/2024-calculated-variables-version4-508.pdf",
            "survey_weighting_guide": "https://www.cdc.gov/brfss/annual_data/2024/pdf/Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2024-508.pdf",
            "cdc_page_release_date": "August 2025",
            "local_download_date": "not recorded in project files",
            "format_discrepancy": "The local CSV and local XPT each expose the same 301-column header. CDC's current annual page describes a 345-variable XPT export; the linked local CDC variable layout has 300 variable rows. Exact header comparison found 296 shared names, five names only in the experiment input, and four layout names absent from the experiment input. See outputs/reports/source_layout_audit.md; this source-version discrepancy remains unresolved.",
            "local_xpt_file": "data/raw/LLCP2024XPT/LLCP2024.XPT",
            "local_xpt_sha256": sha256_file(config.RAW_DIR / "LLCP2024XPT" / "LLCP2024.XPT"),
            "local_xpt_bytes": (config.RAW_DIR / "LLCP2024XPT" / "LLCP2024.XPT").stat().st_size,
            "official_layout_url": "https://www.cdc.gov/brfss/annual_data/2024/llcp_varlayout_24_onecolumn_v2.html",
            "official_layout_local_file": "downloads/2024_variable_layout.html",
            "official_layout_local_sha256": sha256_file(config.PROJECT_ROOT / "downloads" / "2024_variable_layout.html"),
        },
        "target": {
            "source_column": config.TARGET_COL,
            "mapping": {"1": 1, "3": 0},
            "excluded_original_codes": [2, 4, 7, 9, "missing"],
            "positive_class": "self-reported diabetes other than pregnancy-only diabetes",
        },
        "predictors": {
            "count": len(feature_list),
            "names": feature_list,
            "missingness_screen": f"drop features with > {config.MAX_MISSING_FEATURE_PCT:.1f}% missing in training split only",
            "excluded_leakage_fields": config.LEAKAGE_COLS,
            "feature_types": "Explicit lists in scripts/config/config.py; remaining predictors are nominal and one-hot encoded.",
        },
        "splits": {
            "stratified": True,
            "random_seed": config.RANDOM_SEED,
            "ratios": {"train": config.TRAIN_RATIO, "validation": config.VAL_RATIO, "holdout": config.TEST_RATIO},
            "sizes_and_prevalence": split_summary,
        },
        "evaluation": {
            "primary_metric": config.CV_SCORING,
            "reported_metrics": ["macro F1", "macro recall", "accuracy", "weighted F1", "class-specific precision/recall/F1", "confusion matrices", "ROC AUC", "PR-AUC (average precision) where score available"],
            "cv": {"folds": config.CV_FOLDS, "type": "StratifiedKFold", "shuffle": True, "random_seed": config.RANDOM_SEED, "search_budget": "5 sampled hyperparameter configurations per classifier"},
            "models": config.MODEL_NAMES,
            "display_labels": [config.display_model_label(name) for name in config.MODEL_NAMES],
            "selected_model_configurations": selected_model_details,
            "class_imbalance": "Native balanced class weights where supported; XGBoost scale_pos_weight; MLP balanced sample_weight; no resampling; model weights are not survey weights.",
            "tuned_baseline_selection": "Validation PR-AUC (average precision) primary; macro F1 at per-artifact validation-optimal threshold tie-break; no retuning.",
            "classification_threshold": "Locked per-model threshold maximizing validation macro F1 on the 0.05 to 0.95 grid in 0.01 increments; no threshold changes on holdout.",
            "svm_training_limit": {"baseline_max_rows": config.SVM_BASELINE_MAX_ROWS, "tuning_max_rows": config.SVM_TUNING_MAX_ROWS, "stratified": True},
            "majority_baseline": "Most-frequent class chosen using training labels only.",
            "dummy_baseline": "scikit-learn DummyClassifier(strategy='prior'), fitted using training labels only.",
            "holdout_access": "Earlier project versions evaluated six of these classifiers on the same holdout; this rerun does not make it a never-before-seen test set.",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "packages": versions,
            "requirements_lock": "requirements-lock.txt",
            "completed_steps_8_to_10_used_lock": False,
        },
        "outputs": {
            "test_metrics": str(config.FINAL_TEST_METRICS_PATH.relative_to(config.PROJECT_ROOT)),
            "confusion_matrices": "outputs/metrics/final_test_confusion_matrices.csv",
            "bootstrap_intervals": "outputs/metrics/final_test_bootstrap_intervals.csv",
            "validation_metrics": str(config.FULL_VALIDATION_METRICS_PATH.relative_to(config.PROJECT_ROOT)),
            "tuned_baseline_decision": str(config.TUNED_BASELINE_DECISION_PATH.relative_to(config.PROJECT_ROOT)),
            "locked_thresholds": str(config.CHOSEN_THRESHOLDS_PATH.relative_to(config.PROJECT_ROOT)),
        },
    }
    (config.REPORTS_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved comparison report outputs.")


if __name__ == "__main__":
    main()
