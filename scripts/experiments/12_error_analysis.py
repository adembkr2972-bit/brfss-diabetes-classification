import logging
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


SVM_LABEL = config.display_model_label("SVM")


def display_label(model_name: str) -> str:
    return config.display_model_label(model_name)


def prepare_model_for_inference(model) -> None:
    classifier = model.named_steps.get("classifier")
    if classifier is None:
        return
    if hasattr(classifier, "n_jobs"):
        try:
            classifier.n_jobs = 1
        except Exception:
            pass
    if hasattr(classifier, "thread_count"):
        try:
            classifier.thread_count = 1
        except Exception:
            pass


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("12_error_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOGS_DIR / "12_error_analysis.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def outcome_label(true_value: int, predicted_value: int) -> str:
    if true_value == 1 and predicted_value == 1:
        return "TP"
    if true_value == 0 and predicted_value == 0:
        return "TN"
    if true_value == 0 and predicted_value == 1:
        return "FP"
    return "FN"


def subgroup_metrics(frame: pd.DataFrame, variable: str, model_name: str) -> list[dict[str, float]]:
    rows = []
    for subgroup_value, subset in frame.groupby(variable, dropna=False):
        false_positives = int((subset["error_type"] == "FP").sum())
        true_negatives = int((subset["error_type"] == "TN").sum())
        false_negatives = int((subset["error_type"] == "FN").sum())
        true_positives = int((subset["error_type"] == "TP").sum())
        fp_rate = false_positives / (false_positives + true_negatives) if false_positives + true_negatives else np.nan
        fn_rate = false_negatives / (false_negatives + true_positives) if false_negatives + true_positives else np.nan
        rows.append(
            {
                "model_name": model_name,
                "display_label": display_label(model_name),
                "subgroup_variable": variable,
                "subgroup_value": subgroup_value,
                "sample_size": len(subset),
                "false_positive_rate": fp_rate,
                "false_negative_rate": fn_rate,
            }
        )
    return rows


def plot_age_errors(frame: pd.DataFrame, model_name: str) -> None:
    age_frame = (
        frame.groupby("_AGEG5YR")[["predicted_label", "true_label"]]
        .apply(lambda group: pd.Series({"fp_rate": (group["true_label"].eq(0) & group["predicted_label"].eq(1)).mean(),
                                        "fn_rate": (group["true_label"].eq(1) & group["predicted_label"].eq(0)).mean()}))
        .reset_index()
        .sort_values("_AGEG5YR")
    )
    if age_frame.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(age_frame["_AGEG5YR"], age_frame["fp_rate"], label="FP rate")
    plt.plot(age_frame["_AGEG5YR"], age_frame["fn_rate"], label="FN rate")
    plt.xlabel("_AGEG5YR")
    plt.ylabel("Error rate")
    plt.title(f"Error Rate by Age - {display_label(model_name)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"error_rate_by_age_{model_name}.png", dpi=150)
    plt.close()


def feature_importance_plot(model_name: str, model, X_test_frame: pd.DataFrame, y_test_frame: pd.Series) -> None:
    prepare_model_for_inference(model)
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    if hasattr(classifier, "feature_importances_"):
        feature_names = preprocessor.get_feature_names_out()
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        feature_names = preprocessor.get_feature_names_out()
        importance = np.abs(classifier.coef_[0])
    else:
        feature_names = np.array(X_test_frame.columns.tolist())
        sample_size = min(5000, len(X_test_frame))
        sample_features = X_test_frame.sample(n=sample_size, random_state=config.RANDOM_SEED)
        sample_labels = y_test_frame.loc[sample_features.index]
        permutation = permutation_importance(
            model,
            sample_features,
            sample_labels,
            n_repeats=5,
            scoring="f1_macro",
            random_state=config.RANDOM_SEED,
        )
        importance = permutation.importances_mean

    if len(feature_names) != len(importance):
        min_len = min(len(feature_names), len(importance))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        by="importance", ascending=False
    ).head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", orient="h")
    plt.title(f"Top 20 Feature Importance - {display_label(model_name)}")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"feature_importance_{model_name}.png", dpi=150)
    plt.close()


def main() -> None:
    for path in [config.LOGS_DIR, config.REPORTS_DIR, config.FIGURES_DIR]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting error analysis.")

    X_train_frame = pd.read_parquet(config.X_TRAIN_PATH)
    X_test_frame = pd.read_parquet(config.X_TEST_PATH).reset_index(drop=True)
    y_test_frame = pd.read_parquet(config.Y_TEST_PATH)[config.TARGET_BINARY_COL].reset_index(drop=True)
    expected_schema = pd.read_csv(config.TRAIN_SCHEMA_PATH)
    assert X_test_frame.columns.tolist() == X_train_frame.columns.tolist(), "Test columns do not match training columns."
    assert [str(dtype) for dtype in X_test_frame.dtypes.tolist()] == expected_schema["dtype"].tolist(), "Test dtypes do not match training schema."

    shortlist = pd.read_csv(config.MODEL_SHORTLIST_PATH)
    shortlist["shortlisted"] = shortlist["shortlisted"].fillna(0).astype(int)
    shortlisted_models = shortlist.loc[shortlist["shortlisted"] == 1, "model_name"].tolist()

    subgroup_rows = []
    subgroup_vars = ["_AGEG5YR", "_SEX", "_BMI5CAT", "_INCOMG1"]
    failures = 0

    for model_name in shortlisted_models:
        try:
            prediction_path = config.PREDICTIONS_DIR / f"test_predictions_{model_name}.csv"
            model_path = config.MODELS_DIR / f"tuned_{model_name}.joblib"
            if not model_path.exists():
                model_path = config.MODELS_DIR / f"baseline_{model_name}.joblib"
            if not prediction_path.exists():
                logger.warning("Skipping error analysis for %s because predictions are missing.", model_name)
                failures += 1
                continue

            prediction_df = pd.read_csv(prediction_path)
            merged = X_test_frame.copy()
            merged["true_label"] = prediction_df["true_label"]
            merged["predicted_label"] = prediction_df["predicted_label"]
            merged["positive_class_score"] = prediction_df["positive_class_score"]
            merged["display_label"] = display_label(model_name)
            merged["error_type"] = [
                outcome_label(true_value, pred_value)
                for true_value, pred_value in zip(merged["true_label"], merged["predicted_label"])
            ]

            fn_rate = float((merged["error_type"] == "FN").mean())
            logger.warning("%s false negative rate: %.4f", display_label(model_name), fn_rate)

            merged[merged["error_type"].isin(["FP", "FN"])].to_csv(
                config.REPORTS_DIR / f"error_analysis_{model_name}.csv",
                index=False,
            )

            for variable in subgroup_vars:
                if variable in merged.columns:
                    subgroup_rows.extend(subgroup_metrics(merged, variable, model_name))

            plot_age_errors(merged, model_name)
            if model_path.exists():
                feature_importance_plot(model_name, joblib.load(model_path), X_test_frame, y_test_frame)
        except Exception:
            failures += 1
            logger.exception("Error analysis failed for %s", model_name)

    pd.DataFrame(subgroup_rows).to_csv(config.SUBGROUP_ERROR_ANALYSIS_PATH, index=False)
    logger.info("Saved error analysis outputs with %s per-model failures.", failures)


if __name__ == "__main__":
    main()
