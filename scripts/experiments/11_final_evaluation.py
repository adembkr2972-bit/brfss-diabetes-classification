import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


SVM_LABEL = config.display_model_label("SVM")


def display_label(model_name: str) -> str:
    return config.display_model_label(model_name)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("11_final_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOGS_DIR / "11_final_evaluation.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def positive_scores(model, features: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    return None


def metric_row(
    model_name: str,
    y_true,
    y_pred,
    y_score,
    has_probability: bool,
    threshold: float | None = None,
) -> dict[str, float | str | None]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="weighted", zero_division=0
    )
    row = {
        "model_name": model_name,
        "display_label": display_label(model_name),
        "n_test": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f_weighted,
        "precision_class_0": precision[0],
        "recall_class_0": recall[0],
        "f1_class_0": f1[0],
        "support_class_0": int(support[0]),
        "precision_class_1": precision[1],
        "recall_class_1": recall[1],
        "f1_class_1": f1[1],
        "support_class_1": int(support[1]),
        "roc_auc": np.nan,
        "pr_auc_average_precision": np.nan,
        "log_loss": np.nan,
        "classification_threshold": threshold,
        "error": "",
    }
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        row["roc_auc"] = roc_auc_score(y_true, y_score)
        row["pr_auc_average_precision"] = average_precision_score(y_true, y_score)
        if has_probability:
            row["log_loss"] = log_loss(y_true, np.clip(y_score, 1e-9, 1 - 1e-9), labels=[0, 1])
    return row


def confusion_row(model_name: str, y_true, y_pred) -> dict[str, int | str]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model_name": model_name,
        "display_label": display_label(model_name),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def plot_confusion(y_true, y_pred, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
        yticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def bootstrap_intervals(y_true, y_pred, seed: int, repetitions: int = 500) -> dict[str, float]:
    """Respondent-level bootstrap intervals conditional on this test split."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = np.empty((repetitions, 3), dtype=float)
    for i in range(repetitions):
        indices = rng.integers(0, n, size=n)
        tn, fp, fn, tp = confusion_matrix(y_true[indices], y_pred[indices], labels=[0, 1]).ravel()
        rec0 = tn / (tn + fp) if tn + fp else 0.0
        rec1 = tp / (tp + fn) if tp + fn else 0.0
        p0 = tn / (tn + fn) if tn + fn else 0.0
        p1 = tp / (tp + fp) if tp + fp else 0.0
        f0 = 2 * p0 * rec0 / (p0 + rec0) if p0 + rec0 else 0.0
        f1 = 2 * p1 * rec1 / (p1 + rec1) if p1 + rec1 else 0.0
        values[i] = ((tn + tp) / n, (rec0 + rec1) / 2, (f0 + f1) / 2)
    labels = ["accuracy", "recall_macro", "f1_macro"]
    result = {"model_name": "", "bootstrap_repetitions": repetitions}
    for column, label in enumerate(labels):
        low, high = np.percentile(values[:, column], [2.5, 97.5])
        result[f"{label}_ci95_low"] = float(low)
        result[f"{label}_ci95_high"] = float(high)
    return result


def prepare_model_for_inference(model) -> None:
    classifier = model.named_steps.get("classifier")
    if classifier is None:
        return
    if hasattr(classifier, "n_jobs"):
        classifier.n_jobs = 1
    if hasattr(classifier, "thread_count"):
        classifier.thread_count = 1


def current_git_commit() -> str:
    """Return the repository HEAD hash that produced this holdout-stage run."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    for path in [config.LOGS_DIR, config.METRICS_DIR, config.PREDICTIONS_DIR, config.FIGURES_DIR, config.REPORTS_DIR]:
        os.makedirs(path, exist_ok=True)
    logger = setup_logger()
    logger.warning("Evaluating all nine prespecified classifiers on the held-out respondent sample.")
    logger.warning("Earlier versions of six classifiers had already been evaluated on this same holdout; disclose this prior access.")
    started_at = datetime.now().timestamp()
    run_started = float(config.RERUN_STARTED_PATH.read_text(encoding="utf-8").strip())

    X_train = pd.read_parquet(config.X_TRAIN_PATH)
    X_test = pd.read_parquet(config.X_TEST_PATH)
    y_train = pd.read_parquet(config.Y_TRAIN_PATH)[config.TARGET_BINARY_COL].astype(int)
    y_test = pd.read_parquet(config.Y_TEST_PATH)[config.TARGET_BINARY_COL].astype(int)
    threshold_frame = pd.read_csv(config.CHOSEN_THRESHOLDS_PATH)
    threshold_map = threshold_frame.set_index("model_name")["threshold"].astype(float).to_dict()
    artifact_map = threshold_frame.set_index("model_name")["chosen_artifact"].astype(str).to_dict()
    missing_thresholds = sorted(set(config.MODEL_NAMES) - set(threshold_map))
    if missing_thresholds:
        raise ValueError(f"Locked validation thresholds are missing for: {missing_thresholds}")
    expected_schema = pd.read_csv(config.TRAIN_SCHEMA_PATH)
    assert X_test.columns.tolist() == X_train.columns.tolist(), "Test columns do not match training columns."
    assert [str(dtype) for dtype in X_test.dtypes.tolist()] == expected_schema["dtype"].tolist(), "Test dtypes do not match training schema."

    metric_rows = []
    confusion_rows = []
    prediction_arrays: dict[str, np.ndarray] = {}
    roc_payload = []
    failures = []

    for model_name in config.MODEL_NAMES:
        chosen_artifact = artifact_map.get(model_name)
        if chosen_artifact not in {"baseline", "tuned"}:
            raise ValueError(f"Invalid chosen artifact for {model_name}: {chosen_artifact!r}")
        model_path = config.MODELS_DIR / f"{chosen_artifact}_{model_name}.joblib"
        try:
            if not model_path.exists() or model_path.stat().st_mtime <= run_started:
                raise FileNotFoundError(f"No saved tuned or baseline artifact for {model_name}.")
            if model_path.stat().st_mtime > started_at:
                logger.warning("Artifact timestamp for %s is later than the evaluation start.", model_name)
            model = joblib.load(model_path)
            prepare_model_for_inference(model)
            y_score = positive_scores(model, X_test)
            threshold = float(threshold_map[model_name])
            if y_score is not None:
                y_pred = np.asarray(y_score >= threshold, dtype=int)
            else:
                y_pred = np.asarray(model.predict(X_test), dtype=int)
            metrics = metric_row(
                model_name,
                y_test,
                y_pred,
                y_score,
                hasattr(model, "predict_proba"),
                threshold=threshold,
            )
            metric_rows.append(metrics)
            confusion_rows.append(confusion_row(model_name, y_test, y_pred))
            prediction_arrays[model_name] = y_pred
            pd.DataFrame(
                {
                    "true_label": y_test.to_numpy(),
                    "predicted_label": y_pred,
                    "positive_class_score": y_score if y_score is not None else np.nan,
                    "classification_threshold": threshold,
                    "display_label": display_label(model_name),
                }
            ).to_csv(config.PREDICTIONS_DIR / f"test_predictions_{model_name}.csv", index=False)
            plot_confusion(y_test, y_pred, config.FIGURES_DIR / f"confusion_matrix_final_{model_name}.png")
            if y_score is not None:
                roc_payload.append((model_name, *roc_curve(y_test, y_score), metrics["roc_auc"]))
            logger.info("Completed final metrics for %s (macro F1 %.4f).", display_label(model_name), metrics["f1_macro"])
        except Exception as exc:
            logger.exception("Final evaluation failed for %s", model_name)
            metric_rows.append({"model_name": model_name, "error": str(exc)})
            failures.append({"model_name": model_name, "error": str(exc)})

    # Majority-class reference baseline, determined from the training split only.
    majority_class = int(y_train.value_counts().idxmax())
    baseline_pred = np.full(len(y_test), majority_class, dtype=int)
    baseline_metrics = metric_row("MajorityBaseline", y_test, baseline_pred, None, False)
    metric_rows.append(baseline_metrics)
    confusion_rows.append(confusion_row("MajorityBaseline", y_test, baseline_pred))
    prediction_arrays["MajorityBaseline"] = baseline_pred
    pd.DataFrame(
        {
            "true_label": y_test.to_numpy(),
            "predicted_label": baseline_pred,
            "positive_class_score": np.nan,
            "classification_threshold": np.nan,
            "display_label": display_label("MajorityBaseline"),
        }
    ).to_csv(config.PREDICTIONS_DIR / "test_predictions_MajorityBaseline.csv", index=False)
    plot_confusion(y_test, baseline_pred, config.FIGURES_DIR / "confusion_matrix_final_MajorityBaseline.png")

    # Prior baseline estimates the training-sample class prevalence and predicts
    # the most common class at the conventional 0.50 cutoff.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y_train), 1), dtype=np.int8), y_train)
    dummy_features = np.zeros((len(y_test), 1), dtype=np.int8)
    dummy_score = positive_scores(dummy, dummy_features)
    dummy_pred = np.asarray(dummy.predict(dummy_features), dtype=int)
    metric_rows.append(metric_row("DummyClassifier_prior", y_test, dummy_pred, dummy_score, True, threshold=0.5))
    confusion_rows.append(confusion_row("DummyClassifier_prior", y_test, dummy_pred))
    prediction_arrays["DummyClassifier_prior"] = dummy_pred
    pd.DataFrame(
        {
            "true_label": y_test.to_numpy(),
            "predicted_label": dummy_pred,
            "positive_class_score": dummy_score,
            "classification_threshold": 0.5,
            "display_label": display_label("DummyClassifier_prior"),
        }
    ).to_csv(config.PREDICTIONS_DIR / "test_predictions_DummyClassifier_prior.csv", index=False)
    plot_confusion(y_test, dummy_pred, config.FIGURES_DIR / "confusion_matrix_final_DummyClassifier_prior.png")

    final_df = pd.DataFrame(metric_rows).sort_values(by="f1_macro", ascending=False, na_position="last")
    final_df.to_csv(config.FINAL_TEST_METRICS_PATH, index=False)
    pd.DataFrame(confusion_rows).to_csv(config.METRICS_DIR / "final_test_confusion_matrices.csv", index=False)

    uncertainty_rows = []
    for offset, (model_name, y_pred) in enumerate(prediction_arrays.items()):
        interval = bootstrap_intervals(y_test, y_pred, seed=config.RANDOM_SEED + offset)
        interval["model_name"] = model_name
        uncertainty_rows.append(interval)
    pd.DataFrame(uncertainty_rows).to_csv(config.METRICS_DIR / "final_test_bootstrap_intervals.csv", index=False)
    commit_hash = current_git_commit()
    (config.REPORTS_DIR / "test_set_access_note.txt").write_text(
        f"Repository commit at holdout-stage execution: {commit_hash}\n"
        "The held-out split was used by earlier project versions to evaluate six classifiers. "
        "This rerun freezes the approved training/validation protocol before evaluating all nine classifiers, "
        "but the holdout is not a previously untouched external test set. Do not claim that it was never inspected. "
        "Bootstrap intervals are ordinary respondent-level intervals conditional on this split and are not BRFSS design-based intervals.",
        encoding="utf-8",
    )

    if roc_payload:
        plt.figure(figsize=(8, 6))
        for model_name, fpr, tpr, _, auc_value in roc_payload:
            plt.plot(fpr, tpr, label=f"{display_label(model_name)} (AUC={auc_value:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Held-out respondent-sample ROC curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "roc_curve_final_models.png", dpi=150)
        plt.close()

    logger.info("Saved held-out results. Completed classifiers: %s/9; failures: %s.", 9 - len(failures), len(failures))
    if failures:
        raise SystemExit(f"Final evaluation incomplete for {len(failures)} models; see final_test_metrics.csv and log.")


if __name__ == "__main__":
    main()
