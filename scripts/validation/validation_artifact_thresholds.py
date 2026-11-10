"""Validation-only artifact comparison and locked threshold selection.

This script reads X_val/y_val and the already fitted baseline/tuned artifacts.
It never loads the test split, retrains a model, or selects a threshold on test
data. The resulting chosen_thresholds.csv is the source of locked cutoffs for
the later holdout evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
TIE_TOLERANCE = 1e-12


def display_label(model_name: str) -> str:
    return config.display_model_label(model_name)


def get_positive_scores(model, features: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return scores for class 1 without fitting or changing the artifact."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        classes = np.asarray(model.classes_)
        positive_col = int(np.flatnonzero(classes == 1)[0])
        return probabilities[:, positive_col], "predict_proba"
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        classes = np.asarray(model.classes_)
        if len(classes) == 2 and classes[1] != 1:
            scores = -scores
        return scores, "decision_function_raw_margin"
    raise TypeError(f"Model {type(model).__name__} has no probability or decision score.")


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    values = np.asarray(
        [f1_score(y_true, scores >= threshold, average="macro", zero_division=0) for threshold in THRESHOLDS],
        dtype=float,
    )
    maximum = float(values.max())
    tied = np.flatnonzero(np.isclose(values, maximum, rtol=0.0, atol=TIE_TOLERANCE))
    # Deterministic tie rule: prefer the grid point nearest 0.50, then the lower one.
    chosen_index = min(tied, key=lambda i: (abs(float(THRESHOLDS[i]) - 0.5), float(THRESHOLDS[i])))
    return float(THRESHOLDS[chosen_index]), maximum


def cv_best_score(model_name: str) -> float:
    path = config.METRICS_DIR / f"tuning_cv_results_{model_name}.csv"
    if not path.exists():
        return float("nan")
    cv = pd.read_csv(path)
    if "mean_test_score" not in cv or cv.empty:
        return float("nan")
    if "rank_test_score" in cv:
        best = cv.loc[cv["rank_test_score"].idxmin(), "mean_test_score"]
    else:
        best = cv["mean_test_score"].max()
    return float(best)


def main() -> None:
    # Explicitly load only the validation partition.
    X_val = pd.read_parquet(config.X_VAL_PATH)
    y_val = pd.read_parquet(config.Y_VAL_PATH)[config.TARGET_BINARY_COL].astype(int).to_numpy()
    tuned_meta = pd.read_csv(config.TUNED_METRICS_PATH).set_index("model_name")

    prediction_cache: dict[str, tuple[np.ndarray, str]] = {}
    comparison_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []

    for model_name in config.MODEL_NAMES:
        artifact_results: dict[str, dict[str, object]] = {}
        selected_meta = str(tuned_meta.loc[model_name, "selected_artifact"]).strip().lower()

        for artifact_name in ("baseline", "tuned"):
            artifact_path = config.MODELS_DIR / f"{artifact_name}_{model_name}.joblib"
            if not artifact_path.exists():
                raise FileNotFoundError(f"Required existing artifact is missing: {artifact_path}")

            # Step 8 serializes the baseline pipeline into tuned_<model>.joblib
            # when its validation macro F1 is lower. Reuse the same score vector;
            # this avoids running the expensive SVM prediction twice.
            cache_key = str(artifact_path.resolve())
            if artifact_name == "tuned" and selected_meta == "baseline":
                baseline_path = config.MODELS_DIR / f"baseline_{model_name}.joblib"
                cache_key = str(baseline_path.resolve())
            if cache_key not in prediction_cache:
                model = joblib.load(artifact_path if cache_key == str(artifact_path.resolve()) else baseline_path)
                scores, score_source = get_positive_scores(model, X_val)
                if not np.isfinite(scores).all():
                    raise ValueError(f"Non-finite validation scores found for {model_name} ({artifact_name}).")
                prediction_cache[cache_key] = (scores, score_source)
            scores, score_source = prediction_cache[cache_key]
            threshold, threshold_f1 = best_threshold(y_val, scores)
            ap = float(average_precision_score(y_val, scores))
            artifact_results[artifact_name] = {
                "path": str(artifact_path.relative_to(config.PROJECT_ROOT)),
                "scores": scores,
                "score_source": score_source,
                "pr_auc_average_precision": ap,
                "threshold": threshold,
                "threshold_macro_f1": threshold_f1,
            }

        baseline = artifact_results["baseline"]
        tuned = artifact_results["tuned"]
        ap_diff = float(tuned["pr_auc_average_precision"]) - float(baseline["pr_auc_average_precision"])
        f1_diff = float(tuned["threshold_macro_f1"]) - float(baseline["threshold_macro_f1"])
        if ap_diff > TIE_TOLERANCE:
            chosen = "tuned"
            reason = "Tuned artifact selected because its validation PR-AUC (average precision) is higher."
        elif ap_diff < -TIE_TOLERANCE:
            chosen = "baseline"
            reason = "Baseline artifact selected because its validation PR-AUC (average precision) is higher."
        elif f1_diff > TIE_TOLERANCE:
            chosen = "tuned"
            reason = "Validation PR-AUC tied; tuned artifact has higher threshold-optimized validation macro F1."
        elif f1_diff < -TIE_TOLERANCE:
            chosen = "baseline"
            reason = "Validation PR-AUC tied; baseline artifact has higher threshold-optimized validation macro F1."
        else:
            chosen = "baseline"
            reason = "Validation PR-AUC and threshold-optimized macro F1 tied; baseline retained as the stable tie choice."

        chosen_result = artifact_results[chosen]
        common = {
            "model_name": model_name,
            "display_label": display_label(model_name),
            "tuning_search_5fold_cv_best_score": cv_best_score(model_name),
            "cv_score_is_for_tuning_search_not_baseline_cv": True,
        }
        comparison_rows.append(
            {
                **common,
                "baseline_artifact": baseline["path"],
                "baseline_score_source": baseline["score_source"],
                "baseline_validation_pr_auc_average_precision": baseline["pr_auc_average_precision"],
                "baseline_validation_threshold": baseline["threshold"],
                "baseline_validation_macro_f1_at_optimized_threshold": baseline["threshold_macro_f1"],
                "tuned_artifact": tuned["path"],
                "tuned_score_source": tuned["score_source"],
                "tuned_validation_pr_auc_average_precision": tuned["pr_auc_average_precision"],
                "tuned_validation_threshold": tuned["threshold"],
                "tuned_validation_macro_f1_at_optimized_threshold": tuned["threshold_macro_f1"],
                "pr_auc_tuned_minus_baseline": ap_diff,
                "threshold_macro_f1_tuned_minus_baseline": f1_diff,
                "chosen_artifact": chosen,
                "chosen_artifact_path": chosen_result["path"],
                "chosen_reason": reason,
            }
        )
        threshold_rows.append(
            {
                "model_name": model_name,
                "display_label": display_label(model_name),
                "chosen_artifact": chosen,
                "artifact_path": chosen_result["path"],
                "score_source": chosen_result["score_source"],
                "threshold": chosen_result["threshold"],
                "validation_macro_f1_at_threshold": chosen_result["threshold_macro_f1"],
                "threshold_grid_min": float(THRESHOLDS.min()),
                "threshold_grid_max": float(THRESHOLDS.max()),
                "threshold_grid_step": 0.01,
                "tie_rule": "max macro F1; ties choose nearest 0.50, then lower threshold",
            }
        )

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame(comparison_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    comparison_path = config.TUNED_BASELINE_DECISION_PATH
    thresholds_path = config.CHOSEN_THRESHOLDS_PATH
    comparison_df.to_csv(comparison_path, index=False)
    thresholds_df.to_csv(thresholds_path, index=False)

    print("Validation-only tuned vs baseline decision (PR-AUC primary; threshold-optimized macro F1 tie-break):")
    print(
        comparison_df[
            [
                "display_label",
                "baseline_validation_pr_auc_average_precision",
                "baseline_validation_macro_f1_at_optimized_threshold",
                "tuned_validation_pr_auc_average_precision",
                "tuned_validation_macro_f1_at_optimized_threshold",
                "chosen_artifact",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("\nLocked validation thresholds:")
    print(
        thresholds_df[["display_label", "chosen_artifact", "threshold", "validation_macro_f1_at_threshold"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("\nCatBoost/MLPClassifier: 5-fold tuning CV best_score_ next to artifact validation macro F1:")
    for model_name in ("CatBoost", "MLPClassifier"):
        row = comparison_df.loc[comparison_df["model_name"] == model_name].iloc[0]
        cv = row["tuning_search_5fold_cv_best_score"]
        for artifact_name in ("baseline", "tuned"):
            validation_f1 = row[f"{artifact_name}_validation_macro_f1_at_optimized_threshold"]
            gap = abs(float(cv) - float(validation_f1))
            print(
                f"{model_name} {artifact_name}: CV best_score_={float(cv):.4f}; "
                f"validation macro F1={float(validation_f1):.4f}; absolute gap={gap:.4f}."
            )
        print("  Note: baseline has no separate CV best_score_; the displayed CV score belongs to the tuning search.")

    print(f"\nWrote {comparison_path.relative_to(config.PROJECT_ROOT)}")
    print(f"Wrote {thresholds_path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
