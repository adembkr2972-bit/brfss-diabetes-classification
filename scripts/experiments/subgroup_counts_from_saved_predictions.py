"""Read-only subgroup error counts from saved holdout predictions.

This analysis reads only the four explicitly approved subgroup columns from
X_test.parquet and the already-saved CatBoost and Logistic Regression
prediction CSVs. It does not load y_test, split indices, models, or refit/evaluate
any model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


GROUP_COLUMNS = ["_AGEG5YR", "_SEX", "_BMI5CAT", "_INCOMG1"]
MODEL_FILES = {
    "CatBoost": config.PREDICTIONS_DIR / "test_predictions_CatBoost.csv",
    "Logistic Regression": config.PREDICTIONS_DIR / "test_predictions_LogisticRegression.csv",
}
OUTPUT_PATH = config.REPORTS_DIR / "subgroup_counts_and_ci.csv"
RUN_NOTE_PATH = config.OUTPUTS_DIR / "subgroup_counts" / "run_note.txt"
WILSON_Z = 1.959963984540054


def wilson_interval(events: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""
    if total == 0:
        return (float("nan"), float("nan"))
    p_hat = events / total
    z2 = WILSON_Z**2
    denominator = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denominator
    half_width = (
        WILSON_Z
        * np.sqrt(p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total**2))
        / denominator
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def display_group_value(value: object) -> str:
    return "missing" if pd.isna(value) else str(value)


def subgroup_rows(
    model_name: str,
    predictions: pd.DataFrame,
    groups: pd.DataFrame,
) -> list[dict[str, object]]:
    frame = groups.copy()
    frame["true_label"] = predictions["true_label"].to_numpy(dtype=int)
    frame["predicted_label"] = predictions["predicted_label"].to_numpy(dtype=int)
    frame["false_negative"] = (frame["true_label"].eq(1) & frame["predicted_label"].eq(0)).astype(int)
    frame["false_positive"] = (frame["true_label"].eq(0) & frame["predicted_label"].eq(1)).astype(int)

    rows: list[dict[str, object]] = []
    for variable in GROUP_COLUMNS:
        for value, group in frame.groupby(variable, dropna=False, sort=True):
            n = int(len(group))
            positives = int(group["true_label"].sum())
            negatives = n - positives
            false_negatives = int(group["false_negative"].sum())
            false_positives = int(group["false_positive"].sum())
            fnr_ci_low, fnr_ci_high = wilson_interval(false_negatives, positives)
            fpr_ci_low, fpr_ci_high = wilson_interval(false_positives, negatives)
            rows.append(
                {
                    "model": model_name,
                    "subgroup_variable": variable,
                    "group_value": display_group_value(value),
                    "n": n,
                    "positive_count": positives,
                    "negative_count": negatives,
                    "false_negatives": false_negatives,
                    "false_positives": false_positives,
                    "FNR": safe_rate(false_negatives, positives),
                    "FNR_95CI_low": fnr_ci_low,
                    "FNR_95CI_high": fnr_ci_high,
                    "FPR": safe_rate(false_positives, negatives),
                    "FPR_95CI_low": fpr_ci_low,
                    "FPR_95CI_high": fpr_ci_high,
                    "positives_lt_30": positives < 30,
                }
            )
    return rows


def main() -> None:
    # Guard the approved input boundary so the script cannot silently read any
    # additional holdout data source or a different prediction artifact.
    approved_feature_path = config.X_TEST_PATH.resolve()
    expected_feature_path = (config.PROCESSED_DIR / "X_test.parquet").resolve()
    if approved_feature_path != expected_feature_path:
        raise RuntimeError(f"Unexpected approved feature input: {approved_feature_path}")
    expected_prediction_names = {
        "CatBoost": "test_predictions_CatBoost.csv",
        "Logistic Regression": "test_predictions_LogisticRegression.csv",
    }
    if set(MODEL_FILES) != set(expected_prediction_names):
        raise RuntimeError("The prediction input allowlist changed unexpectedly.")
    for model_name, expected_name in expected_prediction_names.items():
        path = MODEL_FILES[model_name].resolve()
        if path.parent != config.PREDICTIONS_DIR.resolve() or path.name != expected_name:
            raise RuntimeError(f"Unexpected prediction input for {model_name}: {path}")
    if not approved_feature_path.is_file():
        raise FileNotFoundError(approved_feature_path)

    # Deliberately request only the four authorized subgroup fields.
    groups = pd.read_parquet(approved_feature_path, columns=GROUP_COLUMNS).reset_index(drop=True)
    if groups.columns.tolist() != GROUP_COLUMNS:
        raise ValueError(f"Unexpected subgroup columns returned: {groups.columns.tolist()}")

    loaded_predictions: dict[str, pd.DataFrame] = {}
    expected_columns = {"true_label", "predicted_label"}
    for model_name, path in MODEL_FILES.items():
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
        prediction = pd.read_csv(path, usecols=["true_label", "predicted_label"])
        if not expected_columns.issubset(prediction.columns):
            raise ValueError(f"Missing labels in saved predictions: {path}")
        if len(prediction) != len(groups):
            raise ValueError(
                f"Row count mismatch: {model_name} predictions={len(prediction)}, "
                f"approved X_test subgroup rows={len(groups)}"
            )
        for column in ("true_label", "predicted_label"):
            values = set(prediction[column].dropna().unique().tolist())
            if not values.issubset({0, 1}) or prediction[column].isna().any():
                raise ValueError(f"{model_name} {column} must contain only binary labels.")
        loaded_predictions[model_name] = prediction

    # Both saved prediction files must describe the same respondents in the
    # same order before subgroup fields are joined positionally.
    reference = loaded_predictions["CatBoost"]
    other = loaded_predictions["Logistic Regression"]
    if not np.array_equal(reference["true_label"].to_numpy(), other["true_label"].to_numpy()):
        raise ValueError("Saved model prediction files have different true-label row order.")

    rows: list[dict[str, object]] = []
    for model_name, prediction in loaded_predictions.items():
        rows.extend(subgroup_rows(model_name, prediction, groups))

    result = pd.DataFrame(rows).sort_values(
        ["model", "subgroup_variable", "group_value"], kind="stable"
    )
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.joinpath("subgroup_counts").mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    note = [
        "Subgroup counts from saved holdout predictions",
        "holdout_read: true (explicitly approved; subgroup features only)",
        f"feature_input: {approved_feature_path}",
        f"feature_columns_read: {', '.join(GROUP_COLUMNS)}",
        "prediction_inputs:",
        *[f"  - {name}: {path.resolve()} (true_label and predicted_label only)" for name, path in MODEL_FILES.items()],
        f"rows_joined_by_position: {len(groups)}",
        "alignment_evidence: scripts/preprocessing/6_split_data.py creates holdout features and labels in the same stratified split and writes both in order; scripts/experiments/11_final_evaluation.py loads X_test and y_test in order, predicts on X_test, and saves true_label/predicted_label arrays without sorting.",
        "holdout_labels_file_read: false",
        "split_indices_file_read: false",
        "models_loaded_or_evaluated: false",
        "definitions: FPR=FP/(FP+TN); FNR=FN/(FN+TP); 95% intervals use the Wilson score interval.",
        "small_positive_group_flag: positives_lt_30 is true when a subgroup has fewer than 30 positive respondents.",
        f"output: {OUTPUT_PATH.resolve()}",
    ]
    RUN_NOTE_PATH.write_text("\n".join(note) + "\n", encoding="utf-8")

    print(f"feature_rows={len(groups)}; prediction_models={len(loaded_predictions)}; output_rows={len(result)}")
    print(f"holdout_read=true (only approved subgroup columns: {', '.join(GROUP_COLUMNS)})")
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {RUN_NOTE_PATH}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
