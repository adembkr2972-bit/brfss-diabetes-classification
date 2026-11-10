"""Validation-only end-to-end rehearsal of final evaluation and reporting.

This script deliberately reads training labels and validation data only. Its
outputs are isolated under outputs/dry_run, except for the paired-comparison
CSV explicitly requested at outputs/metrics/pairwise_comparisons.csv.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


OUTPUT_ROOT = config.OUTPUTS_DIR / "dry_run"
METRICS_DIR = OUTPUT_ROOT / "metrics"
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"
REPORTS_DIR = OUTPUT_ROOT / "reports"
BOOTSTRAP_REPETITIONS = 500
THRESHOLD_MODES = ("default_0.50", "chosen_validation_threshold")
SUBGROUP_VARIABLES = ("_AGEG5YR", "_SEX", "_BMI5CAT", "_INCOMG1")
CI_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "roc_auc",
    "pr_auc_average_precision",
    "mcc",
    "TN",
    "FP",
    "FN",
    "TP",
)


def display_label(model_name: str) -> str:
    return config.display_model_label(model_name)


def prepare_model(model) -> None:
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
    if classifier is None:
        return
    for attribute in ("n_jobs", "thread_count"):
        if hasattr(classifier, attribute):
            try:
                setattr(classifier, attribute, 1)
            except Exception:
                pass


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn = int(np.count_nonzero((y_true == 0) & (y_pred == 0)))
    fp = int(np.count_nonzero((y_true == 0) & (y_pred == 1)))
    fn = int(np.count_nonzero((y_true == 1) & (y_pred == 0)))
    tp = int(np.count_nonzero((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def metrics_from_counts(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    n: int,
    roc_auc: float = np.nan,
    pr_auc: float = np.nan,
) -> dict[str, float | int]:
    recall_0 = tn / (tn + fp) if (tn + fp) else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) else 0.0
    precision_0 = tn / (tn + fn) if (tn + fn) else 0.0
    precision_1 = tp / (tp + fp) if (tp + fp) else 0.0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if precision_0 + recall_0 else 0.0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if precision_1 + recall_1 else 0.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return {
        "accuracy": (tn + tp) / n if n else np.nan,
        "balanced_accuracy": (recall_0 + recall_1) / 2,
        "macro_precision": (precision_0 + precision_1) / 2,
        "macro_recall": (recall_0 + recall_1) / 2,
        "macro_f1": (f1_0 + f1_1) / 2,
        "positive_precision": precision_1,
        "positive_recall": recall_1,
        "positive_f1": f1_1,
        "roc_auc": float(roc_auc),
        "pr_auc_average_precision": float(pr_auc),
        "mcc": float(mcc),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def grouped_auc_ap(
    group_ids_descending: np.ndarray,
    group_count: int,
    y_sample: np.ndarray,
) -> tuple[float, float]:
    positive_by_group = np.bincount(
        group_ids_descending, weights=y_sample, minlength=group_count
    )
    total_by_group = np.bincount(group_ids_descending, minlength=group_count)
    negative_by_group = total_by_group - positive_by_group
    total_positive = float(positive_by_group.sum())
    total_negative = float(negative_by_group.sum())
    if total_positive == 0 or total_negative == 0:
        return np.nan, np.nan

    cumulative_positive = np.cumsum(positive_by_group)
    cumulative_negative = np.cumsum(negative_by_group)
    tpr = np.concatenate(([0.0], cumulative_positive / total_positive))
    fpr = np.concatenate(([0.0], cumulative_negative / total_negative))
    roc_auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))

    cumulative_total = cumulative_positive + cumulative_negative
    precision_at_group = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive, dtype=float),
        where=cumulative_total > 0,
    )
    recall_increase = positive_by_group / total_positive
    average_precision = float(np.sum(recall_increase * precision_at_group))
    return roc_auc, average_precision


def score_point_metrics(y_true: np.ndarray, scores: np.ndarray | None) -> tuple[float, float]:
    if scores is None:
        return np.nan, np.nan
    return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))


def read_validation_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Read only training labels and validation split inputs."""
    X_val = pd.read_parquet(config.X_VAL_PATH).reset_index(drop=True)
    y_val_frame = pd.read_parquet(config.Y_VAL_PATH)
    y_train_frame = pd.read_parquet(config.Y_TRAIN_PATH)
    y_val = y_val_frame[config.TARGET_BINARY_COL].astype(np.int8).to_numpy()
    y_train = y_train_frame[config.TARGET_BINARY_COL].astype(np.int8).to_numpy()
    if len(X_val) != len(y_val):
        raise ValueError(f"Validation feature/label length mismatch: {len(X_val)} vs {len(y_val)}")
    if set(np.unique(y_val)) != {0, 1}:
        raise ValueError("Validation target must contain both binary classes 0 and 1.")
    missing_groups = [column for column in SUBGROUP_VARIABLES if column not in X_val.columns]
    if missing_groups:
        raise ValueError(f"Subgroup fields missing from validation features: {missing_groups}")
    return X_val, y_train, y_val


def infer_selected_models(X_val: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    thresholds = pd.read_csv(config.CHOSEN_THRESHOLDS_PATH)
    if thresholds["model_name"].duplicated().any():
        raise ValueError("chosen_thresholds.csv contains duplicate model names.")
    threshold_map = thresholds.set_index("model_name")["threshold"].astype(float).to_dict()
    artifact_map = thresholds.set_index("model_name")["chosen_artifact"].astype(str).to_dict()
    if set(config.MODEL_NAMES) != set(threshold_map):
        raise ValueError("The frozen threshold table must contain exactly the nine prespecified models.")

    scores_by_model: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    for model_name in config.MODEL_NAMES:
        threshold = float(threshold_map[model_name])
        artifact_name = artifact_map[model_name]
        if artifact_name not in {"baseline", "tuned"}:
            raise ValueError(f"Unexpected artifact choice for {model_name}: {artifact_name}")
        model_path = config.MODELS_DIR / f"{artifact_name}_{model_name}.joblib"
        print(f"Scoring {display_label(model_name)} from {model_path.name} on validation rows only...")
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        model = joblib.load(model_path)
        prepare_model(model)
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"Selected artifact for {model_name} does not expose predict_proba.")
        scores = np.asarray(model.predict_proba(X_val)[:, 1], dtype=np.float64)
        if scores.shape != (len(X_val),) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid validation probability vector for {model_name}.")
        scores_by_model[model_name] = scores
        predictions[f"{model_name}|default_0.50"] = (scores >= 0.5).astype(np.uint8)
        predictions[f"{model_name}|chosen_validation_threshold"] = (scores >= threshold).astype(np.uint8)
        del model
        gc.collect()
    return scores_by_model, predictions


def add_reference_models(
    y_train: np.ndarray,
    y_val: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> None:
    majority_class = int(pd.Series(y_train).value_counts().idxmax())
    predictions["MajorityBaseline|default_0.50"] = np.full(len(y_val), majority_class, dtype=np.uint8)
    predictions["MajorityBaseline|chosen_validation_threshold"] = predictions[
        "MajorityBaseline|default_0.50"
    ].copy()

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y_train), 1), dtype=np.int8), y_train)
    dummy_X = np.zeros((len(y_val), 1), dtype=np.int8)
    dummy_scores = np.asarray(dummy.predict_proba(dummy_X)[:, 1], dtype=np.float64)
    scores_by_model["DummyClassifier_prior"] = dummy_scores
    predictions["DummyClassifier_prior|default_0.50"] = (dummy_scores >= 0.5).astype(np.uint8)
    predictions["DummyClassifier_prior|chosen_validation_threshold"] = predictions[
        "DummyClassifier_prior|default_0.50"
    ].copy()


def point_rows(
    y_val: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    threshold_map: dict[str, float],
    artifact_map: dict[str, str],
) -> list[dict]:
    rows = []
    model_names = list(config.MODEL_NAMES) + ["MajorityBaseline", "DummyClassifier_prior"]
    for model_name in model_names:
        scores = scores_by_model.get(model_name)
        roc_auc, pr_auc = score_point_metrics(y_val, scores)
        for mode in THRESHOLD_MODES:
            key = f"{model_name}|{mode}"
            y_pred = predictions[key]
            tn, fp, fn, tp = binary_counts(y_val, y_pred)
            metric_values = metrics_from_counts(tn, fp, fn, tp, len(y_val), roc_auc, pr_auc)
            threshold = 0.5 if mode == "default_0.50" else threshold_map.get(model_name, 0.5)
            rows.append(
                {
                    "model_name": model_name,
                    "display_label": display_label(model_name),
                    "artifact": artifact_map.get(model_name, "training-label reference"),
                    "threshold_mode": mode,
                    "threshold": float(threshold),
                    "n_validation": len(y_val),
                    **metric_values,
                }
            )
    return rows


def bootstrap_intervals(
    y_val: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    point_records: list[dict],
    repetitions: int,
) -> dict[str, np.ndarray]:
    n = len(y_val)
    model_names = list(config.MODEL_NAMES) + ["MajorityBaseline", "DummyClassifier_prior"]
    row_keys = [(model, mode) for model in model_names for mode in THRESHOLD_MODES]
    values = {key: {metric: np.empty(repetitions, dtype=np.float64) for metric in CI_METRICS} for key in row_keys}

    score_groups: dict[str, tuple[np.ndarray, int]] = {}
    for model_name, scores in scores_by_model.items():
        _, inverse = np.unique(scores, return_inverse=True)
        group_count = int(inverse.max()) + 1
        score_groups[model_name] = (group_count - 1 - inverse, group_count)

    rng = np.random.default_rng(config.RANDOM_SEED + 1300)
    for repetition in range(repetitions):
        indices = rng.integers(0, n, size=n, dtype=np.int32)
        y_sample = y_val[indices]
        auc_ap: dict[str, tuple[float, float]] = {}
        for model_name, (group_ids, group_count) in score_groups.items():
            auc_ap[model_name] = grouped_auc_ap(group_ids[indices], group_count, y_sample)

        for model_name in model_names:
            auc, ap = auc_ap.get(model_name, (np.nan, np.nan))
            for mode in THRESHOLD_MODES:
                y_pred_sample = predictions[f"{model_name}|{mode}"][indices]
                tn, fp, fn, tp = binary_counts(y_sample, y_pred_sample)
                sample_metrics = metrics_from_counts(tn, fp, fn, tp, n, auc, ap)
                key = (model_name, mode)
                for metric in CI_METRICS:
                    values[key][metric][repetition] = sample_metrics[metric]
        if (repetition + 1) % 100 == 0 or repetition + 1 == repetitions:
            print(f"Bootstrap completed: {repetition + 1}/{repetitions}")

    intervals: dict[str, np.ndarray] = {}
    for key, metric_arrays in values.items():
        intervals[f"{key[0]}|{key[1]}"] = metric_arrays["macro_f1"]
    for row in point_records:
        key = (row["model_name"], row["threshold_mode"])
        metric_arrays = values[key]
        for metric in CI_METRICS:
            finite = metric_arrays[metric][np.isfinite(metric_arrays[metric])]
            low, high = np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan)
            row[f"{metric}_ci95_low"] = float(low)
            row[f"{metric}_ci95_high"] = float(high)
        row["bootstrap_repetitions"] = repetitions
        row["bootstrap_method"] = "ordinary respondent bootstrap; percentile 95% interval; conditional on validation split"
    return intervals


def pairwise_comparisons(
    y_val: np.ndarray,
    predictions: dict[str, np.ndarray],
    point_records: list[dict],
    bootstrap_macro_f1: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, list[str]]:
    chosen_rows = [
        row for row in point_records
        if row["model_name"] in config.MODEL_NAMES and row["threshold_mode"] == "chosen_validation_threshold"
    ]
    top_four = [row["model_name"] for row in sorted(chosen_rows, key=lambda row: row["macro_f1"], reverse=True)[:4]]
    point_macro_f1 = {row["model_name"]: row["macro_f1"] for row in chosen_rows}
    rows = []
    for left_index, model_a in enumerate(top_four):
        pred_a = predictions[f"{model_a}|chosen_validation_threshold"]
        for model_b in top_four[left_index + 1:]:
            pred_b = predictions[f"{model_b}|chosen_validation_threshold"]
            boot_diff = (
                bootstrap_macro_f1[f"{model_a}|chosen_validation_threshold"]
                - bootstrap_macro_f1[f"{model_b}|chosen_validation_threshold"]
            )
            finite = boot_diff[np.isfinite(boot_diff)]
            ci_low, ci_high = np.percentile(finite, [2.5, 97.5])
            p_lower = (np.count_nonzero(finite <= 0) + 1) / (len(finite) + 1)
            p_upper = (np.count_nonzero(finite >= 0) + 1) / (len(finite) + 1)
            both_correct = int(np.count_nonzero((pred_a == y_val) & (pred_b == y_val)))
            a_only_correct = int(np.count_nonzero((pred_a == y_val) & (pred_b != y_val)))
            b_only_correct = int(np.count_nonzero((pred_a != y_val) & (pred_b == y_val)))
            neither_correct = int(np.count_nonzero((pred_a != y_val) & (pred_b != y_val)))
            discordant = a_only_correct + b_only_correct
            mcnemar_p = float(binomtest(a_only_correct, discordant, p=0.5, alternative="two-sided").pvalue) if discordant else 1.0
            rows.append(
                {
                    "model_a": model_a,
                    "display_label_a": display_label(model_a),
                    "model_b": model_b,
                    "display_label_b": display_label(model_b),
                    "comparison_threshold": "chosen validation threshold",
                    "macro_f1_difference_a_minus_b": float(point_macro_f1[model_a] - point_macro_f1[model_b]),
                    "paired_bootstrap_ci95_low": float(ci_low),
                    "paired_bootstrap_ci95_high": float(ci_high),
                    "paired_bootstrap_two_sided_p_raw": float(min(1.0, 2 * min(p_lower, p_upper))),
                    "mcnemar_a_only_correct": a_only_correct,
                    "mcnemar_b_only_correct": b_only_correct,
                    "both_correct": both_correct,
                    "neither_correct": neither_correct,
                    "mcnemar_exact_p_raw": mcnemar_p,
                    "bootstrap_repetitions": len(finite),
                    "multiplicity_adjusted": False,
                }
            )
    return pd.DataFrame(rows), top_four


def subgroup_error_rows(
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    model_names = list(config.MODEL_NAMES) + ["MajorityBaseline", "DummyClassifier_prior"]
    for model_name in model_names:
        for mode in THRESHOLD_MODES:
            y_pred = predictions[f"{model_name}|{mode}"]
            for variable in SUBGROUP_VARIABLES:
                grouping = X_val[variable].astype("object").where(X_val[variable].notna(), "<missing>")
                for subgroup_value, indices in grouping.groupby(grouping, dropna=False, sort=False).groups.items():
                    ix = np.asarray(list(indices), dtype=np.int64)
                    y_group = y_val[ix]
                    pred_group = y_pred[ix]
                    tn, fp, fn, tp = binary_counts(y_group, pred_group)
                    positives = int(np.count_nonzero(y_group == 1))
                    rows.append(
                        {
                            "model_name": model_name,
                            "display_label": display_label(model_name),
                            "threshold_mode": mode,
                            "subgroup_variable": variable,
                            "subgroup_value": str(subgroup_value),
                            "group_size": len(ix),
                            "positive_count": positives,
                            "fewer_than_30_positives": positives < 30,
                            "TN": tn,
                            "FP": fp,
                            "FN": fn,
                            "TP": tp,
                            "FPR_FP_over_FP_plus_TN": fp / (fp + tn) if fp + tn else np.nan,
                            "FNR_FN_over_FN_plus_TP": fn / (fn + tp) if fn + tp else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def save_predictions(
    y_val: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    threshold_map: dict[str, float],
    artifact_map: dict[str, str],
) -> None:
    frames = []
    for key, y_pred in predictions.items():
        model_name, threshold_mode = key.split("|", 1)
        threshold = 0.5 if threshold_mode == "default_0.50" else threshold_map.get(model_name, 0.5)
        score = scores_by_model.get(model_name)
        frames.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "display_label": display_label(model_name),
                    "artifact": artifact_map.get(model_name, "training-label reference"),
                    "threshold_mode": threshold_mode,
                    "threshold": float(threshold),
                    "true_label": y_val,
                    "predicted_label": y_pred,
                    "positive_class_score": score if score is not None else np.nan,
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_csv(PREDICTIONS_DIR / "validation_predictions.csv", index=False)


def write_summary(
    metrics_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    top_four: list[str],
    subgroup_df: pd.DataFrame,
    repetitions: int,
    n_validation: int,
) -> None:
    lines = [
        "# Validation-only rehearsal of steps 11–13",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This run uses only `X_val.parquet`, `y_val.parquet`, `y_train.parquet`, the frozen validation threshold table, and selected model artifacts. It does not open or evaluate the held-out split.",
        "",
        f"- Validation rows: {n_validation:,}",
        f"- Classifiers: {len(config.MODEL_NAMES)}; reference baselines: MajorityBaseline and DummyClassifier_prior.",
        f"- Metric rows: {len(metrics_df)} (both default 0.50 and chosen validation thresholds).",
        f"- Bootstrap repetitions: {repetitions}; intervals are ordinary respondent-level percentile intervals conditional on this validation split, not BRFSS design-based intervals.",
        f"- Top four by chosen-threshold validation macro F1: {', '.join(top_four)}.",
        f"- Pairwise comparisons: {len(pairwise_df)} model pairs; raw p-values are not multiplicity-adjusted.",
        f"- Subgroup rows: {len(subgroup_df)}; FPR=FP/(FP+TN), FNR=FN/(FN+TP).",
        "",
        "## Output files",
        "",
        "- `metrics/validation_metrics.csv` — point estimates and 95% bootstrap confidence intervals.",
        "- `metrics/pairwise_comparisons.csv` — paired bootstrap and exact McNemar comparisons for the top four.",
        "- `reports/subgroup_error_analysis.csv` — subgroup sizes, positive counts, confusion counts, conventional FPR/FNR, and small-positive-group flag.",
        "- `predictions/validation_predictions.csv` — validation labels, scores, and both threshold decisions.",
        "- `reports/validation_comparison_table.csv` — model ranking at the frozen threshold.",
        "",
        "These are validation dry-run outputs and must not be reported as final holdout performance.",
    ]
    (REPORTS_DIR / "dry_run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-only equivalents of experimental steps 11-13.")
    parser.add_argument("--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        raise SystemExit("Use at least 100 bootstrap repetitions for this dry run.")
    for directory in (METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print("DRY RUN: validation split only; held-out split access is disabled.")
    X_val, y_train, y_val = read_validation_inputs()
    threshold_frame = pd.read_csv(config.CHOSEN_THRESHOLDS_PATH)
    threshold_map = threshold_frame.set_index("model_name")["threshold"].astype(float).to_dict()
    artifact_map = threshold_frame.set_index("model_name")["chosen_artifact"].astype(str).to_dict()
    scores_by_model, predictions = infer_selected_models(X_val)
    add_reference_models(y_train, y_val, scores_by_model, predictions)

    expected = (len(config.MODEL_NAMES) + 2) * 2
    if len(predictions) != expected:
        raise AssertionError(f"Expected {expected} prediction arrays, received {len(predictions)}.")
    print("[11/13] Calculating both-threshold metrics and bootstrap confidence intervals.")
    point_records = point_rows(y_val, scores_by_model, predictions, threshold_map, artifact_map)
    bootstrap_macro_f1 = bootstrap_intervals(
        y_val, scores_by_model, predictions, point_records, args.bootstrap_repetitions
    )
    metrics_df = pd.DataFrame(point_records)
    metrics_df.to_csv(METRICS_DIR / "validation_metrics.csv", index=False)

    print("[12/13] Calculating subgroup confusion counts, FPR, and FNR.")
    subgroup_df = subgroup_error_rows(X_val, y_val, predictions)
    subgroup_df.to_csv(REPORTS_DIR / "subgroup_error_analysis.csv", index=False)
    save_predictions(y_val, scores_by_model, predictions, threshold_map, artifact_map)

    print("[13/13] Comparing the top four models with paired bootstrap and McNemar tests.")
    pairs_df, top_four = pairwise_comparisons(y_val, predictions, point_records, bootstrap_macro_f1)
    pairs_df.to_csv(METRICS_DIR / "pairwise_comparisons.csv", index=False)
    external_pairwise_path = config.METRICS_DIR / "pairwise_comparisons.csv"
    external_pairwise_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(external_pairwise_path, index=False)

    chosen_table = metrics_df[metrics_df["threshold_mode"] == "chosen_validation_threshold"].copy()
    chosen_table = chosen_table.sort_values("macro_f1", ascending=False)
    chosen_table["rank_by_macro_f1"] = np.arange(1, len(chosen_table) + 1)
    chosen_table.to_csv(REPORTS_DIR / "validation_comparison_table.csv", index=False)
    write_summary(metrics_df, pairs_df, top_four, subgroup_df, args.bootstrap_repetitions, len(y_val))

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "dry-run-on-validation",
        "holdout_read": False,
        "input_data_paths": [
            str(config.X_VAL_PATH.relative_to(config.PROJECT_ROOT)),
            str(config.Y_VAL_PATH.relative_to(config.PROJECT_ROOT)),
            str(config.Y_TRAIN_PATH.relative_to(config.PROJECT_ROOT)),
            str(config.CHOSEN_THRESHOLDS_PATH.relative_to(config.PROJECT_ROOT)),
        ],
        "n_validation": len(y_val),
        "models": list(config.MODEL_NAMES) + ["MajorityBaseline", "DummyClassifier_prior"],
        "threshold_modes": list(THRESHOLD_MODES),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "pairwise_top_four": top_four,
        "outputs": [
            str((METRICS_DIR / "validation_metrics.csv").relative_to(config.PROJECT_ROOT)),
            str((METRICS_DIR / "pairwise_comparisons.csv").relative_to(config.PROJECT_ROOT)),
            str((REPORTS_DIR / "subgroup_error_analysis.csv").relative_to(config.PROJECT_ROOT)),
            str((PREDICTIONS_DIR / "validation_predictions.csv").relative_to(config.PROJECT_ROOT)),
            str(external_pairwise_path.relative_to(config.PROJECT_ROOT)),
        ],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(metrics_df)} metric rows for {len(config.MODEL_NAMES)} classifiers and 2 baselines.")
    print(f"Validation metric table: {METRICS_DIR / 'validation_metrics.csv'}")
    print(f"Pairwise comparisons: {external_pairwise_path}")
    print(f"Subgroup analysis: {REPORTS_DIR / 'subgroup_error_analysis.csv'}")
    print(f"Summary: {REPORTS_DIR / 'dry_run_summary.md'}")


if __name__ == "__main__":
    main()
