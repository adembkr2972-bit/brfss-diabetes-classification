"""Validation-only comparison of baseline weighted and unweighted models."""

from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from validation_experiment_common import (
    BOOTSTRAP_SEED,
    best_threshold,
    chosen_thresholds,
    config,
    display_label,
    ensure_output_dir,
    guarded_path,
    load_train,
    load_validation,
    metric_values,
    model_names,
    positive_scores,
    safe_load_joblib,
    safe_read_csv,
    safe_write_csv,
    safe_write_text,
    start_run,
    write_run_note,
)

guarded_path(__file__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/imbalance_ablation")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--max-train-rows", type=int, default=0, help="Diagnostic cap; zero uses all training rows.")
    return parser.parse_args()


def stratified_subset(features: pd.DataFrame, labels: np.ndarray, row_count: int) -> tuple[pd.DataFrame, np.ndarray]:
    if row_count >= len(labels):
        return features, labels
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=row_count, random_state=config.RANDOM_SEED)
    selected, _ = next(splitter.split(features, labels))
    return features.iloc[selected], labels[selected]


def remove_weighting(model_name: str, pipeline) -> str:
    classifier = pipeline.named_steps["classifier"]
    params = classifier.get_params(deep=False)
    if model_name in {"LogisticRegression", "DecisionTree", "RandomForest", "ExtraTrees", "SVM", "LightGBM"}:
        if "class_weight" not in params:
            raise ValueError(f"{model_name} does not expose the expected class_weight setting.")
        classifier.set_params(class_weight=None)
        return "class_weight=None"
    if model_name == "XGBoost":
        classifier.set_params(scale_pos_weight=1)
        return "scale_pos_weight=1"
    if model_name == "CatBoost":
        params_without_weights = classifier.get_params(deep=False)
        params_without_weights.pop("auto_class_weights", None)
        pipeline.set_params(classifier=type(classifier)(**params_without_weights))
        return "auto_class_weights omitted (CatBoost unweighted default)"
    if model_name == "MLPClassifier":
        return "no sample_weight passed to fit"
    raise ValueError(f"No documented weighting removal rule for {model_name}.")


def metric_row(
    model_name: str,
    arm: str,
    y: np.ndarray,
    scores: np.ndarray,
    threshold_file_row: pd.Series,
    source_path: str,
    fit_rows: int,
) -> dict[str, object]:
    chosen_t = float(threshold_file_row["threshold"])
    threshold, best_f1 = best_threshold(y, scores)
    at_default = metric_values(y, scores, 0.5)
    at_best = metric_values(y, scores, threshold)
    score_row: dict[str, object] = {
        "model_name": model_name,
        "display_label": display_label(model_name),
        "arm": arm,
        "source_artifact": source_path,
        "fit_rows": fit_rows,
        "validation_rows": len(y),
        "chosen_threshold": threshold,
        "chosen_threshold_rule": "max validation macro F1; ties nearest 0.50, then lower threshold",
        "macro_f1_at_0_5": at_default["macro_f1"],
        "macro_f1_at_best_threshold": best_f1,
        "macro_recall_at_best_threshold": at_best["macro_recall"],
        "positive_precision_at_best_threshold": at_best["positive_precision"],
        "positive_recall_at_best_threshold": at_best["positive_recall"],
        "accuracy_at_best_threshold": at_best["accuracy"],
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc_average_precision": float(average_precision_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
    }
    if arm == "weighted" and str(threshold_file_row["chosen_artifact"]).lower() == "baseline":
        reported_f1 = f1_score(y, scores >= chosen_t, average="macro", zero_division=0)
        expected_f1 = float(threshold_file_row["validation_macro_f1_at_threshold"])
        score_row["baseline_chosen_threshold_macro_f1"] = float(reported_f1)
        score_row["baseline_chosen_threshold_f1_difference_vs_file"] = float(reported_f1 - expected_f1)
    else:
        score_row["baseline_chosen_threshold_macro_f1"] = np.nan
        score_row["baseline_chosen_threshold_f1_difference_vs_file"] = np.nan
    return score_row


def paired_bootstrap(
    y: np.ndarray,
    score_map: dict[str, dict[str, np.ndarray]],
    threshold_map: dict[str, dict[str, float]],
    reps: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_names = ["best_threshold_macro_f1", "macro_f1_at_0_5", "pr_auc_average_precision"]
    differences = {name: {metric: np.empty(reps, dtype=float) for metric in metric_names} for name in model_names()}
    for b in range(reps):
        indices = rng.integers(0, len(y), size=len(y))
        yb = y[indices]
        for name in model_names():
            weighted = score_map[name]["weighted"][indices]
            unweighted = score_map[name]["unweighted"][indices]
            for arm, values in (("weighted", weighted), ("unweighted", unweighted)):
                threshold = threshold_map[name][arm]
                if arm == "weighted":
                    pred_w = values >= threshold
                    f1_w_best = f1_score(yb, pred_w, average="macro", zero_division=0)
                    f1_w_default = f1_score(yb, values >= 0.5, average="macro", zero_division=0)
                else:
                    pred_u = values >= threshold
                    f1_u_best = f1_score(yb, pred_u, average="macro", zero_division=0)
                    f1_u_default = f1_score(yb, values >= 0.5, average="macro", zero_division=0)
            differences[name]["best_threshold_macro_f1"][b] = f1_w_best - f1_u_best
            differences[name]["macro_f1_at_0_5"][b] = f1_w_default - f1_u_default
            if np.unique(yb).size == 2:
                ap_w = average_precision_score(yb, weighted)
                ap_u = average_precision_score(yb, unweighted)
                differences[name]["pr_auc_average_precision"][b] = ap_w - ap_u
            else:
                differences[name]["pr_auc_average_precision"][b] = np.nan
        if (b + 1) % 100 == 0 or b + 1 == reps:
            print(f"Bootstrap resamples: {b + 1}/{reps}", flush=True)

    rows: list[dict[str, object]] = []
    for metric in metric_names:
        family_rows: list[dict[str, object]] = []
        for name in model_names():
            arr = differences[name][metric]
            arr = arr[np.isfinite(arr)]
            p_raw = min(1.0, 2 * min((np.sum(arr <= 0) + 1) / (len(arr) + 1), (np.sum(arr >= 0) + 1) / (len(arr) + 1)))
            point_w = score_map[name]["weighted"]
            point_u = score_map[name]["unweighted"]
            if metric == "best_threshold_macro_f1":
                point = f1_score(y, point_w >= threshold_map[name]["weighted"], average="macro", zero_division=0) - f1_score(
                    y, point_u >= threshold_map[name]["unweighted"], average="macro", zero_division=0
                )
            elif metric == "macro_f1_at_0_5":
                point = f1_score(y, point_w >= 0.5, average="macro", zero_division=0) - f1_score(
                    y, point_u >= 0.5, average="macro", zero_division=0
                )
            else:
                point = average_precision_score(y, point_w) - average_precision_score(y, point_u)
            family_rows.append(
                {
                    "model_name": name,
                    "display_label": display_label(name),
                    "metric": metric,
                    "weighted_minus_unweighted": float(point),
                    "ci95_low": float(np.percentile(arr, 2.5)),
                    "ci95_high": float(np.percentile(arr, 97.5)),
                    "p_raw_two_sided": float(p_raw),
                    "bootstrap_reps": reps,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
        order = np.argsort([r["p_raw_two_sided"] for r in family_rows])
        adjusted = np.empty(len(family_rows), dtype=float)
        running = 0.0
        p_values = np.asarray([r["p_raw_two_sided"] for r in family_rows], dtype=float)
        for rank, idx in enumerate(order):
            running = max(running, (len(family_rows) - rank) * p_values[idx])
            adjusted[idx] = min(1.0, running)
        for row, p_holm in zip(family_rows, adjusted):
            row["p_holm_across_nine_for_metric"] = float(p_holm)
            rows.append(row)
    return pd.DataFrame(rows)


def make_summary(metrics: pd.DataFrame, differences: pd.DataFrame, reps: int, capped: bool) -> str:
    best = differences[differences["metric"] == "best_threshold_macro_f1"].set_index("model_name")
    lines = [
        "# Validation weighted-versus-unweighted ablation",
        "",
        "The table reports weighted minus unweighted differences for each model. Confidence intervals are paired respondent bootstrap percentile intervals.",
        "",
        "| Model | Best-threshold macro-F1 difference | 95% percentile CI |",
        "|---|---:|---:|",
    ]
    for name in model_names():
        row = best.loc[name]
        lines.append(
            f"| {display_label(name)} | {row['weighted_minus_unweighted']:.5f} | [{row['ci95_low']:.5f}, {row['ci95_high']:.5f}] |"
        )
    if reps < 1000 or capped:
        sentences = [
            f"This diagnostic run used {reps} bootstrap resamples and {'a capped training subset' if capped else 'the configured training split'}; its intervals are pipeline checks and are not inferential results.",
            "Thresholds were optimized independently for each arm on validation data using the registered grid and tie rule.",
            "The paired resampling used the same respondent indices for both arms and every model.",
            "The intervals should not be used to decide whether weighting changed performance because this is a diagnostic run.",
            "Threshold selection can compensate for weighting, so a small best-threshold difference does not imply weighting was pointless.",
        ]
    else:
        above = [display_label(n) for n in model_names() if best.loc[n, "ci95_low"] > 0]
        below = [display_label(n) for n in model_names() if best.loc[n, "ci95_high"] < 0]
        contains = [display_label(n) for n in model_names() if best.loc[n, "ci95_low"] <= 0 <= best.loc[n, "ci95_high"]]
        sentences = [
            f"The 95% interval for best-threshold macro F1 was wholly above zero for: {', '.join(above) if above else 'none'}.",
            f"It was wholly below zero for: {', '.join(below) if below else 'none'}.",
            f"It included zero for: {', '.join(contains) if contains else 'none'}.",
            "These intervals describe paired respondent resampling on this validation sample and are not BRFSS design-based intervals.",
            "Threshold selection can compensate for weighting, so a small best-threshold difference does not imply weighting was pointless.",
        ]
    sentences.append(
        "For macro-F1 bootstrap replicates, each arm's threshold was held fixed at the value selected on the full validation set; these intervals are conditional on those thresholds and do not include threshold-selection uncertainty."
    )
    lines.extend(["", " ".join(sentences), "", f"Bootstrap resamples: {reps}; seed: {BOOTSTRAP_SEED}."])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 1 or args.max_train_rows < 0:
        raise ValueError("bootstrap reps must be positive and max train rows non-negative")
    out_dir = ensure_output_dir(args.out_dir)
    started_at, started_clock = start_run()
    x_train, y_train = load_train()
    x_val, y_val = load_validation()
    threshold_frame = chosen_thresholds()
    if args.max_train_rows and len(y_train) > args.max_train_rows:
        x_fit, y_fit = stratified_subset(x_train, y_train, args.max_train_rows)
        capped = True
    else:
        x_fit, y_fit = x_train, y_train
        capped = False
    print(f"Loaded {len(y_fit):,} training rows and {len(y_val):,} validation rows.", flush=True)

    score_map: dict[str, dict[str, np.ndarray]] = {}
    threshold_map: dict[str, dict[str, float]] = {}
    metric_rows: list[dict[str, object]] = []
    baseline_dir = guarded_path(config.MODELS_DIR)
    cache_dir = ensure_output_dir(out_dir / "score_cache")

    for model_name in model_names():
        artifact_path = guarded_path(baseline_dir / f"baseline_{model_name}.joblib")
        if not artifact_path.exists():
            raise FileNotFoundError(f"Baseline artifact not found: {artifact_path}")
        cache_path = guarded_path(cache_dir / f"{model_name}.csv")
        if cache_path.exists():
            try:
                cached = safe_read_csv(cache_path)
                valid_cache = (
                    len(cached) == len(y_val)
                    and {"y_true", "weighted_probability", "unweighted_probability", "training_rows_used"}.issubset(cached.columns)
                    and np.array_equal(cached["y_true"].astype(int).to_numpy(), y_val)
                    and int(cached["training_rows_used"].iloc[0]) == len(y_fit)
                )
            except Exception:
                valid_cache = False
            if valid_cache:
                weighted_scores = cached["weighted_probability"].to_numpy(dtype=float)
                unweighted_scores = cached["unweighted_probability"].to_numpy(dtype=float)
                score_map[model_name] = {"weighted": weighted_scores, "unweighted": unweighted_scores}
                threshold_map[model_name] = {}
                selected_row = threshold_frame.loc[model_name]
                for arm, scores in score_map[model_name].items():
                    chosen_t, _ = best_threshold(y_val, scores)
                    threshold_map[model_name][arm] = chosen_t
                    source = str(artifact_path) if arm == "weighted" else f"refit from {artifact_path.name}; resumed cached diagnostic"
                    fit_rows = int(
                        min(len(y_train), config.SVM_BASELINE_MAX_ROWS)
                        if arm == "weighted" and model_name == "SVM"
                        else len(y_train)
                        if arm == "weighted"
                        else cached["fit_rows_unweighted"].iloc[0]
                    )
                    metric_rows.append(metric_row(model_name, arm, y_val, scores, selected_row, source, fit_rows))
                print(f"Reused saved validation score cache for {model_name}.", flush=True)
                continue
        print(f"Scoring and cloning {model_name} from {artifact_path.name}.", flush=True)
        weighted_pipeline = safe_load_joblib(artifact_path)
        weighted_scores = positive_scores(weighted_pipeline, x_val)
        if not hasattr(weighted_pipeline, "named_steps") or "classifier" not in weighted_pipeline.named_steps:
            raise ValueError(f"Unexpected baseline pipeline structure for {model_name}.")
        unweighted_pipeline = clone(weighted_pipeline)
        weighting_removed = remove_weighting(model_name, unweighted_pipeline)
        del weighted_pipeline
        gc.collect()

        if model_name == "SVM" and len(x_fit) > config.SVM_BASELINE_MAX_ROWS:
            unweighted_x, unweighted_y = stratified_subset(x_fit, y_fit, config.SVM_BASELINE_MAX_ROWS)
        else:
            unweighted_x, unweighted_y = x_fit, y_fit
        unweighted_pipeline.fit(unweighted_x, unweighted_y)
        unweighted_scores = positive_scores(unweighted_pipeline, x_val)
        del unweighted_pipeline
        gc.collect()

        score_map[model_name] = {"weighted": weighted_scores, "unweighted": unweighted_scores}
        threshold_map[model_name] = {}
        selected_row = threshold_frame.loc[model_name]
        for arm, scores in score_map[model_name].items():
            chosen_t, _ = best_threshold(y_val, scores)
            threshold_map[model_name][arm] = chosen_t
            source = str(artifact_path) if arm == "weighted" else f"refit from {artifact_path.name}; {weighting_removed}"
            fit_rows = int(
                len(unweighted_y)
                if arm == "unweighted"
                else min(len(y_train), config.SVM_BASELINE_MAX_ROWS)
                if model_name == "SVM"
                else len(y_train)
            )
            metric_rows.append(metric_row(model_name, arm, y_val, scores, selected_row, source, fit_rows))
        cache_frame = pd.DataFrame(
            {
                "y_true": y_val,
                "weighted_probability": weighted_scores,
                "unweighted_probability": unweighted_scores,
                "training_rows_used": len(y_fit),
                "fit_rows_unweighted": len(unweighted_y),
            }
        )
        safe_write_csv(cache_frame, cache_path, index=False)
        print(f"Finished {model_name}.", flush=True)

    metrics = pd.DataFrame(metric_rows)
    differences = paired_bootstrap(y_val, score_map, threshold_map, args.bootstrap_reps)
    safe_write_csv(metrics, out_dir / "ablation_metrics.csv", index=False)
    safe_write_csv(differences, out_dir / "ablation_paired_differences.csv", index=False)
    summary = make_summary(metrics, differences, args.bootstrap_reps, capped)
    safe_write_text(summary, out_dir / "ablation_summary.md")
    details = [
        f"bootstrap_reps={args.bootstrap_reps}; bootstrap_seed={BOOTSTRAP_SEED}",
        f"training_rows_used_for_unweighted_refits={len(y_fit)}; training_rows_original={len(y_train)}",
        f"validation_rows={len(y_val)}; dry_run_train_cap={args.max_train_rows}",
        "weighted arms reused the nine saved baseline artifacts; unweighted arms changed only the documented class-weighting mechanism",
        "SVM unweighted fit used the configured 50,000-row stratified subsample when the available fit split exceeded that cap",
        "macro-F1 bootstrap replicates held each arm threshold fixed at its value selected on the full validation set; intervals do not include threshold-selection uncertainty",
    ]
    elapsed = write_run_note(out_dir, started_at, started_clock, details)
    print(f"Wrote ablation_metrics.csv and ablation_paired_differences.csv in {out_dir} ({elapsed:.1f}s).", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        print("holdout_read: false", flush=True)
