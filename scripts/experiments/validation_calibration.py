"""Calibration diagnostics for the nine chosen artifacts on validation data."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss

from validation_experiment_common import (
    chosen_thresholds,
    config,
    display_label,
    ensure_output_dir,
    guarded_path,
    load_validation,
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
    parser.add_argument("--out-dir", default="outputs/validation_calibration")
    parser.add_argument("--ablation-score-dir", default="outputs/imbalance_ablation/dry_run/score_cache")
    return parser.parse_args()


def calibration_fit(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))

    def objective(params: np.ndarray) -> float:
        linear = params[0] + params[1] * logits
        return float(np.mean(np.logaddexp(0.0, linear) - y * linear))

    result = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Calibration logistic fit failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def equal_frequency_bins(y: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> tuple[list[dict[str, object]], float]:
    order = np.argsort(probabilities, kind="mergesort")
    groups = np.array_split(order, n_bins)
    rows: list[dict[str, object]] = []
    total = len(y)
    ece = 0.0
    for bin_number, indices in enumerate(groups, start=1):
        mean_prediction = float(np.mean(probabilities[indices]))
        observed_rate = float(np.mean(y[indices]))
        count = int(len(indices))
        ece += count / total * abs(mean_prediction - observed_rate)
        rows.append(
            {
                "bin": bin_number,
                "mean_predicted_probability": mean_prediction,
                "observed_positive_rate": observed_rate,
                "bin_count": count,
            }
        )
    return rows, float(ece)


def markdown_summary(metrics: pd.DataFrame) -> str:
    brier_min = metrics.loc[metrics["brier_score"].idxmin()]
    brier_max = metrics.loc[metrics["brier_score"].idxmax()]
    ll_min = metrics.loc[metrics["log_loss"].idxmin()]
    svm = metrics.loc[metrics["model_name"] == "SVM"].iloc[0]
    others = metrics.loc[metrics["model_name"] != "SVM"]
    other_gt_half = (others["fraction_probability_gt_0_5"].min(), others["fraction_probability_gt_0_5"].max())
    other_gt_cut = (others["fraction_probability_gt_chosen_threshold"].min(), others["fraction_probability_gt_chosen_threshold"].max())
    ece_min, ece_max = metrics["ece_10_equal_frequency"].min(), metrics["ece_10_equal_frequency"].max()
    lines = [
        "# Validation calibration summary",
        "",
        f"The lowest Brier score was for {brier_min['display_label']} ({brier_min['brier_score']:.5f}); the highest was for {brier_max['display_label']} ({brier_max['brier_score']:.5f}).",
        f"The lowest log loss was for {ll_min['display_label']} ({ll_min['log_loss']:.5f}); log loss is reported as a validation probability score.",
        f"Across the nine models, calibration intercepts ranged from {metrics['calibration_intercept'].min():.5f} to {metrics['calibration_intercept'].max():.5f}, and slopes ranged from {metrics['calibration_slope'].min():.5f} to {metrics['calibration_slope'].max():.5f}.",
        f"SVM probabilities exceeded 0.50 for {svm['fraction_probability_gt_0_5']:.5f} of validation rows, compared with {other_gt_half[0]:.5f} to {other_gt_half[1]:.5f} for the other models.",
        f"At the saved SVM threshold of {svm['chosen_threshold']:.2f}, its probability exceeded that cutoff for {svm['fraction_probability_gt_chosen_threshold']:.5f} of rows; the other models ranged from {other_gt_cut[0]:.5f} to {other_gt_cut[1]:.5f} at their own saved thresholds.",
        f"Expected calibration error ranged from {ece_min:.5f} to {ece_max:.5f}; ten equal-frequency bins were formed by stable sorting probabilities and splitting rows into ten groups whose counts differ by at most one.",
        "These are descriptive validation results and do not establish why any model's probabilities have a particular distribution.",
    ]
    return "\n\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.out_dir)
    started_at, started_clock = start_run()
    x_val, y_val = load_validation()
    threshold_frame = chosen_thresholds()
    score_columns: dict[str, np.ndarray] = {"y_true": y_val}
    metric_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    cache_dir = guarded_path(config.PROJECT_ROOT / Path(args.ablation_score_dir))

    for model_name in model_names():
        row = threshold_frame.loc[model_name]
        artifact_path = config.PROJECT_ROOT / Path(str(row["artifact_path"]))
        cache_path = guarded_path(cache_dir / f"{model_name}.csv")
        probability_source = f"scored from selected artifact {row['artifact_path']}"
        probabilities = None
        if str(row["chosen_artifact"]).lower() == "baseline" and cache_path.exists():
            cached = safe_read_csv(cache_path)
            valid_cache = (
                len(cached) == len(y_val)
                and {"y_true", "weighted_probability"}.issubset(cached.columns)
                and np.array_equal(cached["y_true"].astype(int).to_numpy(), y_val)
            )
            if valid_cache:
                probabilities = cached["weighted_probability"].to_numpy(dtype=float)
                probability_source = str(cache_path.relative_to(config.PROJECT_ROOT))
        if probabilities is None:
            print(f"Scoring {model_name} on validation rows.", flush=True)
            model = safe_load_joblib(artifact_path)
            probabilities = positive_scores(model, x_val)
            del model
            gc.collect()
        else:
            print(f"Reused {model_name} validation scores from the ablation cache.", flush=True)
        score_columns[f"probability__{model_name}"] = probabilities
        bins, ece = equal_frequency_bins(y_val, probabilities)
        for bin_row in bins:
            reliability_rows.append(
                {
                    "model_name": model_name,
                    "display_label": display_label(model_name),
                    **bin_row,
                }
            )
        intercept, slope = calibration_fit(y_val, probabilities)
        cutoff = float(row["threshold"])
        metric_rows.append(
            {
                "model_name": model_name,
                "display_label": display_label(model_name),
                "artifact_path": str(row["artifact_path"]),
                "validation_probability_source": probability_source,
                "chosen_threshold": cutoff,
                "brier_score": float(brier_score_loss(y_val, probabilities)),
                "log_loss": float(log_loss(y_val, probabilities, labels=[0, 1])),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "ece_10_equal_frequency": ece,
                "fraction_probability_gt_0_5": float(np.mean(probabilities > 0.5)),
                "fraction_probability_gt_chosen_threshold": float(np.mean(probabilities > cutoff)),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    reliability = pd.DataFrame(reliability_rows)
    safe_write_csv(metrics, out_dir / "calibration_metrics.csv", index=False)
    safe_write_csv(reliability, out_dir / "reliability_bins.csv", index=False)
    safe_write_csv(pd.DataFrame(score_columns), out_dir / "validation_scores.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    for ax, model_name in zip(axes.flat, model_names()):
        subset = reliability.loc[reliability["model_name"] == model_name]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.plot(subset["mean_predicted_probability"], subset["observed_positive_rate"], marker="o", linewidth=1.4)
        ax.set_title(display_label(model_name), fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    fig.supxlabel("Mean predicted probability")
    fig.supylabel("Observed positive rate")
    fig.tight_layout()
    fig.savefig(guarded_path(out_dir / "reliability_curves.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    safe_write_text(markdown_summary(metrics), out_dir / "calibration_summary.md")
    elapsed = write_run_note(
        out_dir,
        started_at,
        started_clock,
        [
            f"validation_rows={len(y_val)}",
            "probabilities were scored from the nine chosen artifacts listed in chosen_thresholds.csv",
            "equal-frequency bins use a stable sort followed by ten near-equal row-count groups",
        ],
    )
    print(f"Wrote calibration outputs in {out_dir} ({elapsed:.1f}s).", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        print("holdout_read: false", flush=True)
