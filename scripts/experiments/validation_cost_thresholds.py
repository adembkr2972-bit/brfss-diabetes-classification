"""Threshold and illustrative misclassification-cost analysis on validation scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from validation_experiment_common import (
    THRESHOLDS,
    chosen_thresholds,
    config,
    display_label,
    ensure_output_dir,
    guarded_path,
    model_names,
    safe_read_csv,
    safe_write_csv,
    safe_write_text,
    start_run,
    write_run_note,
)

guarded_path(__file__)

COST_RATIOS = [1, 2, 3, 5, 10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/validation_costs")
    parser.add_argument("--scores-file", default="outputs/validation_calibration/validation_scores.csv")
    return parser.parse_args()


def counts(y: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    pred = scores >= threshold
    actual = y == 1
    tn = int(np.sum(~actual & ~pred))
    fp = int(np.sum(~actual & pred))
    fn = int(np.sum(actual & ~pred))
    tp = int(np.sum(actual & pred))
    return tn, fp, fn, tp


def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.out_dir)
    started_at, started_clock = start_run()
    scores_path = guarded_path(config.PROJECT_ROOT / Path(args.scores_file))
    score_frame = safe_read_csv(scores_path)
    if "y_true" not in score_frame:
        raise ValueError("Validation score cache is missing y_true.")
    y = score_frame["y_true"].astype(int).to_numpy()
    threshold_frame = chosen_thresholds()
    score_map = {name: score_frame[f"probability__{name}"].to_numpy(dtype=float) for name in model_names()}
    grid_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []

    for name in model_names():
        scores = score_map[name]
        saved_t = float(threshold_frame.loc[name, "threshold"])
        f1_grid: dict[float, float] = {}
        for threshold in THRESHOLDS:
            tn, fp, fn, tp = counts(y, scores, float(threshold))
            positive_precision = tp / (tp + fp) if tp + fp else 0.0
            positive_recall = tp / (tp + fn) if tp + fn else 0.0
            f1_pos = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
            f1_neg = (2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
            macro_f1 = (f1_pos + f1_neg) / 2
            f1_grid[float(threshold)] = macro_f1
            grid_rows.append(
                {
                    "model_name": name,
                    "display_label": display_label(name),
                    "threshold": float(threshold),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "positive_precision": positive_precision,
                    "positive_recall": positive_recall,
                    "macro_f1": macro_f1,
                    "saved_macro_f1_optimal_threshold": saved_t,
                }
            )
        macro_max = max(f1_grid.values())
        best_candidates = [t for t, value in f1_grid.items() if abs(value - macro_max) <= 1e-12]
        macro_threshold = min(best_candidates, key=lambda t: (abs(t - 0.5), t))
        for ratio in COST_RATIOS:
            options = []
            for threshold in THRESHOLDS:
                tn, fp, fn, tp = counts(y, scores, float(threshold))
                options.append((fp + ratio * fn, float(threshold), tn, fp, fn, tp))
            min_cost = min(row[0] for row in options)
            optimum = min((row for row in options if row[0] == min_cost), key=lambda row: (abs(row[1] - 0.5), row[1]))
            cost_rows.append(
                {
                    "model_name": name,
                    "display_label": display_label(name),
                    "cost_fn_over_fp": ratio,
                    "minimum_fp_plus_ratio_fn": int(optimum[0]),
                    "cost_optimal_threshold": optimum[1],
                    "macro_f1_optimal_threshold": macro_threshold,
                    "cost_threshold_minus_macro_f1_threshold": optimum[1] - macro_threshold,
                    "macro_f1_at_macro_f1_threshold": f1_grid[macro_threshold],
                    "macro_f1_at_cost_threshold": f1_grid[optimum[1]],
                    "saved_validation_threshold": saved_t,
                    "cost_at_saved_validation_threshold": int(
                        counts(y, scores, saved_t)[1] + ratio * counts(y, scores, saved_t)[2]
                    ),
                }
            )

    cost_frame = pd.DataFrame(cost_rows)
    for ratio in COST_RATIOS:
        subset = cost_frame.loc[cost_frame["cost_fn_over_fp"] == ratio].sort_values(
            ["minimum_fp_plus_ratio_fn", "model_name"], ascending=[True, True]
        )
        for rank, row in enumerate(subset.itertuples(index=False), start=1):
            rank_rows.append(
                {
                    "cost_fn_over_fp": ratio,
                    "model_name": row.model_name,
                    "display_label": row.display_label,
                    "minimum_fp_plus_ratio_fn": row.minimum_fp_plus_ratio_fn,
                    "cost_optimal_threshold": row.cost_optimal_threshold,
                    "rank_by_minimum_cost": rank,
                }
            )
    ranking = pd.DataFrame(rank_rows)
    safe_write_csv(pd.DataFrame(grid_rows), out_dir / "threshold_grid_validation.csv", index=False)
    safe_write_csv(cost_frame, out_dir / "cost_optimal_thresholds.csv", index=False)
    safe_write_csv(ranking, out_dir / "cost_ranking_by_ratio.csv", index=False)

    lines = [
        "# Validation cost-threshold sensitivity",
        "",
        "Cost ratios are illustrative sensitivity settings for cost(FN) / cost(FP), not clinical cost estimates.",
        "",
        "| Ratio | Lowest-cost model | Minimum FP + ratio * FN | Threshold |",
        "|---:|---|---:|---:|",
    ]
    for ratio in COST_RATIOS:
        winner = ranking.loc[ranking["cost_fn_over_fp"] == ratio].sort_values("rank_by_minimum_cost").iloc[0]
        lines.append(
            f"| {ratio} | {winner['display_label']} | {int(winner['minimum_fp_plus_ratio_fn'])} | {winner['cost_optimal_threshold']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The full rank ordering for each ratio is in `cost_ranking_by_ratio.csv`; ties in minimum cost receive sequential ranks in model-name order.",
            "The `cost_optimal_thresholds.csv` file gives each model's cost-optimal threshold and its difference from that model's validation macro-F1-optimal threshold.",
            "All counts and costs use the validation probability cache written by `validation_calibration.py` and the threshold grid from 0.05 through 0.95 in 0.01 steps.",
            "These descriptive results do not establish clinical utility or clinical costs.",
        ]
    )
    safe_write_text("\n".join(lines) + "\n", out_dir / "cost_threshold_summary.md")
    elapsed = write_run_note(
        out_dir,
        started_at,
        started_clock,
        [
            f"validation_rows={len(y)}",
            f"cost_fn_over_fp={COST_RATIOS}",
            f"probability_source={scores_path.relative_to(config.PROJECT_ROOT)}",
            "all scores reused from validation_calibration/validation_scores.csv; no model was refit",
        ],
    )
    print(f"Wrote cost-threshold outputs in {out_dir} ({elapsed:.1f}s).", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        print("holdout_read: false", flush=True)
