import logging
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("10_model_elimination")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOGS_DIR / "10_model_elimination.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def dominated_models(df: pd.DataFrame, active_models: set[str]) -> set[str]:
    dominated = set()
    active_df = df[df["model_name"].isin(active_models)]
    for row in active_df.itertuples(index=False):
        for other in active_df.itertuples(index=False):
            if row.model_name == other.model_name:
                continue
            if (
                other.f1_macro > row.f1_macro
                and other.roc_auc > row.roc_auc
                and other.recall_macro > row.recall_macro
            ):
                dominated.add(row.model_name)
                break
    return dominated


def apply_rules(df: pd.DataFrame, auc_threshold: float, f1_threshold: float, gap_threshold: float) -> tuple[set[str], dict[str, list[str]]]:
    survivors = set(df["model_name"].dropna())
    reasons: dict[str, list[str]] = {model_name: [] for model_name in survivors}

    for row in df.itertuples(index=False):
        if pd.isna(row.roc_auc) or pd.isna(row.f1_macro) or pd.isna(row.overfit_gap):
            survivors.discard(row.model_name)
            reasons.setdefault(row.model_name, []).append("Eliminated due to failed training/evaluation (missing primary metrics)")
    for row in df.itertuples(index=False):
        if pd.notna(row.roc_auc) and row.roc_auc < auc_threshold:
            survivors.discard(row.model_name)
            reasons.setdefault(row.model_name, []).append(f"ROC-AUC below {auc_threshold:.2f}")
    for row in df.itertuples(index=False):
        if row.model_name in survivors and pd.notna(row.f1_macro) and row.f1_macro < f1_threshold:
            survivors.discard(row.model_name)
            reasons.setdefault(row.model_name, []).append(f"F1-macro below {f1_threshold:.2f}")
    for row in df.itertuples(index=False):
        if row.model_name in survivors and pd.notna(row.overfit_gap) and row.overfit_gap > gap_threshold:
            survivors.discard(row.model_name)
            reasons.setdefault(row.model_name, []).append(f"Overfit gap above {gap_threshold:.2f}")

    dominated = dominated_models(df, survivors)
    for model_name in dominated:
        survivors.discard(model_name)
        reasons.setdefault(model_name, []).append("Strictly dominated on F1-macro, ROC-AUC, and recall-macro")

    return survivors, reasons


def main() -> None:
    for path in [config.LOGS_DIR, config.REPORTS_DIR]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting model elimination.")

    validation_df = pd.read_csv(config.FULL_VALIDATION_METRICS_PATH)
    overfit_df = pd.read_csv(config.OVERFITTING_ANALYSIS_PATH)
    merged = validation_df.merge(overfit_df, on="model_name", how="left")
    merged["display_label"] = merged["model_name"].map(config.display_model_label)

    # The research question compares all nine prespecified classifiers, so
    # validation heuristics must not discard weaker models before test review.
    successful = merged["f1_macro"].notna()
    merged["shortlisted"] = (
        merged["model_name"].isin(config.MODEL_NAMES) & successful
    ).astype(int)
    merged.to_csv(config.MODEL_SHORTLIST_PATH, index=False)

    retained = merged.loc[merged["shortlisted"] == 1, "model_name"].tolist()
    failed = merged.loc[
        merged["model_name"].isin(config.MODEL_NAMES) & ~successful, "model_name"
    ].tolist()
    report = [
        "No classifier was removed based on validation performance; the study's comparison includes all nine prespecified models.",
        "Models with completed validation metrics: " + ", ".join(config.display_model_label(name) for name in retained),
        "Models with missing validation metrics: " + (", ".join(config.display_model_label(name) for name in failed) if failed else "none"),
        "MajorityBaseline is reported as a reference baseline and is not one of the nine classifiers.",
    ]
    config.MODEL_ELIMINATION_REPORT_PATH.write_text("\n\n".join(report), encoding="utf-8")
    logger.info(
        "Retained all successfully evaluated prespecified classifiers: %s",
        [config.display_model_label(name) for name in retained],
    )


if __name__ == "__main__":
    main()
