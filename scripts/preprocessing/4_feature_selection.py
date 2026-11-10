import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("4_feature_selection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOGS_DIR / "4_feature_selection.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main() -> None:
    for path in [config.INTERIM_DIR, config.LOGS_DIR]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting feature selection.")

    df = pd.read_parquet(config.BINARIZED_PARQUET_PATH)
    feature_log = []
    candidate_features = list(dict.fromkeys(config.CANDIDATE_FEATURES))

    dynamic_diab_cols = sorted(col for col in df.columns if col.startswith("DIAB"))
    dynamic_exclusions = [col for col in dynamic_diab_cols if col not in config.LEAKAGE_COLS]
    if dynamic_exclusions:
        logger.warning("Dynamic DIAB* exclusions detected: %s", ", ".join(dynamic_exclusions))

    exclusion_reason = {}
    for col in config.LEAKAGE_COLS + dynamic_exclusions:
        exclusion_reason[col] = "leakage"
    for col in config.DROP_COLS:
        exclusion_reason.setdefault(col, "admin_drop")
    for col in config.OPTIONAL_MODULE_FEATURES:
        exclusion_reason.setdefault(col, "module_dependent")

    existing_candidates = []
    for feature in candidate_features:
        if feature not in df.columns:
            feature_log.append({"feature": feature, "decision": "drop", "reason": "not_in_raw_data"})
            continue
        if feature in exclusion_reason:
            feature_log.append({"feature": feature, "decision": "drop", "reason": exclusion_reason[feature]})
            continue
        existing_candidates.append(feature)

    # Determine the feature missingness screen using only the training
    # respondents. The later split script uses this exact seeded split.
    candidate_frame = df[existing_candidates]
    labels = df[config.TARGET_BINARY_COL]
    trainval_frame, _, trainval_labels, _ = train_test_split(
        candidate_frame,
        labels,
        test_size=config.TEST_RATIO,
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )
    train_frame, _, _, _ = train_test_split(
        trainval_frame,
        trainval_labels,
        test_size=config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO),
        stratify=trainval_labels,
        random_state=config.RANDOM_SEED,
    )
    null_pct = train_frame.isna().mean().mul(100)
    final_features = []
    for feature in existing_candidates:
        missing_pct = float(null_pct[feature])
        if missing_pct > config.MAX_MISSING_FEATURE_PCT:
            feature_log.append({"feature": feature, "decision": "drop", "reason": "high_missing_in_training", "training_missing_pct": missing_pct})
        else:
            final_features.append(feature)
            feature_log.append({"feature": feature, "decision": "keep", "reason": "selected", "training_missing_pct": missing_pct})

    fully_null_selected = [feature for feature in final_features if float(null_pct[feature]) == 100.0]
    assert not fully_null_selected, f"Selected features are fully null: {fully_null_selected}"

    selected_df = df[final_features + [config.TARGET_BINARY_COL]].copy()
    selected_df.to_parquet(config.FEATURES_SELECTED_PATH, index=True)
    config.FINAL_FEATURE_LIST_PATH.write_text("\n".join(final_features), encoding="utf-8")
    pd.DataFrame(feature_log).sort_values(by=["decision", "feature"]).to_csv(config.FEATURE_SELECTION_LOG_PATH, index=False)

    logger.info(
        "Retained %s features using a training-only missingness screen (threshold %.1f%%).",
        len(final_features),
        config.MAX_MISSING_FEATURE_PCT,
    )
    logger.info("Saved feature-selected dataset with shape %s.", selected_df.shape)


if __name__ == "__main__":
    main()
