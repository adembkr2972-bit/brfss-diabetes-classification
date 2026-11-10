import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("6_split_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOGS_DIR / "6_split_data.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main() -> None:
    for path in [config.PROCESSED_DIR, config.LOGS_DIR]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting stratified 70/15/15 split.")

    df = pd.read_parquet(config.FEATURES_SELECTED_PATH)
    feature_frame = df.drop(columns=[config.TARGET_BINARY_COL])
    label_series = df[config.TARGET_BINARY_COL]

    trainval_frame, holdout_frame, trainval_labels, holdout_labels = train_test_split(
        feature_frame,
        label_series,
        test_size=config.TEST_RATIO,
        stratify=label_series,
        random_state=config.RANDOM_SEED,
    )

    val_ratio_within_trainval = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_frame, validation_frame, train_labels, validation_labels = train_test_split(
        trainval_frame,
        trainval_labels,
        test_size=val_ratio_within_trainval,
        stratify=trainval_labels,
        random_state=config.RANDOM_SEED,
    )

    positive_rates = {
        "train": float(train_labels.mean()),
        "validation": float(validation_labels.mean()),
        "holdout": float(holdout_labels.mean()),
    }
    assert max(positive_rates.values()) - min(positive_rates.values()) <= 0.01

    train_index = set(train_frame.index)
    validation_index = set(validation_frame.index)
    holdout_index = set(holdout_frame.index)
    assert train_index.isdisjoint(validation_index)
    assert train_index.isdisjoint(holdout_index)
    assert validation_index.isdisjoint(holdout_index)
    assert train_index | validation_index | holdout_index == set(df.index)

    train_frame.to_parquet(config.X_TRAIN_PATH, index=True)
    validation_frame.to_parquet(config.X_VAL_PATH, index=True)
    holdout_frame.to_parquet(config.X_TEST_PATH, index=True)
    train_labels.to_frame(config.TARGET_BINARY_COL).to_parquet(config.Y_TRAIN_PATH, index=True)
    validation_labels.to_frame(config.TARGET_BINARY_COL).to_parquet(config.Y_VAL_PATH, index=True)
    holdout_labels.to_frame(config.TARGET_BINARY_COL).to_parquet(config.Y_TEST_PATH, index=True)

    schema_df = pd.DataFrame(
        {
            "column_name": train_frame.columns.tolist(),
            "dtype": [str(dtype) for dtype in train_frame.dtypes.tolist()],
        }
    )
    schema_df.to_csv(config.TRAIN_SCHEMA_PATH, index=False)

    np.savez(
        config.SPLIT_INDICES_PATH,
        train_indices=np.array(sorted(train_index)),
        val_indices=np.array(sorted(validation_index)),
        test_indices=np.array(sorted(holdout_index)),
    )

    split_summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(train_frame), "positive_rate": positive_rates["train"]},
            {"split": "validation", "rows": len(validation_frame), "positive_rate": positive_rates["validation"]},
            {"split": "holdout", "rows": len(holdout_frame), "positive_rate": positive_rates["holdout"]},
        ]
    )
    split_summary.to_csv(config.SPLIT_SUMMARY_PATH, index=False)
    logger.info("Saved processed splits and index manifests.")


if __name__ == "__main__":
    main()
