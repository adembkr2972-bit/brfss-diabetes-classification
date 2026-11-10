import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("3_target_construction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOGS_DIR / "3_target_construction.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def recode_special_values(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    recoded = df.copy()
    for column, value_map in config.SPECIAL_RECODE_MAPS.items():
        if column not in recoded.columns:
            continue
        cleaned_map = {key: (np.nan if value is None else value) for key, value in value_map.items()}
        recoded[column] = recoded[column].replace(cleaned_map)

    for column in config.GENERAL_RECODE_7_9_COLUMNS:
        if column in recoded.columns and pd.api.types.is_numeric_dtype(recoded[column]):
            recoded[column] = recoded[column].replace({7: np.nan, 9: np.nan})

    for column in config.GENERAL_RECODE_9_ONLY_COLUMNS:
        if column in recoded.columns and pd.api.types.is_numeric_dtype(recoded[column]):
            recoded[column] = recoded[column].replace({9: np.nan})

    # ALCDAY4 combines days per week (101-107) and days per month (201-230).
    # Convert both response formats to mean drinking days per week so that
    # numeric models do not interpret the 100/200 code prefixes as magnitude.
    if "ALCDAY4" in recoded.columns:
        alcohol_days = pd.to_numeric(recoded["ALCDAY4"], errors="coerce")
        weekly = alcohol_days.between(101, 107)
        monthly = alcohol_days.between(201, 230)
        normalized = pd.Series(np.nan, index=recoded.index, dtype=float)
        normalized.loc[weekly] = alcohol_days.loc[weekly] - 100
        normalized.loc[monthly] = (alcohol_days.loc[monthly] - 200) * (7.0 / 30.0)
        normalized.loc[alcohol_days.eq(0)] = 0.0
        recoded["ALCDAY4"] = normalized

    bmi_series = recoded["_BMI5"].dropna() if "_BMI5" in recoded.columns else pd.Series(dtype=float)
    if "_BMI5" in recoded.columns and not bmi_series.empty and bmi_series.quantile(0.5) > 100:
        recoded["_BMI5"] = recoded["_BMI5"] / config.BMI5_SCALE_FACTOR
        logger.info("Applied _BMI5 scale correction.")
    else:
        logger.info("Skipped _BMI5 scale correction.")

    return recoded


def main() -> None:
    for path in [config.INTERIM_DIR, config.LOGS_DIR]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting target construction.")

    raw_df = pd.read_csv(config.RAW_CSV_PATH, low_memory=False)
    raw_rows = len(raw_df)
    value_counts = raw_df[config.TARGET_COL].value_counts(dropna=False).sort_index()

    excluded_rows = pd.DataFrame(
        {
            "diabete4_value": [("NaN" if pd.isna(idx) else idx) for idx in value_counts.index],
            "count": value_counts.values,
            "kept": [idx in config.TARGET_KEEP_VALUES for idx in value_counts.index],
        }
    )
    excluded_rows.to_csv(config.EXCLUDED_ROWS_PATH, index=False)

    df = recode_special_values(raw_df, logger)
    inf_count = int(np.isinf(df.select_dtypes(include=["number"]).to_numpy()).sum())
    assert inf_count == 0, f"Unexpected infinite values detected after recoding: {inf_count}"

    candidate_columns = [col for col in dict.fromkeys(config.CANDIDATE_FEATURES) if col in df.columns]
    candidate_null_pct = df[candidate_columns].isna().mean().mul(100).sort_values(ascending=False)
    logger.info("Post-recode candidate missingness summary: %s", candidate_null_pct.round(4).to_dict())
    fully_null_candidates = candidate_null_pct[candidate_null_pct == 100].index.tolist()
    assert not fully_null_candidates, f"Candidate columns became fully null after recoding: {fully_null_candidates}"

    keep_mask = df[config.TARGET_COL].isin(config.TARGET_KEEP_VALUES)
    filtered_df = df.loc[keep_mask].copy()

    dropped_counts = {
        "value_2": int((raw_df[config.TARGET_COL] == 2).sum()),
        "value_4": int((raw_df[config.TARGET_COL] == 4).sum()),
        "value_7": int((raw_df[config.TARGET_COL] == 7).sum()),
        "value_9": int((raw_df[config.TARGET_COL] == 9).sum()),
        "missing": int(raw_df[config.TARGET_COL].isna().sum()),
    }
    for label, count in dropped_counts.items():
        logger.info("Dropped %s rows: %s", label, count)

    filtered_df[config.TARGET_BINARY_COL] = filtered_df[config.TARGET_COL].map(config.TARGET_KEEP_VALUES).astype(int)
    assert set(filtered_df[config.TARGET_BINARY_COL].unique()) == {0, 1}
    assert int(filtered_df[config.TARGET_BINARY_COL].isna().sum()) == 0

    class_counts = filtered_df[config.TARGET_BINARY_COL].value_counts().sort_index()
    imbalance_ratio = class_counts.max() / class_counts.min()
    logger.info("Target counts: %s", class_counts.to_dict())
    logger.info("Class imbalance ratio: %.4f", imbalance_ratio)
    if imbalance_ratio > 4:
        logger.warning("Class imbalance ratio exceeds 4:1.")

    filtered_df = filtered_df.drop(columns=[config.TARGET_COL])

    accounted_rows = int(keep_mask.sum()) + sum(dropped_counts.values())
    assert accounted_rows == raw_rows, f"Row accounting mismatch: {accounted_rows} vs {raw_rows}"

    filtered_df.to_parquet(config.BINARIZED_PARQUET_PATH, index=True)
    logger.info("Saved binarized dataset with shape %s.", filtered_df.shape)


if __name__ == "__main__":
    main()
