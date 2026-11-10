import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


EXECUTIVE_ORDER_WARNING = (
    "The data set has been modified to comply with President Trump's executive orders. "
    "There are some missing values that may appear to be inconsistencies in the data, "
    "based on a respondent's answers to questions that were removed."
)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("2_data_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOGS_DIR / "2_data_audit.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def sample_values(series: pd.Series) -> str:
    values = series.dropna().astype(str).unique()[:3].tolist()
    return json.dumps(values)


def main() -> None:
    for path in [config.INTERIM_DIR, config.LOGS_DIR]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting read-only data audit.")
    logger.warning(EXECUTIVE_ORDER_WARNING)

    df = pd.read_csv(config.RAW_CSV_PATH, low_memory=False)
    total_rows, total_cols = df.shape
    logger.info("Loaded raw dataset with shape %s.", df.shape)

    total_nulls = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    distribution = (
        df[config.TARGET_COL]
        .value_counts(dropna=False)
        .rename_axis("diabete4_value")
        .reset_index(name="count")
    )
    distribution.to_csv(config.TARGET_DISTRIBUTION_PATH, index=False)

    present_leakage = [col for col in config.LEAKAGE_COLS if col in df.columns]
    missing_leakage = [col for col in config.LEAKAGE_COLS if col not in df.columns]
    for col in missing_leakage:
        logger.warning("Missing expected leakage column: %s", col)

    dynamic_diab_cols = sorted(
        col for col in df.columns if col.startswith("DIAB") and col not in config.LEAKAGE_COLS and col != config.TARGET_COL
    )
    if dynamic_diab_cols:
        logger.warning("Dynamic DIAB* columns outside the fixed leakage list: %s", ", ".join(dynamic_diab_cols))

    candidate_set = set(config.CANDIDATE_FEATURES)
    leakage_set = set(config.LEAKAGE_COLS) | set(dynamic_diab_cols)

    summary_rows = []
    high_missing = []
    near_constant = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_rows) * 100, 4)
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        top_ratio = float(non_null.value_counts(normalize=True).iloc[0]) if not non_null.empty else 0.0
        flags = []

        if col in candidate_set and null_pct > 40:
            flags.append("HIGH_MISSING")
            high_missing.append((col, null_pct))
        if top_ratio > 0.95:
            flags.append("NEAR_CONSTANT")
            near_constant.append((col, round(top_ratio * 100, 4)))

        summary_rows.append(
            {
                "column_name": col,
                "dtype": str(series.dtype),
                "null_count": null_count,
                "null_pct": null_pct,
                "unique_non_null": unique_count,
                "sample_values": sample_values(series),
                "is_candidate": col in candidate_set,
                "is_leakage": col in leakage_set,
                "is_target": col == config.TARGET_COL,
                "flags": ";".join(flags),
            }
        )

    column_summary = pd.DataFrame(summary_rows)
    column_summary.to_csv(config.COLUMN_SUMMARY_PATH, index=False)

    bmi_series = df["_BMI5"].dropna() if "_BMI5" in df.columns else pd.Series(dtype=float)
    bmi_scale_status = "x100 scale detected" if (not bmi_series.empty and bmi_series.quantile(0.5) > 100) else "already corrected"
    logger.info("_BMI5 scale check: %s", bmi_scale_status)

    lines = [
        "Overview",
        f"Rows: {total_rows}",
        f"Columns: {total_cols}",
        f"Total null values: {total_nulls}",
        f"Duplicate rows: {duplicates}",
        "",
        "DIABETE4 Distribution",
    ]
    lines.extend(
        f"Value {'NaN' if pd.isna(row.diabete4_value) else row.diabete4_value}: {int(row.count)}"
        for row in distribution.itertuples(index=False)
    )
    lines.extend(["", "Leakage Column Check"])
    lines.append(f"Present leakage columns: {', '.join(present_leakage) if present_leakage else 'None'}")
    lines.append(f"Missing leakage columns: {', '.join(missing_leakage) if missing_leakage else 'None'}")
    lines.append(f"Dynamic DIAB* flags: {', '.join(dynamic_diab_cols) if dynamic_diab_cols else 'None'}")
    lines.extend(["", "Candidate Feature Null Profile"])
    candidate_profile = column_summary[column_summary["is_candidate"]].sort_values("null_pct", ascending=False)
    lines.extend(
        f"{row.column_name}: {row.null_pct:.4f}% missing ({row.flags or 'OK'})"
        for row in candidate_profile.itertuples(index=False)
    )
    lines.extend(["", "High-Missing Flags"])
    if high_missing:
        lines.extend(f"{name}: {pct:.4f}% missing" for name, pct in high_missing)
    else:
        lines.append("None")
    lines.extend(["", "Near-Constant Flags"])
    if near_constant:
        lines.extend(f"{name}: top value share {pct:.4f}%" for name, pct in near_constant)
    else:
        lines.append("None")
    lines.extend(["", "Data Modification Warning", EXECUTIVE_ORDER_WARNING, "", f"_BMI5 scale audit: {bmi_scale_status}"])

    config.AUDIT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Completed data audit.")


if __name__ == "__main__":
    main()
