import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("5_build_pipelines")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOGS_DIR / "5_build_pipelines.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def classify_features(df: pd.DataFrame, features: list[str]) -> dict[str, list[str]]:
    continuous = [col for col in features if col in config.CONTINUOUS_FEATURES]
    ordinal = [col for col in features if col in config.ORDINAL_FEATURES and col not in continuous]

    assigned = set(continuous) | set(ordinal)
    remaining = [col for col in features if col not in assigned]

    # Assign types from documented variable semantics rather than observed
    # category counts in the full dataset. Remaining variables are nominal.
    binary: list[str] = []
    nominal_low = remaining
    nominal_high: list[str] = []

    return {
        "continuous": continuous,
        "ordinal": ordinal,
        "binary": binary,
        "nominal_low": nominal_low,
        "nominal_high": nominal_high,
    }


def build_transformer(groups: dict[str, list[str]], scale_sensitive: bool) -> ColumnTransformer:
    ordinal_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
    if scale_sensitive:
        ordinal_steps.append(("scaler", StandardScaler()))

    transformers = [
        (
            "missing_indicators",
            MissingIndicator(features="all", error_on_new=False),
            groups["continuous"] + groups["ordinal"] + groups["binary"] + groups["nominal_low"] + groups["nominal_high"],
        ),
        (
            "continuous",
            SkPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    *([("scaler", StandardScaler())] if scale_sensitive else []),
                ]
            ),
            groups["continuous"],
        ),
        ("ordinal", SkPipeline(steps=ordinal_steps), groups["ordinal"]),
        (
            "binary",
            SkPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    *([("scaler", StandardScaler())] if scale_sensitive else []),
                ]
            ),
            groups["binary"],
        ),
        (
            "nominal_low",
            SkPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
                ]
            ),
            groups["nominal_low"],
        ),
        (
            "nominal_high",
            SkPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
                ]
            ),
            groups["nominal_high"],
        ),
    ]

    return ColumnTransformer(
        transformers=[item for item in transformers if item[2]],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def main() -> None:
    for path in [config.INTERIM_DIR, config.LOGS_DIR]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting preprocessing pipeline construction.")

    df = pd.read_parquet(config.FEATURES_SELECTED_PATH)
    features = [line.strip() for line in config.FINAL_FEATURE_LIST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = classify_features(df, features)

    assigned = [col for cols in groups.values() for col in cols]
    assert sorted(assigned) == sorted(features), "Every feature must be assigned."
    assert len(assigned) == len(set(assigned)), "A feature was assigned more than once."

    scale_pipeline = SkPipeline(steps=[("preprocessor", build_transformer(groups, scale_sensitive=True))])
    tree_pipeline = SkPipeline(steps=[("preprocessor", build_transformer(groups, scale_sensitive=False))])

    joblib.dump(scale_pipeline, config.PIPELINE_SCALE_PATH)
    joblib.dump(tree_pipeline, config.PIPELINE_TREE_PATH)

    assignment_rows = [
        {"feature": feature, "feature_type": feature_type}
        for feature_type, feature_list in groups.items()
        for feature in feature_list
    ]
    pd.DataFrame(assignment_rows).sort_values(by=["feature_type", "feature"]).to_csv(config.COLUMN_TYPE_PATH, index=False)
    config.NUMERIC_COLS_PATH.write_text("\n".join(groups["continuous"]), encoding="utf-8")
    categorical = groups["ordinal"] + groups["binary"] + groups["nominal_low"] + groups["nominal_high"]
    config.CATEGORICAL_COLS_PATH.write_text("\n".join(categorical), encoding="utf-8")

    logger.info("Saved unfitted scale-sensitive and tree-based preprocessing pipelines.")


if __name__ == "__main__":
    main()
