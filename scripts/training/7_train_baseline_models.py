import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("7_train_baseline_models")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOGS_DIR / "7_train_baseline_models.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_estimators(scale_pos_weight: float, logger: logging.Logger) -> dict[str, object]:
    estimators = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            solver="saga",
            max_iter=1000,
            random_state=config.RANDOM_SEED,
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            n_jobs=config.MAX_PARALLEL_JOBS,
            random_state=config.RANDOM_SEED,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            class_weight="balanced",
            n_estimators=200,
            n_jobs=config.MAX_PARALLEL_JOBS,
            random_state=config.RANDOM_SEED,
        ),
        "SVM": SVC(
            class_weight="balanced",
            probability=True,
            random_state=config.RANDOM_SEED,
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            early_stopping=True,
            max_iter=500,
            random_state=config.RANDOM_SEED,
        ),
    }

    if XGBClassifier is not None:
        xgb_kwargs = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "random_state": config.RANDOM_SEED,
            "n_jobs": config.MAX_PARALLEL_JOBS,
            "tree_method": "hist",
        }
        if config.USE_GPU:
            xgb_kwargs["device"] = "cuda"
            logger.info("XGBoost GPU mode enabled.")
        estimators["XGBoost"] = XGBClassifier(**xgb_kwargs)
    else:
        logger.warning("xgboost is unavailable; XGBoost baseline will be skipped.")

    if LGBMClassifier is not None:
        lgbm_kwargs = {
            "class_weight": "balanced",
            "n_estimators": 200,
            "learning_rate": 0.1,
            "random_state": config.RANDOM_SEED,
            "verbosity": -1,
            "n_jobs": config.MAX_PARALLEL_JOBS,
        }
        if config.USE_GPU:
            lgbm_kwargs["device_type"] = "gpu"
            logger.info("LightGBM GPU mode enabled.")
        estimators["LightGBM"] = LGBMClassifier(**lgbm_kwargs)
    else:
        logger.warning("lightgbm is unavailable; LightGBM baseline will be skipped.")

    if CatBoostClassifier is not None:
        catboost_kwargs = {
            "auto_class_weights": "Balanced",
            "verbose": 0,
            "random_seed": config.RANDOM_SEED,
            "thread_count": config.MAX_PARALLEL_JOBS,
        }
        if config.USE_GPU:
            catboost_kwargs["task_type"] = "GPU"
            logger.info("CatBoost GPU mode enabled.")
        estimators["CatBoost"] = CatBoostClassifier(**catboost_kwargs)
    else:
        logger.warning("catboost is unavailable; CatBoost baseline will be skipped.")

    return estimators


def positive_scores(model: SkPipeline, features: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(features)
        raw_scores = np.asarray(raw_scores, dtype=float)
        if raw_scores.max() == raw_scores.min():
            return np.full(raw_scores.shape, 0.5)
        return (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    return None


def maybe_subsample_for_svm(features: pd.DataFrame, labels: pd.Series, logger: logging.Logger) -> tuple[pd.DataFrame, pd.Series]:
    if len(features) <= config.SVM_BASELINE_MAX_ROWS:
        return features, labels
    logger.warning(
        "Subsampling SVM baseline training from %s rows to %s rows to avoid quadratic-to-cubic runtime.",
        len(features),
        config.SVM_BASELINE_MAX_ROWS,
    )
    sampled_X, _, sampled_y, _ = train_test_split(
        features,
        labels,
        train_size=config.SVM_BASELINE_MAX_ROWS,
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )
    return sampled_X, sampled_y


def metric_row(model_name: str, y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None, fit_time: float) -> dict[str, float]:
    row = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": np.nan,
        "fit_time_seconds": fit_time,
    }
    if y_score is not None:
        row["roc_auc"] = roc_auc_score(y_true, y_score)
    return row


def failure_row(model_name: str, message: str) -> dict[str, float]:
    return {
        "model_name": model_name,
        "accuracy": np.nan,
        "precision_weighted": np.nan,
        "recall_weighted": np.nan,
        "f1_weighted": np.nan,
        "f1_macro": np.nan,
        "roc_auc": np.nan,
        "fit_time_seconds": np.nan,
        "error": message,
    }


def main() -> None:
    for path in [config.MODELS_DIR, config.LOGS_DIR, config.METRICS_DIR]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting baseline model training.")
    config.RERUN_STARTED_PATH.write_text(str(time.time()), encoding="utf-8")
    logger.info("Recorded current-run marker at %s.", config.RERUN_STARTED_PATH)

    X_train = pd.read_parquet(config.X_TRAIN_PATH)
    y_train = pd.read_parquet(config.Y_TRAIN_PATH)[config.TARGET_BINARY_COL]
    X_val = pd.read_parquet(config.X_VAL_PATH)
    y_val = pd.read_parquet(config.Y_VAL_PATH)[config.TARGET_BINARY_COL]
    expected_schema = pd.read_csv(config.TRAIN_SCHEMA_PATH)

    assert X_val.columns.tolist() == X_train.columns.tolist(), "Validation columns do not match training columns."
    assert [str(dtype) for dtype in X_train.dtypes.tolist()] == expected_schema["dtype"].tolist(), "Training dtypes do not match saved schema."
    assert [str(dtype) for dtype in X_val.dtypes.tolist()] == expected_schema["dtype"].tolist(), "Validation dtypes do not match training schema."

    scale_pipeline = joblib.load(config.PIPELINE_SCALE_PATH)
    tree_pipeline = joblib.load(config.PIPELINE_TREE_PATH)

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / positive_count
    logger.info("Computed XGBoost scale_pos_weight=%.6f", scale_pos_weight)

    estimators = get_estimators(scale_pos_weight, logger)
    results = []
    failures = 0

    for model_name in config.MODEL_NAMES:
        logger.info("Training baseline model: %s", model_name)
        estimator = estimators.get(model_name)
        if estimator is None:
            results.append(failure_row(model_name, "Estimator unavailable in environment."))
            failures += 1
            continue
        fit_X = X_train
        fit_y = y_train
        if model_name == "SVM":
            fit_X, fit_y = maybe_subsample_for_svm(X_train, y_train, logger)

        base_pipeline = scale_pipeline if model_name in config.SCALE_SENSITIVE_MODELS else tree_pipeline
        full_pipeline = SkPipeline(
            steps=[
                ("preprocessor", clone(base_pipeline.named_steps["preprocessor"])),
                ("classifier", estimator),
            ]
        )

        try:
            start = time.time()
            fit_kwargs = {}
            if model_name == "MLPClassifier":
                fit_kwargs["classifier__sample_weight"] = compute_sample_weight("balanced", fit_y)
            full_pipeline.fit(fit_X, fit_y, **fit_kwargs)
            fit_time = time.time() - start

            y_pred = full_pipeline.predict(X_val)
            y_score = positive_scores(full_pipeline, X_val)
            result = metric_row(model_name, y_val, y_pred, y_score, fit_time)
            result["training_rows_fitted"] = len(fit_X)
            result["error"] = ""
            results.append(result)

            joblib.dump(full_pipeline, config.MODELS_DIR / f"baseline_{model_name}.joblib")
        except Exception as exc:
            logger.exception("Baseline training failed for %s", model_name)
            results.append(failure_row(model_name, str(exc)))
            failures += 1

    metrics_df = pd.DataFrame(results)
    metrics_df["display_label"] = metrics_df["model_name"].map(config.display_model_label)
    metrics_df = metrics_df.sort_values(by="f1_macro", ascending=False, na_position="last")
    metrics_df.to_csv(config.BASELINE_METRICS_PATH, index=False)

    top_models = metrics_df.head(3)[["model_name", "f1_macro"]].to_dict(orient="records")
    logger.info("Top baseline models: %s", top_models)
    logger.info("Baseline failures: %s", failures)
    if failures > 2:
        logger.error("More than two baseline models failed; exiting early to avoid cascading invalid downstream runs.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
