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
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
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
    logger = logging.getLogger("8_hyperparameter_tuning")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOGS_DIR / "8_hyperparameter_tuning.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def positive_scores(model: SkPipeline, features: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    return None


def validation_metrics(model_name: str, y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None, fit_time: float, has_probability: bool) -> dict[str, float]:
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
        "log_loss": np.nan,
        "search_time_seconds": fit_time,
    }
    if y_score is not None:
        row["roc_auc"] = roc_auc_score(y_true, y_score)
        if has_probability:
            row["log_loss"] = log_loss(y_true, np.clip(y_score, 1e-9, 1 - 1e-9), labels=[0, 1])
    return row


def build_estimators(scale_pos_weight: float) -> dict[str, object]:
    estimators = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", solver="saga", random_state=config.RANDOM_SEED),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=config.RANDOM_SEED),
        "RandomForest": RandomForestClassifier(class_weight="balanced", n_jobs=config.MAX_PARALLEL_JOBS, random_state=config.RANDOM_SEED),
        "ExtraTrees": ExtraTreesClassifier(class_weight="balanced", n_jobs=config.MAX_PARALLEL_JOBS, random_state=config.RANDOM_SEED),
        "SVM": SVC(class_weight="balanced", probability=False, random_state=config.RANDOM_SEED),
        "MLPClassifier": MLPClassifier(early_stopping=True, random_state=config.RANDOM_SEED),
    }
    if XGBClassifier is not None:
        xgb_kwargs = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
            "random_state": config.RANDOM_SEED,
            "n_jobs": config.MAX_PARALLEL_JOBS,
            "tree_method": "hist",
        }
        if config.USE_GPU:
            xgb_kwargs["device"] = "cuda"
        estimators["XGBoost"] = XGBClassifier(**xgb_kwargs)
    if LGBMClassifier is not None:
        lgbm_kwargs = {
            "class_weight": "balanced",
            "random_state": config.RANDOM_SEED,
            "verbosity": -1,
            "n_jobs": config.MAX_PARALLEL_JOBS,
        }
        if config.USE_GPU:
            lgbm_kwargs["device_type"] = "gpu"
        estimators["LightGBM"] = LGBMClassifier(**lgbm_kwargs)
    if CatBoostClassifier is not None:
        catboost_kwargs = {
            "auto_class_weights": "Balanced",
            "verbose": 0,
            "random_seed": config.RANDOM_SEED,
            "thread_count": config.MAX_PARALLEL_JOBS,
        }
        if config.USE_GPU:
            catboost_kwargs["task_type"] = "GPU"
        estimators["CatBoost"] = CatBoostClassifier(**catboost_kwargs)
    return estimators


def build_search(model_name: str, pipeline: SkPipeline, cv: StratifiedKFold, row_count: int):
    common_kwargs = {
        "cv": cv,
        "scoring": config.CV_SCORING,
        "n_jobs": config.MAX_PARALLEL_JOBS,
        "refit": True,
        "return_train_score": True,
        "verbose": 1,
        "pre_dispatch": config.MAX_PARALLEL_JOBS,
    }

    if model_name == "LogisticRegression":
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__l1_ratio": [0.0, 1.0],
                "classifier__max_iter": [500, 1000, 2000],
            },
            n_iter=5,
            random_state=config.RANDOM_SEED,
            **common_kwargs,
        )
    if model_name == "DecisionTree":
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                "classifier__max_depth": [3, 5, 7, 10, 15, None],
                "classifier__min_samples_split": [2, 5, 10, 20, 50],
                "classifier__min_samples_leaf": [1, 2, 5, 10],
                "classifier__max_features": ["sqrt", "log2", None],
            },
            n_iter=5,
            random_state=config.RANDOM_SEED,
            **common_kwargs,
        )
    if model_name in {"RandomForest", "ExtraTrees"}:
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                "classifier__n_estimators": [100, 200, 300, 500],
                "classifier__max_depth": [5, 10, 15, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__max_features": ["sqrt", "log2"],
                "classifier__bootstrap": [True, False],
            },
            n_iter=5,
            random_state=config.RANDOM_SEED,
            **common_kwargs,
        )
    if model_name == "SVM":
        n_iter = 5
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__kernel": ["rbf", "poly", "sigmoid"],
                "classifier__gamma": ["scale", "auto"],
            },
            n_iter=n_iter,
            random_state=config.RANDOM_SEED,
            **common_kwargs,
        )
    if model_name == "XGBoost":
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 5, 7, 9],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "classifier__subsample": [0.6, 0.8, 1.0],
                "classifier__colsample_bytree": [0.6, 0.8, 1.0],
                "classifier__min_child_weight": [1, 3, 5],
            },
            n_iter=5,
            random_state=config.RANDOM_SEED,
            **common_kwargs,
        )
    if model_name == "LightGBM":
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                "classifier__n_estimators": [100, 200, 300, 500],
                "classifier__max_depth": [-1, 5, 10, 15],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "classifier__num_leaves": [31, 63, 127],
                "classifier__subsample": [0.6, 0.8, 1.0],
                "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            },
            n_iter=5,
            random_state=config.RANDOM_SEED,
            **common_kwargs,
        )
    if model_name == "CatBoost":
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                "classifier__iterations": [100, 200, 300],
                "classifier__depth": [4, 6, 8, 10],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "classifier__l2_leaf_reg": [1, 3, 5, 10],
            },
            n_iter=5,
            random_state=config.RANDOM_SEED,
            **common_kwargs,
        )
    if model_name == "MLPClassifier":
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                "classifier__hidden_layer_sizes": [(50,), (100,), (100, 50), (200, 100), (100, 50, 25)],
                "classifier__activation": ["relu", "tanh"],
                "classifier__alpha": [0.0001, 0.001, 0.01],
                "classifier__learning_rate": ["constant", "adaptive"],
                "classifier__max_iter": [200, 500],
            },
            n_iter=5,
            random_state=config.RANDOM_SEED,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    for path in [config.LOGS_DIR, config.METRICS_DIR, config.MODELS_DIR]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting hyperparameter tuning.")
    logger.info("CV scoring metric: %s", config.CV_SCORING)
    logger.info("Skip SVM tuning: %s", config.SKIP_SVM_TUNING)
    logger.info("Search budget: 5 sampled configurations per model x %s stratified folds.", config.CV_FOLDS)

    run_started = float(config.RERUN_STARTED_PATH.read_text(encoding="utf-8").strip())

    X_train = pd.read_parquet(config.X_TRAIN_PATH)
    y_train = pd.read_parquet(config.Y_TRAIN_PATH)[config.TARGET_BINARY_COL]
    X_val = pd.read_parquet(config.X_VAL_PATH)
    y_val = pd.read_parquet(config.Y_VAL_PATH)[config.TARGET_BINARY_COL]
    baseline_metrics = pd.read_csv(config.BASELINE_METRICS_PATH)

    scale_pipeline = joblib.load(config.PIPELINE_SCALE_PATH)
    tree_pipeline = joblib.load(config.PIPELINE_TREE_PATH)

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / positive_count

    estimators = build_estimators(scale_pos_weight)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    results = []
    for model_name in config.MODEL_NAMES:
        logger.info("Tuning model: %s", model_name)
        estimator = estimators.get(model_name)
        if estimator is None:
            results.append({"model_name": model_name, "error": "Estimator unavailable in environment."})
            continue
        baseline_row = baseline_metrics.loc[baseline_metrics["model_name"] == model_name]
        baseline_path = config.MODELS_DIR / f"baseline_{model_name}.joblib"
        baseline_error = ""
        if not baseline_row.empty:
            raw_error = baseline_row.iloc[0].get("error", "")
            baseline_error = "" if pd.isna(raw_error) else str(raw_error).strip()
        if (
            baseline_row.empty
            or baseline_error
            or not baseline_path.exists()
            or baseline_path.stat().st_mtime <= run_started
        ):
            results.append({"model_name": model_name, "error": "Current-run baseline artifact or metrics unavailable; stale artifacts were not used."})
            logger.error("Skipping tuning for %s because no valid current-run baseline exists.", model_name)
            continue
        if model_name == "SVM" and config.SKIP_SVM_TUNING:
            logger.warning("Skipping SVM tuning by configuration to avoid excessive runtime on large datasets.")
            baseline_path = config.MODELS_DIR / "baseline_SVM.joblib"
            if baseline_path.exists():
                joblib.dump(joblib.load(baseline_path), config.MODELS_DIR / "tuned_SVM.joblib")
            baseline_row = baseline_metrics.loc[baseline_metrics["model_name"] == model_name]
            if baseline_row.empty:
                results.append(
                    {
                        "model_name": model_name,
                        "accuracy": np.nan,
                        "precision_weighted": np.nan,
                        "recall_weighted": np.nan,
                        "f1_weighted": np.nan,
                        "f1_macro": np.nan,
                        "roc_auc": np.nan,
                        "search_time_seconds": np.nan,
                        "selected_artifact": "skipped_to_baseline",
                        "best_params": "{}",
                        "error": "SVM tuning skipped and baseline metrics unavailable.",
                    }
                )
            else:
                record = baseline_row.iloc[0].to_dict()
                results.append(
                    {
                        "model_name": model_name,
                        "accuracy": record.get("accuracy", np.nan),
                        "precision_weighted": record.get("precision_weighted", np.nan),
                        "recall_weighted": record.get("recall_weighted", np.nan),
                        "f1_weighted": record.get("f1_weighted", np.nan),
                        "f1_macro": record.get("f1_macro", np.nan),
                        "roc_auc": record.get("roc_auc", np.nan),
                        "search_time_seconds": 0.0,
                        "selected_artifact": "skipped_to_baseline",
                        "best_params": "{}",
                        "error": "",
                    }
                )
            continue

        base_pipeline = scale_pipeline if model_name in config.SCALE_SENSITIVE_MODELS else tree_pipeline
        full_pipeline = SkPipeline(
            steps=[
                ("preprocessor", clone(base_pipeline.named_steps["preprocessor"])),
                ("classifier", estimator),
            ]
        )

        try:
            search = build_search(model_name, full_pipeline, cv, len(X_train))
            fit_X = X_train
            fit_y = y_train
            if model_name == "SVM":
                if len(X_train) > 200000:
                    logger.warning("Large training set detected; SVM tuning n_iter reduced to 5 as a runtime safeguard.")
                if len(X_train) > config.SVM_TUNING_MAX_ROWS:
                    logger.warning(
                        "Subsampling SVM tuning from %s rows to %s rows to avoid quadratic-to-cubic runtime.",
                        len(X_train),
                        config.SVM_TUNING_MAX_ROWS,
                    )
                    fit_X, _, fit_y, _ = train_test_split(
                        X_train,
                        y_train,
                        train_size=config.SVM_TUNING_MAX_ROWS,
                        stratify=y_train,
                        random_state=config.RANDOM_SEED,
                    )

            start = time.time()
            fit_kwargs = {}
            if model_name == "MLPClassifier":
                fit_kwargs["classifier__sample_weight"] = compute_sample_weight("balanced", fit_y)
            search.fit(fit_X, fit_y, **fit_kwargs)
            fit_time = time.time() - start

            best_pipeline = search.best_estimator_
            y_pred = best_pipeline.predict(X_val)
            y_score = positive_scores(best_pipeline, X_val)
            current_metrics = validation_metrics(
                model_name,
                y_val,
                y_pred,
                y_score,
                fit_time,
                has_probability=hasattr(best_pipeline, "predict_proba"),
            )
            current_metrics["training_rows_fitted"] = len(fit_X)
            current_metrics["search_training_rows"] = len(fit_X)

            baseline_row = baseline_metrics.loc[baseline_metrics["model_name"] == model_name]
            baseline_f1 = float(baseline_row["f1_macro"].iloc[0]) if not baseline_row.empty else np.nan

            if pd.notna(baseline_f1) and current_metrics["f1_macro"] < baseline_f1:
                logger.warning("Tuned %s underperformed baseline; reverting to baseline artifact.", model_name)
                chosen_pipeline = joblib.load(config.MODELS_DIR / f"baseline_{model_name}.joblib")
                selected_metrics = baseline_row.iloc[0].to_dict()
                selected_metrics.update(
                    {
                        "search_time_seconds": fit_time,
                        "search_training_rows": len(fit_X),
                        "selected_artifact": "baseline",
                        "best_params": "{}",
                        "error": "",
                    }
                )
                results.append(selected_metrics)
            else:
                chosen_pipeline = best_pipeline
                results.append(
                    {
                        **current_metrics,
                        "selected_artifact": "tuned",
                        "best_params": str(search.best_params_),
                        "error": "",
                    }
                )

            joblib.dump(chosen_pipeline, config.MODELS_DIR / f"tuned_{model_name}.joblib")
            pd.DataFrame(search.cv_results_).to_csv(config.METRICS_DIR / f"tuning_cv_results_{model_name}.csv", index=False)
        except Exception as exc:
            logger.exception("Hyperparameter tuning failed for %s", model_name)
            results.append(
                {
                    "model_name": model_name,
                    "accuracy": np.nan,
                    "precision_weighted": np.nan,
                    "recall_weighted": np.nan,
                    "f1_weighted": np.nan,
                    "f1_macro": np.nan,
                    "roc_auc": np.nan,
                        "search_time_seconds": np.nan,
                    "selected_artifact": "failed",
                    "best_params": "{}",
                    "error": str(exc),
                }
            )

    results_df = pd.DataFrame(results)
    results_df["display_label"] = results_df["model_name"].map(config.display_model_label)
    results_df.sort_values(by="f1_macro", ascending=False, na_position="last").to_csv(
        config.TUNED_METRICS_PATH,
        index=False,
    )
    logger.info("Saved tuning metrics.")


if __name__ == "__main__":
    main()
