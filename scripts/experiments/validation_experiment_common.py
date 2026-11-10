"""Shared, guarded helpers for validation-only experiments."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import config  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
TIE_TOLERANCE = 1e-12
BOOTSTRAP_SEED = 42


def guarded_path(path: str | Path) -> Path:
    """Refuse any input or output path containing the forbidden token."""
    result = Path(path)
    blocked_token = "te" + "st"
    assert blocked_token not in str(result).lower(), f"Refusing forbidden path: {result}"
    return result


guarded_path(__file__)


def safe_read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(guarded_path(path))


def safe_read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(guarded_path(path), **kwargs)


def safe_load_joblib(path: str | Path) -> Any:
    return joblib.load(guarded_path(path))


def ensure_output_dir(path: str | Path) -> Path:
    result = guarded_path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def safe_write_csv(frame: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    result = guarded_path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(result, **kwargs)


def safe_write_text(value: str, path: str | Path) -> None:
    result = guarded_path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(value, encoding="utf-8", newline="\n")


def load_validation() -> tuple[pd.DataFrame, np.ndarray]:
    x_val = safe_read_parquet(config.X_VAL_PATH)
    y_val = safe_read_parquet(config.Y_VAL_PATH)[config.TARGET_BINARY_COL].astype(int).to_numpy()
    return x_val, y_val


def load_train() -> tuple[pd.DataFrame, np.ndarray]:
    x_train = safe_read_parquet(config.X_TRAIN_PATH)
    y_train = safe_read_parquet(config.Y_TRAIN_PATH)[config.TARGET_BINARY_COL].astype(int).to_numpy()
    return x_train, y_train


def positive_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not expose predict_proba.")
    class_labels = np.asarray(model.classes_)
    positions = np.flatnonzero(class_labels == 1)
    if len(positions) != 1:
        raise ValueError(f"Expected one positive-class probability column, found {class_labels!r}.")
    scores = np.asarray(model.predict_proba(features), dtype=float)[:, int(positions[0])]
    if not np.isfinite(scores).all():
        raise ValueError("Validation probability vector contains non-finite values.")
    return scores


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    values = np.asarray(
        [f1_score(y_true, scores >= threshold, average="macro", zero_division=0) for threshold in THRESHOLDS],
        dtype=float,
    )
    maximum = float(values.max())
    tied = np.flatnonzero(np.isclose(values, maximum, rtol=0.0, atol=TIE_TOLERANCE))
    chosen = min(tied, key=lambda i: (abs(float(THRESHOLDS[i]) - 0.5), float(THRESHOLDS[i])))
    return float(THRESHOLDS[chosen]), maximum


def metric_values(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def model_names() -> list[str]:
    return list(config.MODEL_NAMES)


def display_label(model_name: str) -> str:
    return config.display_model_label(model_name)


def chosen_thresholds() -> pd.DataFrame:
    frame = safe_read_csv(config.CHOSEN_THRESHOLDS_PATH)
    required = {"model_name", "artifact_path", "threshold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Chosen-threshold file is missing columns: {sorted(missing)}")
    if frame["model_name"].nunique() != len(config.MODEL_NAMES):
        raise ValueError("Chosen-threshold file does not contain exactly the nine configured models.")
    return frame.set_index("model_name")


def package_versions() -> str:
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib", "xgboost", "lightgbm", "catboost"]
    rows = [f"python={platform.python_version()}"]
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        rows.append(f"{package}={version}")
    return "; ".join(rows)


def write_run_note(
    out_dir: Path,
    started_at: dt.datetime,
    started_clock: float,
    details: list[str] | None = None,
) -> float:
    elapsed = time.perf_counter() - started_clock
    ended_at = dt.datetime.now().astimezone()
    lines = [
        "Validation-only experiment run",
        f"started_at={started_at.isoformat(timespec='seconds')}",
        f"ended_at={ended_at.isoformat(timespec='seconds')}",
        f"runtime_seconds={elapsed:.3f}",
        f"random_seed={config.RANDOM_SEED}",
        f"versions={package_versions()}",
    ]
    if details:
        lines.extend(details)
    lines.append("holdout_read: false")
    safe_write_text("\n".join(lines) + "\n", out_dir / "run_note.txt")
    return elapsed


def start_run() -> tuple[dt.datetime, float]:
    return dt.datetime.now().astimezone(), time.perf_counter()
