"""
Metrics Utilities
Common metrics and metric selection.
"""

from typing import Dict, Any, Callable
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss,
)
import numpy as np


CLASSIFICATION_METRICS = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted", zero_division=0),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted", zero_division=0),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted", zero_division=0),
}

REGRESSION_METRICS = {
    "mse": mean_squared_error,
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "mae": mean_absolute_error,
    "r2": r2_score,
}


def get_metric_function(metric_name: str, task_type: str) -> Callable:
    """
    Get metric function by name.

    Args:
        metric_name: Name of the metric
        task_type: "classification" or "regression"

    Returns:
        Metric function

    Raises:
        ValueError: If metric not found
    """
    if task_type == "classification":
        if metric_name in CLASSIFICATION_METRICS:
            return CLASSIFICATION_METRICS[metric_name]
    elif task_type == "regression":
        if metric_name in REGRESSION_METRICS:
            return REGRESSION_METRICS[metric_name]

    raise ValueError(f"Metric '{metric_name}' not found for task type '{task_type}'")


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
) -> Dict[str, float]:
    """
    Calculate all available metrics for the task type.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        task_type: "classification" or "regression"

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    if task_type == "classification":
        for metric_name, metric_func in CLASSIFICATION_METRICS.items():
            try:
                metrics[metric_name] = metric_func(y_true, y_pred)
            except Exception as e:
                metrics[metric_name] = None

    elif task_type == "regression":
        for metric_name, metric_func in REGRESSION_METRICS.items():
            try:
                metrics[metric_name] = metric_func(y_true, y_pred)
            except Exception as e:
                metrics[metric_name] = None

    return metrics


def get_sklearn_scoring(metric_name: str) -> str:
    """
    Convert metric name to sklearn scoring string.

    Args:
        metric_name: Metric name

    Returns:
        Sklearn scoring string
    """
    sklearn_scoring_map = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc",
        "mse": "neg_mean_squared_error",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    return sklearn_scoring_map.get(metric_name, metric_name)
