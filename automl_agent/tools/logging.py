"""
Logging Utilities
MLflow integration and general logging.
"""

import logging
import json
from typing import Dict, Any
from pathlib import Path
import mlflow


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with the given name and level.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level))

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def log_config(config: Dict[str, Any], output_dir: str = "output"):
    """
    Log configuration to file.

    Args:
        config: Configuration dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_file = output_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def log_metrics(metrics: Dict[str, Any], output_dir: str = "output"):
    """
    Log metrics to file.

    Args:
        metrics: Metrics dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)


def log_run_summary(summary: Dict[str, Any], output_dir: str = "output"):
    """
    Log run summary to file.

    Args:
        summary: Summary dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_file = output_path / "run_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


class MLflowLogger:
    """MLflow logging wrapper."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MLflow logger.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get("logging", {}).get("use_mlflow", False)

        if self.enabled:
            tracking_uri = config.get("logging", {}).get("mlflow_tracking_uri", "./mlruns")
            experiment_name = config.get("logging", {}).get("experiment_name", "automl_experiment")

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

    def log_run(self, run_name: str, params: Dict, metrics: Dict, model: Any = None):
        """
        Log a run to MLflow.

        Args:
            run_name: Name of the run
            params: Parameters to log
            metrics: Metrics to log
            model: Model to log (optional)
        """
        if not self.enabled:
            return

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            if model is not None:
                mlflow.sklearn.log_model(model, "model")
