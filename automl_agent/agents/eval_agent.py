"""
Evaluation Agent
Función:
- Evaluar y comparar modelos
- Detectar overfitting

Tools:
- sklearn.metrics
- numpy
- mlflow
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import mlflow


class EvaluationAgent:
    """Agent responsible for model evaluation and comparison."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Evaluation Agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_config = config.get("metrics", {})
        self.primary_metric = self.metrics_config.get("primary", "accuracy")
        self.secondary_metrics = self.metrics_config.get("secondary", [])
        self.task_type = self.metrics_config.get("task_type", "classification")

        # Initialize MLflow if enabled
        if config.get("logging", {}).get("use_mlflow", False):
            mlflow.set_tracking_uri(config["logging"].get("mlflow_tracking_uri", "./mlruns"))
            mlflow.set_experiment(config["logging"].get("experiment_name", "automl_experiment"))

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for classification)

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        if self.task_type == "classification":
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])

        elif self.task_type == "regression":
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)

        return metrics

    def evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model

        Returns:
            Evaluation results dictionary
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilities for classification
        y_train_proba = None
        y_test_proba = None
        if self.task_type == "classification" and hasattr(model, "predict_proba"):
            y_train_proba = model.predict_proba(X_train)
            y_test_proba = model.predict_proba(X_test)

        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred, y_train_proba)
        test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_proba)

        # Detect overfitting
        overfit_score = self._detect_overfitting(train_metrics, test_metrics)

        results = {
            "model_name": model_name,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "overfit_score": overfit_score,
            "is_overfitting": overfit_score > 0.1,  # 10% threshold
        }

        # Log to MLflow if enabled
        if self.config.get("logging", {}).get("use_mlflow", False):
            self._log_to_mlflow(model_name, model, results)

        return results

    def _detect_overfitting(
        self, train_metrics: Dict[str, float], test_metrics: Dict[str, float]
    ) -> float:
        """
        Detect overfitting by comparing train and test metrics.

        Args:
            train_metrics: Training metrics
            test_metrics: Test metrics

        Returns:
            Overfitting score (0-1, higher means more overfitting)
        """
        primary_metric = self.primary_metric

        if primary_metric not in train_metrics or primary_metric not in test_metrics:
            return 0.0

        train_score = train_metrics[primary_metric]
        test_score = test_metrics[primary_metric]

        # For metrics where higher is better
        if primary_metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "r2"]:
            return max(0, train_score - test_score)
        # For metrics where lower is better
        else:
            return max(0, test_score - train_score)

    def _log_to_mlflow(self, model_name: str, model: Any, results: Dict[str, Any]):
        """Log results to MLflow."""
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            # Log metrics
            for metric_name, value in results["test_metrics"].items():
                mlflow.log_metric(f"test_{metric_name}", value)

            for metric_name, value in results["train_metrics"].items():
                mlflow.log_metric(f"train_{metric_name}", value)

            mlflow.log_metric("overfit_score", results["overfit_score"])

            # Log model
            mlflow.sklearn.log_model(model, "model")

    def compare_models(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models and rank them.

        Args:
            evaluation_results: List of evaluation results

        Returns:
            Comparison results with rankings
        """
        # Sort by primary metric on test set
        primary_metric = self.primary_metric

        # Determine if higher or lower is better
        higher_is_better = primary_metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "r2"]

        sorted_results = sorted(
            evaluation_results,
            key=lambda x: x["test_metrics"].get(primary_metric, 0),
            reverse=higher_is_better,
        )

        rankings = []
        for rank, result in enumerate(sorted_results, 1):
            rankings.append({
                "rank": rank,
                "model_name": result["model_name"],
                "test_score": result["test_metrics"].get(primary_metric, 0),
                "overfit_score": result["overfit_score"],
            })

        return {
            "rankings": rankings,
            "best_model": rankings[0]["model_name"],
            "best_score": rankings[0]["test_score"],
            "primary_metric": primary_metric,
        }

    def execute(
        self,
        models: List[Dict[str, Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Execute evaluation agent workflow.

        Args:
            models: List of trained models
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Complete evaluation and comparison results
        """
        evaluation_results = []

        for model_info in models:
            model = model_info["optimized_model"]
            model_name = model_info["model_name"]

            result = self.evaluate_model(
                model, X_train, y_train, X_test, y_test, model_name
            )
            evaluation_results.append(result)

        comparison = self.compare_models(evaluation_results)

        return {
            "evaluation_results": evaluation_results,
            "comparison": comparison,
        }
