"""
HPO Agent
Función:
- Optimizar hiperparámetros
- Ejecutar AutoML ligero

Tools:
- FLAML (principal)
- Optuna (secundario)
- joblib (paralelismo)
"""

from typing import Dict, Any, Optional
import numpy as np
from flaml import AutoML
import optuna
from sklearn.model_selection import cross_val_score


class HPOAgent:
    """Agent responsible for hyperparameter optimization."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HPO Agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.hpo_config = config.get("hpo", {})
        self.engine = self.hpo_config.get("engine", "flaml")

    def optimize_with_flaml(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str,
        metric: str,
        time_budget: int = 300,
    ) -> Dict[str, Any]:
        """
        Optimize using FLAML AutoML.

        Args:
            X_train: Training features
            y_train: Training target
            task_type: "classification" or "regression"
            metric: Metric to optimize
            time_budget: Time budget in seconds

        Returns:
            Dictionary with best model and parameters
        """
        automl = AutoML()

        automl_settings = {
            "time_budget": time_budget,
            "metric": metric,
            "task": task_type,
            "log_file_name": "flaml.log",
            "seed": 42,
        }

        automl.fit(X_train, y_train, **automl_settings)

        return {
            "best_model": automl.model,
            "best_estimator": automl.best_estimator,
            "best_config": automl.best_config,
            "best_loss": automl.best_loss,
            "training_time": automl.time_to_find_best_model,
        }

    def optimize_with_optuna(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameter_space: Dict[str, Any],
        metric: str,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Optimize using Optuna.

        Args:
            model: Base model to optimize
            X_train: Training features
            y_train: Training target
            hyperparameter_space: Hyperparameter search space
            metric: Metric to optimize
            n_trials: Number of trials

        Returns:
            Dictionary with best parameters and score
        """
        def objective(trial):
            params = {}
            for param_name, param_values in hyperparameter_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, int) for v in param_values):
                        params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    elif all(isinstance(v, float) for v in param_values):
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)

            model.set_params(**params)
            score = cross_val_score(model, X_train, y_train, cv=3, scoring=metric).mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
        }

    def optimize_pipeline(
        self,
        pipeline_candidate: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Optimize a pipeline candidate.

        Args:
            pipeline_candidate: Pipeline candidate from ModelingAgent
            X_train: Training features
            y_train: Training target

        Returns:
            Optimized pipeline with best parameters
        """
        model = pipeline_candidate["model"]
        model_name = pipeline_candidate["model_name"]
        task_type = pipeline_candidate["task_type"]
        hyperparameters = pipeline_candidate["hyperparameters"]

        metric = self.config["metrics"]["primary"]
        time_budget = self.hpo_config.get("time_budget", 300)
        n_trials = self.hpo_config.get("n_trials", 50)

        if self.engine == "flaml":
            result = self.optimize_with_flaml(
                X_train, y_train, task_type, metric, time_budget
            )
        else:
            result = self.optimize_with_optuna(
                model, X_train, y_train, hyperparameters, metric, n_trials
            )

        return {
            "model_name": model_name,
            "optimized_model": result.get("best_model", model),
            "best_params": result.get("best_config", result.get("best_params", {})),
            "optimization_score": result.get("best_loss", result.get("best_score", 0)),
            "optimization_method": self.engine,
        }

    def execute(
        self,
        pipeline_candidates: list,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> list:
        """
        Execute HPO agent workflow on all candidates.

        Args:
            pipeline_candidates: List of pipeline candidates
            X_train: Training features
            y_train: Training target

        Returns:
            List of optimized pipelines
        """
        optimized_pipelines = []

        for candidate in pipeline_candidates:
            optimized = self.optimize_pipeline(candidate, X_train, y_train)
            optimized_pipelines.append(optimized)

        return optimized_pipelines
