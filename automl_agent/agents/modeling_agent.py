"""
Modeling Agent
Función:
- Seleccionar familias de modelos
- Construir pipelines iniciales

Tools:
- scikit-learn
- lightgbm
- xgboost
- catboost
"""

from typing import Dict, Any, List
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


class ModelingAgent:
    """Agent responsible for model selection and pipeline construction."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Modeling Agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_families = config.get("model_families", [
            "lightgbm", "xgboost", "catboost", "random_forest", "logistic_regression"
        ])

    def get_base_models(self, task_type: str) -> Dict[str, Any]:
        """
        Get base models for given task type.

        Args:
            task_type: "classification" or "regression"

        Returns:
            Dictionary of model name to model instance
        """
        models = {}

        if task_type == "classification":
            if "lightgbm" in self.model_families:
                models["lightgbm"] = lgb.LGBMClassifier(random_state=42, verbose=-1)
            if "xgboost" in self.model_families:
                models["xgboost"] = xgb.XGBClassifier(random_state=42, verbosity=0)
            if "catboost" in self.model_families:
                models["catboost"] = cb.CatBoostClassifier(random_state=42, verbose=False)
            if "random_forest" in self.model_families:
                models["random_forest"] = RandomForestClassifier(random_state=42)
            if "logistic_regression" in self.model_families:
                models["logistic_regression"] = LogisticRegression(random_state=42, max_iter=1000)

        elif task_type == "regression":
            if "lightgbm" in self.model_families:
                models["lightgbm"] = lgb.LGBMRegressor(random_state=42, verbose=-1)
            if "xgboost" in self.model_families:
                models["xgboost"] = xgb.XGBRegressor(random_state=42, verbosity=0)
            if "catboost" in self.model_families:
                models["catboost"] = cb.CatBoostRegressor(random_state=42, verbose=False)
            if "random_forest" in self.model_families:
                models["random_forest"] = RandomForestRegressor(random_state=42)
            if "ridge" in self.model_families:
                models["ridge"] = Ridge(random_state=42)

        return models

    def build_pipeline_candidates(self, dataset_summary: Dict[str, Any], preprocessing_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build initial pipeline candidates.

        Args:
            dataset_summary: Dataset summary from DataAgent
            preprocessing_plan: Preprocessing plan from DataAgent

        Returns:
            List of pipeline candidates
        """
        task_type = dataset_summary["task_type"]
        base_models = self.get_base_models(task_type)

        candidates = []
        for model_name, model in base_models.items():
            candidate = {
                "model_name": model_name,
                "model": model,
                "task_type": task_type,
                "preprocessing": preprocessing_plan,
                "hyperparameters": self._get_default_hyperparameters(model_name),
            }
            candidates.append(candidate)

        return candidates

    def _get_default_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """
        Get default hyperparameter ranges for HPO.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of hyperparameter ranges
        """
        hyperparameters = {
            "lightgbm": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "num_leaves": [15, 31, 63],
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
            },
            "catboost": {
                "iterations": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "depth": [4, 6, 8],
            },
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            },
            "logistic_regression": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"],
            },
            "ridge": {
                "alpha": [0.1, 1.0, 10.0, 100.0],
            },
        }

        return hyperparameters.get(model_name, {})

    def execute(self, dataset_summary: Dict[str, Any], preprocessing_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute modeling agent workflow.

        Args:
            dataset_summary: Dataset summary
            preprocessing_plan: Preprocessing plan

        Returns:
            List of pipeline candidates
        """
        return self.build_pipeline_candidates(dataset_summary, preprocessing_plan)
