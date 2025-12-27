"""
Data Agent
Función:
- Analizar el dataset
- Proponer preprocesado
- Detectar problemas básicos

Tools:
- pandas
- numpy
- ydata-profiling (opcional)
- sklearn.preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class DataAgent:
    """Agent responsible for data analysis and preprocessing."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Data Agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def analyze_dataset(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Analyze dataset and return summary.

        Args:
            data: Input dataframe
            target_column: Name of target column

        Returns:
            Dataset summary dictionary
        """
        from ..tools.data_utils import infer_task_type

        summary = {
            "n_rows": len(data),
            "n_features": len(data.columns) - 1,
            "target_column": target_column,
            "feature_types": {},
            "missing_values": {},
            "target_distribution": {},
            "correlations": {},
            "outliers": {},
        }

        # Analyze feature types
        for col in data.columns:
            if col == target_column:
                continue
            summary["feature_types"][col] = str(data[col].dtype)
            summary["missing_values"][col] = int(data[col].isnull().sum())

        # Infer task type using robust method
        task_type = infer_task_type(data[target_column])
        summary["task_type"] = task_type

        # Analyze target based on task type
        if task_type == "classification":
            summary["target_distribution"] = data[target_column].value_counts().to_dict()
            summary["n_classes"] = data[target_column].nunique()
        else:
            summary["target_distribution"] = {
                "mean": float(data[target_column].mean()),
                "std": float(data[target_column].std()),
                "min": float(data[target_column].min()),
                "max": float(data[target_column].max()),
            }

        return summary

    def propose_preprocessing(self, dataset_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose preprocessing steps based on dataset analysis.

        Args:
            dataset_summary: Summary from analyze_dataset

        Returns:
            Preprocessing plan dictionary
        """
        plan = {
            "handle_missing": [],
            "encode_categorical": [],
            "scale_features": [],
            "feature_engineering": [],
        }

        # Handle missing values
        for col, missing_count in dataset_summary["missing_values"].items():
            if missing_count > 0:
                plan["handle_missing"].append({
                    "column": col,
                    "strategy": "mean" if dataset_summary["feature_types"][col] in ["float64", "int64"] else "most_frequent"
                })

        # Encode categorical features
        for col, dtype in dataset_summary["feature_types"].items():
            if dtype in ["object", "category"]:
                plan["encode_categorical"].append({
                    "column": col,
                    "strategy": "onehot"  # or "label" for tree-based models
                })

        # Scale numerical features
        for col, dtype in dataset_summary["feature_types"].items():
            if dtype in ["float64", "int64"]:
                plan["scale_features"].append({
                    "column": col,
                    "strategy": "standard"
                })

        return plan

    def execute(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Execute data agent workflow.

        Args:
            data: Input dataframe
            target_column: Target column name

        Returns:
            Complete analysis and preprocessing plan
        """
        summary = self.analyze_dataset(data, target_column)
        preprocessing_plan = self.propose_preprocessing(summary)

        return {
            "dataset_summary": summary,
            "preprocessing_plan": preprocessing_plan,
        }
