"""
Data Utilities
Tools for data loading, splitting, and validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.

    Args:
        file_path: Path to data file
        **kwargs: Additional arguments for pandas read functions

    Returns:
        Loaded dataframe
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path, **kwargs)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path, **kwargs)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        validation_size: Proportion of validation set (from training data)
        random_state: Random seed
        stratify: Whether to stratify split (for classification)

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    stratify_param = y if stratify else None

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    # Second split: train vs val
    if validation_size > 0:
        val_size_adjusted = validation_size / (1 - test_size)
        stratify_param_temp = y_temp if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=stratify_param_temp
        )
    else:
        X_train, y_train = X_temp, y_temp
        X_val, y_val = None, None

    return X_train, X_val, X_test, y_train, y_val, y_test


def validate_data(data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Validate dataset for common issues.

    Args:
        data: Input dataframe
        target_column: Name of target column

    Returns:
        Validation report dictionary
    """
    issues = []

    # Check if target column exists
    if target_column not in data.columns:
        issues.append(f"Target column '{target_column}' not found in dataset")

    # Check for empty dataset
    if len(data) == 0:
        issues.append("Dataset is empty")

    # Check for too few samples
    if len(data) < 10:
        issues.append(f"Dataset has only {len(data)} samples, which is too few")

    # Check for missing target values
    if target_column in data.columns:
        missing_target = data[target_column].isnull().sum()
        if missing_target > 0:
            issues.append(f"Target column has {missing_target} missing values")

    # Check for constant features
    constant_features = []
    for col in data.columns:
        if col != target_column and data[col].nunique() == 1:
            constant_features.append(col)

    if constant_features:
        issues.append(f"Constant features detected: {constant_features}")

    # Check for high cardinality categorical features
    high_cardinality = []
    for col in data.select_dtypes(include=['object', 'category']).columns:
        if col != target_column and data[col].nunique() > 100:
            high_cardinality.append(col)

    if high_cardinality:
        issues.append(f"High cardinality features (>100 unique values): {high_cardinality}")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "n_samples": len(data),
        "n_features": len(data.columns) - 1,
    }


def get_feature_target_split(
    data: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.

    Args:
        data: Input dataframe
        target_column: Name of target column

    Returns:
        X (features), y (target)
    """
    X = data.drop(columns=[target_column])
    y = data[target_column].copy()
    return X, y


def infer_task_type(y: pd.Series, max_unique_for_classification: int = 20) -> str:
    """
    Infer whether task is classification or regression based on target.

    Args:
        y: Target variable
        max_unique_for_classification: Max unique values to consider classification

    Returns:
        "classification" or "regression"
    """
    # Remove missing values for analysis
    y_clean = y.dropna()

    if len(y_clean) == 0:
        raise ValueError("Target has no non-null values")

    # If dtype is object or category, it's classification
    if y.dtype in ['object', 'category']:
        return "classification"

    # If dtype is bool, it's classification
    if y.dtype == 'bool':
        return "classification"

    # Check number of unique values
    n_unique = y_clean.nunique()

    # If few unique values, likely classification
    if n_unique <= max_unique_for_classification:
        return "classification"

    # If many unique values and all integers, still might be classification
    # but safer to treat as regression
    if n_unique > max_unique_for_classification:
        # Check if all values are integers
        if np.all(y_clean == y_clean.astype(int)):
            # Could be classification with many classes or regression
            # Use threshold to decide
            return "classification" if n_unique <= 100 else "regression"
        else:
            return "regression"

    return "regression"


def prepare_target(y: pd.Series, task_type: str) -> np.ndarray:
    """
    Prepare target variable for modeling.

    Args:
        y: Target variable
        task_type: "classification" or "regression"

    Returns:
        Prepared target as numpy array
    """
    y_clean = y.copy()

    if task_type == "classification":
        # For classification, ensure integer type
        if y.dtype in ['object', 'category', 'bool']:
            # Use label encoding for categorical targets
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            return le.fit_transform(y_clean)
        else:
            # Convert to int
            return y_clean.astype(int).values
    else:
        # For regression, ensure float type
        return y_clean.astype(float).values
