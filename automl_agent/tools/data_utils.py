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
    errors = []  # Critical errors that prevent execution
    warnings = []  # Non-critical issues that can be auto-fixed

    # Check if target column exists (CRITICAL)
    if target_column not in data.columns:
        errors.append(f"Target column '{target_column}' not found in dataset")

    # Check for empty dataset (CRITICAL)
    if len(data) == 0:
        errors.append("Dataset is empty")

    # Check for too few samples (CRITICAL)
    if len(data) < 10:
        errors.append(f"Dataset has only {len(data)} samples, which is too few")

    # Check for missing target values (WARNING - can be auto-fixed)
    if target_column in data.columns:
        missing_target = data[target_column].isnull().sum()
        if missing_target > 0:
            warnings.append(f"Target column has {missing_target} missing values (will be auto-removed)")

    # Check for constant features (WARNING - can be auto-removed)
    constant_features = []
    for col in data.columns:
        if col != target_column and data[col].nunique() == 1:
            constant_features.append(col)

    if constant_features:
        warnings.append(f"Constant features detected (will be auto-removed): {constant_features}")

    # Check for high cardinality categorical features (WARNING - can be handled)
    high_cardinality = []
    for col in data.select_dtypes(include=['object', 'category']).columns:
        if col != target_column and data[col].nunique() > 100:
            high_cardinality.append(col)

    if high_cardinality:
        warnings.append(f"High cardinality features detected (will be auto-handled): {high_cardinality}")

    return {
        "is_valid": len(errors) == 0,  # Only critical errors block execution
        "errors": errors,
        "warnings": warnings,
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
    n_samples = len(y_clean)

    # Calculate the ratio of unique values to samples
    unique_ratio = n_unique / n_samples

    # If few unique values (< 20) and low unique ratio, likely classification
    if n_unique <= max_unique_for_classification and unique_ratio < 0.5:
        return "classification"

    # If unique ratio is very high (> 0.95), it's likely regression
    # This catches continuous variables with many unique values
    if unique_ratio > 0.95:
        return "regression"

    # For intermediate cases, check the range and distribution
    if np.all(y_clean == y_clean.astype(int)):
        # All values are integers
        value_range = y_clean.max() - y_clean.min()

        # If the range is large relative to unique values, it's likely regression
        # (e.g., scores 0-100, prices, ages)
        if value_range > n_unique * 2:
            return "regression"

        # If few unique values relative to range, and low count, classification
        if n_unique <= 20:
            return "classification"

        # Many unique integer values, treat as regression
        return "regression"
    else:
        # Float values, almost certainly regression
        return "regression"


def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by stripping whitespace.

    Args:
        data: Input dataframe

    Returns:
        Dataframe with cleaned column names
    """
    data.columns = data.columns.str.strip()
    return data


def auto_drop_irrelevant_columns(
    data: pd.DataFrame, target_column: str, verbose: bool = True
) -> Tuple[pd.DataFrame, list]:
    """
    Automatically drop irrelevant columns that don't contribute to ML.

    Drops columns that are:
    - ID-like (named 'id', 'index', etc. or all unique values)
    - High cardinality categorical (>100 unique values for string columns)
    - Mostly missing (>80% null)
    - Constant (only 1 unique value)

    Args:
        data: Input dataframe
        target_column: Name of target column (won't be dropped)
        verbose: Whether to print info about dropped columns

    Returns:
        Cleaned dataframe, list of dropped column names
    """
    data_clean = data.copy()
    dropped_cols = []

    for col in data.columns:
        if col == target_column:
            continue

        n_unique = data[col].nunique()
        n_samples = len(data)
        missing_pct = data[col].isnull().sum() / n_samples

        # Drop ID-like columns (all or almost all unique)
        if n_unique == n_samples or n_unique == n_samples - 1:
            if verbose:
                print(f"[AUTO-DROP] '{col}': ID-like column (all unique values)")
            dropped_cols.append(col)
            continue

        # Drop constant columns
        if n_unique == 1:
            if verbose:
                print(f"[AUTO-DROP] '{col}': Constant column (only 1 unique value)")
            dropped_cols.append(col)
            continue

        # Drop columns with >70% missing values
        if missing_pct > 0.7:
            if verbose:
                print(f"[AUTO-DROP] '{col}': Too many missing values ({missing_pct*100:.1f}%)")
            dropped_cols.append(col)
            continue

        # Drop high cardinality categorical columns (likely useless)
        if data[col].dtype in ['object', 'category']:
            if n_unique > 100:
                # Check if it's truly high cardinality (>50% unique)
                unique_ratio = n_unique / n_samples
                if unique_ratio > 0.5:
                    if verbose:
                        print(f"[AUTO-DROP] '{col}': High cardinality categorical ({n_unique} unique values)")
                    dropped_cols.append(col)

    # Drop all identified columns
    if dropped_cols:
        data_clean = data_clean.drop(columns=dropped_cols)
        if verbose:
            print(f"[OK] Dropped {len(dropped_cols)} irrelevant columns: {dropped_cols}")

    return data_clean, dropped_cols


def clean_target_missing_values(
    data: pd.DataFrame, target_column: str, verbose: bool = True
) -> pd.DataFrame:
    """
    Remove rows where target column has missing values.

    Args:
        data: Input dataframe
        target_column: Name of target column
        verbose: Whether to print info

    Returns:
        Cleaned dataframe
    """
    missing_count = data[target_column].isnull().sum()

    if missing_count > 0:
        data_clean = data.dropna(subset=[target_column]).copy()
        if verbose:
            print(f"[AUTO-CLEAN] Removed {missing_count} rows with missing target values")
            print(f"[OK] Dataset size: {len(data)} -> {len(data_clean)} rows")
        return data_clean

    return data.copy()


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
