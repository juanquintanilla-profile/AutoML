"""
Preprocessing Utilities
Tools for data preprocessing: encoding, scaling, imputation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List, Tuple


class PreprocessingPipeline:
    """Pipeline for data preprocessing."""

    def __init__(self, preprocessing_plan: Dict[str, Any]):
        """
        Initialize preprocessing pipeline.

        Args:
            preprocessing_plan: Plan from DataAgent
        """
        self.plan = preprocessing_plan
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.feature_names = None

    def fit(self, X: pd.DataFrame) -> 'PreprocessingPipeline':
        """
        Fit preprocessing transformers.

        Args:
            X: Training features

        Returns:
            self
        """
        X_copy = X.copy()

        # Handle missing values
        for step in self.plan.get("handle_missing", []):
            col = step["column"]
            strategy = step["strategy"]

            if col in X_copy.columns:
                imputer = SimpleImputer(strategy=strategy)
                X_copy[col] = imputer.fit_transform(X_copy[[col]]).ravel()
                self.imputers[col] = imputer

        # Encode categorical features
        for step in self.plan.get("encode_categorical", []):
            col = step["column"]
            strategy = step["strategy"]

            if col in X_copy.columns:
                if strategy == "onehot":
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    encoder.fit(X_copy[[col]])
                    self.encoders[col] = encoder
                elif strategy == "label":
                    encoder = LabelEncoder()
                    encoder.fit(X_copy[col])
                    self.encoders[col] = encoder

        # Fit scalers for numerical features
        for step in self.plan.get("scale_features", []):
            col = step["column"]
            strategy = step["strategy"]

            if col in X_copy.columns:
                if strategy == "standard":
                    scaler = StandardScaler()
                elif strategy == "minmax":
                    scaler = MinMaxScaler()
                elif strategy == "robust":
                    scaler = RobustScaler()
                else:
                    continue

                scaler.fit(X_copy[[col]])
                self.scalers[col] = scaler

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted transformers.

        Args:
            X: Features to transform

        Returns:
            Transformed features as numpy array
        """
        X_copy = X.copy()

        # Handle missing values
        for col, imputer in self.imputers.items():
            if col in X_copy.columns:
                X_copy[col] = imputer.transform(X_copy[[col]]).ravel()

        # Encode categorical features
        encoded_cols = []
        cols_to_drop = []

        for col, encoder in self.encoders.items():
            if col in X_copy.columns:
                if isinstance(encoder, OneHotEncoder):
                    encoded = encoder.transform(X_copy[[col]])
                    feature_names = encoder.get_feature_names_out([col])
                    encoded_df = pd.DataFrame(
                        encoded, columns=feature_names, index=X_copy.index
                    )
                    encoded_cols.append(encoded_df)
                    cols_to_drop.append(col)
                elif isinstance(encoder, LabelEncoder):
                    X_copy[col] = encoder.transform(X_copy[col])

        # Drop original categorical columns and add encoded ones
        if cols_to_drop:
            X_copy = X_copy.drop(columns=cols_to_drop)
        if encoded_cols:
            X_copy = pd.concat([X_copy] + encoded_cols, axis=1)

        # Scale numerical features
        for col, scaler in self.scalers.items():
            if col in X_copy.columns:
                X_copy[col] = scaler.transform(X_copy[[col]])

        return X_copy.values

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Features

        Returns:
            Transformed features
        """
        return self.fit(X).transform(X)


def apply_preprocessing(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    preprocessing_plan: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, PreprocessingPipeline]:
    """
    Apply preprocessing to train and test sets.

    Args:
        X_train: Training features
        X_test: Test features
        preprocessing_plan: Preprocessing plan

    Returns:
        X_train_processed, X_test_processed, pipeline
    """
    pipeline = PreprocessingPipeline(preprocessing_plan)

    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    return X_train_processed, X_test_processed, pipeline
