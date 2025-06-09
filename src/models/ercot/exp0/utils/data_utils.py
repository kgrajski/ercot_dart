"""
Data Utilities for DART Modeling

Functions for loading, preprocessing, and splitting DART data for model training.
Handles the operational constraint: 6AM data cutoff -> 10AM prediction deadline.
"""

from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def load_dart_data(
    data_path: Union[str, Path] = "data/processed/ercot",
    settlement_points: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load DART data from processed files.

    Args:
        data_path: Path to processed ERCOT data
        settlement_points: List of settlement points to load (default: all)
        start_date: Start date filter (YYYY-MM-DD format)
        end_date: End date filter (YYYY-MM-DD format)

    Returns:
        DataFrame with DART data indexed by datetime
    """
    data_path = Path(data_path)

    # TODO: Implement actual data loading from your processed files
    # This will depend on your file structure from the data processing pipeline

    # Placeholder implementation - replace with actual loading logic
    raise NotImplementedError("Implement based on your processed data file structure")


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    method: str = "time_based",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test splits for time series data.

    Args:
        df: Input DataFrame with datetime index
        test_size: Fraction for test set (most recent data)
        validation_size: Fraction for validation set
        method: Split method ('time_based', 'random', 'walk_forward')

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if method == "time_based":
        # Time-based split (most common for time series)
        n = len(df)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))

        train_df = df.iloc[:val_idx].copy()
        val_df = df.iloc[val_idx:test_idx].copy()
        test_df = df.iloc[test_idx:].copy()

    elif method == "random":
        # Random split (less common for time series)
        from sklearn.model_selection import train_test_split

        train_val, test_df = train_test_split(df, test_size=test_size, random_state=42)
        train_df, val_df = train_test_split(
            train_val, test_size=validation_size, random_state=42
        )

    else:
        raise ValueError(f"Unknown split method: {method}")

    return train_df, val_df, test_df


def create_time_series_splits(
    df: pd.DataFrame, n_splits: int = 5, test_size: int = 24 * 7
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time series cross-validation splits.

    Args:
        df: Input DataFrame
        n_splits: Number of splits
        test_size: Size of each test set (in hours)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    return list(tscv.split(df))


def create_operational_features(
    df: pd.DataFrame, prediction_horizon: int = 24
) -> pd.DataFrame:
    """
    Create features that respect operational constraints.

    ERCOT DAM operational reality:
    - 6AM: Data cutoff for previous day
    - 10AM: Must submit DAM bids for next day
    - Need to predict: T+18 to T+42 hours ahead

    Args:
        df: Input DataFrame with DART data
        prediction_horizon: Hours ahead to predict (24 for next-day)

    Returns:
        DataFrame with operational features
    """
    features_df = df.copy()

    # Add time-based features
    features_df["hour"] = features_df.index.hour
    features_df["day_of_week"] = features_df.index.dayofweek
    features_df["month"] = features_df.index.month
    features_df["is_weekend"] = features_df.index.dayofweek >= 5
    features_df["is_business_hours"] = (features_df["hour"] >= 8) & (
        features_df["hour"] <= 17
    )

    # Add lag features (available at prediction time)
    # These respect the 6AM cutoff constraint
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h to 1 week
    for lag in lag_hours:
        features_df[f"dart_lag_{lag}h"] = features_df["dart"].shift(lag)
        features_df[f"dart_slt_lag_{lag}h"] = features_df["dart_slt"].shift(lag)

    # Add rolling features
    windows = [6, 12, 24, 48, 168]  # 6h to 1 week
    for window in windows:
        features_df[f"dart_mean_{window}h"] = (
            features_df["dart"].rolling(window).mean().shift(1)
        )
        features_df[f"dart_std_{window}h"] = (
            features_df["dart"].rolling(window).std().shift(1)
        )
        features_df[f"dart_slt_mean_{window}h"] = (
            features_df["dart_slt"].rolling(window).mean().shift(1)
        )

    # Sign persistence features (from your EDA showing 80% persistence)
    features_df["dart_sign"] = np.sign(features_df["dart"])
    features_df["dart_sign_lag_1h"] = features_df["dart_sign"].shift(1)
    features_df["dart_sign_lag_24h"] = features_df["dart_sign"].shift(24)

    return features_df


def align_features_targets(
    df: pd.DataFrame,
    target_column: str = "dart",
    prediction_horizon: int = 24,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align features with targets for supervised learning.

    Creates X (features at time t) and y (target at time t + horizon).

    Args:
        df: DataFrame with features and targets
        target_column: Name of target column
        prediction_horizon: Hours ahead to predict
        feature_columns: List of feature columns (auto-detect if None)

    Returns:
        Tuple of (X_features, y_targets) aligned for training
    """
    if feature_columns is None:
        # Auto-detect feature columns (exclude target and datetime)
        exclude_cols = (
            [target_column, "dart", "dart_slt"]
            if target_column not in ["dart", "dart_slt"]
            else [target_column]
        )
        feature_columns = [col for col in df.columns if col not in exclude_cols]

    # Shift target forward by prediction horizon
    y = df[target_column].shift(-prediction_horizon)
    X = df[feature_columns]

    # Remove rows with NaN values
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X_aligned = X[valid_mask].copy()
    y_aligned = y[valid_mask].copy()

    return X_aligned, y_aligned


def filter_business_hours(
    df: pd.DataFrame, business_hours_only: bool = False, exclude_weekends: bool = False
) -> pd.DataFrame:
    """
    Filter data based on business hour patterns from your EDA.

    Args:
        df: Input DataFrame with datetime index
        business_hours_only: If True, keep only 8AM-6PM hours
        exclude_weekends: If True, exclude Saturday/Sunday

    Returns:
        Filtered DataFrame
    """
    mask = pd.Series(True, index=df.index)

    if business_hours_only:
        mask &= (df.index.hour >= 8) & (df.index.hour <= 18)

    if exclude_weekends:
        mask &= df.index.dayofweek < 5

    return df[mask].copy()
