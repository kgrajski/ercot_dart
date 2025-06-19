"""Evaluation Metrics for ERCOT DART Price Prediction.

This module provides comprehensive evaluation metrics for electricity price forecasting,
including standard regression metrics and electricity-specific measures from academic
literature (particularly Lago et al., 2021).

Key Features:
- Standard metrics: R², MAE, RMSE
- Electricity-specific metrics: sMAPE, Directional Accuracy, etc.
- Bootstrap-aware: Calculate metrics across multiple samples with mean/std
- Time-aware: Handle sequential data with proper temporal ordering
- Robust to negative prices: Avoids problematic metrics like MAPE

Metrics Explained:

=== STANDARD REGRESSION METRICS ===

R² (Coefficient of Determination):
- Measures proportion of variance explained by the model (1 - SS_residual/SS_total)
- Well-performing model: R² > 0.7 (electricity markets often 0.3-0.6 due to volatility)
- Warning signs: R² < 0.1 (poor predictive power), R² < 0 (worse than predicting mean)

MAE (Mean Absolute Error):
- Average absolute difference between predictions and actual values ($/MWh)
- Well-performing model: MAE < 10-20 $/MWh for DART prices
- Warning signs: MAE > 50 $/MWh (large systematic errors)

RMSE (Root Mean Squared Error):
- Square root of mean squared errors, penalizes large errors more than MAE ($/MWh)
- Well-performing model: RMSE < 15-30 $/MWh, RMSE/MAE ratio 1.2-1.5
- Warning signs: RMSE >> MAE (many large outlier errors), RMSE > 100 $/MWh

MAPE (Mean Absolute Percentage Error):
- Average absolute percentage error (%) - only calculated for |price| > threshold
- Well-performing model: MAPE < 20-30% for electricity prices
- Warning signs: MAPE > 50% (consistently large relative errors)
- Note: Problematic with near-zero or negative prices, use sMAPE instead

=== ADVANCED DOMAIN-SPECIFIC METRICS ===

sMAPE (Symmetric Mean Absolute Percentage Error):
- Symmetric version of MAPE, handles negative prices better (%)
- Well-performing model: sMAPE < 25-35% for electricity prices
- Warning signs: sMAPE > 60% (poor relative accuracy)

Directional Accuracy:
- Percentage of correct price movement direction predictions (%)
- Well-performing model: > 55-60% (better than random 50%)
- Warning signs: < 45% (worse than random guessing), exactly 50% (no directional skill)

MASE (Mean Absolute Scaled Error):
- MAE relative to naive forecast (persistence model)
- Well-performing model: MASE < 1.0 (better than naive), ideally 0.5-0.8
- Warning signs: MASE > 1.2 (worse than persistence), MASE > 2.0 (much worse than naive)

=== TIME-AWARE METRICS ===

Spike Detection (F1/Precision/Recall):
- Accuracy in predicting price spikes (> 2 standard deviations above mean)
- Well-performing model: F1 > 0.6, Precision > 0.5, Recall > 0.5
- Warning signs: F1 < 0.3 (poor spike detection), Precision < 0.2 (many false alarms)

=== ROBUSTNESS INDICATORS ===

Finite Sample Rate:
- Proportion of valid (non-NaN/infinite) predictions
- Well-performing model: > 95% finite samples
- Warning signs: < 90% finite (numerical instability), < 80% (serious model issues)

Sample Counts:
- Number of valid samples for percentage-based metrics
- Well-performing model: Sample counts close to total samples
- Warning signs: Very low sample counts (most data excluded from metric calculation)

Price Statistics:
- Mean, std, range of actual vs predicted prices
- Well-performing model: Predicted stats similar to actual (within 10-20%)
- Warning signs: Large differences in mean (bias), very different std (over/under-confident)

Usage:
    evaluator = EvaluationMetrics()
    metrics = evaluator.calculate_metrics(y_true, y_pred, timestamps)
    bootstrap_metrics = evaluator.calculate_bootstrap_metrics(y_true_list, y_pred_list)
"""

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class EvaluationMetrics:
    """Comprehensive evaluation metrics for electricity price forecasting.

    This class implements both standard regression metrics and electricity market
    specific evaluation measures, with support for bootstrap sampling and
    time-aware calculations.
    """

    def __init__(self, time_aware: bool = True, price_threshold: float = 1.0):
        """Initialize evaluation metrics calculator.

        Args:
            time_aware: Whether to enforce temporal ordering of predictions
            price_threshold: Minimum absolute price for relative error calculations ($/MWh)
        """
        self.time_aware = time_aware
        self.price_threshold = price_threshold

    def calculate_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics for a single prediction set.

        Args:
            y_true: Actual values (target DART SLT prices)
            y_pred: Predicted values
            timestamps: Optional timestamps for time-aware metrics

        Returns:
            Dictionary with metric names as keys and values as floats

        Raises:
            ValueError: If arrays have different lengths or contain invalid values
        """
        # Convert to numpy arrays for consistent handling
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Validation
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
            )

        if len(y_true) == 0:
            raise ValueError("Empty prediction arrays")

        # Handle time-aware sorting if timestamps provided
        if self.time_aware and timestamps is not None:
            timestamps = np.asarray(timestamps)
            if len(timestamps) != len(y_true):
                raise ValueError(
                    f"Timestamp length ({len(timestamps)}) doesn't match predictions ({len(y_true)})"
                )

            # Sort by timestamps
            sort_idx = np.argsort(timestamps)
            y_true = y_true[sort_idx]
            y_pred = y_pred[sort_idx]
            timestamps = timestamps[sort_idx]

        # Check for finite values
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(finite_mask):
            # Return NaN metrics if no finite values
            return self._get_nan_metrics()

        # Filter to finite values only
        y_true_clean = y_true[finite_mask]
        y_pred_clean = y_pred[finite_mask]

        metrics = {}

        # === STANDARD REGRESSION METRICS ===
        metrics.update(self._calculate_standard_metrics(y_true_clean, y_pred_clean))

        # === ADVANCED DOMAIN-SPECIFIC METRICS ===
        metrics.update(self._calculate_advanced_metrics(y_true_clean, y_pred_clean))

        # === TIME-AWARE METRICS (if applicable) ===
        if timestamps is not None and len(y_true_clean) > 1:
            timestamps_clean = timestamps[finite_mask] if self.time_aware else None
            metrics.update(
                self._calculate_temporal_metrics(
                    y_true_clean, y_pred_clean, timestamps_clean
                )
            )

        # === ROBUSTNESS INDICATORS ===
        metrics.update(self._calculate_robustness_metrics(y_true, y_pred, finite_mask))

        return metrics

    def _calculate_standard_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate standard regression metrics."""
        metrics = {}

        try:
            metrics["r2"] = r2_score(y_true, y_pred)
        except Exception:
            metrics["r2"] = np.nan

        try:
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
        except Exception:
            metrics["mae"] = np.nan

        try:
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        except Exception:
            metrics["rmse"] = np.nan

        # Mean Absolute Percentage Error (with safeguards for near-zero values)
        try:
            # Avoid MAPE for values close to zero (problematic with DART negative prices)
            non_zero_mask = np.abs(y_true) > self.price_threshold
            if np.any(non_zero_mask):
                mape_values = (
                    np.abs(
                        (y_true[non_zero_mask] - y_pred[non_zero_mask])
                        / y_true[non_zero_mask]
                    )
                    * 100
                )
                metrics["mape"] = np.mean(mape_values)
                metrics["mape_sample_count"] = int(np.sum(non_zero_mask))
            else:
                metrics["mape"] = np.nan
                metrics["mape_sample_count"] = 0
        except Exception:
            metrics["mape"] = np.nan
            metrics["mape_sample_count"] = 0

        return metrics

    def _calculate_advanced_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate advanced domain-specific metrics for electricity price forecasting."""
        metrics = {}

        # Symmetric Mean Absolute Percentage Error (better for negative prices)
        try:
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            # Only calculate where denominator is meaningful
            valid_mask = denominator > self.price_threshold
            if np.any(valid_mask):
                smape_values = (
                    np.abs(y_true[valid_mask] - y_pred[valid_mask])
                    / denominator[valid_mask]
                    * 100
                )
                metrics["smape"] = np.mean(smape_values)
                metrics["smape_sample_count"] = int(np.sum(valid_mask))
            else:
                metrics["smape"] = np.nan
                metrics["smape_sample_count"] = 0
        except Exception:
            metrics["smape"] = np.nan
            metrics["smape_sample_count"] = 0

        # Directional Accuracy (percentage of correct price direction predictions)
        try:
            if len(y_true) > 1:
                # Calculate price changes
                true_changes = np.diff(y_true)
                pred_changes = np.diff(y_pred)

                # Direction accuracy: same sign of changes
                correct_directions = np.sign(true_changes) == np.sign(pred_changes)
                metrics["directional_accuracy"] = np.mean(correct_directions) * 100
                metrics["directional_accuracy_sample_count"] = len(correct_directions)
            else:
                metrics["directional_accuracy"] = np.nan
                metrics["directional_accuracy_sample_count"] = 0
        except Exception:
            metrics["directional_accuracy"] = np.nan
            metrics["directional_accuracy_sample_count"] = 0

        # Mean Absolute Scaled Error (MASE) - relative to naive forecast
        try:
            if len(y_true) > 1:
                naive_mae = np.mean(
                    np.abs(np.diff(y_true))
                )  # MAE of naive forecast (persistence)
                if naive_mae > 0:
                    metrics["mase"] = metrics.get("mae", np.nan) / naive_mae
                else:
                    metrics["mase"] = np.nan
            else:
                metrics["mase"] = np.nan
        except Exception:
            metrics["mase"] = np.nan

        return metrics

    def _calculate_temporal_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, timestamps: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate time-aware metrics that depend on sequential ordering."""
        metrics = {}

        # Price spike detection accuracy (define spike as > 2 std deviations)
        try:
            true_mean = np.mean(y_true)
            true_std = np.std(y_true)
            spike_threshold = true_mean + 2 * true_std

            true_spikes = y_true > spike_threshold
            pred_spikes = y_pred > spike_threshold

            if np.any(true_spikes) or np.any(pred_spikes):
                # F1 score for spike detection
                true_positives = int(np.sum(true_spikes & pred_spikes))
                false_positives = int(np.sum(~true_spikes & pred_spikes))
                false_negatives = int(np.sum(true_spikes & ~pred_spikes))

                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else 0
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )

                if precision + recall > 0:
                    metrics["spike_f1_score"] = (
                        2 * (precision * recall) / (precision + recall)
                    )
                else:
                    metrics["spike_f1_score"] = 0

                metrics["spike_precision"] = precision
                metrics["spike_recall"] = recall
                metrics["spike_count_true"] = int(np.sum(true_spikes))
                metrics["spike_count_pred"] = int(np.sum(pred_spikes))
            else:
                metrics["spike_f1_score"] = np.nan
                metrics["spike_precision"] = np.nan
                metrics["spike_recall"] = np.nan
                metrics["spike_count_true"] = 0
                metrics["spike_count_pred"] = 0
        except Exception:
            metrics["spike_f1_score"] = np.nan
            metrics["spike_precision"] = np.nan
            metrics["spike_recall"] = np.nan
            metrics["spike_count_true"] = 0
            metrics["spike_count_pred"] = 0

        return metrics

    def _calculate_robustness_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, finite_mask: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics about data quality and robustness."""
        metrics = {}

        metrics["total_samples"] = len(y_true)
        metrics["finite_samples"] = int(np.sum(finite_mask))
        metrics["finite_sample_rate"] = (
            float(np.sum(finite_mask)) / len(y_true) if len(y_true) > 0 else 0.0
        )

        # Price range statistics
        finite_true = y_true[finite_mask]
        finite_pred = y_pred[finite_mask]

        if len(finite_true) > 0:
            metrics["price_range_true"] = np.max(finite_true) - np.min(finite_true)
            metrics["price_mean_true"] = np.mean(finite_true)
            metrics["price_std_true"] = np.std(finite_true)
        else:
            metrics["price_range_true"] = np.nan
            metrics["price_mean_true"] = np.nan
            metrics["price_std_true"] = np.nan

        if len(finite_pred) > 0:
            metrics["price_range_pred"] = np.max(finite_pred) - np.min(finite_pred)
            metrics["price_mean_pred"] = np.mean(finite_pred)
            metrics["price_std_pred"] = np.std(finite_pred)
        else:
            metrics["price_range_pred"] = np.nan
            metrics["price_mean_pred"] = np.nan
            metrics["price_std_pred"] = np.nan

        return metrics

    def _get_nan_metrics(self) -> Dict[str, float]:
        """Return dictionary with all metrics set to NaN for failed calculations."""
        return {
            "r2": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "mape_sample_count": 0,
            "smape": np.nan,
            "smape_sample_count": 0,
            "directional_accuracy": np.nan,
            "directional_accuracy_sample_count": 0,
            "mase": np.nan,
            "spike_f1_score": np.nan,
            "spike_precision": np.nan,
            "spike_recall": np.nan,
            "spike_count_true": 0,
            "spike_count_pred": 0,
            "total_samples": 0,
            "finite_samples": 0,
            "finite_sample_rate": 0,
            "price_range_true": np.nan,
            "price_mean_true": np.nan,
            "price_std_true": np.nan,
            "price_range_pred": np.nan,
            "price_mean_pred": np.nan,
            "price_std_pred": np.nan,
        }

    def calculate_bootstrap_metrics(
        self,
        y_true_list: List[Union[np.ndarray, pd.Series]],
        y_pred_list: List[Union[np.ndarray, pd.Series]],
        timestamps_list: Optional[List[Union[np.ndarray, pd.Series]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics across multiple bootstrap samples with mean/std summary.

        Args:
            y_true_list: List of true value arrays (one per bootstrap sample)
            y_pred_list: List of predicted value arrays (one per bootstrap sample)
            timestamps_list: Optional list of timestamp arrays

        Returns:
            Dictionary with structure: {metric_name: {'mean': float, 'std': float, 'count': int}}
        """
        if len(y_true_list) != len(y_pred_list):
            raise ValueError("Mismatch in number of bootstrap samples")

        if len(y_true_list) == 0:
            return {}

        # Calculate metrics for each bootstrap sample
        all_metrics = []
        failed_samples = 0

        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            timestamps = timestamps_list[i] if timestamps_list else None

            try:
                sample_metrics = self.calculate_metrics(y_true, y_pred, timestamps)
                all_metrics.append(sample_metrics)
            except Exception as e:
                failed_samples += 1
                continue

        if len(all_metrics) == 0:
            return {"failed_samples": failed_samples, "success_rate": 0.0}

        # Convert to DataFrame for easy aggregation
        metrics_df = pd.DataFrame(all_metrics)

        # Calculate mean and std for each metric
        result = {}
        for metric_name in metrics_df.columns:
            values = metrics_df[metric_name].dropna()  # Remove NaN values

            if len(values) > 0:
                result[metric_name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()) if len(values) > 1 else 0.0,
                    "count": len(values),
                    "nan_count": len(metrics_df) - len(values),
                }
            else:
                result[metric_name] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "count": 0,
                    "nan_count": len(metrics_df),
                }

        # Add summary statistics
        result["_summary"] = {
            "total_bootstrap_samples": len(y_true_list),
            "successful_samples": len(all_metrics),
            "failed_samples": failed_samples,
            "success_rate": len(all_metrics) / len(y_true_list),
        }

        return result

    def get_core_metrics(self, full_metrics: Dict) -> Dict[str, float]:
        """Extract core metrics for backward compatibility with existing code.

        Args:
            full_metrics: Full metrics dictionary from calculate_metrics()

        Returns:
            Dictionary with core metrics (r2, mae, rmse) for compatibility
        """
        return {
            "r2": full_metrics.get("r2", np.nan),
            "mae": full_metrics.get("mae", np.nan),
            "rmse": full_metrics.get("rmse", np.nan),
        }

    def format_for_storage(
        self, metrics: Dict, prefix: str = ""
    ) -> Dict[str, Union[float, int]]:
        """Format metrics dictionary for database/JSON storage.

        Args:
            metrics: Metrics dictionary (from calculate_metrics or calculate_bootstrap_metrics)
            prefix: Optional prefix for metric names (e.g., 'bootstrap_', 'validation_')

        Returns:
            Flattened dictionary suitable for database storage
        """
        storage_dict = {}

        for key, value in metrics.items():
            if isinstance(value, dict):
                # Handle nested dictionaries (from bootstrap metrics)
                for subkey, subvalue in value.items():
                    if subkey != "_summary":  # Skip summary for individual metrics
                        storage_key = f"{prefix}{key}_{subkey}"
                        storage_dict[storage_key] = subvalue
            else:
                # Handle simple values
                storage_key = f"{prefix}{key}"
                storage_dict[storage_key] = value

        return storage_dict
