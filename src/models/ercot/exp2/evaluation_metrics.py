"""Evaluation Metrics for ERCOT DART Price Classification.

This module provides comprehensive evaluation metrics for electricity price classification,
including standard classification metrics and electricity-specific measures adapted for
classification tasks.

Key Features:
- Standard classification metrics: Accuracy, Precision, Recall, F1-Score
- Multi-class support: Weighted averages and per-class metrics
- Class-specific analysis: Confusion matrices and classification reports
- Bootstrap-aware: Calculate metrics across multiple samples with mean/std
- Robust to class imbalance: Handles imbalanced electricity market classification

Metrics Explained:

=== STANDARD CLASSIFICATION METRICS ===

Accuracy:
- Proportion of correctly classified samples (0-1)
- Well-performing model: > 0.7 for electricity market classification
- Warning signs: < 0.6 (poor overall performance), exactly 0.5 for binary (random guessing)

Precision (Weighted):
- Average precision across classes, weighted by support
- Well-performing model: > 0.7 for electricity market classification
- Warning signs: < 0.5 (many false positives)

Recall (Weighted):
- Average recall across classes, weighted by support
- Well-performing model: > 0.7 for electricity market classification
- Warning signs: < 0.5 (many false negatives)

F1-Score (Weighted):
- Harmonic mean of precision and recall, weighted by support
- Well-performing model: > 0.7 for electricity market classification
- Warning signs: < 0.5 (poor balance of precision/recall)

=== PER-CLASS ANALYSIS ===

Per-Class Precision/Recall/F1:
- Individual metrics for each price class (e.g., negative/positive)
- Well-performing model: Balanced performance across classes
- Warning signs: Very poor performance on minority classes

Confusion Matrix:
- Matrix showing true vs predicted class counts
- Well-performing model: High diagonal values, low off-diagonal
- Warning signs: Many off-diagonal errors, especially systematic patterns

=== CLASS DISTRIBUTION ANALYSIS ===

Support (Per-Class):
- Number of samples in each true class
- Important for understanding class balance in dataset
- Warning signs: Very imbalanced classes (>95% in one class)

Classification Report:
- Comprehensive per-class and average metrics
- Provides detailed breakdown of model performance

=== ROBUSTNESS INDICATORS ===

Finite Sample Rate:
- Proportion of valid (non-NaN) predictions
- Well-performing model: > 95% finite samples
- Warning signs: < 90% finite (prediction issues)

Macro vs Weighted Averages:
- Macro: Unweighted average across classes
- Weighted: Average weighted by class support
- Well-performing model: Similar macro and weighted scores
- Warning signs: Large differences (poor minority class performance)

Usage:
    evaluator = EvaluationMetrics()
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    bootstrap_metrics = evaluator.calculate_bootstrap_metrics(y_true_list, y_pred_list)
"""

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class EvaluationMetrics:
    """Comprehensive evaluation metrics for electricity price classification.

    This class implements standard classification metrics and electricity market
    specific evaluation measures, with support for bootstrap sampling and
    multi-class classification.
    """

    def __init__(self, average: str = "weighted", zero_division: int = 0):
        """Initialize evaluation metrics calculator.

        Args:
            average: Averaging strategy for multi-class metrics ('weighted', 'macro', 'micro')
            zero_division: Value to return when there is zero division in metrics
        """
        self.average = average
        self.zero_division = zero_division

    def calculate_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        class_labels: Optional[List[str]] = None,
    ) -> Dict[str, Union[float, List, Dict]]:
        """Calculate comprehensive classification metrics for a single prediction set.

        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            class_labels: Optional list of class label names for reporting

        Returns:
            Dictionary with metric names as keys and values as floats, lists, or dicts

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

        # Check for finite values
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(finite_mask):
            # Return NaN metrics if no finite values
            return self._get_nan_metrics()

        # Filter to finite values only
        y_true_clean = y_true[finite_mask]
        y_pred_clean = y_pred[finite_mask]

        metrics = {}

        # === STANDARD CLASSIFICATION METRICS ===
        metrics.update(self._calculate_standard_metrics(y_true_clean, y_pred_clean))

        # === PER-CLASS ANALYSIS ===
        metrics.update(
            self._calculate_per_class_metrics(y_true_clean, y_pred_clean, class_labels)
        )

        # === CONFUSION MATRIX ===
        metrics.update(self._calculate_confusion_matrix(y_true_clean, y_pred_clean))

        # === ROBUSTNESS INDICATORS ===
        metrics.update(self._calculate_robustness_metrics(y_true, y_pred, finite_mask))

        return metrics

    def _calculate_standard_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate standard classification metrics."""
        metrics = {}

        try:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        except Exception:
            metrics["accuracy"] = np.nan

        try:
            metrics["precision"] = precision_score(
                y_true, y_pred, average=self.average, zero_division=self.zero_division
            )
        except Exception:
            metrics["precision"] = np.nan

        try:
            metrics["recall"] = recall_score(
                y_true, y_pred, average=self.average, zero_division=self.zero_division
            )
        except Exception:
            metrics["recall"] = np.nan

        try:
            metrics["f1"] = f1_score(
                y_true, y_pred, average=self.average, zero_division=self.zero_division
            )
        except Exception:
            metrics["f1"] = np.nan

        # Calculate macro averages as well for comparison
        try:
            metrics["precision_macro"] = precision_score(
                y_true, y_pred, average="macro", zero_division=self.zero_division
            )
        except Exception:
            metrics["precision_macro"] = np.nan

        try:
            metrics["recall_macro"] = recall_score(
                y_true, y_pred, average="macro", zero_division=self.zero_division
            )
        except Exception:
            metrics["recall_macro"] = np.nan

        try:
            metrics["f1_macro"] = f1_score(
                y_true, y_pred, average="macro", zero_division=self.zero_division
            )
        except Exception:
            metrics["f1_macro"] = np.nan

        return metrics

    def _calculate_per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_labels: Optional[List[str]]
    ) -> Dict[str, Union[List, Dict]]:
        """Calculate per-class metrics."""
        metrics = {}

        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=self.zero_division
            )

            metrics["per_class_precision"] = precision.tolist()
            metrics["per_class_recall"] = recall.tolist()
            metrics["per_class_f1"] = f1.tolist()
            metrics["per_class_support"] = support.tolist()

            # If class labels provided, create labeled dictionaries
            if class_labels is not None:
                n_classes = len(precision)
                if len(class_labels) >= n_classes:
                    metrics["per_class_metrics_labeled"] = {
                        class_labels[i]: {
                            "precision": precision[i],
                            "recall": recall[i],
                            "f1": f1[i],
                            "support": support[i],
                        }
                        for i in range(n_classes)
                    }

        except Exception:
            metrics["per_class_precision"] = []
            metrics["per_class_recall"] = []
            metrics["per_class_f1"] = []
            metrics["per_class_support"] = []

        return metrics

    def _calculate_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Union[List, int]]:
        """Calculate confusion matrix and related metrics."""
        metrics = {}

        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()

            # Calculate some summary statistics from confusion matrix
            n_classes = cm.shape[0]
            metrics["n_classes"] = n_classes

            # Diagonal sum (correct predictions)
            metrics["correct_predictions"] = int(np.trace(cm))

            # Total predictions
            metrics["total_predictions"] = int(np.sum(cm))

            # Per-class accuracy (diagonal / row sum)
            class_accuracy = []
            for i in range(n_classes):
                if cm[i, :].sum() > 0:
                    class_accuracy.append(cm[i, i] / cm[i, :].sum())
                else:
                    class_accuracy.append(0.0)
            metrics["per_class_accuracy"] = class_accuracy

        except Exception:
            metrics["confusion_matrix"] = []
            metrics["n_classes"] = 0
            metrics["correct_predictions"] = 0
            metrics["total_predictions"] = 0
            metrics["per_class_accuracy"] = []

        return metrics

    def _calculate_robustness_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, finite_mask: np.ndarray
    ) -> Dict[str, Union[float, int]]:
        """Calculate metrics about data quality and robustness."""
        metrics = {}

        metrics["total_samples"] = len(y_true)
        metrics["finite_samples"] = int(np.sum(finite_mask))
        metrics["finite_sample_rate"] = (
            float(np.sum(finite_mask)) / len(y_true) if len(y_true) > 0 else 0.0
        )

        # Class distribution statistics
        finite_true = y_true[finite_mask]
        finite_pred = y_pred[finite_mask]

        if len(finite_true) > 0:
            unique_true, counts_true = np.unique(finite_true, return_counts=True)
            metrics["n_classes_true"] = len(unique_true)
            metrics["class_distribution_true"] = {
                int(cls): int(count) for cls, count in zip(unique_true, counts_true)
            }
        else:
            metrics["n_classes_true"] = 0
            metrics["class_distribution_true"] = {}

        if len(finite_pred) > 0:
            unique_pred, counts_pred = np.unique(finite_pred, return_counts=True)
            metrics["n_classes_pred"] = len(unique_pred)
            metrics["class_distribution_pred"] = {
                int(cls): int(count) for cls, count in zip(unique_pred, counts_pred)
            }
        else:
            metrics["n_classes_pred"] = 0
            metrics["class_distribution_pred"] = {}

        return metrics

    def _get_nan_metrics(self) -> Dict[str, Union[float, List]]:
        """Return dictionary with all metrics set to NaN for failed calculations."""
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "f1_macro": np.nan,
            "per_class_precision": [],
            "per_class_recall": [],
            "per_class_f1": [],
            "per_class_support": [],
            "confusion_matrix": [],
            "n_classes": 0,
            "correct_predictions": 0,
            "total_predictions": 0,
            "per_class_accuracy": [],
            "total_samples": 0,
            "finite_samples": 0,
            "finite_sample_rate": 0.0,
            "n_classes_true": 0,
            "class_distribution_true": {},
            "n_classes_pred": 0,
            "class_distribution_pred": {},
        }

    def calculate_bootstrap_metrics(
        self,
        y_true_list: List[Union[np.ndarray, pd.Series, List]],
        y_pred_list: List[Union[np.ndarray, pd.Series, List]],
        class_labels: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate bootstrap statistics across multiple prediction sets.

        Args:
            y_true_list: List of true class label arrays
            y_pred_list: List of predicted class label arrays
            class_labels: Optional list of class label names

        Returns:
            Dictionary with metrics as keys and sub-dictionaries with 'mean' and 'std'
        """
        if len(y_true_list) != len(y_pred_list):
            raise ValueError("y_true_list and y_pred_list must have same length")

        if len(y_true_list) == 0:
            return {}

        # Calculate metrics for each bootstrap sample
        all_metrics = []
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            metrics = self.calculate_metrics(y_true, y_pred, class_labels)
            all_metrics.append(metrics)

        # Calculate mean and std for scalar metrics
        bootstrap_stats = {}
        scalar_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "finite_sample_rate",
            "n_classes",
            "correct_predictions",
            "total_predictions",
        ]

        for metric in scalar_metrics:
            values = [
                m.get(metric, np.nan) for m in all_metrics if m.get(metric) is not None
            ]
            if values:
                finite_values = [v for v in values if np.isfinite(v)]
                if finite_values:
                    bootstrap_stats[metric] = {
                        "mean": np.mean(finite_values),
                        "std": np.std(finite_values),
                        "count": len(finite_values),
                    }
                else:
                    bootstrap_stats[metric] = {
                        "mean": np.nan,
                        "std": np.nan,
                        "count": 0,
                    }
            else:
                bootstrap_stats[metric] = {"mean": np.nan, "std": np.nan, "count": 0}

        return bootstrap_stats

    def get_core_metrics(self, full_metrics: Dict) -> Dict[str, float]:
        """Extract core classification metrics for summary reporting.

        Args:
            full_metrics: Full metrics dictionary from calculate_metrics()

        Returns:
            Dictionary with core metrics only
        """
        core_keys = ["accuracy", "precision", "recall", "f1", "finite_sample_rate"]
        return {key: full_metrics.get(key, np.nan) for key in core_keys}

    def format_for_storage(
        self, metrics: Dict, prefix: str = ""
    ) -> Dict[str, Union[float, int]]:
        """Format metrics dictionary for database/CSV storage.

        Args:
            metrics: Metrics dictionary from calculate_metrics()
            prefix: Optional prefix for metric names

        Returns:
            Flattened dictionary suitable for storage
        """
        formatted = {}

        # Store scalar metrics
        scalar_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "n_classes",
            "correct_predictions",
            "total_predictions",
            "finite_sample_rate",
            "total_samples",
            "finite_samples",
        ]

        for key in scalar_metrics:
            if key in metrics:
                formatted[f"{prefix}{key}"] = metrics[key]

        # Store summary of per-class metrics
        if "per_class_precision" in metrics and metrics["per_class_precision"]:
            formatted[f"{prefix}mean_per_class_precision"] = np.mean(
                metrics["per_class_precision"]
            )
            formatted[f"{prefix}min_per_class_precision"] = np.min(
                metrics["per_class_precision"]
            )

        if "per_class_recall" in metrics and metrics["per_class_recall"]:
            formatted[f"{prefix}mean_per_class_recall"] = np.mean(
                metrics["per_class_recall"]
            )
            formatted[f"{prefix}min_per_class_recall"] = np.min(
                metrics["per_class_recall"]
            )

        if "per_class_f1" in metrics and metrics["per_class_f1"]:
            formatted[f"{prefix}mean_per_class_f1"] = np.mean(metrics["per_class_f1"])
            formatted[f"{prefix}min_per_class_f1"] = np.min(metrics["per_class_f1"])

        return formatted
