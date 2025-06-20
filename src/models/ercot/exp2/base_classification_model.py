"""Base Classification Model for Experiment 2.

This module provides the BaseClassificationModel class, which extends BaseExp2Model
to support classification tasks with flexible target transformation strategies.

Key Features:
- Flexible target transformation (manual thresholds, data-driven, K-means clustering)
- Support for binary and multi-class classification
- Incremental complexity progression (sign → threshold → multi-class)
- Classification-specific evaluation metrics
- Trading simulation and profit-based evaluation
- Enhanced temporal feature engineering with cyclical encoding
"""

import json
import os
import pickle
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import holidays
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from src.models.ercot.exp2.base_model import BaseExp2Model


class BaseClassificationModel(BaseExp2Model):
    """Abstract base class for Experiment 2 classification models.

    This class extends BaseExp2Model to support classification tasks with flexible
    target transformation strategies. It maintains all the existing functionality
    while adding classification-specific features and enhanced temporal engineering.
    """

    def __init__(
        self,
        model_type: str,
        output_dir: str,
        settlement_point: str,
        random_state: int = 42,
        feature_scaling: str = "none",
        use_synthetic_data: bool = False,
        use_dart_features: bool = True,
        # Classification-specific parameters
        classification_strategy: str = "sign_only",  # 'sign_only', 'threshold', 'kmeans', 'manual'
        classification_config: Optional[Dict] = None,
    ):
        """Initialize the base classification model.

        Args:
            model_type: Type of model (e.g., 'xgboost_classification')
            output_dir: Directory for saving outputs and artifacts
            settlement_point: Settlement point identifier
            random_state: Random seed for reproducibility
            feature_scaling: Feature scaling method ('none' or 'zscore')
            use_synthetic_data: Whether to inject synthetic data for validation
            use_dart_features: Whether to include DART lag/rolling features
            classification_strategy: Strategy for target transformation
            classification_config: Configuration dictionary for classification strategy
        """
        super().__init__(
            model_type=model_type,
            output_dir=output_dir,
            settlement_point=settlement_point,
            random_state=random_state,
            feature_scaling=feature_scaling,
            use_synthetic_data=use_synthetic_data,
            use_dart_features=use_dart_features,
        )

        self.classification_strategy = classification_strategy
        self.classification_config = classification_config or {}

        # Classification-specific attributes
        self.class_labels = []
        self.class_boundaries = []
        self.target_transformer = None
        self.trading_results = {}

        # Initialize classification strategy
        self._initialize_classification_strategy()

    def _initialize_classification_strategy(self) -> None:
        """Initialize the classification strategy and set up class definitions."""

        if self.classification_strategy == "sign_only":
            # Binary classification: positive vs negative DART
            self.class_labels = ["negative", "positive"]
            self.class_boundaries = [0.0]

        elif self.classification_strategy == "threshold":
            # Binary classification: above/below threshold
            threshold = self.classification_config.get("threshold", 5.0)
            use_absolute = self.classification_config.get("use_absolute", True)

            if use_absolute:
                self.class_labels = ["below_threshold", "above_threshold"]
                self.class_boundaries = [threshold]
            else:
                # Bimodal: separate thresholds for positive and negative
                pos_threshold = self.classification_config.get(
                    "positive_threshold", threshold
                )
                neg_threshold = self.classification_config.get(
                    "negative_threshold", -threshold
                )
                self.class_labels = ["large_negative", "neutral", "large_positive"]
                self.class_boundaries = [neg_threshold, pos_threshold]

        elif self.classification_strategy == "manual":
            # Manual boundaries (like the paper: (-∞,-12), [-12,-5), [-5,5), [5,12), [12,∞))
            boundaries = self.classification_config.get("boundaries", [-12, -5, 5, 12])
            labels = self.classification_config.get(
                "labels", ["very_low", "low", "neutral", "high", "very_high"]
            )
            self.class_boundaries = boundaries
            self.class_labels = labels

        elif self.classification_strategy == "kmeans":
            # Data-driven clustering (will be set after seeing data)
            n_clusters = self.classification_config.get("n_clusters", 3)
            self.n_clusters = n_clusters
            self.class_labels = [f"cluster_{i}" for i in range(n_clusters)]
            # Boundaries will be set after K-means fitting

        else:
            raise ValueError(
                f"Unknown classification strategy: {self.classification_strategy}"
            )

    def load_data(self, dataset) -> None:
        """Load data and apply classification target transformation with enhanced features."""
        # Load data using parent method
        super().load_data(dataset)

        # Apply target transformation
        self._transform_targets()

        # Save transformation details
        self._save_classification_config()

    def _transform_targets(self) -> None:
        """Transform continuous targets to classification labels."""

        if self.classification_strategy == "kmeans":
            # Fit K-means on training data only
            train_targets = self.train_data["dart_slt"].values.reshape(-1, 1)

            kmeans = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
            )
            kmeans.fit(train_targets)

            # Get cluster centers and sort them to create boundaries
            centers = sorted(kmeans.cluster_centers_.flatten())

            # Create boundaries as midpoints between centers
            if len(centers) > 1:
                self.class_boundaries = []
                for i in range(len(centers) - 1):
                    boundary = (centers[i] + centers[i + 1]) / 2
                    self.class_boundaries.append(boundary)

            # Store the fitted kmeans for future use
            self.target_transformer = kmeans

            print(f"K-means clustering results:")
            print(f"  Cluster centers: {centers}")
            print(f"  Class boundaries: {self.class_boundaries}")

        # Apply transformation to both datasets
        self.train_data["target_class"] = self._apply_classification_transform(
            self.train_data["dart_slt"]
        )

        if self.validation_data is not None and len(self.validation_data) > 0:
            self.validation_data["target_class"] = self._apply_classification_transform(
                self.validation_data["dart_slt"]
            )

        # Print class distribution
        self._print_class_distribution()

    def _apply_classification_transform(self, values: pd.Series) -> pd.Series:
        """Apply classification transformation to a series of values."""

        if self.classification_strategy == "sign_only":
            return (values > 0).astype(int)

        elif self.classification_strategy == "threshold":
            threshold = self.class_boundaries[0]
            use_absolute = self.classification_config.get("use_absolute", True)

            if use_absolute:
                return (np.abs(values) > threshold).astype(int)
            else:
                # Bimodal classification
                conditions = [
                    values < self.class_boundaries[0],  # large_negative
                    values > self.class_boundaries[1],  # large_positive
                ]
                return np.select(conditions, [0, 2], default=1)  # neutral = 1

        elif self.classification_strategy in ["manual", "kmeans"]:
            # Multi-class using boundaries
            return pd.cut(
                values,
                bins=[-np.inf] + self.class_boundaries + [np.inf],
                labels=range(len(self.class_labels)),
                include_lowest=True,
            ).astype(int)

        else:
            raise ValueError(
                f"Unknown classification strategy: {self.classification_strategy}"
            )

    def _print_class_distribution(self) -> None:
        """Print class distribution for training and validation sets."""

        print(f"\n📊 Classification Strategy: {self.classification_strategy}")
        print(f"   Class Labels: {self.class_labels}")
        print(f"   Class Boundaries: {self.class_boundaries}")

        # Training set distribution
        train_dist = self.train_data["target_class"].value_counts().sort_index()
        print(f"\n🎯 Training Set Class Distribution:")
        for class_idx, count in train_dist.items():
            class_name = (
                self.class_labels[class_idx]
                if class_idx < len(self.class_labels)
                else f"class_{class_idx}"
            )
            pct = 100 * count / len(self.train_data)
            print(f"   {class_name}: {count:,} samples ({pct:.1f}%)")

        # Validation set distribution (if available)
        if self.validation_data is not None and len(self.validation_data) > 0:
            val_dist = self.validation_data["target_class"].value_counts().sort_index()
            print(f"\n✅ Validation Set Class Distribution:")
            for class_idx, count in val_dist.items():
                class_name = (
                    self.class_labels[class_idx]
                    if class_idx < len(self.class_labels)
                    else f"class_{class_idx}"
                )
                pct = 100 * count / len(self.validation_data)
                print(f"   {class_name}: {count:,} samples ({pct:.1f}%)")

    def _save_classification_config(self) -> None:
        """Save classification configuration and transformation details."""

        config = {
            "classification_strategy": self.classification_strategy,
            "classification_config": self.classification_config,
            "class_labels": self.class_labels,
            "class_boundaries": self.class_boundaries,
            "n_classes": len(self.class_labels),
        }

        # Add strategy-specific details
        if (
            self.classification_strategy == "kmeans"
            and self.target_transformer is not None
        ):
            config[
                "kmeans_centers"
            ] = self.target_transformer.cluster_centers_.flatten().tolist()
            config["kmeans_inertia"] = self.target_transformer.inertia_

        # Save to JSON
        config_path = os.path.join(self.model_dir, "classification_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"💾 Classification configuration saved to: {config_path}")

    def _evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: Optional[pd.DataFrame] = None,
        y_validation: Optional[pd.Series] = None,
        bootstrap_iterations: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate classification model with appropriate metrics."""

        results = {}

        # Get training predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = None
        if hasattr(model, "predict_proba"):
            y_train_proba = model.predict_proba(X_train)

        # Training metrics
        results["train_accuracy"] = accuracy_score(y_train, y_train_pred)
        results["train_precision"] = precision_score(
            y_train, y_train_pred, average="weighted", zero_division=0
        )
        results["train_recall"] = recall_score(
            y_train, y_train_pred, average="weighted", zero_division=0
        )
        results["train_f1"] = f1_score(
            y_train, y_train_pred, average="weighted", zero_division=0
        )

        # Confusion matrix
        results["train_confusion_matrix"] = confusion_matrix(
            y_train, y_train_pred
        ).tolist()

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_train, y_train_pred, average=None, zero_division=0
        )
        results["train_per_class_metrics"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        }

        # Validation metrics (if available)
        if X_validation is not None and y_validation is not None:
            y_val_pred = model.predict(X_validation)
            y_val_proba = None
            if hasattr(model, "predict_proba"):
                y_val_proba = model.predict_proba(X_validation)

            results["validation_accuracy"] = accuracy_score(y_validation, y_val_pred)
            results["validation_precision"] = precision_score(
                y_validation, y_val_pred, average="weighted", zero_division=0
            )
            results["validation_recall"] = recall_score(
                y_validation, y_val_pred, average="weighted", zero_division=0
            )
            results["validation_f1"] = f1_score(
                y_validation, y_val_pred, average="weighted", zero_division=0
            )
            results["validation_confusion_matrix"] = confusion_matrix(
                y_validation, y_val_pred
            ).tolist()

            # Per-class validation metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_validation, y_val_pred, average=None, zero_division=0
            )
            results["validation_per_class_metrics"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "f1": f1.tolist(),
                "support": support.tolist(),
            }

        # Bootstrap evaluation (using parent's bootstrap evaluator but with classification metrics)
        if bootstrap_iterations > 0:
            bootstrap_results = self._evaluate_bootstrap_classification(
                model, X_train, y_train, bootstrap_iterations
            )
            results.update(bootstrap_results)

        return results

    def _evaluate_bootstrap_classification(
        self, model: Any, X: pd.DataFrame, y: pd.Series, iterations: int
    ) -> Dict[str, Any]:
        """Perform bootstrap evaluation for classification."""

        bootstrap_accuracies = []
        bootstrap_precisions = []
        bootstrap_recalls = []
        bootstrap_f1s = []

        for i in range(iterations):
            # Bootstrap sample
            n_samples = len(X)
            bootstrap_indices = np.random.choice(
                n_samples, size=n_samples, replace=True
            )
            X_boot = X.iloc[bootstrap_indices]
            y_boot = y.iloc[bootstrap_indices]

            # Predict on bootstrap sample
            y_pred = model.predict(X_boot)

            # Calculate metrics
            accuracy = accuracy_score(y_boot, y_pred)
            precision = precision_score(
                y_boot, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(y_boot, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_boot, y_pred, average="weighted", zero_division=0)

            bootstrap_accuracies.append(accuracy)
            bootstrap_precisions.append(precision)
            bootstrap_recalls.append(recall)
            bootstrap_f1s.append(f1)

        return {
            "bootstrap_accuracy_mean": np.mean(bootstrap_accuracies),
            "bootstrap_accuracy_std": np.std(bootstrap_accuracies),
            "bootstrap_precision_mean": np.mean(bootstrap_precisions),
            "bootstrap_precision_std": np.std(bootstrap_precisions),
            "bootstrap_recall_mean": np.mean(bootstrap_recalls),
            "bootstrap_recall_std": np.std(bootstrap_recalls),
            "bootstrap_f1_mean": np.mean(bootstrap_f1s),
            "bootstrap_f1_std": np.std(bootstrap_f1s),
        }

    def train_hourly_models(
        self, bootstrap_iterations: int = 5, hours_to_train: Optional[List[int]] = None
    ) -> Dict[int, Dict]:
        """Train classification models for each hour."""

        # Use parent's training logic but with classification targets
        if hours_to_train is None:
            hours_to_train = list(range(1, 25))  # Hours 1-24

        print(f"🎯 Training classification models for hours: {hours_to_train}")

        all_results = {}

        for hour in hours_to_train:
            print(f"\n🕐 Training hour {hour} model...")

            # Filter data for this hour using end_of_hour column (1-24 range)
            # This aligns with business convention: models predict for end-of-delivery hour
            train_hour_data = self.train_data[self.train_data["end_of_hour"] == hour]
            val_hour_data = None
            if self.validation_data is not None:
                val_hour_data = self.validation_data[
                    self.validation_data["end_of_hour"] == hour
                ]

            if len(train_hour_data) == 0:
                print(f"⚠️  No training data for hour {hour}, skipping...")
                continue

            # Prepare features and targets
            X_train = train_hour_data[self.feature_names]
            y_train = train_hour_data["target_class"]

            X_validation = None
            y_validation = None
            if val_hour_data is not None and len(val_hour_data) > 0:
                X_validation = val_hour_data[self.feature_names]
                y_validation = val_hour_data["target_class"]

            # Train model for this hour
            model, results = self._train_model_for_hour(
                X_train, y_train, X_validation, y_validation, bootstrap_iterations, hour
            )

            # Store results
            self.models[hour] = model
            all_results[hour] = results

            print(
                f"✅ Hour {hour} completed - Accuracy: {results.get('train_accuracy', 0):.3f}"
            )

        # Store all results
        self.results = all_results

        # Save results and artifacts
        self._save_results()

        return all_results
