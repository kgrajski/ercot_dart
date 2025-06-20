"""Base Model Module for Experiment 2.

This module provides the BaseExp2Model class, which serves as the foundation for all 
Experiment 2 model implementations. It handles common functionality including:

Key Features:
- Train/test data splitting by year (2024 train, 2025 test)
- Hourly model training (24 separate models)
- Bootstrap resampling for performance evaluation with comprehensive metrics
- Results storage and retrieval
- Model persistence and loading
- Comprehensive evaluation metrics including classification-specific measures
- Visualization and plotting capabilities

Model Types Supported:
- XGBoost Classification
- Random Forest Classification (future)
- Logistic Regression (future)
- Neural Network Classification (future)
"""

import json
import os
import pickle
from abc import ABC
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from src.data.ercot.database import DatabaseProcessor
from src.models.ercot.exp2.bootstrap_evaluator import BootstrapEvaluator
from src.models.ercot.exp2.evaluation_metrics import EvaluationMetrics


class BaseExp2Model(ABC):
    """Abstract base class for Experiment 2 models.

    This class provides common functionality for DART price classification models,
    including data loading, feature scaling, bootstrap evaluation, and results storage.
    Each hour gets its own separate model to capture hour-specific patterns.

    TODO: CODEBASE MODERNIZATION
    - Add type hints throughout entire codebase (typing module)
    - Standardize docstring format (Google/NumPy style)
    - Add mypy configuration and CI integration
    - Update all classes to use consistent patterns
    - Consider dataclasses for configuration objects
    Target: Complete before any major refactoring or team expansion
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
    ):
        """Initialize the base model.

        Args:
            model_type: Type of model (e.g., 'xgboost_classification')
            output_dir: Directory for saving outputs and artifacts
            settlement_point: Settlement point identifier (e.g., 'LZ_HOUSTON_LZ')
            random_state: Random seed for reproducibility
            feature_scaling: Feature scaling method ('none' or 'zscore')
            use_synthetic_data: Whether to inject synthetic data for validation
            use_dart_features: Whether to include DART lag/rolling features (default True)
        """
        self.model_type = model_type
        self.output_dir = output_dir
        self.settlement_point = settlement_point
        self.random_state = random_state
        self.feature_scaling = feature_scaling
        self.use_synthetic_data = use_synthetic_data
        self.use_dart_features = use_dart_features

        # Model components
        self.models = {}
        self.results = {}
        self.predictions = {}  # Store predictions for Analytics Workbench
        self.train_data = None
        self.validation_data = None
        self.feature_names = []

        # Feature scaling components
        self.scalers = {}  # Feature name -> scaler object
        self.scaling_stats = {}  # Feature name -> {mean, std, etc.}

        # Initialize comprehensive metrics evaluator and bootstrap evaluator
        self.evaluator = EvaluationMetrics(average="weighted", zero_division=0)
        self.bootstrap_evaluator = BootstrapEvaluator(self.evaluator, random_state)

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.model_dir = os.path.join(output_dir, model_type)
        os.makedirs(self.model_dir, exist_ok=True)

        # Database for results storage
        self.results_db_path = os.path.join(self.model_dir, "results.db")
        self.db_processor = DatabaseProcessor(self.results_db_path)

    def load_data(self, dataset) -> None:
        """Load and split data from DartSltExp2Dataset with optional feature filtering and scaling.

        Args:
            dataset: DartSltExp2Dataset instance with model-ready data
        """
        # Get all available features from dataset
        all_features = dataset.get_feature_names()

        # Filter features based on use_dart_features parameter
        if self.use_dart_features:
            # Use all features (original behavior)
            self.feature_names = all_features
            print(f"Using ALL features including DART lag/rolling features")
        else:
            # Exclude DART lag and rolling features - use only load/wind/solar/time features
            excluded_patterns = ["dart_slt_lag_", "dart_slt_roll_"]
            filtered_features = [
                feat
                for feat in all_features
                if not any(pattern in feat for pattern in excluded_patterns)
            ]
            self.feature_names = filtered_features

            # Report filtering results
            dart_features = [
                f for f in all_features if any(p in f for p in excluded_patterns)
            ]
            print(f"🔥 EXCLUDED DART FEATURES: {len(dart_features)} features")
            print(f"   Excluded: {dart_features}")
            print(f"✅ REMAINING FEATURES: {len(filtered_features)} features")
            print(f"   Load/Wind/Solar/Time features only")

        self.original_feature_names = self.feature_names.copy()

        # Split data by year: 2024 for train/CV, 2025 for final validation
        df = dataset.df.copy()
        df["year"] = df["utc_ts"].dt.year

        self.train_data = df[df["year"] == 2024].copy()
        self.validation_data = df[df["year"] == 2025].copy()

        print(f"Training data: {len(self.train_data)} samples (2024)")
        print(f"Validation data: {len(self.validation_data)} samples (2025)")
        print(f"Final feature count: {len(self.feature_names)} variables")

        # Inject synthetic data if requested (for validation purposes)
        if self.use_synthetic_data:
            print(f"🧪 INJECTING SYNTHETIC DATA for validation")
            self._inject_synthetic_data()

        # Apply feature scaling if requested
        if self.feature_scaling == "zscore":
            print(f"Applying z-score transformation using training data statistics...")
            self._apply_zscore_transformation()
            self._save_scaling_stats()

    def _apply_zscore_transformation(self) -> None:
        """Apply z-score transformation using training data statistics only.

        Computes mean and std from training data, applies to both train and validation sets.
        Updates feature names to use _z suffix and validates no conflicts exist.
        """
        # Validate that z-scored features don't already exist
        z_features = [f"{name}_z" for name in self.original_feature_names]
        existing_z_features = [
            col for col in self.train_data.columns if col in z_features
        ]
        if existing_z_features:
            raise ValueError(
                f"Z-scored features already exist in dataset: {existing_z_features}"
            )

        # Calculate scaling statistics on training data only
        scaling_stats = {}
        for feature in self.original_feature_names:
            mean_val = self.train_data[feature].mean()
            std_val = self.train_data[feature].std()

            if std_val == 0:
                print(
                    f"WARNING: Feature '{feature}' has zero standard deviation, using 1.0 to avoid division by zero"
                )
                std_val = 1.0

            scaling_stats[feature] = {"mean": mean_val, "std": std_val}

            # Apply transformation to both datasets
            z_feature_name = f"{feature}_z"
            self.train_data[z_feature_name] = (
                self.train_data[feature] - mean_val
            ) / std_val

            if len(self.validation_data) > 0:
                self.validation_data[z_feature_name] = (
                    self.validation_data[feature] - mean_val
                ) / std_val

        # Update feature names to use z-scored versions
        self.feature_names = z_features
        self.scaling_stats = scaling_stats

        print(
            f"Applied z-score transformation to {len(self.original_feature_names)} features"
        )

    def _save_scaling_stats(self) -> None:
        """Save scaling statistics to CSV for reproducibility."""
        if self.scaling_stats is None:
            return

        # Convert to DataFrame format similar to exp_dataset.py pattern
        stats_rows = []
        for feature, stats in self.scaling_stats.items():
            stats_rows.append(
                {
                    "location": self.settlement_point,
                    "feature": feature,
                    "mean": stats["mean"],
                    "std": stats["std"],
                }
            )

        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(self.model_dir, "feature_scaling_stats.csv")
        stats_df.to_csv(stats_path, index=False)

        print(f"Saved feature scaling statistics: {stats_path}")
        print(f"Statistics for {len(stats_rows)} features saved")

    def _inject_synthetic_data(self) -> None:
        """Inject synthetic data with known coefficient patterns for validation.

        Creates synthetic feature values with simple, obvious relationships that
        should be easily visible in the results files:
        - First 5 features: coefficients [1.0, 2.0, 3.0, -1.0, -2.0]
        - Rest: coefficient 0.0 (should be zero/ignored)
        - Small amount of noise for realism

        This allows visual verification that:
        - Linear regression recovers the true coefficients
        - Ridge shrinks coefficients toward zero
        - Lasso zeros out irrelevant features

        IMPORTANT: This method REPLACES the feature values and target values
        in self.train_data and self.validation_data with synthetic data while
        preserving the original dataframe structure (timestamps, metadata, etc.).
        """
        np.random.seed(self.random_state)

        # Define simple, obvious coefficient pattern
        n_features = len(self.feature_names)
        true_coefficients = np.zeros(n_features)

        # Set first 5 features to have obvious, memorable coefficients
        if n_features >= 5:
            true_coefficients[0] = 1.0  # Feature 0: coefficient = 1.0
            true_coefficients[1] = 2.0  # Feature 1: coefficient = 2.0
            true_coefficients[2] = 3.0  # Feature 2: coefficient = 3.0
            true_coefficients[3] = -1.0  # Feature 3: coefficient = -1.0
            true_coefficients[4] = -2.0  # Feature 4: coefficient = -2.0
            # All other features have coefficient = 0.0 (noise features)

        print(f"True coefficients: {true_coefficients[:10]}... (showing first 10)")

        # REPLACE REAL DATA: Generate synthetic data for training set
        # This modifies self.train_data in-place, replacing feature and target values
        self._generate_synthetic_features_and_target(
            self.train_data, true_coefficients, "Training"
        )

        # REPLACE REAL DATA: Generate synthetic data for validation set (if exists)
        # This modifies self.validation_data in-place, replacing feature and target values
        if len(self.validation_data) > 0:
            self._generate_synthetic_features_and_target(
                self.validation_data, true_coefficients, "Validation"
            )

        # Store true coefficients for later comparison
        self.true_coefficients = true_coefficients

    def _generate_synthetic_features_and_target(
        self, data_df: pd.DataFrame, true_coeffs: np.ndarray, dataset_name: str
    ) -> None:
        """Generate synthetic features and target for a specific dataset.

        CRITICAL: This method modifies the input dataframe IN-PLACE, completely
        replacing the feature values and target values with synthetic data.
        The dataframe structure (timestamps, metadata) is preserved.

        Args:
            data_df: The dataframe to modify (self.train_data or self.validation_data)
            true_coeffs: Known true coefficients for synthetic relationship
            dataset_name: Name for logging (e.g., "Training", "Validation")
        """
        n_samples = len(data_df)
        n_features = len(self.feature_names)

        # Generate synthetic features (standard normal for simplicity)
        synthetic_features = np.random.randn(n_samples, n_features)

        # Generate target with known relationship + small noise
        synthetic_target = synthetic_features @ true_coeffs + 0.1 * np.random.randn(
            n_samples
        )

        # REPLACE REAL FEATURE VALUES: Overwrite each feature column with synthetic data
        for i, feature_name in enumerate(self.feature_names):
            data_df[feature_name] = synthetic_features[:, i]  # IN-PLACE REPLACEMENT

        # REPLACE REAL TARGET VALUES: Overwrite target column with synthetic data
        data_df["dart_slt"] = synthetic_target  # IN-PLACE REPLACEMENT

        print(
            f"  {dataset_name} set: REPLACED {n_samples} samples with {n_features} synthetic features"
        )
        print(
            f"  Synthetic target range: [{synthetic_target.min():.2f}, {synthetic_target.max():.2f}]"
        )

    def train_hourly_models(
        self, bootstrap_iterations: int = 5, hours_to_train: Optional[List[int]] = None
    ) -> Dict[int, Dict]:
        """Train separate models for each hour using bootstrap resampling for evaluation.

        Args:
            bootstrap_iterations: Number of bootstrap iterations for model evaluation
            hours_to_train: List of hours to train (1-24). If None, trains all hours.

        Returns:
            Dictionary with results for each hour
        """
        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        hours_to_train = hours_to_train or list(range(1, 25))
        print(f"Training {self.model_type} models for hours: {hours_to_train}")

        for end_hour in hours_to_train:
            print(f"\nTraining model for hour {end_hour}")

            # Filter data for this hour
            hour_train = self.train_data[
                self.train_data["end_of_hour"] == end_hour
            ].copy()
            hour_validation = self.validation_data[
                self.validation_data["end_of_hour"] == end_hour
            ].copy()

            # Report sample counts for this hour
            print(
                f"  Hour {end_hour} samples - Training: {len(hour_train)}, Validation: {len(hour_validation)}"
            )

            if len(hour_train) == 0:
                print(f"No training data for hour {end_hour}, skipping")
                continue

            # Prepare features and target
            X_train = hour_train[self.feature_names]
            y_train = hour_train["dart_slt"]
            X_validation = (
                hour_validation[self.feature_names]
                if len(hour_validation) > 0
                else None
            )
            y_validation = (
                hour_validation["dart_slt"] if len(hour_validation) > 0 else None
            )

            print(
                f"  Hour {end_hour} feature matrix shapes - X_train: {X_train.shape}, X_validation: {X_validation.shape if X_validation is not None else 'None'}"
            )

            # Train model for this hour
            model, hour_results = self._train_model_for_hour(
                X_train,
                y_train,
                X_validation,
                y_validation,
                bootstrap_iterations,
                end_hour,
            )

            # Store results
            self.models[end_hour] = model
            self.results[end_hour] = hour_results

            # Capture predictions for Analytics Workbench
            self._capture_predictions(
                model, X_train, y_train, X_validation, y_validation, end_hour
            )

            print(
                f"  Bootstrap R²: {hour_results['bootstrap_r2_mean']:.4f} ± {hour_results['bootstrap_r2_std']:.4f}"
            )
            if hour_results.get("validation_r2") is not None:
                print(f"  Validation R²: {hour_results['validation_r2']:.4f}")

        # Save all results
        self._save_results()
        return self.results

    @abstractmethod
    def _train_model_for_hour(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: Optional[pd.DataFrame],
        y_validation: Optional[pd.Series],
        bootstrap_iterations: int,
        hour: int,
    ) -> Tuple[Any, Dict]:
        """Train a model for a specific hour. Must be implemented by subclasses.

        Args:
            X_train: Training features (2024 data for this hour)
            y_train: Training target (2024 data for this hour)
            X_validation: Validation features (2025 data for this hour, may be None)
            y_validation: Validation target (2025 data for this hour, may be None)
            bootstrap_iterations: Number of bootstrap iterations (applied to X_train/y_train only)
            hour: Hour being trained (1-24)

        Returns:
            Tuple of (trained_model, results_dict)
        """
        pass

    def _evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: Optional[pd.DataFrame] = None,
        y_validation: Optional[pd.Series] = None,
        bootstrap_iterations: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate model performance using bootstrap resampling + final validation with comprehensive metrics.

        This method delegates to the BootstrapEvaluator for clean separation of concerns.
        All bootstrap and validation evaluation logic is handled by the specialized evaluator.

        CRITICAL EVALUATION WORKFLOW:
        - Bootstrap metrics: Temporary models on bootstrap samples of training data (uncertainty estimation)
        - Validation metrics: MAIN MODEL (trained on ALL training data) evaluated on holdout validation data

        The validation metrics represent the TRUE performance of our deployed model on unseen data.

        Args:
            model: Trained model (ALREADY FITTED ON ALL TRAINING DATA - this is our main/final model)
            X_train: Training features (2024 data for this hour) - only used for bootstrap uncertainty estimation
            y_train: Training target (2024 data for this hour) - only used for bootstrap uncertainty estimation
            X_validation: Validation features (2025 data for this hour) - HOLDOUT DATA for true performance evaluation
            y_validation: Validation target (2025 data for this hour) - HOLDOUT DATA for true performance evaluation
            bootstrap_iterations: Number of bootstrap iterations for performance estimation

        Returns:
            Dictionary with evaluation metrics:
            - bootstrap_*_mean/std: Bootstrap performance estimates (from temporary models on training data)
            - validation_*: FINAL PERFORMANCE METRICS (main model evaluated on holdout validation data)
            - Comprehensive metrics from EvaluationMetrics class
        """
        return self.bootstrap_evaluator.evaluate_model_with_bootstrap(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            bootstrap_iterations=bootstrap_iterations,
        )

    def _save_results(self) -> None:
        """Save models and results to disk."""
        # Save models
        with open(os.path.join(self.model_dir, "models.pk"), "wb") as f:
            pickle.dump(self.models, f)

        # Prepare results for database/CSV (convert list-type fields to CSV-friendly format)
        db_results = {}
        for hour, hour_results in self.results.items():
            # Create a clean copy handling list-type and dict-type fields appropriately
            clean_results = {}
            for k, v in hour_results.items():
                if isinstance(v, list):
                    # Handle specific list types for CSV compatibility
                    if k == "selected_features":
                        clean_results["selected_features_count"] = len(v)
                        clean_results["selected_features_str"] = (
                            ";".join(v) if v else ""
                        )
                    elif k == "zeroed_features":
                        clean_results["zeroed_features_count"] = len(v)
                        # Don't save the actual list - too long for CSV
                    elif k == "coefficients":
                        # Keep coefficients as they're important for analysis
                        clean_results["coefficients_str"] = ";".join(map(str, v))
                    elif k == "feature_names":
                        clean_results["feature_count"] = len(v)
                        # Don't duplicate feature names in CSV
                    elif k == "feature_importances":
                        # Skip feature importance details in CSV (available in pickle)
                        pass
                    elif k in [
                        "per_class_precision",
                        "per_class_recall",
                        "per_class_f1",
                        "per_class_support",
                        "per_class_accuracy",
                    ]:
                        # Store per-class metrics as summary statistics
                        if v:  # Non-empty list
                            clean_results[f"{k}_mean"] = sum(v) / len(v)
                            clean_results[f"{k}_min"] = min(v)
                            clean_results[f"{k}_max"] = max(v)
                        else:
                            clean_results[f"{k}_mean"] = 0.0
                            clean_results[f"{k}_min"] = 0.0
                            clean_results[f"{k}_max"] = 0.0
                    elif k == "confusion_matrix":
                        # Skip confusion matrix in CSV (available in pickle)
                        pass
                    else:
                        # For other lists, just skip to avoid CSV issues
                        pass
                elif isinstance(v, dict):
                    # Handle dictionary fields from classification metrics
                    if k in ["class_distribution_true", "class_distribution_pred"]:
                        # Convert class distribution to summary statistics
                        if v:  # Non-empty dict
                            clean_results[f"{k}_n_classes"] = len(v)
                            clean_results[f"{k}_total_samples"] = sum(v.values())
                            # Convert to JSON string for CSV storage
                            clean_results[f"{k}_json"] = str(v)
                        else:
                            clean_results[f"{k}_n_classes"] = 0
                            clean_results[f"{k}_total_samples"] = 0
                            clean_results[f"{k}_json"] = "{}"
                    elif k == "per_class_metrics_labeled":
                        # Skip labeled metrics in CSV (available in pickle)
                        pass
                    else:
                        # For other dicts, skip to avoid CSV issues
                        pass
                else:
                    # Keep non-list/non-dict fields as-is
                    clean_results[k] = v

            db_results[hour] = clean_results

        # Save results to database (sanitize model_type for SQL table names)
        results_df = pd.DataFrame.from_dict(db_results, orient="index")
        results_df.index.name = "end_hour"
        results_df.reset_index(inplace=True)
        results_df["settlement_point"] = self.settlement_point
        results_df["model_type"] = self.model_type

        # Sanitize model_type for database table name (replace dots and other invalid chars)
        safe_model_type = self.model_type.replace(".", "_").replace("-", "_")
        self.db_processor.save_to_database(results_df, f"results_{safe_model_type}")

        # Save results to CSV with model_type suffix (CSV files can handle dots)
        csv_filename = f"results_{self.model_type}.csv"
        results_df.to_csv(os.path.join(self.model_dir, csv_filename), index=False)

        # Save coefficients in analysis-friendly format
        self._save_coefficients()

        # Save predictions for Analytics Workbench
        self._save_predictions()

        print(f"Models and results saved to: {self.model_dir}")
        print(f"Detailed results in pickle: models.pk")
        print(f"CSV results: {csv_filename}")

    def _save_coefficients(self) -> None:
        """Save model feature importance in analysis-friendly CSV format."""
        if not self.results:
            return

        # Collect all feature importance with feature names
        importance_data = []

        for hour, hour_results in self.results.items():
            # Handle feature importance from classification models (e.g., XGBoost)
            if (
                "feature_importances" in hour_results
                and "feature_names" in hour_results
            ):
                importances = hour_results["feature_importances"]
                feature_names = hour_results["feature_names"]

                for feature, importance in zip(feature_names, importances):
                    importance_data.append(
                        {
                            "end_hour": hour,
                            "feature": feature,
                            "importance": importance,
                            "abs_importance": importance,  # Importances are already non-negative
                            "is_zero": importance == 0,
                            "model_type": self.model_type,
                            "settlement_point": self.settlement_point,
                        }
                    )

            # Handle coefficients from linear models (if any future linear classification models)
            elif "coefficients" in hour_results and "feature_names" in hour_results:
                coefficients = hour_results["coefficients"]
                feature_names = hour_results["feature_names"]

                for feature, coeff in zip(feature_names, coefficients):
                    importance_data.append(
                        {
                            "end_hour": hour,
                            "feature": feature,
                            "importance": abs(
                                coeff
                            ),  # Use absolute value as importance
                            "abs_importance": abs(coeff),
                            "is_zero": coeff == 0,
                            "model_type": self.model_type,
                            "settlement_point": self.settlement_point,
                        }
                    )

        if importance_data:
            # Create feature importance DataFrame
            importance_df = pd.DataFrame(importance_data)

            # Save feature importance CSV
            importance_filename = f"feature_importance_{self.model_type}.csv"
            importance_df.to_csv(
                os.path.join(self.model_dir, importance_filename), index=False
            )

            # Also save a summary by feature (averaged across hours)
            feature_summary = (
                importance_df.groupby("feature")
                .agg(
                    {
                        "importance": ["mean", "std", "count"],
                        "abs_importance": ["mean", "max"],
                        "is_zero": "sum",  # How many hours this feature was zero
                    }
                )
                .round(6)
            )

            # Flatten column names
            feature_summary.columns = [
                "_".join(col).strip() for col in feature_summary.columns
            ]
            feature_summary = feature_summary.reset_index()

            # Save feature summary
            summary_filename = f"feature_summary_{self.model_type}.csv"
            feature_summary.to_csv(
                os.path.join(self.model_dir, summary_filename), index=False
            )

            print(f"Feature importance saved: {importance_filename}")
            print(f"Feature summary saved: {summary_filename}")

            # Print some key stats for immediate feedback
            n_features = len(feature_summary)
            n_zero_features = len(feature_summary[feature_summary["is_zero_sum"] > 0])
            print(
                f"Features: {n_features} total, {n_zero_features} had zero importance in some hours"
            )

    def _capture_predictions(
        self, model, X_train, y_train, X_validation, y_validation, end_hour
    ):
        """Capture predictions for Analytics Workbench."""
        # Get model abbreviation for column naming
        model_abbrev = self._get_model_abbreviation()

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_validation_pred = (
            model.predict(X_validation) if X_validation is not None else None
        )

        # Store training predictions with UTC timestamps from original data
        train_hour_data = self.train_data[
            self.train_data["end_of_hour"] == end_hour
        ].copy()
        train_predictions = []

        for i, (idx, row) in enumerate(train_hour_data.iterrows()):
            train_predictions.append(
                {
                    "utc_ts": row[
                        "utc_ts"
                    ],  # Use UTC timestamp for internal operations
                    "local_ts": row["local_ts"],  # Include local_ts as metadata too
                    "end_hour": end_hour,
                    "dataset_type": "train",
                    "actual_dart_slt": y_train.iloc[i],
                    f"pred_{model_abbrev}": y_train_pred[i],
                    "settlement_point": self.settlement_point,
                    "model_type": self.model_type,
                }
            )

        # Store validation predictions if available
        validation_predictions = []
        if X_validation is not None and y_validation is not None:
            validation_hour_data = self.validation_data[
                self.validation_data["end_of_hour"] == end_hour
            ].copy()

            for i, (idx, row) in enumerate(validation_hour_data.iterrows()):
                validation_predictions.append(
                    {
                        "utc_ts": row[
                            "utc_ts"
                        ],  # Use UTC timestamp for internal operations
                        "local_ts": row["local_ts"],  # Include local_ts as metadata too
                        "end_hour": end_hour,
                        "dataset_type": "validation",
                        "actual_dart_slt": y_validation.iloc[i],
                        f"pred_{model_abbrev}": y_validation_pred[i],
                        "settlement_point": self.settlement_point,
                        "model_type": self.model_type,
                    }
                )

        # Store in predictions dictionary
        self.predictions[end_hour] = {
            "train": train_predictions,
            "validation": validation_predictions,
        }

    def _get_model_abbreviation(self):
        """Get abbreviated model name for column naming."""
        # Map common model types to short abbreviations
        abbreviations = {
            "xgboost_classification": "xgbc",
            "random_forest_classification": "rfc",
            "logistic_regression": "logreg",
            "neural_network_classification": "nnc",
        }

        # Check for exact matches first
        if self.model_type in abbreviations:
            return abbreviations[self.model_type]

        # Fall back to extracting base model type
        for base_type, abbrev in abbreviations.items():
            if base_type in self.model_type:
                return abbrev

        # Ultimate fallback
        return self.model_type.replace("_", "")[:6]

    def _save_predictions(self) -> None:
        """Save predictions for Analytics Workbench."""
        if not self.predictions:
            return

        # Collect all predictions with timestamps and model type
        prediction_data = []

        for end_hour, predictions in self.predictions.items():
            for dataset_type, dataset_predictions in predictions.items():
                for prediction in dataset_predictions:
                    prediction_data.append(
                        {
                            "utc_ts": prediction["utc_ts"],
                            "local_ts": prediction[
                                "local_ts"
                            ],  # Include local_ts as metadata
                            "end_hour": end_hour,
                            "dataset_type": dataset_type,
                            "actual_dart_slt": prediction["actual_dart_slt"],
                            "predicted_dart_slt": prediction[
                                f"pred_{self._get_model_abbreviation()}"
                            ],
                            "settlement_point": self.settlement_point,
                            "model_type": self.model_type,
                        }
                    )

        if prediction_data:
            # Create predictions DataFrame
            prediction_df = pd.DataFrame(prediction_data)

            # CRITICAL: Sort by timestamp to ensure proper time series plotting
            # Keep timezone-naive as established by DartSltExp2Dataset - do NOT modify timezone!
            prediction_df["utc_ts"] = pd.to_datetime(prediction_df["utc_ts"])
            prediction_df = prediction_df.sort_values("utc_ts").reset_index(drop=True)

            # Save predictions CSV
            prediction_filename = f"predictions_{self.model_type}.csv"
            prediction_df.to_csv(
                os.path.join(self.model_dir, prediction_filename), index=False
            )

            print(f"Predictions saved: {prediction_filename}")
