"""Base Model Module for Experiment 0.

This module provides the BaseExp0Model class, which serves as the foundation for all 
Experiment 0 model implementations. It handles common functionality including:

Key Features:
- Train/test data splitting by year (2024 train, 2025 test)
- Hourly model training (24 separate models)
- Bootstrap resampling for performance evaluation with comprehensive metrics
- Results storage and retrieval
- Model persistence and loading
- Comprehensive evaluation metrics including Lago et al. electricity-specific measures
- Visualization and plotting capabilities

Model Types Supported:
- Linear Regression
- Random Forest (future)
- Neural Networks (future)
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
from src.models.ercot.exp0.bootstrap_evaluator import BootstrapEvaluator
from src.models.ercot.exp0.evaluation_metrics import EvaluationMetrics


class BaseExp0Model(ABC):
    """Base class for Experiment 0 model implementations.

    This class provides core functionality for training and evaluating models:
    - Data splitting and validation
    - Hourly model training (24 models per settlement point)
    - Cross-validation and evaluation
    - Results storage and visualization
    - Model persistence

    Subclasses implement specific model types by overriding _train_model_for_hour().
    """

    def __init__(
        self,
        model_type: str,
        output_dir: str,
        settlement_point: str,
        random_state: int = 42,
    ):
        """Initialize base model with common parameters.

        Args:
            model_type: Type of model (e.g., 'linear_regression', 'random_forest')
            output_dir: Directory for saving outputs and artifacts
            settlement_point: Settlement point identifier (e.g., 'LZ_HOUSTON_LZ')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.settlement_point = settlement_point
        self.random_state = random_state

        # Initialize storage
        self.models = {}  # Trained models by hour
        self.results = {}  # Evaluation results by hour
        self.feature_names = None
        self.train_data = None
        self.validation_data = None

        # Initialize comprehensive metrics evaluator
        self.evaluator = EvaluationMetrics(time_aware=True, price_threshold=1.0)
        self.bootstrap_evaluator = BootstrapEvaluator(self.evaluator, random_state)

        # Create output directories
        # The output directory is the same as the model_type, so we don't need to include
        # in the subsidiary files, such as for results.
        self.model_dir = Path(os.path.join(self.output_dir, model_type))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Database for results storage
        self.db_processor = DatabaseProcessor(str(self.model_dir / "results.db"))

    def load_data(self, dataset) -> None:
        """Load and split data from DartSltExp0Dataset.

        Args:
            dataset: DartSltExp0Dataset instance with model-ready data
        """
        self.feature_names = dataset.get_feature_names()

        # Split data by year: 2024 for train/CV, 2025 for final validation
        df = dataset.df.copy()
        df["year"] = df["utc_ts"].dt.year

        self.train_data = df[df["year"] == 2024].copy()
        self.validation_data = df[df["year"] == 2025].copy()

        print(f"Training data: {len(self.train_data)} samples (2024)")
        print(f"Validation data: {len(self.validation_data)} samples (2025)")
        print(f"Features: {len(self.feature_names)} variables")

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

        Args:
            model: Trained model (fitted on full training data)
            X_train: Training features (2024 data for this hour)
            y_train: Training target (2024 data for this hour)
            X_validation: Validation features (2025 data for this hour)
            y_validation: Validation target (2025 data for this hour)
            bootstrap_iterations: Number of bootstrap iterations for performance estimation

        Returns:
            Dictionary with evaluation metrics:
            - bootstrap_*_mean/std: Bootstrap performance estimates (from temporary models)
            - validation_*: Final validation metrics (from main model)
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

        # Save results as JSON (can handle lists)
        with open(os.path.join(self.model_dir, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

        # Prepare results for database/CSV (exclude list-type fields)
        db_results = {}
        for hour, hour_results in self.results.items():
            # Create a clean copy excluding list-type fields that can't be in DataFrame columns
            clean_results = {
                k: v for k, v in hour_results.items() if not isinstance(v, list)
            }
            db_results[hour] = clean_results

        # Save results to database
        results_df = pd.DataFrame.from_dict(db_results, orient="index")
        results_df.index.name = "end_hour"
        results_df.reset_index(inplace=True)
        results_df["settlement_point"] = self.settlement_point
        results_df["model_type"] = self.model_type
        self.db_processor.save_to_database(results_df, f"results")

        # Save results to CSV
        results_df.to_csv(os.path.join(self.model_dir, "results.csv"), index=False)

        print(f"Models and results saved to: {self.model_dir}")
