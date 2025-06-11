"""Base Model Module for Experiment 0.

This module provides the BaseExp0Model class, which serves as the foundation for all 
Experiment 0 model implementations. It handles common functionality including:

Key Features:
- Train/test data splitting by year (2024 train, 2025 test)
- Hourly model training (24 separate models)
- Cross-validation with appropriate strategies
- Results storage and retrieval
- Model persistence and loading
- Comprehensive evaluation metrics
- Visualization and plotting capabilities

Model Types Supported:
- Linear Regression
- Random Forest (future)
- Neural Networks (future)
"""

import json
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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.utils import resample

from src.data.ercot.database import DatabaseProcessor


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

        # Create output directories
        self.model_dir = self.output_dir / model_type
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Database for results storage
        self.db_processor = DatabaseProcessor(
            str(self.model_dir / f"{model_type}_results.db")
        )

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
                f"  Bootstrap R²: {hour_results['cv_r2_mean']:.4f} ± {hour_results['cv_r2_std']:.4f}"
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
        """Evaluate model performance using bootstrap resampling + final validation.

        This method performs TWO types of evaluation:
        1. Bootstrap performance estimation: Creates temporary models on bootstrap
           samples of training data to estimate performance variability
        2. Final validation: Evaluates the main model on holdout validation data

        Args:
            model: Trained model (fitted on full training data)
            X_train: Training features (2024 data for this hour)
            y_train: Training target (2024 data for this hour)
            X_validation: Validation features (2025 data for this hour)
            y_validation: Validation target (2025 data for this hour)
            bootstrap_iterations: Number of bootstrap iterations for performance estimation

        Returns:
            Dictionary with evaluation metrics:
            - cv_r2_mean/std: Bootstrap performance estimates (from temporary models)
            - validation_r2/mae/rmse: Final validation metrics (from main model)
        """
        results = {}

        # === PART 1: Bootstrap Performance Estimation ===
        # Creates temporary models to estimate how performance varies with training data
        bootstrap_scores = []

        print(
            f"  Starting bootstrap evaluation with {bootstrap_iterations} iterations on {len(X_train)} training samples"
        )

        for i in range(bootstrap_iterations):
            # Create bootstrap sample (sample with replacement, same size as original)
            X_boot, y_boot = resample(
                X_train,
                y_train,
                n_samples=len(X_train),
                random_state=self.random_state + i,  # Different seed for each iteration
            )

            # Identify out-of-bag (OOB) samples - samples not in bootstrap
            boot_indices = set(X_boot.index)
            oob_indices = [idx for idx in X_train.index if idx not in boot_indices]

            print(
                f"    Bootstrap {i+1}: Bootstrap sample={len(X_boot)}, OOB samples={len(oob_indices)}"
            )

            if len(oob_indices) < 5:  # Need minimum samples for meaningful evaluation
                # Fallback: use a random 20% of original data as test set
                test_size = max(5, int(0.2 * len(X_train)))
                test_indices = np.random.choice(
                    X_train.index, size=test_size, replace=False
                )
                X_test = X_train.loc[test_indices]
                y_test = y_train.loc[test_indices]
                print(
                    f"    Bootstrap {i+1}: Using fallback random test set of {len(X_test)} samples (OOB too small)"
                )
            else:
                # Use out-of-bag samples as test set
                X_test = X_train.loc[oob_indices]
                y_test = y_train.loc[oob_indices]
                print(
                    f"    Bootstrap {i+1}: Using OOB test set of {len(X_test)} samples"
                )

            # Train TEMPORARY model on bootstrap sample and evaluate on test set
            from sklearn.base import clone

            bootstrap_model = clone(model)  # Temporary model for evaluation only
            bootstrap_model.fit(X_boot, y_boot)

            # Evaluate temporary model
            y_pred = bootstrap_model.predict(X_test)
            score = r2_score(y_test, y_pred)
            bootstrap_scores.append(score)

        # Store bootstrap results (performance estimates)
        bootstrap_scores = np.array(bootstrap_scores)
        results.update(
            {
                "cv_r2_scores": bootstrap_scores.tolist(),  # Keep same name for compatibility
                "cv_r2_mean": bootstrap_scores.mean(),
                "cv_r2_std": bootstrap_scores.std(),
                "n_train_samples": len(X_train),
                "bootstrap_iterations": bootstrap_iterations,
            }
        )

        # === PART 2: Final Validation Evaluation ===
        # Evaluates the MAIN model on true holdout data (2025)
        if X_validation is not None and y_validation is not None:
            y_pred = model.predict(X_validation)  # Main model on holdout data
            results.update(
                {
                    "validation_r2": r2_score(y_validation, y_pred),
                    "validation_mae": mean_absolute_error(y_validation, y_pred),
                    "validation_rmse": np.sqrt(
                        mean_squared_error(y_validation, y_pred)
                    ),
                    "n_validation_samples": len(X_validation),
                }
            )

        return results

    def _save_results(self) -> None:
        """Save models and results to disk."""
        # Save models
        models_file = self.model_dir / f"{self.settlement_point}_models.pkl"
        with open(models_file, "wb") as f:
            pickle.dump(self.models, f)

        # Save results as JSON
        results_file = self.model_dir / f"{self.settlement_point}_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save results to database
        results_df = pd.DataFrame.from_dict(self.results, orient="index")
        results_df.index.name = "end_hour"
        results_df.reset_index(inplace=True)
        results_df["settlement_point"] = self.settlement_point
        results_df["model_type"] = self.model_type

        self.db_processor.save_to_database(
            results_df, f"{self.model_type}_{self.settlement_point}_results"
        )

        print(f"Models and results saved to: {self.model_dir}")
