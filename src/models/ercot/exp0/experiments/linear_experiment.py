"""
Linear Experiment for ERCOT DART Modeling

This module implements a complete linear modeling experiment for DART prediction,
including data loading, preprocessing, model training, and evaluation.
"""

import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from src.models.ercot.exp0.experiments.base_experiment import BaseExperiment
from src.models.ercot.exp0.model_adapters import get_model_adapter
from src.models.ercot.exp0.utils.data_utils import align_features_targets
from src.models.ercot.exp0.utils.data_utils import create_operational_features
from src.models.ercot.exp0.utils.data_utils import create_train_test_split


class LinearExperiment(BaseExperiment):
    """
    Experiment for testing linear baseline models.

    Tests:
    - Different linear model types (linear, ridge, lasso, elasticnet)
    - Hour-specific vs. global models
    - Different feature sets
    - Hyperparameter optimization
    """

    def __init__(
        self,
        experiment_name: str = "linear_baseline",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize linear experiment.

        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
        """
        default_config = {
            "model_types": ["linear", "ridge", "lasso", "elasticnet"],
            "hour_specific_tests": [False, True],
            "target_column": "dart",
            "prediction_horizon": 24,
            "test_size": 0.2,
            "validation_size": 0.2,
            "random_state": 42,
        }

        if config:
            default_config.update(config)

        super().__init__(experiment_name, default_config)

    def run(self) -> Dict[str, Any]:
        """
        Run the linear baseline experiment.

        Returns:
            Dictionary with experiment results
        """
        self.start_experiment()

        try:
            # Step 1: Load and prepare data
            print("\n1. Loading and preparing data...")
            data = self._load_data()

            # Step 2: Create features
            print("\n2. Creating features...")
            features_df = create_operational_features(
                data, self.config["prediction_horizon"]
            )

            # Step 3: Split data
            print("\n3. Splitting data...")
            train_df, val_df, test_df = create_train_test_split(
                features_df,
                test_size=self.config["test_size"],
                validation_size=self.config["validation_size"],
            )

            # Step 4: Align features and targets
            print("\n4. Aligning features and targets...")
            X_train, y_train = align_features_targets(
                train_df,
                target_column=self.config["target_column"],
                prediction_horizon=self.config["prediction_horizon"],
            )
            X_val, y_val = align_features_targets(
                val_df,
                target_column=self.config["target_column"],
                prediction_horizon=self.config["prediction_horizon"],
            )
            X_test, y_test = align_features_targets(
                test_df,
                target_column=self.config["target_column"],
                prediction_horizon=self.config["prediction_horizon"],
            )

            print(f"Training set: {len(X_train)} samples, {X_train.shape[1]} features")
            print(f"Validation set: {len(X_val)} samples")
            print(f"Test set: {len(X_test)} samples")

            # Step 5: Train and evaluate models
            print("\n5. Training and evaluating models...")
            self._train_models(X_train, y_train, X_val, y_val, X_test, y_test)

            # Step 6: Model comparison
            print("\n6. Comparing models...")
            comparison = self.compare_models("rmse")
            self.log_result("model_comparison", comparison.to_dict())

            # Step 7: Best model analysis
            best_model = self.get_best_model("rmse")
            if best_model:
                print("\n7. Analyzing best model...")
                self._analyze_best_model(best_model, X_test, y_test)

        except Exception as e:
            print(f"Experiment failed: {e}")
            self.log_result("error", str(e))
            raise
        finally:
            self.end_experiment()

        return self.results

    def _load_data(self) -> pd.DataFrame:
        """Load DART data for the experiment"""
        # TODO: Implement actual data loading
        # For now, create placeholder
        print("TODO: Implement actual data loading from processed files")
        print("This will connect to your existing data processing pipeline")

        # Placeholder - replace with actual data loading
        raise NotImplementedError(
            "Implement data loading from your processed DART files"
        )

    def _train_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train all model configurations"""

        for model_type in self.config["model_types"]:
            for hour_specific in self.config["hour_specific_tests"]:
                model_name = (
                    f"{model_type}_{'hour_specific' if hour_specific else 'global'}"
                )
                print(f"\nTraining {model_name}...")

                # Create model adapter
                model_adapter = get_model_adapter(
                    "linear",
                    model_type=model_type,
                    hour_specific=hour_specific,
                    standardize=True,
                )

                # Train model
                training_metrics = model_adapter.fit(X_train, y_train, X_val, y_val)

                # Evaluate on test set
                test_metrics = model_adapter.evaluate(X_test, y_test)

                # Combine metrics
                all_metrics = {
                    **training_metrics,
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }

                # Log model
                self.log_model(model_name, model_adapter, all_metrics)

    def _analyze_best_model(self, best_model, X_test, y_test):
        """Analyze the best performing model"""

        # Feature importance
        try:
            importance = best_model.get_feature_importance()
            self.log_result("feature_importance", importance)

            # Top 10 most important features
            sorted_features = sorted(
                importance.items(), key=lambda x: abs(x[1]), reverse=True
            )
            top_features = dict(sorted_features[:10])
            self.log_result("top_10_features", top_features)

            print("\nTop 10 most important features:")
            for feature, coef in top_features.items():
                print(f"  {feature}: {coef:.4f}")

        except Exception as e:
            print(f"Could not analyze feature importance: {e}")

        # Prediction analysis
        predictions = best_model.predict(X_test)
        residuals = y_test - predictions

        # Log prediction statistics
        self.log_result(
            "prediction_stats",
            {
                "mean_prediction": float(np.mean(predictions)),
                "std_prediction": float(np.std(predictions)),
                "mean_residual": float(np.mean(residuals)),
                "std_residual": float(np.std(residuals)),
            },
        )

        print(f"\nPrediction analysis:")
        print(f"  Mean prediction: {np.mean(predictions):.4f}")
        print(f"  Std prediction: {np.std(predictions):.4f}")
        print(f"  Mean residual: {np.mean(residuals):.4f}")
        print(f"  Std residual: {np.std(residuals):.4f}")
