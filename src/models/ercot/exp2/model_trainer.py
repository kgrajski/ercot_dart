"""Model Trainer for Experiment 2.

This module provides the Exp2ModelTrainer class, which serves as the main interface
for training and managing different classification model types for DART price prediction.

Key Features:
- Factory pattern for classification model type selection
- Integration with DartSltExp2Dataset
- Unified interface for all classification model types
- Enhanced temporal feature engineering
- Cross-hour analysis capabilities

Similar to Exp1ModelTrainer pattern, this class orchestrates the classification modeling workflow
while delegating specific model implementation details to specialized classification model classes.
"""

import inspect
import os
import sys
import time
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from src.data.ercot.database import DatabaseProcessor
from src.features.ercot.visualization import COLOR_SEQUENCE
from src.features.ercot.visualization import PROFESSIONAL_COLORS
from src.features.ercot.visualization import SEMANTIC_COLORS
from src.features.ercot.visualization import apply_professional_axis_styling
from src.features.ercot.visualization import get_professional_axis_style
from src.features.ercot.visualization import get_professional_layout
from src.features.utils.utils import inverse_signed_log_transform
from src.models.ercot.exp2.models.xgboost_classification import (
    XGBoostClassificationModel,
)

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class Exp2ModelTrainer:
    """Main trainer class for Experiment 2 classification models.

    This class provides a unified interface for training and managing different
    classification model types, following the same pattern as the Exp1ModelTrainer.

    Supported classification model types:
    - xgboost_classification: XGBoost gradient boosting for classification

    Future implementations:
    - random_forest_classification: Random forest classifier
    - logistic_regression: Logistic regression classifier
    - neural_network_classification: Neural network classifier
    """

    # Available classification model types and their implementations
    MODEL_REGISTRY = {
        "xgboost_classification": XGBoostClassificationModel,
        # Future classification models can be added here:
        # 'random_forest_classification': RandomForestClassificationModel,
        # 'logistic_regression': LogisticRegressionModel,
        # 'neural_network_classification': NeuralNetworkClassificationModel,
    }

    def __init__(
        self, dataset, modeling_dir: str, settlement_point: str, random_state: int = 42
    ):
        """Initialize classification model trainer.

        Args:
            dataset: DartSltExp2Dataset instance with model-ready data
            modeling_dir: Path to the modeling directory; individual model type results will be saved in subdirectories
            settlement_point: Settlement point identifier (e.g., 'LZ_HOUSTON_LZ')
            random_state: Random seed for reproducibility
        """
        # Create modeling directory for training outputs
        self.output_dir = modeling_dir

        self.settlement_point = settlement_point
        self.random_state = random_state

        # This is the dataset that will be golden truth for modeling and analytics.
        self.dataset = dataset

        # Store trained model instances for analytics
        self.trained_models = {}

        # Print dataset info
        print(f"Dataset loaded for {self.settlement_point}")
        print(f"Total samples: {len(dataset)}")
        print(f"Features: {len(dataset.get_feature_names())}")

    def train_model(
        self,
        model_type: str,
        bootstrap_iterations: int = 5,
        hours_to_train: Optional[List[int]] = None,
        feature_scaling: str = None,  # Auto-detect based on model type if None
        # Classification-specific parameters
        classification_strategy: str = "sign_only",
        classification_config: Optional[Dict] = None,
        **model_kwargs,
    ) -> Dict:
        """Train a specific classification model type using bootstrap resampling for evaluation.

        Args:
            model_type: Type of classification model to train (e.g., 'xgboost_classification')
            bootstrap_iterations: Number of bootstrap iterations for model evaluation
            hours_to_train: List of hours to train (1-24). If None, trains all hours.
            feature_scaling: Feature scaling method ('none' or 'zscore').
                           If None, auto-detects: 'none' for tree-based models,
                           'zscore' for linear models
            classification_strategy: Strategy for target transformation ('sign_only', 'threshold', etc.)
            classification_config: Configuration dictionary for classification strategy
            **model_kwargs: Additional parameters for model initialization

        Returns:
            Dictionary with training results

        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in self.MODEL_REGISTRY:
            available_types = list(self.MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Available types: {available_types}"
            )

        # Auto-detect feature scaling if not specified
        if feature_scaling is None:
            if model_type in ["xgboost_classification", "random_forest_classification"]:
                feature_scaling = "none"  # Tree-based models don't need scaling
            else:
                feature_scaling = "zscore"  # Linear models benefit from scaling

        print(f"\nTraining {model_type} model for {self.settlement_point}")
        print(f"Feature scaling: {feature_scaling}")
        print(f"Classification strategy: {classification_strategy}")

        # Create model instance using factory pattern
        model_class = self.MODEL_REGISTRY[model_type]

        # Filter model_kwargs to only include parameters that the model class accepts
        model_signature = inspect.signature(model_class.__init__)
        valid_params = set(model_signature.parameters.keys()) - {"self"}

        # Check if the constructor has **kwargs which can accept additional parameters
        has_kwargs = any(
            param.kind == param.VAR_KEYWORD
            for param in model_signature.parameters.values()
        )

        if has_kwargs:
            # If constructor has **kwargs, pass all parameters
            filtered_kwargs = model_kwargs.copy()
            print(
                f"  Constructor accepts **kwargs - passing all parameters: {sorted(filtered_kwargs.keys())}"
            )
        else:
            # Only filter if no **kwargs (strict parameter checking)
            filtered_kwargs = {
                k: v for k, v in model_kwargs.items() if k in valid_params
            }

            # Log filtered parameters for transparency
            if len(model_kwargs) != len(filtered_kwargs):
                filtered_out = set(model_kwargs.keys()) - set(filtered_kwargs.keys())
                print(f"  Filtered out unsupported parameters: {sorted(filtered_out)}")
                print(f"  Using supported parameters: {sorted(filtered_kwargs.keys())}")

        model = model_class(
            output_dir=str(self.output_dir),
            settlement_point=self.settlement_point,
            random_state=self.random_state,
            feature_scaling=feature_scaling,
            classification_strategy=classification_strategy,
            classification_config=classification_config,
            **filtered_kwargs,
        )

        # Load data into model (ETF features already included in dataset)
        model.load_data(self.dataset)

        # Train hourly models
        results = model.train_hourly_models(
            bootstrap_iterations=bootstrap_iterations, hours_to_train=hours_to_train
        )

        print(f"\n{model_type.title()} training completed!")

        # Store the model instance for analytics
        self.trained_models[model_type] = model

        return results

    def run_experiment(
        self,
        model_types: List[str] = None,
        bootstrap_iterations: int = 10,  # Development default - consider 100+ for production
        hours_to_train: Optional[List[int]] = None,
        **experiment_kwargs,  # Pass through additional parameters to all models
    ) -> Dict[str, Dict]:
        """Run complete experiment with multiple classification model types.

        Args:
            model_types: List of model types to train. If None, uses ['xgboost_classification']
            bootstrap_iterations: Number of bootstrap iterations for model evaluation
                                 - Development: 10-50 (fast iteration)
                                 - Testing: 100-500 (more robust)
                                 - Production: 1000+ (publication quality)
            hours_to_train: List of hours to train (1-24). If None, trains all hours.
            **experiment_kwargs: Additional parameters passed to all models
                                Each model will automatically filter for its supported parameters

        Returns:
            Dictionary with results for each model type
        """
        if model_types is None:
            model_types = ["xgboost_classification"]

        print(f"** Running experiment for {self.settlement_point}")
        print(f"** Model types: {model_types}")

        all_results = {}
        for model_type in model_types:
            print(f"** Training {model_type} model for {self.settlement_point}")

            try:
                # Train the model (24 hourly models with bootstrap resampling)
                # Each model class will automatically filter experiment_kwargs for supported parameters
                results = self.train_model(
                    model_type=model_type,
                    bootstrap_iterations=bootstrap_iterations,
                    hours_to_train=hours_to_train,
                    **experiment_kwargs,  # Pass all parameters - models will filter appropriately
                )
                all_results[model_type] = results

            except Exception as e:
                print(f"ERROR training {model_type}: {e}")
                all_results[model_type] = None

        # Print summary
        print(f"\n** Experiment Summary for {self.settlement_point}:")
        for model_type, results in all_results.items():
            if results:
                # Calculate average accuracy across hours for classification
                avg_accuracy = sum(
                    r.get("train_accuracy", 0) for r in results.values()
                ) / len(results)
                print(
                    f"  {model_type}: {len(results)} hourly models, avg Train Accuracy = {avg_accuracy:.4f}"
                )
            else:
                print(f"  {model_type}: FAILED")

        # Create consolidated model_output dataset
        self._create_model_output_dataset(all_results)

        return all_results

    def _create_model_output_dataset(self, all_results: Dict[str, Dict]) -> None:
        """Create consolidated model_output.csv and model_output.db from live model predictions."""
        try:
            print(f"\n📊 Creating consolidated model_output dataset from live data...")

            # Collect all predictions from trained models' live data
            all_predictions = []
            successful_models = [
                model_type for model_type, results in all_results.items() if results
            ]

            for model_type in successful_models:
                if model_type in self.trained_models:
                    model = self.trained_models[model_type]

                    # Extract predictions from model's live predictions dict
                    if hasattr(model, "predictions") and model.predictions:
                        prediction_data = []

                        for end_hour, predictions in model.predictions.items():
                            for (
                                dataset_type,
                                dataset_predictions,
                            ) in predictions.items():
                                for prediction in dataset_predictions:
                                    prediction_data.append(
                                        {
                                            "utc_ts": prediction[
                                                "utc_ts"
                                            ],  # Use UTC timestamp for internal operations
                                            "local_ts": prediction[
                                                "local_ts"
                                            ],  # Include local_ts as metadata too
                                            "end_hour": end_hour,
                                            "dataset_type": dataset_type,
                                            "actual_dart_slt": prediction[
                                                "actual_dart_slt"
                                            ],
                                            "predicted_dart_slt": prediction[
                                                f"pred_{model._get_model_abbreviation()}"
                                            ],
                                            "settlement_point": self.settlement_point,
                                            "model_type": model.model_type,
                                        }
                                    )

                        if prediction_data:
                            model_predictions = pd.DataFrame(prediction_data)
                            print(
                                f"  📁 Extracted {len(model_predictions)} predictions from live {model_type}"
                            )
                            all_predictions.append(model_predictions)
                    else:
                        print(f"  ⚠️  No live predictions found for {model_type}")
                else:
                    print(f"  ⚠️  No trained model found for {model_type}")

            if all_predictions:
                # Concatenate all predictions
                consolidated_df = pd.concat(all_predictions, ignore_index=True)

                # CRITICAL: Sort by timestamp to ensure proper time series plotting
                # Keep timezone-naive as established by DartSltExp1Dataset - do NOT modify timezone!
                consolidated_df["utc_ts"] = pd.to_datetime(consolidated_df["utc_ts"])
                consolidated_df = consolidated_df.sort_values(
                    ["utc_ts", "model_type"]
                ).reset_index(drop=True)

                # Add inverse signed log transformed DART prices for business use
                # These are the actual $/MWh values that can be compared with scientific literature
                consolidated_df["actual_dart_price"] = inverse_signed_log_transform(
                    consolidated_df["actual_dart_slt"]
                )
                consolidated_df["predicted_dart_price"] = inverse_signed_log_transform(
                    consolidated_df["predicted_dart_slt"]
                )
                consolidated_df["prediction_error_price"] = (
                    consolidated_df["predicted_dart_price"]
                    - consolidated_df["actual_dart_price"]
                )

                # Save consolidated model_output.csv
                output_csv = Path(self.output_dir) / "model_output.csv"
                consolidated_df.to_csv(output_csv, index=False)
                print(f"  ✅ Saved consolidated CSV: {output_csv}")

                # Save consolidated model_output.db
                db_path = Path(self.output_dir) / "model_output.db"
                db_processor = DatabaseProcessor(str(db_path))
                db_processor.save_to_database(consolidated_df, "model_output")
                print(f"  ✅ Saved consolidated DB: {db_path}")

                print(
                    f"📊 Model output dataset: {len(consolidated_df):,} total predictions from {len(successful_models)} models"
                )
                print(f"📊 Columns: {list(consolidated_df.columns)}")

            else:
                print("⚠️  No live predictions found to consolidate")

        except Exception as e:
            print(f"❌ Model output dataset creation failed: {e}")
