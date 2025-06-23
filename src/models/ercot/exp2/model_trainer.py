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
from datetime import datetime
from datetime import timedelta
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
from src.models.ercot.exp2.tmp.phase_1a_analysis.get_optimized_xgboost_params import (
    get_optimized_xgboost_params,
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
        save_results: bool = True,  # Control whether to save non-progressive results
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
            save_results: Control whether to save non-progressive results
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
            bootstrap_iterations=bootstrap_iterations,
            hours_to_train=hours_to_train,
            save_results=save_results,
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

        # Apply Phase 1A optimized parameters for XGBoost models
        optimized_experiment_kwargs = experiment_kwargs.copy()
        if "xgboost_classification" in model_types:
            print(f"\n🎯 Loading Phase 1A optimized XGBoost parameters...")
            try:
                optimized_params, early_stopping = get_optimized_xgboost_params()
                if optimized_params:
                    # Extract model parameters (exclude training-specific parameters)
                    xgb_model_params = {
                        k: v
                        for k, v in optimized_params.items()
                        if k
                        not in [
                            "objective",
                            "eval_metric",
                            "random_state",
                            "verbosity",
                            "n_jobs",
                        ]
                    }

                    # Merge with any existing parameters (experiment_kwargs take precedence)
                    for param, value in xgb_model_params.items():
                        if param not in optimized_experiment_kwargs:
                            optimized_experiment_kwargs[param] = value

                    print(
                        f"   ✅ Applied optimized parameters: {list(xgb_model_params.keys())}"
                    )
                    print(
                        f"   Expected improvements: Reduced overfitting, improved generalization"
                    )
                else:
                    print("   ⚠️ Could not load optimized parameters, using defaults")
            except Exception as e:
                print(f"   ⚠️ Error loading optimized parameters: {e}")
                print("   Using default parameters")

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
                    **optimized_experiment_kwargs,  # Pass all parameters - models will filter appropriately
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

    def run_experiment_progressive(
        self,
        model_types: List[str] = None,
        bootstrap_iterations: int = 10,
        hours_to_train: Optional[List[int]] = None,
        num_weeks: Optional[int] = None,  # None = auto-detect all available weeks
        **experiment_kwargs,
    ) -> Dict[str, Dict]:
        """Run progressive validation experiment - clean parallel to run_experiment.

        Args:
            model_types: List of model types to train. If None, uses ['xgboost_classification']
            bootstrap_iterations: Number of bootstrap iterations for model evaluation
            hours_to_train: List of hours to train (1-24). If None, trains all hours.
            num_weeks: Number of weeks to validate progressively. If None, auto-detects all available weeks.
            **experiment_kwargs: Additional parameters passed to all models

        Returns:
            Dictionary with results for each week and model type
        """
        if model_types is None:
            model_types = ["xgboost_classification"]

        print(f"** Running PROGRESSIVE validation for {self.settlement_point}")
        print(f"** Model types: {model_types}")

        # Auto-detect number of weeks if not specified
        if num_weeks is None:
            max_weeks = self._calculate_max_available_weeks()
            print(f"** Weeks: AUTO-DETECTED {max_weeks} available weeks")
        else:
            max_weeks = num_weeks
            print(f"** Weeks: {num_weeks} (user-specified)")

        # Apply Phase 1A optimized parameters for XGBoost models (same as run_experiment)
        optimized_experiment_kwargs = experiment_kwargs.copy()
        if "xgboost_classification" in model_types:
            print(f"\n🎯 Loading Phase 1A optimized XGBoost parameters...")
            try:
                optimized_params, early_stopping = get_optimized_xgboost_params()
                if optimized_params:
                    xgb_model_params = {
                        k: v
                        for k, v in optimized_params.items()
                        if k
                        not in [
                            "objective",
                            "eval_metric",
                            "random_state",
                            "verbosity",
                            "n_jobs",
                        ]
                    }

                    for param, value in xgb_model_params.items():
                        if param not in optimized_experiment_kwargs:
                            optimized_experiment_kwargs[param] = value

                    print(
                        f"   ✅ Applied optimized parameters: {list(xgb_model_params.keys())}"
                    )
                    print(
                        f"   Expected improvements: Reduced overfitting, improved generalization"
                    )
                else:
                    print("   ⚠️ Could not load optimized parameters, using defaults")
            except Exception as e:
                print(f"   ⚠️ Error loading optimized parameters: {e}")
                print("   Using default parameters")

        # Generate week configurations internally
        weeks_config = self._generate_operational_weeks(max_weeks)

        # Update final count after generation (in case some weeks were skipped)
        actual_weeks = len(weeks_config)
        print(f"** Final week count: {actual_weeks} weeks will be processed")

        # Outer loop over weeks
        all_weeks_results = {}
        all_weeks_predictions = []  # Collect predictions from each week immediately

        for week_config in weeks_config:
            week_num = week_config["week_num"]
            week_desc = week_config["week_description"]
            print(f"\n** Week {week_num}: {week_desc}")

            # Temporarily modify dataset for this week's split
            self._apply_week_split(week_config)

            # Inner loop over model types (SAME as run_experiment)
            week_results = {}
            for model_type in model_types:
                print(f"** Training {model_type} for week {week_num}")

                try:
                    # Same clean delegation as run_experiment!
                    results = self.train_model(
                        model_type=model_type,
                        bootstrap_iterations=bootstrap_iterations,
                        hours_to_train=hours_to_train,
                        save_results=False,  # Don't save non-progressive results during progressive validation
                        **optimized_experiment_kwargs,
                    )

                    # Add week metadata to results
                    self._add_week_metadata(results, week_config)
                    week_results[model_type] = results

                    # IMMEDIATELY collect predictions for this week before model gets overwritten
                    if model_type in self.trained_models:
                        week_predictions = self._extract_week_predictions(
                            model_type, week_config
                        )
                        if week_predictions:
                            all_weeks_predictions.extend(week_predictions)
                            print(
                                f"   ✅ Collected {len(week_predictions)} predictions from {model_type}"
                            )

                except Exception as e:
                    print(f"ERROR training {model_type} for week {week_num}: {e}")
                    week_results[model_type] = None

            all_weeks_results[f"week_{week_num}"] = week_results

            # Restore dataset for next week
            self._restore_original_split()

            print(f"✅ Week {week_num} completed")

        # Create consolidated output using collected predictions
        self._create_progressive_output_from_predictions(all_weeks_predictions)

        # Save progressive predictions and analysis outputs for each model type
        for model_type in model_types:
            if model_type in self.trained_models:
                model = self.trained_models[model_type]

                # Filter predictions for this model type
                model_predictions = [
                    pred
                    for pred in all_weeks_predictions
                    if pred.get("model_type") == model.model_type
                ]

                # Call the model's progressive predictions save method
                if hasattr(model, "_save_progressive_predictions"):
                    model._save_progressive_predictions(model_predictions)

                # Collect results from all weeks for this model type
                model_results = []
                for week_key, week_data in all_weeks_results.items():
                    if model_type in week_data and week_data[model_type] is not None:
                        # Add week metadata to the results
                        week_results = week_data[model_type].copy()
                        model_results.append(week_results)

                # Call the model's progressive analysis save methods
                if hasattr(model, "_save_progressive_results"):
                    model._save_progressive_results(model_results)

                if hasattr(model, "_save_progressive_feature_importance"):
                    model._save_progressive_feature_importance(model_results)

                if hasattr(model, "_save_progressive_feature_summary"):
                    model._save_progressive_feature_summary(model_results)

        print(f"\n🎉 Progressive validation completed! {actual_weeks} weeks")
        return all_weeks_results

    def _calculate_max_available_weeks(self) -> int:
        """Calculate the maximum number of weeks available for progressive validation.

        Returns:
            Maximum number of operational weeks that can fit in the available data
        """
        # Auto-detect data range from dataset
        df_dates = pd.to_datetime(self.dataset.df["utc_ts"])
        data_start = df_dates.min()
        data_end = df_dates.max()

        # Find first Sunday in 2025 for operational weeks
        start_2025 = pd.to_datetime("2025-01-01")
        days_ahead = 6 - start_2025.weekday()  # Days until Sunday (weekday 6)
        if days_ahead == 7:  # If already Sunday
            days_ahead = 0
        first_sunday = start_2025 + timedelta(days=days_ahead)

        # Calculate how many complete weeks fit in the available data
        weeks_available = 0
        current_sunday = first_sunday

        while True:
            week_end = current_sunday + timedelta(days=6)  # Saturday
            if week_end > data_end:
                break
            weeks_available += 1
            current_sunday += timedelta(days=7)  # Next Sunday

        print(
            f"   Data span: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}"
        )
        print(
            f"   First validation week starts: {first_sunday.strftime('%Y-%m-%d')} (Sunday)"
        )
        print(f"   Maximum weeks available: {weeks_available}")

        return weeks_available

    def _generate_operational_weeks(self, num_weeks: int) -> List[Dict]:
        """Generate operational week configurations for progressive validation.

        Args:
            num_weeks: Number of weeks to generate

        Returns:
            List of week configurations with train/validation date ranges
        """
        # Auto-detect data range from dataset
        df_dates = pd.to_datetime(self.dataset.df["utc_ts"])
        data_start = df_dates.min()
        data_end = df_dates.max()

        print(
            f"   Data available: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}"
        )

        # Find first Sunday in 2025 for operational weeks
        start_2025 = pd.to_datetime("2025-01-01")
        days_ahead = 6 - start_2025.weekday()  # Days until Sunday (weekday 6)
        if days_ahead == 7:  # If already Sunday
            days_ahead = 0
        first_sunday = start_2025 + timedelta(days=days_ahead)

        weeks_config = []
        current_sunday = first_sunday

        for week_num in range(1, num_weeks + 1):
            # Calculate week boundaries (Sunday to Saturday)
            week_start = current_sunday
            week_end = current_sunday + timedelta(days=6)  # Saturday

            # Check if we have enough data for this week
            if week_end > data_end:
                print(
                    f"   ⚠️ Week {week_num} ends {week_end.strftime('%Y-%m-%d')}, beyond available data"
                )
                break

            # Training data: All data up to end of previous week
            train_end_date = current_sunday - timedelta(days=1)  # Previous Saturday

            # Generate human-readable description
            week_description = (
                f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}"
            )

            week_config = {
                "week_num": week_num,
                "week_description": week_description,
                "train_end_date": train_end_date.strftime("%Y-%m-%d"),
                "val_start_date": week_start.strftime("%Y-%m-%d"),
                "val_end_date": week_end.strftime("%Y-%m-%d"),
            }

            weeks_config.append(week_config)
            current_sunday += timedelta(days=7)  # Next Sunday

        print(f"   Generated {len(weeks_config)} operational weeks")
        return weeks_config

    def _apply_week_split(self, week_config: Dict) -> None:
        """Temporarily modify dataset for a specific week's train/validation split."""
        train_end = pd.to_datetime(week_config["train_end_date"])
        val_start = pd.to_datetime(week_config["val_start_date"])
        val_end = pd.to_datetime(week_config["val_end_date"])

        # Store original year column
        self.dataset.df["original_year"] = self.dataset.df["utc_ts"].dt.year

        # Create temporary split labels
        self.dataset.df["temp_split"] = "ignore"
        self.dataset.df.loc[
            self.dataset.df["utc_ts"] <= train_end, "temp_split"
        ] = "train"
        self.dataset.df.loc[
            (self.dataset.df["utc_ts"] >= val_start)
            & (self.dataset.df["utc_ts"] <= val_end),
            "temp_split",
        ] = "validation"

        # Temporarily replace year column for models to use
        self.dataset.df.loc[self.dataset.df["temp_split"] == "train", "year"] = 2024
        self.dataset.df.loc[
            self.dataset.df["temp_split"] == "validation", "year"
        ] = 2025
        self.dataset.df.loc[
            self.dataset.df["temp_split"] == "ignore", "year"
        ] = 9999  # Ignored

        train_samples = (self.dataset.df["temp_split"] == "train").sum()
        val_samples = (self.dataset.df["temp_split"] == "validation").sum()
        print(
            f"   Train samples: {train_samples:,} (through {week_config['train_end_date']})"
        )
        print(
            f"   Validation samples: {val_samples:,} ({week_config['val_start_date']} to {week_config['val_end_date']})"
        )

    def _restore_original_split(self) -> None:
        """Restore original year-based dataset split."""
        if "original_year" in self.dataset.df.columns:
            self.dataset.df["year"] = self.dataset.df["original_year"]
            self.dataset.df.drop(columns=["temp_split", "original_year"], inplace=True)

    def _add_week_metadata(self, results: Dict, week_config: Dict) -> None:
        """Add week metadata to model results."""
        for hour, hour_results in results.items():
            hour_results.update(
                {
                    "week_num": week_config["week_num"],
                    "week_description": week_config["week_description"],
                    "validation_mode": "progressive",
                    "train_end_date": week_config["train_end_date"],
                    "val_start_date": week_config["val_start_date"],
                    "val_end_date": week_config["val_end_date"],
                }
            )

    def _extract_week_predictions(
        self, model_type: str, week_config: Dict
    ) -> List[Dict]:
        """Extract predictions from the current week's trained model."""
        week_predictions = []

        if model_type not in self.trained_models:
            return week_predictions

        model = self.trained_models[model_type]

        # Extract predictions from model's live predictions dict
        if hasattr(model, "predictions") and model.predictions:
            for end_hour, predictions in model.predictions.items():
                for dataset_type, dataset_predictions in predictions.items():
                    for prediction in dataset_predictions:
                        prediction_record = {
                            "utc_ts": prediction["utc_ts"],
                            "local_ts": prediction["local_ts"],
                            "end_hour": end_hour,
                            "dataset_type": dataset_type,
                            "actual_dart_slt": prediction["actual_dart_slt"],
                            "predicted_dart_slt": prediction[
                                f"pred_{model._get_model_abbreviation()}"
                            ],
                            "settlement_point": self.settlement_point,
                            "model_type": model.model_type,
                            # Add week metadata
                            "week_num": week_config["week_num"],
                            "week_description": week_config["week_description"],
                            "validation_mode": "progressive",
                            "train_end_date": week_config["train_end_date"],
                            "val_start_date": week_config["val_start_date"],
                            "val_end_date": week_config["val_end_date"],
                        }
                        week_predictions.append(prediction_record)

        return week_predictions

    def _create_progressive_output_from_predictions(
        self, all_predictions: List[Dict]
    ) -> None:
        """Create consolidated progressive validation output from collected predictions."""
        try:
            print(
                f"\n📊 Creating progressive validation output from {len(all_predictions)} collected predictions..."
            )

            if not all_predictions:
                print("⚠️  No predictions collected during progressive validation")
                return

            # Create DataFrame from collected predictions
            consolidated_df = pd.DataFrame(all_predictions)

            # CRITICAL: Sort by week and timestamp for proper time series analysis
            consolidated_df["utc_ts"] = pd.to_datetime(consolidated_df["utc_ts"])
            consolidated_df = consolidated_df.sort_values(
                ["week_num", "utc_ts", "model_type"]
            ).reset_index(drop=True)

            # Add inverse signed log transformed DART prices for business use
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

            # Save consolidated progressive validation outputs
            output_csv = Path(self.output_dir) / "model_output_progressive.csv"
            consolidated_df.to_csv(output_csv, index=False)
            print(f"  ✅ Saved progressive CSV: {output_csv}")

            db_path = Path(self.output_dir) / "model_output_progressive.db"
            db_processor = DatabaseProcessor(str(db_path))
            db_processor.save_to_database(consolidated_df, "model_output_progressive")
            print(f"  ✅ Saved progressive DB: {db_path}")

            # Summary statistics
            weeks_in_data = consolidated_df["week_num"].nunique()
            models_in_data = consolidated_df["model_type"].nunique()
            print(
                f"📊 Progressive validation dataset: {len(consolidated_df):,} total predictions"
            )
            print(f"📊 Coverage: {weeks_in_data} weeks × {models_in_data} models")

            # Show week coverage
            week_summary = (
                consolidated_df.groupby(["week_num", "week_description"])
                .size()
                .reset_index(name="predictions")
            )
            print(f"📊 Week-by-week coverage:")
            for _, row in week_summary.iterrows():
                print(
                    f"   Week {row['week_num']}: {row['predictions']:,} predictions ({row['week_description']})"
                )

        except Exception as e:
            print(f"❌ Progressive model output creation failed: {e}")
            import traceback

            traceback.print_exc()
