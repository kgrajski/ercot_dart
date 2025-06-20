"""Model Trainer for Experiment 1.

This module provides the Exp1ModelTrainer class, which serves as the main interface
for training and managing different model types for DART price prediction.

Key Features:
- Factory pattern for model type selection
- Integration with DartSltExp1Dataset
- Unified interface for all model types

Similar to ERCOTBaseClient pattern, this class orchestrates the modeling workflow
while delegating specific model implementation details to specialized model classes.
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
from src.models.ercot.exp1.models.lasso_regression import LassoRegressionModel
from src.models.ercot.exp1.models.linear_regression import LinearRegressionModel
from src.models.ercot.exp1.models.ridge_regression import RidgeRegressionModel
from src.models.ercot.exp1.models.xgboost_regression import XGBoostRegressionModel

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class Exp1ModelTrainer:
    """Main trainer class for Experiment 1 models.

    This class provides a unified interface for training and managing different
    model types, following the same pattern as the ERCOT data clients.

    Supported model types:
    - linear_regression: Standard linear regression
    - ridge_regression: Ridge regression with L2 regularization
    - lasso_regression: Lasso regression with L1 regularization and feature selection
    - xgboost_regression: XGBoost gradient boosting for non-linear relationships
    - random_forest: Random forest (future implementation)
    - neural_network: Neural networks (future implementation)
    """

    # Available model types and their implementations
    MODEL_REGISTRY = {
        "linear_regression": LinearRegressionModel,
        "ridge_regression": RidgeRegressionModel,
        "lasso_regression": LassoRegressionModel,
        "xgboost_regression": XGBoostRegressionModel,
        # Future models can be added here:
        # 'random_forest': RandomForestModel,
        # 'neural_network': NeuralNetworkModel,
    }

    def __init__(
        self, dataset, modeling_dir: str, settlement_point: str, random_state: int = 42
    ):
        """Initialize model trainer.

        Args:
            dataset: DartSltExp1Dataset instance with model-ready data
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
        **model_kwargs,
    ) -> Dict:
        """Train a specific model type using bootstrap resampling for evaluation.

        Args:
            model_type: Type of model to train (e.g., 'linear_regression')
            bootstrap_iterations: Number of bootstrap iterations for model evaluation
            hours_to_train: List of hours to train (1-24). If None, trains all hours.
            feature_scaling: Feature scaling method ('none' or 'zscore').
                           If None, auto-detects: 'none' for linear_regression,
                           'zscore' for ridge/lasso regression
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
            if model_type in ["linear_regression", "xgboost_regression"]:
                feature_scaling = (
                    "none"  # Linear regression and XGBoost don't need scaling
                )
            else:
                feature_scaling = "zscore"  # Ridge/Lasso benefit from scaling

        print(f"\nTraining {model_type} model for {self.settlement_point}")
        print(f"Feature scaling: {feature_scaling}")

        # Create model instance using factory pattern
        model_class = self.MODEL_REGISTRY[model_type]

        # Filter model_kwargs to only include parameters that the model class accepts
        model_signature = inspect.signature(model_class.__init__)
        valid_params = set(model_signature.parameters.keys()) - {
            "self"
        }  # Exclude 'self'

        # Check if the constructor has **kwargs which can accept additional parameters
        has_kwargs = any(
            param.kind == param.VAR_KEYWORD
            for param in model_signature.parameters.values()
        )

        if has_kwargs:
            # If constructor has **kwargs, pass all parameters (they'll be handled by the model)
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
            **filtered_kwargs,
        )

        #
        # Load data into model
        # This is the dataset that will be golden truth for modeling and analytics.
        # The load_data method is in the base_model.py file.
        #
        model.load_data(self.dataset)

        # Train hourly models
        results = model.train_hourly_models(
            bootstrap_iterations=bootstrap_iterations, hours_to_train=hours_to_train
        )

        print(f"\n{model_type.title()} training completed!")
        print(f"Trained models for {len(results)} hours")

        # Store trained model instance for analytics
        self.trained_models[model_type] = model

        return results

    def run_experiment(
        self,
        model_types: List[str] = None,
        bootstrap_iterations: int = 10,  # Development default - consider 100+ for production
        hours_to_train: Optional[List[int]] = None,
        **experiment_kwargs,  # Pass through additional parameters to all models
    ) -> Dict[str, Dict]:
        """Run complete experiment with multiple model types.

        Args:
            model_types: List of model types to train. If None, uses ['linear_regression']
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
            model_types = ["linear_regression"]

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
                avg_bootstrap_r2 = sum(
                    r["bootstrap_r2_mean"] for r in results.values()
                ) / len(results)
                print(
                    f"  {model_type}: {len(results)} hourly models, avg Bootstrap R² = {avg_bootstrap_r2:.4f}"
                )
            else:
                print(f"  {model_type}: FAILED")

        # Generate Analytics Workbench dashboards
        self._generate_analytics_workbench(all_results)

        # Create consolidated model_output dataset
        self._create_model_output_dataset(all_results)

        return all_results

    def _generate_analytics_workbench(self, all_results: Dict[str, Dict]) -> None:
        """Generate Analytics Workbench dashboards using live data from trained models."""
        try:
            print(
                f"\n🎯 Generating Analytics Workbench dashboards using live model data..."
            )

            # Get successful models and their live artifacts
            successful_models = [
                model_type for model_type, results in all_results.items() if results
            ]
            live_artifacts_dict = {}

            for model_type in successful_models:
                try:
                    # Get the trained model instance with live data
                    if model_type in self.trained_models:
                        model = self.trained_models[model_type]

                        # Extract live artifacts from the model's in-memory data
                        live_artifacts = self._extract_live_artifacts(
                            model, all_results[model_type]
                        )
                        live_artifacts_dict[model_type] = live_artifacts

                        print(f"  📁 Extracted live artifacts for {model_type}")
                    else:
                        print(f"  ⚠️  No trained model found for {model_type}")
                except Exception as e:
                    print(
                        f"  ⚠️  Could not extract live artifacts for {model_type}: {e}"
                    )

            # Generate individual model dashboards using trainer's own methods
            for model_type in successful_models:
                if model_type in live_artifacts_dict:
                    try:
                        print(
                            f"  📊 Creating dashboard for {model_type} using live data..."
                        )

                        # Use trainer's own analytics method
                        dashboard_path = self._create_analytics_dashboard(
                            model_artifacts=live_artifacts_dict[model_type],
                            model_name=model_type,
                            output_path=Path(self.output_dir)
                            / f"analytics_workbench_{model_type}.html",
                        )

                        print(f"    ✅ Dashboard saved: {dashboard_path}")

                    except Exception as e:
                        print(f"    ❌ Failed to create dashboard for {model_type}: {e}")

            # Generate comparative dashboard if multiple models using trainer's own method
            if len(live_artifacts_dict) > 1:
                try:
                    print(f"  📊 Creating comparative dashboard using live data...")

                    comparative_path = self._create_comparative_dashboard(
                        model_artifacts_dict=live_artifacts_dict,
                        model_names=list(live_artifacts_dict.keys()),
                        output_path=Path(self.output_dir)
                        / "analytics_workbench_comparative.html",
                    )

                    print(f"    ✅ Comparative dashboard saved: {comparative_path}")

                except Exception as e:
                    print(f"    ❌ Failed to create comparative dashboard: {e}")

            print(f"🎯 Analytics Workbench generation completed using live data!")

        except Exception as e:
            print(f"❌ Analytics Workbench generation failed: {e}")

    def _extract_live_artifacts(self, model, results: Dict) -> Dict:
        """Extract live artifacts from a trained model's in-memory data structures."""
        artifacts = {}

        # Extract predictions from model's live predictions dict
        if hasattr(model, "predictions") and model.predictions:
            # Convert model.predictions dict to DataFrame format
            prediction_data = []

            for end_hour, predictions in model.predictions.items():
                for dataset_type, dataset_predictions in predictions.items():
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
                                "actual_dart_slt": prediction["actual_dart_slt"],
                                "predicted_dart_slt": prediction[
                                    f"pred_{model._get_model_abbreviation()}"
                                ],
                                "settlement_point": self.settlement_point,
                                "model_type": model.model_type,
                            }
                        )

            if prediction_data:
                predictions_df = pd.DataFrame(prediction_data)
                # Keep timezone-naive as established by DartSltExp1Dataset - do NOT modify timezone!
                predictions_df["utc_ts"] = pd.to_datetime(predictions_df["utc_ts"])
                predictions_df = predictions_df.sort_values("utc_ts").reset_index(
                    drop=True
                )
                artifacts["predictions"] = predictions_df

        # Extract scaling stats from model's live scaling_stats dict
        if hasattr(model, "scaling_stats") and model.scaling_stats:
            stats_rows = []
            for feature, stats in model.scaling_stats.items():
                stats_rows.append(
                    {
                        "location": model.settlement_point,
                        "feature": feature,
                        "mean": stats["mean"],
                        "std": stats["std"],
                    }
                )
            artifacts["scaling_stats"] = pd.DataFrame(stats_rows)

        # Extract feature importance from model's live results
        if hasattr(model, "results") and model.results:
            # Convert live results to feature summary format
            coeff_data = []

            for hour, hour_results in model.results.items():
                # Handle both linear models (coefficients) and XGBoost models (feature_importances)
                if "coefficients" in hour_results and "feature_names" in hour_results:
                    # Linear models: coefficients
                    coefficients = hour_results["coefficients"]
                    feature_names = hour_results["feature_names"]

                    for feature, coeff in zip(feature_names, coefficients):
                        coeff_data.append(
                            {
                                "end_hour": hour,
                                "feature": feature,
                                "coefficient": coeff,
                                "abs_coefficient": abs(coeff),
                                "is_zero": coeff == 0,
                                "model_type": model.model_type,
                                "settlement_point": model.settlement_point,
                            }
                        )

                elif (
                    "feature_importances" in hour_results
                    and "feature_names" in hour_results
                ):
                    # XGBoost models: feature importances (gain-based)
                    importances = hour_results["feature_importances"]
                    feature_names = hour_results["feature_names"]

                    for feature, importance in zip(feature_names, importances):
                        coeff_data.append(
                            {
                                "end_hour": hour,
                                "feature": feature,
                                "coefficient": importance,  # Use importance as coefficient for consistency
                                "abs_coefficient": importance,  # XGBoost importances are already non-negative
                                "is_zero": importance == 0,
                                "model_type": model.model_type,
                                "settlement_point": model.settlement_point,
                            }
                        )

            if coeff_data:
                coeff_df = pd.DataFrame(coeff_data)

                # Create feature summary (averaged across hours)
                feature_summary = (
                    coeff_df.groupby("feature")
                    .agg(
                        {
                            "coefficient": ["mean", "std", "count"],
                            "abs_coefficient": ["mean", "max"],
                            "is_zero": "sum",
                        }
                    )
                    .round(6)
                )

                # Flatten column names
                feature_summary.columns = [
                    "_".join(col).strip() for col in feature_summary.columns
                ]
                feature_summary = feature_summary.reset_index()

                # Add abs_coefficient_mean for compatibility with dashboard
                feature_summary["abs_coefficient_mean"] = feature_summary[
                    "abs_coefficient_mean"
                ]

                artifacts["feature_summary"] = feature_summary

        # Convert training results to DataFrame format for comparative dashboards
        if results:
            results_data = []
            for hour, hour_results in results.items():
                result_row = {"end_of_hour": hour}
                result_row.update(hour_results)
                results_data.append(result_row)

            artifacts["results"] = pd.DataFrame(results_data)

        return artifacts

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

    # ===== ANALYTICS DASHBOARD METHODS =====
    #
    # TIMESTAMP USAGE PATTERN:
    # - All internal operations (merging, sorting, indexing) use utc_ts to avoid DST issues
    # - Display purposes (chart x-axes, hover text) use local_ts for business readability
    # - local_ts is derived from utc_ts when needed for display: utc_ts -> UTC -> America/Chicago
    #
    def _create_analytics_dashboard(
        self,
        model_artifacts: Dict,
        model_name: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate comprehensive 4-panel analysis dashboard using trainer's dataset access.

        Args:
            model_artifacts: Dictionary containing model predictions, scaling stats, etc.
            model_name: Name of model being analyzed
            output_path: Where to save HTML file, optional

        Returns:
            str: Path to generated HTML dashboard
        """
        print(
            f"Creating analytics dashboard for {model_name} using trainer's dataset access..."
        )

        # Prepare dashboard data using trainer's dataset access
        dashboard_data = self._prepare_dashboard_data(model_artifacts, model_name)

        # Create 4-panel subplot figure with wider layout for MacBook Pro 16"
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "DART Price Predictions vs Actual",
                "Load Forecasts (All Contributing Zones)",
                "Wind Generation Forecasts (All Contributing Zones)",
                "Solar Generation Forecasts (All Contributing Zones)",
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,  # Reduced spacing for wider plots
        )

        # Panel 1: DART Predictions (Top-Left)
        self._add_dart_panel(fig, dashboard_data, row=1, col=1)

        # Panel 2: Load Forecasts (Top-Right)
        self._add_load_panel(fig, dashboard_data, row=1, col=2)

        # Panel 3: Wind Generation (Bottom-Left)
        self._add_wind_panel(fig, dashboard_data, row=2, col=1)

        # Panel 4: Solar Generation (Bottom-Right)
        self._add_solar_panel(fig, dashboard_data, row=2, col=2)

        # Apply professional styling with wider layout and 2-column legend
        title = f"Analytics Workbench: {model_name}"

        layout = get_professional_layout(
            title=title,
            height=900,
            width=1600,  # Increased width for MacBook Pro 16"
            showlegend=True,
            legend_position="external_right",
        )

        # Update legend to use better spacing for multiple features
        layout.update(
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=10),
                itemsizing="constant",
                tracegroupgap=0,
                # Configure for better multi-item display
                entrywidth=180,
                entrywidthmode="pixels",
            )
        )

        fig.update_layout(**layout)
        apply_professional_axis_styling(fig, rows=2, cols=2)

        # Enhanced axis labels with synchronized time axes
        fig.update_xaxes(title_text="Date/Time", row=1, col=1, matches="x")
        fig.update_xaxes(title_text="Date/Time", row=1, col=2, matches="x")
        fig.update_xaxes(title_text="Date/Time", row=2, col=1, matches="x")
        fig.update_xaxes(title_text="Date/Time", row=2, col=2, matches="x")

        fig.update_yaxes(title_text="DART Price ($/MWh)", row=1, col=1)
        fig.update_yaxes(title_text="Load Forecast (MW)", row=1, col=2)
        fig.update_yaxes(title_text="Wind Generation (MW)", row=2, col=1)
        fig.update_yaxes(title_text="Solar Generation (MW)", row=2, col=2)

        # Save dashboard
        if output_path is None:
            output_path = (
                Path(self.output_dir) / f"analytics_workbench_{model_name}.html"
            )
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(
            str(output_path),
            include_plotlyjs=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

        print(f"Analytics dashboard saved: {output_path}")
        return str(output_path)

    def _create_comparative_dashboard(
        self,
        model_artifacts_dict: Dict[str, Dict],
        model_names: List[str],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate comparative dashboard for multiple models using trainer's dataset access.

        Args:
            model_artifacts_dict: Dict mapping model names to their artifacts
            model_names: List of model names to compare
            output_path: Where to save HTML file, optional

        Returns:
            str: Path to generated HTML dashboard
        """
        print(
            f"Creating comparative dashboard for models: {model_names} using trainer's dataset access..."
        )

        # Create 2x2 subplot figure for comparison with wider layout for MacBook Pro 16"
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "DART Price Predictions Comparison",
                "Prediction Error Distribution by Model",
                "R² Performance by Hour",
                "MAE Performance by Hour",
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,  # Reduced spacing for wider plots
        )

        # Panel 1: DART Predictions Comparison (Top-Left)
        self._add_comparative_predictions_panel(
            fig, model_artifacts_dict, model_names, row=1, col=1
        )

        # Panel 2: Error Distribution (Top-Right)
        self._add_error_distribution_panel(
            fig, model_artifacts_dict, model_names, row=1, col=2
        )

        # Panel 3: R² by Hour (Bottom-Left)
        self._add_r2_by_hour_panel(fig, model_artifacts_dict, model_names, row=2, col=1)

        # Panel 4: MAE by Hour (Bottom-Right)
        self._add_mae_by_hour_panel(
            fig, model_artifacts_dict, model_names, row=2, col=2
        )

        # Apply professional styling with wider layout
        title = f"Comparative Analytics: {' vs '.join(model_names)}"

        layout = get_professional_layout(
            title=title,
            height=900,
            width=1600,  # Increased width for MacBook Pro 16"
            showlegend=True,
            legend_position="external_right",
        )

        fig.update_layout(**layout)
        apply_professional_axis_styling(fig, rows=2, cols=2)

        # Enhanced axis labels (different for comparative dashboard)
        fig.update_xaxes(title_text="Date/Time", row=1, col=1)
        fig.update_xaxes(title_text="Prediction Error ($/MWh)", row=1, col=2)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=2)

        fig.update_yaxes(title_text="DART Price ($/MWh)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="R² Score", row=2, col=1)
        fig.update_yaxes(title_text="MAE ($/MWh)", row=2, col=2)

        # Save dashboard
        if output_path is None:
            output_path = Path(self.output_dir) / "analytics_workbench_comparative.html"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(
            str(output_path),
            include_plotlyjs=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

        print(f"Comparative dashboard saved: {output_path}")
        return str(output_path)

    def _prepare_dashboard_data(self, model_artifacts: Dict, model_name: str) -> Dict:
        """Prepare dashboard data using raw dataset (correct units) + transformed predictions only."""
        # Get predictions from model artifacts
        if "predictions" not in model_artifacts:
            raise ValueError(
                f"No predictions found in model artifacts for {model_name}"
            )

        predictions = model_artifacts["predictions"].copy()
        predictions["utc_ts"] = pd.to_datetime(predictions["utc_ts"])

        # Ensure column name consistency for merge
        if (
            "end_hour" in predictions.columns
            and "end_of_hour" in self.dataset.df.columns
        ):
            predictions["end_of_hour"] = predictions["end_hour"]

        # SIMPLE APPROACH: Only transform DART predictions (inverse SLT)
        # All other data comes from raw dataset which is already in correct units
        predictions["predicted_dart_actual"] = inverse_signed_log_transform(
            predictions["predicted_dart_slt"]
        )
        predictions["actual_dart_actual"] = inverse_signed_log_transform(
            predictions["actual_dart_slt"]
        )

        # Calculate prediction error in $/MWh
        predictions["prediction_error"] = (
            predictions["predicted_dart_actual"] - predictions["actual_dart_actual"]
        )

        # Merge with raw dataset - features are already in correct MW units
        print(f"🔍 Using raw dataset with correct units:")
        print(f"  Predictions: {len(predictions)} rows")
        print(f"  Raw dataset: {len(self.dataset.df)} rows")

        merged_data = predictions.merge(
            self.dataset.df,
            on=["utc_ts", "local_ts", "end_of_hour"],
            how="left",
            validate="one_to_one",
        )

        if len(merged_data) != len(predictions):
            raise ValueError(
                f"MERGE FAILED: Expected {len(predictions)} rows, got {len(merged_data)}"
            )

        print(f"  ✅ Merge successful: {len(merged_data)} rows")

        # Sort by UTC timestamp
        merged_data = merged_data.sort_values("utc_ts").reset_index(drop=True)

        # Get feature importance
        feature_importance = self._get_feature_importance(model_artifacts)

        return {
            "merged_data": merged_data,
            "feature_importance": feature_importance,
            "predictions": predictions,
        }

    def _get_feature_importance(self, model_artifacts: Dict) -> pd.DataFrame:
        """Extract feature importance from model artifacts with fallback."""
        if "feature_summary" in model_artifacts:
            importance_df = model_artifacts["feature_summary"].copy()
            importance_df["abs_importance"] = importance_df["abs_coefficient_mean"]
            return importance_df.sort_values("abs_importance", ascending=False)
        else:
            # Fallback: use actual feature columns from trainer's dataset
            return pd.DataFrame(
                {
                    "feature": self.dataset.feature_columns[:10],  # Top 10 features
                    "abs_importance": np.linspace(
                        1.0, 0.1, len(self.dataset.feature_columns[:10])
                    ),
                }
            )

    def _add_dart_panel(self, fig, dashboard_data: Dict, row: int, col: int) -> None:
        """Add DART prediction vs actual panel (Top-Left)."""
        data = dashboard_data["merged_data"]

        # Actual DART prices - use local_ts for business display, include both timestamps in hover
        fig.add_trace(
            go.Scatter(
                x=data["local_ts"],
                y=data["actual_dart_actual"],
                name="Actual DART",
                line=dict(color=SEMANTIC_COLORS["positive"], width=2),
                hovertemplate="<b>Actual DART</b><br>Local Time: %{x}<br>UTC Time: %{customdata[0]}<br>Price: $%{y:.2f}/MWh<br>Hour: %{customdata[1]}<extra></extra>",
                customdata=list(zip(data["utc_ts"], data["end_of_hour"])),
            ),
            row=row,
            col=col,
        )

        # Predicted DART prices - use local_ts for business display, include both timestamps in hover
        fig.add_trace(
            go.Scatter(
                x=data["local_ts"],
                y=data["predicted_dart_actual"],
                name="Predicted DART",
                line=dict(color=SEMANTIC_COLORS["negative"], width=1.5),
                hovertemplate="<b>Predicted DART</b><br>Local Time: %{x}<br>UTC Time: %{customdata[0]}<br>Price: $%{y:.2f}/MWh<br>Error: $%{customdata[1]:.2f}/MWh<extra></extra>",
                customdata=list(zip(data["utc_ts"], data["prediction_error"])),
            ),
            row=row,
            col=col,
        )

    def _add_load_panel(self, fig, dashboard_data: Dict, row: int, col: int) -> None:
        """Add load forecast panel using raw dataset features (already in MW)."""
        data = dashboard_data["merged_data"]
        feature_importance = dashboard_data["feature_importance"]

        # Get ALL load forecast features from raw dataset (already in correct MW units)
        # Handle z-scored feature names by mapping back to original dataset column names
        load_features = []
        for f in feature_importance["feature"].tolist():
            if "load_forecast" in f:
                # Map z-scored feature names back to original dataset columns
                original_name = f.replace("_z", "") if f.endswith("_z") else f
                if original_name in data.columns:
                    load_features.append(
                        (f, original_name)
                    )  # (display_name, data_column)

        load_features.sort(key=lambda x: x[0])  # Sort by display name for consistency

        # Use extended color sequence for ALL features
        colors = COLOR_SEQUENCE * (len(load_features) // len(COLOR_SEQUENCE) + 1)

        for i, (feature_name, data_column) in enumerate(load_features):
            # Raw data is already in correct MW units - no transformation needed!
            fig.add_trace(
                go.Scatter(
                    x=data["local_ts"],
                    y=data[data_column],  # Use original column name for data access
                    name=feature_name.replace("_slt", "")
                    .replace("_z", "")
                    .replace("load_forecast_", "Load: "),
                    line=dict(color=colors[i], width=1.5),
                    opacity=0.8,
                    hovertemplate="<b>%{fullData.name}</b><br>Local Time: %{x}<br>UTC Time: %{customdata}<br>Load: %{y:.0f} MW<extra></extra>",
                    customdata=data["utc_ts"],
                ),
                row=row,
                col=col,
            )

    def _add_wind_panel(self, fig, dashboard_data: Dict, row: int, col: int) -> None:
        """Add wind generation panel using raw dataset features (already in MW)."""
        data = dashboard_data["merged_data"]
        feature_importance = dashboard_data["feature_importance"]

        # Get ALL wind features from raw dataset (already in correct MW units)
        # Handle z-scored feature names by mapping back to original dataset column names
        wind_features = []
        for f in feature_importance["feature"].tolist():
            if "wind_generation" in f:
                # Map z-scored feature names back to original dataset columns
                original_name = f.replace("_z", "") if f.endswith("_z") else f
                if original_name in data.columns:
                    wind_features.append(
                        (f, original_name)
                    )  # (display_name, data_column)

        wind_features.sort(key=lambda x: x[0])  # Sort by display name for consistency

        # Use extended color sequence for ALL features
        colors = COLOR_SEQUENCE * (len(wind_features) // len(COLOR_SEQUENCE) + 1)

        for i, (feature_name, data_column) in enumerate(wind_features):
            # Raw data is already in correct MW units - no transformation needed!
            fig.add_trace(
                go.Scatter(
                    x=data["local_ts"],
                    y=data[data_column],  # Use original column name for data access
                    name=feature_name.replace("_slt", "")
                    .replace("_z", "")
                    .replace("wind_generation_", "Wind: "),
                    line=dict(color=colors[i], width=1.5),
                    opacity=0.8,
                    hovertemplate="<b>%{fullData.name}</b><br>Local Time: %{x}<br>UTC Time: %{customdata}<br>Generation: %{y:.0f} MW<extra></extra>",
                    customdata=data["utc_ts"],
                ),
                row=row,
                col=col,
            )

    def _add_solar_panel(self, fig, dashboard_data: Dict, row: int, col: int) -> None:
        """Add solar generation panel using raw dataset features (already in MW)."""
        data = dashboard_data["merged_data"]
        feature_importance = dashboard_data["feature_importance"]

        # Get ALL solar features from raw dataset (already in correct MW units)
        # Handle z-scored feature names by mapping back to original dataset column names
        solar_features = []
        for f in feature_importance["feature"].tolist():
            if "solar" in f:
                # Map z-scored feature names back to original dataset columns
                original_name = f.replace("_z", "") if f.endswith("_z") else f
                if original_name in data.columns:
                    solar_features.append(
                        (f, original_name)
                    )  # (display_name, data_column)

        solar_features.sort(key=lambda x: x[0])  # Sort by display name for consistency

        # Use extended color sequence for ALL features
        colors = COLOR_SEQUENCE * (len(solar_features) // len(COLOR_SEQUENCE) + 1)

        for i, (feature_name, data_column) in enumerate(solar_features):
            # Raw data is already in correct MW units - no transformation needed!
            fig.add_trace(
                go.Scatter(
                    x=data["local_ts"],
                    y=data[data_column],  # Use original column name for data access
                    name=feature_name.replace("_slt", "")
                    .replace("_z", "")
                    .replace("solar_", "Solar: "),
                    line=dict(color=colors[i], width=1.5),
                    opacity=0.8,
                    hovertemplate="<b>%{fullData.name}</b><br>Local Time: %{x}<br>UTC Time: %{customdata}<br>Generation: %{y:.0f} MW<extra></extra>",
                    customdata=data["utc_ts"],
                ),
                row=row,
                col=col,
            )

    def _add_comparative_predictions_panel(
        self,
        fig,
        model_artifacts_dict: Dict[str, Dict],
        model_names: List[str],
        row: int,
        col: int,
    ) -> None:
        """Add comparative predictions panel showing all models."""
        colors = COLOR_SEQUENCE[: len(model_names)]

        # Add actual values once (from first model) - use full merged_data like individual dashboards
        if model_names:
            first_model_data = self._prepare_dashboard_data(
                model_artifacts_dict[model_names[0]], model_names[0]
            )
            # Use merged_data (full dataset) instead of filtering to validation only
            full_data = first_model_data["merged_data"]

            if not full_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=full_data[
                            "local_ts"
                        ],  # Use existing local_ts for business display
                        y=full_data["actual_dart_actual"],
                        name="Actual DART",
                        line=dict(color=SEMANTIC_COLORS["positive"], width=2),
                        hovertemplate="<b>Actual DART</b><br>Local Time: %{x}<br>UTC Time: %{customdata}<br>Price: $%{y:.2f}/MWh<extra></extra>",
                        customdata=full_data["utc_ts"],
                    ),
                    row=row,
                    col=col,
                )

        # Add predicted values for each model - also use full merged_data
        for i, model_name in enumerate(model_names):
            try:
                model_data = self._prepare_dashboard_data(
                    model_artifacts_dict[model_name], model_name
                )
                # Use merged_data (full dataset) instead of filtering to validation only
                full_data = model_data["merged_data"]

                if not full_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=full_data[
                                "local_ts"
                            ],  # Use existing local_ts for business display
                            y=full_data["predicted_dart_actual"],
                            name=f"Predicted ({model_name})",
                            line=dict(color=colors[i], width=1.5),
                            hovertemplate=f"<b>{model_name}</b><br>Local Time: %{{x}}<br>UTC Time: %{{customdata}}<br>Price: $%{{y:.2f}}/MWh<extra></extra>",
                            customdata=full_data["utc_ts"],
                        ),
                        row=row,
                        col=col,
                    )
            except Exception as e:
                print(f"Warning: Could not add {model_name} to comparison: {e}")

    def _add_error_distribution_panel(
        self,
        fig,
        model_artifacts_dict: Dict[str, Dict],
        model_names: List[str],
        row: int,
        col: int,
    ) -> None:
        """Add error distribution comparison panel."""
        colors = COLOR_SEQUENCE[: len(model_names)]

        for i, model_name in enumerate(model_names):
            try:
                model_data = self._prepare_dashboard_data(
                    model_artifacts_dict[model_name], model_name
                )
                validation_data = model_data["predictions"][
                    model_data["predictions"]["dataset_type"] == "validation"
                ]

                if not validation_data.empty:
                    fig.add_trace(
                        go.Histogram(
                            x=validation_data["prediction_error"],
                            name=model_name,
                            opacity=0.7,
                            marker_color=colors[i],
                            hovertemplate=f"<b>{model_name}</b><br>Error: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )
            except Exception as e:
                print(f"Warning: Could not add {model_name} error distribution: {e}")

    def _add_r2_by_hour_panel(
        self,
        fig,
        model_artifacts_dict: Dict[str, Dict],
        model_names: List[str],
        row: int,
        col: int,
    ) -> None:
        """Add R² by hour comparison panel."""
        colors = COLOR_SEQUENCE[: len(model_names)]

        for i, model_name in enumerate(model_names):
            try:
                model_artifacts = model_artifacts_dict[model_name]
                if "results" in model_artifacts:
                    results_df = model_artifacts["results"]

                    fig.add_trace(
                        go.Scatter(
                            x=results_df["end_of_hour"],
                            y=results_df.get(
                                "bootstrap_r2_mean", results_df.get("validation_r2", [])
                            ),
                            name=f"{model_name} R²",
                            line=dict(color=colors[i], width=2),
                            mode="lines+markers",
                            hovertemplate=f"<b>{model_name}</b><br>Hour: %{{x}}<br>R²: %{{y:.3f}}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )
            except Exception as e:
                print(f"Warning: Could not add {model_name} R² data: {e}")

    def _add_mae_by_hour_panel(
        self,
        fig,
        model_artifacts_dict: Dict[str, Dict],
        model_names: List[str],
        row: int,
        col: int,
    ) -> None:
        """Add MAE by hour comparison panel."""
        colors = COLOR_SEQUENCE[: len(model_names)]

        for i, model_name in enumerate(model_names):
            try:
                model_artifacts = model_artifacts_dict[model_name]
                if "results" in model_artifacts:
                    results_df = model_artifacts["results"]

                    fig.add_trace(
                        go.Scatter(
                            x=results_df["end_of_hour"],
                            y=results_df.get(
                                "bootstrap_mae_mean",
                                results_df.get("validation_mae", []),
                            ),
                            name=f"{model_name} MAE",
                            line=dict(color=colors[i], width=2),
                            mode="lines+markers",
                            hovertemplate=f"<b>{model_name}</b><br>Hour: %{{x}}<br>MAE: $%{{y:.2f}}/MWh<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )
            except Exception as e:
                print(f"Warning: Could not add {model_name} MAE data: {e}")
