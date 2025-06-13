"""Model Trainer for Experiment 0.

This module provides the Exp0ModelTrainer class, which serves as the main interface
for training and managing different model types for DART price prediction.

Key Features:
- Factory pattern for model type selection
- Integration with DartSltExp0Dataset
- Unified interface for all model types

Similar to ERCOTBaseClient pattern, this class orchestrates the modeling workflow
while delegating specific model implementation details to specialized model classes.
"""

from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from src.models.ercot.exp0.models.linear_regression import LinearRegressionModel


class Exp0ModelTrainer:
    """Main trainer class for Experiment 0 models.

    This class provides a unified interface for training and managing different
    model types, following the same pattern as the ERCOT data clients.

    Supported model types:
    - linear_regression: Standard linear regression
    - random_forest: Random forest (future implementation)
    - neural_network: Neural networks (future implementation)
    """

    # Available model types and their implementations
    MODEL_REGISTRY = {
        "linear_regression": LinearRegressionModel,
        # Future models can be added here:
        # 'random_forest': RandomForestModel,
        # 'neural_network': NeuralNetworkModel,
    }

    def __init__(
        self, dataset, modeling_dir: str, settlement_point: str, random_state: int = 42
    ):
        """Initialize model trainer.

        Args:
            dataset: DartSltExp0Dataset instance with model-ready data
            modeling_dir: Path to the modeling directory; individual model type results will be saved in subdirectories
            settlement_point: Settlement point identifier (e.g., 'LZ_HOUSTON_LZ')
            random_state: Random seed for reproducibility
        """
        # Create modeling directory for training outputs
        self.output_dir = modeling_dir

        self.settlement_point = settlement_point
        self.random_state = random_state
        self.dataset = dataset

        # Print dataset info
        print(f"Dataset loaded for {self.settlement_point}")
        print(f"Total samples: {len(dataset)}")
        print(f"Features: {len(dataset.get_feature_names())}")

    def train_model(
        self,
        model_type: str,
        bootstrap_iterations: int = 5,
        hours_to_train: Optional[List[int]] = None,
        **model_kwargs,
    ) -> Dict:
        """Train a specific model type using bootstrap resampling for evaluation.

        Args:
            model_type: Type of model to train (e.g., 'linear_regression')
            bootstrap_iterations: Number of bootstrap iterations for model evaluation
            hours_to_train: List of hours to train (1-24). If None, trains all hours.
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

        print(f"\nTraining {model_type} model for {self.settlement_point}")

        # Create model instance using factory pattern
        model_class = self.MODEL_REGISTRY[model_type]
        model = model_class(
            output_dir=str(self.output_dir),
            settlement_point=self.settlement_point,
            random_state=self.random_state,
            **model_kwargs,
        )

        # Load data into model
        model.load_data(self.dataset)

        # Train hourly models
        results = model.train_hourly_models(
            bootstrap_iterations=bootstrap_iterations, hours_to_train=hours_to_train
        )

        print(f"\n{model_type.title()} training completed!")
        print(f"Trained models for {len(results)} hours")

        return results

    def run_experiment(
        self,
        model_types: List[str] = None,
        bootstrap_iterations: int = 10,  # Development default - consider 100+ for production
        hours_to_train: Optional[List[int]] = None,
    ) -> Dict[str, Dict]:
        """Run complete experiment with multiple model types.

        Args:
            model_types: List of model types to train. If None, uses ['linear_regression']
            bootstrap_iterations: Number of bootstrap iterations for model evaluation
                                 - Development: 10-50 (fast iteration)
                                 - Testing: 100-500 (more robust)
                                 - Production: 1000+ (publication quality)
            hours_to_train: List of hours to train (1-24). If None, trains all hours.

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
                results = self.train_model(
                    model_type=model_type,
                    bootstrap_iterations=bootstrap_iterations,
                    hours_to_train=hours_to_train,
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

        return all_results
