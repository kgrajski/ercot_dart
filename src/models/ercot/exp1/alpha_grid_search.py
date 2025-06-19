"""Alpha Grid Search Utility for Ridge and Lasso Regression.

This module provides utilities for testing multiple regularization strengths (alpha values)
for Ridge and Lasso regression models. It supports both simple grid search and more
sophisticated hyperparameter optimization.
"""

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from src.models.ercot.exp1.model_trainer import Exp1ModelTrainer


class AlphaGridSearch:
    """Utility for testing multiple alpha values for regularized regression models.

    This class provides methods for:
    - Grid search over alpha values
    - Comparison of regularization effects
    - Selection of optimal alpha based on validation performance
    """

    def __init__(self, trainer: Exp1ModelTrainer):
        """Initialize alpha grid search.

        Args:
            trainer: Exp1ModelTrainer instance with loaded dataset
        """
        self.trainer = trainer
        self.results = {}

    def search_alpha_values(
        self,
        model_type: str,
        alpha_values: List[float],
        bootstrap_iterations: int = 10,
        hours_to_train: Optional[List[int]] = None,
        **model_kwargs,
    ) -> Dict[float, Dict]:
        """Search over multiple alpha values for a regularized model.

        Args:
            model_type: Type of model ('ridge_regression' or 'lasso_regression')
            alpha_values: List of alpha values to test
            bootstrap_iterations: Number of bootstrap iterations per alpha
            hours_to_train: List of hours to train (1-24). If None, trains all hours.
            **model_kwargs: Additional model parameters (e.g., analyze_feature_selection)

        Returns:
            Dictionary with results for each alpha value
        """
        if model_type not in ["ridge_regression", "lasso_regression"]:
            raise ValueError(f"Model type {model_type} not supported for alpha search")

        print(f"\n=== Alpha Grid Search for {model_type} ===")
        print(f"Testing alpha values: {alpha_values}")
        print(f"Settlement point: {self.trainer.settlement_point}")

        alpha_results = {}

        for alpha in alpha_values:
            print(f"\n--- Testing alpha = {alpha} ---")

            try:
                # Train model with this alpha value
                results = self.trainer.train_model(
                    model_type=model_type,
                    bootstrap_iterations=bootstrap_iterations,
                    hours_to_train=hours_to_train,
                    alpha=alpha,
                    **model_kwargs,
                )

                alpha_results[alpha] = results

                # Calculate summary statistics
                summary = self._calculate_alpha_summary(results, alpha)
                print(f"Alpha {alpha} summary: {summary}")

            except Exception as e:
                print(f"ERROR with alpha {alpha}: {e}")
                alpha_results[alpha] = None

        self.results[model_type] = alpha_results
        return alpha_results

    def _calculate_alpha_summary(self, results: Dict, alpha: float) -> Dict:
        """Calculate summary statistics for a given alpha value."""
        if not results:
            return {"avg_bootstrap_r2": np.nan, "avg_validation_r2": np.nan}

        # Calculate average performance across hours
        bootstrap_r2_values = [
            r.get("bootstrap_r2_mean", np.nan) for r in results.values()
        ]
        validation_r2_values = [
            r.get("validation_r2", np.nan) for r in results.values()
        ]

        summary = {
            "alpha": alpha,
            "avg_bootstrap_r2": np.nanmean(bootstrap_r2_values),
            "std_bootstrap_r2": np.nanstd(bootstrap_r2_values),
            "avg_validation_r2": np.nanmean(validation_r2_values),
            "std_validation_r2": np.nanstd(validation_r2_values),
            "n_hours_trained": len([r for r in results.values() if r is not None]),
        }

        return summary

    def compare_alpha_values(
        self, model_type: str, metric: str = "bootstrap_r2_mean"
    ) -> pd.DataFrame:
        """Compare performance across different alpha values.

        Args:
            model_type: Type of model to compare
            metric: Metric to use for comparison

        Returns:
            DataFrame with comparison results
        """
        if model_type not in self.results:
            raise ValueError(f"No results found for {model_type}")

        comparison_data = []

        for alpha, alpha_results in self.results[model_type].items():
            if alpha_results is None:
                continue

            for hour, hour_results in alpha_results.items():
                if hour_results is None:
                    continue

                row = {
                    "alpha": alpha,
                    "hour": hour,
                    "metric_value": hour_results.get(metric, np.nan),
                }

                # Add additional metrics for context
                row.update(
                    {
                        "bootstrap_r2_mean": hour_results.get(
                            "bootstrap_r2_mean", np.nan
                        ),
                        "validation_r2": hour_results.get("validation_r2", np.nan),
                        "bootstrap_r2_std": hour_results.get(
                            "bootstrap_r2_std", np.nan
                        ),
                    }
                )

                # Add Lasso-specific metrics if available
                if "sparsity_ratio" in hour_results:
                    row["sparsity_ratio"] = hour_results["sparsity_ratio"]
                    row["n_features_selected"] = hour_results["n_features_selected"]

                comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def get_best_alpha(
        self, model_type: str, metric: str = "validation_r2", aggregation: str = "mean"
    ) -> float:
        """Get the best alpha value based on specified metric.

        Args:
            model_type: Type of model
            metric: Metric to optimize
            aggregation: How to aggregate across hours ('mean', 'median')

        Returns:
            Best alpha value
        """
        comparison_df = self.compare_alpha_values(model_type, metric)

        if aggregation == "mean":
            alpha_performance = comparison_df.groupby("alpha")[metric].mean()
        elif aggregation == "median":
            alpha_performance = comparison_df.groupby("alpha")[metric].median()
        else:
            raise ValueError(f"Aggregation {aggregation} not supported")

        best_alpha = alpha_performance.idxmax()
        print(f"Best alpha for {model_type} based on {metric}: {best_alpha}")

        return best_alpha

    @staticmethod
    def get_default_alpha_grids() -> Dict[str, List[float]]:
        """Get default alpha grids for different model types.

        Returns:
            Dictionary with default alpha grids
        """
        return {
            "ridge_regression": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "lasso_regression": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        }

    @staticmethod
    def get_coarse_alpha_grids() -> Dict[str, List[float]]:
        """Get coarse alpha grids for quick testing.

        Returns:
            Dictionary with coarse alpha grids
        """
        return {
            "ridge_regression": [0.01, 1.0, 100.0],
            "lasso_regression": [0.001, 0.1, 10.0],
        }
