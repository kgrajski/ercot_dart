"""Linear Regression Model for Experiment 0."""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.ercot.exp0.base_model import BaseExp0Model


class LinearRegressionModel(BaseExp0Model):
    """Linear Regression implementation for Experiment 0.

    This class implements standard linear regression for DART price prediction.
    Each hour gets its own separate linear model to capture hour-specific patterns.
    """

    def __init__(self, output_dir: str, settlement_point: str, random_state: int = 42):
        """Initialize linear regression model.

        Args:
            output_dir: Directory for saving outputs and artifacts
            settlement_point: Settlement point identifier (e.g., 'LZ_HOUSTON_LZ')
            random_state: Random seed for reproducibility
        """
        super().__init__(
            model_type="linear_regression",
            output_dir=output_dir,
            settlement_point=settlement_point,
            random_state=random_state,
        )

    def _train_model_for_hour(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: Optional[pd.DataFrame],
        y_validation: Optional[pd.Series],
        bootstrap_iterations: int,
        hour: int,
    ) -> Tuple[LinearRegression, Dict]:
        """Train a linear regression model for a specific hour.

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
        # Create and train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        results = self._evaluate_model(
            model, X_train, y_train, X_validation, y_validation, bootstrap_iterations
        )

        # Add linear regression specific metrics
        results.update(
            {
                "coefficients": model.coef_.tolist(),
                "intercept": model.intercept_,
                "feature_names": X_train.columns.tolist(),
            }
        )

        # Feature importance (absolute coefficient values)
        feature_importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "coefficient": model.coef_,
                "abs_coefficient": abs(model.coef_),
            }
        ).sort_values("abs_coefficient", ascending=False)

        results["feature_importance"] = feature_importance.to_dict("records")

        return model, results

    # TODO: Implement plot_feature_importance() with plotly (HTML/PNG/CSV output)
    # For now, removed to focus on core training functionality
