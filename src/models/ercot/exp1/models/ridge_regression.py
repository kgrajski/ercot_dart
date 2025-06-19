"""Ridge Regression Model for Experiment 1."""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.ercot.exp1.base_model import BaseExp1Model


class RidgeRegressionModel(BaseExp1Model):
    """Ridge Regression implementation for Experiment 1.

    Ridge regression adds L2 regularization to linear regression, which:
    - Helps with multicollinearity among load/wind/solar features
    - Improves generalization for the challenging DAM operational constraints
    - Provides stable coefficients even with high-dimensional feature sets

    Each hour gets its own separate Ridge model with the same alpha parameter.
    """

    def __init__(
        self,
        output_dir: str,
        settlement_point: str,
        alpha: float = 10.0,
        random_state: int = 42,
        feature_scaling: str = "zscore",
        **kwargs,
    ):
        """Initialize Ridge regression model.

        Args:
            output_dir: Directory for saving outputs and artifacts
            settlement_point: Settlement point identifier (e.g., 'LZ_HOUSTON_LZ')
            alpha: Ridge regularization strength (L2 penalty coefficient)
            random_state: Random seed for reproducibility
            feature_scaling: Feature scaling method ('none' or 'zscore') - defaults to 'zscore' for Ridge
            **kwargs: Additional parameters passed to base class (e.g., use_synthetic_data)
        """
        # Store Ridge-specific parameters
        self.alpha = alpha

        super().__init__(
            model_type=f"ridge_regression_alpha_{alpha}",
            output_dir=output_dir,
            settlement_point=settlement_point,
            random_state=random_state,
            feature_scaling=feature_scaling,
            **kwargs,
        )

    def _train_model_for_hour(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: Optional[pd.DataFrame],
        y_validation: Optional[pd.Series],
        bootstrap_iterations: int,
        hour: int,
    ) -> Tuple[Ridge, Dict]:
        """Train a Ridge regression model for a specific hour.

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
        # Create and train Ridge regression model
        model = Ridge(
            alpha=self.alpha,
            solver="auto",
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)

        # Evaluate the model
        results = self._evaluate_model(
            model, X_train, y_train, X_validation, y_validation, bootstrap_iterations
        )

        # Add Ridge regression specific metrics
        results.update(
            {
                "coefficients": model.coef_.tolist(),
                "intercept": model.intercept_,
                "feature_names": X_train.columns.tolist(),
                "alpha": self.alpha,
            }
        )

        # Feature importance (absolute coefficient values) - Ridge shrinks but doesn't zero out
        feature_importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "coefficient": model.coef_,
                "abs_coefficient": np.abs(model.coef_),
            }
        ).sort_values("abs_coefficient", ascending=False)

        results["feature_importance"] = feature_importance.to_dict("records")

        # Ridge-specific analysis: coefficient magnitudes and regularization effect
        results.update(
            {
                "max_abs_coefficient": float(np.max(np.abs(model.coef_))),
                "mean_abs_coefficient": float(np.mean(np.abs(model.coef_))),
                "coefficient_l2_norm": float(np.linalg.norm(model.coef_)),
                "coefficient_std": float(np.std(model.coef_)),
            }
        )

        return model, results

    # TODO: Implement regularization path analysis (multiple alpha values)
    # TODO: Implement plot_regularization_effect() with plotly
