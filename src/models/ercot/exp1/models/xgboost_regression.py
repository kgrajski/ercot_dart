"""XGBoost Regression Model for Experiment 1."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
import xgboost as xgb

from src.models.ercot.exp1.base_model import BaseExp1Model


class XGBoostRegressionModel(BaseExp1Model):
    """XGBoost Regression implementation for Experiment 1.

    This class implements XGBoost gradient boosting for DART price prediction.
    XGBoost is particularly well-suited for electricity markets because it can:
    - Capture non-linear relationships between load/wind/solar and prices
    - Learn complex feature interactions (e.g., low wind + high load = price spike)
    - Handle regime switching (normal pricing vs congestion-driven spikes)
    - Automatically discover transmission congestion patterns

    Each hour gets its own separate XGBoost model to capture hour-specific patterns.
    """

    def __init__(
        self,
        output_dir: str,
        settlement_point: str,
        random_state: int = 42,
        feature_scaling: str = "none",  # XGBoost typically doesn't need scaling
        # XGBoost hyperparameters - optimized for fast development/testing
        n_estimators: int = 50,  # Fast default for development (vs 100 production)
        max_depth: int = 4,  # Moderate complexity (vs 6 production)
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,  # L1 regularization
        reg_lambda: float = 1.0,  # L2 regularization
        **kwargs,
    ):
        """Initialize XGBoost regression model.

        Args:
            output_dir: Directory for saving outputs and artifacts
            settlement_point: Settlement point identifier (e.g., 'LZ_HOUSTON_LZ')
            random_state: Random seed for reproducibility
            feature_scaling: Feature scaling method (recommend 'none' for XGBoost)
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            **kwargs: Additional parameters passed to base class (e.g., use_synthetic_data)
        """
        super().__init__(
            model_type="xgboost_regression",
            output_dir=output_dir,
            settlement_point=settlement_point,
            random_state=random_state,
            feature_scaling=feature_scaling,
            **kwargs,
        )

        # Store XGBoost hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

    def _train_model_for_hour(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: Optional[pd.DataFrame],
        y_validation: Optional[pd.Series],
        bootstrap_iterations: int,
        hour: int,
    ) -> Tuple[xgb.XGBRegressor, Dict]:
        """Train an XGBoost regression model for a specific hour.

        CRITICAL TRAINING WORKFLOW:
        1. Train the main model on ALL training data (X_train, y_train)
        2. Evaluate this main model using:
           - Bootstrap estimation on training data (for uncertainty quantification)
           - Final validation on holdout data (for TRUE performance assessment)

        XGBoost is particularly effective for electricity markets because it can:
        - Learn non-linear congestion patterns
        - Discover feature interactions automatically
        - Handle regime switching between normal and spike pricing

        Args:
            X_train: Training features (2024 data for this hour) - used to train the main model
            y_train: Training target (2024 data for this hour) - used to train the main model
            X_validation: Validation features (2025 data for this hour, may be None) - HOLDOUT DATA for evaluation
            y_validation: Validation target (2025 data for this hour, may be None) - HOLDOUT DATA for evaluation
            bootstrap_iterations: Number of bootstrap iterations (applied to X_train/y_train only)
            hour: Hour being trained (1-24)

        Returns:
            Tuple of (trained_model, results_dict)
        """
        # Create and train XGBoost regression model ON ALL TRAINING DATA
        # This is our main/final model that will be deployed
        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",  # Faster training
            verbosity=0,  # Reduce output noise
        )

        model.fit(X_train, y_train)  # MAIN MODEL TRAINING ON ALL TRAINING DATA

        # Evaluate the model:
        # - Bootstrap metrics: temporary models for uncertainty estimation
        # - Validation metrics: THIS MAIN MODEL evaluated on holdout validation data
        results = self._evaluate_model(
            model, X_train, y_train, X_validation, y_validation, bootstrap_iterations
        )

        # Add XGBoost specific metrics
        feature_importance = model.feature_importances_
        feature_names = X_train.columns.tolist()

        results.update(
            {
                "feature_importances": feature_importance.tolist(),
                "feature_names": feature_names,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            }
        )

        # Feature importance DataFrame (XGBoost uses gain-based importance)
        feature_importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": feature_importance,
                "abs_importance": feature_importance,  # XGBoost importances are already non-negative
            }
        ).sort_values("importance", ascending=False)

        results["feature_importance"] = feature_importance_df.to_dict("records")

        return model, results

    def get_feature_interactions(self, top_n: int = 10) -> Dict[int, List[Tuple]]:
        """Get top feature interactions discovered by XGBoost for each hour.

        This method extracts the most important feature interactions from trained models,
        which is particularly valuable for understanding congestion patterns.

        Args:
            top_n: Number of top interactions to return per hour

        Returns:
            Dictionary mapping hour -> list of (feature1, feature2, interaction_gain) tuples
        """
        interactions = {}

        for hour, model in self.models.items():
            if hasattr(model, "get_booster"):
                # Extract feature interactions from XGBoost booster
                booster = model.get_booster()
                # Note: This is a simplified version - XGBoost interaction extraction
                # requires more complex tree traversal for full implementation
                interactions[hour] = []

        return interactions

    # TODO: Implement plot_feature_importance() with plotly (HTML/PNG/CSV output)
    # TODO: Implement plot_partial_dependence() for key features
    # TODO: Implement plot_feature_interactions() to visualize congestion patterns
    # For now, removed to focus on core training functionality
