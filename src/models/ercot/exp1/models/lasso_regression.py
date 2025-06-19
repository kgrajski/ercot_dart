"""Lasso Regression Model for Experiment 1."""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from src.models.ercot.exp1.base_model import BaseExp1Model


class LassoRegressionModel(BaseExp1Model):
    """Lasso Regression implementation for Experiment 1.

    Lasso regression adds L1 regularization to linear regression, which:
    - Performs automatic feature selection by zeroing out coefficients
    - Helps with multicollinearity among load/wind/solar features
    - Creates sparse models that are interpretable
    - Essential for high-dimensional problems with DAM operational constraints

    Each hour gets its own separate Lasso model with the same alpha parameter.
    """

    def __init__(
        self,
        output_dir: str,
        settlement_point: str,
        alpha: float = 0.01,
        max_iter: int = 1000,
        random_state: int = 42,
        feature_scaling: str = "zscore",
        feature_analysis: bool = False,
        stability_tracking: bool = False,
        **kwargs,
    ):
        """Initialize Lasso regression model.

        Args:
            output_dir: Directory for saving outputs and artifacts
            settlement_point: Settlement point identifier (e.g., 'LZ_HOUSTON_LZ')
            alpha: Lasso regularization strength (L1 penalty coefficient)
            max_iter: Maximum number of iterations for convergence
            random_state: Random seed for reproducibility
            feature_scaling: Feature scaling method ('none' or 'zscore') - defaults to 'zscore' for Lasso
            feature_analysis: Whether to perform detailed feature analysis (default False for performance)
            stability_tracking: Whether to track feature selection stability (default False for performance)
            **kwargs: Additional parameters passed to base class (e.g., use_synthetic_data)
        """
        # Store Lasso-specific parameters
        self.alpha = alpha
        self.max_iter = max_iter
        self.feature_analysis = feature_analysis
        self.stability_tracking = stability_tracking

        super().__init__(
            model_type=f"lasso_regression_alpha_{alpha}",
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
    ) -> Tuple[Lasso, Dict]:
        """Train a Lasso regression model for a specific hour.

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
        # Create and train Lasso regression model
        model = Lasso(
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)

        # Evaluate the model
        results = self._evaluate_model(
            model, X_train, y_train, X_validation, y_validation, bootstrap_iterations
        )

        # Add Lasso regression specific metrics
        results.update(
            {
                "coefficients": model.coef_.tolist(),
                "intercept": model.intercept_,
                "feature_names": X_train.columns.tolist(),
                "alpha": self.alpha,
                "max_iter": self.max_iter,
                "n_iter": model.n_iter_,
            }
        )

        # Feature importance (absolute coefficient values) - Many will be exactly zero
        feature_importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "coefficient": model.coef_,
                "abs_coefficient": np.abs(model.coef_),
                "selected": model.coef_ != 0,  # Boolean: was feature selected?
            }
        ).sort_values("abs_coefficient", ascending=False)

        results["feature_importance"] = feature_importance.to_dict("records")

        # Lasso-specific analysis: sparsity and feature selection
        n_features_total = len(model.coef_)
        n_features_selected = np.sum(model.coef_ != 0)
        n_features_zeroed = n_features_total - n_features_selected

        results.update(
            {
                "n_features_total": n_features_total,
                "n_features_selected": n_features_selected,
                "n_features_zeroed": n_features_zeroed,
                "sparsity_ratio": float(n_features_zeroed / n_features_total),
                "selected_features": X_train.columns[model.coef_ != 0].tolist(),
                "zeroed_features": X_train.columns[model.coef_ == 0].tolist(),
            }
        )

        # Optional: Detailed feature selection analysis
        if self.feature_analysis:
            results.update(self._analyze_feature_selection(X_train, model))

        # Optional: Feature stability analysis across bootstrap iterations
        if self.stability_tracking:
            results.update(
                self._analyze_feature_stability(
                    X_train, y_train, bootstrap_iterations, hour
                )
            )

        return model, results

    def _analyze_feature_selection(
        self, X_train: pd.DataFrame, model: Lasso
    ) -> Dict[str, Any]:
        """Analyze which types of features were selected by Lasso.

        Returns:
            Dictionary with feature selection analysis by feature type
        """
        selected_mask = model.coef_ != 0
        selected_features = X_train.columns[selected_mask].tolist()

        # Categorize selected features by type
        feature_categories = {
            "lag_features": [f for f in selected_features if "lag_" in f],
            "rolling_features": [f for f in selected_features if "roll_" in f],
            "load_features": [f for f in selected_features if "load_forecast_" in f],
            "wind_features": [f for f in selected_features if "wind_generation_" in f],
            "solar_features": [f for f in selected_features if "solar_" in f],
            "time_features": [
                f
                for f in selected_features
                if any(t in f for t in ["is_weekend", "is_holiday", "day_of_week_"])
            ],
        }

        # Count selections by category
        selection_counts = {
            cat: len(features) for cat, features in feature_categories.items()
        }

        return {
            "feature_selection_by_category": feature_categories,
            "selection_counts_by_category": selection_counts,
        }

    def _analyze_feature_stability(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        bootstrap_iterations: int,
        hour: int,
    ) -> Dict[str, Any]:
        """Analyze feature selection stability across bootstrap iterations.

        Returns:
            Dictionary with feature stability metrics
        """
        from sklearn.utils import resample

        print(
            f"    Analyzing feature stability across {bootstrap_iterations} bootstrap iterations..."
        )

        feature_selection_counts = pd.Series(0, index=X_train.columns)
        bootstrap_models = []

        for i in range(bootstrap_iterations):
            try:
                # Create bootstrap sample
                X_boot, y_boot = resample(
                    X_train,
                    y_train,
                    n_samples=len(X_train),
                    random_state=self.random_state + i,
                )

                # Train Lasso on bootstrap sample
                bootstrap_model = Lasso(
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    random_state=self.random_state + i,
                )
                bootstrap_model.fit(X_boot, y_boot)
                bootstrap_models.append(bootstrap_model)

                # Count feature selections
                selected_mask = bootstrap_model.coef_ != 0
                feature_selection_counts[selected_mask] += 1

            except Exception as e:
                print(f"    Bootstrap stability iteration {i+1} failed: {str(e)}")
                continue

        # Calculate stability metrics
        selection_frequencies = feature_selection_counts / len(bootstrap_models)

        # Features selected in >50% of bootstrap iterations are considered "stable"
        stable_features = selection_frequencies[
            selection_frequencies > 0.5
        ].index.tolist()
        unstable_features = selection_frequencies[
            (selection_frequencies > 0) & (selection_frequencies <= 0.5)
        ].index.tolist()

        return {
            "feature_selection_frequencies": selection_frequencies.to_dict(),
            "stable_features": stable_features,
            "unstable_features": unstable_features,
            "n_stable_features": len(stable_features),
            "n_unstable_features": len(unstable_features),
            "stability_analyzed": True,
            "stability_bootstrap_iterations": len(bootstrap_models),
        }

    # TODO: Implement regularization path analysis (multiple alpha values)
    # TODO: Implement plot_feature_selection_path() with plotly
