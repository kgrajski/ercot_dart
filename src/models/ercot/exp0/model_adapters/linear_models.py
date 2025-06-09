"""
Linear Model Adapters for ERCOT DART Prediction

This module provides adapters for various linear regression models including:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

All models follow the same adapter interface for consistent usage.
"""

import pickle
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.ercot.exp0.model_adapters.base_adapter import BaseModelAdapter


class LinearModelAdapter(BaseModelAdapter):
    """
    Adapter for scikit-learn linear models with optional standardization.

    Supports hour-specific models and 24-vector output models.
    """

    def __init__(
        self,
        model_type: str = "ridge",
        standardize: bool = True,
        hour_specific: bool = False,
        **model_kwargs,
    ):
        """
        Initialize linear model adapter.

        Args:
            model_type: Type of linear model ('linear', 'ridge', 'lasso', 'elasticnet')
            standardize: Whether to standardize features
            hour_specific: Whether to train separate models per hour (24 models)
            **model_kwargs: Additional arguments for the sklearn model
        """
        super().__init__(
            f"linear_{model_type}",
            {
                "model_type": model_type,
                "standardize": standardize,
                "hour_specific": hour_specific,
                **model_kwargs,
            },
        )

        self.model_type = model_type
        self.standardize = standardize
        self.hour_specific = hour_specific
        self.model_kwargs = model_kwargs

        # Initialize model(s)
        self._init_models()

    def _init_models(self):
        """Initialize the sklearn model(s)"""
        # Model class mapping
        model_classes = {
            "linear": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "elasticnet": ElasticNet,
        }

        if self.model_type not in model_classes:
            raise ValueError(f"Unknown model type: {self.model_type}")

        model_class = model_classes[self.model_type]

        if self.hour_specific:
            # Create 24 separate models for each hour
            self.models = {}
            for hour in range(24):
                if self.standardize:
                    self.models[hour] = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("model", model_class(**self.model_kwargs)),
                        ]
                    )
                else:
                    self.models[hour] = model_class(**self.model_kwargs)
        else:
            # Single model for all hours
            if self.standardize:
                self.model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", model_class(**self.model_kwargs)),
                    ]
                )
            else:
                self.model = model_class(**self.model_kwargs)

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        y: Union[np.ndarray, torch.Tensor, pd.Series],
        X_val: Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor, pd.Series]] = None,
    ) -> Dict[str, Any]:
        """
        Train the linear model(s).

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Training metrics and history
        """
        # Convert inputs to pandas/numpy for consistent handling
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.feature_names = (
            X.columns.tolist()
            if hasattr(X, "columns")
            else [f"feature_{i}" for i in range(X.shape[1])]
        )

        training_metrics = {}

        if self.hour_specific:
            # Train separate model for each hour
            # Assumes X has 'hour' column and y is aligned
            if "hour" not in X.columns:
                raise ValueError("Hour-specific modeling requires 'hour' column in X")

            for hour in range(24):
                hour_mask = X["hour"] == hour
                X_hour = X[hour_mask].drop("hour", axis=1)
                y_hour = y[hour_mask]

                if len(X_hour) > 0:  # Only train if we have data for this hour
                    self.models[hour].fit(X_hour, y_hour)

                    # Calculate training metrics for this hour
                    y_pred = self.models[hour].predict(X_hour)
                    training_metrics[f"hour_{hour:02d}_mse"] = mean_squared_error(
                        y_hour, y_pred
                    )
                    training_metrics[f"hour_{hour:02d}_r2"] = r2_score(y_hour, y_pred)
        else:
            # Train single model
            self.model.fit(X, y)

            # Calculate training metrics
            y_pred = self.model.predict(X)
            training_metrics["train_mse"] = mean_squared_error(y, y_pred)
            training_metrics["train_r2"] = r2_score(y, y_pred)

        # Validation metrics if provided
        self.is_trained = True  # Set this before validation evaluation

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            training_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        self.training_history = training_metrics

        return training_metrics

    def predict(self, X: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the trained model(s).

        Args:
            X: Features for prediction

        Returns:
            Predictions as numpy array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Convert inputs
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        if self.hour_specific:
            # Predict with hour-specific models
            predictions = np.zeros(len(X))

            if "hour" not in X.columns:
                raise ValueError("Hour-specific prediction requires 'hour' column in X")

            for hour in range(24):
                hour_mask = X["hour"] == hour
                X_hour = X[hour_mask].drop("hour", axis=1)

                if len(X_hour) > 0 and hour in self.models:
                    predictions[hour_mask] = self.models[hour].predict(X_hour)

            return predictions
        else:
            # Single model prediction
            return self.model.predict(X)

    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save trained model to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "config": self.config,
            "feature_names": getattr(self, "feature_names", None),
            "training_history": self.training_history,
            "is_trained": self.is_trained,
        }

        if self.hour_specific:
            save_data["models"] = self.models
        else:
            save_data["model"] = self.model

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

    def load_model(self, filepath: Union[str, Path]) -> None:
        """Load trained model from file"""
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        self.config = save_data["config"]
        self.feature_names = save_data.get("feature_names", None)
        self.training_history = save_data["training_history"]
        self.is_trained = save_data["is_trained"]

        if self.hour_specific:
            self.models = save_data["models"]
        else:
            self.model = save_data["model"]

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance/coefficients from trained model(s).

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        if self.hour_specific:
            # Average coefficients across all hour models
            all_coefs = []
            for hour, model in self.models.items():
                if hasattr(model, "coef_"):
                    coefs = model.coef_
                elif hasattr(model, "named_steps"):
                    coefs = model.named_steps["model"].coef_
                else:
                    continue
                all_coefs.append(coefs)

            if all_coefs:
                avg_coefs = np.mean(all_coefs, axis=0)
                return dict(zip(self.feature_names, avg_coefs))
            else:
                return {}
        else:
            # Single model coefficients
            if hasattr(self.model, "coef_"):
                coefs = self.model.coef_
            elif hasattr(self.model, "named_steps"):
                coefs = self.model.named_steps["model"].coef_
            else:
                return {}

            return dict(zip(self.feature_names, coefs))
