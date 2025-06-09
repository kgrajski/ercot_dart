"""
Base Model Adapter Interface

Defines the common interface that all model adapters must implement.
This ensures consistency across different modeling approaches and makes it easy to swap models.
"""

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch


class BaseModelAdapter(ABC):
    """
    Abstract base class for all model adapters.

    Provides common interface for:
    - Training and prediction
    - Model saving/loading
    - Performance evaluation
    - Hyperparameter management
    """

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.training_history = {}

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        y: Union[np.ndarray, torch.Tensor, pd.Series],
        X_val: Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor, pd.Series]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model on the given data.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Dictionary with training metrics and history
        """
        pass

    @abstractmethod
    def predict(
        self, X: Union[np.ndarray, torch.Tensor, pd.DataFrame]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Make predictions on new data.

        Args:
            X: Features for prediction

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save trained model to file"""
        pass

    @abstractmethod
    def load_model(self, filepath: Union[str, Path]) -> None:
        """Load trained model from file"""
        pass

    def evaluate(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        y: Union[np.ndarray, torch.Tensor, pd.Series],
    ) -> Dict[str, float]:
        """
        Evaluate model performance on given data.

        Args:
            X: Features
            y: True targets

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        predictions = self.predict(X)

        # Convert to numpy for consistent metric calculation
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(y, pd.Series):
            y = y.values

        # Calculate standard regression metrics
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score

        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mse)

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "model_name": self.model_name,
            "config": self.config,
            "is_trained": self.is_trained,
            "training_history": self.training_history,
        }
