"""
DART Dataset Class

PyTorch-style dataset for DART prediction tasks.
Handles feature-target alignment and operational constraints.
"""

from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DARTDataset(Dataset):
    """
    PyTorch Dataset for DART prediction.

    Handles the operational constraint: predict T+18 to T+42 hours ahead
    based on data available at 6AM cutoff.
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        y: Union[pd.Series, np.ndarray, torch.Tensor],
        transform: Optional[callable] = None,
        return_tensors: bool = True,
    ):
        """
        Initialize DART dataset.

        Args:
            X: Features (n_samples, n_features)
            y: Targets (n_samples,)
            transform: Optional transform to apply to features
            return_tensors: If True, return PyTorch tensors; else numpy arrays
        """
        self.transform = transform
        self.return_tensors = return_tensors

        # Convert to numpy arrays for consistent handling
        if isinstance(X, pd.DataFrame):
            self.X = X.values.astype(np.float32)
            self.feature_names = X.columns.tolist()
        elif isinstance(X, torch.Tensor):
            self.X = X.detach().cpu().numpy().astype(np.float32)
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            self.X = np.array(X, dtype=np.float32)
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            self.y = y.values.astype(np.float32)
        elif isinstance(y, torch.Tensor):
            self.y = y.detach().cpu().numpy().astype(np.float32)
        else:
            self.y = np.array(y, dtype=np.float32)

        # Validate shapes
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have same length: {len(self.X)} != {len(self.y)}"
            )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, target)
        """
        x = self.X[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.return_tensors:
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

        return x, y

    def get_feature_names(self) -> list:
        """Get feature names"""
        return self.feature_names.copy()

    def to_dataloader(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """
        Create a PyTorch DataLoader from this dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            **kwargs: Additional DataLoader arguments

        Returns:
            PyTorch DataLoader
        """
        from torch.utils.data import DataLoader

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)
