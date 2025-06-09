"""
ERCOT Experiment 0 Modeling Package

This package provides PyTorch-based modeling infrastructure for DART price prediction.
Follows speechBCI project patterns for model adapters, experiments, and utilities.

Key Components:
- model_adapters: Unified interface for different model types
- experiments: Experiment orchestration and configuration
- utils: Data processing, training, and evaluation utilities
- datasets: PyTorch dataset implementations
"""

from src.models.ercot.exp0.datasets import *
from src.models.ercot.exp0.model_adapters import *
from src.models.ercot.exp0.utils import *

__all__ = [
    # Re-export everything from submodules
]

__version__ = "0.1.0"
__experiment__ = "exp0"
