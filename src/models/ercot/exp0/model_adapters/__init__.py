"""
Model Adapters for ERCOT DART Modeling

This module provides a unified interface for different model types through
the adapter pattern. Each model type (linear, tree-based, neural networks)
implements the same interface for consistent training and evaluation.

The adapter pattern allows easy swapping between different model types
while maintaining the same training pipeline.
"""

from src.models.ercot.exp0.model_adapters.base_adapter import BaseModelAdapter
from src.models.ercot.exp0.model_adapters.linear_models import LinearModelAdapter
from src.models.ercot.exp0.model_adapters.tree_models import TreeModelAdapter

# Model registry for easy instantiation
MODEL_REGISTRY = {
    "linear": LinearModelAdapter,
    "ridge": LinearModelAdapter,
    "lasso": LinearModelAdapter,
    "elastic_net": LinearModelAdapter,
    "random_forest": TreeModelAdapter,
    "gradient_boosting": TreeModelAdapter,
    "xgboost": TreeModelAdapter,
}


def get_model_adapter(model_type: str, **kwargs) -> BaseModelAdapter:
    """Factory function to create model adapters.

    Args:
        model_type: Type of model ('linear', 'ridge', 'random_forest', etc.)
        **kwargs: Additional arguments passed to the adapter

    Returns:
        Configured model adapter instance
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    adapter_class = MODEL_REGISTRY[model_type]
    return adapter_class(model_type=model_type, **kwargs)


__all__ = [
    "BaseModelAdapter",
    "LinearModelAdapter",
    "TreeModelAdapter",
    "MODEL_REGISTRY",
    "get_model_adapter",
]
