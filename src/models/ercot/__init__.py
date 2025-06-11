"""ERCOT-specific models and experiments.

This package contains machine learning models and experimental
frameworks specifically designed for ERCOT electricity market
data analysis and prediction.

Structure to be built:
- exp0/: Baseline experiments with linear models
- exp1/: Advanced experiments with tree-based models  
- exp2/: Deep learning experiments
"""

from src.models.ercot.study_dataset import StudyDataset

__all__ = ["StudyDataset"]
