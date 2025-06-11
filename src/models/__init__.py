"""Models package for ERCOT DART prediction.

This package contains machine learning models and experiments
for predicting DART (Day-Ahead Real-Time) price differences.

Structure:
- ercot/: ERCOT-specific models and experiments
  - study_dataset.py: Base class for study datasets
  - exp0/: Experiment 0 components
"""

from src.models.ercot.study_dataset import StudyDataset

__all__ = ["StudyDataset"]
