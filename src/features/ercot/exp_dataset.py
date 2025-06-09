"""Base class for ERCOT experiment datasets."""

import os
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd


class ExpDataset(ABC):
    """Base class for experiment datasets.

    This class provides the foundation for creating and managing experiment-specific
    datasets. It handles data loading, feature generation, and dataset management.

    Each experiment should subclass this and implement the abstract methods to define
    its specific data processing needs.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        input_files: List[str],
        experiment_id: str,
    ):
        """Initialize the experiment dataset.

        Args:
            input_dir: Directory containing processed input data files
            output_dir: Directory where experiment datasets will be saved
            input_files: List of required input filenames
            experiment_id: Unique identifier for this experiment (e.g., 'exp0')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input_files = input_files
        self.experiment_id = experiment_id

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data containers
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.dependent_vars: Optional[pd.DataFrame] = None
        self.independent_vars: Optional[pd.DataFrame] = None
        self.study_data: Optional[pd.DataFrame] = None

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required input data files.

        Returns:
            Dictionary mapping file identifiers to their DataFrames
        """
        for file_name in self.input_files:
            file_path = self.input_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required input file not found: {file_path}")

            # Store data with a clean identifier (remove file extension and suffixes)
            identifier = file_name.split(".")[0]  # Remove extension
            identifier = identifier.split("_clean")[0]  # Remove _clean suffix
            identifier = identifier.split("_transformed")[
                0
            ]  # Remove _transformed suffix

            self.raw_data[identifier] = pd.read_csv(file_path)

        return self.raw_data

    @abstractmethod
    def generate_dependent_vars(self):
        """Generate dependent variables (targets) for the experiment.

        This method should be implemented by each experiment to define its
        specific target variables. The method should store the result in
        self.study_data or update it appropriately.
        """
        pass

    @abstractmethod
    def generate_independent_vars(self):
        """Generate independent variables (features) for the experiment.

        This method should be implemented by each experiment to define its
        specific feature engineering logic. The method should update
        self.study_data with the additional features.
        """
        pass

    @abstractmethod
    def run_eda(self):
        """Run exploratory data analysis on the dataset.

        This method should be implemented by each experiment to define its
        specific EDA requirements.
        """
        pass
