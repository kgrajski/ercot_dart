"""Base class for ERCOT experiment datasets."""

import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from abc import ABC, abstractmethod


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
        experiment_id: str
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
        self.study_dataset: Optional[pd.DataFrame] = None
        
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
            identifier = file_name.split('.')[0]  # Remove extension
            identifier = identifier.split('_clean')[0]  # Remove _clean suffix
            identifier = identifier.split('_transformed')[0]  # Remove _transformed suffix
            
            self.raw_data[identifier] = pd.read_csv(file_path)
            
        return self.raw_data
    
    @abstractmethod
    def generate_dependent_vars(self) -> pd.DataFrame:
        """Generate dependent variables (targets) for the experiment.
        
        This method should be implemented by each experiment to define its
        specific target variables.
        
        Returns:
            DataFrame containing dependent variables
        """
        pass
    
    @abstractmethod
    def generate_independent_vars(self) -> pd.DataFrame:
        """Generate independent variables (features) for the experiment.
        
        This method should be implemented by each experiment to define its
        specific feature engineering logic.
        
        Returns:
            DataFrame containing independent variables
        """
        pass
    
    def save_dataset(self, dataset_type: str = 'study'):
        """Save the experiment dataset.
        
        Args:
            dataset_type: Type of dataset to save ('study', 'train', 'test', etc.)
        """
        if self.dependent_vars is None or self.independent_vars is None:
            raise ValueError("Must generate dependent and independent variables before saving")
            
        # Combine features and targets
        self.study_dataset = pd.concat([
            self.independent_vars,
            self.dependent_vars
        ], axis=1)
        
        # Create filename with experiment ID and dataset type
        filename = f"{self.experiment_id}_{dataset_type}_dataset.csv"
        filepath = self.output_dir / filename
        
        # Save to CSV
        self.study_dataset.to_csv(filepath, index=False)
        print(f"Saved {dataset_type} dataset to: {filepath}")
        
    def load_dataset(self, dataset_type: str = 'study') -> pd.DataFrame:
        """Load a previously saved experiment dataset.
        
        Args:
            dataset_type: Type of dataset to load ('study', 'train', 'test', etc.)
            
        Returns:
            The loaded dataset as a DataFrame
        """
        filename = f"{self.experiment_id}_{dataset_type}_dataset.csv"
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
            
        return pd.read_csv(filepath)
    
    @abstractmethod
    def run_eda(self):
        """Run exploratory data analysis on the dataset.
        
        This method should be implemented by each experiment to define its
        specific EDA requirements.
        """
        pass
    
    def validate_dataset(self) -> bool:
        """Validate the final dataset meets requirements.
        
        Returns:
            True if validation passes
        """
        if self.study_dataset is None:
            raise ValueError("No dataset available to validate")
            
        # Basic validation checks
        if len(self.study_dataset) == 0:
            return False
            
        # Check for any missing values
        if self.study_dataset.isnull().any().any():
            print("Warning: Dataset contains missing values")
            
        return True 