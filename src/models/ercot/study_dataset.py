"""Base class for ERCOT study datasets."""

import os
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from torch.utils.data import Dataset

from src.data.ercot.database import DatabaseProcessor


class StudyDataset(Dataset, ABC):
    """Abstract base class for ERCOT study datasets.

    This class provides the foundation for creating PyTorch-compatible datasets
    from processed study data. It combines the PyTorch Dataset interface with
    utilities for saving data to CSV and database formats.

    Each experiment should subclass this and implement the abstract methods to define
    its specific data loading and processing needs.
    """

    def __init__(
        self,
        experiment_id: str,
        output_dir: Optional[str] = None,
    ):
        """Initialize the study dataset.

        Args:
            experiment_id: Unique identifier for this experiment (e.g., 'exp0')
            output_dir: Directory where outputs will be saved. If None, uses current directory
        """
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data container
        self.df: Optional[pd.DataFrame] = None

    @abstractmethod
    def __len__(self):
        """Return the size of the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get the list of feature column names."""
        pass

    def save_to_csv(self, filename: str, df: Optional[pd.DataFrame] = None) -> Path:
        """Save dataset to CSV file.

        Args:
            filename: Name of the CSV file (with or without .csv extension)
            df: DataFrame to save. If None, uses self.df

        Returns:
            Path: Full path to the saved file
        """
        if df is None:
            if self.df is None:
                raise ValueError("No DataFrame available to save. Load data first.")
            df = self.df

        # Ensure .csv extension
        if not filename.endswith(".csv"):
            filename += ".csv"

        file_path = self.output_dir / filename
        df.to_csv(file_path, index=False)

        print(f"Saved {len(df)} records to {file_path}")
        return file_path

    def save_to_database(
        self,
        table_name: str,
        df: Optional[pd.DataFrame] = None,
        db_filename: Optional[str] = None,
    ) -> Path:
        """Save dataset to SQLite database.

        Args:
            table_name: Name of the database table
            df: DataFrame to save. If None, uses self.df
            db_filename: Database filename. If None, uses experiment_id.db

        Returns:
            Path: Full path to the database file
        """
        if df is None:
            if self.df is None:
                raise ValueError("No DataFrame available to save. Load data first.")
            df = self.df

        # Set default database filename
        if db_filename is None:
            db_filename = f"{self.experiment_id}_dataset.db"
        if not db_filename.endswith(".db"):
            db_filename += ".db"

        db_path = self.output_dir / db_filename

        # Use DatabaseProcessor for saving
        db_processor = DatabaseProcessor(str(db_path))
        db_processor.save_to_database(df, table_name)

        return db_path

    def load_from_database(
        self,
        table_name: str,
        db_filename: Optional[str] = None,
        datetime_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load dataset from SQLite database.

        Args:
            table_name: Name of the database table
            db_filename: Database filename. If None, uses experiment_id.db
            datetime_columns: List of columns to convert to datetime

        Returns:
            pd.DataFrame: Loaded dataset
        """
        # Set default database filename
        if db_filename is None:
            db_filename = f"{self.experiment_id}_dataset.db"
        if not db_filename.endswith(".db"):
            db_filename += ".db"

        db_path = self.output_dir / db_filename

        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        # Use DatabaseProcessor for loading
        db_processor = DatabaseProcessor(str(db_path))
        df = db_processor.read_from_database(table_name, datetime_columns)

        print(f"Loaded {len(df)} records from {table_name} table in {db_path}")
        return df
