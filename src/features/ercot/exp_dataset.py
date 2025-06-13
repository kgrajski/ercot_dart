"""Base class for ERCOT experiment datasets."""

import os
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
        self.independent_vars_data: Optional[pd.DataFrame] = None
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

    def _apply_categorical_encoding(
        self, label_encode_columns=None, one_hot_columns=None
    ):
        """
        Apply categorical encoding to study data.

        - One-hot encode columns in one_hot_columns
        - Label encode columns in label_encode_columns
        - Update independent_vars list to include new feature columns

        Args:
            label_encode_columns: list of columns to label encode (default None)
            one_hot_columns: list of columns to one-hot encode (default None)
        """
        if self.study_data is None:
            raise ValueError("No study data available for encoding.")

        label_encode_columns = label_encode_columns or []
        one_hot_columns = one_hot_columns or []

        # Apply label encoding to specified columns
        for cat_col in label_encode_columns:
            if cat_col in self.study_data.columns:
                le = LabelEncoder()
                self.study_data[cat_col] = le.fit_transform(
                    self.study_data[cat_col].astype(str)
                ).astype("float64")

        # Apply one-hot encoding to specified columns
        new_one_hot_columns = []
        for cat_col in one_hot_columns:
            if cat_col in self.study_data.columns:
                one_hot_df = pd.get_dummies(self.study_data[cat_col], prefix=cat_col)
                self.study_data = pd.concat([self.study_data, one_hot_df], axis=1)
                new_one_hot_columns.extend(one_hot_df.columns.tolist())
                self.study_data.drop(columns=[cat_col], inplace=True)

        # Update independent_vars to include new one-hot features and remove original
        if hasattr(self, "independent_vars"):
            original_independent_vars = self.independent_vars.copy()
            updated_independent_vars = [
                var for var in original_independent_vars if var not in one_hot_columns
            ]
            updated_independent_vars.extend(new_one_hot_columns)
            self._independent_vars_override = updated_independent_vars

    def _create_safe_identifier(self, location: str, location_type: str) -> str:
        """Create a safe filename identifier from location and location_type.

        Uses the pattern: "location (location_type)", then applies safe filename transformations.

        Args:
            location: Location name
            location_type: Location type (e.g., 'LZ', 'HB')

        Returns:
            str: Safe identifier for use in filenames and directories
        """
        point_identifier = f"{location} ({location_type})"
        safe_identifier = (
            point_identifier.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        return safe_identifier

    def _validate_temporal_completeness_single(
        self, df: pd.DataFrame, location: str, location_type: str
    ) -> bool:
        """Validate temporal completeness for a single location+location_type.

        Args:
            df: DataFrame for single location+location_type
            location: Location name
            location_type: Location type

        Returns:
            bool: True if temporal completeness validation passes
        """
        # Extract date and hour components
        df_temp = df.copy()
        df_temp["date"] = df_temp["utc_ts"].dt.date
        df_temp["hour"] = df_temp["utc_ts"].dt.hour

        # Group by date and count unique hours
        date_hour_counts = df_temp.groupby("date")["hour"].nunique()

        # Calculate expected vs actual days
        min_date = min(df_temp["date"])
        max_date = max(df_temp["date"])
        expected_days = (max_date - min_date).days + 1
        actual_days = len(df_temp["date"].unique())

        print(f"    Date range: {min_date} to {max_date}")
        print(f"    Expected days: {expected_days}, Actual days: {actual_days}")

        if actual_days != expected_days:
            print(
                f"    ERROR: Found {actual_days} dates, expected {expected_days} dates"
            )
            return False

        # Check if any date has less than 24 hours
        incomplete_dates = date_hour_counts[date_hour_counts != 24]

        if len(incomplete_dates) > 0:
            for date, hour_count in incomplete_dates.items():
                if (date == min_date) or (date == max_date):
                    print(f"    WARNING: {date}: {hour_count} hours (boundary date)")
                else:
                    print(f"    ERROR: {date}: {hour_count} hours (should be 24)")
                    return False

            # If only boundary dates have issues, still pass
            non_boundary_incomplete = [
                date
                for date in incomplete_dates.index
                if date != min_date and date != max_date
            ]
            if len(non_boundary_incomplete) > 0:
                return False

        print(
            f"    Temporal completeness: ✓ {len(date_hour_counts)} dates with proper hourly coverage"
        )
        return True
