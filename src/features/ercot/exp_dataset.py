"""Base class for ERCOT experiment datasets."""

import os
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


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
        - Label encode columns in label_encode_columns (creates new _le columns)
        - Update independent_vars list to include new feature columns and remove originals

        Args:
            label_encode_columns: list of columns to label encode (default None)
            one_hot_columns: list of columns to one-hot encode (default None)
        """
        if self.study_data is None:
            raise ValueError("No study data available for encoding.")

        label_encode_columns = label_encode_columns or []
        one_hot_columns = one_hot_columns or []

        new_label_columns = []
        remove_columns = []

        # Label encoding: create new columns, do not overwrite originals
        for cat_col in label_encode_columns:
            if cat_col in self.study_data.columns:
                le = LabelEncoder()
                new_col = cat_col + "_le"
                self.study_data[new_col] = le.fit_transform(
                    self.study_data[cat_col].astype(str)
                ).astype("float64")
                new_label_columns.append(new_col)
                remove_columns.append(cat_col)

        # One-hot encoding: as before
        new_one_hot_columns = []
        for cat_col in one_hot_columns:
            if cat_col in self.study_data.columns:
                one_hot_df = pd.get_dummies(self.study_data[cat_col], prefix=cat_col)
                self.study_data = pd.concat([self.study_data, one_hot_df], axis=1)
                new_one_hot_columns.extend(one_hot_df.columns.tolist())
                self.study_data.drop(columns=[cat_col], inplace=True)
                remove_columns.append(cat_col)

        # Update independent_vars to include new encoded features and remove originals
        if hasattr(self, "independent_vars"):
            original_independent_vars = self.independent_vars.copy()
            # Remove all encoded columns (label and one-hot)
            updated_independent_vars = [
                var for var in original_independent_vars if var not in remove_columns
            ]
            # Add new encoded columns (label and one-hot)
            for col in new_label_columns + new_one_hot_columns:
                if col not in updated_independent_vars:
                    updated_independent_vars.append(col)
            self._independent_vars_override = updated_independent_vars

    def append_z_transformed_independent_vars(self, summary_dir=None):
        """
        Appends Z-transformed versions of the independent variables to self.study_data,
        with a '_z' suffix. Saves pre-transformation statistics (mean, std, min, max) for each
        variable per (location, location_type) group to enable consistent transformation of new data.
        Z-transforming is performed within each (location, location_type) group.
        """
        df = self.study_data
        results = []
        summary_rows = []
        group_keys = ["location", "location_type"]
        independent_var_cols = self.independent_vars
        for (location, location_type), group in df.groupby(group_keys):
            group = group.copy()
            valid_cols = [col for col in independent_var_cols if col in group.columns]
            if not valid_cols:
                results.append(group)
                continue
            # Calculate and store pre-transformation statistics
            for col in valid_cols:
                summary_rows.append(
                    {
                        "location": location,
                        "location_type": location_type,
                        "column": col,
                        "mean": group[col].mean(),
                        "std": group[col].std(),
                        "min": group[col].min(),
                        "max": group[col].max(),
                    }
                )
            # Perform Z-transformation
            scaler = StandardScaler()
            z_values = scaler.fit_transform(group[valid_cols].values)
            z_cols = [f"{col}_z" for col in valid_cols]
            for i, col in enumerate(valid_cols):
                group[z_cols[i]] = z_values[:, i]
            results.append(group)
        df_out = pd.concat(results, ignore_index=True)
        self.study_data = df_out
        # Save pre-transformation statistics
        if summary_dir is not None:
            summary = pd.DataFrame(summary_rows)
            summary_path = Path(summary_dir) / "pre_z_transform_stats.csv"
            summary.to_csv(summary_path, index=False)
            print(f"\nSaved pre-transformation statistics: {summary_path}")
            print("\n=== Pre-Transformation Statistics Summary (by location) ===")
            print(summary)

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

    def _generate_random_noise_dataset(self):
        """
        Generate a random noise dataset D_n01 with the same utc_ts and independent variable structure as self.study_data.
        Each independent variable column is replaced with random N(0,1) values (suffix _n01),
        and Z-transformed per (location, location_type) group (suffix _n01_z).
        Returns a DataFrame with utc_ts, ..._n01, ..._n01_z.
        """

        df = self.study_data
        meta_cols = ["utc_ts", "location", "location_type"]
        n_rows = len(df)
        independent_var_cols = self.independent_vars
        noise_data = {
            col + "_n01": np.random.normal(0, 1, n_rows) for col in independent_var_cols
        }
        noise_df = pd.DataFrame(noise_data)
        for col in meta_cols:
            if col in df.columns:
                noise_df[col] = df[col].values
        results = []
        group_keys = ["location", "location_type"]
        for (location, location_type), group in noise_df.groupby(group_keys):
            group = group.copy()
            valid_cols = [col for col in noise_data.keys() if col in group.columns]
            if not valid_cols:
                results.append(group)
                continue
            scaler = StandardScaler()
            z_values = scaler.fit_transform(group[valid_cols].values)
            z_cols = [f"{col}_z" for col in valid_cols]
            for i, col in enumerate(valid_cols):
                group[z_cols[i]] = z_values[:, i]
            results.append(group)
        noise_df_out = pd.concat(results, ignore_index=True)
        keep_cols = (
            meta_cols
            + list(noise_data.keys())
            + [f"{col}_z" for col in noise_data.keys()]
        )
        keep_cols = [col for col in keep_cols if col in noise_df_out.columns]
        return noise_df_out[keep_cols]
