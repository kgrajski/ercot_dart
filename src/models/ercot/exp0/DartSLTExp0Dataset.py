import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.ercot.study_dataset import StudyDataset


class DartSltExp0Dataset(StudyDataset):
    """PyTorch Dataset for ERCOT DART SLT Experiment 0.

    Simple dataset loader for final exp0 datasets. All categorical encoding
    should be done during data preparation (03-kag workflow).

    Each sample includes features, target, and metadata (timestamps, location).

    A critical assumption is that the dataset is of a single location/location type.
    Any batching or parallel processing of multiple locations will use methods outside
    the scope of this class.
    """

    def __init__(
        self,
        spp_dir: str,
        target_column: str = "dart_slt",
        feature_columns: list = None,
        metadata_columns: list = None,
        auto_create_model_ready: bool = True,
        output_dir: str = None,
    ):
        """Initialize the dataset.

        Args:
            spp_dir: Path to the settlement point directory containing final dataset
            target_column: Name of the target variable column
            feature_columns: List of feature column names. If None, uses default features
            metadata_columns: List of metadata columns to preserve. If None, uses defaults
            auto_create_model_ready: If True, automatically creates and saves model-ready dataset
            output_dir: Directory for model_ready outputs (modeling dir, not model_type subdir)
        """
        # Find the final dataset file automatically
        fnames = [
            f for f in os.listdir(spp_dir) if "final_dataset" in f and "exp0" in f
        ]
        if not fnames:
            raise FileNotFoundError(f"No final_dataset file found in {spp_dir}")
        if len(fnames) > 1:
            raise ValueError(
                f"Multiple final_dataset files found in {spp_dir}: {fnames}"
            )

        final_dataset_file_name = os.path.join(spp_dir, fnames[0])
        print(f"Final dataset file: {final_dataset_file_name}")

        # Pass output_dir to parent
        super().__init__(experiment_id="exp0", output_dir=output_dir)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

        # Load the dataset (no dtype specification - let pandas handle it)
        self.df = pd.read_csv(
            final_dataset_file_name,
            parse_dates=["utc_ts", "local_ts"],
            date_format="ISO8601",
        )

        # Strip timezone from utc_ts since it's always UTC anyway
        self.df["utc_ts"] = self.df["utc_ts"].dt.tz_localize(None)

        # Set target column
        self.target_column = target_column

        # Set feature columns
        if feature_columns is None:
            self.feature_columns = self._get_default_features()
        else:
            self.feature_columns = feature_columns

        # Set metadata columns
        if metadata_columns is None:
            self.metadata_columns = [
                "utc_ts",
                "local_ts",
                "end_of_hour",
                "location",
                "location_type",
            ]
        else:
            self.metadata_columns = metadata_columns

        # Convert all feature columns to float64 for PyTorch compatibility
        for col in self.feature_columns:
            self.df[col] = self.df[col].astype("float64")

        # Automatically create model-ready dataset if requested
        if auto_create_model_ready:
            print(f"** Creating model-ready dataset automatically")
            csv_path, db_path = self.create_model_ready_dataset()

            # Store paths for later reference
            self.model_ready_csv_path = csv_path
            self.model_ready_db_path = db_path

    def _get_default_features(self):
        """Get default feature columns including one-hot encoded day_of_week."""
        # Lag features
        lag_features = [
            "dart_slt_lag_24hr",
            "dart_slt_lag_25hr",
            "dart_slt_lag_26hr",
            "dart_slt_lag_168hr",
        ]

        # Rolling features
        roll_features = [
            "dart_slt_roll_mean_24hr",
            "dart_slt_roll_sdev_24hr",
            "dart_slt_roll_mean_168hr",
            "dart_slt_roll_sdev_168hr",
        ]

        # Time features (boolean and one-hot encoded)
        time_features = ["is_weekend", "is_holiday"]

        base_features = lag_features + roll_features + time_features

        # Add day_of_week one-hot encoded columns (dynamically detect them)
        day_of_week_columns = [
            col for col in self.df.columns if col.startswith("day_of_week_")
        ]
        base_features.extend(day_of_week_columns)

        return base_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row
        row = self.df.iloc[idx]

        # Extract features and convert to tensor
        features = torch.tensor(
            row[self.feature_columns].to_numpy(dtype="float64"), dtype=torch.float32
        )

        # Extract target and convert to tensor
        target = torch.tensor(row[self.target_column], dtype=torch.float32)

        # Extract metadata (keep as original types)
        metadata = {col: row[col] for col in self.metadata_columns}

        return {
            "features": features.clone().detach(),
            "target": target.clone().detach(),
            "metadata": metadata,
        }

    def get_feature_names(self):
        """Get feature column names."""
        return self.feature_columns.copy()

    def create_model_ready_dataset(self, filename_base: str = "model_ready"):
        """Create and save a model-ready dataset by reassembling data through __getitem__.

        This method iterates through all samples using __getitem__, reassembles the
        metadata, features, and targets into a single dataframe, and saves it to
        both CSV and database formats.

        Args:
            filename_base: Base name for output files (without extension)

        Returns:
            tuple: (csv_path, db_path) - Paths to saved files
        """
        print(f"Creating model-ready dataset from {len(self)} samples...")

        # Collect all data by iterating through the dataset
        reassembled_data = []

        for idx in range(len(self)):
            sample = self[idx]  # Uses __getitem__

            # Extract data from the sample
            features_tensor = sample["features"]
            target_tensor = sample["target"]
            metadata = sample["metadata"]

            # Convert tensors back to values
            features_values = features_tensor.numpy()
            target_value = target_tensor.item()

            # Create row dictionary starting with metadata
            row_data = metadata.copy()

            # Add target
            row_data[self.target_column] = target_value

            # Add features with their names
            for i, feature_name in enumerate(self.feature_columns):
                row_data[feature_name] = features_values[i]

            reassembled_data.append(row_data)

        # Create DataFrame from reassembled data
        model_ready_df = pd.DataFrame(reassembled_data)

        # Reorder columns: metadata first, then target, then features
        ordered_columns = (
            self.metadata_columns + [self.target_column] + self.feature_columns
        )
        model_ready_df = model_ready_df[ordered_columns]

        print(f"Reassembled dataset shape: {model_ready_df.shape}")

        # Save to CSV and database in the correct output_dir
        csv_path = self.save_to_csv(filename_base, model_ready_df)
        db_path = self.save_to_database(filename_base, model_ready_df, filename_base)

        return csv_path, db_path
