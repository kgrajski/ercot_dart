"""Base ETL module for ERCOT data."""

import os
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd

from src.data.ercot.database import DatabaseProcessor


class ERCOTBaseETL:
    """Base class for ERCOT data ETL operations."""

    def __init__(self, data_dir: str, output_dir: str, db_path: Optional[str] = None):
        """Initialize ETL handler.

        Args:
            data_dir (str): Directory containing raw CSV data files
            output_dir (str): Directory where cleaned data will be saved
            db_path (str, optional): Path to SQLite database file
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_processor = DatabaseProcessor(db_path) if db_path else None

    def get_raw_data(self, endpoint_key: str) -> pd.DataFrame:
        """Get raw data from CSV file.

        Args:
            endpoint_key (str): The endpoint identifier (e.g., "dam_spp")

        Returns:
            pd.DataFrame: Raw data from CSV
        """
        # Find the CSV file
        csv_file = self.data_dir / f"{endpoint_key}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"No CSV file found for endpoint {endpoint_key}")

        print(f"Reading raw data from: {csv_file}")

        # Read CSV with appropriate type handling, automatically parsing timestamp columns
        df = pd.read_csv(csv_file, dtype_backend="numpy_nullable")

        # Debug: Check what we actually have
        print(f"Column names: {list(df.columns)}")
        print(f"Sample utc_ts values: {df['utc_ts'].head(2).tolist()}")

        # Explicitly convert datetime columns
        df["utc_ts"] = pd.to_datetime(df["utc_ts"])
        df["local_ts"] = pd.to_datetime(df["local_ts"])

        print(f"After conversion - utc_ts type: {type(df['utc_ts'][0])}")
        return df

    def save_clean_data(self, df: pd.DataFrame, endpoint_key: str):
        """Save cleaned data to CSV and database.

        Args:
            df (pd.DataFrame): Cleaned DataFrame to save
            endpoint_key (str): The endpoint identifier
        """
        # Create filename with endpoint key and "clean" indicator
        filename = f"{endpoint_key}_clean.csv"
        filepath = self.output_dir / filename

        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Saved cleaned data to: {filepath}")

        # Save to database if configured
        if self.db_processor:
            self.save_to_database(df, f"{endpoint_key}_clean")

    def save_transformed_data(self, df: pd.DataFrame, endpoint_key: str):
        """Save transformed data to CSV and database.

        Args:
            df (pd.DataFrame): Transformed DataFrame to save
            endpoint_key (str): The endpoint identifier
        """
        # Create filename with endpoint key and "transformed" indicator
        filename = f"{endpoint_key}_transformed.csv"
        filepath = self.output_dir / filename

        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Saved transformed data to: {filepath}")

        # Save to database if configured
        if self.db_processor:
            self.save_to_database(df, f"{endpoint_key}_transformed")

    def save_to_database(self, df: pd.DataFrame, endpoint_key: str):
        """Save data to database with ETL-specific handling.
        This method can be overridden by subclasses to provide custom
        data preparation before saving to database.

        Args:
            df (pd.DataFrame): DataFrame to save
            endpoint_key (str): Key identifying the endpoint/table
        """
        if self.db_processor:
            self.db_processor.save_to_database(df, endpoint_key)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data into standardized format.
        This method should be implemented by each client class.

        Args:
            df (pd.DataFrame): Raw DataFrame to clean

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        raise NotImplementedError("Subclasses must implement clean_data()")

    def transform_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transform cleaned data into final format.
        This method can be implemented by client classes that need additional transformation.

        Args:
            df (pd.DataFrame): Cleaned DataFrame to transform

        Returns:
            Optional[pd.DataFrame]: Transformed DataFrame, or None if no transformation needed
        """
        return None

    def validate_temporal_completeness(self, df: pd.DataFrame) -> bool:
        """Validate that every date has all 24 UTC hours.

        Args:
            df (pd.DataFrame): DataFrame with utc_ts column

        Returns:
            bool: True if all dates have complete 24-hour coverage
        """

        # Extract date and hour components
        df_temp = df.copy()
        df_temp["date"] = df_temp["utc_ts"].dt.date
        df_temp["hour"] = df_temp["utc_ts"].dt.hour

        # Group by date and count unique hours
        date_hour_counts = df_temp.groupby("date")["hour"].nunique()

        # Given the maximum and minimum dates, calculate expected number of dates and check if correct
        min_date = min(df_temp["date"])
        max_date = max(df_temp["date"])
        expected_days = (max_date - min_date).days + 1
        actual_days = len(df_temp["date"].unique())

        print(f"Date range: {min_date} to {max_date}")
        print(f"Expected days: {expected_days}, Actual days: {actual_days}")

        if actual_days != expected_days:
            print(f"Error: Found {actual_days} dates, expected {expected_days} dates")
            return False

        # Check if any date has less than 24 hours
        incomplete_dates = date_hour_counts[date_hour_counts != 24]

        if len(incomplete_dates) > 0:
            for date, hour_count in incomplete_dates.items():
                if (date == min(df_temp["date"])) or (date == max(df_temp["date"])):
                    print(f"Warning: {date}: {hour_count} hours")
                    return_flag = True
                else:
                    print(
                        f"Error: Found {len(incomplete_dates)} dates with incomplete hours"
                    )
                    return_flag = False
            return return_flag

        print(
            f"Temporal completeness validated: {len(date_hour_counts)} dates with complete 24-hour coverage"
        )
        return True

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate cleaned data meets requirements.
        This method should be implemented by each client class.

        Args:
            df (pd.DataFrame): Cleaned DataFrame to validate

        Returns:
            bool: True if validation passes
        """
        raise NotImplementedError("Subclasses must implement validate_data()")

    def transform(self) -> pd.DataFrame:
        """Execute the full ETL process.

        Returns:
            pd.DataFrame: Cleaned and validated data, or transformed data if transformation is implemented
        """
        # Get raw data
        df_raw = self.get_raw_data(self.ENDPOINT_KEY)

        # Clean data
        df_clean = self.clean_data(df_raw)

        # Validate temporal completeness first (if this fails, no point in detailed validation)
        if not self.validate_temporal_completeness(df_clean):
            raise ValueError("Temporal completeness validation failed")

        # Run client-specific validation
        if not self.validate_data(df_clean):
            raise ValueError("Data validation failed")

        # Save cleaned data
        self.save_clean_data(df_clean, self.ENDPOINT_KEY)

        # Transform data if implemented
        df_transformed = self.transform_data(df_clean)
        if df_transformed is not None:
            self.save_transformed_data(df_transformed, self.ENDPOINT_KEY)
            return df_transformed

        return df_clean
