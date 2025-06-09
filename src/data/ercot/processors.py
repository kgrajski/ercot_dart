"""Data processing module for ERCOT API responses."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
import pytz
from tqdm import tqdm


class ERCOTProcessor:
    """Processor for ERCOT API response data.

    Timestamp Handling:
    - All timestamps are processed into canonical format
    - local_ts: America/Chicago timezone-aware (NaT for ambiguous times)
    - utc_ts: UTC timezone-naive (canonical reference)
    - is_dst_flag: Boolean DST indicator
    - hour_local: Local hour preserved for business logic

    Usage:
    - Always use utc_ts for data processing pipelines
    - Use local_ts and hour_local only for business logic/display
    - is_dst_flag preserved for validation and reference
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize processor.

        Args:
            output_dir (str, optional): Directory where CSV files will be saved.
                                      If None, CSV files will not be saved.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _set_start_delivery_hour(self, hour_ending):
        """Convert ERCOT hour ending to an internal working delivery hour starting as an integer.

        ERCOT uses various formats for hour ending, including:
        - 1-24 integer (rt_spp, solar_generation, wind_generation, etc.)
        - 01:00 - 24:00 (dam_spp, dam_system_lambda, load_forecast, etc.)

        Convert the hour ending to an integer and then subtract one hour to get the delivery starting hour.`

        The return value is an integer.
        """

        if isinstance(hour_ending, str):
            hour_str = hour_ending.strip()
            if ":" in hour_str:
                start_delivery_hour = int(hour_str.split(":")[0]) - 1
                return start_delivery_hour
        elif isinstance(hour_ending, int):
            hour_num = int(hour_ending)
            if 1 <= hour_num <= 24:
                start_delivery_hour = hour_num - 1
                return start_delivery_hour
            else:
                raise ValueError(f"Hour {hour_num} not in valid range 1-24")
        else:
            raise ValueError(f"hour_ending must be a string or integer")

    def _localize_with_dst(self, row) -> pd.Timestamp:
        """Handle timezone localization with explicit DST logic.

        Args:
            row: DataFrame row containing "local_ts", "hour_ending_std", and "DSTFlag"

        Returns:
            pd.Timestamp: Timezone-aware timestamp in America/Chicago

        Notes:
            - Only ERCOT hour ending "02:00" is ambiguous during Fall DST transition
            - DSTFlag=True means this is the repeated hour (standard time)
            - DSTFlag=False means this is the first occurrence (DST time)
        """
        naive_ts = row["start_delivery_datetime"]

        # Only hour ending 1 is ambiguous during Fall DST
        if row["start_delivery_hour"] == 1:
            # DSTFlag=True means this is the repeated hour (standard time)
            is_dst = not row["DSTFlag"]
            return naive_ts.tz_localize(
                "America/Chicago", ambiguous=is_dst, nonexistent="shift_forward"
            )
        else:
            return naive_ts.tz_localize(
                "America/Chicago", ambiguous=False, nonexistent="shift_forward"
            )

    def add_utc_timestamps(
        self, df: pd.DataFrame, interval_minutes: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Add standardized UTC timestamps and relevant metadata to DataFrame."""

        # For the conversion to UTC we need to work with the delivery starting hour.
        df["start_delivery_hour"] = df["hourEnding"].apply(
            self._set_start_delivery_hour
        )
        print("\n\n", df.head())

        # Calculate delivery start datetime by adding start_delivery_hour to deliveryDate
        df["start_delivery_datetime"] = df.apply(
            lambda row: row["deliveryDate"]
            + pd.Timedelta(hours=row["start_delivery_hour"]),
            axis=1,
        )
        print("\n\n", df.head())

        # Add interval minutes if provided (before timezone localization)
        if interval_minutes is not None:
            df["start_delivery_datetime"] = df[
                "start_delivery_datetime"
            ] + pd.to_timedelta(interval_minutes, unit="minutes")
        print("\n\n", df.head())

        # Apply DST-aware timezone localization row by row for clarity
        df["local_ts"] = df.apply(self._localize_with_dst, axis=1)
        print("\n\n", df.head())

        # Convert to canonical UTC timestamp (timezone-naive since UTC is inherently timezone-agnostic)
        df["utc_ts"] = df["local_ts"].dt.tz_convert("UTC")
        print("\n\n", df.head())

        # Drop temporary columns
        df = df.drop(columns=["start_delivery_datetime", "start_delivery_hour"])
        print("\n\n", df.head())

        return df

    def json_to_df(self, response_data: Dict) -> pd.DataFrame:
        """Convert JSON response data to DataFrame.

        Args:
            response_data (Dict): Dictionary containing response data with fields and data

        Returns:
            pd.DataFrame: Raw DataFrame with typed columns

        Raises:
            ValueError: If response data is missing required fields
        """
        # Validate response structure
        required_fields = ["fields", "data"]
        missing = [f for f in required_fields if f not in response_data]
        if missing:
            raise ValueError(
                f"Response data missing required fields: {', '.join(missing)}"
            )

        # Extract data and field definitions
        data = response_data["data"]
        fields = response_data["fields"]

        # Convert to DataFrame
        if not data:
            # If no data, create empty DataFrame with correct columns
            columns = [field["name"] for field in fields]
            return pd.DataFrame(columns=columns)

        # Create DataFrame from data
        columns = [field["name"] for field in fields]
        df = pd.DataFrame(data, columns=columns)

        # Apply data types based on field definitions
        for field in fields:
            column_name = field["name"]
            data_type = field.get("dataType", "").upper()

            if data_type == "BOOLEAN":
                df[column_name] = df[column_name].astype("boolean")
            elif data_type in ["INTEGER", "INT"]:
                df[column_name] = pd.to_numeric(
                    df[column_name], errors="coerce"
                ).astype("Int64")
            elif data_type in ["DECIMAL", "FLOAT", "DOUBLE"]:
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
            elif data_type in ["DATE", "DATETIME"]:
                df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
            elif data_type in ["VARCHAR", "STRING"]:
                df[column_name] = df[column_name].astype("string")

        return df

    def process_response(self, response) -> pd.DataFrame:
        """Process a single API response to DataFrame.

        Args:
            response: API response object

        Returns:
            pd.DataFrame: Raw DataFrame with typed columns

        Raises:
            ValueError: If response is missing required fields
        """
        # Parse JSON response
        json_response = response.json()

        # Validate response structure
        if "_meta" not in json_response:
            raise ValueError("Response missing _meta field")

        return self.json_to_df(json_response)

    def process_data(self, data: List[Dict]) -> pd.DataFrame:
        """Process a list of raw data records to DataFrame.

        Args:
            data (List[Dict]): List of data records

        Returns:
            pd.DataFrame: Raw DataFrame
        """
        if not data:
            return pd.DataFrame()

        # Create DataFrame
        return pd.DataFrame(data)

    def save_to_csv(self, df: pd.DataFrame, endpoint_key: str, params: Dict) -> Path:
        """Save DataFrame to CSV file.

        Args:
            df (pd.DataFrame): Data to save
            endpoint_key (str): Key identifying the endpoint
            params (Dict): Query parameters used to get the data

        Returns:
            Path: Path to saved CSV file

        Raises:
            ValueError: If output_dir is not set
        """
        if not self.output_dir:
            raise ValueError("output_dir must be set to save CSV files")

        # Create simple filename without timestamp
        filename = f"{endpoint_key}.csv"
        filepath = self.output_dir / filename

        # Save DataFrame
        df.to_csv(filepath, index=False)

        return filepath

    def verify_data(
        self, df: pd.DataFrame, endpoint_key: str, csv_file: Optional[Path] = None
    ):
        """Verify and report on the processed data.

        Args:
            df (pd.DataFrame): Processed data
            endpoint_key (str): Key identifying the endpoint
            csv_file (Path, optional): Path to saved CSV file
        """
        # Verify canonical timestamp columns
        required_cols = {"utc_ts", "local_ts", "hour_local"}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"\nWarning: Missing canonical timestamp columns: {missing}")

        # Verify timezone awareness for local timestamps
        if "local_ts" in df.columns:
            tz = df["local_ts"].dt.tz
            if not (tz and tz.zone == "America/Chicago"):
                print("\nWarning: local_ts is not America/Chicago timezone-aware")

        # Check for ambiguous times (expected during DST transitions)
        if "local_ts" in df.columns:
            ambiguous = df[df["local_ts"].isna()]
            if not ambiguous.empty:
                print(
                    f"\nFound {len(ambiguous)} ambiguous timestamps (expected during DST transitions)"
                )
                print("Example ambiguous times:")
                display_cols = ["utc_ts", "local_ts", "hour_local"]
                print(ambiguous[display_cols].head())

        # Print basic information
        print("\nData Summary:")
        print(f"Endpoint: {endpoint_key}")
        print(f"Records: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        if csv_file:
            print(f"\nData saved to: {csv_file}")

        # Check for missing values
        if df.isna().any().any():
            print("\nWarning: Missing values detected in columns:")
            missing = df.isna().sum()
            missing = missing[missing > 0]
            for col, count in missing.items():
                print(f"  {col}: {count} missing values")
