"""Database processing module for ERCOT data.

This module handles SQLite database operations with automatic datetime string conversion.

Example usage:
    # Save data with datetime columns
    db_processor = DatabaseProcessor("ercot_data.db")
    db_processor.save_to_database(df_with_timestamps, "load_forecast_clean")

    # Read data back with automatic datetime conversion
    df_restored = db_processor.read_from_database("load_forecast_clean")

    # Or specify datetime columns explicitly
    df_restored = db_processor.read_from_database(
        "load_forecast_clean",
        datetime_columns=["utc_ts", "local_ts"]
    )
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd


class DatabaseProcessor:
    """Handles database operations for ERCOT data."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database processor.

        Args:
            db_path (str, optional): Path to SQLite database file.
                                   If None, will create 'ercot_data.db' in current directory.
        """
        if db_path is None:
            db_path = "ercot_data.db"
        self.db_path = Path(db_path)

    def _prepare_dataframe_for_sqlite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for SQLite storage by converting datetime columns to strings.

        This method works on a copy of the DataFrame and converts pandas Timestamp
        objects to ISO8601 string format (YYYY-MM-DD HH:MM:SS.ffffff).

        Args:
            df (pd.DataFrame): DataFrame to prepare

        Returns:
            pd.DataFrame: Copy of DataFrame with datetime columns converted to strings
        """
        df_copy = df.copy()

        # Find and convert datetime columns
        for col in df_copy.columns:
            col_dtype = str(df_copy[col].dtype)

            # Check for datetime64 types (any timezone) or Timestamp objects
            is_datetime_dtype = pd.api.types.is_datetime64_any_dtype(df_copy[col])
            is_datetime_tz = "datetime64" in col_dtype

            has_timestamps = False
            if not df_copy[col].empty:
                first_val = df_copy[col].iloc[0]
                has_timestamps = isinstance(first_val, pd.Timestamp)

            if is_datetime_dtype or is_datetime_tz or has_timestamps:
                try:
                    # Handle timezone-aware datetimes by converting to naive UTC first
                    if (
                        hasattr(df_copy[col].dtype, "tz")
                        and df_copy[col].dtype.tz is not None
                    ):
                        # Convert timezone-aware to UTC then remove timezone info
                        df_copy[col] = (
                            df_copy[col].dt.tz_convert("UTC").dt.tz_localize(None)
                        )

                    # Convert to string
                    df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                except Exception:
                    # If conversion fails, try to handle it as object type
                    df_copy[col] = df_copy[col].astype(str)

        # Final check: ensure no remaining Timestamp objects
        for col in df_copy.columns:
            if not df_copy[col].empty:
                first_val = df_copy[col].iloc[0]
                if isinstance(first_val, pd.Timestamp):
                    df_copy[col] = df_copy[col].astype(str)

        return df_copy

    def create_table_for_client(self, client_key: str, df: pd.DataFrame) -> str:
        """Create SQL table schema for a client's data.

        Args:
            client_key (str): Identifier for the client (e.g., 'load_forecast')
            df (pd.DataFrame): Sample DataFrame to derive schema from

        Returns:
            str: SQL create table statement
        """
        # Map pandas dtypes to SQL types
        dtype_map = {
            "object": "TEXT",
            "int64": "INTEGER",
            "float64": "REAL",
            "datetime64[ns]": "TEXT",  # Store datetime as TEXT in SQLite
            "bool": "BOOLEAN",
        }

        # Generate column definitions
        columns = []
        for col, dtype in df.dtypes.items():
            sql_type = dtype_map.get(str(dtype), "TEXT")
            columns.append(f"{col} {sql_type}")

        # Create table statement
        create_table = f"""
        CREATE TABLE IF NOT EXISTS {client_key} (
            {", ".join(columns)}
        )
        """

        return create_table

    def deduplicate_data(self, df: pd.DataFrame, client_key: str) -> pd.DataFrame:
        """Basic deduplication of data using pandas drop_duplicates.

        Args:
            df (pd.DataFrame): DataFrame to deduplicate
            client_key (str): Identifier for the client (unused for now)

        Returns:
            pd.DataFrame: Deduplicated DataFrame
        """
        # Get count before deduplication
        original_count = len(df)

        # Deduplicate
        deduped_df = df.drop_duplicates(keep="last")

        # Get count after deduplication
        final_count = len(deduped_df)

        # Print warning if duplicates were found
        if final_count < original_count:
            duplicate_count = original_count - final_count
            print(
                f"\nWARNING: Found {duplicate_count} duplicate rows in {client_key} data"
            )
            print(
                f"Original count: {original_count}, After deduplication: {final_count}"
            )

        return deduped_df

    def save_to_database(self, df: pd.DataFrame, client_key: str):
        """Save DataFrame to SQLite database.

        This method converts datetime columns to strings internally for SQLite storage
        but does not modify the original DataFrame passed to it.

        Args:
            df (pd.DataFrame): DataFrame to save (will not be modified)
            client_key (str): Identifier for the client
        """
        # Connect to database
        with sqlite3.connect(self.db_path) as conn:
            # Create table if it doesn't exist
            create_table_sql = self.create_table_for_client(client_key, df)
            conn.execute(create_table_sql)

            # Prepare DataFrame for SQLite (convert datetime to strings on a copy)
            prepared_df = self._prepare_dataframe_for_sqlite(df)

            # Deduplicate and save data
            deduped_df = self.deduplicate_data(prepared_df, client_key)
            deduped_df.to_sql(client_key, conn, if_exists="replace", index=False)

            print(
                f"Saved {len(deduped_df)} records to {client_key} table in {self.db_path}"
            )

    def read_from_database(
        self, client_key: str, datetime_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Read DataFrame from SQLite database with automatic datetime conversion.

        Args:
            client_key (str): Identifier for the client table to read
            datetime_columns (List[str], optional): List of column names to convert
                                                   back to datetime. If None, will
                                                   auto-detect common patterns.

        Returns:
            pd.DataFrame: DataFrame with datetime columns converted back to pandas datetime
        """
        # Connect to database and read data
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {client_key}", conn)

        # Auto-detect datetime columns if not specified
        if datetime_columns is None:
            datetime_columns = []
            for col in df.columns:
                # Look for common datetime column name patterns
                if any(
                    pattern in col.lower()
                    for pattern in ["_ts", "datetime", "date", "time"]
                ):
                    # Check if column contains datetime-like strings
                    if df[col].dtype == "object" and not df[col].empty:
                        sample_value = str(df[col].iloc[0])
                        # Check if it looks like ISO8601 format
                        if "-" in sample_value and ":" in sample_value:
                            datetime_columns.append(col)

        # Convert datetime columns back to pandas datetime
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        print(f"Read {len(df)} records from {client_key} table in {self.db_path}")
        return df
