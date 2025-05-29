"""Base ETL module for ERCOT data."""

import os
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


class ERCOTBaseETL:
    """Base class for ERCOT data ETL operations."""
    
    def __init__(self, data_dir: str, output_dir: str):
        """Initialize ETL handler.
        
        Args:
            data_dir (str): Directory containing raw CSV data files
            output_dir (str): Directory where cleaned data will be saved
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def combine_date_hour(date: str, hour_ending: str | pd.Int64Dtype | int) -> pd.Timestamp:
        """Combine ERCOT delivery date and hour ending into a timestamp.
        
        Handles multiple hour_ending formats:
        - String format ("HH:MM" or "HH:MM:SS")
        - Integer format (Int64 or int, 1-24)
        Handles the special case of "24:00" or "24:00:00" or 24 by converting it to "00:00" of the next day.
        
        Args:
            date (str): Delivery date in format "YYYY-MM-DD"
            hour_ending (str | pd.Int64Dtype | int): Hour ending in either:
                - String format "HH:MM" or "HH:MM:SS" (24-hour clock)
                - Integer format (1-24)
            
        Returns:
            pd.Timestamp: Combined datetime
        """
        # Convert hour_ending to string format if it's an integer type
        if isinstance(hour_ending, (int, pd.Int64Dtype)):
            # Handle hour 24 case for integer input
            if hour_ending == 24:
                base_dt = pd.to_datetime(date)
                return base_dt + timedelta(days=1)
            hour_ending = f"{int(hour_ending):02d}:00"
        elif isinstance(hour_ending, str):
            # Handle hour 24 case for string input ("24:00" or "24:00:00")
            if hour_ending.startswith("24:"):
                base_dt = pd.to_datetime(date)
                return base_dt + timedelta(days=1)
            # If we have "HH:MM:SS" format, strip off the seconds
            if len(hour_ending.split(":")) == 3:
                hour_ending = ":".join(hour_ending.split(":")[:2])
        else:
            # If we get here, we have an unexpected type
            print(f"Warning: Unexpected hour_ending type: {type(hour_ending)}. Value: {hour_ending}")
            # Try to convert to string and proceed
            hour_ending = str(hour_ending)
            
        # For normal hours, parse directly
        return pd.to_datetime(f"{date} {hour_ending}")
    
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
        
        # Read CSV with appropriate type handling
        return pd.read_csv(
            csv_file,
            dtype_backend="numpy_nullable"  # Better string and nullable type handling
        )
    
    def save_clean_data(self, df: pd.DataFrame, endpoint_key: str):
        """Save cleaned data to CSV.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame to save
            endpoint_key (str): The endpoint identifier
        """
        # Create filename with endpoint key and 'clean' indicator
        filename = f"{endpoint_key}_clean.csv"
        filepath = self.output_dir / filename
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Saved cleaned data to: {filepath}")
    
    def save_transformed_data(self, df: pd.DataFrame, endpoint_key: str):
        """Save transformed data to CSV.
        
        Args:
            df (pd.DataFrame): Transformed DataFrame to save
            endpoint_key (str): The endpoint identifier
        """
        # Create filename with endpoint key and 'transformed' indicator
        filename = f"{endpoint_key}_transformed.csv"
        filepath = self.output_dir / filename
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Saved transformed data to: {filepath}")
    
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
            pd.DataFrame: Cleaned and validated data
        """
        # Get raw data
        df_raw = self.get_raw_data(self.ENDPOINT_KEY)
        
        # Clean data
        df_clean = self.clean_data(df_raw)
        
        # Validate
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

    @staticmethod
    def convert_hour_ending(hour_ending: str | pd.Int64Dtype | int) -> int:
        """Convert ERCOT hour ending format to 0-23 integer.
        
        Handles multiple hour_ending formats:
        - String format ("HH:MM" or "HH:MM:SS")
        - Integer format (Int64 or int, 1-24)
        
        Args:
            hour_ending (str | pd.Int64Dtype | int): Hour ending in either:
                - String format "HH:MM" or "HH:MM:SS" (24-hour clock)
                - Integer format (1-24)
            
        Returns:
            int: Hour as 0-23 integer
            
        Example:
            >>> convert_hour_ending("24:00")
            0
            >>> convert_hour_ending("24:00:00")
            0
            >>> convert_hour_ending(24)
            0
            >>> convert_hour_ending("14:00")
            13
            >>> convert_hour_ending(14)
            13
        """
        if isinstance(hour_ending, (int, pd.Int64Dtype)):
            hour = int(hour_ending)
        elif isinstance(hour_ending, str):
            # Strip seconds if present
            if len(hour_ending.split(":")) == 3:
                hour_ending = ":".join(hour_ending.split(":")[:2])
            # Extract hour and convert to integer
            hour = int(hour_ending.split(":")[0])
        else:
            raise ValueError(f"Unexpected hour_ending type: {type(hour_ending)}. Value: {hour_ending}")
            
        # Convert hour 24 to hour 0
        return 0 if hour == 24 else hour - 1 