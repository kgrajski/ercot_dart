"""Base ETL module for ERCOT data."""

import os
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from data.ercot.database import DatabaseProcessor


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
        return pd.read_csv(
            csv_file,
            dtype_backend="numpy_nullable",  # Better string and nullable type handling
            parse_dates=['local_ts', 'utc_ts']  # Parse timestamp columns
        )
    
    def save_clean_data(self, df: pd.DataFrame, endpoint_key: str):
        """Save cleaned data to CSV and database.
        
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
        
        # Save to database if configured
        if self.db_processor:
            self.save_to_database(df, f"{endpoint_key}_clean")
    
    def save_transformed_data(self, df: pd.DataFrame, endpoint_key: str):
        """Save transformed data to CSV and database.
        
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
        """Validate that every hour between earliest and latest utc_ts has at least one row.
        
        This method checks for temporal gaps in the data by ensuring that
        every hour in the time range is represented by at least one record.
        Useful for detecting missing hours in time series data.
        
        Args:
            df (pd.DataFrame): DataFrame to validate (must have utc_ts column)
            
        Returns:
            bool: True if no temporal gaps found, False otherwise
        """
        try:
            if df.empty:
                print("Warning: Cannot validate temporal completeness on empty DataFrame")
                return True
                
            if 'utc_ts' not in df.columns:
                print("Error: DataFrame missing required utc_ts column for temporal validation")
                return False
            
            # Extract hourly timestamps (truncate to hour)
            hourly_timestamps = df['utc_ts'].dt.floor('h')
            
            # Find earliest and latest hours
            earliest_hour = hourly_timestamps.min()
            latest_hour = hourly_timestamps.max()
            
            # Generate complete hour range
            expected_hours = pd.date_range(
                start=earliest_hour, 
                end=latest_hour, 
                freq='h'
            )
            
            # Get unique hours present in data
            actual_hours = set(hourly_timestamps.unique())
            expected_hours_set = set(expected_hours)
            
            # Find missing hours
            missing_hours = expected_hours_set - actual_hours
            
            if missing_hours:
                missing_count = len(missing_hours)
                total_expected = len(expected_hours)
                print(f"Error: Temporal completeness validation failed")
                print(f"Missing {missing_count} hours out of {total_expected} expected hours")
                print(f"Time range: {earliest_hour} to {latest_hour}")
                
                # Show first few missing hours as examples
                sorted_missing = sorted(list(missing_hours))[:5]
                print(f"First missing hours: {[str(h) for h in sorted_missing]}")
                if missing_count > 5:
                    print(f"... and {missing_count - 5} more")
                    
                return False
            
            # Success case
            total_hours = len(expected_hours)
            print(f"Temporal completeness validated: {total_hours} hours from {earliest_hour} to {latest_hour}")
            return True
            
        except Exception as e:
            print(f"Temporal validation error: {str(e)}")
            return False
    
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