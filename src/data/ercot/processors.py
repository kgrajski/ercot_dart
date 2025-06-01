"""Data processing module for ERCOT API responses."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import json
from tqdm import tqdm
import pytz


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
            
    def add_utc_timestamps(self, df: pd.DataFrame, interval_minutes: Optional[pd.Series] = None) -> pd.DataFrame:
        """Add canonical timestamp columns following ERCOT standards.
        
        Takes ERCOT delivery date and hour ending and converts to canonical UTC timestamp.
        Handles different hour ending formats and DST transitions properly.
        
        Hour ending examples in ERCOT's canonical format:
        - "01:00" or "01:00:00" -> the hour ending at 01:00
        - 1 or 1.0 -> the hour ending at 01:00
        - "24:00" or 24 -> the hour ending at 24:00 (same day)
        
        For sub-hourly data (e.g., RT SPP with 15-minute intervals):
        - interval_minutes specifies minutes to add to the base hour
        - For example, interval 2 in hour ending 01:00 would be 01:15
        
        Args:
            df (pd.DataFrame): DataFrame with raw timestamp data
            Must contain:
            - deliveryDate: ERCOT delivery date (string or datetime)
            - hourEnding: Hour ending in various formats
            - DSTFlag: Boolean DST indicator
            interval_minutes (pd.Series, optional): Minutes to add to base hour
                                                  Used for sub-hourly data
            
        Returns:
            pd.DataFrame: DataFrame with canonical timestamp columns:
            - local_ts: America/Chicago timezone-aware (NaT for ambiguous times)
            - utc_ts: UTC timestamp timezone-naive (canonical reference)
            - hour_local: Local hour (preserved for business logic)
        """
        df = df.copy()
        
        if 'deliveryDate' not in df.columns or 'hourEnding' not in df.columns:
            raise ValueError("DataFrame must contain 'deliveryDate' and 'hourEnding' columns")
        
        if 'DSTFlag' not in df.columns:
            raise ValueError("DataFrame must contain 'DSTFlag' column")
        
        # Standardize hourEnding format
        def standardize_hour_ending(hour_ending):
            """Convert various hour ending formats to HH:00 format."""
            if pd.isna(hour_ending):
                return None
                
            # Convert to string and strip any whitespace
            he_str = str(hour_ending).strip()
            
            # Handle integer-like formats (1, 1.0, etc.)
            try:
                he_int = int(float(he_str))
                if not 1 <= he_int <= 24:
                    raise ValueError(f"Hour ending must be between 1 and 24, got {he_int}")
                # Convert hour 24 to hour 0
                return "00:00" if he_int == 24 else f"{he_int:02d}:00"
            except ValueError:
                pass
                
            # Handle HH:MM or HH:MM:SS format
            if ':' in he_str:
                parts = he_str.split(':')
                hour = int(parts[0])
                if not 0 <= hour <= 24:
                    raise ValueError(f"Hour ending must be between 0 and 24, got {hour}")
                # Convert hour 24 to hour 0
                return "00:00" if hour == 24 else f"{hour:02d}:00"
                
            raise ValueError(f"Unrecognized hour ending format: {hour_ending}")
        
        df['hour_ending_std'] = df['hourEnding'].apply(standardize_hour_ending)
        
        # Convert deliveryDate to datetime if it's a string
        if not pd.api.types.is_datetime64_any_dtype(df['deliveryDate']):
            df['deliveryDate'] = pd.to_datetime(df['deliveryDate'])
        
        # For hour 24, we need to add one day to the delivery date
        df['date_str'] = df.apply(
            lambda row: (row['deliveryDate'] + pd.Timedelta(days=1)).strftime('%Y-%m-%d') 
            if row['hour_ending_std'] == "00:00" 
            else row['deliveryDate'].strftime('%Y-%m-%d'),
            axis=1
        )
        
        # Create local timestamp from delivery date and standardized hour ending
        df['local_ts'] = pd.to_datetime(df['date_str'] + ' ' + df['hour_ending_std'])
        
        # Add interval minutes if provided (before timezone localization)
        if interval_minutes is not None:
            df['local_ts'] = df['local_ts'] + pd.to_timedelta(interval_minutes, unit='minutes')
        
        # Localize to America/Chicago using DST flag
        # For fall back (ambiguous times), use DST flag to determine correct time
        # For spring forward (nonexistent times), will result in NaT
        df['local_ts'] = df.apply(
            lambda row: pd.Timestamp(row['local_ts']).tz_localize(
                'America/Chicago',
                ambiguous=row['DSTFlag'],  # Use DSTFlag for ambiguous times
                nonexistent='NaT'  # Mark nonexistent times during spring forward as NaT
            ),
            axis=1
        )
        
        # Extract hour_local before UTC conversion (for business logic)
        df['hour_local'] = df['local_ts'].dt.hour
        
        # Convert to canonical UTC timestamp (timezone-naive since UTC is inherently timezone-agnostic)
        df['utc_ts'] = df['local_ts'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Drop temporary columns
        df = df.drop(columns=['hour_ending_std', 'date_str'])
        
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
            raise ValueError(f"Response data missing required fields: {', '.join(missing)}")
            
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
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("Int64")
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
    
    def verify_data(self, df: pd.DataFrame, endpoint_key: str, csv_file: Optional[Path] = None):
        """Verify and report on the processed data.
        
        Args:
            df (pd.DataFrame): Processed data
            endpoint_key (str): Key identifying the endpoint
            csv_file (Path, optional): Path to saved CSV file
        """
        # Verify canonical timestamp columns
        required_cols = {'utc_ts', 'local_ts', 'hour_local'}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"\nWarning: Missing canonical timestamp columns: {missing}")
        
        # Verify timezone awareness for local timestamps
        if 'local_ts' in df.columns:
            tz = df['local_ts'].dt.tz
            if not (tz and tz.zone == 'America/Chicago'):
                print("\nWarning: local_ts is not America/Chicago timezone-aware")
        
        # Check for ambiguous times (expected during DST transitions)
        if 'local_ts' in df.columns:
            ambiguous = df[df['local_ts'].isna()]
            if not ambiguous.empty:
                print(f"\nFound {len(ambiguous)} ambiguous timestamps (expected during DST transitions)")
                print("Example ambiguous times:")
                display_cols = ['utc_ts', 'local_ts', 'hour_local']
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