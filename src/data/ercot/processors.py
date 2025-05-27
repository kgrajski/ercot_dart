"""Data processing module for ERCOT API responses."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import json
from tqdm import tqdm


class ERCOTProcessor:
    """Processor for ERCOT API response data."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize processor.
        
        Args:
            output_dir (str, optional): Directory where CSV files will be saved.
                                      If None, CSV files will not be saved.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
    def process_response(self, response) -> pd.DataFrame:
        """Process a single API response.
        
        Args:
            response: API response object
            
        Returns:
            pd.DataFrame: Processed data
            
        Raises:
            ValueError: If response is missing required fields
        """
        # Parse JSON response
        json_response = response.json()
        
        # Validate response structure
        required_fields = ["_meta", "fields", "data"]
        missing = [f for f in required_fields if f not in json_response]
        if missing:
            raise ValueError(f"Response missing required fields: {', '.join(missing)}")
            
        # Extract data and field definitions
        data = json_response["data"]
        fields = json_response["fields"]
        
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
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("Int64")  # nullable integer
            elif data_type in ["DECIMAL", "FLOAT", "DOUBLE"]:
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
            elif data_type in ["DATE", "DATETIME"]:
                df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
            elif data_type in ["VARCHAR", "STRING"]:  # Added VARCHAR to handle that type
                df[column_name] = df[column_name].astype("string")
                
        return df
    
    def process_data(self, data: List[Dict]) -> pd.DataFrame:
        """Process a list of data records.
        
        Args:
            data (List[Dict]): List of data records
            
        Returns:
            pd.DataFrame: Processed data
        """
        if not data:
            return pd.DataFrame()
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Convert date columns
        date_columns = [col for col in df.columns if "date" in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors="ignore")
            
        return df
    
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
            
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{endpoint_key}_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        # Save DataFrame
        df.to_csv(filepath, index=False)
        
        # Save parameters to JSON file
        params_file = filepath.with_suffix(".json")
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)
            
        return filepath
    
    def verify_data(self, df: pd.DataFrame, endpoint_key: str, csv_file: Optional[Path] = None):
        """Verify and report on the processed data.
        
        Args:
            df (pd.DataFrame): Processed data
            endpoint_key (str): Key identifying the endpoint
            csv_file (Path, optional): Path to saved CSV file
        """
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
                
        # Check PostedDateTime values if present
        if "PostedDateTime" in df.columns:
            posted_times = pd.to_datetime(df["PostedDateTime"])
            print("\nPostedDateTime range:")
            print(sorted(posted_times.dt.strftime("%Y-%m-%d %H:%M:%S").unique()))