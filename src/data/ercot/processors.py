"""Data processing module for ERCOT API responses."""

from typing import Dict, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime


class ERCOTProcessor:
    """Handles data processing and file operations for ERCOT data."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize processor with optional output directory.
        
        Args:
            output_dir (str, optional): Directory path where CSV files will be saved.
                                      If None, no CSV files will be saved.
                                      If provided, directory will be created if it doesn't exist.
        """
        self.output_dir = None
        if output_dir:
            # Convert to Path object for easier manipulation
            self.output_dir = Path(output_dir)
            # Create directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def json_to_df(self, json_response: Dict) -> pd.DataFrame:
        """Convert ERCOT API JSON response to a pandas DataFrame with proper data types.
        
        Args:
            json_response (dict): JSON response from ERCOT API containing _meta, report, fields, and data
            
        Returns:
            pandas.DataFrame: DataFrame with proper column types based on fields metadata
            
        Raises:
            ValueError: If JSON structure is invalid or missing required fields
        """
        # Validate JSON structure
        required_fields = ['_meta', 'fields', 'data']
        if not all(field in json_response for field in required_fields):
            raise ValueError(f"JSON response missing required fields: {required_fields}")
        
        # Get the data array and fields metadata
        data = json_response['data']
        fields = json_response['fields']
        
        # If no data, return empty DataFrame with correct columns
        if not data:
            columns = [field['name'] for field in fields]
            return pd.DataFrame(columns=columns)
        
        # Create a list of column names in order
        columns = [field['name'] for field in fields]
        
        # Create DataFrame with proper column names
        df = pd.DataFrame(data, columns=columns)
        
        # Apply data types based on fields metadata
        for field in fields:
            column_name = field['name']
            data_type = field.get('dataType', '').upper()
            try:
                if data_type == 'BOOLEAN':
                    df[column_name] = df[column_name].astype('boolean')
                elif data_type in ['INTEGER', 'INT']:
                    df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype('Int64')  # nullable integer
                elif data_type in ['DECIMAL', 'FLOAT', 'DOUBLE']:
                    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
                elif data_type in ['DATE', 'DATETIME']:
                    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
                elif data_type in ['VARCHAR', 'STRING']:  # Added VARCHAR to handle that type
                    df[column_name] = df[column_name].astype('string')
            except Exception as e:
                print(f"Warning: Could not convert column {column_name} to {data_type}: {str(e)}")
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, endpoint_key: str, params: Optional[Dict] = None) -> Optional[Path]:
        """Save DataFrame to CSV in the output directory with a standardized name.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            endpoint_key (str): Key from ENDPOINTS dictionary used to generate data
            params (dict, optional): Parameters used in the API call, used in filename
            
        Returns:
            Path: Path to saved file if output_dir is set, None otherwise
            
        Example filename: load_forecast_2024-03-20_150130.csv
                         (endpoint_key_YYYY-MM-DD_HHMMSS.csv)
        """
        if not self.output_dir:
            return None
            
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        
        # Create filename
        filename = f"{endpoint_key}_{timestamp}.csv"
        
        # Create full path
        file_path = self.output_dir / filename
        
        # Save DataFrame to CSV
        df.to_csv(file_path, index=False)
        
        return file_path
    
    def verify_data(self, df: pd.DataFrame, endpoint_key: str, csv_file: Optional[Path] = None) -> None:
        """Verify the DataFrame contents and output verification results.
        
        Args:
            df (pd.DataFrame): DataFrame to verify
            endpoint_key (str): Key of the endpoint that generated this data
            csv_file (Path, optional): Path to the CSV file if saved
            
        Raises:
            AssertionError: If any verification checks fail
        """
        # Basic DataFrame verification
        assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
        assert not df.empty, "DataFrame should not be empty"
        
        # Print data summary
        print("\nData Summary:")
        print(f"Total records: {len(df)}")
        print("\nColumns:")
        for col in df.columns:
            print(f"- {col}")
        
        # Print first few rows
        print("\nFirst few rows:")
        print(df.head())
        
        # If CSV was saved, verify it
        if csv_file and self.output_dir:
            # Verify file exists
            assert csv_file.exists(), "CSV file should exist"
            assert csv_file.name.startswith(f"{endpoint_key}_"), "CSV filename should start with endpoint key"
            assert csv_file.suffix == ".csv", "File should have .csv extension"
            
            # Verify CSV contents match DataFrame
            df_from_csv = pd.read_csv(csv_file)
            assert df.shape == df_from_csv.shape, "CSV data should match original DataFrame"
            print(f"\nCSV file successfully created and verified at: {csv_file}")
        
        # Print datetime-specific information if available
        if 'PostedDateTime' in df.columns:
            posted_times = pd.to_datetime(df['PostedDateTime'])
            print("\nUnique posted datetimes in the data:")
            print(sorted(posted_times.dt.strftime('%Y-%m-%d %H:%M:%S').unique())) 