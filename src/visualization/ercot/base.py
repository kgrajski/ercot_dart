"""Base visualization module for ERCOT data."""

import os
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from glob import glob


class ERCOTBaseViz:
    """Base class for ERCOT data visualization."""
    
    def __init__(self, data_dir: str, output_dir: str):
        """Initialize visualization handler.
        
        Args:
            data_dir (str): Directory containing CSV data files
            output_dir (str): Directory where HTML plots will be saved
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def combine_date_hour(date: str, hour_ending: str) -> pd.Timestamp:
        """Combine ERCOT delivery date and hour ending into a timestamp.
        
        Handles the special case of "24:00" by converting it to "00:00" of the next day.
        
        Args:
            date (str): Delivery date in format "YYYY-MM-DD"
            hour_ending (str): Hour ending in format "HH:MM" (24-hour clock)
            
        Returns:
            pd.Timestamp: Combined datetime
            
        Example:
            >>> combine_date_hour("2025-06-03", "24:00")
            Timestamp('2025-06-04 00:00:00')
            >>> combine_date_hour("2025-06-03", "14:00")
            Timestamp('2025-06-03 14:00:00')
        """
        # Handle special case of 24:00 before parsing
        if hour_ending == "24:00":
            # Parse the date first
            base_dt = pd.to_datetime(date)
            # Add one day and set time to 00:00
            return base_dt + timedelta(days=1)
        else:
            # For normal hours, parse directly
            return pd.to_datetime(f"{date} {hour_ending}")
        
    def print_column_types(self, df: pd.DataFrame) -> None:
        """Print a formatted table of column names and their data types.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
        """
        # Get max lengths for formatting
        max_col_len = max(len(str(col)) for col in df.columns)
        max_type_len = max(len(str(dtype)) for dtype in df.dtypes)
        
        # Create format string for consistent alignment
        format_str = f"{{:<{max_col_len}}} | {{:<{max_type_len}}}"
        
        # Print header
        print("\nColumn Data Types:")
        print("-" * (max_col_len + max_type_len + 3))  # +3 for " | " separator
        print(format_str.format("Column", "Type"))
        print("-" * (max_col_len + max_type_len + 3))
        
        # Print each column and its type
        for col, dtype in df.dtypes.items():
            print(format_str.format(str(col), str(dtype)))
        print("-" * (max_col_len + max_type_len + 3))
        print()
        
    def get_latest_csv(self, endpoint_key: str) -> Path:
        """Get the path to the latest CSV file for a given endpoint.
        
        Args:
            endpoint_key (str): The endpoint identifier (e.g., "load_forecast")
            
        Returns:
            Path: Path to the latest CSV file
            
        Raises:
            FileNotFoundError: If no CSV files found for the endpoint
        """
        # Find all CSV files for this endpoint
        pattern = self.data_dir / f"{endpoint_key}.csv"
        files = sorted(glob(str(pattern)))
        
        if not files:
            raise FileNotFoundError(f"No CSV files found for endpoint {endpoint_key}")
            
        # Return the latest file (last in sorted order)
        return Path(files[-1])
        
    def get_data(self, endpoint_key: str, show_types: bool = True) -> pd.DataFrame:
        """Get data from CSV file for visualization.
        
        Args:
            endpoint_key (str): The endpoint identifier (e.g., "load_forecast")
            show_types (bool, optional): Whether to display column data types.
                                       Defaults to True.
            
        Returns:
            pd.DataFrame: Data for visualization
        """
        # Get latest CSV file for this endpoint
        csv_file = self.get_latest_csv(endpoint_key)
        print(f"Reading data from: {csv_file}")
        
        # Read CSV with appropriate type handling
        df = pd.read_csv(
            csv_file,
            dtype_backend="numpy_nullable"  # Better string and nullable type handling
        )
        
        if show_types:
            self.print_column_types(df)
            
        return df
    
    def save_plot(self, fig: go.Figure, plot_name: str, endpoint_key: str):
        """Save plotly figure as HTML file.
        
        Args:
            fig (go.Figure): Plotly figure to save
            plot_name (str): Name of the specific plot type (e.g., "daily_forecast")
            endpoint_key (str): The endpoint identifier
        """
        # Create filename with endpoint and plot name
        filename = f"{endpoint_key}_{plot_name}.html"
            
        # Create full path
        filepath = self.output_dir / filename
        
        # Save plot
        fig.write_html(filepath)
        print(f"Plot saved to: {filepath}")
    
    def generate_plots(self):
        """Generate all plots for this client.
        
        This method should be implemented by each client class.
        """
        raise NotImplementedError("Subclasses must implement generate_plots()") 