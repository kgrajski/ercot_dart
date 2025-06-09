"""Base visualization module for ERCOT data."""

import os
from datetime import datetime
from datetime import timedelta
from glob import glob
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
            pd.DataFrame: Data for visualization with proper timestamp parsing
        """
        # Get latest CSV file for this endpoint
        csv_file = self.get_latest_csv(endpoint_key)
        print(f"Reading data from: {csv_file}")

        # Read CSV with proper timestamp parsing
        df = pd.read_csv(
            csv_file,
            dtype_backend="numpy_nullable",  # Better string and nullable type handling
            parse_dates=["utc_ts", "local_ts"],  # Parse timestamp columns
        )

        if show_types:
            self.print_column_types(df)

        return df

    def save_data(self, df: pd.DataFrame, date: str, endpoint_key: str):
        """Save DataFrame as CSV file with consistent naming.

        Args:
            df (pd.DataFrame): Data to save
            date (str): The date to use in the filename (posted_date or delivery_date)
            endpoint_key (str): The endpoint identifier
        """
        # Create filename with endpoint and date
        filename = f"{endpoint_key}_{date}.csv"

        # Create full path
        filepath = self.output_dir / filename

        # Save DataFrame
        df.to_csv(filepath, index=False)
        print(f"Saved DataFrame to: {filepath}")

    def save_plot(self, fig: go.Figure, date: str, endpoint_key: str):
        """Save plotly figure as HTML file.

        Args:
            fig (go.Figure): Plotly figure to save
            date (str): The date to use in the filename (posted_date or delivery_date)
            endpoint_key (str): The endpoint identifier
        """
        # Create filename with endpoint and date
        filename = f"{endpoint_key}_{date}.html"

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
