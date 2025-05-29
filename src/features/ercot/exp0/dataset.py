"""Experiment 0 dataset implementation."""

import pandas as pd
from ..exp_dataset import ExpDataset
from ..visualization import plot_dart_by_location


class Exp0Dataset(ExpDataset):
    """Dataset handler for Experiment 0.
    
    This experiment focuses on:
    - Using transformed RT SPP data with hourly statistics
    - Combining with cleaned data from other sources
    - Creating baseline features for price prediction
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """Initialize Exp0 dataset handler.
        
        Args:
            input_dir: Directory containing processed input data files
            output_dir: Directory where experiment datasets will be saved
        """
        # Define required input files
        input_files = [
            "dam_spp_clean.csv",
            "dam_system_lambda_clean.csv",
            "load_forecast_clean.csv",
            "wind_power_gen_clean.csv",
            "solar_power_gen_clean.csv",
            "rt_spp_transformed.csv"
        ]
        
        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            input_files=input_files,
            experiment_id='exp0'
        )
    
    def generate_dependent_vars(self) -> pd.DataFrame:
        """Generate dependent variables for Exp0.
        
        This implementation calculates DART (Day-Ahead to Real-Time) price differences:
        DART = RT_Price - DAM_Price
        
        The calculation:
        1. Uses the already loaded RT and DAM price data
        2. Merges the datasets on datetime and location
        3. Adds DAM price and calculates the DART difference
        4. Visualizes the DART differences by location
        
        Returns:
            DataFrame containing RT data plus DAM price and DART columns
        """
        # Get the required datasets from raw_data
        rt_df = self.raw_data["rt_spp"]
        dam_df = self.raw_data["dam_spp"]
        
        # Ensure datetime is in proper format
        rt_df["datetime"] = pd.to_datetime(rt_df["datetime"])
        dam_df["datetime"] = pd.to_datetime(dam_df["datetime"])
        
        # Merge RT and DAM data on datetime and location
        # Using left merge to keep all RT records
        result_df = rt_df.merge(
            dam_df[["datetime", "location", "price"]],
            on=["datetime", "location"],
            how="left"
        )
        
        # Rename the DAM price column for clarity
        result_df = result_df.rename(columns={"price": "dam_spp_price"})
        
        # Calculate DART (RT - DAM difference)
        result_df["dart"] = (result_df["price_mean"] - result_df["dam_spp_price"]).round(6)
        
        # Create DART visualization
        plot_dart_by_location(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )
        
        return result_df
    
    def generate_independent_vars(self) -> pd.DataFrame:
        """Generate independent variables for Exp0.
        
        For this initial experiment, we'll focus on:
        - DAM prices and system lambda
        - Load, wind, and solar data
        - Time-based features
        
        Returns:
            DataFrame containing feature variables
        """
        # TODO: Implement feature generation
        # This will be implemented in detail in subsequent steps
        pass
    
    def run_eda(self):
        """Run exploratory data analysis for Exp0.
        
        This will include:
        - Basic statistics of features and targets
        - Correlation analysis
        - Time series plots
        - Distribution plots
        """
        # TODO: Add additional EDA visualizations
        # This will be implemented in detail in subsequent steps
        pass 