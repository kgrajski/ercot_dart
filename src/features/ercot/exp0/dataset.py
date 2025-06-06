"""Experiment 0 dataset implementation."""

import pandas as pd
from ..exp_dataset import ExpDataset
from ..visualization import (
    plot_dart_by_location,
    plot_dart_distributions,
    plot_dart_boxplots,
    plot_dart_qqplots,
    plot_dart_slt_bimodal,
    plot_dart_slt_cumulative,
    plot_dart_slt_by_weekday,
    plot_dart_slt_by_hour,
    plot_dart_slt_power_spectrum,
    plot_dart_slt_power_spectrum_bimodal,
    plot_dart_slt_sign_power_spectrum,
    plot_dart_slt_sign_daily_heatmap,
    plot_dart_slt_spectrogram,
    plot_dart_slt_moving_window_stats,
    plot_dart_slt_sign_transitions,
    plot_dart_slt_kmeans_unimodal,
    plot_dart_slt_kmeans_bimodal,
)


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
            experiment_id="exp0"
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
        rt_df["utc_ts"] = pd.to_datetime(rt_df["utc_ts"])
        dam_df["utc_ts"] = pd.to_datetime(dam_df["utc_ts"])
        
        # Merge RT and DAM data on utc_ts and location
        # Using left merge to keep all RT records
        result_df = rt_df.merge(
            dam_df[["utc_ts", "location", "price"]],
            on=["utc_ts", "location"],
            how="left"
        )
        
        # Rename the DAM price column for clarity
        result_df = result_df.rename(columns={"price": "dam_spp_price"})

        # Rename the RT price column for clarity
        result_df = result_df.rename(columns={"price_mean": "rt_spp_price"})

        # Calculate DART (RT - DAM difference)
        result_df["dart"] = (result_df["rt_spp_price"] - result_df["dam_spp_price"]).round(6)

        # Apply signed log transformation (single source of truth)
        result_df["dart_slt"] = self.signed_log_transform(result_df["dart"])
        
        # Create day of week from local_ts (business time) using apply for mixed timezone handling
        # local_ts contains timezone-aware timestamps that may have different timezones (DST)
        result_df["day_of_week"] = result_df["local_ts"].apply(
            lambda x: pd.to_datetime(x).dayofweek if pd.notna(x) else None
        )

        # Create hour of day from local_ts (business time) using apply for mixed timezone handling
        # local_ts contains timezone-aware timestamps that may have different timezones (DST)
        # Add one hour so that the interpretation is consistent with ERCOT hour means end of delivery hour
        result_df["end_of_hour"] = result_df["local_ts"].apply(
            lambda x: pd.to_datetime(x).hour + 1 if pd.notna(x) else None
        )

        # Create comprehensive DART visualizations
        
        # 1. Time series comparison (raw vs transformed)
        plot_dart_by_location(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )
        
        # 2. Distribution analysis with normal fit and percentiles
        plot_dart_distributions(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )
        
        # 3. Box plots for quartile analysis by location
        plot_dart_boxplots(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )
        
        # 4. Q-Q plots for normality assessment
        plot_dart_qqplots(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )
        
        # 5. Bimodal analysis of signed log transformed DART
        plot_dart_slt_bimodal(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )

        # 6. Cumulative analysis of signed log transformed DART
        plot_dart_slt_cumulative(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )

        # 7. Day-of-week analysis of signed log transformed DART
        plot_dart_slt_by_weekday(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )

        # 8. Hour-of-day analysis of signed log transformed DART
        plot_dart_slt_by_hour(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )

        # 9. Power spectrum analysis of signed log transformed DART
        plot_dart_slt_power_spectrum(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )

        # 10. Bimodal power spectrum analysis of signed log transformed DART
        plot_dart_slt_power_spectrum_bimodal(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )

        # 11. Sign power spectrum analysis of signed log transformed DART
        plot_dart_slt_sign_power_spectrum(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )

        # 12. Sign daily heatmap analysis of signed log transformed DART
        plot_dart_slt_sign_daily_heatmap(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )

        # 13. Spectrogram analysis of signed log transformed DART
        plot_dart_slt_spectrogram(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )

        # 14. Moving window statistics analysis of signed log transformed DART
        plot_dart_slt_moving_window_stats(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )

        # 15. Sign transitions analysis of signed log transformed DART
        plot_dart_slt_sign_transitions(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0"
        )

        # 16. K-means clustering analysis (unimodal) of signed log transformed DART
        plot_dart_slt_kmeans_unimodal(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0",
            max_k=10
        )

        # 17. K-means clustering analysis (bimodal) of signed log transformed DART
        plot_dart_slt_kmeans_bimodal(
            df=result_df,
            output_dir=self.output_dir,
            title_suffix=" - Exp0",
            max_k=10
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