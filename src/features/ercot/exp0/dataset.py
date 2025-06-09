"""Experiment 0 dataset implementation."""

import os

import holidays
import pandas as pd

from src.data.ercot.database import DatabaseProcessor
from src.features.ercot.exp_dataset import ExpDataset
from src.features.ercot.visualization import plot_dart_average_daily_heatmap
from src.features.ercot.visualization import plot_dart_boxplots
from src.features.ercot.visualization import plot_dart_by_location
from src.features.ercot.visualization import plot_dart_distributions
from src.features.ercot.visualization import plot_dart_qqplots
from src.features.ercot.visualization import plot_dart_slt_bimodal
from src.features.ercot.visualization import plot_dart_slt_by_hour
from src.features.ercot.visualization import plot_dart_slt_by_weekday
from src.features.ercot.visualization import plot_dart_slt_cumulative
from src.features.ercot.visualization import plot_dart_slt_kmeans_bimodal
from src.features.ercot.visualization import plot_dart_slt_kmeans_unimodal
from src.features.ercot.visualization import plot_dart_slt_moving_window_stats
from src.features.ercot.visualization import plot_dart_slt_power_spectrum
from src.features.ercot.visualization import plot_dart_slt_power_spectrum_bimodal
from src.features.ercot.visualization import plot_dart_slt_sign_daily_heatmap
from src.features.ercot.visualization import plot_dart_slt_sign_power_spectrum
from src.features.ercot.visualization import plot_dart_slt_sign_transitions
from src.features.ercot.visualization import plot_dart_slt_spectrogram
from src.features.utils import signed_log_transform


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
            "rt_spp_transformed.csv",
        ]

        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            input_files=input_files,
            experiment_id="exp0",
        )

    def generate_dependent_vars(self):
        """Generate dependent variables for Exp0.

        This implementation calculates DART (Day-Ahead to Real-Time) price differences:
        DART = RT_Price - DAM_Price

        The calculation:
        1. Uses the already loaded RT and DAM price data
        2. Merges the datasets on datetime and location
        3. Adds DAM price and calculates the DART difference
        4. Visualizes the DART differences by location

        Stores the result in self.study_data.
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
            how="left",
        )

        # Rename the DAM price column for clarity
        result_df = result_df.rename(columns={"price": "dam_spp_price"})

        # Rename the RT price column for clarity
        result_df = result_df.rename(columns={"price_mean": "rt_spp_price"})

        # Calculate DART (RT - DAM difference)
        result_df["dart"] = (
            result_df["rt_spp_price"] - result_df["dam_spp_price"]
        ).round(6)

        # Apply signed log transformation (single source of truth)
        result_df["dart_slt"] = signed_log_transform(result_df["dart"])

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

        # Create weekend flag (Saturday=5, Sunday=6 in pandas dayofweek)
        result_df["is_weekend"] = result_df["day_of_week"].apply(
            lambda x: x >= 5 if pd.notna(x) else None
        )

        # Create holiday flag for US Federal and Texas State holidays
        # Initialize holiday calendar that includes both US federal and Texas state holidays
        us_holidays = holidays.UnitedStates(state="TX", years=range(2015, 2030))

        # Create holiday flag based on local date (not time)
        result_df["is_holiday"] = result_df["local_ts"].apply(
            lambda x: pd.to_datetime(x).date() in us_holidays if pd.notna(x) else None
        )

        # Store the result in the study_data attribute
        self.study_data = result_df

        # Write the result_df to a csv file, but with the name exp0_study_dataset.csv
        self.study_data.to_csv(
            os.path.join(self.output_dir, "exp0_study_dataset.csv"), index=False
        )

    def _create_safe_identifier(self, location: str, location_type: str) -> str:
        """Create a safe filename identifier from location and location_type.

        Uses the exact same pattern as the visualization functions:
        1. Create point_identifier: "location (location_type)"
        2. Apply safe filename transformations

        Args:
            location: Location name
            location_type: Location type (e.g., 'LZ', 'HB')

        Returns:
            str: Safe identifier for use in filenames and directories
        """
        # Step 1: Create point_identifier exactly like the visualization functions
        point_identifier = f"{location} ({location_type})"

        # Step 2: Apply exact same safe filename transformation as visualization functions
        safe_identifier = (
            point_identifier.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )

        return safe_identifier

    def _generate_independent_vars_single(
        self, df_single: pd.DataFrame, location: str, location_type: str
    ) -> pd.DataFrame:
        """Generate independent variables for a single location+location_type combination.

        Args:
            df_single: DataFrame for single location+location_type, assumed to be sorted by utc_ts
            location: Location name
            location_type: Location type

        Returns:
            pd.DataFrame: DataFrame with added independent variable features
        """
        # Work with a copy
        df = df_single.copy()

        # Sort by time to ensure proper lag/rolling calculations
        df = df.sort_values("utc_ts").reset_index(drop=True)

        # Create lagged features (simple since we're processing single entity)
        lag_hours = [1, 2, 24, 168]  # 1 hour, 2 hours, 1 day, 1 week

        for lag_h in lag_hours:
            col_name = f"dart_slt_lag_{lag_h}hr"
            df[col_name] = df["dart_slt"].shift(lag_h)

        # Create rolling statistics (simple since we're processing single entity)
        roll_hours = [24, 168]  # 1 day, 1 week

        for roll_h in roll_hours:
            # Rolling mean
            roll_mean_col = f"dart_slt_roll_mean_{roll_h}hr"
            df[roll_mean_col] = (
                df["dart_slt"].rolling(window=roll_h, min_periods=1).mean()
            )

            # Rolling standard deviation
            roll_sdev_col = f"dart_slt_roll_sdev_{roll_h}hr"
            df[roll_sdev_col] = (
                df["dart_slt"].rolling(window=roll_h, min_periods=1).std()
            )

        print(f"  Processed {location} [{location_type}]: {len(df):,} records")
        return df

    def generate_independent_vars(self):
        """Generate independent variables for Exp0.

        For this initial experiment, we create time series features from DART SLT:

        Lagged features:
        - dart_slt_lag_1hr: DART SLT value from 1 hour ago
        - dart_slt_lag_2hr: DART SLT value from 2 hours ago
        - dart_slt_lag_24hr: DART SLT value from 24 hours ago (1 day)
        - dart_slt_lag_168hr: DART SLT value from 168 hours ago (7 days)

        Rolling statistics:
        - dart_slt_roll_mean_24hr: 24-hour rolling mean of DART SLT
        - dart_slt_roll_sdev_24hr: 24-hour rolling standard deviation of DART SLT
        - dart_slt_roll_mean_168hr: 168-hour rolling mean of DART SLT
        - dart_slt_roll_sdev_168hr: 168-hour rolling standard deviation of DART SLT

        Processes each location+location_type combination independently and saves
        to individual subdirectories for future orchestration/parallelization.

        Updates self.study_data with combined results.
        """
        if self.study_data is None:
            raise ValueError(
                "No study data available. Run generate_dependent_vars() first."
            )

        print("Processing independent variables by location+location_type...")

        # Process each location+location_type combination independently
        results = []
        feature_summaries = []

        for (location, location_type), group in self.study_data.groupby(
            ["location", "location_type"]
        ):
            # Process this single combination
            processed_group = self._generate_independent_vars_single(
                group, location, location_type
            )
            results.append(processed_group)

            # Create safe identifier for subdirectory
            safe_id = self._create_safe_identifier(location, location_type)
            location_dir = self.output_dir / safe_id
            location_dir.mkdir(parents=True, exist_ok=True)

            # Save this combination's dataset to its own subdirectory
            dataset_file = location_dir / f"exp0_study_dataset_{safe_id}.csv"
            processed_group.to_csv(dataset_file, index=False)

            # Collect feature summary for this combination
            feature_summaries.append(
                {
                    "location": location,
                    "location_type": location_type,
                    "safe_id": safe_id,
                    "records": len(processed_group),
                }
            )

        # Combine all results and sort properly
        self.study_data = pd.concat(results, ignore_index=True)
        self.study_data = self.study_data.sort_values(
            ["location", "location_type", "utc_ts"]
        ).reset_index(drop=True)

        # Add point_identifier column using the same format as visualization functions
        self.study_data["point_identifier"] = self.study_data.apply(
            lambda row: f"{row['location']} ({row['location_type']})", axis=1
        )

        # Save combined dataset to main output directory
        self.study_data.to_csv(
            os.path.join(self.output_dir, "exp0_study_dataset_combined.csv"),
            index=False,
        )

        # Save to database if requested
        # Initialize database processor
        db_path = self.output_dir / "exp0_study_dataset.db"
        db_processor = DatabaseProcessor(str(db_path))

        # Save individual location+location_type datasets to database
        for summary in feature_summaries:
            # Find the corresponding processed data
            location_data = self.study_data[
                (self.study_data["location"] == summary["location"])
                & (self.study_data["location_type"] == summary["location_type"])
            ]

            # Save to database with safe table name
            table_name = f"exp0_{summary['safe_id']}"
            db_processor.save_to_database(location_data, table_name)

        # Save combined dataset to database
        db_processor.save_to_database(self.study_data, "exp0_study_dataset_combined")

        print(f"\nDatabase saved to: {db_path}")

        # Print comprehensive summary
        print(
            f"\nCompleted processing {len(feature_summaries)} location+location_type combinations:"
        )
        for summary in feature_summaries:
            print(
                f"  {summary['location']} [{summary['location_type']}] -> {summary['safe_id']}/: {summary['records']:,} records"
            )

        # Print feature completeness statistics
        lag_hours = [1, 2, 24, 168]
        roll_hours = [24, 168]

        print(f"\nFeature completeness across all combinations:")
        total_records = len(self.study_data)

        for lag_h in lag_hours:
            col_name = f"dart_slt_lag_{lag_h}hr"
            non_null_count = self.study_data[col_name].notna().sum()
            print(
                f"  {col_name}: {non_null_count:,} non-null out of {total_records:,} total ({non_null_count/total_records*100:.1f}%)"
            )

        for roll_h in roll_hours:
            for stat in ["mean", "sdev"]:
                col_name = f"dart_slt_roll_{stat}_{roll_h}hr"
                non_null_count = self.study_data[col_name].notna().sum()
                print(
                    f"  {col_name}: {non_null_count:,} non-null out of {total_records:,} total ({non_null_count/total_records*100:.1f}%)"
                )

    def run_eda(self):
        """Run exploratory data analysis for Exp0.

        This includes comprehensive DART visualizations and analysis:
        - Time series plots and distributions
        - Statistical analysis and transformations
        - Temporal pattern analysis
        - Spectral and clustering analysis
        """
        if self.study_data is None:
            raise ValueError(
                "No study data available. Run generate_dependent_vars() first."
            )

        # Create comprehensive DART visualizations

        # 1. Time series comparison (raw vs transformed)
        plot_dart_by_location(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 2. Distribution analysis with normal fit and percentiles
        plot_dart_distributions(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 3. Box plots for quartile analysis by location
        plot_dart_boxplots(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 4. Q-Q plots for normality assessment
        plot_dart_qqplots(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 5. Bimodal analysis of signed log transformed DART
        plot_dart_slt_bimodal(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 6. Cumulative analysis of signed log transformed DART
        plot_dart_slt_cumulative(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 7. Day-of-week analysis of signed log transformed DART
        plot_dart_slt_by_weekday(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 8. Hour-of-day analysis of signed log transformed DART
        plot_dart_slt_by_hour(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 9. Power spectrum analysis of signed log transformed DART
        plot_dart_slt_power_spectrum(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 10. Bimodal power spectrum analysis of signed log transformed DART
        plot_dart_slt_power_spectrum_bimodal(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 11. Sign power spectrum analysis of signed log transformed DART
        plot_dart_slt_sign_power_spectrum(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 12. Sign daily heatmap analysis of signed log transformed DART
        plot_dart_slt_sign_daily_heatmap(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 13. Spectrogram analysis of signed log transformed DART
        plot_dart_slt_spectrogram(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 14. Moving window statistics analysis of signed log transformed DART
        plot_dart_slt_moving_window_stats(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 15. Sign transitions analysis of signed log transformed DART
        plot_dart_slt_sign_transitions(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )

        # 16. K-means clustering analysis (unimodal) of signed log transformed DART
        plot_dart_slt_kmeans_unimodal(
            df=self.study_data,
            output_dir=self.output_dir,
            title_suffix=" - Exp0",
            max_k=10,
        )

        # 17. K-means clustering analysis (bimodal) of signed log transformed DART
        plot_dart_slt_kmeans_bimodal(
            df=self.study_data,
            output_dir=self.output_dir,
            title_suffix=" - Exp0",
            max_k=10,
        )

        # 18. Average DART daily cycle heatmap analysis
        plot_dart_average_daily_heatmap(
            df=self.study_data, output_dir=self.output_dir, title_suffix=" - Exp0"
        )
