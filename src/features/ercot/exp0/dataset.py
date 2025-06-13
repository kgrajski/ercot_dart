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
from src.features.ercot.visualization import plot_dart_slt_vs_features_by_hour
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

        # Experiment configuration
        self.experiment_name = "exp0"
        self.experiment_description = (
            "Baseline DART price prediction using lagged and rolling features"
        )

        # Feature configuration for this experiment
        # For 24-hour ahead prediction, we need lags that are available at prediction time:
        # - 24hr lag: Same hour, previous day (e.g., predict 6 AM tomorrow using 6 AM yesterday)
        # - 25hr lag: 1 hour before same hour, previous day (e.g., 5 AM yesterday)
        # - 26hr lag: 2 hours before same hour, previous day (e.g., 4 AM yesterday)
        # - 168hr lag: Same hour, previous week (weekly patterns)
        self.lag_hours = [24, 25, 26, 168]  # 1 day, 1 day + 1hr, 1 day + 2hr, 1 week
        self.roll_hours = [24, 168]  # 1 day, 1 week

        # Variable definitions
        self.dependent_vars = ["dart_slt"]  # Target variables for modeling
        self.source_price_columns = {
            "rt_price": "price_mean",  # RT SPP price column name
            "dam_price": "price",  # DAM SPP price column name
        }
        self.target_variables = {
            "dart": "rt_spp_price - dam_spp_price",  # Raw price difference
            "dart_slt": "signed_log_transform(dart)",  # Transformed target
        }

        # Validation parameters
        self.null_check_days_threshold = 1  # Flag nulls occurring after day N
        self.temporal_completeness_hours = 24  # Expected hours per day

        # Naming patterns
        self.dataset_prefix = f"{self.experiment_name}_study_dataset"
        self.final_dataset_prefix = f"{self.experiment_name}_final_dataset"
        self.db_table_prefix = self.experiment_name

    @property
    def independent_vars(self) -> list:
        """Get list of independent variable column names.

        Returns the overridden list if categorical encoding has been applied,
        otherwise returns the original computed list.

        Returns:
            list: Column names for all independent variables
        """
        # Return override if categorical encoding has been applied
        if hasattr(self, "_independent_vars_override"):
            return self._independent_vars_override

        # Original computation
        vars_list = []

        # Add lagged features
        for lag_h in self.lag_hours:
            vars_list.append(f"dart_slt_lag_{lag_h}hr")

        # Add rolling statistics features
        for roll_h in self.roll_hours:
            vars_list.extend(
                [f"dart_slt_roll_mean_{roll_h}hr", f"dart_slt_roll_sdev_{roll_h}hr"]
            )

        return vars_list

    @property
    def all_model_variables(self) -> dict:
        """Get all variables for modeling.

        Returns:
            dict: Dictionary with 'dependent', 'independent', and 'all' variable lists
        """
        return {
            "dependent": self.dependent_vars,
            "independent": self.independent_vars,
            "all": self.dependent_vars + self.independent_vars,
        }

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
        # Note that real-time prices are quoted for location and location type
        # Note that DAM prices are quoted for location only
        # Therefore, when we left merge with real-time prices, the one DAM location price will be applied
        result_df = rt_df.merge(
            dam_df[["utc_ts", "location", "price"]],
            on=["utc_ts", "location"],
            how="left",
        )

        # Rename the DAM price column for clarity
        result_df = result_df.rename(
            columns={self.source_price_columns["dam_price"]: "dam_spp_price"}
        )

        # Rename the RT price column for clarity
        result_df = result_df.rename(
            columns={self.source_price_columns["rt_price"]: "rt_spp_price"}
        )

        # Calculate DART (RT - DAM difference)
        result_df["dart"] = (
            result_df["rt_spp_price"] - result_df["dam_spp_price"]
        ).round(6)

        # Apply signed log transformation (single source of truth)
        result_df["dart_slt"] = signed_log_transform(result_df["dart"])

        # Store the result
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
        self,
        df_single: pd.DataFrame,
        location: str,
        location_type: str,
        lag_hours: list,
        roll_hours: list,
    ) -> pd.DataFrame:
        """Generate independent variables for a single location+location_type combination.

        Args:
            df_single: DataFrame for single location+location_type, assumed to be sorted by utc_ts
            location: Location name
            location_type: Location type
            lag_hours: List of lag hours to create features for
            roll_hours: List of rolling window hours to create features for

        Returns:
            pd.DataFrame: DataFrame with added independent variable features
        """
        # Work with a copy
        df = df_single.copy()

        # Sort by time to ensure proper lag/rolling calculations
        df = df.sort_values("utc_ts").reset_index(drop=True)

        # Create lagged features (simple since we're processing single entity)
        for lag_h in lag_hours:
            col_name = f"dart_slt_lag_{lag_h}hr"
            df[col_name] = df["dart_slt"].shift(lag_h)

        # Create rolling statistics (shifted to end 24 hours before prediction time)
        for roll_h in roll_hours:
            # Rolling mean - shift back 24 hours to avoid data leakage
            roll_mean_col = f"dart_slt_roll_mean_{roll_h}hr"
            df[roll_mean_col] = (
                df["dart_slt"].shift(24).rolling(window=roll_h, min_periods=1).mean()
            )

            # Rolling standard deviation - shift back 24 hours to avoid data leakage
            roll_sdev_col = f"dart_slt_roll_sdev_{roll_h}hr"
            df[roll_sdev_col] = (
                df["dart_slt"].shift(24).rolling(window=roll_h, min_periods=1).std()
            )

        print(f"  Processed {location} [{location_type}]: {len(df):,} records")
        return df

    def generate_independent_vars(self):
        """Generate independent variables for Exp0.

        For 24-hour ahead prediction, we create time series features from DART SLT
        that are available at prediction time (no data leakage):

        Lagged features:
        - dart_slt_lag_24hr: DART SLT value from 24 hours ago (same hour, previous day)
        - dart_slt_lag_25hr: DART SLT value from 25 hours ago (1 hour before, previous day)
        - dart_slt_lag_26hr: DART SLT value from 26 hours ago (2 hours before, previous day)
        - dart_slt_lag_168hr: DART SLT value from 168 hours ago (same hour, previous week)

        Rolling statistics (ending 24 hours before prediction time):
        - dart_slt_roll_mean_24hr: 24-hour rolling mean ending 24 hours ago
        - dart_slt_roll_sdev_24hr: 24-hour rolling standard deviation ending 24 hours ago
        - dart_slt_roll_mean_168hr: 168-hour rolling mean ending 24 hours ago
        - dart_slt_roll_sdev_168hr: 168-hour rolling standard deviation ending 24 hours ago

        Processes each location+location_type combination independently and saves
        to individual subdirectories for future orchestration/parallelization.

        Updates self.study_data with combined results.
        """
        if self.study_data is None:
            raise ValueError(
                "No study data available. Run generate_dependent_vars() first."
            )

        print("Processing independent variables by location+location_type...")

        # =====================================================================
        # Location-independent feature transformations
        # These transformations can be applied to each row independently
        # =====================================================================

        # Create day of week from local_ts (business time) using apply for mixed timezone handling
        #
        # NOTE: It is safe to use day_of_week, is_weekend, and is_holiday as features for forecasting,
        # because these are fully determined by the delivery date and are known in advance for any
        # forecast. There is no information leakage, as long as the holiday calendar is fixed and public.
        self.study_data["day_of_week"] = self.study_data["local_ts"].apply(
            lambda x: pd.to_datetime(x).dayofweek if pd.notna(x) else None
        )

        # Create hour of day from local_ts (business time) using apply for mixed timezone handling
        # local_ts contains timezone-aware timestamps that may have different timezones (DST)
        # Add one hour so that the interpretation is consistent with ERCOT hour means end of delivery hour
        self.study_data["end_of_hour"] = self.study_data["local_ts"].apply(
            lambda x: pd.to_datetime(x).hour + 1 if pd.notna(x) else None
        )

        # Create weekend flag (Saturday=5, Sunday=6 in pandas dayofweek)
        self.study_data["is_weekend"] = self.study_data["day_of_week"].apply(
            lambda x: x >= 5 if pd.notna(x) else None
        )

        # Create holiday flag for US Federal and Texas State holidays
        # Initialize holiday calendar that includes both US federal and Texas state holidays
        us_holidays = holidays.UnitedStates(state="TX", years=range(2015, 2030))

        # Create holiday flag based on local date (not time)
        self.study_data["is_holiday"] = self.study_data["local_ts"].apply(
            lambda x: pd.to_datetime(x).date() in us_holidays if pd.notna(x) else None
        )

        print(
            "Created location-independent features: day_of_week, end_of_hour, is_weekend, is_holiday"
        )

        # =====================================================================
        # Location-dependent feature transformations (lagged and rolling features)
        # These require processing by location+location_type groups
        # =====================================================================

        # Process each location+location_type combination independently
        results = []
        feature_summaries = []

        for (location, location_type), group in self.study_data.groupby(
            ["location", "location_type"]
        ):
            # Process this single combination
            processed_group = self._generate_independent_vars_single(
                group, location, location_type, self.lag_hours, self.roll_hours
            )
            results.append(processed_group)

            # Create safe identifier for subdirectory
            safe_id = self._create_safe_identifier(location, location_type)
            location_dir = self.output_dir / safe_id
            location_dir.mkdir(parents=True, exist_ok=True)

            # Save this combination's dataset to its own subdirectory
            dataset_file = location_dir / f"{self.dataset_prefix}_{safe_id}.csv"
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
            os.path.join(self.output_dir, f"{self.dataset_prefix}_combined.csv"),
            index=False,
        )

        # Save to database if requested
        # Initialize database processor
        db_path = self.output_dir / f"{self.dataset_prefix}.db"
        db_processor = DatabaseProcessor(str(db_path))

        # Save individual location+location_type datasets to database
        for summary in feature_summaries:
            # Find the corresponding processed data
            location_data = self.study_data[
                (self.study_data["location"] == summary["location"])
                & (self.study_data["location_type"] == summary["location_type"])
            ]

            # Save to database with safe table name
            table_name = f"{self.db_table_prefix}_{summary['safe_id']}"
            db_processor.save_to_database(location_data, table_name)

        print(
            f"Saved {len(feature_summaries)} records to {self.db_table_prefix}_{summary['safe_id']} table in {db_path}"
        )

        # Save combined dataset to database
        db_processor.save_to_database(
            self.study_data, f"{self.db_table_prefix}_study_dataset_combined"
        )
        print(
            f"Saved {len(self.study_data)} records to {self.db_table_prefix}_study_dataset_combined table in {db_path}"
        )

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
        print(f"\nFeature completeness across all combinations:")
        total_records = len(self.study_data)

        for lag_h in self.lag_hours:
            col_name = f"dart_slt_lag_{lag_h}hr"
            non_null_count = self.study_data[col_name].notna().sum()
            print(
                f"  {col_name}: {non_null_count:,} non-null out of {total_records:,} total ({non_null_count/total_records*100:.1f}%)"
            )

        for roll_h in self.roll_hours:
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

        # 19. Feature relationship analysis by hour (for modeling stage)
        plot_dart_slt_vs_features_by_hour(
            df=self.study_data,
            output_dir=self.output_dir,
            dependent_vars=self.dependent_vars,
            independent_vars=self.independent_vars,
            title_suffix=" - Exp0",
        )

    def finalize_study_dataset(self):
        """Finalize study dataset by cleaning and validating data.

        This method performs sequential steps:
        1. Apply categorical encoding (one-hot for cyclical, label for boolean features)
        2. Remove rows with null/NaN dependent variables (dart_slt)
        3. Remove rows with null/NaN independent variables
        4. Validate temporal completeness for each location+location_type
        5. Save final clean datasets in both CSV and database formats

        Reports on all cleaning actions taken.
        """
        if self.study_data is None:
            raise ValueError(
                "No study data available. Run generate_independent_vars() first."
            )

        print("Finalizing study dataset...")

        # Step 0: Apply categorical encoding to prepare data for modeling
        print("Applying categorical encoding...")
        self._apply_categorical_encoding()

        # Get configuration from instance variables (updated after encoding)
        dependent_vars = self.dependent_vars
        independent_vars = self.independent_vars

        # Process each location+location_type combination independently
        final_results = []
        cleaning_summaries = []

        for (location, location_type), group in self.study_data.groupby(
            ["location", "location_type"]
        ):
            safe_id = self._create_safe_identifier(location, location_type)
            print(f"\nProcessing {location} [{location_type}]:")

            # Start with the group data
            df_clean = group.copy().sort_values("utc_ts").reset_index(drop=True)
            initial_count = len(df_clean)

            # Step 1: Remove rows with null/NaN dependent variables
            print(f"  Initial records: {initial_count:,}")

            # Check for null dependent variables
            null_dependent_mask = df_clean[dependent_vars].isnull().any(axis=1)
            null_dependent_count = null_dependent_mask.sum()

            if null_dependent_count > 0:
                # Analyze when nulls occur
                null_rows = df_clean[null_dependent_mask]
                min_date = df_clean["utc_ts"].min().date()

                # Check if nulls occur after day 1
                late_nulls = []
                for _, row in null_rows.iterrows():
                    row_date = row["utc_ts"].date()
                    days_from_start = (row_date - min_date).days
                    if days_from_start >= self.null_check_days_threshold:
                        late_nulls.append((row["utc_ts"], days_from_start))

                # Report null removals
                print(
                    f"  Removing {null_dependent_count:,} rows with null dependent variables"
                )
                for _, row in null_rows.iterrows():
                    print(f"    {row['utc_ts']}: null in {dependent_vars}")

                if late_nulls:
                    print(
                        f"  WARNING: {len(late_nulls)} null dependent variables occur after day 1:"
                    )
                    for ts, days in late_nulls:
                        print(f"    {ts} (day {days + 1})")

                # Remove null dependent variables
                df_clean = df_clean[~null_dependent_mask].reset_index(drop=True)

            # Step 2: Remove rows with null/NaN independent variables
            null_independent_mask = df_clean[independent_vars].isnull().any(axis=1)
            null_independent_count = null_independent_mask.sum()

            if null_independent_count > 0:
                # Report which variables have nulls
                null_summary = {}
                for var in independent_vars:
                    null_count = df_clean[var].isnull().sum()
                    if null_count > 0:
                        null_summary[var] = null_count

                print(
                    f"  Removing {null_independent_count:,} rows with null independent variables:"
                )
                for var, count in null_summary.items():
                    print(f"    {var}: {count:,} nulls")

                # Remove null independent variables
                df_clean = df_clean[~null_independent_mask].reset_index(drop=True)

            # Step 3: Validate temporal completeness
            final_count = len(df_clean)
            print(f"  Final records after cleaning: {final_count:,}")

            temporal_complete = self._validate_temporal_completeness_single(
                df_clean, location, location_type
            )

            if not temporal_complete:
                print(
                    f"  WARNING: Temporal completeness validation failed for {location} [{location_type}]"
                )

            # Save individual clean dataset
            location_dir = self.output_dir / safe_id
            location_dir.mkdir(parents=True, exist_ok=True)

            final_dataset_file = (
                location_dir / f"{self.final_dataset_prefix}_{safe_id}.csv"
            )
            df_clean.to_csv(final_dataset_file, index=False)
            print(f"  Saved final dataset: {final_dataset_file}")

            # Collect results
            final_results.append(df_clean)
            cleaning_summaries.append(
                {
                    "location": location,
                    "location_type": location_type,
                    "safe_id": safe_id,
                    "initial_records": initial_count,
                    "final_records": final_count,
                    "removed_dependent_nulls": null_dependent_count,
                    "removed_independent_nulls": null_independent_count,
                    "temporal_complete": temporal_complete,
                }
            )

        # Combine all final results
        self.final_study_data = pd.concat(final_results, ignore_index=True)
        self.final_study_data = self.final_study_data.sort_values(
            ["location", "location_type", "utc_ts"]
        ).reset_index(drop=True)

        # Save combined final dataset
        final_combined_file = (
            self.output_dir / f"{self.final_dataset_prefix}_combined.csv"
        )
        self.final_study_data.to_csv(final_combined_file, index=False)
        print(f"\nSaved combined final dataset: {final_combined_file}")

        # Save to database
        db_path = self.output_dir / f"{self.final_dataset_prefix}.db"
        db_processor = DatabaseProcessor(str(db_path))

        # Save individual location+location_type final datasets to database
        for summary in cleaning_summaries:
            location_data = self.final_study_data[
                (self.final_study_data["location"] == summary["location"])
                & (self.final_study_data["location_type"] == summary["location_type"])
            ]

            table_name = f"{self.db_table_prefix}_final_{summary['safe_id']}"
            db_processor.save_to_database(location_data, table_name)

        print(f"Saved {len(cleaning_summaries)} final datasets to database: {db_path}")

        # Print comprehensive summary
        print(f"\n=== FINAL DATASET SUMMARY ===")
        total_initial = sum(s["initial_records"] for s in cleaning_summaries)
        total_final = sum(s["final_records"] for s in cleaning_summaries)
        total_removed = total_initial - total_final

        print(f"Total initial records: {total_initial:,}")
        print(f"Total final records: {total_final:,}")
        print(
            f"Total removed records: {total_removed:,} ({total_removed/total_initial*100:.1f}%)"
        )

        print(f"\nPer-location summary:")
        for summary in cleaning_summaries:
            removed = summary["initial_records"] - summary["final_records"]
            removal_pct = (
                removed / summary["initial_records"] * 100
                if summary["initial_records"] > 0
                else 0
            )
            status = "✓" if summary["temporal_complete"] else "⚠"
            print(
                f"  {summary['location']} [{summary['location_type']}]: "
                f"{summary['final_records']:,} records ({removed:,} removed, {removal_pct:.1f}%) {status}"
            )

    def _apply_categorical_encoding(self):
        """Apply categorical encoding to study data.

        - One-hot encode cyclical features (day_of_week)
        - Label encode boolean features (is_weekend, is_holiday)
        - Update independent_vars list to include new feature columns
        """
        # Import required libraries
        from sklearn.preprocessing import LabelEncoder

        # Define encoding strategies
        label_encode_columns = ["is_weekend", "is_holiday"]  # Boolean features
        one_hot_columns = ["day_of_week"]  # Cyclical features

        print(f"  Label encoding: {label_encode_columns}")
        print(f"  One-hot encoding: {one_hot_columns}")

        # Apply label encoding to boolean features
        for cat_col in label_encode_columns:
            if cat_col in self.study_data.columns:
                le = LabelEncoder()
                self.study_data[cat_col] = le.fit_transform(
                    self.study_data[cat_col].astype(str)
                ).astype("float64")
                print(f"    Label encoded {cat_col}: {len(le.classes_)} classes")

        # Apply one-hot encoding to cyclical features
        new_one_hot_columns = []
        for cat_col in one_hot_columns:
            if cat_col in self.study_data.columns:
                # Create one-hot encoded columns
                one_hot_df = pd.get_dummies(self.study_data[cat_col], prefix=cat_col)

                # Add to main dataframe
                self.study_data = pd.concat([self.study_data, one_hot_df], axis=1)

                # Track new column names
                new_one_hot_columns.extend(one_hot_df.columns.tolist())

                # Remove original column
                self.study_data.drop(columns=[cat_col], inplace=True)

                print(
                    f"    One-hot encoded {cat_col}: {len(one_hot_df.columns)} features"
                )
                print(f"      New columns: {list(one_hot_df.columns)}")

        # Update independent_vars to include new one-hot features and remove original
        original_independent_vars = self.independent_vars.copy()

        # Remove original cyclical columns from independent vars
        updated_independent_vars = [
            var for var in original_independent_vars if var not in one_hot_columns
        ]

        # Add new one-hot columns
        updated_independent_vars.extend(new_one_hot_columns)

        # Update the instance variable by overriding the property
        self._independent_vars_override = updated_independent_vars

        print(
            f"  Updated independent variables count: {len(original_independent_vars)} → {len(updated_independent_vars)}"
        )
        print(f"  Added one-hot features: {new_one_hot_columns}")

    def _validate_temporal_completeness_single(
        self, df: pd.DataFrame, location: str, location_type: str
    ) -> bool:
        """Validate temporal completeness for a single location+location_type.

        Args:
            df: DataFrame for single location+location_type
            location: Location name
            location_type: Location type

        Returns:
            bool: True if temporal completeness validation passes
        """
        # Extract date and hour components
        df_temp = df.copy()
        df_temp["date"] = df_temp["utc_ts"].dt.date
        df_temp["hour"] = df_temp["utc_ts"].dt.hour

        # Group by date and count unique hours
        date_hour_counts = df_temp.groupby("date")["hour"].nunique()

        # Calculate expected vs actual days
        min_date = min(df_temp["date"])
        max_date = max(df_temp["date"])
        expected_days = (max_date - min_date).days + 1
        actual_days = len(df_temp["date"].unique())

        print(f"    Date range: {min_date} to {max_date}")
        print(f"    Expected days: {expected_days}, Actual days: {actual_days}")

        if actual_days != expected_days:
            print(
                f"    ERROR: Found {actual_days} dates, expected {expected_days} dates"
            )
            return False

        # Check if any date has less than 24 hours
        incomplete_dates = date_hour_counts[date_hour_counts != 24]

        if len(incomplete_dates) > 0:
            for date, hour_count in incomplete_dates.items():
                if (date == min_date) or (date == max_date):
                    print(f"    WARNING: {date}: {hour_count} hours (boundary date)")
                else:
                    print(f"    ERROR: {date}: {hour_count} hours (should be 24)")
                    return False

            # If only boundary dates have issues, still pass
            non_boundary_incomplete = [
                date
                for date in incomplete_dates.index
                if date != min_date and date != max_date
            ]
            if len(non_boundary_incomplete) > 0:
                return False

        print(
            f"    Temporal completeness: ✓ {len(date_hour_counts)} dates with proper hourly coverage"
        )
        return True
