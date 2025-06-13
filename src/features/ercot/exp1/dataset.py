"""Experiment 1 dataset implementation."""

import os

import holidays
import pandas as pd

from src.data.ercot.database import DatabaseProcessor
from src.features.ercot.exp_dataset import ExpDataset
from src.features.ercot.visualization import plot_overlay_scatter
from src.features.utils import signed_log_transform


class Exp1Dataset(ExpDataset):
    """Dataset handler for Experiment 1.

    This experiment focuses on:
    - Using transformed RT SPP data with hourly statistics
    - Combining with cleaned data from other sources
    - Creating baseline features for price prediction
    """

    def __init__(self, input_dir: str, output_dir: str):
        """Initialize Exp1 dataset handler.

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
            experiment_id="exp1",
        )

        # Experiment configuration
        self.experiment_name = "exp1"
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
        self.dependent_vars = ["dart_slt"]  # Target variable for modeling
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
        """Generate dependent variables for Exp1.

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

        # Write the result_df to a csv file, but with the name exp1_study_dataset.csv
        self.study_data.to_csv(
            os.path.join(self.output_dir, "exp1_study_dataset.csv"), index=False
        )

    def generate_independent_vars(self):
        """
        Generate all independent variables for Exp1 by sequentially adding features
        from each data source (DART lags, load, wind, solar, weather, etc.).
        """
        if self.study_data is None:
            raise ValueError(
                "No study data available. Run generate_dependent_vars() first."
            )

        df = self.study_data.copy()

        # Add DART SLT lagged and rolling features
        df = self._add_dart_lagged_and_rolling_features(df)

        # Add load forecast features
        df = self._add_load_features(df)

        # Add wind forecast and actuals
        df = self._add_wind_features(df)

        # Add solar forecast and actuals
        df = self._add_solar_features(df)

        # Add weather features
        df = self._add_weather_features(df)

        # ...add more as needed...

        self.study_data = df

    def _add_dart_lagged_and_rolling_features(self, df):
        """
        Add DART SLT lagged and rolling features to the DataFrame.
        This logic is refactored from the previous _generate_independent_vars_single and
        location-independent feature section.
        """
        # Location-independent features (day_of_week, end_of_hour, is_weekend, is_holiday)
        #
        # NOTE: It is safe to use day_of_week, is_weekend, and is_holiday as features for forecasting,
        # because these are fully determined by the delivery date and are known in advance for any
        # forecast. There is no information leakage, as long as the holiday calendar is fixed and public.
        df["day_of_week"] = df["local_ts"].apply(
            lambda x: pd.to_datetime(x).dayofweek if pd.notna(x) else None
        )
        df["end_of_hour"] = df["local_ts"].apply(
            lambda x: pd.to_datetime(x).hour + 1 if pd.notna(x) else None
        )
        df["is_weekend"] = df["day_of_week"].apply(
            lambda x: x >= 5 if pd.notna(x) else None
        )
        us_holidays = holidays.UnitedStates(state="TX", years=range(2015, 2030))
        df["is_holiday"] = df["local_ts"].apply(
            lambda x: pd.to_datetime(x).date() in us_holidays if pd.notna(x) else None
        )
        print(
            "Created location-independent features: day_of_week, end_of_hour, is_weekend, is_holiday"
        )

        # Location-dependent lagged and rolling features
        results = []
        for (location, location_type), group in df.groupby(
            ["location", "location_type"]
        ):
            group = group.sort_values("utc_ts").reset_index(drop=True)
            # Lagged features
            for lag_h in self.lag_hours:
                col_name = f"dart_slt_lag_{lag_h}hr"
                group[col_name] = group["dart_slt"].shift(lag_h)
            # Rolling features (shifted to end 24 hours before prediction time)
            for roll_h in self.roll_hours:
                roll_mean_col = f"dart_slt_roll_mean_{roll_h}hr"
                group[roll_mean_col] = (
                    group["dart_slt"]
                    .shift(24)
                    .rolling(window=roll_h, min_periods=1)
                    .mean()
                )
                roll_sdev_col = f"dart_slt_roll_sdev_{roll_h}hr"
                group[roll_sdev_col] = (
                    group["dart_slt"]
                    .shift(24)
                    .rolling(window=roll_h, min_periods=1)
                    .std()
                )
            results.append(group)
        updated_df = pd.concat(results, ignore_index=True)
        updated_df = updated_df.sort_values(
            ["location", "location_type", "utc_ts"]
        ).reset_index(drop=True)
        print("Added DART SLT lagged and rolling features.")
        return updated_df

    def _add_load_features(self, df):
        """
        Add load forecast features for each weather zone.
        For each sample (row), use the most recent load forecast (by posted_datetime)
        for the delivery hour, where posted_datetime is ≤ (delivery date - 1 day).
        Adds 8 columns: load_forecast_{zone} for each weather zone.
        """
        load_df = self.raw_data["load_forecast"].copy()
        load_df["utc_ts"] = pd.to_datetime(load_df["utc_ts"])
        load_df["posted_datetime"] = pd.to_datetime(load_df["posted_datetime"])

        def get_latest_forecast_map(load_df):
            """
            For each (delivery_datetime, zone), find the single latest forecast
            posted on or before the day before delivery (i.e., posted_date <= delivery_date - 1).
            Returns a DataFrame with columns: utc_ts, location, load_forecast.
            """
            load_df = load_df.copy()
            load_df["delivery_date"] = load_df["utc_ts"].dt.date
            load_df["posted_date"] = load_df["posted_datetime"].dt.date
            # Only keep forecasts posted on or before the day before delivery
            load_df["max_allowed_posted_date"] = load_df[
                "delivery_date"
            ] - pd.Timedelta(days=1)
            valid_mask = load_df["posted_date"] <= load_df["max_allowed_posted_date"]
            filtered = load_df[valid_mask]
            # For each (utc_ts, location), get the row with the latest posted_datetime
            idx = filtered.groupby(["utc_ts", "location"])["posted_datetime"].idxmax()
            latest = filtered.loc[idx][["utc_ts", "location", "load_forecast"]]
            latest = latest.reset_index(drop=True)
            return latest

        latest_forecasts = get_latest_forecast_map(load_df)
        pivot = latest_forecasts.pivot(
            index="utc_ts", columns="location", values="load_forecast"
        )
        pivot.columns = [f"load_forecast_{zone}" for zone in pivot.columns]
        pivot = pivot.reset_index()

        # Process each location/location_type combination
        results = []
        for (location, location_type), group in df.groupby(
            ["location", "location_type"]
        ):
            # Merge forecasts for this group
            group = group.merge(pivot, on="utc_ts", how="left")

            # Check merge results
            def _check_merge_results(original_df, merged_df, new_columns, context=""):
                n_rows_orig = len(original_df)
                n_rows_merged = len(merged_df)
                if n_rows_merged != n_rows_orig:
                    print(
                        f"[{context}] WARNING: Row count changed after merge! {n_rows_orig} -> {n_rows_merged}"
                    )
                for col in new_columns:
                    n_missing = merged_df[col].isnull().sum()
                    print(
                        f"[{context}] {col}: {n_missing} missing ({n_missing/n_rows_merged:.2%})"
                    )
                    print(
                        f"[{context}] {col}: min={merged_df[col].min()}, max={merged_df[col].max()}, mean={merged_df[col].mean()}"
                    )
                if any(merged_df[col].isnull().all() for col in new_columns):
                    print(
                        f"[{context}] ERROR: At least one new column is entirely null after merge!"
                    )

            _check_merge_results(
                group,
                group,
                list(pivot.columns[1:]),
                context=f"load_forecast_{location}_{location_type}",
            )

            # Create visualization for this location
            safe_id = self._create_safe_identifier(location, location_type)
            location_dir = os.path.join(self.output_dir, safe_id, "eda")
            os.makedirs(location_dir, exist_ok=True)

            feature_cols = [c for c in group.columns if c.startswith("load_forecast_")]
            plot_overlay_scatter(
                df=group,
                feature_cols=feature_cols,
                target_col="dart_slt",
                output_dir=location_dir,
                prefix="load_vs_dart_slt",
                transform_fn=None,
                sample_frac=1.0,
                title_suffix=f" - {location} ({location_type})",
            )

            results.append(group)

        # Combine all results and sort properly
        updated_df = pd.concat(results, ignore_index=True)
        updated_df = updated_df.sort_values(
            ["location", "location_type", "utc_ts"]
        ).reset_index(drop=True)
        return updated_df

    def _add_wind_features(self, df):
        """
        Add wind generation forecast features for each wind zone.
        For each sample (row), use the most recent wind forecast (by posted_datetime)
        for the delivery hour, where posted_datetime is ≤ (delivery date - 1 day).
        Adds columns: wind_generation_{zone} for each wind zone.
        """
        wind_df = self.raw_data["wind_power_gen"].copy()
        wind_df["utc_ts"] = pd.to_datetime(wind_df["utc_ts"])
        wind_df["posted_datetime"] = pd.to_datetime(wind_df["posted_datetime"])

        def get_latest_forecast_map(wind_df):
            """
            For each (delivery_datetime, zone), find the single latest forecast
            posted on or before the day before delivery (i.e., posted_date <= delivery_date - 1).
            Returns a DataFrame with columns: utc_ts, location, wind_generation.
            """
            wind_df = wind_df.copy()
            wind_df["delivery_date"] = wind_df["utc_ts"].dt.date
            wind_df["posted_date"] = wind_df["posted_datetime"].dt.date
            wind_df["max_allowed_posted_date"] = wind_df[
                "delivery_date"
            ] - pd.Timedelta(days=1)
            valid_mask = wind_df["posted_date"] <= wind_df["max_allowed_posted_date"]
            filtered = wind_df[valid_mask]
            idx = filtered.groupby(["utc_ts", "location"])["posted_datetime"].idxmax()
            latest = filtered.loc[idx][["utc_ts", "location", "wind_generation"]]
            latest = latest.reset_index(drop=True)
            return latest

        latest_forecasts = get_latest_forecast_map(wind_df)
        pivot = latest_forecasts.pivot(
            index="utc_ts", columns="location", values="wind_generation"
        )
        pivot.columns = [f"wind_generation_{zone}" for zone in pivot.columns]
        pivot = pivot.reset_index()

        # Process each location/location_type combination
        results = []
        for (location, location_type), group in df.groupby(
            ["location", "location_type"]
        ):
            # Merge forecasts for this group
            group = group.merge(pivot, on="utc_ts", how="left")

            # Check merge results
            def _check_merge_results(original_df, merged_df, new_columns, context=""):
                n_rows_orig = len(original_df)
                n_rows_merged = len(merged_df)
                if n_rows_merged != n_rows_orig:
                    print(
                        f"[{context}] WARNING: Row count changed after merge! {n_rows_orig} -> {n_rows_merged}"
                    )
                for col in new_columns:
                    n_missing = merged_df[col].isnull().sum()
                    print(
                        f"[{context}] {col}: {n_missing} missing ({n_missing/n_rows_merged:.2%})"
                    )
                    print(
                        f"[{context}] {col}: min={merged_df[col].min()}, max={merged_df[col].max()}, mean={merged_df[col].mean()}"
                    )
                if any(merged_df[col].isnull().all() for col in new_columns):
                    print(
                        f"[{context}] ERROR: At least one new column is entirely null after merge!"
                    )

            _check_merge_results(
                group,
                group,
                list(pivot.columns[1:]),
                context=f"wind_generation_{location}_{location_type}",
            )

            # Create visualization for this location
            safe_id = self._create_safe_identifier(location, location_type)
            location_dir = os.path.join(self.output_dir, safe_id, "eda")
            os.makedirs(location_dir, exist_ok=True)

            feature_cols = [
                c for c in group.columns if c.startswith("wind_generation_")
            ]
            plot_overlay_scatter(
                df=group,
                feature_cols=feature_cols,
                target_col="dart_slt",
                output_dir=location_dir,
                prefix="wind_vs_dart_slt",
                transform_fn=None,
                sample_frac=1.0,
                title_suffix=f" - {location} ({location_type})",
            )

            results.append(group)

        # Combine all results and sort properly
        updated_df = pd.concat(results, ignore_index=True)
        updated_df = updated_df.sort_values(
            ["location", "location_type", "utc_ts"]
        ).reset_index(drop=True)
        return updated_df

    def _add_solar_features(self, df):
        """
        Add solar generation forecast features for each solar zone.
        For each sample (row), use the most recent solar forecast (by posted_datetime)
        for the delivery hour, where posted_datetime is ≤ (delivery date - 1 day).
        Adds columns: solar_{zone} for each solar zone.
        """
        solar_df = self.raw_data["solar_power_gen"].copy()
        solar_df["utc_ts"] = pd.to_datetime(solar_df["utc_ts"])
        solar_df["posted_datetime"] = pd.to_datetime(solar_df["posted_datetime"])

        def get_latest_forecast_map(solar_df):
            """
            For each (delivery_datetime, zone), find the single latest forecast
            posted on or before the day before delivery (i.e., posted_date <= delivery_date - 1).
            Returns a DataFrame with columns: utc_ts, location, solar_generation.
            """
            solar_df = solar_df.copy()
            solar_df["delivery_date"] = solar_df["utc_ts"].dt.date
            solar_df["posted_date"] = solar_df["posted_datetime"].dt.date
            solar_df["max_allowed_posted_date"] = solar_df[
                "delivery_date"
            ] - pd.Timedelta(days=1)
            valid_mask = solar_df["posted_date"] <= solar_df["max_allowed_posted_date"]
            filtered = solar_df[valid_mask]
            idx = filtered.groupby(["utc_ts", "location"])["posted_datetime"].idxmax()
            latest = filtered.loc[idx][["utc_ts", "location", "solar_generation"]]
            latest = latest.reset_index(drop=True)
            return latest

        latest_forecasts = get_latest_forecast_map(solar_df)
        pivot = latest_forecasts.pivot(
            index="utc_ts", columns="location", values="solar_generation"
        )
        pivot.columns = [f"solar_{zone}" for zone in pivot.columns]
        pivot = pivot.reset_index()

        # Process each location/location_type combination
        results = []
        for (location, location_type), group in df.groupby(
            ["location", "location_type"]
        ):
            # Merge forecasts for this group
            group = group.merge(pivot, on="utc_ts", how="left")

            # Check merge results
            def _check_merge_results(original_df, merged_df, new_columns, context=""):
                n_rows_orig = len(original_df)
                n_rows_merged = len(merged_df)
                if n_rows_merged != n_rows_orig:
                    print(
                        f"[{context}] WARNING: Row count changed after merge! {n_rows_orig} -> {n_rows_merged}"
                    )
                for col in new_columns:
                    n_missing = merged_df[col].isnull().sum()
                    print(
                        f"[{context}] {col}: {n_missing} missing ({n_missing/n_rows_merged:.2%})"
                    )
                    print(
                        f"[{context}] {col}: min={merged_df[col].min()}, max={merged_df[col].max()}, mean={merged_df[col].mean()}"
                    )
                if any(merged_df[col].isnull().all() for col in new_columns):
                    print(
                        f"[{context}] ERROR: At least one new column is entirely null after merge!"
                    )

            _check_merge_results(
                group,
                group,
                list(pivot.columns[1:]),
                context=f"solar_generation_{location}_{location_type}",
            )

            # Create visualization for this location
            safe_id = self._create_safe_identifier(location, location_type)
            location_dir = os.path.join(self.output_dir, safe_id, "eda")
            os.makedirs(location_dir, exist_ok=True)

            feature_cols = [c for c in group.columns if c.startswith("solar_")]
            plot_overlay_scatter(
                df=group,
                feature_cols=feature_cols,
                target_col="dart_slt",
                output_dir=location_dir,
                prefix="solar_vs_dart_slt",
                transform_fn=None,
                sample_frac=1.0,
                title_suffix=f" - {location} ({location_type})",
            )

            results.append(group)

        # Combine all results and sort properly
        updated_df = pd.concat(results, ignore_index=True)
        updated_df = updated_df.sort_values(
            ["location", "location_type", "utc_ts"]
        ).reset_index(drop=True)
        return updated_df

    def _add_weather_features(self, df):
        """
        Add weather features (e.g., temperature, wind speed, etc.).
        TODO: Implement time-aligned merging and feature engineering for weather data.
        """
        # TODO: Merge weather data, align by utc_ts/location, create features as needed
        print("[TODO] Add weather features (not yet implemented)")
        return df

    def run_eda(self):
        """
        Placeholder for exploratory data analysis and visualization for Exp1.
        Implement new plots and analyses as new independent variables are added.
        """
        print("[TODO] Implement new EDA/visualization for Exp1 features.")

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
        self._apply_categorical_encoding(
            label_encode_columns=["is_weekend", "is_holiday"],
            one_hot_columns=["day_of_week"],
        )

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
