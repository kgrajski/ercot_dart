"""RT Settlement Point Prices ETL client."""

from typing import Optional
import pandas as pd
from etl.ercot.ercot_etl import ERCOTBaseETL


class RTSettlementPointPricesETL(ERCOTBaseETL):
    """
    ETL client for ERCOT RT Settlement Point Prices data.
    
    Note:
    RT Settlement Point Prices represent the Real-Time Market prices
    at various settlement points including Load Zones (LZ_) and Hubs (HB_).
    Data is provided in 15-minute intervals.
    
    Settlement Point Types:
    - HU: Hub (individual hub prices)
    - SH: Settlement Hub (bus average prices)
    - AH: Aggregate Hub (hub average prices)
    - LZ: Load Zone
    
    Important Note on Unique Keys:
    A settlement point is uniquely identified by the combination of:
    - Settlement Point Name (e.g., "LZ_HOUSTON" or "HB_NORTH")
    - Settlement Point Type (e.g., "LZ" or "HU")
    
    While in practice certain settlement points may only have one type
    (e.g., each HB_ point might only appear with type "HU"), we treat
    all points uniformly and use the combination of name + type as the
    unique identifier. This simplifies processing and makes the code
    more robust to potential data changes.
    
    See: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-905-CD
    """
    
    ENDPOINT_KEY = "rt_spp"
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean RT Settlement Point Prices data.
        
        Cleaning steps:
        1. Filter for Load Zones and Hubs by name prefix
        2. Sort by utc timestamp, location, type, hour, and interval
        3. Rename columns to standard format
        
        Args:
            df (pd.DataFrame): Raw DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with one row per 15-minute interval per location
        """
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Filter for Load Zones and Hubs by name prefix only
        mask = df["settlementPoint"].str.startswith(("LZ_", "HB_"))
        df = df.loc[mask].copy()
        
        # Select columns before renaming
        df_clean = df[[
            "utc_ts",
            "local_ts",
            "hour_local",
            "settlementPoint",
            "settlementPointType",
            "deliveryHour",
            "deliveryInterval",
            "settlementPointPrice",
            "DSTFlag"
        ]].copy()
        
        # Sort by utc timestamp, settlement point, type, hour, and interval
        df_clean = df_clean.sort_values([
            "utc_ts",
            "settlementPoint",
            "settlementPointType",
            "deliveryHour",
            "deliveryInterval"
        ])
        
        # Rename columns to standard format (matching DAM clients)
        df_clean = df_clean.rename(columns={
            "settlementPoint": "location",
            "settlementPointType": "location_type",
            "settlementPointPrice": "price",
            "DSTFlag": "dst_flag"
        })
        
        return df_clean
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform cleaned RT Settlement Point Prices data to hourly statistics.
        
        Transformation steps:
        1. Create hourly_utc_ts by truncating utc_ts to hour
        2. Group by hourly_utc_ts, location, and location_type
        3. Calculate mean and standard deviation of prices across the 4 intervals
        4. Round statistics to 6 decimal places
        5. Sort by utc timestamp, location, and type
        6. Reorder columns appropriately
        
        Note: Each hour has 4 intervals with different timestamps
        (e.g., 01:00, 01:15, 01:30, 01:45). We truncate to hour to group these.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame with 15-minute interval data
            
        Returns:
            pd.DataFrame: Transformed DataFrame with hourly price statistics
        """
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Create hourly utc timestamp by truncating to hour
        df.loc[:, "hourly_utc_ts"] = df["utc_ts"].dt.floor('h')
        
        # Also create hourly local timestamp for display purposes
        df.loc[:, "hourly_local_ts"] = df["local_ts"].dt.floor('h')
        
        # Group by hour, location, and type
        df_hourly = df.groupby([
            "hourly_utc_ts",
            "hourly_local_ts",
            "hour_local",
            "location",
            "location_type",
            "dst_flag"
        ]).agg({
            "price": ["mean", "std"]
        }).reset_index()
        
        # Flatten multi-level columns and rename
        df_hourly.columns = [
            "utc_ts",
            "local_ts",
            "hour_local", 
            "location",
            "location_type",
            "dst_flag",
            "price_mean",
            "price_std"
        ]
        
        # Round the statistics to 6 decimal places
        df_hourly.loc[:, "price_mean"] = df_hourly["price_mean"].round(6)
        df_hourly.loc[:, "price_std"] = df_hourly["price_std"].round(6)
        
        # Sort by utc timestamp, location, and type
        df_hourly = df_hourly.sort_values([
            "utc_ts",
            "location",
            "location_type"
        ])
        
        # Reorder columns to match other clients
        df_hourly = df_hourly[[
            "utc_ts",
            "local_ts",
            "hour_local",
            "location",
            "location_type",
            "price_mean",
            "price_std",
            "dst_flag"
        ]]
        
        return df_hourly
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate cleaned RT Settlement Point Prices data.
        
        Simple validation for development - assumes data types are correct
        since we control the entire pipeline.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame to validate
            
        Returns:
            bool: True if validation passes
        """
        try:
            # Check for missing values in key columns
            if df[["utc_ts", "location", "location_type", "price"]].isnull().any().any():
                print("Error: Found missing values in key columns")
                return False
            
            # Check location format
            invalid_locations = df[~df["location"].str.startswith(("LZ_", "HB_"))]["location"].unique()
            if len(invalid_locations) > 0:
                print(f"Error: Found invalid settlement points: {invalid_locations}")
                return False
            
            # Check location type is not empty
            if df["location_type"].isna().any() or (df["location_type"] == "").any():
                print("Error: Found empty or missing location types")
                return False
            
            # Basic row count check
            if len(df) == 0:
                print("Error: No data after cleaning")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False 