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
        1. Convert deliveryDate to datetime
        2. Filter for Load Zones and Hubs by name prefix
        3. Create datetime column from deliveryDate, deliveryHour, and deliveryInterval
        4. Sort by datetime, location, type, hour, and interval
        5. Keep only necessary columns
        
        Args:
            df (pd.DataFrame): Raw DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with one row per 15-minute interval per location
        """
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Convert deliveryDate to datetime
        df.loc[:, "delivery_date"] = pd.to_datetime(df["deliveryDate"]).dt.date
        
        # Filter for Load Zones and Hubs by name prefix only
        mask = df["settlementPoint"].str.startswith(("LZ_", "HB_"))
        df = df.loc[mask].copy()
        
        # Create datetime column combining date, hour, and interval
        df.loc[:, "datetime"] = df.apply(
            lambda row: pd.to_datetime(row["deliveryDate"]) + 
                      pd.Timedelta(hours=row["deliveryHour"]) +
                      pd.Timedelta(minutes=(row["deliveryInterval"] - 1) * 15),
            axis=1
        )
        
        # Select columns before renaming
        df_clean = df[[
            "datetime",
            "settlementPoint",
            "settlementPointType",
            "deliveryHour",
            "deliveryInterval",
            "settlementPointPrice",
            "DSTFlag"
        ]].copy()
        
        # Sort by datetime, settlement point, type, hour, and interval
        df_clean = df_clean.sort_values([
            "datetime",
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
        1. Create hourly_datetime by truncating datetime to hour
        2. Group by hourly_datetime, location, and location_type
        3. Calculate mean and standard deviation of prices across the 4 intervals
        4. Round statistics to 6 decimal places
        5. Sort by datetime, location, and type
        6. Reorder columns to move dst_flag to end
        
        Note: Each hour has 4 intervals with different timestamps
        (e.g., 01:00, 01:15, 01:30, 01:45). We truncate to hour to group these.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame with 15-minute interval data
            
        Returns:
            pd.DataFrame: Transformed DataFrame with hourly price statistics
        """
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Create hourly datetime by truncating to hour
        df.loc[:, "hourly_datetime"] = df["datetime"].dt.floor('h')
        
        # Group by hour, location, and type
        df_hourly = df.groupby([
            "hourly_datetime",
            "location",
            "location_type",
            "dst_flag"
        ]).agg({
            "price": ["mean", "std"]
        }).reset_index()
        
        # Flatten multi-level columns and rename
        df_hourly.columns = [
            "datetime",
            "location",
            "location_type",
            "dst_flag",
            "price_mean",
            "price_std"
        ]
        
        # Round the statistics to 6 decimal places
        df_hourly.loc[:, "price_mean"] = df_hourly["price_mean"].round(6)
        df_hourly.loc[:, "price_std"] = df_hourly["price_std"].round(6)
        
        # Sort by datetime, location, and type (matching DAM clients)
        df_hourly = df_hourly.sort_values([
            "datetime",
            "location",
            "location_type"
        ])
        
        # Reorder columns to move dst_flag to end
        df_hourly = df_hourly[[
            "datetime",
            "location",
            "location_type",
            "price_mean",
            "price_std",
            "dst_flag"
        ]]
        
        return df_hourly
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate cleaned RT Settlement Point Prices data.
        
        Validation rules:
        1. No missing values
        2. datetime is datetime type
        3. price is numeric
        4. location starts with either LZ_ or HB_
        5. location_type is not empty/missing
        6. Data is sorted by datetime, location, type, hour, and interval
        7. One row per datetime, location, type, hour, and interval combination
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame to validate
            
        Returns:
            bool: True if validation passes
        """
        try:
            # Check for missing values
            if df.isnull().any().any():
                print("Error: Found missing values in cleaned data")
                return False
            
            # Check datetime type
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                print("Error: datetime column is not datetime type")
                return False
            
            # Check price is numeric
            if not pd.api.types.is_numeric_dtype(df["price"]):
                print("Error: price column is not numeric")
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
            
            # Check sorting
            sorted_df = df.sort_values([
                "datetime",
                "location",
                "location_type",
                "deliveryHour",
                "deliveryInterval"
            ])
            if not df.equals(sorted_df):
                print("Error: Data is not sorted by datetime, location, type, hour, and interval")
                return False
            
            # Check for duplicates
            duplicates = df.groupby([
                "datetime",
                "location",
                "location_type",
                "deliveryHour",
                "deliveryInterval"
            ]).size()
            if (duplicates > 1).any():
                print("Error: Found duplicate entries for datetime, location, type, hour, and interval")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False 