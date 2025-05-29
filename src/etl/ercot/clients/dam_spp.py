"""DAM Settlement Point Prices ETL module."""

from typing import Optional
import pandas as pd
from etl.ercot.ercot_etl import ERCOTBaseETL


class DAMSettlementPointPricesETL(ERCOTBaseETL):
    """
    ETL client for ERCOT DAM Settlement Point Prices data.
    
    Note:
    DAM Settlement Point Prices represent the Day-Ahead Market prices
    at various settlement points including Load Zones (LZ_) and Hubs (HB_).
    
    See: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-190-CD
    """
    
    ENDPOINT_KEY = "dam_spp"
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DAM Settlement Point Prices data.
        
        Cleaning steps:
        1. Convert deliveryDate to datetime
        2. Filter for Load Zones and Hubs
        3. Create datetime column from deliveryDate and hourEnding
        4. Sort by datetime and settlement point
        5. Keep only necessary columns
        
        Args:
            df (pd.DataFrame): Raw DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with one row per hour per location
        """
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Convert deliveryDate to datetime
        df.loc[:, "delivery_date"] = pd.to_datetime(df["deliveryDate"]).dt.date
        
        # Filter for Load Zones and Hubs
        mask = df["settlementPoint"].str.startswith(("LZ_", "HB_"))
        df = df.loc[mask].copy()
        
        # Create datetime column
        df.loc[:, "datetime"] = df.apply(
            lambda row: self.combine_date_hour(row["deliveryDate"], row["hourEnding"]),
            axis=1
        )
        
        # Sort by datetime and settlement point
        df = df.sort_values(["datetime", "settlementPoint"])
        
        # Select and rename columns
        df_clean = df[[
            "datetime",
            "settlementPoint",
            "settlementPointPrice",
            "DSTFlag"
        ]].copy()
        
        # Rename columns to standard format
        df_clean = df_clean.rename(columns={
            "settlementPoint": "location",
            "settlementPointPrice": "price",
            "DSTFlag": "dst_flag"
        })
        
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate cleaned DAM Settlement Point Prices data.
        
        Validation rules:
        1. No missing values
        2. datetime is datetime type
        3. price is numeric
        4. location starts with either LZ_ or HB_
        5. Data is sorted by datetime and location
        6. One row per hour per location
        
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
            
            # Check location prefixes
            invalid_locations = df[~df["location"].str.startswith(("LZ_", "HB_"))]["location"].unique()
            if len(invalid_locations) > 0:
                print(f"Error: Found locations with invalid prefixes: {invalid_locations}")
                return False
            
            # Check sorting
            if not df.equals(df.sort_values(["datetime", "location"])):
                print("Error: Data is not sorted by datetime and location")
                return False
            
            # Check for duplicates (should be one row per datetime per location)
            duplicates = df.groupby(["datetime", "location"]).size()
            if (duplicates > 1).any():
                print("Error: Found duplicate entries for datetime and location")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False 