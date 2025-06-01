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
        1. Filter for Load Zones and Hubs
        2. Sort by utc timestamp and settlement point
        3. Rename columns to standard format
        
        Args:
            df (pd.DataFrame): Raw DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with one row per hour per location
        """
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Filter for Load Zones and Hubs
        mask = df["settlementPoint"].str.startswith(("LZ_", "HB_"))
        df = df.loc[mask].copy()
        
        # Sort by utc timestamp and settlement point
        df = df.sort_values(["utc_ts", "settlementPoint"])
        
        # Select and rename columns
        df_clean = df[[
            "utc_ts",
            "local_ts",
            "hour_local", 
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
        
        Simple validation for development - assumes data types are correct
        since we control the entire pipeline.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame to validate
            
        Returns:
            bool: True if validation passes
        """
        try:
            # Check for missing values in key columns
            if df[["utc_ts", "location", "price"]].isnull().any().any():
                print("Error: Found missing values in key columns")
                return False
            
            # Check location prefixes are valid
            invalid_locations = df[~df["location"].str.startswith(("LZ_", "HB_"))]["location"].unique()
            if len(invalid_locations) > 0:
                print(f"Error: Found locations with invalid prefixes: {invalid_locations}")
                return False
            
            # Basic row count check
            if len(df) == 0:
                print("Error: No data after cleaning")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False 