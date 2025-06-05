"""DAM System Lambda ETL module."""

from typing import Optional
import pandas as pd
from etl.ercot.ercot_etl import ERCOTBaseETL


class DAMSystemLambdaETL(ERCOTBaseETL):
    """
    ETL client for ERCOT DAM System Lambda data.
    
    Note:
    DAM System Lambda represents the system-wide marginal price of energy 
    for each hour in the Day-Ahead Market.
    
    See: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-523-CD
    """
    
    ENDPOINT_KEY = "dam_system_lambda"
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DAM System Lambda data.
        
        Cleaning steps:
        1. Sort by utc timestamp
        2. Rename columns to standard format
        
        Args:
            df (pd.DataFrame): Raw DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with one row per hour
        """
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Sort by utc timestamp
        df = df.sort_values("utc_ts")
        
        # Select and rename columns
        df_clean = df[[
            "utc_ts",
            "local_ts",
            "systemLambda",
            "DSTFlag"
        ]].copy()
        
        # Rename columns to standard format
        df_clean = df_clean.rename(columns={
            "systemLambda": "price",
            "DSTFlag": "dst_flag"
        })
        
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate cleaned DAM System Lambda data.
        
        Simple validation for development - assumes data types are correct
        since we control the entire pipeline.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame to validate
            
        Returns:
            bool: True if validation passes
        """
        try:
            # Check for missing values in key columns
            if df[["utc_ts", "price"]].isnull().any().any():
                print("Error: Found missing values in key columns")
                return False
            
            # Basic row count check
            if len(df) == 0:
                print("Error: No data after cleaning")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False 