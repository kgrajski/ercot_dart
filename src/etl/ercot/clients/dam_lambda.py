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
        1. Convert deliveryDate to datetime
        2. Create datetime column from deliveryDate and hourEnding
        3. Sort by datetime
        4. Keep only necessary columns
        
        Args:
            df (pd.DataFrame): Raw DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with one row per hour
        """
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Convert deliveryDate to datetime
        df.loc[:, "delivery_date"] = pd.to_datetime(df["deliveryDate"]).dt.date
        
        # Create datetime column
        df.loc[:, "datetime"] = df.apply(
            lambda row: self.combine_date_hour(row["deliveryDate"], row["hourEnding"]),
            axis=1
        )
        
        # Sort by datetime
        df = df.sort_values("datetime")
        
        # Select and rename columns
        df_clean = df[[
            "datetime",
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
        
        Validation rules:
        1. No missing values
        2. datetime is datetime type
        3. price is numeric
        4. Data is sorted by datetime
        5. One row per hour
        
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
            
            # Check sorting
            if not df.equals(df.sort_values("datetime")):
                print("Error: Data is not sorted by datetime")
                return False
            
            # Check for duplicates (should be one row per datetime)
            duplicates = df.groupby("datetime").size()
            if (duplicates > 1).any():
                print("Error: Found duplicate entries for datetime")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False 