"""Load Forecast ETL client."""

from typing import Optional
import pandas as pd
from etl.ercot.ercot_etl import ERCOTBaseETL


class LoadForecastETL(ERCOTBaseETL):
    """
    ETL client for ERCOT Load Forecast data.
    
    Note:
    Load Forecast represents the predicted electricity demand
    for each weather zone in the ERCOT system.
    
    See: https://www.ercot.com/mp/data-products/data-product-details?id=NP3-565-CD
    """
    
    ENDPOINT_KEY = "load_forecast"
    
    # Weather zones in ERCOT system
    WEATHER_ZONES = [
        "coast",
        "east",
        "farWest",
        "north",
        "northCentral",
        "southCentral",
        "southern",
        "west"
    ]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Load Forecast data.
        
        Cleaning steps:
        1. Filter for rows where inUseFlag is True
        2. Convert postedDatetime to datetime
        3. Melt weather zone columns into rows (preserving utc_ts and local_ts)
        4. Sort by posted datetime, utc timestamp, and location
        5. Rename DSTFlag to dst_flag for consistency
        
        Args:
            df (pd.DataFrame): Raw DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with one row per posting time,
                         forecast time, and weather zone
        """
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Filter for rows where inUseFlag is True
        df = df.loc[df["inUseFlag"] == True].copy()
        
        # Convert postedDatetime to datetime
        df.loc[:, "posted_datetime"] = pd.to_datetime(df["postedDatetime"])
        
        # Melt weather zone columns into rows
        df_melted = pd.melt(
            df,
            id_vars=["posted_datetime", "utc_ts", "local_ts", "hour_local", "DSTFlag"],
            value_vars=self.WEATHER_ZONES,
            var_name="location",
            value_name="load_forecast"
        )
        
        # Sort by posting time, utc timestamp, and location
        df_melted = df_melted.sort_values(["posted_datetime", "utc_ts", "location"])
        
        # Rename columns to standard format
        df_clean = df_melted.rename(columns={
            "DSTFlag": "dst_flag"
        })
        
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate cleaned Load Forecast data.
        
        Simple validation for development - assumes data types are correct
        since we control the entire pipeline.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame to validate
            
        Returns:
            bool: True if validation passes
        """
        try:
            # Check for missing values in key columns
            if df[["posted_datetime", "utc_ts", "location", "load_forecast"]].isnull().any().any():
                print("Error: Found missing values in key columns")
                return False
            
            # Check location values are valid
            invalid_locations = df[~df["location"].isin(self.WEATHER_ZONES)]["location"].unique()
            if len(invalid_locations) > 0:
                print(f"Error: Found invalid weather zones: {invalid_locations}")
                return False
            
            # Basic row count check
            if len(df) == 0:
                print("Error: No data after cleaning")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False 