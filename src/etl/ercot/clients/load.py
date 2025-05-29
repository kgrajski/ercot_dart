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
        3. Create datetime column from deliveryDate and hourEnding
        4. Melt weather zone columns into rows
        5. Sort by posted datetime, forecast datetime, and location
        6. Keep only necessary columns
        
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
        
        # Create forecast datetime column
        df.loc[:, "datetime"] = df.apply(
            lambda row: self.combine_date_hour(row["deliveryDate"], row["hourEnding"]),
            axis=1
        )
        
        # Melt weather zone columns into rows
        df_melted = pd.melt(
            df,
            id_vars=["posted_datetime", "datetime", "DSTFlag"],
            value_vars=self.WEATHER_ZONES,
            var_name="location",
            value_name="load_forecast"
        )
        
        # Sort by posting time, forecast time, and location
        df_melted = df_melted.sort_values(["posted_datetime", "datetime", "location"])
        
        # Rename columns to standard format
        df_clean = df_melted.rename(columns={
            "DSTFlag": "dst_flag"
        })
        
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate cleaned Load Forecast data.
        
        Validation rules:
        1. No missing values
        2. posted_datetime is datetime type
        3. datetime is datetime type
        4. load_forecast is numeric
        5. location is one of the valid weather zones
        6. Data is sorted by posted_datetime, datetime, and location
        7. One row per posting time, forecast time, and location combination
        
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
            
            # Check posted_datetime type
            if not pd.api.types.is_datetime64_any_dtype(df["posted_datetime"]):
                print("Error: posted_datetime column is not datetime type")
                return False
            
            # Check datetime type
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                print("Error: datetime column is not datetime type")
                return False
            
            # Check load_forecast is numeric
            if not pd.api.types.is_numeric_dtype(df["load_forecast"]):
                print("Error: load_forecast column is not numeric")
                return False
            
            # Check location values
            invalid_locations = df[~df["location"].isin(self.WEATHER_ZONES)]["location"].unique()
            if len(invalid_locations) > 0:
                print(f"Error: Found invalid weather zones: {invalid_locations}")
                return False
            
            # Check sorting
            if not df.equals(df.sort_values(["posted_datetime", "datetime", "location"])):
                print("Error: Data is not sorted by posted_datetime, datetime, and location")
                return False
            
            # Check for duplicates
            duplicates = df.groupby(["posted_datetime", "datetime", "location"]).size()
            if (duplicates > 1).any():
                print("Error: Found duplicate entries for posted_datetime, datetime, and location")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False 