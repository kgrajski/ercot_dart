"""Wind Generation ETL client."""

from typing import Optional
import pandas as pd
from etl.ercot.ercot_etl import ERCOTBaseETL


class WindGenerationETL(ERCOTBaseETL):
    """
    ETL client for ERCOT Wind Generation data.
    
    Note:
    See: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-742-CD
    """
    
    ENDPOINT_KEY = "wind_power_gen"
    
    # Geographical zones and forecast types from visualization client
    GEOGRAPHICAL_ZONES = [
        "COPHSLCoastal",
        "COPHSLNorth",
        "COPHSLPanhandle",
        "COPHSLSouth",
        "COPHSLSystemWide",
        "COPHSLWest",
        "genCoastal",
        "genNorth",
        "genPanhandle",
        "genSouth",
        "genSystemWide",
        "genWest",
        "HSLSystemWide",
        "STWPFCoastal",
        "STWPFNorth",
        "STWPFPanhandle",
        "STWPFSouth",
        "STWPFSystemWide",
        "STWPFWest",
        "WGRPPCoastal",
        "WGRPPNorth",
        "WGRPPPanhandle",
        "WGRPPSouth",
        "WGRPPSystemWide",
        "WGRPPWest"
    ]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Wind Generation data.
        
        Cleaning steps:
        1. Convert postedDatetime to datetime
        2. Create datetime column from deliveryDate and hourEnding
        3. Melt geographical zone columns into rows
        4. Convert empty strings to NaN and drop rows with NaN
        5. Sort by posted datetime, forecast datetime, and zone
        6. Keep only necessary columns
        
        Args:
            df (pd.DataFrame): Raw DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with one row per posting time,
                         forecast time, and geographical zone
        """
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Convert postedDatetime to datetime
        df.loc[:, "posted_datetime"] = pd.to_datetime(df["postedDatetime"])
        
        # Create forecast datetime column
        df.loc[:, "datetime"] = df.apply(
            lambda row: self.combine_date_hour(row["deliveryDate"], row["hourEnding"]),
            axis=1
        )
        
        # Melt geographical zone columns into rows
        df_melted = pd.melt(
            df,
            id_vars=["posted_datetime", "datetime", "DSTFlag"],
            value_vars=self.GEOGRAPHICAL_ZONES,
            var_name="location",
            value_name="wind_generation"
        )
        
        # Convert empty strings to NaN and drop rows with NaN
        df_melted.loc[:, "wind_generation"] = pd.to_numeric(df_melted["wind_generation"], errors="coerce")
        df_clean = df_melted.dropna(subset=["wind_generation"])
        
        # Sort by posting time, forecast time, and location
        df_clean = df_clean.sort_values(["posted_datetime", "datetime", "location"])
        
        # Rename columns to standard format
        df_clean = df_clean.rename(columns={
            "DSTFlag": "dst_flag"
        })
        
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate cleaned Wind Generation data.
        
        Validation rules:
        1. No missing values
        2. posted_datetime is datetime type
        3. datetime is datetime type
        4. wind_generation is numeric
        5. location is one of the valid geographical zones
        6. Data is sorted by posted_datetime, datetime, and location
        7. One row per posting time, forecast time, and location
        
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
            
            # Check wind_generation is numeric
            if not pd.api.types.is_numeric_dtype(df["wind_generation"]):
                print("Error: wind_generation column is not numeric")
                return False
            
            # Check location values
            invalid_locations = df[~df["location"].isin(self.GEOGRAPHICAL_ZONES)]["location"].unique()
            if len(invalid_locations) > 0:
                print(f"Error: Found invalid geographical zones: {invalid_locations}")
                return False
            
            # Check sorting
            if not df.equals(df.sort_values(["posted_datetime", "datetime", "location"])):
                print("Error: Data is not sorted by posted_datetime, datetime, and location")
                return False
            
            # Check for duplicates
            duplicates = df.groupby(["posted_datetime", "datetime", "location"]).size()
            if (duplicates > 1).any():
                print("Error: Found duplicate entries for posting time, forecast time, and location")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False 