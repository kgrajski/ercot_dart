"""Solar Generation ETL client."""

from typing import Optional
import pandas as pd
from etl.ercot.ercot_etl import ERCOTBaseETL


class SolarGenerationETL(ERCOTBaseETL):
    """
    ETL client for ERCOT Solar Generation data.
    
    Note:
    See: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-745-CD
    """
    
    ENDPOINT_KEY = "solar_power_gen"
    
    # Geographical zones and forecast types from visualization client
    GEOGRAPHICAL_ZONES = [
        "COPHSLCenterEast",
        "COPHSLCenterWest",
        "COPHSLFarEast",
        "COPHSLFarWest",
        "COPHSLNorthWest",
        "COPHSLSouthEast",
        "COPHSLSystemWide",
        "genCenterEast",
        "genCenterWest",
        "genFarEast",
        "genFarWest",
        "genNorthWest",
        "genSouthEast",
        "genSystemWide",
        "HSLSystemWide",
        "PVGRPPCenterEast",
        "PVGRPPCenterWest",
        "PVGRPPFarEast",
        "PVGRPPFarWest",
        "PVGRPPNorthWest",
        "PVGRPPSouthEast",
        "PVGRPPSystemWide",
        "STPPFCenterEast",
        "STPPFCenterWest",
        "STPPFFarEast",
        "STPPFFarWest",
        "STPPFNorthWest",
        "STPPFSouthEast",
        "STPPFSystemWide"
    ]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Solar Generation data.
        
        Cleaning steps:
        1. Convert postedDatetime to datetime
        2. Melt geographical zone columns into rows (preserving utc_ts and local_ts)
        3. Convert empty strings to NaN and drop rows with NaN
        4. Sort by posted datetime, utc timestamp, and zone
        5. Rename DSTFlag to dst_flag for consistency
        
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
        
        # Melt geographical zone columns into rows
        df_melted = pd.melt(
            df,
            id_vars=["posted_datetime", "utc_ts", "local_ts", "hour_local", "DSTFlag"],
            value_vars=self.GEOGRAPHICAL_ZONES,
            var_name="location",
            value_name="solar_generation"
        )
        
        # Convert empty strings to NaN and drop rows with NaN
        df_melted.loc[:, "solar_generation"] = pd.to_numeric(df_melted["solar_generation"], errors="coerce")
        df_clean = df_melted.dropna(subset=["solar_generation"])
        
        # Sort by posting time, utc timestamp, and location
        df_clean = df_clean.sort_values(["posted_datetime", "utc_ts", "location"])
        
        # Rename columns to standard format
        df_clean = df_clean.rename(columns={
            "DSTFlag": "dst_flag"
        })
        
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate cleaned Solar Generation data.
        
        Simple validation for development - assumes data types are correct
        since we control the entire pipeline.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame to validate
            
        Returns:
            bool: True if validation passes
        """
        try:
            # Check for missing values in key columns
            if df[["posted_datetime", "utc_ts", "location", "solar_generation"]].isnull().any().any():
                print("Error: Found missing values in key columns")
                return False
            
            # Check location values are valid
            invalid_locations = df[~df["location"].isin(self.GEOGRAPHICAL_ZONES)]["location"].unique()
            if len(invalid_locations) > 0:
                print(f"Error: Found invalid geographical zones: {invalid_locations}")
                return False
            
            # Basic row count check
            if len(df) == 0:
                print("Error: No data after cleaning")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False 