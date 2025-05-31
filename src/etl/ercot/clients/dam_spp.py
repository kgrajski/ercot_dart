"""DAM Settlement Point Prices ETL module."""

from typing import Optional
import pandas as pd
from etl.ercot.ercot_etl import ERCOTBaseETL


# Maximum number of duplicate examples to show in validation error logs
MAX_DUPLICATE_EXAMPLES = 10


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
            
            # Enhanced duplicate detection
            duplicates = df.groupby(["datetime", "location"]).size()
            duplicate_pairs = duplicates[duplicates > 1]
            
            if not duplicate_pairs.empty:
                total_duplicates = len(duplicate_pairs)
                print(f"\nFound {total_duplicates} datetime-location pairs with duplicate entries")
                print(f"Showing first {min(MAX_DUPLICATE_EXAMPLES, total_duplicates)} examples:")
                print("-" * 50)
                
                for i, ((dt, loc), count) in enumerate(duplicate_pairs.items()):
                    if i >= MAX_DUPLICATE_EXAMPLES:
                        remaining = total_duplicates - MAX_DUPLICATE_EXAMPLES
                        print(f"\n... and {remaining} more duplicate pairs not shown ...")
                        break
                        
                    print(f"\nExample {i + 1}/{min(MAX_DUPLICATE_EXAMPLES, total_duplicates)}:")
                    print(f"DateTime: {dt}")
                    print(f"Location: {loc}")
                    print(f"Number of entries: {count}")
                    
                    # Show the actual duplicate rows
                    dupe_rows = df[
                        (df["datetime"] == dt) & 
                        (df["location"] == loc)
                    ].sort_values("price")
                    
                    print("\nDuplicate rows:")
                    print(dupe_rows.to_string())
                    print("\nOriginal raw data for these entries:")
                    
                    # Get the raw data for these entries
                    raw_df = self.get_raw_data(self.ENDPOINT_KEY)
                    raw_matches = raw_df[
                        (raw_df["deliveryDate"] == dt.strftime("%Y-%m-%d")) &
                        (raw_df["settlementPoint"] == loc)
                    ]
                    print(raw_matches.to_string())
                    print("-" * 50)
                
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False 