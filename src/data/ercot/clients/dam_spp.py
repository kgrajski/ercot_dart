"""DAM Settlement Point Prices client module for ERCOT API."""

from typing import Optional, Dict
from datetime import datetime
import pandas as pd
from ..ercot_data import ERCOTBaseClient


class DAMSettlementPointPricesClient(ERCOTBaseClient):
    """Client for accessing ERCOT DAM Settlement Point Prices data."""
    
    # Endpoint information
    ENDPOINT_KEY = "dam_spp"
    ENDPOINT_PATH = "np4-190-cd/dam_stlmnt_pnt_prices"
    DEFAULT_HOUR_ENDING = "14:00"  # Default to 2 PM drop
    
    # Full list of settlement points we want to collect
    SETTLEMENT_POINTS = [
        "HB_BUSAVG",   # Bus Average Hub
        "HB_HOUSTON",   # Houston Hub
        "HB_HUBAVG",    # Hub Average Hub
        "HB_NORTH",     # North Hub
        "HB_PAN",       # Panhandle Hub
        "HB_SOUTH",     # South Hub
        "HB_WEST",      # West Hub
        "LZ_AEN",       # AEN Load Zone
        "LZ_CPS",       # CPS Load Zone
        "LZ_HOUSTON",   # Houston Load Zone
        "LZ_LCRA",      # LCRA Load Zone
        "LZ_NORTH",     # North Load Zone
        "LZ_RAYBN",     # Rayburn Load Zone
        "LZ_SOUTH",     # South Load Zone
        "LZ_WEST"       # West Load Zone
    ]

    # Focus list of settlement points we want to collect
    SETTLEMENT_POINTS = [
        "LZ_HOUSTON",   # Houston Load Zone
    ]
    
    def _build_query_params(self, current_date: datetime, params: Dict) -> Dict:
        """Override to handle delivery date based parameters for DAM endpoints.
        
        Args:
            current_date (datetime): The date to build parameters for
            params (dict): Original parameters
            
        Returns:
            dict: Query parameters for the specific date
        """
        # For DAM endpoints, we use the date directly as the delivery date
        formatted_date = current_date.strftime("%Y-%m-%d")
        
        # Create parameters dict
        current_params = params.copy()
        # Use the same date for both from and to since we want that specific day's data
        current_params["deliveryDateFrom"] = formatted_date
        current_params["deliveryDateTo"] = formatted_date
        
        # Add settlement point if specified in params
        if "settlementPoint" in params:
            current_params["settlementPoint"] = params["settlementPoint"]
        
        return current_params
    
    def get_dam_spp_data(self, delivery_date_from: str, delivery_date_to: str) -> pd.DataFrame:
        """Get ERCOT Day-Ahead Market Settlement Point Prices.
        
        Args:
            delivery_date_from (str): Start date in YYYY-MM-DD format
            delivery_date_to (str): End date in YYYY-MM-DD format
        
        Returns:
            pandas.DataFrame: DataFrame containing DAM Settlement Point Prices data including:
                - Settlement Point prices
                - Delivery dates and times
                - Settlement Point names (predefined list of Hubs and Load Zones)
                
        Example:
            >>> client = DAMSettlementPointPricesClient()
            >>> df = client.get_dam_spp_data(
            ...     delivery_date_from="2024-01-01",
            ...     delivery_date_to="2024-01-02"
            ... )
        """
        all_data = []
        
        # Get data for each settlement point in our predefined list
        for point in self.SETTLEMENT_POINTS:
            print(f"\nGetting data for {point}")
            params = {
                "deliveryDateFrom": delivery_date_from,
                "deliveryDateTo": delivery_date_to,
                "settlementPoint": point
            }
            
            # Get data for this specific settlement point
            df = self.get_data(self.ENDPOINT_PATH, self.ENDPOINT_KEY, params, 
                               save_output=False, add_utc=True)
            if not df.empty:
                all_data.append(df)
                print(f"Retrieved {len(df)} records")
            else:
                print("No data returned")
        
        # Combine all data if we got any
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal records collected: {len(final_df)}")
            final_df = self._save_data(final_df, self.ENDPOINT_KEY, {})
            return final_df
        else:
            raise ValueError(
                "No data was retrieved for any settlement point. "
                "This could indicate an API error, invalid date range, "
                "or that no data is available for the requested period."
            ) 