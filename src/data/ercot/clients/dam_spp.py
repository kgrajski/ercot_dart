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
    
    def _build_query_params(self, current_date: datetime, params: Dict) -> Dict:
        """Override to handle delivery date based parameters for DAM endpoints.
        
        Args:
            current_date (datetime): The date to build parameters for
            params (dict): Original parameters
            
        Returns:
            dict: Query parameters for the specific date
        """
        # For DAM endpoints, we use the date directly as the delivery date
        formatted_date = current_date.strftime('%Y-%m-%d')
        
        # Create parameters dict
        current_params = params.copy()
        # Use the same date for both from and to since we want that specific day's data
        current_params['deliveryDateFrom'] = formatted_date
        current_params['deliveryDateTo'] = formatted_date
        
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
                - Settlement Point names
                
        Example:
            >>> client = DAMSettlementPointPricesClient()
            >>> df = client.get_dam_spp_data(
            ...     delivery_date_from="2024-01-01",
            ...     delivery_date_to="2024-01-02"
            ... )
        """
        # Set up parameters - use delivery dates for both iteration and API parameters
        params = {
            "deliveryDateFrom": delivery_date_from,    # Used for both iteration and API
            "deliveryDateTo": delivery_date_to         # Used for both iteration and API
        }
        
        return self.get_data(self.ENDPOINT_PATH, self.ENDPOINT_KEY, params) 