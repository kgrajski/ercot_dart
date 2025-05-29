"""Wind generation client module for ERCOT API."""

from typing import Optional, Dict
from datetime import datetime
import pandas as pd
from ..ercot_client import ERCOTBaseClient


class WindGenerationClient(ERCOTBaseClient):
    """Client for accessing ERCOT wind generation data."""
    
    # Endpoint information
    ENDPOINT_KEY = "wind_power_gen"
    ENDPOINT_PATH = "np4-742-cd/wpp_hrly_actual_fcast_geo"
    DEFAULT_HOUR_ENDING = "6:00"  # Default to 6 AM drop
    
    def get_wind_generation_data(self, posted_datetime_from: str, posted_datetime_to: str,
                               posted_hour_ending: Optional[str] = None,
                               hours_before: int = 1) -> pd.DataFrame:
        """Get ERCOT Wind Power Production - Hourly Averaged Actual and Forecasted Values.
        
        Args:
            posted_datetime_from (str): Start date in YYYY-MM-DD format
            posted_datetime_to (str): End date in YYYY-MM-DD format
            posted_hour_ending (str, optional): Hour ending for the daily drop.
                                              This will be appended to the date parameters
                                              as 'YYYY-MM-DDThh:mm'.
                                              Defaults to "6:00" for 6 AM drop.
            hours_before (int, optional): Number of hours before hour_ending to start the query.
                                        Defaults to 1 hour.
        
        Returns:
            pandas.DataFrame: DataFrame containing wind power generation data including:
                - System-wide generation and forecasts
                - Regional generation and forecasts (Panhandle, Coastal, South, West, North)
                - HSL (High Sustained Limit) values
                - COPHSL (Current Operating Plan HSL)
                - STWPF (Short Term Wind Power Forecast)
                - WGRPP (Wind Generation Resource Point-to-Point)
                
        Example:
            >>> client = WindGenerationClient()
            >>> df = client.get_wind_generation_data(
            ...     posted_datetime_from="2024-01-01",
            ...     posted_datetime_to="2024-01-02",
            ...     posted_hour_ending="6:00",  # Will query 2024-01-01T6:00 and 2024-01-02T6:00
            ...     hours_before=1  # Will start query 1 hour before hour_ending
            ... )
        """
        # Set up parameters for hourly drops
        params = {
            "postedDatetimeFrom": posted_datetime_from,
            "postedDatetimeTo": posted_datetime_to,
            "postedHourEnding": posted_hour_ending or self.DEFAULT_HOUR_ENDING,
            "hours_before": hours_before
        }
        
        return self.get_data(self.ENDPOINT_PATH, self.ENDPOINT_KEY, params) 