"""Load forecast client module for ERCOT API."""

from typing import Optional, Dict
from datetime import datetime
import pandas as pd
from ..base import ERCOTBaseClient


class LoadForecastClient(ERCOTBaseClient):
    """Client for accessing ERCOT load forecast data."""
    
    # Endpoint information
    ENDPOINT_KEY = "load_forecast"
    ENDPOINT_PATH = "np3-565-cd/lf_by_model_weather_zone"
    DEFAULT_HOUR_ENDING = "6:00"  # Default to 6 AM drop
    
    def get_load_forecast_data(self, posted_datetime_from: str, posted_datetime_to: str,
                             posted_hour_ending: Optional[str] = None,
                             hours_before: int = 1) -> pd.DataFrame:
        """Get ERCOT Seven-Day Load Forecast by Model and Weather Zone.
        Specifically, for the given date range, we want the 6AM drop.
        
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
            pandas.DataFrame: DataFrame containing the load forecast data
                
        Example:
            >>> client = LoadForecastClient()
            >>> df = client.get_load_forecast_data(
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