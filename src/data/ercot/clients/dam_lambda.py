"""DAM System Lambda client module for ERCOT API."""

from datetime import datetime
from typing import Dict
from typing import Optional

import pandas as pd

from src.data.ercot.ercot_data import ERCOTBaseClient


class DAMSystemLambdaClient(ERCOTBaseClient):
    """Client for accessing ERCOT DAM System Lambda data."""

    # Endpoint information
    ENDPOINT_KEY = "dam_lambda"
    ENDPOINT_PATH = "np4-180-cd/dam_stlmnt_pnt_prices"
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
        formatted_date = current_date.strftime("%Y-%m-%d")

        # Create parameters dict
        current_params = params.copy()
        # Use the same date for both from and to since we want that specific day's data
        current_params["deliveryDateFrom"] = formatted_date
        current_params["deliveryDateTo"] = formatted_date

        return current_params

    def get_dam_lambda_data(
        self, delivery_date_from: str, delivery_date_to: str
    ) -> pd.DataFrame:
        """Get ERCOT Day-Ahead Market System Lambda.

        Args:
            delivery_date_from (str): Start date in YYYY-MM-DD format
            delivery_date_to (str): End date in YYYY-MM-DD format

        Returns:
            pandas.DataFrame: DataFrame containing DAM System Lambda data including:
                - System Lambda values
                - Delivery dates and times

        Example:
            >>> client = DAMSystemLambdaClient()
            >>> df = client.get_dam_lambda_data(
            ...     delivery_date_from="2024-01-01",
            ...     delivery_date_to="2024-01-02"
            ... )
        """
        params = {
            "deliveryDateFrom": delivery_date_from,
            "deliveryDateTo": delivery_date_to,
            "settlementPoint": "SYS_LAMBDA",
        }

        return self.get_data(self.ENDPOINT_PATH, self.ENDPOINT_KEY, params)
