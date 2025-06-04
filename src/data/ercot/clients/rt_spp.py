"""Real-Time Settlement Point Prices client module for ERCOT API."""

import pandas as pd
from ..ercot_data import ERCOTBaseClient


class RTSettlementPointPricesClient(ERCOTBaseClient):
    """Client for accessing ERCOT Real-Time Settlement Point Prices data."""
    
    # Endpoint information
    ENDPOINT_KEY = "rt_spp"
    ENDPOINT_PATH = "np6-905-cd/spp_node_zone_hub"
    
    # List of settlement points we want to collect
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
    
    def _build_query_params(self, current_date, params):
        """Override to handle delivery date based parameters for RT endpoints.
        
        Args:
            current_date: The date to build parameters for
            params: Original parameters
            
        Returns:
            dict: Query parameters for the specific date
        """
        # For RT endpoints, we use the date directly as the delivery date
        formatted_date = current_date.strftime('%Y-%m-%d')
        
        # Create parameters dict
        current_params = params.copy()
        # Use the same date for both from and to since we want that specific day's data
        current_params['deliveryDateFrom'] = formatted_date
        current_params['deliveryDateTo'] = formatted_date
        
        # Add settlement point if specified in params
        if 'settlementPoint' in params:
            current_params['settlementPoint'] = params['settlementPoint']
        
        return current_params
    
    def _prepare_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare timestamp columns for RT SPP data.
        
        Args:
            df: DataFrame with raw RT SPP data
            
        Returns:
            DataFrame with standardized timestamp columns
        """
        if df.empty:
            return df
            
        # Copy deliveryHour to hourEnding - RT SPP uses integers 1-24
        # This spoofs (overrides?) the hourEnding column in the API response
        df['hourEnding'] = df['deliveryHour']
        
        return df
    
    def _adjust_timestamps_for_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust timestamps to include 15-minute intervals.
        
        Args:
            df: DataFrame with hourly timestamps and deliveryInterval column
            
        Returns:
            DataFrame with timestamps adjusted for 15-minute intervals
        """
        if df.empty or 'deliveryInterval' not in df.columns:
            return df
            
        # Calculate minutes to add based on deliveryInterval (1-4)
        # Interval 1 = 0 minutes, Interval 2 = 15 minutes, etc.
        interval_minutes = (df['deliveryInterval'] - 1) * 15
        
        # Add interval minutes to both local_ts and utc_ts
        df['local_ts'] = df['local_ts'] + pd.to_timedelta(interval_minutes, unit='minutes')
        df['utc_ts'] = df['utc_ts'] + pd.to_timedelta(interval_minutes, unit='minutes')
        
        return df

    def get_rt_spp_data(self, delivery_date_from, delivery_date_to):
        """Get ERCOT Real-Time Settlement Point Prices.
        
        Args:
            delivery_date_from: Start date in YYYY-MM-DD format
            delivery_date_to: End date in YYYY-MM-DD format
        
        Returns:
            pandas.DataFrame: DataFrame containing RT Settlement Point Prices data including:
                - Settlement Point prices
                - Delivery dates, hours, and intervals (1-4 for 15-minute periods)
                - Settlement Point names (predefined list of Hubs and Load Zones)
                - Settlement Point types
                - UTC and local timestamps adjusted for 15-minute intervals
                
        Example:
            >>> client = RTSettlementPointPricesClient()
            >>> df = client.get_rt_spp_data(
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
            
            # Get raw data without UTC timestamps
            df = self.get_data(self.ENDPOINT_PATH, self.ENDPOINT_KEY, params, 
                               save_output=False, add_utc=False)
            if not df.empty:
                # Prepare timestamp columns
                df = self._prepare_timestamps(df)
                # Add UTC timestamps (hourly level)
                df = self.processor.add_utc_timestamps(df)
                # Adjust timestamps for 15-minute intervals
                df = self._adjust_timestamps_for_intervals(df)
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
