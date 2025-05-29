"""Real-Time Settlement Point Prices client module for ERCOT API."""

import pandas as pd
from ..ercot_client import ERCOTBaseClient


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
            # Get data but don't save it yet
            response = self.api.make_request(self.ENDPOINT_PATH, params)
            if response.status_code == 200:
                # Process the response data
                df = self.processor.process_response(response)
                if not df.empty:
                    all_data.append(df)
                    print(f"Retrieved {len(df)} records")
                else:
                    print("No data returned")
            else:
                print(f"Failed to get data: {response.status_code}")
        
        # Combine all data if we got any
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal records collected: {len(final_df)}")
            
            # Now save the combined data
            if self.processor.output_dir:
                filepath = self.processor.save_to_csv(final_df, self.ENDPOINT_KEY, {})
                print(f"\nData saved to: {filepath}")
            
            # Save to database if configured
            if self.db_processor:
                self.db_processor.save_to_database(final_df, self.ENDPOINT_KEY)
            
            return final_df
        else:
            # Return empty DataFrame with expected columns if no data
            return pd.DataFrame(columns=[
                "deliveryDate", "hourEnding", "settlementPoint", 
                "settlementPointPrice", "deliveryInterval"
            ]) 