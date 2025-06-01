"""Real-Time Settlement Point Prices visualization module."""

from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..ercot_viz import ERCOTBaseViz


class RTSettlementPointPricesViz(ERCOTBaseViz):
    """
    Client for visualizing ERCOT Real-Time Settlement Point Prices data.
    
    Note:
    RT Settlement Point Prices represent the Real-Time Market prices
    at various settlement points including Load Zones (LZ_) and Hubs (HB_).
    Data is provided in 15-minute intervals.
    
    Important Note on Settlement Points:
    A settlement point is uniquely identified by the combination of:
    - Settlement Point Name (e.g., "LZ_HOUSTON" or "HB_NORTH")
    - Settlement Point Type (e.g., "LZ" or "HU")
    
    Both components are needed to uniquely identify a time series.
    
    See: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-905-CD
    """
    
    ENDPOINT_KEY = "rt_spp"
    
    def plot_rt_spp(self):
        """Create daily RT Settlement Point Prices visualization for each delivery date."""
        # Get data (now includes pre-processed utc_ts and local_ts columns)
        df = self.get_data(self.ENDPOINT_KEY)
        
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Convert deliveryDate to datetime and extract the date component
        df.loc[:, "delivery_date"] = pd.to_datetime(df["deliveryDate"]).dt.date
        
        # Filter for Load Zones and Hubs by name prefix only
        mask = df["settlementPoint"].str.startswith(("LZ_", "HB_"))
        df = df.loc[mask].copy()
        
        # Create unique identifier combining settlement point and type
        df.loc[:, "point_identifier"] = df.apply(
            lambda row: f"{row['settlementPoint']} ({row['settlementPointType']})",
            axis=1
        )
        
        # Get unique delivery dates
        delivery_dates = sorted(df["delivery_date"].unique())
        print(f"\nFound {len(delivery_dates)} unique delivery dates")
        
        # Process each delivery date
        for delivery_date in delivery_dates:
            print(f"\nProcessing RT SPP for delivery date {delivery_date}")
            
            # Filter data for this delivery date
            df_delivery = df.loc[df["delivery_date"] == delivery_date].copy()
            
            # Use local timestamp directly (includes 15-minute interval precision)
            df_delivery.loc[:, "datetime"] = df_delivery["local_ts"]
            
            # Sort by datetime, settlement point, and type for consistent ordering
            df_delivery = df_delivery.sort_values([
                "datetime",
                "settlementPoint",
                "settlementPointType"
            ])
            
            # Create plot
            fig = px.line(df_delivery, 
                         x="datetime", 
                         y="settlementPointPrice",
                         color="point_identifier",
                         title=f"RT Settlement Point Prices - Load Zones and Hubs (Delivery Date {delivery_date})")
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Local Time",
                yaxis_title="Settlement Point Price ($/MWh)",
                legend_title="Settlement Point (Type)"
            )
            
            # Save plot and data
            self.save_plot(fig, str(delivery_date), self.ENDPOINT_KEY)
            self.save_data(df_delivery, str(delivery_date), self.ENDPOINT_KEY)
    
    def generate_plots(self):
        """Generate all plots for RT Settlement Point Prices data."""
        self.plot_rt_spp() 