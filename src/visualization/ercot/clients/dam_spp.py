"""DAM Settlement Point Prices visualization client."""

from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..base import ERCOTBaseViz


class DAMSettlementPointPricesViz(ERCOTBaseViz):
    """
    Client for visualizing ERCOT DAM Settlement Point Prices data.
    
    Note:
    DAM Settlement Point Prices represent the Day-Ahead Market prices
    at various settlement points including Load Zones (LZ_) and Hubs (HB_).
    
    See: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-190-CD
    """
    
    ENDPOINT_KEY = "dam_spp"
    
    def plot_dam_spp(self):
        """Create daily DAM Settlement Point Prices visualization for each delivery date."""
        # Get data
        df = self.get_data(self.ENDPOINT_KEY)
        
        # Convert deliveryDate to datetime and extract the date component
        df["delivery_date"] = pd.to_datetime(df["deliveryDate"]).dt.date
        
        # Filter for Load Zones and Hubs
        mask = df["settlementPoint"].str.startswith(("LZ_", "HB_"))
        df = df[mask]
        
        # Get unique delivery dates
        delivery_dates = sorted(df["delivery_date"].unique())
        print(f"\nFound {len(delivery_dates)} unique delivery dates")
        
        # Process each delivery date
        for delivery_date in delivery_dates:
            print(f"\nProcessing DAM SPP for delivery date {delivery_date}")
            
            # Filter data for this delivery date
            df_delivery = df[df["delivery_date"] == delivery_date]
            
            # Convert deliveryDate and hourEnding to datetime using base class utility
            df_delivery["datetime"] = df_delivery.apply(
                lambda row: self.combine_date_hour(row["deliveryDate"], row["hourEnding"]),
                axis=1
            )
            
            # Sort by datetime and settlement point for consistent ordering
            df_delivery = df_delivery.sort_values(["datetime", "settlementPoint"])
            
            # Save DataFrame to CSV for debugging
            csv_path = self.output_dir / f"dam_spp_{delivery_date}.csv"
            df_delivery.to_csv(csv_path, index=False)
            print(f"Saved DataFrame to: {csv_path}")
            
            # Create plot
            fig = px.line(df_delivery, 
                         x="datetime", 
                         y="settlementPointPrice",
                         color="settlementPoint",
                         title=f"DAM Settlement Point Prices - Load Zones and Hubs (Delivery Date {delivery_date})")
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Hour Ending",
                yaxis_title="Settlement Point Price ($/MWh)",
                legend_title="Settlement Point"
            )
            
            # Save plot
            self.save_plot(fig, delivery_date, self.ENDPOINT_KEY)
    
    def generate_plots(self):
        """Generate all plots for DAM Settlement Point Prices data."""
        self.plot_dam_spp() 