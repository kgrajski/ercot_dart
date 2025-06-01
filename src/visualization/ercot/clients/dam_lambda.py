"""DAM System Lambda visualization module."""

from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..ercot_viz import ERCOTBaseViz


class DAMSystemLambdaViz(ERCOTBaseViz):
    """
    Client for visualizing ERCOT DAM System Lambda data.
    
    Note:
    DAM System Lambda represents the system-wide marginal price of energy 
    for each hour in the Day-Ahead Market.
    
    See: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-523-CD
    """
    
    ENDPOINT_KEY = "dam_system_lambda"
    
    def plot_dam_lambda(self):
        """Create daily DAM System Lambda visualization for each delivery date."""
        # Get data
        df = self.get_data(self.ENDPOINT_KEY)
        
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Convert deliveryDate to datetime and extract the date component
        df.loc[:, "delivery_date"] = pd.to_datetime(df["deliveryDate"]).dt.date
        
        # Get unique delivery dates
        delivery_dates = sorted(df["delivery_date"].unique())
        print(f"\nFound {len(delivery_dates)} unique delivery dates")
        
        # Process each delivery date
        for delivery_date in delivery_dates:
            print(f"\nProcessing DAM Lambda for delivery date {delivery_date}")
            
            # Filter data for this delivery date
            df_delivery = df.loc[df["delivery_date"] == delivery_date].copy()
            
            # Use local timestamp directly (no manual datetime combining needed)
            df_delivery.loc[:, "datetime"] = df_delivery["local_ts"]
            
            # Sort by datetime
            df_delivery = df_delivery.sort_values("datetime")
            
            # Create plot
            fig = px.line(df_delivery, 
                         x="datetime", 
                         y="systemLambda",
                         title=f"DAM System Lambda (Delivery Date {delivery_date})")
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Local Time",
                yaxis_title="System Lambda ($/MWh)",
            )
            
            # Save plot and data
            self.save_plot(fig, str(delivery_date), self.ENDPOINT_KEY)
            self.save_data(df_delivery, str(delivery_date), self.ENDPOINT_KEY)
    
    def generate_plots(self):
        """Generate all plots for DAM System Lambda data."""
        self.plot_dam_lambda() 