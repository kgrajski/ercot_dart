"""DAM System Lambda visualization client."""

from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..base import ERCOTBaseViz


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
        
        # Convert deliveryDate to datetime and extract the date component
        df["delivery_date"] = pd.to_datetime(df["deliveryDate"]).dt.date
        
        # Get unique delivery dates
        delivery_dates = sorted(df["delivery_date"].unique())
        print(f"\nFound {len(delivery_dates)} unique delivery dates")
        
        # Process each delivery date
        for delivery_date in delivery_dates:
            print(f"\nProcessing DAM Lambda for delivery date {delivery_date}")
            
            # Filter data for this delivery date
            df_delivery = df[df["delivery_date"] == delivery_date]
            
            # Convert deliveryDate and hourEnding to datetime using base class utility
            df_delivery["datetime"] = df_delivery.apply(
                lambda row: self.combine_date_hour(row["deliveryDate"], row["hourEnding"]),
                axis=1
            )
            
            # Sort by datetime
            df_delivery = df_delivery.sort_values("datetime")
            
            # Save DataFrame to CSV for debugging
            csv_path = self.output_dir / f"dam_lambda_{delivery_date}.csv"
            df_delivery.to_csv(csv_path, index=False)
            print(f"Saved DataFrame to: {csv_path}")
            
            # Create plot
            fig = px.line(df_delivery, 
                         x="datetime", 
                         y="systemLambda",
                         title=f"DAM System Lambda (Delivery Date {delivery_date})")
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Hour Ending",
                yaxis_title="System Lambda ($/MWh)",
            )
            
            # Save plot
            self.save_plot(fig, delivery_date, self.ENDPOINT_KEY)
    
    def generate_plots(self):
        """Generate all plots for DAM System Lambda data."""
        self.plot_dam_lambda() 