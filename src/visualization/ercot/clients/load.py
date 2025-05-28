"""Load forecast visualization client.

See: https://www.ercot.com/mp/data-products/data-product-details?id=np3-565-cd

"""

from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..base import ERCOTBaseViz


class LoadForecastViz(ERCOTBaseViz):
    """Client for visualizing ERCOT load forecast data."""
    
    ENDPOINT_KEY = "load_forecast"
    WEATHER_ZONES = sorted([
        "coast",
        "east",
        "farWest",
        "north",
        "northCentral",
        "southCentral",
        "southern",
        "west"
    ])
    
    def plot_load_forecast(self):
        """Create daily load forecast by weather zone visualization for each posted date."""
        # Get data
        df = self.get_data(self.ENDPOINT_KEY)

        # We want to subset the data to use only the rows where the inUse column is TRUE
        df = df[df["inUseFlag"] == True]
        print(f"Found {len(df)} rows for this set of posted dates")

        # Convert postedDatetime to datetime and extract the date component
        df["posted_date"] = pd.to_datetime(df["postedDatetime"]).dt.date
        
        # Get unique posted dates
        posted_dates = sorted(df["posted_date"].unique())
        print(f"\nFound {len(posted_dates)} unique posted dates")
        
        # Process each posted date
        for posted_date in posted_dates:
            print(f"\nProcessing forecast posted on {posted_date}")
            
            # Filter data for this posted date
            df_posted = df[df["posted_date"] == posted_date]
            
            # Convert deliveryDate and hourEnding to datetime using base class utility
            df_posted["datetime"] = df_posted.apply(
                lambda row: self.combine_date_hour(row["deliveryDate"], row["hourEnding"]),
                axis=1
            )
            
            # Melt the weather zone columns into a single column
            df_melted = pd.melt(
                df_posted,
                id_vars=["datetime"],
                value_vars=self.WEATHER_ZONES,
                var_name="WeatherZone",
                value_name="LoadForecast"
            )
            
            # Sort by datetime first, then by WeatherZone to maintain the order
            df_melted = df_melted.sort_values(["datetime", "WeatherZone"])
            
            # Create plot
            fig = px.line(df_melted, 
                         x="datetime", 
                         y="LoadForecast",
                         color="WeatherZone",
                         title=f"Load Forecast by Weather Zone (Posted {posted_date})")
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Load Forecast (MW)",
                legend_title="Weather Zone"
            )
            
            # Save plot
            self.save_plot(fig, posted_date, self.ENDPOINT_KEY)
    
    def generate_plots(self):
        """Generate all plots for load forecast data."""
        self.plot_load_forecast()