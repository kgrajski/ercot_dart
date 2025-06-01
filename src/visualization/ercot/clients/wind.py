"""Wind generation forecast visualization client."""

from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..ercot_viz import ERCOTBaseViz


class WindGenerationViz(ERCOTBaseViz):
    """
    Client for visualizing ERCOT wind generation forecast data.

    Note:
    See: https://www.ercot.com/mp/data-products/data-product-details?id=np4-742-cd

    STWPF = "Short Term Wind Power Forecast"
    WGRPP = "Wind Generation Resource Power Potential"
    HSL = "High Sustained Limit"
    COPHSL = "Current Operating Plan High Sustained Limit"

    """
    
    ENDPOINT_KEY = "wind_power_gen"
    GEOGRAPHICAL_ZONES = sorted([
        "COPHSLCoastal",
        "COPHSLNorth",
        "COPHSLPanhandle",
        "COPHSLSouth",
        "COPHSLSystemWide",
        "COPHSLWest",
        "genCoastal",
        "genNorth",
        "genPanhandle",
        "genSouth",
        "genSystemWide",
        "genWest",
        "HSLSystemWide",
        "STWPFCoastal",
        "STWPFNorth",
        "STWPFPanhandle",
        "STWPFSouth",
        "STWPFSystemWide",
        "STWPFWest",
        "WGRPPCoastal",
        "WGRPPNorth",
        "WGRPPPanhandle",
        "WGRPPSouth",
        "WGRPPSystemWide",
        "WGRPPWest",
    ])

    
    def plot_wind_forecast(self):
        """Create daily wind generation forecast by geo zone visualization for each posted date."""
        # Get data (now includes pre-processed utc_ts and local_ts columns)
        df = self.get_data(self.ENDPOINT_KEY)
        
        # Make a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Convert postedDatetime to datetime and extract the date component
        df.loc[:, "posted_date"] = pd.to_datetime(df["postedDatetime"]).dt.date

        # Get unique posted dates
        posted_dates = sorted(df["posted_date"].unique())
        print(f"\nFound {len(posted_dates)} unique posted dates")
        
        # Process each posted date
        for posted_date in posted_dates:
            print(f"\nProcessing forecast posted on {posted_date}")
            
            # Filter data for this posted date
            df_posted = df.loc[df["posted_date"] == posted_date].copy()
            
            # Use local timestamp directly (no manual datetime combining needed)
            df_posted.loc[:, "datetime"] = df_posted["local_ts"]
            
            # Melt the weather zone columns into a single column
            df_melted = pd.melt(
                df_posted,
                id_vars=["datetime"],
                value_vars=self.GEOGRAPHICAL_ZONES,
                var_name="GeoZone_Type",
                value_name="WindGeneration"
            )
            
            # Sort by datetime first, then by GeoZone_Type to maintain the order
            df_melted = df_melted.sort_values(["datetime", "GeoZone_Type"])
            
            # Create plot
            fig = px.line(df_melted, 
                         x="datetime", 
                         y="WindGeneration",
                         color="GeoZone_Type",
                         title=f"Wind Generation Forecast by Geo Zone and Type (Posted {posted_date})")
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Local Time",
                yaxis_title="Wind Generation (MW)",
                legend_title="GeoZone Type"
            )
            
            # Save plot and data
            self.save_plot(fig, str(posted_date), self.ENDPOINT_KEY)
            self.save_data(df_melted, str(posted_date), self.ENDPOINT_KEY)
    
    def generate_plots(self):
        """Generate all plots for wind generation forecast data."""
        self.plot_wind_forecast() 