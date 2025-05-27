"""Solar generation forecast visualization client."""

from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..base import ERCOTBaseViz


class SolarGenerationViz(ERCOTBaseViz):
    """
    Client for visualizing ERCOT solar generation forecast data.

    Note:
    Short Term Photovoltaic Power Forecast (STPPF)
    PhotoVoltaic Generation Resource Power Potential (PVGRPP)
    High Sustained Limits (HSLs)
    Current Operating Plans (COPs)
    See: https://www.ercot.com/files/docs/2022/05/11/PV_Forecast_Update_Workshop_v3_1.pptx

    """
    
    ENDPOINT_KEY = "solar_power_gen"
    GEOGRAPHICAL_ZONES = sorted([
        "COPHSLCenterEast",
        "COPHSLCenterWest",
        "COPHSLFarEast",
        "COPHSLFarWest",
        "COPHSLNorthWest",
        "COPHSLSouthEast",
        "COPHSLSystemWide",
        "genCenterEast",
        "genCenterWest",
        "genFarEast",
        "genFarWest",
        "genNorthWest",
        "genSouthEast",
        "genSystemWide",
        "HSLSystemWide",
        "PVGRPPCenterEast",
        "PVGRPPCenterWest",
        "PVGRPPFarEast",
        "PVGRPPFarWest",
        "PVGRPPNorthWest",
        "PVGRPPSouthEast",
        "PVGRPPSystemWide",
        "STPPFCenterEast",
        "STPPFCenterWest",
        "STPPFFarEast",
        "STPPFFarWest",
        "STPPFNorthWest",
        "STPPFSouthEast",
        "STPPFSystemWide",
    ])

    
    def plot_solar_forecast(self):
        """Create daily solar generation forecast by geo zone visualization for each posted date."""
        # Get data
        df = self.get_data(self.ENDPOINT_KEY)
        
        # Convert postedDatetime to datetime and extract the date component
        df["posted_date"] = pd.to_datetime(df["postedDatetime"]).dt.date

        # Convert the hourEnding column to a datetime object from Int64 type
        df["hourEnding_datetime"] = pd.to_datetime(df["hourEnding"], unit="h")
        
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
                lambda row: self.combine_date_hour(row["deliveryDate"], row["hourEnding_datetime"]),
                axis=1
            )
            
            # Melt the weather zone columns into a single column
            df_melted = pd.melt(
                df_posted,
                id_vars=["datetime"],
                value_vars=self.GEOGRAPHICAL_ZONES,
                var_name="GeoZone_Type",
                value_name="SolarGeneration"
            )
            
            # Sort by datetime first, then by GeoZone_Type to maintain the order
            df_melted = df_melted.sort_values(["datetime", "GeoZone_Type"])
            
            # Save melted DataFrame to CSV for debugging
            csv_path = self.output_dir / f"df_melted_{posted_date}.csv"
            df_melted.to_csv(csv_path, index=False)
            print(f"Saved melted DataFrame to: {csv_path}")
            
            # Create plot
            fig = px.line(df_melted, 
                         x="datetime", 
                         y="SolarGeneration",
                         color="GeoZone_Type",
                         title=f"Solar Generation Forecast by Geo Zone and Type (Posted {posted_date})")
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Solar Generation (MW)",
                legend_title="GeoZone Type"
            )
            
            # Save plot with posted date in filename
            plot_name = f"daily_forecast_{posted_date}"
            self.save_plot(fig, plot_name, self.ENDPOINT_KEY)
    
    def generate_plots(self):
        """Generate all plots for solar generation forecast data."""
        self.plot_solar_forecast() 