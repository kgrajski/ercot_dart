"""Solar generation forecast visualization client."""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.visualization.ercot.ercot_viz import ERCOTBaseViz


class SolarGenerationViz(ERCOTBaseViz):
    """
    Client for visualizing ERCOT solar generation forecast data.

    Note:
    Short Term Photovoltaic Power Forecast (STPPF)
    PhotoVoltaic Generation Resource Power Potential (PVGRPP)
    High Sustained Limits (HSLs)
    Current Operating Plans (COPs)
    See: https://www.ercot.com/files/docs/2022/05/11/PV_Forecast_Update_Workshop_v3_1.pptx

    See: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-745-CD

    """

    ENDPOINT_KEY = "solar_power_gen"
    GEOGRAPHICAL_ZONES = sorted(
        [
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
        ]
    )

    def plot_solar_forecast(self):
        """Create daily solar generation forecast by geo zone visualization for each posted date."""
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
                value_name="SolarGeneration",
            )

            # Sort by datetime first, then by GeoZone_Type to maintain the order
            df_melted = df_melted.sort_values(["datetime", "GeoZone_Type"])

            # Create plot
            fig = px.line(
                df_melted,
                x="datetime",
                y="SolarGeneration",
                color="GeoZone_Type",
                title=f"Solar Generation Forecast by Geo Zone and Type (Posted {posted_date})",
            )

            # Customize layout
            fig.update_layout(
                xaxis_title="Local Time",
                yaxis_title="Solar Generation (MW)",
                legend_title="GeoZone Type",
            )

            # Save plot and data
            self.save_plot(fig, str(posted_date), self.ENDPOINT_KEY)
            self.save_data(df_melted, str(posted_date), self.ENDPOINT_KEY)

    def generate_plots(self):
        """Generate all plots for solar generation forecast data."""
        self.plot_solar_forecast()
