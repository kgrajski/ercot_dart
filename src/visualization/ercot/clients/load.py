"""Load forecast visualization client.

See: https://www.ercot.com/mp/data-products/data-product-details?id=np3-565-cd

"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.visualization.ercot.ercot_viz import ERCOTBaseViz


class LoadForecastViz(ERCOTBaseViz):
    """Client for visualizing ERCOT load forecast data."""

    ENDPOINT_KEY = "load_forecast"
    WEATHER_ZONES = sorted(
        [
            "coast",
            "east",
            "farWest",
            "north",
            "northCentral",
            "southCentral",
            "southern",
            "west",
        ]
    )

    def plot_load_forecast(self):
        """Create daily load forecast by weather zone visualization for each posted date."""
        # Get data (now includes pre-processed utc_ts and local_ts columns)
        df = self.get_data(self.ENDPOINT_KEY)

        # Make a copy to avoid chained assignment warnings
        df = df.copy()

        # Filter to inUse forecasts only
        df = df[df["inUseFlag"] == True].copy()
        print(f"Found {len(df)} rows for this set of posted dates")

        # Convert postedDatetime to datetime and extract the date component
        df.loc[:, "posted_date"] = pd.to_datetime(df["postedDatetime"]).dt.date

        # Get unique posted dates
        posted_dates = sorted(df["posted_date"].unique())
        print(f"\nFound {len(posted_dates)} unique posted dates")

        # Process each posted date
        for posted_date in posted_dates:
            print(f"\nProcessing forecast posted on {posted_date}")

            # Filter data for this posted date
            df_posted = df[df["posted_date"] == posted_date].copy()

            # Use UTC timestamp directly (no manual datetime combining needed)
            df_posted["datetime"] = df_posted["local_ts"]

            # Melt the weather zone columns into a single column
            df_melted = pd.melt(
                df_posted,
                id_vars=["datetime"],
                value_vars=self.WEATHER_ZONES,
                var_name="WeatherZone",
                value_name="LoadForecast",
            )

            # Sort by datetime first, then by WeatherZone to maintain the order
            df_melted = df_melted.sort_values(["datetime", "WeatherZone"])

            # Create plot
            fig = px.line(
                df_melted,
                x="datetime",
                y="LoadForecast",
                color="WeatherZone",
                title=f"Load Forecast by Weather Zone (Posted {posted_date})",
            )

            # Customize layout
            fig.update_layout(
                xaxis_title="Local Time",
                yaxis_title="Load Forecast (MW)",
                legend_title="Weather Zone",
            )

            # Save plot and data
            self.save_plot(fig, str(posted_date), self.ENDPOINT_KEY)
            self.save_data(df_melted, str(posted_date), self.ENDPOINT_KEY)

    def generate_plots(self):
        """Generate all plots for load forecast data."""
        self.plot_load_forecast()
