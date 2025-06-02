"""Visualization utilities for ERCOT feature analysis."""

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_dart_by_location(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create DART (Day-Ahead to Real-Time) price difference visualization.
    
    Args:
        df: DataFrame containing DART data with columns:
           - local_ts: Timestamp for the observation in local time zone
           - location: Settlement point location (e.g., "LZ_HOUSTON")
           - location_type: Type of settlement point
           - price_mean: RT price mean
           - price_std: RT price standard deviation
           - dam_spp_price: DAM settlement point price
           - dart: RT-DAM price difference
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Create unique identifier combining location and type
    df = df.copy()
    df["point_identifier"] = df.apply(
        lambda row: f"{row['location']} ({row['location_type']})",
        axis=1
    )
    
    # Create the main DART plot
    fig = px.line(
        df,
        x="local_ts",
        y="dart",
        color="point_identifier",
        title=f"DART Price Differences by Location{title_suffix}"
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="DART Price Difference ($/MWh)",
        legend_title="Settlement Point (Type)",
        # Add a horizontal line at y=0 to show the zero difference point
        shapes=[
            dict(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=0,
                y1=0,
                line=dict(
                    color="gray",
                    width=1,
                    dash="dash"
                )
            )
        ]
    )
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dart_by_location.html"
    fig.write_html(output_path)
    print(f"DART plot saved to: {output_path}")
    
    # Also save the data
    data_path = output_dir / "dart_by_location.csv"
    df.to_csv(data_path, index=False)
    print(f"DART data saved to: {data_path}") 