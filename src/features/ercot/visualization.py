"""Visualization utilities for ERCOT feature analysis."""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def plot_dart_by_location(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create DART (Day-Ahead to Real-Time) price difference visualization.
    
    Creates a subplot with:
    - Upper plot: Raw DART price differences
    - Lower plot: Signed log transformed DART price differences
    
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
    
    # Apply signed log transformation to DART
    df["dart_slt"] = df["dart"].apply(
        lambda x: np.sign(x) * np.log(1 + abs(x)) if pd.notna(x) else np.nan
    )
    
    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            f"Raw DART Price Differences{title_suffix}",
            f"Signed Log Transformed DART Price Differences{title_suffix}"
        ]
    )
    
    # Get unique point identifiers and colors
    unique_points = df["point_identifier"].unique()
    colors = px.colors.qualitative.Plotly[:len(unique_points)]
    color_map = dict(zip(unique_points, colors))
    
    # Add traces for raw DART (upper plot)
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        fig.add_trace(
            go.Scatter(
                x=point_data["local_ts"],
                y=point_data["dart"],
                mode="lines",
                name=f"{point} (Raw)",
                line=dict(color=color_map[point]),
                legendgroup=point,
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add traces for signed log transformed DART (lower plot)
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        fig.add_trace(
            go.Scatter(
                x=point_data["local_ts"],
                y=point_data["dart_slt"],
                mode="lines",
                name=f"{point} (SLT)",
                line=dict(color=color_map[point], dash="dash"),
                legendgroup=point,
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Add horizontal reference lines at y=0 for both plots
    for row in [1, 2]:
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            row=row, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"DART Price Differences Analysis{title_suffix}",
        height=800,
        legend_title="Settlement Point (Type)"
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="DART Price Difference ($/MWh)", row=1, col=1)
    fig.update_yaxes(title_text="Signed Log DART", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dart_by_location.html"
    fig.write_html(output_path)
    print(f"DART comparison plot saved to: {output_path}")
    
    # Save the enhanced data (now includes dart_slt column)
    data_path = output_dir / "dart_by_location.csv"
    df.to_csv(data_path, index=False)
    print(f"DART data (with SLT) saved to: {data_path}")


def plot_dart_distributions(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create histogram visualization of DART distributions with normal overlay and percentiles.
    
    Creates a subplot with:
    - Upper plot: Raw DART histogram with fitted normal distribution overlay
    - Lower plot: Signed log transformed DART histogram with fitted normal distribution overlay
    
    Args:
        df: DataFrame containing DART data with 'dart' column
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Apply signed log transformation if not already present
    if "dart_slt" not in df.columns:
        df["dart_slt"] = df["dart"].apply(
            lambda x: np.sign(x) * np.log(1 + abs(x)) if pd.notna(x) else np.nan
        )
    
    # Remove NaN values for statistics
    dart_clean = df["dart"].dropna()
    dart_slt_clean = df["dart_slt"].dropna()
    
    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.25,
        subplot_titles=[
            f"Raw DART Distribution{title_suffix}",
            f"Signed Log Transformed DART Distribution{title_suffix}"
        ]
    )
    
    # Plot histograms and overlays for both datasets
    datasets = [
        (dart_clean, "Raw DART", "DART ($/MWh)", 1),
        (dart_slt_clean, "SLT DART", "Signed Log DART", 2)
    ]
    
    for data, name, xlabel, row in datasets:
        # Calculate percentiles
        percentiles = {
            "67th": np.percentile(data, 67),
            "90th": np.percentile(data, 90), 
            "95th": np.percentile(data, 95),
            "99th": np.percentile(data, 99)
        }
        
        # Fit normal distribution
        mu, std = stats.norm.fit(data)
        
        # Create histogram
        fig.add_trace(
            go.Histogram(
                x=data,
                name=f"{name} Histogram",
                nbinsx=50,
                histnorm="probability density",
                opacity=0.7,
                marker_color="lightblue",
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Create fitted normal distribution overlay
        x_range = np.linspace(data.min(), data.max(), 100)
        normal_curve = stats.norm.pdf(x_range, mu, std)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_curve,
                mode="lines",
                name=f"{name} Normal Fit (μ={mu:.2f}, σ={std:.2f})",
                line=dict(color="red", width=2),
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Add percentile lines
        colors = ["orange", "purple", "green", "darkred"]
        for i, (pct_name, pct_value) in enumerate(percentiles.items()):
            fig.add_vline(
                x=pct_value,
                line_dash="dash",
                line_color=colors[i],
                line_width=2,
                annotation_text=f"{pct_name}: {pct_value:.2f}",
                annotation_position="bottom right",
                row=row, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f"DART Distribution Analysis{title_suffix}",
        height=800,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)
    fig.update_yaxes(title_text="Probability Density", row=2, col=1)
    fig.update_xaxes(title_text="DART ($/MWh)", row=1, col=1)
    fig.update_xaxes(title_text="Signed Log DART", row=2, col=1)
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dart_distributions.html"
    fig.write_html(output_path)
    print(f"DART distribution plot saved to: {output_path}")
    
    # Save statistics summary
    stats_summary = pd.DataFrame({
        "Metric": ["Mean", "Std", "67th Percentile", "90th Percentile", "95th Percentile", "99th Percentile"],
        "Raw_DART": [
            dart_clean.mean(), dart_clean.std(),
            np.percentile(dart_clean, 67), np.percentile(dart_clean, 90),
            np.percentile(dart_clean, 95), np.percentile(dart_clean, 99)
        ],
        "SLT_DART": [
            dart_slt_clean.mean(), dart_slt_clean.std(),
            np.percentile(dart_slt_clean, 67), np.percentile(dart_slt_clean, 90),
            np.percentile(dart_slt_clean, 95), np.percentile(dart_slt_clean, 99)
        ]
    })
    
    stats_path = output_dir / "dart_distribution_stats.csv"
    stats_summary.to_csv(stats_path, index=False)
    print(f"DART distribution statistics saved to: {stats_path}")


def plot_dart_boxplots(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create box plot visualization of DART distributions.
    
    Creates a subplot with:
    - Left plot: Raw DART box plot by location
    - Right plot: Signed log transformed DART box plot by location
    
    Args:
        df: DataFrame containing DART data with 'dart' column and 'point_identifier'
        output_dir: Directory where plots will be saved  
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Create point_identifier if not present
    if "point_identifier" not in df.columns:
        df["point_identifier"] = df.apply(
            lambda row: f"{row['location']} ({row['location_type']})",
            axis=1
        )
    
    # Apply signed log transformation if not already present
    if "dart_slt" not in df.columns:
        df["dart_slt"] = df["dart"].apply(
            lambda x: np.sign(x) * np.log(1 + abs(x)) if pd.notna(x) else np.nan
        )
    
    # Create subplot with 1 row, 2 columns (side by side)
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=False,
        horizontal_spacing=0.15,
        subplot_titles=[
            f"Raw DART Box Plots{title_suffix}",
            f"Signed Log Transformed DART Box Plots{title_suffix}"
        ]
    )
    
    # Get unique points for consistent colors
    unique_points = df["point_identifier"].unique()
    colors = px.colors.qualitative.Plotly[:len(unique_points)]
    
    # Add box plots for raw DART (left plot)
    for i, point in enumerate(unique_points):
        point_data = df[df["point_identifier"] == point]
        fig.add_trace(
            go.Box(
                y=point_data["dart"],
                name=point,
                marker_color=colors[i % len(colors)],
                boxpoints="outliers",
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Add box plots for transformed DART (right plot)  
    for i, point in enumerate(unique_points):
        point_data = df[df["point_identifier"] == point]
        fig.add_trace(
            go.Box(
                y=point_data["dart_slt"],
                name=point,
                marker_color=colors[i % len(colors)],
                boxpoints="outliers",
                showlegend=True,
                legendgroup=point
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=f"DART Box Plot Analysis{title_suffix}",
        height=600,
        showlegend=True,
        legend_title="Settlement Point (Type)"
    )
    
    # Update axis labels
    fig.update_yaxes(title_text="DART ($/MWh)", row=1, col=1)
    fig.update_yaxes(title_text="Signed Log DART", row=1, col=2)
    fig.update_xaxes(title_text="Settlement Points", row=1, col=1)
    fig.update_xaxes(title_text="Settlement Points", row=1, col=2)
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dart_boxplots.html"
    fig.write_html(output_path)
    print(f"DART box plot saved to: {output_path}")
    
    # Save box plot statistics
    box_stats = []
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        raw_stats = point_data["dart"].describe()
        slt_stats = point_data["dart_slt"].describe()
        
        box_stats.append({
            "point_identifier": point,
            "raw_q1": raw_stats["25%"],
            "raw_median": raw_stats["50%"], 
            "raw_q3": raw_stats["75%"],
            "raw_iqr": raw_stats["75%"] - raw_stats["25%"],
            "slt_q1": slt_stats["25%"],
            "slt_median": slt_stats["50%"],
            "slt_q3": slt_stats["75%"],
            "slt_iqr": slt_stats["75%"] - slt_stats["25%"]
        })
    
    box_stats_df = pd.DataFrame(box_stats)
    stats_path = output_dir / "dart_boxplot_stats.csv"
    box_stats_df.to_csv(stats_path, index=False)
    print(f"DART box plot statistics saved to: {stats_path}")


def plot_dart_qqplots(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create Q-Q plots for normality assessment of DART distributions.
    
    Creates a subplot with:
    - Upper plot: Raw DART Q-Q plot against normal distribution
    - Lower plot: Signed log transformed DART Q-Q plot against normal distribution
    
    Q-Q plots help assess how closely the data follows a normal distribution.
    Points that fall along the diagonal line indicate normal distribution.
    
    Args:
        df: DataFrame containing DART data with 'dart' column
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Apply signed log transformation if not already present
    if "dart_slt" not in df.columns:
        df["dart_slt"] = df["dart"].apply(
            lambda x: np.sign(x) * np.log(1 + abs(x)) if pd.notna(x) else np.nan
        )
    
    # Remove NaN values for statistics
    dart_clean = df["dart"].dropna()
    dart_slt_clean = df["dart_slt"].dropna()
    
    # Create subplot with 1 row, 2 columns (side by side)
    fig = make_subplots(
        rows=1, cols=2,
        shared_xaxes=False,
        horizontal_spacing=0.15,
        subplot_titles=[
            f"Raw DART Q-Q Plot{title_suffix}",
            f"Signed Log Transformed DART Q-Q Plot{title_suffix}"
        ]
    )
    
    # Plot Q-Q plots for both datasets
    datasets = [
        (dart_clean, "Raw DART", 1),
        (dart_slt_clean, "SLT DART", 2)
    ]
    
    for data, name, col in datasets:
        # Generate Q-Q plot data using scipy
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=None)
        
        # Add scatter plot of actual Q-Q points
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode="markers",
                name=f"{name} Data Points",
                marker=dict(color="blue", size=4, opacity=0.6),
                showlegend=True
            ),
            row=1, col=col
        )
        
        # Add theoretical normal line (perfect fit)
        line_x = np.array([osm.min(), osm.max()])
        line_y = slope * line_x + intercept
        
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name=f"{name} Normal Line (R²={r**2:.3f})",
                line=dict(color="red", width=2),
                showlegend=True
            ),
            row=1, col=col
        )
        
        # Add reference diagonal line (y=x) for perfect normality
        data_range = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        fig.add_trace(
            go.Scatter(
                x=data_range,
                y=data_range,
                mode="lines",
                name=f"{name} Perfect Normal (y=x)",
                line=dict(color="gray", width=1, dash="dash"),
                showlegend=True,
                opacity=0.5
            ),
            row=1, col=col
        )
    
    # Update layout
    fig.update_layout(
        title=f"DART Q-Q Plot Analysis{title_suffix}",
        height=800,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Normal Quantiles", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Normal Quantiles", row=1, col=2)
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dart_qqplots.html"
    fig.write_html(output_path)
    print(f"DART Q-Q plot saved to: {output_path}")
    
    # Save Q-Q statistics summary
    qq_stats = []
    for data, name in [(dart_clean, "Raw_DART"), (dart_slt_clean, "SLT_DART")]:
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=None)
        
        qq_stats.append({
            "Dataset": name,
            "R_squared": r**2,
            "Slope": slope,
            "Intercept": intercept,
            "Correlation": r
        })
    
    qq_stats_df = pd.DataFrame(qq_stats)
    stats_path = output_dir / "dart_qqplot_stats.csv"
    qq_stats_df.to_csv(stats_path, index=False)
    print(f"DART Q-Q plot statistics saved to: {stats_path}") 