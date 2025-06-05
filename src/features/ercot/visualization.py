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
    
    # Verify dart_slt column exists (should be provided by dataset)
    if "dart_slt" not in df.columns:
        raise ValueError("dart_slt column not found. Ensure dataset.py creates this column.")
    
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
                line=dict(color=color_map[point]),
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
    
    # Verify dart_slt column exists (should be provided by dataset)
    if "dart_slt" not in df.columns:
        raise ValueError("dart_slt column not found. Ensure dataset.py creates this column.")
    
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
        positions = ["bottom right", "top right", "bottom left", "top left"]  # Stagger positions
        angles = [45, -45, 45, -45]  # Alternate angling for better spacing
        for i, (pct_name, pct_value) in enumerate(percentiles.items()):
            fig.add_vline(
                x=pct_value,
                line_dash="dash",
                line_color=colors[i],
                line_width=2,
                annotation_text=pct_name,  # Just the percentile name, not the value
                annotation_position=positions[i],
                annotation_font_size=10,  # Smaller font size
                annotation_textangle=angles[i],  # Angle the text
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
    
    # Verify dart_slt column exists (should be provided by dataset)
    if "dart_slt" not in df.columns:
        raise ValueError("dart_slt column not found. Ensure dataset.py creates this column.")
    
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
    
    # Verify dart_slt column exists (should be provided by dataset)
    if "dart_slt" not in df.columns:
        raise ValueError("dart_slt column not found. Ensure dataset.py creates this column.")
    
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


def plot_dart_slt_bimodal(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create bimodal analysis of signed log transformed DART distributions.
    
    Creates a subplot with:
    - Left plot: Negative dart_slt values with histogram, normal fit, and percentiles
    - Right plot: Positive dart_slt values with histogram, normal fit, and percentiles
    
    This helps analyze whether each mode of the bimodal distribution is independently normal.
    
    Args:
        df: DataFrame containing dart_slt column
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Verify dart_slt column exists (should be provided by dataset)
    if "dart_slt" not in df.columns:
        raise ValueError("dart_slt column not found. Ensure dataset.py creates this column.")
    
    # Separate positive and negative values (excluding zeros for cleaner analysis)
    dart_slt_clean = df["dart_slt"].dropna()
    negative_values = dart_slt_clean[dart_slt_clean < 0]
    positive_values = dart_slt_clean[dart_slt_clean > 0]
    
    # Take absolute values of negative values for proper percentile interpretation
    negative_values_abs = negative_values.abs()
    
    print(f"Bimodal analysis: {len(negative_values)} negative values, {len(positive_values)} positive values")
    
    # Create subplot with 1 row, 2 columns (side by side)
    fig = make_subplots(
        rows=1, cols=2,
        shared_xaxes=False,
        horizontal_spacing=0.20,  # Increased spacing between subplots
        subplot_titles=[
            f"Negative DART_SLT Distribution (Absolute Values){title_suffix}",
            f"Positive DART_SLT Distribution{title_suffix}"
        ]
    )
    
    # Plot histograms and overlays for both datasets
    datasets = [
        (negative_values_abs, "Negative DART_SLT (Abs)", 1),
        (positive_values, "Positive DART_SLT", 2)
    ]
    
    bimodal_stats = []
    
    for data, name, col in datasets:
        if len(data) < 10:  # Skip if insufficient data
            print(f"Warning: Insufficient data for {name} ({len(data)} points)")
            continue
            
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
                nbinsx=30,  # Fewer bins since we have fewer data points
                histnorm="probability density",
                opacity=0.7,
                marker_color="lightblue" if "Negative" in name else "lightcoral",
                showlegend=True
            ),
            row=1, col=col
        )
        
        # Create fitted normal distribution overlay
        x_range = np.linspace(data.min(), data.max(), 100)
        normal_curve = stats.norm.pdf(x_range, mu, std)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_curve,
                mode="lines",
                name=f"{name} Normal Fit (μ={mu:.3f}, σ={std:.3f})",
                line=dict(color="red", width=2),
                showlegend=True
            ),
            row=1, col=col
        )
        
        # Add percentile lines
        colors = ["orange", "purple", "green", "darkred"]
        positions = ["bottom right", "top right", "bottom left", "top left"]  # Stagger positions
        angles = [45, -45, 45, -45]  # Alternate angling for better spacing
        for i, (pct_name, pct_value) in enumerate(percentiles.items()):
            fig.add_vline(
                x=pct_value,
                line_dash="dash",
                line_color=colors[i],
                line_width=2,
                annotation_text=pct_name,  # Just the percentile name, not the value
                annotation_position=positions[i],
                annotation_font_size=10,  # Smaller font size
                annotation_textangle=angles[i],  # Angle the text
                row=1, col=col
            )
        
        # Store statistics for CSV output
        bimodal_stats.append({
            "Distribution": name,
            "Count": len(data),
            "Mean": mu,
            "Std": std,
            "67th_Percentile": percentiles["67th"],
            "90th_Percentile": percentiles["90th"],
            "95th_Percentile": percentiles["95th"],
            "99th_Percentile": percentiles["99th"],
            "Min": data.min(),
            "Max": data.max()
        })
    
    # Update layout
    fig.update_layout(
        title={
            "text": f"DART Signed-Log Transform Bimodal Analysis{title_suffix}<br><sub>Note: Negative values shown as absolute values for clearer percentile interpretation</sub>",
            "x": 0.5,
            "xanchor": "center"
        },
        height=800,  # Increased height to accommodate annotations
        showlegend=False,
        yaxis_title="Frequency",
        yaxis2_title="Frequency",
        xaxis_title="Signed-Log Transformed DART (Absolute Value)",
        xaxis2_title="Signed-Log Transformed DART"
    )
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dart_slt_bimodal.html"
    fig.write_html(output_path)
    print(f"DART_SLT bimodal plot saved to: {output_path}")
    
    # Save bimodal statistics
    if bimodal_stats:
        bimodal_stats_df = pd.DataFrame(bimodal_stats)
        stats_filename = f"dart_slt_bimodal_stats_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        bimodal_stats_df.to_csv(stats_filename, index=False)
        print(f"Statistics saved to {stats_filename}")


def plot_dart_slt_cumulative(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create cumulative analysis of signed log transformed DART distributions.
    
    Creates a subplot with:
    - Left plot: Cumulative count of positive and negative dart_slt values
    - Right plot: Cumulative distribution (CDF) of positive and negative dart_slt values
    
    This helps understand the accumulation patterns and relative frequencies of positive vs negative price differences.
    
    Args:
        df: DataFrame containing dart_slt column
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Verify dart_slt column exists (should be provided by dataset)
    if "dart_slt" not in df.columns:
        raise ValueError("dart_slt column not found. Ensure dataset.py creates this column.")
    
    # Separate positive and negative values (excluding zeros for cleaner analysis)
    dart_slt_clean = df["dart_slt"].dropna()
    negative_values = dart_slt_clean[dart_slt_clean < 0]
    positive_values = dart_slt_clean[dart_slt_clean > 0]
    
    # Take absolute values of negative values for proper percentile interpretation
    negative_values_abs = negative_values.abs()
    
    print(f"Cumulative analysis: {len(negative_values)} negative values, {len(positive_values)} positive values")
    
    # Create subplot with 2 rows, 2 columns (2x2 grid)
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=False,
        horizontal_spacing=0.15,
        vertical_spacing=0.15,
        subplot_titles=[
            f"Negative DART_SLT Cumulative Count (Absolute Values){title_suffix}",
            f"Negative DART_SLT Cumulative Distribution (Absolute Values){title_suffix}",
            f"Positive DART_SLT Cumulative Count{title_suffix}",
            f"Positive DART_SLT Cumulative Distribution{title_suffix}"
        ]
    )
    
    # Process both datasets
    datasets = [
        (negative_values_abs, "Negative DART_SLT (Abs)", "blue", 1),  # Row 1 for negative
        (positive_values, "Positive DART_SLT", "red", 2)              # Row 2 for positive
    ]
    
    cumulative_stats = []
    
    for data, name, color, row in datasets:
        if len(data) < 10:  # Skip if insufficient data
            print(f"Warning: Insufficient data for {name} ({len(data)} points)")
            continue
            
        # Sort data for cumulative analysis
        data_sorted = data.sort_values()
        
        # Calculate cumulative statistics
        cumulative_count = np.arange(1, len(data_sorted) + 1)
        cumulative_prob = cumulative_count / len(data_sorted)
        
        # Add cumulative count trace (left column)
        fig.add_trace(
            go.Scatter(
                x=data_sorted,
                y=cumulative_count,
                mode="lines",
                name=f"{name} Count",
                line=dict(color=color, width=2),
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Add cumulative probability trace (right column)
        fig.add_trace(
            go.Scatter(
                x=data_sorted,
                y=cumulative_prob,
                mode="lines",
                name=f"{name} Distribution",
                line=dict(color=color, width=2),
                showlegend=True
            ),
            row=row, col=2
        )
        
        # Store statistics for CSV output
        cumulative_stats.append({
            "Distribution": name,
            "Count": len(data),
            "Proportion": len(data) / len(dart_slt_clean),
            "Min_Value": data.min(),
            "Max_Value": data.max(),
            "Median": data.median(),
            "25th_Percentile": np.percentile(data, 25),
            "75th_Percentile": np.percentile(data, 75)
        })
    
    # Update layout
    fig.update_layout(
        title={
            "text": f"DART Signed-Log Transform Cumulative Analysis{title_suffix}<br><sub>Note: Negative values shown as absolute values for clearer percentile interpretation</sub>",
            "x": 0.5,
            "xanchor": "center"
        },
        height=800,  # Increased height to accommodate 2x2 grid
        showlegend=True
    )
    
    # Update axis labels for all subplots
    # Row 1 (Negative values)
    fig.update_yaxes(title_text="Cumulative Count", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
    fig.update_xaxes(title_text="Signed-Log DART (Absolute Value)", row=1, col=1)
    fig.update_xaxes(title_text="Signed-Log DART (Absolute Value)", row=1, col=2)
    
    # Row 2 (Positive values)  
    fig.update_yaxes(title_text="Cumulative Count", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2)
    fig.update_xaxes(title_text="Signed-Log DART", row=2, col=1)
    fig.update_xaxes(title_text="Signed-Log DART", row=2, col=2)
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dart_slt_cumulative.html"
    fig.write_html(output_path)
    print(f"DART_SLT cumulative plot saved to: {output_path}")
    
    # Save cumulative statistics
    cumulative_stats_df = pd.DataFrame(cumulative_stats)
    stats_path = output_dir / "dart_slt_cumulative_stats.csv"
    cumulative_stats_df.to_csv(stats_path, index=False)
    print(f"DART_SLT cumulative statistics saved to: {stats_path}")


def plot_dart_slt_by_weekday(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create day-of-week analysis of signed log transformed DART distributions.
    
    Creates a bar plot showing:
    - Average and standard deviation of positive dart_slt values by day of week
    - Average and standard deviation of negative dart_slt values (absolute values) by day of week
    - Error bars representing standard deviation
    
    This helps identify systematic patterns in price differences across weekdays vs weekends.
    
    Args:
        df: DataFrame containing dart_slt column and timestamp column
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Verify dart_slt column exists (should be provided by dataset)
    if "dart_slt" not in df.columns:
        raise ValueError("dart_slt column not found. Ensure dataset.py creates this column.")
    
    # Ensure we have the day_of_week column (should be provided by dataset)
    if "day_of_week" not in df.columns:
        raise ValueError("day_of_week column not found. Ensure dataset.py creates this column.")
    
    # Map day_of_week numbers to day names
    day_mapping = {
        0: "Monday",
        1: "Tuesday", 
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }
    
    df["weekday"] = df["day_of_week"].map(day_mapping)
    
    # Define day order for proper plotting
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Separate positive and negative values
    dart_slt_clean = df.dropna(subset=["dart_slt", "weekday"])
    positive_data = dart_slt_clean[dart_slt_clean["dart_slt"] > 0].copy()
    negative_data = dart_slt_clean[dart_slt_clean["dart_slt"] < 0].copy()
    
    # Take absolute values of negative data for proper interpretation
    negative_data["dart_slt_abs"] = negative_data["dart_slt"].abs()
    
    print(f"Weekday analysis: {len(positive_data)} positive values, {len(negative_data)} negative values")
    
    # Calculate statistics by weekday for both datasets
    positive_stats = positive_data.groupby("weekday")["dart_slt"].agg(["mean", "std", "count"]).reset_index()
    negative_stats = negative_data.groupby("weekday")["dart_slt_abs"].agg(["mean", "std", "count"]).reset_index()
    
    # Ensure all days are present (fill missing with NaN)
    positive_stats = positive_stats.set_index("weekday").reindex(day_order).reset_index()
    negative_stats = negative_stats.set_index("weekday").reindex(day_order).reset_index()
    
    # Create the plot
    fig = go.Figure()
    
    # Add positive DART_SLT bars
    fig.add_trace(
        go.Bar(
            x=positive_stats["weekday"],
            y=positive_stats["mean"],
            error_y=dict(
                type="data",
                array=positive_stats["std"],
                visible=True,
                color="darkblue"
            ),
            name="Positive DART_SLT",
            marker_color="lightblue",
            opacity=0.8,
            offsetgroup=1
        )
    )
    
    # Add negative DART_SLT bars (absolute values)
    fig.add_trace(
        go.Bar(
            x=negative_stats["weekday"],
            y=negative_stats["mean"],
            error_y=dict(
                type="data",
                array=negative_stats["std"],
                visible=True,
                color="darkred"
            ),
            name="Negative DART_SLT (Absolute Values)",
            marker_color="lightcoral",
            opacity=0.8,
            offsetgroup=2
        )
    )
    
    # Update layout
    fig.update_layout(
        title={
            "text": f"DART Signed-Log Transform by Delivery Day{title_suffix}<br><sub>Note: Negative values shown as absolute values for comparison with positive values</sub>",
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis_title="Delivery Day",
        yaxis_title="Mean Signed-Log Transformed DART",
        height=600,
        showlegend=True,
        barmode="group",  # Group bars side by side
        bargap=0.2,      # Gap between groups
        bargroupgap=0.1  # Gap between bars in a group
    )
    
    # Ensure days are in correct order
    fig.update_xaxes(categoryorder="array", categoryarray=day_order)
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dart_slt_by_weekday.html"
    fig.write_html(output_path)
    print(f"DART_SLT weekday plot saved to: {output_path}")
    
    # Prepare comprehensive statistics for CSV output
    weekday_stats = []
    
    for day in day_order:
        pos_row = positive_stats[positive_stats["weekday"] == day]
        neg_row = negative_stats[negative_stats["weekday"] == day]
        
        weekday_stats.append({
            "weekday": day,
            "positive_mean": pos_row["mean"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["mean"].iloc[0]) else None,
            "positive_std": pos_row["std"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["std"].iloc[0]) else None,
            "positive_count": pos_row["count"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["count"].iloc[0]) else 0,
            "negative_mean_abs": neg_row["mean"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["mean"].iloc[0]) else None,
            "negative_std_abs": neg_row["std"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["std"].iloc[0]) else None,
            "negative_count": neg_row["count"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["count"].iloc[0]) else 0
        })
    
    # Save weekday statistics
    weekday_stats_df = pd.DataFrame(weekday_stats)
    stats_path = output_dir / "dart_slt_weekday_stats.csv"
    weekday_stats_df.to_csv(stats_path, index=False)
    print(f"DART_SLT weekday statistics saved to: {stats_path}")
    
    # Print summary for quick review
    print("\nWeekday Analysis Summary:")
    print("=" * 50)
    for _, row in weekday_stats_df.iterrows():
        pos_mean = f"{row['positive_mean']:.3f}" if pd.notna(row['positive_mean']) else "N/A"
        neg_mean = f"{row['negative_mean_abs']:.3f}" if pd.notna(row['negative_mean_abs']) else "N/A"
        print(f"{row['weekday']:<10}: Positive={pos_mean}, Negative(Abs)={neg_mean}")


def plot_dart_slt_by_hour(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create hourly analysis of signed log transformed DART distributions.
    
    Creates a bar plot showing:
    - Average and standard deviation of positive dart_slt values by hour of day (local time)
    - Average and standard deviation of negative dart_slt values (absolute values) by hour of day
    - Error bars representing standard deviation
    
    Uses local time since ERCOT operates on Central Time and this provides more intuitive 
    business interpretation (e.g., hour 16 = 4 PM Central Time ending hour).
    
    Args:
        df: DataFrame containing dart_slt column and local_ts timestamp column
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Verify dart_slt column exists (should be provided by dataset)
    if "dart_slt" not in df.columns:
        raise ValueError("dart_slt column not found. Ensure dataset.py creates this column.")
    
    # Ensure we have the end_of_hour column (should be provided by dataset)
    if "end_of_hour" not in df.columns:
        raise ValueError("end_of_hour column not found. Ensure dataset.py creates this column.")
    
    # Use the end_of_hour column directly (no timestamp transformations needed)
    df["hour"] = df["end_of_hour"]
    
    # Separate positive and negative values
    dart_slt_clean = df.dropna(subset=["dart_slt", "hour"])
    positive_data = dart_slt_clean[dart_slt_clean["dart_slt"] > 0].copy()
    negative_data = dart_slt_clean[dart_slt_clean["dart_slt"] < 0].copy()
    
    # Take absolute values of negative data for proper interpretation
    negative_data["dart_slt_abs"] = negative_data["dart_slt"].abs()
    
    print(f"Hourly analysis: {len(positive_data)} positive values, {len(negative_data)} negative values")
    
    # Calculate statistics by hour for both datasets
    positive_stats = positive_data.groupby("hour")["dart_slt"].agg(["mean", "std", "count"]).reset_index()
    negative_stats = negative_data.groupby("hour")["dart_slt_abs"].agg(["mean", "std", "count"]).reset_index()
    
    # Ensure all hours are present (fill missing with NaN) - 1 to 24 (since end_of_hour adds 1)
    all_hours = list(range(1, 25))
    positive_stats = positive_stats.set_index("hour").reindex(all_hours).reset_index()
    negative_stats = negative_stats.set_index("hour").reindex(all_hours).reset_index()
    
    # Create the plot
    fig = go.Figure()
    
    # Add positive DART_SLT bars
    fig.add_trace(
        go.Bar(
            x=positive_stats["hour"],
            y=positive_stats["mean"],
            error_y=dict(
                type="data",
                array=positive_stats["std"],
                visible=True,
                color="darkblue"
            ),
            name="Positive DART_SLT",
            marker_color="lightblue",
            opacity=0.8,
            offsetgroup=1
        )
    )
    
    # Add negative DART_SLT bars (absolute values)
    fig.add_trace(
        go.Bar(
            x=negative_stats["hour"],
            y=negative_stats["mean"],
            error_y=dict(
                type="data",
                array=negative_stats["std"],
                visible=True,
                color="darkred"
            ),
            name="Negative DART_SLT (Absolute Values)",
            marker_color="lightcoral",
            opacity=0.8,
            offsetgroup=2
        )
    )
    
    # Update layout
    fig.update_layout(
        title={
            "text": f"DART Signed-Log Transform by Delivery Hour (Local Time){title_suffix}<br><sub>Note: Negative values shown as absolute values for comparison with positive values</sub>",
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis_title="Delivery Hour (Central Time Ending Hour)",
        yaxis_title="Mean Signed-Log Transformed DART",
        height=600,
        showlegend=True,
        barmode="group",  # Group bars side by side
        bargap=0.2,      # Gap between groups
        bargroupgap=0.1  # Gap between bars in a group
    )
    
    # Set x-axis to show all hours and format nicely
    fig.update_xaxes(
        tickmode="linear",
        tick0=1,
        dtick=1,
        range=[0.5, 24.5]  # Show all hours from 1 to 24 (ending hours)
    )
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dart_slt_by_hour.html"
    fig.write_html(output_path)
    print(f"DART_SLT hourly plot saved to: {output_path}")
    
    # Prepare comprehensive statistics for CSV output
    hourly_stats = []
    
    for hour in all_hours:
        pos_row = positive_stats[positive_stats["hour"] == hour]
        neg_row = negative_stats[negative_stats["hour"] == hour]
        
        hourly_stats.append({
            "hour": hour,
            "positive_mean": pos_row["mean"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["mean"].iloc[0]) else None,
            "positive_std": pos_row["std"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["std"].iloc[0]) else None,
            "positive_count": pos_row["count"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["count"].iloc[0]) else 0,
            "negative_mean_abs": neg_row["mean"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["mean"].iloc[0]) else None,
            "negative_std_abs": neg_row["std"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["std"].iloc[0]) else None,
            "negative_count": neg_row["count"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["count"].iloc[0]) else 0
        })
    
    # Save hourly statistics
    hourly_stats_df = pd.DataFrame(hourly_stats)
    stats_path = output_dir / "dart_slt_hourly_stats.csv"
    hourly_stats_df.to_csv(stats_path, index=False)
    print(f"DART_SLT hourly statistics saved to: {stats_path}")
    
    # Print summary for quick review of key business hours
    print("\nHourly Analysis Summary (Key Business Hours):")
    print("=" * 55)
    key_hours = [7, 8, 9, 10, 17, 18, 19, 20, 21]  # Morning and evening peaks (ending hours)
    for hour in key_hours:
        row = hourly_stats_df[hourly_stats_df["hour"] == hour].iloc[0]
        pos_mean = f"{row['positive_mean']:.3f}" if pd.notna(row['positive_mean']) else "N/A"
        neg_mean = f"{row['negative_mean_abs']:.3f}" if pd.notna(row['negative_mean_abs']) else "N/A"
        print(f"Hour {hour:2d}:00: Positive={pos_mean}, Negative(Abs)={neg_mean}") 