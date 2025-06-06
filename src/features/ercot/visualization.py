"""Visualization utilities for ERCOT feature analysis."""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from ..utils import compute_power_spectrum, compute_kmeans_clustering


# =============================================================================
# PROFESSIONAL STYLING INFRASTRUCTURE
# =============================================================================

# Professional color palette (colorblind-friendly, publication-ready)
PROFESSIONAL_COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Deep magenta 
    'accent': '#F18F01',       # Warm orange
    'neutral': '#C73E1D',      # Deep red
    'success': '#4A7A8A',      # Teal
    'warning': '#8B4513',      # Saddle brown
    'info': '#556B2F',         # Dark olive green
    'purple': '#4B0082',       # Indigo
    'background': '#F5F5F5',   # Light gray
    'text': '#2C3E50',         # Dark blue-gray
    'grid': 'rgba(128,128,128,0.2)',  # Subtle grid
}

# Semantic color mappings for consistency
SEMANTIC_COLORS = {
    'positive': PROFESSIONAL_COLORS['primary'],     # Blue for positive values
    'negative': PROFESSIONAL_COLORS['secondary'],   # Magenta for negative values
    'positive_fill': f"rgba(46, 134, 171, 0.6)",   # Semi-transparent blue
    'negative_fill': f"rgba(162, 59, 114, 0.6)",   # Semi-transparent magenta
    'neutral_line': PROFESSIONAL_COLORS['text'],    # Dark gray for reference lines
    'highlight': PROFESSIONAL_COLORS['accent'],     # Orange for highlights/optimal points
}

# Extended color sequence for multi-series plots
COLOR_SEQUENCE = [
    PROFESSIONAL_COLORS['primary'],    # Blue
    PROFESSIONAL_COLORS['secondary'],  # Magenta
    PROFESSIONAL_COLORS['accent'],     # Orange
    PROFESSIONAL_COLORS['success'],    # Teal
    PROFESSIONAL_COLORS['neutral'],    # Red
    PROFESSIONAL_COLORS['warning'],    # Brown
    PROFESSIONAL_COLORS['info'],       # Olive
    PROFESSIONAL_COLORS['purple'],     # Indigo
]


def get_professional_layout(
    title: str,
    height: int = 600,
    width: int = None,
    showlegend: bool = True,
    legend_position: str = 'auto'
) -> dict:
    """Get professional layout configuration for Plotly figures.
    
    Args:
        title: Main title for the plot
        height: Plot height in pixels
        width: Plot width in pixels (optional)
        showlegend: Whether to show legend
        legend_position: Legend position ('auto', 'upper_left', 'upper_right', 'external_right')
    
    Returns:
        dict: Professional layout configuration
    """
    layout = {
        'title': {
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': PROFESSIONAL_COLORS['text']}
        },
        'height': height,
        'showlegend': showlegend,
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white',
        'margin': dict(t=100, b=60, l=60, r=60),
        'font': {'color': PROFESSIONAL_COLORS['text']}
    }
    
    if width:
        layout['width'] = width
    
    # Configure legend positioning
    if showlegend:
        legend_configs = {
            'upper_left': dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)"),
            'upper_right': dict(x=0.98, y=0.98, xanchor='right', bgcolor="rgba(255,255,255,0.9)"),
            'external_right': dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.9)"),
            'auto': dict(bgcolor="rgba(255,255,255,0.9)")
        }
        
        legend_config = legend_configs.get(legend_position, legend_configs['auto'])
        legend_config.update({
            'bordercolor': PROFESSIONAL_COLORS['text'],
            'borderwidth': 1,
            'font': dict(color=PROFESSIONAL_COLORS['text'], size=11)
        })
        layout['legend'] = legend_config
        
        # Adjust right margin for external legend
        if legend_position == 'external_right':
            layout['margin']['r'] = 150
    
    return layout


def get_professional_axis_style() -> dict:
    """Get professional axis styling configuration.
    
    Returns:
        dict: Axis styling configuration
    """
    return {
        'title_font': dict(color=PROFESSIONAL_COLORS['text'], size=12),
        'tickfont': dict(color=PROFESSIONAL_COLORS['text']),
        'gridcolor': PROFESSIONAL_COLORS['grid'],
        'showgrid': True,
        'zeroline': True,
        'zerolinecolor': PROFESSIONAL_COLORS['grid'],
        'linecolor': PROFESSIONAL_COLORS['text']
    }


def apply_professional_axis_styling(fig, rows: int = 1, cols: int = 1) -> None:
    """Apply professional axis styling to all subplots in a figure.
    
    Args:
        fig: Plotly figure object
        rows: Number of subplot rows
        cols: Number of subplot columns
    """
    axis_style = get_professional_axis_style()
    
    # Handle single plots (go.Figure) vs subplots (make_subplots)
    if rows == 1 and cols == 1:
        # Check if this is a subplot figure or a simple figure
        try:
            # Try to update with row/col (subplot case)
            fig.update_xaxes(**axis_style, row=1, col=1)
            fig.update_yaxes(**axis_style, row=1, col=1)
        except Exception:
            # Fall back to simple figure case (no row/col)
            fig.update_xaxes(**axis_style)
            fig.update_yaxes(**axis_style)
    else:
        # Multiple subplots - use row/col parameters
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                fig.update_xaxes(**axis_style, row=row, col=col)
                fig.update_yaxes(**axis_style, row=row, col=col)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_dart_by_location(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create DART (Day-Ahead to Real-Time) price difference visualization for each settlement point.
    
    Creates separate plots for each settlement point with:
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
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating DART analysis plots for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(point_data)} data points")
        
        # Create subplot with 2 rows
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=[
                f"Raw DART Price Differences - {point}{title_suffix}",
                f"Signed Log Transformed DART Price Differences - {point}{title_suffix}"
            ]
        )
        
        # Add trace for raw DART (upper plot)
        fig.add_trace(
            go.Scatter(
                x=point_data["local_ts"],
                y=point_data["dart"],
                mode="lines",
                name="Raw DART",
                line=dict(color=SEMANTIC_COLORS['positive'], width=2),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add trace for signed log transformed DART (lower plot)
        fig.add_trace(
            go.Scatter(
                x=point_data["local_ts"],
                y=point_data["dart_slt"],
                mode="lines",
                name="SLT DART",
                line=dict(color=SEMANTIC_COLORS['negative'], width=2),
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Add horizontal reference lines at y=0 for both plots
        for row in [1, 2]:
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color=SEMANTIC_COLORS['neutral_line'],
                line_width=1,
                row=row, col=1
            )
        
        # Apply professional layout
        professional_title = f"DART Price Differences Analysis - {point}{title_suffix}"
        layout = get_professional_layout(
            title=professional_title,
            height=800,
            showlegend=True,
            legend_position='upper_left'
        )
        fig.update_layout(**layout)
        
        # Apply professional axis styling
        apply_professional_axis_styling(fig, rows=2, cols=1)
        
        # Update specific axis labels
        fig.update_yaxes(title_text="DART Price Difference ($/MWh)", row=1, col=1)
        fig.update_yaxes(title_text="Signed Log DART", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        # Save individual plot
        output_path = output_dir / f"dart_by_location_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save individual data file
        data_path = output_dir / f"dart_by_location_{safe_filename}.csv"
        point_data.to_csv(data_path, index=False)
        print(f"  Data saved to: {data_path}")
    
    print(f"DART analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_distributions(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create histogram visualization of DART distributions for each settlement point.
    
    Creates separate plots for each settlement point with:
    - Upper plot: Raw DART histogram with fitted normal distribution overlay
    - Lower plot: Signed log transformed DART histogram with fitted normal distribution overlay
    
    Args:
        df: DataFrame containing DART data with 'dart' column, location, and location_type
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
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating distribution analysis plots for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        
        # Remove NaN values for statistics
        dart_clean = point_data["dart"].dropna()
        dart_slt_clean = point_data["dart_slt"].dropna()
        
        if len(dart_clean) < 10 or len(dart_slt_clean) < 10:
            print(f"Warning: Insufficient data for {point}, skipping")
            continue
            
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(dart_clean)} raw, {len(dart_slt_clean)} transformed data points")
        
        # Create subplot with 2 rows
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.25,
            subplot_titles=[
                f"Raw DART Distribution - {point}{title_suffix}",
                f"Signed Log Transformed DART Distribution - {point}{title_suffix}"
            ]
        )
        
        # Store statistics for this settlement point
        point_stats = []
        
        # Plot histograms and overlays for both datasets
        datasets = [
            (dart_clean, "Raw DART", 1),
            (dart_slt_clean, "SLT DART", 2)
        ]
        
        for data, data_type, row in datasets:
            # Fit normal distribution
            mu, std = stats.norm.fit(data)
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=f"{data_type} Histogram",
                    nbinsx=120,  # Increased from 30 to 120 for finer resolution (1/4 current resolution)
                    histnorm="probability density",
                    opacity=0.7,
                    marker_color=SEMANTIC_COLORS['positive_fill'] if row == 1 else SEMANTIC_COLORS['negative_fill'],
                    marker_line=dict(width=1, color=SEMANTIC_COLORS['positive'] if row == 1 else SEMANTIC_COLORS['negative']),
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
                    name=f"{data_type} Normal Fit (μ={mu:.2f}, σ={std:.2f})",
                    line=dict(
                        color=SEMANTIC_COLORS['positive'] if row == 1 else SEMANTIC_COLORS['negative'], 
                        width=2, 
                        dash="dash"
                    ),
                    showlegend=True
                ),
                row=row, col=1
            )
            
            # Calculate percentiles for statistics
            percentiles = {
                "67th": np.percentile(data, 67),
                "90th": np.percentile(data, 90), 
                "95th": np.percentile(data, 95),
                "99th": np.percentile(data, 99)
            }
            
            # Store statistics
            point_stats.append({
                "Settlement_Point": point,
                "Data_Type": data_type,
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
        professional_title = f"DART Distribution Analysis - {point}{title_suffix}"
        layout = get_professional_layout(
            title=professional_title,
            height=800,
            showlegend=True,
            legend_position='upper_right'
        )
        fig.update_layout(**layout)
        
        # Apply professional axis styling
        apply_professional_axis_styling(fig, rows=2, cols=1)
        
        # Update specific axis labels
        fig.update_yaxes(title_text="Probability Density", row=1, col=1)
        fig.update_yaxes(title_text="Probability Density", row=2, col=1)
        fig.update_xaxes(title_text="DART ($/MWh)", row=1, col=1)
        fig.update_xaxes(title_text="Signed Log DART", row=2, col=1)
        
        # Save individual plot
        output_path = output_dir / f"dart_distributions_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save individual statistics
        if point_stats:
            stats_df = pd.DataFrame(point_stats)
            stats_path = output_dir / f"dart_distribution_stats_{safe_filename}.csv"
            stats_df.to_csv(stats_path, index=False)
            print(f"  Statistics saved to: {stats_path}")
    
    print(f"Distribution analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_boxplots(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create box plot visualization of DART distributions for each settlement point.
    
    Creates separate plots for each settlement point with:
    - Left plot: Raw DART box plot
    - Right plot: Signed log transformed DART box plot
    
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
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating box plot analysis for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(point_data)} data points")
        
        # Create subplot with 1 row, 2 columns (side by side)
        fig = make_subplots(
            rows=1, cols=2,
            shared_yaxes=False,
            horizontal_spacing=0.15,
            subplot_titles=[
                f"Raw DART Box Plot - {point}{title_suffix}",
                f"Signed Log Transformed DART Box Plot - {point}{title_suffix}"
            ]
        )
        
        # Add box plot for raw DART (left plot)
        fig.add_trace(
            go.Box(
                y=point_data["dart"],
                name="Raw DART",
                marker_color=SEMANTIC_COLORS['positive'],
                line=dict(color=SEMANTIC_COLORS['positive'], width=2),
                boxpoints="outliers",
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add box plot for transformed DART (right plot)  
        fig.add_trace(
            go.Box(
                y=point_data["dart_slt"],
                name="SLT DART",
                marker_color=SEMANTIC_COLORS['negative'],
                line=dict(color=SEMANTIC_COLORS['negative'], width=2),
                boxpoints="outliers",
                showlegend=True
            ),
            row=1, col=2
        )
        
        # Apply professional layout
        professional_title = f"DART Box Plot Analysis - {point}{title_suffix}"
        layout = get_professional_layout(
            title=professional_title,
            height=600,
            showlegend=True,
            legend_position='upper_right'
        )
        fig.update_layout(**layout)
        
        # Apply professional axis styling
        apply_professional_axis_styling(fig, rows=1, cols=2)
        
        # Update specific axis labels
        fig.update_yaxes(title_text="DART ($/MWh)", row=1, col=1)
        fig.update_yaxes(title_text="Signed Log DART", row=1, col=2)
        fig.update_xaxes(title_text="Data Type", row=1, col=1)
        fig.update_xaxes(title_text="Data Type", row=1, col=2)
        
        # Save individual plot
        output_path = output_dir / f"dart_boxplots_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save individual box plot statistics
        raw_stats = point_data["dart"].describe()
        slt_stats = point_data["dart_slt"].describe()
        
        box_stats = [{
            "Settlement_Point": point,
            "Data_Type": "Raw DART",
            "Count": raw_stats["count"],
            "Mean": raw_stats["mean"],
            "Std": raw_stats["std"],
            "Min": raw_stats["min"],
            "Q1": raw_stats["25%"],
            "Median": raw_stats["50%"], 
            "Q3": raw_stats["75%"],
            "Max": raw_stats["max"],
            "IQR": raw_stats["75%"] - raw_stats["25%"]
        }, {
            "Settlement_Point": point,
            "Data_Type": "SLT DART",
            "Count": slt_stats["count"],
            "Mean": slt_stats["mean"],
            "Std": slt_stats["std"],
            "Min": slt_stats["min"],
            "Q1": slt_stats["25%"],
            "Median": slt_stats["50%"],
            "Q3": slt_stats["75%"],
            "Max": slt_stats["max"],
            "IQR": slt_stats["75%"] - slt_stats["25%"]
        }]
        
        box_stats_df = pd.DataFrame(box_stats)
        stats_path = output_dir / f"dart_boxplot_stats_{safe_filename}.csv"
        box_stats_df.to_csv(stats_path, index=False)
        print(f"  Statistics saved to: {stats_path}")
    
    print(f"Box plot analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_qqplots(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create Q-Q plots for normality assessment of DART distributions for each settlement point.
    
    Creates separate plots for each settlement point with:
    - Left plot: Raw DART Q-Q plot against normal distribution
    - Right plot: Signed log transformed DART Q-Q plot against normal distribution
    
    Q-Q plots help assess how closely the data follows a normal distribution.
    Points that fall along the diagonal line indicate normal distribution.
    
    Args:
        df: DataFrame containing DART data with 'dart' column, location, and location_type
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
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating Q-Q plot analysis for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        
        # Remove NaN values for statistics
        dart_clean = point_data["dart"].dropna()
        dart_slt_clean = point_data["dart_slt"].dropna()
        
        if len(dart_clean) < 10 or len(dart_slt_clean) < 10:
            print(f"Warning: Insufficient data for {point}, skipping")
            continue
            
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(dart_clean)} raw, {len(dart_slt_clean)} transformed data points")
        
        # Create subplot with 1 row, 2 columns (side by side)
        fig = make_subplots(
            rows=1, cols=2,
            shared_xaxes=False,
            horizontal_spacing=0.15,
            subplot_titles=[
                f"Raw DART Q-Q Plot - {point}{title_suffix}",
                f"Signed Log Transformed DART Q-Q Plot - {point}{title_suffix}"
            ]
        )
        
        # Store statistics for this settlement point
        qq_stats = []
        
        # Plot Q-Q plots for both datasets
        datasets = [
            (dart_clean, "Raw DART", 1),
            (dart_slt_clean, "SLT DART", 2)
        ]
        
        for data, data_type, col in datasets:
            # Generate Q-Q plot data using scipy
            (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=None)
            
            # Add scatter plot of actual Q-Q points
            fig.add_trace(
                go.Scatter(
                    x=osm,
                    y=osr,
                    mode="markers",
                    name=f"{data_type} Data",
                    marker=dict(
                        color=SEMANTIC_COLORS['positive'] if col == 1 else SEMANTIC_COLORS['negative'], 
                        size=4, 
                        opacity=0.7
                    ),
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
                    name=f"{data_type} Normal Line (R²={r**2:.3f})",
                    line=dict(
                        color=SEMANTIC_COLORS['positive'] if col == 1 else SEMANTIC_COLORS['negative'], 
                        width=3,
                        dash="dash"
                    ),
                    showlegend=True
                ),
                row=1, col=col
            )
            
            # Store statistics
            qq_stats.append({
                "Settlement_Point": point,
                "Data_Type": data_type,
                "Count": len(data),
                "R_squared": r**2,
                "Slope": slope,
                "Intercept": intercept,
                "Correlation": r
            })
        
        # Apply professional layout
        professional_title = f"DART Q-Q Plot Analysis - {point}{title_suffix}"
        layout = get_professional_layout(
            title=professional_title,
            height=600,
            showlegend=True,
            legend_position='upper_left'
        )
        fig.update_layout(**layout)
        
        # Apply professional axis styling
        apply_professional_axis_styling(fig, rows=1, cols=2)
        
        # Update specific axis labels
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Normal Quantiles", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Normal Quantiles", row=1, col=2)
        
        # Save individual plot
        output_path = output_dir / f"dart_qqplots_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save individual Q-Q statistics
        if qq_stats:
            qq_stats_df = pd.DataFrame(qq_stats)
            stats_path = output_dir / f"dart_qqplot_stats_{safe_filename}.csv"
            qq_stats_df.to_csv(stats_path, index=False)
            print(f"  Statistics saved to: {stats_path}")
    
    print(f"Q-Q plot analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_bimodal(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create bimodal analysis of signed log transformed DART distributions for each settlement point.
    
    Creates separate bimodal analysis for each settlement point, with plots showing:
    - Left subplot: Negative dart_slt values with histogram, normal fit, and percentiles
    - Right subplot: Positive dart_slt values with histogram, normal fit, and percentiles
    
    This helps analyze whether each mode of the bimodal distribution is independently normal
    for each settlement point.
    
    Args:
        df: DataFrame containing dart_slt column, location, and location_type
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
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating bimodal analysis for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        
        # Separate positive and negative values
        dart_slt_clean = point_data["dart_slt"].dropna()
        negative_values = dart_slt_clean[dart_slt_clean < 0]
        positive_values = dart_slt_clean[dart_slt_clean > 0]
        
        # Take absolute values of negative values for proper percentile interpretation
        negative_values_abs = negative_values.abs()
        
        if len(negative_values) < 5 and len(positive_values) < 5:
            print(f"Warning: Insufficient data for {point}, skipping")
            continue
            
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(negative_values)} negative, {len(positive_values)} positive values")
        
        # Create subplot with 1 row, 2 columns
        fig = make_subplots(
            rows=1, cols=2,
            shared_xaxes=False,
            horizontal_spacing=0.15,
            subplot_titles=[
                f"Negative DART_SLT (Absolute Values) - {point}",
                f"Positive DART_SLT - {point}"
            ]
        )
        
        # Store statistics for this settlement point
        bimodal_stats = []
        
        # Plot histograms and overlays for both datasets
        datasets = [
            (negative_values_abs, "Negative DART_SLT (Abs)", 1),
            (positive_values, "Positive DART_SLT", 2)
        ]
        
        for data, name, col in datasets:
            if len(data) < 5:  # Skip if insufficient data
                print(f"Warning: Insufficient data for {point} {name} ({len(data)} points)")
                continue
                
            # Fit normal distribution
            mu, std = stats.norm.fit(data)
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=f"{name}",
                    nbinsx=80,  # Increased from 20 to 80 for finer resolution (1/4 current resolution)
                    histnorm="probability density",
                    opacity=0.7,
                    marker_color=SEMANTIC_COLORS['positive_fill'] if col == 1 else SEMANTIC_COLORS['negative_fill'],
                    marker_line=dict(width=1, color=SEMANTIC_COLORS['positive'] if col == 1 else SEMANTIC_COLORS['negative']),
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
                    name=f"{name} Normal Fit",
                    line=dict(
                        color=SEMANTIC_COLORS['positive'] if col == 1 else SEMANTIC_COLORS['negative'], 
                        width=2, 
                        dash="dash"
                    ),
                    showlegend=True
                ),
                row=1, col=col
            )
            
            # Calculate percentiles for statistics
            percentiles = {
                "67th": np.percentile(data, 67),
                "90th": np.percentile(data, 90), 
                "95th": np.percentile(data, 95),
                "99th": np.percentile(data, 99)
            }
            
            # Store statistics for CSV output
            bimodal_stats.append({
                "Settlement_Point": point,
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
        professional_title = f"DART Signed-Log Transform Bimodal Analysis - {point}{title_suffix}<br><sub style='color:{PROFESSIONAL_COLORS['text']}'>Note: Negative values shown as absolute values for clearer percentile interpretation</sub>"
        layout = get_professional_layout(
            title=professional_title,
            height=600,
            showlegend=True,
            legend_position='upper_right'
        )
        fig.update_layout(**layout)
        
        # Apply professional axis styling
        apply_professional_axis_styling(fig, rows=1, cols=2)
        
        # Update specific axis labels
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Signed-Log DART (Absolute Value)", row=1, col=1)
        fig.update_xaxes(title_text="Signed-Log DART", row=1, col=2)
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_bimodal_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save individual bimodal statistics
        if bimodal_stats:
            bimodal_stats_df = pd.DataFrame(bimodal_stats)
            stats_path = output_dir / f"dart_slt_bimodal_stats_{safe_filename}.csv"
            bimodal_stats_df.to_csv(stats_path, index=False)
            print(f"  Statistics saved to: {stats_path}")
    
    print(f"Bimodal analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_cumulative(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create cumulative analysis of signed log transformed DART distributions for each settlement point.
    
    Creates separate cumulative analysis for each settlement point, with plots showing:
    - Upper left: Negative DART_SLT cumulative count (absolute values)
    - Upper right: Negative DART_SLT cumulative distribution (absolute values)
    - Lower left: Positive DART_SLT cumulative count
    - Lower right: Positive DART_SLT cumulative distribution
    
    This helps understand the accumulation patterns and relative frequencies of positive vs negative 
    price differences at each settlement point.
    
    Args:
        df: DataFrame containing dart_slt column, location, and location_type
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
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating cumulative analysis for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        
        # Separate positive and negative values (excluding zeros for cleaner analysis)
        dart_slt_clean = point_data["dart_slt"].dropna()
        negative_values = dart_slt_clean[dart_slt_clean < 0]
        positive_values = dart_slt_clean[dart_slt_clean > 0]
        
        # Take absolute values of negative values for proper percentile interpretation
        negative_values_abs = negative_values.abs()
        
        if len(negative_values) < 5 and len(positive_values) < 5:
            print(f"Warning: Insufficient data for {point}, skipping")
            continue
            
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(negative_values)} negative values, {len(positive_values)} positive values")
        
        # Create subplot with 2 rows, 2 columns (2x2 grid)
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=False,
            horizontal_spacing=0.15,
            vertical_spacing=0.15,
            subplot_titles=[
                f"Negative DART_SLT Cumulative Count (Abs) - {point}",
                f"Negative DART_SLT Cumulative Distribution (Abs) - {point}",
                f"Positive DART_SLT Cumulative Count - {point}",
                f"Positive DART_SLT Cumulative Distribution - {point}"
            ]
        )
        
        # Store statistics for this settlement point
        cumulative_stats = []
        
        # Process both datasets for this settlement point
        datasets = [
            (negative_values_abs, "Negative DART_SLT (Abs)", "blue", 1),  # Row 1 for negative
            (positive_values, "Positive DART_SLT", "red", 2)              # Row 2 for positive
        ]
        
        for data, name, color, row in datasets:
            if len(data) < 5:  # Skip if insufficient data
                print(f"Warning: Insufficient data for {point} {name} ({len(data)} points)")
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
                    line=dict(color=SEMANTIC_COLORS['positive'] if row == 1 else SEMANTIC_COLORS['negative'], width=2.5),
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
                    line=dict(color=SEMANTIC_COLORS['positive'] if row == 1 else SEMANTIC_COLORS['negative'], width=2.5, dash="dash"),
                    showlegend=True
                ),
                row=row, col=2
            )
            
            # Store statistics for CSV output
            cumulative_stats.append({
                "Settlement_Point": point,
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
        professional_title = f"DART Signed-Log Transform Cumulative Analysis - {point}{title_suffix}<br><sub>Note: Negative values shown as absolute values for clearer percentile interpretation</sub>"
        layout = get_professional_layout(
            title=professional_title,
            height=800,  # Increased height to accommodate 2x2 grid
            showlegend=True
        )
        fig.update_layout(**layout)
        
        # Apply professional axis styling
        apply_professional_axis_styling(fig, rows=2, cols=2)
        
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
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_cumulative_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save individual cumulative statistics
        if cumulative_stats:
            cumulative_stats_df = pd.DataFrame(cumulative_stats)
            stats_path = output_dir / f"dart_slt_cumulative_stats_{safe_filename}.csv"
            cumulative_stats_df.to_csv(stats_path, index=False)
            print(f"  Statistics saved to: {stats_path}")
    
    print(f"Cumulative analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_by_weekday(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create day-of-week analysis of signed log transformed DART distributions for each settlement point.
    
    Creates separate weekday analysis for each settlement point, with bar plots showing:
    - Average and standard deviation of positive dart_slt values by day of week
    - Average and standard deviation of negative dart_slt values (absolute values) by day of week  
    - Error bars representing standard deviation
    
    This helps identify systematic patterns in price differences across weekdays vs weekends
    at each settlement point.
    
    Args:
        df: DataFrame containing dart_slt column, day_of_week column, location, and location_type
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
    
    # Ensure we have the day_of_week column (should be provided by dataset)
    if "day_of_week" not in df.columns:
        raise ValueError("day_of_week column not found. Ensure dataset.py creates this column.")
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating weekday analysis for {len(unique_points)} settlement points")
    
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
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        
        # Separate positive and negative values
        dart_slt_clean = point_data.dropna(subset=["dart_slt", "weekday"])
        positive_data = dart_slt_clean[dart_slt_clean["dart_slt"] > 0].copy()
        negative_data = dart_slt_clean[dart_slt_clean["dart_slt"] < 0].copy()
        
        # Take absolute values of negative data for proper interpretation
        negative_data["dart_slt_abs"] = negative_data["dart_slt"].abs()
        
        if len(positive_data) < 5 and len(negative_data) < 5:
            print(f"Warning: Insufficient data for {point}, skipping")
            continue
            
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(positive_data)} positive values, {len(negative_data)} negative values")
        
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
                    color=SEMANTIC_COLORS['positive'],
                    thickness=2
                ),
                name="Positive DART_SLT",
                marker_color=SEMANTIC_COLORS['positive_fill'],
                marker_line=dict(color=SEMANTIC_COLORS['positive'], width=1),
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
                    color=SEMANTIC_COLORS['negative'],
                    thickness=2
                ),
                name="Negative DART_SLT (Absolute Values)",
                marker_color=SEMANTIC_COLORS['negative_fill'],
                marker_line=dict(color=SEMANTIC_COLORS['negative'], width=1),
                opacity=0.8,
                offsetgroup=2
            )
        )
        
        # Apply professional layout
        professional_title = f"DART Signed-Log Transform by Delivery Day - {point}{title_suffix}<br><sub style='color:{PROFESSIONAL_COLORS['text']}'>Note: Negative values shown as absolute values for comparison with positive values</sub>"
        layout = get_professional_layout(
            title=professional_title,
            height=600,
            showlegend=True,
            legend_position='upper_right'
        )
        fig.update_layout(**layout)
        
        # Apply professional axis styling  
        apply_professional_axis_styling(fig, rows=1, cols=1)
        
        # Update specific axis labels and settings
        fig.update_xaxes(title_text="Delivery Day")
        fig.update_yaxes(title_text="Mean Signed-Log Transformed DART")
        fig.update_layout(
            barmode="group",  # Group bars side by side
            bargap=0.2,      # Gap between groups
            bargroupgap=0.1  # Gap between bars in a group
        )
        
        # Ensure days are in correct order
        fig.update_xaxes(categoryorder="array", categoryarray=day_order)
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_by_weekday_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Store comprehensive statistics for CSV output
        weekday_stats = []
        for day in day_order:
            pos_row = positive_stats[positive_stats["weekday"] == day]
            neg_row = negative_stats[negative_stats["weekday"] == day]
            
            weekday_stats.append({
                "Settlement_Point": point,
                "weekday": day,
                "positive_mean": pos_row["mean"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["mean"].iloc[0]) else None,
                "positive_std": pos_row["std"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["std"].iloc[0]) else None,
                "positive_count": pos_row["count"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["count"].iloc[0]) else 0,
                "negative_mean_abs": neg_row["mean"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["mean"].iloc[0]) else None,
                "negative_std_abs": neg_row["std"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["std"].iloc[0]) else None,
                "negative_count": neg_row["count"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["count"].iloc[0]) else 0
            })
        
        # Save individual weekday statistics
        weekday_stats_df = pd.DataFrame(weekday_stats)
        stats_path = output_dir / f"dart_slt_weekday_stats_{safe_filename}.csv"
        weekday_stats_df.to_csv(stats_path, index=False)
        print(f"  Statistics saved to: {stats_path}")
    
    print(f"Weekday analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_by_hour(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create hourly analysis of signed log transformed DART distributions for each settlement point.
    
    Creates separate hourly analysis for each settlement point, with bar plots showing:
    - Average and standard deviation of positive dart_slt values by hour of day (local time)
    - Average and standard deviation of negative dart_slt values (absolute values) by hour of day
    - Error bars representing standard deviation
    
    Uses local time since ERCOT operates on Central Time and this provides more intuitive 
    business interpretation (e.g., hour 16 = 4 PM Central Time ending hour).
    
    Args:
        df: DataFrame containing dart_slt column, end_of_hour column, location, and location_type
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
    
    # Ensure we have the end_of_hour column (should be provided by dataset)
    if "end_of_hour" not in df.columns:
        raise ValueError("end_of_hour column not found. Ensure dataset.py creates this column.")
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating hourly analysis for {len(unique_points)} settlement points")
    
    # Use the end_of_hour column directly (no timestamp transformations needed)
    df["hour"] = df["end_of_hour"]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        
        # Separate positive and negative values
        dart_slt_clean = point_data.dropna(subset=["dart_slt", "hour"])
        positive_data = dart_slt_clean[dart_slt_clean["dart_slt"] > 0].copy()
        negative_data = dart_slt_clean[dart_slt_clean["dart_slt"] < 0].copy()
        
        # Take absolute values of negative data for proper interpretation
        negative_data["dart_slt_abs"] = negative_data["dart_slt"].abs()
        
        if len(positive_data) < 5 and len(negative_data) < 5:
            print(f"Warning: Insufficient data for {point}, skipping")
            continue
            
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(positive_data)} positive values, {len(negative_data)} negative values")
        
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
                    color=SEMANTIC_COLORS['positive'],
                    thickness=2
                ),
                name="Positive DART_SLT",
                marker_color=SEMANTIC_COLORS['positive_fill'],
                marker_line=dict(color=SEMANTIC_COLORS['positive'], width=1),
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
                    color=SEMANTIC_COLORS['negative'],
                    thickness=2
                ),
                name="Negative DART_SLT (Absolute Values)",
                marker_color=SEMANTIC_COLORS['negative_fill'],
                marker_line=dict(color=SEMANTIC_COLORS['negative'], width=1),
                opacity=0.8,
                offsetgroup=2
            )
        )
        
        # Apply professional layout
        professional_title = f"DART Signed-Log Transform by Delivery Hour (Local Time) - {point}{title_suffix}<br><sub style='color:{PROFESSIONAL_COLORS['text']}'>Note: Negative values shown as absolute values for comparison with positive values</sub>"
        layout = get_professional_layout(
            title=professional_title,
            height=600,
            showlegend=True,
            legend_position='upper_right'
        )
        fig.update_layout(**layout)
        
        # Apply professional axis styling
        apply_professional_axis_styling(fig, rows=1, cols=1)
        
        # Update specific axis labels and settings
        fig.update_xaxes(title_text="Delivery Hour (Central Time Ending Hour)")
        fig.update_yaxes(title_text="Mean Signed-Log Transformed DART")
        fig.update_layout(
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
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_by_hour_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Store comprehensive statistics for CSV output
        hourly_stats = []
        for hour in all_hours:
            pos_row = positive_stats[positive_stats["hour"] == hour]
            neg_row = negative_stats[negative_stats["hour"] == hour]
            
            hourly_stats.append({
                "Settlement_Point": point,
                "hour": hour,
                "positive_mean": pos_row["mean"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["mean"].iloc[0]) else None,
                "positive_std": pos_row["std"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["std"].iloc[0]) else None,
                "positive_count": pos_row["count"].iloc[0] if len(pos_row) > 0 and not pd.isna(pos_row["count"].iloc[0]) else 0,
                "negative_mean_abs": neg_row["mean"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["mean"].iloc[0]) else None,
                "negative_std_abs": neg_row["std"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["std"].iloc[0]) else None,
                "negative_count": neg_row["count"].iloc[0] if len(neg_row) > 0 and not pd.isna(neg_row["count"].iloc[0]) else 0
            })
        
        # Save individual hourly statistics
        hourly_stats_df = pd.DataFrame(hourly_stats)
        stats_path = output_dir / f"dart_slt_hourly_stats_{safe_filename}.csv"
        hourly_stats_df.to_csv(stats_path, index=False)
        print(f"  Statistics saved to: {stats_path}")
    
    print(f"Hourly analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_power_spectrum(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create power spectrum analysis of signed log transformed DART time series for each settlement point.
    
    Performs Fast Fourier Transform (FFT) on the DART_SLT time series to identify
    periodic patterns and frequency components in price differences.
    
    Creates separate power spectrum plots for each settlement point.
    
    Creates plots showing:
    - Power spectrum magnitude in dB vs frequency
    - Frequency axis in cycles per day for business interpretation
    - Identification of dominant frequency components
    
    Args:
        df: DataFrame containing dart_slt column, timestamp data, location, and location_type
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Verify required columns exist
    required_cols = ["dart_slt", "utc_ts", "location", "location_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create unique identifier combining location and type
    df["point_identifier"] = df.apply(
        lambda row: f"{row['location']} ({row['location_type']})",
        axis=1
    )
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating power spectrum analysis for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point].copy()
        
        # Sort by timestamp and remove NaN values
        point_clean = point_data.dropna(subset=["dart_slt", "utc_ts"]).copy()
        point_clean = point_clean.sort_values("utc_ts").reset_index(drop=True)
        
        if len(point_clean) < 24:  # Need at least a day's worth of data
            print(f"Warning: Insufficient data for {point} ({len(point_clean)} points), skipping")
            continue
        
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(point_clean)} data points")
        
        try:
            # Compute power spectrum using the utility function
            spectrum_results = compute_power_spectrum(
                time_series=point_clean["dart_slt"].values,
                timestamps=point_clean["utc_ts"],
                peak_percentile=85
            )
            
            print(f"  Sampling frequency: {spectrum_results['sampling_freq_per_day']:.1f} cycles/day")
            
        except ValueError as e:
            print(f"Warning: Power spectrum computation failed for {point}: {e}")
            continue
        
        # Create the plot
        fig = go.Figure()
        
        # Add power spectrum trace
        fig.add_trace(
            go.Scatter(
                x=spectrum_results["freq_bins_per_day"],
                y=spectrum_results["power_spectrum_db"],
                mode="lines",
                name=f"{point} Power Spectrum",
                line=dict(color=SEMANTIC_COLORS['positive'], width=2.5),
                showlegend=True
            )
        )
        
        # Apply professional layout
        professional_title = f"DART Signed-Log Transform Power Spectrum Analysis - {point}{title_suffix}<br><sub style='color:{PROFESSIONAL_COLORS['text']}'>Frequency domain analysis</sub>"
        layout = get_professional_layout(
            title=professional_title,
            height=600,
            showlegend=True,
            legend_position='upper_right'
        )
        fig.update_layout(**layout)
        
        # Apply professional axis styling
        apply_professional_axis_styling(fig, rows=1, cols=1)
        
        # Update specific axis configuration
        fig.update_xaxes(
            title_text="Frequency (Cycles per Day)",
            type="log",
            range=[-2, 1.5]  # Show from 0.01 to ~30 cycles/day
        )
        fig.update_yaxes(
            title_text="Power Spectral Density (dB)"
        )
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_power_spectrum_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Store statistics for this settlement point using the computed results
        point_stats = {
            "Settlement_Point": point,
            "Data_Points": spectrum_results["n_samples"],
            "Sampling_Freq_CyclesPerDay": spectrum_results["sampling_freq_per_day"],
            "DC_Power_dB": spectrum_results["dc_power_db"],
            "Peak_Count": len(spectrum_results["peak_indices"])
        }
        
        # Add top 3 peaks using the computed results
        for j in range(min(3, len(spectrum_results["peak_frequencies"]))):
            point_stats[f"Peak_{j+1}_Freq_CyclesPerDay"] = spectrum_results["peak_frequencies"][j]
            point_stats[f"Peak_{j+1}_Period_Hours"] = spectrum_results["peak_periods"][j]
            point_stats[f"Peak_{j+1}_Power_dB"] = spectrum_results["peak_powers"][j]
        
        # Save individual power spectrum statistics
        spectrum_stats_df = pd.DataFrame([point_stats])
        stats_path = output_dir / f"dart_slt_power_spectrum_stats_{safe_filename}.csv"
        spectrum_stats_df.to_csv(stats_path, index=False)
        print(f"  Statistics saved to: {stats_path}")
    
    print(f"Power spectrum analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_power_spectrum_bimodal(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create bimodal power spectrum analysis of signed log transformed DART time series for each settlement point.
    
    Performs separate Fast Fourier Transform (FFT) analysis on positive and negative DART_SLT values
    to identify different periodic patterns in each mode of the bimodal distribution.
    
    Creates separate power spectrum plots for each settlement point with:
    - Upper plot: Power spectrum of positive DART_SLT values
    - Lower plot: Power spectrum of absolute values of negative DART_SLT values
    
    This helps understand if positive and negative price differences have different
    underlying dynamics or periodicities.
    
    Args:
        df: DataFrame containing dart_slt column, timestamp data, location, and location_type
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Verify required columns exist
    required_cols = ["dart_slt", "utc_ts", "location", "location_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create unique identifier combining location and type
    df["point_identifier"] = df.apply(
        lambda row: f"{row['location']} ({row['location_type']})",
        axis=1
    )
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating bimodal power spectrum analysis for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point].copy()
        
        # Sort by timestamp and remove NaN values
        point_clean = point_data.dropna(subset=["dart_slt", "utc_ts"]).copy()
        point_clean = point_clean.sort_values("utc_ts").reset_index(drop=True)
        
        if len(point_clean) < 24:  # Need at least a day's worth of data
            print(f"Warning: Insufficient data for {point} ({len(point_clean)} points), skipping")
            continue
        
        # Separate positive and negative values
        positive_data = point_clean[point_clean["dart_slt"] > 0].copy()
        negative_data = point_clean[point_clean["dart_slt"] < 0].copy()
        
        # Take absolute values of negative data for power spectrum analysis
        if len(negative_data) > 0:
            negative_data = negative_data.copy()
            negative_data["dart_slt_abs"] = negative_data["dart_slt"].abs()
        
        # Check if we have sufficient data for both modes
        if len(positive_data) < 24 and len(negative_data) < 24:
            print(f"Warning: Insufficient data for both modes for {point}, skipping")
            continue
        
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(positive_data)} positive values, {len(negative_data)} negative values")
        
        # Create subplot with 2 rows, 1 column
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.25,  # Increased from 0.15 to 0.25 for more room
            subplot_titles=[
                "Positive DART_SLT Power Spectrum",  # Shortened title - point name will be in main title
                "Negative DART_SLT Power Spectrum (Absolute Values)"  # Shortened title
            ]
        )

        # Store statistics for this settlement point
        bimodal_spectrum_stats = []
        
        # Process both datasets
        datasets = [
            (positive_data, "dart_slt", "Positive DART_SLT", "blue", 1),
            (negative_data, "dart_slt_abs", "Negative DART_SLT (Abs)", "red", 2)
        ]
        
        for data, value_col, name, color, row in datasets:
            if len(data) < 24:  # Skip if insufficient data
                print(f"Warning: Insufficient data for {point} {name} ({len(data)} points)")
                continue
                
            try:
                # Compute power spectrum using the utility function
                spectrum_results = compute_power_spectrum(
                    time_series=data[value_col].values,
                    timestamps=data["utc_ts"],
                    peak_percentile=85
                )
                
                print(f"  {name} sampling frequency: {spectrum_results['sampling_freq_per_day']:.1f} cycles/day")
                
            except ValueError as e:
                print(f"Warning: Power spectrum computation failed for {point} {name}: {e}")
                continue
            
            # Add power spectrum trace
            fig.add_trace(
                go.Scatter(
                    x=spectrum_results["freq_bins_per_day"],
                    y=spectrum_results["power_spectrum_db"],
                    mode="lines",
                    name=f"{name}",
                    line=dict(color=SEMANTIC_COLORS['positive'] if row == 1 else SEMANTIC_COLORS['negative'], width=2),
                    showlegend=True
                ),
                row=row, col=1
            )
            
            # Store statistics for this mode using the computed results
            mode_stats = {
                "Settlement_Point": point,
                "Mode": name,
                "Data_Points": spectrum_results["n_samples"],
                "Sampling_Freq_CyclesPerDay": spectrum_results["sampling_freq_per_day"],
                "DC_Power_dB": spectrum_results["dc_power_db"],
                "Peak_Count": len(spectrum_results["peak_indices"])
            }
            
            # Add top 3 peaks using the computed results
            for j in range(min(3, len(spectrum_results["peak_frequencies"]))):
                mode_stats[f"Peak_{j+1}_Freq_CyclesPerDay"] = spectrum_results["peak_frequencies"][j]
                mode_stats[f"Peak_{j+1}_Period_Hours"] = spectrum_results["peak_periods"][j]
                mode_stats[f"Peak_{j+1}_Power_dB"] = spectrum_results["peak_powers"][j]
            
            bimodal_spectrum_stats.append(mode_stats)
        
        # Update layout
        professional_title = f"DART Signed-Log Transform Bimodal Power Spectrum Analysis - {point}{title_suffix}<br><sub style='color:{PROFESSIONAL_COLORS['text']}'>Separate frequency analysis for positive and negative values</sub>"
        layout = get_professional_layout(
            title=professional_title,
            height=900,  # Increased from 800 to 900 for better spacing
            showlegend=True
        )
        fig.update_layout(**layout)
        
        # Update axis labels
        fig.update_yaxes(title_text="Power Spectral Density (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Power Spectral Density (dB)", row=2, col=1)
        fig.update_xaxes(title_text="Frequency (Cycles per Day)", row=2, col=1)
        
        # Set log scale for both x-axes
        fig.update_xaxes(
            type="log",
            range=[-2, 1.5],  # Show from 0.01 to ~30 cycles/day
            row=1, col=1
        )
        fig.update_xaxes(
            type="log",
            range=[-2, 1.5],  # Show from 0.01 to ~30 cycles/day
            row=2, col=1
        )
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_power_spectrum_bimodal_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save individual bimodal power spectrum statistics
        if bimodal_spectrum_stats:
            bimodal_spectrum_stats_df = pd.DataFrame(bimodal_spectrum_stats)
            stats_path = output_dir / f"dart_slt_power_spectrum_bimodal_stats_{safe_filename}.csv"
            bimodal_spectrum_stats_df.to_csv(stats_path, index=False)
            print(f"  Statistics saved to: {stats_path}")
    
    print(f"Bimodal power spectrum analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_sign_power_spectrum(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create power spectrum analysis of DART signed log transform sign sequence for each settlement point.
    
    Converts the DART_SLT time series to a sign sequence where positive values become +1.0
    and negative values become -1.0, then performs FFT analysis to identify periodic patterns
    in the timing of when price differences switch from positive to negative.
    
    This reveals market timing patterns such as daily cycles of when real-time prices
    tend to be above vs below day-ahead prices.
    
    Creates separate plots for each settlement point showing:
    - Power spectrum magnitude in dB vs frequency for the sign sequence
    - Frequency axis in cycles per day for business interpretation
    
    Args:
        df: DataFrame containing dart_slt column, timestamp data, location, and location_type
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Verify required columns exist
    required_cols = ["dart_slt", "utc_ts", "location", "location_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create unique identifier combining location and type
    df["point_identifier"] = df.apply(
        lambda row: f"{row['location']} ({row['location_type']})",
        axis=1
    )
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating sign sequence power spectrum analysis for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point].copy()
        
        # Sort by timestamp and remove NaN values
        point_clean = point_data.dropna(subset=["dart_slt", "utc_ts"]).copy()
        point_clean = point_clean.sort_values("utc_ts").reset_index(drop=True)
        
        if len(point_clean) < 24:  # Need at least a day's worth of data
            print(f"Warning: Insufficient data for {point} ({len(point_clean)} points), skipping")
            continue
        
        # Create sign sequence: +1.0 for positive, -1.0 for negative
        point_clean["dart_slt_sign"] = point_clean["dart_slt"].apply(
            lambda x: 1.0 if x > 0 else -1.0 if x < 0 else 0.0
        )
        
        # Remove any zero values (if DART_SLT was exactly zero)
        point_clean = point_clean[point_clean["dart_slt_sign"] != 0.0]
        
        if len(point_clean) < 24:  # Recheck after removing zeros
            print(f"Warning: Insufficient non-zero data for {point} ({len(point_clean)} points), skipping")
            continue
        
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(point_clean)} sign sequence data points")
        
        try:
            # Compute power spectrum using the utility function on the sign sequence
            spectrum_results = compute_power_spectrum(
                time_series=point_clean["dart_slt_sign"].values,
                timestamps=point_clean["utc_ts"],
                peak_percentile=85
            )
            
            print(f"  Sampling frequency: {spectrum_results['sampling_freq_per_day']:.1f} cycles/day")
            
        except ValueError as e:
            print(f"Warning: Sign sequence power spectrum computation failed for {point}: {e}")
            continue
        
        # Create the plot
        fig = go.Figure()
        
        # Add power spectrum trace
        fig.add_trace(
            go.Scatter(
                x=spectrum_results["freq_bins_per_day"],
                y=spectrum_results["power_spectrum_db"],
                mode="lines",
                name=f"{point} Sign Sequence",
                line=dict(color="green", width=1.5),
                showlegend=True
            )
        )
        
        # Update layout
        fig.update_layout(
            title={
                "text": f"DART Signed-Log Transform Sign Sequence Power Spectrum - {point}{title_suffix}<br><sub>Frequency analysis of positive/negative timing patterns</sub>",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis_title="Frequency (Cycles per Day)",
            yaxis_title="Power Spectral Density (dB)",
            height=600,
            showlegend=True,
            xaxis=dict(
                type="log",
                range=[-2, 1.5],  # Show from 0.01 to ~30 cycles/day
                title="Frequency (Cycles per Day)"
            ),
            yaxis=dict(
                title="Power Spectral Density (dB)"
            )
        )
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_sign_power_spectrum_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Store statistics for this settlement point using the computed results
        point_stats = {
            "Settlement_Point": point,
            "Data_Points": spectrum_results["n_samples"],
            "Positive_Count": int((point_clean["dart_slt_sign"] > 0).sum()),
            "Negative_Count": int((point_clean["dart_slt_sign"] < 0).sum()),
            "Positive_Proportion": float((point_clean["dart_slt_sign"] > 0).mean()),
            "Sampling_Freq_CyclesPerDay": spectrum_results["sampling_freq_per_day"],
            "DC_Power_dB": spectrum_results["dc_power_db"],
            "Peak_Count": len(spectrum_results["peak_indices"])
        }
        
        # Add top 3 peaks using the computed results
        for j in range(min(3, len(spectrum_results["peak_frequencies"]))):
            point_stats[f"Peak_{j+1}_Freq_CyclesPerDay"] = spectrum_results["peak_frequencies"][j]
            point_stats[f"Peak_{j+1}_Period_Hours"] = spectrum_results["peak_periods"][j]
            point_stats[f"Peak_{j+1}_Power_dB"] = spectrum_results["peak_powers"][j]
        
        # Save individual sign sequence power spectrum statistics
        spectrum_stats_df = pd.DataFrame([point_stats])
        stats_path = output_dir / f"dart_slt_sign_power_spectrum_stats_{safe_filename}.csv"
        spectrum_stats_df.to_csv(stats_path, index=False)
        print(f"  Statistics saved to: {stats_path}")
    
    print(f"Sign sequence power spectrum analysis complete: {len(unique_points)} settlement points processed") 


def plot_dart_slt_sign_daily_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create daily cycle heatmap of DART signed log transform positivity rates for each settlement point.
    
    Creates heatmaps showing the proportion of positive DART values by hour of day and day of week.
    This reveals daily and weekly patterns in when real-time prices tend to be above vs below
    day-ahead prices.
    
    Construction approach:
    1. Convert DART_SLT to binary: 1 if positive, 0 if negative
    2. Group by hour of day (1-24) and day of week
    3. Compute positivity rate (proportion of 1s) for each cell
    4. Visualize as heatmap with diverging color scale
    
    Creates separate plots for each settlement point showing:
    - X-axis: Day of week (Monday-Sunday)
    - Y-axis: Hour of day (1-24, ending hour)
    - Color: Positivity rate (0=always negative, 1=always positive, 0.5=balanced)
    
    Args:
        df: DataFrame containing dart_slt column, time columns, location, and location_type
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Verify required columns exist
    required_cols = ["dart_slt", "day_of_week", "end_of_hour", "location", "location_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create unique identifier combining location and type
    df["point_identifier"] = df.apply(
        lambda row: f"{row['location']} ({row['location_type']})",
        axis=1
    )
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating daily cycle heatmaps for {len(unique_points)} settlement points")
    
    # Create day of week mapping for better labels
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
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point].copy()
        
        # Remove NaN values
        point_clean = point_data.dropna(subset=["dart_slt", "weekday", "end_of_hour"]).copy()
        
        if len(point_clean) < 24:  # Need sufficient data
            print(f"Warning: Insufficient data for {point} ({len(point_clean)} points), skipping")
            continue
        
        # Create binary time series: 1 if positive, 0 if negative/zero
        point_clean["dart_positive"] = (point_clean["dart_slt"] > 0).astype(int)
        
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(point_clean)} data points")
        
        # Group by hour and day of week, compute positivity rate
        heatmap_data = point_clean.groupby(["end_of_hour", "weekday"])["dart_positive"].agg([
            'mean',  # positivity rate
            'count', # sample size
            'sum'    # positive count
        ]).reset_index()
        
        # Rename columns for clarity
        heatmap_data.columns = ["hour", "weekday", "positivity_rate", "sample_count", "positive_count"]
        
        # Create pivot table for heatmap
        pivot_data = heatmap_data.pivot(index="hour", columns="weekday", values="positivity_rate")
        
        # Reorder columns to standard week order and ensure all hours are present
        pivot_data = pivot_data.reindex(columns=day_order, fill_value=None)
        all_hours = list(range(1, 25))  # 1 to 24 (ending hours)
        pivot_data = pivot_data.reindex(index=all_hours, fill_value=None)
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdBu',  # Red-Blue diverging scale (Red=negative dominant, Blue=positive dominant)
            zmid=0.5,  # Center the colorscale at 0.5 (balanced)
            zmin=0,
            zmax=1,
            hoverongaps=False,
            hovertemplate='<b>%{y}:00 %{x}</b><br>Positivity Rate: %{z:.2f}<br><extra></extra>',
            colorbar=dict(
                title="Positivity Rate",
                tickmode="linear",
                tick0=0,
                dtick=0.1,
                tickformat=".1f"
            )
        ))
        
        # Update layout
        fig.update_layout(
            title={
                "text": f"DART Daily Cycle Heatmap - {point}{title_suffix}<br><sub>Positivity rate by hour and day of week (Blue=mostly positive, Red=mostly negative)</sub>",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis_title="Day of Week",
            yaxis_title="Hour of Day (Ending Hour)",
            height=600,
            width=800,
            xaxis=dict(
                tickmode="array",
                tickvals=day_order,
                ticktext=day_order
            ),
            yaxis=dict(
                tickmode="linear",
                tick0=1,
                dtick=1,
                autorange="reversed"  # Hour 1 at top, 24 at bottom
            )
        )
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_sign_daily_heatmap_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save detailed statistics
        heatmap_stats = heatmap_data.copy()
        heatmap_stats["Settlement_Point"] = point
        heatmap_stats["negative_count"] = heatmap_stats["sample_count"] - heatmap_stats["positive_count"]
        heatmap_stats["negative_rate"] = 1 - heatmap_stats["positivity_rate"]
        
        # Reorder columns for better readability
        heatmap_stats = heatmap_stats[[
            "Settlement_Point", "weekday", "hour", "sample_count", 
            "positive_count", "negative_count", "positivity_rate", "negative_rate"
        ]]
        
        stats_path = output_dir / f"dart_slt_sign_daily_heatmap_stats_{safe_filename}.csv"
        heatmap_stats.to_csv(stats_path, index=False)
        print(f"  Statistics saved to: {stats_path}")
    
    print(f"Daily cycle heatmap analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_spectrogram(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create spectrogram analysis of signed log transformed DART time series for each settlement point.
    
    Performs Short-Time Fourier Transform (STFT) to show how the frequency content of DART_SLT
    changes over time. This reveals if periodic patterns are stable or evolving.
    
    Creates separate spectrogram plots for each settlement point showing:
    - X-axis: Time
    - Y-axis: Frequency (cycles per day)  
    - Color: Power spectral density (dB)
    - Window analysis to capture temporal evolution of spectral content
    
    Args:
        df: DataFrame containing dart_slt column, timestamp data, location, and location_type
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Verify required columns exist
    required_cols = ["dart_slt", "utc_ts", "location", "location_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create unique identifier combining location and type
    df["point_identifier"] = df.apply(
        lambda row: f"{row['location']} ({row['location_type']})",
        axis=1
    )
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating spectrogram analysis for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point].copy()
        
        # Sort by timestamp and remove NaN values
        point_clean = point_data.dropna(subset=["dart_slt", "utc_ts"]).copy()
        point_clean = point_clean.sort_values("utc_ts").reset_index(drop=True)
        
        if len(point_clean) < 168:  # Need at least a week's worth of data for meaningful spectrogram
            print(f"Warning: Insufficient data for {point} ({len(point_clean)} points), skipping")
            continue
        
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(point_clean)} data points")
        
        try:
            # Calculate sampling parameters
            timestamps = pd.to_datetime(point_clean["utc_ts"])
            time_diffs = timestamps.diff().dropna()
            median_dt = time_diffs.median()
            sampling_freq_hz = 1 / median_dt.total_seconds()
            
            # Perform STFT with appropriate window size
            window_size = min(168, len(point_clean) // 4)  # ~1 week or 1/4 of data
            overlap = window_size // 2  # 50% overlap
            
            from scipy.signal import spectrogram
            frequencies, times_stft, Sxx = spectrogram(
                point_clean["dart_slt"].values,
                fs=sampling_freq_hz,
                window='hann',
                nperseg=window_size,
                noverlap=overlap,
                scaling='density'
            )
            
            # Convert to dB and cycles per day
            Sxx_db = 10 * np.log10(Sxx + 1e-12)
            frequencies_cpd = frequencies * 86400  # Convert Hz to cycles per day
            
            # Convert time indices to actual timestamps for x-axis
            time_indices = times_stft * len(point_clean) / (len(point_clean) / sampling_freq_hz)
            time_labels = [timestamps.iloc[int(min(i, len(timestamps) - 1))] for i in time_indices]
            
        except Exception as e:
            print(f"Warning: Spectrogram computation failed for {point}: {e}")
            continue
        
        # Create the spectrogram plot
        fig = go.Figure(data=go.Heatmap(
            z=Sxx_db,
            x=time_labels,
            y=frequencies_cpd,
            colorscale='Viridis',
            hovertemplate='<b>Time: %{x}</b><br>Frequency: %{y:.2f} cycles/day<br>Power: %{z:.1f} dB<br><extra></extra>',
            colorbar=dict(
                title="Power Spectral Density (dB)"
            )
        ))
        
        # Update layout
        fig.update_layout(
            title={
                "text": f"DART Signed-Log Transform Spectrogram - {point}{title_suffix}<br><sub>Time-frequency analysis showing spectral evolution</sub>",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis_title="Time",
            yaxis_title="Frequency (Cycles per Day)",
            height=600,
            yaxis=dict(
                range=[0, min(10, frequencies_cpd.max())]  # Focus on 0-10 cycles/day
            )
        )
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_spectrogram_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save spectrogram statistics
        spectrogram_stats = [{
            "Settlement_Point": point,
            "Data_Points": len(point_clean),
            "Window_Size": window_size,
            "Overlap": overlap,
            "Time_Windows": len(times_stft),
            "Frequency_Bins": len(frequencies_cpd),
            "Max_Frequency_CPD": frequencies_cpd.max(),
            "Sampling_Freq_Hz": sampling_freq_hz,
            "Mean_Power_dB": Sxx_db.mean(),
            "Max_Power_dB": Sxx_db.max()
        }]
        
        spectrogram_stats_df = pd.DataFrame(spectrogram_stats)
        stats_path = output_dir / f"dart_slt_spectrogram_stats_{safe_filename}.csv"
        spectrogram_stats_df.to_csv(stats_path, index=False)
        print(f"  Statistics saved to: {stats_path}")
    
    print(f"Spectrogram analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_moving_window_stats(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    window_hours: int = 168  # Default to 1 week window
) -> None:
    """Create moving window statistics analysis of signed log transformed DART for each settlement point.
    
    Computes rolling statistics over time windows to reveal temporal structure and non-stationarity
    in the DART_SLT time series. This helps identify if statistical properties change over time.
    
    Creates separate plots for each settlement point with multiple subplots showing:
    - Rolling mean and standard deviation
    - Rolling skewness and kurtosis
    - Rolling percentage of positive values
    - Rolling autocorrelation at lag 24 (daily pattern strength)
    
    Args:
        df: DataFrame containing dart_slt column, timestamp data, location, and location_type
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
        window_hours: Size of rolling window in hours (default: 168 = 1 week)
    """
    # Prepare data
    df = df.copy()
    
    # Verify required columns exist
    required_cols = ["dart_slt", "utc_ts", "location", "location_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create unique identifier combining location and type
    df["point_identifier"] = df.apply(
        lambda row: f"{row['location']} ({row['location_type']})",
        axis=1
    )
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating moving window statistics analysis for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point].copy()
        
        # Sort by timestamp and remove NaN values
        point_clean = point_data.dropna(subset=["dart_slt", "utc_ts"]).copy()
        point_clean = point_clean.sort_values("utc_ts").reset_index(drop=True)
        
        if len(point_clean) < window_hours * 2:  # Need at least 2 windows worth of data
            print(f"Warning: Insufficient data for {point} ({len(point_clean)} points), skipping")
            continue
        
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(point_clean)} data points with {window_hours}h window")
        
        # Convert to pandas datetime index for rolling operations
        point_clean = point_clean.set_index('utc_ts')
        point_clean.index = pd.to_datetime(point_clean.index)
        
        # Compute rolling statistics
        window_str = f"{window_hours}h"
        
        rolling_mean = point_clean["dart_slt"].rolling(window_str, min_periods=24).mean()
        rolling_std = point_clean["dart_slt"].rolling(window_str, min_periods=24).std()
        rolling_skew = point_clean["dart_slt"].rolling(window_str, min_periods=24).skew()
        rolling_kurt = point_clean["dart_slt"].rolling(window_str, min_periods=24).apply(lambda x: x.kurtosis())
        rolling_positive_pct = point_clean["dart_slt"].rolling(window_str, min_periods=24).apply(lambda x: (x > 0).mean() * 100)
        
        # Compute rolling autocorrelation at lag 24 (daily pattern)
        def rolling_autocorr_24(series):
            if len(series) < 48:  # Need at least 2 days
                return np.nan
            return series.autocorr(lag=24)
        
        rolling_autocorr = point_clean["dart_slt"].rolling(window_str, min_periods=48).apply(rolling_autocorr_24)
        
        # Create subplot with 2x2 grid
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.15,
            horizontal_spacing=0.15,
            subplot_titles=[
                f"Rolling Mean ± Std Dev ({window_hours}h window)",
                f"Rolling Skewness & Kurtosis ({window_hours}h window)", 
                f"Rolling Positive Rate % ({window_hours}h window)",
                f"Rolling Daily Autocorrelation ({window_hours}h window)"
            ]
        )
        
        # Plot 1: Rolling mean and std
        fig.add_trace(
            go.Scatter(
                x=rolling_mean.index,
                y=rolling_mean.values,
                mode="lines",
                name="Rolling Mean",
                line=dict(color="blue", width=2),
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_std.index,
                y=rolling_std.values,
                mode="lines",
                name="Rolling Std Dev",
                line=dict(color="red", width=2),
                yaxis="y2",
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Plot 2: Rolling skewness and kurtosis
        fig.add_trace(
            go.Scatter(
                x=rolling_skew.index,
                y=rolling_skew.values,
                mode="lines",
                name="Rolling Skewness",
                line=dict(color="green", width=2),
                showlegend=True
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_kurt.index,
                y=rolling_kurt.values,
                mode="lines",
                name="Rolling Kurtosis",
                line=dict(color="orange", width=2),
                yaxis="y4",
                showlegend=True
            ),
            row=1, col=2
        )
        
        # Plot 3: Rolling positive percentage
        fig.add_trace(
            go.Scatter(
                x=rolling_positive_pct.index,
                y=rolling_positive_pct.values,
                mode="lines",
                name="Positive Rate %",
                line=dict(color="purple", width=2),
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Add 50% reference line
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Plot 4: Rolling autocorrelation
        fig.add_trace(
            go.Scatter(
                x=rolling_autocorr.index,
                y=rolling_autocorr.values,
                mode="lines",
                name="Daily Autocorr (lag=24h)",
                line=dict(color="brown", width=2),
                showlegend=True
            ),
            row=2, col=2
        )
        
        # Add 0 reference line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                "text": f"DART Signed-Log Transform Moving Window Statistics - {point}{title_suffix}<br><sub>Rolling statistics analysis with {window_hours}h window</sub>",
                "x": 0.5,
                "xanchor": "center"
            },
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_yaxes(title_text="DART_SLT", row=1, col=1)
        fig.update_yaxes(title_text="Skewness", row=1, col=2)
        fig.update_yaxes(title_text="Positive Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="Autocorrelation", row=2, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_moving_window_stats_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save moving window statistics
        window_stats = pd.DataFrame({
            "timestamp": rolling_mean.index,
            "rolling_mean": rolling_mean.values,
            "rolling_std": rolling_std.values,
            "rolling_skewness": rolling_skew.values,
            "rolling_kurtosis": rolling_kurt.values,
            "rolling_positive_pct": rolling_positive_pct.values,
            "rolling_daily_autocorr": rolling_autocorr.values
        })
        
        window_stats["Settlement_Point"] = point
        window_stats = window_stats.dropna()  # Remove NaN values from beginning
        
        stats_path = output_dir / f"dart_slt_moving_window_stats_{safe_filename}.csv"
        window_stats.to_csv(stats_path, index=False)
        print(f"  Statistics saved to: {stats_path}")
    
    print(f"Moving window statistics analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_sign_transitions(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """Create sign transition analysis of signed log transformed DART for each settlement point.
    
    Analyzes the patterns in transitions between positive and negative DART_SLT values
    to understand market switching behavior and persistence patterns.
    
    Creates separate plots for each settlement point with multiple subplots showing:
    - Transition matrix heatmap (positive->positive, positive->negative, etc.)
    - Run length distributions (how long positive/negative streaks last)
    - Transition timing analysis by hour of day
    - Transition probability by day of week
    
    Args:
        df: DataFrame containing dart_slt column, time columns, location, and location_type
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
    """
    # Prepare data
    df = df.copy()
    
    # Verify required columns exist
    required_cols = ["dart_slt", "day_of_week", "end_of_hour", "location", "location_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create unique identifier combining location and type
    df["point_identifier"] = df.apply(
        lambda row: f"{row['location']} ({row['location_type']})",
        axis=1
    )
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating sign transition analysis for {len(unique_points)} settlement points")
    
    # Create day mapping
    day_mapping = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    df["weekday"] = df["day_of_week"].map(day_mapping)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point].copy()
        
        # Remove NaN values and sort by time
        point_clean = point_data.dropna(subset=["dart_slt", "weekday", "end_of_hour"]).copy()
        point_clean = point_clean.sort_values("utc_ts").reset_index(drop=True)
        
        if len(point_clean) < 48:  # Need at least 2 days of data
            print(f"Warning: Insufficient data for {point} ({len(point_clean)} points), skipping")
            continue
        
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(point_clean)} data points")
        
        # Create sign sequence
        point_clean["dart_sign"] = np.where(point_clean["dart_slt"] > 0, 1, -1)
        
        # Analyze transitions
        transitions = []
        run_lengths_pos = []
        run_lengths_neg = []
        current_run_length = 1
        current_sign = point_clean["dart_sign"].iloc[0]
        
        for i in range(1, len(point_clean)):
            prev_sign = point_clean["dart_sign"].iloc[i-1]
            curr_sign = point_clean["dart_sign"].iloc[i]
            
            # Record transition
            transitions.append({
                "from_state": "Positive" if prev_sign == 1 else "Negative",
                "to_state": "Positive" if curr_sign == 1 else "Negative",
                "hour": point_clean["end_of_hour"].iloc[i],
                "weekday": point_clean["weekday"].iloc[i]
            })
            
            # Track run lengths
            if curr_sign == current_sign:
                current_run_length += 1
            else:
                # End of run
                if current_sign == 1:
                    run_lengths_pos.append(current_run_length)
                else:
                    run_lengths_neg.append(current_run_length)
                current_run_length = 1
                current_sign = curr_sign
        
        # Add final run
        if current_sign == 1:
            run_lengths_pos.append(current_run_length)
        else:
            run_lengths_neg.append(current_run_length)
        
        transitions_df = pd.DataFrame(transitions)
        
        # Create 2x2 subplot
        fig = make_subplots(
            rows=2, cols=2,
            vertical_spacing=0.2,
            horizontal_spacing=0.1,  # Reduce horizontal spacing since we have more width
            subplot_titles=[
                "State Transition Probabilities<br><sub>Positive ↔ Negative DART persistence</sub>",
                "Run Length Distributions<br><sub>Duration of consecutive positive/negative periods</sub>",
                "Sign Changes by Hour of Day<br><sub>When do positive/negative switches occur?</sub>", 
                "Sign Changes by Day of Week<br><sub>Are switches more common on certain days?</sub>"
            ]
        )
        
        # Plot 1: Transition matrix
        trans_matrix = transitions_df.groupby(["from_state", "to_state"]).size().unstack(fill_value=0)
        trans_matrix_norm = trans_matrix.div(trans_matrix.sum(axis=1), axis=0)  # Normalize by row
        
        fig.add_trace(
            go.Heatmap(
                z=trans_matrix_norm.values,
                x=trans_matrix_norm.columns,
                y=trans_matrix_norm.index,
                colorscale="Blues",
                hovertemplate="From: %{y}<br>To: %{x}<br>Probability: %{z:.3f}<extra></extra>",
                showscale=False,  # Remove the colorbar
                text=trans_matrix_norm.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 14, "color": "white"},
                showlegend=False  # Turn off legend since it has its own colorbar
            ),
            row=1, col=1
        )
        
        # Plot 2: Run length distributions
        max_length = max(max(run_lengths_pos) if run_lengths_pos else 0,
                        max(run_lengths_neg) if run_lengths_neg else 0)
        
        bins = list(range(1, min(max_length + 2, 25)))  # Cap at 24 hours
        
        if run_lengths_pos:
            fig.add_trace(
                go.Histogram(
                    x=run_lengths_pos,
                    name="Positive Runs",
                    xbins=dict(start=0.5, end=max(bins)+0.5, size=1),
                    marker_color="blue",
                    opacity=0.7,
                    legendgroup="runs",
                    showlegend=True
                ),
                row=1, col=2
            )
        
        if run_lengths_neg:
            fig.add_trace(
                go.Histogram(
                    x=run_lengths_neg,
                    name="Negative Runs", 
                    xbins=dict(start=0.5, end=max(bins)+0.5, size=1),
                    marker_color="red",
                    opacity=0.7,
                    legendgroup="runs",
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # Plot 3: Transitions by hour (only actual state changes)
        hour_transitions = transitions_df[transitions_df["from_state"] != transitions_df["to_state"]]
        hour_counts = hour_transitions.groupby("hour").size().reindex(range(1, 25), fill_value=0)
        
        fig.add_trace(
            go.Bar(
                x=hour_counts.index,
                y=hour_counts.values,
                name="Sign Changes",
                marker_color="green",
                hovertemplate="Hour %{x}: %{y} sign changes<extra></extra>",
                showlegend=False  # Turn off legend for this bar chart
            ),
            row=2, col=1
        )
        
        # Plot 4: Transitions by weekday (only actual state changes)
        weekday_transitions = hour_transitions.groupby("weekday").size().reindex(
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fill_value=0
        )
        
        fig.add_trace(
            go.Bar(
                x=weekday_transitions.index,
                y=weekday_transitions.values,
                name="Sign Changes",
                marker_color="orange",
                hovertemplate="%{x}: %{y} sign changes<extra></extra>",
                showlegend=False  # Turn off legend for this bar chart
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                "text": f"DART Signed-Log Transform Sign Transition Analysis - {point}{title_suffix}<br><sub>Analysis of positive/negative switching patterns</sub>",
                "x": 0.5,
                "xanchor": "center"
            },
            height=900,  # Increased from 800
            width=1400,  # Increased from 1200 
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.95,
                xanchor="right", 
                x=0.98,  # Position in upper right of the run length subplot
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1
            )
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Run Length (Consecutive Hours)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency Count", row=1, col=2)
        fig.update_xaxes(title_text="Hour of Day (1-24)", row=2, col=1)
        fig.update_yaxes(title_text="Number of Sign Changes", row=2, col=1)
        fig.update_xaxes(title_text="Day of Week", row=2, col=2)
        fig.update_yaxes(title_text="Number of Sign Changes", row=2, col=2)
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_sign_transitions_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save transition statistics
        transition_stats = [{
            "Settlement_Point": point,
            "Total_Transitions": len(transitions_df[transitions_df["from_state"] != transitions_df["to_state"]]),
            "Pos_to_Neg_Transitions": len(transitions_df[(transitions_df["from_state"] == "Positive") & 
                                                        (transitions_df["to_state"] == "Negative")]),
            "Neg_to_Pos_Transitions": len(transitions_df[(transitions_df["from_state"] == "Negative") & 
                                                        (transitions_df["to_state"] == "Positive")]),
            "Pos_to_Pos_Persistence": len(transitions_df[(transitions_df["from_state"] == "Positive") & 
                                                        (transitions_df["to_state"] == "Positive")]),
            "Neg_to_Neg_Persistence": len(transitions_df[(transitions_df["from_state"] == "Negative") & 
                                                        (transitions_df["to_state"] == "Negative")]),
            "Mean_Positive_Run_Length": np.mean(run_lengths_pos) if run_lengths_pos else 0,
            "Mean_Negative_Run_Length": np.mean(run_lengths_neg) if run_lengths_neg else 0,
            "Max_Positive_Run_Length": max(run_lengths_pos) if run_lengths_pos else 0,
            "Max_Negative_Run_Length": max(run_lengths_neg) if run_lengths_neg else 0
        }]
        
        transition_stats_df = pd.DataFrame(transition_stats)
        stats_path = output_dir / f"dart_slt_sign_transitions_stats_{safe_filename}.csv"
        transition_stats_df.to_csv(stats_path, index=False)
        print(f"  Statistics saved to: {stats_path}")
    
    print(f"Sign transition analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_kmeans_unimodal(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    max_k: int = 10
) -> None:
    """Create K-means clustering analysis of signed log transformed DART for each settlement point (unimodal).
    
    Performs K-means clustering on the entire DART_SLT distribution to identify natural groupings
    in the data. Creates plots showing the elbow method for optimal K selection and
    the resulting cluster centers overlaid on the data distribution.
    
    Creates separate plots for each settlement point with:
    - Left subplot: Elbow curve (inertia vs K) for optimal K selection
    - Right subplot: Histogram of DART_SLT with vertical lines at cluster centers
    
    Args:
        df: DataFrame containing dart_slt column, location, and location_type
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
        max_k: Maximum number of clusters to evaluate (default: 10)
    """
    # Define professional color palette (colorblind-friendly)
    colors = {
        'primary': '#2E86AB',      # Professional blue
        'secondary': '#A23B72',    # Deep magenta 
        'accent': '#F18F01',       # Warm orange
        'neutral': '#C73E1D',      # Deep red
        'success': '#4A7A8A',      # Teal
        'background': '#F5F5F5',   # Light gray
        'text': '#2C3E50',         # Dark blue-gray
        'histogram': 'rgba(46, 134, 171, 0.6)',  # Semi-transparent primary
    }
    
    # Prepare data
    df = df.copy()
    
    # Create point_identifier if not present
    if "point_identifier" not in df.columns:
        df["point_identifier"] = df.apply(
            lambda row: f"{row['location']} ({row['location_type']})",
            axis=1
        )
    
    # Verify dart_slt column exists
    if "dart_slt" not in df.columns:
        raise ValueError("dart_slt column not found. Ensure dataset.py creates this column.")
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating K-means unimodal analysis for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        
        # Remove NaN values
        dart_slt_clean = point_data["dart_slt"].dropna()
        
        if len(dart_slt_clean) < max_k * 10:  # Need sufficient data for clustering
            print(f"Warning: Insufficient data for {point} ({len(dart_slt_clean)} points), skipping")
            continue
        
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(dart_slt_clean)} data points with max_k={max_k}")
        
        try:
            # Perform K-means clustering analysis
            kmeans_results = compute_kmeans_clustering(
                time_series=dart_slt_clean.values,
                max_k=max_k,
                random_state=42
            )
            
            print(f"  Optimal K: {kmeans_results['optimal_k']}")
            
        except Exception as e:
            print(f"Warning: K-means clustering failed for {point}: {e}")
            continue
        
        # Create subplot with 1 row, 2 columns
        fig = make_subplots(
            rows=1, cols=2,
            shared_yaxes=False,
            horizontal_spacing=0.15,
            subplot_titles=[
                f"Elbow Method Analysis",
                f"Distribution with K={kmeans_results['optimal_k']} Clusters"
            ]
        )
        
        # Plot 1: Elbow curve (inertia vs K)
        fig.add_trace(
            go.Scatter(
                x=kmeans_results["k_values"],
                y=kmeans_results["inertias"],
                mode="lines+markers",
                name="Inertia",
                line=dict(color=colors['primary'], width=3),
                marker=dict(size=8, color=colors['primary']),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Highlight optimal K
        optimal_idx = kmeans_results["optimal_k"] - 1  # Convert to 0-based index
        fig.add_trace(
            go.Scatter(
                x=[kmeans_results["optimal_k"]],
                y=[kmeans_results["inertias"][optimal_idx]],
                mode="markers",
                name=f"Optimal K={kmeans_results['optimal_k']}",
                marker=dict(
                    color=colors['accent'], 
                    size=14, 
                    symbol="diamond",
                    line=dict(width=2, color=colors['text'])
                ),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Plot 2: Distribution with cluster centers
        fig.add_trace(
            go.Histogram(
                x=dart_slt_clean,
                name="DART_SLT Distribution",
                nbinsx=60,
                histnorm="probability density",
                opacity=0.8,
                marker_color=colors['histogram'],
                marker_line=dict(width=1, color=colors['primary']),
                showlegend=True
            ),
            row=1, col=2
        )
        
        # Add vertical lines for cluster centers with professional styling
        cluster_colors = [colors['secondary'], colors['success'], colors['neutral'], 
                         colors['accent'], '#8B4513', '#556B2F', '#4B0082', '#DC143C']
        
        # Calculate annotation positions to avoid overlap
        y_max = max(np.histogram(dart_slt_clean, bins=60, density=True)[0])
        annotation_heights = [y_max * 0.9, y_max * 0.8, y_max * 0.7, y_max * 0.85, 
                            y_max * 0.75, y_max * 0.65, y_max * 0.95, y_max * 0.6]
        
        for i, center in enumerate(kmeans_results["cluster_centers"]):
            color = cluster_colors[i % len(cluster_colors)]
            fig.add_vline(
                x=center,
                line_dash="dash",
                line_color=color,
                line_width=2.5,
                row=1, col=2
            )
            
            # Add annotation as a scatter point instead of vline annotation to avoid overlap
            fig.add_trace(
                go.Scatter(
                    x=[center],
                    y=[annotation_heights[i % len(annotation_heights)]],
                    mode="markers+text",
                    marker=dict(
                        color=color,
                        size=8,
                        symbol="diamond"
                    ),
                    text=[f"C{i+1}: {center:.2f}"],
                    textposition="top center",
                    textfont=dict(size=10, color=color),
                    showlegend=False,
                    name=""
                ),
                row=1, col=2
            )
        
        # Update layout with professional styling and better spacing
        fig.update_layout(
            title={
                "text": f"K-means Clustering Analysis (Unimodal) - {point}{title_suffix}<br><sub style='color:{colors['text']}'>Elbow method for optimal cluster identification</sub>",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16, "color": colors['text']}
            },
            height=650,  # Increased height to accommodate annotations
            width=1300,  # Increased width for better spacing
            showlegend=True,
            legend=dict(
                x=0.02,  # Position legend in upper left to avoid histogram area
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=colors['text'],
                borderwidth=1,
                font=dict(color=colors['text'], size=11)
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(t=100, b=60, l=60, r=60)  # Add margins for better spacing
        )
        
        # Update axis labels with professional styling
        fig.update_xaxes(
            title_text="Number of Clusters (K)", 
            row=1, col=1,
            title_font=dict(color=colors['text']),
            tickfont=dict(color=colors['text']),
            gridcolor='rgba(128,128,128,0.2)'
        )
        fig.update_yaxes(
            title_text="Inertia (Within-cluster sum of squares)", 
            row=1, col=1,
            title_font=dict(color=colors['text']),
            tickfont=dict(color=colors['text']),
            gridcolor='rgba(128,128,128,0.2)'
        )
        fig.update_xaxes(
            title_text="DART_SLT Value", 
            row=1, col=2,
            title_font=dict(color=colors['text']),
            tickfont=dict(color=colors['text']),
            gridcolor='rgba(128,128,128,0.2)'
        )
        fig.update_yaxes(
            title_text="Probability Density", 
            row=1, col=2,
            title_font=dict(color=colors['text']),
            tickfont=dict(color=colors['text']),
            gridcolor='rgba(128,128,128,0.2)'
        )
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_kmeans_unimodal_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save clustering statistics
        cluster_stats = [{
            "Settlement_Point": point,
            "Data_Points": len(dart_slt_clean),
            "Max_K_Tested": max_k,
            "Optimal_K": kmeans_results["optimal_k"],
            "Optimal_Inertia": kmeans_results["inertias"][optimal_idx],
            "Cluster_Centers": list(kmeans_results["cluster_centers"])
        }]
        
        # Add individual cluster information
        for i, center in enumerate(kmeans_results["cluster_centers"]):
            cluster_mask = kmeans_results["labels"] == i
            cluster_size = np.sum(cluster_mask)
            cluster_stats[0][f"Cluster_{i+1}_Center"] = center
            cluster_stats[0][f"Cluster_{i+1}_Size"] = cluster_size
            cluster_stats[0][f"Cluster_{i+1}_Proportion"] = cluster_size / len(dart_slt_clean)
        
        cluster_stats_df = pd.DataFrame(cluster_stats)
        stats_path = output_dir / f"dart_slt_kmeans_unimodal_stats_{safe_filename}.csv"
        cluster_stats_df.to_csv(stats_path, index=False)
        print(f"  Statistics saved to: {stats_path}")
    
    print(f"K-means unimodal analysis complete: {len(unique_points)} settlement points processed")


def plot_dart_slt_kmeans_bimodal(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    max_k: int = 10
) -> None:
    """Create K-means clustering analysis of signed log transformed DART for each settlement point (bimodal).
    
    Performs separate K-means clustering on positive and negative (absolute value) DART_SLT values
    to identify natural groupings within each mode of the bimodal distribution.
    
    Creates separate plots for each settlement point with:
    - Upper left: Elbow curve for positive DART_SLT values
    - Upper right: Positive DART_SLT histogram with cluster centers
    - Lower left: Elbow curve for negative DART_SLT values (absolute)
    - Lower right: Negative DART_SLT histogram (absolute) with cluster centers
    
    Args:
        df: DataFrame containing dart_slt column, location, and location_type
        output_dir: Directory where plots will be saved
        title_suffix: Optional suffix to add to plot title
        max_k: Maximum number of clusters to evaluate (default: 10)
    """
    # Define professional color palette (colorblind-friendly)
    colors = {
        'primary': '#2E86AB',      # Professional blue
        'secondary': '#A23B72',    # Deep magenta 
        'accent': '#F18F01',       # Warm orange
        'neutral': '#C73E1D',      # Deep red
        'success': '#4A7A8A',      # Teal
        'background': '#F5F5F5',   # Light gray
        'text': '#2C3E50',         # Dark blue-gray
        'positive': 'rgba(46, 134, 171, 0.6)',    # Semi-transparent blue
        'negative': 'rgba(162, 59, 114, 0.6)',    # Semi-transparent magenta
    }
    
    # Prepare data
    df = df.copy()
    
    # Create point_identifier if not present
    if "point_identifier" not in df.columns:
        df["point_identifier"] = df.apply(
            lambda row: f"{row['location']} ({row['location_type']})",
            axis=1
        )
    
    # Verify dart_slt column exists
    if "dart_slt" not in df.columns:
        raise ValueError("dart_slt column not found. Ensure dataset.py creates this column.")
    
    # Get unique settlement points
    unique_points = df["point_identifier"].unique()
    print(f"Creating K-means bimodal analysis for {len(unique_points)} settlement points")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate separate plot for each settlement point
    for point in unique_points:
        point_data = df[df["point_identifier"] == point]
        
        # Separate positive and negative values
        dart_slt_clean = point_data["dart_slt"].dropna()
        positive_values = dart_slt_clean[dart_slt_clean > 0]
        negative_values = dart_slt_clean[dart_slt_clean < 0].abs()  # Take absolute values
        
        if len(positive_values) < max_k * 5 and len(negative_values) < max_k * 5:
            print(f"Warning: Insufficient data for both modes for {point}, skipping")
            continue
        
        # Create safe filename
        safe_filename = point.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        print(f"Processing {point}: {len(positive_values)} positive, {len(negative_values)} negative values")
        
        # Create subplot with 2 rows, 2 columns
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=False,
            vertical_spacing=0.25,
            horizontal_spacing=0.12,
            subplot_titles=[
                "Positive Values: Elbow Analysis",
                "Positive Values: Distribution with Clusters", 
                "Negative Values (Abs): Elbow Analysis",
                "Negative Values (Abs): Distribution with Clusters"
            ]
        )
        
        # Store clustering results for both modes
        bimodal_cluster_stats = []
        
        # Process both datasets
        datasets = [
            (positive_values, "Positive DART_SLT", colors['primary'], colors['positive'], 1),
            (negative_values, "Negative DART_SLT (Abs)", colors['secondary'], colors['negative'], 2)
        ]
        
        for data, name, line_color, hist_color, row in datasets:
            if len(data) < max_k * 5:  # Skip if insufficient data
                print(f"Warning: Insufficient data for {point} {name} ({len(data)} points)")
                continue
            
            try:
                # Perform K-means clustering analysis
                kmeans_results = compute_kmeans_clustering(
                    time_series=data.values,
                    max_k=max_k,
                    random_state=42
                )
                
                print(f"  {name} Optimal K: {kmeans_results['optimal_k']}")
                
            except Exception as e:
                print(f"Warning: K-means clustering failed for {point} {name}: {e}")
                continue
            
            # Plot elbow curve (left column)
            fig.add_trace(
                go.Scatter(
                    x=kmeans_results["k_values"],
                    y=kmeans_results["inertias"],
                    mode="lines+markers",
                    name=f"{name} Inertia",
                    line=dict(color=line_color, width=3),
                    marker=dict(size=6, color=line_color),
                    showlegend=True
                ),
                row=row, col=1
            )
            
            # Highlight optimal K
            optimal_idx = kmeans_results["optimal_k"] - 1
            fig.add_trace(
                go.Scatter(
                    x=[kmeans_results["optimal_k"]],
                    y=[kmeans_results["inertias"][optimal_idx]],
                    mode="markers",
                    name=f"{name} K={kmeans_results['optimal_k']}",
                    marker=dict(
                        color=colors['accent'], 
                        size=12, 
                        symbol="diamond",
                        line=dict(width=2, color=colors['text'])
                    ),
                    showlegend=True
                ),
                row=row, col=1
            )
            
            # Plot distribution with cluster centers (right column)
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=f"{name} Distribution",
                    nbinsx=40,
                    histnorm="probability density",
                    opacity=0.8,
                    marker_color=hist_color,
                    marker_line=dict(width=1, color=line_color),
                    showlegend=True
                ),
                row=row, col=2
            )
            
            # Add vertical lines for cluster centers with smart positioning
            cluster_colors = [colors['accent'], colors['neutral'], colors['success'], 
                            '#8B4513', '#556B2F', '#4B0082', '#DC143C', '#FF6347']
            
            # Calculate histogram for annotation positioning
            hist_counts, hist_edges = np.histogram(data, bins=40, density=True)
            y_max = max(hist_counts)
            annotation_heights = [y_max * 0.9, y_max * 0.8, y_max * 0.7, y_max * 0.85, 
                                y_max * 0.75, y_max * 0.65, y_max * 0.95, y_max * 0.6]
            
            for i, center in enumerate(kmeans_results["cluster_centers"]):
                color = cluster_colors[i % len(cluster_colors)]
                
                # Add vertical line
                fig.add_vline(
                    x=center,
                    line_dash="dash",
                    line_color=color,
                    line_width=2,
                    row=row, col=2
                )
                
                # Add cluster center as scatter point with text
                fig.add_trace(
                    go.Scatter(
                        x=[center],
                        y=[annotation_heights[i % len(annotation_heights)]],
                        mode="markers+text",
                        marker=dict(
                            color=color,
                            size=6,
                            symbol="diamond"
                        ),
                        text=[f"C{i+1}: {center:.2f}"],
                        textposition="top center",
                        textfont=dict(size=9, color=color),
                        showlegend=False,
                        name=""
                    ),
                    row=row, col=2
                )
            
            # Store statistics for this mode
            mode_stats = {
                "Settlement_Point": point,
                "Mode": name,
                "Data_Points": len(data),
                "Max_K_Tested": max_k,
                "Optimal_K": kmeans_results["optimal_k"],
                "Optimal_Inertia": kmeans_results["inertias"][optimal_idx],
                "Cluster_Centers": list(kmeans_results["cluster_centers"])
            }
            
            # Add individual cluster information
            for i, center in enumerate(kmeans_results["cluster_centers"]):
                cluster_mask = kmeans_results["labels"] == i
                cluster_size = np.sum(cluster_mask)
                mode_stats[f"Cluster_{i+1}_Center"] = center
                mode_stats[f"Cluster_{i+1}_Size"] = cluster_size
                mode_stats[f"Cluster_{i+1}_Proportion"] = cluster_size / len(data)
            
            bimodal_cluster_stats.append(mode_stats)
        
        # Update layout with professional styling and better spacing
        fig.update_layout(
            title={
                "text": f"K-means Clustering Analysis (Bimodal) - {point}{title_suffix}<br><sub style='color:{colors['text']}'>Separate clustering analysis for positive and negative values</sub>",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16, "color": colors['text']}
            },
            height=950,  # Increased height for 2x2 layout
            width=1400,  # Increased width for better spacing
            showlegend=True,
            legend=dict(
                x=1.02,  # Position legend outside plot area to the right
                y=1,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=colors['text'],
                borderwidth=1,
                font=dict(color=colors['text'], size=10)
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(t=100, b=60, l=60, r=150)  # Extra right margin for external legend
        )
        
        # Update axis labels with professional styling
        for row in [1, 2]:
            fig.update_xaxes(
                title_text="Number of Clusters (K)", 
                row=row, col=1,
                title_font=dict(color=colors['text'], size=12),
                tickfont=dict(color=colors['text']),
                gridcolor='rgba(128,128,128,0.2)'
            )
            fig.update_yaxes(
                title_text="Inertia", 
                row=row, col=1,
                title_font=dict(color=colors['text'], size=12),
                tickfont=dict(color=colors['text']),
                gridcolor='rgba(128,128,128,0.2)'
            )
            fig.update_xaxes(
                title_text="DART_SLT Value", 
                row=row, col=2,
                title_font=dict(color=colors['text'], size=12),
                tickfont=dict(color=colors['text']),
                gridcolor='rgba(128,128,128,0.2)'
            )
            fig.update_yaxes(
                title_text="Probability Density", 
                row=row, col=2,
                title_font=dict(color=colors['text'], size=12),
                tickfont=dict(color=colors['text']),
                gridcolor='rgba(128,128,128,0.2)'
            )
        
        # Save individual plot
        output_path = output_dir / f"dart_slt_kmeans_bimodal_{safe_filename}.html"
        fig.write_html(output_path)
        print(f"  Plot saved to: {output_path}")
        
        # Save bimodal clustering statistics
        if bimodal_cluster_stats:
            bimodal_cluster_stats_df = pd.DataFrame(bimodal_cluster_stats)
            stats_path = output_dir / f"dart_slt_kmeans_bimodal_stats_{safe_filename}.csv"
            bimodal_cluster_stats_df.to_csv(stats_path, index=False)
            print(f"  Statistics saved to: {stats_path}")
    
    print(f"K-means bimodal analysis complete: {len(unique_points)} settlement points processed")