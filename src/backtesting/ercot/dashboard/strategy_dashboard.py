"""Strategy Deep-Dive Dashboard for ERCOT Trading Strategies.

This module creates detailed hour-specific dashboards showing financial performance,
prediction accuracy, feature evolution, and trade analysis.
"""

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.features.ercot.visualization import COLOR_SEQUENCE
from src.features.ercot.visualization import PROFESSIONAL_COLORS
from src.features.ercot.visualization import apply_professional_axis_styling
from src.features.ercot.visualization import get_professional_layout


def create_strategy_dashboard(
    strategy_results: Dict,
    target_hour: int,
    output_path: str,
    settlement_point: str = "ERCOT_SOUTH_HUB",
):
    """Create detailed strategy dashboard for a specific hour.

    Args:
        strategy_results: Dictionary of strategy results
        target_hour: Hour to focus on (0-23)
        output_path: Path to save HTML dashboard
        settlement_point: Settlement point for title
    """
    print(f"📊 Creating strategy dashboard for hour {target_hour:02d}...")

    # Create 2x2 subplot layout
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Portfolio Value Over Time - Hour {target_hour:02d}",
            f"Prediction Accuracy - Hour {target_hour:02d}",
            f"Trade P&L Distribution - Hour {target_hour:02d}",
            f"Weekly Performance - Hour {target_hour:02d}",
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Color assignment
    strategy_colors = {}
    for i, strategy_name in enumerate(strategy_results.keys()):
        strategy_colors[strategy_name] = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]

    # Collect data for CSV export
    csv_data = []

    # Process each strategy
    for strategy_name, results in strategy_results.items():
        if not results or not results["trades"]:
            continue

        color = strategy_colors[strategy_name]

        # Filter data for target hour
        trades_df = pd.DataFrame(results["trades"])
        hour_trades = trades_df[trades_df["entry_hour"] == target_hour].copy()

        portfolio_df = pd.DataFrame(results["portfolio_values"])
        hour_portfolio = portfolio_df[portfolio_df["end_hour"] == target_hour].copy()

        if hour_trades.empty:
            continue

        # Convert timestamps
        hour_trades["entry_time"] = pd.to_datetime(hour_trades["entry_time"])
        hour_portfolio["utc_ts"] = pd.to_datetime(hour_portfolio["utc_ts"])

        # Add strategy info for CSV export
        hour_trades_export = hour_trades.copy()
        hour_trades_export["strategy"] = strategy_name
        csv_data.append(hour_trades_export)

        # Plot 1: Portfolio Value Over Time
        fig.add_trace(
            go.Scatter(
                x=hour_portfolio["utc_ts"],
                y=hour_portfolio["portfolio_value"],
                mode="lines+markers",
                name=f"{strategy_name} Portfolio",
                line=dict(color=color, width=2),
                marker=dict(size=4),
                showlegend=True,
                hovertemplate="<b>%{fullData.name}</b><br>Time: %{x}<br>Portfolio Value: $%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Plot 2: Prediction Accuracy
        # Create bins for actual vs predicted comparison
        hour_trades["correct_prediction"] = (hour_trades["actual_dart_slt"] > 0) == (
            hour_trades["entry_prediction"] > 0
        )

        accuracy_over_time = (
            hour_trades.set_index("entry_time")
            .resample("W")["correct_prediction"]
            .mean()
            * 100
        )

        fig.add_trace(
            go.Scatter(
                x=accuracy_over_time.index,
                y=accuracy_over_time.values,
                mode="lines+markers",
                name=f"{strategy_name} Accuracy",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                showlegend=False,
                hovertemplate="<b>%{fullData.name}</b><br>Week: %{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Plot 3: P&L Distribution
        fig.add_trace(
            go.Histogram(
                x=hour_trades["pnl"],
                name=f"{strategy_name} P&L",
                marker_color=color,
                opacity=0.7,
                nbinsx=20,
                showlegend=False,
                hovertemplate="<b>%{fullData.name}</b><br>P&L Range: %{x}<br>Count: %{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Plot 4: Weekly Performance
        if "week_num" in hour_trades.columns:
            weekly_pnl = hour_trades.groupby("week_num")["pnl"].sum()

            fig.add_trace(
                go.Bar(
                    x=weekly_pnl.index,
                    y=weekly_pnl.values,
                    name=f"{strategy_name} Weekly P&L",
                    marker_color=color,
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate="<b>%{fullData.name}</b><br>Week: %{x}<br>P&L: $%{y:.2f}<extra></extra>",
                ),
                row=2,
                col=2,
            )

    # Update layout with professional styling
    layout = get_professional_layout(
        title=f"ERCOT Strategy Deep-Dive: Hour {target_hour:02d} - {settlement_point}",
        height=800,
        showlegend=True,
        legend_position="upper_right",
    )

    fig.update_layout(**layout)
    apply_professional_axis_styling(fig, rows=2, cols=2)

    # Update axis labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="P&L ($)", row=2, col=1)
    fig.update_xaxes(title_text="Week Number", row=2, col=2)

    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Weekly P&L ($)", row=2, col=2)

    # Add hour-specific statistics
    if strategy_results:
        first_strategy = list(strategy_results.values())[0]
        trades_df = pd.DataFrame(first_strategy["trades"])
        hour_trades = trades_df[trades_df["entry_hour"] == target_hour]

        if not hour_trades.empty:
            hour_stats = {
                "total_trades": len(hour_trades),
                "win_rate": (hour_trades["pnl"] > 0).mean() * 100,
                "avg_pnl": hour_trades["pnl"].mean(),
                "total_pnl": hour_trades["pnl"].sum(),
                "accuracy": (
                    (hour_trades["actual_dart_slt"] > 0)
                    == (hour_trades["entry_prediction"] > 0)
                ).mean()
                * 100,
            }

            annotations_text = (
                f"<b>Hour {target_hour:02d} Statistics</b><br>"
                f"Total Trades: {hour_stats['total_trades']}<br>"
                f"Win Rate: {hour_stats['win_rate']:.1f}%<br>"
                f"Avg P&L: ${hour_stats['avg_pnl']:.2f}<br>"
                f"Total P&L: ${hour_stats['total_pnl']:.2f}<br>"
                f"Prediction Accuracy: {hour_stats['accuracy']:.1f}%"
            )

            fig.add_annotation(
                text=annotations_text,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=PROFESSIONAL_COLORS["text"],
                borderwidth=1,
                font=dict(size=10, color=PROFESSIONAL_COLORS["text"]),
            )

    # Save HTML dashboard
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    print(f"   ✅ Strategy dashboard saved: {output_path}")

    # Save PNG version
    png_path = output_path.replace(".html", ".png")
    try:
        fig.write_image(png_path, width=1200, height=800, scale=2)
        print(f"   ✅ Strategy dashboard PNG saved: {png_path}")
    except Exception as e:
        print(
            f"   ⚠️  Could not save PNG (install kaleido with: pip install kaleido): {e}"
        )

    # Save corresponding CSV file with trade details
    if csv_data:
        combined_trades = pd.concat(csv_data, ignore_index=True)
        csv_path = output_path.replace(".html", ".csv")
        combined_trades.to_csv(csv_path, index=False)
        print(f"   ✅ Strategy trades CSV saved: {csv_path}")


def create_trade_analysis_table(
    strategy_results: Dict, target_hour: int
) -> pd.DataFrame:
    """Create detailed trade analysis table for a specific hour."""
    all_trades = []

    for strategy_name, results in strategy_results.items():
        if not results or not results["trades"]:
            continue

        trades_df = pd.DataFrame(results["trades"])
        hour_trades = trades_df[trades_df["entry_hour"] == target_hour].copy()

        if not hour_trades.empty:
            hour_trades["strategy"] = strategy_name
            hour_trades["correct_prediction"] = (
                hour_trades["actual_dart_slt"] > 0
            ) == (hour_trades["entry_prediction"] > 0)
            all_trades.append(hour_trades)

    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)

        # Select relevant columns for analysis
        analysis_cols = [
            "strategy",
            "entry_time",
            "direction",
            "entry_prediction",
            "actual_dart_slt",
            "pnl",
            "correct_prediction",
            "week_num",
        ]

        return combined_trades[analysis_cols].sort_values("entry_time")

    return pd.DataFrame()


def calculate_hourly_risk_metrics(strategy_results: Dict, target_hour: int) -> Dict:
    """Calculate detailed risk metrics for a specific hour."""
    risk_metrics = {}

    for strategy_name, results in strategy_results.items():
        if not results or not results["trades"]:
            continue

        trades_df = pd.DataFrame(results["trades"])
        hour_trades = trades_df[trades_df["entry_hour"] == target_hour]

        if hour_trades.empty:
            continue

        pnl_series = hour_trades["pnl"]

        # Calculate risk metrics
        metrics = {
            "var_95": np.percentile(pnl_series, 5),  # Value at Risk (95%)
            "cvar_95": pnl_series[
                pnl_series <= np.percentile(pnl_series, 5)
            ].mean(),  # Conditional VaR
            "max_loss": pnl_series.min(),
            "max_gain": pnl_series.max(),
            "volatility": pnl_series.std(),
            "skewness": pnl_series.skew(),
            "kurtosis": pnl_series.kurtosis(),
            "profit_factor": pnl_series[pnl_series > 0].sum()
            / abs(pnl_series[pnl_series < 0].sum())
            if (pnl_series < 0).any()
            else float("inf"),
        }

        risk_metrics[strategy_name] = metrics

    return risk_metrics


def create_hours_overlay_dashboard(
    strategy_results: Dict, output_path: str, settlement_point: str = "ERCOT_SOUTH_HUB"
):
    """Create hours overlay dashboard showing all hours on the same plots.

    This addresses the conceptual issue with per-hour portfolio values by showing:
    1. Cumulative returns by hour (not absolute portfolio values)
    2. Prediction accuracy trends by hour
    3. Weekly performance comparison across hours

    Args:
        strategy_results: Dictionary of strategy results
        output_path: Path to save HTML dashboard
        settlement_point: Settlement point for title
    """
    print(f"📊 Creating hours overlay dashboard...")

    # Create 3x1 subplot layout (3 rows, 1 column for better use of horizontal space)
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "Cumulative Returns by Hour",
            "Prediction Accuracy by Hour",
            "Weekly Performance by Hour",
        ],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        vertical_spacing=0.12,  # Increased spacing to prevent label overlap
    )

    # Process each strategy (though typically just one for this view)
    for strategy_name, results in strategy_results.items():
        if not results or not results["trades"]:
            continue

        trades_df = pd.DataFrame(results["trades"])
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])

        # Get all unique hours
        available_hours = sorted(trades_df["entry_hour"].unique())

        # Color scheme: use different colors for different hours
        for i, hour in enumerate(available_hours):
            color = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]
            hour_trades = trades_df[trades_df["entry_hour"] == hour].copy()

            if hour_trades.empty:
                continue

            # Plot 1: Cumulative Returns by Hour (not absolute portfolio value)
            hour_trades_sorted = hour_trades.sort_values("entry_time")
            cumulative_returns = hour_trades_sorted["pnl"].cumsum()

            fig.add_trace(
                go.Scatter(
                    x=hour_trades_sorted["entry_time"],
                    y=cumulative_returns,
                    mode="lines+markers",
                    name=f"Hour {hour:02d}",
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    legendgroup=f"hour_{hour}",
                    hovertemplate=f"<b>Hour {hour:02d}</b><br>Time: %{{x}}<br>Cumulative Return: $%{{y:,.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Plot 2: Prediction Accuracy by Hour (weekly rolling)
            hour_trades["correct_prediction"] = (
                hour_trades["actual_dart_slt"] > 0
            ) == (hour_trades["entry_prediction"] > 0)

            # Weekly accuracy calculation
            weekly_accuracy = (
                hour_trades.set_index("entry_time")
                .resample("W")["correct_prediction"]
                .mean()
                * 100
            )

            if len(weekly_accuracy) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=weekly_accuracy.index,
                        y=weekly_accuracy.values,
                        mode="lines+markers",
                        name=f"Hour {hour:02d}",
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                        legendgroup=f"hour_{hour}",
                        showlegend=False,
                        hovertemplate=f"<b>Hour {hour:02d}</b><br>Week: %{{x}}<br>Accuracy: %{{y:.1f}}%<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

            # Plot 3: Weekly Performance by Hour
            if "week_num" in hour_trades.columns:
                weekly_pnl = hour_trades.groupby("week_num")["pnl"].sum()

                # Use slight offset for x-position to avoid overlap
                x_offset = (i - len(available_hours) / 2) * 0.02

                fig.add_trace(
                    go.Bar(
                        x=weekly_pnl.index + x_offset,
                        y=weekly_pnl.values,
                        name=f"Hour {hour:02d}",
                        marker_color=color,
                        opacity=0.7,
                        legendgroup=f"hour_{hour}",
                        showlegend=False,
                        hovertemplate=f"<b>Hour {hour:02d}</b><br>Week: %{{x}}<br>P&L: $%{{y:.2f}}<extra></extra>",
                    ),
                    row=3,
                    col=1,
                )

    # Update layout with professional styling
    layout = get_professional_layout(
        title=f"ERCOT Strategy Hours Overlay: All Hours - {settlement_point}",
        height=1000,  # Increased height for 3-row layout
        showlegend=True,
        legend_position="upper_right",
    )

    # Add more room for the legend on the right
    layout.update(
        {
            "margin": {
                "l": 80,
                "r": 200,
                "t": 100,
                "b": 80,
            },  # Increased right margin for legend
            "legend": {
                "orientation": "v",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.02,  # Position legend outside the plot area
                "font": {"size": 10},
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": PROFESSIONAL_COLORS["text"],
                "borderwidth": 1,
            },
        }
    )

    fig.update_layout(**layout)
    apply_professional_axis_styling(fig, rows=3, cols=1)

    # Update axis labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Week", row=2, col=1)
    fig.update_xaxes(title_text="Week Number", row=3, col=1)

    fig.update_yaxes(title_text="Cumulative Return ($)", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_yaxes(title_text="Weekly P&L ($)", row=3, col=1)

    # Save HTML dashboard
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    print(f"   ✅ Hours overlay dashboard saved: {output_path}")

    # Save PNG version
    png_path = output_path.replace(".html", ".png")
    try:
        fig.write_image(
            png_path, width=1400, height=1000, scale=2
        )  # Increased width for legend space
        print(f"   ✅ Hours overlay PNG saved: {png_path}")
    except Exception as e:
        print(
            f"   ⚠️  Could not save PNG (install kaleido with: pip install kaleido): {e}"
        )
