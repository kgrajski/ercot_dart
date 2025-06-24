"""Strategy Deep-Dive Dashboard for ERCOT Trading Strategies.

This module creates detailed hour-specific dashboards showing financial performance,
prediction accuracy, feature evolution, and trade analysis.

ARCHITECTURE: This module is now DISPLAY-ONLY. All metrics calculations are 
handled by the ERCOTTradeAnalytics class (single source of truth).
"""

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtesting.ercot.analytics.trade_analytics import create_analytics_engine
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

    REFACTORED: Now uses centralized analytics engine for all calculations.

    Args:
        strategy_results: Dictionary of strategy results
        target_hour: Hour to focus on (0-23)
        output_path: Path to save HTML dashboard
        settlement_point: Settlement point for title
    """
    print(f"📊 Creating strategy dashboard for hour {target_hour:02d}...")

    # SINGLE SOURCE OF TRUTH: Create analytics engine
    analytics = create_analytics_engine(strategy_results)

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

        # GET DATA FROM SINGLE SOURCE OF TRUTH
        trades_df = analytics.get_trades_dataframe(strategy_name)
        hour_trades = trades_df[trades_df["entry_hour"] == target_hour].copy()

        # Portfolio data (still from original results for now)
        portfolio_df = pd.DataFrame(results["portfolio_values"])
        hour_portfolio = portfolio_df[portfolio_df["end_hour"] == target_hour].copy()

        if hour_trades.empty:
            continue

        # Convert portfolio timestamps
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

        # Plot 2: Prediction Accuracy - USING SINGLE SOURCE OF TRUTH
        weekly_accuracy = analytics.get_weekly_accuracy_by_hour(
            strategy_name, target_hour
        )

        if not weekly_accuracy.empty:
            fig.add_trace(
                go.Scatter(
                    x=weekly_accuracy["week_start"],
                    y=weekly_accuracy["accuracy_pct"],
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

        # Plot 4: Weekly Performance - USING SINGLE SOURCE OF TRUTH
        weekly_performance = analytics.get_weekly_performance_by_hour(
            strategy_name, target_hour
        )

        if not weekly_performance.empty:
            fig.add_trace(
                go.Bar(
                    x=weekly_performance["week_num"],
                    y=weekly_performance["total_pnl"],
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

    # Add hour-specific statistics - USING SINGLE SOURCE OF TRUTH
    if strategy_results:
        first_strategy = list(strategy_results.keys())[0]
        hour_stats = analytics.get_hour_statistics_summary(first_strategy, target_hour)

        if hour_stats:
            annotations_text = (
                f"<b>Hour {target_hour:02d} Statistics</b><br>"
                f"Total Trades: {hour_stats['total_trades_formatted']}<br>"
                f"Win Rate: {hour_stats['win_rate_formatted']}<br>"
                f"Avg P&L: {hour_stats['avg_pnl_formatted']}<br>"
                f"Total P&L: {hour_stats['total_pnl_formatted']}<br>"
                f"Prediction Accuracy: {hour_stats['prediction_accuracy_formatted']}"
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
    """Create detailed trade analysis table for a specific hour.

    REFACTORED: Now uses centralized analytics engine.
    """
    analytics = create_analytics_engine(strategy_results)
    all_trades = []

    for strategy_name in strategy_results.keys():
        trades_df = analytics.get_trades_dataframe(strategy_name)
        hour_trades = trades_df[trades_df["entry_hour"] == target_hour].copy()

        if not hour_trades.empty:
            hour_trades["strategy"] = strategy_name
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
            "correct_prediction",  # Using the authoritative field
            "week_num",
        ]

        available_cols = [
            col for col in analysis_cols if col in combined_trades.columns
        ]
        return combined_trades[available_cols].sort_values("entry_time")

    return pd.DataFrame()


def calculate_hourly_risk_metrics(strategy_results: Dict, target_hour: int) -> Dict:
    """Calculate detailed risk metrics for a specific hour.

    REFACTORED: Now uses centralized analytics engine.
    """
    analytics = create_analytics_engine(strategy_results)
    risk_metrics = {}

    for strategy_name in strategy_results.keys():
        trades_df = analytics.get_trades_dataframe(strategy_name)
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

    REFACTORED: Now uses centralized analytics engine for all calculations.

    Args:
        strategy_results: Dictionary of strategy results
        output_path: Path to save HTML dashboard
        settlement_point: Settlement point for title
    """
    print(f"📊 Creating hours overlay dashboard...")

    # SINGLE SOURCE OF TRUTH: Create analytics engine
    analytics = create_analytics_engine(strategy_results)

    # Get strategy name for subtitle
    strategy_name = list(strategy_results.keys())[0] if strategy_results else "Unknown"

    # Create 3x1 subplot layout (3 rows, 1 column for better use of horizontal space)
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "Per Hour Contribution to Cumulative Returns Time Series",
            "Weekly Per Hour Contribution to Cumulative Returns Time Series",
            "Weekly Per Hour Accuracy Rate Time Series",
        ],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        vertical_spacing=0.12,  # Increased spacing to prevent label overlap
    )

    # Process each strategy (though typically just one for this view)
    for strategy_name_loop, results in strategy_results.items():
        if not results or not results["trades"]:
            continue

        # GET ALL HOURS FROM SINGLE SOURCE OF TRUTH
        hourly_summary = analytics.get_all_hours_summary(strategy_name_loop)
        available_hours = sorted(hourly_summary["hour"].tolist())

        # Color scheme: use different colors for different hours
        for i, hour in enumerate(available_hours):
            color = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]

            # Plot 1: Cumulative Returns by Hour - USING SINGLE SOURCE OF TRUTH
            cumulative_data = analytics.get_cumulative_returns_by_hour(
                strategy_name_loop, hour
            )

            if not cumulative_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=cumulative_data["entry_time"],
                        y=cumulative_data["cumulative_returns"],
                        mode="lines+markers",
                        name=f"Hour {hour:02d}",
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        legendgroup=f"hour_{hour}",
                        hovertemplate=f"<b>Hour {hour:02d}</b><br>Date: %{{x|%Y-%m-%d}}<br>Cumulative Return: $%{{y:,.2f}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

            # Plot 2: Weekly Performance by Hour - USING SINGLE SOURCE OF TRUTH
            weekly_performance = analytics.get_weekly_performance_by_hour(
                strategy_name_loop, hour
            )

            if not weekly_performance.empty:
                # Use integer week numbers for better display
                week_numbers = weekly_performance["week_num"].round().astype(int)
                # Use slight offset for x-position to avoid overlap but keep centered
                x_offset = (i - len(available_hours) / 2) * 0.03

                fig.add_trace(
                    go.Bar(
                        x=week_numbers + x_offset,
                        y=weekly_performance["total_pnl"],
                        name=f"Hour {hour:02d}",
                        marker_color=color,
                        opacity=0.7,
                        legendgroup=f"hour_{hour}",
                        showlegend=False,
                        hovertemplate=f"<b>Hour {hour:02d}</b><br>Week: %{{x}}<br>P&L: $%{{y:.2f}}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

            # Plot 3: Prediction Accuracy by Hour - USING SINGLE SOURCE OF TRUTH
            weekly_accuracy = analytics.get_weekly_accuracy_by_hour(
                strategy_name_loop, hour
            )

            if not weekly_accuracy.empty:
                fig.add_trace(
                    go.Scatter(
                        x=weekly_accuracy["week_start"],
                        y=weekly_accuracy["accuracy_pct"],
                        mode="lines+markers",
                        name=f"Hour {hour:02d}",
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                        legendgroup=f"hour_{hour}",
                        showlegend=False,
                        hovertemplate=f"<b>Hour {hour:02d}</b><br>Week Start: %{{x|%Y-%m-%d}}<br>Accuracy: %{{y:.1f}}%<extra></extra>",
                    ),
                    row=3,
                    col=1,
                )

    # Create title with strategy subtitle
    main_title = f"ERCOT Strategy Hours Overlay: All Hours - {settlement_point}"
    subtitle = f"Trading Strategy: {strategy_name.upper()}"
    full_title = (
        f"{main_title}<br><span style='font-size: 16px;'><b>{subtitle}</b></span>"
    )

    # Update layout with professional styling
    layout = get_professional_layout(
        title=full_title,
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

    # Update axis labels with improved formatting
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(
        title_text="Week Number",
        row=2,
        col=1,
        dtick=1,  # Show integer week numbers
        tickmode="linear",  # Ensure linear spacing
    )
    fig.update_xaxes(title_text="Week", row=3, col=1)

    fig.update_yaxes(title_text="Cumulative Return ($)", row=1, col=1)
    fig.update_yaxes(title_text="Weekly P&L ($)", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=3, col=1)

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
