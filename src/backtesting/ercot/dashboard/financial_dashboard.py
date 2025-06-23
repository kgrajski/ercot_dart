"""Financial Summary Dashboard for ERCOT Trading Strategies.

This module creates an overview dashboard showing financial performance
across all hours and strategies, following the visualization patterns from exp2.
"""

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.features.ercot.visualization import COLOR_SEQUENCE
from src.features.ercot.visualization import PROFESSIONAL_COLORS
from src.features.ercot.visualization import apply_professional_axis_styling
from src.features.ercot.visualization import get_professional_layout


def create_financial_dashboard(
    strategy_results: Dict, output_path: str, settlement_point: str = "ERCOT_SOUTH_HUB"
):
    """Create financial summary dashboard showing all hours overlaid.

    Args:
        strategy_results: Dictionary of strategy results
        output_path: Path to save HTML dashboard
        settlement_point: Settlement point for title
    """
    print(f"📊 Creating financial summary dashboard...")

    # Create 2x2 subplot layout
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Cumulative P&L by Hour",
            "Win Rate by Hour",
            "Sharpe Ratio by Hour",
            "Total Trades by Hour",
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

    # Collect summary data for CSV export
    summary_data = []

    # Process each strategy
    for strategy_name, results in strategy_results.items():
        if not results or not results["trades"]:
            continue

        # Convert trades to DataFrame
        trades_df = pd.DataFrame(results["trades"])

        # Group by hour for aggregation
        hourly_stats = (
            trades_df.groupby("entry_hour")
            .agg(
                {
                    "pnl": ["sum", "count", lambda x: (x > 0).sum()],
                }
            )
            .round(2)
        )

        # Flatten column names
        hourly_stats.columns = ["total_pnl", "total_trades", "winning_trades"]
        hourly_stats["win_rate"] = (
            hourly_stats["winning_trades"] / hourly_stats["total_trades"] * 100
        ).fillna(0)

        # Calculate Sharpe ratio approximation per hour
        hourly_stats[
            "sharpe_approx"
        ] = 0.0  # Use float instead of int to avoid dtype issues
        for hour in hourly_stats.index:
            hour_trades = trades_df[trades_df["entry_hour"] == hour]["pnl"]
            if len(hour_trades) > 1 and hour_trades.std() > 0:
                hourly_stats.loc[hour, "sharpe_approx"] = (
                    hour_trades.mean() / hour_trades.std()
                )

        # Add strategy column for CSV export
        hourly_stats_export = hourly_stats.copy()
        hourly_stats_export["strategy"] = strategy_name
        hourly_stats_export["hour"] = hourly_stats_export.index
        summary_data.append(hourly_stats_export)

        color = strategy_colors[strategy_name]

        # Plot 1: Cumulative P&L by Hour
        fig.add_trace(
            go.Scatter(
                x=hourly_stats.index,
                y=hourly_stats["total_pnl"],
                mode="lines+markers",
                name=f"{strategy_name} P&L",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                showlegend=True,
                hovertemplate="<b>%{fullData.name}</b><br>Hour: %{x}<br>P&L: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Plot 2: Win Rate by Hour
        fig.add_trace(
            go.Scatter(
                x=hourly_stats.index,
                y=hourly_stats["win_rate"],
                mode="lines+markers",
                name=f"{strategy_name} Win Rate",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                showlegend=False,
                hovertemplate="<b>%{fullData.name}</b><br>Hour: %{x}<br>Win Rate: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Plot 3: Sharpe Ratio by Hour
        fig.add_trace(
            go.Scatter(
                x=hourly_stats.index,
                y=hourly_stats["sharpe_approx"],
                mode="lines+markers",
                name=f"{strategy_name} Sharpe",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                showlegend=False,
                hovertemplate="<b>%{fullData.name}</b><br>Hour: %{x}<br>Sharpe: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Plot 4: Total Trades by Hour
        fig.add_trace(
            go.Bar(
                x=hourly_stats.index,
                y=hourly_stats["total_trades"],
                name=f"{strategy_name} Trades",
                marker_color=color,
                opacity=0.7,
                showlegend=False,
                hovertemplate="<b>%{fullData.name}</b><br>Hour: %{x}<br>Trades: %{y}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    # Update layout with professional styling
    layout = get_professional_layout(
        title=f"ERCOT Trading Strategy Performance - {settlement_point}",
        height=800,
        showlegend=True,
        legend_position="upper_right",
    )

    fig.update_layout(**layout)
    apply_professional_axis_styling(fig, rows=2, cols=2)

    # Update axis labels
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=2)

    fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Number of Trades", row=2, col=2)

    # Add annotations with key statistics
    if strategy_results:
        first_strategy = list(strategy_results.values())[0]
        metrics = first_strategy["performance_metrics"]

        annotations_text = (
            f"<b>Strategy Performance Summary</b><br>"
            f"Total Return: {metrics.get('total_return_pct', 0):+.2f}%<br>"
            f"Win Rate: {metrics.get('win_rate_pct', 0):.1f}%<br>"
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.3f}<br>"
            f"Max DD: {metrics.get('max_drawdown_pct', 0):.2f}%"
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

    # Save HTML dashboard with project-standard config
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    print(f"   ✅ Financial dashboard saved: {output_path}")

    # Save PNG version
    png_path = output_path.replace(".html", ".png")
    try:
        fig.write_image(png_path, width=1200, height=800, scale=2)
        print(f"   ✅ Financial dashboard PNG saved: {png_path}")
    except Exception as e:
        print(
            f"   ⚠️  Could not save PNG (install kaleido with: pip install kaleido): {e}"
        )

    # Save corresponding CSV file
    if summary_data:
        combined_summary = pd.concat(summary_data, ignore_index=True)
        csv_path = output_path.replace(".html", ".csv")
        combined_summary.to_csv(csv_path, index=False)
        print(f"   ✅ Financial summary CSV saved: {csv_path}")


def create_performance_summary_table(strategy_results: Dict) -> pd.DataFrame:
    """Create summary table of strategy performance metrics."""
    summary_data = []

    for strategy_name, results in strategy_results.items():
        if not results:
            continue

        metrics = results["performance_metrics"]
        summary_data.append(
            {
                "Strategy": strategy_name,
                "Total Return (%)": f"{metrics.get('total_return_pct', 0):+.2f}",
                "Win Rate (%)": f"{metrics.get('win_rate_pct', 0):.1f}",
                "Total Trades": f"{metrics.get('total_trades', 0):,}",
                "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.3f}",
                "Max Drawdown (%)": f"{metrics.get('max_drawdown_pct', 0):.2f}",
                "Final Capital ($)": f"{metrics.get('final_capital', 0):,.2f}",
            }
        )

    return pd.DataFrame(summary_data)
