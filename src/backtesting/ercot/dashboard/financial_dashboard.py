"""Financial Summary Dashboard for ERCOT Trading Strategies.

This module creates an overview dashboard showing financial performance
across all hours and strategies, following the visualization patterns from exp2.

ARCHITECTURE: This module is now DISPLAY-ONLY. All metrics calculations are 
handled by the ERCOTTradeAnalytics class (single source of truth).
"""

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtesting.ercot.analytics.trade_analytics import create_analytics_engine
from src.features.ercot.visualization import COLOR_SEQUENCE
from src.features.ercot.visualization import PROFESSIONAL_COLORS
from src.features.ercot.visualization import apply_professional_axis_styling
from src.features.ercot.visualization import get_professional_layout


def create_financial_dashboard(
    strategy_results: Dict, output_path: str, settlement_point: str = "ERCOT_SOUTH_HUB"
):
    """Create financial summary dashboard showing all hours overlaid.

    REFACTORED: Now uses centralized analytics engine for all calculations.

    Args:
        strategy_results: Dictionary of strategy results
        output_path: Path to save HTML dashboard
        settlement_point: Settlement point for title
    """
    print(f"📊 Creating financial summary dashboard...")

    # Filter out None results before processing
    valid_strategy_results = {
        name: results
        for name, results in strategy_results.items()
        if results is not None and results.get("trades")
    }

    if not valid_strategy_results:
        print("   ⚠️  No valid strategy results to display in dashboard")
        # Create empty dashboard file to satisfy caller expectations
        with open(output_path, "w") as f:
            f.write(
                "<html><body><h1>No Valid Strategy Results</h1><p>All strategies failed or had no trades.</p></body></html>"
            )
        return str(output_path)

    # SINGLE SOURCE OF TRUTH: Create analytics engine
    analytics = create_analytics_engine(valid_strategy_results)

    # Get strategy name for subtitle
    strategy_name = (
        list(valid_strategy_results.keys())[0] if valid_strategy_results else "Unknown"
    )

    # Create 2x2 subplot layout
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Contribution to Cumulative Returns by Hour",
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

    # Color assignment for strategies (using professional color sequence)
    strategy_names = list(valid_strategy_results.keys())
    strategy_colors = {}
    for i, strategy_name in enumerate(strategy_names):
        # Use the professional color sequence, cycling through if needed
        strategy_colors[strategy_name] = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]

    # Collect summary data for CSV export
    summary_data = []

    # Collect strategy performance for subtitle
    strategy_performance_lines = []

    # Process each strategy
    for strategy_name_loop, results in valid_strategy_results.items():
        if not results or not results["trades"]:
            continue

        # GET HOURLY METRICS FROM SINGLE SOURCE OF TRUTH
        hourly_stats = analytics.get_hourly_metrics(strategy_name_loop)

        if hourly_stats.empty:
            continue

        # Add strategy column for CSV export
        hourly_stats_export = hourly_stats.copy()
        hourly_stats_export["strategy"] = strategy_name_loop
        summary_data.append(hourly_stats_export)

        # Get strategy color and performance metrics
        strategy_color = strategy_colors[strategy_name_loop]
        performance_metrics = results.get("performance_metrics", {})
        total_return_pct = performance_metrics.get("total_return_pct", 0)
        final_capital = performance_metrics.get("final_capital", 0)

        # Add to subtitle performance summary
        strategy_performance_lines.append(
            f"{strategy_name_loop.upper()}: {total_return_pct:+.2f}% return (${final_capital:,.0f} final capital)"
        )

        # Plot 1: Contribution to Cumulative Returns by Hour (no legend)
        fig.add_trace(
            go.Bar(
                x=hourly_stats["hour"],
                y=hourly_stats["total_pnl"],
                name=f"{strategy_name_loop.upper()}",
                marker_color=strategy_color,
                opacity=0.7,
                showlegend=False,  # No legend - performance shown in subtitle
                hovertemplate="<b>Hour %{x}</b><br>P&L Contribution: $%{y:.2f}<extra></extra>",
                legendgroup=strategy_name_loop,  # Group all traces for this strategy
            ),
            row=1,
            col=1,
        )

        # Plot 2: Win Rate by Hour (same color, no legend)
        fig.add_trace(
            go.Bar(
                x=hourly_stats["hour"],
                y=hourly_stats["win_rate_pct"],
                name=f"{strategy_name_loop} Win Rate",
                marker_color=strategy_color,
                opacity=0.7,
                showlegend=False,
                hovertemplate="<b>Hour %{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>",
                legendgroup=strategy_name_loop,
            ),
            row=1,
            col=2,
        )

        # Plot 3: Sharpe Ratio by Hour (same color, no legend)
        fig.add_trace(
            go.Bar(
                x=hourly_stats["hour"],
                y=hourly_stats["sharpe_ratio"],
                name=f"{strategy_name_loop} Sharpe",
                marker_color=strategy_color,
                opacity=0.7,
                showlegend=False,
                hovertemplate="<b>Hour %{x}</b><br>Sharpe Ratio: %{y:.3f}<extra></extra>",
                legendgroup=strategy_name_loop,
            ),
            row=2,
            col=1,
        )

        # Plot 4: Total Trades by Hour (same color, no legend)
        fig.add_trace(
            go.Bar(
                x=hourly_stats["hour"],
                y=hourly_stats["total_trades"],
                name=f"{strategy_name_loop} Trades",
                marker_color=strategy_color,
                opacity=0.7,
                showlegend=False,
                hovertemplate="<b>Hour %{x}</b><br>Total Trades: %{y}<extra></extra>",
                legendgroup=strategy_name_loop,
            ),
            row=2,
            col=2,
        )

    # Create title with performance subtitle (horizontal format for scalability)
    main_title = f"ERCOT Trading Strategy Performance - {settlement_point}"
    subtitle = " • ".join(
        strategy_performance_lines
    )  # Horizontal layout with bullet separators
    full_title = f"{main_title}<br><span style='font-size: 16px;'><b>{subtitle}</b></span>"  # Bigger, bold font for key results

    # Update layout with professional styling
    layout = get_professional_layout(
        title=full_title,
        height=800,
        showlegend=False,  # No legend needed - performance shown in title
    )

    # No legend configuration needed since performance is in subtitle

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

    # Save CSV version of summary data
    if summary_data:
        csv_data = pd.concat(summary_data, ignore_index=True)
        csv_path = output_path.replace(".html", ".csv")
        csv_data.to_csv(csv_path, index=False)
        print(f"   ✅ Financial summary CSV saved: {csv_path}")

    return str(output_path)


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
