"""Demo script for ERCOT backtesting framework.

This script creates synthetic prediction data and demonstrates the backtesting
workflow to validate our implementation works correctly.
"""

import sys
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtesting.ercot.strategy_runner import ERCOTStrategyRunner


def create_synthetic_predictions(num_weeks=3, hours_per_day=24):
    """Create synthetic prediction data for testing."""
    print("🧪 Creating synthetic prediction data...")

    # Start date
    start_date = datetime(2025, 1, 1)

    # Generate data
    data = []

    for week in range(num_weeks):
        for day in range(7):  # 7 days per week
            for hour in range(hours_per_day):
                timestamp = start_date + timedelta(weeks=week, days=day, hours=hour)

                # Create synthetic DART values with some realistic patterns
                # Make some hours more volatile than others
                if hour in [7, 8, 18, 19]:  # Peak hours
                    volatility = 2.0
                    bias = 0.1  # Slight positive bias during peaks
                else:
                    volatility = 1.0
                    bias = 0.0

                # Actual DART (what really happened)
                actual_dart = np.random.normal(bias, volatility)

                # Predicted DART (what model predicted)
                # Add some correlation to actual + noise to simulate model accuracy
                prediction_accuracy = 0.65  # 65% correlation with actual
                predicted_dart = prediction_accuracy * actual_dart + (
                    1 - prediction_accuracy
                ) * np.random.normal(0, volatility)

                data.append(
                    {
                        "utc_ts": timestamp,
                        "end_hour": hour,
                        "actual_dart_slt": actual_dart,
                        "predicted_dart_slt": predicted_dart,
                        "week_num": week + 1,
                        "week_description": f"Week {week + 1} (Jan {1 + week*7}-{7 + week*7}, 2025)",
                    }
                )

    df = pd.DataFrame(data)

    print(f"   ✅ Created {len(df)} synthetic predictions")
    print(f"   Date range: {df['utc_ts'].min()} to {df['utc_ts'].max()}")
    print(f"   Hours covered: {sorted(df['end_hour'].unique())}")
    print(f"   Weeks: {df['week_num'].nunique()}")

    return df


def main():
    """Run the backtesting demo."""
    print("🚀 ERCOT Backtesting Framework Demo")
    print("=" * 50)

    # Create synthetic data
    predictions_df = create_synthetic_predictions(num_weeks=3)

    # Save to temporary file
    temp_predictions_file = "temp_synthetic_predictions.csv"
    predictions_df.to_csv(temp_predictions_file, index=False)
    print(f"📁 Saved synthetic data to: {temp_predictions_file}")

    try:
        # Initialize strategy runner
        print("\n📊 Initializing strategy runner...")
        runner = ERCOTStrategyRunner(
            output_dir="demo_backtesting_results", settlement_point="SYNTHETIC_HUB"
        )

        # Run backtests
        print("\n🎯 Running backtests...")
        results = runner.run_backtest(
            predictions_file=temp_predictions_file,
            strategy_types=["naive"],
            initial_capital=10000.0,
            transaction_cost=0.5,
        )

        # Analyze results
        print("\n📈 Results Analysis:")
        print("=" * 40)

        for strategy_name, result in results.items():
            if result is None:
                print(f"❌ {strategy_name}: Failed to run")
                continue

            metrics = result["performance_metrics"]
            trades = result["trades"]

            print(f"✅ {strategy_name.upper()} Strategy Results:")
            print(f"   Total Trades: {len(trades)}")
            print(f"   Win Rate: {metrics.get('win_rate_pct', 0):.1f}%")
            print(f"   Total Return: {metrics.get('total_return_pct', 0):+.2f}%")
            print(f"   Total P&L: ${metrics.get('total_pnl', 0):.2f}")
            print(f"   Final Capital: ${metrics.get('final_capital', 0):,.2f}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
            print(
                f"   Transaction Costs: ${metrics.get('total_transaction_costs', 0):.2f}"
            )

            # Show sample trades
            if trades:
                print(f"\n   Sample Trades (first 5):")
                trades_df = pd.DataFrame(trades)
                sample_cols = [
                    "entry_time",
                    "direction",
                    "entry_prediction",
                    "actual_dart",
                    "pnl",
                ]
                print(trades_df[sample_cols].head().to_string(index=False))

        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Results saved to: demo_backtesting_results/SYNTHETIC_HUB/")
        print(
            f"🌐 Open financial dashboard: demo_backtesting_results/SYNTHETIC_HUB/dashboards/financial_summary.html"
        )

        # Clean up
        import os

        if os.path.exists(temp_predictions_file):
            os.remove(temp_predictions_file)
            print(f"🧹 Cleaned up temporary file: {temp_predictions_file}")

        return 0

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
