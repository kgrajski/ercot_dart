"""05-kag-exp0-ercot-backtest: ERCOT Trading Strategy Backtesting Workflow

This script implements backtesting for ERCOT electricity market trading strategies
using predictions from the exp2 progressive validation workflow.

The data comes from the 04-kag-exp2-ercot-modeling.py script.
Specifically, the predictions_progressive_xgboost_classification.csv file.

Key Features:
- Factory pattern for strategy registration and execution
- Interactive HTML dashboards inspired by exp2
- Financial summary dashboard (all hours overlaid)
- Strategy deep-dive dashboards (single hour focus)
- Risk metrics and transaction cost modeling
- Extensible framework for additional strategies
"""

import os
import sys
import time
from pathlib import Path

# Add project root to Python path for direct script execution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backtesting.ercot.strategy_runner import ERCOTStrategyRunner


def main():
    """Main function for ERCOT strategy backtesting."""

    script_name = "05-kag-exp0-ercot-backtest"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***")

    # Configuration - follows exp2 directory patterns
    processed_data_date = "2025-06-04"
    data_exp = "exp2"  # Source experiment for data
    backtest_exp = "exp0"  # Backtesting experiment identifier
    strategy_types = ["naive"]
    initial_capital = 10000.0
    transaction_cost = 0.5

    # Optional: Filter to specific hours (None = all hours)
    # target_hours = [16, 17, 18, 19]  # Peak hours only
    # target_hours = [1, 2, 3, 4]     # Off-peak hours only
    target_hours = None  # All hours

    # Directory setup - following exp2 modeling pattern
    root_dir = "/Users/kag/Documents/Projects/"
    project_dir = os.path.join(root_dir, "ercot_dart")
    data_dir = os.path.join(project_dir, "data/studies", data_exp, processed_data_date)

    print(f"\n** ERCOT Trading Strategy Backtesting")
    print(f"   Source experiment: {data_exp}")
    print(f"   Backtest experiment: {backtest_exp}")
    print(f"   Data directory: {data_dir}")
    print(f"   Strategy types: {strategy_types}")
    print(f"   Initial capital: ${initial_capital:,.2f}")
    print(f"   Transaction cost: ${transaction_cost:.2f} per trade")
    if target_hours:
        print(f"   Target hours: {target_hours}")
    else:
        print(f"   Target hours: All hours")

    # Generate the list of subdirectories in the data directory (settlement points)
    spp_loc_list = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    print(f"Subdirectories in {data_dir}: {spp_loc_list}")

    # Process each settlement point independently
    for spp_loc in spp_loc_list:
        print(f"\n** Running backtests for {spp_loc}")

        # Directory structure following exp2 patterns
        # Base path: data_dir/{settlement_point}
        base_study_path = Path(data_dir) / spp_loc

        # Source data paths (from modeling experiment)
        predictions_file = (
            base_study_path
            / "modeling"
            / "xgboost_classification"
            / "predictions_progressive_xgboost_classification.csv"
        )

        # Output paths (parallel to modeling, under backtest/{backtest_exp})
        output_base_path = base_study_path / "backtest" / backtest_exp

        print(f"   Predictions file: {predictions_file}")
        print(f"   Output base path: {output_base_path}")

        try:
            # Check if predictions file exists
            if not predictions_file.exists():
                print(
                    f"⚠️  Skipping {spp_loc} - predictions file not found: {predictions_file}"
                )
                continue

            # Initialize strategy runner with exp2-style paths
            print("   Initializing strategy runner...")
            strategy_runner = ERCOTStrategyRunner(
                settlement_point=spp_loc,
                output_base_path=output_base_path,
                target_hours=target_hours,
            )

            # Run backtests
            print("   Running backtests...")
            results = strategy_runner.run_backtest(
                predictions_file=str(predictions_file),
                strategy_types=strategy_types,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
            )

            # Print summary for this settlement point
            print(f"\n   Backtest Results Summary for {spp_loc}")
            print("   " + "-" * 40)

            successful_strategies = [
                name for name, result in results.items() if result is not None
            ]

            for strategy in successful_strategies:
                metrics = results[strategy]["performance_metrics"]
                print(f"     {strategy.upper()}:")
                print(
                    f"       Total Return: {metrics.get('total_return_pct', 0):+.2f}%"
                )
                print(f"       Win Rate: {metrics.get('win_rate_pct', 0):.1f}%")
                print(f"       Total Trades: {metrics.get('total_trades', 0):,}")
                print(f"       Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                print(f"       Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
                print(f"       Final Capital: ${metrics.get('final_capital', 0):,.2f}")

            # File locations following new simplified structure
            print(f"\n   Results Location:")
            print(f"     Base path: {output_base_path}")
            for strategy in successful_strategies:
                strategy_dir = output_base_path / strategy
                print(f"     {strategy.upper()} strategy: {strategy_dir}")
                if (strategy_dir / "financial_summary.html").exists():
                    print(
                        f"       Financial dashboard: {strategy_dir / 'financial_summary.html'}"
                    )
                if (strategy_dir / "strategy_deep_dive").exists():
                    print(
                        f"       Deep-dive dashboards: {strategy_dir / 'strategy_deep_dive'}/"
                    )

            # Create dashboards
            dashboard_paths = strategy_runner.create_dashboards(results)

            print(f"   ✅ Completed backtesting for {spp_loc}")

        except Exception as e:
            print(f"   ❌ Error during backtesting for {spp_loc}: {e}")
            import traceback

            traceback.print_exc()
            continue  # Continue with next settlement point

        print("-" * 60)

    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"*** {script_name} - END ***")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
