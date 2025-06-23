"""ERCOT Strategy Runner - Factory and Orchestrator for Trading Strategies.

This module provides the main interface for running backtests on ERCOT trading strategies,
following the same factory pattern used in the modeling components.
"""

import json
import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd

from src.backtesting.ercot.dashboard.financial_dashboard import (
    create_financial_dashboard,
)
from src.backtesting.ercot.dashboard.strategy_dashboard import (
    create_hours_overlay_dashboard,
)
from src.backtesting.ercot.dashboard.strategy_dashboard import create_strategy_dashboard
from src.backtesting.ercot.strategies.naive_strategy import NaiveStrategy


class ERCOTStrategyRunner:
    """Main orchestrator for ERCOT backtesting strategies.

    This class provides a unified interface for running backtests across multiple
    strategies, following the same pattern as the model trainers.
    """

    # Available strategy types and their implementations (factory pattern)
    STRATEGY_REGISTRY = {
        "naive": NaiveStrategy,
        # Future strategies can be added here:
        # "momentum": MomentumStrategy,
        # "mean_reversion": MeanReversionStrategy,
    }

    def __init__(
        self,
        settlement_point: str,
        output_base_path: str,
        target_hours: Optional[List[int]] = None,
    ):
        """Initialize strategy runner.

        Args:
            settlement_point: Settlement point name
            output_base_path: Base path for outputs (e.g., .../backtest/exp0)
            target_hours: Optional list of hours to filter for (1-24)
        """
        self.settlement_point = settlement_point
        self.output_base_path = Path(output_base_path)
        self.target_hours = target_hours

        # Create base output directory
        self.output_base_path.mkdir(parents=True, exist_ok=True)

        print(f"📁 Strategy runner initialized for {settlement_point}")
        print(f"   Base output path: {self.output_base_path}")

    def run_backtest(
        self,
        predictions_file: str,
        strategy_types: List[str] = None,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.5,
        target_hours: List[int] = None,
        **strategy_kwargs,
    ) -> Dict:
        """Run backtest for multiple strategies.

        Args:
            predictions_file: Path to predictions CSV file
            strategy_types: List of strategy types to run. If None, uses ['naive']
            initial_capital: Starting capital for strategies
            transaction_cost: Fixed cost per trade
            target_hours: List of hours to filter (e.g., [16, 17, 18] for peak hours).
                         If None, uses all hours.
            **strategy_kwargs: Additional parameters passed to strategies

        Returns:
            Dictionary with results for each strategy
        """
        if strategy_types is None:
            strategy_types = ["naive"]

        print(f"\n🚀 Running ERCOT backtests for {self.settlement_point}")
        print(f"   Predictions file: {predictions_file}")
        print(f"   Strategy types: {strategy_types}")
        if target_hours:
            print(f"   Target hours: {target_hours}")

        # Load predictions data
        predictions_df = self._load_predictions(predictions_file)

        # Filter by target hours if specified
        if target_hours:
            original_count = len(predictions_df)
            predictions_df = predictions_df[
                predictions_df["end_hour"].isin(target_hours)
            ]
            filtered_count = len(predictions_df)
            print(
                f"   Filtered to target hours: {original_count:,} → {filtered_count:,} predictions"
            )

            if filtered_count == 0:
                print(f"⚠️  No predictions found for target hours {target_hours}")
                return {}

        # Run each strategy
        all_results = {}
        for strategy_type in strategy_types:
            strategy_label = f"{strategy_type}"
            if target_hours:
                hours_str = "_".join(map(str, sorted(target_hours)))
                strategy_label = f"{strategy_type}_hours_{hours_str}"

            print(f"\n** Running {strategy_label} strategy")

            try:
                # Create strategy instance
                results = self._run_single_strategy(
                    strategy_type=strategy_type,
                    strategy_label=strategy_label,
                    predictions_df=predictions_df,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    **strategy_kwargs,
                )

                all_results[strategy_label] = results

            except Exception as e:
                print(f"❌ Error running {strategy_label} strategy: {e}")
                all_results[strategy_label] = None

        # Store results for dashboard creation
        self.backtest_results = all_results

        # Create dashboards
        self.create_dashboards(all_results)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _load_predictions(self, predictions_file: str) -> pd.DataFrame:
        """Load and validate predictions data."""
        print(f"📊 Loading predictions from: {predictions_file}")

        if not os.path.exists(predictions_file):
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

        df = pd.read_csv(predictions_file)

        # Validate required columns
        required_cols = ["utc_ts", "actual_dart_slt", "predicted_dart_slt", "end_hour"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"Missing required columns in predictions file: {missing_cols}"
            )

        # Convert timestamp
        df["utc_ts"] = pd.to_datetime(df["utc_ts"])

        print(f"   ✅ Loaded {len(df):,} predictions")
        print(f"   Date range: {df['utc_ts'].min()} to {df['utc_ts'].max()}")
        print(f"   Hours covered: {sorted(df['end_hour'].unique())}")

        return df

    def _run_single_strategy(
        self,
        strategy_type: str,
        strategy_label: str,
        predictions_df: pd.DataFrame,
        initial_capital: float,
        transaction_cost: float,
        **strategy_kwargs,
    ) -> Dict:
        """Run backtest for a single strategy."""

        if strategy_type not in self.STRATEGY_REGISTRY:
            available_types = list(self.STRATEGY_REGISTRY.keys())
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. Available: {available_types}"
            )

        # Create strategy instance with label for identification
        strategy_class = self.STRATEGY_REGISTRY[strategy_type]
        strategy = strategy_class(
            strategy_name=strategy_label,  # Use label instead of type for unique identification
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            output_dir=str(self.output_base_path),
            **strategy_kwargs,
        )

        # Run backtest
        results = strategy.execute_backtest(predictions_df)

        # Save strategy results
        self._save_strategy_results(strategy_label, results)

        return results

    def _save_strategy_results(self, strategy_name: str, results: Dict) -> str:
        """Save strategy results to files.

        Args:
            strategy_name: Name of the strategy
            results: Strategy results dictionary

        Returns:
            Path to saved results directory
        """
        # Create strategy-specific directory
        strategy_dir = self.output_base_path / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Save results JSON
        results_file = strategy_dir / "strategy_results.json"
        with open(results_file, "w") as f:
            # Convert any numpy types to Python types for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2, default=str)

        # Save trades CSV
        if results["trades"]:
            trades_df = pd.DataFrame(results["trades"])
            trades_file = strategy_dir / "trades.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"   💾 Trades saved: {trades_file}")

        print(f"💾 Strategy results saved to: {strategy_dir}")
        return str(strategy_dir)

    def create_dashboards(self, strategy_results: Dict[str, Dict]) -> Dict[str, str]:
        """Create dashboards for all strategies.

        Args:
            strategy_results: Dictionary mapping strategy names to results

        Returns:
            Dictionary mapping dashboard types to file paths
        """
        print(f"\n📊 Creating dashboards for {len(strategy_results)} strategies...")

        dashboard_paths = {}

        # Create financial summary dashboard (all strategies overlaid)
        if strategy_results:
            # Use first strategy's directory as the base for the financial summary
            first_strategy = list(strategy_results.keys())[0]
            financial_output = (
                self.output_base_path / first_strategy / "financial_summary.html"
            )

            create_financial_dashboard(
                strategy_results=strategy_results,
                output_path=str(financial_output),
                settlement_point=self.settlement_point,
            )
            dashboard_paths["financial_summary"] = str(financial_output)
            print(f"   ✅ Financial summary dashboard: {financial_output}")

        # Create strategy-specific deep-dive dashboards
        deep_dive_paths = []
        for strategy_name, results in strategy_results.items():
            if not results or not results["trades"]:
                continue

            strategy_dir = self.output_base_path / strategy_name

            # Create hours overlay dashboard directly in strategy folder
            hours_overlay_output = strategy_dir / "hours_overlay.html"
            create_hours_overlay_dashboard(
                strategy_results={strategy_name: results},
                output_path=str(hours_overlay_output),
                settlement_point=self.settlement_point,
            )

            dashboard_paths[f"{strategy_name}_hours_overlay"] = str(
                hours_overlay_output
            )
            print(f"   ✅ Strategy hours overlay: {hours_overlay_output}")

        return dashboard_paths

    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, "item"):  # numpy scalars
            return obj.item()
        elif hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()
        else:
            return obj

    def _print_summary(self, all_results: Dict):
        """Print summary of backtest results."""
        print(f"\n📈 Backtest Summary for {self.settlement_point}")
        print("=" * 60)

        for strategy_name, results in all_results.items():
            if results is None:
                print(f"❌ {strategy_name.upper()}: FAILED")
                continue

            metrics = results["performance_metrics"]
            print(f"✅ {strategy_name.upper()}:")
            print(f"   Total Return: {metrics.get('total_return_pct', 0):+.2f}%")
            print(f"   Win Rate: {metrics.get('win_rate_pct', 0):.1f}%")
            print(f"   Total Trades: {metrics.get('total_trades', 0):,}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"   Final Capital: ${metrics.get('final_capital', 0):,.2f}")
            print()

    def get_strategy_comparison(self) -> pd.DataFrame:
        """Get comparison table of strategy performance metrics."""
        if not self.backtest_results:
            return pd.DataFrame()

        comparison_data = []
        for strategy_name, results in self.backtest_results.items():
            if results and results["performance_metrics"]:
                metrics = results["performance_metrics"].copy()
                metrics["strategy"] = strategy_name
                comparison_data.append(metrics)

        return pd.DataFrame(comparison_data)
