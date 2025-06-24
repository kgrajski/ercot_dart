"""Trade Analytics - Single Source of Truth for ERCOT Backtesting Metrics.

This module provides centralized calculation of all trading metrics, ensuring
consistent results across dashboards and reports. Dashboards should ONLY
call these functions for data, never recalculate metrics themselves.

Architecture Principles:
- Single Source of Truth: All metrics calculated here
- Separation of Concerns: Dashboards only do visualization
- Immutable Results: Return copies to prevent modification
- Comprehensive Coverage: All metrics needed by any visualization
"""

from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd


class ERCOTTradeAnalytics:
    """Centralized analytics engine for ERCOT trading strategy results.

    This class provides the single source of truth for all metrics calculations.
    Dashboard components should never recalculate metrics - they should only
    consume the pre-calculated results from this class.
    """

    def __init__(self, strategy_results: Dict):
        """Initialize analytics with strategy results.

        Args:
            strategy_results: Raw strategy results from backtest execution
        """
        self.strategy_results = strategy_results
        self._trades_df_cache = {}
        self._portfolio_df_cache = {}
        self._metrics_cache = {}

    def get_trades_dataframe(self, strategy_name: str) -> pd.DataFrame:
        """Get trades DataFrame with standardized columns and types.

        Returns:
            Clean DataFrame with all trades for the strategy
        """
        if strategy_name not in self._trades_df_cache:
            if (
                strategy_name not in self.strategy_results
                or not self.strategy_results[strategy_name]
                or not self.strategy_results[strategy_name]["trades"]
            ):
                return pd.DataFrame()

            df = pd.DataFrame(self.strategy_results[strategy_name]["trades"])

            # Standardize timestamp columns
            df["entry_time"] = pd.to_datetime(df["entry_time"])
            if "exit_time" in df.columns:
                df["exit_time"] = pd.to_datetime(df["exit_time"])

            # Ensure required columns exist with proper types
            required_columns = {
                "entry_hour": "int64",
                "pnl": "float64",
                "actual_dart_slt": "int64",
                "correct_prediction": "bool",
                "direction": "object",
                "position_size": "float64",
            }

            for col, dtype in required_columns.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)

            self._trades_df_cache[strategy_name] = df.copy()

        return self._trades_df_cache[strategy_name].copy()

    def get_hourly_metrics(self, strategy_name: str) -> pd.DataFrame:
        """Calculate comprehensive hourly metrics for a strategy.

        This is the SINGLE SOURCE OF TRUTH for hourly analysis.
        All dashboards should use this instead of recalculating.

        Returns:
            DataFrame with metrics for each hour (1-24)
        """
        cache_key = f"{strategy_name}_hourly_metrics"
        if cache_key not in self._metrics_cache:
            trades_df = self.get_trades_dataframe(strategy_name)
            if trades_df.empty:
                return pd.DataFrame()

            # Group by hour and calculate all metrics
            hourly_metrics = []

            for hour in range(1, 25):
                hour_trades = trades_df[trades_df["entry_hour"] == hour]

                if hour_trades.empty:
                    continue

                metrics = {
                    "hour": hour,
                    "total_trades": len(hour_trades),
                    "total_pnl": hour_trades["pnl"].sum(),
                    "avg_pnl": hour_trades["pnl"].mean(),
                    "winning_trades": (hour_trades["pnl"] > 0).sum(),
                    "losing_trades": (hour_trades["pnl"] < 0).sum(),
                    "win_rate_pct": (hour_trades["pnl"] > 0).mean() * 100,
                    # SINGLE SOURCE OF TRUTH: Prediction accuracy
                    "prediction_accuracy_pct": hour_trades["correct_prediction"].mean()
                    * 100,
                    "max_win": hour_trades["pnl"].max(),
                    "max_loss": hour_trades["pnl"].min(),
                    "pnl_std": hour_trades["pnl"].std(),
                    "cumulative_pnl": hour_trades["pnl"].cumsum().iloc[-1],
                }

                # Risk metrics
                if len(hour_trades) > 1:
                    returns = hour_trades["pnl"] / hour_trades["position_size"]
                    if returns.std() > 0:
                        metrics["sharpe_ratio"] = returns.mean() / returns.std()
                    else:
                        metrics["sharpe_ratio"] = 0.0
                else:
                    metrics["sharpe_ratio"] = 0.0

                hourly_metrics.append(metrics)

            self._metrics_cache[cache_key] = pd.DataFrame(hourly_metrics)

        return self._metrics_cache[cache_key].copy()

    def get_weekly_accuracy_by_hour(
        self, strategy_name: str, hour: int
    ) -> pd.DataFrame:
        """Calculate weekly prediction accuracy for a specific hour.

        Args:
            strategy_name: Strategy to analyze
            hour: Hour to focus on (1-24)

        Returns:
            DataFrame with weekly accuracy percentages
        """
        trades_df = self.get_trades_dataframe(strategy_name)
        hour_trades = trades_df[trades_df["entry_hour"] == hour].copy()

        if hour_trades.empty:
            return pd.DataFrame()

        # Calculate weekly accuracy using the SINGLE SOURCE OF TRUTH
        weekly_accuracy = (
            hour_trades.set_index("entry_time")
            .resample("W")["correct_prediction"]  # Use the authoritative field
            .agg(["mean", "count"])
            .reset_index()
        )

        weekly_accuracy.columns = ["week_start", "accuracy_pct", "trade_count"]
        weekly_accuracy["accuracy_pct"] *= 100  # Convert to percentage

        return weekly_accuracy

    def get_cumulative_returns_by_hour(
        self, strategy_name: str, hour: int
    ) -> pd.DataFrame:
        """Calculate cumulative returns time series for a specific hour.

        Args:
            strategy_name: Strategy to analyze
            hour: Hour to focus on (1-24)

        Returns:
            DataFrame with timestamp and cumulative returns
        """
        trades_df = self.get_trades_dataframe(strategy_name)
        hour_trades = trades_df[trades_df["entry_hour"] == hour].copy()

        if hour_trades.empty:
            return pd.DataFrame()

        # Sort by time and calculate cumulative returns
        hour_trades_sorted = hour_trades.sort_values("entry_time")
        hour_trades_sorted["cumulative_returns"] = hour_trades_sorted["pnl"].cumsum()

        return hour_trades_sorted[
            ["entry_time", "cumulative_returns", "pnl", "correct_prediction"]
        ].copy()

    def get_weekly_performance_by_hour(
        self, strategy_name: str, hour: int
    ) -> pd.DataFrame:
        """Calculate weekly P&L performance for a specific hour.

        Args:
            strategy_name: Strategy to analyze
            hour: Hour to focus on (1-24)

        Returns:
            DataFrame with weekly P&L totals
        """
        trades_df = self.get_trades_dataframe(strategy_name)
        hour_trades = trades_df[trades_df["entry_hour"] == hour].copy()

        if hour_trades.empty or "week_num" not in hour_trades.columns:
            return pd.DataFrame()

        weekly_pnl = (
            hour_trades.groupby("week_num")["pnl"]
            .agg(["sum", "count", "mean"])
            .reset_index()
        )

        weekly_pnl.columns = ["week_num", "total_pnl", "trade_count", "avg_pnl"]

        return weekly_pnl

    def get_hour_statistics_summary(self, strategy_name: str, hour: int) -> Dict:
        """Get comprehensive statistics summary for a specific hour.

        This provides the authoritative summary stats for dashboard annotations.

        Args:
            strategy_name: Strategy to analyze
            hour: Hour to focus on (1-24)

        Returns:
            Dictionary with all key statistics
        """
        hourly_metrics = self.get_hourly_metrics(strategy_name)
        hour_metrics = hourly_metrics[hourly_metrics["hour"] == hour]

        if hour_metrics.empty:
            return {}

        metrics = hour_metrics.iloc[0].to_dict()

        # Add formatted versions for display
        metrics.update(
            {
                "total_trades_formatted": f"{metrics['total_trades']:,}",
                "win_rate_formatted": f"{metrics['win_rate_pct']:.1f}%",
                "avg_pnl_formatted": f"${metrics['avg_pnl']:.2f}",
                "total_pnl_formatted": f"${metrics['total_pnl']:.2f}",
                "prediction_accuracy_formatted": f"{metrics['prediction_accuracy_pct']:.1f}%",
            }
        )

        return metrics

    def get_all_hours_summary(self, strategy_name: str) -> pd.DataFrame:
        """Get summary of all hours for overview dashboards.

        Returns:
            DataFrame with key metrics for all hours (1-24)
        """
        return self.get_hourly_metrics(strategy_name)

    def validate_data_consistency(self) -> Dict[str, List[str]]:
        """Validate that all data is internally consistent.

        Returns:
            Dictionary mapping strategy names to list of validation errors
        """
        validation_errors = {}

        for strategy_name in self.strategy_results.keys():
            errors = []
            trades_df = self.get_trades_dataframe(strategy_name)

            if trades_df.empty:
                errors.append("No trades data available")
                continue

            # Check for missing required fields
            required_fields = [
                "correct_prediction",
                "pnl",
                "actual_dart_slt",
                "entry_hour",
            ]
            missing_fields = [f for f in required_fields if f not in trades_df.columns]
            if missing_fields:
                errors.append(f"Missing required fields: {missing_fields}")

            # Check for data type consistency
            if "correct_prediction" in trades_df.columns:
                if not trades_df["correct_prediction"].dtype == bool:
                    errors.append("correct_prediction field is not boolean type")

            # Check for logical consistency
            if len(trades_df) > 0:
                accuracy = trades_df["correct_prediction"].mean()
                total_pnl = trades_df["pnl"].sum()

                # Flag suspiciously high accuracy with negative returns
                if accuracy > 0.9 and total_pnl < -100:
                    errors.append(
                        f"High accuracy ({accuracy:.1%}) with significant losses (${total_pnl:.2f}) - possible calculation error"
                    )

            validation_errors[strategy_name] = errors

        return validation_errors


def create_analytics_engine(strategy_results: Dict) -> ERCOTTradeAnalytics:
    """Factory function to create analytics engine with validation.

    Args:
        strategy_results: Raw strategy results from backtest

    Returns:
        Configured analytics engine

    Raises:
        ValueError: If strategy results are invalid
    """
    if not strategy_results:
        raise ValueError("No strategy results provided")

    analytics = ERCOTTradeAnalytics(strategy_results)

    # Validate data consistency
    validation_errors = analytics.validate_data_consistency()

    # Report any validation issues
    for strategy_name, errors in validation_errors.items():
        if errors:
            print(f"⚠️  Data validation warnings for {strategy_name}:")
            for error in errors:
                print(f"   - {error}")

    return analytics
