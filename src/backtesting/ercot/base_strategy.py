"""Base Strategy Class for ERCOT Trading Strategies.

This module provides the abstract base class for all ERCOT trading strategies,
following the same patterns used in the modeling components.
"""

import os
import traceback
import warnings
from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from src.features.utils.utils import inverse_signed_log_transform


class StrategyValidationError(Exception):
    """Custom exception for strategy validation failures."""

    pass


class BaseStrategy(ABC):
    """Abstract base class for ERCOT trading strategies.

    This class provides the template method pattern for strategy execution
    and defines the interface that all concrete strategies must implement.

    Key concepts:
    - Strategies operate on hourly predictions from classification models
    - Each trade represents a bet on DART direction (positive/negative)
    - Profits/losses are realized based on actual DART movements
    - Transaction costs are applied to each trade
    """

    def __init__(
        self,
        strategy_name: str,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.5,  # Fixed cost per trade (Wang et al. style)
        output_dir: str = None,
        enable_validation: bool = True,  # Enable comprehensive validation checks
        enable_debug_logging: bool = True,  # Enable detailed debug output
    ):
        """Initialize the base strategy.

        Args:
            strategy_name: Name of the strategy for identification
            initial_capital: Starting capital for the strategy
            transaction_cost: Fixed cost per trade in dollars
            output_dir: Directory for saving strategy results
            enable_validation: Whether to enable validation checks
            enable_debug_logging: Whether to enable debug logging
        """
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.output_dir = output_dir or "."
        self.enable_validation = enable_validation
        self.enable_debug_logging = enable_debug_logging

        # Results storage
        self.trades = []  # List of executed trades
        self.portfolio_values = []  # Time series of portfolio value
        self.performance_metrics = {}  # Summary metrics

        # Current state
        self.current_capital = initial_capital
        self.current_position_size = 0  # Current open position size
        self.current_position_entry_price = 0  # Normalized entry price
        self.current_position_entry_hour = None  # Normalized entry hour
        self.current_position_entry_time = None  # Normalized entry time
        self.current_position_entry_prediction = (
            None  # Store entry prediction for trade analysis
        )

        # Validation and monitoring
        self.validation_errors = []  # Track validation errors
        self.method_call_counts = {}  # Track method calls for monitoring

        if self.enable_validation:
            self._validate_initialization()

    def _validate_initialization(self):
        """Validate strategy initialization parameters."""
        errors = []

        if self.initial_capital <= 0:
            errors.append(
                f"Initial capital must be positive, got {self.initial_capital}"
            )

        if self.transaction_cost < 0:
            errors.append(
                f"Transaction cost cannot be negative, got {self.transaction_cost}"
            )

        if self.transaction_cost >= self.initial_capital:
            errors.append(
                f"Transaction cost ({self.transaction_cost}) cannot exceed initial capital ({self.initial_capital})"
            )

        if not self.strategy_name or not isinstance(self.strategy_name, str):
            errors.append(
                f"Strategy name must be a non-empty string, got {self.strategy_name}"
            )

        if errors:
            error_msg = (
                f"🚨 STRATEGY INITIALIZATION FAILED for {self.strategy_name}:\n"
                + "\n".join(f"  ❌ {error}" for error in errors)
            )
            raise StrategyValidationError(error_msg)

        if self.enable_debug_logging:
            print(f"✅ Strategy '{self.strategy_name}' initialized successfully")
            print(f"   💰 Initial Capital: ${self.initial_capital:,.2f}")
            print(f"   💸 Transaction Cost: ${self.transaction_cost:.2f}")

    def _track_method_call(self, method_name: str):
        """Track method calls for monitoring."""
        self.method_call_counts[method_name] = (
            self.method_call_counts.get(method_name, 0) + 1
        )

    def _validate_signal(self, signal: Dict, method_context: str = "") -> None:
        """Validate trading signal structure."""
        if not self.enable_validation:
            return

        if not isinstance(signal, dict):
            raise StrategyValidationError(
                f"🚨 SIGNAL VALIDATION FAILED {method_context}: "
                f"Signal must be a dictionary, got {type(signal)}"
            )

        required_keys = ["direction", "confidence"]
        missing_keys = [key for key in required_keys if key not in signal]
        if missing_keys:
            raise StrategyValidationError(
                f"🚨 SIGNAL VALIDATION FAILED {method_context}: "
                f"Missing required keys: {missing_keys}. Got keys: {list(signal.keys())}"
            )

        if signal["direction"] not in ["long", "short"]:
            raise StrategyValidationError(
                f"🚨 SIGNAL VALIDATION FAILED {method_context}: "
                f"Direction must be 'long' or 'short', got '{signal['direction']}'"
            )

        confidence = signal["confidence"]
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            raise StrategyValidationError(
                f"🚨 SIGNAL VALIDATION FAILED {method_context}: "
                f"Confidence must be a number between 0 and 1, got {confidence}"
            )

    def _validate_position_size(
        self, position_size: float, available_capital: float, method_context: str = ""
    ) -> None:
        """Validate position size."""
        if not self.enable_validation:
            return

        if not isinstance(position_size, (int, float)):
            raise StrategyValidationError(
                f"🚨 POSITION SIZE VALIDATION FAILED {method_context}: "
                f"Position size must be a number, got {type(position_size)}"
            )

        if position_size < 0:
            raise StrategyValidationError(
                f"🚨 POSITION SIZE VALIDATION FAILED {method_context}: "
                f"Position size cannot be negative, got {position_size}"
            )

        if position_size > available_capital:
            raise StrategyValidationError(
                f"🚨 POSITION SIZE VALIDATION FAILED {method_context}: "
                f"Position size ({position_size}) exceeds available capital ({available_capital})"
            )

    def _safe_method_call(self, method_func, method_name: str, *args, **kwargs):
        """Safely call a method with comprehensive error handling."""
        self._track_method_call(method_name)

        try:
            if self.enable_debug_logging:
                print(
                    f"🔧 Calling {method_name} with args: {len(args)} positional, {len(kwargs)} keyword"
                )

            result = method_func(*args, **kwargs)

            if self.enable_debug_logging:
                print(f"✅ {method_name} completed successfully")

            return result

        except Exception as e:
            error_msg = (
                f"🚨 CRITICAL ERROR in {self.strategy_name}.{method_name}:\n"
                f"   ❌ Error: {str(e)}\n"
                f"   📍 Args: {args}\n"
                f"   📍 Kwargs: {kwargs}\n"
                f"   📊 Method call count: {self.method_call_counts.get(method_name, 0)}\n"
                f"   🔍 Full traceback:\n{traceback.format_exc()}"
            )

            print(error_msg)
            self.validation_errors.append(
                {
                    "method": method_name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "args": str(args),
                    "kwargs": str(kwargs),
                }
            )

            # Re-raise with additional context
            raise StrategyValidationError(
                f"Method {method_name} failed: {str(e)}"
            ) from e

    @abstractmethod
    def generate_signal(self, prediction_row: pd.Series) -> Dict:
        """Generate trading signal from prediction data.

        Args:
            prediction_row: Single row from predictions DataFrame

        Returns:
            Dictionary with signal information:
            - direction: "long" or "short"
            - confidence: 0.0 to 1.0
            - size_multiplier: Optional position size multiplier
        """
        pass

    @abstractmethod
    def calculate_position_size(self, signal: Dict, available_capital: float) -> float:
        """Calculate position size for the trade.

        Args:
            signal: Signal dictionary from generate_signal()
            available_capital: Available capital for trading

        Returns:
            Position size in dollars
        """
        pass

    def execute_backtest(self, predictions_df: pd.DataFrame) -> Dict:
        """Execute backtest with operational trading model.

        Enhanced to handle probability scores and raw DART values:
        - predicted_dart_slt: Binary prediction (0/1) - main signal
        - predicted_prob_class_0/1: XGBoost probability scores
        - prediction_confidence: Model confidence (max probability)
        - actual_dart_slt_raw: Raw DART values before classification
        """
        print(f"🚀 Starting backtest with {len(predictions_df)} predictions...")

        # Validate predictions DataFrame
        if self.enable_validation:
            self._validate_predictions_dataframe(predictions_df)

        # Reset strategy state
        self._reset_state()

        # Sort by UTC timestamp to ensure chronological processing
        predictions_df = predictions_df.sort_values("utc_ts").reset_index(drop=True)

        # Initialize trades list
        trades = []
        processing_errors = []

        # Process each prediction in chronological order
        for i, row in predictions_df.iterrows():
            try:
                is_last_prediction = i == len(predictions_df) - 1

                # Close existing position if any
                if self.current_position_size != 0:
                    trade = self._close_position(row, trades)
                    if trade:
                        trades.append(trade)

                # Open new position (unless this is the last prediction)
                if not is_last_prediction:
                    # Generate signal with validation
                    signal = self._safe_method_call(
                        self.generate_signal, "generate_signal", row
                    )
                    self._validate_signal(signal, f"at row {i}")

                    # Calculate position size with validation
                    position_size = self._safe_method_call(
                        self.calculate_position_size,
                        "calculate_position_size",
                        signal,
                        self.current_capital,
                    )
                    self._validate_position_size(
                        position_size, self.current_capital, f"at row {i}"
                    )

                    if (
                        position_size != 0
                        and abs(position_size) <= self.current_capital
                    ):
                        # Record position entry
                        self.current_position_size = position_size
                        self.current_position_entry_price = 1.0  # Normalized entry
                        self.current_position_entry_hour = row["end_hour"]
                        self.current_position_entry_time = row["utc_ts"]
                        self.current_position_entry_prediction = row.get(
                            "predicted_dart_slt", 0
                        )  # Store binary prediction

                        # Update capital
                        self.current_capital -= abs(position_size)  # Transaction cost

                        if self.enable_debug_logging:
                            print(
                                f"  📈 Opened position: ${position_size:+.0f} @ hour {row['end_hour']} ({row['utc_ts']})"
                            )

            except Exception as e:
                error_info = {
                    "row_index": i,
                    "timestamp": row.get("utc_ts"),
                    "hour": row.get("end_hour"),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                processing_errors.append(error_info)

                print(f"🚨 ERROR processing row {i}: {str(e)}")
                if self.enable_debug_logging:
                    print(f"   Full traceback: {traceback.format_exc()}")

                # Continue processing other rows
                continue

            # Record portfolio value at this point in time (after each prediction)
            self.portfolio_values.append(
                {
                    "utc_ts": row["utc_ts"],
                    "end_hour": row["end_hour"],
                    "portfolio_value": self.current_capital,
                    "week_num": row.get("week_num"),
                    "week_description": row.get("week_description"),
                    "has_position": self.current_position_size != 0,
                }
            )

        # Check for critical errors
        if processing_errors:
            print(
                f"⚠️  {len(processing_errors)} processing errors occurred during backtest"
            )
            if (
                len(processing_errors) > len(predictions_df) * 0.1
            ):  # More than 10% errors
                raise StrategyValidationError(
                    f"🚨 BACKTEST FAILED: Too many processing errors ({len(processing_errors)}/{len(predictions_df)})"
                )

        # Assign trades to instance variable for metrics calculation
        self.trades = trades

        # Calculate final performance metrics
        try:
            performance_metrics = self._calculate_performance_metrics()
        except Exception as e:
            print(f"🚨 ERROR calculating performance metrics: {str(e)}")
            performance_metrics = {"error": str(e)}

        # Validation summary
        if self.enable_validation and self.enable_debug_logging:
            self._print_validation_summary()

        if performance_metrics.get("total_trades", 0) > 0:
            print(
                f"✅ Backtest completed: {performance_metrics['total_trades']} trades, "
                f"{performance_metrics['total_return_pct']:+.2f}% return"
            )
        else:
            print("⚠️  Backtest completed with no trades executed")

        return {
            "trades": trades,
            "performance_metrics": performance_metrics,
            "final_portfolio_value": self.current_capital
            + sum(trade.get("pnl", 0) for trade in trades),
            "processing_errors": processing_errors,
            "validation_errors": self.validation_errors,
            "method_call_counts": self.method_call_counts,
        }

    def _validate_predictions_dataframe(self, predictions_df: pd.DataFrame) -> None:
        """Validate the predictions DataFrame structure."""
        required_columns = [
            "utc_ts",
            "predicted_dart_slt",
            "actual_dart_slt",
            "end_hour",
        ]
        missing_columns = [
            col for col in required_columns if col not in predictions_df.columns
        ]

        if missing_columns:
            raise StrategyValidationError(
                f"🚨 PREDICTIONS DATAFRAME VALIDATION FAILED: "
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {list(predictions_df.columns)}"
            )

        if len(predictions_df) == 0:
            raise StrategyValidationError(
                "🚨 PREDICTIONS DATAFRAME VALIDATION FAILED: Empty DataFrame"
            )

        # Check for data quality issues
        null_counts = predictions_df[required_columns].isnull().sum()
        if null_counts.any():
            print(f"⚠️  WARNING: Null values found in required columns:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"   {col}: {count} null values")

        if self.enable_debug_logging:
            print(
                f"✅ Predictions DataFrame validated: {len(predictions_df)} rows, {len(predictions_df.columns)} columns"
            )

    def _print_validation_summary(self) -> None:
        """Print a summary of validation results."""
        print(f"\n📊 VALIDATION SUMMARY for {self.strategy_name}:")
        print(f"   🔧 Method Calls: {dict(self.method_call_counts)}")
        print(f"   ❌ Validation Errors: {len(self.validation_errors)}")

        if self.validation_errors:
            print("   🚨 Error Details:")
            for i, error in enumerate(
                self.validation_errors[-3:], 1
            ):  # Show last 3 errors
                print(f"      {i}. {error['method']}: {error['error']}")
        else:
            print("   ✅ No validation errors detected")

    def _reset_state(self):
        """Reset strategy state for new backtest."""
        self.current_capital = self.initial_capital
        self.current_position_size = 0
        self.current_position_entry_price = 0
        self.current_position_entry_hour = None
        self.current_position_entry_time = None
        self.current_position_entry_prediction = None
        self.trades = []
        self.portfolio_values = []
        self.performance_metrics = {}

        # Reset monitoring state
        self.validation_errors = []
        self.method_call_counts = {}

        if self.enable_debug_logging:
            print(f"🔄 Strategy state reset for {self.strategy_name}")
            print(f"   💰 Capital: ${self.current_capital:,.2f}")
            print(f"   📊 Position: {self.current_position_size}")

        # Validate state after reset
        if self.enable_validation:
            assert (
                self.current_capital == self.initial_capital
            ), f"Capital reset failed: {self.current_capital} != {self.initial_capital}"
            assert (
                self.current_position_size == 0
            ), f"Position reset failed: {self.current_position_size} != 0"
            assert (
                len(self.trades) == 0
            ), f"Trades reset failed: {len(self.trades)} != 0"

    def _process_prediction(
        self, row: pd.Series, idx: int, is_last_prediction: bool = False
    ):
        """Process a single prediction following operational trading model.

        Operational flow:
        1. Close any existing position using actual DART results
        2. Open new position based on current prediction (unless last prediction)
        3. Record portfolio state

        Args:
            row: Current prediction row
            idx: Row index for tracking
            is_last_prediction: True if this is the final prediction (don't open new position)
        """
        # Step 1: Close existing position if any (settle previous trade)
        if self.current_position_size != 0:
            trade = self._close_position(row, self.trades)
            if trade:
                self.trades.append(trade)

        # Step 2: Open new position based on current prediction (unless last)
        if not is_last_prediction:
            # Generate trading signal
            signal = self.generate_signal(row)

            # Calculate position size
            position_size = self.calculate_position_size(signal, self.current_capital)

            # Open new position if signal indicates trade and we have capital
            if (
                position_size > 0
                and signal.get("direction")
                and self.current_capital > self.transaction_cost
            ):
                self.current_position_size = position_size
                self.current_position_entry_price = 1.0  # Normalized entry
                self.current_position_entry_hour = row["end_hour"]
                self.current_position_entry_time = row["utc_ts"]
                self.current_position_entry_prediction = row.get(
                    "predicted_dart_slt", 0
                )  # Store binary prediction

                # Update capital
                self.current_capital -= abs(position_size)  # Transaction cost

                print(
                    f"  📈 Opened position: ${position_size:+.0f} @ hour {row['end_hour']} ({row['utc_ts']})"
                )

        # Step 3: Record portfolio value at this point in time
        self.portfolio_values.append(
            {
                "utc_ts": row["utc_ts"],
                "end_hour": row["end_hour"],
                "portfolio_value": self.current_capital,
                "week_num": row.get("week_num"),
                "week_description": row.get("week_description"),
                "has_position": self.current_position_size != 0,
            }
        )

    def _close_position(self, row: pd.Series, trades: List[Dict]) -> Dict:
        """Close the current position and calculate P&L.

        Args:
            row: Current prediction row containing actual DART results
            trades: List of completed trades (for context)

        Returns:
            Dict: Trade record if position was closed, None otherwise
        """
        if self.current_position_size == 0:
            return None

        # Get actual DART movement for P&L calculation
        actual_dart = row["actual_dart_slt"]  # 0 or 1 from classification

        # Determine P&L based on position direction and actual outcome
        # Long wins when actual_dart = 1 (RT > DA), Short wins when actual_dart = 0 (RT ≤ DA)
        if self.current_position_size > 0:  # Long position
            pnl = (
                abs(self.current_position_size)
                if actual_dart == 1
                else -abs(self.current_position_size)
            )
        else:  # Short position
            pnl = (
                abs(self.current_position_size)
                if actual_dart == 0
                else -abs(self.current_position_size)
            )

        # Apply transaction cost on position close
        pnl -= self.transaction_cost

        # Update capital with realized P&L
        self.current_capital += pnl

        # Get position direction for recording
        direction = "long" if self.current_position_size > 0 else "short"

        # Create trade record
        trade = {
            "entry_time": self.current_position_entry_time,
            "exit_time": row["utc_ts"],
            "entry_hour": self.current_position_entry_hour,
            "exit_hour": row["end_hour"],
            "direction": direction,
            "position_size": abs(self.current_position_size),
            "actual_dart_slt": actual_dart,
            "pnl": pnl,
            "week_num": row.get("week_num"),
            "capital_after": self.current_capital,
            # Additional analysis fields
            "correct_prediction": (self.current_position_size > 0 and actual_dart == 1)
            or (self.current_position_size < 0 and actual_dart == 0),
            "trade_duration_hours": self._calculate_trade_duration(
                self.current_position_entry_time, row["utc_ts"]
            ),
            "entry_prediction": self.current_position_entry_prediction,
        }

        # Debug output
        direction_emoji = "📈" if direction == "long" else "📉"
        result_emoji = "✅" if pnl > 0 else "❌"
        print(
            f"  {result_emoji} {direction_emoji} Closed {direction}: ${pnl:+.2f} (actual: {actual_dart}) @ hour {row['end_hour']}"
        )

        # Clear position
        self.current_position_size = 0
        self.current_position_entry_price = 0
        self.current_position_entry_hour = None
        self.current_position_entry_time = None
        self.current_position_entry_prediction = None

        return trade

    def _calculate_trade_duration(self, entry_time, exit_time) -> float:
        """Calculate trade duration in hours."""
        try:
            if isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)
            if isinstance(exit_time, str):
                exit_time = pd.to_datetime(exit_time)
            return (exit_time - entry_time).total_seconds() / 3600.0
        except:
            return 0.0

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate strategy performance metrics.

        This method computes all financial metrics displayed in the dashboard's financial summary.
        The metrics follow standard quantitative finance conventions adapted for electricity trading.

        Calculation Details:
        ====================

        1. **Total Return (%)**:
           - Formula: (Total P&L / Initial Capital) × 100
           - Represents the percentage return on invested capital
           - Example: $262 profit on $10,000 = +2.62%

        2. **Win Rate (%)**:
           - Formula: (Number of Profitable Trades / Total Trades) × 100
           - Percentage of trades that generated positive P&L
           - Example: 158 winning trades out of 289 total = 54.7%

        3. **Total Trades**:
           - Simple count of completed round-trip trades
           - Each trade involves: open position → daily settlement → close position

        4. **Sharpe Ratio**:
           - Formula: (Mean Portfolio Return / Std Dev Portfolio Return) × √(365 × 24)
           - Annualized risk-adjusted return metric
           - Uses portfolio value percentage changes, annualized for electricity markets
           - ERCOT operates 24/7, 365 days per year (not 252 trading days like financial markets)
           - Higher values indicate better risk-adjusted performance

        5. **Max Drawdown (%)**:
           - Formula: Min((Portfolio Value - Running Maximum) / Running Maximum) × 100
           - Largest peak-to-trough decline in portfolio value
           - Measures worst-case downside risk
           - Example: -2.62% means portfolio fell 2.62% from its highest point

        6. **Final Capital**:
           - End-of-backtest portfolio value in dollars
           - Initial Capital + Total P&L - Transaction Costs

        P&L Calculation Logic:
        =====================
        Each trade's P&L is calculated as follows:

        - **Long Position**: Bet that Real-Time price > Day-Ahead price
          - Win: +$1.00 when actual_dart_slt = 1 (RT > DA)
          - Lose: -$1.00 when actual_dart_slt = 0 (RT ≤ DA)

        - **Short Position**: Bet that Real-Time price ≤ Day-Ahead price
          - Win: +$1.00 when actual_dart_slt = 0 (RT ≤ DA)
          - Lose: -$1.00 when actual_dart_slt = 1 (RT > DA)

        - **Transaction Costs**: $0.50 deducted on both entry and exit
          - Total per round-trip trade: $1.00
          - Net P&L = Base P&L - Transaction Costs

        Trading Model:
        ==============
        - Each hourly prediction opens a new $1 position
        - Previous position automatically closes at settlement
        - No position sizing or risk management (naive strategy)
        - Capital requirement: sufficient funds for next trade + transaction costs

        Returns:
            Dict containing all calculated performance metrics
        """
        if not self.trades:
            return {}

        trades_df = pd.DataFrame(self.trades)
        portfolio_df = pd.DataFrame(self.portfolio_values)

        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["pnl"] > 0])
        losing_trades = len(trades_df[trades_df["pnl"] < 0])

        total_pnl = trades_df["pnl"].sum()
        total_return_pct = (total_pnl / self.initial_capital) * 100

        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # Risk metrics
        returns = portfolio_df["portfolio_value"].pct_change().dropna()

        sharpe_ratio = 0
        max_drawdown = 0
        if len(returns) > 1:
            sharpe_ratio = (
                (returns.mean() / returns.std()) * np.sqrt(365 * 24)
                if returns.std() > 0
                else 0
            )  # Annualized

            # Calculate max drawdown
            portfolio_values = portfolio_df["portfolio_value"]
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max
            max_drawdown = drawdown.min() * 100  # Convert to percentage

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": win_rate,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "final_capital": self.current_capital,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "avg_trade_pnl": trades_df["pnl"].mean() if total_trades > 0 else 0,
            "total_transaction_costs": total_trades
            * 2
            * self.transaction_cost,  # 2x for entry/exit
        }

    def save_results(self, results: Dict):
        """Save strategy results to files."""
        # Create strategy output directory
        strategy_dir = os.path.join(self.output_dir, f"strategy_{self.strategy_name}")
        os.makedirs(strategy_dir, exist_ok=True)

        # Save trades
        if results["trades"]:
            trades_df = pd.DataFrame(results["trades"])
            trades_df.to_csv(os.path.join(strategy_dir, "trades.csv"), index=False)

        # Save portfolio values
        if results["portfolio_values"]:
            portfolio_df = pd.DataFrame(results["portfolio_values"])
            portfolio_df.to_csv(
                os.path.join(strategy_dir, "portfolio_values.csv"), index=False
            )

        # Save performance metrics
        metrics_df = pd.DataFrame([results["performance_metrics"]])
        metrics_df.to_csv(
            os.path.join(strategy_dir, "performance_metrics.csv"), index=False
        )

        print(f"💾 Strategy results saved to: {strategy_dir}")
