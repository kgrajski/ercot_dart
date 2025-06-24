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
    """Abstract base class for ERCOT trading strategies using per-hour independent betting.

    This class implements a per-hour independent betting model where each prediction
    generates exactly one bet with even money payouts. This approach treats each
    hourly prediction as a standalone betting opportunity rather than a traditional
    position-holding strategy.

    Betting Model:
    ==============
    - **Independence**: Each row/hour is processed independently with no state carryover
    - **Even Money**: Win = +$1.00 profit, Lose = -$1.00 loss (before transaction costs)
    - **Transaction Costs**: $0.10 per transaction × 2 = $0.20 total per bet
    - **Net Payouts**: Win = +$0.80, Lose = -$1.20 (after transaction costs)

    Key Concepts:
    =============
    - Each hourly prediction triggers one complete bet cycle (place → resolve → settle)
    - Profits/losses are realized immediately based on actual DART movements
    - No position holding across multiple hours - each bet is self-contained
    - Capital management through independent bet sizing per prediction

    Data Structure:
    ===============
    Each prediction row contains:
    - predicted_dart_slt: Binary prediction (0=short, 1=long)
    - actual_dart_slt: Actual outcome (0=RT≤DA, 1=RT>DA)
    - Probability scores and confidence metrics for analysis

    This model is well-suited for backtesting classification models on electricity
    markets where each prediction represents a distinct trading opportunity.
    """

    def __init__(
        self,
        strategy_name: str,
        transaction_cost: float,  # Required parameter - no default
        initial_capital: float = 10000.0,
        output_dir: str = None,
        enable_validation: bool = True,  # Enable comprehensive validation checks
        enable_debug_logging: bool = True,  # Enable detailed debug output
    ):
        """Initialize the base strategy.

        Args:
            strategy_name: Name of the strategy for identification
            transaction_cost: Fixed cost per transaction (entry or exit) in dollars
            initial_capital: Starting capital for the strategy
            output_dir: Directory for saving strategy results
            enable_validation: Whether to enable validation checks
            enable_debug_logging: Whether to enable debug logging
        """
        self.strategy_name = strategy_name
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.output_dir = output_dir or "."
        self.enable_validation = enable_validation
        self.enable_debug_logging = enable_debug_logging

        # Results storage
        self.trades = []  # List of executed trades
        self.portfolio_values = []  # Time series of portfolio value
        self.performance_metrics = {}  # Summary metrics

        # Current state
        self.current_capital = initial_capital

        # Validation and monitoring
        self.validation_errors = []  # Track validation errors
        self.method_call_counts = {}  # Track method calls for monitoring

        if self.enable_validation:
            self._validate_initialization()

        if self.enable_debug_logging:
            print(f"✅ Strategy '{self.strategy_name}' initialized successfully")
            print(f"   💰 Initial Capital: ${self.initial_capital:,.2f}")
            print(f"   💸 Transaction Cost: ${self.transaction_cost:.2f}")

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

    def initialize_strategy(self, predictions_df: pd.DataFrame) -> None:
        """Initialize strategy with full predictions dataset.

        This method is called before backtesting starts to allow the strategy
        to analyze the full dataset and set parameters. Base implementation
        does nothing - strategies can override this for custom initialization.

        Args:
            predictions_df: Full predictions DataFrame
        """
        # Base implementation does nothing - strategies can override
        pass

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

        if signal["direction"] not in ["long", "short", "no_trade"]:
            raise StrategyValidationError(
                f"🚨 SIGNAL VALIDATION FAILED {method_context}: "
                f"Direction must be 'long', 'short', or 'no_trade', got '{signal['direction']}'"
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
        """Generate trading signal for independent bet from prediction data.

        This method determines the bet direction (long/short) and confidence level
        for a single independent bet based on the prediction row. Each call generates
        exactly one betting decision with no dependency on previous bets.

        Independent Betting Context:
        ============================
        - Called once per prediction row to generate one bet
        - No state carried from previous predictions
        - Signal determines bet direction: long (RT > DA) or short (RT ≤ DA)
        - Confidence can be used for bet sizing or filtering

        Args:
            prediction_row: Single row from predictions DataFrame containing:
                - predicted_dart_slt: Binary prediction (0/1)
                - predicted_prob_class_0/1: XGBoost probability scores
                - prediction_confidence: Model confidence
                - Other prediction metadata

        Returns:
            Dictionary with signal information for this independent bet:
            - direction: "long" (bet RT > DA) or "short" (bet RT ≤ DA)
            - confidence: 0.0 to 1.0 (betting confidence level)
            - size_multiplier: Optional position size multiplier for this bet
        """
        pass

    @abstractmethod
    def calculate_position_size(self, signal: Dict, available_capital: float) -> float:
        """Calculate bet size for single independent bet.

        This method determines how much capital to wager on this specific bet.
        In the independent betting model, each bet is sized independently based
        on the signal and available capital, with no consideration of other bets.

        Independent Betting Context:
        ============================
        - Called once per bet to determine stake amount
        - No position sizing across multiple bets
        - Must account for transaction costs ($0.10 × 2 = $0.20 per bet)
        - Available capital is current capital after previous bets

        Even Money Betting:
        ===================
        - Bet size represents the stake amount (typically $1.00)
        - Win: Get back 2x bet size (stake + profit)
        - Lose: Forfeit entire bet size
        - Net after transaction costs: Win = +$0.80, Lose = -$1.20

        Args:
            signal: Signal dictionary from generate_signal() containing:
                - direction: "long" or "short"
                - confidence: 0.0 to 1.0
                - size_multiplier: Optional multiplier
            available_capital: Current capital available for this bet

        Returns:
            Bet size in dollars (positive for long, negative for short)
            Must be ≤ available_capital to account for transaction costs
        """
        pass

    def execute_backtest(self, predictions_df: pd.DataFrame) -> Dict:
        """Execute backtest with per-hour independent betting model.

        Each row represents one complete bet cycle:
        - Place bet based on predicted_dart_slt
        - Resolve bet using actual_dart_slt
        - Calculate P&L with even money payouts
        - Record trade and update capital

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

        # Process each prediction as independent bet
        for i, row in predictions_df.iterrows():
            try:
                # Generate signal for this bet
                signal = self._safe_method_call(
                    self.generate_signal, "generate_signal", row
                )
                self._validate_signal(signal, f"at row {i}")

                # Calculate bet size
                bet_size = self._safe_method_call(
                    self.calculate_position_size,
                    "calculate_position_size",
                    signal,
                    self.current_capital,
                )
                self._validate_position_size(
                    bet_size, self.current_capital, f"at row {i}"
                )

                # Process bet if signal generated and sufficient capital (skip no_trade signals)
                if (
                    signal.get("direction") != "no_trade"
                    and bet_size != 0
                    and abs(bet_size) <= self.current_capital
                ):
                    trade = self._process_independent_bet(row, signal, bet_size)
                    if trade:
                        trades.append(trade)

                        if self.enable_debug_logging:
                            direction = "long" if bet_size > 0 else "short"
                            result_emoji = "✅" if trade["pnl"] > 0 else "❌"
                            direction_emoji = "📈" if direction == "long" else "📉"
                            print(
                                f"  {result_emoji} {direction_emoji} {direction} bet: ${trade['pnl']:+.2f} (actual: {trade['actual_dart_slt']}) @ hour {row['end_hour']}"
                            )
                elif (
                    signal.get("direction") == "no_trade" and self.enable_debug_logging
                ):
                    print(
                        f"  ⏸️  No trade @ hour {row['end_hour']}: {signal.get('reason', 'no reason provided')}"
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

            # Record portfolio value after each bet
            self.portfolio_values.append(
                {
                    "utc_ts": row["utc_ts"],
                    "end_hour": row["end_hour"],
                    "portfolio_value": self.current_capital,
                    "week_num": row.get("week_num"),
                    "week_description": row.get("week_description"),
                    "has_position": False,  # No persistent positions in independent betting
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
            "final_portfolio_value": self.current_capital,
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
        self.trades = []
        self.portfolio_values = []
        self.performance_metrics = {}

        # Reset monitoring state
        self.validation_errors = []
        self.method_call_counts = {}

        if self.enable_debug_logging:
            print(f"🔄 Strategy state reset for {self.strategy_name}")
            print(f"   💰 Capital: ${self.current_capital:,.2f}")
            print(f"   🎰 Betting model: Independent per-hour bets")

        # Validate state after reset
        if self.enable_validation:
            assert (
                self.current_capital == self.initial_capital
            ), f"Capital reset failed: {self.current_capital} != {self.initial_capital}"
            assert (
                len(self.trades) == 0
            ), f"Trades reset failed: {len(self.trades)} != 0"

    def _process_independent_bet(
        self, row: pd.Series, signal: Dict, bet_size: float
    ) -> Dict:
        """Process a single independent bet cycle.

        Each bet is completely independent:
        1. Place bet based on prediction (deduct stake + transaction cost)
        2. Resolve bet using actual outcome (even money payout)
        3. Update capital and return trade record

        Args:
            row: Prediction row containing prediction and actual outcome
            signal: Trading signal from generate_signal()
            bet_size: Bet size from calculate_position_size()

        Returns:
            Dict: Complete trade record for this bet
        """
        # Get prediction and actual outcome
        actual_dart = row["actual_dart_slt"]  # 0 or 1 from classification
        predicted_dart = row.get("predicted_dart_slt", 0)

        # Determine bet direction from signal/bet_size
        direction = "long" if bet_size > 0 else "short"

        # Calculate P&L using even money betting rules
        # Long wins when actual_dart = 1 (RT > DA), Short wins when actual_dart = 0 (RT ≤ DA)
        if bet_size > 0:  # Long bet
            pnl = abs(bet_size) if actual_dart == 1 else -abs(bet_size)
        else:  # Short bet
            pnl = abs(bet_size) if actual_dart == 0 else -abs(bet_size)

        # Apply transaction cost (entry + exit transactions)
        pnl -= 2 * self.transaction_cost

        # Update capital with net P&L
        self.current_capital += pnl

        # Determine if prediction was correct
        correct_prediction = (
            bet_size > 0 and actual_dart == 1
        ) or (  # Long bet, RT > DA
            bet_size < 0 and actual_dart == 0
        )  # Short bet, RT ≤ DA

        # Create complete trade record
        trade = {
            "entry_time": row["utc_ts"],
            "exit_time": row["utc_ts"],  # Same time for independent bets
            "entry_hour": row["end_hour"],
            "exit_hour": row["end_hour"],  # Same hour for independent bets
            "direction": direction,
            "position_size": abs(bet_size),
            "actual_dart_slt": actual_dart,
            "pnl": pnl,
            "week_num": row.get("week_num"),
            "capital_after": self.current_capital,
            "correct_prediction": correct_prediction,
            "trade_duration_hours": 0.0,  # Independent bets have no duration
            "entry_prediction": predicted_dart,
            # Additional fields for analysis
            "predicted_prob_class_0": row.get("predicted_prob_class_0"),
            "predicted_prob_class_1": row.get("predicted_prob_class_1"),
            "prediction_confidence": row.get("prediction_confidence"),
            "actual_dart_slt_raw": row.get("actual_dart_slt_raw"),
        }

        return trade

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate strategy performance metrics for independent betting model.

        This method computes financial metrics for the per-hour independent betting
        strategy with even money payouts. All metrics are adapted for the unique
        characteristics of independent betting rather than traditional position holding.

        Independent Betting Model:
        ==========================
        - Each prediction generates exactly one independent bet
        - No position holding - each bet resolves immediately
        - Even money payouts: Win = +$1.00, Lose = -$1.00 (before costs)
        - Transaction cost: $0.10 per transaction × 2 = $0.20 per bet
        - Net payouts: Win = +$0.80, Lose = -$1.20

        Calculation Details:
        ====================

        1. **Total Return (%)**:
           - Formula: (Total P&L / Initial Capital) × 100
           - Represents cumulative return from all independent bets
           - Example: $168 profit on $10,000 = +1.68%

        2. **Win Rate (%)**:
           - Formula: (Number of Profitable Bets / Total Bets) × 100
           - Percentage of independent bets that generated positive P&L
           - Critical metric: Need >60% win rate to overcome transaction costs
           - Example: 200 winning bets out of 336 total = 59.5%

        3. **Total Trades**:
           - Count of independent bets placed (one per prediction row)
           - Each "trade" is one complete bet cycle: place → resolve → settle

        4. **Sharpe Ratio**:
           - Formula: (Mean Portfolio Return / Std Dev Portfolio Return) × √(365 × 24)
           - Annualized risk-adjusted return for independent betting
           - Based on portfolio value changes after each bet
           - ERCOT operates 24/7, so annualized using 365 × 24 periods

        5. **Max Drawdown (%)**:
           - Largest cumulative loss from peak portfolio value
           - Important for independent betting due to transaction cost drag
           - Shows worst-case capital erosion during losing streaks

        6. **Final Capital**:
           - End portfolio value: Initial Capital + Total P&L from all bets

        P&L Calculation (Even Money Betting):
        ====================================
        For each independent bet:

        - **Long Bet** (predict RT > DA):
          - Win: +$1.00 when actual_dart_slt = 1 (RT > DA)
          - Lose: -$1.00 when actual_dart_slt = 0 (RT ≤ DA)

        - **Short Bet** (predict RT ≤ DA):
          - Win: +$1.00 when actual_dart_slt = 0 (RT ≤ DA)
          - Lose: -$1.00 when actual_dart_slt = 1 (RT > DA)

        - **Transaction Cost**: $0.10 per transaction × 2 = $0.20 per bet
        - **Net P&L**: Base P&L - $0.20 transaction cost

        Break-Even Analysis:
        ===================
        - Fair coin (50% win rate): Expected loss due to transaction costs
        - Break-even: ~60% win rate needed (0.60 × $0.80 + 0.40 × (-$1.20) = 0)
        - Model edge required to overcome 20 cent transaction cost per bet

        Returns:
            Dict containing all calculated performance metrics for independent betting
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
            * self.transaction_cost,  # Round-trip cost per independent bet
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
