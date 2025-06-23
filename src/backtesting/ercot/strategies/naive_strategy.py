"""Naive Trading Strategy for ERCOT Markets.

This strategy implements the simplest possible approach:
- Bet $1 on each prediction
- Go long if prediction is positive DART (RT > DA)
- Go short if prediction is negative DART (RT < DA)
- Settle previous bet before making new one
"""

from typing import Dict

import pandas as pd

from src.backtesting.ercot.base_strategy import BaseStrategy


class NaiveStrategy(BaseStrategy):
    """Naive $1-per-prediction strategy.

    This strategy represents the simplest possible trading approach:
    - Fixed position size of $1 per trade
    - Direction based purely on classification prediction
    - No risk management or position sizing logic
    """

    def __init__(
        self,
        strategy_name: str = "naive",
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.5,
        output_dir: str = None,
    ):
        """Initialize naive strategy.

        Args:
            strategy_name: Name/label for this strategy instance
            initial_capital: Starting capital for the strategy
            transaction_cost: Fixed cost per trade in dollars
            output_dir: Directory for saving strategy results
        """
        super().__init__(
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            output_dir=output_dir,
        )

    def generate_signal(self, prediction_row: pd.Series) -> Dict:
        """Generate trading signal from prediction.

        Simple logic for classification predictions:
        - predicted_dart_slt = 1 (positive class) → go long (bet RT > DA)
        - predicted_dart_slt = 0 (negative class) → go short (bet RT < DA)

        Args:
            prediction_row: Single row from predictions DataFrame

        Returns:
            Dictionary with signal information
        """
        # Get the classification prediction (0 or 1)
        predicted_class = prediction_row.get("predicted_dart_slt", 0)

        # Convert classification to trading direction
        if predicted_class == 1:
            direction = "long"  # Bet that real-time will be higher than day-ahead
            signal_value = 1.0
        else:  # predicted_class == 0
            direction = "short"  # Bet that real-time will be lower than day-ahead
            signal_value = -1.0

        return {
            "direction": direction,
            "confidence": 1.0,  # Always confident in naive strategy
            "signal_value": signal_value,
        }

    def calculate_position_size(self, signal: Dict, available_capital: float) -> float:
        """Calculate position size - always $1 for naive strategy.

        Args:
            signal: Signal dictionary from generate_signal()
            available_capital: Available capital for trading

        Returns:
            Position size in dollars (always $1)
        """
        # Check if we have enough capital for the trade
        min_required = 1.0 + (2 * self.transaction_cost)  # $1 + entry/exit costs

        if available_capital >= min_required:
            return 1.0
        else:
            # Not enough capital for full trade
            return 0.0

    def __str__(self):
        """String representation of the strategy."""
        return f"NaiveStrategy(capital=${self.initial_capital:,.2f}, tx_cost=${self.transaction_cost:.2f})"

    def __repr__(self):
        """Detailed representation of the strategy."""
        return self.__str__()
