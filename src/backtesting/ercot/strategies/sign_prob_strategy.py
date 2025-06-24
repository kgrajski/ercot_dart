"""Probability-Based Trading Strategy for ERCOT Markets.

This strategy implements a more sophisticated approach than naive trading:
- Only trade predictions with confidence above a specified percentile threshold
- Use prediction_confidence (max of predicted_prob_class_0, predicted_prob_class_1)
- Calculate hour-specific confidence thresholds (24 thresholds, one per hour)
- Same $1 position size as naive, but much more selective
"""

import os
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.backtesting.ercot.base_strategy import BaseStrategy
from src.features.ercot.visualization import PROFESSIONAL_COLORS
from src.features.ercot.visualization import apply_professional_axis_styling
from src.features.ercot.visualization import get_professional_layout


class SignProbStrategy(BaseStrategy):
    """Probability-based trading strategy with hour-specific thresholds.

    This strategy only trades when the model has high confidence for that specific hour:
    - Calculate hour-specific confidence thresholds from all predictions (24 thresholds)
    - Only trade when prediction_confidence > hour_threshold
    - Same position sizing as naive strategy ($1 per trade)
    - Should result in fewer trades but higher quality predictions adapted per hour

    ⚠️  **Note**: Current implementation has data leakage (uses backtesting data for
    thresholds). See initialize_strategy() docstring for details and TODO improvements.
    """

    def __init__(
        self,
        transaction_cost: float,  # Required parameter - must be provided by caller
        strategy_name: str = "sign_prob",
        initial_capital: float = 10000.0,
        output_dir: str = None,
        prob_percentile: float = 0.95,  # New: Confidence percentile threshold
        **kwargs,
    ):
        """Initialize probability-based strategy.

        Args:
            strategy_name: Name/label for this strategy instance
            initial_capital: Starting capital for the strategy
            transaction_cost: Fixed cost per trade in dollars
            output_dir: Directory for saving strategy results
            prob_percentile: Percentile threshold for prediction confidence (0.0-1.0)
        """
        super().__init__(
            strategy_name=strategy_name,
            transaction_cost=transaction_cost,
            initial_capital=initial_capital,
            output_dir=output_dir,
        )

        self.prob_percentile = prob_percentile
        self.hour_thresholds: Dict[int, float] = {}  # Store 24 hour-specific thresholds
        self.total_predictions_count = 0
        self.predictions_above_threshold = 0
        self.hour_stats = {}  # Store per-hour statistics

    def initialize_strategy(self, predictions_df: pd.DataFrame) -> None:
        """Initialize strategy with full predictions dataset to calculate hour-specific thresholds.

        This method calculates confidence thresholds for each hour (1-24) and creates
        visualization and CSV outputs.

        ⚠️  **KNOWN DATA LEAKAGE - TODO FOR FUTURE IMPROVEMENT**

        Currently, this method calculates thresholds using the SAME dataset that will be
        used for backtesting. This creates a form of data leakage where we're using
        future information to set our trading thresholds.

        The result is that we get exactly N trades per hour (where N = percentile_level *
        total_predictions_per_hour), which is mathematically correct but unrealistic
        compared to real-world deployment.

        **Future Improvement (TODO):**
        - Use validation set confidence distributions from step 04- modeling output
        - Calculate thresholds from model training/validation performance rather than
          the backtesting dataset itself
        - This would create more realistic trade distributions with natural variation
          per hour as confidence levels fluctuate over time

        **Files to explore for validation thresholds:**
        - Look for validation results in modeling output from 04-kag-exp2-ercot-modeling.py
        - Extract confidence distributions from cross-validation or holdout validation
        - Apply those historical thresholds to the backtesting predictions

        For now, we accept this limitation as the strategy still provides valuable
        insights into hour-specific filtering effectiveness vs naive approaches.

        Args:
            predictions_df: Full predictions DataFrame (creates data leakage as documented above)
        """
        super().initialize_strategy(predictions_df)

        # Calculate hour-specific confidence thresholds
        self._calculate_hour_thresholds(predictions_df)

        # Save thresholds and create visualizations
        self._save_hour_thresholds()
        self._create_threshold_plot()

        print(f"   📊 SignProb Strategy Initialization (Hour-Specific Thresholds):")
        print(f"      Percentile level: {self.prob_percentile:.1%}")
        print(f"      Total predictions: {self.total_predictions_count:,}")
        print(
            f"      Predictions above thresholds: {self.predictions_above_threshold:,} ({self.predictions_above_threshold/self.total_predictions_count:.1%})"
        )
        print(
            f"      Threshold range: {min(self.hour_thresholds.values()):.4f} - {max(self.hour_thresholds.values()):.4f}"
        )

    def _calculate_hour_thresholds(self, predictions_df: pd.DataFrame) -> None:
        """Calculate confidence thresholds for each hour."""
        self.total_predictions_count = len(predictions_df)
        total_above_threshold = 0

        # Get confidence values
        if "prediction_confidence" in predictions_df.columns:
            confidence_col = "prediction_confidence"
        elif (
            "predicted_prob_class_0" in predictions_df.columns
            and "predicted_prob_class_1" in predictions_df.columns
        ):
            # Fallback: calculate confidence from prob columns
            predictions_df["prediction_confidence"] = np.maximum(
                predictions_df["predicted_prob_class_0"],
                predictions_df["predicted_prob_class_1"],
            )
            confidence_col = "prediction_confidence"
        else:
            raise ValueError(
                "Cannot find prediction confidence columns (prediction_confidence or predicted_prob_class_0/1)"
            )

        # Calculate threshold for each hour
        for hour in range(1, 25):
            hour_data = predictions_df[predictions_df["end_hour"] == hour]

            if len(hour_data) == 0:
                # No data for this hour, use global median as fallback
                global_confidence = predictions_df[confidence_col].dropna()
                threshold = np.percentile(
                    global_confidence, 50
                )  # Use median as safe fallback
                print(
                    f"      ⚠️  No data for hour {hour}, using global median: {threshold:.4f}"
                )
            else:
                confidence_values = hour_data[confidence_col].dropna()
                threshold = np.percentile(confidence_values, self.prob_percentile * 100)
                above_threshold = (confidence_values >= threshold).sum()
                total_above_threshold += above_threshold

                # Store hour statistics
                self.hour_stats[hour] = {
                    "total_predictions": len(confidence_values),
                    "above_threshold": above_threshold,
                    "threshold": threshold,
                    "percentage_above": (above_threshold / len(confidence_values)) * 100
                    if len(confidence_values) > 0
                    else 0,
                }

            self.hour_thresholds[hour] = threshold

        self.predictions_above_threshold = total_above_threshold

    def _save_hour_thresholds(self) -> None:
        """Save hour-specific thresholds to CSV file."""
        if not self.output_dir:
            return

        # Create strategy output directory
        strategy_dir = os.path.join(self.output_dir, self.strategy_name)
        os.makedirs(strategy_dir, exist_ok=True)

        # Prepare threshold data
        threshold_data = []
        for hour in range(1, 25):
            stats = self.hour_stats.get(hour, {})
            threshold_data.append(
                {
                    "hour": hour,
                    "threshold": self.hour_thresholds[hour],
                    "total_predictions": stats.get("total_predictions", 0),
                    "predictions_above_threshold": stats.get("above_threshold", 0),
                    "percentage_above_threshold": stats.get("percentage_above", 0.0),
                    "percentile_level": self.prob_percentile * 100,
                }
            )

        # Save to CSV
        threshold_df = pd.DataFrame(threshold_data)
        csv_path = os.path.join(strategy_dir, "hour_thresholds.csv")
        threshold_df.to_csv(csv_path, index=False)
        print(f"   💾 Hour thresholds saved: {csv_path}")

    def _create_threshold_plot(self) -> None:
        """Create professional visualization of hour-specific thresholds."""
        if not self.output_dir:
            return

        # Create strategy output directory
        strategy_dir = os.path.join(self.output_dir, self.strategy_name)
        os.makedirs(strategy_dir, exist_ok=True)

        # Prepare data for plotting
        hours = list(range(1, 25))
        thresholds = [self.hour_thresholds[hour] for hour in hours]
        predictions_above = [
            self.hour_stats.get(hour, {}).get("above_threshold", 0) for hour in hours
        ]

        # Create plot
        fig = go.Figure()

        # Bar chart of thresholds
        fig.add_trace(
            go.Bar(
                x=hours,
                y=thresholds,
                name="Confidence Threshold",
                marker_color=PROFESSIONAL_COLORS["primary"],
                opacity=0.8,
                hovertemplate="<b>Hour %{x}</b><br>Threshold: %{y:.4f}<br>Predictions Above: %{customdata}<extra></extra>",
                customdata=predictions_above,
            )
        )

        # Update layout with professional styling
        layout = get_professional_layout(
            title=f"Hour-Specific Confidence Thresholds<br><sub>Strategy: {self.strategy_name.upper()} ({self.prob_percentile:.0%} percentile per hour)</sub>",
            height=600,
            showlegend=False,
        )

        fig.update_layout(**layout)
        apply_professional_axis_styling(fig)

        # Update axis labels
        fig.update_xaxes(
            title_text="Hour of Day",
            tickmode="array",
            tickvals=list(range(1, 25)),
            ticktext=[str(h) for h in range(1, 25)],
        )
        fig.update_yaxes(title_text="Confidence Threshold")

        # Save HTML and PNG
        html_path = os.path.join(strategy_dir, "hour_thresholds_plot.html")
        fig.write_html(
            html_path,
            include_plotlyjs=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )
        print(f"   📊 Threshold plot saved: {html_path}")

        # Save PNG version
        png_path = os.path.join(strategy_dir, "hour_thresholds_plot.png")
        try:
            fig.write_image(png_path, width=1200, height=600, scale=2)
            print(f"   📊 Threshold plot PNG saved: {png_path}")
        except Exception as e:
            print(f"   ⚠️  Could not save PNG: {e}")

    def generate_signal(self, prediction_row: pd.Series) -> Dict:
        """Generate trading signal from prediction if confidence exceeds hour-specific threshold.

        Only trades when prediction confidence exceeds the calculated threshold for that hour.

        Args:
            prediction_row: Single row from predictions DataFrame

        Returns:
            Dictionary with signal information (may indicate no trade)
        """
        # Check if hour thresholds have been set
        if not self.hour_thresholds:
            raise ValueError(
                "Strategy not initialized - call initialize_strategy() first"
            )

        # Get hour and corresponding threshold
        hour = prediction_row.get("end_hour")
        if hour not in self.hour_thresholds:
            return {
                "direction": "no_trade",
                "confidence": 0.0,
                "signal_value": 0.0,
                "reason": f"no threshold available for hour {hour}",
            }

        hour_threshold = self.hour_thresholds[hour]

        # Get prediction confidence
        confidence = prediction_row.get("prediction_confidence")
        if confidence is None:
            # Fallback: calculate from prob columns
            prob_class_0 = prediction_row.get("predicted_prob_class_0", 0.5)
            prob_class_1 = prediction_row.get("predicted_prob_class_1", 0.5)
            confidence = max(prob_class_0, prob_class_1)

        # Only trade if confidence exceeds hour-specific threshold
        if confidence < hour_threshold:
            return {
                "direction": "no_trade",
                "confidence": confidence,
                "signal_value": 0.0,
                "reason": f"confidence ({confidence:.4f}) < hour {hour} threshold ({hour_threshold:.4f})",
            }

        # High confidence for this hour - generate signal based on prediction
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
            "confidence": confidence,
            "signal_value": signal_value,
            "reason": f"high_confidence ({confidence:.4f} >= hour {hour} threshold {hour_threshold:.4f})",
        }

    def calculate_position_size(self, signal: Dict, available_capital: float) -> float:
        """Calculate position size - $1 for high-confidence trades, $0 for no-trade.

        Args:
            signal: Signal dictionary from generate_signal()
            available_capital: Available capital for trading

        Returns:
            Position size in dollars ($1 for trades, $0 for no-trade)
        """
        # No trade if signal indicates no_trade
        if signal.get("direction") == "no_trade":
            return 0.0

        # Check if we have enough capital for the trade
        min_required = 1.0 + (2 * self.transaction_cost)  # $1 + entry/exit costs

        if available_capital >= min_required:
            return 1.0
        else:
            # Not enough capital for full trade
            return 0.0

    def __str__(self):
        """String representation of the strategy."""
        return f"SignProbStrategy(capital=${self.initial_capital:,.2f}, tx_cost=${self.transaction_cost:.2f}, percentile={self.prob_percentile:.1%}, hour_specific=True)"

    def __repr__(self):
        """Detailed representation of the strategy."""
        return self.__str__()
