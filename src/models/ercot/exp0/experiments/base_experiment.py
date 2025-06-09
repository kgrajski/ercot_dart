"""
Base Experiment Class

Provides a framework for running and tracking DART prediction experiments.
Similar to the speechBCI experiment management pattern.
"""

import json
from abc import ABC
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd


class BaseExperiment(ABC):
    """
    Abstract base class for DART prediction experiments.

    Provides common infrastructure for:
    - Experiment configuration and tracking
    - Result logging and persistence
    - Model comparison and evaluation
    """

    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        output_dir: Optional[str] = None,
    ):
        """
        Initialize experiment.

        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration dictionary
            output_dir: Directory to save results (default: data/experiments/exp0)
        """
        self.experiment_name = experiment_name
        self.config = config
        self.output_dir = Path(output_dir or f"data/experiments/exp0/{experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Experiment tracking
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.models = {}

        # Save initial config
        self._save_config()

    def _save_config(self):
        """Save experiment configuration"""
        config_file = self.output_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment.

        Returns:
            Dictionary with experiment results
        """
        pass

    def start_experiment(self):
        """Mark experiment start time"""
        self.start_time = datetime.now()
        print(f"Starting experiment: {self.experiment_name}")
        print(f"Start time: {self.start_time}")

    def end_experiment(self):
        """Mark experiment end time and save results"""
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time

        print(f"Experiment completed: {self.experiment_name}")
        print(f"Duration: {duration}")

        # Save final results
        self._save_results()

    def _save_results(self):
        """Save experiment results"""
        results_file = self.output_dir / "results.json"

        final_results = {
            "experiment_name": self.experiment_name,
            "start_time": str(self.start_time),
            "end_time": str(self.end_time),
            "duration": (
                str(self.end_time - self.start_time)
                if self.end_time and self.start_time
                else None
            ),
            "config": self.config,
            "results": self.results,
        }

        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

    def log_result(self, key: str, value: Any):
        """Log a result value"""
        self.results[key] = value
        print(f"Logged result: {key} = {value}")

    def log_model(self, name: str, model_adapter, metrics: Dict[str, float]):
        """Log a trained model and its metrics"""
        self.models[name] = {"model": model_adapter, "metrics": metrics}

        # Save model to disk
        model_file = self.output_dir / f"model_{name}.pkl"
        model_adapter.save_model(model_file)

        print(f"Logged model: {name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    def compare_models(self, metric: str = "rmse") -> pd.DataFrame:
        """
        Compare models by a specific metric.

        Args:
            metric: Metric to compare by

        Returns:
            DataFrame with model comparison
        """
        if not self.models:
            print("No models to compare")
            return pd.DataFrame()

        comparison_data = []
        for name, model_info in self.models.items():
            metrics = model_info["metrics"]
            if metric in metrics:
                comparison_data.append(
                    {"model": name, "metric": metric, "value": metrics[metric]}
                )

        if not comparison_data:
            print(f"No models have metric: {metric}")
            return pd.DataFrame()

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("value")

        print(f"\nModel comparison by {metric}:")
        print(df.to_string(index=False))

        return df

    def get_best_model(self, metric: str = "rmse", minimize: bool = True):
        """
        Get the best model by a specific metric.

        Args:
            metric: Metric to optimize
            minimize: If True, lower is better; if False, higher is better

        Returns:
            Best model adapter
        """
        if not self.models:
            return None

        best_model = None
        best_value = float("inf") if minimize else float("-inf")

        for name, model_info in self.models.items():
            metrics = model_info["metrics"]
            if metric in metrics:
                value = metrics[metric]
                if (minimize and value < best_value) or (
                    not minimize and value > best_value
                ):
                    best_value = value
                    best_model = model_info["model"]

        return best_model
