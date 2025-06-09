#!/usr/bin/env python3
"""
Development Script for ERCOT DART Modeling - Experiment 0

Main entry point for running DART prediction experiments.
Follows the workflow naming convention: 04-kag-exp0-modeling.py

Usage:
    python src/workflow/04-kag-exp0-modeling.py --experiment linear --config config.json
    python src/workflow/04-kag-exp0-modeling.py --experiment linear --hour-specific
    python src/workflow/04-kag-exp0-modeling.py --list-experiments
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

# Add src to path for imports (from workflow directory)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.ercot.exp0.experiments import LinearExperiment
from src.models.ercot.exp0.model_adapters import get_model_adapter


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ERCOT DART Modeling Development")

    # Experiment selection
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default="linear",
        choices=["linear", "tree", "neural"],
        help="Type of experiment to run",
    )

    # Configuration
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to configuration JSON file"
    )

    # Quick options (override config)
    parser.add_argument(
        "--hour-specific", action="store_true", help="Use hour-specific models"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ridge",
        choices=["linear", "ridge", "lasso", "elasticnet"],
        help="Type of linear model",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="dart",
        choices=["dart", "dart_slt"],
        help="Target variable to predict",
    )

    # Experiment management
    parser.add_argument("--name", type=str, default=None, help="Custom experiment name")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Custom output directory"
    )

    # Utilities
    parser.add_argument(
        "--list-experiments", action="store_true", help="List available experiments"
    )
    parser.add_argument(
        "--test-model", type=str, default=None, help="Test a specific model adapter"
    )

    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    return parser.parse_args()


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    if config_path is None:
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def create_experiment_config(args) -> Dict[str, Any]:
    """Create experiment configuration from args"""
    config = {}

    # Load base config from file if provided
    if args.config:
        config.update(load_config(args.config))

    # Override with command line arguments
    if args.hour_specific:
        config["hour_specific_tests"] = [True]

    if args.model_type:
        config["model_types"] = [args.model_type]

    if args.target:
        config["target_column"] = args.target

    return config


def run_linear_experiment(args) -> Dict[str, Any]:
    """Run linear modeling experiment"""
    print("🚀 Starting Linear DART Experiment")
    print("=" * 50)

    # Create configuration
    config = create_experiment_config(args)

    # Create experiment name
    experiment_name = args.name or f"linear_{args.model_type}"
    if args.hour_specific:
        experiment_name += "_hourly"

    print(f"Experiment: {experiment_name}")
    print(f"Config: {config}")

    if args.dry_run:
        print("🔍 DRY RUN - Would execute experiment with above config")
        return {}

    # Run experiment
    experiment = LinearExperiment(experiment_name=experiment_name, config=config)

    if args.output_dir:
        experiment.output_dir = Path(args.output_dir)

    return experiment.run()


def test_model_adapter(model_type: str, debug: bool = False):
    """Test a specific model adapter"""
    print(f"🧪 Testing Model Adapter: {model_type}")
    print("=" * 50)

    try:
        # Create model adapter
        adapter = get_model_adapter(model_type)
        print(f"✅ Successfully created {model_type} adapter")
        print(f"   Class: {adapter.__class__.__name__}")
        print(f"   Config: {adapter.config}")

        # Test with dummy data if in debug mode
        if debug:
            import numpy as np
            import pandas as pd

            print("\n🔍 Testing with dummy data...")

            # Create dummy data
            n_samples, n_features = 1000, 10
            X = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=[f"feature_{i}" for i in range(n_features)],
            )
            y = pd.Series(np.random.randn(n_samples))

            print(f"   Data shape: X={X.shape}, y={y.shape}")

            # Test training
            print("   Training model...")
            metrics = adapter.fit(X[:800], y[:800], X[800:900], y[800:900])
            print(f"   Training metrics: {metrics}")

            # Test prediction
            print("   Making predictions...")
            predictions = adapter.predict(X[900:])
            print(f"   Predictions shape: {predictions.shape}")

            # Test evaluation
            print("   Evaluating model...")
            eval_metrics = adapter.evaluate(X[900:], y[900:])
            print(f"   Evaluation metrics: {eval_metrics}")

            print("✅ Model adapter test completed successfully")

    except Exception as e:
        print(f"❌ Model adapter test failed: {e}")
        if debug:
            import traceback

            traceback.print_exc()


def list_experiments():
    """List available experiments"""
    print("📋 Available Experiments")
    print("=" * 50)

    experiments = {
        "linear": {
            "description": "Linear baseline models (Phase 1)",
            "models": ["linear", "ridge", "lasso", "elasticnet"],
            "status": "✅ Implemented",
        },
        "tree": {
            "description": "Tree-based models (Phase 2)",
            "models": ["random_forest", "xgboost", "lightgbm"],
            "status": "🚧 Planned",
        },
        "neural": {
            "description": "Neural network models (Phase 3)",
            "models": ["mlp", "lstm", "transformer"],
            "status": "🚧 Planned",
        },
    }

    for exp_name, exp_info in experiments.items():
        print(f"\n🔬 {exp_name.upper()}")
        print(f"   Description: {exp_info['description']}")
        print(f"   Models: {', '.join(exp_info['models'])}")
        print(f"   Status: {exp_info['status']}")


def main():
    """Main entry point"""
    args = parse_args()

    if args.debug:
        print("🐛 Debug mode enabled")

    try:
        if args.list_experiments:
            list_experiments()

        elif args.test_model:
            test_model_adapter(args.test_model, args.debug)

        elif args.experiment == "linear":
            results = run_linear_experiment(args)
            if results:
                print("\n✅ Experiment completed successfully!")

        elif args.experiment in ["tree", "neural"]:
            print(f"❌ {args.experiment.upper()} experiments not yet implemented")
            print("   These are planned for Phase 2 and Phase 3")

        else:
            print(f"❌ Unknown experiment: {args.experiment}")

    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
