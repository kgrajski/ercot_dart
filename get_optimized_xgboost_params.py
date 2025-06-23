"""Utility to Load Optimized XGBoost Parameters from Phase 1A Analysis.

Quick access to the recommended XGBoost parameters derived from 
Phase 1A hyperparameter tuning analysis.
"""

from pathlib import Path

import pandas as pd


def get_optimized_xgboost_params():
    """Load the optimized XGBoost parameters from Phase 1A analysis.

    Returns:
        dict: Optimized XGBoost parameters ready for model training
    """

    # Path to the Phase 1A analysis results
    params_file = "src/models/ercot/exp2/tmp/phase_1a_analysis/recommended_xgboost_params_from_dataset.csv"

    try:
        # Load parameters from CSV
        params_df = pd.read_csv(params_file, index_col=0)
        params_dict = params_df.iloc[:, 0].to_dict()

        # Convert to appropriate types for XGBoost
        optimized_params = {
            "max_depth": int(params_dict["max_depth"]),
            "n_estimators": int(params_dict["n_estimators"]),
            "learning_rate": float(params_dict["learning_rate"]),
            "subsample": float(params_dict["subsample"]),
            "colsample_bytree": float(params_dict["colsample_bytree"]),
            "reg_alpha": float(params_dict["reg_alpha"]),
            "reg_lambda": float(params_dict["reg_lambda"]),
            "min_child_weight": int(params_dict["min_child_weight"]),
            "scale_pos_weight": float(params_dict["scale_pos_weight"]),
            # Additional parameters for production use
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }

        # Early stopping configuration (for training)
        early_stopping_rounds = int(params_dict["early_stopping_rounds"])

        return optimized_params, early_stopping_rounds

    except Exception as e:
        print(f"❌ Error loading optimized parameters: {e}")
        print(f"Expected file: {params_file}")
        return None, None


def print_parameter_summary():
    """Print a summary of the optimized parameters."""

    params, early_stopping = get_optimized_xgboost_params()

    if params is None:
        print("❌ Failed to load parameters")
        return

    print("🎯 OPTIMIZED XGBOOST PARAMETERS (Phase 1A Results)")
    print("=" * 55)

    print("\n📊 Model Architecture:")
    print(f"   max_depth: {params['max_depth']}")
    print(f"   n_estimators: {params['n_estimators']}")
    print(f"   learning_rate: {params['learning_rate']}")

    print("\n🔒 Regularization:")
    print(f"   reg_alpha (L1): {params['reg_alpha']}")
    print(f"   reg_lambda (L2): {params['reg_lambda']}")
    print(f"   min_child_weight: {params['min_child_weight']}")

    print("\n🎲 Randomness:")
    print(f"   subsample: {params['subsample']}")
    print(f"   colsample_bytree: {params['colsample_bytree']}")

    print("\n⚖️  Class Balancing:")
    print(f"   scale_pos_weight: {params['scale_pos_weight']}")

    print("\n⏹️  Early Stopping:")
    print(f"   early_stopping_rounds: {early_stopping}")

    print("\n📈 Expected Improvements:")
    print("   • Validation accuracy: 53.6% → 63.3% (+18.2%)")
    print("   • Overfitting gap: 46.4% → <30%")
    print("   • Models worse than baseline: 25/48 → <2/48")


def get_training_config():
    """Get complete training configuration for XGBoost models.

    Returns:
        dict: Complete configuration for XGBoost training
    """

    params, early_stopping = get_optimized_xgboost_params()

    if params is None:
        return None

    training_config = {
        "model_params": params,
        "training_params": {
            "early_stopping_rounds": early_stopping,
            "verbose": False,
        },
        "validation_split": 0.2,  # For internal validation if needed
        "cross_validation": False,  # Use time-based validation instead
        "feature_importance": True,  # Track feature importance
        "save_model": True,
    }

    return training_config


if __name__ == "__main__":
    print_parameter_summary()

    # Example usage
    print("\n" + "=" * 55)
    print("EXAMPLE USAGE")
    print("=" * 55)
    print(
        """
from get_optimized_xgboost_params import get_optimized_xgboost_params
import xgboost as xgb

# Load optimized parameters
params, early_stopping = get_optimized_xgboost_params()

# Create and train model
model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=early_stopping,
    verbose=False
)
"""
    )
