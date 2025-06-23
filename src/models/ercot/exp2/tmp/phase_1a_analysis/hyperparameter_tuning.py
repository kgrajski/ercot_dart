"""Hyperparameter Tuning for XGBoost Classification.

Uses Optuna to optimize XGBoost hyperparameters with focus on reducing
overfitting and improving validation performance for DART sign prediction.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings("ignore")


def objective(trial, X_train, y_train, X_val, y_val, hour, settlement_point):
    """Optuna objective function for XGBoost hyperparameter optimization."""

    # Define search space with focus on reducing overfitting
    params = {
        # Core tree parameters
        "max_depth": trial.suggest_int("max_depth", 2, 5),  # Reduced from 6
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        # Regularization (key for overfitting)
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        # Learning parameters
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 25, 200),
        # Class balancing (from our analysis)
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 2.0),
        # Fixed parameters
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "verbosity": 0,
    }

    # Train model
    model = xgb.XGBClassifier(**params)

    # Use early stopping to prevent overfitting
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False,
    )

    # Predictions
    y_val_pred = model.predict(X_val)
    y_train_pred = model.predict(X_train)

    # Calculate metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)

    # Primary objective: maximize validation accuracy
    # Secondary objective: minimize overfitting gap
    overfitting_gap = train_accuracy - val_accuracy

    # Composite score: prioritize validation performance, penalize overfitting
    score = val_accuracy - 0.1 * max(0, overfitting_gap - 0.1)  # Allow some overfitting

    # Store additional metrics for analysis
    trial.set_user_attr("val_accuracy", val_accuracy)
    trial.set_user_attr("train_accuracy", train_accuracy)
    trial.set_user_attr("val_f1", val_f1)
    trial.set_user_attr("overfitting_gap", overfitting_gap)
    trial.set_user_attr("hour", hour)
    trial.set_user_attr("settlement_point", settlement_point)

    return score


def tune_representative_hours():
    """Tune hyperparameters for representative hours to find optimal configurations."""

    print("🔧 XGBoost Hyperparameter Tuning with Optuna")
    print("=" * 60)

    # Load data
    settlement_points = ["LZ_HOUSTON_LZ", "LZ_HOUSTON_LZEW"]
    base_path = "data/studies/exp2/2025-06-04"

    # Representative hours from our analysis
    representative_hours = {
        "best": 17,  # 74.2% accuracy
        "worst": 4,  # 39.4% accuracy
        "median": 22,  # 58.4% accuracy
    }

    all_results = []

    for sp in settlement_points:
        print(f"\n📊 Processing {sp}")

        # Load data
        file_path = f"{base_path}/{sp}/modeling/model_ready.csv"
        df = pd.read_csv(file_path)
        df["utc_ts"] = pd.to_datetime(df["utc_ts"])
        df["year"] = df["utc_ts"].dt.year
        df["dart_sign"] = (df["dart_slt"] > 0).astype(int)

        # Split by year
        train_data = df[df["year"] == 2024].copy()
        val_data = df[df["year"] == 2025].copy()

        # Get feature names (same logic as dataset)
        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                "utc_ts",
                "local_ts",
                "end_of_hour",
                "dart_slt",
                "year",
                "dart_sign",
            ]
        ]

        print(f"   Features: {len(feature_cols)}")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")

        for hour_type, hour in representative_hours.items():
            print(f"\n   🎯 Tuning Hour {hour} ({hour_type} performer)")

            # Filter data for this hour
            hour_train = train_data[train_data["end_of_hour"] == hour].copy()
            hour_val = val_data[val_data["end_of_hour"] == hour].copy()

            if len(hour_train) == 0 or len(hour_val) == 0:
                print(f"      ⚠️  Insufficient data for Hour {hour}")
                continue

            # Prepare features and targets
            X_train = hour_train[feature_cols]
            y_train = hour_train["dart_sign"]
            X_val = hour_val[feature_cols]
            y_val = hour_val["dart_sign"]

            print(
                f"      Training: {len(X_train)} samples, {y_train.mean():.1%} positive"
            )
            print(
                f"      Validation: {len(X_val)} samples, {y_val.mean():.1%} positive"
            )

            # Create Optuna study
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(random_state=42),
                study_name=f"{sp}_hour_{hour}_{hour_type}",
            )

            # Optimize
            study.optimize(
                lambda trial: objective(
                    trial, X_train, y_train, X_val, y_val, hour, sp
                ),
                n_trials=50,  # Reasonable for initial exploration
                timeout=300,  # 5 minutes per hour
            )

            # Extract best results
            best_trial = study.best_trial
            best_params = best_trial.params
            best_score = best_trial.value

            print(f"      ✅ Best validation score: {best_score:.3f}")
            print(
                f"      📈 Best validation accuracy: {best_trial.user_attrs['val_accuracy']:.3f}"
            )
            print(
                f"      📉 Overfitting gap: {best_trial.user_attrs['overfitting_gap']:.3f}"
            )

            # Store results
            result = {
                "settlement_point": sp,
                "hour": hour,
                "hour_type": hour_type,
                "best_score": best_score,
                "val_accuracy": best_trial.user_attrs["val_accuracy"],
                "train_accuracy": best_trial.user_attrs["train_accuracy"],
                "val_f1": best_trial.user_attrs["val_f1"],
                "overfitting_gap": best_trial.user_attrs["overfitting_gap"],
                **best_params,  # Include all hyperparameters
            }
            all_results.append(result)

    # Analyze results
    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("TUNING RESULTS SUMMARY")
    print("=" * 60)

    # Overall improvement
    baseline_accuracy = 0.536  # From our analysis
    avg_tuned_accuracy = results_df["val_accuracy"].mean()
    improvement = avg_tuned_accuracy - baseline_accuracy

    print(f"Baseline accuracy: {baseline_accuracy:.3f}")
    print(f"Tuned accuracy: {avg_tuned_accuracy:.3f}")
    print(f"Improvement: {improvement:.3f} ({improvement/baseline_accuracy*100:+.1f}%)")

    # Best performing configurations
    best_overall = results_df.loc[results_df["val_accuracy"].idxmax()]
    print(f"\nBest overall result:")
    print(
        f"  {best_overall['settlement_point']} Hour {best_overall['hour']} ({best_overall['hour_type']})"
    )
    print(f"  Validation accuracy: {best_overall['val_accuracy']:.3f}")
    print(f"  Overfitting gap: {best_overall['overfitting_gap']:.3f}")

    # Parameter analysis
    print(f"\nOptimal parameter ranges:")
    param_cols = [
        "max_depth",
        "learning_rate",
        "n_estimators",
        "reg_alpha",
        "reg_lambda",
        "scale_pos_weight",
    ]
    for param in param_cols:
        if param in results_df.columns:
            print(
                f"  {param}: {results_df[param].min():.3f} - {results_df[param].max():.3f} (avg: {results_df[param].mean():.3f})"
            )

    # Save results
    results_df.to_csv("hyperparameter_tuning_results.csv", index=False)
    print(f"\n💾 Detailed results saved to: hyperparameter_tuning_results.csv")

    # Generate recommended parameters for full experiment
    recommended_params = {
        "max_depth": int(results_df["max_depth"].median()),
        "min_child_weight": int(results_df["min_child_weight"].median()),
        "learning_rate": results_df["learning_rate"].median(),
        "n_estimators": int(results_df["n_estimators"].median()),
        "subsample": results_df["subsample"].median(),
        "colsample_bytree": results_df["colsample_bytree"].median(),
        "reg_alpha": results_df["reg_alpha"].median(),
        "reg_lambda": results_df["reg_lambda"].median(),
        "scale_pos_weight": results_df["scale_pos_weight"].median(),
    }

    print(f"\n🎯 RECOMMENDED PARAMETERS FOR FULL EXPERIMENT:")
    print("=" * 60)
    for param, value in recommended_params.items():
        print(f"{param}: {value}")

    # Save recommended parameters
    pd.Series(recommended_params).to_csv("recommended_xgboost_params.csv")
    print(f"\n💾 Recommended parameters saved to: recommended_xgboost_params.csv")

    return results_df, recommended_params


if __name__ == "__main__":
    try:
        results, recommended = tune_representative_hours()
        print("\n✅ Hyperparameter tuning complete!")
    except Exception as e:
        print(f"❌ Error during tuning: {e}")
        import traceback

        traceback.print_exc()
