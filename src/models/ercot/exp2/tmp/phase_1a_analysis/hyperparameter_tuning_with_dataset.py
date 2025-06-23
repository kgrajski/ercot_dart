"""Hyperparameter Tuning Using DartSLTExp2Dataset.

Uses DartSLTExp2Dataset to properly load features and RandomForest with Optuna 
to find optimal parameters. This avoids data type issues and reuses existing logic.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.ercot.exp2.DartSLTExp2Dataset import DartSltExp2Dataset

warnings.filterwarnings("ignore")


def objective(trial, X_train, y_train, X_val, y_val, hour, settlement_point):
    """Optuna objective function for RandomForest hyperparameter optimization."""

    # Define search space focused on reducing overfitting
    params = {
        # Core tree parameters
        "n_estimators": trial.suggest_int("n_estimators", 25, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        # Randomness for regularization
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "bootstrap": True,  # Always use bootstrap
        # Class balancing
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample", None]
        ),
        # Fixed parameters
        "random_state": 42,
        "n_jobs": 1,  # Keep simple for reliability
    }

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

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
    score = val_accuracy - 0.1 * max(0, overfitting_gap - 0.2)  # Allow some overfitting

    # Store additional metrics for analysis
    trial.set_user_attr("val_accuracy", val_accuracy)
    trial.set_user_attr("train_accuracy", train_accuracy)
    trial.set_user_attr("val_f1", val_f1)
    trial.set_user_attr("overfitting_gap", overfitting_gap)
    trial.set_user_attr("hour", hour)
    trial.set_user_attr("settlement_point", settlement_point)

    return score


def tune_representative_hours():
    """Tune hyperparameters for representative hours using DartSLTExp2Dataset."""

    print("🔧 RandomForest Hyperparameter Tuning with Optuna + DartSLTExp2Dataset")
    print("=" * 70)

    # Settlement points and paths
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

        # Use DartSLTExp2Dataset to properly load data
        spp_dir = f"{base_path}/{sp}"
        modeling_dir = f"{spp_dir}/modeling"

        try:
            # Initialize dataset (automatically finds final dataset)
            dataset = DartSltExp2Dataset(spp_dir=spp_dir, output_dir=modeling_dir)

            # Get feature names using the same logic as the model
            feature_names = dataset.get_feature_names()
            print(f"   Features from dataset: {len(feature_names)}")
            print(f"   Sample features: {feature_names[:5]}...")

            # Get the dataframe
            df = dataset.df.copy()
            df["year"] = df["utc_ts"].dt.year
            df["dart_sign"] = (df["dart_slt"] > 0).astype(int)

            # Split by year
            train_data = df[df["year"] == 2024].copy()
            val_data = df[df["year"] == 2025].copy()

            print(f"   Training samples: {len(train_data)}")
            print(f"   Validation samples: {len(val_data)}")

            # Verify all features are numeric
            feature_data = train_data[feature_names]
            non_numeric = feature_data.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric) > 0:
                print(f"   ⚠️ Non-numeric features found: {list(non_numeric)}")
                # Filter out non-numeric features
                numeric_features = feature_data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                feature_names = numeric_features
                print(f"   ✅ Using {len(feature_names)} numeric features")

        except Exception as e:
            print(f"   ❌ Error loading dataset for {sp}: {e}")
            continue

        for hour_type, hour in representative_hours.items():
            print(f"\n   🎯 Tuning Hour {hour} ({hour_type} performer)")

            # Filter data for this hour
            hour_train = train_data[train_data["end_of_hour"] == hour].copy()
            hour_val = val_data[val_data["end_of_hour"] == hour].copy()

            if len(hour_train) == 0 or len(hour_val) == 0:
                print(f"      ⚠️  Insufficient data for Hour {hour}")
                continue

            # Prepare features and targets
            X_train = hour_train[feature_names]
            y_train = hour_train["dart_sign"]
            X_val = hour_val[feature_names]
            y_val = hour_val["dart_sign"]

            print(
                f"      Training: {len(X_train)} samples, {y_train.mean():.1%} positive"
            )
            print(
                f"      Validation: {len(X_val)} samples, {y_val.mean():.1%} positive"
            )
            print(f"      Feature matrix shape: {X_train.shape}")

            # Verify no missing values
            if X_train.isnull().any().any() or X_val.isnull().any().any():
                print(f"      ⚠️  Missing values detected, filling with median")
                X_train = X_train.fillna(X_train.median())
                X_val = X_val.fillna(
                    X_train.median()
                )  # Use training median for validation

            # Check for edge cases
            if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
                print(f"      ⚠️  Single class in data, skipping Hour {hour}")
                continue

            # Create Optuna study
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(),
                study_name=f"{sp}_hour_{hour}_{hour_type}",
            )

            # Optimize with timeout for reliability
            try:
                study.optimize(
                    lambda trial: objective(
                        trial, X_train, y_train, X_val, y_val, hour, sp
                    ),
                    n_trials=30,  # Reduced for faster execution
                    timeout=180,  # 3 minutes per hour
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
                    "n_features": len(feature_names),
                    **best_params,  # Include all hyperparameters
                }
                all_results.append(result)

            except Exception as e:
                print(f"      ❌ Error tuning Hour {hour}: {e}")
                continue

    if not all_results:
        print("❌ No successful tuning results!")
        return None, None

    # Analyze results
    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("TUNING RESULTS SUMMARY")
    print("=" * 70)

    # Overall improvement
    baseline_accuracy = 0.536  # From our analysis
    avg_tuned_accuracy = results_df["val_accuracy"].mean()
    improvement = avg_tuned_accuracy - baseline_accuracy

    print(f"Baseline accuracy (XGBoost): {baseline_accuracy:.3f}")
    print(f"Tuned accuracy (RandomForest): {avg_tuned_accuracy:.3f}")
    print(f"Improvement: {improvement:.3f} ({improvement/baseline_accuracy*100:+.1f}%)")

    # Best performing configurations
    best_overall = results_df.loc[results_df["val_accuracy"].idxmax()]
    print(f"\nBest overall result:")
    print(
        f"  {best_overall['settlement_point']} Hour {best_overall['hour']} ({best_overall['hour_type']})"
    )
    print(f"  Validation accuracy: {best_overall['val_accuracy']:.3f}")
    print(f"  Overfitting gap: {best_overall['overfitting_gap']:.3f}")
    print(f"  Features used: {best_overall['n_features']}")

    # Parameter analysis
    print(f"\nOptimal parameter patterns:")
    if "max_depth" in results_df.columns:
        print(
            f"  Max depth: {results_df['max_depth'].min()} - {results_df['max_depth'].max()} (median: {results_df['max_depth'].median():.0f})"
        )
    if "n_estimators" in results_df.columns:
        print(
            f"  N estimators: {results_df['n_estimators'].min()} - {results_df['n_estimators'].max()} (median: {results_df['n_estimators'].median():.0f})"
        )
    if "min_samples_split" in results_df.columns:
        print(
            f"  Min samples split: {results_df['min_samples_split'].min()} - {results_df['min_samples_split'].max()} (median: {results_df['min_samples_split'].median():.0f})"
        )
    if "min_samples_leaf" in results_df.columns:
        print(
            f"  Min samples leaf: {results_df['min_samples_leaf'].min()} - {results_df['min_samples_leaf'].max()} (median: {results_df['min_samples_leaf'].median():.0f})"
        )

    # Class weight effectiveness
    if "class_weight" in results_df.columns:
        print(f"\nClass weight effectiveness:")
        for cw in results_df["class_weight"].unique():
            subset = results_df[results_df["class_weight"] == cw]
            avg_acc = subset["val_accuracy"].mean()
            avg_gap = subset["overfitting_gap"].mean()
            count = len(subset)
            print(
                f"  {cw}: {avg_acc:.3f} accuracy, {avg_gap:.3f} overfitting gap ({count} results)"
            )

    # Save results
    results_df.to_csv("hyperparameter_tuning_results_with_dataset.csv", index=False)
    print(
        f"\n💾 Detailed results saved to: hyperparameter_tuning_results_with_dataset.csv"
    )

    # Translate insights to XGBoost recommendations
    print(f"\n🎯 XGBOOST PARAMETER RECOMMENDATIONS:")
    print("=" * 70)
    print("Based on RandomForest tuning patterns:")

    # Conservative parameters based on RF insights
    avg_max_depth = (
        results_df["max_depth"].median() if "max_depth" in results_df.columns else 3
    )
    avg_n_est = (
        results_df["n_estimators"].median()
        if "n_estimators" in results_df.columns
        else 50
    )
    avg_min_split = (
        results_df["min_samples_split"].median()
        if "min_samples_split" in results_df.columns
        else 5
    )

    # Best class weight strategy
    best_class_weight = None
    if "class_weight" in results_df.columns:
        cw_performance = results_df.groupby("class_weight")["val_accuracy"].mean()
        best_class_weight = cw_performance.idxmax()

    xgb_recommendations = {
        "max_depth": max(2, min(5, int(avg_max_depth))),  # Conservative depth
        "n_estimators": max(25, min(100, int(avg_n_est))),  # Conservative trees
        "learning_rate": 0.05,  # Slower learning to reduce overfitting
        "subsample": 0.8,  # Row randomness
        "colsample_bytree": 0.8,  # Feature randomness
        "reg_alpha": 1.0,  # L1 regularization
        "reg_lambda": 2.0,  # L2 regularization
        "min_child_weight": max(1, int(avg_min_split / 2)),  # Conservative splits
        "scale_pos_weight": 1.35,  # From our class analysis
        "early_stopping_rounds": 10,  # Prevent overfitting
    }

    for param, value in xgb_recommendations.items():
        print(f"  {param}: {value}")

    if best_class_weight:
        print(f"\nBest class balancing strategy: {best_class_weight}")
        if best_class_weight in ["balanced", "balanced_subsample"]:
            print("  → Use scale_pos_weight for XGBoost instead")

    # Save recommendations
    pd.Series(xgb_recommendations).to_csv("recommended_xgboost_params_from_dataset.csv")
    print(
        f"\n💾 XGBoost recommendations saved to: recommended_xgboost_params_from_dataset.csv"
    )

    return results_df, xgb_recommendations


if __name__ == "__main__":
    try:
        results, xgb_params = tune_representative_hours()
        if results is not None:
            print("\n✅ Hyperparameter tuning complete!")
        else:
            print("\n❌ Hyperparameter tuning failed!")
    except Exception as e:
        print(f"❌ Error during tuning: {e}")
        import traceback

        traceback.print_exc()
