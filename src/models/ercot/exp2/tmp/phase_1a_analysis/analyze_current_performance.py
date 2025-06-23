"""Current Model Performance Analysis.

Analyzes the existing XGBoost classification results to understand
which hours and classes are most problematic.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def analyze_current_performance():
    """Analyze current XGBoost classification performance."""

    print("📊 Current XGBoost Classification Performance Analysis")
    print("=" * 70)

    settlement_points = ["LZ_HOUSTON_LZ", "LZ_HOUSTON_LZEW"]
    base_path = "data/studies/exp2/2025-06-04"

    all_results = []

    # Load results from both settlement points
    for sp in settlement_points:
        results_path = f"{base_path}/{sp}/modeling/xgboost_classification/results_xgboost_classification.csv"
        try:
            df = pd.read_csv(results_path)
            df["settlement_point"] = sp
            all_results.append(df)
            print(f"✅ Loaded {sp} results: {len(df)} hours")
        except Exception as e:
            print(f"❌ Error loading {sp}: {e}")

    if not all_results:
        print("No results loaded. Exiting.")
        return

    # Combine results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Overall performance summary
    print("\n" + "=" * 70)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 70)

    avg_train_acc = combined_results["train_accuracy"].mean()
    avg_val_acc = combined_results["validation_accuracy"].mean()
    avg_val_f1 = combined_results["validation_f1"].mean()

    print(f"Average Training Accuracy: {avg_train_acc:.3f}")
    print(f"Average Validation Accuracy: {avg_val_acc:.3f}")
    print(f"Average Validation F1: {avg_val_f1:.3f}")
    print(f"Overfitting Gap: {avg_train_acc - avg_val_acc:.3f}")

    # Per-hour performance analysis
    print("\n" + "=" * 70)
    print("PER-HOUR PERFORMANCE (Validation Accuracy)")
    print("=" * 70)

    # Average across both settlement points
    hourly_performance = (
        combined_results.groupby("end_hour")
        .agg(
            {
                "validation_accuracy": ["mean", "std"],
                "validation_f1": ["mean", "std"],
                "train_accuracy": "mean",
            }
        )
        .round(3)
    )

    # Flatten column names
    hourly_performance.columns = [
        "val_acc_mean",
        "val_acc_std",
        "val_f1_mean",
        "val_f1_std",
        "train_acc",
    ]
    hourly_performance["overfitting_gap"] = (
        hourly_performance["train_acc"] - hourly_performance["val_acc_mean"]
    )

    print("Hour  Val_Acc  Val_F1   Train_Acc  Overfitting_Gap")
    print("-" * 50)
    for hour in range(1, 25):
        if hour in hourly_performance.index:
            row = hourly_performance.loc[hour]
            print(
                f"{hour:4d}  {row['val_acc_mean']:.3f}   {row['val_f1_mean']:.3f}    {row['train_acc']:.3f}      {row['overfitting_gap']:.3f}"
            )

    # Identify best and worst performing hours
    best_hours = hourly_performance.nlargest(5, "val_acc_mean")
    worst_hours = hourly_performance.nsmallest(5, "val_acc_mean")

    print("\n" + "=" * 70)
    print("BEST PERFORMING HOURS (Validation Accuracy)")
    print("=" * 70)
    for hour, row in best_hours.iterrows():
        print(
            f"Hour {hour:2d}: {row['val_acc_mean']:.3f} accuracy, {row['val_f1_mean']:.3f} F1"
        )

    print("\n" + "=" * 70)
    print("WORST PERFORMING HOURS (Validation Accuracy)")
    print("=" * 70)
    for hour, row in worst_hours.iterrows():
        print(
            f"Hour {hour:2d}: {row['val_acc_mean']:.3f} accuracy, {row['val_f1_mean']:.3f} F1"
        )

    # Settlement point comparison
    print("\n" + "=" * 70)
    print("SETTLEMENT POINT COMPARISON")
    print("=" * 70)

    sp_performance = (
        combined_results.groupby("settlement_point")
        .agg(
            {
                "validation_accuracy": ["mean", "std"],
                "validation_f1": ["mean", "std"],
                "train_accuracy": "mean",
            }
        )
        .round(3)
    )

    for sp in settlement_points:
        sp_data = combined_results[combined_results["settlement_point"] == sp]
        val_acc_mean = sp_data["validation_accuracy"].mean()
        val_acc_std = sp_data["validation_accuracy"].std()
        val_f1_mean = sp_data["validation_f1"].mean()
        train_acc = sp_data["train_accuracy"].mean()

        print(f"\n{sp}:")
        print(f"  Validation Accuracy: {val_acc_mean:.3f} ± {val_acc_std:.3f}")
        print(f"  Validation F1: {val_f1_mean:.3f}")
        print(f"  Training Accuracy: {train_acc:.3f}")
        print(f"  Overfitting Gap: {train_acc - val_acc_mean:.3f}")

    # Random chance baseline
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    # Load class distribution analysis
    try:
        class_dist = pd.read_csv("class_distribution_analysis.csv", index_col=0)
        overall_ratio = class_dist.loc["overall_ratio"]

        # Random chance accuracy for imbalanced dataset
        neg_pct = 57.5  # From our analysis
        pos_pct = 42.5
        random_accuracy = (
            max(neg_pct, pos_pct) / 100
        )  # Baseline: always predict majority class

        print(f"Random Baseline (majority class): {random_accuracy:.3f}")
        print(f"Current Average Validation: {avg_val_acc:.3f}")
        print(f"Improvement over baseline: {avg_val_acc - random_accuracy:.3f}")

        # Hours that are worse than random baseline
        poor_hours = combined_results[
            combined_results["validation_accuracy"] < random_accuracy
        ]
        if len(poor_hours) > 0:
            print(
                f"\n⚠️  Hours performing worse than baseline: {len(poor_hours)} out of {len(combined_results)}"
            )
            for _, row in poor_hours.iterrows():
                print(
                    f"   Hour {row['end_hour']} ({row['settlement_point']}): {row['validation_accuracy']:.3f}"
                )
        else:
            print(f"\n✅ All hours perform better than random baseline")

    except Exception as e:
        print(f"Could not load class distribution analysis: {e}")

    # Generate tuning recommendations
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING RECOMMENDATIONS")
    print("=" * 70)

    # Representative hours for tuning
    worst_hour = worst_hours.index[0]
    best_hour = best_hours.index[0]
    median_hours = hourly_performance.sort_values("val_acc_mean")
    median_hour = median_hours.index[len(median_hours) // 2]

    print(f"Representative hours for grid search:")
    print(
        f"  Best performer: Hour {best_hour} ({best_hours.loc[best_hour, 'val_acc_mean']:.3f} accuracy)"
    )
    print(
        f"  Worst performer: Hour {worst_hour} ({worst_hours.loc[worst_hour, 'val_acc_mean']:.3f} accuracy)"
    )
    print(
        f"  Median performer: Hour {median_hour} ({hourly_performance.loc[median_hour, 'val_acc_mean']:.3f} accuracy)"
    )

    print(f"\nOverfitting indicators:")
    high_overfitting = hourly_performance[hourly_performance["overfitting_gap"] > 0.4]
    print(f"  Hours with >40% overfitting gap: {len(high_overfitting)}")
    if len(high_overfitting) > 0:
        print(
            f"  Worst overfitting: Hour {high_overfitting['overfitting_gap'].idxmax()} ({high_overfitting['overfitting_gap'].max():.3f} gap)"
        )

    # Save analysis
    analysis_results = {
        "avg_validation_accuracy": avg_val_acc,
        "avg_overfitting_gap": avg_train_acc - avg_val_acc,
        "best_hour": best_hour,
        "worst_hour": worst_hour,
        "median_hour": median_hour,
        "hours_worse_than_baseline": len(poor_hours) if "poor_hours" in locals() else 0,
    }

    pd.Series(analysis_results).to_csv("current_performance_analysis.csv")
    hourly_performance.to_csv("hourly_performance_analysis.csv")

    print(
        f"\n💾 Analysis saved to: current_performance_analysis.csv, hourly_performance_analysis.csv"
    )

    return analysis_results, hourly_performance


if __name__ == "__main__":
    try:
        summary, hourly_data = analyze_current_performance()
        print("\n✅ Performance analysis complete!")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
