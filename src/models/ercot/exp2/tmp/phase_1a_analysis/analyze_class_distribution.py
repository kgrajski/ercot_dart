"""Class Distribution Analysis for ERCOT DART Sign Prediction.

This script analyzes the distribution of positive vs negative DART prices
across different dimensions to understand class imbalance issues.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def analyze_class_distribution():
    """Comprehensive analysis of DART sign class distribution."""

    print("🔍 ERCOT DART Sign Distribution Analysis")
    print("=" * 60)

    # Load both settlement point datasets
    settlement_points = ["LZ_HOUSTON_LZ", "LZ_HOUSTON_LZEW"]
    base_path = "data/studies/exp2/2025-06-04"

    all_data = []

    for sp in settlement_points:
        file_path = f"{base_path}/{sp}/modeling/model_ready.csv"
        try:
            df = pd.read_csv(file_path)
            df["settlement_point"] = sp
            df["utc_ts"] = pd.to_datetime(df["utc_ts"])
            df["year"] = df["utc_ts"].dt.year
            df["month"] = df["utc_ts"].dt.month
            df["dart_sign"] = (df["dart_slt"] > 0).astype(
                int
            )  # 1 for positive, 0 for negative
            all_data.append(df)
            print(f"✅ Loaded {sp}: {len(df)} samples")
        except Exception as e:
            print(f"❌ Error loading {sp}: {e}")

    if not all_data:
        print("No data loaded. Exiting.")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} total samples")

    # Overall distribution
    print("\n" + "=" * 60)
    print("OVERALL SIGN DISTRIBUTION")
    print("=" * 60)

    overall_dist = combined_df["dart_sign"].value_counts().sort_index()
    overall_pct = (
        combined_df["dart_sign"].value_counts(normalize=True).sort_index() * 100
    )

    print(f"Negative DART (0): {overall_dist[0]:,} samples ({overall_pct[0]:.1f}%)")
    print(f"Positive DART (1): {overall_dist[1]:,} samples ({overall_pct[1]:.1f}%)")
    print(f"Imbalance ratio: {overall_dist[1] / overall_dist[0]:.3f}")

    # By settlement point
    print("\n" + "=" * 60)
    print("BY SETTLEMENT POINT")
    print("=" * 60)

    for sp in settlement_points:
        sp_data = combined_df[combined_df["settlement_point"] == sp]
        sp_dist = sp_data["dart_sign"].value_counts().sort_index()
        sp_pct = sp_data["dart_sign"].value_counts(normalize=True).sort_index() * 100

        print(f"\n{sp}:")
        print(f"  Negative: {sp_dist[0]:,} ({sp_pct[0]:.1f}%)")
        print(f"  Positive: {sp_dist[1]:,} ({sp_pct[1]:.1f}%)")
        print(f"  Ratio: {sp_dist[1] / sp_dist[0]:.3f}")

    # By year
    print("\n" + "=" * 60)
    print("BY YEAR (Train vs Validation)")
    print("=" * 60)

    for year in sorted(combined_df["year"].unique()):
        year_data = combined_df[combined_df["year"] == year]
        year_dist = year_data["dart_sign"].value_counts().sort_index()
        year_pct = (
            year_data["dart_sign"].value_counts(normalize=True).sort_index() * 100
        )

        dataset_type = "Training (2024)" if year == 2024 else "Validation (2025)"
        print(f"\n{dataset_type}:")
        print(f"  Negative: {year_dist[0]:,} ({year_pct[0]:.1f}%)")
        print(f"  Positive: {year_dist[1]:,} ({year_pct[1]:.1f}%)")
        print(f"  Ratio: {year_dist[1] / year_dist[0]:.3f}")

    # By hour
    print("\n" + "=" * 60)
    print("BY HOUR (Top 5 Most Imbalanced)")
    print("=" * 60)

    hourly_ratios = []
    for hour in range(1, 25):
        hour_data = combined_df[combined_df["end_of_hour"] == hour]
        if len(hour_data) > 0:
            hour_dist = hour_data["dart_sign"].value_counts().sort_index()
            if len(hour_dist) == 2:  # Both classes present
                ratio = hour_dist[1] / hour_dist[0]
                hourly_ratios.append(
                    {
                        "hour": hour,
                        "negative": hour_dist[0],
                        "positive": hour_dist[1],
                        "ratio": ratio,
                        "total": len(hour_data),
                    }
                )

    hourly_df = pd.DataFrame(hourly_ratios)
    hourly_df = hourly_df.sort_values("ratio")

    print("\nMost negative-biased hours:")
    for _, row in hourly_df.head(5).iterrows():
        print(
            f"  Hour {int(row['hour']):2d}: {int(row['negative']):3d} neg, {int(row['positive']):3d} pos (ratio: {row['ratio']:.3f})"
        )

    print("\nMost positive-biased hours:")
    for _, row in hourly_df.tail(5).iterrows():
        print(
            f"  Hour {int(row['hour']):2d}: {int(row['negative']):3d} neg, {int(row['positive']):3d} pos (ratio: {row['ratio']:.3f})"
        )

    # Monthly analysis
    print("\n" + "=" * 60)
    print("BY MONTH")
    print("=" * 60)

    monthly_stats = []
    for month in sorted(combined_df["month"].unique()):
        month_data = combined_df[combined_df["month"] == month]
        month_dist = month_data["dart_sign"].value_counts().sort_index()
        if len(month_dist) == 2:
            ratio = month_dist[1] / month_dist[0]
            pct_pos = month_dist[1] / (month_dist[0] + month_dist[1]) * 100
            monthly_stats.append(
                {
                    "month": month,
                    "negative": month_dist[0],
                    "positive": month_dist[1],
                    "pct_positive": pct_pos,
                    "ratio": ratio,
                }
            )

    monthly_df = pd.DataFrame(monthly_stats)
    print(
        f"{'Month':<6} {'Negative':<10} {'Positive':<10} {'% Positive':<12} {'Ratio':<8}"
    )
    print("-" * 50)
    for _, row in monthly_df.iterrows():
        print(
            f"{int(row['month']):5d} {int(row['negative']):9d} {int(row['positive']):9d} {row['pct_positive']:10.1f}% {row['ratio']:7.3f}"
        )

    # Generate summary for hyperparameter tuning
    print("\n" + "=" * 60)
    print("SUMMARY FOR HYPERPARAMETER TUNING")
    print("=" * 60)

    worst_hours = hourly_df.head(3)["hour"].tolist()
    best_hours = hourly_df.tail(3)["hour"].tolist()
    median_idx = len(hourly_df) // 2
    median_hour = hourly_df.iloc[median_idx]["hour"]

    print(f"Representative hours for tuning:")
    print(f"  Most imbalanced: {worst_hours}")
    print(f"  Least imbalanced: {best_hours}")
    print(f"  Median balance: Hour {median_hour}")

    # Class weights recommendation
    overall_neg = overall_dist[0]
    overall_pos = overall_dist[1]
    scale_pos_weight = overall_neg / overall_pos

    print(f"\nRecommended XGBoost scale_pos_weight: {scale_pos_weight:.3f}")
    print(
        f"This will balance the {overall_pct[0]:.1f}% negative vs {overall_pct[1]:.1f}% positive split"
    )

    # Save detailed analysis
    output_file = "class_distribution_analysis.csv"
    analysis_summary = {
        "overall_negative": overall_dist[0],
        "overall_positive": overall_dist[1],
        "overall_ratio": overall_dist[1] / overall_dist[0],
        "recommended_scale_pos_weight": scale_pos_weight,
        "worst_hour": worst_hours[0],
        "best_hour": best_hours[0],
        "median_hour": median_hour,
    }

    pd.Series(analysis_summary).to_csv(output_file)
    print(f"\n💾 Detailed analysis saved to: {output_file}")

    return analysis_summary, hourly_df


if __name__ == "__main__":
    try:
        summary, hourly_data = analyze_class_distribution()
        print("\n✅ Analysis complete!")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
