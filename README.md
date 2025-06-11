# ERCOT DART Price Prediction

## Project Overview

ERCOT DART (Day-Ahead Real-Time) price prediction system for the Texas electricity market. This project analyzes and predicts the differences between day-ahead market (DAM) and real-time market (RTM) settlement prices in ERCOT, helping market participants understand price dynamics and make informed trading decisions.

**Key Features:**
- Historical ERCOT price data collection via official API
- Comprehensive exploratory data analysis of DART patterns
- Time series feature engineering for price prediction
- Machine learning models for DART forecasting
- Rich visualization suite for market analysis


## Temporal Dynamics

![DART Time Series](reports/figures/initial_dart_houston/dart_by_location_LZ_HOUSTON_LZ.png)

The raw and transformed DART series show:
- Frequent, high-amplitude price excursions
- Short-lived spikes (typically 1–3 hours)
- The **Signed Log Transform (SLT)** retains directional information while compressing extreme values:

  `SLT(x) = sign(x) * log(1 + abs(x))`

This helps reveal structure while preserving magnitude asymmetry.


---

## Distributional Behavior

![Raw vs SLT Distributions](reports/figures/initial_dart_houston/dart_distributions_LZ_HOUSTON_LZ.png)

- The raw DART distribution is sharply peaked near zero with long tails.
- SLT symmetrizes and normalizes magnitude for clearer tail inspection and segmentation.

---

![Bimodal Histogram](reports/figures/initial_dart_houston/dart_slt_bimodal_LZ_HOUSTON_LZ.png)

Separate histograms of positive and negative SLT values (shown as absolute):
- Reveal clear right-skew in both distributions
- Indicate heavy-tailed, non-Gaussian behavior
- Suggest different dynamics for positive vs negative deviations

---

## K-Means Cluster Analysis

To identify natural regime groupings:

![KMeans Bimodal](reports/figures/initial_dart_houston/dart_slt_kmeans_bimodal_LZ_HOUSTON_LZ.png)

Using K-means clustering on **positive and negative SLT magnitudes separately**:
- Optimal segmentation is **three clusters** on each side
- These correspond to mild, moderate, and severe pricing deviations

---

![KMeans Unimodal](reports/figures/initial_dart_houston/dart_slt_kmeans_unimodal_LZ_HOUSTON_LZ.png)

When clustering the **signed SLT values together**:
- Optimal **two-cluster solution** aligns with DART polarity (positive vs negative)
- Suggests sign alone captures the dominant regime division

---

## Moving Window Behavior

![Rolling Statistics](reports/figures/initial_dart_houston/dart_slt_moving_window_stats_LZ_HOUSTON_LZ.png)

Using a 168-hour rolling window:
- Standard deviation, skewness, and kurtosis vary significantly over time
- Indicates changing volatility and shape
- Positive rate drifts seasonally, suggesting persistent market bias

---

## Cyclic and Frequency Structure

![SLT Spectrum](reports/figures/initial_dart_houston/dart_slt_power_spectrum_LZ_HOUSTON_LZ.png)

Power spectrum of the SLT series:
- Strong peak at **1 cycle/day**, confirming diurnal periodicity
- Additional harmonic content indicates layered temporal structure

---

![Sign Heatmap](reports/figures/initial_dart_houston/dart_slt_sign_daily_heatmap_LZ_HOUSTON_LZ.png)

Hourly sign probability by day of week and hour shows:
- Consistently **negative DARTs** during business hours
  - Counterintuitive?  Could make sense if DAM forecasts tend to be conservative?
- Higher **positive DART** probability overnight and early morning
- Weekends display flatter patterns

---

![DART Average Heatmap](reports/figures/initial_dart_houston/dart_average_daily_heatmap_LZ_HOUSTON_LZ.png)

Hourly average DART by day of week and hour shows:
- Interpreted from the perspective of a **resource generator bidding into DAM**
  - **Positive DART (RT > DAM)** → **loss**: the generator misses potential upside by locking in a lower DAM price (red)
  - **Negative DART (RT < DAM)** → **gain**: the generator benefits from locking in a higher DAM price ahead of a lower real-time outcome (blue)
- Interpret very carefully, because we're looking at an aggregate view
  - Timespan of interest: **early weekday mornings**
  - **Gains** are more sporadic and tend to cluster overnight or late at night
  - Highlights the **risk asymmetry** and the value of selective hedging
  - These are first impressions only.  Do not trade based on this figure.

---

![Sign Spectrum](reports/figures/initial_dart_houston/dart_slt_sign_power_spectrum_LZ_HOUSTON_LZ.png)

Spectral analysis of the DART sign sequence shows:
- Strong daily cycle
- Temporal structure in sign alternation, not just magnitude

---

## Sign Transition Behavior

![Sign Transitions](reports/figures/initial_dart_houston/dart_slt_sign_transitions_LZ_HOUSTON_LZ.png)

Sign transition summary:
- ~80% persistence in sign from hour to hour
- Most runs last just 1–3 hours, but longer runs do occur
- Transitions often cluster at **1AM, 9AM, and 11PM**, possibly linked to load ramping or forecast updates

---

## Next Steps

These insights establish a strong foundation for downstream forecasting and classification models. They suggest:
- Rich temporal and categorical structure in DART behavior
- Availability of regime classification as a supplement to raw regression
- Opportunities for integrating time-aware and probability-based models
