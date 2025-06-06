# ercot_dart

## First Look: ERCOT Real-Time vs DAM Settlement Point Prices Using LZ (Houston) as an Example

This initial analysis explores the hourly differences between ERCOT real-time market (RTM) and day-ahead market (DAM) settlement prices — commonly referred to as **DART** (RTM minus DAM). We use **LZ_HOUSTON** as a representative settlement point. The focus is exploratory: understanding DART's statistical behavior, periodic structure, and temporal dynamics.

The dataset spans **January 1, 2024 through June 5, 2025**, and the data products needed to support these computations were downloaded via the [ERCOT Public API](https://www.ercot.com/mp/data-products).

---

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

Hourly sign probability by day of week shows:
- Consistently **negative DARTs** during business hours
- Higher **positive DART** probability overnight and early morning
- Weekends display flatter patterns

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
- Usefulness of regime classification over raw regression
- Opportunities for integrating time-aware and probability-
