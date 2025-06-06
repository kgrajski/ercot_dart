# ercot_dart

## First Look: ERCOT Real-Time vs DAM Settlement Point Prices Using LZ (Houston) as an Example

This initial analysis explores the hourly differences between ERCOT real-time market (RTM) and day-ahead market (DAM) settlement prices — commonly referred to as **DART** (RTM minus DAM). We use **LZ_HOUSTON** as a representative settlement point. The focus is exploratory: understanding DART's statistical behavior, periodic structure, and temporal dynamics. The dataset spans **January 1, 2024 through June 5, 2025**, and was downloaded from ERCOT’s Public API.

---

## Temporal Dynamics

![DART Time Series](reports/figures/initial_dart_houston/dart_by_location_LZ_HOUSTON_LZ.png)

Raw and Signed Log Transformed (SLT) DART series both show:
- Frequent, high-amplitude price excursions
- Short-lived spikes, typically lasting 1–3 hours
- SLT transformation reveals structure while compressing extremes

---

## Distributional Behavior

![Raw vs SLT Distributions](reports/figures/initial_dart_houston/dart_distributions_LZ_HOUSTON_LZ.png)

The raw DART distribution is sharply peaked near zero with long tails. The SLT view:
- Symmetrizes the data
- Highlights heavy tails and potential regime separation
- Lends itself to further density- or cluster-based analyses

---

![Bimodal Histogram](reports/figures/initial_dart_houston/dart_slt_bimodal_LZ_HOUSTON_LZ.png)

A closer look at SLT-positive and SLT-negative distributions (absolute-valued) reveals:
- Right-skewed histograms in both directions
- Departure from normality despite moderately good fit in log-space
- Significant asymmetry between positive and negative excursions

---

## Cluster Analysis

To further characterize DART behaviors by intensity:

![KMeans Bimodal](reports/figures/initial_dart_houston/dart_slt_kmeans_bimodal_LZ_HOUSTON_LZ.png)

Separate K-means cluster analyses for positive and negative SLT values suggest:
- **Three clusters** are a natural segmentation for both sides
- Cluster centers capture mild, moderate, and extreme DART conditions

![KMeans Unimodal](reports/figures/initial_dart_houston/dart_slt_kmeans_unimodal_LZ_HOUSTON_LZ.png)

When considered unimodally (signed SLT values together), **two clusters** dominate:
- A likely regime switch between negative and positive pricing
- Useful for coarse predictive modeling or early warning indicators

---

## Moving Window Dynamics

![Rolling Statistics](reports/figures/initial_dart_houston/dart_slt_moving_window_stats_LZ_HOUSTON_LZ.png)

Using 168-hour (weekly) rolling windows:
- Standard deviation, skewness, and kurtosis exhibit temporal clustering
- Some intervals show high volatility and non-Gaussian shapes
- Positive sign rate fluctuates seasonally, with sustained deviation from 50%

---

## Cyclic and Frequency Structure

![SLT Spectrum](reports/figures/initial_dart_houston/dart_slt_power_spectrum_LZ_HOUSTON_LZ.png)

SLT power spectrum highlights:
- A strong peak near **1 cycle/day** (diurnal behavior)
- Secondary structure across multiple frequencies, suggesting load/congestion interaction

![Sign Heatmap](reports/figures/initial_dart_houston/dart_slt_sign_daily_heatmap_LZ_HOUSTON_LZ.png)

Hourly positivity rates by day-of-week show:
- **Persistent negativity** during business hours
- More positive DARTs overnight and early morning
- Saturdays and Sundays are more balanced

![Sign Spectrum](reports/figures/initial_dart_houston/dart_slt_sign_power_spectrum_LZ_HOUSTON_LZ.png)

Binary sign sequence analysis confirms:
- **Strong diurnal periodicity**
- Reaffirmed structure in the direction of the DART signal, not just magnitude

---

## Sign Transition Behavior

![Sign Transition Summary](reports/figures/initial_dart_houston/dart_slt_sign_transitions_LZ_HOUSTON_LZ.png)

Key insights from sign-transition analysis:
- High persistence in both positive and negative DART signs
- Most regime changes are short but non-random
- Elevated switching at **1AM, 9AM, and 11PM**, aligned with operational or market shifts

---

These findings offer a strong empirical foundation for DART forecasting, risk-aware bidding, and congestion-sensitivity modeling. Future work will move beyond descriptive analytics into probabilistic classification, time-series modeling, and ultimately prescriptive bidding support.

---
