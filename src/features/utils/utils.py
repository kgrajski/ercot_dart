"""Utility functions for feature engineering and analysis."""

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def signed_log_transform(data):
    """Apply signed logarithm transformation to price differences.

    The signed logarithm transformation is defined as:
    z = sign(x) * log(1 + |x|)

    This transformation:
    - Preserves the sign of the input (positive stays positive, negative stays negative)
    - Compresses large values using logarithmic scaling
    - Maps zero to zero
    - Is symmetric: signed_log(-x) = -signed_log(x)
    - Properly handles missing values (NaN)

    This is particularly useful for electricity price differences (like DART)
    which can have extreme values and benefit from outlier-resistant transformations.

    Args:
        data: pandas Series or scalar value

    Returns:
        pandas Series (if input is Series) or scalar with same structure as input

    Example:
        # For a single value
        transformed = signed_log_transform(15.7)

        # For a pandas Series (e.g., DART column)
        df["dart_transformed"] = signed_log_transform(df["dart"])
    """
    if isinstance(data, pd.Series):
        # Pandas-native approach with proper NaN handling
        return data.apply(
            lambda x: np.sign(x) * np.log(1 + abs(x)) if pd.notna(x) else np.nan
        )
    else:
        # Handle scalar values
        if pd.notna(data):
            return np.sign(data) * np.log(1 + abs(data))
        else:
            return np.nan


def compute_power_spectrum(
    time_series: np.ndarray, timestamps: pd.Series, peak_percentile: float = 85
) -> dict:
    """Compute power spectrum analysis of a time series using FFT.

    Performs Fast Fourier Transform (FFT) on the input time series to identify
    periodic patterns and frequency components.

    Args:
        time_series: 1D numpy array of time series values
        timestamps: Pandas Series of timestamps corresponding to the time series
        peak_percentile: Percentile threshold for identifying spectral peaks (default: 85)

    Returns:
        dict containing:
            - freq_bins_per_day: Frequency bins in cycles per day
            - power_spectrum_db: Power spectral density in dB
            - sampling_freq_per_day: Sampling frequency in cycles per day
            - n_samples: Number of samples in the time series
            - dc_power_db: DC component power in dB
            - peak_indices: List of indices of spectral peaks
            - peak_frequencies: List of peak frequencies in cycles per day
            - peak_periods: List of peak periods in hours
            - peak_powers: List of peak powers in dB
    """
    # Convert timestamps to pandas datetime if not already
    timestamps = pd.to_datetime(timestamps)

    # Calculate sampling frequency
    time_diffs = timestamps.diff().dropna()
    median_dt = time_diffs.median()

    if median_dt.total_seconds() == 0:
        raise ValueError("Zero time differences found in timestamps")

    sampling_freq_hz = 1 / median_dt.total_seconds()  # Hz
    sampling_freq_per_day = sampling_freq_hz * 86400  # cycles per day

    # Perform FFT
    n_samples = len(time_series)
    fft_values = np.fft.fft(time_series)

    # Calculate one-sided power spectrum
    n_oneside = n_samples // 2
    fft_magnitude = np.abs(fft_values[:n_oneside])

    # Convert to power spectral density and then to dB
    power_spectrum = (fft_magnitude**2) / n_samples
    power_spectrum[1:] *= 2  # Double power for positive frequencies (except DC)

    # Convert to dB
    power_spectrum_db = 10 * np.log10(power_spectrum + 1e-12)

    # Calculate frequency axis in cycles per day
    freq_bins_hz = np.fft.fftfreq(n_samples, d=1 / sampling_freq_hz)[:n_oneside]
    freq_bins_per_day = freq_bins_hz * 86400  # Convert Hz to cycles per day

    # Find dominant frequencies (spectral peaks)
    peak_indices = []
    peak_frequencies = []
    peak_periods = []
    peak_powers = []

    if len(power_spectrum_db) > 1:
        peak_threshold = np.percentile(power_spectrum_db[1:], peak_percentile)

        for j in range(1, len(power_spectrum_db)):
            if power_spectrum_db[j] > peak_threshold:
                # Check if it's a local maximum
                is_peak = True
                window = 3
                for k in range(
                    max(1, j - window), min(len(power_spectrum_db), j + window + 1)
                ):
                    if k != j and power_spectrum_db[k] > power_spectrum_db[j]:
                        is_peak = False
                        break
                if is_peak:
                    peak_indices.append(j)
                    freq_cpd = freq_bins_per_day[j]
                    peak_frequencies.append(freq_cpd)
                    period_hours = 24 / freq_cpd if freq_cpd > 0 else float("inf")
                    peak_periods.append(period_hours)
                    peak_powers.append(power_spectrum_db[j])

    return {
        "freq_bins_per_day": freq_bins_per_day,
        "power_spectrum_db": power_spectrum_db,
        "sampling_freq_per_day": sampling_freq_per_day,
        "n_samples": n_samples,
        "dc_power_db": power_spectrum_db[0],
        "peak_indices": peak_indices,
        "peak_frequencies": peak_frequencies,
        "peak_periods": peak_periods,
        "peak_powers": peak_powers,
    }


def compute_kmeans_clustering(
    time_series: np.ndarray, max_k: int = 10, random_state: int = 42
) -> dict:
    """Perform K-means clustering analysis on a time series to identify natural groupings.

    Computes K-means clustering for K=1 to max_k and identifies optimal K using
    elbow method (inertia) and silhouette analysis. Designed for signed log transformed
    financial data which should be well-behaved.

    Args:
        time_series: 1D numpy array of time series values
        max_k: Maximum number of clusters to evaluate (default: 10)
        random_state: Random state for reproducible results (default: 42)

    Returns:
        dict containing:
            - k_values: Array of K values tested (1 to max_k)
            - inertias: Within-cluster sum of squares for each K
            - silhouette_scores: Silhouette scores for each K (None for K=1)
            - optimal_k: Optimal K selected using elbow method
            - optimal_model: Fitted KMeans model with optimal K
            - cluster_centers: Cluster centers for optimal K
            - labels: Cluster labels for each data point using optimal K
            - elbow_k: K value at elbow point in inertia curve
    """
    # Remove any NaN or infinite values
    time_series = time_series[np.isfinite(time_series)]

    if len(time_series) < max_k * 5:
        raise ValueError(
            f"Insufficient data points ({len(time_series)}) for clustering with max_k={max_k}"
        )

    # Reshape data for sklearn (needs 2D array)
    X = time_series.reshape(-1, 1)

    # Suppress sklearn warnings during clustering (these are expected with financial data)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        warnings.filterwarnings("ignore", message=".*overflow encountered.*")
        warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
        warnings.filterwarnings("ignore", message=".*divide by zero.*")

        # Initialize storage for results
        k_values = np.arange(1, max_k + 1)
        inertias = []
        silhouette_scores_list = []

        # Fit K-means for each K value
        models = {}
        for k in k_values:
            try:
                # Use improved parameters for better stability
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init=10,  # Multiple initializations
                    max_iter=300,  # Standard iterations
                    init="k-means++",  # Smart initialization
                    algorithm="lloyd",  # Most stable algorithm
                )

                kmeans.fit(X)
                models[k] = kmeans

                # Store inertia (within-cluster sum of squares)
                inertias.append(kmeans.inertia_)

                # Calculate silhouette score (only meaningful for K >= 2)
                if k >= 2 and len(np.unique(kmeans.labels_)) > 1:
                    sil_score = silhouette_score(X, kmeans.labels_)
                    silhouette_scores_list.append(sil_score)
                else:
                    silhouette_scores_list.append(None)

            except Exception as e:
                print(f"Warning: K-means failed for K={k}: {e}")
                # Use a reasonable fallback value for failed clustering
                inertias.append(inertias[-1] * 1.5 if inertias else 1e6)
                silhouette_scores_list.append(None)
                models[k] = None

    # Convert to numpy arrays
    inertias = np.array(inertias)

    # Find optimal K using elbow method
    optimal_k = find_elbow_point(k_values, inertias)

    # Get optimal model and results
    optimal_model = models.get(optimal_k)
    if optimal_model is None:
        # Fallback to K=2 if optimal model failed
        optimal_k = 2
        optimal_model = models.get(optimal_k)
        if optimal_model is None:
            raise ValueError("All K-means clustering attempts failed")

    cluster_centers = optimal_model.cluster_centers_.flatten()
    labels = optimal_model.labels_

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores_list,
        "optimal_k": optimal_k,
        "optimal_model": optimal_model,
        "cluster_centers": cluster_centers,
        "labels": labels,
        "elbow_k": optimal_k,
        "models": models,
    }


def find_elbow_point(k_values: np.ndarray, inertias: np.ndarray) -> int:
    """Find the elbow point in the K-means inertia curve using the knee detection method.

    Uses the "knee" detection algorithm to find the point of maximum curvature
    in the inertia vs K curve.

    Args:
        k_values: Array of K values
        inertias: Array of inertia values corresponding to each K

    Returns:
        int: Optimal K value at the elbow point
    """
    # Ensure inputs are numpy arrays
    k_values = np.array(k_values)
    inertias = np.array(inertias)

    # Handle edge cases
    if len(k_values) < 3:
        return max(2, k_values[0])

    # Check for invalid inertias
    if not np.all(np.isfinite(inertias)):
        # Find first valid K value >= 2
        valid_mask = np.isfinite(inertias) & (k_values >= 2)
        if np.any(valid_mask):
            return k_values[valid_mask][0]
        else:
            return 2

    # Normalize the data to [0, 1] range for knee detection
    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    inertia_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())

    # Calculate distances from each point to the line connecting first and last points
    distances = []
    for i in range(len(k_norm)):
        # Distance from point to line formula
        x0, y0 = k_norm[i], inertia_norm[i]
        x1, y1 = k_norm[0], inertia_norm[0]  # First point
        x2, y2 = k_norm[-1], inertia_norm[-1]  # Last point

        # Distance from point (x0, y0) to line from (x1, y1) to (x2, y2)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if denominator == 0:
            distance = 0
        else:
            distance = (
                abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / denominator
            )
        distances.append(distance)

    # Find the point with maximum distance (the elbow)
    elbow_idx = np.argmax(distances)
    optimal_k = k_values[elbow_idx]

    # Ensure we return at least K=2 (K=1 is trivial)
    return max(2, optimal_k)
