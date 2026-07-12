"""
Image encoding utilities for NIR spectral signals.

Supported encodings:
- RP: Recurrence Plot
- GASF: Gramian Angular Summation Field
- GADF: Gramian Angular Difference Field
- MTF: Markov Transition Field
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

SUPPORTED_ENCODINGS = ("rp", "gasf", "gadf", "mtf")


def normalize_encoding_method(method):
    return str(method).lower()


def resample_series(series, target_length=None):
    """Resample a 1D series to a fixed length with linear interpolation."""
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 1:
        raise ValueError(f"Expected a 1D spectral series, got shape {series.shape}")
    if target_length is None or target_length <= 0 or len(series) == target_length:
        return series
    x_old = np.linspace(0.0, 1.0, num=len(series), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(x_new, x_old, series).astype(np.float32)


def minmax_scale(series, feature_range=(-1.0, 1.0)):
    """Scale a 1D series into a target range."""
    series = np.asarray(series, dtype=np.float32)
    min_val = float(np.min(series))
    max_val = float(np.max(series))
    low, high = feature_range
    if np.isclose(max_val, min_val):
        return np.full_like(series, (low + high) / 2.0, dtype=np.float32)
    scaled = (series - min_val) / (max_val - min_val)
    return (scaled * (high - low) + low).astype(np.float32)


def phase_space_reconstruction(series, m=1, tau=1):
    """Phase space reconstruction via time-delay embedding."""
    if m < 1:
        raise ValueError(f"Embedding dimension m must be >= 1, got {m}")
    if tau < 1:
        raise ValueError(f"Embedding delay tau must be >= 1, got {tau}")
    series = np.asarray(series, dtype=np.float32)
    n = len(series)
    if m == 1:
        return series.reshape(-1, 1)

    n_vectors = n - (m - 1) * tau
    if n_vectors <= 0:
        return series.reshape(-1, 1)

    vectors = np.zeros((n_vectors, m), dtype=np.float32)
    for i in range(n_vectors):
        for j in range(m):
            vectors[i, j] = series[i + j * tau]
    return vectors


def compute_rp_threshold(X_data, percentile=10, n_sample=30, m=1, tau=1, random_state=42,
                         target_length=None):
    """Compute a global RP threshold from a subset of samples."""
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    rng = np.random.RandomState(random_state)
    n_samples = X_data.shape[0]
    sample_indices = rng.choice(n_samples, min(n_sample, n_samples), replace=False)

    all_dists = []
    for idx in sample_indices:
        series = resample_series(X_data[idx], target_length=target_length)
        vectors = phase_space_reconstruction(series, m=m, tau=tau)
        if len(vectors) > 1:
            dists = pdist(vectors, metric='euclidean')
            if dists.size > 0:
                all_dists.append(dists)

    if not all_dists:
        return 0.0
    all_dists = np.concatenate(all_dists)
    return float(np.percentile(all_dists, percentile))


def recurrence_plot(series, threshold, m=1, tau=1, target_length=None):
    """Convert a 1D series into a recurrence plot image."""
    series = resample_series(series, target_length=target_length)
    vectors = phase_space_reconstruction(series, m=m, tau=tau)
    if len(vectors) == 0:
        return np.zeros((1, 1), dtype=np.float32)
    dist_matrix = squareform(pdist(vectors, metric='euclidean'))
    return (dist_matrix <= threshold).astype(np.float32)


def gramian_angular_summation_field(series, target_length=None):
    """Generate a GASF image."""
    series = resample_series(series, target_length=target_length)
    scaled = minmax_scale(series, feature_range=(-1.0, 1.0))
    scaled = np.clip(scaled, -1.0, 1.0)
    phi = np.arccos(scaled)
    gasf = np.cos(phi[:, None] + phi[None, :])
    return gasf.astype(np.float32)


def gramian_angular_difference_field(series, target_length=None):
    """Generate a GADF image."""
    series = resample_series(series, target_length=target_length)
    scaled = minmax_scale(series, feature_range=(-1.0, 1.0))
    scaled = np.clip(scaled, -1.0, 1.0)
    phi = np.arccos(scaled)
    gadf = np.sin(phi[:, None] - phi[None, :])
    return gadf.astype(np.float32)


def _quantile_bins(series, n_bins=8):
    series = np.asarray(series, dtype=np.float32)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(series, quantiles)
    bins = np.unique(bins)
    if len(bins) <= 2:
        return np.zeros_like(series, dtype=np.int64), np.array([series.min(), series.max()], dtype=np.float32)
    states = np.digitize(series, bins[1:-1], right=True)
    return states.astype(np.int64), bins.astype(np.float32)


def markov_transition_field(series, n_bins=8, target_length=None):
    """Generate a Markov Transition Field image."""
    if n_bins < 2:
        raise ValueError(f"MTF requires n_bins >= 2, got {n_bins}")
    series = resample_series(series, target_length=target_length)
    states, bins = _quantile_bins(series, n_bins=n_bins)
    n_states = max(int(states.max()) + 1, 1)
    if len(series) == 0:
        return np.zeros((1, 1), dtype=np.float32)

    transition = np.zeros((n_states, n_states), dtype=np.float32)
    for i in range(len(states) - 1):
        transition[states[i], states[i + 1]] += 1.0

    row_sum = transition.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    transition = transition / row_sum

    mtf = transition[states[:, None], states[None, :]]
    return mtf.astype(np.float32)


def encode_series(series, method='rp', target_length=None, rp_threshold=None, rp_m=1, rp_tau=1,
                  mtf_bins=8):
    """Dispatch helper for supported encodings."""
    method = normalize_encoding_method(method)
    if method == 'rp':
        if rp_threshold is None:
            raise ValueError("rp_threshold is required for RP encoding.")
        return recurrence_plot(series, rp_threshold, m=rp_m, tau=rp_tau, target_length=target_length)
    if method == 'gasf':
        return gramian_angular_summation_field(series, target_length=target_length)
    if method == 'gadf':
        return gramian_angular_difference_field(series, target_length=target_length)
    if method == 'mtf':
        return markov_transition_field(series, n_bins=mtf_bins, target_length=target_length)
    raise ValueError(f"Unsupported encoding method: {method}")
