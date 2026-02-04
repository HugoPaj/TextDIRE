"""
DivEye-style surprisal diversity features for Text-DIRE.

Based on the insight that human text has irregular reconstruction patterns
(errors cluster and vary) while AI text reconstructs uniformly.

Features extracted:
- Distributional: mean, variance, skewness, kurtosis
- First-order temporal: diff_mean, diff_variance
- Second-order temporal: diff2_variance, diff2_entropy, diff2_autocorr
"""

import numpy as np
from scipy import stats


def extract_diveye_features(token_correctness: np.ndarray) -> dict:
    """
    Extract 9 DivEye features from per-token reconstruction correctness.

    Args:
        token_correctness: Boolean array of shape (n_tokens,) where True = correct

    Returns:
        Dictionary with 9 features:
        - mean_accuracy: Average reconstruction accuracy
        - variance: Spread in per-token correctness
        - skewness: Asymmetry (human: more rare failures)
        - kurtosis: Heavy-tailed behavior
        - diff_mean: Mean of position-to-position accuracy changes
        - diff_variance: Variance of sequential changes
        - diff2_variance: Variance of change-in-change
        - diff2_entropy: Entropy of second differences (discretized)
        - diff2_autocorr: Autocorrelation of second differences
    """
    # Convert to float (1.0 = correct, 0.0 = incorrect)
    x = token_correctness.astype(float)
    n = len(x)

    if n < 4:
        return _default_features()

    # Distributional features
    mean_acc = np.mean(x)
    variance = np.var(x, ddof=1) if n > 1 else 0.0
    skewness = stats.skew(x) if n > 2 else 0.0
    kurtosis = stats.kurtosis(x) if n > 3 else 0.0

    # First-order differences
    dx = np.diff(x)  # Shape: (n-1,)
    diff_mean = np.mean(dx) if len(dx) > 0 else 0.0
    diff_variance = np.var(dx, ddof=1) if len(dx) > 1 else 0.0

    # Second-order differences
    d2x = np.diff(dx)  # Shape: (n-2,)
    diff2_variance = np.var(d2x, ddof=1) if len(d2x) > 1 else 0.0

    # Entropy of discretized second differences
    if len(d2x) > 0:
        # Discretize to bins: negative, zero, positive
        bins = np.digitize(d2x, bins=[-0.5, 0.5])
        _, counts = np.unique(bins, return_counts=True)
        probs = counts / counts.sum()
        diff2_entropy = -np.sum(probs * np.log(probs + 1e-10))
    else:
        diff2_entropy = 0.0

    # Autocorrelation of second differences
    if len(d2x) > 1:
        d2x_centered = d2x - np.mean(d2x)
        var_d2x = np.var(d2x)
        if var_d2x > 1e-10:
            autocorr_full = np.correlate(d2x_centered, d2x_centered, mode='full')
            # Lag-0 autocorrelation normalized
            diff2_autocorr = autocorr_full[len(d2x_centered)] / (var_d2x * len(d2x))
        else:
            diff2_autocorr = 0.0
    else:
        diff2_autocorr = 0.0

    return {
        'mean_accuracy': float(mean_acc),
        'variance': float(variance),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'diff_mean': float(diff_mean),
        'diff_variance': float(diff_variance),
        'diff2_variance': float(diff2_variance),
        'diff2_entropy': float(diff2_entropy),
        'diff2_autocorr': float(diff2_autocorr),
    }


def _default_features() -> dict:
    """Return default feature values for sequences too short to analyze."""
    return {
        'mean_accuracy': 0.5,
        'variance': 0.0,
        'skewness': 0.0,
        'kurtosis': 0.0,
        'diff_mean': 0.0,
        'diff_variance': 0.0,
        'diff2_variance': 0.0,
        'diff2_entropy': 0.0,
        'diff2_autocorr': 0.0,
    }


def features_to_array(features: dict) -> np.ndarray:
    """Convert feature dictionary to numpy array in consistent order."""
    keys = [
        'mean_accuracy', 'variance', 'skewness', 'kurtosis',
        'diff_mean', 'diff_variance',
        'diff2_variance', 'diff2_entropy', 'diff2_autocorr'
    ]
    return np.array([features[k] for k in keys])


def get_feature_names() -> list:
    """Return ordered list of feature names."""
    return [
        'mean_accuracy', 'variance', 'skewness', 'kurtosis',
        'diff_mean', 'diff_variance',
        'diff2_variance', 'diff2_entropy', 'diff2_autocorr'
    ]
