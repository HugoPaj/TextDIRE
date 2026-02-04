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


def extract_late_stage_features(token_correctness: np.ndarray, window_size: int = 20) -> dict:
    """
    Late-stage stability features - AI text stabilizes in second half.
    Based on TSD paper (arxiv.org/abs/2601.04833).

    Args:
        token_correctness: Boolean array of shape (n_tokens,) where True = correct
        window_size: Window size for local volatility calculation

    Returns:
        Dictionary with 2 features:
        - derivative_dispersion: Std of |diff| in second half
        - local_volatility: Mean of local stds in sliding window
    """
    x = token_correctness.astype(float)
    n = len(x)

    if n < 10:
        return {'derivative_dispersion': 0.0, 'local_volatility': 0.0}

    # Use second half only
    second_half = x[n // 2:]

    # Derivative Dispersion: std of |diff| in second half
    diffs = np.abs(np.diff(second_half))
    derivative_dispersion = np.std(diffs) if len(diffs) > 1 else 0.0

    # Local Volatility: mean of local stds in sliding window
    local_stds = []
    for i in range(max(1, len(second_half) - window_size)):
        window = second_half[i:i + window_size]
        if len(window) > 1:
            local_stds.append(np.std(window))
    local_volatility = np.mean(local_stds) if local_stds else 0.0

    return {
        'derivative_dispersion': float(derivative_dispersion),
        'local_volatility': float(local_volatility),
    }


def extract_stylometric_features(text: str) -> dict:
    """
    Text-level stylometric features.
    Based on Ghostbuster and LOG-AID papers.

    Args:
        text: The raw text string

    Returns:
        Dictionary with 3 features:
        - type_token_ratio: Vocabulary richness (unique words / total words)
        - avg_sentence_length: Average number of words per sentence
        - sentence_length_variance: Variance in sentence lengths
    """
    words = text.lower().split()
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    if not words:
        return {
            'type_token_ratio': 0.0,
            'avg_sentence_length': 0.0,
            'sentence_length_variance': 0.0
        }

    # Type-Token Ratio (vocabulary richness)
    type_token_ratio = len(set(words)) / len(words)

    # Sentence statistics
    sentence_lengths = [len(s.split()) for s in sentences] if sentences else [0]
    avg_sentence_length = np.mean(sentence_lengths)
    sentence_length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

    return {
        'type_token_ratio': float(type_token_ratio),
        'avg_sentence_length': float(avg_sentence_length),
        'sentence_length_variance': float(sentence_length_variance),
    }


def extract_all_features(token_correctness: np.ndarray, text: str) -> dict:
    """
    Extract all 14 features (9 DivEye + 2 late-stage + 3 stylometric).

    Args:
        token_correctness: Boolean array of shape (n_tokens,) where True = correct
        text: The raw text string

    Returns:
        Dictionary with 14 features
    """
    features = {}
    features.update(extract_diveye_features(token_correctness))
    features.update(extract_late_stage_features(token_correctness))
    features.update(extract_stylometric_features(text))
    return features


def get_all_feature_names() -> list:
    """Return ordered list of all 14 feature names."""
    return [
        # DivEye features (9)
        'mean_accuracy', 'variance', 'skewness', 'kurtosis',
        'diff_mean', 'diff_variance',
        'diff2_variance', 'diff2_entropy', 'diff2_autocorr',
        # Late-stage features (2)
        'derivative_dispersion', 'local_volatility',
        # Stylometric features (3)
        'type_token_ratio', 'avg_sentence_length', 'sentence_length_variance',
    ]
