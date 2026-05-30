"""
Evaluation utilities for Text-DIRE experiments.

Includes AUROC computation, plotting, and statistical analysis.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result of detector evaluation."""
    auroc: float
    accuracy: float
    threshold: float
    precision: float
    recall: float
    f1: float

    # Optional detailed metrics
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None


def _oriented_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    pos_label: int = 1,
) -> tuple[np.ndarray, float]:
    """Return scores oriented so higher values indicate the positive class."""
    from sklearn.metrics import roc_auc_score

    auroc = roc_auc_score(labels, scores)
    if auroc < 0.5:
        return -scores, 1 - auroc
    return scores, auroc


def compute_auroc(
    labels: list[int],
    scores: list[float],
    pos_label: int = 1,
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        labels: True labels (0 or 1)
        scores: Prediction scores (higher = more likely positive)
        pos_label: Which label is positive

    Returns:
        AUROC score (0.5 = random, 1.0 = perfect)
    """
    _, auroc = _oriented_scores(np.array(labels), np.array(scores), pos_label)
    return auroc


def evaluate_detector(
    labels: list[int],
    scores: list[float],
    pos_label: int = 1,
    threshold: Optional[float] = None,
) -> EvaluationResult:
    """
    Comprehensive evaluation of a detector.

    Args:
        labels: True labels (0 = human, 1 = AI)
        scores: Prediction scores
        pos_label: Which label is positive (default: 1 = AI)
        threshold: Decision threshold (if None, uses optimal from ROC)

    Returns:
        EvaluationResult with all metrics
    """
    from sklearn.metrics import (
        roc_curve,
        precision_recall_fscore_support,
        accuracy_score,
    )

    labels = np.array(labels)
    scores = np.array(scores)

    scores, auroc = _oriented_scores(labels, scores, pos_label)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)

    # Find optimal threshold (Youden's J statistic)
    if threshold is None:
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = thresholds[best_idx]

    # Make predictions at threshold
    predictions = (scores >= threshold).astype(int)

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        pos_label=pos_label,
        average='binary',
        zero_division=0,
    )

    return EvaluationResult(
        auroc=auroc,
        accuracy=accuracy,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )


def plot_distributions(
    human_scores: list[float],
    ai_scores: list[float],
    title: str = "Score Distribution",
    xlabel: str = "Score",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """
    Plot score distributions for human and AI texts.

    Args:
        human_scores: Scores for human texts
        ai_scores: Scores for AI texts
        title: Plot title
        xlabel: X-axis label
        save_path: Path to save figure (if provided)
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(
        human_scores,
        ax=ax,
        color='blue',
        alpha=0.5,
        label='Human',
        stat='density',
        bins=20,
    )
    sns.histplot(
        ai_scores,
        ax=ax,
        color='red',
        alpha=0.5,
        label='AI',
        stat='density',
        bins=20,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    # Add statistics annotation
    human_mean = np.mean(human_scores)
    ai_mean = np.mean(ai_scores)
    ax.axvline(human_mean, color='blue', linestyle='--', alpha=0.7)
    ax.axvline(ai_mean, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_roc_curves(
    results: dict[str, EvaluationResult],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
):
    """
    Plot ROC curves for multiple detectors.

    Args:
        results: Dictionary mapping detector name to EvaluationResult
        save_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['blue', 'orange', 'green', 'red', 'purple']

    for i, (name, result) in enumerate(results.items()):
        if result.fpr is not None and result.tpr is not None:
            color = colors[i % len(colors)]
            ax.plot(
                result.fpr,
                result.tpr,
                label=f'{name} (AUC={result.auroc:.3f})',
                color=color,
            )

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_experiment_summary(
    dire_results: list[dict],
    perplexity_results: list[dict],
    mask_ratios: list[float],
    save_path: Optional[str] = None,
):
    """
    Create comprehensive experiment summary plot.

    Args:
        dire_results: DIRE computation results
        perplexity_results: Perplexity computation results
        mask_ratios: List of mask ratios used
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Extract data
    human_dire = {r: [] for r in mask_ratios}
    ai_dire = {r: [] for r in mask_ratios}

    for result in dire_results:
        label = result.get("label", 0)
        for ratio in mask_ratios:
            error_key = f"error_{ratio}"
            if error_key in result:
                if label == 0:
                    human_dire[ratio].append(result[error_key])
                else:
                    ai_dire[ratio].append(result[error_key])

    human_ppl = [r["perplexity"] for r in perplexity_results if r.get("label", 0) == 0]
    ai_ppl = [r["perplexity"] for r in perplexity_results if r.get("label", 0) == 1]

    # Plot 1: Best DIRE distribution
    best_ratio = mask_ratios[len(mask_ratios) // 2]  # Use middle ratio
    ax1 = axes[0, 0]

    if human_dire[best_ratio] and ai_dire[best_ratio]:
        sns.histplot(human_dire[best_ratio], ax=ax1, color='blue', alpha=0.5, label='Human', stat='density', bins=20)
        sns.histplot(ai_dire[best_ratio], ax=ax1, color='red', alpha=0.5, label='AI', stat='density', bins=20)
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Text-DIRE Distribution (mask ratio={best_ratio})')
        ax1.legend()

    # Plot 2: AUROC by mask ratio
    ax2 = axes[0, 1]

    aurocs = []
    for ratio in mask_ratios:
        if human_dire[ratio] and ai_dire[ratio]:
            all_scores = human_dire[ratio] + ai_dire[ratio]
            all_labels = [0] * len(human_dire[ratio]) + [1] * len(ai_dire[ratio])
            auroc = compute_auroc(all_labels, all_scores)
            aurocs.append(auroc)
        else:
            aurocs.append(0.5)

    bars = ax2.bar([str(r) for r in mask_ratios], aurocs, color='steelblue')
    ax2.set_xlabel('Mask Ratio')
    ax2.set_ylabel('AUROC')
    ax2.set_title('AUROC by Mask Ratio')
    ax2.set_ylim(0.4, 1.0)

    for bar, auroc in zip(bars, aurocs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auroc:.3f}', ha='center', va='bottom')

    # Plot 3: Perplexity distribution
    ax3 = axes[1, 0]

    if human_ppl and ai_ppl:
        # Cap for visualization
        human_ppl_capped = [min(p, 500) for p in human_ppl]
        ai_ppl_capped = [min(p, 500) for p in ai_ppl]

        sns.histplot(human_ppl_capped, ax=ax3, color='blue', alpha=0.5, label='Human', stat='density', bins=20)
        sns.histplot(ai_ppl_capped, ax=ax3, color='red', alpha=0.5, label='AI', stat='density', bins=20)
        ax3.set_xlabel('Perplexity (capped at 500)')
        ax3.set_ylabel('Density')
        ax3.set_title('Perplexity Distribution')
        ax3.legend()

    # Plot 4: Method comparison
    ax4 = axes[1, 1]

    method_names = []
    method_aurocs = []

    for ratio in mask_ratios:
        if human_dire[ratio] and ai_dire[ratio]:
            all_scores = human_dire[ratio] + ai_dire[ratio]
            all_labels = [0] * len(human_dire[ratio]) + [1] * len(ai_dire[ratio])
            auroc = compute_auroc(all_labels, all_scores)
            method_names.append(f'DIRE-{ratio}')
            method_aurocs.append(auroc)

    if human_ppl and ai_ppl:
        all_ppl = human_ppl + ai_ppl
        all_labels = [0] * len(human_ppl) + [1] * len(ai_ppl)
        ppl_auroc = compute_auroc(all_labels, [-p for p in all_ppl])
        method_names.append('Perplexity')
        method_aurocs.append(ppl_auroc)

    if method_names:
        colors = ['steelblue'] * len(mask_ratios) + ['orange']
        ax4.barh(method_names, method_aurocs, color=colors[:len(method_names)])
        ax4.set_xlabel('AUROC')
        ax4.set_title('Method Comparison')
        ax4.set_xlim(0.4, 1.0)
        ax4.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

        for i, auroc in enumerate(method_aurocs):
            ax4.text(auroc + 0.01, i, f'{auroc:.3f}', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def statistical_test(
    human_scores: list[float],
    ai_scores: list[float],
) -> dict:
    """
    Perform statistical tests comparing human and AI score distributions.

    Returns:
        Dictionary with test statistics and p-values
    """
    from scipy import stats

    results = {}

    # T-test
    t_stat, t_pvalue = stats.ttest_ind(human_scores, ai_scores)
    results["t_test"] = {
        "statistic": t_stat,
        "p_value": t_pvalue,
    }

    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(
        human_scores,
        ai_scores,
        alternative='two-sided',
    )
    results["mann_whitney"] = {
        "statistic": u_stat,
        "p_value": u_pvalue,
    }

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(human_scores) - 1) * np.std(human_scores, ddof=1)**2 +
         (len(ai_scores) - 1) * np.std(ai_scores, ddof=1)**2) /
        (len(human_scores) + len(ai_scores) - 2)
    )
    cohens_d = (np.mean(human_scores) - np.mean(ai_scores)) / pooled_std

    results["effect_size"] = {
        "cohens_d": cohens_d,
        "interpretation": interpret_cohens_d(cohens_d),
    }

    return results


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_auroc_ci(
    labels: list[int],
    scores: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Compute bootstrap confidence interval for AUROC.

    Args:
        labels: True labels
        scores: Prediction scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed

    Returns:
        Dictionary with point estimate and CI bounds
    """
    from sklearn.metrics import roc_auc_score

    np.random.seed(random_state)

    labels = np.array(labels)
    scores = np.array(scores)
    n_samples = len(labels)

    # Point estimate
    point_auroc = roc_auc_score(labels, scores)
    if point_auroc < 0.5:
        scores = -scores
        point_auroc = 1 - point_auroc

    # Bootstrap
    bootstrap_aurocs = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_labels = labels[indices]
        boot_scores = scores[indices]

        # Skip if only one class
        if len(np.unique(boot_labels)) < 2:
            continue

        try:
            auroc = roc_auc_score(boot_labels, boot_scores)
            if auroc < 0.5:
                auroc = 1 - auroc
            bootstrap_aurocs.append(auroc)
        except Exception:
            continue

    bootstrap_aurocs = np.array(bootstrap_aurocs)

    # Compute CI
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_aurocs, alpha / 2 * 100)
    upper = np.percentile(bootstrap_aurocs, (1 - alpha / 2) * 100)

    return {
        "auroc": point_auroc,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "std": np.std(bootstrap_aurocs),
        "n_bootstrap": len(bootstrap_aurocs),
    }


def bootstrap_metric_ci(
    labels: list[int],
    predictions: list[int],
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Compute bootstrap CI for any metric function.

    Args:
        labels: True labels
        predictions: Predicted labels
        metric_fn: Function(labels, predictions) -> float
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed

    Returns:
        Dictionary with point estimate and CI bounds
    """
    np.random.seed(random_state)

    labels = np.array(labels)
    predictions = np.array(predictions)
    n_samples = len(labels)

    # Point estimate
    point_estimate = metric_fn(labels, predictions)

    # Bootstrap
    bootstrap_values = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_labels = labels[indices]
        boot_preds = predictions[indices]

        try:
            value = metric_fn(boot_labels, boot_preds)
            bootstrap_values.append(value)
        except Exception:
            continue

    bootstrap_values = np.array(bootstrap_values)

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_values, alpha / 2 * 100)
    upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)

    return {
        "estimate": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "std": np.std(bootstrap_values),
    }


# =============================================================================
# Statistical Tests for Method Comparison
# =============================================================================

def mcnemar_test(
    labels: list[int],
    predictions_a: list[int],
    predictions_b: list[int],
) -> dict:
    """
    McNemar's test for comparing two classifiers.

    Tests whether the disagreement between two classifiers is significant.

    Args:
        labels: True labels
        predictions_a: Predictions from classifier A
        predictions_b: Predictions from classifier B

    Returns:
        Dictionary with test statistic and p-value
    """
    from scipy import stats

    labels = np.array(labels)
    predictions_a = np.array(predictions_a)
    predictions_b = np.array(predictions_b)

    # Build contingency table
    # b = A wrong, B right
    # c = A right, B wrong
    correct_a = predictions_a == labels
    correct_b = predictions_b == labels

    b = np.sum(~correct_a & correct_b)  # A wrong, B right
    c = np.sum(correct_a & ~correct_b)  # A right, B wrong

    # McNemar test statistic (with continuity correction)
    if b + c == 0:
        return {
            "statistic": 0,
            "p_value": 1.0,
            "b": b,
            "c": c,
            "significant": False,
        }

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "b": int(b),
        "c": int(c),
        "significant": p_value < 0.05,
    }


def compare_aurocs_delong(
    labels: list[int],
    scores_a: list[float],
    scores_b: list[float],
) -> dict:
    """
    DeLong test for comparing two AUROC values.

    Args:
        labels: True labels
        scores_a: Scores from method A
        scores_b: Scores from method B

    Returns:
        Dictionary with test results
    """
    from scipy import stats

    labels = np.array(labels)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    # Compute AUROCs
    from sklearn.metrics import roc_auc_score

    auroc_a = roc_auc_score(labels, scores_a)
    auroc_b = roc_auc_score(labels, scores_b)

    # Handle inverted scores
    if auroc_a < 0.5:
        scores_a = -scores_a
        auroc_a = 1 - auroc_a
    if auroc_b < 0.5:
        scores_b = -scores_b
        auroc_b = 1 - auroc_b

    # Simplified DeLong variance estimation via bootstrap
    n_bootstrap = 1000
    auroc_diffs = []

    n = len(labels)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue

        try:
            a_auroc = roc_auc_score(labels[idx], scores_a[idx])
            b_auroc = roc_auc_score(labels[idx], scores_b[idx])
            auroc_diffs.append(a_auroc - b_auroc)
        except Exception:
            continue

    if not auroc_diffs:
        return {
            "auroc_a": auroc_a,
            "auroc_b": auroc_b,
            "difference": auroc_a - auroc_b,
            "p_value": 1.0,
            "significant": False,
        }

    auroc_diffs = np.array(auroc_diffs)
    diff_mean = np.mean(auroc_diffs)
    diff_std = np.std(auroc_diffs)

    # Z-test
    z_score = diff_mean / diff_std if diff_std > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return {
        "auroc_a": auroc_a,
        "auroc_b": auroc_b,
        "difference": auroc_a - auroc_b,
        "z_score": z_score,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


# =============================================================================
# Multiple Comparison Correction
# =============================================================================

def bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> dict:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Significance level

    Returns:
        Dictionary with corrected threshold and significant tests
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    significant = [p < corrected_alpha for p in p_values]

    return {
        "original_alpha": alpha,
        "corrected_alpha": corrected_alpha,
        "n_tests": n_tests,
        "p_values": p_values,
        "significant": significant,
        "n_significant": sum(significant),
    }


def benjamini_hochberg(
    p_values: list[float],
    alpha: float = 0.05,
) -> dict:
    """
    Benjamini-Hochberg procedure for FDR control.

    Args:
        p_values: List of p-values
        alpha: False discovery rate

    Returns:
        Dictionary with adjusted p-values and significant tests
    """
    n_tests = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]

    # BH adjusted p-values
    adjusted = np.zeros(n_tests)
    for i in range(n_tests - 1, -1, -1):
        if i == n_tests - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n_tests / (i + 1))

    # Map back to original order
    adjusted_original = np.zeros(n_tests)
    adjusted_original[sorted_idx] = adjusted

    significant = adjusted_original < alpha

    return {
        "alpha": alpha,
        "n_tests": n_tests,
        "p_values": p_values,
        "adjusted_p_values": adjusted_original.tolist(),
        "significant": significant.tolist(),
        "n_significant": sum(significant),
    }


def comprehensive_evaluation(
    labels: list[int],
    scores: list[float],
    method_name: str = "Method",
    n_bootstrap: int = 1000,
) -> dict:
    """
    Comprehensive evaluation with confidence intervals.

    Args:
        labels: True labels
        scores: Prediction scores
        method_name: Name of the method
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with all evaluation metrics and CIs
    """
    labels = np.array(labels)
    scores = np.array(scores)

    # Basic evaluation
    basic_result = evaluate_detector(labels, scores)

    # Bootstrap CI for AUROC
    auroc_ci = bootstrap_auroc_ci(labels, scores, n_bootstrap)

    oriented_scores, _ = _oriented_scores(labels, scores)
    predictions = (oriented_scores >= basic_result.threshold).astype(int)

    # Bootstrap CI for accuracy
    from sklearn.metrics import accuracy_score
    accuracy_ci = bootstrap_metric_ci(
        labels, predictions,
        lambda label_values, pred_values: accuracy_score(label_values, pred_values),
        n_bootstrap,
    )

    # Statistical tests
    stat_result = statistical_test(
        scores[labels == 0].tolist(),
        scores[labels == 1].tolist(),
    )

    return {
        "method": method_name,
        "auroc": basic_result.auroc,
        "auroc_ci_lower": auroc_ci["ci_lower"],
        "auroc_ci_upper": auroc_ci["ci_upper"],
        "accuracy": basic_result.accuracy,
        "accuracy_ci_lower": accuracy_ci["ci_lower"],
        "accuracy_ci_upper": accuracy_ci["ci_upper"],
        "f1": basic_result.f1,
        "precision": basic_result.precision,
        "recall": basic_result.recall,
        "threshold": basic_result.threshold,
        "effect_size": stat_result["effect_size"]["cohens_d"],
        "mann_whitney_p": stat_result["mann_whitney"]["p_value"],
        "n_samples": len(labels),
    }


def format_results_summary(
    results: dict,
    mask_ratios: list[float] = None,
) -> str:
    """
    Format evaluation results as a readable string.

    Args:
        results: Dictionary of results (from evaluate_and_plot)
        mask_ratios: List of mask ratios used

    Returns:
        Formatted string summary
    """
    lines = [
        "=" * 60,
        "TEXT-DIRE EXPERIMENT RESULTS",
        "=" * 60,
        "",
    ]

    if mask_ratios is None:
        mask_ratios = [0.3, 0.5, 0.7]

    for ratio in mask_ratios:
        key = f"mask_{ratio}"
        if key in results:
            r = results[key]
            lines.extend([
                f"DIRE (mask ratio = {ratio}):",
                f"  Human texts - Mean error: {r['human_mean']:.4f} (SD: {r['human_std']:.4f})",
                f"  AI texts    - Mean error: {r['ai_mean']:.4f} (SD: {r['ai_std']:.4f})",
                f"  AUROC: {r['auroc']:.4f}",
                "",
            ])

    if "perplexity_baseline" in results:
        r = results["perplexity_baseline"]
        lines.extend([
            "Perplexity Baseline (GPT-2):",
            f"  Human texts - Mean: {r['human_mean']:.2f} (SD: {r['human_std']:.2f})",
            f"  AI texts    - Mean: {r['ai_mean']:.2f} (SD: {r['ai_std']:.2f})",
            f"  AUROC: {r['auroc']:.4f}",
            "",
        ])

    # Best method
    best_auroc = 0
    best_method = None

    for key, r in results.items():
        if r.get("auroc", 0) > best_auroc:
            best_auroc = r["auroc"]
            best_method = key

    if best_method:
        lines.extend([
            "-" * 60,
            f"Best method: {best_method} (AUROC = {best_auroc:.4f})",
            "-" * 60,
        ])

    return "\n".join(lines)
