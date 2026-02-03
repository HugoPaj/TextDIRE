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
    from sklearn.metrics import roc_auc_score

    auroc = roc_auc_score(labels, scores)

    # If AUROC < 0.5, the score direction is inverted
    if auroc < 0.5:
        auroc = 1 - auroc

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
        roc_auc_score,
        roc_curve,
        precision_recall_fscore_support,
        accuracy_score,
    )

    labels = np.array(labels)
    scores = np.array(scores)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)
    auroc = roc_auc_score(labels, scores)

    # Handle inverted scores
    if auroc < 0.5:
        fpr, tpr, thresholds = roc_curve(labels, -scores, pos_label=pos_label)
        auroc = 1 - auroc
        scores = -scores

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
