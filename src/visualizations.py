"""
Publication-Ready Visualizations for Text-DIRE.

Generates figures suitable for arXiv paper:
1. Score distributions (human vs AI per model)
2. ROC curves for all methods
3. Ablation heatmaps
4. Token-level error heatmaps
5. t-SNE of DIRE feature vectors
"""

import numpy as np
from typing import Optional, Union
from pathlib import Path


# Publication style settings
PUBLICATION_STYLE = {
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'text.usetex': False,  # Set True if LaTeX available
    'axes.grid': True,
    'grid.alpha': 0.3,
}

# Color scheme for paper
COLORS = {
    'human': '#2E86AB',      # Blue
    'ai': '#E94F37',         # Red
    'dire': '#1E88E5',       # Deep blue
    'detectgpt': '#FFC107',  # Amber
    'binoculars': '#4CAF50', # Green
    'perplexity': '#9C27B0', # Purple
    'fast_detectgpt': '#FF5722',  # Deep orange
}


def setup_publication_style():
    """Apply publication-ready matplotlib style."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(PUBLICATION_STYLE)


def plot_score_distributions(
    human_scores: list[float],
    ai_scores: list[float],
    title: str = "DIRE Score Distribution",
    xlabel: str = "Reconstruction Error",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
    ai_source: str = "AI",
) -> 'plt.Figure':
    """
    Plot score distributions for human and AI texts.

    Publication-quality histogram with overlapping distributions.

    Args:
        human_scores: Scores for human texts
        ai_scores: Scores for AI texts
        title: Plot title
        xlabel: X-axis label
        save_path: Path to save figure
        figsize: Figure size
        ai_source: AI source name for legend

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot distributions
    sns.histplot(
        human_scores, ax=ax,
        color=COLORS['human'], alpha=0.6,
        label='Human', stat='density', bins=30,
        edgecolor='white', linewidth=0.5,
    )
    sns.histplot(
        ai_scores, ax=ax,
        color=COLORS['ai'], alpha=0.6,
        label=ai_source, stat='density', bins=30,
        edgecolor='white', linewidth=0.5,
    )

    # Add mean lines
    human_mean = np.mean(human_scores)
    ai_mean = np.mean(ai_scores)
    ax.axvline(human_mean, color=COLORS['human'], linestyle='--', alpha=0.8, linewidth=2)
    ax.axvline(ai_mean, color=COLORS['ai'], linestyle='--', alpha=0.8, linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(loc='upper right', frameon=True, fancybox=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_multi_model_distributions(
    scores_by_source: dict[str, tuple[list[float], list[float]]],
    title: str = "DIRE Scores by AI Model",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
) -> 'plt.Figure':
    """
    Plot score distributions for multiple AI sources.

    Args:
        scores_by_source: Dict mapping source name to (human_scores, ai_scores)
        title: Overall title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    setup_publication_style()

    num_sources = len(scores_by_source)
    cols = min(2, num_sources)
    rows = (num_sources + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_sources == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (source, (human_scores, ai_scores)) in enumerate(scores_by_source.items()):
        ax = axes[idx]

        sns.histplot(
            human_scores, ax=ax,
            color=COLORS['human'], alpha=0.6,
            label='Human', stat='density', bins=25,
        )
        sns.histplot(
            ai_scores, ax=ax,
            color=COLORS['ai'], alpha=0.6,
            label=source, stat='density', bins=25,
        )

        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Density')
        ax.set_title(f'{source}')
        ax.legend(loc='upper right', fontsize=9)

    # Hide unused subplots
    for idx in range(num_sources, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray, float]],
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
) -> 'plt.Figure':
    """
    Plot ROC curves for multiple methods.

    Args:
        results: Dict mapping method name to (fpr, tpr, auroc)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Color map for methods
    method_colors = {
        'DIRE': COLORS['dire'],
        'DetectGPT': COLORS['detectgpt'],
        'Fast-DetectGPT': COLORS['fast_detectgpt'],
        'Binoculars': COLORS['binoculars'],
        'Perplexity': COLORS['perplexity'],
    }

    for method, (fpr, tpr, auroc) in results.items():
        # Get color
        color = method_colors.get(method, None)
        if color is None:
            # Try prefix matching
            for key, c in method_colors.items():
                if key.lower() in method.lower():
                    color = c
                    break
            else:
                color = 'gray'

        ax.plot(fpr, tpr, label=f'{method} (AUC={auroc:.3f})',
               color=color, linewidth=2)

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right', frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_ablation_heatmap(
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "Ablation Study",
    xlabel: str = "",
    ylabel: str = "",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    cmap: str = "RdYlGn",
    vmin: float = 0.5,
    vmax: float = 1.0,
) -> 'plt.Figure':
    """
    Plot ablation study results as a heatmap.

    Args:
        data: 2D array of values
        row_labels: Labels for rows
        col_labels: Labels for columns
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
        vmin, vmax: Value range for colormap

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data, ax=ax,
        annot=True, fmt='.3f',
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        cbar_kws={'label': 'AUROC'},
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_token_error_heatmap(
    tokens: list[str],
    errors: list[bool],
    confidences: list[float],
    title: str = "Token-Level Reconstruction",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 3),
) -> 'plt.Figure':
    """
    Plot token-level error visualization for a single text.

    Args:
        tokens: List of tokens
        errors: List of booleans (True = error)
        confidences: Model confidence for each token
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    setup_publication_style()

    # Limit tokens for visualization
    max_tokens = 50
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        errors = errors[:max_tokens]
        confidences = confidences[:max_tokens]

    fig, ax = plt.subplots(figsize=figsize)

    # Create color array
    colors = []
    for err, conf in zip(errors, confidences):
        if err:
            # Red for errors, darker = lower confidence
            colors.append((1, 0.2, 0.2, 0.5 + 0.5 * (1 - conf)))
        else:
            # Green for correct, darker = higher confidence
            colors.append((0.2, 0.8, 0.2, 0.3 + 0.7 * conf))

    # Plot tokens as colored boxes
    for i, (token, color) in enumerate(zip(tokens, colors)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5))
        ax.text(i + 0.5, 0.5, token[:8], ha='center', va='center', fontsize=8, rotation=45)

    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.2, 0.8, 0.2, 0.8), label='Correct'),
        Patch(facecolor=(1, 0.2, 0.2, 0.8), label='Error'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_tsne_features(
    features: np.ndarray,
    labels: list[int],
    sources: Optional[list[str]] = None,
    title: str = "t-SNE of DIRE Features",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    perplexity: int = 30,
) -> 'plt.Figure':
    """
    Plot t-SNE visualization of DIRE feature vectors.

    Args:
        features: Feature vectors (N x D)
        labels: Labels (0=human, 1=AI)
        sources: Optional source names for color coding
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        perplexity: t-SNE perplexity parameter

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    setup_publication_style()

    # Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=figsize)

    if sources:
        # Color by source
        unique_sources = list(set(sources))
        cmap = plt.cm.get_cmap('tab10', len(unique_sources))

        for i, source in enumerate(unique_sources):
            mask = [s == source for s in sources]
            ax.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                c=[cmap(i)], label=source, alpha=0.6, s=30,
            )
    else:
        # Color by label
        human_mask = [l == 0 for l in labels]
        ai_mask = [l == 1 for l in labels]

        ax.scatter(
            embeddings[human_mask, 0], embeddings[human_mask, 1],
            c=COLORS['human'], label='Human', alpha=0.6, s=30,
        )
        ax.scatter(
            embeddings[ai_mask, 0], embeddings[ai_mask, 1],
            c=COLORS['ai'], label='AI', alpha=0.6, s=30,
        )

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    ax.legend(loc='best', frameon=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_method_comparison_bar(
    results: dict[str, dict[str, float]],
    metric: str = "auroc",
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> 'plt.Figure':
    """
    Plot bar chart comparing methods across AI sources.

    Args:
        results: Dict[method][ai_source] = metric_value
        metric: Metric to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    setup_publication_style()

    methods = list(results.keys())
    if not methods:
        return None

    ai_sources = list(results[methods[0]].keys())
    x = np.arange(len(ai_sources))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        values = [results[method].get(src, 0) for src in ai_sources]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('AI Source')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(ai_sources)
    ax.legend(loc='lower right', ncol=2)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_paper_figures(
    output_dir: str = "paper/figures",
    **kwargs
):
    """
    Create all figures for the paper.

    This is a convenience function that creates placeholder figures.
    In practice, you would call individual plotting functions with
    actual experimental data.
    """
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Creating paper figures...")
    print(f"Output directory: {output_dir}")

    # Create placeholder figures
    figures = [
        "fig1_score_distributions.pdf",
        "fig2_roc_curves.pdf",
        "fig3_ablation_heatmap.pdf",
        "fig4_method_comparison.pdf",
        "fig5_tsne_features.pdf",
        "fig6_token_errors.pdf",
    ]

    for fig_name in figures:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Placeholder: {fig_name}",
               ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        save_path = Path(output_dir) / fig_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Created: {save_path}")

    print("Done!")


if __name__ == "__main__":
    print("Publication Visualizations Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - plot_score_distributions")
    print("  - plot_multi_model_distributions")
    print("  - plot_roc_curves")
    print("  - plot_ablation_heatmap")
    print("  - plot_token_error_heatmap")
    print("  - plot_tsne_features")
    print("  - plot_method_comparison_bar")
    print("  - create_paper_figures")
