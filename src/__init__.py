"""
Text-DIRE: Diffusion Reconstruction Error for AI Text Detection

This package implements the core functionality for computing Text-DIRE scores
using mask-based diffusion models like LLaDA.

Expanded for arXiv publication with:
- Multi-source data loading (WikiText, Reddit, CNN, arXiv)
- RAID/MAGE benchmark integration
- Monte Carlo DIRE estimation
- Multi-metric scoring
- Ensemble scoring
- Multiple diffusion model support
- Full baseline implementations (DetectGPT, Fast-DetectGPT, Binoculars)
- Token-level analysis
- Publication-ready visualizations
- Statistical analysis with bootstrap CIs
"""

# Core DIRE functionality
from .dire import (
    TextDIRE,
    compute_dire_score,
    compute_dire_score_mc,
    ensemble_dire_score,
    mask_tokens,
    DIREResult,
    MCDIREResult,
    EnsembleDIREResult,
)

# Data loading
from .data import (
    load_human_texts,
    load_ai_texts,
    load_datasets,
    load_raid_benchmark,
    load_mage_benchmark,
    load_multi_source_human_texts,
    create_combined_dataset,
    TextDataset,
)

# Baseline methods
from .baselines import (
    compute_perplexity,
    PerplexityDetector,
    DetectGPTBaseline,
    FastDetectGPT,
    Binoculars,
)

# Evaluation
from .evaluate import (
    compute_auroc,
    evaluate_detector,
    plot_distributions,
    bootstrap_auroc_ci,
    mcnemar_test,
    comprehensive_evaluation,
)

# Diffusion models
from .diffusion_models import (
    load_diffusion_model,
    get_available_models,
    MultiModelDIRE,
)

# AI text generation
from .ai_text_generator import (
    AITextGenerator,
    get_available_models as get_ai_models,
)

# Analysis
from .analysis import (
    analyze_reconstruction,
    aggregate_analyses,
    compare_human_vs_ai_patterns,
)

# Visualizations
from .visualizations import (
    plot_score_distributions,
    plot_roc_curves,
    plot_ablation_heatmap,
    plot_tsne_features,
    create_paper_figures,
)

__version__ = "0.2.0"

__all__ = [
    # DIRE
    "TextDIRE",
    "compute_dire_score",
    "compute_dire_score_mc",
    "ensemble_dire_score",
    "mask_tokens",
    "DIREResult",
    "MCDIREResult",
    "EnsembleDIREResult",

    # Data
    "load_human_texts",
    "load_ai_texts",
    "load_datasets",
    "load_raid_benchmark",
    "load_mage_benchmark",
    "load_multi_source_human_texts",
    "create_combined_dataset",
    "TextDataset",

    # Baselines
    "compute_perplexity",
    "PerplexityDetector",
    "DetectGPTBaseline",
    "FastDetectGPT",
    "Binoculars",

    # Evaluation
    "compute_auroc",
    "evaluate_detector",
    "plot_distributions",
    "bootstrap_auroc_ci",
    "mcnemar_test",
    "comprehensive_evaluation",

    # Diffusion models
    "load_diffusion_model",
    "get_available_models",
    "MultiModelDIRE",

    # AI generation
    "AITextGenerator",

    # Analysis
    "analyze_reconstruction",
    "aggregate_analyses",
    "compare_human_vs_ai_patterns",

    # Visualizations
    "plot_score_distributions",
    "plot_roc_curves",
    "plot_ablation_heatmap",
    "plot_tsne_features",
    "create_paper_figures",
]
