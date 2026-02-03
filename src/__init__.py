"""
Text-DIRE: Diffusion Reconstruction Error for AI Text Detection

This package implements the core functionality for computing Text-DIRE scores
using mask-based diffusion models like LLaDA.
"""

from .dire import TextDIRE, compute_dire_score, mask_tokens
from .data import load_human_texts, load_ai_texts, load_datasets
from .baselines import compute_perplexity, PerplexityDetector
from .evaluate import compute_auroc, plot_distributions, evaluate_detector

__all__ = [
    "TextDIRE",
    "compute_dire_score",
    "mask_tokens",
    "load_human_texts",
    "load_ai_texts",
    "load_datasets",
    "compute_perplexity",
    "PerplexityDetector",
    "compute_auroc",
    "plot_distributions",
    "evaluate_detector",
]
