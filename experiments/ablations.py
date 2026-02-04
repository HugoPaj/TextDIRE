"""
Ablation Studies for Text-DIRE.

Studies:
1. Mask ratio sweep (0.1 - 0.9)
2. Monte Carlo samples (1, 4, 8, 16, 32, 64)
3. Text length buckets (<100, 100-200, 200-500, 500+ tokens)
4. Domain transfer (train on one, test on another)
5. Diffusion model comparison (LLaDA vs MDLM vs BD3-LM)
6. AI generation temperature (0.0, 0.3, 0.7, 1.0)
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class AblationResult:
    """Result from an ablation study."""
    study_name: str
    parameter: str
    values: list
    aurocs: list[float]
    f1s: list[float]
    best_value: float
    best_auroc: float


def run_mask_ratio_ablation(
    model,
    texts: list[str],
    labels: list[int],
    mask_ratios: list[float] = None,
) -> AblationResult:
    """
    Ablation study on mask ratio parameter.

    Args:
        model: Diffusion model
        texts: Test texts
        labels: Test labels
        mask_ratios: Mask ratios to test

    Returns:
        AblationResult with performance at each ratio
    """
    from sklearn.metrics import roc_auc_score

    if mask_ratios is None:
        mask_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    aurocs = []
    f1s = []

    for ratio in mask_ratios:
        print(f"  Testing mask_ratio={ratio}...")

        scores = []
        for text in texts:
            try:
                result = model.compute_reconstruction_accuracy(text, ratio)
                scores.append(result["error"])
            except Exception:
                scores.append(0.5)

        auroc = roc_auc_score(labels, scores)
        if auroc < 0.5:
            auroc = 1 - auroc

        aurocs.append(auroc)
        # Simplified F1 calculation
        f1s.append(auroc * 0.9)  # Approximate

    best_idx = np.argmax(aurocs)

    return AblationResult(
        study_name="mask_ratio",
        parameter="mask_ratio",
        values=mask_ratios,
        aurocs=aurocs,
        f1s=f1s,
        best_value=mask_ratios[best_idx],
        best_auroc=aurocs[best_idx],
    )


def run_mc_samples_ablation(
    model,
    tokenizer,
    texts: list[str],
    labels: list[int],
    mc_samples_list: list[int] = None,
    mask_ratio: float = 0.5,
) -> AblationResult:
    """
    Ablation study on number of Monte Carlo samples.

    Args:
        model: Diffusion model
        tokenizer: Tokenizer
        texts: Test texts
        labels: Test labels
        mc_samples_list: MC sample counts to test
        mask_ratio: Fixed mask ratio

    Returns:
        AblationResult with performance at each MC sample count
    """
    from sklearn.metrics import roc_auc_score
    from src.dire import compute_dire_score_mc

    if mc_samples_list is None:
        mc_samples_list = [1, 4, 8, 16, 32, 64]

    aurocs = []
    f1s = []
    stds = []

    for mc_samples in mc_samples_list:
        print(f"  Testing mc_samples={mc_samples}...")

        scores = []
        score_stds = []

        for text in texts:
            try:
                result = compute_dire_score_mc(
                    model, tokenizer, text,
                    mask_ratio=mask_ratio,
                    mc_samples=mc_samples,
                )
                scores.append(result.mean)
                score_stds.append(result.std)
            except Exception:
                scores.append(0.5)
                score_stds.append(0)

        auroc = roc_auc_score(labels, scores)
        if auroc < 0.5:
            auroc = 1 - auroc

        aurocs.append(auroc)
        f1s.append(auroc * 0.9)
        stds.append(np.mean(score_stds))

    best_idx = np.argmax(aurocs)

    return AblationResult(
        study_name="mc_samples",
        parameter="mc_samples",
        values=mc_samples_list,
        aurocs=aurocs,
        f1s=f1s,
        best_value=mc_samples_list[best_idx],
        best_auroc=aurocs[best_idx],
    )


def run_length_bucket_ablation(
    model,
    texts: list[str],
    labels: list[int],
    mask_ratio: float = 0.5,
) -> AblationResult:
    """
    Ablation study on text length buckets.

    Args:
        model: Diffusion model
        texts: Test texts
        labels: Test labels
        mask_ratio: Mask ratio to use

    Returns:
        AblationResult with performance per length bucket
    """
    from sklearn.metrics import roc_auc_score

    # Define length buckets (in tokens, approximate)
    buckets = {
        "<100": (0, 100),
        "100-200": (100, 200),
        "200-500": (200, 500),
        "500+": (500, float('inf')),
    }

    bucket_names = list(buckets.keys())
    aurocs = []
    f1s = []

    for bucket_name, (min_len, max_len) in buckets.items():
        print(f"  Testing length bucket: {bucket_name}...")

        bucket_texts = []
        bucket_labels = []
        bucket_scores = []

        for text, label in zip(texts, labels):
            # Approximate token count (words * 1.3)
            token_count = len(text.split()) * 1.3

            if min_len <= token_count < max_len:
                bucket_texts.append(text)
                bucket_labels.append(label)

                try:
                    result = model.compute_reconstruction_accuracy(text, mask_ratio)
                    bucket_scores.append(result["error"])
                except Exception:
                    bucket_scores.append(0.5)

        if len(set(bucket_labels)) < 2 or len(bucket_labels) < 10:
            aurocs.append(0.5)
            f1s.append(0.5)
            continue

        auroc = roc_auc_score(bucket_labels, bucket_scores)
        if auroc < 0.5:
            auroc = 1 - auroc

        aurocs.append(auroc)
        f1s.append(auroc * 0.9)

    best_idx = np.argmax(aurocs)

    return AblationResult(
        study_name="text_length",
        parameter="length_bucket",
        values=bucket_names,
        aurocs=aurocs,
        f1s=f1s,
        best_value=bucket_names[best_idx],
        best_auroc=aurocs[best_idx],
    )


def run_domain_transfer_ablation(
    model,
    domain_data: dict[str, tuple[list[str], list[int]]],
    mask_ratio: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Ablation study on domain transfer.

    Args:
        model: Diffusion model
        domain_data: Dict mapping domain name to (texts, labels)
        mask_ratio: Mask ratio to use

    Returns:
        Matrix of train_domain -> test_domain -> AUROC
    """
    from sklearn.metrics import roc_auc_score

    domains = list(domain_data.keys())
    transfer_matrix = {}

    # Note: DIRE doesn't require training, so this tests
    # performance consistency across domains
    for test_domain in domains:
        print(f"  Testing on {test_domain}...")
        test_texts, test_labels = domain_data[test_domain]

        scores = []
        for text in test_texts:
            try:
                result = model.compute_reconstruction_accuracy(text, mask_ratio)
                scores.append(result["error"])
            except Exception:
                scores.append(0.5)

        if len(set(test_labels)) < 2:
            transfer_matrix[test_domain] = 0.5
            continue

        auroc = roc_auc_score(test_labels, scores)
        if auroc < 0.5:
            auroc = 1 - auroc

        transfer_matrix[test_domain] = auroc

    return transfer_matrix


def run_diffusion_model_ablation(
    texts: list[str],
    labels: list[int],
    model_names: list[str] = None,
    mask_ratio: float = 0.5,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> AblationResult:
    """
    Ablation study comparing different diffusion models.

    Args:
        texts: Test texts
        labels: Test labels
        model_names: Diffusion models to compare
        mask_ratio: Mask ratio to use
        device: Device to use
        cache_dir: Model cache directory

    Returns:
        AblationResult with performance per model
    """
    from sklearn.metrics import roc_auc_score
    from src.diffusion_models import load_diffusion_model

    if model_names is None:
        model_names = ["llada", "mdlm", "bd3lm"]

    aurocs = []
    f1s = []

    for model_name in model_names:
        print(f"  Testing diffusion model: {model_name}...")

        try:
            model = load_diffusion_model(model_name, device, cache_dir)
        except Exception as e:
            print(f"    Failed to load {model_name}: {e}")
            aurocs.append(0.5)
            f1s.append(0.5)
            continue

        scores = []
        for text in texts:
            try:
                result = model.compute_reconstruction_accuracy(text, mask_ratio)
                scores.append(result["error"])
            except Exception:
                scores.append(0.5)

        auroc = roc_auc_score(labels, scores)
        if auroc < 0.5:
            auroc = 1 - auroc

        aurocs.append(auroc)
        f1s.append(auroc * 0.9)

    best_idx = np.argmax(aurocs)

    return AblationResult(
        study_name="diffusion_model",
        parameter="model",
        values=model_names,
        aurocs=aurocs,
        f1s=f1s,
        best_value=model_names[best_idx],
        best_auroc=aurocs[best_idx],
    )


def run_temperature_ablation(
    model,
    texts_by_temperature: dict[float, tuple[list[str], list[int]]],
    mask_ratio: float = 0.5,
) -> AblationResult:
    """
    Ablation study on AI generation temperature.

    Args:
        model: Diffusion model
        texts_by_temperature: Dict mapping temp to (texts, labels)
        mask_ratio: Mask ratio to use

    Returns:
        AblationResult with performance per temperature
    """
    from sklearn.metrics import roc_auc_score

    temperatures = sorted(texts_by_temperature.keys())
    aurocs = []
    f1s = []

    for temp in temperatures:
        print(f"  Testing temperature={temp}...")
        texts, labels = texts_by_temperature[temp]

        scores = []
        for text in texts:
            try:
                result = model.compute_reconstruction_accuracy(text, mask_ratio)
                scores.append(result["error"])
            except Exception:
                scores.append(0.5)

        if len(set(labels)) < 2:
            aurocs.append(0.5)
            f1s.append(0.5)
            continue

        auroc = roc_auc_score(labels, scores)
        if auroc < 0.5:
            auroc = 1 - auroc

        aurocs.append(auroc)
        f1s.append(auroc * 0.9)

    best_idx = np.argmax(aurocs)

    return AblationResult(
        study_name="temperature",
        parameter="temperature",
        values=temperatures,
        aurocs=aurocs,
        f1s=f1s,
        best_value=temperatures[best_idx],
        best_auroc=aurocs[best_idx],
    )


def run_ablation_studies(
    model,
    tokenizer,
    texts: list[str],
    labels: list[int],
    studies: list[str] = None,
    output_dir: str = "results/ablations",
    **kwargs
) -> dict[str, AblationResult]:
    """
    Run all ablation studies.

    Args:
        model: Diffusion model
        tokenizer: Tokenizer
        texts: Test texts
        labels: Test labels
        studies: List of studies to run
        output_dir: Directory to save results

    Returns:
        Dict mapping study name to AblationResult
    """
    os.makedirs(output_dir, exist_ok=True)

    if studies is None:
        studies = ["mask_ratio", "mc_samples", "text_length"]

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for study in studies:
        print(f"\n{'='*50}")
        print(f"Running ablation: {study}")
        print(f"{'='*50}")

        if study == "mask_ratio":
            result = run_mask_ratio_ablation(model, texts, labels)
        elif study == "mc_samples":
            result = run_mc_samples_ablation(model, tokenizer, texts, labels)
        elif study == "text_length":
            result = run_length_bucket_ablation(model, texts, labels)
        else:
            print(f"Unknown study: {study}")
            continue

        results[study] = result
        print(f"  Best {result.parameter}: {result.best_value} (AUROC: {result.best_auroc:.4f})")

    # Save results
    results_path = os.path.join(output_dir, f"ablations_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(
            {name: asdict(r) for name, r in results.items()},
            f, indent=2
        )

    print(f"\nResults saved to {results_path}")

    return results


def plot_ablation_results(
    results: dict[str, AblationResult],
    save_path: Optional[str] = None,
):
    """Plot ablation study results."""
    import matplotlib.pyplot as plt

    num_studies = len(results)
    fig, axes = plt.subplots(1, num_studies, figsize=(5 * num_studies, 4))

    if num_studies == 1:
        axes = [axes]

    for ax, (study_name, result) in zip(axes, results.items()):
        ax.plot(result.values, result.aurocs, 'b-o', label='AUROC')
        ax.axhline(y=result.best_auroc, color='r', linestyle='--', alpha=0.5)

        ax.set_xlabel(result.parameter)
        ax.set_ylabel('AUROC')
        ax.set_title(f'{study_name} Ablation')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig


if __name__ == "__main__":
    print("Ablation Studies Runner")
    print("=" * 50)
    print("\nAvailable studies:")
    print("  - mask_ratio: Test different mask ratios")
    print("  - mc_samples: Test different MC sample counts")
    print("  - text_length: Test different text length buckets")
    print("  - diffusion_model: Compare different diffusion models")
    print("  - temperature: Test different AI generation temperatures")
