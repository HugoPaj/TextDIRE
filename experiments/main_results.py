"""
Main Results Experiment for Text-DIRE Paper.

Runs all detection methods on all AI sources and generates the main
comparison table for the paper.

Methods:
- DIRE-LLaDA (mask ratios 0.3, 0.5, 0.7)
- DIRE-MDLM (mask ratios 0.3, 0.5, 0.7)
- DetectGPT
- Fast-DetectGPT
- Binoculars
- Perplexity

AI Sources:
- GPT-5.2
- GPT-5-mini
- Claude Sonnet 4.5
- Claude Haiku 4.5

Metrics:
- AUROC
- F1@optimal
- Accuracy@optimal
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class MethodResult:
    """Results for a single method on a single AI source."""
    method: str
    ai_source: str
    auroc: float
    f1: float
    accuracy: float
    precision: float
    recall: float
    threshold: float
    num_samples: int


@dataclass
class ExperimentResults:
    """Complete experiment results."""
    timestamp: str
    methods: list[str]
    ai_sources: list[str]
    results: list[MethodResult]
    human_sources: list[str]
    config: dict


def evaluate_method(
    labels: list[int],
    scores: list[float],
    method_name: str,
    ai_source: str,
) -> MethodResult:
    """
    Evaluate a detection method's scores.

    Args:
        labels: True labels (0=human, 1=AI)
        scores: Detection scores
        method_name: Name of the method
        ai_source: AI source being detected

    Returns:
        MethodResult with all metrics
    """
    from sklearn.metrics import (
        roc_auc_score, roc_curve,
        precision_recall_fscore_support,
        accuracy_score,
    )

    labels = np.array(labels)
    scores = np.array(scores)

    # Compute AUROC
    auroc = roc_auc_score(labels, scores)

    # Handle inverted scores
    if auroc < 0.5:
        scores = -scores
        auroc = 1 - auroc

    # Find optimal threshold using Youden's J
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]

    # Compute metrics at optimal threshold
    predictions = (scores >= optimal_threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions,
        pos_label=1,
        average='binary',
        zero_division=0,
    )

    return MethodResult(
        method=method_name,
        ai_source=ai_source,
        auroc=auroc,
        f1=f1,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        threshold=optimal_threshold,
        num_samples=len(labels),
    )


def run_main_experiment(
    human_texts: list[str],
    ai_texts_by_source: dict[str, list[str]],
    diffusion_models: list[str] = None,
    mask_ratios: list[float] = None,
    run_detectgpt: bool = True,
    run_fast_detectgpt: bool = True,
    run_binoculars: bool = True,
    run_perplexity: bool = True,
    detectgpt_perturbations: int = 100,
    output_dir: str = "results",
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> ExperimentResults:
    """
    Run the main experiment comparing all methods across all AI sources.

    Args:
        human_texts: List of human-written texts
        ai_texts_by_source: Dict mapping AI source name to list of texts
        diffusion_models: Diffusion models to test (default: ["llada"])
        mask_ratios: Mask ratios for DIRE (default: [0.3, 0.5, 0.7])
        run_detectgpt: Whether to run DetectGPT
        run_fast_detectgpt: Whether to run Fast-DetectGPT
        run_binoculars: Whether to run Binoculars
        run_perplexity: Whether to run perplexity baseline
        detectgpt_perturbations: Number of perturbations for DetectGPT
        output_dir: Directory to save results
        device: Device to use
        cache_dir: Model cache directory

    Returns:
        ExperimentResults with all method comparisons
    """
    from src.diffusion_models import load_diffusion_model
    from src.baselines import (
        PerplexityDetector,
        DetectGPTBaseline,
        FastDetectGPT,
        Binoculars,
    )

    if diffusion_models is None:
        diffusion_models = ["llada"]

    if mask_ratios is None:
        mask_ratios = [0.3, 0.5, 0.7]

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    timestamp = datetime.now().isoformat()

    # Prepare labels
    human_labels = [0] * len(human_texts)

    # Load diffusion models for DIRE
    dire_models = {}
    for model_name in diffusion_models:
        print(f"\nLoading diffusion model: {model_name}")
        try:
            dire_models[model_name] = load_diffusion_model(
                model_name, device, cache_dir
            )
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")

    # Load baseline detectors
    detectors = {}

    if run_perplexity:
        print("\nLoading perplexity detector...")
        detectors["perplexity"] = PerplexityDetector(
            model_name="gpt2", device=device, cache_dir=cache_dir
        )

    if run_fast_detectgpt:
        print("\nLoading Fast-DetectGPT...")
        detectors["fast_detectgpt"] = FastDetectGPT(
            model_name="gpt2-medium", device=device, cache_dir=cache_dir
        )

    if run_binoculars:
        print("\nLoading Binoculars...")
        detectors["binoculars"] = Binoculars(
            observer_model="gpt2",
            performer_model="gpt2-medium",
            device=device,
            cache_dir=cache_dir,
        )

    if run_detectgpt:
        print("\nLoading DetectGPT (this may take a while)...")
        detectors["detectgpt"] = DetectGPTBaseline(
            scoring_model="gpt2-medium",
            perturbation_model="t5-large",
            device=device,
            cache_dir=cache_dir,
        )

    # Run experiments for each AI source
    for ai_source, ai_texts in ai_texts_by_source.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {ai_source}")
        print(f"{'='*60}")

        ai_labels = [1] * len(ai_texts)
        all_texts = human_texts + ai_texts
        all_labels = human_labels + ai_labels

        # DIRE with each diffusion model
        for model_name, dire_model in dire_models.items():
            for mask_ratio in mask_ratios:
                method_name = f"DIRE-{model_name}-{mask_ratio}"
                print(f"\nRunning {method_name}...")

                scores = []
                for text in all_texts:
                    try:
                        result = dire_model.compute_reconstruction_accuracy(
                            text, mask_ratio
                        )
                        scores.append(result["error"])
                    except Exception as e:
                        scores.append(0.5)  # Neutral score on error

                eval_result = evaluate_method(
                    all_labels, scores, method_name, ai_source
                )
                all_results.append(eval_result)
                print(f"  AUROC: {eval_result.auroc:.4f}, F1: {eval_result.f1:.4f}")

        # Perplexity baseline
        if "perplexity" in detectors:
            print(f"\nRunning Perplexity...")
            ppl_results = detectors["perplexity"].compute_perplexities(
                all_texts, progress_bar=True
            )
            # Lower perplexity = more AI-like, so negate for scoring
            scores = [-r.perplexity for r in ppl_results]

            # Pad if some failed
            while len(scores) < len(all_labels):
                scores.append(-50)  # Neutral

            eval_result = evaluate_method(
                all_labels[:len(scores)], scores, "Perplexity", ai_source
            )
            all_results.append(eval_result)
            print(f"  AUROC: {eval_result.auroc:.4f}, F1: {eval_result.f1:.4f}")

        # Fast-DetectGPT
        if "fast_detectgpt" in detectors:
            print(f"\nRunning Fast-DetectGPT...")
            fdg_results = detectors["fast_detectgpt"].detect(
                all_texts, progress_bar=True
            )
            scores = [r.score for r in fdg_results]

            while len(scores) < len(all_labels):
                scores.append(0)

            eval_result = evaluate_method(
                all_labels[:len(scores)], scores, "Fast-DetectGPT", ai_source
            )
            all_results.append(eval_result)
            print(f"  AUROC: {eval_result.auroc:.4f}, F1: {eval_result.f1:.4f}")

        # Binoculars
        if "binoculars" in detectors:
            print(f"\nRunning Binoculars...")
            bino_results = detectors["binoculars"].detect(
                all_texts, progress_bar=True
            )
            # Lower score = more AI-like
            scores = [-r.score for r in bino_results]

            while len(scores) < len(all_labels):
                scores.append(0)

            eval_result = evaluate_method(
                all_labels[:len(scores)], scores, "Binoculars", ai_source
            )
            all_results.append(eval_result)
            print(f"  AUROC: {eval_result.auroc:.4f}, F1: {eval_result.f1:.4f}")

        # DetectGPT (slow, run last)
        if "detectgpt" in detectors:
            print(f"\nRunning DetectGPT ({detectgpt_perturbations} perturbations)...")
            dgpt_results = detectors["detectgpt"].detect(
                all_texts,
                num_perturbations=detectgpt_perturbations,
                progress_bar=True,
            )
            scores = [r.curvature for r in dgpt_results]

            while len(scores) < len(all_labels):
                scores.append(0)

            eval_result = evaluate_method(
                all_labels[:len(scores)], scores, "DetectGPT", ai_source
            )
            all_results.append(eval_result)
            print(f"  AUROC: {eval_result.auroc:.4f}, F1: {eval_result.f1:.4f}")

    # Compile results
    experiment_results = ExperimentResults(
        timestamp=timestamp,
        methods=list(set(r.method for r in all_results)),
        ai_sources=list(ai_texts_by_source.keys()),
        results=all_results,
        human_sources=["wikitext"],  # Update based on actual sources
        config={
            "diffusion_models": diffusion_models,
            "mask_ratios": mask_ratios,
            "detectgpt_perturbations": detectgpt_perturbations,
        }
    )

    # Save results
    results_path = os.path.join(output_dir, f"main_results_{timestamp.replace(':', '-')}.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": experiment_results.timestamp,
            "methods": experiment_results.methods,
            "ai_sources": experiment_results.ai_sources,
            "human_sources": experiment_results.human_sources,
            "config": experiment_results.config,
            "results": [asdict(r) for r in experiment_results.results],
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Print summary table
    print_results_table(experiment_results)

    return experiment_results


def print_results_table(results: ExperimentResults):
    """Print a formatted results table."""
    print("\n" + "=" * 80)
    print("MAIN RESULTS TABLE")
    print("=" * 80)

    # Group by method
    methods = sorted(set(r.method for r in results.results))
    ai_sources = results.ai_sources

    # Header
    header = f"{'Method':<25}" + "".join(f"{src:<15}" for src in ai_sources) + f"{'Avg':<10}"
    print(header)
    print("-" * len(header))

    # Rows
    for method in methods:
        row = f"{method:<25}"
        aurocs = []

        for ai_source in ai_sources:
            matching = [r for r in results.results
                       if r.method == method and r.ai_source == ai_source]
            if matching:
                auroc = matching[0].auroc
                aurocs.append(auroc)
                row += f"{auroc:.3f}{'':>10}"
            else:
                row += f"{'N/A':<15}"

        if aurocs:
            avg = np.mean(aurocs)
            row += f"{avg:.3f}"

        print(row)

    print("=" * 80)


def generate_latex_table(results: ExperimentResults) -> str:
    """Generate LaTeX table for paper."""
    methods = sorted(set(r.method for r in results.results))
    ai_sources = results.ai_sources

    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{AI Text Detection Results (AUROC)}\n"
    latex += "\\label{tab:main_results}\n"

    # Column spec
    latex += "\\begin{tabular}{l" + "c" * len(ai_sources) + "c}\n"
    latex += "\\toprule\n"

    # Header
    latex += "Method & " + " & ".join(ai_sources) + " & Avg \\\\\n"
    latex += "\\midrule\n"

    # Rows
    for method in methods:
        row = method.replace("_", "\\_") + " & "
        aurocs = []

        for ai_source in ai_sources:
            matching = [r for r in results.results
                       if r.method == method and r.ai_source == ai_source]
            if matching:
                auroc = matching[0].auroc
                aurocs.append(auroc)
                row += f"{auroc:.3f} & "
            else:
                row += "-- & "

        if aurocs:
            row += f"{np.mean(aurocs):.3f}"
        else:
            row += "--"

        latex += row + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


if __name__ == "__main__":
    # Example usage
    print("Main Results Experiment Runner")
    print("=" * 50)
    print("\nUsage:")
    print("  from experiments.main_results import run_main_experiment")
    print("  results = run_main_experiment(human_texts, ai_texts_by_source)")
    print("\nOr run via Modal for GPU acceleration.")
