"""
RAID Benchmark Evaluation for Text-DIRE.

Evaluates Text-DIRE on the RAID benchmark (ACL 2024 / COLING 2025)
for direct comparison with published methods.

RAID: https://raid-bench.xyz
Paper: https://arxiv.org/abs/2405.07940
GitHub: https://github.com/liamdugan/raid
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
class RAIDEvalResult:
    """Evaluation results on RAID benchmark."""
    method: str
    overall_auroc: float
    overall_f1: float
    domain_aurocs: dict[str, float]
    model_aurocs: dict[str, float]
    attack_aurocs: dict[str, float]
    num_samples: int


RAID_DOMAINS = [
    "arxiv", "books", "news", "recipes",
    "reddit", "reviews", "wikipedia", "poetry"
]

RAID_MODELS = [
    "gpt4", "gpt3", "chatgpt", "cohere",
    "llama", "mistral", "mpt", "human"
]

RAID_ATTACKS = [
    "none", "paraphrase", "perturb_char", "perturb_word",
    "homoglyph", "number", "whitespace", "misspelling",
    "upper_lower", "article_deletion", "alternative_spelling"
]


def evaluate_on_raid(
    method_name: str,
    score_fn,
    domains: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    attacks: Optional[list[str]] = None,
    num_samples: Optional[int] = None,
    output_dir: str = "results/raid",
) -> RAIDEvalResult:
    """
    Evaluate a detection method on the RAID benchmark.

    Args:
        method_name: Name of the detection method
        score_fn: Function that takes text and returns detection score
                  (higher = more AI-like)
        domains: RAID domains to evaluate on (default: all)
        models: RAID models to evaluate on (default: all)
        attacks: Attack types to evaluate (default: all)
        num_samples: Limit samples for faster evaluation
        output_dir: Directory to save results

    Returns:
        RAIDEvalResult with comprehensive evaluation metrics
    """
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    from src.data import load_raid_benchmark

    os.makedirs(output_dir, exist_ok=True)

    if domains is None:
        domains = RAID_DOMAINS

    if attacks is None:
        attacks = RAID_ATTACKS

    print(f"Loading RAID benchmark...")
    raid_data = load_raid_benchmark(
        split="test",
        domains=domains,
        models=models,
        attacks=attacks,
        num_samples=num_samples,
    )

    print(f"Loaded {len(raid_data)} samples")

    # Compute scores
    print(f"Computing {method_name} scores...")
    scores = []
    valid_labels = []
    valid_sources = []

    from tqdm import tqdm
    for text, label, source in tqdm(
        zip(raid_data.texts, raid_data.labels, raid_data.sources),
        total=len(raid_data.texts),
        desc=method_name
    ):
        try:
            score = score_fn(text)
            scores.append(score)
            valid_labels.append(label)
            valid_sources.append(source)
        except Exception as e:
            continue

    scores = np.array(scores)
    valid_labels = np.array(valid_labels)

    # Overall AUROC
    overall_auroc = roc_auc_score(valid_labels, scores)
    if overall_auroc < 0.5:
        scores = -scores
        overall_auroc = 1 - overall_auroc

    # Optimal threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(valid_labels, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    predictions = (scores >= optimal_threshold).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(
        valid_labels, predictions, pos_label=1, average='binary', zero_division=0
    )

    # Per-domain AUROC
    domain_aurocs = {}
    for domain in domains:
        domain_mask = [domain in s for s in valid_sources]
        if sum(domain_mask) > 0:
            domain_labels = valid_labels[domain_mask]
            domain_scores = scores[domain_mask]
            if len(set(domain_labels)) > 1:
                auroc = roc_auc_score(domain_labels, domain_scores)
                if auroc < 0.5:
                    auroc = 1 - auroc
                domain_aurocs[domain] = auroc

    # Per-model AUROC
    model_aurocs = {}
    for model in RAID_MODELS:
        model_mask = [model in s for s in valid_sources]
        if sum(model_mask) > 0:
            model_labels = valid_labels[model_mask]
            model_scores = scores[model_mask]
            if len(set(model_labels)) > 1:
                auroc = roc_auc_score(model_labels, model_scores)
                if auroc < 0.5:
                    auroc = 1 - auroc
                model_aurocs[model] = auroc

    # Per-attack AUROC
    attack_aurocs = {}
    for attack in attacks:
        attack_mask = [attack in s for s in valid_sources]
        if sum(attack_mask) > 0:
            attack_labels = valid_labels[attack_mask]
            attack_scores = scores[attack_mask]
            if len(set(attack_labels)) > 1:
                auroc = roc_auc_score(attack_labels, attack_scores)
                if auroc < 0.5:
                    auroc = 1 - auroc
                attack_aurocs[attack] = auroc

    result = RAIDEvalResult(
        method=method_name,
        overall_auroc=overall_auroc,
        overall_f1=f1,
        domain_aurocs=domain_aurocs,
        model_aurocs=model_aurocs,
        attack_aurocs=attack_aurocs,
        num_samples=len(valid_labels),
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"raid_{method_name}_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\nResults saved to {results_path}")
    print_raid_results(result)

    return result


def print_raid_results(result: RAIDEvalResult):
    """Print formatted RAID evaluation results."""
    print("\n" + "=" * 60)
    print(f"RAID Benchmark Results: {result.method}")
    print("=" * 60)

    print(f"\nOverall Performance:")
    print(f"  AUROC: {result.overall_auroc:.4f}")
    print(f"  F1: {result.overall_f1:.4f}")
    print(f"  Samples: {result.num_samples}")

    print(f"\nPer-Domain AUROC:")
    for domain, auroc in sorted(result.domain_aurocs.items()):
        print(f"  {domain:<15} {auroc:.4f}")

    print(f"\nPer-Model AUROC:")
    for model, auroc in sorted(result.model_aurocs.items()):
        print(f"  {model:<15} {auroc:.4f}")

    print(f"\nPer-Attack AUROC:")
    for attack, auroc in sorted(result.attack_aurocs.items()):
        print(f"  {attack:<20} {auroc:.4f}")


def format_for_leaderboard(result: RAIDEvalResult) -> dict:
    """
    Format results for RAID leaderboard submission.

    Returns dict matching the RAID submission format.
    """
    return {
        "model_name": result.method,
        "overall": {
            "auroc": result.overall_auroc,
            "f1": result.overall_f1,
        },
        "by_domain": result.domain_aurocs,
        "by_generator": result.model_aurocs,
        "by_attack": result.attack_aurocs,
        "num_samples": result.num_samples,
    }


def compare_methods_on_raid(
    methods: dict[str, callable],
    output_dir: str = "results/raid",
    **kwargs
) -> dict[str, RAIDEvalResult]:
    """
    Compare multiple methods on RAID benchmark.

    Args:
        methods: Dict mapping method name to score function
        output_dir: Directory to save results
        **kwargs: Additional arguments passed to evaluate_on_raid

    Returns:
        Dict mapping method name to RAIDEvalResult
    """
    results = {}

    for method_name, score_fn in methods.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")

        result = evaluate_on_raid(
            method_name=method_name,
            score_fn=score_fn,
            output_dir=output_dir,
            **kwargs
        )
        results[method_name] = result

    # Print comparison table
    print("\n" + "=" * 80)
    print("RAID BENCHMARK COMPARISON")
    print("=" * 80)

    header = f"{'Method':<25} {'Overall':<10} {'Avg Domain':<12} {'Avg Attack':<12}"
    print(header)
    print("-" * len(header))

    for method_name, result in results.items():
        overall = result.overall_auroc
        avg_domain = np.mean(list(result.domain_aurocs.values())) if result.domain_aurocs else 0
        avg_attack = np.mean(list(result.attack_aurocs.values())) if result.attack_aurocs else 0

        print(f"{method_name:<25} {overall:.4f}{'':>4} {avg_domain:.4f}{'':>6} {avg_attack:.4f}")

    return results


if __name__ == "__main__":
    print("RAID Benchmark Evaluation")
    print("=" * 50)
    print("\nUsage:")
    print("  from experiments.raid_evaluation import evaluate_on_raid")
    print("  result = evaluate_on_raid('DIRE', score_function)")
    print("\nOr compare multiple methods:")
    print("  results = compare_methods_on_raid({'DIRE': fn1, 'DetectGPT': fn2})")
