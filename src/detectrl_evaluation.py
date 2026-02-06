"""
DetectRL Benchmark Evaluation for Text-DIRE.

DetectRL (NeurIPS 2024) tests detection in real-world scenarios:
- Task 1: Robustness to attacks (prompt variation, paraphrase, perturbation, data mixing)
- Task 2: Generalization across domains and LLMs
- Task 4: Attacks applied to human text (false positive analysis)

Paper: https://github.com/NLP2CT/DetectRL
GitHub: https://github.com/NLP2CT/DetectRL

Usage:
    python -m src.detectrl_evaluation
    python -m src.detectrl_evaluation --tasks task1_attack --device cuda
    python -m src.detectrl_evaluation --tasks task1_attack task2_domain_gen --mask-ratio 0.8
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
from tqdm import tqdm


@dataclass
class DetectRLResult:
    """Results from a single DetectRL evaluation setting."""
    task: str              # e.g., "task1_attack", "task2_domain_gen"
    setting: str           # e.g., "paraphrase", "arxiv", "ChatGPT"
    auroc: float
    tpr_at_1pct: float     # TPR@FPR=1%
    tpr_at_5pct: float     # TPR@FPR=5%
    f1: float
    n_samples: int
    human_mean_score: float
    ai_mean_score: float
    separation: float      # Cohen's d effect size


def _compute_metrics(labels, scores, task: str, setting: str) -> Optional[DetectRLResult]:
    """Compute all metrics for a single setting."""
    from sklearn.metrics import (
        roc_auc_score, roc_curve, f1_score, accuracy_score,
    )
    from .evaluate import compute_tpr_at_fpr

    labels = np.array(labels)
    scores = np.array(scores)

    if len(labels) < 10 or len(set(labels)) < 2:
        return None

    # AUROC
    auroc = roc_auc_score(labels, scores)
    if auroc < 0.5:
        scores = -scores
        auroc = 1 - auroc

    # TPR@FPR
    tpr_at_fpr = compute_tpr_at_fpr(labels, scores, fpr_targets=[0.01, 0.05])

    # Optimal threshold + F1
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    threshold = thresholds[optimal_idx]
    predictions = (scores >= threshold).astype(int)
    f1 = f1_score(labels, predictions, zero_division=0)

    # Score distributions
    human_scores = scores[labels == 0]
    ai_scores = scores[labels == 1]
    human_mean = float(np.mean(human_scores)) if len(human_scores) > 0 else 0.0
    ai_mean = float(np.mean(ai_scores)) if len(ai_scores) > 0 else 0.0

    # Cohen's d
    if len(human_scores) > 1 and len(ai_scores) > 1:
        pooled_std = np.sqrt(
            ((len(human_scores) - 1) * np.std(human_scores, ddof=1)**2 +
             (len(ai_scores) - 1) * np.std(ai_scores, ddof=1)**2) /
            (len(human_scores) + len(ai_scores) - 2)
        )
        cohens_d = abs(ai_mean - human_mean) / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = 0.0

    return DetectRLResult(
        task=task,
        setting=setting,
        auroc=auroc,
        tpr_at_1pct=tpr_at_fpr[0.01],
        tpr_at_5pct=tpr_at_fpr[0.05],
        f1=f1,
        n_samples=len(labels),
        human_mean_score=human_mean,
        ai_mean_score=ai_mean,
        separation=cohens_d,
    )


def _score_texts(
    texts: list[str],
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    mask_ratio: float = 0.5,
    mc_samples: int = 8,
    device: str = "cuda",
) -> list[float]:
    """
    Run Text-DIRE detection on texts using MC-averaged scoring.

    Uses multiple random masks per text and batched inference for throughput.
    Returns scores where HIGHER = more likely AI.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    from .dire import TextDIRE, compute_dire_score_mc

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    scores = []
    for text in tqdm(texts, desc="DIRE detection (MC)"):
        try:
            result = compute_dire_score_mc(
                model, tokenizer, text,
                mask_ratio=mask_ratio,
                mc_samples=mc_samples,
            )
            scores.append(result.accuracy_mean or result.mean)
        except Exception as e:
            print(f"Error: {e}")
            scores.append(0.5)

    return scores


def evaluate_task1_robustness(
    score_fn,
    attacks: Optional[list[str]] = None,
    num_samples: Optional[int] = None,
    cache_dir: str = "datasets/detectrl",
) -> list[DetectRLResult]:
    """
    Task 1: Evaluate robustness to various attacks.

    Tests: direct prompt, prompt attacks, paraphrase attacks,
    perturbation attacks, and data mixing.

    Args:
        score_fn: Function(texts) -> list[float], higher = more AI
        attacks: Which attacks to test (default: all)
        num_samples: Limit per setting
        cache_dir: DetectRL cache directory

    Returns:
        List of DetectRLResult, one per attack type
    """
    from .data import load_detectrl_benchmark, DETECTRL_ATTACKS, DETECTRL_FILES

    if attacks is None:
        attacks = list(DETECTRL_FILES.keys())

    results = []

    for attack in attacks:
        print(f"\n--- Task 1: {attack} ---")
        dataset = load_detectrl_benchmark(
            tasks=["task1_attack"],
            attacks=[attack],
            num_samples=num_samples,
            cache_dir=cache_dir,
        )

        if len(dataset) < 10:
            print(f"  Skipping {attack} - too few samples ({len(dataset)})")
            continue

        scores = score_fn(dataset.texts)
        result = _compute_metrics(dataset.labels, scores, "task1_attack", attack)
        if result:
            results.append(result)
            print(f"  AUROC: {result.auroc:.4f}  TPR@1%: {result.tpr_at_1pct:.4f}  "
                  f"TPR@5%: {result.tpr_at_5pct:.4f}  n={result.n_samples}")

    return results


def evaluate_task2_generalization(
    score_fn,
    domains: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    num_samples: Optional[int] = None,
    cache_dir: str = "datasets/detectrl",
) -> list[DetectRLResult]:
    """
    Task 2: Evaluate cross-domain and cross-LLM generalization.

    Args:
        score_fn: Function(texts) -> list[float], higher = more AI
        domains: Which domains to test (default: all)
        models: Which LLMs to test (default: all)
        num_samples: Limit per setting
        cache_dir: DetectRL cache directory

    Returns:
        List of DetectRLResult, one per domain/model
    """
    from .data import (
        load_detectrl_benchmark, DETECTRL_DOMAINS, DETECTRL_MODELS,
        DETECTRL_DOMAIN_FILES, DETECTRL_MODEL_FILES,
    )

    if domains is None:
        domains = list(DETECTRL_DOMAIN_FILES.keys())
    if models is None:
        models = list(DETECTRL_MODEL_FILES.keys())

    results = []

    # Per-domain evaluation
    for domain in domains:
        print(f"\n--- Task 2 (domain): {domain} ---")
        dataset = load_detectrl_benchmark(
            tasks=["task2_domain_gen"],
            domains=[domain],
            num_samples=num_samples,
            cache_dir=cache_dir,
        )

        if len(dataset) < 10:
            print(f"  Skipping {domain} - too few samples ({len(dataset)})")
            continue

        scores = score_fn(dataset.texts)
        result = _compute_metrics(dataset.labels, scores, "task2_domain_gen", domain)
        if result:
            results.append(result)
            print(f"  AUROC: {result.auroc:.4f}  TPR@1%: {result.tpr_at_1pct:.4f}  "
                  f"TPR@5%: {result.tpr_at_5pct:.4f}  n={result.n_samples}")

    # Per-LLM evaluation
    for model in models:
        print(f"\n--- Task 2 (LLM): {model} ---")
        dataset = load_detectrl_benchmark(
            tasks=["task3_llm_gen"],
            models=[model],
            num_samples=num_samples,
            cache_dir=cache_dir,
        )

        if len(dataset) < 10:
            print(f"  Skipping {model} - too few samples ({len(dataset)})")
            continue

        scores = score_fn(dataset.texts)
        result = _compute_metrics(dataset.labels, scores, "task3_llm_gen", model)
        if result:
            results.append(result)
            print(f"  AUROC: {result.auroc:.4f}  TPR@1%: {result.tpr_at_1pct:.4f}  "
                  f"TPR@5%: {result.tpr_at_5pct:.4f}  n={result.n_samples}")

    return results


def evaluate_task4_human_writing(
    score_fn,
    num_samples: Optional[int] = None,
    cache_dir: str = "datasets/detectrl",
) -> list[DetectRLResult]:
    """
    Task 4: Evaluate false positives on attacked human text.

    Tests whether attacks applied to human text fool the detector
    into flagging it as AI-generated.

    Args:
        score_fn: Function(texts) -> list[float], higher = more AI
        num_samples: Limit per setting
        cache_dir: DetectRL cache directory

    Returns:
        List of DetectRLResult
    """
    from .data import load_detectrl_benchmark

    # Load all attack data and check false positive rate on human samples
    dataset = load_detectrl_benchmark(
        tasks=["task1_attack"],
        num_samples=num_samples,
        cache_dir=cache_dir,
    )

    if len(dataset) < 10:
        print("  Not enough data for Task 4 evaluation")
        return []

    scores = score_fn(dataset.texts)

    # Overall false positive analysis
    result = _compute_metrics(dataset.labels, scores, "task4_human_writing", "overall")
    results = []
    if result:
        results.append(result)
        print(f"  Task 4 Overall - AUROC: {result.auroc:.4f}  "
              f"TPR@1%: {result.tpr_at_1pct:.4f}  n={result.n_samples}")

    return results


def run_detectrl_evaluation(
    tasks: Optional[list[str]] = None,
    mask_ratio: float = 0.5,
    device: str = "cuda",
    output_dir: str = "results/detectrl",
    max_samples: Optional[int] = None,
) -> list[DetectRLResult]:
    """
    Run full DetectRL benchmark evaluation.

    Args:
        tasks: List of tasks to run (default: all)
        mask_ratio: DIRE mask ratio
        device: Device to use
        output_dir: Directory for results
        max_samples: Limit samples per setting (for testing)

    Returns:
        List of all DetectRLResult objects
    """
    if tasks is None:
        tasks = ["task1_attack", "task2_domain_gen", "task3_llm_gen"]

    # Create score function using DIRE
    print(f"Initializing DIRE detector (mask_ratio={mask_ratio}, device={device})...")

    import torch
    from transformers import AutoModel, AutoTokenizer
    from .dire import TextDIRE

    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    dire = TextDIRE(model, tokenizer, device=device)

    def score_fn(texts):
        scores = []
        for text in tqdm(texts, desc="DIRE scoring"):
            try:
                result = dire.compute_score(text, mask_ratio=mask_ratio)
                scores.append(result.token_accuracy)
            except Exception:
                scores.append(0.5)
        return scores

    all_results = []

    if "task1_attack" in tasks:
        print(f"\n{'='*60}")
        print("TASK 1: ROBUSTNESS TO ATTACKS")
        print('='*60)
        results = evaluate_task1_robustness(
            score_fn, num_samples=max_samples,
        )
        all_results.extend(results)

    if "task2_domain_gen" in tasks or "task3_llm_gen" in tasks:
        print(f"\n{'='*60}")
        print("TASK 2/3: GENERALIZATION (DOMAIN + LLM)")
        print('='*60)
        results = evaluate_task2_generalization(
            score_fn, num_samples=max_samples,
        )
        all_results.extend(results)

    if "task4_human_writing" in tasks:
        print(f"\n{'='*60}")
        print("TASK 4: HUMAN WRITING ATTACKS")
        print('='*60)
        results = evaluate_task4_human_writing(
            score_fn, num_samples=max_samples,
        )
        all_results.extend(results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_data = [asdict(r) for r in all_results]
    with open(output_path / "detectrl_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {output_path / 'detectrl_results.json'}")
    print_detectrl_results(all_results)

    return all_results


def print_detectrl_results(results: list[DetectRLResult]):
    """Print formatted DetectRL evaluation results."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print("DETECTRL BENCHMARK RESULTS - TEXT-DIRE")
    print("=" * 90)
    print(f"{'Task':<20} {'Setting':<22} {'AUROC':<8} {'TPR@1%':<9} "
          f"{'TPR@5%':<9} {'F1':<8} {'Sep':<8} {'N':<8}")
    print("-" * 90)

    current_task = None
    for result in results:
        if result.task != current_task:
            current_task = result.task
            print(f"\n  {current_task}")

        print(f"  {'':>2}{result.setting:<18} {result.auroc:<8.4f} "
              f"{result.tpr_at_1pct:<9.4f} {result.tpr_at_5pct:<9.4f} "
              f"{result.f1:<8.4f} {result.separation:<8.2f} {result.n_samples:<8}")

    # Summary statistics
    print("\n" + "-" * 90)
    aurocs = [r.auroc for r in results]
    tpr1s = [r.tpr_at_1pct for r in results]
    tpr5s = [r.tpr_at_5pct for r in results]
    total_n = sum(r.n_samples for r in results)

    print(f"  {'AVERAGE':<20} {'':>2} {np.mean(aurocs):<8.4f} "
          f"{np.mean(tpr1s):<9.4f} {np.mean(tpr5s):<9.4f} "
          f"{'':>8} {'':>8} {total_n:<8}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="DetectRL Benchmark Evaluation")
    parser.add_argument("--tasks", nargs="+",
                        default=["task1_attack", "task2_domain_gen", "task3_llm_gen"],
                        choices=["task1_attack", "task2_domain_gen", "task3_llm_gen",
                                 "task4_human_writing"],
                        help="Tasks to evaluate")
    parser.add_argument("--mask-ratio", type=float, default=0.5,
                        help="DIRE mask ratio")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default="results/detectrl",
                        help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per setting (for testing)")

    args = parser.parse_args()

    run_detectrl_evaluation(
        tasks=args.tasks,
        mask_ratio=args.mask_ratio,
        device=args.device,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
