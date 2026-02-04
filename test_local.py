"""
Quick local test of detection methods (no GPU needed).

Tests perplexity and Fast-DetectGPT on the generated dataset.
For DIRE (requires GPU), use Modal.

Usage:
    python test_local.py
    python test_local.py --dataset datasets/combined_modern.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Local detection test")
    parser.add_argument("--dataset", default="datasets/combined_modern.json",
                        help="Path to combined dataset")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max samples to test (for speed)")

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        print("Run 'python generate_and_test.py' first to generate the dataset.")
        return

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    human_texts = data["human_texts"][:args.max_samples]
    ai_texts = data["ai_texts"][:args.max_samples]

    all_texts = human_texts + ai_texts
    all_labels = [0] * len(human_texts) + [1] * len(ai_texts)

    print(f"Testing on {len(human_texts)} human + {len(ai_texts)} AI samples")

    # Test perplexity baseline
    print("\n" + "=" * 50)
    print("PERPLEXITY BASELINE (GPT-2)")
    print("=" * 50)

    try:
        from src.baselines import PerplexityDetector
        from src.evaluate import comprehensive_evaluation

        print("Loading GPT-2...")
        detector = PerplexityDetector(model_name="gpt2")

        print("Computing perplexity...")
        results = detector.compute_perplexities(all_texts, progress_bar=True)

        # Lower perplexity = more AI-like, so negate
        scores = [-r.perplexity for r in results]

        # Evaluate
        eval_result = comprehensive_evaluation(all_labels[:len(scores)], scores, "Perplexity")

        print(f"\nResults:")
        print(f"  AUROC: {eval_result['auroc']:.4f} ({eval_result['auroc_ci_lower']:.4f}-{eval_result['auroc_ci_upper']:.4f})")
        print(f"  Accuracy: {eval_result['accuracy']:.4f}")
        print(f"  F1: {eval_result['f1']:.4f}")

        # Per-class stats
        human_ppl = [-s for s, l in zip(scores, all_labels[:len(scores)]) if l == 0]
        ai_ppl = [-s for s, l in zip(scores, all_labels[:len(scores)]) if l == 1]

        import numpy as np
        print(f"\n  Human perplexity: {np.mean(human_ppl):.2f} (+/- {np.std(human_ppl):.2f})")
        print(f"  AI perplexity: {np.mean(ai_ppl):.2f} (+/- {np.std(ai_ppl):.2f})")

    except Exception as e:
        print(f"Perplexity test failed: {e}")

    # Test Fast-DetectGPT
    print("\n" + "=" * 50)
    print("FAST-DETECTGPT")
    print("=" * 50)

    try:
        from src.baselines import FastDetectGPT

        print("Loading model...")
        detector = FastDetectGPT(model_name="gpt2")

        print("Computing scores...")
        results = detector.detect(all_texts, progress_bar=True)
        scores = [r.score for r in results]

        eval_result = comprehensive_evaluation(all_labels[:len(scores)], scores, "Fast-DetectGPT")

        print(f"\nResults:")
        print(f"  AUROC: {eval_result['auroc']:.4f} ({eval_result['auroc_ci_lower']:.4f}-{eval_result['auroc_ci_upper']:.4f})")
        print(f"  Accuracy: {eval_result['accuracy']:.4f}")
        print(f"  F1: {eval_result['f1']:.4f}")

    except Exception as e:
        print(f"Fast-DetectGPT test failed: {e}")

    print("\n" + "=" * 50)
    print("To run DIRE (requires GPU), use:")
    print("  modal run modal_app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
