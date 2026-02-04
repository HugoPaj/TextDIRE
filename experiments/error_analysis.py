"""
Error Analysis for Text-DIRE.

Provides detailed analysis of detection failures:
1. Confusion matrices per model
2. False positive/negative case studies
3. Token-level error patterns by POS tags
4. Length vs accuracy correlation
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ConfusionMatrix:
    """Confusion matrix for binary classification."""
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        return self.true_positive / (self.true_positive + self.false_positive) if (self.true_positive + self.false_positive) > 0 else 0

    @property
    def recall(self) -> float:
        return self.true_positive / (self.true_positive + self.false_negative) if (self.true_positive + self.false_negative) > 0 else 0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    @property
    def accuracy(self) -> float:
        total = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        return (self.true_positive + self.true_negative) / total if total > 0 else 0


@dataclass
class ErrorCase:
    """A single error case for analysis."""
    text: str
    true_label: int
    predicted_label: int
    score: float
    error_type: str  # "false_positive" or "false_negative"
    text_length: int
    word_count: int
    source: Optional[str] = None
    token_analysis: Optional[dict] = None


def compute_confusion_matrix(
    labels: list[int],
    predictions: list[int],
) -> ConfusionMatrix:
    """Compute confusion matrix from labels and predictions."""
    tp = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 0)

    return ConfusionMatrix(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )


def find_optimal_threshold(
    labels: list[int],
    scores: list[float],
) -> float:
    """Find optimal classification threshold using Youden's J."""
    from sklearn.metrics import roc_curve

    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)

    return thresholds[optimal_idx]


def extract_error_cases(
    texts: list[str],
    labels: list[int],
    scores: list[float],
    threshold: float,
    sources: Optional[list[str]] = None,
    max_cases: int = 50,
) -> tuple[list[ErrorCase], list[ErrorCase]]:
    """
    Extract false positive and false negative cases.

    Returns:
        Tuple of (false_positives, false_negatives)
    """
    false_positives = []
    false_negatives = []

    predictions = [1 if s >= threshold else 0 for s in scores]

    for i, (text, label, pred, score) in enumerate(zip(texts, labels, predictions, scores)):
        source = sources[i] if sources else None

        if label == 0 and pred == 1:
            # False positive (human classified as AI)
            error_case = ErrorCase(
                text=text[:500],  # Truncate for storage
                true_label=label,
                predicted_label=pred,
                score=score,
                error_type="false_positive",
                text_length=len(text),
                word_count=len(text.split()),
                source=source,
            )
            false_positives.append(error_case)

        elif label == 1 and pred == 0:
            # False negative (AI classified as human)
            error_case = ErrorCase(
                text=text[:500],
                true_label=label,
                predicted_label=pred,
                score=score,
                error_type="false_negative",
                text_length=len(text),
                word_count=len(text.split()),
                source=source,
            )
            false_negatives.append(error_case)

    # Sort by score (most confident errors first)
    false_positives.sort(key=lambda x: -x.score)
    false_negatives.sort(key=lambda x: x.score)

    return false_positives[:max_cases], false_negatives[:max_cases]


def analyze_pos_patterns(
    error_cases: list[ErrorCase],
    nlp=None,
) -> dict[str, dict]:
    """
    Analyze POS tag patterns in error cases.

    Args:
        error_cases: List of error cases
        nlp: spaCy NLP model (optional, will load if not provided)

    Returns:
        Dict with POS tag statistics
    """
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            print("spaCy not available, skipping POS analysis")
            return {}

    pos_counts = Counter()
    tag_counts = Counter()
    dep_counts = Counter()

    for case in error_cases:
        doc = nlp(case.text)

        for token in doc:
            pos_counts[token.pos_] += 1
            tag_counts[token.tag_] += 1
            dep_counts[token.dep_] += 1

    return {
        "pos_distribution": dict(pos_counts.most_common(20)),
        "tag_distribution": dict(tag_counts.most_common(20)),
        "dep_distribution": dict(dep_counts.most_common(20)),
    }


def analyze_length_correlation(
    texts: list[str],
    labels: list[int],
    scores: list[float],
    threshold: float,
) -> dict:
    """
    Analyze correlation between text length and detection accuracy.

    Returns:
        Dict with length-based accuracy statistics
    """
    # Define length buckets
    buckets = [
        (0, 100, "<100"),
        (100, 200, "100-200"),
        (200, 500, "200-500"),
        (500, 1000, "500-1000"),
        (1000, float('inf'), "1000+"),
    ]

    bucket_stats = {}

    for min_len, max_len, bucket_name in buckets:
        bucket_indices = [
            i for i, t in enumerate(texts)
            if min_len <= len(t.split()) < max_len
        ]

        if not bucket_indices:
            continue

        bucket_labels = [labels[i] for i in bucket_indices]
        bucket_scores = [scores[i] for i in bucket_indices]
        bucket_preds = [1 if s >= threshold else 0 for s in bucket_scores]

        correct = sum(1 for l, p in zip(bucket_labels, bucket_preds) if l == p)
        accuracy = correct / len(bucket_labels)

        bucket_stats[bucket_name] = {
            "count": len(bucket_labels),
            "accuracy": accuracy,
            "human_count": bucket_labels.count(0),
            "ai_count": bucket_labels.count(1),
        }

    return bucket_stats


def analyze_source_performance(
    texts: list[str],
    labels: list[int],
    scores: list[float],
    sources: list[str],
    threshold: float,
) -> dict[str, dict]:
    """
    Analyze detection performance per source.

    Returns:
        Dict mapping source to performance metrics
    """
    from sklearn.metrics import roc_auc_score

    source_stats = {}
    unique_sources = set(sources)

    for source in unique_sources:
        indices = [i for i, s in enumerate(sources) if s == source]

        if len(indices) < 10:
            continue

        source_labels = [labels[i] for i in indices]
        source_scores = [scores[i] for i in indices]
        source_preds = [1 if s >= threshold else 0 for s in source_scores]

        correct = sum(1 for l, p in zip(source_labels, source_preds) if l == p)
        accuracy = correct / len(source_labels)

        # Compute AUROC if we have both classes
        if len(set(source_labels)) > 1:
            auroc = roc_auc_score(source_labels, source_scores)
            if auroc < 0.5:
                auroc = 1 - auroc
        else:
            auroc = None

        source_stats[source] = {
            "count": len(source_labels),
            "accuracy": accuracy,
            "auroc": auroc,
            "human_count": source_labels.count(0),
            "ai_count": source_labels.count(1),
        }

    return source_stats


def run_error_analysis(
    texts: list[str],
    labels: list[int],
    scores: list[float],
    sources: Optional[list[str]] = None,
    output_dir: str = "results/error_analysis",
    analyze_pos: bool = True,
) -> dict:
    """
    Run comprehensive error analysis.

    Args:
        texts: Test texts
        labels: True labels
        scores: Detection scores
        sources: Optional source identifiers
        output_dir: Directory to save results
        analyze_pos: Whether to run POS analysis (requires spaCy)

    Returns:
        Dict with all analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find optimal threshold
    threshold = find_optimal_threshold(labels, scores)
    predictions = [1 if s >= threshold else 0 for s in scores]

    print(f"Optimal threshold: {threshold:.4f}")

    # Confusion matrix
    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)

    cm = compute_confusion_matrix(labels, predictions)
    print(f"  True Positive:  {cm.true_positive}")
    print(f"  True Negative:  {cm.true_negative}")
    print(f"  False Positive: {cm.false_positive}")
    print(f"  False Negative: {cm.false_negative}")
    print(f"\n  Precision: {cm.precision:.4f}")
    print(f"  Recall:    {cm.recall:.4f}")
    print(f"  F1 Score:  {cm.f1:.4f}")
    print(f"  Accuracy:  {cm.accuracy:.4f}")

    # Extract error cases
    print("\n" + "=" * 50)
    print("ERROR CASE EXTRACTION")
    print("=" * 50)

    false_positives, false_negatives = extract_error_cases(
        texts, labels, scores, threshold, sources
    )

    print(f"  False Positives: {len(false_positives)}")
    print(f"  False Negatives: {len(false_negatives)}")

    # Length correlation
    print("\n" + "=" * 50)
    print("LENGTH VS ACCURACY")
    print("=" * 50)

    length_stats = analyze_length_correlation(texts, labels, scores, threshold)
    for bucket, stats in length_stats.items():
        print(f"  {bucket}: {stats['accuracy']:.3f} ({stats['count']} samples)")

    # Source performance
    source_stats = None
    if sources:
        print("\n" + "=" * 50)
        print("PERFORMANCE BY SOURCE")
        print("=" * 50)

        source_stats = analyze_source_performance(
            texts, labels, scores, sources, threshold
        )
        for source, stats in sorted(source_stats.items(), key=lambda x: -x[1]['count']):
            auroc_str = f"{stats['auroc']:.3f}" if stats['auroc'] else "N/A"
            print(f"  {source}: acc={stats['accuracy']:.3f}, auroc={auroc_str} ({stats['count']})")

    # POS analysis
    pos_analysis = None
    if analyze_pos and (false_positives or false_negatives):
        print("\n" + "=" * 50)
        print("POS TAG ANALYSIS")
        print("=" * 50)

        try:
            if false_positives:
                fp_pos = analyze_pos_patterns(false_positives)
                print("\nFalse Positives - Top POS tags:")
                for pos, count in list(fp_pos.get("pos_distribution", {}).items())[:5]:
                    print(f"    {pos}: {count}")

            if false_negatives:
                fn_pos = analyze_pos_patterns(false_negatives)
                print("\nFalse Negatives - Top POS tags:")
                for pos, count in list(fn_pos.get("pos_distribution", {}).items())[:5]:
                    print(f"    {pos}: {count}")

            pos_analysis = {
                "false_positives": fp_pos if false_positives else {},
                "false_negatives": fn_pos if false_negatives else {},
            }

        except Exception as e:
            print(f"  POS analysis skipped: {e}")

    # Compile results
    results = {
        "timestamp": timestamp,
        "threshold": threshold,
        "confusion_matrix": asdict(cm),
        "length_analysis": length_stats,
        "source_analysis": source_stats,
        "pos_analysis": pos_analysis,
        "false_positives": [asdict(fp) for fp in false_positives[:10]],
        "false_negatives": [asdict(fn) for fn in false_negatives[:10]],
        "statistics": {
            "total_samples": len(texts),
            "total_errors": cm.false_positive + cm.false_negative,
            "error_rate": (cm.false_positive + cm.false_negative) / len(texts),
        }
    }

    # Save results
    results_path = os.path.join(output_dir, f"error_analysis_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")

    return results


def print_error_examples(
    error_cases: list[ErrorCase],
    num_examples: int = 5,
    error_type: str = "errors",
):
    """Print example error cases for manual inspection."""
    print(f"\n{error_type.upper()} - Example Cases:")
    print("-" * 60)

    for i, case in enumerate(error_cases[:num_examples]):
        print(f"\n[{i+1}] Score: {case.score:.4f}, Words: {case.word_count}")
        print(f"    Source: {case.source}")
        print(f"    Text: {case.text[:200]}...")


if __name__ == "__main__":
    print("Error Analysis Runner")
    print("=" * 50)
    print("\nUsage:")
    print("  from experiments.error_analysis import run_error_analysis")
    print("  results = run_error_analysis(texts, labels, scores)")
