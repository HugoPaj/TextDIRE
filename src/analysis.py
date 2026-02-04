"""
Token-Type Analysis for Text-DIRE.

Analyzes which types of tokens are harder/easier to reconstruct,
providing insights into what makes AI text different from human text.

Analysis dimensions:
- Part-of-speech tags
- Word frequency
- Position in sentence
- Semantic category
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict


@dataclass
class TokenAnalysis:
    """Analysis results for a single token."""
    token: str
    position: int
    was_correct: bool
    confidence: float
    pos_tag: Optional[str] = None
    dependency: Optional[str] = None
    word_frequency: Optional[float] = None


@dataclass
class TextAnalysisResult:
    """Complete analysis for a single text."""
    text: str
    overall_accuracy: float
    token_analyses: list[TokenAnalysis]
    pos_accuracy: dict[str, float] = field(default_factory=dict)
    position_accuracy: dict[str, float] = field(default_factory=dict)
    frequency_accuracy: dict[str, float] = field(default_factory=dict)


@dataclass
class AggregateAnalysis:
    """Aggregated analysis across multiple texts."""
    num_texts: int
    overall_accuracy: float
    pos_accuracy: dict[str, float]
    pos_counts: dict[str, int]
    position_accuracy: dict[str, float]
    frequency_bins_accuracy: dict[str, float]
    hardest_tokens: list[tuple[str, float]]
    easiest_tokens: list[tuple[str, float]]


def load_word_frequencies(freq_file: Optional[str] = None) -> dict[str, float]:
    """
    Load word frequency data.

    Returns dict mapping word to log frequency (normalized).
    """
    # Use a simple frequency approximation based on word length
    # In practice, you'd load from a file like Google 1-grams
    frequencies = {}

    # Common words have higher frequency
    common_words = {
        'the': 1.0, 'a': 0.95, 'an': 0.9, 'is': 0.85, 'are': 0.8,
        'was': 0.75, 'were': 0.7, 'be': 0.65, 'been': 0.6, 'being': 0.55,
        'have': 0.5, 'has': 0.45, 'had': 0.4, 'do': 0.35, 'does': 0.3,
        'did': 0.25, 'will': 0.2, 'would': 0.15, 'could': 0.1, 'should': 0.05,
        'and': 0.95, 'or': 0.85, 'but': 0.8, 'if': 0.75, 'then': 0.7,
        'that': 0.9, 'this': 0.85, 'these': 0.7, 'those': 0.65,
        'it': 0.9, 'its': 0.7, 'they': 0.85, 'them': 0.8, 'their': 0.75,
        'he': 0.8, 'she': 0.75, 'him': 0.7, 'her': 0.7, 'his': 0.65,
        'in': 0.9, 'on': 0.85, 'at': 0.8, 'to': 0.95, 'for': 0.85,
        'of': 0.95, 'with': 0.85, 'by': 0.8, 'from': 0.75, 'about': 0.65,
    }

    frequencies.update(common_words)
    return frequencies


def get_frequency_bin(word: str, frequencies: dict[str, float]) -> str:
    """Categorize word into frequency bin."""
    freq = frequencies.get(word.lower(), 0)

    if freq > 0.8:
        return "very_common"
    elif freq > 0.5:
        return "common"
    elif freq > 0.2:
        return "moderate"
    elif freq > 0:
        return "rare"
    else:
        return "unknown"


def get_position_bin(position: int, seq_len: int) -> str:
    """Categorize token position into bins."""
    relative_pos = position / seq_len

    if relative_pos < 0.1:
        return "start"
    elif relative_pos < 0.3:
        return "early"
    elif relative_pos < 0.7:
        return "middle"
    elif relative_pos < 0.9:
        return "late"
    else:
        return "end"


def analyze_reconstruction(
    model,
    tokenizer,
    text: str,
    mask_ratio: float = 0.5,
    max_length: int = 512,
    nlp=None,
) -> TextAnalysisResult:
    """
    Analyze token-level reconstruction for a text.

    Args:
        model: Diffusion model
        tokenizer: Tokenizer
        text: Input text
        mask_ratio: Mask ratio for DIRE
        max_length: Maximum sequence length
        nlp: spaCy NLP model for POS tagging

    Returns:
        TextAnalysisResult with per-token analysis
    """
    import torch
    from src.dire import mask_tokens

    device = next(model.parameters()).device

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    original_ids = inputs["input_ids"]
    seq_len = original_ids.shape[1]

    # Get mask token ID
    mask_token_id = getattr(tokenizer, 'mask_token_id', 126336)

    # Create mask
    masked_ids, mask_positions = mask_tokens(original_ids, mask_ratio, mask_token_id)

    # Get predictions
    with torch.no_grad():
        outputs = model(masked_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        probs = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

    # Get word frequencies
    frequencies = load_word_frequencies()

    # Get POS tags if spaCy available
    pos_tags = {}
    dependencies = {}
    if nlp:
        try:
            doc = nlp(text)
            for i, token in enumerate(doc):
                pos_tags[i] = token.pos_
                dependencies[i] = token.dep_
        except Exception:
            pass

    # Analyze each masked token
    token_analyses = []
    masked_indices = mask_positions[0].nonzero(as_tuple=True)[0]

    for idx in masked_indices:
        pos = idx.item()
        orig_token_id = original_ids[0, pos].item()
        pred_token_id = predictions[0, pos].item()

        orig_token = tokenizer.decode([orig_token_id]).strip()
        was_correct = orig_token_id == pred_token_id
        confidence = probs[0, pos, pred_token_id].item()

        analysis = TokenAnalysis(
            token=orig_token,
            position=pos,
            was_correct=was_correct,
            confidence=confidence,
            pos_tag=pos_tags.get(pos),
            dependency=dependencies.get(pos),
            word_frequency=frequencies.get(orig_token.lower()),
        )
        token_analyses.append(analysis)

    # Compute aggregated statistics
    overall_accuracy = sum(1 for t in token_analyses if t.was_correct) / len(token_analyses) if token_analyses else 0

    # POS accuracy
    pos_accuracy = {}
    pos_correct = defaultdict(int)
    pos_total = defaultdict(int)
    for t in token_analyses:
        if t.pos_tag:
            pos_total[t.pos_tag] += 1
            if t.was_correct:
                pos_correct[t.pos_tag] += 1
    for pos in pos_total:
        pos_accuracy[pos] = pos_correct[pos] / pos_total[pos]

    # Position accuracy
    position_accuracy = {}
    pos_bin_correct = defaultdict(int)
    pos_bin_total = defaultdict(int)
    for t in token_analyses:
        bin_name = get_position_bin(t.position, seq_len)
        pos_bin_total[bin_name] += 1
        if t.was_correct:
            pos_bin_correct[bin_name] += 1
    for bin_name in pos_bin_total:
        position_accuracy[bin_name] = pos_bin_correct[bin_name] / pos_bin_total[bin_name]

    # Frequency accuracy
    frequency_accuracy = {}
    freq_correct = defaultdict(int)
    freq_total = defaultdict(int)
    for t in token_analyses:
        freq_bin = get_frequency_bin(t.token, frequencies)
        freq_total[freq_bin] += 1
        if t.was_correct:
            freq_correct[freq_bin] += 1
    for freq_bin in freq_total:
        frequency_accuracy[freq_bin] = freq_correct[freq_bin] / freq_total[freq_bin]

    return TextAnalysisResult(
        text=text,
        overall_accuracy=overall_accuracy,
        token_analyses=token_analyses,
        pos_accuracy=pos_accuracy,
        position_accuracy=position_accuracy,
        frequency_accuracy=frequency_accuracy,
    )


def aggregate_analyses(
    results: list[TextAnalysisResult],
) -> AggregateAnalysis:
    """
    Aggregate analysis results across multiple texts.

    Args:
        results: List of TextAnalysisResult objects

    Returns:
        AggregateAnalysis with overall statistics
    """
    if not results:
        return AggregateAnalysis(
            num_texts=0,
            overall_accuracy=0,
            pos_accuracy={},
            pos_counts={},
            position_accuracy={},
            frequency_bins_accuracy={},
            hardest_tokens=[],
            easiest_tokens=[],
        )

    # Overall accuracy
    total_correct = sum(
        sum(1 for t in r.token_analyses if t.was_correct)
        for r in results
    )
    total_tokens = sum(len(r.token_analyses) for r in results)
    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    # Aggregate POS accuracy
    pos_correct = defaultdict(int)
    pos_total = defaultdict(int)
    for r in results:
        for t in r.token_analyses:
            if t.pos_tag:
                pos_total[t.pos_tag] += 1
                if t.was_correct:
                    pos_correct[t.pos_tag] += 1

    pos_accuracy = {pos: pos_correct[pos] / pos_total[pos] for pos in pos_total}
    pos_counts = dict(pos_total)

    # Aggregate position accuracy
    position_correct = defaultdict(int)
    position_total = defaultdict(int)
    for r in results:
        for pos, acc in r.position_accuracy.items():
            # This is approximate - we don't have the raw counts
            count = sum(1 for t in r.token_analyses if get_position_bin(t.position, len(r.text.split())) == pos)
            position_total[pos] += count
            position_correct[pos] += int(acc * count)

    position_accuracy = {
        pos: position_correct[pos] / position_total[pos]
        for pos in position_total if position_total[pos] > 0
    }

    # Aggregate frequency accuracy
    freq_correct = defaultdict(int)
    freq_total = defaultdict(int)
    frequencies = load_word_frequencies()
    for r in results:
        for t in r.token_analyses:
            freq_bin = get_frequency_bin(t.token, frequencies)
            freq_total[freq_bin] += 1
            if t.was_correct:
                freq_correct[freq_bin] += 1

    frequency_accuracy = {
        freq: freq_correct[freq] / freq_total[freq]
        for freq in freq_total if freq_total[freq] > 0
    }

    # Find hardest and easiest tokens
    token_correct = defaultdict(int)
    token_total = defaultdict(int)
    for r in results:
        for t in r.token_analyses:
            token_total[t.token.lower()] += 1
            if t.was_correct:
                token_correct[t.token.lower()] += 1

    token_accuracy = {
        token: token_correct[token] / token_total[token]
        for token in token_total if token_total[token] >= 5
    }

    hardest = sorted(token_accuracy.items(), key=lambda x: x[1])[:20]
    easiest = sorted(token_accuracy.items(), key=lambda x: -x[1])[:20]

    return AggregateAnalysis(
        num_texts=len(results),
        overall_accuracy=overall_accuracy,
        pos_accuracy=pos_accuracy,
        pos_counts=pos_counts,
        position_accuracy=position_accuracy,
        frequency_bins_accuracy=frequency_accuracy,
        hardest_tokens=hardest,
        easiest_tokens=easiest,
    )


def compare_human_vs_ai_patterns(
    human_results: list[TextAnalysisResult],
    ai_results: list[TextAnalysisResult],
) -> dict:
    """
    Compare reconstruction patterns between human and AI text.

    Returns:
        Dict with comparative statistics
    """
    human_agg = aggregate_analyses(human_results)
    ai_agg = aggregate_analyses(ai_results)

    comparison = {
        "overall_accuracy": {
            "human": human_agg.overall_accuracy,
            "ai": ai_agg.overall_accuracy,
            "difference": human_agg.overall_accuracy - ai_agg.overall_accuracy,
        },
        "pos_accuracy_diff": {},
        "position_accuracy_diff": {},
        "frequency_accuracy_diff": {},
    }

    # POS differences
    all_pos = set(human_agg.pos_accuracy.keys()) | set(ai_agg.pos_accuracy.keys())
    for pos in all_pos:
        human_acc = human_agg.pos_accuracy.get(pos, 0)
        ai_acc = ai_agg.pos_accuracy.get(pos, 0)
        comparison["pos_accuracy_diff"][pos] = {
            "human": human_acc,
            "ai": ai_acc,
            "diff": human_acc - ai_acc,
        }

    # Position differences
    all_positions = set(human_agg.position_accuracy.keys()) | set(ai_agg.position_accuracy.keys())
    for pos in all_positions:
        human_acc = human_agg.position_accuracy.get(pos, 0)
        ai_acc = ai_agg.position_accuracy.get(pos, 0)
        comparison["position_accuracy_diff"][pos] = {
            "human": human_acc,
            "ai": ai_acc,
            "diff": human_acc - ai_acc,
        }

    # Frequency differences
    all_freqs = set(human_agg.frequency_bins_accuracy.keys()) | set(ai_agg.frequency_bins_accuracy.keys())
    for freq in all_freqs:
        human_acc = human_agg.frequency_bins_accuracy.get(freq, 0)
        ai_acc = ai_agg.frequency_bins_accuracy.get(freq, 0)
        comparison["frequency_accuracy_diff"][freq] = {
            "human": human_acc,
            "ai": ai_acc,
            "diff": human_acc - ai_acc,
        }

    return comparison


def print_analysis_summary(analysis: AggregateAnalysis, label: str = ""):
    """Print a summary of the analysis."""
    print(f"\n{'='*50}")
    print(f"TOKEN ANALYSIS SUMMARY {label}")
    print(f"{'='*50}")

    print(f"\nOverall Accuracy: {analysis.overall_accuracy:.4f}")
    print(f"Number of Texts: {analysis.num_texts}")

    print("\nPOS Tag Accuracy:")
    for pos, acc in sorted(analysis.pos_accuracy.items(), key=lambda x: -x[1]):
        count = analysis.pos_counts.get(pos, 0)
        print(f"  {pos:<8} {acc:.4f} (n={count})")

    print("\nPosition Accuracy:")
    for pos, acc in sorted(analysis.position_accuracy.items(), key=lambda x: x[0]):
        print(f"  {pos:<10} {acc:.4f}")

    print("\nWord Frequency Accuracy:")
    for freq, acc in sorted(analysis.frequency_bins_accuracy.items()):
        print(f"  {freq:<12} {acc:.4f}")

    print("\nHardest Tokens (lowest accuracy):")
    for token, acc in analysis.hardest_tokens[:10]:
        print(f"  {token:<15} {acc:.4f}")

    print("\nEasiest Tokens (highest accuracy):")
    for token, acc in analysis.easiest_tokens[:10]:
        print(f"  {token:<15} {acc:.4f}")


if __name__ == "__main__":
    print("Token Analysis Module")
    print("=" * 50)
    print("\nUsage:")
    print("  from src.analysis import analyze_reconstruction")
    print("  result = analyze_reconstruction(model, tokenizer, text)")
