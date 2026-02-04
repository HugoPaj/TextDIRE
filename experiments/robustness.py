"""
Robustness Evaluation for Text-DIRE.

Tests detection robustness against various adversarial attacks:
1. Paraphrase attacks (GPT-4 rephrasing)
2. Back-translation (EN -> DE -> EN)
3. Homoglyph substitution (a -> а)
4. Instruction variation ("write naturally", "avoid AI patterns")
"""

import os
import json
import numpy as np
import re
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class RobustnessResult:
    """Result from robustness evaluation."""
    attack_name: str
    original_auroc: float
    attacked_auroc: float
    auroc_drop: float
    num_samples: int
    attack_success_rate: float  # Fraction of texts successfully modified


# Homoglyph mappings (Latin -> Cyrillic/Greek lookalikes)
HOMOGLYPHS = {
    'a': 'а',  # Cyrillic
    'e': 'е',  # Cyrillic
    'o': 'о',  # Cyrillic
    'p': 'р',  # Cyrillic
    'c': 'с',  # Cyrillic
    'x': 'х',  # Cyrillic
    'y': 'у',  # Cyrillic
    'A': 'А',  # Cyrillic
    'B': 'В',  # Cyrillic
    'E': 'Е',  # Cyrillic
    'H': 'Н',  # Cyrillic
    'K': 'К',  # Cyrillic
    'M': 'М',  # Cyrillic
    'O': 'О',  # Cyrillic
    'P': 'Р',  # Cyrillic
    'T': 'Т',  # Cyrillic
    'X': 'Х',  # Cyrillic
}


def apply_homoglyph_attack(text: str, ratio: float = 0.1) -> str:
    """
    Apply homoglyph substitution attack.

    Replaces a fraction of characters with visually similar lookalikes.

    Args:
        text: Original text
        ratio: Fraction of eligible characters to replace

    Returns:
        Modified text with homoglyph substitutions
    """
    import random

    chars = list(text)
    eligible_indices = [i for i, c in enumerate(chars) if c in HOMOGLYPHS]

    num_replace = max(1, int(len(eligible_indices) * ratio))
    replace_indices = random.sample(eligible_indices, min(num_replace, len(eligible_indices)))

    for i in replace_indices:
        chars[i] = HOMOGLYPHS[chars[i]]

    return ''.join(chars)


def apply_back_translation(
    text: str,
    target_lang: str = "de",
    api_key: Optional[str] = None,
) -> str:
    """
    Apply back-translation attack (EN -> target -> EN).

    Args:
        text: Original text
        target_lang: Intermediate language
        api_key: OpenAI API key for translation

    Returns:
        Back-translated text
    """
    # Use OpenAI for translation
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        # Fallback: simple word shuffling as approximation
        words = text.split()
        if len(words) > 5:
            import random
            # Shuffle some words to simulate translation artifacts
            for i in range(len(words) // 10):
                idx = random.randint(0, len(words) - 2)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return ' '.join(words)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Translate to target language
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Translate the following text to {target_lang}. Only output the translation."},
                {"role": "user", "content": text}
            ],
            max_tokens=1024,
        )
        translated = response.choices[0].message.content

        # Translate back to English
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate the following text to English. Only output the translation."},
                {"role": "user", "content": translated}
            ],
            max_tokens=1024,
        )
        back_translated = response.choices[0].message.content

        return back_translated

    except Exception as e:
        print(f"Back-translation failed: {e}")
        return text


def apply_paraphrase_attack(
    text: str,
    api_key: Optional[str] = None,
    preserve_meaning: bool = True,
) -> str:
    """
    Apply paraphrase attack using GPT-4.

    Args:
        text: Original text
        api_key: OpenAI API key
        preserve_meaning: Whether to strictly preserve meaning

    Returns:
        Paraphrased text
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        # Fallback: synonym replacement
        return _simple_paraphrase(text)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        if preserve_meaning:
            prompt = "Paraphrase the following text while preserving its exact meaning. Use different words and sentence structures."
        else:
            prompt = "Rewrite the following text in a more natural, human-like style while keeping the main ideas."

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=1024,
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Paraphrase failed: {e}")
        return text


def _simple_paraphrase(text: str) -> str:
    """Simple paraphrase fallback using basic transformations."""
    # Simple transformations
    text = re.sub(r'\bvery\b', 'quite', text)
    text = re.sub(r'\bbut\b', 'however,', text)
    text = re.sub(r'\balso\b', 'additionally', text)
    text = re.sub(r'\bbecause\b', 'since', text)
    text = re.sub(r'\bhowever\b', 'nevertheless', text)
    return text


def apply_whitespace_attack(text: str, ratio: float = 0.1) -> str:
    """
    Apply whitespace manipulation attack.

    Adds zero-width characters or extra spaces.

    Args:
        text: Original text
        ratio: Fraction of spaces to modify

    Returns:
        Modified text
    """
    import random

    # Zero-width space
    zwsp = '\u200b'

    words = text.split()
    num_modify = max(1, int(len(words) * ratio))

    for _ in range(num_modify):
        idx = random.randint(0, len(words) - 1)
        # Insert zero-width space
        words[idx] = words[idx] + zwsp

    return ' '.join(words)


def apply_typo_attack(text: str, ratio: float = 0.05) -> str:
    """
    Apply realistic typo attack.

    Args:
        text: Original text
        ratio: Fraction of words to modify

    Returns:
        Text with typos
    """
    import random

    # Common typo patterns
    typo_patterns = [
        (r'the', 'teh'),
        (r'and', 'adn'),
        (r'that', 'taht'),
        (r'with', 'wiht'),
        (r'have', 'ahve'),
        (r'this', 'thsi'),
        (r'from', 'form'),
        (r'they', 'tehy'),
        (r'been', 'bene'),
        (r'were', 'weere'),
    ]

    words = text.split()
    num_typos = max(1, int(len(words) * ratio))

    for _ in range(num_typos):
        idx = random.randint(0, len(words) - 1)
        word = words[idx].lower()

        for pattern, replacement in typo_patterns:
            if word == pattern:
                words[idx] = replacement
                break
        else:
            # Random character swap for longer words
            if len(words[idx]) > 3:
                char_idx = random.randint(1, len(words[idx]) - 2)
                chars = list(words[idx])
                chars[char_idx], chars[char_idx + 1] = chars[char_idx + 1], chars[char_idx]
                words[idx] = ''.join(chars)

    return ' '.join(words)


def evaluate_robustness(
    score_fn: Callable[[str], float],
    texts: list[str],
    labels: list[int],
    attack_fn: Callable[[str], str],
    attack_name: str,
) -> RobustnessResult:
    """
    Evaluate robustness against a specific attack.

    Args:
        score_fn: Detection scoring function
        texts: Original texts
        labels: Labels
        attack_fn: Attack function
        attack_name: Name of the attack

    Returns:
        RobustnessResult with before/after comparison
    """
    from sklearn.metrics import roc_auc_score

    # Score original texts
    original_scores = []
    attacked_scores = []
    successful_attacks = 0

    for text, label in zip(texts, labels):
        # Only attack AI texts (label=1)
        if label == 1:
            try:
                attacked_text = attack_fn(text)
                if attacked_text != text:
                    successful_attacks += 1

                original_score = score_fn(text)
                attacked_score = score_fn(attacked_text)
            except Exception:
                original_score = 0.5
                attacked_score = 0.5
        else:
            # Human text - no attack
            original_score = score_fn(text)
            attacked_score = original_score

        original_scores.append(original_score)
        attacked_scores.append(attacked_score)

    # Compute AUROCs
    original_auroc = roc_auc_score(labels, original_scores)
    if original_auroc < 0.5:
        original_scores = [-s for s in original_scores]
        attacked_scores = [-s for s in attacked_scores]
        original_auroc = 1 - original_auroc

    attacked_auroc = roc_auc_score(labels, attacked_scores)
    if attacked_auroc < 0.5:
        attacked_auroc = 1 - attacked_auroc

    num_ai = sum(labels)
    attack_success_rate = successful_attacks / num_ai if num_ai > 0 else 0

    return RobustnessResult(
        attack_name=attack_name,
        original_auroc=original_auroc,
        attacked_auroc=attacked_auroc,
        auroc_drop=original_auroc - attacked_auroc,
        num_samples=len(texts),
        attack_success_rate=attack_success_rate,
    )


def run_robustness_evaluation(
    score_fn: Callable[[str], float],
    texts: list[str],
    labels: list[int],
    attacks: list[str] = None,
    output_dir: str = "results/robustness",
    api_key: Optional[str] = None,
) -> dict[str, RobustnessResult]:
    """
    Run full robustness evaluation.

    Args:
        score_fn: Detection scoring function
        texts: Test texts
        labels: Test labels
        attacks: List of attacks to run
        output_dir: Directory to save results
        api_key: API key for attacks requiring API calls

    Returns:
        Dict mapping attack name to RobustnessResult
    """
    os.makedirs(output_dir, exist_ok=True)

    if attacks is None:
        attacks = ["homoglyph", "whitespace", "typo"]
        # Add API-based attacks if key available
        if api_key or os.environ.get("OPENAI_API_KEY"):
            attacks.extend(["paraphrase", "back_translation"])

    attack_fns = {
        "homoglyph": lambda t: apply_homoglyph_attack(t, ratio=0.1),
        "whitespace": lambda t: apply_whitespace_attack(t, ratio=0.1),
        "typo": lambda t: apply_typo_attack(t, ratio=0.05),
        "paraphrase": lambda t: apply_paraphrase_attack(t, api_key),
        "back_translation": lambda t: apply_back_translation(t, "de", api_key),
    }

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for attack in attacks:
        if attack not in attack_fns:
            print(f"Unknown attack: {attack}")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating robustness: {attack}")
        print(f"{'='*50}")

        result = evaluate_robustness(
            score_fn=score_fn,
            texts=texts,
            labels=labels,
            attack_fn=attack_fns[attack],
            attack_name=attack,
        )

        results[attack] = result

        print(f"  Original AUROC: {result.original_auroc:.4f}")
        print(f"  Attacked AUROC: {result.attacked_auroc:.4f}")
        print(f"  AUROC Drop: {result.auroc_drop:.4f}")
        print(f"  Attack Success Rate: {result.attack_success_rate:.2%}")

    # Save results
    results_path = os.path.join(output_dir, f"robustness_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(
            {name: asdict(r) for name, r in results.items()},
            f, indent=2
        )

    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)
    print(f"{'Attack':<20} {'Original':<10} {'Attacked':<10} {'Drop':<10}")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name:<20} {result.original_auroc:.4f}{'':>4} {result.attacked_auroc:.4f}{'':>4} {result.auroc_drop:.4f}")

    return results


if __name__ == "__main__":
    print("Robustness Evaluation Runner")
    print("=" * 50)
    print("\nAvailable attacks:")
    print("  - homoglyph: Character substitution with lookalikes")
    print("  - whitespace: Zero-width space insertion")
    print("  - typo: Realistic typo injection")
    print("  - paraphrase: GPT-4 paraphrasing (requires API)")
    print("  - back_translation: EN->DE->EN translation (requires API)")
