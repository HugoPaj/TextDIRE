"""
Shared scoring helpers for local FastAPI and Vercel serverless handlers.
"""

from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass
from typing import Literal


DEFAULT_MASK_RATIOS = [0.3, 0.5, 0.7]
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    scores: dict[float, float]
    notes: list[str]


def provider_name() -> str:
    return os.getenv("TEXTDIRE_PROVIDER", "demo").strip().lower() or "demo"


def resolve_mask_ratios(mode: str, requested: list[float] | None) -> list[float]:
    if requested:
        ratios = requested
    elif mode == "fast":
        ratios = [0.5]
    elif mode == "careful":
        ratios = [0.3, 0.5, 0.7]
    else:
        ratios = DEFAULT_MASK_RATIOS

    valid = sorted({round(float(r), 2) for r in ratios if 0.05 <= float(r) <= 0.95})
    if not valid:
        raise ValueError("Choose mask ratios between 0.05 and 0.95.")
    return valid


def score_text(text: str, mask_ratios: list[float]) -> ProviderResult:
    provider = provider_name()
    if provider == "modal":
        return score_with_modal(text, mask_ratios)
    if provider != "demo":
        raise RuntimeError(f"Unknown TEXTDIRE_PROVIDER={provider!r}. Use 'demo' or 'modal'.")
    return score_with_demo_signal(text, mask_ratios)


def score_with_modal(text: str, mask_ratios: list[float]) -> ProviderResult:
    try:
        import modal
    except Exception as exc:  # pragma: no cover - depends on runtime deps
        raise RuntimeError(
            "Modal provider is enabled, but the Modal package could not be imported. "
            "Install requirements and configure Modal credentials."
        ) from exc

    app_name = os.getenv("TEXTDIRE_MODAL_APP", "text-dire").strip() or "text-dire"
    function_name = (
        os.getenv("TEXTDIRE_MODAL_FUNCTION", "compute_dire_scores_batch").strip()
        or "compute_dire_scores_batch"
    )

    try:
        compute_dire_scores_batch = _lookup_modal_function(modal, app_name, function_name)
        results = compute_dire_scores_batch.remote([text], [1], mask_ratios)
    except Exception as exc:  # pragma: no cover - depends on Modal service state
        raise RuntimeError(
            "Modal provider is enabled, but the deployed Text-DIRE function could not run. "
            "Deploy it with `modal deploy modal_app.py` and verify Modal auth. "
            f"Lookup target: {app_name}.{function_name}. Underlying error: {exc}"
        ) from exc

    if not results:
        raise RuntimeError("Modal returned no DIRE results for this text.")

    row = results[0]
    scores: dict[float, float] = {}
    for ratio in mask_ratios:
        key = f"error_{ratio}"
        if key in row:
            scores[ratio] = float(row[key])

    if not scores:
        raise RuntimeError("Modal response did not include reconstruction error scores.")

    return ProviderResult(
        provider="modal",
        scores=scores,
        notes=["Scored with LLaDA-8B on Modal GPU inference."],
    )


def score_with_demo_signal(text: str, mask_ratios: list[float]) -> ProviderResult:
    """
    Lightweight local stand-in for UI testing.

    This is intentionally labeled as demo. It does not implement DIRE; it only
    creates stable, text-dependent scores so the website can be exercised before
    Modal GPU inference is configured.
    """
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    unique_ratio = len(set(words)) / max(1, len(words))
    long_words = sum(1 for word in words if len(word) >= 8) / max(1, len(words))
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_lengths = [len(re.findall(r"[A-Za-z0-9']+", sentence)) for sentence in sentences]
    sentence_variation = (
        statistics.pstdev(sentence_lengths) / max(1.0, statistics.fmean(sentence_lengths))
        if len(sentence_lengths) > 1
        else 0.0
    )
    repeated_bigram_rate = _repeated_bigram_rate(words)

    base_error = (
        0.38
        + (unique_ratio * 0.18)
        + (long_words * 0.16)
        + min(sentence_variation, 1.0) * 0.14
        - min(repeated_bigram_rate * 5.0, 0.18)
    )
    base_error = max(0.18, min(0.82, base_error))

    scores = {
        ratio: max(0.0, min(1.0, base_error + ((ratio - 0.5) * 0.08)))
        for ratio in mask_ratios
    }

    return ProviderResult(
        provider="demo",
        scores=scores,
        notes=[
            "Demo mode is active. Set TEXTDIRE_PROVIDER=modal to use real DIRE GPU scoring.",
        ],
    )


def classify_score(score: float) -> tuple[Literal["human", "ai", "uncertain"], float]:
    distance = score - DEFAULT_THRESHOLD
    confidence = min(0.99, 0.5 + abs(distance) * 1.6)
    if abs(distance) < 0.035:
        return "uncertain", max(0.35, confidence - 0.2)
    if distance > 0:
        return "human", confidence
    return "ai", confidence


def build_analysis_response(
    provider_result: ProviderResult,
    elapsed_seconds: float,
) -> dict:
    score = statistics.fmean(provider_result.scores.values())
    prediction, confidence = classify_score(score)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "score": round(score, 4),
        "threshold": DEFAULT_THRESHOLD,
        "provider": provider_result.provider,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "breakdown": [
            {
                "mask_ratio": ratio,
                "reconstruction_error": value,
                "token_accuracy": max(0.0, min(1.0, 1.0 - value)),
            }
            for ratio, value in sorted(provider_result.scores.items())
        ],
        "notes": [
            *provider_result.notes,
            "Text-DIRE is a research signal, not proof of authorship.",
            "Short, heavily edited, translated, or domain-specific text can be unreliable.",
        ],
    }


def _lookup_modal_function(modal_module, app_name: str, function_name: str):
    function_cls = modal_module.Function
    if hasattr(function_cls, "from_name"):
        return function_cls.from_name(app_name, function_name)
    return function_cls.lookup(app_name, function_name)


def _repeated_bigram_rate(words: list[str]) -> float:
    if len(words) < 4:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return 1.0 - (len(set(bigrams)) / len(bigrams))
