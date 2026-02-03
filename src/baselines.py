"""
Baseline methods for AI text detection.

Includes perplexity-based detection and other common approaches.
"""

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class PerplexityResult:
    """Result of perplexity computation."""
    perplexity: float
    loss: float
    num_tokens: int


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    max_length: int = 512,
    stride: int = 256,
) -> PerplexityResult:
    """
    Compute perplexity of text using an autoregressive language model.

    Lower perplexity often indicates AI-generated text, as AI models
    tend to produce more "typical" outputs.

    Args:
        model: Language model (e.g., GPT-2)
        tokenizer: Corresponding tokenizer
        text: Input text
        max_length: Maximum sequence length for sliding window
        stride: Stride for sliding window

    Returns:
        PerplexityResult with perplexity, loss, and token count
    """
    device = next(model.parameters()).device

    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    if seq_len <= max_length:
        # Short text - compute directly
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        perplexity = np.exp(loss)
        return PerplexityResult(
            perplexity=perplexity,
            loss=loss,
            num_tokens=seq_len,
        )

    # Long text - use sliding window
    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()

        # Only compute loss on tokens not seen in previous window
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    loss = total_nll / seq_len
    perplexity = torch.exp(loss).item()

    return PerplexityResult(
        perplexity=perplexity,
        loss=loss.item(),
        num_tokens=seq_len,
    )


class PerplexityDetector:
    """
    Perplexity-based AI text detector.

    Uses the observation that AI-generated text often has lower perplexity
    (more "typical" according to a language model) than human text.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize perplexity detector.

        Args:
            model_name: Hugging Face model name (e.g., "gpt2", "gpt2-medium")
            device: Device to use (auto-detected if None)
            cache_dir: Directory to cache models
        """
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    def compute_perplexity(self, text: str, **kwargs) -> PerplexityResult:
        """Compute perplexity for a single text."""
        return compute_perplexity(self.model, self.tokenizer, text, **kwargs)

    def compute_perplexities(
        self,
        texts: list[str],
        progress_bar: bool = True,
    ) -> list[PerplexityResult]:
        """
        Compute perplexity for multiple texts.

        Args:
            texts: List of texts
            progress_bar: Whether to show progress bar

        Returns:
            List of PerplexityResult objects
        """
        results = []

        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Computing perplexity")
            except ImportError:
                iterator = texts
        else:
            iterator = texts

        for text in iterator:
            if not text or len(text.strip()) < 10:
                continue

            try:
                result = self.compute_perplexity(text)
                results.append(result)
            except Exception as e:
                print(f"Error computing perplexity: {e}")
                continue

        return results

    def predict(
        self,
        text: str,
        threshold: float = 50.0,
    ) -> tuple[str, float]:
        """
        Predict whether text is human or AI-generated.

        Args:
            text: Text to classify
            threshold: Perplexity threshold (lower -> AI)

        Returns:
            Tuple of (prediction, perplexity)
        """
        result = self.compute_perplexity(text)

        if result.perplexity < threshold:
            return "ai", result.perplexity
        else:
            return "human", result.perplexity


class BurstinessDetector:
    """
    Burstiness-based AI text detector.

    Measures the variance in perplexity across a text. Human text tends
    to have higher variance (more "bursty") than AI text.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    def compute_burstiness(
        self,
        text: str,
        window_size: int = 50,
    ) -> dict:
        """
        Compute burstiness metrics for text.

        Args:
            text: Input text
            window_size: Size of sliding window for local perplexity

        Returns:
            Dictionary with burstiness metrics
        """
        # Split into sentences or windows
        words = text.split()
        if len(words) < window_size:
            window_size = len(words) // 2

        if window_size < 5:
            return {"burstiness": 0, "mean_ppl": 0, "std_ppl": 0}

        # Compute perplexity for each window
        local_ppls = []

        for i in range(0, len(words) - window_size, window_size // 2):
            window_text = " ".join(words[i:i + window_size])

            try:
                result = compute_perplexity(
                    self.model,
                    self.tokenizer,
                    window_text,
                )
                local_ppls.append(result.perplexity)
            except Exception:
                continue

        if not local_ppls:
            return {"burstiness": 0, "mean_ppl": 0, "std_ppl": 0}

        mean_ppl = np.mean(local_ppls)
        std_ppl = np.std(local_ppls)

        # Burstiness = coefficient of variation
        burstiness = std_ppl / mean_ppl if mean_ppl > 0 else 0

        return {
            "burstiness": burstiness,
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "num_windows": len(local_ppls),
        }


class DetectGPTBaseline:
    """
    Simplified DetectGPT-style baseline.

    Measures how much the log probability changes when text is perturbed.
    AI text tends to lie at local maxima of the model's probability surface.

    Note: This is a simplified version - full DetectGPT uses multiple perturbations.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    def _get_log_prob(self, text: str) -> float:
        """Get average log probability of text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            return -outputs.loss.item()

    def _perturb_text(self, text: str, num_perturbations: int = 5) -> list[str]:
        """
        Simple perturbation: randomly swap adjacent words.
        """
        import random

        words = text.split()
        perturbations = []

        for _ in range(num_perturbations):
            perturbed = words.copy()

            # Swap 5% of adjacent word pairs
            num_swaps = max(1, len(words) // 20)

            for _ in range(num_swaps):
                if len(perturbed) < 2:
                    break
                idx = random.randint(0, len(perturbed) - 2)
                perturbed[idx], perturbed[idx + 1] = perturbed[idx + 1], perturbed[idx]

            perturbations.append(" ".join(perturbed))

        return perturbations

    def compute_curvature(
        self,
        text: str,
        num_perturbations: int = 5,
    ) -> dict:
        """
        Compute probability curvature for text.

        AI text tends to have negative curvature (perturbations decrease probability).

        Args:
            text: Input text
            num_perturbations: Number of perturbations to average

        Returns:
            Dictionary with original and perturbed log probs
        """
        original_log_prob = self._get_log_prob(text)

        perturbations = self._perturb_text(text, num_perturbations)
        perturbed_log_probs = [self._get_log_prob(p) for p in perturbations]

        mean_perturbed = np.mean(perturbed_log_probs)

        # Curvature estimate: original - mean(perturbed)
        # Positive = original is at local maximum (AI-like)
        curvature = original_log_prob - mean_perturbed

        return {
            "original_log_prob": original_log_prob,
            "mean_perturbed_log_prob": mean_perturbed,
            "curvature": curvature,
            "num_perturbations": num_perturbations,
        }
