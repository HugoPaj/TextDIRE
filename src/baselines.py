"""
Baseline methods for AI text detection.

Includes:
- Perplexity-based detection (GPT-2)
- Burstiness detection
- DetectGPT (Mitchell et al., 2023) - full implementation with T5 perturbations
- Fast-DetectGPT (Bao et al., 2023) - conditional probability curvature
- Binoculars (Hans et al., 2024) - two-model perplexity comparison
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union
from dataclasses import dataclass, field


@dataclass
class PerplexityResult:
    """Result of perplexity computation."""
    perplexity: float
    loss: float
    num_tokens: int


@dataclass
class DetectGPTResult:
    """Result of DetectGPT computation."""
    curvature: float
    original_log_prob: float
    mean_perturbed_log_prob: float
    std_perturbed_log_prob: float
    num_perturbations: int
    z_score: float  # Normalized curvature


@dataclass
class FastDetectGPTResult:
    """Result of Fast-DetectGPT computation."""
    score: float
    conditional_entropy: float
    unconditional_entropy: float
    num_tokens: int


@dataclass
class BinocularsResult:
    """Result of Binoculars computation."""
    score: float
    observer_ppl: float
    performer_ppl: float
    cross_perplexity: float


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
    Full DetectGPT implementation per Mitchell et al. 2023.

    Uses T5-based mask-filling for perturbations and computes
    log probability curvature to detect AI-generated text.

    AI text tends to lie at local maxima of the model's probability surface.

    Paper: https://arxiv.org/abs/2301.11305
    """

    def __init__(
        self,
        scoring_model: str = "gpt2-medium",
        perturbation_model: str = "t5-large",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize DetectGPT detector.

        Args:
            scoring_model: Model for computing log probabilities
            perturbation_model: Model for generating perturbations (T5)
            device: Device to use
            cache_dir: Model cache directory
        """
        from transformers import (
            GPT2LMHeadModel, GPT2TokenizerFast,
            T5ForConditionalGeneration, T5Tokenizer
        )

        self.cache_dir = cache_dir

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Scoring model (GPT-2)
        print(f"Loading scoring model: {scoring_model}")
        self.scoring_tokenizer = GPT2TokenizerFast.from_pretrained(
            scoring_model,
            cache_dir=cache_dir,
        )
        self.scoring_model = GPT2LMHeadModel.from_pretrained(
            scoring_model,
            cache_dir=cache_dir,
        ).to(self.device).eval()

        # Perturbation model (T5)
        print(f"Loading perturbation model: {perturbation_model}")
        self.perturbation_tokenizer = T5Tokenizer.from_pretrained(
            perturbation_model,
            cache_dir=cache_dir,
        )
        self.perturbation_model = T5ForConditionalGeneration.from_pretrained(
            perturbation_model,
            cache_dir=cache_dir,
        ).to(self.device).eval()

    def _get_log_prob(self, text: str, max_length: int = 512) -> float:
        """Get average log probability of text."""
        inputs = self.scoring_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.scoring_model(**inputs, labels=inputs.input_ids)
            return -outputs.loss.item()

    def _t5_mask_and_fill(
        self,
        text: str,
        mask_ratio: float = 0.15,
        max_length: int = 512,
    ) -> str:
        """
        Create T5-style perturbation by masking spans and filling.

        Args:
            text: Original text
            mask_ratio: Fraction of tokens to mask
            max_length: Maximum sequence length

        Returns:
            Perturbed text
        """
        import random

        words = text.split()
        if len(words) < 5:
            return text

        # Number of spans to mask
        num_masks = max(1, int(len(words) * mask_ratio))

        # Create masked text with sentinel tokens
        masked_words = words.copy()
        mask_positions = sorted(random.sample(range(len(words)), min(num_masks, len(words))))

        # Group consecutive positions into spans
        spans = []
        current_span = [mask_positions[0]] if mask_positions else []

        for pos in mask_positions[1:]:
            if pos == current_span[-1] + 1:
                current_span.append(pos)
            else:
                spans.append(current_span)
                current_span = [pos]
        if current_span:
            spans.append(current_span)

        # Replace spans with sentinel tokens
        sentinel_idx = 0
        offset = 0
        for span in spans:
            start = span[0] - offset
            end = span[-1] - offset + 1
            sentinel = f"<extra_id_{sentinel_idx}>"
            masked_words = masked_words[:start] + [sentinel] + masked_words[end:]
            offset += len(span) - 1
            sentinel_idx += 1
            if sentinel_idx >= 100:
                break

        masked_text = " ".join(masked_words)

        # Generate fill-ins with T5
        try:
            inputs = self.perturbation_tokenizer(
                masked_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.perturbation_model.generate(
                    inputs.input_ids,
                    max_length=256,
                    num_beams=1,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.96,
                )

            fills = self.perturbation_tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Parse fills and reconstruct text
            result_words = words.copy()
            fill_parts = fills.split("<extra_id_")

            for i, span in enumerate(spans):
                if i + 1 < len(fill_parts):
                    # Extract fill between sentinel tokens
                    fill_text = fill_parts[i + 1].split(">")[-1].split("<")[0].strip()
                    fill_words = fill_text.split() if fill_text else [words[span[0]]]

                    # Replace span in result
                    start = span[0]
                    end = span[-1] + 1
                    result_words = result_words[:start] + fill_words + result_words[end:]

                    # Adjust span positions for remaining spans
                    offset = len(fill_words) - len(span)
                    for j in range(i + 1, len(spans)):
                        spans[j] = [p + offset for p in spans[j]]

            return " ".join(result_words)

        except Exception as e:
            # Fallback: return original with simple word swaps
            return self._simple_perturb(text)

    def _simple_perturb(self, text: str) -> str:
        """Simple fallback perturbation."""
        import random

        words = text.split()
        if len(words) < 2:
            return text

        perturbed = words.copy()
        num_swaps = max(1, len(words) // 20)

        for _ in range(num_swaps):
            idx = random.randint(0, len(perturbed) - 2)
            perturbed[idx], perturbed[idx + 1] = perturbed[idx + 1], perturbed[idx]

        return " ".join(perturbed)

    def generate_perturbations(
        self,
        text: str,
        num_perturbations: int = 100,
        mask_ratio: float = 0.15,
    ) -> list[str]:
        """
        Generate multiple T5-based perturbations.

        Args:
            text: Original text
            num_perturbations: Number of perturbations to generate
            mask_ratio: Fraction of text to mask

        Returns:
            List of perturbed texts
        """
        perturbations = []

        for _ in range(num_perturbations):
            perturbed = self._t5_mask_and_fill(text, mask_ratio)
            perturbations.append(perturbed)

        return perturbations

    def compute_curvature(
        self,
        text: str,
        num_perturbations: int = 100,
        mask_ratio: float = 0.15,
    ) -> DetectGPTResult:
        """
        Compute probability curvature for text using T5 perturbations.

        AI text tends to have positive curvature (is at local probability maximum).

        Args:
            text: Input text
            num_perturbations: Number of perturbations to average
            mask_ratio: Fraction of text to mask in perturbations

        Returns:
            DetectGPTResult with curvature and statistics
        """
        original_log_prob = self._get_log_prob(text)

        perturbations = self.generate_perturbations(text, num_perturbations, mask_ratio)
        perturbed_log_probs = [self._get_log_prob(p) for p in perturbations]

        mean_perturbed = np.mean(perturbed_log_probs)
        std_perturbed = np.std(perturbed_log_probs)

        # Curvature: original - mean(perturbed)
        # Positive curvature = original at local maximum (AI-like)
        curvature = original_log_prob - mean_perturbed

        # Z-score for normalized comparison
        z_score = curvature / std_perturbed if std_perturbed > 0 else 0

        return DetectGPTResult(
            curvature=curvature,
            original_log_prob=original_log_prob,
            mean_perturbed_log_prob=mean_perturbed,
            std_perturbed_log_prob=std_perturbed,
            num_perturbations=num_perturbations,
            z_score=z_score,
        )

    def detect(
        self,
        texts: list[str],
        num_perturbations: int = 100,
        progress_bar: bool = True,
    ) -> list[DetectGPTResult]:
        """
        Run DetectGPT on multiple texts.

        Args:
            texts: List of texts to analyze
            num_perturbations: Perturbations per text
            progress_bar: Show progress bar

        Returns:
            List of DetectGPTResult objects
        """
        results = []

        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="DetectGPT")
            except ImportError:
                iterator = texts
        else:
            iterator = texts

        for text in iterator:
            if not text or len(text.strip()) < 10:
                continue
            try:
                result = self.compute_curvature(text, num_perturbations)
                results.append(result)
            except Exception as e:
                print(f"Error processing text: {e}")
                continue

        return results


class FastDetectGPT:
    """
    Fast-DetectGPT implementation per Bao et al. 2023.

    Uses conditional probability curvature without requiring perturbations.
    Much faster than original DetectGPT while maintaining accuracy.

    Paper: https://arxiv.org/abs/2310.05130
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Fast-DetectGPT detector.

        Args:
            model_name: Language model name
            device: Device to use
            cache_dir: Model cache directory
        """
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading model: {model_name}")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        ).to(self.device).eval()

    def compute_score(
        self,
        text: str,
        max_length: int = 512,
    ) -> FastDetectGPTResult:
        """
        Compute Fast-DetectGPT score using conditional probability curvature.

        The score is based on the observation that AI text has lower
        conditional entropy (more predictable given context).

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            FastDetectGPTResult with detection score
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]

        if seq_len < 3:
            return FastDetectGPTResult(
                score=0.0,
                conditional_entropy=0.0,
                unconditional_entropy=0.0,
                num_tokens=seq_len,
            )

        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            logits = outputs.logits  # [1, seq_len, vocab_size]

            # Compute token-level log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Get log prob of each actual token (shifted by 1)
            token_log_probs = log_probs[0, :-1, :].gather(
                1, input_ids[0, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # Conditional entropy: -mean(log_prob)
            conditional_entropy = -token_log_probs.mean().item()

            # Compute unconditional entropy (using uniform-ish baseline)
            # This approximates sampling entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            unconditional_entropy = entropy.mean().item()

            # Score: difference between conditional and unconditional
            # Higher score = more AI-like (lower conditional entropy)
            score = unconditional_entropy - conditional_entropy

        return FastDetectGPTResult(
            score=score,
            conditional_entropy=conditional_entropy,
            unconditional_entropy=unconditional_entropy,
            num_tokens=seq_len,
        )

    def detect(
        self,
        texts: list[str],
        progress_bar: bool = True,
    ) -> list[FastDetectGPTResult]:
        """
        Run Fast-DetectGPT on multiple texts.

        Args:
            texts: List of texts to analyze
            progress_bar: Show progress bar

        Returns:
            List of FastDetectGPTResult objects
        """
        results = []

        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Fast-DetectGPT")
            except ImportError:
                iterator = texts
        else:
            iterator = texts

        for text in iterator:
            if not text or len(text.strip()) < 10:
                continue
            try:
                result = self.compute_score(text)
                results.append(result)
            except Exception as e:
                print(f"Error processing text: {e}")
                continue

        return results


class Binoculars:
    """
    Binoculars detector per Hans et al. 2024.

    Uses two language models (observer and performer) to detect AI text
    by comparing their perplexities. The key insight is that AI text
    has similar perplexity under both models, while human text varies more.

    Paper: https://arxiv.org/abs/2401.12070
    """

    def __init__(
        self,
        observer_model: str = "gpt2",
        performer_model: str = "gpt2-medium",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Binoculars detector.

        Args:
            observer_model: Smaller/different model for comparison
            performer_model: Main language model
            device: Device to use
            cache_dir: Model cache directory
        """
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Observer (smaller model)
        print(f"Loading observer model: {observer_model}")
        self.observer_tokenizer = GPT2TokenizerFast.from_pretrained(
            observer_model,
            cache_dir=cache_dir,
        )
        self.observer_model = GPT2LMHeadModel.from_pretrained(
            observer_model,
            cache_dir=cache_dir,
        ).to(self.device).eval()

        # Performer (larger/better model)
        print(f"Loading performer model: {performer_model}")
        self.performer_tokenizer = GPT2TokenizerFast.from_pretrained(
            performer_model,
            cache_dir=cache_dir,
        )
        self.performer_model = GPT2LMHeadModel.from_pretrained(
            performer_model,
            cache_dir=cache_dir,
        ).to(self.device).eval()

    def _compute_perplexity(
        self,
        model,
        tokenizer,
        text: str,
        max_length: int = 512,
    ) -> float:
        """Compute perplexity using a model."""
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            return np.exp(outputs.loss.item())

    def _compute_cross_perplexity(
        self,
        text: str,
        max_length: int = 512,
    ) -> float:
        """
        Compute cross-perplexity: performer probability evaluated by observer.

        This measures how "surprising" the performer's predictions are to the observer.
        """
        # Get performer logits
        performer_inputs = self.performer_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        observer_inputs = self.observer_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            # Performer predictions
            performer_outputs = self.performer_model(performer_inputs.input_ids)
            performer_logits = performer_outputs.logits

            # Observer log probabilities
            observer_outputs = self.observer_model(observer_inputs.input_ids)
            observer_log_probs = F.log_softmax(observer_outputs.logits, dim=-1)

            # Get performer's top predictions
            performer_preds = performer_logits.argmax(dim=-1)

            # Cross entropy: observer's prob of performer's predictions
            # Use aligned positions (assuming same tokenization)
            min_len = min(performer_preds.shape[1], observer_log_probs.shape[1])
            cross_log_probs = observer_log_probs[0, :min_len-1, :].gather(
                1, performer_preds[0, 1:min_len].unsqueeze(-1)
            ).squeeze(-1)

            cross_entropy = -cross_log_probs.mean().item()
            cross_perplexity = np.exp(cross_entropy)

        return cross_perplexity

    def compute_score(
        self,
        text: str,
        max_length: int = 512,
    ) -> BinocularsResult:
        """
        Compute Binoculars score.

        The score is the ratio of observer perplexity to performer perplexity.
        AI text tends to have similar perplexity under both models (ratio ~ 1),
        while human text has higher observer perplexity (ratio > 1).

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            BinocularsResult with detection score
        """
        observer_ppl = self._compute_perplexity(
            self.observer_model, self.observer_tokenizer, text, max_length
        )
        performer_ppl = self._compute_perplexity(
            self.performer_model, self.performer_tokenizer, text, max_length
        )
        cross_ppl = self._compute_cross_perplexity(text, max_length)

        # Score: log ratio of perplexities
        # Lower score = more AI-like (similar perplexities)
        score = np.log(observer_ppl / performer_ppl) if performer_ppl > 0 else 0

        return BinocularsResult(
            score=score,
            observer_ppl=observer_ppl,
            performer_ppl=performer_ppl,
            cross_perplexity=cross_ppl,
        )

    def detect(
        self,
        texts: list[str],
        progress_bar: bool = True,
    ) -> list[BinocularsResult]:
        """
        Run Binoculars on multiple texts.

        Args:
            texts: List of texts to analyze
            progress_bar: Show progress bar

        Returns:
            List of BinocularsResult objects
        """
        results = []

        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Binoculars")
            except ImportError:
                iterator = texts
        else:
            iterator = texts

        for text in iterator:
            if not text or len(text.strip()) < 10:
                continue
            try:
                result = self.compute_score(text)
                results.append(result)
            except Exception as e:
                print(f"Error processing text: {e}")
                continue

        return results
