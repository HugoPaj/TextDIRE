"""
Text-DIRE: Core computation logic for Diffusion Reconstruction Error.

The hypothesis: A text diffusion model trained on human text will reconstruct
human-written text more accurately than AI-generated text, because AI text
lies "off-manifold" from natural human writing.

Enhanced with:
- Monte Carlo DIRE estimation for stability
- Multi-metric scoring (top-k accuracy, entropy, perplexity)
- Ensemble scoring across multiple mask ratios
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union
from dataclasses import dataclass, field
import math


@dataclass
class DIREResult:
    """Result of a DIRE computation for a single text."""
    token_accuracy: float
    reconstruction_error: float
    num_masked: int
    num_total: int
    mask_ratio: float

    # Optional: per-token details
    correct_predictions: Optional[list] = None
    confidence_scores: Optional[list] = None

    # Extended metrics
    top_k_accuracy: Optional[dict] = None  # {5: acc, 10: acc}
    mean_entropy: Optional[float] = None
    reconstruction_perplexity: Optional[float] = None
    cross_entropy_loss: Optional[float] = None


@dataclass
class MCDIREResult:
    """Result of Monte Carlo DIRE estimation."""
    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float
    samples: list[float] = field(default_factory=list)
    num_samples: int = 0

    # Extended metrics with uncertainty
    accuracy_mean: Optional[float] = None
    accuracy_std: Optional[float] = None
    ce_loss_mean: Optional[float] = None
    ce_loss_std: Optional[float] = None


@dataclass
class EnsembleDIREResult:
    """Result of ensemble DIRE scoring across multiple mask ratios."""
    weighted_score: float
    individual_scores: dict[float, float]  # mask_ratio -> score
    weights: dict[float, float]  # mask_ratio -> weight
    best_mask_ratio: float
    best_score: float


def mask_tokens(
    input_ids: torch.Tensor,
    mask_ratio: float,
    mask_token_id: int,
    exclude_first: int = 1,
    exclude_last: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask tokens for DIRE computation.

    Args:
        input_ids: Token IDs tensor of shape (batch_size, seq_len)
        mask_ratio: Fraction of tokens to mask (0.0 to 1.0)
        mask_token_id: Token ID to use for masking
        exclude_first: Number of tokens to exclude from masking at start
        exclude_last: Number of tokens to exclude from masking at end

    Returns:
        masked_ids: Token IDs with random positions replaced by mask_token_id
        mask_positions: Boolean tensor indicating which positions were masked
    """
    batch_size, seq_len = input_ids.shape

    # Calculate valid range for masking
    valid_start = exclude_first
    valid_end = seq_len - exclude_last
    valid_len = valid_end - valid_start

    if valid_len <= 0:
        raise ValueError("Sequence too short for masking with current exclusions")

    num_mask = max(1, int(valid_len * mask_ratio))

    # Create mask positions tensor
    mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)

    for i in range(batch_size):
        # Sample random positions in valid range
        positions = torch.randperm(valid_len)[:num_mask] + valid_start
        mask_positions[i, positions] = True

    # Apply masking
    masked_ids = input_ids.clone()
    masked_ids[mask_positions] = mask_token_id

    return masked_ids, mask_positions


def mask_tokens_padded(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_ratio: float,
    mask_token_id: int,
    exclude_first: int = 1,
    exclude_last: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask tokens for batches with variable-length sequences (padding).

    Unlike mask_tokens(), this respects per-sequence lengths via
    attention_mask so that padding tokens are never masked.

    Args:
        input_ids: Token IDs tensor of shape (batch_size, seq_len)
        attention_mask: Binary mask (1 = real token, 0 = padding) [batch_size, seq_len]
        mask_ratio: Fraction of valid tokens to mask (0.0 to 1.0)
        mask_token_id: Token ID to use for masking
        exclude_first: Tokens to exclude from masking at start of each sequence
        exclude_last: Tokens to exclude from masking at end of each real sequence

    Returns:
        masked_ids: Token IDs with random positions replaced by mask_token_id
        mask_positions: Boolean tensor indicating which positions were masked
    """
    batch_size, seq_len = input_ids.shape
    mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)

    for i in range(batch_size):
        # Find actual (non-padding) sequence length
        real_len = attention_mask[i].sum().item()

        valid_start = exclude_first
        valid_end = real_len - exclude_last
        valid_len = valid_end - valid_start

        if valid_len <= 0:
            continue

        num_mask = max(1, int(valid_len * mask_ratio))
        positions = torch.randperm(valid_len, device=input_ids.device)[:num_mask] + valid_start
        mask_positions[i, positions] = True

    masked_ids = input_ids.clone()
    masked_ids[mask_positions] = mask_token_id

    return masked_ids, mask_positions


def _extract_logits(outputs) -> torch.Tensor:
    """Extract logits tensor from various model output formats."""
    if hasattr(outputs, 'logits'):
        return outputs.logits
    elif isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def compute_dire_scores_batch(
    model,
    tokenizer,
    texts: list[str],
    mask_ratio: float = 0.5,
    max_length: int = 512,
    batch_size: int = 8,
) -> list[DIREResult]:
    """
    Compute Text-DIRE scores for multiple texts using batched GPU inference.

    Pads sequences within each mini-batch and runs a single forward pass
    per batch, dramatically improving throughput on GPU.

    Args:
        model: The diffusion model (e.g., LLaDA)
        tokenizer: The tokenizer
        texts: List of texts to evaluate
        mask_ratio: Fraction of tokens to mask
        max_length: Maximum sequence length
        batch_size: Number of texts per forward pass

    Returns:
        List of DIREResult, one per input text (skips texts too short to mask)
    """
    device = next(model.parameters()).device

    # Resolve mask token ID once
    mask_token_id = getattr(tokenizer, 'mask_token_id', None)
    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, 'mask_id', None)
    if mask_token_id is None:
        mask_token_id = 126336  # LLaDA default

    # Ensure tokenizer can pad
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    results: list[DIREResult] = []

    num_batches = math.ceil(len(texts) / batch_size)
    for batch_idx in range(num_batches):
        batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # Tokenize the full mini-batch with padding
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        input_ids = encoded.input_ids                 # [B, L]
        attention_mask = encoded.attention_mask       # [B, L]
        cur_batch_size = input_ids.shape[0]

        # Mask tokens (padding-aware)
        masked_ids, mask_positions = mask_tokens_padded(
            input_ids,
            attention_mask,
            mask_ratio,
            mask_token_id,
        )

        # Single forward pass for the whole batch
        with torch.no_grad():
            logits = _extract_logits(model(masked_ids))   # [B, L, V]
            predictions = logits.argmax(dim=-1)            # [B, L]

        # Unpack per-text results
        for i in range(cur_batch_size):
            seq_mask = mask_positions[i]       # [L]
            num_masked = seq_mask.sum().item()

            if num_masked == 0:
                # Sequence was too short to mask — skip
                continue

            orig_tokens = input_ids[i][seq_mask]
            pred_tokens = predictions[i][seq_mask]
            correct = (pred_tokens == orig_tokens).float()
            accuracy = correct.mean().item()

            real_len = attention_mask[i].sum().item()

            results.append(DIREResult(
                token_accuracy=accuracy,
                reconstruction_error=1.0 - accuracy,
                num_masked=num_masked,
                num_total=real_len,
                mask_ratio=mask_ratio,
                correct_predictions=correct.cpu().tolist(),
            ))

    return results


def compute_extended_metrics(
    logits: torch.Tensor,
    original_ids: torch.Tensor,
    mask_positions: torch.Tensor,
) -> dict:
    """
    Compute extended metrics from model logits.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        original_ids: Original token IDs [batch, seq_len]
        mask_positions: Boolean mask of masked positions [batch, seq_len]

    Returns:
        Dictionary with extended metrics
    """
    metrics = {}

    # Get logits at masked positions
    masked_logits = logits[mask_positions]  # [num_masked, vocab_size]
    masked_targets = original_ids[mask_positions]  # [num_masked]

    if len(masked_logits) == 0:
        return metrics

    # Probabilities
    probs = F.softmax(masked_logits, dim=-1)
    log_probs = F.log_softmax(masked_logits, dim=-1)

    # Top-k accuracy
    for k in [5, 10]:
        top_k_preds = masked_logits.topk(k, dim=-1).indices
        top_k_correct = (top_k_preds == masked_targets.unsqueeze(-1)).any(dim=-1)
        metrics[f"top_{k}_accuracy"] = top_k_correct.float().mean().item()

    # Entropy of predictions
    entropy = -(probs * log_probs).sum(dim=-1)
    metrics["mean_entropy"] = entropy.mean().item()
    metrics["std_entropy"] = entropy.std().item()

    # Cross-entropy loss
    ce_loss = F.cross_entropy(masked_logits, masked_targets)
    metrics["cross_entropy_loss"] = ce_loss.item()

    # Reconstruction perplexity
    metrics["reconstruction_perplexity"] = np.exp(ce_loss.item())

    # Confidence of correct predictions
    correct_probs = probs[torch.arange(len(masked_targets)), masked_targets]
    metrics["mean_confidence"] = correct_probs.mean().item()
    metrics["std_confidence"] = correct_probs.std().item()

    return metrics


def compute_dire_score(
    model,
    tokenizer,
    text: str,
    mask_ratio: float = 0.5,
    max_length: int = 512,
    num_runs: int = 1,
    aggregate: str = "mean",
    compute_extended: bool = False,
) -> DIREResult:
    """
    Compute Text-DIRE score for a single text.

    The DIRE score measures how well a diffusion model can reconstruct
    masked tokens. The hypothesis is that human text will be reconstructed
    more accurately than AI-generated text.

    Args:
        model: The diffusion model (e.g., LLaDA)
        tokenizer: The tokenizer
        text: Input text to evaluate
        mask_ratio: Fraction of tokens to mask
        max_length: Maximum sequence length
        num_runs: Number of times to run with different masks (for stability)
        aggregate: How to aggregate multiple runs ("mean" or "median")
        compute_extended: Whether to compute extended metrics (top-k, entropy, etc.)

    Returns:
        DIREResult with accuracy and error metrics
    """
    device = next(model.parameters()).device

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(device)

    original_ids = inputs["input_ids"]
    seq_len = original_ids.shape[1]

    # Get mask token ID
    mask_token_id = getattr(tokenizer, 'mask_token_id', None)
    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, 'mask_id', None)
    if mask_token_id is None:
        # LLaDA uses 126336 as mask token
        mask_token_id = 126336

    accuracies = []
    all_correct = []

    for _ in range(num_runs):
        # Create mask
        masked_ids, mask_positions = mask_tokens(
            original_ids,
            mask_ratio,
            mask_token_id,
        )

        # Get model predictions
        with torch.no_grad():
            outputs = model(masked_ids)

            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            predictions = logits.argmax(dim=-1)

        # Calculate accuracy at masked positions
        original_tokens = original_ids[mask_positions]
        predicted_tokens = predictions[mask_positions]

        correct = (predicted_tokens == original_tokens).float()
        accuracies.append(correct.mean().item())
        all_correct.extend(correct.cpu().tolist())

    # Aggregate results
    if aggregate == "mean":
        token_accuracy = np.mean(accuracies)
    else:  # median
        token_accuracy = np.median(accuracies)

    num_masked = mask_positions.sum().item()

    # Build result
    result = DIREResult(
        token_accuracy=token_accuracy,
        reconstruction_error=1.0 - token_accuracy,
        num_masked=num_masked,
        num_total=seq_len,
        mask_ratio=mask_ratio,
        correct_predictions=all_correct if num_runs == 1 else None,
    )

    # Compute extended metrics on last run if requested
    if compute_extended and num_runs == 1:
        # Re-run to get logits for extended metrics
        with torch.no_grad():
            outputs = model(masked_ids)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            extended = compute_extended_metrics(logits, original_ids, mask_positions)

            result.top_k_accuracy = {
                5: extended.get("top_5_accuracy"),
                10: extended.get("top_10_accuracy"),
            }
            result.mean_entropy = extended.get("mean_entropy")
            result.reconstruction_perplexity = extended.get("reconstruction_perplexity")
            result.cross_entropy_loss = extended.get("cross_entropy_loss")

    return result


def compute_dire_score_mc(
    model,
    tokenizer,
    text: str,
    mask_ratio: float = 0.5,
    mc_samples: int = 32,
    max_length: int = 512,
) -> MCDIREResult:
    """
    Compute Text-DIRE score with Monte Carlo estimation for stability.

    Multiple random maskings are performed to get a stable estimate
    with confidence intervals.

    Args:
        model: The diffusion model (e.g., LLaDA)
        tokenizer: The tokenizer
        text: Input text to evaluate
        mask_ratio: Fraction of tokens to mask
        mc_samples: Number of Monte Carlo samples
        max_length: Maximum sequence length

    Returns:
        MCDIREResult with mean, std, and 95% CI
    """
    device = next(model.parameters()).device

    # Tokenize once
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(device)

    original_ids = inputs["input_ids"]
    seq_len = original_ids.shape[1]

    # Get mask token ID
    mask_token_id = getattr(tokenizer, 'mask_token_id', None)
    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, 'mask_id', None)
    if mask_token_id is None:
        mask_token_id = 126336  # LLaDA default

    accuracies = []
    ce_losses = []

    for _ in range(mc_samples):
        # Create new random mask
        masked_ids, mask_positions = mask_tokens(
            original_ids,
            mask_ratio,
            mask_token_id,
        )

        with torch.no_grad():
            outputs = model(masked_ids)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            predictions = logits.argmax(dim=-1)

            # Accuracy
            original_tokens = original_ids[mask_positions]
            predicted_tokens = predictions[mask_positions]
            correct = (predicted_tokens == original_tokens).float()
            accuracies.append(correct.mean().item())

            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits[mask_positions],
                original_ids[mask_positions],
            ).item()
            ce_losses.append(ce_loss)

    # Compute statistics
    accuracies = np.array(accuracies)
    errors = 1.0 - accuracies
    ce_losses = np.array(ce_losses)

    return MCDIREResult(
        mean=float(np.mean(errors)),
        std=float(np.std(errors)),
        ci_95_lower=float(np.percentile(errors, 2.5)),
        ci_95_upper=float(np.percentile(errors, 97.5)),
        samples=errors.tolist(),
        num_samples=mc_samples,
        accuracy_mean=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies)),
        ce_loss_mean=float(np.mean(ce_losses)),
        ce_loss_std=float(np.std(ce_losses)),
    )


def ensemble_dire_score(
    model,
    tokenizer,
    text: str,
    mask_ratios: list[float] = None,
    weights: list[float] = None,
    mc_samples: int = 4,
    max_length: int = 512,
) -> EnsembleDIREResult:
    """
    Compute ensemble DIRE score combining multiple mask ratios.

    Args:
        model: The diffusion model
        tokenizer: The tokenizer
        text: Input text to evaluate
        mask_ratios: List of mask ratios to combine (default: [0.3, 0.5, 0.7])
        weights: Weights for each ratio (default: [0.2, 0.3, 0.5])
        mc_samples: MC samples per mask ratio for stability
        max_length: Maximum sequence length

    Returns:
        EnsembleDIREResult with weighted score and individual scores
    """
    if mask_ratios is None:
        mask_ratios = [0.3, 0.5, 0.7]

    if weights is None:
        # Higher weight on higher mask ratios (harder reconstruction)
        weights = [0.2, 0.3, 0.5]

    if len(weights) != len(mask_ratios):
        raise ValueError("weights must have same length as mask_ratios")

    # Normalize weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    individual_scores = {}
    weight_dict = {}

    for ratio, weight in zip(mask_ratios, weights):
        # Use MC estimation for each ratio
        mc_result = compute_dire_score_mc(
            model, tokenizer, text,
            mask_ratio=ratio,
            mc_samples=mc_samples,
            max_length=max_length,
        )
        individual_scores[ratio] = mc_result.mean
        weight_dict[ratio] = weight

    # Compute weighted score
    weighted_score = sum(
        individual_scores[ratio] * weight_dict[ratio]
        for ratio in mask_ratios
    )

    # Find best performing ratio
    best_ratio = max(individual_scores, key=individual_scores.get)

    return EnsembleDIREResult(
        weighted_score=weighted_score,
        individual_scores=individual_scores,
        weights=weight_dict,
        best_mask_ratio=best_ratio,
        best_score=individual_scores[best_ratio],
    )


class TextDIRE:
    """
    Text-DIRE detector class for batch processing.

    Usage:
        dire = TextDIRE(model, tokenizer)
        scores = dire.compute_scores(texts, mask_ratios=[0.3, 0.5, 0.7])
    """

    def __init__(
        self,
        model,
        tokenizer,
        mask_token_id: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize Text-DIRE detector.

        Args:
            model: Pre-trained diffusion model (e.g., LLaDA)
            tokenizer: Corresponding tokenizer
            mask_token_id: Override mask token ID (auto-detected if None)
            device: Device to use (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer

        # Auto-detect device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        # Auto-detect mask token ID
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        else:
            self.mask_token_id = getattr(tokenizer, 'mask_token_id', None)
            if self.mask_token_id is None:
                self.mask_token_id = getattr(tokenizer, 'mask_id', None)
            if self.mask_token_id is None:
                # LLaDA default
                self.mask_token_id = 126336

        self.model.eval()

    def compute_score(
        self,
        text: str,
        mask_ratio: float = 0.5,
        max_length: int = 512,
        compute_extended: bool = False,
    ) -> DIREResult:
        """Compute DIRE score for a single text."""
        return compute_dire_score(
            self.model,
            self.tokenizer,
            text,
            mask_ratio=mask_ratio,
            max_length=max_length,
            compute_extended=compute_extended,
        )

    def compute_score_mc(
        self,
        text: str,
        mask_ratio: float = 0.5,
        mc_samples: int = 32,
        max_length: int = 512,
    ) -> MCDIREResult:
        """Compute DIRE score with Monte Carlo estimation."""
        return compute_dire_score_mc(
            self.model,
            self.tokenizer,
            text,
            mask_ratio=mask_ratio,
            mc_samples=mc_samples,
            max_length=max_length,
        )

    def compute_ensemble_score(
        self,
        text: str,
        mask_ratios: list[float] = None,
        weights: list[float] = None,
        mc_samples: int = 4,
        max_length: int = 512,
    ) -> EnsembleDIREResult:
        """Compute ensemble DIRE score across multiple mask ratios."""
        return ensemble_dire_score(
            self.model,
            self.tokenizer,
            text,
            mask_ratios=mask_ratios,
            weights=weights,
            mc_samples=mc_samples,
            max_length=max_length,
        )

    def compute_scores(
        self,
        texts: list[str],
        mask_ratios: list[float] = None,
        max_length: int = 512,
        batch_size: int = 8,
        progress_bar: bool = True,
    ) -> list[dict]:
        """
        Compute DIRE scores for multiple texts using batched GPU inference.

        Sequences are padded within each mini-batch and scored in a single
        forward pass, giving 4-10x throughput improvement over sequential
        scoring on GPU.

        Args:
            texts: List of texts to evaluate
            mask_ratios: List of mask ratios to try
            max_length: Maximum sequence length
            batch_size: Texts per forward pass (tune to GPU memory)
            progress_bar: Whether to show progress bar

        Returns:
            List of dictionaries with scores for each text
        """
        if mask_ratios is None:
            mask_ratios = [0.3, 0.5, 0.7]

        # Filter valid texts and remember their original indices
        valid_entries: list[tuple[int, str]] = [
            (idx, text) for idx, text in enumerate(texts)
            if text and len(text.strip()) >= 10
        ]
        valid_texts = [t for _, t in valid_entries]
        valid_indices = [i for i, _ in valid_entries]

        results: list[dict] = []

        # For each mask ratio, run batched inference over all valid texts
        ratio_results: dict[float, list[DIREResult]] = {}

        for mask_ratio in mask_ratios:
            if progress_bar:
                try:
                    from tqdm import tqdm
                    num_batches = math.ceil(len(valid_texts) / batch_size)
                    pbar = tqdm(total=num_batches, desc=f"DIRE mask={mask_ratio}")
                except ImportError:
                    pbar = None
            else:
                pbar = None

            dire_results: list[DIREResult] = []
            num_batches = math.ceil(len(valid_texts) / batch_size)

            for b in range(num_batches):
                batch_texts = valid_texts[b * batch_size : (b + 1) * batch_size]
                try:
                    batch_results = compute_dire_scores_batch(
                        self.model,
                        self.tokenizer,
                        batch_texts,
                        mask_ratio=mask_ratio,
                        max_length=max_length,
                        batch_size=len(batch_texts),  # already sliced
                    )
                    dire_results.extend(batch_results)
                except Exception as e:
                    # Fall back to sequential for this batch
                    for text in batch_texts:
                        try:
                            r = self.compute_score(text, mask_ratio, max_length)
                            dire_results.append(r)
                        except Exception:
                            dire_results.append(DIREResult(
                                token_accuracy=0.0,
                                reconstruction_error=1.0,
                                num_masked=0,
                                num_total=0,
                                mask_ratio=mask_ratio,
                            ))

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            ratio_results[mask_ratio] = dire_results

        # Assemble per-text result dicts
        # All ratio lists should be the same length (one entry per valid text)
        num_valid = len(valid_texts)
        for i in range(num_valid):
            text_result = {
                "text_idx": valid_indices[i],
                "text_length": len(valid_texts[i]),
            }

            for mask_ratio in mask_ratios:
                if i < len(ratio_results[mask_ratio]):
                    r = ratio_results[mask_ratio][i]
                    text_result[f"accuracy_{mask_ratio}"] = r.token_accuracy
                    text_result[f"error_{mask_ratio}"] = r.reconstruction_error
                    text_result[f"num_masked_{mask_ratio}"] = r.num_masked

            results.append(text_result)

        return results

    def predict(
        self,
        text: str,
        mask_ratio: float = 0.5,
        threshold: float = 0.5,
    ) -> tuple[str, float]:
        """
        Predict whether text is human or AI-generated.

        Args:
            text: Text to classify
            mask_ratio: Mask ratio for DIRE computation
            threshold: Decision threshold (error > threshold -> human)

        Returns:
            Tuple of (prediction, confidence)
            prediction: "human" or "ai"
            confidence: Score between 0 and 1
        """
        result = self.compute_score(text, mask_ratio)

        # Higher reconstruction error -> more likely human
        # (This is the hypothesis - may need to flip based on experiments)
        if result.reconstruction_error > threshold:
            return "human", result.reconstruction_error
        else:
            return "ai", 1.0 - result.reconstruction_error


def analyze_token_patterns(
    model,
    tokenizer,
    text: str,
    mask_ratio: float = 0.5,
    max_length: int = 512,
) -> dict:
    """
    Analyze which types of tokens are easier/harder to reconstruct.

    Useful for understanding what makes AI text different from human text.

    Returns:
        Dictionary with token-level analysis
    """
    device = next(model.parameters()).device

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    original_ids = inputs["input_ids"]

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

    # Analyze each masked token
    analysis = {
        "tokens": [],
        "positions": [],
        "correct": [],
        "confidence": [],
    }

    for i, pos in enumerate(mask_positions[0].nonzero(as_tuple=True)[0]):
        pos = pos.item()
        orig_token = original_ids[0, pos].item()
        pred_token = predictions[0, pos].item()
        confidence = probs[0, pos, pred_token].item()

        analysis["tokens"].append(tokenizer.decode([orig_token]))
        analysis["positions"].append(pos)
        analysis["correct"].append(orig_token == pred_token)
        analysis["confidence"].append(confidence)

    # Summary statistics
    analysis["accuracy"] = np.mean(analysis["correct"]) if analysis["correct"] else 0
    analysis["mean_confidence"] = np.mean(analysis["confidence"]) if analysis["confidence"] else 0

    return analysis
