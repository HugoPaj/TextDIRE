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
        progress_bar: bool = True,
    ) -> list[dict]:
        """
        Compute DIRE scores for multiple texts.

        Args:
            texts: List of texts to evaluate
            mask_ratios: List of mask ratios to try
            max_length: Maximum sequence length
            progress_bar: Whether to show progress bar

        Returns:
            List of dictionaries with scores for each text
        """
        if mask_ratios is None:
            mask_ratios = [0.3, 0.5, 0.7]

        results = []

        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(enumerate(texts), total=len(texts), desc="Computing DIRE")
            except ImportError:
                iterator = enumerate(texts)
        else:
            iterator = enumerate(texts)

        for idx, text in iterator:
            if not text or len(text.strip()) < 10:
                continue

            text_result = {"text_idx": idx, "text_length": len(text)}

            try:
                for mask_ratio in mask_ratios:
                    result = self.compute_score(text, mask_ratio, max_length)
                    text_result[f"accuracy_{mask_ratio}"] = result.token_accuracy
                    text_result[f"error_{mask_ratio}"] = result.reconstruction_error
                    text_result[f"num_masked_{mask_ratio}"] = result.num_masked

                results.append(text_result)

            except Exception as e:
                print(f"Error processing text {idx}: {e}")
                continue

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
