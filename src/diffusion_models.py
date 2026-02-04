"""
Multi-Diffusion Model Support for Text-DIRE.

Provides a unified interface for loading and using different text diffusion models:
- LLaDA-8B (GSAI-ML/LLaDA-8B-Base)
- MDLM (kuleshov-group/mdlm-owt)
- BD3-LM (kuleshov-group/bd3lms)

This allows testing DIRE across multiple diffusion architectures to show
generalization of the detection method.
"""

import torch
from typing import Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class DiffusionModelConfig:
    """Configuration for a diffusion model."""
    name: str
    model_id: str
    mask_token_id: int
    requires_transformers_version: Optional[str] = None
    max_sequence_length: int = 512
    supports_bf16: bool = True
    model_type: str = "masked"  # "masked", "continuous", "discrete"


# Model configurations
MODEL_CONFIGS = {
    "llada": DiffusionModelConfig(
        name="LLaDA-8B",
        model_id="GSAI-ML/LLaDA-8B-Base",
        mask_token_id=126336,
        requires_transformers_version="4.38.2",
        max_sequence_length=2048,
        supports_bf16=True,
        model_type="masked",
    ),
    "mdlm": DiffusionModelConfig(
        name="MDLM",
        model_id="kuleshov-group/mdlm-owt",
        mask_token_id=50257,  # GPT-2 tokenizer mask (may need adjustment)
        max_sequence_length=1024,
        supports_bf16=True,
        model_type="masked",
    ),
    "bd3lm": DiffusionModelConfig(
        name="BD3-LM",
        model_id="kuleshov-group/bd3lms",
        mask_token_id=50257,
        max_sequence_length=1024,
        supports_bf16=True,
        model_type="discrete",
    ),
}


class TextDiffusionModel(ABC):
    """Abstract base class for text diffusion models."""

    def __init__(self, config: DiffusionModelConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self, cache_dir: Optional[str] = None):
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        pass

    def mask_tokens(
        self,
        input_ids: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tokens for DIRE computation.

        Args:
            input_ids: Original token IDs [batch_size, seq_len]
            mask_ratio: Fraction of tokens to mask

        Returns:
            Tuple of (masked_ids, mask_positions)
        """
        batch_size, seq_len = input_ids.shape

        # Calculate number of tokens to mask (exclude first/last)
        valid_start = 1
        valid_end = seq_len - 1
        valid_len = valid_end - valid_start

        if valid_len <= 0:
            raise ValueError("Sequence too short for masking")

        num_mask = max(1, int(valid_len * mask_ratio))

        # Create mask
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)

        for i in range(batch_size):
            positions = torch.randperm(valid_len, device=input_ids.device)[:num_mask] + valid_start
            mask_positions[i, positions] = True

        # Apply mask
        masked_ids = input_ids.clone()
        masked_ids[mask_positions] = self.config.mask_token_id

        return masked_ids, mask_positions

    def compute_reconstruction_accuracy(
        self,
        text: str,
        mask_ratio: float = 0.5,
        max_length: Optional[int] = None,
    ) -> dict:
        """
        Compute reconstruction accuracy for a text.

        Args:
            text: Input text
            mask_ratio: Fraction of tokens to mask
            max_length: Maximum sequence length

        Returns:
            Dictionary with accuracy metrics
        """
        if max_length is None:
            max_length = self.config.max_sequence_length

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(self.device)

        # Mask
        masked_ids, mask_positions = self.mask_tokens(input_ids, mask_ratio)

        # Forward
        with torch.no_grad():
            logits = self.forward(masked_ids)
            predictions = logits.argmax(dim=-1)

        # Compute accuracy
        original_tokens = input_ids[mask_positions]
        predicted_tokens = predictions[mask_positions]
        correct = (predicted_tokens == original_tokens).float()

        return {
            "accuracy": correct.mean().item(),
            "error": 1.0 - correct.mean().item(),
            "num_masked": mask_positions.sum().item(),
            "num_total": input_ids.shape[1],
        }


class LLaDAModel(TextDiffusionModel):
    """LLaDA-8B diffusion model wrapper."""

    def load(self, cache_dir: Optional[str] = None):
        """Load LLaDA model and tokenizer."""
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading {self.config.name}...")
        print(f"Note: Requires transformers=={self.config.requires_transformers_version}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        dtype = torch.bfloat16 if self.config.supports_bf16 else torch.float32

        self.model = AutoModel.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        ).to(self.device).eval()

        print(f"{self.config.name} loaded successfully")

        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        outputs = self.model(input_ids)
        if hasattr(outputs, 'logits'):
            return outputs.logits
        return outputs


class MDLMModel(TextDiffusionModel):
    """MDLM (Masked Diffusion Language Model) wrapper."""

    def load(self, cache_dir: Optional[str] = None):
        """Load MDLM model and tokenizer."""
        try:
            # MDLM may have custom loading requirements
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            print(f"Loading {self.config.name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

            # Update mask token ID from tokenizer if available
            if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id:
                self.config.mask_token_id = self.tokenizer.mask_token_id

            dtype = torch.bfloat16 if self.config.supports_bf16 else torch.float32

            self.model = AutoModelForMaskedLM.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                cache_dir=cache_dir,
            ).to(self.device).eval()

            print(f"{self.config.name} loaded successfully")

        except Exception as e:
            print(f"Error loading MDLM with AutoModelForMaskedLM: {e}")
            print("Trying AutoModel...")

            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

            self.model = AutoModel.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.config.supports_bf16 else torch.float32,
                cache_dir=cache_dir,
            ).to(self.device).eval()

        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        outputs = self.model(input_ids)
        if hasattr(outputs, 'logits'):
            return outputs.logits
        if hasattr(outputs, 'last_hidden_state'):
            # May need to project to vocab
            raise NotImplementedError("MDLM may require custom projection layer")
        return outputs


class BD3LMModel(TextDiffusionModel):
    """BD3-LM (Block Discrete Denoising Diffusion) wrapper."""

    def load(self, cache_dir: Optional[str] = None):
        """Load BD3-LM model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer

            print(f"Loading {self.config.name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

            if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id:
                self.config.mask_token_id = self.tokenizer.mask_token_id

            dtype = torch.bfloat16 if self.config.supports_bf16 else torch.float32

            self.model = AutoModel.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                cache_dir=cache_dir,
            ).to(self.device).eval()

            print(f"{self.config.name} loaded successfully")

        except Exception as e:
            print(f"Error loading BD3-LM: {e}")
            raise

        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        outputs = self.model(input_ids)
        if hasattr(outputs, 'logits'):
            return outputs.logits
        return outputs


# Model factory
MODEL_CLASSES = {
    "llada": LLaDAModel,
    "mdlm": MDLMModel,
    "bd3lm": BD3LMModel,
}


def get_available_models() -> list[str]:
    """Get list of available diffusion models."""
    return list(MODEL_CONFIGS.keys())


def get_model_config(model_name: str) -> DiffusionModelConfig:
    """Get configuration for a specific model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]


def load_diffusion_model(
    model_name: str,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> TextDiffusionModel:
    """
    Load a text diffusion model by name.

    Args:
        model_name: Model name ("llada", "mdlm", "bd3lm")
        device: Device to use
        cache_dir: Directory to cache model

    Returns:
        Loaded TextDiffusionModel instance
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    model_class = MODEL_CLASSES[model_name]

    model = model_class(config, device)
    model.load(cache_dir)

    return model


class MultiModelDIRE:
    """
    DIRE detector that supports multiple diffusion models.

    Useful for comparing detection performance across different
    diffusion architectures.
    """

    def __init__(
        self,
        model_names: list[str] = None,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize multi-model DIRE detector.

        Args:
            model_names: List of model names to load
            device: Device to use
            cache_dir: Directory to cache models
        """
        if model_names is None:
            model_names = ["llada"]

        self.models = {}
        self.device = device
        self.cache_dir = cache_dir

        for name in model_names:
            print(f"Loading {name}...")
            self.models[name] = load_diffusion_model(name, device, cache_dir)

    def compute_scores(
        self,
        text: str,
        mask_ratio: float = 0.5,
    ) -> dict[str, dict]:
        """
        Compute DIRE scores using all loaded models.

        Args:
            text: Input text
            mask_ratio: Mask ratio for DIRE

        Returns:
            Dictionary mapping model name to score dict
        """
        results = {}

        for name, model in self.models.items():
            try:
                scores = model.compute_reconstruction_accuracy(text, mask_ratio)
                results[name] = scores
            except Exception as e:
                print(f"Error computing score with {name}: {e}")
                results[name] = {"error": str(e)}

        return results

    def compare_models(
        self,
        texts: list[str],
        labels: list[int],
        mask_ratio: float = 0.5,
    ) -> dict[str, dict]:
        """
        Compare model performance on a dataset.

        Args:
            texts: List of texts
            labels: Labels (0=human, 1=AI)
            mask_ratio: Mask ratio for DIRE

        Returns:
            Dictionary with performance metrics per model
        """
        from sklearn.metrics import roc_auc_score

        results = {}

        for name in self.models.keys():
            scores = []
            valid_labels = []

            for text, label in zip(texts, labels):
                try:
                    score = self.models[name].compute_reconstruction_accuracy(
                        text, mask_ratio
                    )
                    scores.append(score["error"])
                    valid_labels.append(label)
                except Exception:
                    continue

            if scores:
                auroc = roc_auc_score(valid_labels, scores)
                if auroc < 0.5:
                    auroc = 1 - auroc

                results[name] = {
                    "auroc": auroc,
                    "num_samples": len(scores),
                    "mean_error_human": sum(
                        s for s, l in zip(scores, valid_labels) if l == 0
                    ) / max(1, valid_labels.count(0)),
                    "mean_error_ai": sum(
                        s for s, l in zip(scores, valid_labels) if l == 1
                    ) / max(1, valid_labels.count(1)),
                }

        return results


if __name__ == "__main__":
    print("Text-DIRE Multi-Model Support")
    print("=" * 50)

    print("\nAvailable diffusion models:")
    for name in get_available_models():
        config = get_model_config(name)
        print(f"  {name}: {config.name}")
        print(f"    Model ID: {config.model_id}")
        print(f"    Mask Token: {config.mask_token_id}")
        print(f"    Type: {config.model_type}")
        print()

    print("To load a model:")
    print("  model = load_diffusion_model('llada')")
    print("  result = model.compute_reconstruction_accuracy(text)")
