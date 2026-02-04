"""
Tests for DIRE computation module.
"""

import pytest
import torch
import numpy as np


class TestMaskTokens:
    """Tests for mask_tokens function."""

    def test_mask_tokens_basic(self):
        """Test basic masking functionality."""
        from src.dire import mask_tokens

        # Create simple input
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        mask_ratio = 0.5
        mask_token_id = 0

        masked_ids, mask_positions = mask_tokens(
            input_ids, mask_ratio, mask_token_id
        )

        # Check shapes
        assert masked_ids.shape == input_ids.shape
        assert mask_positions.shape == input_ids.shape

        # Check that some positions are masked
        assert mask_positions.sum() > 0

        # Check that masked positions have mask token
        assert (masked_ids[mask_positions] == mask_token_id).all()

    def test_mask_tokens_excludes_boundaries(self):
        """Test that first and last tokens are not masked."""
        from src.dire import mask_tokens

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        # Run multiple times to check consistency
        for _ in range(10):
            _, mask_positions = mask_tokens(
                input_ids, 0.5, 0,
                exclude_first=1, exclude_last=1
            )

            # First and last should never be masked
            assert not mask_positions[0, 0]
            assert not mask_positions[0, -1]

    def test_mask_tokens_ratio(self):
        """Test that mask ratio is approximately correct."""
        from src.dire import mask_tokens

        input_ids = torch.tensor([[1] * 100])
        mask_ratio = 0.3

        _, mask_positions = mask_tokens(input_ids, mask_ratio, 0)

        # Expected: ~30% of 98 valid positions (excluding first/last)
        expected = int(98 * mask_ratio)
        actual = mask_positions.sum().item()

        # Allow some tolerance
        assert abs(actual - expected) <= 5

    def test_mask_tokens_short_sequence(self):
        """Test handling of short sequences."""
        from src.dire import mask_tokens

        input_ids = torch.tensor([[1, 2, 3]])

        # Should still work with short sequences
        masked_ids, mask_positions = mask_tokens(input_ids, 0.5, 0)

        assert masked_ids.shape == input_ids.shape


class TestDIREResult:
    """Tests for DIREResult dataclass."""

    def test_dire_result_creation(self):
        """Test DIREResult creation."""
        from src.dire import DIREResult

        result = DIREResult(
            token_accuracy=0.8,
            reconstruction_error=0.2,
            num_masked=10,
            num_total=50,
            mask_ratio=0.5,
        )

        assert result.token_accuracy == 0.8
        assert result.reconstruction_error == 0.2
        assert result.num_masked == 10

    def test_dire_result_with_optional_fields(self):
        """Test DIREResult with optional fields."""
        from src.dire import DIREResult

        result = DIREResult(
            token_accuracy=0.8,
            reconstruction_error=0.2,
            num_masked=10,
            num_total=50,
            mask_ratio=0.5,
            correct_predictions=[True, False, True],
            top_k_accuracy={5: 0.9, 10: 0.95},
        )

        assert result.correct_predictions == [True, False, True]
        assert result.top_k_accuracy[5] == 0.9


class TestMCDIREResult:
    """Tests for MCDIREResult dataclass."""

    def test_mc_dire_result(self):
        """Test MCDIREResult creation."""
        from src.dire import MCDIREResult

        result = MCDIREResult(
            mean=0.3,
            std=0.05,
            ci_95_lower=0.2,
            ci_95_upper=0.4,
            samples=[0.25, 0.30, 0.35],
            num_samples=3,
        )

        assert result.mean == 0.3
        assert result.std == 0.05
        assert result.num_samples == 3


class TestEnsembleDIREResult:
    """Tests for EnsembleDIREResult dataclass."""

    def test_ensemble_dire_result(self):
        """Test EnsembleDIREResult creation."""
        from src.dire import EnsembleDIREResult

        result = EnsembleDIREResult(
            weighted_score=0.35,
            individual_scores={0.3: 0.25, 0.5: 0.35, 0.7: 0.40},
            weights={0.3: 0.2, 0.5: 0.3, 0.7: 0.5},
            best_mask_ratio=0.7,
            best_score=0.40,
        )

        assert result.weighted_score == 0.35
        assert result.best_mask_ratio == 0.7


class TestExtendedMetrics:
    """Tests for extended metrics computation."""

    def test_compute_extended_metrics(self):
        """Test extended metrics calculation."""
        from src.dire import compute_extended_metrics

        # Create mock data
        vocab_size = 100
        seq_len = 10
        batch_size = 1

        logits = torch.randn(batch_size, seq_len, vocab_size)
        original_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_positions[0, 3:6] = True

        metrics = compute_extended_metrics(logits, original_ids, mask_positions)

        # Check that all expected metrics are present
        assert "top_5_accuracy" in metrics
        assert "top_10_accuracy" in metrics
        assert "mean_entropy" in metrics
        assert "cross_entropy_loss" in metrics
        assert "reconstruction_perplexity" in metrics

        # Check value ranges
        assert 0 <= metrics["top_5_accuracy"] <= 1
        assert 0 <= metrics["top_10_accuracy"] <= 1
        assert metrics["mean_entropy"] >= 0
        assert metrics["reconstruction_perplexity"] >= 1


class TestTextDIREClass:
    """Tests for TextDIRE class (requires mock model)."""

    def test_text_dire_initialization(self):
        """Test TextDIRE initialization with mock model."""
        from src.dire import TextDIRE

        # Create a mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(10, 10)

            def forward(self, x):
                # Return mock logits
                batch, seq = x.shape
                return type('obj', (object,), {
                    'logits': torch.randn(batch, seq, 1000)
                })()

        class MockTokenizer:
            def __init__(self):
                self.mask_token_id = 0

            def __call__(self, text, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
                }

            def decode(self, ids):
                return "test"

        model = MockModel()
        tokenizer = MockTokenizer()

        dire = TextDIRE(model, tokenizer)

        assert dire.mask_token_id == 0
        assert dire.model is not None


# Integration tests (skipped by default, require GPU)
@pytest.mark.skip(reason="Requires GPU and LLaDA model")
class TestDIREIntegration:
    """Integration tests requiring actual model."""

    def test_compute_dire_score_real(self):
        """Test DIRE score computation with real model."""
        from src.dire import compute_dire_score
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Base",
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Base",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()

        text = "The quick brown fox jumps over the lazy dog."

        result = compute_dire_score(model, tokenizer, text, mask_ratio=0.5)

        assert 0 <= result.token_accuracy <= 1
        assert result.reconstruction_error == 1 - result.token_accuracy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
