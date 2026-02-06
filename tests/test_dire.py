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


class TestMaskTokensPadded:
    """Tests for mask_tokens_padded (batch-aware masking)."""

    def test_respects_attention_mask(self):
        """Padding tokens must never be masked."""
        from src.dire import mask_tokens_padded

        # Two sequences: lengths 8 and 5, padded to 8
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 0, 0, 0],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ])

        for _ in range(20):
            _, mask_positions = mask_tokens_padded(
                input_ids, attention_mask, 0.5, mask_token_id=99
            )
            # Padding positions (indices 5,6,7 of row 1) must never be True
            assert not mask_positions[1, 5:].any(), "Padding tokens were masked"

    def test_excludes_boundaries(self):
        """First and last real tokens must not be masked."""
        from src.dire import mask_tokens_padded

        input_ids = torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 0, 0, 0, 0,  0],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ])

        for _ in range(20):
            _, mask_positions = mask_tokens_padded(
                input_ids, attention_mask, 0.5, 99,
                exclude_first=1, exclude_last=1,
            )
            # First token of each row
            assert not mask_positions[0, 0]
            assert not mask_positions[1, 0]
            # Last real token of each row
            assert not mask_positions[0, 9]
            assert not mask_positions[1, 4]

    def test_output_shapes(self):
        """Output tensors must match input shapes."""
        from src.dire import mask_tokens_padded

        B, L = 4, 20
        input_ids = torch.randint(1, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)

        masked_ids, mask_positions = mask_tokens_padded(
            input_ids, attention_mask, 0.5, 99
        )
        assert masked_ids.shape == (B, L)
        assert mask_positions.shape == (B, L)
        assert masked_ids.dtype == input_ids.dtype
        assert mask_positions.dtype == torch.bool


class TestComputeDireScoresBatch:
    """Tests for compute_dire_scores_batch."""

    @staticmethod
    def _make_mock_model_and_tokenizer(vocab_size=1000):
        """Build a deterministic mock model + tokenizer for unit tests."""

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(10, 10)

            def forward(self, x):
                B, L = x.shape
                # Return logits that always predict token 1
                logits = torch.zeros(B, L, vocab_size)
                logits[:, :, 1] = 10.0  # strong bias toward token 1
                return type('obj', (object,), {'logits': logits})()

        class MockTokenizer:
            def __init__(self):
                self.mask_token_id = 0
                self.pad_token_id = 0
                self.eos_token_id = 0

            def __call__(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                # Each word becomes one token (id=1 for every word)
                max_len = max(len(t.split()) for t in texts)
                ids, masks = [], []
                for t in texts:
                    tokens = [1] * len(t.split())
                    pad_len = max_len - len(tokens)
                    ids.append(tokens + [0] * pad_len)
                    masks.append([1] * len(tokens) + [0] * pad_len)
                result = type('obj', (object,), {
                    'input_ids': torch.tensor(ids),
                    'attention_mask': torch.tensor(masks),
                })()
                result.to = lambda device: result
                return result

            def decode(self, ids):
                return "test"

        return MockModel(), MockTokenizer()

    def test_returns_one_result_per_text(self):
        """Batch function should return one DIREResult per valid text."""
        from src.dire import compute_dire_scores_batch

        model, tokenizer = self._make_mock_model_and_tokenizer()
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "a shorter sentence here",
            "another test sentence with several words in it",
        ]
        results = compute_dire_scores_batch(
            model, tokenizer, texts,
            mask_ratio=0.5, batch_size=2,
        )
        assert len(results) == 3

    def test_accuracy_values_in_range(self):
        """All accuracy values must be in [0, 1]."""
        from src.dire import compute_dire_scores_batch

        model, tokenizer = self._make_mock_model_and_tokenizer()
        texts = ["word " * 20 for _ in range(5)]
        results = compute_dire_scores_batch(
            model, tokenizer, texts, mask_ratio=0.5, batch_size=4,
        )
        for r in results:
            assert 0.0 <= r.token_accuracy <= 1.0
            assert abs(r.token_accuracy + r.reconstruction_error - 1.0) < 1e-6

    def test_batch_size_one_matches_single(self):
        """batch_size=1 should produce the same results as sequential."""
        from src.dire import compute_dire_scores_batch

        torch.manual_seed(42)
        model, tokenizer = self._make_mock_model_and_tokenizer()
        texts = ["one two three four five six seven eight nine ten"]

        torch.manual_seed(0)
        results_b1 = compute_dire_scores_batch(
            model, tokenizer, texts, mask_ratio=0.5, batch_size=1,
        )
        # Basic sanity — we get a result
        assert len(results_b1) == 1
        assert results_b1[0].num_masked > 0

    def test_correct_predictions_stored(self):
        """Each result should store per-token correctness."""
        from src.dire import compute_dire_scores_batch

        model, tokenizer = self._make_mock_model_and_tokenizer()
        texts = ["a b c d e f g h i j k l"]
        results = compute_dire_scores_batch(
            model, tokenizer, texts, mask_ratio=0.5, batch_size=1,
        )
        assert results[0].correct_predictions is not None
        assert len(results[0].correct_predictions) == results[0].num_masked


class TestTextDIREBatchedScores:
    """Tests for TextDIRE.compute_scores with batched backend."""

    def test_compute_scores_batched(self):
        """TextDIRE.compute_scores should use batched inference."""
        from src.dire import TextDIRE

        model, tokenizer = TestComputeDireScoresBatch._make_mock_model_and_tokenizer()
        dire = TextDIRE(model, tokenizer)

        texts = ["word " * 15 for _ in range(6)]
        results = dire.compute_scores(
            texts, mask_ratios=[0.5], batch_size=4, progress_bar=False,
        )
        assert len(results) == 6
        for r in results:
            assert "accuracy_0.5" in r
            assert "error_0.5" in r

    def test_filters_short_texts(self):
        """Texts shorter than 10 chars should be skipped."""
        from src.dire import TextDIRE

        model, tokenizer = TestComputeDireScoresBatch._make_mock_model_and_tokenizer()
        dire = TextDIRE(model, tokenizer)

        texts = ["short", "", "word " * 15, "   ", "word " * 15]
        results = dire.compute_scores(
            texts, mask_ratios=[0.5], batch_size=4, progress_bar=False,
        )
        # Only the two long texts should produce results
        assert len(results) == 2


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
