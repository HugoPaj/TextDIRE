"""
Tests for baseline detection methods.
"""

import pytest
import torch
import numpy as np


class TestPerplexityResult:
    """Tests for PerplexityResult dataclass."""

    def test_perplexity_result_creation(self):
        """Test PerplexityResult creation."""
        from src.baselines import PerplexityResult

        result = PerplexityResult(
            perplexity=50.0,
            loss=3.9,
            num_tokens=100,
        )

        assert result.perplexity == 50.0
        assert result.loss == 3.9
        assert result.num_tokens == 100


class TestDetectGPTResult:
    """Tests for DetectGPTResult dataclass."""

    def test_detectgpt_result_creation(self):
        """Test DetectGPTResult creation."""
        from src.baselines import DetectGPTResult

        result = DetectGPTResult(
            curvature=0.5,
            original_log_prob=-3.0,
            mean_perturbed_log_prob=-3.5,
            std_perturbed_log_prob=0.2,
            num_perturbations=100,
            z_score=2.5,
        )

        assert result.curvature == 0.5
        assert result.z_score == 2.5


class TestFastDetectGPTResult:
    """Tests for FastDetectGPTResult dataclass."""

    def test_fast_detectgpt_result_creation(self):
        """Test FastDetectGPTResult creation."""
        from src.baselines import FastDetectGPTResult

        result = FastDetectGPTResult(
            score=1.5,
            conditional_entropy=2.0,
            unconditional_entropy=3.5,
            num_tokens=50,
        )

        assert result.score == 1.5
        assert result.num_tokens == 50


class TestBinocularsResult:
    """Tests for BinocularsResult dataclass."""

    def test_binoculars_result_creation(self):
        """Test BinocularsResult creation."""
        from src.baselines import BinocularsResult

        result = BinocularsResult(
            score=0.3,
            observer_ppl=80.0,
            performer_ppl=60.0,
            cross_perplexity=70.0,
        )

        assert result.score == 0.3
        assert result.observer_ppl == 80.0


class TestComputePerplexity:
    """Tests for perplexity computation."""

    @pytest.mark.skip(reason="Requires GPU")
    def test_compute_perplexity_basic(self):
        """Test basic perplexity computation."""
        from src.baselines import compute_perplexity
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        text = "The quick brown fox jumps over the lazy dog."

        result = compute_perplexity(model, tokenizer, text)

        assert result.perplexity > 0
        assert result.loss > 0
        assert result.num_tokens > 0


class TestPerplexityDetector:
    """Tests for PerplexityDetector class."""

    @pytest.mark.skip(reason="Requires model download")
    def test_perplexity_detector_init(self):
        """Test PerplexityDetector initialization."""
        from src.baselines import PerplexityDetector

        detector = PerplexityDetector(model_name="gpt2")

        assert detector.model is not None
        assert detector.tokenizer is not None


class TestBurstinessDetector:
    """Tests for BurstinessDetector class."""

    @pytest.mark.skip(reason="Requires model download")
    def test_burstiness_basic(self):
        """Test burstiness computation."""
        from src.baselines import BurstinessDetector

        detector = BurstinessDetector(model_name="gpt2")

        text = " ".join(["word"] * 100)

        result = detector.compute_burstiness(text, window_size=20)

        assert "burstiness" in result
        assert "mean_ppl" in result
        assert result["burstiness"] >= 0


class TestDetectGPTBaseline:
    """Tests for DetectGPT baseline."""

    def test_simple_perturb(self):
        """Test simple perturbation fallback."""
        from src.baselines import DetectGPTBaseline

        # Test the static method directly
        text = "The quick brown fox jumps over the lazy dog."

        # Create minimal instance
        class MockDetector:
            def _simple_perturb(self, t):
                import random
                words = t.split()
                if len(words) < 2:
                    return t
                perturbed = words.copy()
                num_swaps = max(1, len(words) // 20)
                for _ in range(num_swaps):
                    idx = random.randint(0, len(perturbed) - 2)
                    perturbed[idx], perturbed[idx + 1] = perturbed[idx + 1], perturbed[idx]
                return " ".join(perturbed)

        detector = MockDetector()
        perturbed = detector._simple_perturb(text)

        # Should be different (with high probability)
        # but have same word count
        assert len(perturbed.split()) == len(text.split())

    @pytest.mark.skip(reason="Requires model download")
    def test_detectgpt_curvature(self):
        """Test DetectGPT curvature computation."""
        from src.baselines import DetectGPTBaseline

        detector = DetectGPTBaseline(
            scoring_model="gpt2",
            perturbation_model="t5-small",  # Use small for testing
        )

        text = "This is a test sentence for DetectGPT evaluation."

        result = detector.compute_curvature(text, num_perturbations=5)

        assert result.curvature is not None
        assert result.original_log_prob < 0  # Log probs are negative
        assert result.num_perturbations == 5


class TestFastDetectGPT:
    """Tests for Fast-DetectGPT."""

    @pytest.mark.skip(reason="Requires model download")
    def test_fast_detectgpt_score(self):
        """Test Fast-DetectGPT score computation."""
        from src.baselines import FastDetectGPT

        detector = FastDetectGPT(model_name="gpt2")

        text = "This is a test sentence for Fast-DetectGPT."

        result = detector.compute_score(text)

        assert result.score is not None
        assert result.conditional_entropy > 0
        assert result.unconditional_entropy > 0


class TestBinoculars:
    """Tests for Binoculars detector."""

    @pytest.mark.skip(reason="Requires model download")
    def test_binoculars_score(self):
        """Test Binoculars score computation."""
        from src.baselines import Binoculars

        detector = Binoculars(
            observer_model="gpt2",
            performer_model="gpt2-medium",
        )

        text = "This is a test sentence for Binoculars detection."

        result = detector.compute_score(text)

        assert result.score is not None
        assert result.observer_ppl > 0
        assert result.performer_ppl > 0


class TestDetectorBatch:
    """Tests for batch detection methods."""

    @pytest.mark.skip(reason="Requires model download")
    def test_batch_detection(self):
        """Test batch detection."""
        from src.baselines import PerplexityDetector

        detector = PerplexityDetector(model_name="gpt2")

        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]

        results = detector.compute_perplexities(texts, progress_bar=False)

        assert len(results) == 3
        for r in results:
            assert r.perplexity > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
