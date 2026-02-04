"""
Tests for data loading module.
"""

import pytest
import numpy as np


class TestTextDataset:
    """Tests for TextDataset class."""

    def test_text_dataset_creation(self):
        """Test basic TextDataset creation."""
        from src.data import TextDataset

        dataset = TextDataset(
            texts=["Hello", "World", "Test"],
            labels=[0, 1, 0],
            sources=["human", "ai", "human"],
        )

        assert len(dataset) == 3
        assert dataset.texts[0] == "Hello"

    def test_get_human_texts(self):
        """Test filtering human texts."""
        from src.data import TextDataset

        dataset = TextDataset(
            texts=["Human1", "AI1", "Human2", "AI2"],
            labels=[0, 1, 0, 1],
            sources=["h", "a", "h", "a"],
        )

        human = dataset.get_human_texts()
        assert len(human) == 2
        assert "Human1" in human
        assert "Human2" in human

    def test_get_ai_texts(self):
        """Test filtering AI texts."""
        from src.data import TextDataset

        dataset = TextDataset(
            texts=["Human1", "AI1", "Human2", "AI2"],
            labels=[0, 1, 0, 1],
            sources=["h", "a", "h", "a"],
        )

        ai = dataset.get_ai_texts()
        assert len(ai) == 2
        assert "AI1" in ai
        assert "AI2" in ai

    def test_get_texts_by_source(self):
        """Test filtering by source."""
        from src.data import TextDataset

        dataset = TextDataset(
            texts=["t1", "t2", "t3", "t4"],
            labels=[0, 1, 0, 1],
            sources=["wiki", "hc3", "wiki", "gpt"],
        )

        wiki = dataset.get_texts_by_source("wiki")
        assert len(wiki) == 2

    def test_filter_by_length(self):
        """Test length-based filtering."""
        from src.data import TextDataset

        dataset = TextDataset(
            texts=["short", "this is a medium length text", "this is a very long text with many more words"],
            labels=[0, 0, 1],
            sources=["a", "b", "c"],
        )

        filtered = dataset.filter_by_length(min_words=3, max_words=10)
        assert len(filtered) == 2

    def test_split_basic(self):
        """Test train/val/test splitting."""
        from src.data import TextDataset

        # Create dataset with enough samples
        n = 100
        dataset = TextDataset(
            texts=[f"text_{i}" for i in range(n)],
            labels=[i % 2 for i in range(n)],
            sources=["src"] * n,
        )

        train, val, test = dataset.split(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            stratify=False,
        )

        # Check approximate sizes
        assert 55 <= len(train) <= 65
        assert 15 <= len(val) <= 25
        assert 15 <= len(test) <= 25

        # Check no overlap
        train_texts = set(train.texts)
        val_texts = set(val.texts)
        test_texts = set(test.texts)

        assert len(train_texts & val_texts) == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts & test_texts) == 0

    def test_to_dict_and_from_dict(self):
        """Test serialization."""
        from src.data import TextDataset

        original = TextDataset(
            texts=["a", "b"],
            labels=[0, 1],
            sources=["x", "y"],
            metadata={"test": 123},
        )

        d = original.to_dict()
        restored = TextDataset.from_dict(d)

        assert restored.texts == original.texts
        assert restored.labels == original.labels
        assert restored.sources == original.sources
        assert restored.metadata == original.metadata


class TestDatasetStats:
    """Tests for dataset statistics."""

    def test_get_dataset_stats(self):
        """Test computing dataset statistics."""
        from src.data import get_dataset_stats

        texts = [
            "one two three four five",
            "six seven eight nine ten eleven",
            "twelve thirteen",
        ]

        stats = get_dataset_stats(texts)

        assert stats["num_samples"] == 3
        assert stats["min_words"] == 2
        assert stats["max_words"] == 6
        assert 3 < stats["mean_words"] < 5


class TestAvailableSources:
    """Tests for source listings."""

    def test_get_available_human_sources(self):
        """Test listing human sources."""
        from src.data import get_available_human_sources

        sources = get_available_human_sources()

        assert "wikitext" in sources
        assert "reddit" in sources
        assert "cnn" in sources
        assert "arxiv" in sources

    def test_get_available_benchmarks(self):
        """Test listing benchmarks."""
        from src.data import get_available_benchmarks

        benchmarks = get_available_benchmarks()

        assert "raid" in benchmarks
        assert "mage" in benchmarks
        assert "m4gt" in benchmarks


# Integration tests (require network access)
@pytest.mark.skip(reason="Requires network access")
class TestDataLoadingIntegration:
    """Integration tests for data loading."""

    def test_load_human_texts_wikitext(self):
        """Test loading WikiText."""
        from src.data import load_human_texts

        texts = load_human_texts("wikitext", num_samples=10)

        assert len(texts) == 10
        assert all(len(t.split()) >= 50 for t in texts)

    def test_load_ai_texts_hc3(self):
        """Test loading HC3."""
        from src.data import load_ai_texts

        texts = load_ai_texts("hc3", num_samples=10)

        assert len(texts) == 10

    def test_load_datasets_balanced(self):
        """Test loading balanced dataset."""
        from src.data import load_datasets

        dataset = load_datasets(num_samples=10)

        human_count = dataset.labels.count(0)
        ai_count = dataset.labels.count(1)

        assert human_count == ai_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
