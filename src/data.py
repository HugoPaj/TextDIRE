"""
Data loading utilities for Text-DIRE experiments.

Provides functions to load human text from multiple sources (WikiText, Reddit,
CNN/DailyMail, arXiv) and AI-generated text (HC3, API-generated).

Also includes integration with standard benchmarks: RAID, MAGE, M4GT-Bench.
"""

import json
import os
from typing import Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TextDataset:
    """Container for a text classification dataset."""
    texts: list[str]
    labels: list[int]  # 0 = human, 1 = AI
    sources: list[str]  # Source identifier for each text
    metadata: dict = field(default_factory=dict)  # Additional metadata

    def __len__(self):
        return len(self.texts)

    def get_human_texts(self) -> list[str]:
        return [t for t, l in zip(self.texts, self.labels) if l == 0]

    def get_ai_texts(self) -> list[str]:
        return [t for t, l in zip(self.texts, self.labels) if l == 1]

    def get_texts_by_source(self, source: str) -> list[str]:
        """Get texts from a specific source."""
        return [t for t, s in zip(self.texts, self.sources) if s == source]

    def filter_by_length(self, min_words: int = 0, max_words: int = float('inf')) -> 'TextDataset':
        """Filter dataset by word count."""
        filtered_texts = []
        filtered_labels = []
        filtered_sources = []

        for text, label, source in zip(self.texts, self.labels, self.sources):
            word_count = len(text.split())
            if min_words <= word_count <= max_words:
                filtered_texts.append(text)
                filtered_labels.append(label)
                filtered_sources.append(source)

        return TextDataset(
            texts=filtered_texts,
            labels=filtered_labels,
            sources=filtered_sources,
            metadata=self.metadata
        )

    def split(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        stratify: bool = True,
        seed: int = 42,
    ) -> tuple['TextDataset', 'TextDataset', 'TextDataset']:
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            stratify: Whether to stratify by label and source
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train, val, test) TextDatasets
        """
        import numpy as np
        from sklearn.model_selection import train_test_split

        np.random.seed(seed)

        indices = list(range(len(self.texts)))

        if stratify:
            # Create stratification key combining label and source
            strat_key = [f"{l}_{s}" for l, s in zip(self.labels, self.sources)]
        else:
            strat_key = None

        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=[strat_key[i] for i in indices] if strat_key else None,
            random_state=seed,
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            stratify=[strat_key[i] for i in temp_idx] if strat_key else None,
            random_state=seed,
        )

        def make_subset(indices):
            return TextDataset(
                texts=[self.texts[i] for i in indices],
                labels=[self.labels[i] for i in indices],
                sources=[self.sources[i] for i in indices],
                metadata=self.metadata,
            )

        return make_subset(train_idx), make_subset(val_idx), make_subset(test_idx)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "texts": self.texts,
            "labels": self.labels,
            "sources": self.sources,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TextDataset':
        """Create from dictionary."""
        return cls(
            texts=data["texts"],
            labels=data["labels"],
            sources=data["sources"],
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str):
        """Save dataset to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TextDataset':
        """Load dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def load_human_texts(
    source: str = "wikitext",
    num_samples: int = 100,
    min_words: int = 50,
    max_length: Optional[int] = None,
) -> list[str]:
    """
    Load human-written text samples.

    Args:
        source: Data source ("wikitext", "wikitext-2", or "c4")
        num_samples: Number of samples to load
        min_words: Minimum words per sample
        max_length: Maximum character length (None for no limit)

    Returns:
        List of human-written text samples
    """
    from datasets import load_dataset

    texts = []

    if source == "wikitext" or source == "wikitext-103":
        try:
            dataset = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                split="test",
            )

            for item in dataset:
                text = item["text"].strip()

                # Filter out headers and empty lines
                if not text or text.startswith("="):
                    continue

                # Check minimum length
                if len(text.split()) < min_words:
                    continue

                # Check maximum length
                if max_length and len(text) > max_length:
                    text = text[:max_length]

                texts.append(text)

                if len(texts) >= num_samples:
                    break

        except Exception as e:
            print(f"Error loading WikiText-103: {e}")
            # Fallback to WikiText-2
            return load_human_texts("wikitext-2", num_samples, min_words, max_length)

    elif source == "wikitext-2":
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )

        for item in dataset:
            text = item["text"].strip()

            if not text or text.startswith("="):
                continue

            if len(text.split()) < min_words:
                continue

            if max_length and len(text) > max_length:
                text = text[:max_length]

            texts.append(text)

            if len(texts) >= num_samples:
                break

    elif source == "c4":
        dataset = load_dataset(
            "c4",
            "en",
            split="validation",
            streaming=True,
        )

        for item in dataset:
            text = item["text"].strip()

            if len(text.split()) < min_words:
                continue

            if max_length and len(text) > max_length:
                text = text[:max_length]

            texts.append(text)

            if len(texts) >= num_samples:
                break

    elif source == "bookcorpus":
        dataset = load_dataset(
            "bookcorpus",
            split="train",
            streaming=True,
        )

        for item in dataset:
            text = item["text"].strip()

            if len(text.split()) < min_words:
                continue

            if max_length and len(text) > max_length:
                text = text[:max_length]

            texts.append(text)

            if len(texts) >= num_samples:
                break

    elif source == "reddit" or source == "writingprompts":
        # Reddit WritingPrompts - creative writing
        try:
            dataset = load_dataset(
                "euclaise/writingprompts",
                split="train",
                streaming=True,
            )

            for item in dataset:
                # Get the story (response), not the prompt
                text = item.get("story", item.get("text", "")).strip()

                if not text:
                    continue

                # Filter out short responses
                if len(text.split()) < min_words:
                    continue

                if max_length and len(text) > max_length:
                    text = text[:max_length]

                texts.append(text)

                if len(texts) >= num_samples:
                    break

        except Exception as e:
            print(f"Error loading WritingPrompts: {e}")
            # Fallback to alternative reddit dataset
            try:
                dataset = load_dataset(
                    "reddit",
                    split="train",
                    streaming=True,
                )
                for item in dataset:
                    text = item.get("content", item.get("body", "")).strip()
                    if text and len(text.split()) >= min_words:
                        if max_length and len(text) > max_length:
                            text = text[:max_length]
                        texts.append(text)
                        if len(texts) >= num_samples:
                            break
            except Exception:
                print("Fallback reddit dataset also failed")

    elif source == "cnn" or source == "cnn_dailymail":
        # CNN/DailyMail - news articles
        try:
            dataset = load_dataset(
                "cnn_dailymail",
                "3.0.0",
                split="test",
            )

            for item in dataset:
                text = item.get("article", "").strip()

                if not text:
                    continue

                if len(text.split()) < min_words:
                    continue

                if max_length and len(text) > max_length:
                    text = text[:max_length]

                texts.append(text)

                if len(texts) >= num_samples:
                    break

        except Exception as e:
            print(f"Error loading CNN/DailyMail: {e}")

    elif source == "arxiv":
        # arXiv abstracts - academic writing
        try:
            dataset = load_dataset(
                "scientific_papers",
                "arxiv",
                split="test",
                streaming=True,
            )

            for item in dataset:
                # Use the abstract
                text = item.get("abstract", "").strip()

                if not text:
                    continue

                if len(text.split()) < min_words:
                    continue

                if max_length and len(text) > max_length:
                    text = text[:max_length]

                texts.append(text)

                if len(texts) >= num_samples:
                    break

        except Exception as e:
            print(f"Error loading arXiv: {e}, trying alternative...")
            # Fallback to arxiv_dataset
            try:
                dataset = load_dataset(
                    "togethercomputer/RedPajama-Data-1T-Sample",
                    split="train",
                    streaming=True,
                )
                for item in dataset:
                    if item.get("meta", {}).get("source") == "arxiv":
                        text = item.get("text", "").strip()
                        if text and len(text.split()) >= min_words:
                            if max_length and len(text) > max_length:
                                text = text[:max_length]
                            texts.append(text)
                            if len(texts) >= num_samples:
                                break
            except Exception:
                print("Fallback arxiv source also failed")

    elif source == "news" or source == "ag_news":
        # AG News - news articles (shorter)
        try:
            dataset = load_dataset(
                "ag_news",
                split="test",
            )

            for item in dataset:
                text = item.get("text", "").strip()

                if not text or len(text.split()) < min_words:
                    continue

                if max_length and len(text) > max_length:
                    text = text[:max_length]

                texts.append(text)

                if len(texts) >= num_samples:
                    break

        except Exception as e:
            print(f"Error loading AG News: {e}")

    elif source == "imdb":
        # IMDB reviews - mixed sentiment text
        try:
            dataset = load_dataset(
                "imdb",
                split="test",
            )

            for item in dataset:
                text = item.get("text", "").strip()

                if not text or len(text.split()) < min_words:
                    continue

                if max_length and len(text) > max_length:
                    text = text[:max_length]

                texts.append(text)

                if len(texts) >= num_samples:
                    break

        except Exception as e:
            print(f"Error loading IMDB: {e}")

    else:
        raise ValueError(f"Unknown source: {source}. Available: wikitext, wikitext-2, c4, bookcorpus, reddit, cnn, arxiv, news, imdb")

    print(f"Loaded {len(texts)} human text samples from {source}")
    return texts


def load_ai_texts(
    source: str = "hc3",
    num_samples: int = 100,
    min_words: int = 30,
    max_length: Optional[int] = None,
) -> list[str]:
    """
    Load AI-generated text samples.

    Args:
        source: Data source ("hc3", "gpt-wiki-intro", or "aigenerated")
        num_samples: Number of samples to load
        min_words: Minimum words per sample
        max_length: Maximum character length (None for no limit)

    Returns:
        List of AI-generated text samples
    """
    from datasets import load_dataset

    texts = []

    if source == "hc3":
        try:
            # HC3: Human vs ChatGPT dataset
            dataset = load_dataset(
                "Hello-SimpleAI/HC3",
                "all",
                split="train",
                trust_remote_code=True,
            )

            for item in dataset:
                # Get ChatGPT answers
                if not item["chatgpt_answers"]:
                    continue

                for answer in item["chatgpt_answers"]:
                    text = answer.strip()

                    if len(text.split()) < min_words:
                        continue

                    if max_length and len(text) > max_length:
                        text = text[:max_length]

                    texts.append(text)

                    if len(texts) >= num_samples:
                        break

                if len(texts) >= num_samples:
                    break

        except Exception as e:
            print(f"Error loading HC3: {e}")
            # Fallback
            return load_ai_texts("gpt-wiki-intro", num_samples, min_words, max_length)

    elif source == "gpt-wiki-intro":
        try:
            # GPT-generated Wikipedia intros
            dataset = load_dataset(
                "aadityaubhat/GPT-wiki-intro",
                split="train",
            )

            for item in dataset:
                text = item.get("generated_intro", "").strip()

                if not text:
                    continue

                if len(text.split()) < min_words:
                    continue

                if max_length and len(text) > max_length:
                    text = text[:max_length]

                texts.append(text)

                if len(texts) >= num_samples:
                    break

        except Exception as e:
            print(f"Error loading GPT-wiki-intro: {e}")
            raise

    elif source == "aigenerated":
        try:
            # AI-generated news/text dataset
            dataset = load_dataset(
                "NicolaiSivesworker/chatgpt-vs-human",
                split="train",
            )

            for item in dataset:
                if item.get("source", "") != "chatgpt":
                    continue

                text = item.get("text", "").strip()

                if not text or len(text.split()) < min_words:
                    continue

                if max_length and len(text) > max_length:
                    text = text[:max_length]

                texts.append(text)

                if len(texts) >= num_samples:
                    break

        except Exception as e:
            print(f"Error loading aigenerated dataset: {e}")
            raise

    else:
        raise ValueError(f"Unknown source: {source}")

    print(f"Loaded {len(texts)} AI text samples from {source}")
    return texts


def load_datasets(
    num_samples: int = 100,
    human_source: str = "wikitext",
    ai_source: str = "hc3",
    balance: bool = True,
) -> TextDataset:
    """
    Load combined dataset with human and AI texts.

    Args:
        num_samples: Number of samples per class
        human_source: Source for human texts
        ai_source: Source for AI texts
        balance: Whether to balance classes

    Returns:
        TextDataset with combined texts, labels, and sources
    """
    print(f"Loading {num_samples} samples each of human and AI text...")

    human_texts = load_human_texts(human_source, num_samples)
    ai_texts = load_ai_texts(ai_source, num_samples)

    # Balance if requested
    if balance:
        min_samples = min(len(human_texts), len(ai_texts))
        human_texts = human_texts[:min_samples]
        ai_texts = ai_texts[:min_samples]
        print(f"Balanced to {min_samples} samples per class")

    # Combine
    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)
    sources = [human_source] * len(human_texts) + [ai_source] * len(ai_texts)

    return TextDataset(texts=texts, labels=labels, sources=sources)


def load_hc3_paired(num_samples: int = 100) -> dict:
    """
    Load paired human and AI answers to the same questions from HC3.

    This is useful for controlled comparisons where both human and AI
    are answering the same question.

    Returns:
        Dictionary with questions, human_answers, and ai_answers
    """
    from datasets import load_dataset

    dataset = load_dataset("Hello-SimpleAI/HC3", "all", split="train")

    pairs = {
        "questions": [],
        "human_answers": [],
        "ai_answers": [],
    }

    for item in dataset:
        if not item["human_answers"] or not item["chatgpt_answers"]:
            continue

        # Get first answer from each
        human = item["human_answers"][0].strip()
        ai = item["chatgpt_answers"][0].strip()

        # Filter short answers
        if len(human.split()) < 20 or len(ai.split()) < 20:
            continue

        pairs["questions"].append(item["question"])
        pairs["human_answers"].append(human)
        pairs["ai_answers"].append(ai)

        if len(pairs["questions"]) >= num_samples:
            break

    print(f"Loaded {len(pairs['questions'])} paired Q&A samples")
    return pairs


def get_dataset_stats(texts: list[str]) -> dict:
    """
    Compute basic statistics for a text dataset.

    Returns:
        Dictionary with length statistics
    """
    import numpy as np

    word_counts = [len(t.split()) for t in texts]
    char_counts = [len(t) for t in texts]

    return {
        "num_samples": len(texts),
        "mean_words": np.mean(word_counts),
        "std_words": np.std(word_counts),
        "min_words": np.min(word_counts),
        "max_words": np.max(word_counts),
        "mean_chars": np.mean(char_counts),
        "std_chars": np.std(char_counts),
    }


# =============================================================================
# Standard Benchmark Integration
# =============================================================================

def load_raid_benchmark(
    split: str = "test",
    domains: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    attacks: Optional[list[str]] = None,
    num_samples: Optional[int] = None,
) -> TextDataset:
    """
    Load the RAID benchmark dataset (ACL 2024 / COLING 2025).

    RAID is the largest AI text detection benchmark with 6M+ generations,
    11 models, 8 domains, and 11 adversarial attacks.

    Paper: https://arxiv.org/abs/2405.07940
    Leaderboard: https://raid-bench.xyz
    GitHub: https://github.com/liamdugan/raid

    Args:
        split: Dataset split ("train", "test")
        domains: Filter by domains (e.g., ["arxiv", "news", "reddit"])
                 Available: arxiv, books, news, recipes, reddit, reviews, wikipedia, poetry
        models: Filter by generator models
                Available: gpt4, gpt3, chatgpt, cohere, llama, mistral, mpt, etc.
        attacks: Filter by attack types
                 Available: none, paraphrase, perturb_char, perturb_word,
                           homoglyph, number, whitespace, misspelling,
                           upper_lower, article_deletion, alternative_spelling
        num_samples: Limit number of samples (None for all)

    Returns:
        TextDataset with RAID samples
    """
    from datasets import load_dataset

    print(f"Loading RAID benchmark ({split} split)...")

    try:
        # Load RAID from HuggingFace
        dataset = load_dataset("liamdugan/raid", split=split)

        texts = []
        labels = []
        sources = []

        for item in dataset:
            # Filter by domain
            if domains and item.get("domain") not in domains:
                continue

            # Filter by model
            if models and item.get("model") not in models:
                continue

            # Filter by attack type
            if attacks and item.get("attack") not in attacks:
                continue

            text = item.get("text", item.get("generation", "")).strip()
            if not text:
                continue

            # Label: 0 = human, 1 = AI
            label = 0 if item.get("label") == "human" or item.get("model") == "human" else 1

            # Source combines domain, model, and attack info
            source = f"raid_{item.get('domain', 'unknown')}_{item.get('model', 'unknown')}"
            if item.get("attack") and item.get("attack") != "none":
                source += f"_{item.get('attack')}"

            texts.append(text)
            labels.append(label)
            sources.append(source)

            if num_samples and len(texts) >= num_samples:
                break

        print(f"Loaded {len(texts)} samples from RAID benchmark")
        print(f"  Human: {labels.count(0)}, AI: {labels.count(1)}")

        return TextDataset(
            texts=texts,
            labels=labels,
            sources=sources,
            metadata={
                "benchmark": "RAID",
                "split": split,
                "domains": domains,
                "models": models,
                "attacks": attacks,
            }
        )

    except Exception as e:
        print(f"Error loading RAID benchmark: {e}")
        raise


def load_mage_benchmark(
    split: str = "test",
    domains: Optional[list[str]] = None,
    generators: Optional[list[str]] = None,
    num_samples: Optional[int] = None,
) -> TextDataset:
    """
    Load the MAGE benchmark dataset.

    MAGE contains 447.7k samples across 7 domains and 27 generators.

    Paper: Li et al., 2023

    Args:
        split: Dataset split
        domains: Filter by domains
        generators: Filter by generator models
        num_samples: Limit number of samples

    Returns:
        TextDataset with MAGE samples
    """
    from datasets import load_dataset

    print(f"Loading MAGE benchmark ({split} split)...")

    try:
        # Try loading MAGE from HuggingFace
        # Note: Dataset name may vary
        dataset = load_dataset("yaful/MAGE", split=split)

        texts = []
        labels = []
        sources = []

        for item in dataset:
            # Filter by domain if specified
            if domains and item.get("domain") not in domains:
                continue

            # Filter by generator if specified
            if generators and item.get("generator") not in generators:
                continue

            text = item.get("text", "").strip()
            if not text:
                continue

            # Determine label (varies by dataset format)
            if "label" in item:
                label = int(item["label"])
            elif item.get("generator") in ["human", None]:
                label = 0
            else:
                label = 1

            source = f"mage_{item.get('domain', 'unknown')}_{item.get('generator', 'unknown')}"

            texts.append(text)
            labels.append(label)
            sources.append(source)

            if num_samples and len(texts) >= num_samples:
                break

        print(f"Loaded {len(texts)} samples from MAGE benchmark")

        return TextDataset(
            texts=texts,
            labels=labels,
            sources=sources,
            metadata={
                "benchmark": "MAGE",
                "split": split,
            }
        )

    except Exception as e:
        print(f"Error loading MAGE benchmark: {e}")
        print("Falling back to alternative loading method...")

        # Return empty dataset if not available
        return TextDataset(texts=[], labels=[], sources=[], metadata={"error": str(e)})


def load_m4gt_benchmark(
    split: str = "test",
    languages: Optional[list[str]] = None,
    num_samples: Optional[int] = None,
) -> TextDataset:
    """
    Load the M4GT-Bench multilingual benchmark.

    M4GT-Bench supports 8 languages for cross-lingual AI text detection.

    Paper: Wang et al., 2024

    Args:
        split: Dataset split
        languages: Filter by language codes (e.g., ["en", "de", "zh"])
        num_samples: Limit number of samples

    Returns:
        TextDataset with M4GT samples
    """
    from datasets import load_dataset

    print(f"Loading M4GT benchmark ({split} split)...")

    try:
        dataset = load_dataset("m4gt/m4gt-bench", split=split)

        texts = []
        labels = []
        sources = []

        for item in dataset:
            lang = item.get("language", "en")

            if languages and lang not in languages:
                continue

            text = item.get("text", "").strip()
            if not text:
                continue

            label = int(item.get("label", 0))
            source = f"m4gt_{lang}_{item.get('model', 'unknown')}"

            texts.append(text)
            labels.append(label)
            sources.append(source)

            if num_samples and len(texts) >= num_samples:
                break

        print(f"Loaded {len(texts)} samples from M4GT benchmark")

        return TextDataset(
            texts=texts,
            labels=labels,
            sources=sources,
            metadata={
                "benchmark": "M4GT",
                "split": split,
                "languages": languages,
            }
        )

    except Exception as e:
        print(f"Error loading M4GT benchmark: {e}")
        return TextDataset(texts=[], labels=[], sources=[], metadata={"error": str(e)})


# =============================================================================
# Multi-Source Dataset Creation
# =============================================================================

def load_multi_source_human_texts(
    sources: Optional[list[str]] = None,
    samples_per_source: int = 500,
    min_words: int = 50,
    max_length: Optional[int] = None,
) -> TextDataset:
    """
    Load human texts from multiple diverse sources.

    Args:
        sources: List of sources to use. Defaults to all available.
                 Available: wikitext, reddit, cnn, arxiv
        samples_per_source: Number of samples per source
        min_words: Minimum words per sample
        max_length: Maximum character length

    Returns:
        TextDataset with texts from all sources
    """
    if sources is None:
        sources = ["wikitext", "reddit", "cnn", "arxiv"]

    all_texts = []
    all_labels = []
    all_sources = []

    for source in sources:
        print(f"Loading {samples_per_source} samples from {source}...")

        try:
            texts = load_human_texts(
                source=source,
                num_samples=samples_per_source,
                min_words=min_words,
                max_length=max_length,
            )

            all_texts.extend(texts)
            all_labels.extend([0] * len(texts))
            all_sources.extend([source] * len(texts))

        except Exception as e:
            print(f"Warning: Failed to load from {source}: {e}")
            continue

    print(f"Loaded {len(all_texts)} total human text samples")

    return TextDataset(
        texts=all_texts,
        labels=all_labels,
        sources=all_sources,
        metadata={"type": "human", "sources": sources}
    )


def load_multi_source_ai_texts(
    ai_dataset_path: Optional[str] = None,
    models: Optional[list[str]] = None,
    samples_per_model: int = 500,
) -> TextDataset:
    """
    Load AI-generated texts from multiple models.

    Args:
        ai_dataset_path: Path to generated AI text dataset JSON
                        (from ai_text_generator.py)
        models: List of models to include. Defaults to all available.
        samples_per_model: Number of samples per model

    Returns:
        TextDataset with AI texts from all models
    """
    if ai_dataset_path and os.path.exists(ai_dataset_path):
        # Load from generated dataset
        with open(ai_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_texts = []
        all_labels = []
        all_sources = []

        for model, samples in data.items():
            if models and model not in models:
                continue

            for sample in samples[:samples_per_model]:
                text = sample.get("text", "")
                if text:
                    all_texts.append(text)
                    all_labels.append(1)
                    all_sources.append(model)

        return TextDataset(
            texts=all_texts,
            labels=all_labels,
            sources=all_sources,
            metadata={"type": "ai", "models": models or list(data.keys())}
        )

    # Fallback: load from HuggingFace datasets
    print("No AI dataset path provided, loading from HC3...")
    return load_ai_texts_as_dataset("hc3", samples_per_model)


def load_ai_texts_as_dataset(
    source: str,
    num_samples: int,
    min_words: int = 30,
) -> TextDataset:
    """Load AI texts and return as TextDataset."""
    texts = load_ai_texts(source, num_samples, min_words)

    return TextDataset(
        texts=texts,
        labels=[1] * len(texts),
        sources=[source] * len(texts),
        metadata={"type": "ai", "source": source}
    )


def create_combined_dataset(
    human_sources: Optional[list[str]] = None,
    ai_dataset_path: Optional[str] = None,
    ai_models: Optional[list[str]] = None,
    samples_per_source: int = 500,
    balance: bool = True,
) -> TextDataset:
    """
    Create a combined dataset with human and AI texts.

    Args:
        human_sources: Human text sources to use
        ai_dataset_path: Path to generated AI dataset
        ai_models: AI models to include
        samples_per_source: Samples per source/model
        balance: Whether to balance human and AI samples

    Returns:
        Combined TextDataset
    """
    # Load human texts
    human_data = load_multi_source_human_texts(
        sources=human_sources,
        samples_per_source=samples_per_source,
    )

    # Load AI texts
    ai_data = load_multi_source_ai_texts(
        ai_dataset_path=ai_dataset_path,
        models=ai_models,
        samples_per_model=samples_per_source,
    )

    # Balance if requested
    if balance:
        min_samples = min(len(human_data), len(ai_data))
        human_texts = human_data.texts[:min_samples]
        human_labels = human_data.labels[:min_samples]
        human_sources = human_data.sources[:min_samples]

        ai_texts = ai_data.texts[:min_samples]
        ai_labels = ai_data.labels[:min_samples]
        ai_sources = ai_data.sources[:min_samples]
    else:
        human_texts = human_data.texts
        human_labels = human_data.labels
        human_sources_list = human_data.sources

        ai_texts = ai_data.texts
        ai_labels = ai_data.labels
        ai_sources = ai_data.sources

    # Combine
    combined = TextDataset(
        texts=human_texts + ai_texts,
        labels=human_labels + ai_labels,
        sources=human_sources + ai_sources,
        metadata={
            "human_sources": human_data.metadata.get("sources", []),
            "ai_sources": ai_data.metadata.get("models", []),
            "balanced": balance,
        }
    )

    print(f"Combined dataset: {len(combined)} samples")
    print(f"  Human: {combined.labels.count(0)}")
    print(f"  AI: {combined.labels.count(1)}")

    return combined


def get_available_human_sources() -> list[str]:
    """Get list of available human text sources."""
    return [
        "wikitext",      # WikiText-103 (encyclopedic)
        "wikitext-2",    # WikiText-2 (smaller)
        "c4",            # C4 (web text)
        "bookcorpus",    # BookCorpus (books)
        "reddit",        # Reddit WritingPrompts (creative)
        "cnn",           # CNN/DailyMail (news)
        "arxiv",         # arXiv abstracts (academic)
        "news",          # AG News (news, shorter)
        "imdb",          # IMDB reviews
    ]


# =============================================================================
# DetectRL Benchmark (NeurIPS 2024)
# =============================================================================

DETECTRL_DOMAINS = ["arxiv", "writing_prompt", "xsum", "yelp_review"]
DETECTRL_MODELS = ["ChatGPT", "Llama-2-70b", "Claude-instant", "Google-PaLM"]
DETECTRL_ATTACKS = ["direct_prompt", "prompt_attacks", "paraphrase_attacks", "perturbation_attacks", "data_mixing"]

# Mapping from task/attack names to GitHub file paths (relative to Benchmark/Benchmark_Data/)
DETECTRL_FILES = {
    "direct_prompt": "Direct_Prompt/direct_prompt_test.json",
    "prompt_attacks": "Prompt_Attacks/prompt_attacks_llm_test.json",
    "paraphrase_attacks": "Paraphrase_Attacks/paraphrase_attacks_llm_test.json",
    "perturbation_attacks": "Perturbation_Attacks/perturbation_attacks_llm_test.json",
    "data_mixing": "Data_Mixing/data_mixing_attacks_test.json",
}

# Multi-domain files
DETECTRL_DOMAIN_FILES = {
    domain: f"Multi_Domain/multi_domains_{domain}_test.json"
    for domain in DETECTRL_DOMAINS
}

# Multi-LLM files
DETECTRL_MODEL_FILES = {
    model: f"Multi_LLM/multi_llms_{model}_test.json"
    for model in DETECTRL_MODELS
}

# Task groupings for evaluation
DETECTRL_TASKS = {
    "task1_attack": ["direct_prompt", "prompt_attacks", "paraphrase_attacks",
                     "perturbation_attacks", "data_mixing"],
    "task2_domain_gen": list(DETECTRL_DOMAINS),
    "task3_llm_gen": list(DETECTRL_MODELS),
}


def _download_detectrl_file(relative_path: str, cache_dir: str) -> Optional[str]:
    """
    Download a single DetectRL file from GitHub, caching locally.

    Args:
        relative_path: Path relative to Benchmark_Data/ in the repo
        cache_dir: Local directory to cache files

    Returns:
        Path to the downloaded file, or None if download failed
    """
    import urllib.request
    import urllib.error

    local_path = os.path.join(cache_dir, relative_path.replace("/", os.sep))

    # Return cached version if it exists
    if os.path.exists(local_path):
        return local_path

    # Download from GitHub
    base_url = "https://raw.githubusercontent.com/NLP2CT/DetectRL/main/Benchmark/Benchmark_Data"
    url = f"{base_url}/{relative_path}"

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        print(f"  Downloading {relative_path}...")
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except urllib.error.URLError as e:
        print(f"  Warning: Failed to download {relative_path}: {e}")
        return None


def _parse_detectrl_json(file_path: str) -> list[dict]:
    """Parse a DetectRL JSON file into a list of records."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # DetectRL files may be a list of dicts or a dict of lists
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Handle {"human": [...], "llm": [...]} format
        records = []
        for key, items in data.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str):
                        label = "human" if key.lower() in ("human", "human_text") else "llm"
                        records.append({"text": item, "label": label, "data_type": key})
                    elif isinstance(item, dict):
                        if "label" not in item:
                            item["label"] = "human" if key.lower() in ("human", "human_text") else "llm"
                        records.append(item)
        return records

    return []


def load_detectrl_benchmark(
    split: str = "test",
    tasks: Optional[list[str]] = None,
    domains: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    attacks: Optional[list[str]] = None,
    num_samples: Optional[int] = None,
    cache_dir: str = "datasets/detectrl",
) -> TextDataset:
    """
    Load the DetectRL benchmark dataset (NeurIPS 2024).

    DetectRL is designed to test detection in real-world scenarios including
    prompt variation, paraphrase attacks, perturbation attacks, and data mixing.

    Paper: https://arxiv.org/abs/2410.xxxxx
    GitHub: https://github.com/NLP2CT/DetectRL

    Args:
        split: Dataset split ("test" only for benchmark)
        tasks: Filter by task groups (e.g., ["task1_attack", "task2_domain_gen"])
        domains: Filter by domains (subset of DETECTRL_DOMAINS)
        models: Filter by LLM types (subset of DETECTRL_MODELS)
        attacks: Filter by attack types (subset of DETECTRL_ATTACKS)
        num_samples: Limit number of samples (None for all)
        cache_dir: Directory to cache downloaded files

    Returns:
        TextDataset with DetectRL samples
    """
    print(f"Loading DetectRL benchmark ({split} split)...")

    os.makedirs(cache_dir, exist_ok=True)

    # Determine which files to download based on filters
    files_to_load = {}  # source_name -> relative_path

    if tasks:
        for task in tasks:
            if task in DETECTRL_TASKS:
                for setting in DETECTRL_TASKS[task]:
                    if task == "task1_attack" and setting in DETECTRL_FILES:
                        files_to_load[f"task1_{setting}"] = DETECTRL_FILES[setting]
                    elif task == "task2_domain_gen" and setting in DETECTRL_DOMAIN_FILES:
                        files_to_load[f"task2_{setting}"] = DETECTRL_DOMAIN_FILES[setting]
                    elif task == "task3_llm_gen" and setting in DETECTRL_MODEL_FILES:
                        files_to_load[f"task3_{setting}"] = DETECTRL_MODEL_FILES[setting]
    else:
        # Load all files by default
        if attacks:
            for attack in attacks:
                if attack in DETECTRL_FILES:
                    files_to_load[f"task1_{attack}"] = DETECTRL_FILES[attack]
        else:
            for name, path in DETECTRL_FILES.items():
                files_to_load[f"task1_{name}"] = path

        if domains:
            for domain in domains:
                if domain in DETECTRL_DOMAIN_FILES:
                    files_to_load[f"task2_{domain}"] = DETECTRL_DOMAIN_FILES[domain]
        else:
            for domain, path in DETECTRL_DOMAIN_FILES.items():
                files_to_load[f"task2_{domain}"] = path

        if models:
            for model in models:
                if model in DETECTRL_MODEL_FILES:
                    files_to_load[f"task3_{model}"] = DETECTRL_MODEL_FILES[model]
        else:
            for model, path in DETECTRL_MODEL_FILES.items():
                files_to_load[f"task3_{model}"] = path

    texts = []
    labels = []
    sources = []

    for source_name, relative_path in files_to_load.items():
        file_path = _download_detectrl_file(relative_path, cache_dir)
        if file_path is None:
            continue

        records = _parse_detectrl_json(file_path)

        for record in records:
            text = record.get("text", "").strip()
            if not text:
                continue

            # Label mapping: "human" -> 0, "llm" -> 1
            label_str = record.get("label", "").lower()
            if label_str in ("human", "human_text", "0"):
                label = 0
            else:
                label = 1

            texts.append(text)
            labels.append(label)
            sources.append(f"detectrl_{source_name}")

            if num_samples and len(texts) >= num_samples:
                break

        if num_samples and len(texts) >= num_samples:
            break

    print(f"Loaded {len(texts)} samples from DetectRL benchmark")
    print(f"  Human: {labels.count(0)}, AI: {labels.count(1)}")

    return TextDataset(
        texts=texts,
        labels=labels,
        sources=sources,
        metadata={
            "benchmark": "DetectRL",
            "split": split,
            "tasks": tasks,
            "domains": domains,
            "models": models,
            "attacks": attacks,
        }
    )


def get_available_benchmarks() -> dict:
    """Get dictionary of available benchmarks."""
    return {
        "raid": {
            "description": "RAID benchmark (ACL 2024) - 6M+ samples, 11 models, 8 domains, 11 attacks",
            "url": "https://raid-bench.xyz",
            "load_func": "load_raid_benchmark",
        },
        "detectrl": {
            "description": "DetectRL benchmark (NeurIPS 2024) - Real-world scenarios, attacks, domain/LLM transfer",
            "url": "https://github.com/NLP2CT/DetectRL",
            "load_func": "load_detectrl_benchmark",
        },
        "mage": {
            "description": "MAGE benchmark - 447.7k samples, 7 domains, 27 generators",
            "load_func": "load_mage_benchmark",
        },
        "m4gt": {
            "description": "M4GT-Bench - Multilingual benchmark, 8 languages",
            "load_func": "load_m4gt_benchmark",
        },
    }
