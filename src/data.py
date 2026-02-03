"""
Data loading utilities for Text-DIRE experiments.

Provides functions to load human text (WikiText) and AI-generated text (HC3).
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class TextDataset:
    """Container for a text classification dataset."""
    texts: list[str]
    labels: list[int]  # 0 = human, 1 = AI
    sources: list[str]  # Source identifier for each text

    def __len__(self):
        return len(self.texts)

    def get_human_texts(self) -> list[str]:
        return [t for t, l in zip(self.texts, self.labels) if l == 0]

    def get_ai_texts(self) -> list[str]:
        return [t for t, l in zip(self.texts, self.labels) if l == 1]


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

    else:
        raise ValueError(f"Unknown source: {source}")

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
