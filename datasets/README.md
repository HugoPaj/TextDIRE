# Datasets Directory

This directory stores generated and cached datasets for Text-DIRE experiments.

## Structure

```
datasets/
├── human/          # Cached human text samples
├── ai/             # Generated AI text samples
├── ai_cache/       # Cache for API-generated texts
└── splits.json     # Train/val/test split definitions
```

## Human Text Sources

- **wikitext**: WikiText-103 encyclopedic text
- **reddit**: Reddit WritingPrompts creative writing
- **cnn**: CNN/DailyMail news articles
- **arxiv**: arXiv paper abstracts

## AI Text Sources

Generated using `src/ai_text_generator.py`:

- **GPT-5.2**: OpenAI flagship model
- **GPT-5-mini**: OpenAI efficient model
- **Claude Sonnet 4.5**: Anthropic balanced model
- **Claude Haiku 4.5**: Anthropic fast model

## Generating Datasets

```python
from src.ai_text_generator import AITextGenerator

generator = AITextGenerator()
dataset = generator.generate_dataset(
    models=["gpt-5.2", "claude-sonnet-4.5"],
    num_samples_per_model=500,
    save_path="datasets/ai/generated_texts.json"
)
```

## Loading Datasets

```python
from src.data import create_combined_dataset

dataset = create_combined_dataset(
    human_sources=["wikitext", "reddit", "cnn", "arxiv"],
    ai_dataset_path="datasets/ai/generated_texts.json",
    samples_per_source=500
)

train, val, test = dataset.split(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
```

## Benchmarks

For standard benchmarks, use:

```python
from src.data import load_raid_benchmark, load_mage_benchmark

raid = load_raid_benchmark(split="test")
mage = load_mage_benchmark(split="test")
```
