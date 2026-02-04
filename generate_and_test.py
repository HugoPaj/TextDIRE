"""
Generate AI text from modern models and test DIRE detection.

Usage:
    python generate_and_test.py --samples 100
    python generate_and_test.py --samples 100 --models gpt-4o claude-sonnet-4-5-20250929
"""

import argparse
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_dataset(models: list[str], num_samples: int, output_path: str):
    """Generate AI text dataset from specified models."""
    from src.ai_text_generator import AITextGenerator

    print(f"Generating {num_samples} samples from each model...")
    print(f"Models: {models}")

    generator = AITextGenerator()

    dataset = generator.generate_dataset(
        models=models,
        num_samples_per_model=num_samples,
        save_path=output_path,
    )

    # Print summary
    for model, texts in dataset.items():
        print(f"  {model}: {len(texts)} samples")

    return dataset


def load_human_texts(num_samples: int):
    """Load human texts from WikiText."""
    from src.data import load_human_texts

    print(f"Loading {num_samples} human text samples...")
    texts = load_human_texts("wikitext", num_samples=num_samples)
    print(f"  Loaded {len(texts)} human samples")

    return texts


def main():
    parser = argparse.ArgumentParser(description="Generate and test AI text detection")
    parser.add_argument("--samples", type=int, default=100, help="Samples per model")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini", "claude-haiku-4-5-20251009"],
                        help="Models to test")
    parser.add_argument("--output", default="datasets/ai/modern_ai_texts.json",
                        help="Output path for generated texts")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, use existing dataset")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Step 1: Generate AI texts (or load existing)
    if args.skip_generation and os.path.exists(args.output):
        print(f"Loading existing dataset from {args.output}")
        with open(args.output, "r", encoding="utf-8") as f:
            ai_dataset = json.load(f)
    else:
        ai_dataset = generate_dataset(args.models, args.samples, args.output)
        # Convert GeneratedText objects to dicts if needed
        ai_dataset = {
            model: [t.text if hasattr(t, 'text') else t['text'] for t in texts]
            for model, texts in ai_dataset.items()
        }

    # Step 2: Load human texts
    total_ai_samples = sum(len(texts) for texts in ai_dataset.values())
    human_texts = load_human_texts(total_ai_samples)

    # Step 3: Prepare data for Modal
    print("\nPreparing data for DIRE evaluation...")

    # Combine all AI texts
    all_ai_texts = []
    ai_sources = []
    for model, texts in ai_dataset.items():
        for text in texts:
            if isinstance(text, dict):
                all_ai_texts.append(text.get('text', ''))
            else:
                all_ai_texts.append(text)
            ai_sources.append(model)

    # Balance
    min_samples = min(len(human_texts), len(all_ai_texts))
    human_texts = human_texts[:min_samples]
    all_ai_texts = all_ai_texts[:min_samples]
    ai_sources = ai_sources[:min_samples]

    # Save combined dataset for Modal
    combined_path = "datasets/combined_modern.json"
    combined = {
        "human_texts": human_texts,
        "ai_texts": all_ai_texts,
        "ai_sources": ai_sources,
    }

    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\nDataset ready: {combined_path}")
    print(f"  Human: {len(human_texts)} samples")
    print(f"  AI: {len(all_ai_texts)} samples")
    print(f"  Models: {list(ai_dataset.keys())}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
To run DIRE on this dataset, use Modal:

    modal run modal_app.py

Or for a quick local test with perplexity baseline:

    python test_local.py
    """)


if __name__ == "__main__":
    main()
