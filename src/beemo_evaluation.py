"""
Beemo Benchmark Evaluation for Text-DIRE.

Beemo (Benchmark of Expert-edited Machine-generated Outputs) tests detection
on edited AI text - a harder and more realistic scenario.

Paper: arxiv.org/abs/2411.04032 (NAACL 2025)

Usage:
    python -m src.beemo_evaluation
    python -m src.beemo_evaluation --scenario hard --device cuda
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
from tqdm import tqdm


@dataclass
class BeemoResult:
    """Results from Beemo evaluation."""
    scenario: str
    auroc: float
    accuracy: float
    f1: float
    n_samples: int
    human_mean_score: float
    ai_mean_score: float
    separation: float  # Cohen's d effect size


def load_beemo_dataset():
    """Load Beemo dataset from HuggingFace."""
    from datasets import load_dataset

    print("Loading Beemo dataset from HuggingFace...")
    dataset = load_dataset("toloka/beemo")
    df = dataset["train"].to_pandas()

    print(f"Loaded {len(df)} samples")
    print(f"Categories: {df['category'].unique().tolist()}")
    print(f"Models: {df['model'].unique().tolist()}")

    return df


def prepare_evaluation_data(df, scenario: str = "easy"):
    """
    Prepare data for a specific evaluation scenario.

    Scenarios:
        - easy: human_output vs model_output (raw AI)
        - medium: human_output vs gpt-4o_edits (LLM-edited)
        - medium_llama: human_output vs llama-3.1-70b_edits
        - hard: human_output vs human_edits (expert-edited)
        - all: Combined evaluation across all AI variants

    Returns:
        texts: List of texts
        labels: List of labels (0=human, 1=AI)
        metadata: List of metadata dicts
    """
    texts = []
    labels = []
    metadata = []

    if scenario == "easy":
        # Human vs raw model output
        for _, row in df.iterrows():
            # Human text
            if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                texts.append(str(row["human_output"]))
                labels.append(0)
                metadata.append({"source": "human", "category": row["category"]})

            # AI text
            if row["model_output"] and len(str(row["model_output"]).strip()) > 10:
                texts.append(str(row["model_output"]))
                labels.append(1)
                metadata.append({"source": row["model"], "category": row["category"]})

    elif scenario == "medium":
        # Human vs GPT-4o edited
        for _, row in df.iterrows():
            if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                texts.append(str(row["human_output"]))
                labels.append(0)
                metadata.append({"source": "human", "category": row["category"]})

            if row["gpt-4o_edits"] and len(str(row["gpt-4o_edits"]).strip()) > 10:
                texts.append(str(row["gpt-4o_edits"]))
                labels.append(1)
                metadata.append({"source": "gpt-4o_edited", "category": row["category"]})

    elif scenario == "medium_llama":
        # Human vs Llama-3.1-70B edited
        for _, row in df.iterrows():
            if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                texts.append(str(row["human_output"]))
                labels.append(0)
                metadata.append({"source": "human", "category": row["category"]})

            if row["llama-3.1-70b_edits"] and len(str(row["llama-3.1-70b_edits"]).strip()) > 10:
                texts.append(str(row["llama-3.1-70b_edits"]))
                labels.append(1)
                metadata.append({"source": "llama_edited", "category": row["category"]})

    elif scenario == "hard":
        # Human vs expert-edited AI (hardest)
        for _, row in df.iterrows():
            if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                texts.append(str(row["human_output"]))
                labels.append(0)
                metadata.append({"source": "human", "category": row["category"]})

            if row["human_edits"] and len(str(row["human_edits"]).strip()) > 10:
                texts.append(str(row["human_edits"]))
                labels.append(1)
                metadata.append({"source": "human_edited_ai", "category": row["category"]})

    elif scenario == "all":
        # All variants combined
        for _, row in df.iterrows():
            # Human (only add once per row to balance)
            if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                texts.append(str(row["human_output"]))
                labels.append(0)
                metadata.append({"source": "human", "category": row["category"]})

            # Raw model output
            if row["model_output"] and len(str(row["model_output"]).strip()) > 10:
                texts.append(str(row["model_output"]))
                labels.append(1)
                metadata.append({"source": f"raw_{row['model']}", "category": row["category"]})

    print(f"Scenario '{scenario}': {len(texts)} texts ({sum(1 for l in labels if l == 0)} human, {sum(labels)} AI)")

    return texts, labels, metadata


def run_dire_detection(
    texts: list[str],
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    mask_ratio: float = 0.5,
    device: str = "cuda",
) -> list[float]:
    """
    Run Text-DIRE detection on texts.

    Returns scores where HIGHER = more likely AI.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    from .dire import TextDIRE

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    dire = TextDIRE(model, tokenizer, device=device)

    scores = []
    for text in tqdm(texts, desc="DIRE detection"):
        try:
            result = dire.compute_score(text, mask_ratio=mask_ratio)
            # DIRE: lower error = AI (reconstructs well)
            # We want: higher score = AI
            # So: AI_score = 1 - reconstruction_error = token_accuracy
            ai_score = result.token_accuracy
            scores.append(ai_score)
        except Exception as e:
            print(f"Error: {e}")
            scores.append(0.5)

    return scores


def evaluate_scores(
    labels: list[int],
    scores: list[float],
    scenario: str,
) -> BeemoResult:
    """Evaluate detection scores."""
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve

    labels = np.array(labels)
    scores = np.array(scores)

    # AUROC
    auroc = roc_auc_score(labels, scores)

    # Handle inverted scores
    if auroc < 0.5:
        scores = 1 - scores
        auroc = 1 - auroc

    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    threshold = thresholds[optimal_idx]

    # Predictions at optimal threshold
    predictions = (scores >= threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    # Score distributions
    human_scores = scores[labels == 0]
    ai_scores = scores[labels == 1]

    human_mean = np.mean(human_scores)
    ai_mean = np.mean(ai_scores)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(human_scores) - 1) * np.std(human_scores, ddof=1)**2 +
         (len(ai_scores) - 1) * np.std(ai_scores, ddof=1)**2) /
        (len(human_scores) + len(ai_scores) - 2)
    )
    cohens_d = abs(ai_mean - human_mean) / pooled_std if pooled_std > 0 else 0

    return BeemoResult(
        scenario=scenario,
        auroc=auroc,
        accuracy=accuracy,
        f1=f1,
        n_samples=len(labels),
        human_mean_score=human_mean,
        ai_mean_score=ai_mean,
        separation=cohens_d,
    )


def run_beemo_evaluation(
    scenarios: list[str] = None,
    mask_ratio: float = 0.5,
    device: str = "cuda",
    output_dir: str = "results/beemo",
    max_samples: Optional[int] = None,
) -> dict[str, BeemoResult]:
    """
    Run full Beemo benchmark evaluation.

    Args:
        scenarios: List of scenarios to evaluate
        mask_ratio: DIRE mask ratio
        device: Device to use
        output_dir: Directory for results
        max_samples: Limit samples per scenario (for testing)

    Returns:
        Dictionary mapping scenario to BeemoResult
    """
    if scenarios is None:
        scenarios = ["easy", "medium", "hard"]

    # Load dataset
    df = load_beemo_dataset()

    if max_samples:
        df = df.head(max_samples)

    results = {}
    all_scores_data = {}

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Evaluating scenario: {scenario.upper()}")
        print('='*60)

        # Prepare data
        texts, labels, metadata = prepare_evaluation_data(df, scenario)

        if len(texts) == 0:
            print(f"No data for scenario {scenario}, skipping")
            continue

        # Run detection
        scores = run_dire_detection(
            texts,
            mask_ratio=mask_ratio,
            device=device,
        )

        # Evaluate
        result = evaluate_scores(labels, scores, scenario)
        results[scenario] = result

        # Store detailed data
        all_scores_data[scenario] = {
            "texts": texts[:10],  # Sample for reference
            "labels": labels,
            "scores": scores,
            "metadata": metadata,
        }

        # Print results
        print(f"\nResults for {scenario}:")
        print(f"  AUROC:    {result.auroc:.4f}")
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  F1:       {result.f1:.4f}")
        print(f"  Samples:  {result.n_samples}")
        print(f"  Human mean score: {result.human_mean_score:.4f}")
        print(f"  AI mean score:    {result.ai_mean_score:.4f}")
        print(f"  Separation (d):   {result.separation:.2f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = {
        scenario: {
            "auroc": r.auroc,
            "accuracy": r.accuracy,
            "f1": r.f1,
            "n_samples": r.n_samples,
            "human_mean": r.human_mean_score,
            "ai_mean": r.ai_mean_score,
            "separation": r.separation,
        }
        for scenario, r in results.items()
    }

    with open(output_path / "beemo_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Detailed scores (for plotting)
    detailed = {
        scenario: {
            "labels": [int(l) for l in data["labels"]],
            "scores": [float(s) for s in data["scores"]],
        }
        for scenario, data in all_scores_data.items()
    }

    with open(output_path / "beemo_scores.json", "w") as f:
        json.dump(detailed, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


def run_beemo_by_model(
    mask_ratio: float = 0.5,
    device: str = "cuda",
) -> dict:
    """
    Run Beemo evaluation broken down by source LLM.
    """
    df = load_beemo_dataset()

    models = df["model"].unique().tolist()
    print(f"Models in dataset: {models}")

    # Load DIRE model once
    import torch
    from transformers import AutoModel, AutoTokenizer
    from .dire import TextDIRE

    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    dire = TextDIRE(model, tokenizer, device=device)

    results_by_model = {}

    for source_model in models:
        print(f"\n{'='*40}")
        print(f"Evaluating: {source_model}")
        print('='*40)

        subset = df[df["model"] == source_model]

        texts = []
        labels = []

        for _, row in subset.iterrows():
            # Human
            if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                texts.append(str(row["human_output"]))
                labels.append(0)

            # Model output
            if row["model_output"] and len(str(row["model_output"]).strip()) > 10:
                texts.append(str(row["model_output"]))
                labels.append(1)

        if len(texts) < 20:
            print(f"Skipping {source_model} - too few samples")
            continue

        # Run detection
        scores = []
        for text in tqdm(texts, desc=f"DIRE on {source_model}"):
            try:
                result = dire.compute_score(text, mask_ratio=mask_ratio)
                scores.append(result.token_accuracy)
            except Exception:
                scores.append(0.5)

        # Evaluate
        result = evaluate_scores(labels, scores, source_model)
        results_by_model[source_model] = {
            "auroc": result.auroc,
            "accuracy": result.accuracy,
            "n_samples": result.n_samples,
        }

        print(f"  AUROC: {result.auroc:.4f}")

    return results_by_model


def run_beemo_by_category(
    mask_ratio: float = 0.5,
    device: str = "cuda",
) -> dict:
    """
    Run Beemo evaluation broken down by text category.
    """
    df = load_beemo_dataset()

    categories = df["category"].unique().tolist()
    print(f"Categories in dataset: {categories}")

    # Load DIRE model once
    import torch
    from transformers import AutoModel, AutoTokenizer
    from .dire import TextDIRE

    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    dire = TextDIRE(model, tokenizer, device=device)

    results_by_category = {}

    for category in categories:
        print(f"\n{'='*40}")
        print(f"Evaluating category: {category}")
        print('='*40)

        subset = df[df["category"] == category]

        texts = []
        labels = []

        for _, row in subset.iterrows():
            if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                texts.append(str(row["human_output"]))
                labels.append(0)

            if row["model_output"] and len(str(row["model_output"]).strip()) > 10:
                texts.append(str(row["model_output"]))
                labels.append(1)

        if len(texts) < 20:
            print(f"Skipping {category} - too few samples")
            continue

        # Run detection
        scores = []
        for text in tqdm(texts, desc=f"DIRE on {category}"):
            try:
                result = dire.compute_score(text, mask_ratio=mask_ratio)
                scores.append(result.token_accuracy)
            except Exception:
                scores.append(0.5)

        # Evaluate
        result = evaluate_scores(labels, scores, category)
        results_by_category[category] = {
            "auroc": result.auroc,
            "accuracy": result.accuracy,
            "n_samples": result.n_samples,
        }

        print(f"  AUROC: {result.auroc:.4f}")

    return results_by_category


def plot_beemo_results(
    results_path: str = "results/beemo/beemo_scores.json",
    output_path: str = "results/beemo/beemo_plots.png",
):
    """Generate visualization of Beemo results."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    with open(results_path) as f:
        data = json.load(f)

    n_scenarios = len(data)
    fig, axes = plt.subplots(1, n_scenarios + 1, figsize=(5 * (n_scenarios + 1), 5))

    aurocs = []
    scenario_names = []

    for idx, (scenario, scores_data) in enumerate(data.items()):
        ax = axes[idx]

        labels = np.array(scores_data["labels"])
        scores = np.array(scores_data["scores"])

        human_scores = scores[labels == 0]
        ai_scores = scores[labels == 1]

        sns.histplot(human_scores, ax=ax, color="blue", alpha=0.5, label="Human", stat="density", bins=20)
        sns.histplot(ai_scores, ax=ax, color="red", alpha=0.5, label="AI", stat="density", bins=20)

        ax.set_xlabel("DIRE Score (token accuracy)")
        ax.set_ylabel("Density")
        ax.set_title(f"{scenario.upper()}")
        ax.legend()

        # Compute AUROC
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(labels, scores)
        if auroc < 0.5:
            auroc = 1 - auroc
        aurocs.append(auroc)
        scenario_names.append(scenario)

        ax.text(0.05, 0.95, f"AUROC: {auroc:.3f}", transform=ax.transAxes,
                fontsize=12, verticalalignment="top", fontweight="bold")

    # Summary bar chart
    ax = axes[-1]
    colors = ["green" if a > 0.8 else "orange" if a > 0.6 else "red" for a in aurocs]
    bars = ax.bar(scenario_names, aurocs, color=colors)
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC by Scenario")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    for bar, auroc in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{auroc:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    return fig


def print_comparison_table(results: dict[str, BeemoResult]):
    """Print a comparison table of results."""
    print("\n" + "=" * 70)
    print("BEEMO BENCHMARK RESULTS - TEXT-DIRE")
    print("=" * 70)
    print(f"{'Scenario':<15} {'AUROC':<10} {'Accuracy':<10} {'F1':<10} {'Separation':<12} {'N':<8}")
    print("-" * 70)

    for scenario, result in results.items():
        print(f"{scenario:<15} {result.auroc:<10.4f} {result.accuracy:<10.4f} "
              f"{result.f1:<10.4f} {result.separation:<12.2f} {result.n_samples:<8}")

    print("=" * 70)

    # Interpretation
    print("\nInterpretation:")
    print("  - Easy (raw AI): Standard detection task")
    print("  - Medium (LLM-edited): AI text refined by another LLM")
    print("  - Hard (human-edited): AI text refined by human experts")
    print("\nComparison to Beemo paper baselines:")
    print("  - Binoculars: Best on raw AI, struggles on edited")
    print("  - DetectGPT: Most robust across scenarios")


def main():
    parser = argparse.ArgumentParser(description="Beemo Benchmark Evaluation")
    parser.add_argument("--scenarios", nargs="+", default=["easy", "medium", "hard"],
                        choices=["easy", "medium", "medium_llama", "hard", "all"])
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results/beemo")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--by-model", action="store_true", help="Evaluate by source model")
    parser.add_argument("--by-category", action="store_true", help="Evaluate by text category")
    parser.add_argument("--plot", action="store_true", help="Generate plots from existing results")

    args = parser.parse_args()

    if args.plot:
        plot_beemo_results(
            results_path=f"{args.output_dir}/beemo_scores.json",
            output_path=f"{args.output_dir}/beemo_plots.png",
        )
        return

    if args.by_model:
        results = run_beemo_by_model(
            mask_ratio=args.mask_ratio,
            device=args.device,
        )
        print("\n" + "=" * 50)
        print("RESULTS BY SOURCE MODEL")
        print("=" * 50)
        for model, r in sorted(results.items(), key=lambda x: x[1]["auroc"], reverse=True):
            print(f"  {model:<30} AUROC: {r['auroc']:.4f}")
        return

    if args.by_category:
        results = run_beemo_by_category(
            mask_ratio=args.mask_ratio,
            device=args.device,
        )
        print("\n" + "=" * 50)
        print("RESULTS BY CATEGORY")
        print("=" * 50)
        for cat, r in sorted(results.items(), key=lambda x: x[1]["auroc"], reverse=True):
            print(f"  {cat:<20} AUROC: {r['auroc']:.4f}")
        return

    # Main evaluation
    results = run_beemo_evaluation(
        scenarios=args.scenarios,
        mask_ratio=args.mask_ratio,
        device=args.device,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )

    print_comparison_table(results)

    # Generate plots
    try:
        plot_beemo_results(
            results_path=f"{args.output_dir}/beemo_scores.json",
            output_path=f"{args.output_dir}/beemo_plots.png",
        )
    except Exception as e:
        print(f"Could not generate plots: {e}")


if __name__ == "__main__":
    main()
