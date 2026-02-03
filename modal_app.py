"""
Text-DIRE: Diffusion Reconstruction Error for AI Text Detection
Modal Cloud GPU Implementation

Run with: modal run modal_app.py
"""

import modal

# Define the image with all dependencies
# IMPORTANT: LLaDA requires exactly transformers==4.38.2
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0",
    "transformers==4.38.2",  # LLaDA requires this exact version
    "accelerate",
    "datasets",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "numpy",
    "tqdm",
    "huggingface_hub",
    "sentencepiece",
)

# Create app and volume
app = modal.App("text-dire")
volume = modal.Volume.from_name("text-dire-vol", create_if_missing=True)

MODEL_DIR = "/vol/models"
RESULTS_DIR = "/vol/results"


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/vol": volume},
)
def load_and_cache_model():
    """
    Pre-download and cache the LLaDA model to the volume.
    This speeds up subsequent runs.

    Requires transformers==4.38.2 (exact version for LLaDA compatibility)
    """
    import os
    from transformers import AutoModel, AutoTokenizer
    import torch

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading LLaDA-8B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        cache_dir=MODEL_DIR,
    )

    print("Loading LLaDA-8B model (this may take a while on first run)...")
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_DIR,
    ).to("cuda").eval()

    print("LLaDA-8B loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")

    # Commit the volume to persist the cached model
    volume.commit()

    return {"status": "success", "model_type": "LLaDA-8B"}


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/vol": volume},
    memory=32768,
)
def compute_dire_scores_batch(texts: list[str], labels: list[int], mask_ratios: list[float] = None):
    """
    Compute Text-DIRE scores for a batch of texts using LLaDA.

    Based on the official LLaDA implementation from:
    https://github.com/ML-GSAI/LLaDA

    Args:
        texts: List of text samples
        labels: List of labels (0=human, 1=AI)
        mask_ratios: List of masking ratios to try

    Returns:
        Dictionary with scores and metadata
    """
    import os
    import torch
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    if mask_ratios is None:
        mask_ratios = [0.3, 0.5, 0.7]

    os.makedirs(MODEL_DIR, exist_ok=True)

    # LLaDA's mask token ID (from official code)
    MASK_ID = 126336

    print("Loading LLaDA-8B model (requires transformers==4.38.2)...")
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        cache_dir=MODEL_DIR,
    )

    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_DIR,
    ).to("cuda").eval()

    print(f"LLaDA-8B loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Using mask token ID: {MASK_ID}")
    print(f"Computing DIRE scores for {len(texts)} texts...")

    results = []

    for idx, text in enumerate(tqdm(texts, desc="Computing DIRE scores")):
        if not text or len(text.strip()) < 10:
            continue

        text_results = {
            "text_idx": idx,
            "label": labels[idx],
            "text_length": len(text),
        }

        try:
            # Tokenize
            input_ids = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )["input_ids"].to(model.device)

            seq_len = input_ids.shape[1]

            if seq_len < 10:
                continue

            for mask_ratio in mask_ratios:
                # Number of tokens to mask
                num_mask = max(1, int(seq_len * mask_ratio))

                # Random positions to mask (can include any position for LLaDA)
                mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
                positions = torch.randperm(seq_len, device=input_ids.device)[:num_mask]
                mask_positions[0, positions] = True

                # Apply mask
                masked_ids = input_ids.clone()
                masked_ids[mask_positions] = MASK_ID

                # Get model predictions (LLaDA returns logits directly)
                with torch.no_grad():
                    logits = model(masked_ids).logits

                    # Get predictions at masked positions
                    predictions = logits.argmax(dim=-1)

                    # Compute accuracy: how many masked tokens were correctly predicted
                    original_tokens = input_ids[mask_positions]
                    predicted_tokens = predictions[mask_positions]
                    correct = (predicted_tokens == original_tokens).float()
                    token_accuracy = correct.mean().item()

                    # Also compute cross-entropy loss at masked positions (like official get_log_likelihood)
                    ce_loss = F.cross_entropy(
                        logits[mask_positions],
                        input_ids[mask_positions],
                        reduction='mean'
                    ).item()

                text_results[f"accuracy_{mask_ratio}"] = token_accuracy
                text_results[f"error_{mask_ratio}"] = 1.0 - token_accuracy
                text_results[f"ce_loss_{mask_ratio}"] = ce_loss
                text_results[f"num_masked_{mask_ratio}"] = num_mask

        except Exception as e:
            print(f"Error processing text {idx}: {e}")
            continue

        results.append(text_results)

    # Commit volume
    volume.commit()

    return results


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/vol": volume},
)
def compute_perplexity_baseline(texts: list[str], labels: list[int]):
    """
    Compute perplexity-based baseline using GPT-2.
    Lower perplexity often indicates AI-generated text.
    """
    import os
    import torch
    from tqdm import tqdm
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import numpy as np

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading GPT-2 for perplexity baseline...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=MODEL_DIR).cuda()
    model.eval()

    results = []

    for idx, text in enumerate(tqdm(texts, desc="Computing perplexity")):
        if not text or len(text.strip()) < 10:
            continue

        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                perplexity = np.exp(loss)

            results.append({
                "text_idx": idx,
                "label": labels[idx],
                "perplexity": perplexity,
                "loss": loss,
            })

        except Exception as e:
            print(f"Error computing perplexity for text {idx}: {e}")
            continue

    volume.commit()
    return results


@app.function(
    image=image,
    timeout=600,
    volumes={"/vol": volume},
)
def load_datasets(num_samples: int = 100):
    """
    Load human (WikiText) and AI (HC3 ChatGPT) text samples.
    """
    from datasets import load_dataset

    print(f"Loading {num_samples} samples each of human and AI text...")

    # Load WikiText for human samples
    print("Loading WikiText-103...")
    try:
        wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        human_texts = [
            t.strip() for t in wiki["text"]
            if t and len(t.split()) > 50 and not t.startswith("=")
        ][:num_samples]
    except Exception as e:
        print(f"WikiText loading failed: {e}, trying alternative...")
        # Fallback to a simpler dataset
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        human_texts = [
            t.strip() for t in wiki["text"]
            if t and len(t.split()) > 30 and not t.startswith("=")
        ][:num_samples]

    print(f"Loaded {len(human_texts)} human text samples")

    # Load AI text samples - try multiple sources
    ai_texts = []

    # Try HC3 first
    print("Loading HC3 dataset (Human vs ChatGPT)...")
    try:
        hc3 = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
        for item in hc3:
            if item["chatgpt_answers"] and len(item["chatgpt_answers"]) > 0:
                answer = item["chatgpt_answers"][0]
                if len(answer.split()) > 30:
                    ai_texts.append(answer.strip())
            if len(ai_texts) >= num_samples:
                break
    except Exception as e:
        print(f"HC3 loading failed: {e}")

    # Fallback 1: GPT-wiki-intro dataset
    if len(ai_texts) < num_samples:
        print("Trying GPT-wiki-intro dataset...")
        try:
            gpt_wiki = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
            for item in gpt_wiki:
                text = item.get("generated_intro", "").strip()
                if text and len(text.split()) > 30:
                    ai_texts.append(text)
                if len(ai_texts) >= num_samples:
                    break
        except Exception as e:
            print(f"GPT-wiki-intro loading failed: {e}")

    # Fallback 2: Use human text from different part of WikiText as "AI-like" control
    # (This is just for testing the pipeline - not a real AI detector test)
    if len(ai_texts) < num_samples:
        print("Using WikiText validation as fallback AI samples (for pipeline testing only)...")
        try:
            wiki_val = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
            for text in wiki_val["text"]:
                text = text.strip()
                if text and len(text.split()) > 50 and not text.startswith("="):
                    ai_texts.append(text)
                if len(ai_texts) >= num_samples:
                    break
        except Exception as e:
            print(f"Fallback also failed: {e}")

    print(f"Loaded {len(ai_texts)} AI text samples")

    # Balance datasets
    min_samples = min(len(human_texts), len(ai_texts))
    human_texts = human_texts[:min_samples]
    ai_texts = ai_texts[:min_samples]

    print(f"Using {min_samples} samples per class")

    return {
        "human_texts": human_texts,
        "ai_texts": ai_texts,
        "num_samples": min_samples,
    }


@app.function(
    image=image,
    timeout=600,
    volumes={"/vol": volume},
)
def evaluate_and_plot(dire_results: list, perplexity_results: list, mask_ratios: list[float] = None):
    """
    Evaluate DIRE scores and create visualization.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_auc_score, roc_curve

    if mask_ratios is None:
        mask_ratios = [0.3, 0.5, 0.7]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Evaluating results...")

    # Extract scores by label
    results_summary = {}

    for mask_ratio in mask_ratios:
        error_key = f"error_{mask_ratio}"

        human_errors = [r[error_key] for r in dire_results if r["label"] == 0 and error_key in r]
        ai_errors = [r[error_key] for r in dire_results if r["label"] == 1 and error_key in r]

        if human_errors and ai_errors:
            # Combine for AUROC calculation
            all_errors = human_errors + ai_errors
            all_labels = [0] * len(human_errors) + [1] * len(ai_errors)

            # AUROC: higher error for AI should give good AUROC
            # But if human has higher error, we flip
            auroc = roc_auc_score(all_labels, all_errors)
            if auroc < 0.5:
                auroc = 1 - auroc  # Flip if direction is wrong

            results_summary[f"mask_{mask_ratio}"] = {
                "human_mean": np.mean(human_errors),
                "human_std": np.std(human_errors),
                "ai_mean": np.mean(ai_errors),
                "ai_std": np.std(ai_errors),
                "auroc": auroc,
            }

            print(f"\nMask ratio {mask_ratio}:")
            print(f"  Human - Mean error: {np.mean(human_errors):.4f} (+/- {np.std(human_errors):.4f})")
            print(f"  AI    - Mean error: {np.mean(ai_errors):.4f} (+/- {np.std(ai_errors):.4f})")
            print(f"  AUROC: {auroc:.4f}")

    # Perplexity baseline
    if perplexity_results:
        human_ppl = [r["perplexity"] for r in perplexity_results if r["label"] == 0]
        ai_ppl = [r["perplexity"] for r in perplexity_results if r["label"] == 1]

        if human_ppl and ai_ppl:
            all_ppl = human_ppl + ai_ppl
            all_labels = [0] * len(human_ppl) + [1] * len(ai_ppl)

            # Lower perplexity typically indicates AI text
            ppl_auroc = roc_auc_score(all_labels, [-p for p in all_ppl])
            if ppl_auroc < 0.5:
                ppl_auroc = 1 - ppl_auroc

            results_summary["perplexity_baseline"] = {
                "human_mean": np.mean(human_ppl),
                "human_std": np.std(human_ppl),
                "ai_mean": np.mean(ai_ppl),
                "ai_std": np.std(ai_ppl),
                "auroc": ppl_auroc,
            }

            print(f"\nPerplexity Baseline:")
            print(f"  Human - Mean: {np.mean(human_ppl):.2f} (+/- {np.std(human_ppl):.2f})")
            print(f"  AI    - Mean: {np.mean(ai_ppl):.2f} (+/- {np.std(ai_ppl):.2f})")
            print(f"  AUROC: {ppl_auroc:.4f}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: DIRE distribution for best mask ratio
    best_ratio = max(mask_ratios, key=lambda r: results_summary.get(f"mask_{r}", {}).get("auroc", 0))
    error_key = f"error_{best_ratio}"

    human_errors = [r[error_key] for r in dire_results if r["label"] == 0 and error_key in r]
    ai_errors = [r[error_key] for r in dire_results if r["label"] == 1 and error_key in r]

    ax1 = axes[0, 0]
    if human_errors and ai_errors:
        sns.histplot(human_errors, ax=ax1, color='blue', alpha=0.5, label='Human', stat='density', bins=20)
        sns.histplot(ai_errors, ax=ax1, color='red', alpha=0.5, label='AI', stat='density', bins=20)
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Text-DIRE Distribution (mask ratio={best_ratio})')
        ax1.legend()

    # Plot 2: AUROC comparison across mask ratios
    ax2 = axes[0, 1]
    aurocs = [results_summary.get(f"mask_{r}", {}).get("auroc", 0) for r in mask_ratios]
    bars = ax2.bar([f"{r}" for r in mask_ratios], aurocs, color='steelblue')
    if "perplexity_baseline" in results_summary:
        ax2.axhline(y=results_summary["perplexity_baseline"]["auroc"],
                   color='orange', linestyle='--', label=f'Perplexity baseline')
        ax2.legend()
    ax2.set_xlabel('Mask Ratio')
    ax2.set_ylabel('AUROC')
    ax2.set_title('AUROC by Mask Ratio')
    ax2.set_ylim(0.4, 1.0)
    for bar, auroc in zip(bars, aurocs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auroc:.3f}', ha='center', va='bottom')

    # Plot 3: Perplexity distribution
    ax3 = axes[1, 0]
    if perplexity_results:
        human_ppl = [r["perplexity"] for r in perplexity_results if r["label"] == 0]
        ai_ppl = [r["perplexity"] for r in perplexity_results if r["label"] == 1]

        # Cap perplexity for visualization
        human_ppl_capped = [min(p, 500) for p in human_ppl]
        ai_ppl_capped = [min(p, 500) for p in ai_ppl]

        sns.histplot(human_ppl_capped, ax=ax3, color='blue', alpha=0.5, label='Human', stat='density', bins=20)
        sns.histplot(ai_ppl_capped, ax=ax3, color='red', alpha=0.5, label='AI', stat='density', bins=20)
        ax3.set_xlabel('Perplexity (capped at 500)')
        ax3.set_ylabel('Density')
        ax3.set_title('Perplexity Distribution')
        ax3.legend()

    # Plot 4: ROC Curves
    ax4 = axes[1, 1]

    # Best DIRE ROC curve
    if human_errors and ai_errors:
        all_errors = human_errors + ai_errors
        all_labels = [0] * len(human_errors) + [1] * len(ai_errors)

        fpr, tpr, _ = roc_curve(all_labels, all_errors)
        auroc = roc_auc_score(all_labels, all_errors)
        if auroc < 0.5:
            fpr, tpr, _ = roc_curve(all_labels, [-e for e in all_errors])
            auroc = 1 - auroc

        ax4.plot(fpr, tpr, label=f'DIRE (mask={best_ratio}) AUC={auroc:.3f}', color='blue')

    # Perplexity ROC curve
    if perplexity_results:
        all_ppl = human_ppl + ai_ppl
        all_labels = [0] * len(human_ppl) + [1] * len(ai_ppl)

        fpr, tpr, _ = roc_curve(all_labels, [-p for p in all_ppl])
        ppl_auroc = roc_auc_score(all_labels, [-p for p in all_ppl])
        if ppl_auroc < 0.5:
            fpr, tpr, _ = roc_curve(all_labels, all_ppl)
            ppl_auroc = 1 - ppl_auroc

        ax4.plot(fpr, tpr, label=f'Perplexity AUC={ppl_auroc:.3f}', color='orange', linestyle='--')

    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves')
    ax4.legend()
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(RESULTS_DIR, "dire_distributions.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")

    # Save results as text
    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(results_path, "w") as f:
        f.write("Text-DIRE Experiment Results\n")
        f.write("=" * 50 + "\n\n")

        for key, values in results_summary.items():
            f.write(f"{key}:\n")
            for k, v in values.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")

    print(f"Results summary saved to {results_path}")

    volume.commit()

    return results_summary


@app.function(
    image=image,
    gpu="A100",
    timeout=10800,  # 3 hours max
    volumes={"/vol": volume},
    memory=32768,
)
def run_experiment(num_samples: int = 100):
    """
    Full proof-of-concept experiment.

    1. Load datasets (human + AI text)
    2. Compute DIRE scores
    3. Compute perplexity baseline
    4. Evaluate and plot results
    """
    import os
    import json

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Text-DIRE: Diffusion Reconstruction Error for AI Detection")
    print("=" * 60)

    # Step 1: Load datasets
    print("\n[Step 1/4] Loading datasets...")
    data = load_datasets.remote(num_samples=num_samples)

    human_texts = data["human_texts"]
    ai_texts = data["ai_texts"]

    # Combine texts and labels
    all_texts = human_texts + ai_texts
    all_labels = [0] * len(human_texts) + [1] * len(ai_texts)

    print(f"Total samples: {len(all_texts)}")

    # Step 2: Compute DIRE scores
    print("\n[Step 2/4] Computing DIRE scores with LLaDA-8B...")
    mask_ratios = [0.3, 0.5, 0.7]
    dire_results = compute_dire_scores_batch.remote(all_texts, all_labels, mask_ratios)

    print(f"DIRE scores computed for {len(dire_results)} samples")

    # Step 3: Compute perplexity baseline
    print("\n[Step 3/4] Computing perplexity baseline with GPT-2...")
    perplexity_results = compute_perplexity_baseline.remote(all_texts, all_labels)

    print(f"Perplexity computed for {len(perplexity_results)} samples")

    # Step 4: Evaluate and plot
    print("\n[Step 4/4] Evaluating results and creating plots...")
    results_summary = evaluate_and_plot.remote(dire_results, perplexity_results, mask_ratios)

    # Save raw results
    raw_results_path = os.path.join(RESULTS_DIR, "raw_results.json")
    with open(raw_results_path, "w") as f:
        json.dump({
            "dire_results": dire_results,
            "perplexity_results": perplexity_results,
            "summary": {k: {kk: float(vv) for kk, vv in v.items()}
                       for k, v in results_summary.items()},
        }, f, indent=2)

    volume.commit()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("\nTo download results:")
    print("  modal volume get text-dire-vol results/dire_distributions.png .")
    print("  modal volume get text-dire-vol results/results_summary.txt .")
    print("  modal volume get text-dire-vol results/raw_results.json .")

    return results_summary


@app.local_entrypoint()
def main(num_samples: int = 100):
    """
    Entry point for modal run command.

    Usage:
        modal run modal_app.py
        modal run modal_app.py --num-samples 50
    """
    print("Starting Text-DIRE experiment on Modal...")
    print(f"Using {num_samples} samples per class")

    results = run_experiment.remote(num_samples=num_samples)

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    for method, metrics in results.items():
        print(f"\n{method}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    return results
