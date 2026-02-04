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


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/vol": volume},
    memory=32768,
)
def compute_dire_scores_mc(
    texts: list[str],
    labels: list[int],
    mask_ratio: float = 0.5,
    mc_samples: int = 32,
):
    """
    Compute DIRE scores with Monte Carlo estimation for stability.

    Args:
        texts: List of text samples
        labels: Labels for each text
        mask_ratio: Mask ratio for DIRE
        mc_samples: Number of MC samples per text

    Returns:
        List of result dictionaries with mean, std, CI
    """
    import os
    import torch
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    os.makedirs(MODEL_DIR, exist_ok=True)

    MASK_ID = 126336

    print("Loading LLaDA-8B model...")
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

    print(f"Computing MC DIRE scores ({mc_samples} samples) for {len(texts)} texts...")

    results = []

    for idx, text in enumerate(tqdm(texts, desc="MC DIRE")):
        if not text or len(text.strip()) < 10:
            continue

        try:
            input_ids = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )["input_ids"].to(model.device)

            seq_len = input_ids.shape[1]
            if seq_len < 10:
                continue

            accuracies = []
            ce_losses = []

            for _ in range(mc_samples):
                num_mask = max(1, int(seq_len * mask_ratio))
                mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
                positions = torch.randperm(seq_len, device=input_ids.device)[:num_mask]
                mask_positions[0, positions] = True

                masked_ids = input_ids.clone()
                masked_ids[mask_positions] = MASK_ID

                with torch.no_grad():
                    logits = model(masked_ids).logits
                    predictions = logits.argmax(dim=-1)

                    original_tokens = input_ids[mask_positions]
                    predicted_tokens = predictions[mask_positions]
                    correct = (predicted_tokens == original_tokens).float()
                    accuracies.append(correct.mean().item())

                    ce_loss = F.cross_entropy(
                        logits[mask_positions],
                        input_ids[mask_positions],
                    ).item()
                    ce_losses.append(ce_loss)

            errors = [1.0 - a for a in accuracies]

            results.append({
                "text_idx": idx,
                "label": labels[idx],
                "error_mean": np.mean(errors),
                "error_std": np.std(errors),
                "error_ci_lower": np.percentile(errors, 2.5),
                "error_ci_upper": np.percentile(errors, 97.5),
                "accuracy_mean": np.mean(accuracies),
                "ce_loss_mean": np.mean(ce_losses),
                "mc_samples": mc_samples,
            })

        except Exception as e:
            print(f"Error processing text {idx}: {e}")
            continue

    volume.commit()
    return results


@app.function(
    image=image.pip_install("openai", "anthropic"),
    timeout=7200,
    volumes={"/vol": volume},
)
def generate_ai_texts_batch(
    num_samples: int = 100,
    models: list[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    openai_key: str = None,
    anthropic_key: str = None,
):
    """
    Generate AI texts from multiple models using OpenAI and Anthropic APIs.

    Args:
        num_samples: Number of samples per model
        models: List of model names
        temperature: Sampling temperature
        max_tokens: Maximum tokens per generation

    Returns:
        Dict mapping model name to list of generated texts
    """
    import os
    import json
    from tqdm import tqdm

    if models is None:
        models = [
            "gpt-5.2",           # OpenAI flagship
            "gpt-5-mini",        # OpenAI efficient
            "claude-sonnet-4-5-20250929",  # Anthropic balanced
            "claude-haiku-4-5-20251009",   # Anthropic fast
        ]

    # Diverse prompts for generation
    prompt_templates = [
        "Explain the concept of {topic} in detail.",
        "Write a short essay about {topic}.",
        "What are the key aspects of {topic}?",
        "Describe {topic} and its importance.",
        "Discuss the future of {topic}.",
    ]

    topics = [
        "artificial intelligence", "climate change", "space exploration",
        "quantum computing", "renewable energy", "genetic engineering",
        "blockchain technology", "cybersecurity", "virtual reality",
        "machine learning", "sustainable development", "digital privacy",
        "autonomous vehicles", "biotechnology", "nanotechnology",
        "global economics", "social media", "education reform",
        "healthcare innovation", "urban planning", "democracy",
        "philosophy of mind", "modern art", "scientific method",
    ]

    # Generate prompts
    prompts = []
    for i in range(num_samples):
        template = prompt_templates[i % len(prompt_templates)]
        topic = topics[i % len(topics)]
        prompts.append(template.format(topic=topic))

    # Get API keys from parameters or environment
    openai_api_key = openai_key or os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")

    results = {}

    for model in models:
        print(f"\n{'='*50}")
        print(f"Generating {num_samples} samples from {model}")
        print(f"{'='*50}")

        model_results = []

        try:
            if model.startswith("gpt") or model.startswith("o1"):
                from openai import OpenAI
                if not openai_api_key:
                    print(f"ERROR: No OpenAI API key. Set OPENAI_API_KEY or create 'api-keys' secret.")
                    continue
                client = OpenAI(api_key=openai_api_key)

                for prompt in tqdm(prompts, desc=model):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant. Provide detailed, well-written responses."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        text = response.choices[0].message.content
                        if text and len(text.strip()) > 50:
                            model_results.append(text)
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

            elif model.startswith("claude"):
                from anthropic import Anthropic
                if not anthropic_api_key:
                    print(f"ERROR: No Anthropic API key. Set ANTHROPIC_API_KEY or create 'api-keys' secret.")
                    continue
                client = Anthropic(api_key=anthropic_api_key)

                for prompt in tqdm(prompts, desc=model):
                    try:
                        response = client.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system="You are a helpful assistant. Provide detailed, well-written responses.",
                            messages=[{"role": "user", "content": prompt}],
                        )
                        text = response.content[0].text
                        if text and len(text.strip()) > 50:
                            model_results.append(text)
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

            print(f"Generated {len(model_results)} samples from {model}")
            results[model] = model_results

        except Exception as e:
            print(f"Failed to generate from {model}: {e}")
            results[model] = []

    # Save to volume
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "generated_ai_texts.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()
    print(f"\nSaved generated texts to {output_path}")

    return results


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={"/vol": volume},
    memory=32768,
)
def run_full_experiment(
    num_samples: int = 500,
    ai_sources: list[str] = None,
    mask_ratios: list[float] = None,
):
    """
    Run full experiment comparing DIRE with baselines.

    This is the main experiment function for the paper.
    """
    import os
    import json
    from datetime import datetime

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if ai_sources is None:
        ai_sources = ["hc3"]

    if mask_ratios is None:
        mask_ratios = [0.3, 0.5, 0.7]

    timestamp = datetime.now().isoformat()

    print("=" * 60)
    print("TEXT-DIRE FULL EXPERIMENT")
    print("=" * 60)

    # Load datasets
    print("\n[1/4] Loading datasets...")
    data = load_datasets.remote(num_samples=num_samples)

    human_texts = data["human_texts"]
    ai_texts = data["ai_texts"]
    all_texts = human_texts + ai_texts
    all_labels = [0] * len(human_texts) + [1] * len(ai_texts)

    # DIRE scores
    print("\n[2/4] Computing DIRE scores...")
    dire_results = compute_dire_scores_batch.remote(all_texts, all_labels, mask_ratios)

    # Perplexity baseline
    print("\n[3/4] Computing perplexity baseline...")
    perplexity_results = compute_perplexity_baseline.remote(all_texts, all_labels)

    # Evaluation
    print("\n[4/4] Evaluating and creating visualizations...")
    eval_results = evaluate_and_plot.remote(dire_results, perplexity_results, mask_ratios)

    # Save comprehensive results
    results = {
        "timestamp": timestamp,
        "config": {
            "num_samples": num_samples,
            "ai_sources": ai_sources,
            "mask_ratios": mask_ratios,
        },
        "dire_results": dire_results,
        "perplexity_results": perplexity_results,
        "evaluation": eval_results,
    }

    results_path = os.path.join(RESULTS_DIR, f"full_experiment_{timestamp.replace(':', '-')}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    volume.commit()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {results_path}")

    return eval_results


@app.function(
    image=image.pip_install("openai", "anthropic"),
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={"/vol": volume},
    memory=32768,
)
def run_modern_ai_experiment(
    num_samples: int = 100,
    models: list[str] = None,
    mask_ratios: list[float] = None,
    openai_key: str = None,
    anthropic_key: str = None,
):
    """
    Full experiment: Generate AI text from modern models and run DIRE detection.

    Args:
        num_samples: Samples per AI model
        models: AI models to test
        mask_ratios: DIRE mask ratios

    Returns:
        Results dictionary with per-model AUROC scores
    """
    import os
    import json
    import torch
    import numpy as np
    from datetime import datetime
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    if models is None:
        models = [
            "gpt-5.2",
            "gpt-5-mini",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251009",
        ]

    if mask_ratios is None:
        mask_ratios = [0.3, 0.5, 0.7]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("TEXT-DIRE: Modern AI Models Experiment")
    print("=" * 60)

    # Step 1: Generate AI texts
    print("\n[Step 1/4] Generating AI texts from modern models...")
    ai_texts_by_model = generate_ai_texts_batch.remote(
        num_samples=num_samples,
        models=models,
        openai_key=openai_key,
        anthropic_key=anthropic_key,
    )

    # Step 2: Load human texts
    print("\n[Step 2/4] Loading human texts...")
    from datasets import load_dataset

    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    human_texts = [
        t.strip() for t in wiki["text"]
        if t and len(t.split()) > 50 and not t.startswith("=")
    ]

    # Match total AI samples
    total_ai = sum(len(texts) for texts in ai_texts_by_model.values())
    human_texts = human_texts[:total_ai]
    print(f"Loaded {len(human_texts)} human samples")

    # Step 3: Load DIRE model
    print("\n[Step 3/4] Loading LLaDA-8B for DIRE...")
    MASK_ID = 126336

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

    print("LLaDA-8B loaded!")

    # Step 4: Compute DIRE scores for each model
    print("\n[Step 4/4] Computing DIRE scores...")

    results = {}

    for ai_model, ai_texts in ai_texts_by_model.items():
        if not ai_texts:
            print(f"Skipping {ai_model} (no samples)")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {ai_model} ({len(ai_texts)} samples)")
        print(f"{'='*50}")

        # Balance with human texts
        n_samples = min(len(human_texts), len(ai_texts))
        eval_human = human_texts[:n_samples]
        eval_ai = ai_texts[:n_samples]

        all_texts = eval_human + eval_ai
        all_labels = [0] * len(eval_human) + [1] * len(eval_ai)

        model_results = {"n_samples": n_samples}

        for mask_ratio in mask_ratios:
            print(f"  Mask ratio {mask_ratio}...")

            scores = []
            for text in tqdm(all_texts, desc=f"DIRE-{mask_ratio}"):
                if not text or len(text.strip()) < 20:
                    scores.append(0.5)
                    continue

                try:
                    input_ids = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )["input_ids"].to("cuda")

                    seq_len = input_ids.shape[1]
                    if seq_len < 5:
                        scores.append(0.5)
                        continue

                    num_mask = max(1, int(seq_len * mask_ratio))
                    mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
                    positions = torch.randperm(seq_len, device=input_ids.device)[:num_mask]
                    mask_positions[0, positions] = True

                    masked_ids = input_ids.clone()
                    masked_ids[mask_positions] = MASK_ID

                    with torch.no_grad():
                        logits = model(masked_ids).logits
                        predictions = logits.argmax(dim=-1)

                        original_tokens = input_ids[mask_positions]
                        predicted_tokens = predictions[mask_positions]
                        correct = (predicted_tokens == original_tokens).float()
                        accuracy = correct.mean().item()
                        scores.append(1.0 - accuracy)  # Error

                except Exception as e:
                    scores.append(0.5)
                    continue

            # Compute metrics
            human_errors = scores[:len(eval_human)]
            ai_errors = scores[len(eval_human):]

            auroc = roc_auc_score(all_labels, scores)
            if auroc < 0.5:
                auroc = 1 - auroc

            model_results[f"mask_{mask_ratio}"] = {
                "auroc": auroc,
                "human_mean": np.mean(human_errors),
                "human_std": np.std(human_errors),
                "ai_mean": np.mean(ai_errors),
                "ai_std": np.std(ai_errors),
            }

            print(f"    AUROC: {auroc:.4f}")
            print(f"    Human error: {np.mean(human_errors):.4f} (+/- {np.std(human_errors):.4f})")
            print(f"    AI error: {np.mean(ai_errors):.4f} (+/- {np.std(ai_errors):.4f})")

        results[ai_model] = model_results

    # Save results
    output_path = os.path.join(RESULTS_DIR, f"modern_ai_results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'DIRE-0.3':<12} {'DIRE-0.5':<12} {'DIRE-0.7':<12}")
    print("-" * 70)

    for ai_model, res in results.items():
        row = f"{ai_model:<35}"
        for mr in mask_ratios:
            key = f"mask_{mr}"
            if key in res:
                row += f"{res[key]['auroc']:.4f}      "
            else:
                row += "N/A         "
        print(row)

    print("=" * 70)
    print(f"\nResults saved to: {output_path}")

    return results


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={"/vol": volume},
    memory=32768,
)
def run_beemo_multistep(
    scenarios: list[str] = None,
    mask_ratio: float = 0.8,
    num_steps: int = 3,
    confidence_threshold: float = 0.5,
    max_samples: int = None,
):
    """
    Run Beemo benchmark with MULTI-STEP diffusion DIRE.

    Instead of single-step mask->predict, this iteratively:
    1. Mask tokens
    2. Predict
    3. Re-mask low-confidence predictions
    4. Predict again
    5. Repeat for num_steps

    This mimics actual diffusion model inference more closely.

    Args:
        scenarios: List of scenarios to evaluate
        mask_ratio: Initial mask ratio
        num_steps: Number of diffusion steps
        confidence_threshold: Re-mask predictions below this confidence
        max_samples: Limit samples (None = use all)
    """
    import os
    import json
    import torch
    import torch.nn.functional as F
    import numpy as np
    from datetime import datetime
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_dataset

    if scenarios is None:
        scenarios = ["easy", "medium", "hard"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("TEXT-DIRE: Multi-Step Diffusion Beemo Evaluation")
    print(f"Steps: {num_steps}, Confidence threshold: {confidence_threshold}")
    print("=" * 60)

    # Load Beemo dataset
    print("\n[1/3] Loading Beemo dataset...")
    dataset = load_dataset("toloka/beemo")
    df = dataset["train"].to_pandas()

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

    print(f"Loaded {len(df)} samples")

    # Load LLaDA model
    print("\n[2/3] Loading LLaDA-8B...")
    MASK_ID = 126336

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

    print("LLaDA-8B loaded!")

    def compute_multistep_dire_score(text):
        """
        Multi-step DIRE: iteratively refine predictions.

        Step 1: Mask tokens, predict
        Step 2+: Re-mask low-confidence predictions, predict again
        Final: Score based on final predictions vs original
        """
        if not text or len(str(text).strip()) < 10:
            return {"accuracy": 0.5, "ce_loss": 5.0, "avg_confidence": 0.5}

        try:
            input_ids = tokenizer(
                str(text),
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )["input_ids"].to("cuda")

            seq_len = input_ids.shape[1]
            if seq_len < 5:
                return {"accuracy": 0.5, "ce_loss": 5.0, "avg_confidence": 0.5}

            # Initial masking
            num_mask = max(1, int(seq_len * mask_ratio))
            mask_positions = torch.zeros(seq_len, dtype=torch.bool, device="cuda")
            positions = torch.randperm(seq_len, device="cuda")[:num_mask]
            mask_positions[positions] = True

            # Track which positions are still masked
            current_masked = mask_positions.clone()
            current_ids = input_ids.clone()
            current_ids[0, current_masked] = MASK_ID

            all_confidences = []

            for step in range(num_steps):
                with torch.no_grad():
                    logits = model(current_ids).logits
                    probs = F.softmax(logits, dim=-1)
                    predictions = logits.argmax(dim=-1)

                    # Get confidence for each masked position
                    if current_masked.sum() > 0:
                        masked_probs = probs[0, current_masked]
                        masked_preds = predictions[0, current_masked]

                        # Confidence = probability of predicted token
                        confidences = masked_probs[
                            torch.arange(masked_preds.size(0), device="cuda"),
                            masked_preds
                        ]
                        all_confidences.append(confidences.mean().item())

                        if step < num_steps - 1:
                            # Fill in high-confidence predictions
                            high_conf_mask = confidences >= confidence_threshold

                            # Get indices of currently masked positions
                            masked_indices = current_masked.nonzero(as_tuple=True)[0]

                            # Update current_ids with high-confidence predictions
                            for idx, (pos, conf) in enumerate(zip(masked_indices, confidences)):
                                if conf >= confidence_threshold:
                                    current_ids[0, pos] = masked_preds[idx]
                                    current_masked[pos] = False

                            # If all positions filled, break early
                            if current_masked.sum() == 0:
                                break

            # Final evaluation: compare final predictions to original
            with torch.no_grad():
                logits = model(current_ids).logits
                probs = F.softmax(logits, dim=-1)
                predictions = logits.argmax(dim=-1)

                # Score on ALL originally masked positions
                original_tokens = input_ids[0, mask_positions]
                predicted_tokens = predictions[0, mask_positions]

                correct = (predicted_tokens == original_tokens).float()
                accuracy = correct.mean().item()

                # Cross-entropy loss
                ce_loss = F.cross_entropy(
                    logits[0, mask_positions],
                    input_ids[0, mask_positions]
                ).item()

                # Average confidence across steps
                avg_confidence = np.mean(all_confidences) if all_confidences else 0.5

            return {
                "accuracy": accuracy,
                "ce_loss": ce_loss,
                "avg_confidence": avg_confidence,
            }

        except Exception as e:
            print(f"Error: {e}")
            return {"accuracy": 0.5, "ce_loss": 5.0, "avg_confidence": 0.5}

    # Evaluate each scenario
    print(f"\n[3/3] Evaluating scenarios with {num_steps}-step diffusion...")
    results = {}

    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario.upper()}")
        print('='*50)

        texts = []
        labels = []

        if scenario == "easy":
            for _, row in df.iterrows():
                if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                    texts.append(str(row["human_output"]))
                    labels.append(0)
                if row["model_output"] and len(str(row["model_output"]).strip()) > 10:
                    texts.append(str(row["model_output"]))
                    labels.append(1)

        elif scenario == "medium":
            for _, row in df.iterrows():
                if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                    texts.append(str(row["human_output"]))
                    labels.append(0)
                if row["gpt-4o_edits"] and len(str(row["gpt-4o_edits"]).strip()) > 10:
                    texts.append(str(row["gpt-4o_edits"]))
                    labels.append(1)

        elif scenario == "hard":
            for _, row in df.iterrows():
                if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                    texts.append(str(row["human_output"]))
                    labels.append(0)
                if row["human_edits"] and len(str(row["human_edits"]).strip()) > 10:
                    texts.append(str(row["human_edits"]))
                    labels.append(1)

        print(f"Samples: {len(texts)} ({sum(1 for l in labels if l == 0)} human, {sum(labels)} AI)")

        if len(texts) < 20:
            print(f"Skipping {scenario} - too few samples")
            continue

        # Compute multi-step DIRE scores
        all_results = []
        for text in tqdm(texts, desc=f"Multi-step DIRE on {scenario}"):
            all_results.append(compute_multistep_dire_score(text))

        labels = np.array(labels)

        # Try both accuracy and CE loss as scores
        accuracy_scores = np.array([r["accuracy"] for r in all_results])
        ce_scores = np.array([r["ce_loss"] for r in all_results])
        confidence_scores = np.array([r["avg_confidence"] for r in all_results])

        # Evaluate accuracy-based
        auroc_acc = roc_auc_score(labels, accuracy_scores)
        if auroc_acc < 0.5:
            accuracy_scores = 1 - accuracy_scores
            auroc_acc = 1 - auroc_acc

        # Evaluate CE loss-based (lower loss = more AI-like)
        auroc_ce = roc_auc_score(labels, -ce_scores)  # Negate: lower loss -> higher score
        if auroc_ce < 0.5:
            auroc_ce = 1 - auroc_ce

        # Find best metric
        best_metric = "accuracy" if auroc_acc >= auroc_ce else "ce_loss"
        best_auroc = max(auroc_acc, auroc_ce)

        scores = accuracy_scores if best_metric == "accuracy" else -ce_scores

        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        threshold = thresholds[optimal_idx]

        predictions = (scores >= threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        # Score distributions
        human_scores = scores[labels == 0]
        ai_scores = scores[labels == 1]

        pooled_std = np.sqrt(
            ((len(human_scores) - 1) * np.std(human_scores, ddof=1)**2 +
             (len(ai_scores) - 1) * np.std(ai_scores, ddof=1)**2) /
            (len(human_scores) + len(ai_scores) - 2)
        )
        cohens_d = abs(np.mean(ai_scores) - np.mean(human_scores)) / pooled_std if pooled_std > 0 else 0

        results[scenario] = {
            "auroc_accuracy": float(auroc_acc),
            "auroc_ce_loss": float(auroc_ce),
            "best_auroc": float(best_auroc),
            "best_metric": best_metric,
            "accuracy": float(accuracy),
            "f1": float(f1),
            "n_samples": len(labels),
            "human_mean": float(np.mean(human_scores)),
            "ai_mean": float(np.mean(ai_scores)),
            "separation": float(cohens_d),
            "num_steps": num_steps,
            "confidence_threshold": confidence_threshold,
        }

        print(f"  AUROC (accuracy):  {auroc_acc:.4f}")
        print(f"  AUROC (CE loss):   {auroc_ce:.4f}")
        print(f"  Best metric:       {best_metric} ({best_auroc:.4f})")
        print(f"  Accuracy:          {accuracy:.4f}")
        print(f"  F1:                {f1:.4f}")
        print(f"  Separation:        {cohens_d:.2f}")

    # Save results
    output_path = os.path.join(RESULTS_DIR, f"beemo_multistep_{num_steps}steps_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()

    # Print summary
    print("\n" + "=" * 70)
    print(f"BEEMO MULTI-STEP DIRE RESULTS ({num_steps} steps)")
    print("=" * 70)
    print(f"{'Scenario':<15} {'AUROC(acc)':<12} {'AUROC(CE)':<12} {'Best':<12} {'Separation':<12}")
    print("-" * 70)

    for scenario, r in results.items():
        print(f"{scenario:<15} {r['auroc_accuracy']:<12.4f} {r['auroc_ce_loss']:<12.4f} "
              f"{r['best_auroc']:<12.4f} {r['separation']:<12.2f}")

    print("=" * 70)
    print(f"\nResults saved to: {output_path}")

    return results


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={"/vol": volume},
    memory=32768,
)
def run_beemo_benchmark(
    scenarios: list[str] = None,
    mask_ratio: float = 0.5,
    max_samples: int = None,
):
    """
    Run Beemo benchmark (NAACL 2025) for AI text detection.

    Beemo tests on edited AI text - a harder and more realistic scenario.
    Paper: arxiv.org/abs/2411.04032

    Scenarios:
        - easy: human vs raw AI output
        - medium: human vs GPT-4o edited AI
        - hard: human vs human-edited AI

    Usage:
        modal run modal_app.py --experiment beemo
    """
    import os
    import json
    import torch
    import numpy as np
    from datetime import datetime
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_dataset

    if scenarios is None:
        scenarios = ["easy", "medium", "hard"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("TEXT-DIRE: Beemo Benchmark Evaluation")
    print("=" * 60)

    # Load Beemo dataset
    print("\n[1/3] Loading Beemo dataset from HuggingFace...")
    dataset = load_dataset("toloka/beemo")
    df = dataset["train"].to_pandas()

    if max_samples:
        df = df.head(max_samples)

    print(f"Loaded {len(df)} samples")
    print(f"Categories: {df['category'].unique().tolist()}")
    print(f"Models: {df['model'].unique().tolist()}")

    # Load LLaDA model
    print("\n[2/3] Loading LLaDA-8B for DIRE...")
    MASK_ID = 126336

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

    print("LLaDA-8B loaded!")

    def compute_dire_score(text):
        """Compute DIRE score for a single text."""
        if not text or len(str(text).strip()) < 10:
            return 0.5

        try:
            input_ids = tokenizer(
                str(text),
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )["input_ids"].to("cuda")

            seq_len = input_ids.shape[1]
            if seq_len < 5:
                return 0.5

            num_mask = max(1, int(seq_len * mask_ratio))
            mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
            positions = torch.randperm(seq_len, device=input_ids.device)[:num_mask]
            mask_positions[0, positions] = True

            masked_ids = input_ids.clone()
            masked_ids[mask_positions] = MASK_ID

            with torch.no_grad():
                logits = model(masked_ids).logits
                predictions = logits.argmax(dim=-1)

                original_tokens = input_ids[mask_positions]
                predicted_tokens = predictions[mask_positions]
                correct = (predicted_tokens == original_tokens).float()
                accuracy = correct.mean().item()

            # Higher accuracy = more likely AI (reconstructs well)
            return accuracy

        except Exception as e:
            return 0.5

    # Evaluate each scenario
    print("\n[3/3] Evaluating scenarios...")
    results = {}

    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario.upper()}")
        print('='*50)

        texts = []
        labels = []

        if scenario == "easy":
            # Human vs raw model output
            for _, row in df.iterrows():
                if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                    texts.append(str(row["human_output"]))
                    labels.append(0)
                if row["model_output"] and len(str(row["model_output"]).strip()) > 10:
                    texts.append(str(row["model_output"]))
                    labels.append(1)

        elif scenario == "medium":
            # Human vs GPT-4o edited
            for _, row in df.iterrows():
                if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                    texts.append(str(row["human_output"]))
                    labels.append(0)
                if row["gpt-4o_edits"] and len(str(row["gpt-4o_edits"]).strip()) > 10:
                    texts.append(str(row["gpt-4o_edits"]))
                    labels.append(1)

        elif scenario == "hard":
            # Human vs expert-edited AI (hardest)
            for _, row in df.iterrows():
                if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                    texts.append(str(row["human_output"]))
                    labels.append(0)
                if row["human_edits"] and len(str(row["human_edits"]).strip()) > 10:
                    texts.append(str(row["human_edits"]))
                    labels.append(1)

        print(f"Samples: {len(texts)} ({sum(1 for l in labels if l == 0)} human, {sum(labels)} AI)")

        if len(texts) < 20:
            print(f"Skipping {scenario} - too few samples")
            continue

        # Compute DIRE scores
        scores = []
        for text in tqdm(texts, desc=f"DIRE on {scenario}"):
            scores.append(compute_dire_score(text))

        labels = np.array(labels)
        scores = np.array(scores)

        # Compute metrics
        auroc = roc_auc_score(labels, scores)
        if auroc < 0.5:
            scores = 1 - scores
            auroc = 1 - auroc

        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        threshold = thresholds[optimal_idx]

        predictions = (scores >= threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        # Score distributions
        human_scores = scores[labels == 0]
        ai_scores = scores[labels == 1]

        # Effect size
        pooled_std = np.sqrt(
            ((len(human_scores) - 1) * np.std(human_scores, ddof=1)**2 +
             (len(ai_scores) - 1) * np.std(ai_scores, ddof=1)**2) /
            (len(human_scores) + len(ai_scores) - 2)
        )
        cohens_d = abs(np.mean(ai_scores) - np.mean(human_scores)) / pooled_std if pooled_std > 0 else 0

        results[scenario] = {
            "auroc": float(auroc),
            "accuracy": float(accuracy),
            "f1": float(f1),
            "n_samples": len(labels),
            "human_mean": float(np.mean(human_scores)),
            "human_std": float(np.std(human_scores)),
            "ai_mean": float(np.mean(ai_scores)),
            "ai_std": float(np.std(ai_scores)),
            "separation": float(cohens_d),
        }

        print(f"  AUROC:      {auroc:.4f}")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  F1:         {f1:.4f}")
        print(f"  Human mean: {np.mean(human_scores):.4f}")
        print(f"  AI mean:    {np.mean(ai_scores):.4f}")
        print(f"  Separation: {cohens_d:.2f}")

    # Save results
    output_path = os.path.join(RESULTS_DIR, f"beemo_results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()

    # Print summary table
    print("\n" + "=" * 70)
    print("BEEMO BENCHMARK RESULTS - TEXT-DIRE")
    print("=" * 70)
    print(f"{'Scenario':<15} {'AUROC':<10} {'Accuracy':<10} {'F1':<10} {'Separation':<12}")
    print("-" * 70)

    for scenario, r in results.items():
        print(f"{scenario:<15} {r['auroc']:<10.4f} {r['accuracy']:<10.4f} "
              f"{r['f1']:<10.4f} {r['separation']:<12.2f}")

    print("=" * 70)
    print(f"\nResults saved to: {output_path}")

    return results


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={"/vol": volume},
    memory=32768,
)
def run_beemo_by_model(
    mask_ratio: float = 0.5,
    max_samples: int = None,
):
    """
    Run Beemo evaluation broken down by SOURCE MODEL.
    
    This helps understand why EASY (diverse models) vs MEDIUM (GPT-4o edited)
    show different performance - by showing per-model AUROC.
    
    Usage:
        modal run modal_app.py --experiment beemo-by-model
    """
    import os
    import json
    import torch
    import numpy as np
    from datetime import datetime
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_dataset

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("TEXT-DIRE: Beemo Per-Model Analysis")
    print("=" * 60)

    # Load Beemo dataset
    print("\n[1/3] Loading Beemo dataset from HuggingFace...")
    dataset = load_dataset("toloka/beemo")
    df = dataset["train"].to_pandas()

    if max_samples:
        df = df.head(max_samples)

    print(f"Loaded {len(df)} samples")
    models_in_data = df["model"].unique().tolist()
    print(f"Source models in dataset: {models_in_data}")
    print(f"Categories: {df['category'].unique().tolist()}")

    # Load LLaDA model
    print("\n[2/3] Loading LLaDA-8B for DIRE...")
    MASK_ID = 126336

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

    print("LLaDA-8B loaded!")

    def compute_dire_score(text):
        """Compute DIRE score for a single text."""
        if not text or len(str(text).strip()) < 10:
            return 0.5

        try:
            input_ids = tokenizer(
                str(text),
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )["input_ids"].to("cuda")

            seq_len = input_ids.shape[1]
            if seq_len < 5:
                return 0.5

            num_mask = max(1, int(seq_len * mask_ratio))
            mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
            positions = torch.randperm(seq_len, device=input_ids.device)[:num_mask]
            mask_positions[0, positions] = True

            masked_ids = input_ids.clone()
            masked_ids[mask_positions] = MASK_ID

            with torch.no_grad():
                logits = model(masked_ids).logits
                predictions = logits.argmax(dim=-1)

                original_tokens = input_ids[mask_positions]
                predicted_tokens = predictions[mask_positions]
                correct = (predicted_tokens == original_tokens).float()
                accuracy = correct.mean().item()

            # Higher accuracy = more likely AI (reconstructs well)
            return accuracy

        except Exception as e:
            return 0.5

    # Evaluate by source model
    print("\n[3/3] Evaluating by source model...")
    results_by_model = {}

    for source_model in models_in_data:
        print(f"\n{'='*50}")
        print(f"Evaluating source model: {source_model}")
        print('='*50)

        subset = df[df["model"] == source_model]

        texts = []
        labels = []
        sources = []

        for _, row in subset.iterrows():
            # Human text
            if row["human_output"] and len(str(row["human_output"]).strip()) > 10:
                texts.append(str(row["human_output"]))
                labels.append(0)
                sources.append("human")

            # Model output (raw AI)
            if row["model_output"] and len(str(row["model_output"]).strip()) > 10:
                texts.append(str(row["model_output"]))
                labels.append(1)
                sources.append(source_model)

        print(f"Samples: {len(texts)} ({sum(1 for l in labels if l == 0)} human, {sum(labels)} AI)")

        if len(texts) < 20:
            print(f"Skipping {source_model} - too few samples")
            continue

        # Compute DIRE scores
        scores = []
        for text in tqdm(texts, desc=f"DIRE on {source_model}"):
            scores.append(compute_dire_score(text))

        labels = np.array(labels)
        scores = np.array(scores)

        # Compute metrics
        auroc = roc_auc_score(labels, scores)
        if auroc < 0.5:
            scores = 1 - scores
            auroc = 1 - auroc

        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        threshold = thresholds[optimal_idx]

        predictions = (scores >= threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        # Score distributions
        human_scores = scores[labels == 0]
        ai_scores = scores[labels == 1]

        # Effect size
        pooled_std = np.sqrt(
            ((len(human_scores) - 1) * np.std(human_scores, ddof=1)**2 +
             (len(ai_scores) - 1) * np.std(ai_scores, ddof=1)**2) /
            (len(human_scores) + len(ai_scores) - 2)
        )
        cohens_d = abs(np.mean(ai_scores) - np.mean(human_scores)) / pooled_std if pooled_std > 0 else 0

        results_by_model[source_model] = {
            "auroc": float(auroc),
            "accuracy": float(accuracy),
            "f1": float(f1),
            "n_samples": len(labels),
            "human_mean": float(np.mean(human_scores)),
            "human_std": float(np.std(human_scores)),
            "ai_mean": float(np.mean(ai_scores)),
            "ai_std": float(np.std(ai_scores)),
            "separation": float(cohens_d),
        }

        print(f"  AUROC:      {auroc:.4f}")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  F1:         {f1:.4f}")
        print(f"  Human mean: {np.mean(human_scores):.4f}")
        print(f"  AI mean:    {np.mean(ai_scores):.4f}")
        print(f"  Separation: {cohens_d:.2f}")

    # Save results
    output_path = os.path.join(RESULTS_DIR, f"beemo_by_model_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results_by_model, f, indent=2)

    volume.commit()

    # Print summary table sorted by AUROC
    print("\n" + "=" * 80)
    print("BEEMO BY SOURCE MODEL - SORTED BY AUROC")
    print("=" * 80)
    print(f"{'Model':<25} {'AUROC':<10} {'AI Mean':<12} {'Human Mean':<12} {'Separation':<12} {'N':<8}")
    print("-" * 80)

    for model_name, r in sorted(results_by_model.items(), key=lambda x: x[1]["auroc"], reverse=True):
        print(f"{model_name:<25} {r['auroc']:<10.4f} {r['ai_mean']:<12.4f} "
              f"{r['human_mean']:<12.4f} {r['separation']:<12.2f} {r['n_samples']:<8}")

    print("=" * 80)
    print(f"\nResults saved to: {output_path}")

    # Print analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    best_model = max(results_by_model.items(), key=lambda x: x[1]["auroc"])
    worst_model = min(results_by_model.items(), key=lambda x: x[1]["auroc"])
    
    print(f"\nBest detected model:  {best_model[0]} (AUROC: {best_model[1]['auroc']:.4f})")
    print(f"Worst detected model: {worst_model[0]} (AUROC: {worst_model[1]['auroc']:.4f})")
    print(f"\nAUROC range: {worst_model[1]['auroc']:.4f} - {best_model[1]['auroc']:.4f}")
    
    avg_auroc = np.mean([r["auroc"] for r in results_by_model.values()])
    print(f"Average AUROC across models: {avg_auroc:.4f}")

    return results_by_model


@app.local_entrypoint()
def main(
    num_samples: int = 100,
    experiment: str = "basic",
    mc_samples: int = 32,
    models: str = None,
    openai_key: str = None,
    anthropic_key: str = None,
    mask_ratio: float = 0.5,
):
    """
    Entry point for modal run command.

    Usage:
        modal run modal_app.py
        modal run modal_app.py --num-samples 50
        modal run modal_app.py --experiment full --num-samples 500
        modal run modal_app.py --experiment modern --num-samples 100 --openai-key YOUR_KEY --anthropic-key YOUR_KEY
        modal run modal_app.py --experiment mc --mc-samples 64
        modal run modal_app.py --experiment beemo
        modal run modal_app.py --experiment beemo --mask-ratio 0.7
        modal run modal_app.py --experiment beemo-by-model
        modal run modal_app.py --experiment beemo-multistep --num-samples 200

    Args:
        num_samples: Number of samples per class
        experiment: Experiment type (basic, full, modern, mc, beemo, beemo-by-model, beemo-multistep)
        mc_samples: MC samples for mc experiment
        models: Comma-separated list of models for modern experiment
        openai_key: OpenAI API key (required for modern experiment)
        anthropic_key: Anthropic API key (required for modern experiment)
        mask_ratio: DIRE mask ratio (default 0.5, try 0.7 for harder reconstruction)
    """
    import os

    # Try to load from .env file first
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed locally, that's fine

    # Try to load keys from environment if not provided
    if not openai_key:
        openai_key = os.environ.get("OPENAI_API_KEY")
    if not anthropic_key:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    print("Starting Text-DIRE experiment on Modal...")
    print(f"Experiment type: {experiment}")
    print(f"Using {num_samples} samples per class")

    if experiment == "beemo-by-model":
        print(f"\nRunning Beemo per-source-model analysis (mask_ratio={mask_ratio})...")
        results = run_beemo_by_model.remote(
            mask_ratio=mask_ratio,
            max_samples=num_samples if num_samples != 100 else None,
        )

        print("\n" + "=" * 60)
        print("BEEMO BY-MODEL ANALYSIS COMPLETE")
        print("=" * 60)

        # Results are already printed by the function
        return results

    elif experiment == "beemo":
        print(f"\nRunning Beemo benchmark evaluation (mask_ratio={mask_ratio})...")
        results = run_beemo_benchmark.remote(
            scenarios=["easy", "medium", "hard"],
            mask_ratio=mask_ratio,
            max_samples=num_samples if num_samples != 100 else None,  # None = use all
        )

        print("\n" + "=" * 60)
        print("BEEMO BENCHMARK COMPLETE")
        print("=" * 60)

        for scenario, metrics in results.items():
            print(f"\n{scenario.upper()}:")
            print(f"  AUROC: {metrics['auroc']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")

        return results

    elif experiment == "beemo-multistep":
        print(f"\nRunning Beemo with MULTI-STEP diffusion (mask_ratio={mask_ratio})...")
        results = run_beemo_multistep.remote(
            scenarios=["easy", "medium", "hard"],
            mask_ratio=mask_ratio,
            num_steps=3,  # 3 diffusion steps
            confidence_threshold=0.5,
            max_samples=num_samples if num_samples != 100 else None,
        )

        print("\n" + "=" * 60)
        print("BEEMO MULTI-STEP COMPLETE")
        print("=" * 60)

        for scenario, metrics in results.items():
            print(f"\n{scenario.upper()}:")
            print(f"  AUROC (accuracy): {metrics['auroc_accuracy']:.4f}")
            print(f"  AUROC (CE loss):  {metrics['auroc_ce_loss']:.4f}")
            print(f"  Best AUROC:       {metrics['best_auroc']:.4f}")
            print(f"  Best metric:      {metrics['best_metric']}")

        return results

    elif experiment == "modern":
        # Parse models if provided
        model_list = None
        if models:
            model_list = [m.strip() for m in models.split(",")]

        # Check for API keys
        if not openai_key and not anthropic_key:
            print("WARNING: No API keys provided. Set OPENAI_API_KEY/ANTHROPIC_API_KEY in environment")
            print("or pass --openai-key and --anthropic-key arguments.")

        results = run_modern_ai_experiment.remote(
            num_samples=num_samples,
            models=model_list,
            openai_key=openai_key,
            anthropic_key=anthropic_key,
        )

        print("\n" + "=" * 60)
        print("MODERN AI EXPERIMENT COMPLETE")
        print("=" * 60)

        for model, metrics in results.items():
            print(f"\n{model}:")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    print(f"  {key}: AUROC={value.get('auroc', 'N/A'):.4f}")
                else:
                    print(f"  {key}: {value}")

        return results

    elif experiment == "full":
        results = run_full_experiment.remote(num_samples=num_samples)
    elif experiment == "mc":
        data = load_datasets.remote(num_samples=num_samples)
        all_texts = data["human_texts"] + data["ai_texts"]
        all_labels = [0] * len(data["human_texts"]) + [1] * len(data["ai_texts"])

        results = compute_dire_scores_mc.remote(
            all_texts, all_labels,
            mask_ratio=0.5,
            mc_samples=mc_samples,
        )
        print(f"\nMC DIRE results: {len(results)} samples computed")
        return results
    else:
        results = run_experiment.remote(num_samples=num_samples)

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    for method, metrics in results.items():
        print(f"\n{method}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    return results
