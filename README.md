# Text-DIRE: Diffusion Reconstruction Error for AI Text Detection

A novel AI text detector based on the DIRE (Diffusion Reconstruction Error) method. This adapts the approach that has been successful for detecting AI-generated images to the NLP domain.

## Core Hypothesis

A text diffusion model trained on human text will reconstruct human-written text more accurately than AI-generated text, because AI text lies "off-manifold" from natural human writing.

## How It Works

1. **Mask tokens**: Randomly mask X% of tokens in the input text
2. **Reconstruct**: Use LLaDA-8B (a text diffusion model) to predict the masked tokens
3. **Measure error**: Compare predictions to original tokens
4. **Classify**: Higher reconstruction error may indicate human text (hypothesis to test)

## Requirements

- Python 3.11+
- Modal account (for cloud GPU access)

## Quick Start

```bash
# Install Modal
pip install modal

# Authenticate (first time only)
modal setup

# Run the experiment (uses A100 GPU on Modal)
modal run modal_app.py

# Or with custom sample count
modal run modal_app.py --num-samples 50

# Download results
modal volume get text-dire-vol results/dire_distributions.png .
modal volume get text-dire-vol results/results_summary.txt .
```

## Web App

This repository includes a small FastAPI website for trying Text-DIRE from a
browser.

```bash
pip install -r requirements.txt
uvicorn src.web_app:app --reload
```

Open http://127.0.0.1:8000.

By default the site runs in `demo` mode so the interface works without a GPU.
Demo mode is not DIRE; it is a lightweight placeholder signal for local UI
testing. To use real GPU-backed Text-DIRE scoring through Modal:

```bash
modal setup
modal deploy modal_app.py
uvicorn src.web_app:app --reload
```

Set `TEXTDIRE_PROVIDER=modal` in `.env` before starting the server. Use
`TEXTDIRE_PROVIDER=demo` when you only want to try the interface locally.
The web app looks up the deployed Modal app `text-dire` and function
`compute_dire_scores_batch`; override these with `TEXTDIRE_MODAL_APP` and
`TEXTDIRE_MODAL_FUNCTION` if you deploy under different names.

The frontend and API do not need a GPU. The real DIRE inference worker does,
because LLaDA-8B is too large for practical CPU serving.

## Vercel Deployment

The Vercel deployment serves the static website from `web/` and uses Python
serverless functions in `api/`. Vercel does not run LLaDA-8B directly; the
serverless API calls the deployed Modal GPU function.

Deploy the GPU worker first:

```bash
modal deploy modal_app.py
```

Then set these Vercel environment variables:

```env
TEXTDIRE_PROVIDER=modal
TEXTDIRE_MODAL_APP=text-dire
TEXTDIRE_MODAL_FUNCTION=compute_dire_scores_batch
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

For a preview deployment that only exercises the UI, use:

```env
TEXTDIRE_PROVIDER=demo
```

The production request path is:

```text
Browser -> Vercel /api/analyze -> Modal compute_dire_scores_batch -> Vercel -> Browser
```

## Project Structure

```
text-dire/
├── modal_app.py           # Main Modal app with all functions
├── src/
│   ├── __init__.py
│   ├── dire.py            # Text-DIRE computation logic
│   ├── data.py            # Data loading utilities
│   ├── baselines.py       # Perplexity baseline
│   └── evaluate.py        # AUROC, plotting utilities
├── run_local.py           # Local script to trigger Modal runs
├── requirements.txt       # Local requirements
└── README.md
```

## Modal Functions

| Function | Description | GPU |
|----------|-------------|-----|
| `run_experiment()` | Full proof-of-concept experiment | A100 |
| `compute_dire_scores_batch()` | Compute DIRE scores for texts | A100 |
| `compute_perplexity_baseline()` | Compute perplexity baseline | A100 |
| `load_datasets()` | Load human/AI text samples | None |
| `evaluate_and_plot()` | Evaluate and create visualizations | None |
| `load_and_cache_model()` | Pre-cache LLaDA model | A100 |

## Datasets

- **Human text**: WikiText-103 (encyclopedic articles)
- **AI text**: HC3 (Human vs ChatGPT answers)

## Output

The experiment produces:

1. **dire_distributions.png**: Visualization of DIRE score distributions
2. **results_summary.txt**: Text summary of all metrics
3. **raw_results.json**: Raw data for further analysis

### Expected Metrics

- DIRE AUROC for each mask ratio (0.3, 0.5, 0.7)
- Perplexity baseline AUROC
- Distribution statistics (mean, std) per class

## Cost Estimate

- A100 GPU: ~$2.78/hour on Modal
- Full experiment: ~10-20 minutes
- Estimated cost: ~$0.50-1.00

## Local Testing

For testing without GPU:

```python
from src.dire import TextDIRE, compute_dire_score
from src.data import load_datasets
from src.evaluate import evaluate_detector, plot_distributions
```

## Technical Details

### LLaDA-8B

LLaDA is a mask-based diffusion model for text:
- **Forward process**: Randomly mask tokens at increasing rates
- **Reverse process**: Predict all masked tokens simultaneously
- **Model**: GSAI-ML/LLaDA-8B-Base (8B parameters)

### DIRE Computation

```python
def compute_dire_score(model, tokenizer, text, mask_ratio=0.5):
    # 1. Tokenize
    inputs = tokenizer(text, return_tensors="pt")

    # 2. Mask random tokens
    masked_ids, positions = mask_tokens(inputs.input_ids, mask_ratio)

    # 3. Model predicts masked tokens
    outputs = model(masked_ids)
    predictions = outputs.logits.argmax(dim=-1)

    # 4. Compare to original
    correct = (predictions[positions] == inputs.input_ids[positions])
    accuracy = correct.float().mean()

    return {"accuracy": accuracy, "error": 1 - accuracy}
```

## Troubleshooting

### Model loading fails
The first run downloads LLaDA-8B (~16GB). If it fails:
```bash
# Just cache the model
modal run modal_app.py::load_and_cache_model
```

### Out of memory
Reduce batch size or use smaller samples:
```bash
modal run modal_app.py --num-samples 30
```

### Modal authentication
```bash
modal setup  # Re-authenticate
```

## Research Notes

This is an experimental detector. The hypothesis that human text has higher reconstruction error is based on:

1. AI models (like GPT) produce text that is "typical" for their training distribution
2. A diffusion model trained on human text learns the human text manifold
3. AI text may lie slightly off this manifold, making it easier to reconstruct

The experiment will test whether this hypothesis holds empirically.

## Future Work

- [ ] Test with different mask ratios
- [ ] Try other diffusion models (smaller/larger)
- [ ] Test on more diverse datasets
- [ ] Combine DIRE with perplexity for ensemble
- [ ] Analyze which token types are most discriminative

## License

MIT
