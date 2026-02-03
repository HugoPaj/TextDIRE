#!/usr/bin/env python3
"""
Local script to trigger Text-DIRE experiments on Modal.

Usage:
    python run_local.py                    # Run full experiment
    python run_local.py --samples 50       # Run with 50 samples per class
    python run_local.py --cache-model      # Just cache the model
    python run_local.py --download         # Download results from Modal

Requirements:
    pip install modal
    modal setup  # First-time authentication
"""

import argparse
import subprocess
import sys


def run_experiment(num_samples: int = 100):
    """Run the full Text-DIRE experiment on Modal."""
    print("=" * 60)
    print("TEXT-DIRE: Diffusion Reconstruction Error for AI Detection")
    print("=" * 60)
    print()
    print(f"Running experiment with {num_samples} samples per class...")
    print("This will use Modal cloud GPUs (A100-40GB)")
    print()

    cmd = ["modal", "run", "modal_app.py", "--num-samples", str(num_samples)]

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running Modal: {e}")
        return e.returncode
    except FileNotFoundError:
        print("Error: Modal CLI not found. Please install with 'pip install modal'")
        print("Then run 'modal setup' to authenticate.")
        return 1


def cache_model():
    """Pre-cache the LLaDA model to Modal Volume."""
    print("Caching LLaDA-8B model to Modal Volume...")
    print("This may take a while on first run (~16GB download)")

    # Create a small script to just load the model
    cache_script = '''
import modal_app
modal_app.load_and_cache_model.remote()
print("Model cached successfully!")
'''

    cmd = ["modal", "run", "modal_app.py::load_and_cache_model"]

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error caching model: {e}")
        return e.returncode


def download_results(output_dir: str = "."):
    """Download results from Modal Volume."""
    print("Downloading results from Modal Volume...")

    files_to_download = [
        "results/dire_distributions.png",
        "results/results_summary.txt",
        "results/raw_results.json",
    ]

    for file_path in files_to_download:
        cmd = ["modal", "volume", "get", "text-dire-vol", file_path, output_dir]
        print(f"  Downloading {file_path}...")

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"    -> Saved to {output_dir}")
        except subprocess.CalledProcessError:
            print(f"    -> File not found (experiment may not have run yet)")

    print()
    print("Download complete!")


def list_volume_contents():
    """List contents of the Modal Volume."""
    print("Contents of text-dire-vol:")

    cmd = ["modal", "volume", "ls", "text-dire-vol"]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error listing volume: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Text-DIRE: AI Text Detection using Diffusion Reconstruction Error"
    )

    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=100,
        help="Number of samples per class (default: 100)",
    )

    parser.add_argument(
        "--cache-model",
        action="store_true",
        help="Just cache the model to Modal Volume",
    )

    parser.add_argument(
        "--download",
        "-d",
        action="store_true",
        help="Download results from Modal Volume",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="Output directory for downloaded results",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List contents of Modal Volume",
    )

    args = parser.parse_args()

    if args.cache_model:
        sys.exit(cache_model())

    if args.download:
        download_results(args.output_dir)
        sys.exit(0)

    if args.list:
        list_volume_contents()
        sys.exit(0)

    # Default: run experiment
    sys.exit(run_experiment(args.samples))


if __name__ == "__main__":
    main()
