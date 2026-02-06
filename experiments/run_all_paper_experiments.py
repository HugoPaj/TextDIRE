"""
Run All Paper Experiments for Text-DIRE.

This script orchestrates running all experiments needed for the paper:
1. beemo-zeroshot: Zero-shot scoring methods (DIRE, TTR, DIRE-TTR)
2. beemo-enhanced: XGBoost + 42 features (trained classifier)
3. Mask ratio ablation
4. Per-model analysis

Usage:
    # Run all experiments
    python experiments/run_all_paper_experiments.py --all

    # Run specific experiment
    python experiments/run_all_paper_experiments.py --experiment beemo-zeroshot

    # Run on Modal cloud
    modal run experiments/run_all_paper_experiments.py --experiment beemo-zeroshot

    # Quick test with 200 samples
    python experiments/run_all_paper_experiments.py --all --num-samples 200
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# Experiment configurations
EXPERIMENTS = {
    'beemo-zeroshot': {
        'description': 'Zero-shot scoring methods (DIRE, TTR, DIRE-TTR)',
        'modal_command': 'modal run modal_app.py --experiment beemo-zeroshot',
    },
    'beemo-enhanced': {
        'description': 'XGBoost + 42 features (trained classifier)',
        'modal_command': 'modal run modal_app.py --experiment beemo-enhanced',
    },
    'beemo-by-model': {
        'description': 'Per-model analysis',
        'modal_command': 'modal run modal_app.py --experiment beemo-by-model',
    },
    'beemo-multistep': {
        'description': 'Multi-step diffusion',
        'modal_command': 'modal run modal_app.py --experiment beemo-multistep',
    },
    'beemo-logscale': {
        'description': 'Log-scale scoring',
        'modal_command': 'modal run modal_app.py --experiment beemo-logscale',
    },
    'detectrl': {
        'description': 'DetectRL real-world benchmark (NeurIPS 2024)',
        'modal_command': 'modal run modal_app.py --experiment detectrl',
    },
    'detectrl-full': {
        'description': 'DetectRL full leaderboard (all 13 metrics)',
        'modal_command': 'modal run modal_app.py --experiment detectrl-full',
    },
    'raid-adversarial': {
        'description': 'RAID adversarial subset evaluation (attacks only)',
        'modal_command': 'modal run modal_app.py --experiment raid-adversarial',
    },
}

MASK_RATIOS = [0.3, 0.5, 0.7, 0.8, 0.9]


def run_command(command: str, dry_run: bool = False) -> int:
    """Run a shell command and return exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {command}")
    print('='*60)

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return 0

    result = subprocess.run(command, shell=True)
    return result.returncode


def run_experiment(
    experiment: str,
    num_samples: int = None,
    mask_ratio: float = None,
    dry_run: bool = False,
) -> int:
    """Run a single experiment."""
    if experiment not in EXPERIMENTS:
        print(f"Unknown experiment: {experiment}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        return 1

    config = EXPERIMENTS[experiment]
    print(f"\n{'#'*60}")
    print(f"# Experiment: {experiment}")
    print(f"# Description: {config['description']}")
    print('#'*60)

    command = config['modal_command']

    if num_samples:
        command += f" --num-samples {num_samples}"

    if mask_ratio:
        command += f" --mask-ratio {mask_ratio}"

    return run_command(command, dry_run)


def run_mask_ratio_ablation(
    num_samples: int = None,
    dry_run: bool = False,
) -> dict:
    """Run ablation across mask ratios."""
    print("\n" + "="*60)
    print("MASK RATIO ABLATION")
    print("="*60)

    results = {}
    for ratio in MASK_RATIOS:
        print(f"\n>>> Mask ratio: {ratio}")
        command = f"modal run modal_app.py --experiment beemo-zeroshot --mask-ratio {ratio}"
        if num_samples:
            command += f" --num-samples {num_samples}"

        exit_code = run_command(command, dry_run)
        results[ratio] = {'exit_code': exit_code}

    return results


def run_all_experiments(
    num_samples: int = None,
    dry_run: bool = False,
    skip_ablation: bool = False,
) -> dict:
    """Run all experiments for the paper."""
    print("\n" + "#"*60)
    print("# RUNNING ALL PAPER EXPERIMENTS")
    print("#"*60)

    start_time = datetime.now()
    results = {'experiments': {}, 'ablation': {}}

    # Run main experiments
    for experiment in EXPERIMENTS:
        exit_code = run_experiment(experiment, num_samples, dry_run=dry_run)
        results['experiments'][experiment] = {'exit_code': exit_code}

    # Run mask ratio ablation
    if not skip_ablation:
        results['ablation'] = run_mask_ratio_ablation(num_samples, dry_run)

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "#"*60)
    print("# EXPERIMENT SUMMARY")
    print("#"*60)
    print(f"Duration: {duration}")
    print("\nExperiment Results:")
    for exp, result in results['experiments'].items():
        status = "OK" if result['exit_code'] == 0 else f"FAILED ({result['exit_code']})"
        print(f"  {exp}: {status}")

    if 'ablation' in results and results['ablation']:
        print("\nAblation Results:")
        for ratio, result in results['ablation'].items():
            status = "OK" if result['exit_code'] == 0 else f"FAILED ({result['exit_code']})"
            print(f"  mask_ratio={ratio}: {status}")

    return results


def collect_results(output_dir: str = "results") -> dict:
    """Collect all experiment results into a single JSON file for paper generation."""
    output_dir = Path(output_dir)
    paper_results = {
        'main_results': {},
        'ablation': {},
        'per_model': {},
        'score_distributions': {},
        'roc_curves': {},
        'feature_importance': {},
    }

    # Look for beemo results
    beemo_results_path = output_dir / "beemo" / "beemo_results.json"
    if beemo_results_path.exists():
        with open(beemo_results_path) as f:
            beemo = json.load(f)
        print(f"Loaded: {beemo_results_path}")

        # Map to paper format
        for scenario in ['easy', 'medium', 'hard']:
            if scenario in beemo:
                paper_results['main_results'][f'DIRE alone'] = {
                    'easy': beemo.get('easy', {}).get('auroc', 0),
                    'medium': beemo.get('medium', {}).get('auroc', 0),
                    'hard': beemo.get('hard', {}).get('auroc', 0),
                }

    # Look for enhanced results
    enhanced_path = output_dir / "beemo-enhanced" / "results.json"
    if enhanced_path.exists():
        with open(enhanced_path) as f:
            enhanced = json.load(f)
        print(f"Loaded: {enhanced_path}")

        paper_results['main_results']['XGBoost + 42 features'] = {
            'easy': enhanced.get('easy', {}).get('auroc', 0),
            'medium': enhanced.get('medium', {}).get('auroc', 0),
            'hard': enhanced.get('hard', {}).get('auroc', 0),
        }

        # Feature importance
        if 'feature_importance' in enhanced:
            paper_results['feature_importance'] = enhanced['feature_importance']

    # Look for scores for distributions
    scores_path = output_dir / "beemo" / "beemo_scores.json"
    if scores_path.exists():
        with open(scores_path) as f:
            scores = json.load(f)
        print(f"Loaded: {scores_path}")

        for scenario, data in scores.items():
            labels = data.get('labels', [])
            score_values = data.get('scores', [])
            paper_results['score_distributions'][scenario] = {
                'human_scores': [s for s, l in zip(score_values, labels) if l == 0],
                'ai_scores': [s for s, l in zip(score_values, labels) if l == 1],
            }

    # Save collected results
    output_path = output_dir / "paper_results.json"
    with open(output_path, 'w') as f:
        json.dump(paper_results, f, indent=2)
    print(f"\nSaved collected results to: {output_path}")

    return paper_results


def generate_paper_outputs(
    results_path: str = "results/paper_results.json",
    output_dir: str = "paper",
):
    """Generate all paper figures and tables from collected results."""
    from src.paper_figures import generate_all_figures
    from src.latex_tables import generate_all_tables

    print("\n" + "#"*60)
    print("# GENERATING PAPER OUTPUTS")
    print("#"*60)

    # Generate figures
    generate_all_figures(results_path, f"{output_dir}/figures")

    # Generate tables
    generate_all_tables(results_path, f"{output_dir}/tables")

    print("\nPaper outputs generated successfully!")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all paper experiments for Text-DIRE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments with full dataset
    python experiments/run_all_paper_experiments.py --all

    # Quick test with 200 samples
    python experiments/run_all_paper_experiments.py --all --num-samples 200

    # Run specific experiment
    python experiments/run_all_paper_experiments.py --experiment beemo-zeroshot

    # Dry run (show commands without executing)
    python experiments/run_all_paper_experiments.py --all --dry-run

    # Collect results and generate paper outputs
    python experiments/run_all_paper_experiments.py --collect --generate
        """
    )

    parser.add_argument("--all", action="store_true",
                        help="Run all experiments")
    parser.add_argument("--experiment", type=str,
                        help="Run specific experiment",
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--ablation", action="store_true",
                        help="Run mask ratio ablation only")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--mask-ratio", type=float, default=None,
                        help="Mask ratio for single experiment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip mask ratio ablation when running --all")
    parser.add_argument("--collect", action="store_true",
                        help="Collect results from completed experiments")
    parser.add_argument("--generate", action="store_true",
                        help="Generate paper figures and tables")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing experiment results")

    args = parser.parse_args()

    if args.collect:
        collect_results(args.results_dir)

    if args.generate:
        generate_paper_outputs(
            f"{args.results_dir}/paper_results.json",
            "paper"
        )

    if args.all:
        run_all_experiments(
            num_samples=args.num_samples,
            dry_run=args.dry_run,
            skip_ablation=args.skip_ablation,
        )
    elif args.experiment:
        run_experiment(
            args.experiment,
            num_samples=args.num_samples,
            mask_ratio=args.mask_ratio,
            dry_run=args.dry_run,
        )
    elif args.ablation:
        run_mask_ratio_ablation(
            num_samples=args.num_samples,
            dry_run=args.dry_run,
        )
    elif not args.collect and not args.generate:
        parser.print_help()


if __name__ == "__main__":
    main()
