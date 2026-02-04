"""
Text-DIRE Experiments Package

Contains experiment runners for:
- Main results (method comparison across AI sources)
- RAID benchmark evaluation
- Ablation studies
- Robustness evaluation
- Error analysis
"""

from .main_results import run_main_experiment
from .raid_evaluation import evaluate_on_raid
from .ablations import run_ablation_studies
from .robustness import run_robustness_evaluation
from .error_analysis import run_error_analysis

__all__ = [
    "run_main_experiment",
    "evaluate_on_raid",
    "run_ablation_studies",
    "run_robustness_evaluation",
    "run_error_analysis",
]
