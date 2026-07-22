"""
Pacchetto evaluation: metriche e benchmarking dei decoder QEC.
"""

from .metrics import compute_ler, compute_ler_per_round, find_pseudo_threshold, find_decoder_threshold
from .benchmark import run_benchmark_custom, run_benchmark_pymatching

__all__ = [
    "compute_ler",
    "compute_ler_per_round",
    "find_pseudo_threshold",
    "find_decoder_threshold",
    "run_benchmark_custom",
    "run_benchmark_pymatching",
]
