"""
Pacchetto data: generazione e caricamento dei dataset per il benchmarking QEC.
"""

from .synthetic_dataset import generate_synthetic_data, SyntheticDataset
from .google_sycamore import GoogleSycamoreLoader

__all__ = [
    "generate_synthetic_data",
    "SyntheticDataset",
    "GoogleSycamoreLoader",
]
