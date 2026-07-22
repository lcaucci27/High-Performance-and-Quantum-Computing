"""
Generazione di dati sintetici per il training e la valutazione dei decoder QEC.
Usa Stim per simulare il rotated surface code con rumore depolarizzante circuit-level.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import stim


@dataclass
class SyntheticDataset:
    """
    Contenitore per un dataset sintetico generato con Stim.

    Attributes
    ----------
    syndromes : np.ndarray, shape (N, d²-1)
        Bitstring di detection event (ancilla triggerate).
    logical_flips : np.ndarray, shape (N,)
        Ground truth del flip logico (0/1) per ogni shot.
    d : int
        Distanza del codice.
    p : float
        Physical error rate usato per la simulazione.
    """
    syndromes:    np.ndarray
    logical_flips: np.ndarray
    d:            int
    p:            float


def _build_stim_circuit(d: int, p: float, rounds: int = 1) -> stim.Circuit:
    """
    Costruisce il circuito Stim per il rotated surface code con rumore depolarizzante.

    Usa stim.Circuit.generated() che fornisce direttamente il circuito
    per il surface code con il modello di rumore circuit-level standard.

    Parameters
    ----------
    d : int
        Distanza del codice (code distance).
    p : float
        Physical error rate per il canale di rumore.
    rounds : int
        Numero di rounds di misura degli ancilla.

    Returns
    -------
    stim.Circuit
        Circuito con rumore depolarizzante circuit-level.
    """
    # stim.Circuit.generated supporta 'surface_code:rotated_memory_x'
    # che è il rotated surface code con osservabile X logico
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=rounds,
        distance=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    return circuit


def generate_synthetic_data(
    d: int,
    p: float,
    n_shots: int = 10_000,
    rounds: int = 1,
    seed: int = 42,
) -> SyntheticDataset:
    """
    Genera un dataset sintetico per distanza d e error rate p.

    Campiona n_shots shot dal circuito Stim, estrae:
      - detection events (sindromi degli ancilla)
      - logical observable flips (ground truth)

    Parameters
    ----------
    d : int
        Distanza del rotated surface code.
    p : float
        Physical error rate per il modello di rumore.
    n_shots : int
        Numero di shot da campionare.
    rounds : int
        Numero di rounds del circuito QEC.
    seed : int
        Seed per la riproducibilità.

    Returns
    -------
    SyntheticDataset
        Dataset con sindromi e flip logici.
    """
    circuit = _build_stim_circuit(d, p, rounds=rounds)
    sampler  = circuit.compile_detector_sampler(seed=seed)

    # Campiona detection events e observable flips insieme
    detection_events, observable_flips = sampler.sample(
        shots=n_shots,
        separate_observables=True,
    )

    # detection_events: shape (n_shots, n_detectors), bool
    # observable_flips: shape (n_shots, n_observables), bool
    syndromes    = detection_events.astype(np.float32)
    logical_flip = observable_flips[:, 0].astype(int)  # primo osservabile logico

    return SyntheticDataset(
        syndromes=syndromes,
        logical_flips=logical_flip,
        d=d,
        p=p,
    )


def generate_training_data(
    d: int,
    p_train: float,
    n_shots: int = 50_000,
    seed: int = 0,
) -> SyntheticDataset:
    """
    Genera dati di training per LLD/HLD vicino al pseudo-threshold MWPM.

    Parameters
    ----------
    d : int
        Distanza del codice.
    p_train : float
        Error rate di training (tipicamente vicino al p_th MWPM per d).
    n_shots : int
        Numero di shot di training.
    seed : int
        Seed per riproducibilità.

    Returns
    -------
    SyntheticDataset
        Dataset di training.
    """
    return generate_synthetic_data(d, p_train, n_shots=n_shots, seed=seed)


def generate_all_eval_data(
    distances: List[int],
    p_values: List[float],
    n_shots: int = 10_000,
    seed: int = 42,
) -> Dict[Tuple[int, float], SyntheticDataset]:
    """
    Genera tutti i dataset di valutazione per le combinazioni (d, p).

    Parameters
    ----------
    distances : list of int
        Distanze da simulare.
    p_values : list of float
        Valori di physical error rate.
    n_shots : int
        Shot per combinazione.
    seed : int
        Seed base (incrementato per ogni combinazione).

    Returns
    -------
    dict
        Mappa (d, p) → SyntheticDataset.
    """
    results: Dict[Tuple[int, float], SyntheticDataset] = {}
    s = seed
    total = len(distances) * len(p_values)
    done  = 0
    for d in distances:
        print(f"  d={d}: {len(p_values)} p-values × {n_shots} shot...", end="", flush=True)
        for p in p_values:
            results[(d, p)] = generate_synthetic_data(d, p, n_shots=n_shots, seed=s)
            s += 1
            done += 1
        print(f" OK ({done}/{total})")
    return results
