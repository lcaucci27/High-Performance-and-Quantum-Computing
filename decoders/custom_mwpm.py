"""
Decoder MWPM "degradato": usa PyMatching ma con un error rate fisso (P_ASSUMED)
invece del valore reale di p. Questo crea una degradazione sistematica e DETERMINISTICA
che il decoder HLD può imparare a correggere.
"""

import numpy as np
import pymatching
import stim

from .base_decoder import BaseDecoder

# Error rate assunto dal matcher, intenzionalmente fisso e "sbagliato"
# rispetto al p reale. Questo crea un bias sistematico e apprendibile.
#
# P_ASSUMED=0.30 è scelto per massimizzare il segnale PM!=Custom nel range di training.
# edge_weight = log(0.7/0.3) ≈ 0.85  vs  ottimale ≈ 4.6 a p=0.010 (fattore 5.4×).
# Questo genera PM!=Custom ≈ 5-12% nel range [0.004, 0.013] → zona informativa per HLD.
# Analogo al PED del paper (chain-routing, ~10-15% errori a p_th).
P_ASSUMED = 0.30


def _build_circuit(d: int, p: float, rounds: int = 1) -> stim.Circuit:
    """Costruisce il circuito Stim per il rotated surface code."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=rounds,
        distance=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def _build_matcher(d: int, p: float) -> pymatching.Matching:
    """Costruisce il matcher PyMatching dal detector error model."""
    circuit = _build_circuit(d, p)
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


class CustomMWPMDecoder(BaseDecoder):
    """
    Decoder MWPM con error rate assunto fisso (P_ASSUMED = 0.30).

    La degradazione è DETERMINISTICA: la stessa sindrome produce sempre
    la stessa correzione. Questo lo rende peggiore di PyMatching ottimale
    (che usa il p reale) ma apprendibile dal decoder HLD.

    Failure modes sistematici:
      - p << P_ASSUMED: MWPM sovrastima gli errori → over-matching
      - p >> P_ASSUMED: MWPM sottostima gli errori → under-matching
      - p ≈ P_ASSUMED: comportamento quasi-ottimale

    HLD impara a correggere questi failure mode imparando il pattern
    PM(sindrome) XOR Custom(sindrome), che è deterministico per una data sindrome.
    """

    def __init__(self, d: int, p_assumed: float = P_ASSUMED) -> None:
        """
        Parameters
        ----------
        d : int
            Distanza del codice.
        p_assumed : float
            Error rate usato per costruire il matcher (default=P_ASSUMED=0.30).
        """
        super().__init__(d)
        self.p_assumed = p_assumed
        # Il matcher è costruito una volta sola: decodifica DETERMINISTICA
        self.matcher = _build_matcher(d, p_assumed)

    def predict(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Decodifica deterministica: stessa sindrome → sempre stessa correzione.

        Parameters
        ----------
        syndromes : np.ndarray, shape (N, d²-1), dtype float/bool
        Returns
        -------
        np.ndarray, shape (N,), dtype int
        """
        preds = self.matcher.decode_batch(syndromes.astype(np.uint8))
        return preds[:, 0].astype(int)


def run_mwpm(syndrome: np.ndarray, d: int, p: float = P_ASSUMED) -> int:
    """
    MWPM su singola sindrome (lento, ricostruisce il matcher ad ogni chiamata).

    Parameters
    ----------
    syndrome : np.ndarray, shape (d²-1,)
    d : int
    p : float
    Returns
    -------
    int : 0 o 1
    """
    matcher = _build_matcher(d, p)
    pred = matcher.decode(syndrome.astype(np.uint8))
    return int(pred[0]) if hasattr(pred, "__len__") else int(pred)
