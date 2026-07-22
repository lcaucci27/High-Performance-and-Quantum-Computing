"""
Classe base astratta per tutti i decoder QEC del progetto.
Ogni decoder concreto deve implementare il metodo predict().
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseDecoder(ABC):
    """
    Interfaccia comune a tutti i decoder.

    Ogni decoder riceve una matrice di sindromi e restituisce
    un array di predizioni del flip logico (0 o 1 per ogni shot).
    """

    def __init__(self, d: int) -> None:
        """
        Parameters
        ----------
        d : int
            Distanza del surface code (d=3, 5, 7, ...).
        """
        self.d = d

    @abstractmethod
    def predict(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Predice il flip logico per ogni shot.

        Parameters
        ----------
        syndromes : np.ndarray, shape (N, d²-1)
            Bitstring di ancilla syndrome per N shot.

        Returns
        -------
        np.ndarray, shape (N,), dtype int
            Predizione del flip logico: 0 (nessun errore) o 1 (errore).
        """
        ...

    def train(
        self,
        syndromes: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Addestra il decoder (opzionale, solo per decoder NN).

        Parameters
        ----------
        syndromes : np.ndarray
            Dati di training.
        labels : np.ndarray
            Etichette di training.
        """
        # default: nessun training (decoder classici)
        pass
