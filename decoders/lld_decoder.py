"""
Low-Level Decoder (LLD) per il rotated surface code.
Classificatore binario diretto: sindrome → flip logico.
Nessuna struttura QEC: la NN fa pure pattern matching sul syndrome bitstring.

Architettura fedele a Overwater, Babaie & Sebastiano (arXiv:2202.05741):
  - 2 hidden layer, SQNL activation (Sec. III.B, Eq. 13)
  - d²-1 input (sindrome ancilla), 1 output logit (flip logico Z)
  - Hidden sizes: h1=256, h2=64  (Table III, configurazione migliore)
  - Training al p_th di MWPM per distanza (Sec. IV.A)
  - BCEWithLogitsLoss (numericamente stabile vs BCE+Sigmoid)
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .activations import SQNL
from .base_decoder import BaseDecoder

class _LLDNet(nn.Module):
    """
    Rete feedforward per il Low-Level Decoder.

    Architettura (Overwater et al. 2022, Sec. III.B e Table III):
      Linear(d²-1, h1) → SQNL → Linear(h1, h2) → SQNL → Linear(h2, 1)

    CPU default:  h1=256, h2=64  (Table III)
    GPU enhanced: h1=512, h2=256 (maggiore capacità per distanze grandi)

    Output: logit di P(flip logico Z).  Nessuna Sigmoid in rete:
    usa BCEWithLogitsLoss per stabilità numerica.
    """

    def __init__(self, d: int, h1: int = 256, h2: int = 64) -> None:
        super().__init__()
        n_in = d * d - 1
        self.net = nn.Sequential(
            nn.Linear(n_in, h1),
            SQNL(),
            nn.Linear(h1, h2),
            SQNL(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output shape: (batch,), logit del flip logico."""
        return self.net(x).squeeze(-1)


class LLDDecoder(BaseDecoder):
    """
    Low-Level Decoder (LLD): classificatore binario su sindrome pura.

    Non usa MWPM né struttura del codice: la NN deve imparare
    implicitamente la decodifica da sola. Questo la rende il decoder
    con le peggiori performance attese, specialmente a grandi distanze
    (assenza di QEC scaling corretto, Fig. 6 del paper).

    Addestrato al p_th di MWPM per ogni distanza (Sec. IV.A):
    questo è il punto in cui il decoder ha più segnale da imparare
    (50% errori logici attesi da MWPM al threshold).
    """

    def __init__(self, d: int, h1: int = 256, h2: int = 64, device: Optional[str] = None) -> None:
        super().__init__(d)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _LLDNet(d, h1=h1, h2=h2).to(self.device)
        self._h1 = h1
        self._h2 = h2
        self._trained = False

    def train(  # type: ignore[override]
        self,
        syndromes: np.ndarray,
        logical_flips: np.ndarray,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 20,
    ) -> None:
        """
        Addestra la rete LLD con BCEWithLogitsLoss sul flip logico.

        Il training al p_th di MWPM (passato dall'esterno come p di generazione)
        segue Sec. IV.A del paper: "we sample at the physical error rate
        corresponding to the p_th of the MWPM algorithm for that distance."

        Parameters
        ----------
        syndromes     : np.ndarray, shape (N, d²-1)
        logical_flips : np.ndarray, shape (N,), 0 o 1
        """
        X = torch.tensor(syndromes,                   dtype=torch.float32, device=self.device)
        Y = torch.tensor(logical_flips.astype(float), dtype=torch.float32, device=self.device)

        loader    = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(epochs):
            total = 0.0
            for x_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(x_b), y_b)
                loss.backward()
                optimizer.step()
                total += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"    LLD d={self.d} [{epoch+1:2d}/{epochs}] loss={total/len(loader):.4f}")

        self._trained = True
        self.model.eval()

    def predict(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Predice il flip logico: threshold a 0.5 sulla probabilità.

        Parameters
        ----------
        syndromes : np.ndarray, shape (N, d²-1)
        Returns
        -------
        np.ndarray, shape (N,), dtype int
        """
        if not self._trained:
            return np.zeros(len(syndromes), dtype=int)

        self.model.eval()
        with torch.no_grad():
            X      = torch.tensor(syndromes, dtype=torch.float32, device=self.device)
            logits = self.model(X).cpu().numpy()

        return (logits > 0.0).astype(int)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Salva i pesi del modello su disco."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "d": self.d,
            "h1": self._h1,
            "h2": self._h2,
            "trained": self._trained,
        }, path)

    def load(self, path: str) -> bool:
        """Carica i pesi del modello da disco. Restituisce True se ok."""
        if not os.path.isfile(path):
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self._trained = ckpt.get("trained", True)
        self.model.eval()
        return True
