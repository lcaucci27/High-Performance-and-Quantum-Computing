"""
High-Level Decoder (HLD): PED deterministico + rete neurale classificatrice.

Architettura fedele a Overwater, Babaie & Sebastiano (arXiv:2202.05741):
  - PED (Pure Error Decoder): CustomMWPM con P_ASSUMED fisso, deterministico,
    produce sempre la stessa predizione per una data sindrome (analogo al
    chain-routing PED del paper, Sec. III.A).
  - NN (Sec. III.B): prende la SOLA sindrome (d²-1 input) e predice se il PED
    fa un errore logico. 2 hidden layer, SQNL, hidden sizes 256+64 (Table III).
  - Combined: flip_finale = PED_pred XOR NN_correction.

Target di training (Sec. IV):
  NN impara: PM(s) XOR PED(s), dove PM = PyMatching ottimale con p corretto.
  Poiché PED e PM sono ENTRAMBI deterministici, il target è una funzione
  deterministica della sindrome → zero label noise → apprendimento pulito.
  Ground truth (actual_flip) NON va usato perché è stocastico per la stessa
  sindrome (degeneracy QEC) → label noise → NN apprende output ≈ 0 → non scatta.

Differenza rispetto a LLD: il PED gestisce la parte "pure error", riducendo
il task della NN alla sola classificazione logica (molto più semplice).
Questo è il motivo per cui HLD scala meglio con la distanza (QEC scaling).
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .activations import SQNL
from .base_decoder import BaseDecoder
from .custom_mwpm import P_ASSUMED, CustomMWPMDecoder

# Dimensioni hidden layer (Table III del paper: 256 + 64)
_H1 = 256
_H2 = 64


class _HLDNet(nn.Module):
    """
    Rete classificatrice per la correzione logica residuale del PED.

    Input (Sec. III.B del paper): SOLA sindrome ancilla, d²-1 valori.
    Output: logit di P(flip_reale != PED) = P(PED ha fatto un errore logico).

    Architettura: 2 hidden layer con SQNL (come nel paper, Sec. III.B-V.B).
    CPU default:  h1=256, h2=64  (Table III del paper)
    GPU enhanced: h1=512, h2=256 (maggiore capacità per catturare correlazioni
                                   nel circuit-level noise che MWPM ignora)
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
        """Output shape: (batch,), logit."""
        return self.net(x).squeeze(-1)


class HLDDecoder(BaseDecoder):
    """
    High-Level Decoder: CustomMWPM (PED) + rete neurale classificatrice.

    CustomMWPM con P_ASSUMED fisso serve come PED deterministico.
    La NN impara quando il PED sbaglia rispetto a PyMatching ottimale.

    QEC scaling (Sec. V.C-E del paper):
      Sotto-threshold: HLD d=7 < d=5 < d=3 in LER.
    """

    def __init__(
        self,
        d: int,
        p_assumed: float = P_ASSUMED,
        h1: int = 256,
        h2: int = 64,
        max_fire_rate: float = 0.15,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(d)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._base = CustomMWPMDecoder(d, p_assumed=p_assumed)
        self.model = _HLDNet(d, h1=h1, h2=h2).to(self.device)
        self._h1 = h1
        self._h2 = h2
        self._max_fire_rate = max_fire_rate
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(  # type: ignore[override]
        self,
        syndromes: np.ndarray,
        logical_flips: np.ndarray,
        lr: float = 3e-3,
        batch_size: int = 256,
        epochs: int = 100,
        pos_weight_cap: float = 40.0,
    ) -> None:
        """
        Addestra la NN sul residuo del PED: PM_pred(s) XOR PED(s).

        logical_flips deve essere la PREDIZIONE DI PYMATCHING (oracle), non il
        ground truth. Il target è deterministico → zero label noise.

        Usa CosineAnnealingLR: LR parte da `lr` e decade a 1e-5 in `epochs` steps.
        Questo converge più rapidamente di StepLR, specialmente per d grandi.

        pos_weight cap a 40.0: con p vicino a p_th PM!=Custom è ~4-8%,
        pos_weight ~12-24 → bilanciamento adeguato. Safety valve (fire_rate)
        previene overfiring a inference.

        Parameters
        ----------
        syndromes     : np.ndarray, shape (N, d²-1)
        logical_flips : np.ndarray, shape (N,), predizioni oracle (PyMatching)
        """
        base      = self._base.predict(syndromes)
        residuals = (logical_flips.astype(int) ^ base).astype(np.float32)

        X = torch.tensor(syndromes.astype(np.float32), dtype=torch.float32, device=self.device)
        Y = torch.tensor(residuals,                    dtype=torch.float32, device=self.device)

        # pos_weight: bilancia la classe rara (PM!=Custom).
        # cap configurabile per d: per d>=7 il segnale è sparso (~1.6% positivi),
        # cap=5 lascia i negativi dominare 12:1. Cap=10 riduce a 6:1.
        # Safety valve (max_fire_rate) previene overfiring a inference.
        n_pos = float(residuals.sum())
        n_neg = float(len(residuals)) - n_pos
        raw_w = n_neg / max(n_pos, 1.0)
        pos_w = torch.tensor(min(raw_w, pos_weight_cap), dtype=torch.float32, device=self.device)

        loader    = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        baseline_err = float(residuals.mean())
        print(f"    HLD d={self.d}  PM!=Custom {baseline_err*100:.1f}%  "
              f"pos_weight={min(raw_w, pos_weight_cap):.2f}  N={len(residuals)}")

        self.model.train()
        for epoch in range(epochs):
            total = 0.0
            for x_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(x_b), y_b)
                loss.backward()
                optimizer.step()
                total += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"    HLD d={self.d} [{epoch+1:3d}/{epochs}] "
                      f"loss={total/len(loader):.4f}  lr={cur_lr:.2e}")

        self._trained = True
        self.model.eval()

        # Diagnostica finale: stima fire_rate su training data per verificare calibrazione
        with torch.no_grad():
            logits_all = self.model(X).cpu().numpy()
        train_fire = float((logits_all > 0.5).mean())
        print(f"    HLD d={self.d} training completato — "
              f"fire_rate_train={train_fire:.3f}  target={baseline_err:.3f}")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Predice il flip logico: PED + correzione NN.

        Soglia 0.5 sul logit (= prob ~62%).
        Safety valve: se fire_rate > max_fire_rate → fallback a PED base.

        Parameters
        ----------
        syndromes : np.ndarray, shape (N, d²-1)
        Returns
        -------
        np.ndarray, shape (N,), dtype int
        """
        base = self._base.predict(syndromes)

        if not self._trained:
            return base

        self.model.eval()
        with torch.no_grad():
            X      = torch.tensor(syndromes.astype(np.float32), dtype=torch.float32, device=self.device)
            logits = self.model(X).cpu().numpy()
            nn_residual = (logits > 0.5).astype(int)

        fire_rate = float(nn_residual.mean())
        if fire_rate > self._max_fire_rate:
            return base

        return (base ^ nn_residual).astype(int)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Salva i pesi del modello e la configurazione su disco."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "d": self.d,
            "h1": self._h1,
            "h2": self._h2,
            "trained": self._trained,
        }, path)

    def load(self, path: str) -> bool:
        """
        Carica i pesi del modello da disco.
        Restituisce True se il caricamento ha avuto successo, False altrimenti.
        """
        if not os.path.isfile(path):
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self._trained = ckpt.get("trained", True)
        self.model.eval()
        return True
