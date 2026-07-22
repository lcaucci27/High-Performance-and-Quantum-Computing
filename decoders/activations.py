"""
Funzioni di attivazione custom per le reti neurali del progetto QEC.
Implementa SQNL (Square Non-Linearity) da Overwater et al. (arXiv:2202.05741).
"""

import torch
import torch.nn as nn


class SQNL(nn.Module):
    """
    Square Non-Linearity activation function.

    Definita per regioni:
        x < -2        →  -1
        -2 ≤ x < 0   →  x + x²/4
        0 ≤ x ≤ 2    →  x - x²/4
        x > 2         →  +1

    Secondo Overwater et al. questa formulazione garantisce
    ~31% di miglioramento rispetto a TanH in hardware quantistico.

    Note: la formula originale nel paper usa intervallo [-2, 2],
    non [-1, 1]. Qui implementiamo la versione corretta del paper.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applica SQNL element-wise al tensore x.

        Parameters
        ----------
        x : torch.Tensor
            Input di qualsiasi forma.

        Returns
        -------
        torch.Tensor
            Output della stessa forma di x, con valori in [-1, 1].
        """
        # Regione x < -2: output = -1
        out = torch.full_like(x, -1.0)

        # Regione -2 ≤ x < 0: output = x + x²/4
        mask_neg = (x >= -2.0) & (x < 0.0)
        out = torch.where(mask_neg, x + x * x / 4.0, out)

        # Regione 0 ≤ x ≤ 2: output = x - x²/4
        mask_pos = (x >= 0.0) & (x <= 2.0)
        out = torch.where(mask_pos, x - x * x / 4.0, out)

        # Regione x > 2: output = +1
        mask_sat = x > 2.0
        out = torch.where(mask_sat, torch.ones_like(x), out)

        return out
