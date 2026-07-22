"""
Utility condivise per tutti i plot QEC.
Stile pseudoth.png: log-log, diagonale y=x, cerchio colorato al pseudo-threshold.
Il p_th è riportato nella label della legenda, niente testo inline che si sovrappone.
"""

import os
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics import find_pseudo_threshold_log

# ── Stile globale ──────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    try:
        plt.style.use("seaborn-paper")
    except OSError:
        pass

matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   12,
    "legend.fontsize":   9,
    "figure.dpi":       150,
    "lines.linewidth":  1.8,
    "lines.markersize": 5,
})


# ── API pubblica ───────────────────────────────────────────────────────────

def pseudo_threshold_label(base_name: str, p_values: List[float], ler_values: List[float]) -> str:
    """
    Costruisce la label per la legenda includendo il p_th.

    Invece di annotare inline (causa overlap), il valore è nella legenda:
      "HLD  (p_th = 0.082)"

    Parameters
    ----------
    base_name : str
        Nome base del decoder/distanza.
    p_values, ler_values : dati della curva.

    Returns
    -------
    str
        Label con p_th, o solo base_name se p_th non trovato.
    """
    p_th = find_pseudo_threshold_log(p_values, ler_values)
    if p_th and p_th > 5e-4:
        return f"{base_name}  (p$_{{\\rm th}}$={p_th:.3f})"
    return base_name


def mark_pseudo_threshold(
    ax: plt.Axes,
    p_values: List[float],
    ler_values: List[float],
    color: str,
) -> Optional[float]:
    """
    Disegna il cerchio colorato al pseudo-threshold (senza testo inline).

    Il p_th è già riportato nella legenda tramite pseudo_threshold_label.

    Parameters
    ----------
    ax : asse target
    p_values, ler_values : dati della curva
    color : colore del cerchio

    Returns
    -------
    float or None : valore del pseudo-threshold
    """
    p_th = find_pseudo_threshold_log(p_values, ler_values)
    if p_th and p_th > 5e-4:
        ax.plot(
            p_th, p_th,
            "o",
            color=color,
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=6,
        )
    return p_th


def setup_log_ax(ax: plt.Axes, title: str = "", ylabel: str = "Logical Error Rate (LER)") -> None:
    """Configura asse log-log con griglia maggiore e minore."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate (PER)", labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.grid(True, which="major", linestyle="-",  alpha=0.30, linewidth=0.8)
    ax.grid(True, which="minor", linestyle=":",  alpha=0.15, linewidth=0.5)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)


def draw_diagonal(ax: plt.Axes, p_min: float = 1e-3, p_max: float = 0.2) -> None:
    """Diagonale LER = PER in nero tratteggiato."""
    p_line = np.logspace(np.log10(p_min), np.log10(p_max), 300)
    ax.plot(p_line, p_line, "k--", linewidth=1.5, label="LER = PER", zorder=2, alpha=0.8)


def plot_ler_curve(
    ax: plt.Axes,
    p_values: List[float],
    ler_values: List[float],
    color: str,
    label: str,
    marker: str = "o",
) -> None:
    """Traccia una curva LER vs PER in log-log, filtrando zeri."""
    valid = [(p, l) for p, l in zip(p_values, ler_values) if p > 0 and l > 0]
    if not valid:
        return
    ps, ls = zip(*valid)
    ax.plot(ps, ls, marker=marker, color=color,
            linewidth=1.8, markersize=5, label=label, zorder=3, alpha=0.9)


def add_qec_benefit_region(ax: plt.Axes, p_min: float, p_max: float) -> None:
    """Aggiunge la regione verde "QEC aiuta" (sotto la diagonale y=x)."""
    p_fill = np.logspace(np.log10(p_min), np.log10(p_max), 200)
    # Riempi un paio di decadi sotto la diagonale; autoscale_log_ax taglierà poi i limiti
    ax.fill_between(p_fill, p_fill * 1e-3, p_fill,
                    alpha=0.06, color="forestgreen", zorder=0)


def autoscale_log_ax(ax: plt.Axes, padding: float = 0.6) -> None:
    """
    Imposta i limiti y al range effettivo delle linee già tracciate.

    Esclude le linee di riferimento (diagonale, fill) e usa solo
    i dati reali delle curve LER. Aggiunge `padding` decadi di margine.
    """
    y_vals: List[float] = []
    for line in ax.get_lines():
        yd = np.asarray(line.get_ydata(), dtype=float)
        yd = yd[(yd > 0) & np.isfinite(yd)]
        y_vals.extend(yd.tolist())
    if y_vals:
        ymin = float(min(y_vals))
        ymax = float(max(y_vals))
        ax.set_ylim(
            max(ymin / (10 ** padding), 1e-6),
            min(ymax * (10 ** padding), 1.0),
        )


def save_figure(fig: plt.Figure, base_path: str, filename: str, dpi: int = 300) -> None:
    """Salva la figura solo in PNG."""
    os.makedirs(base_path, exist_ok=True)
    png_path = os.path.join(base_path, f"{filename}.png")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    print(f"  Salvato: {png_path}")


# ── Compatibilità con vecchi chiamanti ────────────────────────────────────
def add_pseudo_threshold(
    ax: plt.Axes,
    p_values: List[float],
    ler_values: List[float],
    color: str,
    label_prefix: str = "",
) -> Optional[float]:
    """Wrapper di compatibilità: disegna solo il cerchio, senza testo."""
    return mark_pseudo_threshold(ax, p_values, ler_values, color)
