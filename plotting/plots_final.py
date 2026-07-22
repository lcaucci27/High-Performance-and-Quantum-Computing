"""
3 plot definitivi del benchmark QEC.

A: LER vs PER per distanza, tutti e 4 i decoder (LLD / Custom MWPM / HLD / PyMatching)
   su un unico asse per distanza. Sostituisce i doppi plot 1+4.

B: Scalabilità con d, 4 subplot (LLD / Custom MWPM / HLD / PyMatching).
   LLD: QEC scaling invertito (nessun beneficio).
   Custom MWPM: scaling intermedio (MWPM con p_assumed fisso).
   HLD e PyMatching: d=7 < d=5 < d=3 a basso p (QEC funziona).

C: Google Sycamore, stile Jung et al. 2024, Figure 12.
    Scatter di LER/round (%) per decoder su hardware reale.
    Un punto per ogni esperimento (center × rounds). Linea = mediana.
    Dati: Google Quantum AI, Nature 614 (2023), Zenodo.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from config import DISTANCES, P_VALUES, RESULTS_FINAL
from plotting.plot_utils import (
    add_qec_benefit_region,
    autoscale_log_ax,
    draw_diagonal,
    mark_pseudo_threshold,
    plot_ler_curve,
    pseudo_threshold_label,
    save_figure,
    setup_log_ax,
)

# ── Palette decoder ────────────────────────────────────────────────────────
_C4 = {
    "lld":         "#D62728",
    "custom_mwpm": "#FF7F0E",
    "hld":         "#1F77B4",
    "pymatching":  "#2CA02C",
}
_M4 = {"lld": "^", "custom_mwpm": "s", "hld": "o", "pymatching": "D"}
_N4 = {
    "lld":         "LLD",
    "custom_mwpm": "Custom MWPM",
    "hld":         "HLD",
    "pymatching":  "PyMatching (ottimale)",
}
_ORDER4 = ["lld", "custom_mwpm", "hld", "pymatching"]

# ── Palette distanze ────────────────────────────────────────────────────────
_CDIST = {3: "tab:red", 5: "tab:green", 7: "tab:blue", 9: "tab:purple"}
_MDIST = {3: "^", 5: "s", 7: "o", 9: "D"}

# ── Palette Plot C ─────────────────────────────────────────────────────────
# Sintetici (addestrati su Stim) + hardware-trained (fine-tune su Sycamore)
_OSYC = ["lld", "custom_mwpm", "hld", "pymatching_our", "lld_hw", "hld_hw"]
_CSYC = {
    "lld":            "#D62728",
    "custom_mwpm":    "#FF7F0E",
    "hld":            "#1F77B4",
    "pymatching_our": "#2CA02C",
    "lld_hw":         "#8B0000",   # rosso scuro, LLD fine-tuned su hardware
    "hld_hw":         "#003580",   # blu scuro, HLD fine-tuned su hardware
}
_NSYC = {
    "lld":            "LLD\n(sintetico)",
    "custom_mwpm":    "Custom\nMWPM",
    "hld":            "HLD\n(sintetico)",
    "pymatching_our": "PyMatching\n(ours)",
    "lld_hw":         "LLD\n(hardware)",
    "hld_hw":         "HLD\n(hardware)",
}


def _get(results: Dict, dec: str, d: int, p_values: List[float]):
    if dec not in results or d not in results[dec]:
        return [], []
    dr = results[dec][d]
    ps = sorted(p for p in p_values if p in dr)
    return ps, [dr[p] for p in ps]


# ── Plot A ─────────────────────────────────────────────────────────────────

def plot_A_ler_vs_per(
    all_results: Dict,
    distances: List[int] = DISTANCES,
    p_values: List[float] = P_VALUES,
    save_dir: str = RESULTS_FINAL,
) -> None:
    """
    Plot A: LER vs PER per distanza, tutti e 4 i decoder sullo stesso asse.

    Ordinamento atteso a ogni p:
      LLD >= Custom MWPM >= HLD >= PyMatching (ottimale)
    Il pseudo-threshold (cerchio colorato) quantifica la qualita' di ogni decoder.
    PyMatching e' il riferimento teorico: usa il p reale per la calibrazione.
    """
    fig, axes = plt.subplots(1, len(distances), figsize=(5.5 * len(distances), 4.8))
    if len(distances) == 1:
        axes = [axes]

    fig.suptitle(
        "Plot A — LER vs PER per distanza  ·  Tutti i decoder a confronto",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for i, d in enumerate(distances):
        ax = axes[i]
        setup_log_ax(ax, title=f"d = {d}")

        all_p = [p for dec in _ORDER4 for p in _get(all_results, dec, d, p_values)[0]]
        if all_p:
            draw_diagonal(ax, p_min=min(all_p) * 0.7, p_max=max(all_p) * 1.3)
            add_qec_benefit_region(ax, min(all_p) * 0.7, max(all_p) * 1.3)

        for dec in _ORDER4:
            ps, ls = _get(all_results, dec, d, p_values)
            if not ps:
                continue
            label = pseudo_threshold_label(_N4[dec], ps, ls)
            plot_ler_curve(ax, ps, ls, color=_C4[dec], label=label, marker=_M4[dec])
            mark_pseudo_threshold(ax, ps, ls, _C4[dec])

        autoscale_log_ax(ax)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=7.5)

    plt.tight_layout()
    save_figure(fig, save_dir, "plotA_ler_vs_per")
    plt.close(fig)


# ── Plot B ─────────────────────────────────────────────────────────────────

def plot_B_scalability(
    all_results: Dict,
    distances: List[int] = DISTANCES,
    p_values: List[float] = P_VALUES,
    save_dir: str = RESULTS_FINAL,
) -> None:
    """
    Plot B: Scalabilita' con la distanza, 4 subplot (LLD / Custom MWPM / HLD / PyMatching).

    LLD:         QEC scaling INVERTITO, d=7 ha LER piu' alta di d=3.
                 Dimostra che senza struttura QEC la NN non scala.
    Custom MWPM: scaling intermedio, MWPM con P_ASSUMED=0.30 fisso (PED deterministico),
                 subottimale per p << 0.30 ma bias sistematico apprendibile dall'HLD.
    HLD:         QEC scaling CORRETTO, d=7 < d=5 < d=3 a basso p.
                 Custom MWPM come PED + NN classificatrice (paper approach).
    PyMatching:  QEC scaling OTTIMALE, riferimento teorico superiore.
    """
    decoders_b = ["lld", "custom_mwpm", "hld", "pymatching"]

    fig, axes = plt.subplots(1, len(decoders_b), figsize=(5.5 * len(decoders_b), 4.8),
                              sharey=False)

    fig.suptitle(
        "Plot B — Scalabilita' con la distanza  ·  4 decoder a confronto",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for i, dec in enumerate(decoders_b):
        ax = axes[i]
        setup_log_ax(ax, title=_N4[dec])

        all_p = [p for d in distances for p in _get(all_results, dec, d, p_values)[0]]
        if all_p:
            draw_diagonal(ax, p_min=min(all_p) * 0.7, p_max=max(all_p) * 1.3)
            add_qec_benefit_region(ax, min(all_p) * 0.7, max(all_p) * 1.3)

        for d in distances:
            ps, ls = _get(all_results, dec, d, p_values)
            if not ps:
                continue
            label = pseudo_threshold_label(f"d={d}", ps, ls)
            plot_ler_curve(ax, ps, ls, color=_CDIST[d], label=label, marker=_MDIST[d])
            mark_pseudo_threshold(ax, ps, ls, _CDIST[d])

        autoscale_log_ax(ax)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    save_figure(fig, save_dir, "plotB_scalability")
    plt.close(fig)


# ── Plot C ─────────────────────────────────────────────────────────────────

def plot_C_sycamore(
    sycamore_data: Dict,
    distances: List[int] = DISTANCES,
    save_dir: str = RESULTS_FINAL,
) -> None:
    """
    Plot C: Google Sycamore, LER a r=1 su hardware reale.

    6 decoder a confronto:
      Sintetici (addestrati su Stim): LLD, Custom MWPM, HLD, PyMatching
      Hardware (fine-tuned su Sycamore): LLD_hw, HLD_hw

    Separatore visivo tra le due famiglie.
    Dati: Google QAI, Nature 614 (2023), Zenodo.
    """
    our_ler   = sycamore_data.get("our_ler_r1", {})
    dist_found = [d for d in [3, 5] if d in distances]
    if not dist_found:
        print("  [Plot C] Nessuna distanza Sycamore compatibile — saltato.")
        return
    if not any(our_ler.get(dec) for dec in _OSYC):
        print("  [Plot C] Nessun dato LER r=1 disponibile — saltato.")
        return

    _D_MARKER = {3: "v", 5: "o"}
    _D_OFFSET = {3: -0.13, 5: +0.13}
    _D_COLOR  = {3: "#C0392B", 5: "#2471A3"}
    _D_LABEL  = {3: "d = 3", 5: "d = 5"}

    # posizioni x: 0-3 sintetici | gap | 5-6 hardware-trained
    x_pos = {"lld": 0, "custom_mwpm": 1, "hld": 2, "pymatching_our": 3,
              "lld_hw": 5, "hld_hw": 6}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle(
        "Plot C — Google Sycamore  ·  Hardware reale (r=1, ~50 000 shot/centro)\n"
        "LER (%)  |  sintetici (sinistra)  vs  fine-tuned su hardware (destra)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    all_ler_vals = []
    legend_done  = set()

    for dec in _OSYC:
        ler_d = our_ler.get(dec, {})
        xp    = x_pos[dec]
        for d in dist_found:
            val = ler_d.get(d)
            if val is None or val <= 0:
                continue
            x     = xp + _D_OFFSET[d]
            label = _D_LABEL[d] if d not in legend_done else "_nolegend_"
            ax.plot(x, val * 100,
                    marker=_D_MARKER[d], color=_D_COLOR[d],
                    markersize=11, markeredgecolor="white", markeredgewidth=0.8,
                    linestyle="none", zorder=4, label=label)
            legend_done.add(d)
            all_ler_vals.append(val * 100)
            ax.text(x + 0.09, val * 100, f"{val*100:.2f}%",
                    color=_D_COLOR[d], fontsize=7.5, va="center", fontweight="bold")

    # Separatore visivo tra sintetici e hardware-trained
    ax.axvline(x=4, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([_NSYC[d] for d in _OSYC], fontsize=8.5)

    ax.set_ylabel("LER (%)", fontsize=12)
    ax.set_yscale("log")
    ax.set_xlim(-0.6, 7.0)
    if all_ler_vals:
        ax.set_ylim(min(all_ler_vals) * 0.3, max(all_ler_vals) * 5.0)

    ax.grid(True, which="major", linestyle="-",  alpha=0.25, linewidth=0.8)
    ax.grid(True, which="minor", linestyle=":",  alpha=0.12, linewidth=0.5)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    if all_ler_vals:
        ybot = min(all_ler_vals) * 0.4
        ax.text(1.5, ybot, "Addestrati su Stim", fontsize=8.5,
                color="#555555", ha="center", va="bottom", style="italic")
        ax.text(5.5, ybot, "Fine-tuned su Sycamore", fontsize=8.5,
                color="#555555", ha="center", va="bottom", style="italic")

    ax.text(0.01, 0.02, "Dati: Google QAI, Nature 614 (2023)",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            ha="left", color="gray", style="italic")

    plt.tight_layout()
    save_figure(fig, save_dir, "plotC_sycamore")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────

def plot_all(
    all_results: Dict,
    sycamore_data: Optional[Dict] = None,
    distances: List[int] = DISTANCES,
    p_values: List[float] = P_VALUES,
    save_dir: str = RESULTS_FINAL,
) -> None:
    """Genera i 3 plot definitivi in save_dir."""
    print("  Plot A (LER vs PER — tutti i decoder per distanza)...")
    plot_A_ler_vs_per(all_results, distances, p_values, save_dir)

    print("  Plot B (scalabilita' con d — LLD / HLD / PyMatching)...")
    plot_B_scalability(all_results, distances, p_values, save_dir)

    print("  Plot C (Google Sycamore — LER r=1 tutti i decoder)...")
    has_c = bool(sycamore_data and sycamore_data.get("our_ler_r1"))
    if has_c:
        plot_C_sycamore(sycamore_data, distances, save_dir)
    else:
        print("  [Plot C] Sycamore non disponibile.")
