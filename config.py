"""
Configurazione globale del progetto di benchmarking QEC.
Contiene costanti, path e parametri condivisi tra tutti i moduli.
"""

# ── Distanze del surface code da simulare ──────────────────────────────────
DISTANCES = [3, 5, 7, 9]

# ── Sweep di physical error rate per i dati sintetici ──────────────────────
import numpy as np
P_VALUES = np.logspace(np.log10(0.001), np.log10(0.15), 18).tolist()

# ── Shots per combinazione (d, p) sui dati sintetici ──────────────────────
N_SHOTS = 10_000

# ── Path del dataset Google Sycamore (Zenodo) ─────────────────────────────
GOOGLE_DATASET_PATH = r"C:\Users\caucc\OneDrive\Desktop\google_qec3v5_experiment_data"

# ── Parametri training reti neurali ───────────────────────────────────────
NN_LR         = 1e-3
NN_BATCH_SIZE = 128
NN_EPOCHS     = 20

# ── Pseudo-threshold di PyMatching (MWPM ottimale) per noise CIRCUIT-LEVEL ──
# Questi sono i p_th per stim circuit-level depolarizing (NON code-level).
# Derivati dai plot sperimentali (dove la curva PyMatching attraversa y=x).
# Nota: i valori code-level del paper (0.0825, 0.1037, 0.1137) NON si applicano qui.
# LLD e HLD vengono addestrati vicino a questi p_th per massimizzare il segnale.
MWPM_PTH = {3: 0.003, 5: 0.007, 7: 0.010, 9: 0.013}

# ── p di training HLD per distanza ────────────────────────────────────────
# Solo p vicini a p_th (pseudo-threshold di MWPM per circuit-level noise).
# PM!=Custom è massimo qui (~4-8%), il che garantisce un segnale adeguato
# per la rete e pos_weight elevato senza dominanza eccessiva dei negativi.
# Safety valve in HLDDecoder (fire_rate > 0.30 → fallback a CustomMWPM)
# gestisce il regime a basso p senza bisogno di regolarizzatore.
HLD_TRAIN_P = {
    3: [0.002, 0.003, 0.005],
    5: [0.005, 0.007, 0.010],
    7: [0.007, 0.010, 0.013],
    9: [0.010, 0.013, 0.016],
}

# ── Output directories ────────────────────────────────────────────────────
import os as _os
_BASE = _os.path.dirname(_os.path.abspath(__file__))
RESULTS_CUSTOM     = _os.path.join(_BASE, "results", "custom")
RESULTS_PYMATCHING = _os.path.join(_BASE, "results", "pymatching")
RESULTS_FINAL      = _os.path.join(_BASE, "results", "final")

# ── Palette colori: una per distanza, una per decoder ─────────────────────
DIST_COLORS    = {3: "tab:red", 5: "tab:green", 7: "tab:blue", 9: "tab:purple"}
DECODER_COLORS = {
    "lld":               "tab:red",
    "custom_mwpm":       "tab:orange",
    "hld":               "tab:blue",
    "pymatching":        "tab:purple",
    "correlated_matching": "tab:cyan",
    "belief_matching":   "tab:olive",
}
