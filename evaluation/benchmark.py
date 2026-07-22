"""
Benchmarking unificato: addestra LLD e HLD e valuta tutti e 4 i decoder.

Modalità CPU: conservativa, ~20-35 min, rete 256+64, training moderato.
Modalità GPU: aggressiva, ~60-90 min, rete 512+256, training pesante.

Checkpoint:
  CPU → results/checkpoints/cpu/  (hld_d{d}.pt, lld_d{d}.pt)
  GPU → results/checkpoints/gpu/  (hld_d{d}.pt, lld_d{d}.pt)
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (DISTANCES, HLD_TRAIN_P, MWPM_PTH, N_SHOTS, P_VALUES,
                    RESULTS_FINAL)
from data.synthetic_dataset import SyntheticDataset, generate_all_eval_data, generate_synthetic_data
from decoders import CustomMWPMDecoder, HLDDecoder, LLDDecoder
from evaluation.metrics import compute_ler

BenchmarkResults = Dict[str, Dict[int, Dict[float, float]]]

# ── Checkpoint directories ─────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_CPU_DIR = os.path.join(_BASE, "results", "checkpoints", "cpu")
CKPT_GPU_DIR = os.path.join(_BASE, "results", "checkpoints", "gpu")


# ─── Run configurations ────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """
    Configurazione completa del benchmark: architettura NN + schedule di training.

    CPU_CONFIG: conservativa, ~20-35 min su CPU.
                hld_shots_each ridotto + CosineAnnealingLR per convergenza rapida.
    GPU_CONFIG: aggressiva, ~60-90 min su GPU, progettata per avvicinarsi o
                battere PyMatching catturando correlazioni nel circuit-level noise.
    """
    # NN architecture
    h1: int = 256           # primo hidden layer
    h2: int = 64            # secondo hidden layer
    # LLD training
    lld_shots:  int = 8_000
    lld_epochs: int = 15
    lld_lr:     float = 1e-3
    # HLD training: multi-p (d=3, d=5)
    hld_shots_each: int = 12_000  # per valore di p (usa HLD_TRAIN_P da config)
    hld_epochs:     int = 100
    hld_lr:         float = 3e-3  # LR iniziale CosineAnnealingLR
    hld_shots_large_d: int = 20_000  # unused, kept for backward compat
    # Safety valve
    hld_max_fire_rate: float = 0.30
    # Batch size
    batch_size: int = 256


# ── CPU: rete paper, training veloce con CosineAnnealingLR ────────────────
CPU_CONFIG = RunConfig()

# ── GPU: rete più grande, training pesante ────────────────────────────────
GPU_CONFIG = RunConfig(
    h1=512, h2=256,
    lld_shots=30_000,  lld_epochs=40,  lld_lr=5e-4,
    hld_shots_each=20_000, hld_epochs=150, hld_lr=2e-3,
    hld_shots_large_d=50_000,
    hld_max_fire_rate=0.30,
    batch_size=512,
)


def _gen_lld_data(d: int, p: float, shots: int) -> Tuple[np.ndarray, np.ndarray]:
    """Genera training data LLD al p_th di MWPM per distanza d."""
    ds = generate_synthetic_data(d, p, n_shots=shots, seed=42)
    return ds.syndromes, ds.logical_flips


def _gen_multi_p(d: int, p_list: List[float], shots: int) -> Tuple[np.ndarray, np.ndarray]:
    """Genera training data su più valori di p e li concatena."""
    all_s, all_f = [], []
    for i, p in enumerate(p_list):
        ds = generate_synthetic_data(d, p, n_shots=shots, seed=i * 17)
        all_s.append(ds.syndromes)
        all_f.append(ds.logical_flips)
    return np.concatenate(all_s), np.concatenate(all_f)


def _gen_hld_data(
    d: int, p_list: List[float], shots: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera dati di training per HLD, oracle = PyMatching ottimale.

    Target passato a HLD.train = PM_pred (predizione PyMatching).
    Internamente HLD.train fa: residual = PM_pred XOR Custom_MWPM_pred.
    Il target è deterministico → zero label noise → apprendimento pulito.
    """
    import pymatching
    from data.synthetic_dataset import _build_stim_circuit

    all_s, all_pm = [], []
    for i, p in enumerate(p_list):
        ds = generate_synthetic_data(d, p, n_shots=shots, seed=i * 31 + 7)
        circuit = _build_stim_circuit(d, p)
        dem     = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        preds    = matcher.decode_batch(ds.syndromes.astype(np.uint8))
        pm_preds = preds[:, 0].astype(int)

        base     = CustomMWPMDecoder(d).predict(ds.syndromes)
        disagree = float((pm_preds != base).mean())
        pm_ler   = float((pm_preds != ds.logical_flips).mean())
        print(f"    d={d} p={p:.3f}: PM!=Custom {disagree*100:.1f}%  PM_LER={pm_ler*100:.1f}%")

        all_s.append(ds.syndromes)
        all_pm.append(pm_preds)
    return np.concatenate(all_s), np.concatenate(all_pm)


def _eval_fn(
    fn,
    eval_data: Dict,
    distances: List[int],
    p_values: List[float],
    label: str,
) -> Dict[int, Dict[float, float]]:
    """Valuta una callable fn(d, syndromes) → predictions su tutti (d, p)."""
    results = {d: {} for d in distances}
    for d in distances:
        print(f"  {label:18s} d={d}", end="", flush=True)
        for p in p_values:
            data = eval_data[(d, p)]
            preds = fn(d, data.syndromes)
            results[d][p] = compute_ler(preds, data.logical_flips)
            print(".", end="", flush=True)
        lo, hi = min(results[d].values()), max(results[d].values())
        print(f"  LER [{lo:.4f}–{hi:.4f}]")
    return results


def run_full_benchmark(
    distances: Optional[List[int]] = None,
    p_values: Optional[List[float]] = None,
    n_shots: int = N_SHOTS,
    cfg: RunConfig = CPU_CONFIG,
    mode: str = "cpu",
) -> BenchmarkResults:
    """
    Benchmark completo: allena LLD e HLD, valuta tutti e 4 i decoder.
    Salva i pesi dopo training in results/checkpoints/{mode}/.

    Parameters
    ----------
    mode : "cpu" o "gpu", determina la cartella checkpoint
    """
    if distances is None: distances = DISTANCES
    if p_values  is None: p_values  = P_VALUES

    ckpt_dir = CKPT_GPU_DIR if mode == "gpu" else CKPT_CPU_DIR
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n" + "="*65)
    print("BENCHMARK  —  LLD / Custom MWPM / HLD / PyMatching")
    print(f"Modalità: {mode.upper()}  h1={cfg.h1} h2={cfg.h2}")
    print(f"LLD  {cfg.lld_shots}shot×{cfg.lld_epochs}ep  lr={cfg.lld_lr:.0e}")
    print(f"HLD  {cfg.hld_shots_each}shot/p ×{cfg.hld_epochs}ep  lr={cfg.hld_lr:.0e}  CosineAnnealingLR  pos_weight_cap=40")
    print(f"Eval {len(distances)} dist × {len(p_values)} p × {n_shots} shot")
    print(f"Checkpoints → {ckpt_dir}")
    print("="*65)

    # ── 1. Eval data ──────────────────────────────────────────────────────
    print(f"\n[1/3] Generazione eval data...")
    eval_data = generate_all_eval_data(distances, p_values, n_shots=n_shots)

    # ── 2. Training ────────────────────────────────────────────────────────
    print("\n[2/3] Training NN...")
    lld_dec: Dict[int, LLDDecoder] = {}
    hld_dec: Dict[int, HLDDecoder] = {}

    for d in distances:
        hld_p_list = HLD_TRAIN_P[d]
        p_lld = MWPM_PTH[d]
        print(f"\n  ── d = {d} ({'input=' + str(d*d-1) + ' features'}) ─────────────────")

        # LLD
        print(f"  LLD: {cfg.lld_shots} shot @ p_th={p_lld:.4f}")
        syn_l, flip_l = _gen_lld_data(d, p_lld, cfg.lld_shots)
        lld = LLDDecoder(d, h1=cfg.h1, h2=cfg.h2)
        lld.train(syn_l, flip_l, lr=cfg.lld_lr, batch_size=cfg.batch_size, epochs=cfg.lld_epochs)
        lld_path = os.path.join(ckpt_dir, f"lld_d{d}.pt")
        lld.save(lld_path)
        lld_dec[d] = lld

        # HLD: multi-p vicino a p_th, pos_weight_cap=40 (default HLDDecoder)
        n_hld = len(hld_p_list) * cfg.hld_shots_each
        print(f"  HLD: {n_hld} shot su p={hld_p_list}  target=PM_oracle")
        syn_h, flip_h = _gen_hld_data(d, hld_p_list, cfg.hld_shots_each)
        hld = HLDDecoder(d, h1=cfg.h1, h2=cfg.h2, max_fire_rate=cfg.hld_max_fire_rate)
        hld.train(syn_h, flip_h, lr=cfg.hld_lr, batch_size=cfg.batch_size,
                  epochs=cfg.hld_epochs)
        hld_path = os.path.join(ckpt_dir, f"hld_d{d}.pt")
        hld.save(hld_path)
        print(f"  [ckpt] Salvato: {os.path.basename(hld_path)}")
        hld_dec[d] = hld

    # ── 3. Valutazione ────────────────────────────────────────────────────
    print("\n[3/3] Valutazione...")

    mwpm = {d: CustomMWPMDecoder(d) for d in distances}
    res_mwpm = _eval_fn(lambda d, s: mwpm[d].predict(s),    eval_data, distances, p_values, "Custom MWPM")
    res_lld  = _eval_fn(lambda d, s: lld_dec[d].predict(s), eval_data, distances, p_values, "LLD")
    res_hld  = _eval_fn(lambda d, s: hld_dec[d].predict(s), eval_data, distances, p_values, "HLD")

    # PyMatching ottimale
    import pymatching
    from data.synthetic_dataset import _build_stim_circuit

    res_pm: Dict[int, Dict[float, float]] = {d: {} for d in distances}
    for d in distances:
        print(f"  {'PyMatching':18s} d={d}", end="", flush=True)
        for p in p_values:
            data    = eval_data[(d, p)]
            circuit = _build_stim_circuit(d, p)
            dem     = circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            preds        = matcher.decode_batch(data.syndromes.astype(np.uint8))
            res_pm[d][p] = compute_ler(preds[:, 0].astype(int), data.logical_flips)
            print(".", end="", flush=True)
        lo, hi = min(res_pm[d].values()), max(res_pm[d].values())
        print(f"  LER [{lo:.4f}–{hi:.4f}]")

    # ── Riepilogo ─────────────────────────────────────────────────────────
    print("\n  Confronto decoder — LER medio su tutti i p:")
    for d in distances:
        avg_lld  = np.mean(list(res_lld[d].values()))
        avg_mwpm = np.mean(list(res_mwpm[d].values()))
        avg_hld  = np.mean(list(res_hld[d].values()))
        avg_pm   = np.mean(list(res_pm[d].values()))
        pct_hld  = 100 * (avg_mwpm - avg_hld) / avg_mwpm if avg_mwpm > 0 else 0
        pct_pm   = 100 * (avg_mwpm - avg_pm)  / avg_mwpm if avg_mwpm > 0 else 0
        print(f"    d={d}: LLD={avg_lld:.4f}  MWPM={avg_mwpm:.4f}  "
              f"HLD={avg_hld:.4f} ({pct_hld:+.1f}%)  PM={avg_pm:.4f} ({pct_pm:+.1f}%)")

    # Scalabilità sotto-threshold: verifica gerarchia d a p≈p_th
    print("\n  Scalabilità a p≈p_th (atteso: PM < HLD < MWPM < LLD per ogni d):")
    for d in distances:
        p_near = min(p_values, key=lambda p: abs(p - MWPM_PTH[d]))
        print(f"    d={d} @ p={p_near:.4f}: "
              f"LLD={res_lld[d][p_near]:.4f}  MWPM={res_mwpm[d][p_near]:.4f}  "
              f"HLD={res_hld[d][p_near]:.4f}  PM={res_pm[d][p_near]:.4f}")

    final = {
        "lld":         res_lld,
        "custom_mwpm": res_mwpm,
        "hld":         res_hld,
        "pymatching":  res_pm,
    }
    cache = os.path.join(RESULTS_FINAL, mode, "benchmark_results.json")
    save_results(final, cache)
    print(f"[Cache] Salvato → {cache}")
    print("\n[Benchmark] Completato.")
    return final


def run_eval_from_checkpoints(
    distances=None,
    p_values=None,
    n_shots: int = N_SHOTS,
    cfg: RunConfig = CPU_CONFIG,
    mode: str = "cpu",
) -> Optional[BenchmarkResults]:
    """
    Valuta i 4 decoder dai checkpoint esistenti, senza rifare il training.
    Usato da main.py quando --skip-benchmark è attivo e la cache JSON manca.
    Restituisce None se mancano checkpoint.
    """
    if distances is None: distances = DISTANCES
    if p_values  is None: p_values  = P_VALUES

    ckpt_dir = CKPT_GPU_DIR if mode == "gpu" else CKPT_CPU_DIR
    missing  = [
        os.path.join(ckpt_dir, f"{pfx}_d{d}.pt")
        for d in distances for pfx in ("lld", "hld")
        if not os.path.isfile(os.path.join(ckpt_dir, f"{pfx}_d{d}.pt"))
    ]
    if missing:
        print("[EvalOnly] Checkpoint mancanti:")
        for m in missing: print(f"    {m}")
        return None

    print(f"\n[EvalOnly] Checkpoint {mode.upper()} — skip training, solo eval")
    print(f"  {len(distances)} dist × {len(p_values)} p × {n_shots} shot")

    print("[1/2] Generazione eval data...")
    eval_data = generate_all_eval_data(distances, p_values, n_shots=n_shots)

    lld_dec: Dict[int, LLDDecoder] = {}
    hld_dec: Dict[int, HLDDecoder] = {}
    for d in distances:
        lld = LLDDecoder(d, h1=cfg.h1, h2=cfg.h2)
        lld.load(os.path.join(ckpt_dir, f"lld_d{d}.pt"))
        lld_dec[d] = lld
        hld = HLDDecoder(d, h1=cfg.h1, h2=cfg.h2)
        hld.load(os.path.join(ckpt_dir, f"hld_d{d}.pt"))
        hld_dec[d] = hld
        print(f"  [ckpt] d={d} lld+hld caricati")

    print("[2/2] Valutazione...")
    mwpm     = {d: CustomMWPMDecoder(d) for d in distances}
    res_mwpm = _eval_fn(lambda d, s: mwpm[d].predict(s),    eval_data, distances, p_values, "Custom MWPM")
    res_lld  = _eval_fn(lambda d, s: lld_dec[d].predict(s), eval_data, distances, p_values, "LLD")
    res_hld  = _eval_fn(lambda d, s: hld_dec[d].predict(s), eval_data, distances, p_values, "HLD")

    import pymatching
    from data.synthetic_dataset import _build_stim_circuit

    res_pm: Dict[int, Dict[float, float]] = {d: {} for d in distances}
    for d in distances:
        print(f"  {'PyMatching':18s} d={d}", end="", flush=True)
        for p in p_values:
            data    = eval_data[(d, p)]
            circuit = _build_stim_circuit(d, p)
            dem     = circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            preds   = matcher.decode_batch(data.syndromes.astype(np.uint8))
            res_pm[d][p] = compute_ler(preds[:, 0].astype(int), data.logical_flips)
            print(".", end="", flush=True)
        lo, hi = min(res_pm[d].values()), max(res_pm[d].values())
        print(f"  LER [{lo:.4f}–{hi:.4f}]")

    final = {"lld": res_lld, "custom_mwpm": res_mwpm, "hld": res_hld, "pymatching": res_pm}
    cache = os.path.join(RESULTS_FINAL, mode, "benchmark_results.json")
    save_results(final, cache)
    print(f"[Cache] Salvato → {cache}")
    return final


def save_results(results: BenchmarkResults, path: str) -> None:
    """Salva i risultati del benchmark su JSON (chiavi convertite a stringa)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serial = {
        dec: {str(d): {str(p): v for p, v in by_p.items()}
              for d, by_p in by_d.items()}
        for dec, by_d in results.items()
    }
    with open(path, "w") as f:
        json.dump(serial, f, indent=2)


def load_results(path: str) -> Optional[BenchmarkResults]:
    """Carica risultati dal JSON. Restituisce None se il file non esiste."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {
        dec: {int(d): {float(p): v for p, v in by_p.items()}
              for d, by_p in by_d.items()}
        for dec, by_d in raw.items()
    }


def run_benchmark_custom(distances=None, p_values=None, n_shots=N_SHOTS, cfg=CPU_CONFIG) -> BenchmarkResults:
    return run_full_benchmark(distances, p_values, n_shots, cfg)


def run_benchmark_pymatching(distances=None, p_values=None, n_shots=N_SHOTS, cfg=CPU_CONFIG) -> BenchmarkResults:
    return run_full_benchmark(distances, p_values, n_shots, cfg)


def evaluate_decoders_on_sycamore(
    shots_by_d: Dict,
    ckpt_dir: str = CKPT_CPU_DIR,
    h1: int = CPU_CONFIG.h1,
    h2: int = CPU_CONFIG.h2,
) -> Dict[str, Dict[int, float]]:
    """
    Valuta i nostri 4 decoder su dati Sycamore hardware (r=1 soltanto).

    LLD e HLD vengono caricati dai checkpoint salvati durante il benchmark.
    PyMatching usa il DEM hardware calibrato (circuit_detector_error_model.dem).
    CustomMWPM usa P_ASSUMED=0.30 (deterministico).

    Parameters
    ----------
    shots_by_d : dict {d: List[SycamoreShot]}
        Dataset Sycamore già caricato (da GoogleSycamoreLoader.load_all).
    ckpt_dir : str
        Directory dei checkpoint (cpu/ o gpu/).
    h1, h2 : int
        Dimensioni hidden layer, devono corrispondere al checkpoint.

    Returns
    -------
    dict: decoder_name → {d: LER_r1}
        LER a r=1 su hardware per ogni decoder e distanza.
        Decoder mancanti (checkpoint non trovato) sono assenti dal dict.
    """
    import pymatching
    import stim as _stim

    results: Dict[str, Dict[int, float]] = {
        "lld": {}, "custom_mwpm": {}, "hld": {}, "pymatching_our": {},
        "lld_hw": {}, "hld_hw": {},
    }

    for d, shots in shots_by_d.items():
        # Solo r=1 con detection events disponibili
        r1 = [s for s in shots if s.rounds == 1 and s.detection_events is not None]
        if not r1:
            continue

        syndromes  = np.concatenate([s.detection_events    for s in r1]).astype(np.float32)
        labels     = np.concatenate([s.actual_flips        for s in r1])
        pm_oracle  = np.concatenate([s.predicted_pymatching for s in r1])
        n_total    = len(labels)
        if n_total == 0:
            continue

        print(f"  [Sycamore eval] d={d} r=1: {n_total} shot da {len(r1)} centri")

        # ── Train/test split 80/20 per hardware-trained ────────────────────
        rng   = np.random.RandomState(42)
        idx   = rng.permutation(n_total)
        split = int(n_total * 0.8)
        tr, te = idx[:split], idx[split:]
        syn_tr, pm_tr = syndromes[tr], pm_oracle[tr]
        syn_te, lbl_te        = syndromes[te], labels[te]
        print(f"    split: {len(tr)} train  {len(te)} test")

        # ── PyMatching, per-centro con DEM hardware calibrato ─────────────
        pm_errors, pm_total = 0, 0
        for shot in r1:
            if shot.dem_path and os.path.exists(shot.dem_path) and shot.detection_events is not None:
                try:
                    dem     = _stim.DetectorErrorModel.from_file(shot.dem_path)
                    matcher = pymatching.Matching.from_detector_error_model(dem)
                    syns    = shot.detection_events.astype(np.uint8)
                    preds   = matcher.decode_batch(syns)[:, 0].astype(int)
                    pm_errors += int((preds != shot.actual_flips[:len(preds)]).sum())
                    pm_total  += len(preds)
                except Exception as e:
                    print(f"    [PyMatching center={shot.center}] errore: {e}")
        if pm_total > 0:
            results["pymatching_our"][d] = pm_errors / pm_total

        # ── CustomMWPM, deterministico, su tutti gli shot ─────────────────
        mwpm  = CustomMWPMDecoder(d)
        preds = mwpm.predict(syndromes)
        results["custom_mwpm"][d] = float((preds != labels).mean())

        # ── LLD / HLD sintetici (checkpoint esistenti, tutti gli shot) ──────
        lld_path = os.path.join(ckpt_dir, f"lld_d{d}.pt")
        hld_path = os.path.join(ckpt_dir, f"hld_d{d}.pt")

        if os.path.exists(lld_path):
            try:
                lld = LLDDecoder(d, h1=h1, h2=h2)
                lld.load(lld_path)
                results["lld"][d] = float((lld.predict(syndromes) != labels).mean())
            except Exception as e:
                print(f"    [LLD] checkpoint error: {e}")

        if os.path.exists(hld_path):
            try:
                hld = HLDDecoder(d, h1=h1, h2=h2)
                hld.load(hld_path)
                results["hld"][d] = float((hld.predict(syndromes) != labels).mean())
            except Exception as e:
                print(f"    [HLD] checkpoint error: {e}")

        # ── LLD_hw / HLD_hw, fine-tune su dati hardware reali ─────────────
        # Partono dai pesi .pt (non da zero) e si adattano al rumore Sycamore.
        # Oracle: predizioni PM di Google (già ottimali per l'hardware).
        if os.path.exists(lld_path):
            try:
                lld_hw = LLDDecoder(d, h1=h1, h2=h2)
                lld_hw.load(lld_path)                       # parte dai pesi esistenti
                lld_hw.train(syn_tr, pm_tr, epochs=40)      # fine-tune su hardware
                results["lld_hw"][d] = float((lld_hw.predict(syn_te) != lbl_te).mean())
                print(f"    LLD_hw trained on {len(syn_tr)} hw shots")
            except Exception as e:
                print(f"    [LLD_hw] errore: {e}")

        if os.path.exists(hld_path):
            try:
                hld_hw = HLDDecoder(d, h1=h1, h2=h2)
                hld_hw.load(hld_path)                       # parte dai pesi esistenti
                hld_hw.train(syn_tr, pm_tr, epochs=60)      # fine-tune su hardware
                results["hld_hw"][d] = float((hld_hw.predict(syn_te) != lbl_te).mean())
                print(f"    HLD_hw trained on {len(syn_tr)} hw shots")
            except Exception as e:
                print(f"    [HLD_hw] errore: {e}")

        # Log
        for dec, val in [(k, results[k].get(d)) for k in results]:
            if val is not None:
                print(f"    {dec:16s} d={d}: LER={val*100:.3f}%")

    return results
