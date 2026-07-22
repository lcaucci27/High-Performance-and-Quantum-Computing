"""
Entry point del progetto QEC benchmarking.
Esegue il benchmark completo e genera i 3 plot definitivi (A, B, C).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DISTANCES, GOOGLE_DATASET_PATH, N_SHOTS, P_VALUES, RESULTS_FINAL
from data.google_sycamore import GoogleSycamoreLoader
from evaluation.benchmark import (CPU_CONFIG, GPU_CONFIG, CKPT_CPU_DIR, CKPT_GPU_DIR,
                                   run_full_benchmark, run_eval_from_checkpoints,
                                   load_results, evaluate_decoders_on_sycamore)
from plotting.plots_final import plot_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QEC Decoder Benchmarking")
    p.add_argument("--skip-sycamore",  action="store_true")
    p.add_argument("--skip-benchmark", action="store_true",
                   help="Salta il benchmark NN (utile per re-plottare con risultati cached)")
    p.add_argument("--distances",      nargs="+", type=int, default=DISTANCES)
    p.add_argument("--n-shots",        type=int,  default=N_SHOTS)
    p.add_argument("--gpu",            action="store_true",
                   help="Usa GPU_CONFIG: rete 512+256, training pesante (60-120 min su GPU)")
    return p.parse_args()


def load_sycamore(distances: list) -> dict:
    """
    Carica il dataset Sycamore e restituisce:
      {
        "per":          {d: {decoder: (per_list, ler_round_list)}},
        "rounds":       {d: {decoder: (rounds_list, ler_list)}},
        "epsilon_L":    {d: {decoder: (eps, std)}},
        "google_ler_r1":{decoder: {d: LER}},   ← LER a r=1 decoder Google
        "raw_shots":    {d: List[SycamoreShot]} ← per evaluate_decoders_on_sycamore
      }
    """
    sycamore_dists = [d for d in [3, 5] if d in distances]
    if not sycamore_dists or not os.path.isdir(GOOGLE_DATASET_PATH):
        print(f"  [Sycamore] Path non trovato o non richiesto: {GOOGLE_DATASET_PATH}")
        return {}

    print(f"\n[Sycamore] Caricamento da: {GOOGLE_DATASET_PATH}")
    loader    = GoogleSycamoreLoader(GOOGLE_DATASET_PATH)
    all_shots = loader.load_all(distances=sycamore_dists)

    per_data     = {}
    rounds_data  = {}
    epsilon_data = {}

    for d, shots in all_shots.items():
        if not shots:
            continue
        per_data[d]     = loader.compute_ler_per_decoder(shots)
        rounds_data[d]  = loader.compute_ler_vs_rounds(shots)
        epsilon_data[d] = loader.compute_epsilon_L(shots)

        n_exp = len(shots)
        print(f"  d={d}: {n_exp} esperimenti caricati.")
        for dec, (eps, std) in epsilon_data[d].items():
            if eps is not None:
                print(f"    {dec}: εL = {eps:.3f}% ± {std:.3f}%")

    # LER a r=1 per i decoder Google (confronto diretto con i nostri)
    google_ler_r1 = loader.compute_google_ler_r1(all_shots)
    for dec, by_d in google_ler_r1.items():
        for d, ler in by_d.items():
            print(f"  [Google r=1] {dec} d={d}: LER={ler*100:.3f}%")

    return {
        "per":           per_data,
        "rounds":        rounds_data,
        "epsilon_L":     epsilon_data,
        "google_ler_r1": google_ler_r1,
        "raw_shots":     all_shots,
    }


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 70)
    print("QEC DECODER BENCHMARKING")
    print(f"Distanze: {args.distances} | Shot: {args.n_shots} | p-values: {len(P_VALUES)} punti")
    print("=" * 70)

    cfg = GPU_CONFIG if args.gpu else CPU_CONFIG
    print(f"Modalità: {'GPU (aggressiva)' if args.gpu else 'CPU (conservativa)'}  "
          f"h1={cfg.h1} h2={cfg.h2}")

    mode     = "gpu" if args.gpu else "cpu"
    ckpt_dir = CKPT_GPU_DIR if args.gpu else CKPT_CPU_DIR
    save_dir = os.path.join(RESULTS_FINAL, mode)
    os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()

    # ── Sycamore ──────────────────────────────────────────────────────────
    sycamore_data = {}
    if not args.skip_sycamore:
        sycamore_data = load_sycamore(args.distances)

    # ── Benchmark o caricamento da cache ──────────────────────────────────
    all_results = None
    cache_path  = os.path.join(save_dir, "benchmark_results.json")

    if args.skip_benchmark:
        # 1. prova cache JSON
        all_results = load_results(cache_path)
        if all_results is not None:
            print(f"[Cache] Caricato da {cache_path}")
        else:
            # 2. nessuna cache → ricalcola da checkpoint senza ritraining
            print("[Cache] Non trovata — ricalcolo da checkpoint esistenti...")
            all_results = run_eval_from_checkpoints(
                distances=args.distances,
                p_values=P_VALUES,
                n_shots=args.n_shots,
                cfg=cfg,
                mode=mode,
            )
    else:
        all_results = run_full_benchmark(
            distances=args.distances,
            p_values=P_VALUES,
            n_shots=args.n_shots,
            cfg=cfg,
            mode=mode,
        )

    # ── Valutazione decoder su hardware Sycamore (r=1) ────────────────────
    if sycamore_data and sycamore_data.get("raw_shots"):
        print("\n[Sycamore] Valutazione decoder nostri su hardware r=1...")
        our_ler = evaluate_decoders_on_sycamore(
            shots_by_d=sycamore_data["raw_shots"],
            ckpt_dir=ckpt_dir,
            h1=cfg.h1,
            h2=cfg.h2,
        )
        sycamore_data["our_ler_r1"] = our_ler

    # ── Plot ──────────────────────────────────────────────────────────────
    if all_results is not None:
        print("\n" + "=" * 60)
        print(f"GENERAZIONE PLOT ({mode.upper()}) → {save_dir}")
        print("=" * 60)
        plot_all(
            all_results=all_results,
            sycamore_data=sycamore_data,
            distances=args.distances,
            p_values=P_VALUES,
            save_dir=save_dir,
        )
    else:
        print("[ERRORE] Nessun risultato disponibile — impossibile generare i plot.")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Completato in {elapsed:.1f}s")
    print(f"  {save_dir}/  → plotA, plotB, plotC")
    print("=" * 70)


if __name__ == "__main__":
    main()
