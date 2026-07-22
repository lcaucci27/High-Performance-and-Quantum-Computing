"""
Loader del dataset Google Sycamore (Zenodo) per la valutazione dei decoder QEC.
Carica le predizioni di pymatching, correlated_matching e belief_matching
dal dataset sperimentale, ed estrae il PER dal circuito Stim calibrato.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import stim
import yaml


@dataclass
class SycamoreShot:
    """
    Dati per un singolo esperimento Sycamore (distance + rounds + center).

    Attributes
    ----------
    distance : int
        Distanza del codice (3 o 5).
    rounds : int
        Numero di rounds QEC (1, 3, 5, ..., 25).
    center : str
        Identificatore del centro del patch (es. '3_5').
    n_shots : int
        Numero di shot sperimentali.
    actual_flips : np.ndarray, shape (n_shots,)
        Flip logico effettivo misurato.
    predicted_pymatching : np.ndarray, shape (n_shots,)
        Predizioni di pymatching.
    predicted_correlated : np.ndarray, shape (n_shots,)
        Predizioni di correlated matching.
    predicted_belief : np.ndarray, shape (n_shots,)
        Predizioni di belief matching.
    per : float
        Physical error rate estratto dal circuito calibrato.
    detection_events : np.ndarray or None, shape (n_shots, n_det), dtype uint8
        Detection events hardware (ancilla triggerate), None se non disponibili.
    dem_path : str or None
        Percorso al file circuit_detector_error_model.dem per PyMatching.
    """
    distance:              int
    rounds:                int
    center:                str
    n_shots:               int
    actual_flips:          np.ndarray
    predicted_pymatching:  np.ndarray
    predicted_correlated:  np.ndarray
    predicted_belief:      np.ndarray
    per:                   float
    detection_events:      Optional[np.ndarray] = None
    dem_path:              Optional[str] = None


def _parse_01_file(path: str) -> np.ndarray:
    """
    Legge un file .01 (formato ASCII Sycamore) come array di 0 e 1.

    Formato: ogni riga contiene il carattere ASCII '0' o '1' seguito da newline.
    I byte nel file sono: 48 ('0'), 49 ('1'), 10 (newline).

    Parameters
    ----------
    path : str
        Percorso al file .01.

    Returns
    -------
    np.ndarray, dtype int
        Array di 0 e 1, shape (n_shots,).
    """
    raw = np.fromfile(path, dtype=np.uint8)
    # Rimuovi newline (0x0A = 10), converti ASCII: '0'=48 → 0, '1'=49 → 1
    data = raw[raw != 10] - 48
    return data.astype(int)


# Mantieni compatibilità con il vecchio nome usato internamente
_parse_binary_file = _parse_01_file


def _extract_per_from_stim(circuit: stim.Circuit) -> float:
    """
    Estrae il Physical Error Rate dal circuito Stim calibrato.

    Il PER è calcolato come media dei valori DEPOLARIZE2 presenti
    nel circuito, che rappresenta il rumore a due qubit calibrato
    sull'hardware Sycamore.

    Parameters
    ----------
    circuit : stim.Circuit
        Circuito Stim già caricato.

    Returns
    -------
    float
        Media dei valori DEPOLARIZE2 come stima del PER.
    """
    depol2_values: List[float] = []
    for instruction in circuit:
        if hasattr(instruction, 'name') and instruction.name == "DEPOLARIZE2":
            for arg in instruction.gate_args_copy():
                depol2_values.append(float(arg))

    if depol2_values:
        return float(np.mean(depol2_values))

    # Fallback: cerca DEPOLARIZE1
    depol1_values: List[float] = []
    for instruction in circuit:
        if hasattr(instruction, 'name') and instruction.name == "DEPOLARIZE1":
            for arg in instruction.gate_args_copy():
                depol1_values.append(float(arg))
    if depol1_values:
        return float(np.mean(depol1_values))
    return 0.005


def _parse_properties(props_path: str) -> Dict:
    """Legge il file properties.yml di un esperimento Sycamore."""
    with open(props_path, "r") as f:
        return yaml.safe_load(f)


class GoogleSycamoreLoader:
    """
    Loader per il dataset Google Sycamore dal path locale (Zenodo).

    Struttura attesa delle directory:
      {code}_b{basis}_d{distance}_r{rounds}_center_{row}_{col}/

    Files per directory:
      - properties.yml
      - obs_flips_actual.01
      - obs_flips_predicted_by_pymatching.01
      - obs_flips_predicted_by_correlated_matching.01
      - circuit_noisy.stim
      - detection_events.b8            (detection events hardware, n_det bit/shot)
      - circuit_detector_error_model.dem  (DEM per PyMatching)
    """

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)

    def _find_experiment_dirs(self) -> List[Path]:
        dirs = []
        seen_names: set = set()
        for item in self.base_path.iterdir():
            if not item.is_dir() or item.name.startswith("repetition_code"):
                continue
            # Deduplicazione: su OneDrive possono comparire path logici duplicati
            if item.name in seen_names:
                continue
            if "obs_flips_actual.01" in [f.name for f in item.iterdir()]:
                dirs.append(item)
                seen_names.add(item.name)
        return sorted(dirs)

    def _load_single_experiment(self, exp_dir: Path) -> Optional[SycamoreShot]:
        required_files = [
            "properties.yml",
            "obs_flips_actual.01",
            "obs_flips_predicted_by_pymatching.01",
            "obs_flips_predicted_by_correlated_matching.01",
            "circuit_noisy.stim",
        ]
        for fname in required_files:
            if not (exp_dir / fname).exists():
                return None

        props    = _parse_properties(str(exp_dir / "properties.yml"))
        distance = int(props.get("distance", props.get("d", 3)))
        rounds   = int(props.get("rounds",   props.get("r", 1)))
        n_shots  = int(props.get("shots",    50_000))

        dir_name = exp_dir.name
        parts    = dir_name.split("_center_")
        center   = parts[1] if len(parts) > 1 else "unknown"

        # Carica i flip (formato ASCII .01)
        actual  = _parse_01_file(str(exp_dir / "obs_flips_actual.01"))
        pm_pred = _parse_01_file(str(exp_dir / "obs_flips_predicted_by_pymatching.01"))
        cm_pred = _parse_01_file(str(exp_dir / "obs_flips_predicted_by_correlated_matching.01"))
        bm_path = exp_dir / "obs_flips_predicted_by_belief_matching.01"
        bm_pred = _parse_01_file(str(bm_path)) if bm_path.exists() else pm_pred.copy()

        n = min(len(actual), n_shots)

        # Carica il circuito Stim (usato per PER + n_detectors)
        circuit  = stim.Circuit.from_file(str(exp_dir / "circuit_noisy.stim"))
        per      = _extract_per_from_stim(circuit)
        n_det    = circuit.num_detectors  # ancilla per round (d²-1 per r=1)

        # Carica detection events (detection_events.b8: n_det bit/shot, NO observable)
        detection_events = None
        det_path = exp_dir / "detection_events.b8"
        if det_path.exists():
            try:
                raw_evts = stim.read_shot_data_file(
                    path=str(det_path),
                    format="b8",
                    num_measurements=n_det,
                    bit_packed=False,
                )
                detection_events = raw_evts[:n].astype(np.uint8)
            except Exception:
                detection_events = None

        # Path al DEM per PyMatching hardware-calibrato
        dem_candidate = exp_dir / "circuit_detector_error_model.dem"
        dem_path = str(dem_candidate) if dem_candidate.exists() else None

        return SycamoreShot(
            distance=distance,
            rounds=rounds,
            center=center,
            n_shots=n,
            actual_flips=actual[:n],
            predicted_pymatching=pm_pred[:n],
            predicted_correlated=cm_pred[:n],
            predicted_belief=bm_pred[:n],
            per=per,
            detection_events=detection_events,
            dem_path=dem_path,
        )

    def load_all(
        self, distances: Optional[List[int]] = None
    ) -> Dict[int, List[SycamoreShot]]:
        """
        Carica tutti gli esperimenti Sycamore disponibili.

        Parameters
        ----------
        distances : list of int, optional
            Se specificato, filtra per distanza. Default: [3, 5].

        Returns
        -------
        dict: distance → list of SycamoreShot
            Tutti gli esperimenti raggruppati per distanza, ordinati per rounds.
        """
        if distances is None:
            distances = [3, 5]

        results: Dict[int, List[SycamoreShot]] = {d: [] for d in distances}
        exp_dirs = self._find_experiment_dirs()

        if not exp_dirs:
            print(f"  [Sycamore] Nessuna directory trovata in {self.base_path}")
            return results

        for exp_dir in exp_dirs:
            shot = self._load_single_experiment(exp_dir)
            if shot is not None and shot.distance in distances:
                results[shot.distance].append(shot)
                print(f"  [Sycamore] d={shot.distance} r={shot.rounds} "
                      f"center={shot.center} PER={shot.per:.4f} "
                      f"det_evts={'OK' if shot.detection_events is not None else 'N/A'}")

        for d in distances:
            results[d].sort(key=lambda s: s.rounds)

        return results

    def compute_ler_per_decoder(
        self,
        shots: List[SycamoreShot],
    ) -> Dict[str, Tuple[List[float], List[float]]]:
        """
        Calcola LER/round e PER per ogni decoder Google su una lista di shot.

        Returns
        -------
        dict: decoder_name → (list of PER, list of LER/round)
        """
        decoders = {
            "pymatching":          "predicted_pymatching",
            "correlated_matching": "predicted_correlated",
            "belief_matching":     "predicted_belief",
        }

        results: Dict[str, Tuple[List[float], List[float]]] = {
            name: ([], []) for name in decoders
        }

        for shot in shots:
            for dec_name, attr in decoders.items():
                predicted = getattr(shot, attr)
                actual    = shot.actual_flips
                n         = min(len(predicted), len(actual))
                if n == 0:
                    continue
                ler       = np.sum(predicted[:n] != actual[:n]) / n
                ler_round = ler / shot.rounds if shot.rounds > 0 else ler
                results[dec_name][0].append(shot.per)
                results[dec_name][1].append(ler_round)

        return results

    def compute_ler_vs_rounds(
        self,
        shots: List[SycamoreShot],
    ) -> Dict[str, Tuple[List[int], List[float]]]:
        """Calcola LER vs numero di rounds per ogni decoder Google."""
        from collections import defaultdict

        decoders = {
            "pymatching":          "predicted_pymatching",
            "correlated_matching": "predicted_correlated",
            "belief_matching":     "predicted_belief",
        }

        err_count: Dict[str, Dict[int, List[int]]] = {
            dec: defaultdict(lambda: [0, 0]) for dec in decoders
        }

        for shot in shots:
            for dec_name, attr in decoders.items():
                predicted = getattr(shot, attr)
                actual    = shot.actual_flips
                n         = min(len(predicted), len(actual))
                if n == 0:
                    continue
                errors = int(np.sum(predicted[:n] != actual[:n]))
                err_count[dec_name][shot.rounds][0] += errors
                err_count[dec_name][shot.rounds][1] += n

        results: Dict[str, Tuple[List[int], List[float]]] = {}
        for dec_name in decoders:
            rounds_sorted = sorted(err_count[dec_name].keys())
            ler_list = [
                err_count[dec_name][r][0] / err_count[dec_name][r][1]
                if err_count[dec_name][r][1] > 0 else 0.0
                for r in rounds_sorted
            ]
            results[dec_name] = (rounds_sorted, ler_list)

        return results

    def compute_epsilon_L(
        self,
        shots: List[SycamoreShot],
        r_min: int = 3,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """
        Stima εL (logical error rate per ciclo) con fit esponenziale.

        Modello: F_L(r) = (1 - 2 εL)^(r - r0)
        Fit lineare su log(F_L) vs r per r >= r_min.

        Returns
        -------
        dict: decoder_name → (epsilon_L_percent, uncertainty_percent)
        """
        from collections import defaultdict

        decoders = {
            "pymatching":          "predicted_pymatching",
            "correlated_matching": "predicted_correlated",
            "belief_matching":     "predicted_belief",
        }

        err_count: Dict[str, Dict[int, List[int]]] = {
            dec: defaultdict(lambda: [0, 0]) for dec in decoders
        }
        for shot in shots:
            for dec_name, attr in decoders.items():
                predicted = getattr(shot, attr)
                actual    = shot.actual_flips
                n         = min(len(predicted), len(actual))
                if n == 0:
                    continue
                errors = int(np.sum(predicted[:n] != actual[:n]))
                err_count[dec_name][shot.rounds][0] += errors
                err_count[dec_name][shot.rounds][1] += n

        results: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        for dec_name in decoders:
            ec         = err_count[dec_name]
            rounds_all = sorted(r for r in ec if r >= r_min and ec[r][1] > 0)

            r_arr, fl_arr = [], []
            for r in rounds_all:
                total_err, total_n = ec[r]
                pl = total_err / total_n
                fl = 1.0 - 2.0 * pl
                if fl > 0:
                    r_arr.append(r)
                    fl_arr.append(fl)

            if len(r_arr) < 3:
                results[dec_name] = (None, None)
                continue

            r_np = np.array(r_arr, dtype=np.float64)
            lfl  = np.log(np.array(fl_arr, dtype=np.float64))

            try:
                coeffs, cov = np.polyfit(r_np, lfl, 1, cov=True)
            except (np.linalg.LinAlgError, ValueError):
                results[dec_name] = (None, None)
                continue

            slope     = coeffs[0]
            slope_std = float(np.sqrt(max(cov[0, 0], 0.0)))
            eps_l     = (1.0 - np.exp(slope)) / 2.0
            eps_l_std = abs(-np.exp(slope) / 2.0) * slope_std

            results[dec_name] = (float(eps_l * 100), float(eps_l_std * 100))

        return results

    def compute_google_ler_r1(
        self,
        shots_by_d: Dict[int, List[SycamoreShot]],
    ) -> Dict[str, Dict[int, float]]:
        """
        Calcola LER a r=1 per i 3 decoder Google su ciascuna distanza.

        Aggrega tutti i centri con rounds==1 per distanza d.

        Returns
        -------
        dict: decoder_name → {d: LER}
        """
        decoders = {
            "pymatching":          "predicted_pymatching",
            "correlated_matching": "predicted_correlated",
            "belief_matching":     "predicted_belief",
        }
        results: Dict[str, Dict[int, float]] = {dec: {} for dec in decoders}

        for d, shots in shots_by_d.items():
            r1 = [s for s in shots if s.rounds == 1]
            if not r1:
                continue
            for dec_name, attr in decoders.items():
                total_err = sum(
                    int(np.sum(getattr(s, attr)[:s.n_shots] != s.actual_flips[:s.n_shots]))
                    for s in r1
                )
                total_n = sum(s.n_shots for s in r1)
                if total_n > 0:
                    results[dec_name][d] = total_err / total_n

        return results
