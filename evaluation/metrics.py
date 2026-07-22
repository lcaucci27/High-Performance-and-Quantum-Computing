"""
Metriche di valutazione per i decoder QEC.
Calcola Logical Error Rate (LER) e LER/round, e trova pseudo-threshold e decoder threshold.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def compute_ler(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Calcola il Logical Error Rate (LER).

    LER = (numero di predizioni sbagliate) / (totale shot)

    Parameters
    ----------
    predictions : np.ndarray, shape (N,)
        Predizioni del flip logico dal decoder.
    ground_truth : np.ndarray, shape (N,)
        Ground truth del flip logico.

    Returns
    -------
    float
        LER in [0, 1].
    """
    n = len(predictions)
    if n == 0:
        return 0.0
    errors = np.sum(predictions != ground_truth)
    return float(errors) / n


def compute_ler_per_round(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    rounds: int,
) -> float:
    """
    Calcola il Logical Error Rate per round.

    LER/round = LER / rounds

    Parameters
    ----------
    predictions : np.ndarray, shape (N,)
        Predizioni del flip logico.
    ground_truth : np.ndarray, shape (N,)
        Ground truth.
    rounds : int
        Numero di rounds QEC.

    Returns
    -------
    float
        LER/round.
    """
    ler = compute_ler(predictions, ground_truth)
    return ler / rounds if rounds > 0 else ler


def find_pseudo_threshold(
    p_values: List[float],
    ler_values: List[float],
) -> Optional[float]:
    """
    Trova il pseudo-threshold: il valore di p in cui LER = PER (curva y = x).

    Il pseudo-threshold è il punto in cui il decoder inizia a peggiorare
    rispetto a nessuna correzione. Al di sopra di esso, il QEC fa danni.

    Parameters
    ----------
    p_values : list of float
        Valori di physical error rate (asse x).
    ler_values : list of float
        Corrispondenti valori di LER (asse y).

    Returns
    -------
    float or None
        Il pseudo-threshold p_th, o None se non trovato.
    """
    if len(p_values) < 2:
        return None

    p_arr   = np.array(p_values)
    ler_arr = np.array(ler_values)

    # Differenza tra LER e la diagonale y=x
    diff = ler_arr - p_arr

    # Cerca un cambio di segno (zero crossing)
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0:
            # Interpolazione lineare per trovare il crossing
            try:
                f_interp = interp1d(
                    [p_arr[i], p_arr[i + 1]],
                    [diff[i], diff[i + 1]],
                    kind="linear",
                )
                p_th = brentq(f_interp, p_arr[i], p_arr[i + 1])
                return float(p_th)
            except ValueError:
                continue

    return None


def find_decoder_threshold(
    p_values_1: List[float],
    ler_values_1: List[float],
    p_values_2: List[float],
    ler_values_2: List[float],
) -> Optional[float]:
    """
    Trova il decoder threshold: il p in cui due curve LER si intersecano.

    Il decoder threshold è il punto in cui, aumentando la distanza del codice,
    si inizia a ottenere un vantaggio (LER decresce con d crescente).

    Parameters
    ----------
    p_values_1, ler_values_1 : liste per la prima curva (distanza minore)
    p_values_2, ler_values_2 : liste per la seconda curva (distanza maggiore)

    Returns
    -------
    float or None
        Il decoder threshold p_dec, o None se non trovato.
    """
    # Trova il range di p comune tra le due curve
    p_min = max(min(p_values_1), min(p_values_2))
    p_max = min(max(p_values_1), max(p_values_2))

    if p_min >= p_max:
        return None

    # Griglia comune di p
    p_common = np.linspace(p_min, p_max, 200)

    try:
        f1 = interp1d(p_values_1, ler_values_1, kind="linear", bounds_error=False,
                      fill_value="extrapolate")
        f2 = interp1d(p_values_2, ler_values_2, kind="linear", bounds_error=False,
                      fill_value="extrapolate")

        ler1 = f1(p_common)
        ler2 = f2(p_common)
        diff = ler1 - ler2

        # Cerca zero crossing
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] <= 0:
                try:
                    f_diff = interp1d(p_common[i:i+2], diff[i:i+2], kind="linear")
                    p_dec  = brentq(f_diff, p_common[i], p_common[i + 1])
                    return float(p_dec)
                except ValueError:
                    continue

    except Exception:
        pass

    return None


def find_pseudo_threshold_log(
    p_values: List[float],
    ler_values: List[float],
) -> Optional[float]:
    """
    Versione log-log del pseudo-threshold (più stabile numericamente).

    Lavora in spazio log-log e interpola per trovare il crossing con y=x.

    Parameters
    ----------
    p_values : list of float
        Physical error rate.
    ler_values : list of float
        LER corrispondenti.

    Returns
    -------
    float or None
        Il pseudo-threshold in scala originale.
    """
    # Filtra valori zero per il log
    valid = [(p, l) for p, l in zip(p_values, ler_values) if p > 0 and l > 0]
    if len(valid) < 2:
        return find_pseudo_threshold(p_values, ler_values)

    log_p   = np.log10([v[0] for v in valid])
    log_ler = np.log10([v[1] for v in valid])

    # In log-log, y=x diventa log(LER) = log(p) → differenza = 0
    diff = log_ler - log_p

    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0:
            try:
                f_interp = interp1d(log_p[i:i+2], diff[i:i+2], kind="linear")
                log_p_th = brentq(f_interp, log_p[i], log_p[i + 1])
                return float(10 ** log_p_th)
            except ValueError:
                continue

    return None
