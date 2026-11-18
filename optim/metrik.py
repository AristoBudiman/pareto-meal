"""Metrik evaluasi kualitas Pareto front: Hypervolume, Spacing, dan IGD."""

import numpy as np

TITIK_ACUAN = np.array([300.0, 15.0])  # titik terburuk untuk Hypervolume


def hypervolume(F: np.ndarray, titik_acuan: np.ndarray = TITIK_ACUAN) -> float:
    """Menghitung Hypervolume ternormalisasi [0, 1] terhadap titik acuan."""
    if len(F) == 0:
        return 0.0
    F = F[(F[:, 0] < titik_acuan[0]) & (F[:, 1] < titik_acuan[1])]
    if len(F) == 0:
        return 0.0
    F = F[np.lexsort((F[:, 1], F[:, 0]))]
    luas, terbaik = 0.0, titik_acuan[1]
    for i, p in enumerate(F):
        x_berikut = F[i + 1, 0] if i + 1 < len(F) else titik_acuan[0]
        terbaik = min(terbaik, p[1])
        luas += (x_berikut - p[0]) * (titik_acuan[1] - terbaik)
    return float(luas / (titik_acuan[0] * titik_acuan[1]))  # dinormalisasi 0-1


def spacing(F: np.ndarray, titik_acuan: np.ndarray = TITIK_ACUAN) -> float:
    """Menghitung sebaran (Spacing) solusi pada ruang objektif ternormalisasi."""
    if len(F) < 2:
        return float(np.nan)
    Fn = F / titik_acuan
    d = [
        min(np.abs(Fn[i] - Fn[j]).sum() for j in range(len(Fn)) if j != i)
        for i in range(len(Fn))
    ]
    return float(np.sqrt(np.sum((np.mean(d) - np.array(d)) ** 2) / (len(d) - 1)))


def igd(
    F: np.ndarray, F_sejati: np.ndarray, titik_acuan: np.ndarray = TITIK_ACUAN
) -> float:
    """Menghitung Inverted Generational Distance (IGD) terhadap front sejati."""
    if len(F) == 0 or len(F_sejati) == 0:
        return float(np.nan)
    A, B = F / titik_acuan, F_sejati / titik_acuan
    return float(np.mean([np.min(np.linalg.norm(A - b, axis=1)) for b in B]))
