"""Metrik evaluasi kualitas Pareto front: Hypervolume, Spacing, dan IGD."""

import numpy as np

# Titik terburuk, dipakai untuk HV dan normalisasi
TITIK_ACUAN = np.array([300.0, 15.0])


def hypervolume(F: np.ndarray, titik_acuan: np.ndarray = TITIK_ACUAN) -> float:
    """Menghitung Hypervolume ternormalisasi [0, 1] terhadap titik acuan dari titik unik."""
    if len(F) == 0:
        return 0.0
    F = np.unique(F, axis=0)
    F = F[(F[:, 0] < titik_acuan[0]) & (F[:, 1] < titik_acuan[1])]
    if len(F) == 0:
        return 0.0

    F = F[np.argsort(F[:, 0])]
    luas = 0.0
    f2_terkecil = titik_acuan[1]

    # Luas dihitung per pita vertikal dari kiri ke kanan
    for i in range(len(F)):
        f1_kanan = F[i + 1, 0] if i + 1 < len(F) else titik_acuan[0]
        f2_terkecil = min(f2_terkecil, F[i, 1])
        luas += (f1_kanan - F[i, 0]) * (titik_acuan[1] - f2_terkecil)

    return float(luas / (titik_acuan[0] * titik_acuan[1]))


def spacing(F: np.ndarray, titik_acuan: np.ndarray = TITIK_ACUAN) -> float:
    """Menghitung sebaran (Spacing) titik unik pada ruang objektif ternormalisasi."""
    if len(F) == 0:
        return float(np.nan)
    F = np.unique(F, axis=0) / titik_acuan
    if len(F) < 2:
        return float(np.nan)

    jarak_terdekat = []
    for i in range(len(F)):
        jarak = np.abs(F - F[i]).sum(axis=1)
        jarak[i] = np.inf  # jarak ke diri sendiri tidak dihitung
        jarak_terdekat.append(jarak.min())

    return float(np.std(jarak_terdekat, ddof=1))


def igd(
    F: np.ndarray, F_sejati: np.ndarray, titik_acuan: np.ndarray = TITIK_ACUAN
) -> float:
    """Menghitung Inverted Generational Distance (IGD) dari titik unik terhadap front sejati."""
    if len(F) == 0 or len(F_sejati) == 0:
        return float(np.nan)

    hasil = np.unique(F, axis=0) / titik_acuan
    sejati = np.unique(F_sejati, axis=0) / titik_acuan

    jarak = [np.linalg.norm(hasil - titik, axis=1).min() for titik in sejati]
    return float(np.mean(jarak))
