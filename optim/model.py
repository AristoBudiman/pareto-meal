"""Model optimasi multi-objektif: evaluasi, dominasi Pareto, dan brute force."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optim.data import DataMenu


@dataclass(frozen=True)
class Konfigurasi:
    """Target gizi dan batas maksimal per satu kali makan."""

    target_kalori: float = 600.0
    target_protein: float = 25.0
    target_serat: float = 8.0
    maks_natrium: float = 800.0
    maks_gula: float = 17.0
    maks_lemak_jenuh: float = 7.0


def nama_menu(x: np.ndarray | list[int], data: DataMenu) -> str:
    """Membentuk nama kombinasi menu dari vektor x = [utama, lauk, minuman]."""
    return (
        f"{data.utama['Item'][int(x[0])]} + "
        f"{data.nama_lauk[int(x[1])]} + "
        f"{data.minum['Item'][int(x[2])]}"
    )


def hitung_gizi(X: np.ndarray, data: DataMenu) -> np.ndarray:
    """Menghitung total 6 komponen gizi untuk array indeks X berukuran (n, 3)."""
    if len(X) == 0:
        return np.empty((0, 6), dtype=float)
    return data.gizi_utama[X[:, 0]] + data.gizi_lauk[X[:, 1]] + data.gizi_minum[X[:, 2]]


def evaluasi(
    X: np.ndarray, data: DataMenu, konfig: Konfigurasi
) -> tuple[np.ndarray, np.ndarray]:
    """Mengevaluasi objektif (n, 2) dan status lolos batasan untuk X berukuran (n, 3)."""
    if len(X) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=bool)
    total = hitung_gizi(X, data)
    kal, prot, serat, natrium, gula, lemak = total.T
    f1 = np.abs(kal - konfig.target_kalori)
    f2 = np.clip(konfig.target_protein - prot, 0, None) + np.clip(
        konfig.target_serat - serat, 0, None
    )
    # Penalty: besar pelanggaran (dinormalisasi terhadap batasnya)
    pelanggaran = (
        np.clip(natrium - konfig.maks_natrium, 0, None) / konfig.maks_natrium
        + np.clip(gula - konfig.maks_gula, 0, None) / konfig.maks_gula
        + np.clip(lemak - konfig.maks_lemak_jenuh, 0, None) / konfig.maks_lemak_jenuh
    )
    F = np.c_[f1 + 1000 * pelanggaran, f2 + 100 * pelanggaran]
    return F, pelanggaran == 0


def non_dominasi(F: np.ndarray) -> list[int]:
    """Mengembalikan indeks solusi yang tidak didominasi solusi lain (minimisasi)."""
    return [
        i
        for i in range(len(F))
        if not np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
    ]


def ambil_pareto(
    X: np.ndarray, data: DataMenu, konfig: Konfigurasi
) -> tuple[np.ndarray, np.ndarray]:
    """Mengambil solusi unik yang lolos batasan dan non-dominasi."""
    if len(X) == 0:
        return np.empty((0, 3), dtype=int), np.empty((0, 2), dtype=float)
    X = np.unique(X, axis=0)
    F, lolos = evaluasi(X, data, konfig)
    X, F = X[lolos], F[lolos]
    if len(F) == 0:
        return X, F
    idx = non_dominasi(F)
    X, F = X[idx], F[idx]
    F, u = np.unique(F, axis=0, return_index=True)
    return X[u], F


def brute_force(
    data: DataMenu, konfig: Konfigurasi
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mencari Pareto front sejati dan jumlah kombinasi lolos batasan."""
    semua = (
        np.array(
            np.meshgrid(
                *[np.arange(b + 1) for b in data.batas_atas], indexing="ij"
            )
        )
        .reshape(3, -1)
        .T
    )
    F_semua, lolos = evaluasi(semua, data, konfig)
    jumlah_lolos = int(lolos.sum())
    if jumlah_lolos == 0:
        return np.empty((0, 3), dtype=int), np.empty((0, 2), dtype=float), 0

    X_lolos, F_lolos = semua[lolos], F_semua[lolos]
    urut = np.lexsort((F_lolos[:, 1], F_lolos[:, 0]))
    idx_pf, terbaik = [], np.inf
    for i in urut:  # cara cepat mencari front untuk 2 objektif
        if F_lolos[i, 1] < terbaik:
            idx_pf.append(i)
            terbaik = F_lolos[i, 1]
    X_sejati, F_sejati = X_lolos[idx_pf], F_lolos[idx_pf]
    return X_sejati, F_sejati, jumlah_lolos


def tabel_solusi(X: np.ndarray, F: np.ndarray, data: DataMenu) -> pd.DataFrame:
    """Menyusun tabel DataFrame solusi Pareto lengkap dengan rincian gizi."""
    kolom = [
        "Nama Menu",
        "Kalori (kkal)",
        "Protein (g)",
        "Serat (g)",
        "Natrium (mg)",
        "Gula (g)",
        "Lemak Jenuh (g)",
        "f1",
        "f2",
    ]
    if len(X) == 0:
        return pd.DataFrame(columns=kolom)

    total = hitung_gizi(X, data)
    baris = []
    for x, g, f in zip(X, total, F):
        baris.append(
            {
                "Nama Menu": nama_menu(x, data),
                "Kalori (kkal)": float(g[0]),
                "Protein (g)": float(g[1]),
                "Serat (g)": float(g[2]),
                "Natrium (mg)": float(g[3]),
                "Gula (g)": float(g[4]),
                "Lemak Jenuh (g)": float(g[5]),
                "f1": float(f[0]),
                "f2": float(f[1]),
            }
        )
    return pd.DataFrame(baris, columns=kolom).sort_values(["f1", "f2"]).reset_index(drop=True)


def pilih_rekomendasi(
    X: np.ndarray, F: np.ndarray, data: DataMenu
) -> list[dict[str, object]]:
    """Memilih 3 menu rekomendasi dari front Pareto (logika sel terakhir notebook)."""
    if len(X) == 0 or len(F) == 0:
        return []

    pilihan = {
        "Paling dekat target kalori": int(np.argmin(F[:, 0])),
        "Paling lengkap protein & serat": int(np.argmin(F[:, 1])),
        "Kompromi": int(np.argmin(np.linalg.norm(F / F.max(0).clip(1), axis=1))),
    }
    hasil = []
    for label, i in pilihan.items():
        x = X[i]
        total = hitung_gizi(np.array([x]), data)[0]
        hasil.append(
            {
                "label": label,
                "nama_menu": nama_menu(x, data),
                "kalori": float(total[0]),
                "protein": float(total[1]),
                "serat": float(total[2]),
                "natrium": float(total[3]),
                "gula": float(total[4]),
                "lemak_jenuh": float(total[5]),
                "f1": float(F[i, 0]),
                "f2": float(F[i, 1]),
            }
        )
    return hasil
