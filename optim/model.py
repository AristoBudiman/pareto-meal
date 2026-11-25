"""Model optimasi multi-objektif: evaluasi, dominasi Pareto, dan brute force."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from optim.data import DataMenu
from optim.metrik import TITIK_ACUAN


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
        f"{data.lauk['Item'][int(x[1])]} + "
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
    kalori, protein, serat, natrium, gula, lemak = total.T

    f1 = np.abs(kalori - konfig.target_kalori)

    kurang_protein = np.maximum(konfig.target_protein - protein, 0)
    kurang_serat = np.maximum(konfig.target_serat - serat, 0)
    f2 = kurang_protein + kurang_serat

    # Kelebihan dibagi batasnya supaya ketiganya setara
    lebih_natrium = np.maximum(natrium - konfig.maks_natrium, 0) / konfig.maks_natrium
    lebih_gula = np.maximum(gula - konfig.maks_gula, 0) / konfig.maks_gula
    lebih_lemak = np.maximum(lemak - konfig.maks_lemak_jenuh, 0) / konfig.maks_lemak_jenuh
    pelanggaran = lebih_natrium + lebih_gula + lebih_lemak

    # Pengali dibuat besar supaya solusi yang melanggar selalu kalah
    F = np.column_stack([f1 + 1000 * pelanggaran, f2 + 100 * pelanggaran])
    lolos = pelanggaran == 0
    return F, lolos


def non_dominasi(F: np.ndarray) -> list[int]:
    """Mengembalikan indeks solusi yang tidak didominasi solusi lain (minimisasi)."""
    hasil = []
    for i in range(len(F)):
        kalah = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if not kalah.any():
            hasil.append(i)
    return hasil


def ambil_pareto(
    X: np.ndarray, data: DataMenu, konfig: Konfigurasi
) -> tuple[np.ndarray, np.ndarray]:
    """Mengambil solusi unik yang lolos batasan dan non-dominasi (menu kembar tetap disimpan)."""
    if len(X) == 0:
        return np.empty((0, 3), dtype=int), np.empty((0, 2), dtype=float)
    X = np.unique(X, axis=0)
    F, lolos = evaluasi(X, data, konfig)
    X, F = X[lolos], F[lolos]
    if len(X) == 0:
        return X, F
    idx = non_dominasi(F)
    return X[idx], F[idx]


def brute_force(
    data: DataMenu, konfig: Konfigurasi
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mencari Pareto front sejati dari semua kombinasi menggunakan non_dominasi."""
    semua = np.array(
        list(product(range(len(data.utama)), range(len(data.lauk)), range(len(data.minum))))
    )
    F_semua, lolos = evaluasi(semua, data, konfig)
    jumlah_lolos = int(lolos.sum())
    if jumlah_lolos == 0:
        return np.empty((0, 3), dtype=int), np.empty((0, 2), dtype=float), 0

    X_lolos = semua[lolos]
    F_lolos = F_semua[lolos]

    idx = non_dominasi(F_lolos)
    X_sejati = X_lolos[idx]
    F_sejati = F_lolos[idx]

    # Urutkan dari f1 terkecil supaya enak dibaca
    urutan = np.argsort(F_sejati[:, 0])
    X_sejati, F_sejati = X_sejati[urutan], F_sejati[urutan]
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
    return pd.DataFrame(baris, columns=kolom).sort_values(["f1", "f2", "Nama Menu"]).reset_index(drop=True)


def pilih_rekomendasi(
    X: np.ndarray,
    F: np.ndarray,
    data: DataMenu,
    titik_acuan: np.ndarray = TITIK_ACUAN,
) -> list[dict[str, object]]:
    """Memilih 3 titik rekomendasi dari front Pareto dan menyertakan semua menu di tiap titik."""
    if len(X) == 0 or len(F) == 0:
        return []

    F_norm = F / titik_acuan
    pilihan = {
        "Paling dekat target kalori": F[np.argmin(F[:, 0])],
        "Paling lengkap protein dan serat": F[np.argmin(F[:, 1])],
        "Kompromi": F[np.argmin(np.linalg.norm(F_norm, axis=1))],
    }

    hasil = []
    for label, titik in pilihan.items():
        sama = np.all(F == titik, axis=1)
        menu_list = []
        for x in X[sama]:
            total = hitung_gizi(np.array([x]), data)[0]
            menu_list.append(
                {
                    "nama_menu": nama_menu(x, data),
                    "kalori": float(total[0]),
                    "protein": float(total[1]),
                    "serat": float(total[2]),
                    "natrium": float(total[3]),
                    "gula": float(total[4]),
                    "lemak_jenuh": float(total[5]),
                }
            )
        hasil.append(
            {
                "label": label,
                "f1": float(titik[0]),
                "f2": float(titik[1]),
                "menus": menu_list,
            }
        )
    return hasil
