"""Implementasi algoritma metaheuristik MOGA dan MOPSO serta runner eksperimen."""

import time
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
from scipy.stats import ranksums

from optim.data import DataMenu
from optim.metrik import hypervolume, igd, spacing
from optim.model import (
    Konfigurasi,
    ambil_pareto,
    brute_force,
    evaluasi,
    nama_menu,
    non_dominasi,
)


def moga(
    rng: np.random.Generator,
    data: DataMenu,
    konfig: Konfigurasi,
    N: int = 50,
    generasi: int = 400,
    p_mutasi: float = 0.2,
    sigma: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-Objective Genetic Algorithm (Fonseca-Fleming + fitness sharing)."""
    batas_atas = data.batas_atas
    X = np.column_stack([rng.integers(0, b + 1, N) for b in batas_atas])
    arsip = X.copy()
    for g in range(generasi):
        F, _ = evaluasi(X, data, konfig)
        # Ranking Fonseca-Fleming: 1 + jumlah solusi yang mendominasi
        rank = 1 + np.array([np.sum(np.all(F <= f, 1) & np.any(F < f, 1)) for f in F])
        # Fitness sharing agar solusi tersebar
        Fn = (F - F.min(0)) / (np.ptp(F, 0) + 1e-9)
        jarak = np.linalg.norm(Fn[:, None] - Fn[None], axis=2)
        share = np.where(jarak < sigma, 1 - jarak / sigma, 0).sum(1)
        fitness = (1 / rank) / share

        def turnamen() -> np.ndarray:
            a, b = rng.integers(0, N, 2)
            return X[a] if fitness[a] > fitness[b] else X[b]

        anak = []
        for _ in range(N):
            p1, p2 = turnamen(), turnamen()
            child = np.where(rng.random(3) < 0.5, p1, p2)  # crossover uniform
            for j in range(3):  # mutasi
                if rng.random() < p_mutasi:
                    child[j] = rng.integers(0, batas_atas[j] + 1)
            anak.append(child)
        X = np.array(anak)
        arsip = np.vstack([arsip, X])
        if g % 20 == 0:
            Xa, _ = ambil_pareto(arsip, data, konfig)
            if len(Xa):
                arsip = Xa
    return ambil_pareto(arsip, data, konfig)


def mopso(
    rng: np.random.Generator,
    data: DataMenu,
    konfig: Konfigurasi,
    N: int = 50,
    iterasi: int = 400,
    w: float = 0.5,
    c1: float = 1.5,
    c2: float = 1.5,
    p_acak: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-Objective Particle Swarm Optimization dengan arsip eksternal."""
    batas_atas = data.batas_atas
    bulat = lambda P: np.clip(np.rint(P), 0, batas_atas).astype(int)
    X = np.column_stack([rng.uniform(0, b, N) for b in batas_atas])
    V = np.zeros_like(X)
    F, _ = evaluasi(bulat(X), data, konfig)
    pbest, F_pbest = X.copy(), F.copy()
    arsip, F_arsip = bulat(X), F.copy()
    k = non_dominasi(F_arsip)
    arsip, F_arsip = arsip[k], F_arsip[k]

    for _ in range(iterasi):
        # Pilih leader dari arsip
        leader = arsip[rng.integers(0, len(arsip), N)].astype(float)
        r1, r2 = rng.random((N, 3)), rng.random((N, 3))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (leader - X)
        X = np.clip(X + V, 0, batas_atas)
        if rng.random() < p_acak:  # turbulensi agar tidak terjebak
            X[rng.integers(0, N)] = [rng.uniform(0, b) for b in batas_atas]
        F, _ = evaluasi(bulat(X), data, konfig)
        lebih_baik = np.all(F <= F_pbest, 1) & np.any(F < F_pbest, 1)
        pbest[lebih_baik], F_pbest[lebih_baik] = X[lebih_baik], F[lebih_baik]
        # Update arsip eksternal
        arsip, F_arsip = np.vstack([arsip, bulat(X)]), np.vstack([F_arsip, F])
        k = non_dominasi(F_arsip)
        arsip, F_arsip = arsip[k], F_arsip[k]
        arsip, u = np.unique(arsip, axis=0, return_index=True)
        F_arsip = F_arsip[u]
    return ambil_pareto(arsip, data, konfig)


def _ke_json_nilai(nilai: float) -> float | None:
    """Mengubah NaN menjadi None agar valid untuk serialisasi JSON."""
    return None if np.isnan(nilai) else float(nilai)


def jalankan_eksperimen(
    data: DataMenu,
    konfig: Konfigurasi | None = None,
    jumlah_run: int = 30,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Menjalankan eksperimen perbandingan MOGA dan MOPSO sebanyak `jumlah_run` kali."""
    if konfig is None:
        konfig = Konfigurasi()

    X_sejati, F_sejati, jumlah_lolos = brute_force(data, konfig)
    daftar_algo = [("MOGA", moga), ("MOPSO", mopso)]
    total_langkah = len(daftar_algo) * jumlah_run
    langkah = 0

    hasil_algo = {}
    hv_mentah = {}
    for nama, algoritma in daftar_algo:
        catatan = {
            "HV": [],
            "Spacing": [],
            "IGD": [],
            "Jumlah solusi": [],
            "Waktu (s)": [],
            "front": [],
        }
        for seed in range(jumlah_run):
            mulai = time.time()
            X, F = algoritma(np.random.default_rng(seed), data, konfig)
            waktu = time.time() - mulai
            catatan["Waktu (s)"].append(float(waktu))
            catatan["HV"].append(float(hypervolume(F)))
            catatan["Spacing"].append(float(spacing(F)))
            catatan["IGD"].append(float(igd(F, F_sejati)))
            catatan["Jumlah solusi"].append(int(len(F)))
            catatan["front"].append((X, F))

            langkah += 1
            if progress_callback is not None:
                progress_callback(langkah, total_langkah, f"{nama} run {seed + 1}/{jumlah_run}")

        hv_mentah[nama] = catatan["HV"]
        idx_terbaik = int(np.argmax(catatan["HV"]))
        X_terbaik, F_terbaik = catatan["front"][idx_terbaik]

        metrik_keys = ["HV", "Spacing", "IGD", "Jumlah solusi", "Waktu (s)"]
        ringkasan = {}
        metrik_json = {}
        for m in metrik_keys:
            arr = np.array(catatan[m], dtype=float)
            ringkasan[m] = {
                "mean": _ke_json_nilai(float(np.nanmean(arr))),
                "std": _ke_json_nilai(float(np.nanstd(arr))),
            }
            metrik_json[m] = [_ke_json_nilai(float(v)) for v in arr]

        hasil_algo[nama] = {
            "metrik_per_run": metrik_json,
            "ringkasan": ringkasan,
            "run_terbaik": {
                "run_index": idx_terbaik,
                "seed": idx_terbaik,
                "hv": float(catatan["HV"][idx_terbaik]),
                "X": X_terbaik.astype(int).tolist(),
                "F": F_terbaik.astype(float).tolist(),
                "menu": [nama_menu(x, data) for x in X_terbaik],
            },
        }

    uji = ranksums(hv_mentah["MOGA"], hv_mentah["MOPSO"])
    p_value = float(uji.pvalue)

    return {
        "konfigurasi": asdict(konfig),
        "jumlah_run": jumlah_run,
        "total_kombinasi": int(data.total_kombinasi),
        "jumlah_kombinasi_lolos": int(jumlah_lolos),
        "front_sejati": {
            "X": X_sejati.astype(int).tolist(),
            "F": F_sejati.astype(float).tolist(),
            "hv": float(hypervolume(F_sejati)),
            "menu": [nama_menu(x, data) for x in X_sejati],
        },
        "wilcoxon_hv": {
            "statistic": float(uji.statistic),
            "p_value": p_value,
            "alpha": 0.05,
            "signifikan": bool(p_value < 0.05),
        },
        "algoritma": hasil_algo,
    }
