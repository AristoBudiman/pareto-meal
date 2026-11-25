"""Implementasi MOGA (Fonseca & Fleming 1993), MOPSO (Coello dkk. 2004), dan eksperimen."""

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
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-Objective Genetic Algorithm mengikuti Fonseca & Fleming (1993)."""
    batas_atas = data.batas_atas
    # Sigma share untuk 2 objektif yang sudah dinormalisasi
    sigma = 2 / (N - 1)

    X = np.column_stack([rng.integers(0, b + 1, N) for b in batas_atas])
    arsip = X.copy()

    for g in range(generasi):
        F, _ = evaluasi(X, data, konfig)

        # Rank = 1 + jumlah solusi yang mendominasi
        rank = np.array(
            [1 + np.sum(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F]
        )

        # Fitness diinterpolasi linear dari N (rank terbaik) sampai 1 (rank terburuk)
        urutan = np.argsort(rank, kind="stable")
        fitness = np.zeros(N)
        fitness[urutan] = np.linspace(N, 1, N)

        for r in np.unique(rank):
            anggota = np.where(rank == r)[0]

            # Solusi dengan rank yang sama diberi fitness yang sama
            fitness[anggota] = fitness[anggota].mean()

            # Sharing hanya di antara solusi dengan rank yang sama
            Fr = F[anggota]
            Fr = (Fr - Fr.min(axis=0)) / (np.ptp(Fr, axis=0) + 1e-9)
            jarak = np.abs(Fr[:, None] - Fr[None]).max(axis=2)
            niche = np.where(jarak < sigma, 1 - jarak / sigma, 0).sum(axis=1)

            total = fitness[anggota].sum()
            fitness[anggota] = fitness[anggota] / niche
            fitness[anggota] *= total / fitness[anggota].sum()

        # Stochastic universal sampling
        kumulatif = np.cumsum(fitness)
        langkah = kumulatif[-1] / N
        titik = rng.uniform(0, langkah) + langkah * np.arange(N)
        terpilih = np.minimum(np.searchsorted(kumulatif, titik), N - 1)
        rng.shuffle(terpilih)

        anak = []
        for i in range(0, N, 2):
            p1, p2 = X[terpilih[i]], X[terpilih[i + 1]]

            # Crossover uniform, tiap gen diambil dari salah satu induk
            ambil_p1 = rng.random(3) < 0.5
            for child in (np.where(ambil_p1, p1, p2), np.where(ambil_p1, p2, p1)):
                for j in range(3):
                    if rng.random() < p_mutasi:
                        child[j] = rng.integers(0, batas_atas[j] + 1)
                anak.append(child)

        X = np.array(anak)
        arsip = np.vstack([arsip, X])

        # Arsip dirapikan tiap 20 generasi supaya tidak terlalu besar
        if g % 20 == 0:
            X_arsip, _ = ambil_pareto(arsip, data, konfig)
            if len(X_arsip) > 0:
                arsip = X_arsip

    return ambil_pareto(arsip, data, konfig)


def kotak_grid(F: np.ndarray, n_divisi: int) -> np.ndarray:
    """Nomor hypercube tiap solusi di ruang objektif."""
    bawah, atas = F.min(axis=0), F.max(axis=0)
    lebar = (atas - bawah) / n_divisi + 1e-9
    posisi = np.minimum(((F - bawah) / lebar).astype(int), n_divisi - 1)
    return posisi[:, 0] * n_divisi + posisi[:, 1]


def mopso(
    rng: np.random.Generator,
    data: DataMenu,
    konfig: Konfigurasi,
    N: int = 50,
    iterasi: int = 400,
    w: float = 0.4,
    n_divisi: int = 30,
    maks_arsip: int = 100,
    laju_mutasi: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-Objective Particle Swarm Optimization mengikuti Coello dkk. (2004)."""
    batas_atas = data.batas_atas
    bulat = lambda P: np.clip(np.rint(P), 0, batas_atas).astype(int)

    X = np.column_stack([rng.uniform(0, b, N) for b in batas_atas])
    V = np.zeros_like(X)
    F, _ = evaluasi(bulat(X), data, konfig)

    pbest, F_pbest = X.copy(), F.copy()

    idx = non_dominasi(F)
    arsip, F_arsip = bulat(X)[idx], F[idx]

    for t in range(iterasi):
        # Pilih leader: hypercube yang lebih sepi punya peluang lebih besar
        kotak = kotak_grid(F_arsip, n_divisi)
        isi_kotak, jumlah = np.unique(kotak, return_counts=True)
        peluang = (10 / jumlah) / (10 / jumlah).sum()
        leader = np.empty_like(X)
        for i in range(N):
            k = rng.choice(isi_kotak, p=peluang)
            leader[i] = arsip[rng.choice(np.where(kotak == k)[0])]

        r1, r2 = rng.random((N, 3)), rng.random((N, 3))
        V = w * V + r1 * (pbest - X) + r2 * (leader - X)
        X = X + V

        # Kalau keluar batas, posisi ditahan di batas dan arah kecepatan dibalik
        keluar = (X < 0) | (X > batas_atas)
        X = np.clip(X, 0, batas_atas)
        V[keluar] *= -1

        # Mutasi, peluangnya mengecil sampai 0 di akhir iterasi
        pm = (1 - t / iterasi) ** (5 / laju_mutasi)
        for i in range(N):
            if rng.random() < pm:
                j = rng.integers(3)
                rentang = batas_atas[j] * pm
                X[i, j] = np.clip(
                    X[i, j] + rng.uniform(-rentang, rentang), 0, batas_atas[j]
                )

        F, _ = evaluasi(bulat(X), data, konfig)

        # Update arsip, hanya simpan yang non-dominasi
        arsip = np.vstack([arsip, bulat(X)])
        F_arsip = np.vstack([F_arsip, F])
        arsip, unik = np.unique(arsip, axis=0, return_index=True)
        F_arsip = F_arsip[unik]
        idx = non_dominasi(F_arsip)
        arsip, F_arsip = arsip[idx], F_arsip[idx]

        # Kalau arsip penuh, buang dari hypercube yang paling ramai
        while len(arsip) > maks_arsip:
            kotak = kotak_grid(F_arsip, n_divisi)
            isi_kotak, jumlah = np.unique(kotak, return_counts=True)
            terpadat = isi_kotak[np.argmax(jumlah)]
            buang = rng.choice(np.where(kotak == terpadat)[0])
            arsip, F_arsip = np.delete(arsip, buang, axis=0), np.delete(
                F_arsip, buang, axis=0
            )

        # Update pbest
        for i in range(N):
            baru_menang = np.all(F[i] <= F_pbest[i]) and np.any(F[i] < F_pbest[i])
            lama_menang = np.all(F_pbest[i] <= F[i]) and np.any(F_pbest[i] < F[i])
            if baru_menang or (not lama_menang and rng.random() < 0.5):
                pbest[i], F_pbest[i] = X[i], F[i]

    return ambil_pareto(arsip, data, konfig)


def _ke_json_nilai(nilai: float) -> float | None:
    """Mengubah NaN menjadi None agar valid untuk serialisasi JSON."""
    return None if np.isnan(nilai) else float(nilai)


def _hitung_titik_sejati_ditemukan(F: np.ndarray, F_sejati_unik: np.ndarray) -> int:
    """Menghitung berapa banyak titik unik front sejati yang ditemukan dalam satu run."""
    if len(F) == 0 or len(F_sejati_unik) == 0:
        return 0
    F_unik = np.unique(F, axis=0)
    return int(
        sum(
            1
            for titik in F_sejati_unik
            if np.any(np.all(F_unik == titik, axis=1))
        )
    )


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
    F_sejati_unik = (
        np.unique(F_sejati, axis=0)
        if len(F_sejati) > 0
        else np.empty((0, 2), dtype=float)
    )
    jumlah_titik_sejati = int(len(F_sejati_unik))

    daftar_algo = [("MOGA", moga), ("MOPSO", mopso)]
    total_langkah = len(daftar_algo) * jumlah_run
    langkah = 0

    hasil_algo = {}
    metrik_mentah = {"HV": {}, "IGD": {}}
    for nama, algoritma in daftar_algo:
        catatan = {
            "HV": [],
            "Spacing": [],
            "IGD": [],
            "Jumlah titik": [],
            "Waktu (detik)": [],
        }
        front = []
        titik_sejati_per_run = []

        for seed in range(jumlah_run):
            # Seed sama untuk kedua algoritma supaya adil dan bisa diulang
            rng = np.random.default_rng(seed)

            mulai = time.time()
            X, F = algoritma(rng, data, konfig)
            catatan["Waktu (detik)"].append(float(time.time() - mulai))

            catatan["HV"].append(float(hypervolume(F)))
            catatan["Spacing"].append(float(spacing(F)))
            catatan["IGD"].append(float(igd(F, F_sejati)))
            catatan["Jumlah titik"].append(
                int(len(np.unique(F, axis=0))) if len(F) > 0 else 0
            )
            titik_sejati_per_run.append(
                _hitung_titik_sejati_ditemukan(F, F_sejati_unik)
            )
            front.append((X, F))

            langkah += 1
            if progress_callback is not None:
                progress_callback(
                    langkah, total_langkah, f"{nama} run {seed + 1}/{jumlah_run}"
                )

        metrik_mentah["HV"][nama] = catatan["HV"]
        metrik_mentah["IGD"][nama] = catatan["IGD"]

        idx_terbaik = int(np.argmax(catatan["HV"]))
        X_terbaik, F_terbaik = front[idx_terbaik]

        metrik_keys = ["HV", "Spacing", "IGD", "Jumlah titik", "Waktu (detik)"]
        ringkasan = {}
        metrik_json = {}
        for m in metrik_keys:
            arr = np.array(catatan[m], dtype=float)
            ringkasan[m] = {
                "mean": _ke_json_nilai(float(np.nanmean(arr))),
                "std": _ke_json_nilai(float(np.nanstd(arr))),
            }
            metrik_json[m] = [_ke_json_nilai(float(v)) for v in arr]

        distribusi_titik_sejati = {
            str(k): int(sum(1 for v in titik_sejati_per_run if v == k))
            for k in range(jumlah_titik_sejati + 1)
        }

        hasil_algo[nama] = {
            "metrik_per_run": metrik_json,
            "ringkasan": ringkasan,
            "titik_sejati_per_run": titik_sejati_per_run,
            "distribusi_titik_sejati": distribusi_titik_sejati,
            "run_terbaik": {
                "run_index": idx_terbaik,
                "seed": idx_terbaik,
                "hv": float(catatan["HV"][idx_terbaik]),
                "X": X_terbaik.astype(int).tolist(),
                "F": F_terbaik.astype(float).tolist(),
                "menu": [nama_menu(x, data) for x in X_terbaik],
            },
        }

    wilcoxon = {}
    for m in ["HV", "IGD"]:
        uji = ranksums(metrik_mentah[m]["MOGA"], metrik_mentah[m]["MOPSO"])
        p_val = float(uji.pvalue)
        wilcoxon[m] = {
            "statistic": float(uji.statistic),
            "p_value": p_val,
            "alpha": 0.05,
            "signifikan": bool(p_val < 0.05),
        }

    return {
        "konfigurasi": asdict(konfig),
        "jumlah_run": jumlah_run,
        "total_kombinasi": int(data.total_kombinasi),
        "jumlah_kombinasi_lolos": int(jumlah_lolos),
        "front_sejati": {
            "X": X_sejati.astype(int).tolist(),
            "F": F_sejati.astype(float).tolist(),
            "jumlah_titik": jumlah_titik_sejati,
            "jumlah_menu": int(len(X_sejati)),
            "hv": float(hypervolume(F_sejati)),
            "spacing": _ke_json_nilai(float(spacing(F_sejati))),
            "menu": [nama_menu(x, data) for x in X_sejati],
        },
        "wilcoxon": wilcoxon,
        "algoritma": hasil_algo,
    }
