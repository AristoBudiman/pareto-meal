"""Implementasi MOGA, MOPSO, eksperimen utama, dan eksperimen multi-skenario."""

import time
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
from scipy.stats import ranksums

from optim.data import DataMenu
from optim.metrik import a12, hitung_titik_acuan, hypervolume, igd, spacing
from optim.model import (
    Konfigurasi,
    ambil_pareto,
    brute_force,
    evaluasi,
    nama_menu,
    non_dominasi,
)

SKENARIO_DEFAULT: dict[str, Konfigurasi] = {
    "S1 Baseline": Konfigurasi(
        target_kalori=600.0,
        target_protein=25.0,
        target_serat=8.0,
        maks_natrium=800.0,
        maks_gula=17.0,
        maks_lemak_jenuh=7.0,
    ),
    "S2 Batasan sedang": Konfigurasi(
        target_kalori=600.0,
        target_protein=25.0,
        target_serat=8.0,
        maks_natrium=700.0,
        maks_gula=12.0,
        maks_lemak_jenuh=6.0,
    ),
    "S3 Batasan ketat": Konfigurasi(
        target_kalori=600.0,
        target_protein=25.0,
        target_serat=8.0,
        maks_natrium=600.0,
        maks_gula=10.0,
        maks_lemak_jenuh=5.0,
    ),
    "S4 Target 800 kkal": Konfigurasi(
        target_kalori=800.0,
        target_protein=35.0,
        target_serat=10.0,
        maks_natrium=800.0,
        maks_gula=17.0,
        maks_lemak_jenuh=7.0,
    ),
}


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
    sigma = 2 / (N - 1)

    X = np.column_stack([rng.integers(0, b + 1, N) for b in batas_atas])
    arsip = X.copy()

    for g in range(generasi):
        F, _ = evaluasi(X, data, konfig)

        rank = np.array(
            [1 + np.sum(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F]
        )

        urutan = np.argsort(rank, kind="stable")
        fitness = np.zeros(N)
        fitness[urutan] = np.linspace(N, 1, N)

        for r in np.unique(rank):
            anggota = np.where(rank == r)[0]
            fitness[anggota] = fitness[anggota].mean()

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

            ambil_p1 = rng.random(3) < 0.5
            for child in (np.where(ambil_p1, p1, p2), np.where(ambil_p1, p2, p1)):
                for j in range(3):
                    if rng.random() < p_mutasi:
                        child[j] = rng.integers(0, batas_atas[j] + 1)
                anak.append(child)

        X = np.array(anak)
        arsip = np.vstack([arsip, X])

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

        keluar = (X < 0) | (X > batas_atas)
        X = np.clip(X, 0, batas_atas)
        V[keluar] *= -1

        pm = (1 - t / iterasi) ** (5 / laju_mutasi)
        for i in range(N):
            if rng.random() < pm:
                j = rng.integers(3)
                rentang = batas_atas[j] * pm
                X[i, j] = np.clip(
                    X[i, j] + rng.uniform(-rentang, rentang), 0, batas_atas[j]
                )

        F, _ = evaluasi(bulat(X), data, konfig)

        arsip = np.vstack([arsip, bulat(X)])
        F_arsip = np.vstack([F_arsip, F])
        arsip, unik = np.unique(arsip, axis=0, return_index=True)
        F_arsip = F_arsip[unik]
        idx = non_dominasi(F_arsip)
        arsip, F_arsip = arsip[idx], F_arsip[idx]

        while len(arsip) > maks_arsip:
            kotak = kotak_grid(F_arsip, n_divisi)
            isi_kotak, jumlah = np.unique(kotak, return_counts=True)
            terpadat = isi_kotak[np.argmax(jumlah)]
            buang = rng.choice(np.where(kotak == terpadat)[0])
            arsip, F_arsip = np.delete(arsip, buang, axis=0), np.delete(
                F_arsip, buang, axis=0
            )

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
    """Menghitung berapa banyak titik unik front sejati yang benar-benar ditemukan."""
    if len(F) == 0 or len(F_sejati_unik) == 0:
        return 0
    return int(sum(np.any(np.all(F == t, axis=1)) for t in F_sejati_unik))


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
    titik_acuan = hitung_titik_acuan(F_sejati)
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
            rng = np.random.default_rng(seed)

            mulai = time.time()
            X, F = algoritma(rng, data, konfig)
            catatan["Waktu (detik)"].append(float(time.time() - mulai))

            catatan["HV"].append(float(hypervolume(F, titik_acuan)))
            catatan["Spacing"].append(float(spacing(F, titik_acuan)))
            catatan["IGD"].append(float(igd(F, F_sejati, titik_acuan)))
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
        kw = {"nan_policy": "omit"} if m == "IGD" else {}
        uji = ranksums(metrik_mentah[m]["MOGA"], metrik_mentah[m]["MOPSO"], **kw)
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
        "titik_acuan": [float(titik_acuan[0]), float(titik_acuan[1])],
        "front_sejati": {
            "X": X_sejati.astype(int).tolist(),
            "F": F_sejati.astype(float).tolist(),
            "jumlah_titik": jumlah_titik_sejati,
            "jumlah_menu": int(len(X_sejati)),
            "hv": float(hypervolume(F_sejati, titik_acuan)),
            "spacing": _ke_json_nilai(float(spacing(F_sejati, titik_acuan))),
            "menu": [nama_menu(x, data) for x in X_sejati],
        },
        "wilcoxon": wilcoxon,
        "algoritma": hasil_algo,
    }


def jalankan_multi_skenario(
    data: DataMenu,
    skenario: dict[str, Konfigurasi] | None = None,
    jumlah_run: int = 30,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Menjalankan eksperimen 30 run pada beberapa skenario target dan batasan gizi."""
    daftar_skenario = SKENARIO_DEFAULT if skenario is None else skenario
    total_langkah = len(daftar_skenario) * 2 * jumlah_run
    langkah = 0

    hasil_skenario = {}
    rekap_unggul: dict[str, int] = {}

    for nama_s, konfig in daftar_skenario.items():
        X_sejati, F_sejati, jumlah_lolos = brute_force(data, konfig)
        titik_acuan = hitung_titik_acuan(F_sejati)
        sejati_unik = (
            np.unique(F_sejati, axis=0)
            if len(F_sejati) > 0
            else np.empty((0, 2), dtype=float)
        )
        n_sejati = int(len(sejati_unik))
        hv_sejati = float(hypervolume(F_sejati, titik_acuan))

        catat = {}
        run_terbaik = {}
        for nama, algoritma in [("MOGA", moga), ("MOPSO", mopso)]:
            m: dict[str, list] = {
                "HV": [],
                "IGD": [],
                "Jumlah titik": [],
                "Titik sejati": [],
            }
            fronts = []
            for seed in range(jumlah_run):
                _, F = algoritma(np.random.default_rng(seed), data, konfig)
                m["HV"].append(float(hypervolume(F, titik_acuan)))
                m["IGD"].append(
                    _ke_json_nilai(float(igd(F, F_sejati, titik_acuan)))
                )
                m["Jumlah titik"].append(
                    int(len(np.unique(F, axis=0))) if len(F) > 0 else 0
                )
                m["Titik sejati"].append(
                    _hitung_titik_sejati_ditemukan(F, sejati_unik)
                )
                fronts.append(F)

                langkah += 1
                if progress_callback is not None:
                    progress_callback(
                        langkah,
                        total_langkah,
                        f"{nama_s} - {nama} run {seed + 1}/{jumlah_run}",
                    )

            idx_best = int(np.argmax(m["HV"]))
            catat[nama] = m
            run_terbaik[nama] = {
                "seed": idx_best,
                "hv": float(m["HV"][idx_best]),
                "F": fronts[idx_best].astype(float).tolist(),
            }

        mg, mp = catat["MOGA"], catat["MOPSO"]
        p_hv = float(ranksums(mg["HV"], mp["HV"]).pvalue)
        igd_mg = [np.nan if v is None else v for v in mg["IGD"]]
        igd_mp = [np.nan if v is None else v for v in mp["IGD"]]
        p_igd = float(ranksums(igd_mg, igd_mp, nan_policy="omit").pvalue)
        efek = float(a12(mg["HV"], mp["HV"]))

        if p_hv >= 0.05:
            unggul = "seri"
        else:
            unggul = "MOGA" if np.mean(mg["HV"]) > np.mean(mp["HV"]) else "MOPSO"

        rekap_unggul[unggul] = rekap_unggul.get(unggul, 0) + 1

        lengkap_moga = float(np.mean(np.array(mg["Titik sejati"]) == n_sejati) * 100)
        lengkap_mopso = float(np.mean(np.array(mp["Titik sejati"]) == n_sejati) * 100)

        hasil_skenario[nama_s] = {
            "konfigurasi": asdict(konfig),
            "lolos": int(jumlah_lolos),
            "n_sejati": n_sejati,
            "titik_acuan": [float(titik_acuan[0]), float(titik_acuan[1])],
            "hv_sejati": hv_sejati,
            "F_sejati": F_sejati.astype(float).tolist(),
            "metrik": catat,
            "run_terbaik": run_terbaik,
            "ringkasan": {
                "hv_moga_mean": float(np.mean(mg["HV"])),
                "hv_moga_std": float(np.std(mg["HV"])),
                "hv_mopso_mean": float(np.mean(mp["HV"])),
                "hv_mopso_std": float(np.std(mp["HV"])),
                "igd_moga_mean": _ke_json_nilai(float(np.nanmean(igd_mg))),
                "igd_mopso_mean": _ke_json_nilai(float(np.nanmean(igd_mp))),
                "lengkap_moga": lengkap_moga,
                "lengkap_mopso": lengkap_mopso,
                "p_hv": p_hv,
                "p_igd": p_igd,
                "a12_hv": efek,
                "unggul": unggul,
            },
        }

    return {
        "jumlah_run": jumlah_run,
        "skenario": hasil_skenario,
        "rekap": rekap_unggul,
    }
