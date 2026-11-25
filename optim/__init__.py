"""Paket optimasi multi-objektif pemilihan menu McDonald's."""

from optim.data import DataMenu, KOLOM_GIZI, muat_data
from optim.model import (
    Konfigurasi,
    ambil_pareto,
    brute_force,
    evaluasi,
    hitung_gizi,
    nama_menu,
    non_dominasi,
    pilih_rekomendasi,
    tabel_solusi,
)
from optim.metrik import a12, hitung_titik_acuan, hypervolume, igd, spacing
from optim.algoritma import (
    SKENARIO_DEFAULT,
    jalankan_eksperimen,
    jalankan_multi_skenario,
    kotak_grid,
    moga,
    mopso,
)

__all__ = [
    "DataMenu",
    "KOLOM_GIZI",
    "Konfigurasi",
    "SKENARIO_DEFAULT",
    "a12",
    "ambil_pareto",
    "brute_force",
    "evaluasi",
    "hitung_gizi",
    "hitung_titik_acuan",
    "hypervolume",
    "igd",
    "jalankan_eksperimen",
    "jalankan_multi_skenario",
    "kotak_grid",
    "moga",
    "mopso",
    "muat_data",
    "nama_menu",
    "non_dominasi",
    "pilih_rekomendasi",
    "spacing",
    "tabel_solusi",
]
