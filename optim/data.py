"""Modul pemuatan dan pengelompokan dataset menu McDonald's."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KOLOM_GIZI = [
    "Calories",
    "Protein",
    "Dietary Fiber",
    "Sodium",
    "Sugars",
    "Saturated Fat",
]

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "menu.csv"


@dataclass(frozen=True)
class DataMenu:
    """Menyimpan dataframe per slot dan matriks nutrisi siap hitung."""

    utama: pd.DataFrame
    lauk: pd.DataFrame
    minum: pd.DataFrame
    gizi_utama: np.ndarray
    gizi_lauk: np.ndarray
    gizi_minum: np.ndarray
    batas_atas: np.ndarray
    nama_lauk: list[str]
    total_kombinasi: int


def muat_data(path_csv: str | Path = DEFAULT_CSV_PATH) -> DataMenu:
    """Memuat menu.csv dan mengelompokkannya menjadi slot utama, lauk, dan minuman."""
    df = pd.read_csv(path_csv)

    kategori_utama = ["Breakfast", "Beef & Pork", "Chicken & Fish", "Salads"]
    kategori_minum = ["Beverages", "Coffee & Tea", "Smoothies & Shakes"]

    utama = df[df["Category"].isin(kategori_utama)]
    minum = df[df["Category"].isin(kategori_minum)]

    # Opsi tanpa lauk, semua nilai gizinya 0 dan diletakkan di index 0
    tanpa_lauk = pd.DataFrame([{kolom: 0 for kolom in df.columns}])
    tanpa_lauk["Item"] = "(tanpa lauk)"
    lauk = pd.concat([tanpa_lauk, df[df["Category"] == "Snacks & Sides"]])

    # Index dipakai sebagai nomor menu di kromosom
    utama = utama.reset_index(drop=True)
    lauk = lauk.reset_index(drop=True)
    minum = minum.reset_index(drop=True)

    gizi_utama = utama[KOLOM_GIZI].values
    gizi_lauk = lauk[KOLOM_GIZI].values
    gizi_minum = minum[KOLOM_GIZI].values
    batas_atas = np.array([len(utama) - 1, len(lauk) - 1, len(minum) - 1])
    nama_lauk = list(lauk["Item"])
    total_kombinasi = len(utama) * len(lauk) * len(minum)

    return DataMenu(
        utama=utama,
        lauk=lauk,
        minum=minum,
        gizi_utama=gizi_utama,
        gizi_lauk=gizi_lauk,
        gizi_minum=gizi_minum,
        batas_atas=batas_atas,
        nama_lauk=nama_lauk,
        total_kombinasi=total_kombinasi,
    )
