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

    # Buang porsi sharing (McNuggets 20 & 40 piece)
    df = df[~df["Item"].str.contains(r"\((?:20|40) piece\)")]

    # Kelompokkan menjadi 3 slot
    utama = df[
        df["Category"].isin(["Breakfast", "Beef & Pork", "Chicken & Fish", "Salads"])
        & (df["Item"] != "Hash Brown")
    ].reset_index(drop=True)
    lauk = pd.concat(
        [
            df[df["Category"] == "Snacks & Sides"],
            df[df["Item"] == "Hash Brown"],
        ]
    ).reset_index(drop=True)
    minum = df[
        df["Category"].isin(["Beverages", "Coffee & Tea", "Smoothies & Shakes"])
    ].reset_index(drop=True)

    gizi_utama = utama[KOLOM_GIZI].values
    # Baris 0 = tanpa lauk
    gizi_lauk = np.vstack([np.zeros(len(KOLOM_GIZI)), lauk[KOLOM_GIZI].values])
    gizi_minum = minum[KOLOM_GIZI].values
    batas_atas = np.array(
        [len(gizi_utama) - 1, len(gizi_lauk) - 1, len(gizi_minum) - 1]
    )
    nama_lauk = ["(tanpa lauk)"] + list(lauk["Item"])
    total_kombinasi = len(utama) * (len(lauk) + 1) * len(minum)

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
