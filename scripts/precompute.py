"""Menjalankan eksperimen utama 30 run dan multi-skenario (S1-S4) lalu menyimpan ke JSON."""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from optim.algoritma import jalankan_eksperimen, jalankan_multi_skenario
from optim.data import muat_data
from optim.model import Konfigurasi


def main() -> None:
    data = muat_data(ROOT_DIR / "menu.csv")
    konfig = Konfigurasi()

    print(f"Menu utama : {len(data.utama)}")
    print(f"Lauk       : {len(data.lauk)}")
    print(f"Minuman    : {len(data.minum)}")
    print(f"Total kombinasi: {data.total_kombinasi}")

    print("\nMenjalankan eksperimen utama (30 run)...")
    hasil = jalankan_eksperimen(data, konfig, jumlah_run=30)

    print("Menjalankan eksperimen multi-skenario (S1-S4, 30 run per skenario)...")
    hasil["multi_skenario"] = jalankan_multi_skenario(data, jumlah_run=30)

    output_path = ROOT_DIR / "data" / "hasil_eksperimen.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False, allow_nan=False)

    fs = hasil["front_sejati"]
    print(
        f"\nLolos batasan   : {hasil['jumlah_kombinasi_lolos']} "
        f"({hasil['jumlah_kombinasi_lolos'] / hasil['total_kombinasi'] * 100:.2f}%)"
    )
    print(f"Titik acuan     : {hasil['titik_acuan']}")
    print(
        f"Front sejati    : {fs['jumlah_titik']} titik, {fs['jumlah_menu']} menu | "
        f"HV = {fs['hv']:.4f} | Spacing = {fs['spacing']:.4f}"
    )

    for nama, info in hasil["algoritma"].items():
        print(f"\n=== {nama} ===")
        for m, stat in info["ringkasan"].items():
            print(f"  {m:15s}: {stat['mean']:.4f} ± {stat['std']:.4f}")
        print(f"  Distribusi titik sejati: {info['distribusi_titik_sejati']}")

    for m, w in hasil["wilcoxon"].items():
        status = "signifikan" if w["signifikan"] else "tidak signifikan"
        print(f"\nWilcoxon {m}: p = {w['p_value']:.4f} -> {status}")

    print("\n=== Ringkasan Multi-skenario ===")
    for nama_s, s in hasil["multi_skenario"]["skenario"].items():
        r = s["ringkasan"]
        print(
            f"  {nama_s:20s} | Lolos={s['lolos']:4d} | Acuan={s['titik_acuan']} | "
            f"Sejati={s['n_sejati']} | HV_sejati={s['hv_sejati']:.4f} | "
            f"HV_MOGA={r['hv_moga_mean']:.4f} | HV_MOPSO={r['hv_mopso_mean']:.4f} | "
            f"Lengkap={r['lengkap_moga']:.0f}%/{r['lengkap_mopso']:.0f}% | "
            f"A12={r['a12_hv']:.2f} | Unggul={r['unggul']}"
        )
    print(f"Rekap: {hasil['multi_skenario']['rekap']}")
    print(f"\nTersimpan ke: {output_path}")


if __name__ == "__main__":
    main()
