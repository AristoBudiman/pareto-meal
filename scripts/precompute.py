"""Menjalankan eksperimen 30 run dengan konfigurasi default dan menyimpan hasil ke JSON."""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from optim.algoritma import jalankan_eksperimen
from optim.data import muat_data
from optim.model import Konfigurasi


def main() -> None:
    data = muat_data(ROOT_DIR / "menu.csv")
    konfig = Konfigurasi()

    print(f"Menu utama : {len(data.utama)}")
    print(f"Lauk       : {len(data.lauk)} (+1 pilihan tanpa lauk)")
    print(f"Minuman    : {len(data.minum)}")
    print(f"Total kombinasi: {data.total_kombinasi}")

    hasil = jalankan_eksperimen(data, konfig, jumlah_run=30)

    output_path = ROOT_DIR / "data" / "hasil_eksperimen.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(
        f"\nKombinasi lolos batasan: {hasil['jumlah_kombinasi_lolos']} "
        f"({hasil['jumlah_kombinasi_lolos'] / hasil['total_kombinasi'] * 100:.2f}%)"
    )
    print(
        f"Front sejati: {len(hasil['front_sejati']['F'])} solusi | "
        f"HV = {hasil['front_sejati']['hv']:.4f}"
    )
    for nama, info in hasil["algoritma"].items():
        print(f"\n=== {nama} ===")
        for m, stat in info["ringkasan"].items():
            print(f"  {m:15s}: {stat['mean']:.4f} ± {stat['std']:.4f}")
    w = hasil["wilcoxon_hv"]
    status = "signifikan" if w["signifikan"] else "tidak signifikan"
    print(f"\nWilcoxon rank-sum (HV): p = {w['p_value']:.4f} -> {status}")
    print(f"Tersimpan ke: {output_path}")


if __name__ == "__main__":
    main()
