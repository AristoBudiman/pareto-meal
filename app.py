"""Aplikasi demo Streamlit untuk optimasi multi-objektif menu McDonald's."""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from optim.algoritma import jalankan_eksperimen, moga, mopso
from optim.data import DataMenu, muat_data
from optim.metrik import hypervolume
from optim.model import (
    Konfigurasi,
    ambil_pareto,
    brute_force,
    pilih_rekomendasi,
    tabel_solusi,
)

ROOT_DIR = Path(__file__).resolve().parent
HASIL_JSON_PATH = ROOT_DIR / "data" / "hasil_eksperimen.json"


@st.cache_data
def muat_data_cached() -> DataMenu:
    """Memuat dan mengelompokkan dataset menu.csv dengan cache Streamlit."""
    return muat_data(ROOT_DIR / "menu.csv")


@st.cache_data
def brute_force_cached(
    target_kalori: float,
    target_protein: float,
    target_serat: float,
    maks_natrium: float,
    maks_gula: float,
    maks_lemak_jenuh: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Menjalankan brute force dengan cache berdasarkan konfigurasi target dan batasan."""
    data = muat_data_cached()
    konfig = Konfigurasi(
        target_kalori=target_kalori,
        target_protein=target_protein,
        target_serat=target_serat,
        maks_natrium=maks_natrium,
        maks_gula=maks_gula,
        maks_lemak_jenuh=maks_lemak_jenuh,
    )
    return brute_force(data, konfig)


@st.cache_data
def muat_hasil_eksperimen_json() -> dict:
    """Membaca hasil precompute 30 run dari data/hasil_eksperimen.json."""
    with HASIL_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def buat_grafik_pareto(
    F_sejati: np.ndarray,
    fronts: list[tuple[str, np.ndarray, str, str]],
    judul: str,
) -> plt.Figure:
    """Membuat grafik Pareto front (f1 vs f2) dengan front sejati sebagai acuan."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if len(F_sejati) > 0:
        ax.plot(F_sejati[:, 0], F_sejati[:, 1], "k--", alpha=0.4)
        ax.scatter(
            F_sejati[:, 0],
            F_sejati[:, 1],
            s=160,
            facecolors="none",
            edgecolors="k",
            linewidths=1.5,
            label="Front sejati (brute force)",
            zorder=3,
        )

    for label, F, warna, penanda in fronts:
        if len(F) > 0:
            ax.scatter(
                F[:, 0],
                F[:, 1],
                c=warna,
                marker=penanda,
                s=70,
                label=label,
                zorder=4,
            )

    ax.set_xlabel("f1 = |kalori - target| (kkal)")
    ax.set_ylabel("f2 = kekurangan protein + serat (g)")
    ax.set_title(judul)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def buat_grafik_boxplot(hasil_json: dict) -> plt.Figure:
    """Membuat boxplot perbandingan HV dan IGD untuk MOGA dan MOPSO."""
    moga_hv = hasil_json["algoritma"]["MOGA"]["metrik_per_run"]["HV"]
    mopso_hv = hasil_json["algoritma"]["MOPSO"]["metrik_per_run"]["HV"]
    moga_igd = [
        v for v in hasil_json["algoritma"]["MOGA"]["metrik_per_run"]["IGD"] if v is not None
    ]
    mopso_igd = [
        v for v in hasil_json["algoritma"]["MOPSO"]["metrik_per_run"]["IGD"] if v is not None
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].boxplot([moga_hv, mopso_hv], tick_labels=["MOGA", "MOPSO"], patch_artist=True)
    axes[0].set_title("Distribusi Hypervolume (HV) - 30 Run")
    axes[0].set_ylabel("HV (lebih tinggi lebih baik)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].boxplot([moga_igd, mopso_igd], tick_labels=["MOGA", "MOPSO"], patch_artist=True)
    axes[1].set_title("Distribusi IGD - 30 Run")
    axes[1].set_ylabel("IGD (lebih rendah lebih baik)")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def jalankan_mode_a(
    data: DataMenu,
    konfig: Konfigurasi,
    pilihan_algo: str,
    seed: int,
) -> dict:
    """Menjalankan brute force dan algoritma terpilih untuk Tab 1."""
    X_sejati, F_sejati, jumlah_lolos = brute_force_cached(
        konfig.target_kalori,
        konfig.target_protein,
        konfig.target_serat,
        konfig.maks_natrium,
        konfig.maks_gula,
        konfig.maks_lemak_jenuh,
    )

    daftar = (
        [("MOGA", moga), ("MOPSO", mopso)]
        if pilihan_algo == "Keduanya"
        else [("MOGA", moga)]
        if pilihan_algo == "MOGA"
        else [("MOPSO", mopso)]
    )

    hasil_algo = {}
    for nama, fn_algo in daftar:
        mulai = time.time()
        X_a, F_a = fn_algo(np.random.default_rng(seed), data, konfig)
        waktu = time.time() - mulai
        hasil_algo[nama] = {
            "X": X_a,
            "F": F_a,
            "waktu": waktu,
            "hv": hypervolume(F_a),
        }

    return {
        "konfig": konfig,
        "pilihan_algo": pilihan_algo,
        "seed": seed,
        "X_sejati": X_sejati,
        "F_sejati": F_sejati,
        "jumlah_lolos": jumlah_lolos,
        "hv_sejati": hypervolume(F_sejati),
        "hasil_algo": hasil_algo,
    }


def tampilkan_kartu_rekomendasi(rekomendasi: list[dict[str, object]]) -> None:
    """Menampilkan 3 kartu rekomendasi menu dalam 3 kolom."""
    kolom = st.columns(3)
    for col, item in zip(kolom, rekomendasi):
        with col:
            with st.container(border=True):
                st.markdown(f"**{item['label']}**")
                st.write(item["nama_menu"])
                st.caption(f"f1 = {item['f1']:.0f} kkal | f2 = {item['f2']:.0f} g")
                st.markdown(
                    f"- Kalori: **{item['kalori']:.0f} kkal**\n"
                    f"- Protein: **{item['protein']:.0f} g** | Serat: **{item['serat']:.0f} g**\n"
                    f"- Natrium: **{item['natrium']:.0f} mg** | Gula: **{item['gula']:.0f} g**\n"
                    f"- Lemak jenuh: **{item['lemak_jenuh']:.1f} g**"
                )


def render_tab_rekomendasi(data: DataMenu) -> None:
    """Merender konten Tab 1: Rekomendasi Menu (Mode A)."""
    st.subheader("Rekomendasi Kombinasi Menu McDonald's")
    st.caption(
        "Atur target gizi, batasan nutrisi, pilihan algoritma, dan seed pada sidebar, "
        "lalu klik tombol **Cari menu**."
    )

    hasil = st.session_state.get("hasil_mode_a")
    if hasil is None:
        st.info("Klik **Cari menu** di sidebar untuk memulai pencarian kombinasi menu.")
        return

    jumlah_lolos = hasil["jumlah_lolos"]
    persen_lolos = (jumlah_lolos / data.total_kombinasi) * 100
    konfig: Konfigurasi = hasil["konfig"]

    if jumlah_lolos == 0:
        st.metric(
            "Kombinasi lolos batasan",
            f"0 / {data.total_kombinasi:,}",
            delta="0.00%",
            delta_color="off",
        )
        st.warning(
            "Tidak ada kombinasi menu yang memenuhi seluruh batasan gizi tersebut. "
            "Silakan longgarkan batas maksimum natrium, gula, atau lemak jenuh pada sidebar."
        )
        return

    # Baris ringkasan singkat
    hasil_algo = hasil["hasil_algo"]
    kolom_info = st.columns(2 + len(hasil_algo))
    kolom_info[0].metric(
        "Kombinasi lolos batasan",
        f"{jumlah_lolos:,} ({persen_lolos:.2f}%)",
        help=f"Dari total {data.total_kombinasi:,} kombinasi menu.",
    )
    kolom_info[1].metric(
        "HV Front Sejati",
        f"{hasil['hv_sejati']:.4f}",
        help=f"{len(hasil['F_sejati'])} solusi pada front sejati.",
    )
    for idx, (nama, info) in enumerate(hasil_algo.items(), start=2):
        kolom_info[idx].metric(
            f"HV {nama}",
            f"{info['hv']:.4f}",
            delta=f"{info['waktu']:.2f} dtk | {len(info['F'])} solusi",
            delta_color="off",
        )

    # Grafik Pareto front
    st.markdown("### Grafik Pareto Front")
    gaya = {"MOGA": ("tab:blue", "o"), "MOPSO": ("tab:orange", "x")}
    fronts_plot = []
    for nama, info in hasil_algo.items():
        warna, penanda = gaya[nama]
        fronts_plot.append((f"{nama} (seed {hasil['seed']})", info["F"], warna, penanda))

    fig = buat_grafik_pareto(
        hasil["F_sejati"],
        fronts_plot,
        "Pareto Front Pemilihan Menu McDonald's",
    )
    st.pyplot(fig)
    plt.close(fig)

    # Tabel solusi Pareto dari algoritma
    st.markdown("### Tabel Solusi Pareto Algoritma")
    for nama, info in hasil_algo.items():
        if len(hasil_algo) > 1:
            st.markdown(f"**{nama}**")
        if len(info["F"]) == 0:
            st.warning(
                f"Algoritma {nama} pada seed {hasil['seed']} belum menemukan solusi yang lolos semua batasan."
            )
        else:
            df_tabel = tabel_solusi(info["X"], info["F"], data)
            st.dataframe(
                df_tabel,
                width="stretch",
                hide_index=True,
                column_config={
                    "Kalori (kkal)": st.column_config.NumberColumn(format="%.0f"),
                    "Protein (g)": st.column_config.NumberColumn(format="%.0f"),
                    "Serat (g)": st.column_config.NumberColumn(format="%.0f"),
                    "Natrium (mg)": st.column_config.NumberColumn(format="%.0f"),
                    "Gula (g)": st.column_config.NumberColumn(format="%.0f"),
                    "Lemak Jenuh (g)": st.column_config.NumberColumn(format="%.1f"),
                    "f1": st.column_config.NumberColumn(format="%.0f"),
                    "f2": st.column_config.NumberColumn(format="%.0f"),
                },
            )

    # 3 kartu rekomendasi
    st.markdown("### 3 Rekomendasi Menu")
    kumpulan_X = [info["X"] for info in hasil_algo.values() if len(info["X"]) > 0]
    if kumpulan_X:
        X_gabung = np.vstack(kumpulan_X)
        X_rek, F_rek = ambil_pareto(X_gabung, data, konfig)
        if len(hasil_algo) > 1:
            st.caption("Dipilih dari gabungan solusi non-dominasi MOGA dan MOPSO.")
        else:
            nama_tunggal = next(iter(hasil_algo.keys()))
            st.caption(f"Dipilih dari solusi Pareto yang ditemukan oleh {nama_tunggal}.")
        rekomendasi = pilih_rekomendasi(X_rek, F_rek, data)
        tampilkan_kartu_rekomendasi(rekomendasi)
    else:
        st.info(
            "Algoritma tidak menemukan solusi yang lolos batasan pada seed ini. "
            "Berikut 3 rekomendasi acuan dari Pareto front sejati (brute force):"
        )
        rekomendasi_sejati = pilih_rekomendasi(hasil["X_sejati"], hasil["F_sejati"], data)
        tampilkan_kartu_rekomendasi(rekomendasi_sejati)


def render_tab_perbandingan(data: DataMenu) -> None:
    """Merender konten Tab 2: Perbandingan Algoritma (Mode B)."""
    st.subheader("Perbandingan Eksperimen 30 Run: MOGA vs MOPSO")

    if "eksperimen_data" not in st.session_state:
        if not HASIL_JSON_PATH.exists():
            st.error(
                f"Berkas `{HASIL_JSON_PATH}` tidak ditemukan. "
                "Jalankan `python scripts/precompute.py` terlebih dahulu."
            )
            return
        st.session_state["eksperimen_data"] = muat_hasil_eksperimen_json()

    hasil = st.session_state["eksperimen_data"]
    konfig = hasil["konfigurasi"]
    st.caption(
        f"Konfigurasi default: Target {konfig['target_kalori']:.0f} kkal, "
        f"Protein {konfig['target_protein']:.0f} g, Serat {konfig['target_serat']:.0f} g | "
        f"Batas Natrium ≤ {konfig['maks_natrium']:.0f} mg, Gula ≤ {konfig['maks_gula']:.0f} g, "
        f"Lemak jenuh ≤ {konfig['maks_lemak_jenuh']:.0f} g | "
        f"{hasil['jumlah_run']} run independen (seed 0–{hasil['jumlah_run'] - 1})."
    )

    # Ringkasan Front Sejati & Uji Wilcoxon
    w = hasil["wilcoxon_hv"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Kombinasi lolos batasan",
        f"{hasil['jumlah_kombinasi_lolos']:,}",
        delta=f"{hasil['jumlah_kombinasi_lolos'] / hasil['total_kombinasi'] * 100:.2f}%",
        delta_color="off",
    )
    c2.metric(
        "Front sejati",
        f"{len(hasil['front_sejati']['F'])} solusi",
        delta=f"HV = {hasil['front_sejati']['hv']:.4f}",
        delta_color="off",
    )
    c3.metric("Wilcoxon rank-sum (HV) p-value", f"{w['p_value']:.4f}")
    c4.metric(
        "Kesimpulan (alpha = 0.05)",
        "Signifikan" if w["signifikan"] else "Tidak signifikan",
    )

    if w["signifikan"]:
        st.success(
            f"Hasil uji Wilcoxon rank-sum pada metrik HV menunjukkan p = {w['p_value']:.4f} (< 0.05), "
            "sehingga terdapat perbedaan kinerja yang signifikan secara statistik antara MOGA dan MOPSO."
        )
    else:
        st.info(
            f"Hasil uji Wilcoxon rank-sum pada metrik HV menunjukkan p = {w['p_value']:.4f} (≥ 0.05), "
            "sehingga tidak terdapat perbedaan yang signifikan secara statistik antara MOGA dan MOPSO pada α = 0.05."
        )

    # Tabel metrik rata-rata ± std
    st.markdown("### Tabel Metrik Kinerja (Rata-rata ± Standar Deviasi)")
    daftar_metrik = ["HV", "Spacing", "IGD", "Jumlah solusi", "Waktu (s)"]
    baris_tabel = []
    for m in daftar_metrik:
        s_moga = hasil["algoritma"]["MOGA"]["ringkasan"][m]
        s_mopso = hasil["algoritma"]["MOPSO"]["ringkasan"][m]
        baris_tabel.append(
            {
                "Metrik": m,
                "MOGA": f"{s_moga['mean']:.4f} ± {s_moga['std']:.4f}",
                "MOPSO": f"{s_mopso['mean']:.4f} ± {s_mopso['std']:.4f}",
            }
        )
    st.dataframe(pd.DataFrame(baris_tabel), width="stretch", hide_index=True)

    # Boxplot HV dan IGD
    st.markdown("### Distribusi HV dan IGD (30 Run)")
    fig_box = buat_grafik_boxplot(hasil)
    st.pyplot(fig_box)
    plt.close(fig_box)

    # Grafik front run terbaik MOGA dan MOPSO vs front sejati
    st.markdown("### Front Run Terbaik vs Front Sejati")
    F_sejati = np.array(hasil["front_sejati"]["F"], dtype=float)
    F_moga_best = np.array(hasil["algoritma"]["MOGA"]["run_terbaik"]["F"], dtype=float)
    F_mopso_best = np.array(hasil["algoritma"]["MOPSO"]["run_terbaik"]["F"], dtype=float)

    fig_pf = buat_grafik_pareto(
        F_sejati,
        [
            (
                f"MOGA (run terbaik, seed {hasil['algoritma']['MOGA']['run_terbaik']['seed']})",
                F_moga_best,
                "tab:blue",
                "o",
            ),
            (
                f"MOPSO (run terbaik, seed {hasil['algoritma']['MOPSO']['run_terbaik']['seed']})",
                F_mopso_best,
                "tab:orange",
                "x",
            ),
        ],
        "Pareto Front Run Terbaik MOGA & MOPSO vs Front Sejati",
    )
    st.pyplot(fig_pf)
    plt.close(fig_pf)

    # Expander opsional untuk menjalankan ulang 30 run
    with st.expander("Jalankan ulang 30 run"):
        st.warning(
            "Peringatan: Menjalankan ulang eksperimen 30 run untuk MOGA dan MOPSO "
            "membutuhkan waktu sekitar 1 menit."
        )
        if st.button("Mulai jalankan ulang 30 run", key="btn_rerun_30"):
            bar_progres = st.progress(0, text="Memulai eksperimen...")

            def update_progres(langkah: int, total: int, teks: str) -> None:
                bar_progres.progress(langkah / total, text=f"Menjalankan {teks}...")

            hasil_baru = jalankan_eksperimen(
                data,
                Konfigurasi(),
                jumlah_run=30,
                progress_callback=update_progres,
            )
            bar_progres.empty()
            st.session_state["eksperimen_data"] = hasil_baru
            st.rerun()


def render_tab_tentang(data: DataMenu) -> None:
    """Merender konten Tab 3: Tentang proyek."""
    st.subheader("Tentang Proyek")

    st.markdown(
        """
### Tujuan
Aplikasi ini merupakan demo proyek mata kuliah **Kecerdasan Komputasional** untuk
optimasi multi-objektif pemilihan kombinasi menu McDonald's dalam satu kali makan,
sekaligus membandingkan kinerja algoritma **MOGA** dan **MOPSO**.
Pendekatannya bersifat realistis (*harm reduction*): apabila seseorang makan di restoran cepat saji,
kombinasi menu apa yang paling mendekati target kalori dan kebutuhan protein serta serat
tanpa melampaui batas aman natrium, gula, dan lemak jenuh.

### Dataset
Data berasal dari **McDonald's Nutrition Facts (Kaggle)**. Setelah membuang porsi *sharing*
(`McNuggets 20 & 40 piece`), item dikelompokkan menjadi 3 slot:
- **Menu utama**: {n_utama} item (`Breakfast`, `Beef & Pork`, `Chicken & Fish`, `Salads`, kecuali `Hash Brown`)
- **Lauk pendamping**: {n_lauk} item (`Snacks & Sides` + `Hash Brown`) ditambah **1 opsi tanpa lauk**
- **Minuman**: {n_minum} item (`Beverages`, `Coffee & Tea`, `Smoothies & Shakes`)
- **Total ruang pencarian**: {total_komb:,} kombinasi

### Representasi Solusi
Setiap kandidat kombinasi direpresentasikan sebagai vektor integer 3 gen:
`x = [indeks_utama, indeks_lauk, indeks_minuman]`, dengan `indeks_lauk = 0` berarti **tanpa lauk**.

### Fungsi Objektif & Penanganan Batasan
Kedua fungsi objektif berikut **diminimalkan** secara bersamaan:
- **f1** = `|total kalori - target kalori|` (deviasi kalori dalam kkal)
- **f2** = `max(0, target protein - total protein) + max(0, target serat - total serat)` (kekurangan protein dan serat dalam gram)

Batasan harian per satu kali makan (default: natrium ≤ 800 mg, gula ≤ 17 g, lemak jenuh ≤ 7 g)
ditangani menggunakan **fungsi penalti** yang dinormalisasi terhadap batas masing-masing:
- `pelanggaran = max(0, natrium - maks_natrium)/maks_natrium + max(0, gula - maks_gula)/maks_gula + max(0, lemak - maks_lemak)/maks_lemak`
- Nilai penalti ditambahkan sebesar `1000 × pelanggaran` pada `f1` dan `100 × pelanggaran` pada `f2`.
  Hanya solusi dengan `pelanggaran == 0` yang diambil ke Pareto front akhir.

### Algoritma & Front Sejati
- **MOGA**: menggunakan *Fonseca-Fleming ranking*, *fitness sharing*, seleksi turnamen, *crossover uniform*, dan mutasi acak.
- **MOPSO**: menggunakan partikel kontinu yang dibulatkan ke indeks diskrit, arsip eksternal non-dominasi, pemilihan *leader* dari arsip, dan turbulensi acak.
- **Front Sejati (Brute Force)**: mengevaluasi seluruh {total_komb:,} kombinasi sebagai acuan *ground truth*.

### Metrik Evaluasi
- **Hypervolume (HV)**: luas area yang didominasi front terhadap titik acuan `[300, 15]`, dinormalisasi ke `[0, 1]` (lebih tinggi lebih baik).
- **Spacing**: kerataan jarak antar solusi pada front ternormalisasi (lebih rendah lebih baik).
- **IGD (Inverted Generational Distance)**: rata-rata jarak minimum titik front sejati ke front hasil algoritma (lebih rendah lebih baik).
- **Uji Statistik**: **Wilcoxon rank-sum test** pada nilai HV dari 30 run independen dengan tingkat signifikansi `α = 0.05`.

### Referensi
1. **Sahin, O., & Aytekin-Sahin, G. (2024).** *Open-source multi-objective optimization software for menu planning*. Expert Systems with Applications, 252, 124213.
2. **Ahourag, A., Elkari, B., El Moutaouakil, K., et al. (2024).** *A Multiobjective Diet Planning Model for Diabetic Patients in the Moroccan Health Context Using Particle Swarm Intelligence*. Statistics, Optimization and Information Computing, 12, 605–616.
        """.format(
            n_utama=len(data.utama),
            n_lauk=len(data.lauk),
            n_minum=len(data.minum),
            total_komb=data.total_kombinasi,
        )
    )


def main() -> None:
    st.set_page_config(
        page_title="Optimasi Menu McDonald's — MOGA vs MOPSO",
        page_icon="🍔",
        layout="wide",
    )

    data = muat_data_cached()

    st.title("Optimasi Multi-Objektif Pemilihan Menu McDonald's")
    st.caption("Perbandingan MOGA dan MOPSO untuk Kombinasi Menu Satu Kali Makan")

    # Sidebar untuk Mode A
    with st.sidebar:
        st.header("Pengaturan Pencarian Menu")
        st.subheader("Target Gizi (1× Makan)")
        target_kalori = st.number_input(
            "Target kalori (kkal)",
            min_value=0,
            max_value=3000,
            value=600,
            step=50,
        )
        target_protein = st.number_input(
            "Target protein (g)",
            min_value=0,
            max_value=200,
            value=25,
            step=1,
        )
        target_serat = st.number_input(
            "Target serat (g)",
            min_value=0,
            max_value=100,
            value=8,
            step=1,
        )

        st.subheader("Batasan Nutrisi Maksimum")
        maks_natrium = st.number_input(
            "Maksimum natrium (mg)",
            min_value=1,
            max_value=5000,
            value=800,
            step=50,
        )
        maks_gula = st.number_input(
            "Maksimum gula (g)",
            min_value=1,
            max_value=200,
            value=17,
            step=1,
        )
        maks_lemak_jenuh = st.number_input(
            "Maksimum lemak jenuh (g)",
            min_value=1,
            max_value=100,
            value=7,
            step=1,
        )

        st.subheader("Algoritma & Seed")
        pilihan_algo = st.selectbox(
            "Pilih algoritma",
            options=["Keduanya", "MOGA", "MOPSO"],
            index=0,
        )
        seed = st.number_input(
            "Seed acak",
            min_value=0,
            max_value=100000,
            value=0,
            step=1,
        )

        tombol_cari = st.button("Cari menu", type="primary", width="stretch")

    konfig_input = Konfigurasi(
        target_kalori=float(target_kalori),
        target_protein=float(target_protein),
        target_serat=float(target_serat),
        maks_natrium=float(maks_natrium),
        maks_gula=float(maks_gula),
        maks_lemak_jenuh=float(maks_lemak_jenuh),
    )

    if tombol_cari or "hasil_mode_a" not in st.session_state:
        with st.spinner("Mencari kombinasi menu optimal..."):
            st.session_state["hasil_mode_a"] = jalankan_mode_a(
                data,
                konfig_input,
                pilihan_algo,
                int(seed),
            )

    tab1, tab2, tab3 = st.tabs(
        ["Rekomendasi Menu", "Perbandingan Algoritma", "Tentang"]
    )
    with tab1:
        render_tab_rekomendasi(data)
    with tab2:
        render_tab_perbandingan(data)
    with tab3:
        render_tab_tentang(data)


if __name__ == "__main__":
    main()
