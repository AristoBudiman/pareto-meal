"""Aplikasi demo Streamlit untuk optimasi multi-objektif menu McDonald's."""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from optim.algoritma import (
    jalankan_eksperimen,
    jalankan_multi_skenario,
    moga,
    mopso,
)
from optim.data import DataMenu, muat_data
from optim.metrik import hitung_titik_acuan, hypervolume
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
    """Membaca hasil precompute dari data/hasil_eksperimen.json."""
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
        Fs = F_sejati[np.argsort(F_sejati[:, 0])]
        ax.plot(Fs[:, 0], Fs[:, 1], "k--", alpha=0.4)
        ax.scatter(
            Fs[:, 0],
            Fs[:, 1],
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


def buat_boxplot_hv_skenario(nama_s: str, sken: dict) -> plt.Figure:
    """Membuat boxplot HV 30 run untuk satu skenario dengan garis HV front sejati."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    hv_moga = sken["metrik"]["MOGA"]["HV"]
    hv_mopso = sken["metrik"]["MOPSO"]["HV"]
    ax.boxplot([hv_moga, hv_mopso], tick_labels=["MOGA", "MOPSO"], patch_artist=True)
    ax.axhline(
        sken["hv_sejati"],
        color="k",
        ls="--",
        alpha=0.6,
        label=f"HV front sejati ({sken['hv_sejati']:.4f})",
    )
    ax.set_title(f"Distribusi HV — {nama_s}")
    ax.set_ylabel("HV")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
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
    titik_acuan = hitung_titik_acuan(F_sejati)

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
        jumlah_titik = int(len(np.unique(F_a, axis=0))) if len(F_a) > 0 else 0
        hasil_algo[nama] = {
            "X": X_a,
            "F": F_a,
            "waktu": waktu,
            "hv": hypervolume(F_a, titik_acuan) if jumlah_lolos > 0 else 0.0,
            "jumlah_titik": jumlah_titik,
        }

    jumlah_titik_sejati = int(len(np.unique(F_sejati, axis=0))) if len(F_sejati) > 0 else 0
    return {
        "konfig": konfig,
        "pilihan_algo": pilihan_algo,
        "seed": seed,
        "X_sejati": X_sejati,
        "F_sejati": F_sejati,
        "titik_acuan": titik_acuan,
        "jumlah_titik_sejati": jumlah_titik_sejati,
        "jumlah_lolos": jumlah_lolos,
        "hv_sejati": hypervolume(F_sejati, titik_acuan) if jumlah_lolos > 0 else 0.0,
        "hasil_algo": hasil_algo,
    }


def tampilkan_kartu_rekomendasi(rekomendasi: list[dict[str, object]]) -> None:
    """Menampilkan 3 kartu rekomendasi menu dalam 3 kolom (semua menu per titik)."""
    kolom = st.columns(3)
    for col, item in zip(kolom, rekomendasi):
        with col:
            with st.container(border=True):
                st.markdown(f"**{item['label']}**")
                st.caption(f"f1 = {item['f1']:.0f} kkal | f2 = {item['f2']:.0f} g")
                menus = item["menus"]
                for idx, menu in enumerate(menus):
                    if idx > 0:
                        st.divider()
                    st.write(menu["nama_menu"])
                    st.markdown(
                        f"- Kalori: **{menu['kalori']:.0f} kkal**\n"
                        f"- Protein: **{menu['protein']:.0f} g** | Serat: **{menu['serat']:.0f} g**\n"
                        f"- Natrium: **{menu['natrium']:.0f} mg** | Gula: **{menu['gula']:.0f} g**\n"
                        f"- Lemak jenuh: **{menu['lemak_jenuh']:.1f} g**"
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
    titik_acuan: np.ndarray = hasil["titik_acuan"]

    if jumlah_lolos == 0:
        st.metric("Kombinasi lolos batasan", f"0 / {data.total_kombinasi:,}")
        st.caption("0.00% dari total kombinasi")
        st.warning(
            "Tidak ada kombinasi menu yang memenuhi seluruh batasan gizi tersebut. "
            "Silakan longgarkan batas maksimum natrium, gula, atau lemak jenuh pada sidebar."
        )
        return

    # Baris ringkasan singkat (caption biasa agar tidak memicu panah delta)
    hasil_algo = hasil["hasil_algo"]
    kolom_info = st.columns(2 + len(hasil_algo))
    with kolom_info[0]:
        st.metric("Kombinasi lolos batasan", f"{jumlah_lolos:,} ({persen_lolos:.2f}%)")
        st.caption(f"Dari {data.total_kombinasi:,} total kombinasi")
    with kolom_info[1]:
        st.metric("HV Front Sejati", f"{hasil['hv_sejati']:.4f}")
        st.caption(
            f"{hasil['jumlah_titik_sejati']} titik ({len(hasil['F_sejati'])} menu) | "
            f"Acuan [{titik_acuan[0]:.0f}, {titik_acuan[1]:.0f}]"
        )
    for idx, (nama, info) in enumerate(hasil_algo.items(), start=2):
        with kolom_info[idx]:
            st.metric(f"HV {nama}", f"{info['hv']:.4f}")
            st.caption(
                f"Waktu: {info['waktu']:.2f} dtk | {info['jumlah_titik']} titik ({len(info['F'])} menu)"
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
        rekomendasi = pilih_rekomendasi(X_rek, F_rek, data, titik_acuan=titik_acuan)
        tampilkan_kartu_rekomendasi(rekomendasi)
    else:
        st.info(
            "Algoritma tidak menemukan solusi yang lolos batasan pada seed ini. "
            "Berikut 3 rekomendasi acuan dari Pareto front sejati (brute force):"
        )
        rekomendasi_sejati = pilih_rekomendasi(
            hasil["X_sejati"], hasil["F_sejati"], data, titik_acuan=titik_acuan
        )
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
    fs = hasil["front_sejati"]
    acuan = hasil["titik_acuan"]
    st.caption(
        f"Konfigurasi default: Target {konfig['target_kalori']:.0f} kkal, "
        f"Protein {konfig['target_protein']:.0f} g, Serat {konfig['target_serat']:.0f} g | "
        f"Batas Natrium ≤ {konfig['maks_natrium']:.0f} mg, Gula ≤ {konfig['maks_gula']:.0f} g, "
        f"Lemak jenuh ≤ {konfig['maks_lemak_jenuh']:.0f} g | "
        f"{hasil['jumlah_run']} run independen (seed 0–{hasil['jumlah_run'] - 1})."
    )

    # Ringkasan Front Sejati & Uji Wilcoxon (HV dan IGD)
    w_hv = hasil["wilcoxon"]["HV"]
    w_igd = hasil["wilcoxon"]["IGD"]
    persen_lolos = hasil["jumlah_kombinasi_lolos"] / hasil["total_kombinasi"] * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Kombinasi lolos batasan", f"{hasil['jumlah_kombinasi_lolos']:,}")
        st.caption(f"{persen_lolos:.2f}% dari {hasil['total_kombinasi']:,}")
    with c2:
        st.metric("Front sejati", f"{fs['jumlah_titik']} titik, {fs['jumlah_menu']} menu")
        st.caption(
            f"HV = {fs['hv']:.4f} | Spacing = {fs['spacing']:.4f} | "
            f"Acuan [{acuan[0]:.0f}, {acuan[1]:.0f}]"
        )
    with c3:
        st.metric("Wilcoxon HV (p-value)", f"{w_hv['p_value']:.4f}")
        st.caption("Signifikan (α = 0.05)" if w_hv["signifikan"] else "Tidak signifikan (α = 0.05)")
    with c4:
        st.metric("Wilcoxon IGD (p-value)", f"{w_igd['p_value']:.4f}")
        st.caption("Signifikan (α = 0.05)" if w_igd["signifikan"] else "Tidak signifikan (α = 0.05)")

    if w_hv["signifikan"] and w_igd["signifikan"]:
        st.success(
            f"Uji Wilcoxon rank-sum menunjukkan perbedaan signifikan antara MOGA dan MOPSO "
            f"pada HV (p = {w_hv['p_value']:.4f}) maupun IGD (p = {w_igd['p_value']:.4f}) pada α = 0.05."
        )
    else:
        st.info(
            f"Uji Wilcoxon rank-sum (α = 0.05): HV p = {w_hv['p_value']:.4f} "
            f"({'signifikan' if w_hv['signifikan'] else 'tidak signifikan'}), "
            f"IGD p = {w_igd['p_value']:.4f} "
            f"({'signifikan' if w_igd['signifikan'] else 'tidak signifikan'})."
        )

    # Tabel metrik rata-rata ± std & Tabel titik front sejati per run
    kol_kiri, kol_kanan = st.columns([3, 2])
    with kol_kiri:
        st.markdown("### Tabel Metrik Kinerja (Rata-rata ± Standar Deviasi)")
        daftar_metrik = ["HV", "Spacing", "IGD", "Jumlah titik", "Waktu (detik)"]
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

    with kol_kanan:
        st.markdown("### Titik Front Sejati Ditemukan per Run")
        dist_moga = hasil["algoritma"]["MOGA"]["distribusi_titik_sejati"]
        dist_mopso = hasil["algoritma"]["MOPSO"]["distribusi_titik_sejati"]
        baris_dist = []
        for k in range(fs["jumlah_titik"] + 1):
            key = str(k)
            baris_dist.append(
                {
                    "Titik Front Sejati": f"{k} titik",
                    "MOGA (run)": dist_moga.get(key, 0),
                    "MOPSO (run)": dist_mopso.get(key, 0),
                }
            )
        st.dataframe(pd.DataFrame(baris_dist), width="stretch", hide_index=True)

    # Boxplot HV dan IGD
    st.markdown("### Distribusi HV dan IGD (30 Run)")
    fig_box = buat_grafik_boxplot(hasil)
    st.pyplot(fig_box)
    plt.close(fig_box)

    # Grafik front run terbaik MOGA dan MOPSO vs front sejati
    st.markdown("### Front Run Terbaik vs Front Sejati")
    F_sejati = np.array(fs["F"], dtype=float)
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

    # Bagian Multi-skenario
    st.divider()
    st.markdown("## Multi-skenario")
    st.markdown(
        """
Supaya kesimpulan perbandingan tidak bergantung pada satu konfigurasi saja, eksperimen diulang pada 4 skenario target dan batasan gizi.
Setiap skenario dicari ulang front sejatinya dengan *brute force* dan dihitung ulang titik acuannya (`2 × nilai terburuk front sejati`), lalu MOGA dan MOPSO masing-masing dijalankan 30 kali (`seed` 0–29):

- **S1 Baseline**: konfigurasi eksperimen utama (target `600 kkal`, protein `25 g`, serat `8 g`; natrium ≤ `800 mg`, gula ≤ `17 g`, lemak jenuh ≤ `7 g`).
- **S2 Batasan sedang**: natrium ≤ `700 mg`, gula ≤ `12 g`, lemak jenuh ≤ `6 g` (memperketat batasan dari baseline).
- **S3 Batasan ketat**: natrium ≤ `600 mg`, gula ≤ `10 g`, lemak jenuh ≤ `5 g` (menguji saat kombinasi layak makin sedikit).
- **S4 Target 800 kkal**: target kalori `800 kkal`, protein `35 g`, serat `10 g` (untuk kebutuhan energi yang lebih tinggi).
        """
    )

    multi = hasil.get("multi_skenario")
    if multi:
        sken_dict = multi["skenario"]
        baris_sken = []
        for nama_s, h in sken_dict.items():
            r = h["ringkasan"]
            baris_sken.append(
                {
                    "Skenario": nama_s,
                    "Front sejati": h["n_sejati"],
                    "HV sejati": round(h["hv_sejati"], 4),
                    "HV MOGA": f"{r['hv_moga_mean']:.4f} ± {r['hv_moga_std']:.4f}",
                    "HV MOPSO": f"{r['hv_mopso_mean']:.4f} ± {r['hv_mopso_std']:.4f}",
                    "IGD MOGA": f"{r['igd_moga_mean']:.4f}",
                    "IGD MOPSO": f"{r['igd_mopso_mean']:.4f}",
                    "Lengkap MOGA": f"{r['lengkap_moga']:.0f}%",
                    "Lengkap MOPSO": f"{r['lengkap_mopso']:.0f}%",
                    "p HV": round(r["p_hv"], 4),
                    "p IGD": round(r["p_igd"], 4),
                    "A12 HV": round(r["a12_hv"], 2),
                    "Unggul": r["unggul"],
                }
            )

        st.markdown("### Tabel Ringkasan Multi-skenario")
        st.dataframe(pd.DataFrame(baris_sken), width="stretch", hide_index=True)

        rekap_teks = ", ".join(
            f"**{k} {v}**" for k, v in multi["rekap"].items()
        )
        st.success(
            f"**Rekap:** {rekap_teks} dari {len(baris_sken)} skenario "
            f"(berdasarkan perbedaan HV signifikan menurut uji Wilcoxon, p < 0.05)."
        )
        st.caption(
            "Kolom **Lengkap** = persentase run yang menemukan seluruh titik front sejati. "
            "**A12 > 0.5** berarti HV MOGA cenderung lebih besar (patokan Vargha-Delaney: 0.56 kecil, 0.64 sedang, 0.71 besar)."
        )

        st.markdown("### Visualisasi per Skenario")
        pilihan_sken = st.selectbox(
            "Pilih skenario",
            options=list(sken_dict.keys()),
            index=0,
        )
        sken_aktif = sken_dict[pilihan_sken]
        r_aktif = sken_aktif["ringkasan"]
        acuan_s = sken_aktif["titik_acuan"]
        st.caption(
            f"**{pilihan_sken}** — Lolos batasan: **{sken_aktif['lolos']:,}** kombinasi | "
            f"Front sejati: **{sken_aktif['n_sejati']} titik** (HV = {sken_aktif['hv_sejati']:.4f}) | "
            f"Titik acuan: `[{acuan_s[0]:.0f}, {acuan_s[1]:.0f}]` | "
            f"Lengkap: MOGA {r_aktif['lengkap_moga']:.0f}%, MOPSO {r_aktif['lengkap_mopso']:.0f}% | "
            f"A12 HV: {r_aktif['a12_hv']:.2f}"
        )

        col_grafik1, col_grafik2 = st.columns(2)
        with col_grafik1:
            F_sejati_s = np.array(sken_aktif["F_sejati"], dtype=float)
            F_moga_s = np.array(sken_aktif["run_terbaik"]["MOGA"]["F"], dtype=float)
            F_mopso_s = np.array(sken_aktif["run_terbaik"]["MOPSO"]["F"], dtype=float)
            fig_s = buat_grafik_pareto(
                F_sejati_s,
                [
                    (
                        f"MOGA (seed {sken_aktif['run_terbaik']['MOGA']['seed']})",
                        F_moga_s,
                        "tab:blue",
                        "o",
                    ),
                    (
                        f"MOPSO (seed {sken_aktif['run_terbaik']['MOPSO']['seed']})",
                        F_mopso_s,
                        "tab:orange",
                        "x",
                    ),
                ],
                f"Pareto Front Run Terbaik — {pilihan_sken}",
            )
            st.pyplot(fig_s)
            plt.close(fig_s)

        with col_grafik2:
            fig_box_s = buat_boxplot_hv_skenario(pilihan_sken, sken_aktif)
            st.pyplot(fig_box_s)
            plt.close(fig_box_s)

    # Expander opsional untuk menjalankan ulang 30 run
    with st.expander("Jalankan ulang 30 run"):
        st.warning(
            "Peringatan: Menjalankan ulang eksperimen utama dan seluruh 4 skenario (masing-masing 30 run "
            "untuk MOGA dan MOPSO) membutuhkan waktu sekitar 3–4 menit."
        )
        if st.button("Mulai jalankan ulang 30 run", key="btn_rerun_30"):
            bar_progres = st.progress(0, text="Memulai eksperimen...")
            total_utama = 60
            total_sken = 4 * 60
            total_semua = total_utama + total_sken

            def update_utama(langkah: int, total: int, teks: str) -> None:
                bar_progres.progress(
                    langkah / total_semua,
                    text=f"Eksperimen utama: {teks}...",
                )

            def update_sken(langkah: int, total: int, teks: str) -> None:
                bar_progres.progress(
                    (total_utama + langkah) / total_semua,
                    text=f"Multi-skenario: {teks}...",
                )

            hasil_baru = jalankan_eksperimen(
                data,
                Konfigurasi(),
                jumlah_run=30,
                progress_callback=update_utama,
            )
            hasil_baru["multi_skenario"] = jalankan_multi_skenario(
                data,
                jumlah_run=30,
                progress_callback=update_sken,
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
Data yang dipakai adalah **McDonald's Nutrition Facts (Kaggle)** berisi 260 item menu beserta
kandungan gizinya. Item dibagi menjadi 3 slot sesuai kategori di dataset tanpa membuang porsi besar
dari awal (item yang tidak cocok tersaring oleh batasan gizi):
- **Menu utama**: {n_utama} item (`Breakfast`, `Beef & Pork`, `Chicken & Fish`, `Salads`)
- **Lauk pendamping**: {n_lauk} item (`Snacks & Sides`, ditambah 1 opsi `(tanpa lauk)` bernilai gizi 0 di indeks 0)
- **Minuman**: {n_minum} item (`Beverages`, `Coffee & Tea`, `Smoothies & Shakes`)
- **Total ruang pencarian**: {total_komb:,} kombinasi (kategori `Desserts` tidak dipakai)

### Representasi Solusi
Setiap solusi terdiri dari 3 bilangan bulat:
`x = [indeks_utama, indeks_lauk, indeks_minuman]`, dengan `indeks_lauk = 0` berarti **(tanpa lauk)**.

### Fungsi Objektif & Penanganan Batasan
Kedua fungsi objektif berikut **diminimalkan** secara bersamaan:
- **f1** = `|total kalori - target kalori|` (selisih kalori dalam kkal)
- **f2** = `max(0, target protein - protein) + max(0, target serat - serat)` (kekurangan protein dan serat dalam gram)

Batasan harian per satu kali makan (default: natrium ≤ 800 mg, gula ≤ 17 g, lemak jenuh ≤ 7 g)
ditangani menggunakan **fungsi penalti** yang dinormalisasi terhadap batas masing-masing:
- `pelanggaran = max(0, natrium - maks_natrium)/maks_natrium + max(0, gula - maks_gula)/maks_gula + max(0, lemak - maks_lemak)/maks_lemak`
- Nilai penalti ditambahkan sebesar `1000 × pelanggaran` pada `f1` dan `100 × pelanggaran` pada `f2`.
  Hanya solusi dengan `pelanggaran == 0` yang masuk ke Pareto front akhir; jika ada beberapa menu dengan nilai `f1` dan `f2` yang sama, semuanya tetap disimpan.

### Algoritma & Front Sejati
- **MOGA (Fonseca & Fleming, 1993)**:
  1. Setiap solusi diberi `rank = 1 + jumlah solusi lain yang mendominasinya` (non-dominasi = rank 1).
  2. Populasi diurutkan berdasarkan rank, lalu fitness diberikan secara linear dari `N` (terbaik) sampai `1` (terburuk), dan solusi dengan rank sama mendapat nilai rata-ratanya.
  3. *Fitness sharing* dilakukan di ruang objektif hanya antar solusi ber-rank sama menggunakan norma tak hingga pada objektif ternormalisasi dengan `σ = 2 / (N - 1)`, lalu diskalakan ulang agar total fitness tiap rank tetap.
  4. Induk dipilih dengan *stochastic universal sampling*, dikawinkan dengan *crossover uniform*, dan dimutasi; seluruh anak disimpan ke arsip yang dirapikan tiap 20 generasi.
- **MOPSO (Coello, Pulido & Lechuga, 2004)**:
  1. Ruang objektif dibagi menjadi grid (*hypercube* `30 × 30`). *Leader* dipilih dengan *roulette wheel* yang memberi peluang lebih besar pada *hypercube* yang lebih sepi.
  2. Kecepatan dan posisi partikel kontinu diperbarui dan dibulatkan ke indeks menu terdekat saat dievaluasi; partikel yang keluar batas ditahan di batas dan arah kecepatannya dibalik.
  3. Mutasi diberikan dengan peluang yang mengecil seiring iterasi (`pm = (1 - t/iterasi)^(5/laju_mutasi)`).
  4. Arsip eksternal (maksimal 100 solusi) diisi solusi non-dominasi; jika penuh, solusi dibuang secara acak dari *hypercube* terpadat. `pbest` diperbarui bila posisi baru mendominasi atau acak 50% bila saling non-dominasi.
- **Front Sejati (Brute Force)**: mengevaluasi seluruh {total_komb:,} kombinasi sebagai acuan *ground truth*.

### Titik Acuan, Metrik Evaluasi & Multi-skenario
- **Titik Acuan Dinamis**: ditentukan dari `np.maximum(2 * F_SEJATI.max(axis=0), 1.0)` berdasarkan Pareto front sejati dari konfigurasi yang sedang aktif (pada baseline default bernilai `[300, 12]`). Jika target atau batasan di sidebar diubah ataupun antar-skenario berbeda, titik acuan ikut menyesuaikan secara otomatis, termasuk untuk normalisasi metrik dan pilihan **Kompromi**.
- Semua metrik dihitung dari **titik objektif unik** (`np.unique`):
  - **Hypervolume (HV)**: luas daerah yang didominasi front terhadap titik acuan, dinormalisasi ke `[0, 1]` (makin besar makin baik).
  - **Spacing**: simpangan baku jarak tiap titik ke tetangga terdekatnya pada ruang ternormalisasi (makin kecil makin rata).
  - **IGD (Inverted Generational Distance)**: rata-rata jarak minimum dari tiap titik front sejati ke titik hasil algoritma (makin kecil makin dekat dan lengkap).
- **Uji Statistik & Effect Size**:
  - **Wilcoxon rank-sum test** pada nilai **HV** dan **IGD** dari 30 run independen (`α = 0.05`).
  - **Vargha-Delaney A12** untuk HV: mengukur peluang nilai HV MOGA lebih besar daripada HV MOPSO (`A12 > 0.5` menguntungkan MOGA; patokan besar efek: `0.56` kecil, `0.64` sedang, `0.71` besar).
- **Eksperimen Multi-skenario**: pengujian diulang pada 4 skenario (**S1 Baseline**, **S2 Batasan sedang**, **S3 Batasan ketat**, dan **S4 Target 800 kkal**) untuk memastikan kesimpulan konsisten di berbagai kondisi batasan dan target kalori, termasuk persentase run yang berhasil menemukan seluruh titik front sejati (**Lengkap**).

### Referensi
1. **Fonseca, C. M., & Fleming, P. J. (1993).** *Genetic Algorithms for Multiobjective Optimization: Formulation, Discussion and Generalization*. Proceedings of the Fifth International Conference on Genetic Algorithms (ICGA), 416–423.
2. **Coello, C. A. C., Pulido, G. T., & Lechuga, M. S. (2004).** *Handling multiple objectives with particle swarm optimization*. IEEE Transactions on Evolutionary Computation, 8(3), 256–279.
3. **Sahin, O., & Aytekin-Sahin, G. (2024).** *Open-source multi-objective optimization software for menu planning*. Expert Systems with Applications, 252, 124213.
4. **Ahourag, A., Elkari, B., El Moutaouakil, K., et al. (2024).** *A Multiobjective Diet Planning Model for Diabetic Patients in the Moroccan Health Context Using Particle Swarm Intelligence*. Statistics, Optimization and Information Computing, 12, 605–616.
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
