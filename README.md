# Optimasi Multi-Objektif Pemilihan Menu McDonald's (MOGA vs MOPSO)

Aplikasi demo Streamlit untuk proyek mata kuliah **Kecerdasan Komputasional**: mencari kombinasi menu McDonald's untuk satu kali makan (1 menu utama, maksimal 1 lauk, dan 1 minuman) menggunakan optimasi multi-objektif, serta membandingkan kinerja **MOGA** dan **MOPSO** terhadap **Pareto front sejati** (*brute force*).

## Struktur Proyek

```text
pareto-meal/
├── app.py                      # Entry point aplikasi Streamlit (3 tab)
├── menu.csv                    # Dataset McDonald's Nutrition Facts
├── requirements.txt            # Dependensi Python dengan versi terpin
├── data/
│   └── hasil_eksperimen.json   # Hasil precompute 30 run (MOGA vs MOPSO)
├── optim/
│   ├── __init__.py
│   ├── data.py                 # Pemuatan dan pengelompokan data menu
│   ├── model.py                # Konfigurasi, evaluasi, non-dominasi, brute force
│   ├── metrik.py               # Hypervolume, Spacing, IGD
│   └── algoritma.py            # MOGA, MOPSO, dan runner eksperimen
└── scripts/
    └── precompute.py           # Skrip untuk menghasilkan data/hasil_eksperimen.json
```

## Cara Menjalankan Lokal

1. Pastikan Python 3.10+ sudah terpasang.
2. (Opsional) Buat dan aktifkan virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
5. Buka URL lokal yang tampil di terminal (umumnya `http://localhost:8501`).

## Cara Regenerate `data/hasil_eksperimen.json`

File `data/hasil_eksperimen.json` sudah tersedia di repositori agar tab **Perbandingan Algoritma** langsung terbuka tanpa menunggu proses komputasi 30 run. Jika ingin menghasilkan ulang file tersebut:

```bash
python scripts/precompute.py
```

Skrip akan menjalankan brute force serta 30 run independen (`seed` 0–29) untuk MOGA dan MOPSO dengan konfigurasi default (target 600 kkal, protein 25 g, serat 8 g; batas natrium 800 mg, gula 17 g, lemak jenuh 7 g), lalu menimpa `data/hasil_eksperimen.json`.

## Langkah Deploy ke Streamlit Community Cloud (Repo Private)

1. Push seluruh repositori ini (termasuk `menu.csv`, `data/hasil_eksperimen.json`, dan `requirements.txt`) ke repositori GitHub **private** Anda.
2. Buka [share.streamlit.io](https://share.streamlit.io/) dan masuk menggunakan akun GitHub yang memiliki akses ke repositori tersebut.
3. Saat pertama kali menghubungkan repo private, klik **Authorize streamlit** / berikan izin akses ke repositori private pada GitHub OAuth settings bila diminta.
4. Klik **Create app** → pilih **Deploy a public app from GitHub** (atau pilih opsi deploy dari repo Anda).
5. Isi konfigurasi deployment:
   - **Repository**: `<username>/<nama-repo>`
   - **Branch**: `main`
   - **Main file path**: `app.py`
6. Klik **Deploy** dan tunggu hingga proses instalasi `requirements.txt` serta startup Streamlit selesai.
