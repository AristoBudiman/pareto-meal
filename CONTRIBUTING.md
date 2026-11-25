# Panduan Teknis Pengembang (CONTRIBUTING)

Catatan ini berisi alur kerja teknis, aturan perubahan kode, dan panduan deployment untuk anggota tim proyek.

## 1. Lingkungan dan Cara Menjalankan Lokal

- Versi Python yang digunakan adalah **Python 3.12** (disamakan dengan environment Streamlit Community Cloud).
- Pustaka yang terpasang ada di `requirements.txt` dengan versi terpin (`streamlit`, `pandas`, `numpy`, `scipy`, `matplotlib`).

Langkah menjalankan secara lokal:

1. Buat dan aktifkan virtual environment (opsional, disarankan):
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```
2. Pasang seluruh dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Buka alamat lokal yang muncul di terminal (biasanya `http://localhost:8501`).

## 2. Sumber Kebenaran Logika

- File `KK_Project_Final.ipynb` adalah **sumber kebenaran** untuk seluruh logika pemrosesan data, fungsi evaluasi, algoritma, metrik, skenario, dan cara penggunaan seed.
- Apabila ada perubahan pada logika di notebook, modul di folder `optim/` wajib ikut disamakan tanpa mengubah rumus atau perilaku aslinya:
  - `optim/data.py`: pemuatan `menu.csv` dan pengelompokan slot menu.
  - `optim/model.py`: `Konfigurasi`, fungsi `evaluasi`, `non_dominasi`, `ambil_pareto`, `brute_force`, dan `pilih_rekomendasi`.
  - `optim/metrik.py`: `hitung_titik_acuan`, `hypervolume`, `spacing`, `igd`, dan `a12`.
  - `optim/algoritma.py`: `moga`, `mopso`, `kotak_grid`, `jalankan_eksperimen`, dan `jalankan_multi_skenario`.

## 3. Regenerate `data/hasil_eksperimen.json`

File `data/hasil_eksperimen.json` menyimpan hasil komputasi 30 run untuk eksperimen utama dan 4 skenario (`S1`–`S4`) agar tab **Perbandingan Algoritma** di Streamlit langsung terbuka tanpa menunggu proses komputasi ulang setiap kali halaman dibuka.

Jalankan perintah berikut untuk membangkitkan ulang file JSON tersebut:

```bash
python scripts/precompute.py
```

Script ini **wajib dijalankan ulang** (dan hasilnya di-commit) setiap kali ada perubahan pada:
- implementasi atau logika algoritma (`moga` / `mopso`)
- parameter algoritma atau jumlah run
- nilai target atau batasan gizi default (`Konfigurasi`)
- daftar skenario pada `SKENARIO_DEFAULT`
- cara perhitungan titik acuan atau metrik evaluasi (`HV`, `Spacing`, `IGD`, `A12`)

## 4. Alur Kerja Git dan Aturan Tim

- Selalu jalankan `git pull` sebelum mulai mengubah kode agar tidak terjadi konflik dengan perubahan anggota tim lain.
- Sebelum melakukan `git push`, uji aplikasi secara lokal dengan `streamlit run app.py` dan pastikan:
  1. Ketiga tab (**Rekomendasi Menu**, **Perbandingan Algoritma**, dan **Tentang**) terbuka tanpa error.
  2. Pencarian menu berjalan normal, termasuk saat batasan dibuat sangat ketat hingga tidak ada kombinasi yang lolos (harus tampil pesan peringatan, bukan traceback error).
- Aturan kode dan eksperimen:
  - Jangan menambah library baru ke `requirements.txt` tanpa kesepakatan tim.
  - Tidak ada tuning parameter otomatis (misalnya Optuna); gunakan parameter default sesuai `KK_Project_Final.ipynb`.
  - Parameter eksperimen (`N`, jumlah iterasi/generasi) dan `seed` (`0`–`29`) harus identik untuk MOGA dan MOPSO agar perbandingan tetap adil.

## 5. Deploy dan Reboot di Streamlit Community Cloud

Aplikasi dideploy ke Streamlit Community Cloud dari repositori GitHub private dengan konfigurasi berikut:

- **Main file path**: `app.py`
- **Python version**: `3.12`
- **Branch**: `main`

Langkah deploy awal:
1. Pastikan seluruh perubahan terbaru (termasuk `menu.csv`, `data/hasil_eksperimen.json`, dan `requirements.txt`) sudah di-push ke branch `main`.
2. Buka [share.streamlit.io](https://share.streamlit.io/) dan masuk menggunakan akun GitHub yang memiliki akses ke repositori tersebut.
3. Izinkan akses repositori private pada pengaturan OAuth GitHub bila diminta.
4. Klik **Create app**, pilih repositori, branch `main`, main file `app.py`, lalu pilih **Python 3.12** di pengaturan lanjutan (*Advanced settings*) sebelum klik **Deploy**.

Cara reboot aplikasi di Streamlit Community Cloud:
- Saat ada pembaruan `data/hasil_eksperimen.json` atau perubahan `requirements.txt` yang perlu memuat ulang cache atau environment server, buka dashboard app di Streamlit Community Cloud, klik menu titik tiga (**⋮**) atau **Manage app** di pojok kanan bawah, lalu pilih **Reboot app** (atau **Clear cache** jika hanya perlu membersihkan `@st.cache_data`).
