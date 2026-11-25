[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pareto-meal-project-kk.streamlit.app)

# Optimasi Multi-Objektif Pemilihan Menu McDonald's: Perbandingan MOGA dan MOPSO

Proyek ini disusun oleh Kelompok 1 untuk mata kuliah Kecerdasan Komputasional, Program Studi Informatika, Universitas Sebelas Maret (UNS). Proyek ini memodelkan pemilihan kombinasi menu McDonald's untuk satu kali makan sebagai masalah optimasi multi-objektif dan membandingkan kinerja algoritma MOGA dan MOPSO terhadap Pareto front sejati. Hasil optimasi dapat dicoba langsung melalui antarmuka web Streamlit.

## Latar Belakang

Kebiasaan makan di luar rumah sering menjadi tantangan bagi pelaku diet karena makanan cepat saji umumnya padat kalori, tinggi natrium, tinggi gula, dan tinggi lemak jenuh. Di sisi lain, ada kalanya pelaku diet tetap makan di restoran cepat saji karena keterbatasan waktu atau situasi sosial, sehingga dibutuhkan pilihan menu yang paling aman dengan pendekatan *harm reduction*. Masalahnya, dua tujuan nutrisi saling bertentangan: kombinasi menu yang mendekati target kalori sering kali kekurangan protein dan serat, sedangkan menu yang mencukupi protein dan serat cenderung melebihi target kalori.

## Model Masalah

Dataset yang digunakan adalah *McDonald's Nutrition Facts* dari Kaggle yang berisi 260 item menu beserta informasi gizinya. Item dikelompokkan menjadi tiga slot sesuai kategori dataset, tanpa membuang porsi besar di awal agar penyaringan murni dilakukan oleh batasan gizi (kategori *Desserts* tidak digunakan):

- **Menu utama**: 90 item (`Breakfast`, `Beef & Pork`, `Chicken & Fish`, `Salads`)
- **Lauk pendamping**: 14 item (`Snacks & Sides`, ditambah satu opsi `(tanpa lauk)` bernilai gizi 0 pada indeks 0)
- **Minuman**: 150 item (`Beverages`, `Coffee & Tea`, `Smoothies & Shakes`)

Total ruang pencarian terdiri dari `90 × 14 × 150 = 189.000` kombinasi. Setiap kandidat solusi direpresentasikan sebagai vektor integer 3 gen `x = [indeks_utama, indeks_lauk, indeks_minuman]`. Kedua fungsi objektif diminimalkan secara bersamaan:

- **f1** = `|total kalori - target kalori|` (selisih kalori dalam kkal)
- **f2** = `max(0, target protein - total protein) + max(0, target serat - total serat)` (kekurangan protein dan serat dalam gram)

Batasan nutrisi per satu kali makan diterapkan pada natrium (default ≤ 800 mg), gula (≤ 17 g), dan lemak jenuh (≤ 7 g). Pelanggaran terhadap ketiga batas tersebut dinormalisasi terhadap masing-masing batasnya lalu ditambahkan sebagai fungsi penalti sebesar `1000 × pelanggaran` pada `f1` dan `100 × pelanggaran` pada `f2`, sehingga solusi yang melanggar selalu kalah dari solusi yang memenuhi seluruh batasan.

## Metode

- **Brute Force**: mengevaluasi seluruh 189.000 kombinasi dan menyaring solusi non-dominasi yang lolos batasan untuk memperoleh Pareto front sejati sebagai acuan (*ground truth*).
- **MOGA (Fonseca & Fleming, 1993)**: menggunakan *Pareto ranking*, pemberian fitness linear yang dirata-ratakan per rank, *fitness sharing* di antara solusi ber-rank sama dengan norma tak hingga pada objektif ternormalisasi (`σ = 2 / (N - 1)`), seleksi *stochastic universal sampling*, *crossover uniform*, mutasi, dan arsip solusi.
- **MOPSO (Coello, Pulido & Lechuga, 2004)**: membagi ruang objektif menjadi grid *hypercube* (`30 × 30`), memilih *leader* dengan *roulette wheel* yang memprioritaskan *hypercube* lebih sepi, memperbarui kecepatan dan posisi partikel kontinu yang dibulatkan saat evaluasi, membalik kecepatan di batas, menerapkan mutasi menurun, serta mengelola arsip eksternal berkapasitas 100 solusi.

## Evaluasi

Kedua algoritma dibandingkan pada 30 run independen dengan seed `0` sampai `29` menggunakan parameter populasi dan iterasi yang setara (`N = 50`, `400` generasi/iterasi). Metrik dihitung dari titik objektif unik terhadap titik acuan dinamis `2 × nilai terburuk front sejati` (minimal `1,0`):

- **Hypervolume (HV)**: luas area yang didominasi front hingga titik acuan, dinormalisasi ke `[0, 1]`.
- **Spacing**: simpangan baku jarak antar titik ke tetangga terdekatnya pada ruang ternormalisasi.
- **IGD (Inverted Generational Distance)**: rata-rata jarak minimum dari setiap titik front sejati ke titik hasil algoritma.
- **Uji statistik dan effect size**: uji Wilcoxon rank-sum pada taraf signifikansi `α = 0,05` untuk HV dan IGD, serta ukuran efek Vargha-Delaney **A12** untuk HV.

## Hasil Ringkas

Pada konfigurasi default (target 600 kkal, protein 25 g, serat 8 g; batas natrium 800 mg, gula 17 g, lemak jenuh 7 g), terdapat **1.262** kombinasi lolos batasan (`0,67%`) dengan front sejati sebanyak **4 titik (5 menu)**, titik acuan `[300, 12]`, HV front sejati `0,9000`, dan Spacing `0,2000`.

### Eksperimen Utama (30 Run)

| Metrik | MOGA | MOPSO |
| --- | --- | --- |
| HV | 0.8744 ± 0.0393 | 0.7653 ± 0.1057 |
| Spacing | 0.2056 ± 0.0702 | 0.0568 ± 0.0957 |
| IGD | 0.0360 ± 0.0328 | 0.1295 ± 0.0537 |
| Jumlah titik | 3.5000 ± 0.5627 | 2.3000 ± 0.4583 |
| Titik sejati ditemukan per run (0 / 1 / 2 / 3 / 4) | 1 / 1 / 9 / 10 / 9 | 1 / 14 / 8 / 7 / 0 |

Uji Wilcoxon rank-sum menunjukkan perbedaan signifikan antara MOGA dan MOPSO baik pada HV (`p = 0,0000`) maupun IGD (`p = 0,0000`).

### Rekap Multi-Skenario

| Skenario | Lolos | Front sejati | HV sejati | HV MOGA | HV MOPSO | Lengkap MOGA | Lengkap MOPSO | A12 HV | Unggul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 Baseline | 1262 | 4 | 0.9000 | 0.8744 ± 0.0393 | 0.7653 ± 0.1057 | 30% | 0% | 0.84 | MOGA |
| S2 Batasan sedang | 747 | 4 | 0.8547 | 0.8356 ± 0.0200 | 0.8072 ± 0.0316 | 37% | 3% | 0.79 | MOGA |
| S3 Batasan ketat | 493 | 3 | 0.5077 | 0.5069 ± 0.0027 | 0.4625 ± 0.0406 | 90% | 3% | 0.98 | MOGA |
| S4 Target 800 kkal | 1262 | 5 | 0.6868 | 0.6740 ± 0.0179 | 0.6014 ± 0.0833 | 17% | 0% | 0.85 | MOGA |

Secara keseluruhan, **MOGA unggul di 4 dari 4 skenario** (`p < 0,05` dan `A12` di rentang `0,79–0,98`).

## Struktur Folder

```text
pareto-meal/
├── app.py                      # Aplikasi web Streamlit
├── KK_Project_Final.ipynb      # Notebook utama eksperimen dan sumber logika
├── menu.csv                    # Dataset McDonald's Nutrition Facts
├── requirements.txt            # Daftar pustaka Python
├── CONTRIBUTING.md             # Panduan teknis untuk anggota tim
├── data/
│   └── hasil_eksperimen.json   # Hasil precompute eksperimen utama dan multi-skenario
├── optim/
│   ├── __init__.py
│   ├── data.py                 # Pemuatan dan pengelompokan data menu
│   ├── model.py                # Evaluasi, non-dominasi, brute force, rekomendasi
│   ├── metrik.py               # Titik acuan dinamis, HV, Spacing, IGD, A12
│   └── algoritma.py            # MOGA, MOPSO, runner eksperimen dan skenario
└── scripts/
    └── precompute.py           # Skrip pembangkit data/hasil_eksperimen.json
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Panduan teknis pengembangan, pembaruan hasil precompute, dan deployment tersedia di [CONTRIBUTING.md](CONTRIBUTING.md).

## Referensi

1. Sahin, O., & Aytekin-Sahin, G. (2024). Open-source multi-objective optimization software for menu planning. *Expert Systems with Applications*, 252, 124213.
2. Ahourag, A., Elkari, B., El Moutaouakil, K., et al. (2024). A Multiobjective Diet Planning Model for Diabetic Patients in the Moroccan Health Context Using Particle Swarm Intelligence. *Statistics, Optimization and Information Computing*, 12, 605–616.
3. Fonseca, C. M., & Fleming, P. J. (1993). Genetic Algorithms for Multiobjective Optimization: Formulation, Discussion and Generalization. *Proceedings of the Fifth International Conference on Genetic Algorithms (ICGA)*, 416–423.
4. Coello, C. A. C., Pulido, G. T., & Lechuga, M. S. (2004). Handling multiple objectives with particle swarm optimization. *IEEE Transactions on Evolutionary Computation*, 8(3), 256–279.
5. Vargha, A., & Delaney, H. D. (2000). A Critique and Improvement of the CL Common Language Effect Size Statistics of McGraw and Wong. *Journal of Educational and Behavioral Statistics*, 25(2), 101–132.
6. McDonald's. *Nutrition Facts for McDonald's Menu* [Dataset]. Kaggle. https://www.kaggle.com/datasets/mcdonalds/nutrition-facts
