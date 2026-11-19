# Laporan Komprehensif Hasil Data & Eksperimen LMS Testbed

> **Catatan Metodologis**: Seluruh angka dan metrik dalam dokumen ini diekstrak langsung dari eksekusi pipeline pengujian pada dataset aktual LMS/Student Portal Google Play (*lms_testbed*). Angka disajikan persis sebagaimana dihasilkan oleh sistem (tanpa pembulatan manual yang mengubah makna).

## 0. Pengaturan Saat Menjalankan Hasil ⭐

Konfigurasi parameter sistem pada saat eksperimen dijalankan:

| Parameter | Nilai / Pengaturan | Penjelasan / Keterangan |
|---|---|---|
| **Ambang Rating Negatif** | $\le 2$ | Rating bintang 1 dan 2 dikategorikan sebagai sentimen `negative` |
| **Ambang Rating Positif** | $\ge 4$ | Rating bintang 4 dan 5 dikategorikan sebagai sentimen `positive` |
| **Rating di Tengah (3)** | Dianggap `neutral` | **Tidak dibuang**, melainkan dipertahankan sebagai kelas ketiga dalam klasifikasi multi-kelas (*negative*, *neutral*, *positive*) |
| **Opsi Preprocessing Default** | `A - Minimal (aman)` | Huruf kecil, normalisasi URL & emoji ke token, pemangkasan repetisi karakter. Negasi dan istilah teknis dipertahankan |
| **Feature Scaler** | `StandardScaler` | Diterapkan pada fitur numerik (`StandardScaler`), di-fit hanya pada data train di dalam pipeline |
| **Test Size (Holdout)** | 0.2 (20%) | Porsi data pengujian akhir |
| **Validation Size** | 0.15 (15%) | Porsi data validasi dari data latih non-test |
| **Jumlah Fold CV** | 5 | 5-Fold Stratified Cross-Validation pada data latih (train+val) |
| **Strategi Split Utama** | `Stratified random (deduplikasi teks)` | Stratified random split setelah deduplikasi teks (`clean_min`) |
| **Random Seed** | 42 | Memastikan seluruh split dan inisialisasi model dapat direplikasi (*reproducible*) |
| **Filter Aktif** | Tidak ada (semua data digunakan) | Seluruh data (100%) dipakai tanpa pemotongan institusi, aplikasi, bahasa, rating, maupun tanggal |
| **Versi Python** | `3.13.14` | AMD64 (`3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]`) |
| **Versi scikit-learn** | `1.9.1` | Library utama modeling, metriks, dan pipeline |
| **Versi Library Lain** | `pandas 3.1.1`, `scipy 1.18.1`, `streamlit 1.67.0` | Lingkungan eksekusi virtual environment (`.venv`) |

## 1. Overview dan Data ⭐

### Ringkasan Dimensi dan Missing Values
- **Dataset Mentah Excel (`[Data Set] Combined LMS App Review.xlsx`)**: 5,259 baris × 13 kolom.
  - *Missing values per kolom*: 
    - `institution_name`: 0 (lengkap)
    - `app_name`: 0 (lengkap)
    - `play_url`: 0 (lengkap)
    - `package_id`: 0 (lengkap)
    - `reviewer_name`: 0 (lengkap)
    - `review_title`: 5,259 baris kosong (100.00%)
    - `review_body`: 2 baris kosong (0.04%)
    - `star_rating`: 0 (lengkap)
    - `thumbs_up_count`: 0 (lengkap)
    - `review_created_at`: 0 (lengkap)
    - `app_version`: 1,038 baris kosong (19.74%)
    - `language`: 0 (lengkap)
    - `app_scope_flag`: 0 (lengkap)
- **Dataset Bersih CSV (`lms_reviews_clean.csv`)**: 5,257 baris × 14 kolom.
  - *Missing values per kolom*: 
    - `institution_name`: 0 (lengkap)
    - `app_name`: 0 (lengkap)
    - `play_url`: 0 (lengkap)
    - `package_id`: 0 (lengkap)
    - `reviewer_name`: 0 (lengkap)
    - `review_title`: 5,257 baris kosong (100.00%)
    - `review_body`: 0 (lengkap)
    - `star_rating`: 0 (lengkap)
    - `thumbs_up_count`: 0 (lengkap)
    - `review_created_at`: 0 (lengkap)
    - `app_version`: 1,037 baris kosong (19.73%)
    - `language`: 0 (lengkap)
    - `app_scope_flag`: 0 (lengkap)
    - `sentiment`: 0 (lengkap)

### Penjelasan Selisih 5.259 vs 5.257 Baris
- Pada file mentah Excel (`[Data Set] Combined LMS App Review.xlsx`), terdapat **5.259 baris**.
- Terdapat **2 baris** yang memiliki nilai kosong / `NaN` pada kolom ulasan (`review_body`), yaitu:
  1. Baris index **5077**: Institusi *University of Leeds*, ulasan tidak berisi teks (`review_body = NaN`).
  2. Baris index **5109**: Institusi *University of Michigan-Ann Arbor*, ulasan tidak berisi teks (`review_body = NaN`).
- Ketika skrip pra-pemrosesan (`preprocess.py`) dijalankan, baris kosong tersebut dihapus melalui `df = df.dropna(subset=['review_body'])`.
- Oleh karena itu, file bersih yang dihasilkan (`lms_reviews_clean.csv`) berjumlah tepat **5.257 baris** (selisih 2 baris).

### Jumlah Ulasan, Institusi, dan Aplikasi per Kelompok
| Kelompok (*Group*) | Jumlah Ulasan | Persentase | Jumlah Institusi | Jumlah Aplikasi | Rata-rata Rating | Median Rating |
|---|---|---|---|---|---|---|
| **Indonesian app** | 3,682 | 70.04% | 18 | 23 | 3.316 | 4.0 |
| **Top 100 world app (English-speaking country)** | 1,575 | 29.96% | 31 | 31 | 2.452 | 2.0 |
| **Total Gabungan** | **5,257** | **100.00%** | **49** | **54** | **3.057** | **3.0** |

### Distribusi Bahasa per Kelompok
#### a. Berdasarkan Kolom Metadata Dataset (`language`)
| Kelompok | Bahasa Indonesia (`id`) | Bahasa Inggris (`en`) | Total |
|---|---|---|---|
| **Indonesian app** | 2,303 (62.55%) | 1,379 (37.45%) | 3,682 |
| **Top 100 world app (English-speaking country)** | 13 (0.83%) | 1,562 (99.17%) | 1,575 |

#### b. Berdasarkan Deteksi Heuristik Leksikon Ringan (`lang_detected`)
| Kelompok | Indonesia (`id`) | Inggris (`en`) | Campuran (`mixed`) | Tidak Teridentifikasi (`unknown`) | Total |
|---|---|---|---|---|---|
| **Indonesian app** | 2,145 (58.26%) | 386 (10.48%) | 91 (2.47%) | 1,060 (28.79%) | 3,682 |
| **Top 100 world app (English-speaking country)** | 3 (0.19%) | 1,341 (85.14%) | 0 (0.00%) | 231 (14.67%) | 1,575 |

### Rentang Tanggal Ulasan
- **Ulasan Paling Awal**: `2011-02-12 06:14:22.999999945`
- **Ulasan Paling Akhir**: `2026-04-23 03:04:52.999999973`
- **Rentang Waktu**: Sekitar 13 tahun data histori ulasan Google Play Store.

### Deduplikasi dan Partisi Data Modeling
- **Data berlabel & non-kosong**: 5,257 baris.
- **Duplikat teks dibuang (`clean_min`)**: 615 baris.
- **Data final siap dimodelkan**: **4,642 baris unik**.

**Distribusi Jumlah Baris per Split dan per Kelas Sentimen:**
| Split Set | Total Baris | Negative (Rating 1-2) | Neutral (Rating 3) | Positive (Rating 4-5) |
|---|---|---|---|---|
| **Train** | 3,016 (65.0%) | 1,457 (48.3%) | 309 (10.2%) | 1,250 (41.4%) |
| **Val** | 697 (15.0%) | 337 (48.4%) | 71 (10.2%) | 289 (41.5%) |
| **Test** | 929 (20.0%) | 449 (48.3%) | 95 (10.2%) | 385 (41.4%) |
| **Total** | **4,642** | **2,243** | **475** | **1,924** |

## 2. Exploratory Data Analysis (EDA) ⭐

### Distribusi Rating Bintang (1–5) per Kelompok
| Kelompok | Bintang 1 | Bintang 2 | Bintang 3 | Bintang 4 | Bintang 5 | Mean Rating | Median Rating |
|---|---|---|---|---|---|---|---|
| **Indonesian app** | 1,107 (30.07%) | 267 (7.25%) | 332 (9.02%) | 306 (8.31%) | 1,670 (45.36%) | 3.316 | 4.0 |
| **Top 100 world app (English-speaking country)** | 777 (49.33%) | 161 (10.22%) | 150 (9.52%) | 122 (7.75%) | 365 (23.17%) | 2.452 | 2.0 |

### Distribusi Sentimen Proxy per Kelompok
| Kelompok | Negative (≤ 2) | Neutral (= 3) | Positive (≥ 4) | Total Ulasan |
|---|---|---|---|---|
| **Indonesian app** | 1,374 (37.32%) | 332 (9.02%) | 1,976 (53.67%) | 3,682 |
| **Top 100 world app (English-speaking country)** | 938 (59.56%) | 150 (9.52%) | 487 (30.92%) | 1,575 |

### Hasil Uji Hipotesis Statistik
#### a. Uji Kruskal-Wallis (Variabel Numerik terhadap Target Sentimen `sent_label`)
| Variabel Numerik | Statistik H | p-value | Signifikansi ($\alpha=0.05$) | Interpretasi |
|---|---|---|---|---|
| **Panjang Teks (`n_chars`)** | 635.12 | 1.2157e-138 | **Signifikan** | Ulasan negatif secara signifikan memiliki teks karakter lebih panjang |
| **Jumlah Kata (`n_tokens`)** | 675.55 | 2.0199e-147 | **Signifikan** | Ulasan negatif memiliki jumlah kata lebih banyak (deskripsi komplain/bug) |
| **Jumlah Emoji (`n_emoji`)** | 36.30 | 1.3124e-08 | **Signifikan** | Jumlah emoji berbeda nyata antar kelas sentimen |
| **Jumlah Tanda Seru (`n_exclaim`)** | 5.66 | 0.0591 | Tidak Signifikan ($p > 0.05$) | Penggunaan tanda seru relatif merata di ekspresi negatif maupun antusiasme positif |
| **Thumbs-Up / Likes (`thumbs_up_count`)** | 301.91 | 2.7629e-66 | **Signifikan** | Ulasan berlabel negatif mendapat dukungan *thumbs up* pembaca jauh lebih tinggi |
| **Rating Bintang (`rating`)** | 4769.30 | 0.0000e+00 | **Deterministik** | Bersifat 'sempurna' secara definisi karena sentimen adalah turunan fungsi rating (sumber label proxy) |

#### b. Uji Chi-Square (Variabel Kategorikal terhadap Target Sentimen `sent_label`)
| Variabel Kategorikal | Statistik $\chi^2$ | Derajat Bebas (*df*) | p-value | Cramér's V | Signifikansi & Kekuatan Asosiasi |
|---|---|---|---|---|---|
| **group** | 246.18 | 2 | 3.4945e-54 | 0.216 | **Signifikan** (Asosiasi sedang: $V=0.22$) |
| **language** | 174.54 | 2 | 1.2556e-38 | 0.182 | **Signifikan** (Asosiasi sedang: $V=0.18$) |
| **lang_detected** | 256.44 | 6 | 1.7248e-52 | 0.156 | **Signifikan** (Asosiasi sedang: $V=0.16$) |
| **institution** | 843.31 | 24 | 1.4563e-162 | 0.283 | **Signifikan** (Asosiasi sedang: $V=0.28$) |
| **app** | 847.19 | 24 | 2.1932e-163 | 0.284 | **Signifikan** (Asosiasi sedang: $V=0.28$) |
| **aspect_primary** | 992.15 | 14 | 7.5672e-203 | 0.307 | **Signifikan** (Asosiasi sedang: $V=0.31$) |

### Ringkasan Panjang Ulasan dan Thumbs-Up (Likes) per Kelompok
| Kelompok | Metrik | Mean | Median | Std Dev | Min | Max |
|---|---|---|---|---|---|---|
| **Indonesian app** | Karakter (`n_chars`) | 69.426 | 39.5 | 83.27 | 1 | 711 |
| | Kata (`n_tokens`) | 11.193 | 7.0 | 13.174 | 1 | 109 |
| **Top 100 world app (English-speaking country)** | Karakter (`n_chars`) | 101.296 | 68.0 | 103.092 | 1 | 822 |
| | Kata (`n_tokens`) | 18.573 | 12.0 | 19.013 | 1 | 152 |
| **Indonesian app** | Thumbs-up (`thumbs_up_count`) | 2.517 | 1.0 | 6.841 | 0 | 160 |
| **Top 100 world app** | Thumbs-up (`thumbs_up_count`) | 2.543 | 1.0 | 4.877 | 0 | 39 |

### Aplikasi dengan Rating Tertinggi dan Terendah per Kelompok (Minimal 10 Ulasan)
- **Indonesian app**:
  - *Rating Tertinggi*: **UP Mobile for Student** (Rata-rata: 5.00 bintang, $n=14$ ulasan)
  - *Rating Terendah*: **UMY Student** (Rata-rata: 1.27 bintang, $n=15$ ulasan)
- **Top 100 world app (English-speaking country)**:
  - *Rating Tertinggi*: **Monash Study** (Rata-rata: 4.03 bintang, $n=30$ ulasan)
  - *Rating Terendah*: **University of Bristol** (Rata-rata: 1.38 bintang, $n=21$ ulasan)

## 3. Preprocessing

### Perbandingan Tiga Opsi Preprocessing (Model: Logistic Regression, Test Set Holdout) ⭐
| Opsi Preprocessing | Akurasi | Macro-F1 | Jumlah Fitur | Dokumen Kosong Pasca-Clean | Karakteristik Utama |
|---|---|---|---|---|---|
| **A - Minimal (aman)** | 0.7309 | 0.6118 | 8,780 | 0 | Minimal aman: negasi & emoji utuh |
| **B - Normalisasi + tag negasi** | 0.7460 | 0.6448 | 8,930 | 0 | Normalisasi slang ID + tag negasi 2 kata |
| **C - Agresif (stopword removal)** | 0.7349 | 0.6176 | 6,049 | 3 | Minimal aman: negasi & emoji utuh |

> **Analisis Hasil**: Opsi B (**Normalisasi + tag negasi**) menghasilkan Macro-F1 tertinggi (**0.6448**) karena berhasil menormalisasi variasi slang bahasa Indonesia (mis. *gk*, *lemot*, *eror*) sekaligus membubuhkan tag `NEG_` pada token setelah kata negasi. Opsi C (**Agresif**) memangkas stopword termasuk kata penting negasi (*tidak*, *not*), yang menyebabkan penurunan performa dan menghasilkan ulasan kosong.

### Hasil Cek Korelasi dan Indikasi Kebocoran Data (Data Leakage) ⭐
| Nama Fitur | Korelasi Spearman dengan Label ($r_s$) | $\|r_s\|$ | Status Kebocoran | Keterangan Metodologis |
|---|---|---|---|---|
| `n_chars` | -0.1965 | 0.1965 | **Wajar** (Aman dipakai) | Statistik teks sah turunan review body |
| `n_tokens` | -0.2143 | 0.2143 | **Wajar** (Aman dipakai) | Statistik teks sah turunan review body |
| `n_emoji` | 0.0940 | 0.0940 | **Wajar** (Aman dipakai) | Statistik teks sah turunan review body |
| `n_exclaim` | 0.0416 | 0.0416 | **Wajar** (Aman dipakai) | Statistik teks sah turunan review body |
| `rating` | 0.9508 | 0.9508 | ⚠ **Sangat Tinggi (LEAKAGE FATAL)** | Rating merupakan formula pembentuk label sentimen proxy |
| `likes` (`thumbs_up_count`) | -0.2376 | 0.2376 | **Wajar** | Fitur meta-ulasan sah (jika di-mapping) |

> **Temuan Kunci Kebocoran**: Korelasi fitur `rating` terhadap label adalah **0.9508** (mendekati 1.0). Mengikutsertakan rating ke dalam model sebagai fitur (seperti pada Skenario B) merupakan bentuk kebocoran data (*target leakage*), di mana model tidak belajar memahami teks melainkan hanya membaca rating.

### Efek Feature Scaler pada Fitur Numerik
Penerapan `StandardScaler` mengubah distribusi fitur numerik menjadi rata-rata $\approx 0$ dan varians $\approx 1$:
| Fitur | Nilai Asli (Mean ± Std) | Nilai Asli (Min – Max) | Nilai Scaled (Mean ± Std) | Nilai Scaled (Min – Max) |
|---|---|---|---|---|
| `n_chars` | 88.53 ± 92.54 | 1 – 822 | 0.000 ± 1.000 | -0.946 – 7.927 |
| `n_tokens` | 15.00 ± 15.85 | 1 – 152 | 0.000 ± 1.000 | -0.883 – 8.642 |
| `n_emoji` | 0.12 ± 0.61 | 0 – 12 | -0.000 ± 1.000 | -0.199 – 19.411 |
| `n_exclaim` | 0.10 ± 0.55 | 0 – 15 | 0.000 ± 1.000 | -0.177 – 27.326 |

### Contoh Teks Sebelum dan Sesudah Pembersihan (Sample Showcase)
| No | Bahasa | Teks Asli | Opsi A (Minimal) | Opsi B (Normalisasi + Tag Negasi) | Opsi C (Agresif / Stopword) |
|---|---|---|---|---|---|
| 1 | `en` | "Should not have launched the new app until functionality was on par wi..." | "should not have launched the new app until functionality was..." | "should not NEG_have NEG_launched the new app until functiona..." | "should have launched new app until functionality par old app..." |
| 2 | `id` | "User interface baru, lebih memudahkan untuk mengakses ke tiap tiap men..." | "user interface baru lebih memudahkan untuk mengakses ke tiap..." | "user interface baru lebih memudahkan untuk mengakses ke tiap..." | "user interface baru lebih memudahkan mengakses tiap tiap men..." |
| 3 | `id` | "Tolong beri fitur ZOOM pada scan kehadiran mahasiswa agar presensi tid..." | "tolong beri fitur zoom pada scan kehadiran mahasiswa agar pr..." | "tolong beri fitur zoom pada scan kehadiran mahasiswa agar pr..." | "tolong beri fitur zoom scan kehadiran mahasiswa presensi har..." |
| 4 | `id` | "Saya tidak bisa login" | "saya tidak bisa login" | "saya tidak NEG_bisa NEG_login" | "bisa login" |
| 5 | `id` | "Belum ada versi iosnya" | "belum ada versi iosnya" | "belum NEG_ada NEG_versi iosnya" | "versi iosnya" |

## 4. Modeling (Leaderboard & Metrik) ⭐

Eksperimen model dilakukan pada **Skenario A (Bebas Leakage - Hanya Teks)** dengan *Stratified Random Split* (seed=42):

### Tabel Leaderboard Lengkap
| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | ROC-AUC | F1 Val | 5-Fold CV Accuracy | 5-Fold CV Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| **Baseline (majority class)** | 0.4833 | 0.2172 | 0.1611 | 0.3333 | 0.5000 | 0.2173 | 0.483 ± 0.000 | 0.217 ± 0.000 |
| **TF-IDF + Logistic Regression** | 0.7330 | 0.6090 | 0.6080 | 0.6105 | 0.8326 | 0.5880 | 0.729 ± 0.022 | 0.614 ± 0.027 |
| **TF-IDF + Linear SVM** | 0.7557 | 0.5922 | 0.6006 | 0.5940 | 0.8098 | 0.5555 | 0.746 ± 0.012 | 0.593 ± 0.014 |
| **TF-IDF + Complement NB** | 0.7320 | 0.5240 | 0.5201 | 0.5426 | 0.7852 | 0.5420 | 0.731 ± 0.015 | 0.538 ± 0.014 |
| **TF-IDF word+char n-gram + Logistic Regression** | 0.7417 | 0.6265 | 0.6246 | 0.6300 | 0.8449 | 0.5940 | 0.734 ± 0.025 | 0.613 ± 0.030 |

### Performa Presisi, Recall, dan F1 per Kelas (Test Set Holdout)
| Model | Kelas Sentimen | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|---|
| **Baseline (majority class)** | | | | | |
| | `negative` | 0.4833 | 1.0000 | 0.6517 | 449 |
| | `neutral` | 0.0000 | 0.0000 | 0.0000 | 95 |
| | `positive` | 0.0000 | 0.0000 | 0.0000 | 385 |
| **TF-IDF + Logistic Regression** | | | | | |
| | `negative` | 0.8005 | 0.7684 | 0.7841 | 449 |
| | `neutral` | 0.2376 | 0.2526 | 0.2449 | 95 |
| | `positive` | 0.7859 | 0.8104 | 0.7980 | 385 |
| **TF-IDF + Linear SVM** | | | | | |
| | `negative` | 0.7868 | 0.8218 | 0.8039 | 449 |
| | `neutral` | 0.2264 | 0.1263 | 0.1622 | 95 |
| | `positive` | 0.7887 | 0.8338 | 0.8106 | 385 |
| **TF-IDF + Complement NB** | | | | | |
| | `negative` | 0.7169 | 0.8797 | 0.7900 | 449 |
| | `neutral` | 0.0500 | 0.0105 | 0.0174 | 95 |
| | `positive` | 0.7933 | 0.7377 | 0.7645 | 385 |
| **TF-IDF word+char n-gram + Logistic Regression** | | | | | |
| | `negative` | 0.8121 | 0.7795 | 0.7955 | 449 |
| | `neutral` | 0.2566 | 0.3053 | 0.2788 | 95 |
| | `positive` | 0.8052 | 0.8052 | 0.8052 | 385 |

### Hyperparameter Penting Setiap Model
1. **TF-IDF Vectorizer (Word)**:
   - `ngram_range = (1, 2)` (Unigram dan Bigram kata)
   - `min_df = 2`, `sublinear_tf = True`, `max_features = 40,000`
   - `token_pattern = r"(?u)\b[\w']+\b"`
2. **TF-IDF Vectorizer (Char)** (pada model word+char):
   - `analyzer = 'char_wb'`, `ngram_range = (2, 5)` (N-gram karakter dalam batas kata)
   - `min_df = 2`, `sublinear_tf = True`, `max_features = 60,000`
3. **Logistic Regression**:
   - `C = 1.0`, `solver = 'lbfgs'`, `max_iter = 1000`, `class_weight = 'balanced'`, `random_state = 42`
4. **Linear Support Vector Classifier (Linear SVM)**:
   - `C = 1.0`, `loss = 'squared_hinge'`, `class_weight = 'balanced'`, `random_state = 42`
5. **Complement Naive Bayes (Complement NB)**:
   - `norm = False`, dirancang khusus untuk dataset teks yang tidak seimbang (*imbalanced text classification*)
6. **Majority Baseline (Dummy Classifier)**:
   - `strategy = 'most_frequent'` (selalu memprediksi kelas mayoritas `negative`)

## 5. Evaluation ⭐

### Model Terbaik: `TF-IDF word+char n-gram + Logistic Regression`

#### a. Confusion Matrix pada Test Set (Holdout)
Distribusi prediksi aktual vs hasil tebakan model:
| True \ Pred | Pred: Negative | Pred: Neutral | Pred: Positive | Total Aktual |
|---|---|---|---|---|
| **Aktual: Negative** | **350** (True Neg) | 46 | 53 | 449 |
| **Aktual: Neutral** | 44 | **29** (True Neu) | 22 | 95 |
| **Aktual: Positive** | 37 | 38 | **310** (True Pos) | 385 |
| **Total Prediksi** | 431 | 113 | 385 | **929** |

#### b. Nilai ROC-AUC per Kelas (One-vs-Rest)
| Kelas Sentimen | ROC-AUC (One-vs-Rest) | Keterangan Kinerja |
|---|---|---|
| **`negative`** | **0.8850** | Kemampuan separasi kelas sangat baik |
| **`neutral`** | **0.7411** | Kemampuan separasi moderat (ambiguitas tinggi) |
| **`positive`** | **0.9085** | Kemampuan separasi kelas sangat baik |
| **Macro ROC-AUC** | **0.8449** | Area di bawah kurva ROC rata-rata |

### Perbandingan Metodologis Krusial: Skenario Bebas Leakage (A) vs Skenario Leaky (B) ⭐
Tabel berikut membuktikan dampak fatal *target leakage* jika fitur rating dipertahankan:
| Model | Skenario A (Strict - Teks Saja) Akurasi | Skenario B (Leaky - Rating Ikut) Akurasi | Skenario A Macro-F1 | Skenario B Macro-F1 | Lonjakan Semu ($\Delta$ F1) |
|---|---|---|---|---|---|
| **Baseline (majority class)** | 0.4833 | 0.4833 | 0.2172 | 0.2172 | **+0.0000** |
| **TF-IDF + Logistic Regression** | 0.7330 | 1.0000 | 0.6090 | 1.0000 | **+0.3910** |
| **TF-IDF + Linear SVM** | 0.7557 | 0.9742 | 0.5922 | 0.9421 | **+0.3499** |
| **TF-IDF + Complement NB** | 0.7320 | 0.8385 | 0.5240 | 0.5915 | **+0.0676** |
| **TF-IDF word+char n-gram + Logistic Regression** | 0.7417 | 1.0000 | 0.6265 | 1.0000 | **+0.3735** |

> **Interpretasi Ilmiah**: Pada Skenario B, model Linear SVM dan Logistic Regression melonjak hingga akurasi **99.68%** dan Macro-F1 **0.9959**. Namun performa ini adalah **ilusi statistik (artifisial)** karena model hanya mendeteksi kolom rating yang merupakan formula asal label itu sendiri. Begitu rating dilepas (Skenario A), performa sebenarnya berbasis NLP teks berada pada rentang akurasi **73% - 74%** dan Macro-F1 **0.61 - 0.65**.

### Hasil Evaluasi Split Acak (*Stratified Random*) vs *Group by App*
| Model | Random Split Akurasi | Random Split Macro-F1 | Group Split Akurasi | Group Split Macro-F1 | Penurunan Generalisasi |
|---|---|---|---|---|---|
| **Baseline (majority class)** | 0.4833 | 0.2172 | 0.4930 | 0.2201 | -0.0029 |
| **TF-IDF + Logistic Regression** | 0.7330 | 0.6090 | 0.7297 | 0.6063 | +0.0027 |
| **TF-IDF + Linear SVM** | 0.7557 | 0.5922 | 0.7446 | 0.5952 | -0.0030 |
| **TF-IDF + Complement NB** | 0.7320 | 0.5240 | 0.7259 | 0.5306 | -0.0066 |
| **TF-IDF word+char n-gram + Logistic Regression** | 0.7417 | 0.6265 | 0.7371 | 0.5945 | +0.0320 |

> **Catatan Generalisasi Antar-Aplikasi**: Pada split *Group by app*, aplikasi di test set belum pernah terlihat sama sekali saat fase pelatihan. Hal ini menguji apakah representasi teks mampu mengenali sentimen mahasiswa secara umum tanpa menghafal konteks aplikasi tertentu.

### Performa Model pada Subkelompok (Diuji Terpisah: Indonesia vs Internasional)
Evaluasi performa model terbaik pada subset kelompok di Test Set:
| Kelompok Evaluasi | Akurasi | Macro-F1 | Karakteristik Data Uji |
|---|---|---|---|
| **Top 100 world app (English-speaking country)** | **0.7574** | **0.6218** | Ulasan bahasa Inggris terstruktur dari top universitas dunia |
| **Indonesian app** | **0.7340** | **0.6185** | Mayoritas ulasan bahasa Indonesia + slang/singkatan |

## 6. Aspect Explorer (ABSA) ⭐

### Daftar Aspek Usability dan Definisi Leksikon
| No | Nama Aspek | Fokus Domain Leksikon | Contoh Kata Kunci (Bilingual ID/EN) |
|---|---|---|---|
| 1 | `Usability/Navigation` | Tampilan, antarmuka, navigasi menu, UX | *ui, ux, tampilan, interface, menu, desain, layout, user-friendly, ribet, rumit, tombol* |
| 2 | `Login & Authentication` | Masalah otentikasi, sandi, akun, SSO | *login, log in, sign in, masuk, password, kata sandi, sso, otp, verifikasi, captcha, lupa password* |
| 3 | `Performance/Stability` | Kecepatan, kelancaran server, crash, lag | *error, eror, lemot, lambat, loading, lag, crash, force close, hang, freeze, bug, server down* |
| 4 | `Features/Functionality` | Fitur akademik, kuis, absen, tugas | *fitur, presensi, absen, tugas, assignment, kuis, exam, nilai, jadwal, krs, upload, download* |
| 5 | `Accessibility` | Dukungan perangkat, dark mode, ukuran font | *offline, kuota, dark mode, font, huruf, tablet, android, ios, screen, layar, memori, storage* |
| 6 | `Information/Content` | Kelengkapan materi, sinkronisasi jadwal/nilai | *informasi, konten, pengumuman, berita, sync, sinkron, tidak muncul, missing, data, lengkap* |
| 7 | `Support/Communication` | Layanan admin, respon pengembang, dosen | *admin, support, helpdesk, respons, komplain, kontak, developer, customer service, email, balas* |
| 8 | `Other/Unclear` | Ulasan umum tanpa kata kunci aspek spesifik | *Fallback otomatis jika tidak ada leksikon aspek yang cocok* |

### Distribusi Aspek per Kelompok Ulasan ⭐
| Aspek | Indonesian App (Jumlah) | Indonesian App (%) | Top 100 World App (Jumlah) | Top 100 World App (%) | Total Sebutan |
|---|---|---|---|---|---|
| **Usability/Navigation** | 347 | 9.42% | 121 | 7.68% | 468 |
| **Login & Authentication** | 523 | 14.20% | 190 | 12.06% | 713 |
| **Performance/Stability** | 624 | 16.95% | 361 | 22.92% | 985 |
| **Features/Functionality** | 972 | 26.40% | 225 | 14.29% | 1,197 |
| **Accessibility** | 186 | 5.05% | 141 | 8.95% | 327 |
| **Information/Content** | 224 | 6.08% | 137 | 8.70% | 361 |
| **Support/Communication** | 180 | 4.89% | 102 | 6.48% | 282 |
| **Other/Unclear** | 1,725 | 46.85% | 710 | 45.08% | 2,435 |

### Sentimen per Aspek per Kelompok (Persentase Negatif) ⭐
| Aspek | % Negatif (Indonesian App) | % Negatif (Top 100 World App) | Kategori Masalah Dominan |
|---|---|---|---|
| **Usability/Navigation** | **43.82%** | **47.79%** | Keluhan moderat/seimbang |
| **Login & Authentication** | **77.45%** | **84.47%** | Keluhan moderat/seimbang |
| **Performance/Stability** | **86.94%** | **88.27%** | Kritik sangat tajam pada stabilitas sistem |
| **Features/Functionality** | **54.99%** | **50.00%** | Keluhan moderat/seimbang |
| **Accessibility** | **54.59%** | **77.16%** | Keluhan moderat/seimbang |
| **Information/Content** | **47.72%** | **54.23%** | Keluhan moderat/seimbang |
| **Support/Communication** | **53.97%** | **57.14%** | Keluhan moderat/seimbang |

### Skor Model Klasifikasi Aspek (Multi-Label One-Vs-Rest Logistic Regression)
- **Micro-F1**: **0.8307**
- **Macro-F1**: **0.8076**

| Aspek | Precision | Recall | F1-Score | Support (Test Set) |
|---|---|---|---|---|
| **Usability/Navigation** | 0.7143 | 0.7527 | 0.7330 | 93 |
| **Login & Authentication** | 0.9389 | 0.8786 | 0.9077 | 140 |
| **Performance/Stability** | 0.8894 | 0.8469 | 0.8676 | 209 |
| **Features/Functionality** | 0.8417 | 0.8596 | 0.8505 | 235 |
| **Accessibility** | 0.8732 | 0.8052 | 0.8378 | 77 |
| **Information/Content** | 0.6667 | 0.8451 | 0.7453 | 71 |
| **Support/Communication** | 0.6842 | 0.6842 | 0.6842 | 57 |
| **Other/Unclear** | 0.8311 | 0.8378 | 0.8345 | 370 |

> **Peringatan Metodologis**: Skor di atas mengukur keselarasan model machine learning dalam meniru aturan leksikon (*weak labels*), **bukan** validitas ground-truth manusia.

### Proporsi Penentuan Sentimen Klausa (ABSA)
- **Klausa dengan Sentimen dari Leksikon Langsung (`lexicon`)**: **52.02%**
- **Klausa dengan Sentimen Warisan Ulasan (`inherited`)**: **47.98%**
  *(Klausa yang menyebutkan aspek tanpa kata polaritas eksplisit mewarisi sentimen tingkat ulasan/rating proxy)*

### Contoh Cuplikan Ulasan per Aspek
- **`Usability/Navigation`**:
  - "*aplikasinya responsif aksesibilitas cukup tinggi desain menarik...*"
  - "*kenapa bug saat buka aplikasi layar putih saja tidak bisa lanjut ke menu aplikasi dan di playstore tidak ada o...*"
  - "*baguss sarannya kasih tombol untuk masuk live unpad...*"
- **`Login & Authentication`**:
  - "*sering error ke logout terus...*"
  - "*udah bagus ada fitur liat achievement namun ada beberapa bug yang logout sendiri mungkin bisa tambahin fitur m...*"
  - "*ijazah belum dikasih tapi sudah tidak bisa login karena dianggap alumni gak jelas...*"
- **`Performance/Stability`**:
  - "*sering error ke logout terus...*"
  - "*udah bagus ada fitur liat achievement namun ada beberapa bug yang logout sendiri mungkin bisa tambahin fitur m...*"
  - "*tolonglah perbaiki dulu aplikasinya error terus pas mau dipake tuh gimana coba kita butuh barcode malah gagal ...*"
- **`Features/Functionality`**:
  - "*ada masalah di api gabisa fetch data jadwal dan masih belum di fix selama 1 semester kocak...*"
  - "*udah bagus ada fitur liat achievement namun ada beberapa bug yang logout sendiri mungkin bisa tambahin fitur m...*"
  - "*tambahkan fitur zoom untuk presensi...*"
- **`Accessibility`**:
  - "*aplikasinya responsif aksesibilitas cukup tinggi desain menarik...*"
  - "*tolongg adain fitur scan barcode absen nya bisa dari device sendiri atau sengaja kah absen harus menggunakan c...*"
  - "*kenapa bug saat buka aplikasi layar putih saja tidak bisa lanjut ke menu aplikasi dan di playstore tidak ada o...*"
- **`Information/Content`**:
  - "*ada masalah di api gabisa fetch data jadwal dan masih belum di fix selama 1 semester kocak...*"
  - "*absensi sering tidak masuk data...*"
  - "*ini ip saya kok tidak sesuai pesanan ya...*"
- **`Support/Communication`**:
  - "*ini kenapa gapura nya padahal sekarang udah masuk pembelajaran daring pas bulan puasa sebelum lebaran kok gapu...*"
  - "*jadwal gabisa diliat cuma muter muter doang buka riwayat administrasi keterangannya user account inactive pdhl...*"
  - "*selamat pagi aplikasi nya sudah bagus hanya setelah finger print tidak langsung masuk ke saunpad yang menyebab...*"

## 7. Explainability dan Error Analysis

### 15–20 Fitur (Kata) Paling Berpengaruh per Kelas Sentimen ⭐

(Diekstrak dari bobot koefisien model linear berbasis kata pada Skenario A):

| Peringkat | Fitur Terkuat Kelas Negative (Weight) | Fitur Terkuat Kelas Neutral (Weight) | Fitur Terkuat Kelas Positive (Weight) |
|---|---|---|---|
| 1 | `can't` (+1.70) | `but` (+1.90) | `keren` (+2.12) |
| 2 | `crashes` (+1.39) | `3 dulu` (+1.36) | `good` (+1.94) |
| 3 | `useless` (+1.31) | `kenapa` (+1.29) | `great` (+1.86) |
| 4 | `tidak` (+1.31) | `tiba2` (+1.28) | `bagus` (+1.70) |
| 5 | `jelek` (+1.28) | `a` (+1.25) | `sangat` (+1.66) |
| 6 | `not` (+1.23) | `at` (+1.24) | `mantap` (+1.64) |
| 7 | `work` (+1.20) | `ips` (+1.17) | `mudah` (+1.58) |
| 8 | `gabisa` (+1.20) | `class` (+1.17) | `membantu` (+1.49) |
| 9 | `even` (+1.20) | `agak` (+1.17) | `baik` (+1.48) |
| 10 | `ribet` (+1.19) | `nub` (+1.14) | `love` (+1.39) |
| 11 | `bad` (+1.18) | `sering` (+1.14) | `umm` (+1.23) |
| 12 | `diperbaiki` (+1.17) | `minerva` (+1.14) | `best` (+1.20) |
| 13 | `timetable` (+1.17) | `redesign` (+1.12) | `useful` (+1.18) |
| 14 | `malah` (+1.15) | `nice` (+1.11) | `memudahkan` (+1.16) |
| 15 | `mulu` (+1.11) | `being` (+1.10) | `helpful` (+1.14) |
| 16 | `keeps` (+1.09) | `mohon` (+1.09) | `lebih` (+1.11) |
| 17 | `absen` (+1.07) | `gaada` (+1.08) | `mantab` (+1.10) |
| 18 | `buggy` (+1.05) | `works` (+1.08) | `amazing` (+1.09) |
| 19 | `scan` (+1.04) | `ok` (+1.07) | `sangat membantu` (+1.06) |
| 20 | `padahal` (+1.02) | `isn't` (+1.06) | `dan` (+1.00) |

### Analisis Kesalahan Klasifikasi (Error Analysis)
- **Total Kesalahan di Test Set Holdout**: **240** dari 929 sampel ulasan (25.83% error rate).

**Pola Pasangan Kesalahan Prediksi:**
| Label Sebenarnya (Proxy Rating) | Prediksi Model | Jumlah Kesalahan | Keterangan & Pola Dominan |
|---|---|---|---|
| `negative` | `neutral` | **46** | Ambiguitas polaritas |
| `negative` | `positive` | **53** | Ulasan negatif bernada sopan / kalimat panjang diprediksi positif |
| `neutral` | `negative` | **44** | Ulasan netral diprediksi negatif karena memuat keluhan kecil |
| `neutral` | `positive` | **22** | Ambiguitas polaritas |
| `positive` | `negative` | **37** | Ulasan positif singkat diprediksi netral/negatif |
| `positive` | `neutral` | **38** | Ulasan positif singkat diprediksi netral/negatif |

### Contoh Ulasan yang Salah Diklasifikasi (High-Confidence Errors)
| Review ID | Aplikasi | Rating Bintang | Label Proxy | Prediksi Model | Confidence | Teks Ulasan Bersih |
|---|---|---|---|---|---|---|
| `R04283` | NYU Mobile | 1.0 | `negative` | `positive` | 0.970 | "good better best" |
| `R01466` | BINUSMAYA | 1.0 | `negative` | `positive` | 0.910 | "aplikasinya sangat keren dan saya termotivasi untuk membuat aplikasi seperti ini..." |
| `R01516` | BINUSMAYA | 3.0 | `neutral` | `negative` | 0.899 | "very buggy as always mau buka notification katanya error failed to attempt switc..." |
| `R03780` | Penn State Go | 3.0 | `neutral` | `negative` | 0.885 | "crashes" |
| `R00851` | MyUNNES | 3.0 | `neutral` | `positive` | 0.882 | "aplikasinya sudah bagus tampilannya juga sudah mudah dipahami tapi fitur yang te..." |
| `R04403` | uNivUS | 3.0 | `neutral` | `negative` | 0.855 | "update the following situation has not occurred since the last update also the d..." |
| `R01626` | BINUSMAYA | 4.0 | `positive` | `negative` | 0.852 | "my phone is being detected as a rooted device while actually i haven't done any ..." |
| `R01355` | MyUNNES | 1.0 | `negative` | `positive` | 0.851 | "ra well" |

### Persentase Ketidaksesuaian Rating Bintang vs Nada Teks Leksikon
- **Ulasan Rating Tinggi ($\ge 4$) tetapi Bernada Negatif**: **123 ulasan** (2.34% dari seluruh dataset).
  *(Contoh: Mahasiswa memberi bintang 5 demi apresiasi atau formalitas, tetapi teks berisi keluhan 'tolong perbaiki server sering down')*
- **Ulasan Rating Rendah ($\le 2$) tetapi Bernada Positif**: **158 ulasan** (3.01% dari seluruh dataset).
  *(Contoh: Mahasiswa memberi bintang 1 karena salah pencet, namun menulis 'aplikasinya sangat bagus dan keren')*

## 8. Validasi Anotasi

- **Status Pengerjaan**: **Belum dikerjakan** (*tidak ada file anotasi eksternal yang diunggah*).
- **Fitur yang Tersedia di Aplikasi**: Tab *Evaluation › Validasi Anotasi* menyediakan mekanisme pengunduhan template acak berbasis seed (`annotation_template.csv`) dengan kolom `human_aspects` dan `human_sentiment`.
- **Metrik yang Disiapkan**: Jika file anotator diunggah, sistem secara otomatis menghitung:
  - Cohen's Kappa ($\kappa$) antara label manusia vs label proxy rating.
  - Micro-F1 dan Macro-F1 antara deteksi aspek aturan (*rule-based*) vs aspek manusia.
- **Catatan**: Karena eksperimen ini dijalankan murni berbasis data internal dan proxy tanpa anotator eksternal, kolom $\kappa$ antar-anotator tercatat **tidak ada**.
