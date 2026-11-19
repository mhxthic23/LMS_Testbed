# LMS Review Mining Testbed

Aplikasi Streamlit untuk pipeline data mining/NLP pada ulasan Google Play aplikasi LMS & student portal
(Indonesia vs internasional): **Overview → EDA → Preprocessing → Modeling → Evaluation**, plus Aspect Explorer,
Explainability & Error Analysis, dan Review Explorer.

## Setup
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Dataset dibaca otomatis dari `data/processed/lms_reviews_clean.csv` (atau `data/raw/*.xlsx`), atau unggah CSV/XLSX di sidebar.
Kolom dideteksi dari nama aktual dan bisa diubah di sidebar › *Pemetaan kolom*.

## Fitur interaktif (sidebar)
Upload CSV/XLSX · ambang rating→sentimen · **Leakage scenario** (A: teks saja / B: rating+likes ikut = leakage demo) ·
**Feature scaler** (Standard/MinMax/Robust/None) · test size · validation size · CV folds · pilihan model · strategi split ·
preprocessing teks · seed · filter institusi/aplikasi/kelompok/bahasa/rating/tanggal · pencarian.

Tab: Overview · EDA (quality, rating, **chi-square & Kruskal-Wallis vs sentimen**, Indonesia vs Internasional) ·
Preprocessing (**engineered feature matrix**, efek scaler, cek korelasi/leakage, text cleaning) · Modeling (**Train models**, quick leaderboard) ·
Evaluation (**Model comparison**, confusion matrix, ROC, aspek, validasi anotasi) · Aspect Explorer (ABSA) ·
Explainability (**playground prediksi**, fitur berpengaruh, error analysis) · Review Explorer + ekspor CSV.

## Struktur
```
app.py                  entry point, sidebar (data, konfigurasi, filter), tab
src/config.py           seed, leksikon aspek/sentimen, kamus slang/emoji
src/data.py             load, deteksi kolom, parsing rating, kualitas data, filter
src/text.py             preprocessing code-mixed, emoji, negasi, deteksi bahasa
src/labeling.py         label proxy rating, aspek (weak label), ABSA klausa
src/modeling.py         split, model, CV, metrik, fitur berpengaruh
src/ui.py               CSS & chart helper
src/views/*.py          isi tiap tab
```

## Metode
- **Sentimen**: label *proxy* dari rating (ambang di sidebar). Rating tidak pernah menjadi fitur.
- **Preprocessing** (3 opsi, dibandingkan empiris): A minimal, B slang+tag negasi, C agresif (stopword). Tanpa terjemahan.
- **Split**: train/val/test reproducible (seed tampil); duplikat teks dihapus sebelum split; opsi *group by app*.
- **Model**: baseline mayoritas, TF-IDF + LR / Linear SVM / Complement NB, TF-IDF word+char + LR. Metrik: accuracy, macro & per-kelas P/R/F1, confusion matrix, K-fold CV.
- **Aspek**: multi-label dari leksikon bilingual (+ Other/Unclear); model TF-IDF+LR meniru aturan tsb.
- **ABSA**: sentimen per klausa (leksikon + negasi); klausa tanpa polaritas mewarisi sentimen review (ditandai `inherited`).
- **Validasi**: tab Evaluation › *Validasi anotasi* (unduh sampel, isi manual, unggah, hitung metrik & Cohen's κ).

## Asumsi & keterbatasan
- Rating ≠ sentimen sebenarnya (mis. rating 5 berisi keluhan; rating 3 ambigu).
- Skor model aspek/ABSA mengukur kesesuaian dengan leksikon, **bukan** validitas, sampai ada anotasi manusia.
- Deteksi bahasa = heuristik leksikon; leksikon aspek/sentimen terbatas dan perlu diperluas.
- Perbandingan Indonesia vs internasional terpengaruh perbedaan aplikasi, bahasa, populasi, dan jumlah review.
- Model transformer (IndoBERT/XLM-R) belum disertakan; bisa ditambah sebagai model lanjutan.
