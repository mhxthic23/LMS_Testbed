
from pathlib import Path
import pandas as pd

# Lokasi file
INPUT_FILE = Path("data/raw/[Data Set] Combined LMS App Review.xlsx")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "lms_reviews_clean.csv"

# Baca sheet review
df = pd.read_excel(
    INPUT_FILE,
    sheet_name="Combined_Reviews"
)

print("Jumlah data awal:", len(df))

# Hapus review kosong
df = df.dropna(subset=["review_body"]).copy()

# Ubah review menjadi string dan rapikan spasi
df["review_body"] = (
    df["review_body"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Hapus review yang kosong setelah dibersihkan
df = df[df["review_body"] != ""]

# Konversi tanggal Excel ke datetime
df["review_created_at"] = pd.to_datetime(
    df["review_created_at"],
    unit="D",
    origin="1899-12-30",
    errors="coerce"
)

# Buat label sentimen berdasarkan rating
def get_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df["sentiment"] = df["star_rating"].apply(get_sentiment)

# Simpan hasil preprocessing
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n=== HASIL PREPROCESSING ===")
print("Jumlah data bersih:", len(df))
print("Jumlah data yang dihapus:", 5259 - len(df))

print("\nDistribusi sentimen:")
print(df["sentiment"].value_counts())

print("\nDistribusi cakupan aplikasi:")
print(df["app_scope_flag"].value_counts())

print("\nDistribusi bahasa:")
print(df["language"].value_counts())

print("\nFile tersimpan di:", OUTPUT_FILE)
