
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

files = (
    list(DATA_DIR.glob("*.xlsx"))
    + list(DATA_DIR.glob("*.xls"))
    + list(DATA_DIR.glob("*.csv"))
)

if not files:
    print("Dataset tidak ditemukan!")

else:
    file_path = files[0]

    print("Dataset:", file_path.name)

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(
            file_path,
            sheet_name="Combined_Reviews"
        )

    print("\n=== INFORMASI DATASET ===")
    print("Jumlah baris:", df.shape[0])
    print("Jumlah kolom:", df.shape[1])

    print("\n=== NAMA KOLOM ===")
    print(df.columns.tolist())

    print("\n=== 10 DATA PERTAMA ===")
    print(df.head(10).to_string())

    print("\n=== DATA KOSONG ===")
    print(df.isnull().sum())

    print("\n=== DUPLIKAT ===")
    print(df.duplicated().sum())

    print("\n=== TIPE DATA ===")
    print(df.dtypes)

    print("\n=== DATASET SELESAI DIPERIKSA ===")
