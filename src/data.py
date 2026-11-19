"""Loading, deteksi/pemetaan kolom, standardisasi, kualitas data, dan filter."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COLUMN_CANDIDATES, DEFAULT_PATHS


def _key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def read_any(file_or_path):
    name = getattr(file_or_path, "name", str(file_or_path)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_or_path)
    try:
        return pd.read_csv(file_or_path)
    except UnicodeDecodeError:
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        return pd.read_csv(file_or_path, encoding="latin-1")


def find_local_file():
    for p in DEFAULT_PATHS:
        if Path(p).exists():
            return p
    return None


def detect_columns(df):
    """Cocokkan kolom aktual ke nama kanonik berdasarkan kandidat nama (tanpa mengarang kolom)."""
    by_key = {_key(c): c for c in df.columns}
    found = {}
    for canon, cands in COLUMN_CANDIDATES.items():
        found[canon] = next((by_key[_key(c)] for c in cands if _key(c) in by_key), None)
    return found


WORD_RATING = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5}


def parse_rating(series):
    """Rating bisa berupa teks ('5 stars', 'four'). Petakan eksplisit, bukan asumsi numerik."""
    def conv(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return float(v)
        s = str(v).strip().lower()
        if s in WORD_RATING:
            return float(WORD_RATING[s])
        m = re.search(r"(\d+(?:[.,]\d+)?)", s)
        return float(m.group(1).replace(",", ".")) if m else np.nan
    out = series.map(conv)
    return out.where(out.between(1, 5))


def standardize(raw, mapping):
    """Bangun DataFrame dengan nama kolom kanonik dari pemetaan kolom."""
    df = pd.DataFrame(index=raw.index)
    for canon, col in mapping.items():
        df[canon] = raw[col] if col and col in raw.columns else np.nan
    df["rating_raw"] = df["rating"]
    df["rating"] = parse_rating(df["rating"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce")
    for c in ["institution", "app", "group", "language"]:
        df[c] = df[c].astype("object").where(df[c].notna(), "Unknown").astype(str)
    df["review_text"] = df["review_text"].astype("object").where(df["review_text"].notna(), "").astype(str)
    df["review_title"] = df["review_title"].astype("object").where(df["review_title"].notna(), "").astype(str)
    if mapping.get("review_id") is None or df["review_id"].isna().all():
        df["review_id"] = [f"R{i:05d}" for i in range(len(df))]
    df["review_id"] = df["review_id"].astype(str)
    return df.reset_index(drop=True)


def quality_report(df):
    n = len(df)
    txt = df["review_text"].str.strip()
    rep = {
        "n_rows": n,
        "n_apps": df["app"].nunique(),
        "n_institutions": df["institution"].nunique(),
        "missing_rating": int(df["rating"].isna().sum()),
        "empty_text": int((txt == "").sum()),
        "duplicate_rows": int(df.duplicated(subset=["app", "review_text", "rating"]).sum()),
        "duplicate_texts": int(txt[txt != ""].duplicated().sum()),
        "short_text": int(df["n_tokens"].between(1, 2).sum()),
        "emoji_only": int(df["emoji_only"].sum()),
    }
    miss = (df.replace({"": np.nan, "Unknown": np.nan}).isna().mean() * 100).round(2)
    miss = miss.drop(labels=["rating_raw"], errors="ignore").rename("missing_%").reset_index()
    miss.columns = ["column", "missing_%"]
    return rep, miss


def apply_filters(df, institutions, apps, groups, languages, date_range, ratings, search, include_unrated=True):
    m = df["institution"].isin(institutions) & df["app"].isin(apps) & df["group"].isin(groups)
    m &= df["lang_detected"].isin(languages)
    rated = df["rating"].isin(ratings)
    m &= rated | (df["rating"].isna() & include_unrated)
    if date_range and len(date_range) == 2 and df["date"].notna().any():
        d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        m &= (df["date"].isna()) | ((df["date"] >= d0) & (df["date"] < d1))
    if search:
        m &= df["review_text"].str.contains(search, case=False, na=False, regex=False)
    return df[m]
