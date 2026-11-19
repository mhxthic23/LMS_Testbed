"""Pelabelan: sentimen proxy dari rating, aspek berbasis leksikon (weak labels), dan ABSA klausa."""
import re

import numpy as np
import pandas as pd

from .config import (ASPECT_NAMES, ASPECTS, CAPABILITY, NEG_WORDS, NEGATIONS, OTHER, POS_WORDS)
from .text import base_clean

_ASPECT_RE = {a: re.compile(r"(?<![a-z])(?:" + "|".join(kws) + ")") for a, kws in ASPECTS.items()}
_SPLIT_RE = re.compile(r"[.!?;\n]+|\b(?:tapi|namun|tetapi|but|however|sedangkan|padahal)\b")


def rating_to_sentiment(rating, neg_max=2, pos_min=4):
    """LABEL PROXY: rating bintang -> sentimen. Bukan label manusia."""
    if pd.isna(rating):
        return np.nan
    if rating <= neg_max:
        return "negative"
    if rating >= pos_min:
        return "positive"
    return "neutral"


def detect_aspects(text):
    t = base_clean(text)
    found = [a for a, rx in _ASPECT_RE.items() if rx.search(t)]
    return found or [OTHER]


def lexicon_polarity(text):
    """Skor polaritas klausa dengan pembalikan negasi (jendela 3 kata). Return (label|None, skor)."""
    toks = re.findall(r"[\w']+", base_clean(text))
    score = 0
    for i, w in enumerate(toks):
        window = toks[max(0, i - 3):i]
        negated = any(x in NEGATIONS for x in window)
        if w in POS_WORDS:
            score += -1 if negated else 1
        elif w in NEG_WORDS:
            score += 1 if negated else -1
        elif w in CAPABILITY and negated:
            score -= 1
    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return None, 0


def absa(text, fallback):
    """ABSA tingkat klausa: aspek yang disebut + sentimen klausa (leksikon); jika klausa netral,
    mewarisi sentimen tingkat-review (ditandai source='inherited')."""
    rows = []
    for clause in _SPLIT_RE.split(str(text).lower()):
        clause = clause.strip()
        if len(clause) < 3:
            continue
        aspects = [a for a in detect_aspects(clause) if a != OTHER]
        if not aspects:
            continue
        label, _ = lexicon_polarity(clause)
        src = "lexicon"
        if label is None:
            label, src = (fallback if isinstance(fallback, str) else "neutral"), "inherited"
        for a in aspects:
            rows.append({"aspect": a, "clause": clause, "sentiment": label, "source": src})
    return rows


def enrich(df, mode, neg_max, pos_min):
    """Tambahkan kolom turunan (dipakai semua tab)."""
    from .text import clean_text, detect_language, text_flags
    out = df.copy()
    flags = out["review_text"].map(text_flags).apply(pd.Series)
    out = pd.concat([out, flags], axis=1)
    out["clean_min"] = out["review_text"].map(base_clean)
    out["clean_model"] = out["review_text"].map(lambda t: clean_text(t, mode))
    out["lang_detected"] = out["review_text"].map(detect_language)
    out["sent_label"] = out["rating"].map(lambda r: rating_to_sentiment(r, neg_max, pos_min))
    out["aspects"] = out["review_text"].map(detect_aspects)
    out["aspects_str"] = out["aspects"].map("; ".join)
    out["lex_sentiment"] = out["review_text"].map(lambda t: lexicon_polarity(t)[0] or "neutral")
    out["n_chars"] = out["review_text"].str.len()
    out["n_exclaim"] = out["review_text"].str.count("!")
    out["aspect_primary"] = out["aspects"].map(lambda a: a[0])
    out["month"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out


def single_row(text, rating, mode, neg_max, pos_min):
    """Bangun 1 baris terproses dari teks bebas (untuk playground prediksi)."""
    df = pd.DataFrame({"review_text": [text], "rating": [rating], "likes": [0.0], "date": pd.to_datetime([None]),
                       "group": ["Unknown"], "institution": ["Unknown"], "app": ["Unknown"], "review_id": ["X"]})
    return enrich(df, mode, neg_max, pos_min)
