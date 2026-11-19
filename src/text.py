"""Preprocessing multibahasa/code-mixed: normalisasi, emoji, slang, negasi, deteksi bahasa."""
import re

from .config import (EMOJI_MAP, EN_HINT, ID_HINT, NEGATIONS, SLANG, STOP_EN, STOP_ID)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\U0001F000-\U0001F2FF\u2B50\u2764]")
REPEAT_RE = re.compile(r"(.)\1{2,}")

PREPROC_MODES = {
    "A - Minimal (aman)": "Huruf kecil, URL->token, emoji->token, kurangi huruf berulang. Negasi, emoji, istilah teknis dipertahankan.",
    "B - Normalisasi + tag negasi": "A + normalisasi slang/singkatan (tanpa terjemahan bahasa) + penanda negasi (NEG_kata) pada 2 kata setelah negasi.",
    "C - Agresif (stopword removal)": "A + slang + hapus stopword ID/EN (termasuk negasi). Dipakai sebagai pembanding: biasanya merusak makna.",
}


def base_clean(text):
    t = "" if text is None else str(text)
    t = t.lower()
    t = URL_RE.sub(" url ", t)
    for e, tok in EMOJI_MAP.items():
        t = t.replace(e, tok)
    t = EMOJI_RE.sub(" emo_other ", t)
    t = REPEAT_RE.sub(r"\1\1", t)
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_slang(tokens):
    out = []
    for w in tokens:
        out.extend(SLANG.get(w, w).split())
    return out


def tag_negation(tokens, window=2):
    out, left = [], 0
    for w in tokens:
        if w in NEGATIONS:
            out.append(w)
            left = window
        elif left > 0 and not w.startswith("emo_"):
            out.append("NEG_" + w)
            left -= 1
        else:
            out.append(w)
    return out


def clean_text(text, mode):
    t = base_clean(text)
    if mode.startswith("A"):
        return t
    toks = normalize_slang(t.split())
    if mode.startswith("B"):
        return " ".join(tag_negation(toks))
    toks = [w for w in toks if w not in STOP_ID and w not in STOP_EN and w not in NEGATIONS and len(w) > 1]
    return " ".join(toks)


def text_flags(raw):
    s = "" if raw is None else str(raw)
    has_letter = bool(re.search(r"[^\W\d_]", EMOJI_RE.sub("", s)))
    has_emoji = bool(EMOJI_RE.search(s)) or any(e in s for e in EMOJI_MAP)
    n_tokens = len(base_clean(s).split())
    return {
        "n_tokens": n_tokens,
        "emoji_only": has_emoji and not has_letter,
        "has_emoji": has_emoji,
        "n_emoji": len(EMOJI_RE.findall(s)),
    }


def detect_language(text):
    """Heuristik leksikon ringan: id / en / mixed / unknown. Bukan penerjemahan."""
    toks = re.findall(r"[a-z']+", base_clean(text))
    if not toks:
        return "unknown"
    n_id = sum(w in ID_HINT for w in toks)
    n_en = sum(w in EN_HINT for w in toks)
    if n_id == 0 and n_en == 0:
        return "unknown"
    if n_id and n_en and min(n_id, n_en) / (n_id + n_en) >= 0.25:
        return "mixed"
    return "id" if n_id > n_en else "en"
