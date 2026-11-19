"""Helper tampilan: CSS, kartu metrik, chart (Altair), glosarium."""
import altair as alt
import streamlit as st

SENT_COLORS = alt.Scale(domain=["negative", "neutral", "positive"], range=["#d1495b", "#9aa5b1", "#2a9d8f"])
CSS = """
<style>
.block-container{padding-top:2.2rem;max-width:1250px}
.hero{padding:1.1rem 1.4rem;border-radius:14px;background:linear-gradient(120deg,#1f3a5f,#2a6f97);color:#fff;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.9rem;color:#fff} .hero p{margin:.3rem 0 0;opacity:.9}
.card{border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:.8rem 1rem;margin-bottom:.6rem}
.warn{border-left:5px solid #e9a23b;background:rgba(233,162,59,.12);padding:.6rem .9rem;border-radius:6px;margin:.5rem 0}
.good{border-left:5px solid #2a9d8f;background:rgba(42,157,143,.12);padding:.6rem .9rem;border-radius:6px;margin:.5rem 0}
.stTabs [data-baseweb="tab"]{font-weight:600}
</style>
"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def warn(msg):
    st.markdown(f'<div class="warn">⚠️ {msg}</div>', unsafe_allow_html=True)


def good(msg):
    st.markdown(f'<div class="good">✅ {msg}</div>', unsafe_allow_html=True)


def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


def bar(df, x, y, color=None, horizontal=False, height=300, title=None, sent_color=False, stack=True):
    enc = dict(x=alt.X(x, sort=None) if not horizontal else alt.X(y), y=alt.Y(y) if not horizontal else alt.Y(x, sort="-x"))
    if color:
        enc["color"] = alt.Color(color, scale=SENT_COLORS) if sent_color else alt.Color(color)
        if not stack and not horizontal:
            enc["xOffset"] = alt.XOffset(color)
    ch = alt.Chart(df, title=title or "").mark_bar().encode(**enc, tooltip=list(df.columns)).properties(height=height)
    st.altair_chart(ch, use_container_width=True)


def line(df, x, y, color=None, height=300, title=None, sent_color=False):
    enc = dict(x=x, y=y)
    if color:
        enc["color"] = alt.Color(color, scale=SENT_COLORS) if sent_color else alt.Color(color)
    st.altair_chart(alt.Chart(df, title=title or "").mark_line(point=True).encode(**enc, tooltip=list(df.columns))
                    .properties(height=height), use_container_width=True)


def heatmap_cm(cm, labels, height=280):
    import pandas as pd
    d = pd.DataFrame([(labels[i], labels[j], int(cm[i][j])) for i in range(len(labels)) for j in range(len(labels))],
                     columns=["Aktual", "Prediksi", "n"])
    base = alt.Chart(d).encode(x=alt.X("Prediksi:N", sort=labels), y=alt.Y("Aktual:N", sort=labels))
    st.altair_chart(base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"))) +
                    base.mark_text(fontSize=15).encode(text="n:Q"), use_container_width=True)


def glossary():
    with st.expander("📖 Glosarium singkat istilah data mining"):
        st.markdown("""
- **TF-IDF**: bobot kata; tinggi bila sering muncul di satu review tapi jarang di review lain.
- **Macro-F1**: rata-rata F1 antar kelas (tiap kelas bobot sama) – lebih jujur daripada accuracy saat kelas tidak seimbang.
- **Precision / Recall**: dari semua prediksi kelas X, berapa yang benar / dari semua data kelas X, berapa yang ditemukan.
- **Cross-validation (CV)**: data dibagi K lipatan; model diuji bergantian untuk menilai stabilitas skor.
- **Data leakage**: informasi test/label bocor ke training sehingga skor tampak terlalu bagus.
- **Label proxy**: label pengganti (di sini rating bintang → sentimen) – bukan penilaian manusia atas teks.
- **Weak label**: label otomatis dari aturan/leksikon; perlu divalidasi manusia.
- **ABSA**: sentimen per aspek (mis. *login* negatif, *tampilan* positif) dalam satu review.
- **Code-mixing**: campuran bahasa dalam satu kalimat (Indonesia + Inggris).
""")
