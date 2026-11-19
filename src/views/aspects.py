import pandas as pd
import streamlit as st

from .. import ui
from ..config import ASPECT_NAMES, OTHER, SENT_LABELS
from ..labeling import absa


@st.cache_data(show_spinner="Menghitung ABSA ...")
def absa_table(ids, texts, fallbacks):
    rows = []
    for rid, t, fb in zip(ids, texts, fallbacks):
        for r in absa(t, fb):
            rows.append({"review_id": rid, **r})
    return pd.DataFrame(rows, columns=["review_id", "aspect", "clause", "sentiment", "source"])


def render(ctx):
    fdf = ctx["fdf"]
    st.subheader("Aspect Explorer & ABSA")
    if fdf.empty:
        st.info("Tidak ada data sesuai filter.")
        return
    ex = fdf.explode("aspects").rename(columns={"aspects": "aspect"})
    freq = ex.groupby("aspect").size().rename("n").reset_index().sort_values("n", ascending=False)
    freq["persen_review"] = (freq["n"] / len(fdf) * 100).round(1)
    st.markdown("#### Frekuensi aspek (1 review bisa > 1 aspek)")
    ui.bar(freq, "n:Q", "aspect:N", horizontal=True, height=300)
    ui.warn("Aspek ditentukan oleh **leksikon kata kunci** (weak label). Review tanpa kata kunci → Other/Unclear. Belum tervalidasi manusia.")

    ab = absa_table(tuple(fdf["review_id"]), tuple(fdf["review_text"]), tuple(fdf["sent_label"].fillna("neutral")))
    t1, t2, t3 = st.tabs(["Sentimen per aspek (ABSA)", "Tren waktu", "Contoh klausa"])
    with t1:
        if ab.empty:
            st.info("Tidak ada klausa beraspek terdeteksi.")
        else:
            src = st.radio("Sumber sentimen", ["Semua", "Hanya leksikon klausa (lebih ketat)"], horizontal=True)
            a2 = ab if src == "Semua" else ab[ab["source"] == "lexicon"]
            g = a2.groupby(["aspect", "sentiment"]).size().rename("n").reset_index()
            g["persen"] = g["n"] / g.groupby("aspect")["n"].transform("sum") * 100
            ui.bar(g, "aspect:N", "persen:Q", color="sentiment:N", sent_color=True, title="Komposisi sentimen per aspek (%)", height=340)
            st.dataframe(g.pivot(index="aspect", columns="sentiment", values="n").fillna(0).astype(int), use_container_width=True)
            inh = (ab["source"] == "inherited").mean() * 100
            st.caption(f"{inh:.0f}% klausa tidak punya kata polaritas sehingga mewarisi sentimen review (proxy rating). ABSA ini lanjutan eksperimental, belum tervalidasi.")
            gg = ab.merge(fdf[["review_id", "group"]], on="review_id")
            gg = gg.groupby(["group", "aspect"]).agg(n=("sentiment", "size"), negatif=("sentiment", lambda s: (s == "negative").mean() * 100)).reset_index()
            ui.bar(gg, "aspect:N", "negatif:Q", color="group:N", stack=False, title="% klausa negatif per aspek, per kelompok", height=300)
    with t2:
        if not fdf["month"].notna().any():
            st.info("Kolom tanggal tidak tersedia/valid.")
        else:
            pick = st.multiselect("Aspek", ASPECT_NAMES + [OTHER], default=ASPECT_NAMES[:3])
            mode = st.radio("Tampilkan", ["Jumlah review", "% negatif (proxy rating)"], horizontal=True)
            d = ex.dropna(subset=["month"])
            d = d[d["aspect"].isin(pick)]
            if d.empty:
                st.info("Pilih aspek atau longgarkan filter.")
            elif mode == "Jumlah review":
                ui.line(d.groupby(["month", "aspect"]).size().rename("n").reset_index(), "month:T", "n:Q", color="aspect:N")
            else:
                dd = d.groupby(["month", "aspect"])["sent_label"].apply(lambda s: (s == "negative").mean() * 100).rename("persen_negatif").reset_index()
                ui.line(dd, "month:T", "persen_negatif:Q", color="aspect:N")
                st.caption("Titik dengan sedikit review sangat berfluktuasi; interpretasi hati-hati.")
    with t3:
        asp = st.selectbox("Aspek", ASPECT_NAMES)
        sent = st.selectbox("Sentimen", ["(semua)"] + SENT_LABELS)
        v = ab[ab["aspect"] == asp]
        if sent != "(semua)":
            v = v[v["sentiment"] == sent]
        st.dataframe(v.head(200), hide_index=True, use_container_width=True)
    st.download_button("⬇ Ekspor tabel ABSA (CSV)", ab.to_csv(index=False).encode("utf-8-sig"), "absa_clauses.csv", "text/csv")
