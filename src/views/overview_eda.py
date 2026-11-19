import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from .. import ui
from ..config import ASPECT_NAMES, OTHER, SENT_LABELS
from ..data import quality_report


def render_overview(ctx):
    df, fdf, raw = ctx["df"], ctx["fdf"], ctx["raw"]
    lab = df["sent_label"].value_counts(normalize=True)
    ui.metric_row([("Rows", f"{len(raw):,}"), ("Columns", raw.shape[1]),
                   ("Missing values", f"{int(raw.isna().sum().sum()):,}"),
                   ("Positive rate (proxy)", f"{lab.get('positive', 0):.1%}"),
                   ("Negative rate (proxy)", f"{lab.get('negative', 0):.1%}")])
    st.markdown("### Raw data sample")
    n_show = st.slider("Jumlah baris ditampilkan", 5, 100, 10, step=5)
    st.dataframe(raw.head(n_show), use_container_width=True)
    st.markdown("#### Ringkasan setelah filter")
    ui.metric_row([
        ("Review", f"{len(fdf):,}"), ("Aplikasi", fdf["app"].nunique()), ("Institusi", fdf["institution"].nunique()),
        ("Rata-rata rating", f"{fdf['rating'].mean():.2f}" if fdf["rating"].notna().any() else "-"),
        ("Kelompok", fdf["group"].nunique()),
    ])
    st.markdown("#### Alur pipeline")
    st.markdown("""
| Tahap | Isi | Tab |
|---|---|---|
| 1. Overview | Struktur & ringkasan data | Overview |
| 2. EDA | Kualitas data, bahasa, rating, perbandingan Indonesia vs Internasional | EDA |
| 3. Preprocessing | Normalisasi code-mixed, emoji, slang, negasi + perbandingan opsi | Preprocessing |
| 4. Modeling | Sentiment → Aspek → ABSA, baseline + split tanpa kebocoran | Modeling |
| 5. Evaluation | Metrik, confusion matrix, CV, validasi anotasi | Evaluation |
| Lanjutan | Aspect Explorer, Explainability & Error Analysis, Review Explorer + ekspor | tab terkait |
""")
    st.markdown("#### Pemetaan kolom yang terdeteksi")
    mp = pd.DataFrame({"Kolom kanonik": list(ctx["mapping"].keys()),
                       "Kolom di file": [v if v else "— (tidak ada)" for v in ctx["mapping"].values()]})
    st.dataframe(mp, hide_index=True, use_container_width=True)
    ui.glossary()
    with st.expander("🎓 Ringkasan metodologi untuk dosen"):
        st.markdown(f"""
1. **Data**: ulasan Google Play aplikasi LMS/student portal; kolom dideteksi dari nama aktual, bukan diasumsikan.
2. **Label sentimen = PROXY** dari rating bintang (≤{ctx['cfg']['neg_max']} negatif, ≥{ctx['cfg']['pos_min']} positif, selainnya netral). Rating **tidak pernah** menjadi fitur model (mencegah leakage).
3. **Preprocessing** mempertahankan negasi, emoji, dan istilah teknis; tidak ada terjemahan otomatis. Tiga varian dibandingkan secara empiris.
4. **Split** train/val/test reproducible (seed={ctx['cfg']['seed']}); teks duplikat dihapus sebelum split; tersedia split *group-by-app* untuk menguji generalisasi ke aplikasi yang belum dilihat.
5. **Model**: baseline mayoritas, TF-IDF + LR / Linear SVM / Complement NB, dan varian char n-gram. Evaluasi: accuracy, macro P/R/F1, per kelas, confusion matrix, K-fold CV.
6. **Aspek** = *weak label* dari leksikon dwibahasa (multi-label + Other/Unclear). Skor model aspek mengukur kesesuaian dengan aturan, **bukan** validitas; validasi memakai sampel anotasi manusia (tab Evaluation).
7. **ABSA** = sentimen tingkat klausa (leksikon + negasi); klausa tanpa polaritas mewarisi sentimen review dan ditandai. Belum divalidasi tanpa anotasi.
8. **Keterbatasan**: label proxy bisa tidak selaras dengan teks (mis. rating 5 tapi berisi keluhan); perbandingan Indonesia vs internasional dipengaruhi perbedaan aplikasi, bahasa, populasi pengguna, dan jumlah review.
""")


def render_eda(ctx):
    t1, t2, t3, t4 = st.tabs(["🧪 Quality & Language", "⭐ Rating & Sentiment", "📐 Variable vs Target", "🌍 Indonesia vs International"])
    df, fdf = ctx["df"], ctx["fdf"]
    with t1:
        rep, miss = quality_report(fdf)
        ui.metric_row([("Teks kosong", rep["empty_text"]), ("Rating kosong/tak valid", rep["missing_rating"]),
                       ("Duplikat (app+teks+rating)", rep["duplicate_rows"]), ("Duplikat teks", rep["duplicate_texts"]),
                       ("Pendek (≤2 kata)", rep["short_text"]), ("Emoji-only", rep["emoji_only"])])
        if rep["empty_text"] or rep["duplicate_texts"]:
            ui.warn("Teks kosong dikeluarkan dari modeling dan duplikat teks dihapus sebelum split.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Nilai kosong per kolom (%)**")
            st.dataframe(miss, hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Review per aplikasi** (flag jika < 30)")
            per_app = fdf.groupby(["app", "group"]).agg(n=("review_id", "count"), mean_rating=("rating", "mean")).reset_index()
            per_app["cukup (≥30)"] = per_app["n"] >= 30
            st.dataframe(per_app.sort_values("n"), hide_index=True, use_container_width=True, height=300)
        st.markdown("**Bahasa**")
        c1, c2 = st.columns(2)
        with c1:
            ld = fdf["language"].value_counts().rename_axis("language").reset_index(name="n")
            ui.bar(ld, "language:N", "n:Q", title="Kolom bahasa di dataset", height=240)
        with c2:
            dd = fdf["lang_detected"].value_counts().rename_axis("terdeteksi").reset_index(name="n")
            ui.bar(dd, "terdeteksi:N", "n:Q", title="Deteksi heuristik (id/en/mixed/unknown)", height=240)
        st.caption("Deteksi bahasa = heuristik leksikon ringan; teks sangat pendek/emoji menjadi 'unknown'. Periksa manual contoh 'mixed'.")
    with t2:
        r = fdf["rating"].value_counts().sort_index().rename_axis("rating").reset_index(name="n")
        s = fdf["sent_label"].value_counts().reindex(SENT_LABELS).fillna(0).rename_axis("sentiment").reset_index(name="n")
        c1, c2 = st.columns(2)
        with c1:
            ui.bar(r.astype({"rating": str}), "rating:N", "n:Q", title="Distribusi rating bintang", height=280)
        with c2:
            ui.bar(s, "sentiment:N", "n:Q", color="sentiment:N", sent_color=True, title="Label proxy sentimen", height=280)
        tot = s["n"].sum()
        if tot:
            mx = s["n"].max() / tot
            (ui.warn if mx > 0.5 else ui.good)(f"Kelas terbesar = {mx:.0%} data. "
                                                  + ("Kelas tidak seimbang: gunakan macro-F1, bukan accuracy saja." if mx > 0.5 else "Distribusi cukup seimbang."))
        ui.warn("Sentimen di sini adalah **proxy dari rating bintang**, bukan label manusia. Rating dan isi teks bisa tidak konsisten.")
        if fdf["month"].notna().any():
            tr = fdf.dropna(subset=["month"]).groupby("month").agg(n=("review_id", "count"), mean_rating=("rating", "mean")).reset_index()
            c1, c2 = st.columns(2)
            with c1:
                ui.line(tr, "month:T", "n:Q", title="Volume review per bulan")
            with c2:
                ui.line(tr, "month:T", "mean_rating:Q", title="Rata-rata rating per bulan")
    with t3:
        var_vs_target(fdf)
    with t4:
        comparison(fdf)


def comparison(fdf):
    if fdf["group"].nunique() < 2:
        st.info("Pilih ≥ 2 kelompok (cakupan aplikasi) di filter sidebar untuk membandingkan.")
    g = fdf.groupby("group")
    tab = g.agg(review=("review_id", "count"), aplikasi=("app", "nunique"), institusi=("institution", "nunique"),
                rata_rating=("rating", "mean"), median_rating=("rating", "median"),
                review_per_app=("review_id", lambda x: len(x) / max(1, fdf.loc[x.index, "app"].nunique()))).round(3)
    st.dataframe(tab, use_container_width=True)
    sent = fdf.groupby(["group", "sent_label"]).size().rename("n").reset_index()
    sent["persen"] = sent["n"] / sent.groupby("group")["n"].transform("sum") * 100
    asp = fdf.explode("aspects").groupby(["group", "aspects"]).size().rename("n").reset_index()
    asp["persen_review"] = asp["n"] / asp["group"].map(g.size()) * 100
    c1, c2 = st.columns(2)
    with c1:
        ui.bar(sent, "group:N", "persen:Q", color="sent_label:N", sent_color=True, title="Komposisi sentimen proxy (%)")
    with c2:
        ui.bar(asp, "aspects:N", "persen_review:Q", color="group:N", stack=False, title="% review yang menyebut aspek")
    rt = fdf.groupby(["group", "rating"]).size().rename("n").reset_index()
    rt["persen"] = rt["n"] / rt.groupby("group")["n"].transform("sum") * 100
    ui.bar(rt.astype({"rating": str}), "rating:N", "persen:Q", color="group:N", stack=False, title="Distribusi rating per kelompok (%)")
    ui.warn("**Batasan perbandingan**: (1) jumlah review & aplikasi tidak sama; (2) aplikasi berbeda fitur/skala pengguna; "
            "(3) bahasa & budaya memengaruhi gaya rating; (4) sampel Google Play tidak mewakili seluruh mahasiswa; "
            "(5) sentimen berasal dari rating (proxy). Selisih rata-rata bukan bukti sebab-akibat.")


def _lump(series, top=12):
    keep = series.value_counts().head(top).index
    return series.where(series.isin(keep), "Other (lumped)")


def var_vs_target(fdf):
    d = fdf[fdf["sent_label"].notna()]
    if d["sent_label"].nunique() < 2:
        st.info("Butuh minimal 2 kelas sentimen pada data terfilter.")
        return
    st.markdown("### Categorical variable vs. Sentiment (chi-square test)")
    cats = {"Kelompok (Indonesia/Intl)": "group", "Bahasa (kolom dataset)": "language", "Bahasa terdeteksi": "lang_detected",
            "Institusi": "institution", "Aplikasi": "app", "Aspek utama": "aspect_primary"}
    if d["version"].notna().any():
        cats["Versi aplikasi"] = "version"
    label = st.selectbox("Choose a categorical variable", list(cats))
    col = cats[label]
    x = _lump(d[col].astype(str))
    ct = pd.crosstab(x, d["sent_label"]).reindex(columns=SENT_LABELS, fill_value=0)
    ct = ct.loc[:, ct.sum() > 0]
    c1, c2 = st.columns([3, 1])
    with c1:
        st.dataframe(ct, use_container_width=True)
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, pval, dof, exp = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        c2.metric("Chi-square statistic", f"{chi2:.2f}")
        c2.metric("p-value", f"{pval:.4g}")
        c2.caption(("Significant" if pval < 0.05 else "Not significant") + f" at 0.05 · Cramér's V = {v:.2f}")
        if (exp < 5).mean() > 0.2:
            ui.warn("Lebih dari 20% sel punya frekuensi harapan < 5; hasil chi-square kurang andal.")
    else:
        c2.info("Butuh ≥ 2 kategori & ≥ 2 kelas.")
    pct = st.checkbox("Tampilkan sebagai persen", value=False)
    long = ct.reset_index().melt(id_vars=ct.index.name or "index", var_name="sentiment", value_name="n")
    xname = long.columns[0]
    ch = alt.Chart(long).mark_bar().encode(
        x=alt.X(f"{xname}:N", sort="-y", title=label), y=alt.Y("n:Q", stack="normalize" if pct else True, title="Persen" if pct else "Jumlah"),
        color=alt.Color("sentiment:N", scale=ui.SENT_COLORS), tooltip=[xname, "sentiment", "n"]).properties(height=340)
    st.altair_chart(ch, use_container_width=True)
    st.caption("Asosiasi statistik, bukan sebab-akibat. 'Signifikan' pada n besar bisa tetap berefek kecil – lihat Cramér's V.")

    st.markdown("### Numeric variable vs. Sentiment")
    nums = {"Panjang teks (karakter)": "n_chars", "Jumlah kata": "n_tokens", "Jumlah emoji": "n_emoji",
            "Jumlah tanda seru": "n_exclaim", "Likes": "likes", "⚠ Rating (sumber label → leakage)": "rating"}
    nl = st.selectbox("Choose a numeric variable", list(nums))
    nc = nums[nl]
    dd = d.dropna(subset=[nc])
    if nc == "rating":
        ui.warn("Rating adalah sumber label proxy; hubungannya dengan sentimen 'sempurna' by construction. Jangan dipakai sebagai fitur (lihat Skenario A/B).")
    if dd.empty:
        st.info("Variabel ini kosong pada data terfilter.")
    else:
        box = alt.Chart(dd).mark_boxplot(extent="min-max").encode(
            x=alt.X("sent_label:N", sort=SENT_LABELS, title="sentiment"), y=alt.Y(f"{nc}:Q", title=nl),
            color=alt.Color("sent_label:N", scale=ui.SENT_COLORS, legend=None)).properties(height=320)
        c1, c2 = st.columns([3, 1])
        c1.altair_chart(box, use_container_width=True)
        groups = [g[nc].values for _, g in dd.groupby("sent_label") if len(g) > 1]
        if len(groups) >= 2 and np.ptp(dd[nc].values) > 0:
            try:
                h, p = stats.kruskal(*groups)
                c2.metric("Kruskal-Wallis H", f"{h:.2f}")
                c2.metric("p-value", f"{p:.4g}")
                c2.caption("Significant at 0.05" if p < 0.05 else "Not significant at 0.05")
            except ValueError:
                c2.info("Uji tidak dapat dihitung (nilai identik).")
        c2.dataframe(dd.groupby("sent_label")[nc].describe()[["count", "mean", "50%"]].round(2))
    st.markdown("### Korelasi antar variabel numerik (Spearman)")
    cols = [c for c in ["n_chars", "n_tokens", "n_emoji", "n_exclaim", "likes", "rating"] if d[c].notna().sum() > 2 and d[c].nunique() > 1]
    if len(cols) >= 2:
        cm = d[cols].corr(method="spearman")
        lg = cm.reset_index().melt("index")
        base = alt.Chart(lg).encode(x=alt.X("index:N", title=None), y=alt.Y("variable:N", title=None))
        st.altair_chart(base.mark_rect().encode(color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]))) +
                        base.mark_text().encode(text=alt.Text("value:Q", format=".2f")), use_container_width=True)
