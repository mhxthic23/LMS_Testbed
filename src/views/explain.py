import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .. import ui
from ..config import SENT_LABELS
from ..labeling import absa, detect_aspects, single_row
from ..modeling import feature_frame, top_features


def render_explain(ctx):
    st.subheader("Explainability & Error Analysis")
    out = st.session_state.get("sent")
    if not out:
        st.info("Latih model di tab Modeling terlebih dahulu.")
        return
    res = out["results"]
    names = [n for n in res if not n.startswith("Baseline")]
    if not names:
        st.info("Pilih setidaknya satu model non-baseline.")
        return
    best = max(names, key=lambda n: res[n]["test"]["f1_macro"])
    name = st.selectbox("Model", names, index=names.index(best))
    r = res[name]
    t0, t1, t2, t3 = st.tabs(["🎮 Coba prediksi", "🔑 Fitur berpengaruh", "❌ Prediksi salah", "🔍 Perlu diperiksa manusia"])
    with t0:
        playground(ctx, out, name)
    with t1:
        tf = top_features(r["pipe"])
        if tf is None or "char" in name:
            st.info("Penjelasan fitur ditampilkan untuk model linear/NB berbasis kata (n-gram char sulit dibaca).")
        else:
            for c in tf["class"].unique():
                with st.expander(f"Kata/fitur terkuat untuk kelas: {c}", expanded=True):
                    ui.bar(tf[tf["class"] == c], "weight:Q", "feature:N", horizontal=True, height=320)
            if out["feat_cfg"]["scenario"] == "B":
                ui.warn("Skenario B: perhatikan fitur [num] rating — ia mendominasi prediksi karena bocor dari label.")
            st.caption("Bobot besar = kata mendorong prediksi ke kelas tsb (korelasi, bukan sebab-akibat). Perhatikan token NEG_* dan emo_*.")
    te = out["test_df"].copy()
    te["pred"], te["confidence"] = r["test_pred"], r["test_conf"]
    te["benar"] = te["pred"] == te["sent_label"]
    cols = ["review_id", "app", "review_text", "rating", "sent_label", "pred", "confidence", "lex_sentiment"]
    with t2:
        wrong = te[~te["benar"]]
        st.metric("Salah di test set", f"{len(wrong)} / {len(te)}")
        pair = wrong.groupby(["sent_label", "pred"]).size().rename("n").reset_index().sort_values("n", ascending=False)
        st.dataframe(pair.rename(columns={"sent_label": "label proxy", "pred": "prediksi"}), hide_index=True, use_container_width=True)
        st.dataframe(wrong.sort_values("confidence", ascending=False)[cols].head(200), hide_index=True, use_container_width=True)
        st.caption("Urut dari keyakinan tertinggi: salah-yakin sering menandakan label proxy yang keliru, bukan model yang buruk.")
    with t3:
        thr = st.slider("Ambang keyakinan rendah (kuantil)", 0.05, 0.5, 0.2)
        low = te["confidence"] <= te["confidence"].quantile(thr)
        conflict = ((te["sent_label"] == "positive") & (te["lex_sentiment"] == "negative")) | \
                   ((te["sent_label"] == "negative") & (te["lex_sentiment"] == "positive"))
        short = te["n_tokens"] <= 2
        te["alasan"] = ""
        te.loc[low, "alasan"] += "ambigu/keyakinan rendah; "
        te.loc[conflict, "alasan"] += "rating vs leksikon bertentangan; "
        te.loc[short | te["emoji_only"], "alasan"] += "teks sangat pendek/emoji; "
        te.loc[~te["benar"] & ~low, "alasan"] += "salah prediksi; "
        flag = te[te["alasan"] != ""]
        st.metric("Review ditandai", f"{len(flag)} ({len(flag) / max(1, len(te)):.0%} test set)")
        st.dataframe(flag[cols + ["alasan"]], hide_index=True, use_container_width=True)
        st.download_button("⬇ Ekspor daftar review untuk dianotasi (CSV)", flag[cols + ["alasan"]].to_csv(index=False).encode("utf-8-sig"),
                           "review_to_check.csv", "text/csv")


def render_explorer(ctx):
    st.subheader("Review Explorer & Ekspor")
    fdf = ctx["fdf"]
    q = st.text_input("Cari kata/frasa pada review (di dalam hasil filter)", placeholder="login, error, tugas ...")
    v = fdf[fdf["review_text"].str.contains(q, case=False, na=False, regex=False)] if q else fdf
    out = st.session_state.get("sent")
    if out:
        names = list(out["results"])
        pick = st.selectbox("Tambahkan prediksi dari model", ["(tidak)"] + names)
        if pick != "(tidak)":
            v = v.copy()
            vv = v.copy()
            vv["clean_model"] = vv["clean_model"].where(vv["clean_model"] != "", "kosong")
            v["pred_sentiment"] = out["results"][pick]["pipe"].predict(feature_frame(vv))
            v["in_test_set"] = v["review_id"].isin(out["test_df"]["review_id"])
            st.caption("Prediksi pada data train bersifat in-sample (terlalu optimis); gunakan kolom in_test_set untuk evaluasi jujur.")
    cols = ["review_id", "institution", "app", "group", "lang_detected", "rating", "date", "review_text", "sent_label",
            "lex_sentiment", "aspects_str"] + [c for c in ["pred_sentiment", "in_test_set"] if c in v.columns]
    st.dataframe(v[cols], hide_index=True, use_container_width=True, height=420)
    st.caption(f"{len(v):,} review ditampilkan.")
    st.download_button("⬇ Ekspor hasil (CSV)", v[cols].to_csv(index=False).encode("utf-8-sig"), "lms_review_analysis.csv", "text/csv")


def playground(ctx, out, model_name):
    cfg = ctx["cfg"]
    st.markdown("Ketik review sendiri (Indonesia/Inggris/campuran) dan lihat prediksi, aspek, serta sentimen per aspek.")
    ex = st.selectbox("Contoh cepat", ["(tulis sendiri)", "aplikasinya error dan susah di remember password",
                                       "tampilan bagus, tapi login sering gagal", "not working after update, cant open 😡", "mantap, tugas mudah diupload 👍"])
    text = st.text_area("Teks review", value="" if ex == "(tulis sendiri)" else ex, height=90)
    rating = None
    if out["feat_cfg"]["scenario"] == "B":
        rating = st.slider("Rating (dipakai model karena Skenario B)", 1, 5, 3)
    if not text.strip():
        st.caption("Masukkan teks untuk melihat hasil.")
        return
    row = single_row(text, float(rating) if rating else np.nan, cfg["mode"], cfg["neg_max"], cfg["pos_min"])
    pipe = out["results"][model_name]["pipe"]
    X = feature_frame(row.assign(clean_model=row["clean_model"].where(row["clean_model"] != "", "kosong")))
    pred = pipe.predict(X)[0]
    clf = pipe.named_steps["clf"]
    sc = pipe.predict_proba(X)[0] if hasattr(clf, "predict_proba") else pipe.decision_function(X)[0]
    c1, c2 = st.columns([1, 2])
    c1.metric("Prediksi sentimen", pred)
    c1.caption(f"Bahasa terdeteksi: {row['lang_detected'].iloc[0]} · Teks bersih: `{row['clean_model'].iloc[0]}`")
    probs = pd.DataFrame({"kelas": list(clf.classes_), "skor": sc})
    c2.altair_chart(alt.Chart(probs).mark_bar().encode(x=alt.X("skor:Q", title="probabilitas" if hasattr(clf, "predict_proba") else "skor margin"),
                    y=alt.Y("kelas:N", sort=SENT_LABELS), color=alt.Color("kelas:N", scale=ui.SENT_COLORS, legend=None)).properties(height=140),
                    use_container_width=True)
    st.markdown("**Aspek terdeteksi:** " + ", ".join(detect_aspects(text)))
    ab = absa(text, pred)
    if ab:
        st.dataframe(pd.DataFrame(ab), hide_index=True, use_container_width=True)
    st.caption("Aspek & ABSA berbasis leksikon (belum tervalidasi). Prediksi model bisa salah pada teks sangat pendek atau sarkasme.")
