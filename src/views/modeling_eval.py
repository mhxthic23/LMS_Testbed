import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support, roc_curve
from sklearn.preprocessing import MultiLabelBinarizer

from .. import ui
from ..config import ASPECT_NAMES, OTHER, SENT_LABELS
from ..modeling import calc_metrics, prepare_model_df, train_aspects, train_sentiment

NUM_COLS = ["Accuracy", "Recall (negative)", "Precision (negative)", "Macro F1", "ROC-AUC"]
METRIC_CHOICES = ["Accuracy", "Recall (negative)", "Precision (negative)", "Macro F1", "ROC-AUC",
                  "Precision (macro)", "Recall (macro)", "F1 val"]


def _signature(ctx):
    c = ctx["cfg"]
    return (c["mode"], c["seed"], c["split"], c["test_size"], c["val_size"], c["cv"], tuple(c["models"]),
            c["neg_max"], c["pos_min"], c["train_filtered"], c["scenario"], c["scaler"], c["use_cat"], len(ctx["train_df"]))


def leaderboard(out):
    rows = []
    for name, r in out["results"].items():
        cv = r["cv"]
        rows.append({"Model": name, "Accuracy": r["test"]["accuracy"], "Recall (negative)": r["test"]["recall_neg"],
                     "Precision (negative)": r["test"]["precision_neg"], "Macro F1": r["test"]["f1_macro"],
                     "ROC-AUC": r["auc"], "Precision (macro)": r["test"]["precision_macro"],
                     "Recall (macro)": r["test"]["recall_macro"], "F1 val": r["val"]["f1_macro"],
                     "CV Acc (mean±std)": f"{cv['acc_mean']:.3f} ± {cv['acc_std']:.3f}" if "acc_mean" in cv else "n/a",
                     "CV Macro-F1 (mean±std)": f"{cv['f1_mean']:.3f} ± {cv['f1_std']:.3f}" if "f1_mean" in cv else "n/a"})
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)


def render_modeling(ctx):
    cfg = ctx["cfg"]
    m, info = prepare_model_df(ctx["train_df"])
    st.markdown(f"Training on **{len(cfg['models'])}** selected model(s): " + ", ".join(cfg["models"]))
    c1, c2, c3 = st.columns(3)
    c1.metric("Data berlabel & non-kosong", f"{info['labeled_nonempty']:,}")
    c2.metric("Duplikat teks dihapus", f"{info.get('duplicates_removed', 0):,}")
    c3.metric("Siap dimodelkan", f"{len(m):,}")
    st.caption(f"Skenario **{cfg['scenario_label']}** · Scaler **{cfg['scaler']}** · Split **{cfg['split_label']}** · seed **{cfg['seed']}** · "
               f"test {cfg['test_size']:.0%} · val {cfg['val_size']:.0%} · CV {cfg['cv']}-fold")
    if cfg["scenario"] == "B":
        ui.warn("Skenario B memasukkan **rating** (sumber label proxy) dan likes sebagai fitur → **data leakage**. "
                "Skor akan tampak sangat tinggi tetapi tidak valid. Bandingkan dengan Skenario A.")
    else:
        ui.good("Skenario A: fitur hanya teks + statistik teks; rating & likes dibuang. TF-IDF dan scaler di-fit di dalam pipeline (hanya data train/fold).")
    if len(m) < 60 or m["sent_label"].nunique() < 2 or m["sent_label"].value_counts().min() < max(6, cfg["cv"]):
        st.error("Data terlalu sedikit atau satu kelas terlalu jarang untuk training/CV. Longgarkan filter atau ubah ambang rating.")
        return
    if st.button("🚀 Train models", type="primary"):
        bar = st.progress(0.0, text="Menyiapkan ...")
        try:
            out = train_sentiment(m, cfg["split"], cfg["test_size"], cfg["val_size"], cfg["cv"], cfg["seed"],
                                  cfg["models"], cfg["scenario"], cfg["scaler"], cfg["use_cat"],
                                  progress=lambda p, t: bar.progress(p, text=t))
            out["aspect"] = train_aspects(m, out["split"], cfg["seed"])
            out["info"], out["sig"], out["m"] = info, _signature(ctx), m
            st.session_state["sent"] = out
        except Exception as e:
            st.error(f"Training gagal: {e}")
        bar.empty()
    out = st.session_state.get("sent")
    if not out:
        st.info("Belum ada hasil. Atur konfigurasi di sidebar lalu klik **Train models**.")
        return
    if out["sig"] != _signature(ctx):
        ui.warn("Konfigurasi/filter berubah sejak training terakhir. Tabel di bawah masih hasil lama; klik Train models lagi.")
    st.markdown("### Quick leaderboard")
    lb = leaderboard(out)
    fmt = {c: "{:.4f}" for c in NUM_COLS}
    st.dataframe(lb[["Model"] + NUM_COLS + ["CV Acc (mean±std)"]].style.format(fmt, na_rep="n/a")
                 .highlight_max(subset=NUM_COLS, color="#c9f0d4"), hide_index=True, use_container_width=True)
    sz = out["sizes"]
    tr, va, te = out["split"]
    ov = len(set(out["m"]["app"].iloc[tr]) & set(out["m"]["app"].iloc[te]))
    st.caption(f"Split: train {sz['train']:,} · val {sz['val']:,} · test {sz['test']:,} · aplikasi per split {out['apps']} · "
               f"aplikasi di train & test sekaligus: {ov}. Sel hijau = terbaik per kolom. Bandingkan dengan baseline; satu skor ≠ model akurat.")
    if out["aspect"]:
        a = out["aspect"]
        st.markdown(f"### Tahap 2 – Klasifikasi aspek (multi-label) · micro-F1 {a['micro_f1']:.3f} · macro-F1 {a['macro_f1']:.3f}")
        ui.warn("Target = weak label dari leksikon. Skor ini mengukur seberapa baik model meniru aturan, **bukan** kebenaran aspek.")
        st.dataframe(a["table"].round(3), hide_index=True, use_container_width=True)
        if a["skipped"]:
            st.caption("Dilewati (positif < 2 di train): " + ", ".join(a["skipped"]))
    st.caption("Tahap 3 – ABSA (sentimen per aspek) tersedia di tab Aspect Explorer.")


def render_evaluation(ctx):
    out = st.session_state.get("sent")
    st.markdown("## Model comparison")
    if not out:
        st.info("Latih model terlebih dahulu di tab Modeling.")
        return
    lb = leaderboard(out)
    metric = st.selectbox("Metric to compare", METRIC_CHOICES)
    d = lb[["Model", metric]].dropna()
    ch = alt.Chart(d).mark_bar().encode(x=alt.X(metric + ":Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Model:N", sort="-x", title=None),
                                        tooltip=["Model", metric]).properties(height=60 * len(d) + 60)
    st.altair_chart(ch, use_container_width=True)
    res = out["results"]
    base = next((r for n, r in res.items() if n.startswith("Baseline")), None)
    best = lb.iloc[0]
    if base is not None and not best["Model"].startswith("Baseline"):
        dd = best["Macro F1"] - base["test"]["f1_macro"]
        (ui.good if dd > 0.02 else ui.warn)(f"Model terbaik ({best['Model']}) vs baseline: selisih macro-F1 = {dd:+.3f}. "
                                            + ("Ada peningkatan nyata." if dd > 0.02 else "Peningkatan kecil; hati-hati menyimpulkan."))
    if out["feat_cfg"]["scenario"] == "B":
        ui.warn("Hasil ini berasal dari **Skenario B (leaky)**: rating ikut sebagai fitur. Jangan dilaporkan sebagai performa model sebenarnya.")
    st.download_button("⬇ Ekspor leaderboard (CSV)", lb.to_csv(index=False).encode("utf-8-sig"), "leaderboard_sentiment.csv", "text/csv")
    t1, t2, t3, t4 = st.tabs(["🔬 Detail model", "📉 ROC curve", "🧩 Aspek", "🧑‍🏫 Validasi anotasi"])
    with t1:
        detail(out, lb)
    with t2:
        roc_tab(out)
    with t3:
        if not out.get("aspect"):
            st.info("Data aspek belum cukup untuk dilatih.")
        else:
            a = out["aspect"]
            ui.metric_row([("Micro-F1", f"{a['micro_f1']:.3f}"), ("Macro-F1", f"{a['macro_f1']:.3f}")])
            ui.bar(a["table"].melt("aspect", ["precision", "recall", "f1"]), "aspect:N", "value:Q", color="variable:N",
                   stack=False, title="Metrik per aspek (terhadap weak label)", height=320)
            st.dataframe(a["table"].round(3), hide_index=True, use_container_width=True)
            ui.warn("Tanpa ground-truth manusia, model aspek/ABSA **tidak boleh** dinyatakan tervalidasi.")
    with t4:
        annotation(ctx)


def detail(out, lb):
    res = out["results"]
    name = st.selectbox("Model", list(res.keys()), index=list(res.keys()).index(lb.iloc[0]["Model"]))
    r = res[name]
    split = st.radio("Data evaluasi", ["test", "val"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Metrik per kelas**")
        st.dataframe(r[split]["per_class"].round(3), hide_index=True, use_container_width=True)
        cv = r["cv"]
        if "f1_mean" in cv:
            st.markdown(f"**{cv['folds']}-fold CV:** accuracy {cv['acc_mean']:.3f} ± {cv['acc_std']:.3f} · macro-F1 {cv['f1_mean']:.3f} ± {cv['f1_std']:.3f}")
        else:
            st.caption(f"CV tidak tersedia: {cv.get('error')}")
        cd = out["class_dist"].rename_axis("kelas").reset_index(name="n")
        ui.bar(cd, "kelas:N", "n:Q", color="kelas:N", sent_color=True, height=180, title="Distribusi kelas (data model)")
    with c2:
        st.markdown("**Confusion matrix**")
        ui.heatmap_cm(r[split]["cm"], SENT_LABELS)
    st.caption("Accuracy bisa tinggi hanya karena kelas dominan; lihat recall kelas *neutral* (rating 3 ambigu) dan macro-F1. "
               "Selisih besar test vs CV → varians tinggi atau split group yang sulit.")


def roc_tab(out):
    res = out["results"]
    name = st.selectbox("Model untuk ROC", list(res.keys()), key="roc_model")
    r = res[name]
    if np.isnan(r["auc"]):
        st.info("ROC-AUC tidak tersedia (test set tidak memuat ketiga kelas atau model tidak lengkap).")
        return
    rows = []
    for i, c in enumerate(SENT_LABELS):
        fpr, tpr, _ = roc_curve((r["test_y"] == c).astype(int), r["test_scores"][:, i])
        rows += [{"FPR": a, "TPR": b, "kelas": c} for a, b in zip(fpr, tpr)]
    d = pd.DataFrame(rows)
    line = alt.Chart(d).mark_line().encode(x="FPR:Q", y="TPR:Q", color=alt.Color("kelas:N", scale=ui.SENT_COLORS))
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
    st.altair_chart((line + diag).properties(height=340), use_container_width=True)
    st.caption(f"ROC-AUC macro one-vs-rest = {r['auc']:.3f} (0.5 = acak). AUC tinggi tidak menjamin prediksi kelas netral baik.")


def annotation(ctx):
    fdf, cfg = ctx["fdf"], ctx["cfg"]
    st.markdown("""**Alur anotasi manusia (ground-truth kecil)**
1. Unduh sampel acak (reproducible dengan seed). 2. Isi `human_aspects` (pisahkan dengan `;`, nama aspek persis atau `Other/Unclear`) dan `human_sentiment` (negative/neutral/positive).
3. Unggah kembali untuk menghitung metrik. Idealnya ≥ 2 anotator dan hitung agreement.""")
    st.caption("Nama aspek valid: " + " | ".join(ASPECT_NAMES + [OTHER]))
    nmax = max(30, min(300, len(fdf)))
    n = st.slider("Ukuran sampel", 30, nmax, min(100, nmax), step=10) if nmax > 30 else 30
    pool = fdf[fdf["review_text"].str.strip() != ""]
    if len(pool):
        s = pool.sample(min(n, len(pool)), random_state=cfg["seed"])
        tmpl = pd.DataFrame({"review_id": s["review_id"], "app": s["app"], "review_text": s["review_text"], "rating": s["rating"],
                             "pred_aspects_rule": s["aspects_str"], "pred_sentiment_proxy": s["sent_label"],
                             "human_aspects": "", "human_sentiment": ""})
        st.download_button("⬇ Unduh template anotasi (CSV)", tmpl.to_csv(index=False).encode("utf-8-sig"), "annotation_template.csv", "text/csv")
    up = st.file_uploader("Unggah file anotasi yang sudah diisi", type=["csv"], key="annot")
    if not up:
        return
    ann = pd.read_csv(up)
    need = {"review_id", "human_aspects", "human_sentiment"}
    if not need.issubset(ann.columns):
        st.error(f"Kolom wajib tidak lengkap. Dibutuhkan: {sorted(need)}")
        return
    ann["review_id"] = ann["review_id"].astype(str)
    j = ann.merge(ctx["df"][["review_id", "aspects", "sent_label"]], on="review_id", how="inner")
    ja = j[j["human_aspects"].notna() & (j["human_aspects"].astype(str).str.strip() != "")]
    if ja.empty:
        st.warning("Belum ada baris dengan human_aspects terisi.")
    else:
        valid = set(ASPECT_NAMES + [OTHER])
        H = ja["human_aspects"].map(lambda s: [x.strip() for x in str(s).split(";") if x.strip() in valid])
        mlb = MultiLabelBinarizer(classes=ASPECT_NAMES + [OTHER])
        yh, yp = mlb.fit_transform(H), mlb.transform(ja["aspects"])
        p, r, f, sup = precision_recall_fscore_support(yh, yp, zero_division=0)
        st.markdown(f"**Aspek (aturan vs manusia), n = {len(ja)}** · micro-F1 {f1_score(yh, yp, average='micro', zero_division=0):.3f}")
        st.dataframe(pd.DataFrame({"aspect": mlb.classes_, "precision": p, "recall": r, "f1": f, "support_human": sup}).round(3),
                     hide_index=True, use_container_width=True)
    js = j[j["human_sentiment"].isin(SENT_LABELS)]
    if not js.empty:
        pr = js["sent_label"].fillna("neutral")
        mt = calc_metrics(js["human_sentiment"], pr)
        st.markdown(f"**Sentimen manusia vs label proxy rating, n = {len(js)}:** accuracy {mt['accuracy']:.3f} · macro-F1 {mt['f1_macro']:.3f} · Cohen's κ {cohen_kappa_score(js['human_sentiment'], pr):.3f}")
        st.dataframe(mt["per_class"].round(3), hide_index=True, use_container_width=True)
        st.caption("Skor rendah berarti rating bintang kurang valid sebagai pengganti label sentimen teks.")
