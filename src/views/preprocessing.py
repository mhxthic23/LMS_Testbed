import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .. import ui
from ..modeling import NUM_BASE, NUM_LEAK, _tfidf, feature_frame, make_features, prepare_model_df
from ..text import PREPROC_MODES, clean_text, detect_language


@st.cache_data(show_spinner=False)
def compare_modes(texts, labels, seed, test_size):
    out = []
    X = pd.Series(texts)
    y = pd.Series(labels)
    tr, te = train_test_split(range(len(X)), test_size=test_size, stratify=y, random_state=seed)
    for mode in PREPROC_MODES:
        Xc = X.map(lambda t: clean_text(t, mode))
        keep = Xc.str.len() > 0
        trk, tek = [i for i in tr if keep.iloc[i]], [i for i in te if keep.iloc[i]]
        pipe_vec = _tfidf(2 if len(X) >= 500 else 1)
        Xtr = pipe_vec.fit_transform(Xc.iloc[trk])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed).fit(Xtr, y.iloc[trk])
        pred = clf.predict(pipe_vec.transform(Xc.iloc[tek]))
        out.append({"opsi": mode, "macro_F1_test": f1_score(y.iloc[tek], pred, average="macro"),
                    "n_fitur": Xtr.shape[1], "n_kosong_setelah_clean": int((~keep).sum())})
    return pd.DataFrame(out)


def render(ctx):
    cfg = ctx["cfg"]
    st.markdown(f"**Scenario:** {cfg['scenario_label']}")
    st.markdown(f"**Scaler:** {cfg['scaler']} · **Test size:** {cfg['test_size']} · **CV folds:** {cfg['cv']} · **Preprocessing teks:** {cfg['mode']}")
    t1, t2 = st.tabs(["🧮 Engineered feature matrix", "🧹 Text cleaning (multilingual & code-mixed)"])
    with t1:
        feature_matrix(ctx)
    with t2:
        text_cleaning(ctx)


def feature_matrix(ctx):
    cfg = ctx["cfg"]
    m, info = prepare_model_df(ctx["fdf"])
    if len(m) < 20:
        st.info("Data terfilter terlalu sedikit untuk membangun matriks fitur.")
        return
    X = feature_frame(m)
    ct = make_features(cfg["scenario"], cfg["scaler"], cfg["use_cat"], len(m))
    with st.spinner("Membangun matriks fitur ..."):
        M = ct.fit_transform(X).tocsr()
    names = np.array(ct.get_feature_names_out())
    is_tf = np.array([n.startswith("tfidf__") for n in names])
    is_num = np.array([n.startswith("num__") for n in names])
    is_cat = np.array([n.startswith("cat__") for n in names])
    st.markdown("### Engineered feature matrix")
    st.markdown(f"Shape after encoding: **{M.shape[0]:,} rows × {M.shape[1]:,} features** "
                f"(TF-IDF {is_tf.sum():,} + numerik {is_num.sum()} + kategorikal {is_cat.sum()})")
    st.caption(f"{info.get('duplicates_removed', 0):,} duplikat teks sudah dibuang; baris kosong/tanpa label tidak ikut.")
    c1, c2 = st.columns(2)
    n_rows = c1.slider("Jumlah baris ditampilkan", 5, 50, 10)
    k = c2.slider("Jumlah kolom TF-IDF teratas (berdasar frekuensi dokumen)", 5, 60, 15)
    idx_tf = np.where(is_tf)[0]
    dfreq = np.asarray((M[:, idx_tf] > 0).sum(axis=0)).ravel()
    top = idx_tf[np.argsort(-dfreq)[:k]]
    cols = np.concatenate([np.where(is_num | is_cat)[0], top])
    disp = pd.DataFrame(M[:n_rows][:, cols].toarray(), columns=[n.split("__", 1)[1] for n in names[cols]])
    disp.insert(0, "label", m["sent_label"].values[:n_rows])
    st.dataframe(disp.round(3), use_container_width=True)
    if cfg["scenario"] == "B":
        ui.warn("Skenario B: kolom **rating** dan **likes** ikut sebagai fitur (leakage). Lihat tabel korelasi di bawah.")
    st.markdown("### Efek scaler pada fitur numerik")
    num_cols = NUM_BASE + (NUM_LEAK if cfg["scenario"] == "B" else [])
    raw = X[num_cols].describe().loc[["mean", "std", "min", "max"]].T
    sc = pd.DataFrame(M[:, np.where(is_num)[0]].toarray(), columns=num_cols).describe().loc[["mean", "std", "min", "max"]].T
    cmp = raw.join(sc, lsuffix=" (asli)", rsuffix=" (scaled)")
    st.dataframe(cmp.round(3), use_container_width=True)
    st.caption("TF-IDF tidak di-scale (sudah ternormalisasi). Scaler di-fit hanya pada data train di dalam pipeline saat Train models.")
    st.markdown("### Cek kebocoran: korelasi fitur numerik dengan label")
    y = m["sent_label"].map({"negative": -1, "neutral": 0, "positive": 1}).reset_index(drop=True)
    chk = pd.DataFrame({"fitur": NUM_BASE + NUM_LEAK,
                        "|korelasi Spearman|": [abs(X[c].corr(y, method="spearman")) for c in NUM_BASE + NUM_LEAK]})
    chk["dipakai di skenario ini"] = chk["fitur"].isin(num_cols)
    chk["status"] = np.where(chk["|korelasi Spearman|"] > 0.6, "⚠ sangat tinggi → indikasi leakage", "wajar")
    st.dataframe(chk.round(3), hide_index=True, use_container_width=True)
    st.caption("Fitur yang berkorelasi hampir sempurna dengan label hampir pasti 'sumber' label, bukan prediktor sah.")


def text_cleaning(ctx):
    fdf, cfg = ctx["fdf"], ctx["cfg"]
    st.subheader("Multilingual & Code-Mixed Preprocessing")
    st.markdown(f"**Opsi aktif:** `{cfg['mode']}` — {PREPROC_MODES[cfg['mode']]}")
    st.caption("Tidak ada terjemahan otomatis. Slang hanya dinormalkan dalam bahasa aslinya (mis. *gk* → *tidak*).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teks kosong", int((fdf["review_text"].str.strip() == "").sum()))
    c2.metric("Emoji-only", int(fdf["emoji_only"].sum()))
    c3.metric("Mengandung emoji", int(fdf["has_emoji"].sum()))
    c4.metric("Code-mixed (heuristik)", int((fdf["lang_detected"] == "mixed").sum()))

    st.markdown("#### Contoh sebelum / sesudah")
    kind = st.radio("Jenis contoh", ["Acak", "Code-mixed", "Mengandung negasi", "Emoji", "Sangat pendek"], horizontal=True)
    pool = fdf[fdf["review_text"].str.strip() != ""]
    if kind == "Code-mixed":
        pool = pool[pool["lang_detected"] == "mixed"]
    elif kind == "Mengandung negasi":
        pool = pool[pool["clean_min"].str.contains(r"\b(?:tidak|gak|ga|nggak|not|never|belum|cant|can't|don't)\b")]
    elif kind == "Emoji":
        pool = pool[pool["has_emoji"]]
    elif kind == "Sangat pendek":
        pool = pool[pool["n_tokens"] <= 2]
    if pool.empty:
        st.info("Tidak ada contoh untuk kategori ini pada filter saat ini.")
    else:
        s = pool.sample(min(10, len(pool)), random_state=cfg["seed"])
        view = pd.DataFrame({"Asli": s["review_text"], "A-Minimal": s["review_text"].map(lambda t: clean_text(t, list(PREPROC_MODES)[0])),
                             "B-Normalisasi+Negasi": s["review_text"].map(lambda t: clean_text(t, list(PREPROC_MODES)[1])),
                             "C-Agresif": s["review_text"].map(lambda t: clean_text(t, list(PREPROC_MODES)[2])),
                             "Bahasa": s["lang_detected"]})
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.caption("Perhatikan opsi C: negasi ('tidak', 'not') ikut terhapus sehingga makna bisa terbalik.")

    st.markdown("#### Perbandingan opsi preprocessing (Logistic Regression, test set)")
    lab = fdf[fdf["sent_label"].notna() & (fdf["review_text"].str.strip() != "")].drop_duplicates("clean_min")
    if len(lab) < 100 or lab["sent_label"].nunique() < 2 or lab["sent_label"].value_counts().min() < 5:
        st.info("Data terfilter terlalu sedikit/tidak seimbang untuk perbandingan (butuh ≥100 review & ≥5 per kelas).")
        return
    if st.button("▶ Jalankan perbandingan opsi preprocessing"):
        with st.spinner("Membandingkan 3 opsi ..."):
            res = compare_modes(tuple(lab["review_text"]), tuple(lab["sent_label"]), cfg["seed"], cfg["test_size"])
        st.session_state["prep_cmp"] = res
    if "prep_cmp" in st.session_state:
        res = st.session_state["prep_cmp"]
        st.dataframe(res.round(4), hide_index=True, use_container_width=True)
        ui.bar(res, "opsi:N", "macro_F1_test:Q", title="Macro-F1 per opsi preprocessing", height=240)
        st.caption("Satu split saja → selisih kecil bisa hanya noise. Gunakan CV di tab Modeling untuk keputusan akhir.")
