"""LMS Review Mining Testbed – entry point Streamlit."""
import streamlit as st

from src import ui
from src.config import COLUMN_CANDIDATES, SEED
from src.data import apply_filters, detect_columns, find_local_file, read_any, standardize
from src.labeling import enrich
from src.modeling import AVAILABLE_MODELS, SCALERS, SCENARIOS, SPLIT_STRATEGIES
from src.text import PREPROC_MODES
from src.views import aspects, explain, modeling_eval, overview_eda, preprocessing

st.set_page_config(page_title="LMS Review Mining Testbed", page_icon="📊", layout="wide")
ui.setup()


@st.cache_data(show_spinner="Membaca dataset ...")
def load_raw(path):
    return read_any(path)


@st.cache_data(show_spinner="Memproses teks & label ...")
def build(raw, mapping_items, mode, neg_max, pos_min):
    return enrich(standardize(raw, dict(mapping_items)), mode, neg_max, pos_min)


# =============================================================== SIDEBAR
sb = st.sidebar
sb.title("⚙️ Testbed Configuration")
up = sb.file_uploader("Upload LMS review dataset (optional)", type=["csv", "xlsx", "xls"],
                      help="CSV/XLSX Google Play review. Jika kosong, dibaca dari data/processed/lms_reviews_clean.csv")
map_slot = sb.container()

try:
    if up is not None:
        raw, source = read_any(up), f"Upload: {up.name}"
    else:
        path = find_local_file()
        if path is None:
            st.error("Dataset tidak ditemukan. Unggah file di sidebar atau letakkan di `data/processed/lms_reviews_clean.csv`.")
            st.stop()
        raw, source = load_raw(path), f"Lokal: {path}"
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()
sb.caption(source)

detected = detect_columns(raw)
with map_slot.expander("Pemetaan kolom"):
    opts = ["(tidak ada)"] + list(raw.columns)
    mapping = {}
    for canon in COLUMN_CANDIDATES:
        cur = detected[canon] if detected[canon] else "(tidak ada)"
        ch = st.selectbox(canon, opts, index=opts.index(cur), key=f"map_{canon}")
        mapping[canon] = None if ch == "(tidak ada)" else ch
missing_req = [c for c in ("review_text", "rating") if not mapping[c]]
if missing_req:
    st.error(f"Kolom wajib belum terpetakan: {missing_req}. Atur di sidebar › Pemetaan kolom. Kolom tersedia: {list(raw.columns)}")
    st.stop()

neg_max = sb.slider("Negative threshold on rating (≤)", 1, 3, 2)
pos_min = sb.slider("Positive threshold on rating (≥)", 3, 5, 4)
if pos_min <= neg_max:
    sb.error("Ambang positif harus > ambang negatif.")
    st.stop()
sb.divider()
scenario_label = sb.radio("Leakage scenario", list(SCENARIOS))
scaler = sb.selectbox("Feature scaler", list(SCALERS))
test_size = sb.slider("Test size (holdout)", 0.10, 0.40, 0.20, 0.05)
val_size = sb.slider("Validation size", 0.10, 0.30, 0.15, 0.05)
cv = sb.slider("Cross-validation folds (K)", 3, 10, 5)
sb.divider()
models = sb.multiselect("Models to train", AVAILABLE_MODELS, default=AVAILABLE_MODELS[:4])
split_label = sb.selectbox("Split strategy", list(SPLIT_STRATEGIES))
mode = sb.selectbox("Text preprocessing", list(PREPROC_MODES))
use_cat = sb.checkbox("Sertakan metadata kategorikal (bahasa & kelompok, one-hot)", value=False)
train_filtered = sb.checkbox("Latih hanya pada data terfilter", value=False)
seed = sb.number_input("Random seed", 0, 99999, SEED)

df = build(raw, tuple(mapping.items()), mode, neg_max, pos_min)

sb.divider()
sb.subheader("🔎 Filter")
inst = sb.multiselect("Institusi", sorted(df["institution"].unique()), default=sorted(df["institution"].unique()))
app_opts = sorted(df[df["institution"].isin(inst)]["app"].unique())
apps = sb.multiselect("Aplikasi", app_opts, default=app_opts)
grp_opts = sorted(df["group"].unique())
groups = sb.multiselect("Negara/kelompok", grp_opts, default=grp_opts)
lang_opts = sorted(df["lang_detected"].unique())
langs = sb.multiselect("Bahasa (terdeteksi)", lang_opts, default=lang_opts)
rat_opts = sorted(df["rating"].dropna().unique())
ratings = sb.multiselect("Rating", rat_opts, default=rat_opts)
date_range = None
if df["date"].notna().any():
    d0, d1 = df["date"].min().date(), df["date"].max().date()
    date_range = sb.date_input("Rentang tanggal", (d0, d1), min_value=d0, max_value=d1)
    if len(date_range) != 2:
        date_range = (d0, d1)
search = sb.text_input("Cari kata di review")

fdf = apply_filters(df, inst, apps, groups, langs, date_range, ratings, search)
cfg = dict(seed=int(seed), mode=mode, neg_max=neg_max, pos_min=pos_min, split=SPLIT_STRATEGIES[split_label],
           split_label=split_label, test_size=test_size, val_size=val_size, cv=cv, models=models,
           train_filtered=train_filtered, scenario=SCENARIOS[scenario_label], scenario_label=scenario_label,
           scaler=scaler, use_cat=use_cat)
ctx = dict(raw=raw, df=df, fdf=fdf, mapping=mapping, cfg=cfg, train_df=fdf if train_filtered else df)

# =============================================================== MAIN
ui.hero("📊 LMS Review Mining Testbed",
        "End-to-end pipeline pada ulasan Google Play aplikasi LMS / student portal (Indonesia vs Internasional) — "
        "Overview · EDA · Preprocessing · Modeling · Evaluation, semua dapat diatur dari sidebar.")
if fdf.empty:
    st.warning("Tidak ada review yang cocok dengan filter. Longgarkan filter di sidebar.")
    st.stop()

tabs = st.tabs(["📋 Overview", "🔍 EDA", "🛠️ Preprocessing", "🚀 Modeling", "📊 Evaluation",
                "🧭 Aspect Explorer", "💡 Explainability", "📝 Review Explorer"])
with tabs[0]:
    overview_eda.render_overview(ctx)
with tabs[1]:
    overview_eda.render_eda(ctx)
with tabs[2]:
    preprocessing.render(ctx)
with tabs[3]:
    if models:
        modeling_eval.render_modeling(ctx)
    else:
        st.info("Pilih minimal satu model di sidebar.")
with tabs[4]:
    modeling_eval.render_evaluation(ctx)
with tabs[5]:
    aspects.render(ctx)
with tabs[6]:
    explain.render_explain(ctx)
with tabs[7]:
    explain.render_explorer(ctx)
