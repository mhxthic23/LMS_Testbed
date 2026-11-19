"""Training, split bebas-kebocoran, evaluasi sentimen & aspek."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler, label_binarize
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.svm import LinearSVC

from .config import ASPECT_NAMES, OTHER, SENT_LABELS

AVAILABLE_MODELS = ["Baseline (majority class)", "TF-IDF + Logistic Regression", "TF-IDF + Linear SVM",
                    "TF-IDF + Complement NB", "TF-IDF word+char n-gram + Logistic Regression"]
SPLIT_STRATEGIES = {
    "Stratified random (deduplikasi teks)": "random",
    "Group by aplikasi (aplikasi test tidak ada di train)": "group",
}


def _tfidf(min_df=2):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True, max_features=40000,
                           token_pattern=r"(?u)\b[\w']+\b")


NUM_BASE = ["n_chars", "n_tokens", "n_emoji", "n_exclaim"]   # turunan teks (aman)
NUM_LEAK = ["rating", "likes"]                               # Skenario B: rating = sumber label -> LEAKAGE
CAT_COLS = ["lang_detected", "group"]
SCALERS = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler,
           "None (tanpa scaling)": None}
SCENARIOS = {"A: Strict - hanya teks (buang rating & likes)": "A",
             "B: Leaky demo - pertahankan rating & likes": "B"}


def feature_frame(m):
    """DataFrame fitur mentah: kolom 'text' + numerik + kategorikal. Rating/likes hanya dipakai jika skenario B."""
    m = m.reset_index(drop=True)
    f = pd.DataFrame({"text": m["clean_model"].values})
    for c in NUM_BASE:
        f[c] = pd.to_numeric(m[c], errors="coerce").fillna(0).values
    f["rating"] = pd.to_numeric(m["rating"], errors="coerce").fillna(0).values
    f["likes"] = pd.to_numeric(m["likes"], errors="coerce").fillna(0).values
    f["lang_detected"], f["group"] = m["lang_detected"].values, m["group"].values
    return f


def make_features(scenario="A", scaler="StandardScaler", use_cat=False, n_docs=1000, char=False, nonneg=False):
    md = 2 if n_docs >= 500 else 1
    text_vec = make_union(_tfidf(md), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=md,
                                                      sublinear_tf=True, max_features=60000)) if char else _tfidf(md)
    sname = "MinMaxScaler" if (nonneg and scaler in ("StandardScaler", "RobustScaler")) else scaler
    scaler_obj = SCALERS[sname]() if SCALERS[sname] else "passthrough"
    num = NUM_BASE + (NUM_LEAK if scenario == "B" else [])
    tr = [("tfidf", text_vec, "text"), ("num", scaler_obj, num)]
    if use_cat:
        tr.append(("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS))
    return ColumnTransformer(tr, sparse_threshold=1.0)


def build_model(name, seed, n_docs=1000, scenario="A", scaler="StandardScaler", use_cat=False):
    feats = make_features(scenario, scaler, use_cat, n_docs, char="char" in name, nonneg="Complement" in name)
    if name.startswith("Baseline"):
        clf = DummyClassifier(strategy="most_frequent")
    elif "Logistic" in name:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    elif "SVM" in name:
        clf = LinearSVC(class_weight="balanced", random_state=seed)
    else:
        clf = ComplementNB()
    return Pipeline([("feat", feats), ("clf", clf)])


def prepare_model_df(df, dedup=True):
    m = df[df["sent_label"].notna() & (df["clean_model"].str.len() > 0)]
    info = {"labeled_nonempty": len(m)}
    if dedup:
        before = len(m)
        m = m.drop_duplicates(subset="clean_min")
        info["duplicates_removed"] = before - len(m)
    return m.reset_index(drop=True), info


def make_splits(m, strategy, test_size, val_size, seed):
    """Return index arrays (train, val, test). Group split => aplikasi tidak bocor lintas split."""
    idx = np.arange(len(m))
    y, groups = m["sent_label"].values, m["app"].values
    val_rel = val_size / (1 - test_size)
    if strategy == "group" and len(set(groups)) >= 5:
        tv, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(idx, y, groups))
        tr_i, va_i = next(GroupShuffleSplit(1, test_size=val_rel, random_state=seed).split(tv, y[tv], groups[tv]))
        return tv[tr_i], tv[va_i], te
    tv, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    tr, va = train_test_split(tv, test_size=val_rel, stratify=y[tv], random_state=seed)
    return tr, va, te


def calc_metrics(y_true, y_pred, labels=SENT_LABELS):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "recall_neg": r[0], "precision_neg": p[0],
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p.mean(), "recall_macro": r.mean(), "f1_macro": f.mean(),
        "per_class": pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": s}),
        "cm": confusion_matrix(y_true, y_pred, labels=labels),
    }


def confidence(pipe, X):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X).max(axis=1)
    s = np.sort(pipe.decision_function(X), axis=1)
    return s[:, -1] - s[:, -2]


def class_scores(pipe, X):
    clf = pipe.named_steps["clf"]
    return pipe.predict_proba(X) if hasattr(clf, "predict_proba") else pipe.decision_function(X)


def auc_ovr(y_true, scores, classes):
    try:
        if list(classes) != SENT_LABELS or len(set(y_true)) < 3:
            return np.nan
        return roc_auc_score(label_binarize(y_true, classes=SENT_LABELS), scores, average="macro")
    except Exception:
        return np.nan


def train_sentiment(m, strategy, test_size, val_size, cv_folds, seed, model_names, scenario="A",
                    scaler="StandardScaler", use_cat=False, progress=None):
    tr, va, te = make_splits(m, strategy, test_size, val_size, seed)
    X, y, groups = feature_frame(m), m["sent_label"].reset_index(drop=True), m["app"].reset_index(drop=True)
    trva = np.concatenate([tr, va])
    if strategy == "group" and groups.iloc[trva].nunique() >= cv_folds:
        cv, cv_groups = GroupKFold(n_splits=cv_folds), groups.iloc[trva]
    else:
        cv, cv_groups = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed), None
    results = {}
    for i, name in enumerate(model_names):
        if progress:
            progress(i / len(model_names), f"Melatih {name} ...")
        pipe = build_model(name, seed, len(m), scenario, scaler, use_cat)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        res = {"pipe": pipe}
        for tag, ix in (("val", va), ("test", te)):
            pred = pipe.predict(X.iloc[ix])
            res[tag] = calc_metrics(y.iloc[ix], pred)
        res["test_pred"], res["test_conf"] = pipe.predict(X.iloc[te]), confidence(pipe, X.iloc[te])
        res["test_scores"], res["test_y"] = class_scores(pipe, X.iloc[te]), y.iloc[te].values
        res["auc"] = auc_ovr(res["test_y"], res["test_scores"], pipe.named_steps["clf"].classes_)
        try:
            cvr = cross_validate(build_model(name, seed, len(m), scenario, scaler, use_cat), X.iloc[trva], y.iloc[trva], cv=cv,
                                 groups=cv_groups, scoring={"acc": "accuracy", "f1": "f1_macro"})
            res["cv"] = {"acc_mean": cvr["test_acc"].mean(), "acc_std": cvr["test_acc"].std(),
                         "f1_mean": cvr["test_f1"].mean(), "f1_std": cvr["test_f1"].std(), "folds": cv_folds}
        except Exception as e:  # CV bisa gagal pada data kecil/kelas jarang
            res["cv"] = {"error": str(e)}
        results[name] = res
    if progress:
        progress(1.0, "Selesai")
    sizes = {"train": len(tr), "val": len(va), "test": len(te)}
    apps = {k: m["app"].iloc[v].nunique() for k, v in (("train", tr), ("val", va), ("test", te))}
    test_df = m.iloc[te].copy()
    return {"results": results, "sizes": sizes, "apps": apps, "test_df": test_df, "split": (tr, va, te),
            "strategy": strategy, "seed": seed, "feat_cfg": {"scenario": scenario, "scaler": scaler, "use_cat": use_cat}, "class_dist": y.value_counts().reindex(SENT_LABELS).fillna(0)}


def aspect_matrix(m):
    Y = pd.DataFrame(0, index=m.index, columns=ASPECT_NAMES + [OTHER])
    for i, asp in zip(m.index, m["aspects"]):
        for a in asp:
            Y.loc[i, a] = 1
    return Y


def train_aspects(m, split, seed):
    """Multi-label OvR LR pada weak labels (leksikon). Skor = kesesuaian dengan aturan, BUKAN validitas."""
    tr, va, te = split
    trva = np.concatenate([tr, va])
    Y = aspect_matrix(m)
    X = m["clean_model"]
    keep = [c for c in Y.columns if 2 <= Y.iloc[trva][c].sum() < len(trva)]
    if not keep:
        return None
    pipe = Pipeline([("tfidf", _tfidf(1 if len(m) < 500 else 2)),
                     ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced",
                                                                    random_state=seed)))])
    pipe.fit(X.iloc[trva], Y.iloc[trva][keep].values)
    pred = pipe.predict(X.iloc[te])
    yt = Y.iloc[te][keep].values
    p, r, f, s = precision_recall_fscore_support(yt, pred, zero_division=0)
    table = pd.DataFrame({"aspect": keep, "precision": p, "recall": r, "f1": f, "support": s.astype(int)})
    return {"table": table, "micro_f1": f1_score(yt, pred, average="micro", zero_division=0),
            "macro_f1": f1_score(yt, pred, average="macro", zero_division=0), "pipe": pipe, "keep": keep,
            "skipped": [c for c in Y.columns if c not in keep]}


def top_features(pipe, n=15):
    """Fitur paling berpengaruh per kelas (model linear / NB)."""
    clf, vec = pipe.named_steps["clf"], pipe.named_steps["feat"]
    names = np.array([n.replace("tfidf__", "").replace("num__", "[num] ").replace("cat__", "[cat] ")
                      for n in vec.get_feature_names_out()])
    if hasattr(clf, "coef_"):
        W = clf.coef_
        if W.shape[0] == 1:
            W = np.vstack([-W[0], W[0]])
        classes = clf.classes_ if W.shape[0] == len(clf.classes_) else ["neg", "pos"]
    elif hasattr(clf, "feature_log_prob_"):
        W, classes = clf.feature_log_prob_, clf.classes_
        W = W - W.mean(axis=0)
    else:
        return None
    rows = []
    for ci, c in enumerate(classes):
        for j in np.argsort(W[ci])[::-1][:n]:
            rows.append({"class": c, "feature": names[j], "weight": float(W[ci][j])})
    return pd.DataFrame(rows)
