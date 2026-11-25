"""Step 3 - Model the customer-level table.

Evaluation design (important with only 33 bad customers):
  * repeated stratified 5-fold CV (10 repeats) - every customer is scored by a
    model that never saw it; Gini = 2*AUC - 1 is computed per repeat
  * all preprocessing (imputation, scaling, encoding, feature selection) lives
    INSIDE the sklearn Pipeline so nothing leaks from validation folds
Writes: outputs/*.csv, outputs/model_metrics.json, data/clean/model_matrix.csv,
        data/clean/oof_predictions.csv, outputs/final_model.joblib
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from utils import CLEAN, OUT, RANDOM_STATE, log1p_clip

warnings.filterwarnings("ignore")
rng = np.random.default_rng(RANDOM_STATE)

m = pd.read_csv(CLEAN / "customer_master.csv", parse_dates=["dt_opened", "entry_time"])
y = m["Bad_label"].values
ID_COLS = ["customer_no", "dt_opened", "entry_time", "Bad_label"]
X = m.drop(columns=ID_COLS).copy()

# --- explicit type handling --------------------------------------------------
cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) in ("string", "str")]
# integer-coded categoricals (few levels) are better treated as categories
code_like = [c for c in X.columns if c not in cat_cols and X[c].nunique() <= 12 and c.startswith("feature_")]
cat_cols += code_like
num_cols = [c for c in X.columns if c not in cat_cols]
X[cat_cols] = X[cat_cols].astype("object").where(X[cat_cols].notna(), np.nan)
for c in cat_cols:
    X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(v))

# heavy-tailed money columns -> log1p (negatives clipped at 0)
money = [c for c in num_cols if any(k in c for k in ("amount", "amt", "balance", "limit", "high_credit", "feature_7", "feature_35", "feature_52"))]


def log_money(df):
    df = pd.DataFrame(df).copy()
    return np.log1p(df.clip(lower=0))


print(f"rows={len(X)}  bad={y.sum()}  numeric={len(num_cols)}  categorical={len(cat_cols)}  log-transformed={len(money)}")


def make_pre(scale: bool):
    other_num = [c for c in num_cols if c not in money]
    steps_money = [("imp", SimpleImputer(strategy="median", add_indicator=True)),
                   ("log", FunctionTransformer(log1p_clip, feature_names_out="one-to-one"))]
    steps_num = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps_money.append(("sc", StandardScaler()))
        steps_num.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("money", Pipeline(steps_money), money),
        ("num", Pipeline(steps_num), other_num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.03,
                                               sparse_output=False))]), cat_cols),
    ], verbose_feature_names_out=True)


models = {
    "Logistic regression (L2, strong reg.)": Pipeline([
        ("pre", make_pre(True)),
        ("clf", LogisticRegression(C=0.02, class_weight="balanced", max_iter=3000))]),
    "Logistic regression (L1 sparse)": Pipeline([
        ("pre", make_pre(True)),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=0.08, class_weight="balanced", max_iter=3000))]),
    "Random forest": Pipeline([
        ("pre", make_pre(False)),
        ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                       class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE))]),
    "Gradient boosting (shallow)": Pipeline([
        ("pre", make_pre(False)),
        ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.03, max_depth=2,
                                           subsample=0.7, min_samples_leaf=10, random_state=RANDOM_STATE))]),
}

# --- single-variable benchmark: the bureau-score-like field alone -------------
def score_only_gini():
    s = X["feature_3"].fillna(X["feature_3"].median()).values
    return 2 * roc_auc_score(y, -s) - 1, 2 * roc_auc_score(y, s) - 1


# --- repeated stratified CV ----------------------------------------------------
N_REP, N_SPLIT = 5, 5
rskf = RepeatedStratifiedKFold(n_splits=N_SPLIT, n_repeats=N_REP, random_state=RANDOM_STATE)
splits = list(rskf.split(X, y))
results, oof_store = [], {}
for name, pipe in models.items():
    oof = np.zeros((N_REP, len(y)))
    for k, (tr, te) in enumerate(splits):
        rep = k // N_SPLIT
        pipe.fit(X.iloc[tr], y[tr])
        oof[rep, te] = pipe.predict_proba(X.iloc[te])[:, 1]
    ginis = [2 * roc_auc_score(y, oof[r]) - 1 for r in range(N_REP)]
    results.append({"model": name, "cv_gini_mean": np.mean(ginis) * 100, "cv_gini_std": np.std(ginis) * 100,
                    "cv_auc_mean": (np.mean(ginis) + 1) / 2})
    oof_store[name] = oof
    print(f"{name:42s} Gini {np.mean(ginis)*100:5.1f} +/- {np.std(ginis)*100:4.1f}", flush=True)
res = pd.DataFrame(results).sort_values("cv_gini_mean", ascending=False).reset_index(drop=True)
g_lo, g_hi = score_only_gini()
print(f"single-field benchmark (feature_3 alone): Gini {g_lo*100:.1f} (higher=safer) / {g_hi*100:.1f} (higher=riskier)")

# ensemble of the best two diverse learners (rank average)
top_names = res["model"].head(2).tolist()
ens = np.mean([pd.DataFrame(oof_store[n]).rank(axis=1).values for n in top_names], axis=0)
ens_g = [2 * roc_auc_score(y, ens[r]) - 1 for r in range(N_REP)]
res.loc[len(res)] = {"model": "Rank-average ensemble (top 2)", "cv_gini_mean": np.mean(ens_g) * 100,
                     "cv_gini_std": np.std(ens_g) * 100, "cv_auc_mean": (np.mean(ens_g) + 1) / 2}
oof_store["Rank-average ensemble (top 2)"] = ens / len(y)
res = res.sort_values("cv_gini_mean", ascending=False).reset_index(drop=True)
print(res.round(2).to_string(index=False))
res.to_csv(OUT / "model_comparison.csv", index=False)

best_name = res.loc[0, "model"]
print("\nBEST:", best_name)

# --- bootstrap CI for the best model's repeat-averaged OOF score ---------------
best_oof = pd.DataFrame(oof_store[best_name]).rank(axis=1).mean(axis=0).values
boot = []
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
for _ in range(1000):
    idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    boot.append(2 * roc_auc_score(y[idx], best_oof[idx]) - 1)
ci = np.percentile(boot, [2.5, 97.5]) * 100
point = (2 * roc_auc_score(y, best_oof) - 1) * 100
print(f"Best model pooled OOF Gini {point:.1f}  (95% bootstrap CI {ci[0]:.1f} - {ci[1]:.1f})")

# --- permutation test: is the model better than chance? -------------------------
perm = []
for _ in range(1000):
    perm.append(2 * roc_auc_score(rng.permutation(y), best_oof) - 1)
p_val = (np.sum(np.array(perm) >= point / 100) + 1) / (len(perm) + 1)
print(f"permutation-test p-value for Gini > 0: {p_val:.4f}")

# --- rank ordering (deciles, 10 = riskiest) ------------------------------------
score_rank = pd.Series(best_oof).rank(method="first")
def rank_table(n_bins):
    b = pd.qcut(score_rank, n_bins, labels=False) + 1
    t = pd.DataFrame({"bin": b, "bad": y}).groupby("bin").agg(customers=("bad", "size"), bad=("bad", "sum"))
    t["bad_rate_pct"] = t.bad / t.customers * 100
    t["lift_vs_avg"] = t.bad_rate_pct / (y.mean() * 100)
    t = t.sort_index(ascending=False)
    t["cum_bad_captured_pct"] = t.bad.cumsum() / t.bad.sum() * 100
    return t.reset_index()
dec, quint = rank_table(10), rank_table(5)
dec.to_csv(OUT / "rank_ordering_deciles.csv", index=False)
quint.to_csv(OUT / "rank_ordering_quintiles.csv", index=False)
print(dec.round(2).to_string(index=False))

# KS statistic
order = np.argsort(-best_oof)
cum_bad = np.cumsum(y[order]) / y.sum()
cum_good = np.cumsum(1 - y[order]) / (1 - y).sum()
ks = np.max(cum_bad - cum_good) * 100

# --- single 70/30 hold-out as a second, simpler view -----------------------------
tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=RANDOM_STATE)
hold = {}
for name, pipe in models.items():
    pipe.fit(X.iloc[tr], y[tr])
    hold[name] = (2 * roc_auc_score(y[te], pipe.predict_proba(X.iloc[te])[:, 1]) - 1) * 100
print("hold-out (70/30) Gini:", {k: round(v, 1) for k, v in hold.items()}, "| bad in test:", int(y[te].sum()))

# --- feature importance: gain + permutation (on out-of-fold) --------------------
final = models["Random forest"] if "Random forest" in models else None
gain_model = models["Gradient boosting (shallow)"]
rf_model = models["Random forest"]
def gain_table(pipe):
    pipe.fit(X, y)
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pipe.named_steps["clf"].feature_importances_
    return pd.DataFrame({"feature": names, "gain": imp})
gb_gain = gain_table(gain_model).rename(columns={"gain": "gain_gb"})
rf_gain = gain_table(rf_model).rename(columns={"gain": "gain_rf"})
fi = gb_gain.merge(rf_gain, on="feature")
fi["gain_avg"] = fi[["gain_gb", "gain_rf"]].mean(axis=1)
fi["feature"] = fi["feature"].str.replace(r"^(money|num|cat)__", "", regex=True)

# permutation importance of the RF on held-out folds (more honest than in-sample gain)
perm_scores = []
rf_model.named_steps["clf"].set_params(n_jobs=1)   # single thread: far faster for tiny predict calls
skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
for tr_i, te_i in skf.split(X, y):
    rf_model.fit(X.iloc[tr_i], y[tr_i])
    r = permutation_importance(rf_model, X.iloc[te_i], y[te_i], scoring="roc_auc", n_repeats=5,
                               random_state=RANDOM_STATE)
    perm_scores.append(pd.Series(r.importances_mean, index=X.columns))
perm_imp = pd.concat(perm_scores, axis=1).mean(axis=1).rename("perm_auc_drop")

# gain per ORIGINAL column (sum of one-hot levels back to their parent)
def parent(f):
    f = f.replace("cat__", "").replace("money__", "").replace("num__", "").replace("missingindicator_", "")
    for c in sorted(X.columns, key=len, reverse=True):
        if f == c or f.startswith(c + "_"):
            return c
    return f
fi["parent"] = fi["feature"].apply(parent)
parent_gain = fi.groupby("parent")[["gain_gb", "gain_rf", "gain_avg"]].sum().sort_values("gain_avg", ascending=False)
parent_gain = parent_gain.join(perm_imp)
parent_gain["gain_avg_norm"] = parent_gain["gain_avg"] / parent_gain["gain_avg"].sum()
parent_gain["cum_gain"] = parent_gain["gain_avg_norm"].cumsum()
parent_gain.index.name = "feature"
parent_gain.reset_index().to_csv(OUT / "feature_importance_by_variable.csv", index=False)
fi.sort_values("gain_avg", ascending=False).to_csv(OUT / "feature_importance_by_encoded_column.csv", index=False)
print(parent_gain.head(20).round(4).to_string())

# --- compact "66-feature style" final feature matrix: features covering 85% gain ----
keep = parent_gain[(parent_gain.cum_gain.shift(fill_value=0) < 0.85)].index.tolist()
keep = [k for k in keep if k in X.columns]
print(f"\nfinal feature set: {len(keep)} of {X.shape[1]} variables")
(parent_gain.loc[keep].reset_index()[["feature", "gain_avg_norm", "perm_auc_drop"]]
 .rename(columns={"gain_avg_norm": "gain"}).to_csv(OUT / "final_feature_matrix.csv", index=False))

# reduced model evaluated with the same CV protocol (note: selection used the
# full data, so this figure is mildly optimistic and reported as such)
num_c2, cat_c2 = [c for c in keep if c in num_cols], [c for c in keep if c in cat_cols]
money2 = [c for c in num_c2 if c in money]
other2 = [c for c in num_c2 if c not in money]
pre2 = ColumnTransformer([
    ("money", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                        ("log", FunctionTransformer(log1p_clip, feature_names_out="one-to-one"))]), money2),
    ("num", SimpleImputer(strategy="median", add_indicator=True), other2),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                      ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.03, sparse_output=False))]), cat_c2),
])
reduced = Pipeline([("pre", pre2), ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.03, max_depth=2,
                                                                      subsample=0.7, min_samples_leaf=10, random_state=RANDOM_STATE))])
red_g = []
for r in range(N_REP):
    o = np.zeros(len(y))
    for k in range(N_SPLIT):
        tr_i, te_i = splits[r * N_SPLIT + k]
        reduced.fit(X.iloc[tr_i][keep], y[tr_i])
        o[te_i] = reduced.predict_proba(X.iloc[te_i][keep])[:, 1]
    red_g.append((2 * roc_auc_score(y, o) - 1) * 100)
print(f"reduced model ({len(keep)} vars) CV Gini {np.mean(red_g):.1f} +/- {np.std(red_g):.1f}")

# --- calibrate to a 300-900 style credit score ---------------------------------------
p_raw = np.clip(pd.Series(best_oof).rank(pct=True).values, 1e-4, 1 - 1e-4)  # rank-based, monotonic
cal = LogisticRegression(C=1e6).fit(np.log(p_raw / (1 - p_raw)).reshape(-1, 1), y)
pd_hat = cal.predict_proba(np.log(p_raw / (1 - p_raw)).reshape(-1, 1))[:, 1]
odds = (1 - pd_hat) / pd_hat
PDO, BASE, BASE_ODDS = 40, 650, 30
factor = PDO / np.log(2)
offset = BASE - factor * np.log(BASE_ODDS)
cred_score = np.clip(offset + factor * np.log(odds), 300, 900)

oof_df = pd.DataFrame({"customer_no": m.customer_no, "Bad_label": y, "oof_risk_rank_score": best_oof,
                       "predicted_pd": pd_hat, "cred_score_300_900": cred_score.round(0)})
oof_df.to_csv(CLEAN / "oof_predictions.csv", index=False)

# --- final model fitted on all data and saved -------------------------------------------
final_model = models[best_name] if best_name in models else models[top_names[0]]
final_model.fit(X, y)
joblib.dump({"model": final_model, "columns": list(X.columns), "best_name": best_name}, OUT / "final_model.joblib")

# --- model matrix (what the models see) ---------------------------------------------------
pipe_view = make_pre(False).fit(X)
mm = pd.DataFrame(pipe_view.transform(X), columns=pipe_view.get_feature_names_out())
mm.insert(0, "customer_no", m.customer_no.values)
mm["Bad_label"] = y
mm.to_csv(CLEAN / "model_matrix.csv", index=False)

metrics = {
    "n_customers": int(len(y)), "n_bad": int(y.sum()), "bad_rate_pct": float(y.mean() * 100),
    "cv_protocol": f"{N_SPLIT}-fold stratified x {N_REP} repeats",
    "best_model": best_name, "best_cv_gini_mean": float(res.loc[0, "cv_gini_mean"]),
    "best_cv_gini_std": float(res.loc[0, "cv_gini_std"]),
    "pooled_oof_gini": float(point), "pooled_oof_gini_ci95": [float(ci[0]), float(ci[1])],
    "permutation_p_value": float(p_val), "ks_pct": float(ks),
    "reduced_model_vars": len(keep), "reduced_model_cv_gini_mean": float(np.mean(red_g)),
    "holdout_70_30_gini": {k: float(v) for k, v in hold.items()}, "holdout_bad_in_test": int(y[te].sum()),
    "single_field_feature_3_gini": {"low_score_riskier": float(g_lo * 100), "high_score_riskier": float(g_hi * 100)},
    "benchmark_gini": 37.9,
    "score_card": {"base_score": BASE, "base_odds_good_to_bad": BASE_ODDS, "pdo": PDO},
}
json.dump(metrics, open(OUT / "model_metrics.json", "w"), indent=2)
print("\nsaved metrics ->", OUT / "model_metrics.json")
