"""
CREDIT RISK SCORECARD - COMPLETE PROJECT IN ONE FILE
====================================================
Predicts which credit-card customers become "bad" (30+ days past due) and
builds a 300-900 credit score.

Run from the project folder (needs data/raw/accounts_raw.csv,
demographics_raw.csv, enquiries_raw.csv):

    pip install -r requirements.txt
    python pipeline.py          # about 5-10 minutes (modelling is the slow part)

Sections
    0 Setup / helpers      3 EDA charts
    1 Data cleaning        4 Modelling + validation + feature gain
    2 Feature engineering  5 Model charts   6 Report card

Outputs: data/clean/*.csv, charts/*.png, outputs/*.csv|json|joblib
To load the saved model elsewhere, define the same log1p_clip() function first.
"""


# ============================================================================
# SECTION 0 - Setup, paths and helper functions
# ============================================================================
from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent      # running as a script
except NameError:
    ROOT = Path.cwd()                              # running in a notebook
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CHARTS = ROOT / "charts"
OUT = ROOT / "outputs"
for _p in (CLEAN, CHARTS, OUT):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Standard Indian credit-bureau product codes (assumption - the client did not
# supply a code book, so these are documented as an assumption in the README).
PRODUCT_LABELS = {
    0: "Other", 1: "Auto loan", 2: "Housing loan", 3: "Property loan",
    4: "Loan against shares", 5: "Personal loan", 6: "Consumer loan",
    7: "Gold loan", 8: "Education loan", 9: "Loan to professional",
    10: "Credit card", 11: "Leasing", 12: "Overdraft", 13: "Two-wheeler loan",
    15: "Loan against deposits", 17: "Commercial vehicle loan",
    32: "Other (32)", 51: "Business loan", 52: "Business loan (priority)",
    53: "Business loan (other)", 54: "Business loan (other)",
    55: "Business loan (other)", 56: "Business loan (other)",
    57: "Business loan (other)", 40: "Other (40)",
}
SECURED_CODES = {1, 2, 3, 4, 7, 13, 15, 17}


def parse_date(s: pd.Series, max_year: int = 2016) -> pd.Series:
    """Parse dd-Mon-yy strings. Two-digit years that land in the future
    (e.g. '03-Sep-65' -> 2065) are rolled back by 100 years."""
    out = pd.to_datetime(s, format="%d-%b-%y", errors="coerce")
    future = out.dt.year > max_year
    out = out.where(~future, out - pd.DateOffset(years=100))
    return out


def clean_history(s: pd.Series) -> pd.Series:
    """Strip stray quote marks from payment-history strings."""
    return s.astype("string").str.replace('"', "", regex=False).str.strip()


def tokens(hist: str):
    """Split a payment-history string into 3-character tokens
    (most recent month first)."""
    if hist is None or pd.isna(hist) or hist == "":
        return []
    return re.findall(r".{3}", hist)


def token_to_dpd(tok: str):
    """000/STD -> 0 days past due, numeric -> that many days, XXX -> NaN."""
    if tok == "STD":
        return 0.0
    if tok == "XXX":
        return np.nan
    if tok.isdigit():
        return float(int(tok))
    return np.nan


def gini(y_true, y_score) -> float:
    """Gini = 2 * AUC - 1 (reported on the 0-100 scale elsewhere)."""
    from sklearn.metrics import roc_auc_score
    return 2 * roc_auc_score(y_true, y_score) - 1


def log1p_clip(a):
    """log(1+x) with negatives clipped to 0 (module-level so models can be pickled)."""
    return np.log1p(np.clip(a, 0, None))


# ============================================================================
# SECTION 1 - Data cleaning (accounts, enquiries, demographics)
# ============================================================================
import numpy as np
import pandas as pd

log = []


def note(table, issue, rows, action):
    log.append({"table": table, "issue": issue, "rows_or_cols_affected": rows, "action": action})
    print(f"[{table}] {issue}: {rows} -> {action}")


# ---------------------------------------------------------------- accounts
acc = pd.read_csv(RAW / "accounts_raw.csv")
n0 = len(acc)
dups = acc.duplicated().sum()
note("accounts", "exact duplicate rows", int(dups), "dropped (same account reported twice)" if dups else "none found")
acc = acc.drop_duplicates()

for c in ["dt_opened", "upload_dt", "opened_dt", "last_paymt_dt", "closed_dt",
          "reporting_dt", "paymt_str_dt", "paymt_end_dt"]:
    bad = acc[c].notna().sum() - parse_date(acc[c]).notna().sum()
    acc[c] = parse_date(acc[c])
    if bad:
        note("accounts", f"unparseable dates in {c}", int(bad), "set to NaT")

# payment history: strip quotes, join the two history blocks (recent -> older)
acc["paymenthistory1"] = clean_history(acc["paymenthistory1"])
acc["paymenthistory2"] = clean_history(acc["paymenthistory2"])
acc["payment_history"] = (acc["paymenthistory1"].fillna("") + acc["paymenthistory2"].fillna(""))
acc["history_months"] = acc["payment_history"].str.len() // 3
note("accounts", "payment history strings wrapped in stray quotes", int(len(acc)), "quotes stripped; history1+history2 concatenated")

# per-account delinquency summary (days past due, DPD)
dpd = acc["payment_history"].apply(lambda h: [token_to_dpd(t) for t in tokens(h)])
acc["max_dpd"] = dpd.apply(lambda v: np.nanmax(v) if len(v) and not np.all(np.isnan(v)) else np.nan)
acc["n_30plus"] = dpd.apply(lambda v: int(np.nansum(np.array(v) >= 30)) if len(v) else 0)
acc["months_since_first_30plus"] = dpd.apply(
    lambda v: next((i for i, x in enumerate(v) if x >= 30), np.nan) if len(v) else np.nan)

# 100 % empty / very sparse column
for c in acc.columns:
    pct = acc[c].isna().mean()
    if pct == 1.0:
        note("accounts", f"column {c} is 100% empty", 1, "dropped")
        acc = acc.drop(columns=c)

# logical checks
chk = {
    "closed before opened": (acc.closed_dt < acc.opened_dt).sum(),
    "last payment before opened": (acc.last_paymt_dt < acc.opened_dt).sum(),
    "opened after bureau upload date": (acc.opened_dt > acc.upload_dt).sum(),
    "negative balance": (acc.cur_balance_amt < 0).sum(),
    "balance > high credit amount": (acc.cur_balance_amt > acc.high_credit_amt).sum(),
}
for k, v in chk.items():
    note("accounts", f"logic check - {k}", int(v), "flagged only (no row removed)")
acc["is_open"] = acc["closed_dt"].isna().astype(int)
acc["product"] = acc["acct_type"].map(PRODUCT_LABELS).fillna("Other")
acc["is_secured"] = acc["acct_type"].isin(SECURED_CODES).astype(int)
acc["has_credit_limit"] = acc["creditlimit"].notna().astype(int)
note("accounts", "creditlimit/cashlimit/rateofinterest/paymentfrequency/actualpaymentamount mostly null",
     "76-88% null", "kept as NaN (information only exists for credit cards / some loans)")

acc.to_csv(CLEAN / "accounts_clean.csv", index=False)

# --------------------------------------------------------------- enquiries
enq = pd.read_csv(RAW / "enquiries_raw.csv")
blank = enq[["enquiry_dt", "enq_purpose", "enq_amt"]].isna().all(axis=1)
note("enquiries", "blank enquiry records (ids only, no enquiry details)", int(blank.sum()), "dropped - customer treated as having no enquiry history")
enq = enq[~blank].copy()
dups = enq.duplicated().sum()
note("enquiries", "exact duplicate rows", int(dups), "kept - repeat identical enquiries are a real behaviour signal")
for c in ["dt_opened", "upload_dt", "enquiry_dt"]:
    enq[c] = parse_date(enq[c])
enq["days_before_application"] = (enq["dt_opened"] - enq["enquiry_dt"]).dt.days
neg = (enq.days_before_application < 0).sum()
note("enquiries", "enquiry dated AFTER the application", int(neg), "flagged only" if neg else "none found")
note("enquiries", "enq_amt equal to 0 or placeholder (<=1000)", int((enq.enq_amt <= 1000).sum()), "kept; treated as small-ticket/credit-card style enquiry")
enq["purpose_label"] = enq["enq_purpose"].map(PRODUCT_LABELS).fillna("Other")
enq["is_secured_purpose"] = enq["enq_purpose"].isin(SECURED_CODES).astype(int)
enq.to_csv(CLEAN / "enquiries_clean.csv", index=False)

# ------------------------------------------------------------ demographics
dem = pd.read_csv(RAW / "demographics_raw.csv")
note("demographics", "duplicate customer_no", int(dem.customer_no.duplicated().sum()), "none - one row per customer")
note("demographics", "target Bad_label distribution", f"{int(dem.Bad_label.sum())} bad / {len(dem)}", "severe class imbalance -> stratified CV, AUC/Gini metrics")

dem["dt_opened"] = parse_date(dem["dt_opened"])
dem["entry_time"] = parse_date(dem["entry_time"])

# columns removed and why
constant = [c for c in dem.columns if c.startswith("feature_") and dem[c].nunique(dropna=True) <= 1]
mostly_null = [c for c in dem.columns if c.startswith("feature_") and dem[c].isna().mean() > 0.9 and c not in constant]
dup_cols = {"feature_59": "feature_11", "feature_68": "feature_34", "feature_69": "feature_35"}
pii = ["feature_20", "feature_21", "feature_22", "feature_47", "feature_77", "feature_24", "feature_45", "feature_38"]
note("demographics", "constant / all-null columns", ",".join(constant), "dropped (no information)")
note("demographics", ">90% null columns", ",".join(mostly_null), "dropped (too sparse to learn from 1000 rows)")
note("demographics", "exact duplicate columns", ",".join(f"{k}=={v}" for k, v in dup_cols.items()), "dropped the copy")

# derive safe features from PII columns BEFORE dropping them
dob = parse_date(dem["feature_21"], max_year=2000)
dem["age_at_application"] = ((dem["dt_opened"] - dob).dt.days / 365.25).round(1)
bad_age = ((dem.age_at_application < 18) | (dem.age_at_application > 90)).sum()
note("demographics", "implausible age (<18 or >90)", int(bad_age), "set to NaN" if bad_age else "none")
dem.loc[(dem.age_at_application < 18) | (dem.age_at_application > 90), "age_at_application"] = np.nan
dem["email_is_free_provider"] = dem["feature_24"].fillna("").str.upper().str.contains(
    "GMAIL|YAHOO|REDIFF|HOTMAIL|VSNL|SIFY|OUTLOOK|LIVE").astype(int)
dem["has_email"] = dem["feature_24"].notna().astype(int)
dem["has_second_email"] = dem["feature_45"].notna().astype(int)
dem["days_entry_to_open"] = (dem["dt_opened"] - dem["entry_time"]).dt.days

# placeholder values
dem.loc[dem["feature_35"] <= 1, "feature_35"] = np.nan   # income-like field, 1 = "not provided"
dem.loc[dem["feature_7"] <= 1, "feature_7"] = np.nan
note("demographics", "income-like field feature_35 uses placeholder 1", int((pd.read_csv(RAW / 'demographics_raw.csv')['feature_35'] <= 1).sum()), "placeholder -> NaN + missing flag")
dem["income_missing"] = dem["feature_35"].isna().astype(int)
dem["feature_3_missing"] = dem["feature_3"].isna().astype(int)
note("demographics", "feature_3 (score-like, range 553-796) missing", int(dem.feature_3.isna().sum()), "kept NaN, missing flag added, median-imputed inside the model pipeline")

# feature_75 holds mixed junk date formats ('0000-00-00', '6/15/65' ...) -> boolean flag
dem["f75_has_valid_value"] = (~dem["feature_75"].isin(["0000-00-00"])).astype(int)
# feature_70 'Mon-yy' / '0-0' strings
dem["f70_is_missing_code"] = (dem["feature_70"] == "0-0").astype(int)

# strip whitespace from text columns, standardise case
for c in dem.select_dtypes(include=["object", "string"]).columns:
    dem[c] = dem[c].astype("string").str.strip()
dem["feature_32"] = dem["feature_32"].replace({"Paren": "Parents", "Compa": "Company", "Rente": "Rented"})
note("demographics", "truncated category labels in feature_32 (Paren/Compa/Rente)", int(dem.feature_32.notna().sum()), "expanded to Parents/Company/Rented (assumption)")

drop = set(constant) | set(mostly_null) | set(dup_cols) | set(pii) | {"feature_29", "feature_44", "feature_66", "feature_15", "feature_16", "feature_17", "feature_63", "feature_75", "feature_70", "feature_2"}
note("demographics", "PII / identifier / free-text / high-cardinality code columns", ",".join(sorted(drop - set(constant) - set(mostly_null) - set(dup_cols))), "dropped from clean file (privacy & no generalisable signal); derived safe features kept")
dem = dem.drop(columns=[c for c in drop if c in dem.columns])
dem.to_csv(CLEAN / "demographics_clean.csv", index=False)

pd.DataFrame(log).to_csv(OUT / "data_quality_log.csv", index=False)
print("\nclean shapes:", {"accounts": acc.shape, "enquiries": enq.shape, "demographics": dem.shape})


# ============================================================================
# SECTION 2 - Feature engineering (customer-level table)
# ============================================================================
import numpy as np
import pandas as pd

acc = pd.read_csv(CLEAN / "accounts_clean.csv", parse_dates=["dt_opened", "opened_dt", "last_paymt_dt", "closed_dt"])
enq = pd.read_csv(CLEAN / "enquiries_clean.csv", parse_dates=["dt_opened", "enquiry_dt"])
dem = pd.read_csv(CLEAN / "demographics_clean.csv", parse_dates=["dt_opened", "entry_time"])
acc["payment_history"] = acc["payment_history"].fillna("").astype(str)


# ------------------------------------------------------- account features
def bucket_count(h, lo, hi):
    v = np.array([token_to_dpd(t) for t in tokens(h)], dtype=float)
    return int(np.sum((v >= lo) & (v <= hi))) if len(v) else 0


acc["cnt_dpd_1_29"] = acc["payment_history"].apply(lambda h: bucket_count(h, 1, 29))
acc["cnt_dpd_30_plus"] = acc["payment_history"].apply(lambda h: bucket_count(h, 30, 10 ** 4))
acc["diff_lastpaymt_opened_days"] = (acc["last_paymt_dt"] - acc["opened_dt"]).dt.days


def agg_accounts(g):
    tot_lim = g["creditlimit"].sum(min_count=1)
    tot_bal = g["cur_balance_amt"].sum()
    bal_lim = g.loc[g.creditlimit.notna(), "cur_balance_amt"].sum()
    mean_bal = g["cur_balance_amt"].mean()
    denom = (g["creditlimit"].mean() + g["cashlimit"].mean(skipna=True)) if g["creditlimit"].notna().any() else np.nan
    ratio = bal_lim / tot_lim if tot_lim and tot_lim > 0 else np.nan
    util_trend = ratio / (mean_bal / denom) if denom and denom > 0 and mean_bal > 0 and ratio == ratio else np.nan
    return pd.Series({
        "acc_n_accounts": len(g),
        "acc_n_open": int(g.is_open.sum()),
        "acc_n_closed": int((1 - g.is_open).sum()),
        "acc_total_high_credit": g.high_credit_amt.sum(),
        "acc_total_balance": tot_bal,
        "acc_total_credit_limit": tot_lim,
        "acc_ratio_balance_creditlimit": ratio,
        "acc_utilisation_trend": util_trend,
        "acc_mean_diff_lastpaymt_opened": g.diff_lastpaymt_opened_days.mean(),
        "acc_total_diff_lastpaymt_opened": g.diff_lastpaymt_opened_days.sum(),
        "acc_payment_history_mean_length": g.history_months.mean(),
        "acc_avg_dpd_0_29_bucket": g.cnt_dpd_1_29.mean(),
        "acc_total_30plus_months": g.cnt_dpd_30_plus.sum(),
        "acc_n_accounts_ever_30plus": int((g.cnt_dpd_30_plus > 0).sum()),
        "acc_min_months_last_30_plus": g.months_since_first_30plus.min(),
        "acc_max_dpd": g.max_dpd.max(),
        "acc_perc_secured": g.is_secured.mean(),
        "acc_n_credit_cards": int((g.acct_type == 10).sum()),
    })


acc_feat = acc.groupby(["customer_no", "dt_opened"]).apply(agg_accounts, include_groups=False).reset_index()

# ------------------------------------------------------ enquiry features
def agg_enquiries(g):
    d = g["days_before_application"]
    return pd.Series({
        "enq_count_total": len(g),
        "enq_count_recency_90": int((d <= 90).sum()),
        "enq_count_recency_365": int((d <= 365).sum()),
        "enq_mean_diff_open_enquiry_dt": d.mean(),
        "enq_days_since_last": d.min(),
        "enq_total_amount": g.enq_amt.sum(),
        "enq_mean_amount": g.enq_amt.mean(),
        "enq_max_amount": g.enq_amt.max(),
        "enq_n_distinct_purposes": g.enq_purpose.nunique(),
        "enq_max_freq_purpose": g.enq_purpose.mode().iloc[0],
        "enq_perc_unsecured": 1 - g.is_secured_purpose.mean(),
    })


enq_feat = enq.groupby(["customer_no", "dt_opened"]).apply(agg_enquiries, include_groups=False).reset_index()

# --------------------------------------------------------------- join
master = dem.merge(acc_feat, on=["customer_no", "dt_opened"], how="left")
master = master.merge(enq_feat, on=["customer_no", "dt_opened"], how="left")
master["has_bureau_accounts"] = master["acc_n_accounts"].notna().astype(int)
master["has_enquiry_history"] = master["enq_count_total"].notna().astype(int)

# --------------------------------------------------- coverage diagnostic
cov = pd.DataFrame({
    "source": ["demographics (population)", "accounts - customers in table", "accounts - matched to demographics",
               "enquiries - customers in table", "enquiries - matched to demographics"],
    "customers": [len(dem), acc.customer_no.nunique(), int(master.has_bureau_accounts.sum()),
                  enq.customer_no.nunique(), int(master.has_enquiry_history.sum())],
})
cov["pct_of_population"] = (cov.customers / len(dem) * 100).round(1)
bad_acc = master.loc[master.has_bureau_accounts == 1, "Bad_label"].agg(["sum", "count"]).tolist()
bad_enq = master.loc[master.has_enquiry_history == 1, "Bad_label"].agg(["sum", "count"]).tolist()
cov["bad_customers_inside"] = [int(master.Bad_label.sum()), np.nan, int(bad_acc[0]), np.nan, int(bad_enq[0])]
cov.to_csv(OUT / "join_coverage.csv", index=False)
print(cov.to_string(index=False))

master.to_csv(CLEAN / "customer_master.csv", index=False)
print("\ncustomer_master:", master.shape)


# ============================================================================
# SECTION 3 - Exploratory data analysis and charts
# ============================================================================
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GREY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9aa0a6"

m = pd.read_csv(CLEAN / "customer_master.csv", parse_dates=["dt_opened"])
acc = pd.read_csv(CLEAN / "accounts_clean.csv")
enq = pd.read_csv(CLEAN / "enquiries_clean.csv", parse_dates=["dt_opened", "enquiry_dt"])
y = m.Bad_label
base_rate = y.mean() * 100


def save(name):
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", name)


# 1 ---------------------------------------------------------- target + coverage
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
cnt = y.value_counts().sort_index()
ax[0].bar(["Good (0)", "Bad (1)"], cnt.values, color=[BLUE, RED])
for i, v in enumerate(cnt.values):
    ax[0].text(i, v + 12, f"{v}  ({v / len(y) * 100:.1f}%)", ha="center", fontweight="bold")
ax[0].set_title(f"Only {int(cnt[1])} of {len(y)} customers are 'bad' ({base_rate:.1f}%)")
ax[0].set_ylabel("Customers")
ax[0].set_ylim(0, cnt.max() * 1.15)
cov = pd.read_csv(OUT / "join_coverage.csv")
labels = ["Demographics\n(all customers)", "Accounts table\nmatched", "Enquiries table\nmatched"]
vals = [cov.customers[0], cov.customers[2], cov.customers[4]]
bars = ax[1].bar(labels, vals, color=[BLUE, ORANGE, GREEN])
for b, v in zip(bars, vals):
    ax[1].text(b.get_x() + b.get_width() / 2, v + 15, f"{v}  ({v / len(y) * 100:.1f}%)", ha="center", fontweight="bold")
ax[1].set_title("Bureau history exists for very few customers")
ax[1].set_ylabel("Customers with data")
ax[1].set_ylim(0, 1150)
fig.suptitle("Data foundation: severe class imbalance and thin bureau coverage", fontsize=14, fontweight="bold", y=1.03)
save("eda_01_target_and_coverage.png")

# 2 --------------------------------------------------------- missingness
dem_cols = [c for c in m.columns if c not in ("customer_no", "dt_opened", "entry_time", "Bad_label")
            and not c.startswith(("acc_", "enq_", "has_"))]
miss = (m[dem_cols].isna().mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0].head(15)
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.barh(miss.index[::-1], miss.values[::-1], color=ORANGE)
for i, v in enumerate(miss.values[::-1]):
    ax.text(v + 1, i, f"{v:.0f}%", va="center")
ax.set_xlabel("% missing")
ax.set_title("Missing values in application data (columns with any missing)")
ax.set_xlim(0, 100)
save("eda_02_missing_values.png")

# 3 --------------------------------------------- bad rate by category (n>=15)
cat_candidates = ["feature_1", "feature_32", "feature_27", "feature_37", "feature_36", "feature_46",
                  "feature_28", "feature_72", "feature_4", "feature_55", "feature_76", "feature_26"]
rows = []
for c in cat_candidates:
    if c not in m.columns:
        continue
    t = m.assign(v=m[c].astype("string").fillna("missing")).groupby("v").Bad_label.agg(["size", "sum"])
    t["bad_rate_pct"] = t["sum"] / t["size"] * 100
    t["variable"] = c
    rows.append(t.reset_index().rename(columns={"v": "level", "size": "customers", "sum": "bad"}))
cat_tbl = pd.concat(rows)
cat_tbl.to_csv(OUT / "category_bad_rates.csv", index=False)

show = ["feature_1", "feature_32", "feature_27", "feature_37"]
titles = {"feature_1": "Card product", "feature_32": "Residence type", "feature_27": "Education", "feature_37": "Industry"}
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
for ax, c in zip(axes.ravel(), show):
    t = cat_tbl[(cat_tbl.variable == c) & (cat_tbl.customers >= 15)].sort_values("bad_rate_pct")
    ax.barh(t.level.astype(str), t.bad_rate_pct, color=[RED if v > base_rate * 1.5 else BLUE for v in t.bad_rate_pct])
    for i, (v, n) in enumerate(zip(t.bad_rate_pct, t.customers)):
        ax.text(v + 0.1, i, f"{v:.1f}%  (n={n})", va="center", fontsize=9)
    ax.axvline(base_rate, color="k", ls="--", lw=1)
    ax.set_title(titles[c])
    ax.set_xlabel("Bad rate %")
    ax.set_xlim(0, max(t.bad_rate_pct.max() * 1.45, base_rate * 2))
fig.suptitle(f"Bad rate by segment (dashed line = portfolio {base_rate:.1f}%; segments with n<15 hidden)",
             fontsize=14, fontweight="bold", y=1.01)
save("eda_03_bad_rate_by_segment.png")

# 4 ------------------------------------ key numeric distributions by outcome
fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
specs = [("feature_3", "Score-like field (feature_3)", False), ("age_at_application", "Age at application (yrs)", False),
         ("feature_7", "Limit/amount field (feature_7, log scale)", True), ("feature_35", "Income-like field (feature_35, log)", True)]
for ax, (c, ttl, lg) in zip(axes, specs):
    d = m[[c, "Bad_label"]].dropna()
    d["plot"] = np.log10(d[c].clip(lower=1)) if lg else d[c]
    sns.boxplot(data=d, x="Bad_label", y="plot", ax=ax, palette=[BLUE, RED], showfliers=False, width=0.5)
    sns.stripplot(data=d[d.Bad_label == 1], x="Bad_label", y="plot", ax=ax, color="k", size=3, alpha=0.6)
    ax.set_xticklabels(["Good", "Bad"])
    ax.set_xlabel("")
    ax.set_ylabel("log10 value" if lg else "")
    ax.set_title(ttl, fontsize=11)
fig.suptitle("Bad customers have lower limit/income-like values; age and feature_3 barely differ", fontsize=14, fontweight="bold", y=1.03)
save("eda_04_numeric_by_outcome.png")

# 5 ----------------------- univariate Gini of numeric fields & information value
num = m.select_dtypes("number").drop(columns=["customer_no", "Bad_label"])
res = []
for c in num.columns:
    s = num[c]
    if s.notna().sum() < 50 or s.nunique() < 2:
        continue
    filled = s.fillna(s.median())
    auc = roc_auc_score(y, filled)
    res.append({"feature": c, "gini_pct": (2 * auc - 1) * 100, "abs_gini_pct": abs(2 * auc - 1) * 100,
                "pct_missing": s.isna().mean() * 100})
uni = pd.DataFrame(res).sort_values("abs_gini_pct", ascending=False)
uni.to_csv(OUT / "univariate_numeric_gini.csv", index=False)
top = uni.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(9.5, 6))
ax.barh(top.feature, top.gini_pct, color=[BLUE if v < 0 else ORANGE for v in top.gini_pct])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Single-variable Gini (%)   (negative = higher value -> LOWER risk)")
ax.set_title("Strongest single numeric predictors (each used alone)")
save("eda_05_univariate_gini.png")


def information_value(series, target, bins=5):
    d = pd.DataFrame({"x": series, "y": target})
    if pd.api.types.is_numeric_dtype(d.x) and d.x.nunique() > bins:
        d["b"] = pd.qcut(d.x, bins, duplicates="drop").astype("string").fillna("missing")
    else:
        d["b"] = d.x.astype("string").fillna("missing")
    g = d.groupby("b").y.agg(["sum", "count"])
    g["good"] = g["count"] - g["sum"]
    g = g + 0.5   # Laplace smoothing - tiny counts make raw WOE explode
    woe = np.log((g["good"] / g["good"].sum()) / (g["sum"] / g["sum"].sum()))
    return float(((g["good"] / g["good"].sum() - g["sum"] / g["sum"].sum()) * woe).sum())


iv_rows = []
for c in m.columns:
    if c in ("customer_no", "dt_opened", "entry_time", "Bad_label") or m[c].nunique() < 2:
        continue
    if m[c].dtype == "object" or str(m[c].dtype) in ("string", "str"):
        if m[c].nunique() > 12:      # many levels x 33 bad customers => IV is inflated by noise
            continue
    iv_rows.append({"feature": c, "n_levels_or_bins": int(min(m[c].nunique(), 5)) if pd.api.types.is_numeric_dtype(m[c]) and m[c].nunique() > 5 else int(m[c].nunique()),
                    "information_value": information_value(m[c], y)})
iv = pd.DataFrame(iv_rows).sort_values("information_value", ascending=False)
iv["strength"] = pd.cut(iv.information_value, [-1, 0.02, 0.1, 0.3, 0.5, 99],
                        labels=["useless", "weak", "medium", "strong", "suspiciously strong"])
iv.to_csv(OUT / "information_value.csv", index=False)
t = iv.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(9.5, 6))
ax.barh(t.feature, t.information_value, color=GREEN)
ax.axvline(0.1, color="k", ls="--", lw=1)
ax.text(0.102, 0.2, "0.1 = medium", fontsize=9)
ax.set_xlabel("Information value (Laplace-smoothed, 5 bins)")
ax.set_title("Information value: which variables separate good from bad?")
save("eda_06_information_value.png")

# 7 ----------------------------------------- time trend of applications
mm = m.assign(month=m.dt_opened.dt.to_period("M").dt.to_timestamp()).groupby("month").Bad_label.agg(["size", "sum"])
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.bar(mm.index.strftime("%b-%y"), mm["size"], color=BLUE, label="Applications")
ax.bar(mm.index.strftime("%b-%y"), mm["sum"], color=RED, label="Bad customers")
ax.set_ylabel("Customers")
ax2 = ax.twinx()
ax2.plot(mm.index.strftime("%b-%y"), mm["sum"] / mm["size"] * 100, color="k", marker="o", label="Bad rate % (right)")
ax2.set_ylabel("Bad rate %")
ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax.set_title("92% of applications come from Apr-May 2015; later months are too small to judge")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right")
save("eda_07_applications_over_time.png")

# 8 ---------------------------------------------------- account-level views
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
prod = acc["product"].value_counts().head(8)
axes[0].barh(prod.index[::-1], prod.values[::-1], color=BLUE)
axes[0].set_title("Accounts by product")
axes[0].set_xlabel("Accounts")
all_tok = pd.Series([t for h in acc.payment_history.dropna().astype(str) for t in tokens(h)])
def bucket(t):
    d = token_to_dpd(t)
    if t == "XXX":
        return "No data (XXX)"
    if d == 0:
        return "Current (000/STD)"
    if d < 30:
        return "1-29 DPD"
    if d < 90:
        return "30-89 DPD"
    return "90+ DPD"
bk = all_tok.map(bucket).value_counts().reindex(["Current (000/STD)", "No data (XXX)", "1-29 DPD", "30-89 DPD", "90+ DPD"]).fillna(0)
axes[1].bar(range(len(bk)), bk.values, color=[GREEN, GREY, ORANGE, RED, "#7a1f27"])
axes[1].set_xticks(range(len(bk)))
axes[1].set_xticklabels([b.replace(" (", "\n(") for b in bk.index], fontsize=9)
axes[1].set_yscale("log")
for i, v in enumerate(bk.values):
    axes[1].text(i, v * 1.15, f"{int(v):,}\n({v / bk.sum() * 100:.1f}%)", ha="center", fontsize=9)
axes[1].set_title("Monthly payment status across all accounts")
axes[1].set_ylabel("Account-months (log scale)")
axes[1].set_ylim(1, bk.max() * 8)
card = acc[(acc.acct_type == 10) & acc.creditlimit.notna() & (acc.creditlimit > 0)].copy()
card["util"] = (card.cur_balance_amt / card.creditlimit).clip(0, 1.5)
axes[2].hist(card.util, bins=25, color=ORANGE, edgecolor="white")
axes[2].set_title(f"Credit-card utilisation (n={len(card)})")
axes[2].set_xlabel("Balance / limit")
axes[2].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
fig.suptitle("Bureau accounts: unsecured-heavy, mostly clean, a long tail of severe delinquency", fontsize=14, fontweight="bold", y=1.03)
save("eda_08_accounts_overview.png")

# 9 --------------------------------------------------- enquiry views
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
ep = enq["purpose_label"].value_counts().head(8)
axes[0].barh(ep.index[::-1], ep.values[::-1], color=GREEN)
axes[0].set_title("Enquiries by purpose")
per = enq.groupby("customer_no").size()
axes[1].hist(per, bins=15, color=BLUE, edgecolor="white")
axes[1].set_title(f"Enquiries per customer (median {per.median():.0f}, max {per.max()})")
axes[1].set_xlabel("# enquiries on record")
axes[2].hist(enq.days_before_application.dropna().clip(upper=2000), bins=30, color=ORANGE, edgecolor="white")
axes[2].axvline(365, color="k", ls="--")
axes[2].axvline(90, color="k", ls=":")
axes[2].set_title("Days between enquiry and application")
axes[2].set_xlabel("Days (clipped at 2000)")
fig.suptitle("Enquiry behaviour (only 50 customers have enquiry history)", fontsize=14, fontweight="bold", y=1.03)
save("eda_09_enquiries_overview.png")

# 10 --------------------------------------------- correlation heat-map
top_num = [c for c in uni.feature.head(14) if c in m.columns]
corr = m[top_num + ["Bad_label"]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(10, 8.2))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=0.3, ax=ax)
ax.set_title("Spearman correlation: top numeric predictors vs. outcome")
save("eda_10_correlation_heatmap.png")

print("\nTop information-value variables:\n", iv.head(10).round(3).to_string(index=False))
print("\nTop univariate Gini:\n", uni.head(10).round(1).to_string(index=False))


# ============================================================================
# SECTION 4 - Modelling, validation, rank ordering, feature gain
# ============================================================================
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


# ============================================================================
# SECTION 5 - Model evaluation charts
# ============================================================================
import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GREY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9aa0a6"
BENCH = 37.9
BENCH_DEC = {10: 11.713, 9: 8.053, 8: 6.003, 7: 3.959, 6: 4.246, 5: 3.660, 4: 2.053, 3: 2.050, 2: 2.050, 1: 1.318}

mt = json.load(open(OUT / "model_metrics.json"))
cmp_ = pd.read_csv(OUT / "model_comparison.csv")
dec = pd.read_csv(OUT / "rank_ordering_deciles.csv")
oof = pd.read_csv(CLEAN / "oof_predictions.csv")
fi = pd.read_csv(OUT / "feature_importance_by_variable.csv")
y, s = oof.Bad_label.values, oof.oof_risk_rank_score.values


def save(name):
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", name)


# 1 model comparison -------------------------------------------------------------
c = cmp_.sort_values("cv_gini_mean")
fig, ax = plt.subplots(figsize=(10, 4.8))
cols = [GREEN if i == len(c) - 1 or n.startswith("Rank-average") and False else BLUE for i, n in enumerate(c.model)]
ax.barh(c.model, c.cv_gini_mean, xerr=c.cv_gini_std, color=cols, capsize=4)
for i, (v, sd) in enumerate(zip(c.cv_gini_mean, c.cv_gini_std)):
    ax.text(v + sd + 0.8, i, f"{v:.1f}", va="center", fontweight="bold")
ax.axvline(BENCH, color=RED, ls="--", lw=1.5)
ax.text(BENCH + 0.4, -0.45, f"Reference Gini {BENCH}", color=RED, fontsize=9)
ax.set_xlabel("Cross-validated Gini (%)  - mean +/- std over 5x5-fold repeats")
ax.set_xlim(0, 55)
ax.set_title("Simple, regularised models beat the reference; deeper models overfit 33 bad customers")
save("model_01_comparison.png")

# 2 ROC + bootstrap uncertainty -----------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
boot = np.array([2 * roc_auc_score(y[i], s[i]) - 1 for i in
                 (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(2000))]) * 100
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fpr, tpr, _ = roc_curve(y, s)
ax[0].plot(fpr, tpr, color=BLUE, lw=2.5, label=f"Out-of-fold ROC (Gini {mt['pooled_oof_gini']:.1f})")
ax[0].plot([0, 1], [0, 1], color=GREY, ls="--", label="Random")
ax[0].fill_between(fpr, fpr, tpr, color=BLUE, alpha=0.12)
ax[0].set_xlabel("False positive rate (good customers flagged)")
ax[0].set_ylabel("True positive rate (bad customers caught)")
ax[0].set_title("ROC curve - every customer scored out-of-fold")
ax[0].legend(loc="lower right")
ax[1].hist(boot, bins=40, color=BLUE, edgecolor="white", alpha=0.85)
lo, hi = np.percentile(boot, [2.5, 97.5])
ax[1].axvline(lo, color="k", ls=":")
ax[1].axvline(hi, color="k", ls=":")
ax[1].axvline(BENCH, color=RED, ls="--", lw=2, label=f"Reference {BENCH}")
ax[1].axvline(0, color=GREY, lw=1)
share = (boot > BENCH).mean() * 100
ax[1].set_title(f"With 33 bads the Gini is uncertain: 95% CI {lo:.0f} to {hi:.0f}")
ax[1].set_xlabel("Bootstrap Gini (%)")
ax[1].set_ylabel("Resamples")
ax[1].text(0.62, 0.75, f"{share:.0f}% of resamples\nexceed {BENCH}", transform=ax[1].transAxes, fontsize=10)
ax[1].legend(loc="upper right")
save("model_02_roc_and_uncertainty.png")

# 3 rank ordering --------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
d = dec.sort_values("bin")
x = np.arange(1, 11)
ax[0].bar(x - 0.2, d.bad_rate_pct, 0.4, color=BLUE, label="This model (out-of-fold)")
ax[0].bar(x + 0.2, [BENCH_DEC[i] for i in x], 0.4, color=GREY, label="Reference model")
ax[0].axhline(mt["bad_rate_pct"], color="k", ls="--", lw=1)
for xi, v in zip(x, d.bad_rate_pct):
    ax[0].text(xi - 0.2, v + 0.2, f"{v:.0f}%", ha="center", fontsize=8)
ax[0].set_xticks(x)
ax[0].set_xlabel("Risk decile (10 = riskiest 10% of customers)")
ax[0].set_ylabel("Bad rate %")
ax[0].set_title("Rank ordering: top 2 deciles hold 58% of bad customers")
ax[0].legend()
g = dec.sort_values("bin", ascending=False)
pct_pop = np.arange(1, 11) * 10
ax[1].plot(np.r_[0, pct_pop], np.r_[0, g.cum_bad_captured_pct], marker="o", color=BLUE, lw=2.5, label="Model")
ax[1].plot([0, 100], [0, 100], color=GREY, ls="--", label="Random")
for px, py in zip(pct_pop[:3], g.cum_bad_captured_pct.values[:3]):
    ax[1].annotate(f"{py:.0f}%", (px, py), textcoords="offset points", xytext=(6, -14))
ax[1].set_xlabel("% of customers reviewed (riskiest first)")
ax[1].set_ylabel("% of bad customers captured")
ax[1].set_title(f"Cumulative gains (KS = {mt['ks_pct']:.0f})")
ax[1].legend(loc="lower right")
save("model_03_rank_ordering.png")

# 4 feature importance -----------------------------------------------------------------
t = fi.head(15).iloc[::-1]
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
ax[0].barh(t.feature, t.gain_avg_norm * 100, color=BLUE)
ax[0].set_xlabel("Gain share (%)  - average of boosting and forest")
ax[0].set_title("In-sample gain (what the trees use)")
ax[1].barh(t.feature, t.perm_auc_drop, color=[GREEN if v > 0 else RED for v in t.perm_auc_drop])
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_xlabel("AUC drop when shuffled on held-out folds")
ax[1].set_title("Out-of-sample permutation importance (green = truly useful)")
save("model_04_feature_importance.png")

# 5 credit score ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
bins = np.linspace(oof.cred_score_300_900.min(), oof.cred_score_300_900.max(), 25)
ax[0].hist(oof.cred_score_300_900[y == 0], bins=bins, color=BLUE, alpha=0.75, density=True, label="Good")
ax[0].hist(oof.cred_score_300_900[y == 1], bins=bins, color=RED, alpha=0.75, density=True, label="Bad")
ax[0].set_xlabel("Credit score (higher = safer)")
ax[0].set_ylabel("Density")
ax[0].set_title("Bad customers concentrate at low scores")
ax[0].legend()
band = pd.qcut(oof.cred_score_300_900.rank(method="first"), 5, labels=False)
tb = oof.assign(band=band).groupby("band").agg(lo=("cred_score_300_900", "min"), hi=("cred_score_300_900", "max"),
                                                 n=("Bad_label", "size"), bad=("Bad_label", "sum"))
tb["rate"] = tb.bad / tb.n * 100
labels = [f"{int(a)}-{int(b)}" for a, b in zip(tb.lo, tb.hi)]
ax[1].bar(labels, tb.rate, color=[RED, ORANGE, GREY, BLUE, GREEN])
for i, (r, b) in enumerate(zip(tb.rate, tb.bad)):
    ax[1].text(i, r + 0.2, f"{r:.1f}%\n({int(b)} bad)", ha="center", fontsize=9)
ax[1].set_xlabel("Credit-score band (quintiles, lowest = riskiest)")
ax[1].set_ylabel("Bad rate %")
ax[1].set_title("Score bands separate risk")
ax[1].set_ylim(0, tb.rate.max() * 1.35)
tb.reset_index().to_csv(OUT / "score_band_summary.csv", index=False)
save("model_05_credit_score.png")
print(tb.round(1))


# ============================================================================
# SECTION 6 - One-page model report card
# ============================================================================
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, RED, GREY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9aa0a6"
mt = json.load(open(OUT / "model_metrics.json"))
cmp_ = pd.read_csv(OUT / "model_comparison.csv").sort_values("cv_gini_mean")
dec = pd.read_csv(OUT / "rank_ordering_deciles.csv").sort_values("bin")
fi = pd.read_csv(OUT / "feature_importance_by_variable.csv").head(8).iloc[::-1]
sb = pd.read_csv(OUT / "score_band_summary.csv")
top2 = dec.sort_values("bin", ascending=False).cum_bad_captured_pct.iloc[1]

fig = plt.figure(figsize=(14, 10), facecolor="white")
gs = fig.add_gridspec(3, 4, height_ratios=[0.7, 1.6, 1.6], hspace=0.55, wspace=0.45)
fig.suptitle("Bank GoodCredit - Credit Risk Scorecard: Model Report Card", fontsize=18, fontweight="bold", y=0.985)

kpis = [("CV Gini", f"{mt['best_cv_gini_mean']:.1f}", f"+/- {mt['best_cv_gini_std']:.1f}  (reference 37.9)", BLUE),
        ("KS statistic", f"{mt['ks_pct']:.0f}", "separation of bad vs good", GREEN),
        ("Top-decile bad rate", f"{dec.bad_rate_pct.iloc[-1]:.0f}%", f"vs {mt['bad_rate_pct']:.1f}% average (3.6x lift)", ORANGE),
        ("Bad captured in top 20%", f"{top2:.0f}%", "of all bad customers", RED)]
for i, (t, v, sub, c) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=c, alpha=0.12))
    ax.text(0.5, 0.78, t, ha="center", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.42, v, ha="center", fontsize=26, fontweight="bold", color=c, transform=ax.transAxes)
    ax.text(0.5, 0.12, sub, ha="center", fontsize=8.5, transform=ax.transAxes)

ax = fig.add_subplot(gs[1, :2])
ax.bar(dec.bin, dec.bad_rate_pct, color=[RED if b >= 9 else BLUE for b in dec.bin])
ax.axhline(mt["bad_rate_pct"], color="k", ls="--", lw=1)
ax.set_title("Rank ordering by risk decile", fontweight="bold")
ax.set_xlabel("Decile (10 = riskiest)"); ax.set_ylabel("Bad rate %"); ax.set_xticks(range(1, 11))

ax = fig.add_subplot(gs[1, 2:])
ax.barh(cmp_.model.str.replace(" (strong reg.)", "").str.replace(" (top 2)", ""), cmp_.cv_gini_mean, xerr=cmp_.cv_gini_std, color=BLUE, capsize=3)
ax.axvline(37.9, color=RED, ls="--"); ax.set_xlim(0, 55)
ax.set_title("Model comparison (CV Gini)", fontweight="bold"); ax.tick_params(axis="y", labelsize=8)

ax = fig.add_subplot(gs[2, :2])
ax.barh(fi.feature, fi.gain_avg_norm * 100, color=GREEN)
ax.set_title("Top features by gain share", fontweight="bold"); ax.set_xlabel("Gain %")

ax = fig.add_subplot(gs[2, 2:])
lab = [f"{int(a)}-{int(b)}" for a, b in zip(sb.lo, sb.hi)]
ax.bar(lab, sb.rate, color=[RED, ORANGE, GREY, BLUE, GREEN])
for i, r in enumerate(sb.rate): ax.text(i, r + 0.2, f"{r:.1f}%", ha="center", fontsize=9)
ax.set_title("Bad rate by credit-score band (lowest first)", fontweight="bold"); ax.set_xlabel("Credit score"); ax.set_ylabel("Bad rate %"); ax.set_ylim(0, sb.rate.max() * 1.2)

lo, hi = mt["pooled_oof_gini_ci95"]
fig.text(0.5, 0.012, f"Caution: only {mt['n_bad']} bad customers - 95% CI for Gini {lo:.0f}-{hi:.0f}; bureau tables link to just 8 (accounts) and 50 (enquiries) customers. "
         "Pilot for review-tiering, not automated decisions.", ha="center", fontsize=9, color=RED)
plt.savefig(CHARTS / "report_card.png", dpi=150, bbox_inches="tight")
print("saved report_card.png")
