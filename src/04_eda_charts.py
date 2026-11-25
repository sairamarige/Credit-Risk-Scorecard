"""Step 4 - Exploratory analysis charts + univariate screening tables.

Writes: charts/eda_*.png, outputs/information_value.csv,
        outputs/univariate_numeric_gini.csv, outputs/category_bad_rates.csv
"""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from utils import CLEAN, CHARTS, OUT, tokens, token_to_dpd

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
