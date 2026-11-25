"""Step 5 - Model evaluation charts (reads outputs written by 03_modeling.py)."""
import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from utils import CLEAN, CHARTS, OUT, RANDOM_STATE

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
