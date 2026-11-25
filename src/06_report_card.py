"""Step 6 - One-page model report card (PNG) from saved outputs."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import CHARTS, OUT

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
