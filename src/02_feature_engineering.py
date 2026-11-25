"""Step 2 - Build one row per customer.

Account and enquiry tables have many rows per customer; they are aggregated
to customer level and joined to demographics on (customer_no, dt_opened).

Writes: data/clean/customer_master.csv, outputs/join_coverage.csv
"""
import numpy as np
import pandas as pd
from utils import CLEAN, OUT, tokens, token_to_dpd

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
