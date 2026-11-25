"""Step 1 - Clean the raw tables.

Reads  : data/raw/{accounts,demographics,enquiries}_raw.csv
Writes : data/clean/{accounts,demographics,enquiries}_clean.csv
         outputs/data_quality_log.csv
"""
import numpy as np
import pandas as pd
from utils import RAW, CLEAN, OUT, parse_date, clean_history, tokens, token_to_dpd, PRODUCT_LABELS, SECURED_CODES

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
