"""Shared helpers for the credit-risk scorecard project."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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
