# Data dictionary (cleaned files)

Column meanings for `feature_*` fields are **inferred** from value ranges; the source data is anonymised.

## customer_master.csv (1 row per customer, modelling table)
| Column group | Examples | Meaning |
|---|---|---|
| Identifiers | `customer_no`, `dt_opened`, `entry_time` | Key = (customer_no, dt_opened). Not used as model inputs. |
| Target | `Bad_label` | 1 = reached 30+ days past due, 0 = did not |
| Application | `feature_1` (card product), `feature_3` (score-like), `feature_7` (amount/limit-like), `feature_32` (residence type), `feature_35` (income-like, placeholder 1 -> NaN), `feature_27` (education), `feature_36/37` (employer type/industry) | cleaned demographic fields |
| Derived | `age_at_application`, `email_is_free_provider`, `has_email`, `has_second_email`, `days_entry_to_open`, `income_missing`, `feature_3_missing` | engineered from raw/PII columns |
| Account (`acc_*`) | `acc_ratio_balance_creditlimit`, `acc_utilisation_trend`, `acc_avg_dpd_0_29_bucket`, `acc_min_months_last_30_plus`, `acc_max_dpd` | bureau tradeline aggregates (populated for 8 customers) |
| Enquiry (`enq_*`) | `enq_count_recency_365`, `enq_count_recency_90`, `enq_mean_diff_open_enquiry_dt`, `enq_max_freq_purpose`, `enq_perc_unsecured` | enquiry aggregates (populated for 50 customers) |
| Coverage flags | `has_bureau_accounts`, `has_enquiry_history` | 1 if the customer exists in that table |

## accounts_clean.csv
Dates parsed; `payment_history` = history1 + history2 (quotes removed, 3-character monthly tokens, newest first); `history_months`, `max_dpd`, `n_30plus`, `months_since_first_30plus`, `is_open`, `product`, `is_secured`, `has_credit_limit` added. `amt_past_due` removed (100% empty).

## enquiries_clean.csv
Dates parsed; `days_before_application`, `purpose_label`, `is_secured_purpose` added. Blank record removed.

## demographics_clean.csv
Constant, >90%-null, duplicate and personal-identifier columns removed; see `outputs/data_quality_log.csv`.

## model_matrix.csv
Fully encoded matrix (imputed, one-hot, log-money) the tree models see, plus `Bad_label`.

## oof_predictions.csv
Out-of-fold risk rank score, calibrated probability of default and 300-900 credit score per customer.

## Code assumptions
Product/purpose codes follow standard Indian credit-bureau coding (10 = credit card, 5/6 = personal/consumer loan, 1 = auto, 2 = housing ...). Secured = {1,2,3,4,7,13,15,17}.
