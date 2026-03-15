"""
generate_data.py
Generates a realistic synthetic churn dataset with ~15% churn rate.
Run: python scripts/generate_data.py
Output: data/churn_frame.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 8000  # number of customer-cycle rows

Path("data").mkdir(exist_ok=True)

# ── Base customer attributes ───────────────────────────────────────────────
tenure_months     = np.random.exponential(scale=24, size=N).clip(1, 120).round(1)
plan_tier         = np.random.choice(["basic","standard","premium"], size=N, p=[0.40,0.40,0.20])
region            = np.random.choice(["north","south","east","west","central"], size=N)
is_autopay        = np.random.choice([True, False], size=N, p=[0.60, 0.40])
is_discounted     = np.random.choice([True, False], size=N, p=[0.30, 0.70])
has_family_bundle = np.random.choice([True, False], size=N, p=[0.25, 0.75])

plan_price = {"basic": 299, "standard": 599, "premium": 999}
billing_amount = np.array([plan_price[p] for p in plan_tier], dtype=float)
billing_amount += np.random.normal(0, 30, N)  # small variation
billing_amount = billing_amount.clip(100)

# ── Usage signals ──────────────────────────────────────────────────────────
monthly_usage_hours = np.random.gamma(shape=3, scale=8, size=N).clip(0, 200)
active_days         = np.random.binomial(30, p=0.65, size=N).clip(0, 30)
login_count         = np.random.poisson(lam=12, size=N).clip(0, 60)
avg_session_min     = np.random.gamma(shape=2, scale=15, size=N).clip(1, 120)
device_count        = np.random.randint(1, 5, size=N).astype(float)
add_on_count        = np.random.poisson(lam=1.2, size=N).clip(0, 6).astype(float)

# ── Support signals ────────────────────────────────────────────────────────
support_tickets    = np.random.poisson(lam=0.8, size=N).clip(0, 10).astype(float)
sla_breaches       = np.random.poisson(lam=0.2, size=N).clip(0, 5).astype(float)

# ── Marketing signals ──────────────────────────────────────────────────────
promotions_redeemed   = np.random.poisson(lam=0.5, size=N).clip(0, 5).astype(float)
email_opens           = np.random.poisson(lam=3,   size=N).clip(0, 20).astype(float)
email_clicks          = (email_opens * np.random.uniform(0, 0.4, N)).round().astype(float)
last_campaign_days_ago= np.random.exponential(scale=15, size=N).clip(0, 90).round(1)
last_payment_days_ago = np.random.exponential(scale=10, size=N).clip(0, 60).round(1)

# ── NPS score ─────────────────────────────────────────────────────────────
nps_score = np.random.choice(range(0, 11), size=N,
              p=[0.03,0.03,0.04,0.05,0.08,0.10,0.12,0.15,0.17,0.13,0.10])

# ── Cycle dates (simulate 24 months of history) ───────────────────────────
cycle_start_days = np.random.randint(0, 720, size=N)
base_date = pd.Timestamp("2023-01-01")
cycle_start = [base_date + pd.Timedelta(days=int(d)) for d in cycle_start_days]
cycle_end   = [s + pd.Timedelta(days=30) for s in cycle_start]

customer_ids = [f"CUST_{str(i).zfill(5)}" for i in range(N)]

# ── Churn probability (logistic model with realistic drivers) ──────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

engagement = active_days / 30.0
support_int = support_tickets + 3 * sla_breaches
price_sens  = billing_amount / (tenure_months + 1)

logit = (
    - 0.04 * tenure_months          # longer tenure → less churn
    - 2.5  * engagement             # higher engagement → less churn
    + 0.15 * support_int            # more support issues → more churn
    - 0.8  * is_autopay.astype(int) # autopay → less churn
    + 0.6  * is_discounted.astype(int) * -1  # discount → less churn
    - 0.04 * nps_score              # higher NPS → less churn
    + 0.003 * price_sens            # price-to-tenure pressure
    + 0.02 * last_payment_days_ago  # payment recency risk
    - 0.5  * has_family_bundle.astype(int)
    + np.random.normal(0, 0.5, N)   # noise
    + 0.5                           # intercept → ~15% base rate
)

churn_prob = sigmoid(logit)
churned_next_cycle = (np.random.uniform(size=N) < churn_prob).astype(int)

print(f"Churn rate: {churned_next_cycle.mean():.2%}")

# ── Assemble DataFrame ─────────────────────────────────────────────────────
df = pd.DataFrame({
    "customer_id":           customer_ids,
    "cycle_start":           [s.strftime("%Y-%m-%d") for s in cycle_start],
    "cycle_end":             [e.strftime("%Y-%m-%d") for e in cycle_end],
    "billing_amount":        billing_amount.round(2),
    "last_payment_days_ago": last_payment_days_ago,
    "plan_tier":             plan_tier,
    "tenure_months":         tenure_months,
    "monthly_usage_hours":   monthly_usage_hours.round(2),
    "active_days":           active_days.astype(float),
    "login_count":           login_count.astype(float),
    "avg_session_min":       avg_session_min.round(2),
    "device_count":          device_count,
    "add_on_count":          add_on_count,
    "support_tickets":       support_tickets,
    "sla_breaches":          sla_breaches,
    "promotions_redeemed":   promotions_redeemed,
    "email_opens":           email_opens,
    "email_clicks":          email_clicks,
    "last_campaign_days_ago":last_campaign_days_ago,
    "nps_score":             nps_score.astype(float),
    "region":                region,
    "is_autopay":            is_autopay,
    "is_discounted":         is_discounted,
    "has_family_bundle":     has_family_bundle,
    "churned_next_cycle":    churned_next_cycle,
})

df.to_csv("data/churn_frame.csv", index=False)
print(f"Saved {len(df)} rows → data/churn_frame.csv")
print(df.head(3).to_string())
