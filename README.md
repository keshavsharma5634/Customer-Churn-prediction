# 🔮 Customer Churn Prediction Model

> End-to-end ML system: data → features → XGBoost → FastAPI → Next.js Dashboard → MLOps

---

## 📁 Project Structure

```
churn_project/
├── data/                        ← Generated datasets
│   ├── churn_frame.csv          ← Synthetic raw data (8,000 rows, ~12% churn)
│   └── churn_frame.parquet      ← Type-enforced Parquet (after Notebook 01)
│
├── notebooks/                   ← 7 Jupyter notebooks (run in order)
│   ├── 01_ingest.ipynb          ← Data schema & ingestion
│   ├── 02_eda.ipynb             ← EDA + leakage guardrails
│   ├── 03_features.ipynb        ← Feature engineering + ColumnTransformer
│   ├── 04_baselines.ipynb       ← LogReg + RandomForest baselines
│   ├── 05_xgboost_optuna.ipynb  ← XGBoost + Optuna tuning + calibration
│   ├── 06_evaluation.ipynb      ← PR-AUC, SHAP, actionable segments
│   └── 07_mlops.ipynb           ← Drift detection + monitoring
│
├── models/                      ← Saved model artifacts
│   ├── preprocessor.joblib      ← Fitted ColumnTransformer
│   └── churn_calibrated.joblib  ← Calibrated XGBoost (final model)
│
├── serving/
│   └── app.py                   ← FastAPI scoring service
│
├── apps/web/
│   └── dashboard.html           ← Success-Ops dashboard (standalone HTML)
│
├── docker/
│   └── Dockerfile               ← Production container
│
├── scripts/
│   ├── generate_data.py         ← Synthetic data generator
│   └── build_notebooks.py       ← Notebook builder
│
└── requirements.txt
```

---

## 🚀 Quick Start

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate Data
```bash
python scripts/generate_data.py
# → data/churn_frame.csv (8,000 rows, ~12% churn rate)
```

### Step 3 — Run Notebooks in Order
```bash
jupyter lab
# Open notebooks/ folder and run 01 → 02 → 03 → 04 → 05 → 06 → 07
```

### Step 4 — Start FastAPI Server
```bash
uvicorn serving.app:app --reload --port 8000
# API docs → http://localhost:8000/docs
```

### Step 5 — Open Dashboard
```bash
# Open apps/web/dashboard.html in your browser
```

---

## 📒 Notebook Guide

| # | Notebook | Key Output |
|---|----------|-----------|
| 01 | Data Ingestion | `churn_frame.parquet` |
| 02 | EDA + Leakage | Distribution plots, time-based split |
| 03 | Feature Engineering | `preprocessor.joblib`, 5 new features |
| 04 | Baselines (LR + RF) | PR-AUC benchmarks |
| 05 | XGBoost + Optuna | `churn_calibrated.joblib` |
| 06 | Evaluation + SHAP | `churn_scores.parquet`, segment playbook |
| 07 | MLOps | PSI drift report, Brier monitoring |

---

## 🌐 FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/score` | Predict churn probability for a customer |
| POST | `/explain` | Get top 5 churn drivers |
| GET | `/topk?k=20` | Top-K at-risk customers |

**Example /score request:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "billing_amount": 599,
    "plan_tier": "standard",
    "tenure_months": 6,
    "active_days": 8,
    "monthly_usage_hours": 5,
    "login_count": 3,
    "support_tickets": 2,
    "sla_breaches": 1,
    "nps_score": 3,
    "is_autopay": false,
    "is_discounted": false,
    "has_family_bundle": false,
    "last_payment_days_ago": 20,
    "avg_session_min": 12,
    "device_count": 1,
    "add_on_count": 0,
    "promotions_redeemed": 0,
    "email_opens": 1,
    "email_clicks": 0,
    "last_campaign_days_ago": 30,
    "region": "north"
  }'
```

**Response:**
```json
{
  "churn_prob": 0.7842,
  "segment": "high",
  "suggested_action": "Priority support callback + apology credit"
}
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -f docker/Dockerfile -t churn-api .

# Run container
docker run -p 8000:8000 churn-api
```

---

## 🧠 Model Architecture

```
Raw CSV
   ↓
add_features()        ← 5 engineered business features
   ↓
ColumnTransformer     ← StandardScaler (num) + OneHotEncoder (cat)
   ↓
XGBClassifier         ← Tuned by Optuna (30 trials, PR-AUC objective)
   ↓
CalibratedClassifierCV ← Isotonic calibration (cv=3)
   ↓
churn_calibrated.joblib
```

### Engineered Features
| Feature | Formula | Signal |
|---------|---------|--------|
| `engagement_rate` | active_days / 30 | How often customer is active |
| `usage_per_login` | hours / logins | Session depth |
| `support_intensity` | tickets + 3×SLA_breaches | Weighted frustration |
| `email_ctr` | clicks / opens | Marketing responsiveness |
| `price_to_tenure` | billing / tenure | Value perception |

---

## 📈 Key Metrics

| Metric | Description |
|--------|-------------|
| **PR-AUC** | Primary metric (handles imbalance better than ROC-AUC) |
| **ROC-AUC** | Overall discriminative power |
| **Lift@10%** | How much better than random at top 10% |
| **Brier Score** | Calibration quality (lower = better) |

---

## 🏗️ Industry Applications

- **Telecom**: Monthly plan churn watchlists
- **SaaS**: Success-ops triage queue
- **OTT**: Subscription renewal risk scoring
- **Fintech**: Loan app inactivity churn
- **D2C**: Reorder / retention campaigns

---

## 📚 Interview Q&A

**Q: Why PR-AUC over ROC-AUC?**
A: With ~12% churn rate (imbalanced), ROC-AUC can be overly optimistic. PR-AUC focuses on the minority class and better reflects business value.

**Q: How do you prevent leakage?**
A: Exclude all post-cycle features + use `TimeSeriesSplit` so training always precedes validation in time.

**Q: Why isotonic calibration?**
A: Raw tree model probabilities are often poorly calibrated. Isotonic calibration ensures P(churn=0.7) really means 70% chance, enabling reliable top-K ranking.

**Q: How does Optuna work?**
A: Optuna uses TPE (Tree-structured Parzen Estimator) to intelligently explore the hyperparameter space, converging on high PR-AUC configurations in ~30 trials.

**Q: What triggers retraining?**
A: PSI > 0.25 on any key feature (significant distribution shift) OR PR-AUC drops > 15% from baseline.
