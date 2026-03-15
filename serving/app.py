"""
serving/app.py
FastAPI Churn Scoring Service
Endpoints:
  POST /score    → churn probability + segment
  POST /explain  → top SHAP-based drivers
  GET  /health   → service health check
  GET  /topk     → top-K at-risk customers from batch scores
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Scores customer churn probability and explains key drivers.",
    version="1.0.0",
)

# Allow Next.js dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ──────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_calibrated.joblib")
model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")


# ── Request schema ─────────────────────────────────────────────────────────
class Customer(BaseModel):
    billing_amount:         float = Field(..., example=599.0)
    last_payment_days_ago:  float = Field(..., example=5.0)
    plan_tier:              str   = Field(..., example="standard")
    tenure_months:          float = Field(..., example=18.0)
    monthly_usage_hours:    float = Field(..., example=22.5)
    active_days:            float = Field(..., example=20.0)
    login_count:            float = Field(..., example=10.0)
    avg_session_min:        float = Field(..., example=25.0)
    device_count:           float = Field(..., example=2.0)
    add_on_count:           float = Field(..., example=1.0)
    support_tickets:        float = Field(..., example=0.0)
    sla_breaches:           float = Field(..., example=0.0)
    promotions_redeemed:    float = Field(..., example=1.0)
    email_opens:            float = Field(..., example=3.0)
    email_clicks:           float = Field(..., example=1.0)
    last_campaign_days_ago: float = Field(..., example=10.0)
    nps_score:              float = Field(..., example=7.0)
    region:                 str   = Field(..., example="north")
    is_autopay:             bool  = Field(..., example=True)
    is_discounted:          bool  = Field(..., example=False)
    has_family_bundle:      bool  = Field(..., example=False)


def _add_features(data: dict) -> dict:
    """Add engineered features to raw customer dict."""
    d = data.copy()
    d["engagement_rate"]   = min(d["active_days"] / 30.0, 1.0)
    d["usage_per_login"]   = d["monthly_usage_hours"] / (d["login_count"] + 1e-3)
    d["support_intensity"] = d["support_tickets"] + 3 * d["sla_breaches"]
    d["email_ctr"]         = d["email_clicks"] / (d["email_opens"] + 1e-3)
    d["price_to_tenure"]   = d["billing_amount"] / (d["tenure_months"] + 1e-3)
    return d


def _segment(prob: float) -> str:
    if prob >= 0.50:
        return "high"
    elif prob >= 0.25:
        return "medium"
    return "low"


def _retention_action(data: dict, prob: float) -> str:
    if prob < 0.25:
        return "No action needed"
    if data["engagement_rate"] < 0.3 and data["price_to_tenure"] > 20:
        return "Retention discount + plan downshift"
    if data["support_intensity"] > 3:
        return "Priority support callback + apology credit"
    if data["tenure_months"] > 24 and data["engagement_rate"] < 0.4:
        return "Re-activation campaign (feature tips)"
    if not data["is_autopay"] and data["last_payment_days_ago"] > 15:
        return "Autopay nudge + payment incentive"
    return "Personalized retention offer"


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/score")
def score(c: Customer):
    """Return churn probability and risk segment for a single customer."""
    data = _add_features(c.model_dump())
    X    = pd.DataFrame([data])
    prob = float(model.predict_proba(X)[0, 1])
    return {
        "churn_prob":       round(prob, 4),
        "segment":          _segment(prob),
        "suggested_action": _retention_action(data, prob),
    }


@app.post("/explain")
def explain(c: Customer):
    """Return lightweight top-5 churn driver hints."""
    data = _add_features(c.model_dump())
    
    # Rule-based explanations (SHAP computed in notebooks for global view)
    drivers = []
    if data["engagement_rate"] < 0.4:
        drivers.append({"feature": "engagement_rate", "direction": "↓ low",
                         "note": "Customer active only {:.0%} of days".format(data["engagement_rate"])})
    if data["support_intensity"] > 2:
        drivers.append({"feature": "support_intensity", "direction": "↑ high",
                         "note": f"{data['support_tickets']:.0f} tickets, {data['sla_breaches']:.0f} SLA breaches"})
    if data["nps_score"] < 5:
        drivers.append({"feature": "nps_score", "direction": "↓ low",
                         "note": f"NPS score is {data['nps_score']:.0f}/10"})
    if data["price_to_tenure"] > 25:
        drivers.append({"feature": "price_to_tenure", "direction": "↑ high",
                         "note": "High billing relative to tenure"})
    if not data["is_autopay"]:
        drivers.append({"feature": "is_autopay", "direction": "⚠ off",
                         "note": "No autopay — payment friction"})
    
    # Fill up to 5
    fallbacks = ["tenure_months", "usage_per_login", "email_ctr", "last_payment_days_ago", "add_on_count"]
    for fb in fallbacks:
        if len(drivers) >= 5:
            break
        if not any(d["feature"] == fb for d in drivers):
            drivers.append({"feature": fb, "direction": "—", "note": "Moderate signal"})
    
    return {"top_drivers": drivers[:5]}


@app.get("/topk")
def topk(k: int = 20):
    """Return top-K highest churn risk customers from batch scores."""
    scores_path = "data/churn_scores.parquet"
    if not os.path.exists(scores_path):
        raise HTTPException(status_code=404, detail="Batch scores not found. Run Notebook 06 first.")
    
    df = pd.read_parquet(scores_path)
    top = df.sort_values("churn_prob", ascending=False).head(k)
    
    return top[[
        "customer_id", "churn_prob", "risk_tier",
        "suggested_action", "plan_tier", "tenure_months",
        "billing_amount", "nps_score"
    ]].to_dict(orient="records")
