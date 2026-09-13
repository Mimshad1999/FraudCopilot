"""
Risk-Scoring Tool
=================
Wraps the trained sklearn pipeline as a callable "tool" the agent
graph can invoke, exactly the way your Airtel risk-scoring framework
would be exposed to any downstream system.
"""

import pickle
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "risk_model.pkl"

with open(MODEL_PATH, "rb") as f:
    _model = pickle.load(f)

FEATURE_COLUMNS = ["amount_inr", "hour", "is_odd_hour", "is_foreign_city",
                    "is_high_risk_merchant", "is_new_device_for_customer",
                    "merchant", "city", "device_type"]


def score_transaction(txn: dict, device: dict, rule_report) -> dict:
    """Builds the feature row and returns a fraud probability + label."""
    row = {
        "amount_inr": txn["amount_inr"],
        "hour": txn.get("hour", pd.to_datetime(txn["timestamp"]).hour),
        "is_odd_hour": int(any(r.rule_name == "odd_hour" and r.triggered for r in rule_report.results)),
        "is_foreign_city": int(any(r.rule_name == "foreign_location" and r.triggered for r in rule_report.results)),
        "is_high_risk_merchant": int(any(r.rule_name == "high_risk_merchant" and r.triggered for r in rule_report.results)),
        "is_new_device_for_customer": int(device.get("is_new_device_for_customer", 0)),
        "merchant": txn["merchant"],
        "city": txn["city"],
        "device_type": device.get("device_type", "Unknown"),
    }
    X = pd.DataFrame([row])[FEATURE_COLUMNS]
    proba = _model.predict_proba(X)[0, 1]
    label = "High Risk" if proba >= 0.5 else "Low Risk"

    return {
        "fraud_probability": round(float(proba), 4),
        "predicted_label": label,
    }
