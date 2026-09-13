"""
Rule-Based Fraud Check Engine
=============================
Deterministic, explainable checks that run alongside the ML model.
This is the kind of logic real fraud teams keep even in an
LLM-driven system, because rules are auditable and instantly
explainable to a compliance officer — the LLM's job is to reason
*over* these signals, not replace them.
"""

from dataclasses import dataclass, field
from typing import List
import pandas as pd

HIGH_RISK_MERCHANTS = {"Crypto Exchange XYZ", "Foreign Exchange Kiosk", "Unknown Merchant Ltd"}
FOREIGN_CITIES = {"Lagos", "Dubai"}
ODD_HOURS = set(range(0, 5))  # 12am - 4:59am
HIGH_VALUE_MULTIPLIER = 4  # flag if amount > 4x customer's historical average


@dataclass
class RuleResult:
    rule_name: str
    triggered: bool
    detail: str


@dataclass
class RuleCheckReport:
    transaction_id: str
    results: List[RuleResult] = field(default_factory=list)

    @property
    def triggered_rules(self):
        return [r for r in self.results if r.triggered]

    @property
    def rule_score(self):
        """Simple count-based score; each triggered rule = 1 point."""
        return len(self.triggered_rules)

    def to_dict(self):
        return {
            "transaction_id": self.transaction_id,
            "rule_score": self.rule_score,
            "triggered_rules": [r.rule_name for r in self.triggered_rules],
            "details": [r.detail for r in self.triggered_rules],
        }


def check_rules(transaction: dict, device: dict, customer_history: pd.DataFrame) -> RuleCheckReport:
    """
    transaction: dict with keys amount_inr, merchant, city, timestamp, hour
    device: dict with keys device_type, is_new_device_for_customer
    customer_history: DataFrame of this customer's past transactions
    """
    report = RuleCheckReport(transaction_id=transaction["transaction_id"])

    # Rule 1: High-risk merchant category
    is_high_risk_merchant = transaction["merchant"] in HIGH_RISK_MERCHANTS
    report.results.append(RuleResult(
        "high_risk_merchant", is_high_risk_merchant,
        f"Merchant '{transaction['merchant']}' is in the high-risk category."
    ))

    # Rule 2: Foreign / unusual location
    is_foreign = transaction["city"] in FOREIGN_CITIES
    report.results.append(RuleResult(
        "foreign_location", is_foreign,
        f"Transaction location '{transaction['city']}' is flagged as high-risk/foreign."
    ))

    # Rule 3: Odd-hour transaction
    hour = transaction.get("hour")
    is_odd_hour = hour in ODD_HOURS if hour is not None else False
    report.results.append(RuleResult(
        "odd_hour", is_odd_hour,
        f"Transaction occurred at {hour}:00, outside typical active hours."
    ))

    # Rule 4: New/unrecognized device
    is_new_device = bool(device.get("is_new_device_for_customer", 0))
    report.results.append(RuleResult(
        "new_device", is_new_device,
        f"Transaction used device '{device.get('device_id', 'unknown')}', not previously seen for this customer."
    ))

    # Rule 5: Amount significantly above customer's historical average
    if len(customer_history) > 0:
        avg_amount = customer_history["amount_inr"].mean()
        is_amount_spike = transaction["amount_inr"] > avg_amount * HIGH_VALUE_MULTIPLIER
    else:
        avg_amount = None
        is_amount_spike = False
    detail = (
        f"Amount ₹{transaction['amount_inr']:.2f} is {transaction['amount_inr']/avg_amount:.1f}x "
        f"the customer's historical average of ₹{avg_amount:.2f}."
        if avg_amount else "No transaction history available for comparison."
    )
    report.results.append(RuleResult("amount_spike", is_amount_spike, detail))

    return report
