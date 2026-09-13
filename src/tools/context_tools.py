"""
Context Retrieval Tools
=======================
These functions simulate what, in a real bank, would be API calls to
internal systems (core banking, device-fingerprinting service, case
management system). Here they just query the synthetic CSVs — but the
interface is written so you could swap these for real API calls
without touching the agent graph.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data"

_transactions = pd.read_csv(DATA_DIR / "transactions.csv")
_devices = pd.read_csv(DATA_DIR / "device_ip_logs.csv")
_transactions["hour"] = pd.to_datetime(_transactions["timestamp"]).dt.hour


def get_transaction(transaction_id: str) -> dict:
    """Fetch the core transaction record."""
    row = _transactions[_transactions["transaction_id"] == transaction_id]
    if row.empty:
        raise ValueError(f"Transaction {transaction_id} not found.")
    return row.iloc[0].to_dict()


def get_device_info(transaction_id: str) -> dict:
    """Fetch device/IP metadata associated with a transaction."""
    row = _devices[_devices["transaction_id"] == transaction_id]
    if row.empty:
        raise ValueError(f"No device record for transaction {transaction_id}.")
    return row.iloc[0].to_dict()


def get_customer_history(customer_id: str, exclude_transaction_id: str = None,
                          lookback: int = 20) -> pd.DataFrame:
    """Fetch this customer's recent transaction history, most recent first."""
    hist = _transactions[_transactions["customer_id"] == customer_id]
    if exclude_transaction_id:
        hist = hist[hist["transaction_id"] != exclude_transaction_id]
    return hist.sort_values("timestamp", ascending=False).head(lookback)


def get_full_context(transaction_id: str) -> dict:
    """Convenience wrapper the Context Agent calls to gather everything
    about a flagged transaction in one shot."""
    txn = get_transaction(transaction_id)
    device = get_device_info(transaction_id)
    history = get_customer_history(txn["customer_id"], exclude_transaction_id=transaction_id)

    return {
        "transaction": txn,
        "device": device,
        "customer_history_summary": {
            "num_past_transactions": len(history),
            "avg_amount": round(history["amount_inr"].mean(), 2) if len(history) else None,
            "common_cities": history["city"].value_counts().head(3).to_dict() if len(history) else {},
            "common_merchants": history["merchant"].value_counts().head(3).to_dict() if len(history) else {},
        },
        "customer_history_df": history,  # kept for rule engine; drop before sending to LLM
    }
