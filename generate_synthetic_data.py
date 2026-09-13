"""
Synthetic Data Generator — Agentic Fraud Investigation Copilot
================================================================
Generates 4 datasets needed for the project, no real data or paid
services required:

  1. transactions.csv       - core transaction records
  2. device_ip_logs.csv     - device/IP metadata per transaction
  3. past_cases.csv         - short investigation case notes (for RAG)
  4. labeled_dataset.csv    - joined + labeled table for training/eval
                              of your risk-scoring model

Run:
    python generate_synthetic_data.py

Requires: pandas, numpy   (no internet / API keys needed)
"""

import random
import string
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

N_TRANSACTIONS = 2000
FRAUD_RATE = 0.06  # ~6% fraud, mimics real-world class imbalance

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
               "Ananya", "Diya", "Ishaan", "Kabir", "Anika", "Meera", "Rohan",
               "Priya", "Neha", "Karan", "Sanya", "Aryan", "Riya"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Reddy", "Iyer", "Kumar", "Singh",
              "Nair", "Das", "Chatterjee", "Patel", "Mehta", "Joshi", "Rao"]

MERCHANTS = ["Amazon", "Flipkart", "Swiggy", "Zomato", "BigBasket", "Myntra",
             "IRCTC", "PayU Recharge", "Electricity Board", "Uber", "Ola",
             "Local Kirana Store", "Cash Withdrawal ATM", "Crypto Exchange XYZ",
             "Foreign Exchange Kiosk", "Unknown Merchant Ltd"]

CITIES = [
    ("Delhi", 28.6139, 77.2090), ("Mumbai", 19.0760, 72.8777),
    ("Bangalore", 12.9716, 77.5946), ("Gurgaon", 28.4595, 77.0266),
    ("Pune", 18.5204, 73.8567), ("Chennai", 13.0827, 80.2707),
    ("Kolkata", 22.5726, 88.3639), ("Hyderabad", 17.3850, 78.4867),
    ("Jaipur", 26.9124, 75.7873), ("Lagos", 6.5244, 3.3792),         # foreign anomaly
    ("Dubai", 25.2048, 55.2708),                                     # foreign anomaly
]

DEVICE_TYPES = ["Android Phone", "iPhone", "Desktop Browser", "Tablet"]


def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def random_device_id():
    return "DEV-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def generate_transactions(n):
    rows = []
    base_time = datetime(2026, 1, 1)

    # Give each customer a "home city" and typical amount range so we can
    # inject realistic anomalies (location jump, amount spike, velocity).
    customers = [
        {
            "customer_id": f"CUST{1000+i}",
            "name": random_name(),
            "home_city": random.choice(CITIES[:9]),  # domestic home cities only
            "typical_amount": random.randint(200, 5000),
        }
        for i in range(300)
    ]

    for i in range(n):
        cust = random.choice(customers)
        is_fraud = random.random() < FRAUD_RATE
        txn_time = base_time + timedelta(
            days=random.randint(0, 250),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        if is_fraud:
            # Inject classic fraud patterns
            pattern = random.choice(["amount_spike", "geo_mismatch", "odd_hour", "new_merchant_high_value"])
            amount = round(cust["typical_amount"] * random.uniform(5, 20), 2)
            city = random.choice(CITIES[9:]) if pattern == "geo_mismatch" else cust["home_city"]
            merchant = random.choice(["Crypto Exchange XYZ", "Foreign Exchange Kiosk", "Unknown Merchant Ltd"])
            if pattern == "odd_hour":
                txn_time = txn_time.replace(hour=random.choice([1, 2, 3, 4]))
        else:
            amount = round(cust["typical_amount"] * random.uniform(0.5, 1.5), 2)
            city = cust["home_city"]
            merchant = random.choice(MERCHANTS[:-3])  # exclude the suspicious ones

        rows.append({
            "transaction_id": f"TXN{100000+i}",
            "customer_id": cust["customer_id"],
            "customer_name": cust["name"],
            "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
            "amount_inr": amount,
            "merchant": merchant,
            "city": city[0],
            "latitude": city[1],
            "longitude": city[2],
            "is_fraud": int(is_fraud),
        })

    return pd.DataFrame(rows)


def generate_device_logs(transactions_df):
    rows = []
    # Give each customer a small pool of "usual" devices; fraud txns
    # sometimes use a brand-new/unusual device.
    customer_devices = {}

    for _, txn in transactions_df.iterrows():
        cust_id = txn["customer_id"]
        if cust_id not in customer_devices:
            customer_devices[cust_id] = [
                {"device_id": random_device_id(), "device_type": random.choice(DEVICE_TYPES), "ip": random_ip()}
                for _ in range(random.randint(1, 2))
            ]

        if txn["is_fraud"] and random.random() < 0.7:
            # new/unrecognized device for this customer
            device = {"device_id": random_device_id(), "device_type": random.choice(DEVICE_TYPES), "ip": random_ip()}
        else:
            device = random.choice(customer_devices[cust_id])

        rows.append({
            "transaction_id": txn["transaction_id"],
            "device_id": device["device_id"],
            "device_type": device["device_type"],
            "ip_address": device["ip"],
            "is_new_device_for_customer": int(device["device_id"] not in [d["device_id"] for d in customer_devices[cust_id]]) if txn["is_fraud"] else 0,
        })

    return pd.DataFrame(rows)


CASE_TEMPLATES_FRAUD = [
    "Customer {name} reported an unrecognized transaction of ₹{amount} at {merchant}. "
    "Device used did not match customer's registered devices. Investigation confirmed unauthorized access; "
    "account was temporarily frozen and funds reversed.",
    "Multiple high-value transactions from {city} within a short window, inconsistent with customer's usual "
    "spending pattern in {home_city}. Flagged for geo-velocity mismatch. Confirmed as fraud after customer denied transactions.",
    "Transaction of ₹{amount} at {merchant} occurred at {hour}:00, an unusual hour for this customer. "
    "Combined with a new device login, this was escalated and confirmed as account takeover.",
    "Customer's card was used for a transaction at {merchant}, a merchant category associated with prior fraud cases. "
    "Amount was significantly above customer's typical transaction size. Confirmed fraudulent after review.",
]

CASE_TEMPLATES_LEGIT = [
    "Customer {name} made a transaction of ₹{amount} at {merchant} in {city}, consistent with prior spending history. "
    "No anomalies found; case closed as legitimate.",
    "Transaction flagged by automated rule due to slightly elevated amount, but device and location matched customer's "
    "usual profile. Confirmed legitimate after customer verification call.",
    "Routine purchase at {merchant}; amount and location consistent with customer behavior. No further action needed.",
]


def generate_past_cases(transactions_df, device_df, n_cases=250):
    merged = transactions_df.merge(device_df, on="transaction_id")
    sample = merged.sample(n=min(n_cases, len(merged)), random_state=42)

    rows = []
    for i, txn in sample.iterrows():
        if txn["is_fraud"]:
            template = random.choice(CASE_TEMPLATES_FRAUD)
        else:
            template = random.choice(CASE_TEMPLATES_LEGIT)

        note = template.format(
            name=txn["customer_name"],
            amount=txn["amount_inr"],
            merchant=txn["merchant"],
            city=txn["city"],
            home_city=txn["city"],
            hour=pd.to_datetime(txn["timestamp"]).hour,
        )

        rows.append({
            "case_id": f"CASE{5000+i}",
            "transaction_id": txn["transaction_id"],
            "customer_id": txn["customer_id"],
            "verdict": "Fraud Confirmed" if txn["is_fraud"] else "Legitimate",
            "case_note": note,
        })

    return pd.DataFrame(rows)


def build_labeled_dataset(transactions_df, device_df):
    """Joined, feature-ready table you can feed straight into your
    existing Logistic Regression / Random Forest risk model."""
    merged = transactions_df.merge(device_df, on="transaction_id")
    merged["hour"] = pd.to_datetime(merged["timestamp"]).dt.hour
    merged["is_odd_hour"] = merged["hour"].apply(lambda h: 1 if h in [0, 1, 2, 3, 4] else 0)
    merged["is_foreign_city"] = merged["city"].isin(["Lagos", "Dubai"]).astype(int)
    merged["is_high_risk_merchant"] = merged["merchant"].isin(
        ["Crypto Exchange XYZ", "Foreign Exchange Kiosk", "Unknown Merchant Ltd"]
    ).astype(int)

    cols = [
        "transaction_id", "customer_id", "amount_inr", "merchant", "city",
        "hour", "is_odd_hour", "is_foreign_city", "is_high_risk_merchant",
        "device_type", "is_new_device_for_customer", "is_fraud",
    ]
    return merged[cols]


def main():
    print("Generating transactions...")
    transactions_df = generate_transactions(N_TRANSACTIONS)

    print("Generating device/IP logs...")
    device_df = generate_device_logs(transactions_df)

    print("Generating past case notes (for RAG)...")
    past_cases_df = generate_past_cases(transactions_df, device_df)

    print("Building labeled dataset for model training...")
    labeled_df = build_labeled_dataset(transactions_df, device_df)

    transactions_df.to_csv("transactions.csv", index=False)
    device_df.to_csv("device_ip_logs.csv", index=False)
    past_cases_df.to_csv("past_cases.csv", index=False)
    labeled_df.to_csv("labeled_dataset.csv", index=False)

    print("\nDone. Files written:")
    print(f"  transactions.csv       ({len(transactions_df)} rows)")
    print(f"  device_ip_logs.csv     ({len(device_df)} rows)")
    print(f"  past_cases.csv         ({len(past_cases_df)} rows)")
    print(f"  labeled_dataset.csv    ({len(labeled_df)} rows)")
    print(f"\nFraud rate: {transactions_df['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()
