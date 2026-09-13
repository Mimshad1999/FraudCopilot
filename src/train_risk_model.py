"""
Risk-Scoring Model Trainer
==========================
Trains a Random Forest classifier on the labeled fraud dataset and
saves it as a pickled tool the agent system can call.

Run:
    python src/train_risk_model.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_PATH = Path(__file__).parent.parent / "data" / "labeled_dataset.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "risk_model.pkl"

NUMERIC_FEATURES = ["amount_inr", "hour", "is_odd_hour", "is_foreign_city",
                     "is_high_risk_merchant", "is_new_device_for_customer"]
CATEGORICAL_FEATURES = ["merchant", "city", "device_type"]


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",  # important: dataset is imbalanced
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("=== Model Evaluation ===")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
