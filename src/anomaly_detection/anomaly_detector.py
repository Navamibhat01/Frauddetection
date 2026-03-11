"""
anomaly_detector.py
===================
Phases 8–11: Unsupervised Anomaly Detection & Risk Scoring

Pipeline:
1. Load GNN transaction embeddings
2. Apply Isolation Forest
3. Convert anomaly scores → risk scores
4. Normalize scores (with fallback if needed)
5. Generate BLOCK / MONITOR / ALLOW decisions
6. Save results
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

EMB_PATH = os.path.join(BASE_DIR, "data", "graph", "txn_embeddings.npy")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fraud_scores.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------
def main():

    print("=" * 60)
    print("PHASE 8 — Isolation Forest on GNN Transaction Embeddings")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load embeddings
    # ---------------------------------------------------------
    X = np.load(EMB_PATH)

    print("Transaction embeddings loaded:", X.shape)

    # ---------------------------------------------------------
    # Train Isolation Forest
    # ---------------------------------------------------------
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Isolation Forest...")
    iso.fit(X)

    scores = iso.decision_function(X)

    # convert so higher = more anomalous
    anomaly_scores = -scores

    # ---------------------------------------------------------
    # Normalize risk scores
    # ---------------------------------------------------------
    print("\nNormalizing risk scores...")

    if anomaly_scores.max() - anomaly_scores.min() < 1e-6:

        # fallback if scores are flat
        ranks = anomaly_scores.argsort().argsort()
        risk_scores = ranks / ranks.max()

        print("Used rank-based fallback normalization.")

    else:

        scaler = MinMaxScaler()

        risk_scores = scaler.fit_transform(
            anomaly_scores.reshape(-1, 1)
        ).flatten()

    # ---------------------------------------------------------
    # Decision Logic
    # ---------------------------------------------------------
    decisions = []
    predicted_fraud = []

    for score in risk_scores:

        if score > 0.80:
            decisions.append("BLOCK")
            predicted_fraud.append(1)

        elif score > 0.60:
            decisions.append("MONITOR")
            predicted_fraud.append(0)

        else:
            decisions.append("ALLOW")
            predicted_fraud.append(0)

    # ---------------------------------------------------------
    # Generate Transaction IDs
    # ---------------------------------------------------------
    txn_ids = [f"txn_{i}" for i in range(len(risk_scores))]

    # ---------------------------------------------------------
    # Create DataFrame
    # ---------------------------------------------------------
    df = pd.DataFrame({
        "transaction_id": txn_ids,
        "risk_score": np.round(risk_scores, 4),
        "predicted_fraud": predicted_fraud,
        "decision": decisions
    })

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------
    df.to_csv(OUTPUT_PATH, index=False)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n------------------------------")
    print("PIPELINE SUMMARY")
    print("------------------------------")

    print(df["decision"].value_counts())

    print("------------------------------")
    print("Average Risk Score:", df["risk_score"].mean())

    print("=" * 60)

    print("\nResults saved at:", OUTPUT_PATH)


# ---------------------------------------------------------
# RUN SCRIPT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()