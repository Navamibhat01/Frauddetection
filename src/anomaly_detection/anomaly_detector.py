"""
anomaly_detector.py
===================
Phases 8–11: Unsupervised Anomaly Detection & Risk Scoring
---------------------------------------------------------
Pipeline:
  8.  Isolation Forest on GNN transaction embeddings
  9.  Robust risk normalization (rank fallback)
 10.  Percentile-based fraud selection
 11.  BLOCK / MONITOR / ALLOW decisions
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

GRAPH_PATH = os.path.join(BASE_DIR, "data", "graph", "transaction_graph.pkl")
EMB_PATH   = os.path.join(BASE_DIR, "data", "graph", "txn_embeddings.npy")
OUT_DIR    = os.path.join(BASE_DIR, "data", "output")
OUT_PATH   = os.path.join(OUT_DIR, "fraud_scores.csv")

os.makedirs(OUT_DIR, exist_ok=True)


def main():

    print("=" * 60)
    print("PHASE 8 — Isolation Forest on GNN Embeddings")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load embeddings
    # ---------------------------------------------------------
    X = np.load(EMB_PATH)
    print(f"Loaded embeddings: {X.shape}")

    # ---------------------------------------------------------
    # Load graph (transaction IDs)
    # ---------------------------------------------------------
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    txn_ids = [n for n, d in G.nodes(data=True) if d.get("type") == "transaction"]

    if len(txn_ids) != X.shape[0]:
        print("Warning: Txn ID mismatch — using indices.")
        txn_ids = [f"txn_{i}" for i in range(X.shape[0])]

    # ---------------------------------------------------------
    # Isolation Forest
    # ---------------------------------------------------------
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )

    print("\nFitting Isolation Forest...")
    iso.fit(X)

    raw_scores = iso.decision_function(X)

    # Convert so higher = more anomalous
    anomaly_scores = -raw_scores

    # ---------------------------------------------------------
    # PHASE 9 — Robust Risk Scaling
    # ---------------------------------------------------------
    max_score = anomaly_scores.max()

    if max_score == 0 or np.isnan(max_score):
        # Rank fallback if flat
        ranks = anomaly_scores.argsort().argsort()
        risk_scores = ranks / ranks.max()
        print("Used rank-based risk (flat score fallback).")
    else:
        risk_scores = anomaly_scores / max_score

    # ---------------------------------------------------------
    # PHASE 10 — Rank-based Fraud Selection (Top 5%)
    # ---------------------------------------------------------
    cutoff = int(0.95 * len(risk_scores))
    sorted_idx = np.argsort(risk_scores)

    fraud_mask = np.zeros(len(risk_scores), dtype=bool)
    fraud_mask[sorted_idx[cutoff:]] = True

    decisions = []
    synthetic_fraud = []

    for score, is_fraud in zip(risk_scores, fraud_mask):

        if is_fraud:
            decisions.append("BLOCK")
            synthetic_fraud.append(1)

        elif score > 0.5:
            decisions.append("MONITOR")
            synthetic_fraud.append(0)

        else:
            decisions.append("ALLOW")
            synthetic_fraud.append(0)

    # ---------------------------------------------------------
    # PHASE 11 — Save CSV
    # ---------------------------------------------------------
    df = pd.DataFrame({
        "txn_id": txn_ids,
        "risk_score": np.round(risk_scores, 4),
        "is_fraud_synthetic": synthetic_fraud,
        "decision": decisions
    })

    df.to_csv(OUT_PATH, index=False)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n------------------------------")
    print("PIPELINE SUMMARY")
    print("------------------------------")
    print(df["decision"].value_counts())
    print("------------------------------")
    print(f"Average Risk Score: {df['risk_score'].mean():.4f}")
    print("=" * 60)

    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()