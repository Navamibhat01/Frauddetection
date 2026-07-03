import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score
import time

# 1. PATHS
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EMB_PATH = os.path.join(BASE_DIR, "data", "graph", "txn_embeddings.npy")
LBL_PATH = os.path.join(BASE_DIR, "data", "graph", "txn_labels.npy")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "fraud_dataset_cleaned.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "output", "fraud_risk_results.csv")

# 2. LOAD DATA
X = np.load(EMB_PATH)
df = pd.read_csv(DATA_PATH)

# --- DYNAMIC SYNC LOGIC ---
# If new transactions were added, we must use the latest available embeddings
# and ensure the labels (y_true) match the current CSV length
current_count = len(X)
df = df.iloc[:current_count].reset_index(drop=True)

# If you want to include new transactions, you MUST run train_gnn.py first.
# Here we load the labels and ensure they match the embeddings perfectly.
y_true = np.load(LBL_PATH)[:current_count]

print(f"Adaptive Mode: Processing all {current_count} transactions.")

# 3. TRAIN & PREDICT
# We use a slightly higher contamination to account for new incoming fraud patterns
clf = IsolationForest(
    n_estimators=300, 
    contamination=0.16, 
    random_state=42, 
    n_jobs=-1
)

start_time = time.time()
y_pred_raw = clf.fit_predict(X) 
total_time = time.time() - start_time

# 4. CALCULATE RISK SCORES
decision_scores = clf.decision_function(X)
min_ds, max_ds = decision_scores.min(), decision_scores.max()
risk_scores = 100 * (1 - (decision_scores - min_ds) / (max_ds - min_ds))

# 5. GENERATE METRICS
y_pred_binary = np.where(y_pred_raw == -1, 1, 0)

print("\n" + "="*45)
print("GNN-BASED FRAUD DETECTION (LIVE SYNC)")
print("="*45)
print(f"Accuracy: {np.mean(y_pred_binary == y_true):.2%}")
print(f"Total Transactions Processed: {len(df)}")
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred_binary, target_names=['Legitimate', 'Fraud']))

# 6. SAVE EVERYTHING (Including the new transactions)
results_df = pd.DataFrame({
    'transaction_id': df['transaction_id'],
    'actual_label': y_true,
    'predicted_label': y_pred_binary,
    'risk_score': np.round(risk_scores, 2)
})

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print("-" * 45)
print(f"Live Results saved to: {OUTPUT_PATH}")
print("="*45)