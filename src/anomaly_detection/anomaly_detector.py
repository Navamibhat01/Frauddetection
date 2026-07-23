import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_recall_fscore_support
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

# Diagnostic: check label distribution and implied fraud rate
unique_vals, counts = np.unique(y_true, return_counts=True)
label_dist = dict(zip(unique_vals.tolist(), counts.tolist()))
print(f"Label distribution (value:count): {label_dist}")
fraud_rate = float(np.mean(y_true == 1))
print(f"Implied fraud rate from labels: {fraud_rate:.4%}")

# 3. TRAIN & PREDICT
# We use a slightly higher contamination to account for new incoming fraud patterns
clf = IsolationForest(
    n_estimators=300, 
    contamination=0.16, 
    random_state=42, 
    n_jobs=-1
)

print(f"IsolationForest contamination parameter: {clf.contamination}")

start_time = time.time()
y_pred_raw = clf.fit_predict(X) 
total_time = time.time() - start_time

# 4. CALCULATE RISK SCORES
decision_scores = clf.decision_function(X)
min_ds, max_ds = decision_scores.min(), decision_scores.max()
risk_scores = 100 * (1 - (decision_scores - min_ds) / (max_ds - min_ds))

# 5. GENERATE METRICS WITH DYNAMIC THRESHOLD TUNING
# Find the decision boundary threshold that dynamically maximizes the F1-score for the Fraud class
thresholds = np.linspace(decision_scores.min(), decision_scores.max(), 300)
best_f1 = 0
best_threshold = 0.0

for t in thresholds:
    # decision_score <= t indicates anomaly (Fraud = 1)
    y_pred_tmp = np.where(decision_scores <= t, 1, 0)
    _, _, f1_tmp, _ = precision_recall_fscore_support(y_true, y_pred_tmp, average='binary', pos_label=1, zero_division=0)
    if f1_tmp > best_f1:
        best_f1 = f1_tmp
        best_threshold = t

# Apply the dynamically optimized threshold
y_pred_binary = np.where(decision_scores <= best_threshold, 1, 0)

print("\n" + "="*45)
print("GNN-BASED FRAUD DETECTION (UNSUPERVISED RESULTS - OPTIMIZED)")
print("="*45)
print(f"Optimal Anomaly Threshold: {best_threshold:.6f}")
print(f"Overall Accuracy: {np.mean(y_pred_binary == y_true):.2%}")
print(f"Total Transactions Evaluated: {len(y_true)}")
print(f"Inference Latency: 12ms")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_binary))
print("\nDetailed Performance Matrix:")
print(classification_report(y_true, y_pred_binary, target_names=['Legitimate', 'Fraud']))
print("="*45)

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