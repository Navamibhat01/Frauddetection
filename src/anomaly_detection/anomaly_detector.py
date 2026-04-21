import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import time

# 1. PATHS
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EMB_PATH = os.path.join(BASE_DIR, "data", "graph", "txn_embeddings.npy")
LBL_PATH = os.path.join(BASE_DIR, "data", "graph", "txn_labels.npy")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "fraud_dataset_cleaned.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "output", "fraud_risk_results.csv")

# 2. LOAD DATA
X = np.load(EMB_PATH)
y_true = np.load(LBL_PATH)
df = pd.read_csv(DATA_PATH)

print(f"⚙️ Processing {len(X)} transaction embeddings...")

# 3. TRAIN & PREDICT
# contamination=0.15 matches your ~17% fraud rate in the dataset
clf = IsolationForest(contamination=0.15, random_state=42, n_jobs=-1)

start_time = time.time()
y_pred_raw = clf.fit_predict(X) 
total_time = time.time() - start_time

# 4. CALCULATE RISK SCORES (0-100)
# Lower decision_function values = higher anomaly
decision_scores = clf.decision_function(X)
min_ds, max_ds = decision_scores.min(), decision_scores.max()
# Scaling: 100 is High Risk, 0 is Low Risk
risk_scores = 100 * (1 - (decision_scores - min_ds) / (max_ds - min_ds))

# 5. GENERATE METRICS
# Convert Isolation Forest output (-1: anomaly, 1: normal) to binary (1: fraud, 0: legit)
y_pred_binary = [1 if p == -1 else 0 for p in y_pred_raw]

print("\n" + "="*45)
print("🚀 GNN-BASED FRAUD DETECTION METRICS")
print("="*45)
print(f"⏱️ Avg Latency: {(total_time/len(X))*1000:.4f} ms per txn")
print(f"📊 Accuracy: {np.mean(np.array(y_pred_binary) == y_true):.2%}")
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred_binary, target_names=['Legitimate', 'Fraud']))

# 6. SAVE DETAILED RESULTS
results_df = pd.DataFrame({
    'transaction_id': df['transaction_id'],
    'actual_label': y_true,
    'predicted_label': y_pred_binary,
    'risk_score': np.round(risk_scores, 2)
})

# Categorize Risk
results_df['risk_level'] = pd.cut(results_df['risk_score'], 
                                 bins=[0, 35, 75, 100], 
                                 labels=['Low', 'Medium', 'High'],
                                 include_lowest=True)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print("-" * 45)
print(f"✅ Results & Risk Scores saved to:\n   {OUTPUT_PATH}")
print("="*45)

# 7. TEMPORAL STABILITY (The "Adaptive" Proof)
chunks = 4
c_size = len(y_true) // chunks
print("\n📈 Adaptive Stability (F1-Score across time windows):")
for i in range(chunks):
    f1 = f1_score(y_true[i*c_size:(i+1)*c_size], y_pred_binary[i*c_size:(i+1)*c_size])
    print(f"   Window {i+1}: {f1:.4f}")