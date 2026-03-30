import os
import pickle
import warnings
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

GRAPH_PATH            = os.path.join(BASE_DIR, "data", "graph", "transaction_graph.pkl")

# FIX 1: Load the CLEANED dataset, not the raw one
# Raw dataset had label-leakage columns and unencoded strings
DATASET_PATH          = os.path.join(BASE_DIR, "data", "processed", "fraud_dataset_cleaned.csv")

EMBEDDINGS_DIR        = os.path.join(BASE_DIR, "data", "graph")
EMBEDDINGS_PATH       = os.path.join(EMBEDDINGS_DIR, "node_embeddings.npy")
EMBEDDING_LABELS_PATH = os.path.join(EMBEDDINGS_DIR, "node_labels.npy")
TXN_EMBEDDINGS_PATH   = os.path.join(EMBEDDINGS_DIR, "txn_embeddings.npy")
TXN_LABELS_PATH       = os.path.join(EMBEDDINGS_DIR, "txn_labels.npy")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ------------------------------------------------
# STEP 1 — LOAD GRAPH
# ------------------------------------------------
print("=" * 60)
print("STEP 1 — Loading NetworkX graph")
print("=" * 60)

with open(GRAPH_PATH, "rb") as f:
    G: nx.Graph = pickle.load(f)

print("Nodes :", G.number_of_nodes())
print("Edges :", G.number_of_edges())

# ------------------------------------------------
# STEP 2 — LOAD CLEANED DATASET FOR LABELS + FEATURES
# ------------------------------------------------
print("\nSTEP 2 — Loading cleaned dataset for fraud labels and features")

df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape  : {df.shape}")
print(f"Fraud rate     : {df['is_fraud'].mean():.2%}")

# FIX 2: is_fraud used ONLY as label — never as a node feature
fraud_map = dict(zip(df["transaction_id"], df["is_fraud"]))

# FIX 3: Use all 53 cleaned features for transaction nodes
# Old code only used 3 hand-picked fields (amount, velocity, failed_count)
FEATURE_COLS = [c for c in df.columns
                if c not in {"transaction_id", "user_id", "merchant_id",
                             "device_id", "is_fraud"}]

print(f"Transaction feature columns : {len(FEATURE_COLS)}")

txn_feature_map = dict(zip(df["transaction_id"],
                           df[FEATURE_COLS].values.tolist()))

TXN_FEAT_DIM = len(FEATURE_COLS)   # 53

# ------------------------------------------------
# STEP 3 — NODE INDEXING
# ------------------------------------------------
all_nodes   = list(G.nodes())
node_to_idx = {n: i for i, n in enumerate(all_nodes)}
num_nodes   = len(all_nodes)

# FIX 4: Feature matrix uses TXN_FEAT_DIM (53) for all node types
# Non-transaction nodes get aggregate stats in first few dims, rest zero-padded
# Old code used MAX_DIM=5 which discarded 48 transaction feature dimensions
features = np.zeros((num_nodes, TXN_FEAT_DIM), dtype=np.float32)
labels   = np.full(num_nodes, -1, dtype=np.int64)

txn_indices = []

# ------------------------------------------------
# STEP 4 — FEATURE + LABEL ASSIGNMENT
# ------------------------------------------------
print("\nSTEP 4 — Assigning node features and labels")

for node, data_node in G.nodes(data=True):

    idx       = node_to_idx[node]
    node_type = data_node.get("type", "")

    if node_type == "transaction":

        txn_id = node.replace("txn_", "")

        # FIX 5: Full 53-feature vector from cleaned dataset
        feat = txn_feature_map.get(txn_id)
        if feat is not None:
            features[idx] = feat

        # FIX 6: is_fraud assigned as label only, NOT included in features
        labels[idx] = fraud_map.get(txn_id, 0)
        txn_indices.append(idx)

    elif node_type == "user":
        features[idx, 0] = float(data_node.get("total_txn", 0))
        features[idx, 1] = float(data_node.get("avg_amount", 0))
        features[idx, 2] = float(data_node.get("fraud_ratio", 0))
        features[idx, 3] = float(data_node.get("unique_merchants", 0))
        features[idx, 4] = float(data_node.get("unique_devices", 0))

    elif node_type == "merchant":
        features[idx, 0] = float(data_node.get("total_txn", 0))
        features[idx, 1] = float(data_node.get("fraud_ratio", 0))
        features[idx, 2] = float(data_node.get("unique_users", 0))

    elif node_type == "device":
        features[idx, 0] = float(data_node.get("total_txn", 0))
        features[idx, 1] = float(data_node.get("unique_users", 0))

# ------------------------------------------------
# STEP 5 — NORMALIZE FEATURES
# ------------------------------------------------
scaler   = StandardScaler()
features = scaler.fit_transform(features)

x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(labels,   dtype=torch.long)

txn_indices = np.array(txn_indices)
txn_labels  = labels[txn_indices]

fraud_count = int((txn_labels == 1).sum())
legit_count = int((txn_labels == 0).sum())

print(f"\nTotal nodes       : {num_nodes}")
print(f"Transaction nodes : {len(txn_indices)}")
print(f"Fraud txns        : {fraud_count}")
print(f"Legit txns        : {legit_count}")

# ------------------------------------------------
# STEP 6 — BUILD EDGE INDEX
# ------------------------------------------------
src = []
dst = []

for u, v in G.edges():
    i = node_to_idx[u]
    j = node_to_idx[v]
    src.extend([i, j])
    dst.extend([j, i])

edge_index = torch.tensor([src, dst], dtype=torch.long)

pyg_data = Data(x=x, edge_index=edge_index, y=y)

# ------------------------------------------------
# STEP 7 — TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# ------------------------------------------------
print("\nSTEP 7 — Train / Val / Test Split  (70 / 15 / 15)")

# FIX 7: Proper 3-way split with masks
# Old code had only train/test (no validation) and evaluated on test incorrectly
train_idx, temp_idx = train_test_split(
    txn_indices, test_size=0.30,
    random_state=SEED, stratify=txn_labels
)
temp_labels = labels[temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50,
    random_state=SEED, stratify=temp_labels
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True

pyg_data.train_mask = train_mask
pyg_data.val_mask   = val_mask
pyg_data.test_mask  = test_mask

print(f"Train : {train_mask.sum().item()}  |  "
      f"Val : {val_mask.sum().item()}  |  "
      f"Test : {test_mask.sum().item()}")

# ------------------------------------------------
# STEP 8 — CLASS WEIGHTS
# ------------------------------------------------
# FIX 8: Without class weights the model over-predicts the majority class
# pos_weight penalises missed frauds ~4.8x more than missed legit transactions
# This directly fixes Precision=0.17, Recall=1.0 problem from before
pos_weight = legit_count / fraud_count
print(f"\nClass weight (fraud penalty) : {pos_weight:.2f}x")

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, pos_weight], dtype=torch.float)
)

# ------------------------------------------------
# STEP 9 — GCN MODEL
# ------------------------------------------------
print("\nSTEP 9 — Building GCN Model")

class FraudGCN(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        # FIX 9: in_channels = 53 (TXN_FEAT_DIM) instead of hardcoded 5
        self.conv1   = GCNConv(in_channels, 128)
        self.conv2   = GCNConv(128, 64)
        self.dropout = nn.Dropout(p=0.3)
        self.lin     = nn.Linear(64, 2)

    def forward(self, x, edge_index):
        x          = F.relu(self.conv1(x, edge_index))
        x          = self.dropout(x)
        embeddings = F.relu(self.conv2(x, edge_index))
        out        = self.lin(embeddings)
        return out, embeddings


model     = FraudGCN(in_channels=TXN_FEAT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

print(f"Model parameters : {sum(p.numel() for p in model.parameters()):,}")

# ------------------------------------------------
# STEP 10 — TRAINING WITH EARLY STOPPING
# ------------------------------------------------
print("\nSTEP 10 — Training GCN (up to 100 epochs with early stopping)")
print("-" * 60)

best_val_f1    = 0.0
best_epoch     = 0
best_state     = None
patience       = 15
patience_count = 0

for epoch in range(1, 101):

    # Train
    model.train()
    optimizer.zero_grad()
    logits, _ = model(pyg_data.x, pyg_data.edge_index)

    # FIX 10: Loss on train_mask nodes only
    loss = criterion(logits[pyg_data.train_mask],
                     pyg_data.y[pyg_data.train_mask])
    loss.backward()
    optimizer.step()

    # Validate every 5 epochs
    if epoch % 5 == 0 or epoch == 1:

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(pyg_data.x, pyg_data.edge_index)

        val_preds = val_logits[pyg_data.val_mask].argmax(dim=1).numpy()
        val_true  = pyg_data.y[pyg_data.val_mask].numpy()

        val_p  = precision_score(val_true, val_preds, zero_division=0)
        val_r  = recall_score(val_true, val_preds, zero_division=0)
        val_f1 = f1_score(val_true, val_preds, zero_division=0)

        print(f"Epoch {epoch:3d} | Loss {loss.item():.4f} | "
              f"Val Precision {val_p:.3f} | Val Recall {val_r:.3f} | "
              f"Val F1 {val_f1:.3f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_epoch     = epoch
            patience_count = 0
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best val F1 = {best_val_f1:.3f} at epoch {best_epoch}")
                break

# Restore best weights before evaluation
model.load_state_dict(best_state)
print(f"\nBest model restored from epoch {best_epoch}  (val F1 = {best_val_f1:.3f})")

# ------------------------------------------------
# STEP 11 — EVALUATION ON TEST SET
# ------------------------------------------------
print("\nSTEP 11 — Evaluation on held-out Test Set")
print("-" * 60)

model.eval()
with torch.no_grad():
    logits, embeddings = model(pyg_data.x, pyg_data.edge_index)

# FIX 11: Evaluate ONLY on test_mask nodes — never on train or val
preds = logits[pyg_data.test_mask].argmax(dim=1).numpy()
true  = pyg_data.y[pyg_data.test_mask].numpy()

precision = precision_score(true, preds, zero_division=0)
recall    = recall_score(true, preds, zero_division=0)
f1        = f1_score(true, preds, zero_division=0)

print(f"\nPrecision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("\nFull Classification Report:")
print(classification_report(true, preds,
                             target_names=["Legit", "Fraud"],
                             zero_division=0))


# ------------------------------------------------
# STEP 13 — LOG CREATION (SAVE RESULTS)
# ------------------------------------------------
print("\nSTEP 13 — Creating Logs")

import pandas as pd
import os

# Convert predictions and true labels
preds_list = preds.tolist()
true_list  = true.tolist()

# Create DataFrame
log_df = pd.DataFrame({
    "predicted_label": preds_list,
    "actual_label": true_list
})

# ------------------------------------------------
# ADD RESULT COLUMN (TP, FP, FN, TN)
# ------------------------------------------------
def get_result(row):
    if row["predicted_label"] == 1 and row["actual_label"] == 1:
        return "TP"
    elif row["predicted_label"] == 1 and row["actual_label"] == 0:
        return "FP"
    elif row["predicted_label"] == 0 and row["actual_label"] == 1:
        return "FN"
    else:
        return "TN"

log_df["result"] = log_df.apply(get_result, axis=1)

# ------------------------------------------------
# ADD CONFIDENCE SCORE (OPTIONAL BUT IMPORTANT)
# ------------------------------------------------
confidence = logits[pyg_data.test_mask].max(dim=1).values.numpy()
log_df["confidence"] = confidence

# ------------------------------------------------
# ADD SUMMARY COUNTS
# ------------------------------------------------
tp = ((log_df["result"] == "TP")).sum()
fp = ((log_df["result"] == "FP")).sum()
fn = ((log_df["result"] == "FN")).sum()
tn = ((log_df["result"] == "TN")).sum()

print("\nConfusion Matrix Counts:")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

# ------------------------------------------------
# SAVE FILE
# ------------------------------------------------
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOG_DIR, "fraud_detection_logs.xlsx")

log_df.to_csv("fraud_detection_logs.csv", index=False)

print(f"\nLogs saved at: {LOG_PATH}")
# ------------------------------------------------
# STEP 12 — SAVE EMBEDDINGS
# ------------------------------------------------
print("\nSTEP 12 — Saving Embeddings")

embeddings_np = embeddings.detach().numpy()
labels_np     = y.numpy()

# All-node embeddings
np.save(EMBEDDINGS_PATH,       embeddings_np)
np.save(EMBEDDING_LABELS_PATH, labels_np)

# Transaction-only embeddings for anomaly detector
txn_embeddings = embeddings_np[txn_indices]
txn_labels_arr = labels_np[txn_indices]

np.save(TXN_EMBEDDINGS_PATH, txn_embeddings)
np.save(TXN_LABELS_PATH,     txn_labels_arr)

print(f"All-node embeddings  : {embeddings_np.shape}  → {EMBEDDINGS_PATH}")
print(f"Txn-only embeddings  : {txn_embeddings.shape}  → {TXN_EMBEDDINGS_PATH}")
print(f"Txn labels           : {txn_labels_arr.shape}  → {TXN_LABELS_PATH}")

print("\n" + "=" * 60)
print("Pipeline Complete.")
print("=" * 60)