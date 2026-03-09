"""
train_gnn.py
============
UPI Fraud Detection — GCN Training Pipeline
--------------------------------------------
Steps covered:
  1. Load the NetworkX graph from transaction_graph.pkl
  2. Convert it to PyTorch Geometric (PyG) HeteroData / homogeneous Data
  3. Build a 2-layer Graph Convolutional Network (GCN)
  4. Train on transaction nodes for binary fraud classification
  5. Evaluate with precision, recall, and F1-score
  6. Extract and save node embeddings
"""

import os
import pickle
import warnings

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

# ─────────────────────────────────────────────────────────────────────────────
# 0. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRAPH_PATH = os.path.join(BASE_DIR, "data", "graph", "transaction_graph.pkl")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "graph")
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "node_embeddings.npy")
EMBEDDING_LABELS_PATH = os.path.join(EMBEDDINGS_DIR, "node_labels.npy")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD NETWORKX GRAPH
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading NetworkX graph")
print("=" * 60)

with open(GRAPH_PATH, "rb") as f:
    G: nx.Graph = pickle.load(f)

print(f"  Nodes : {G.number_of_nodes():,}")
print(f"  Edges : {G.number_of_edges():,}")

# Quick sample to inspect available attributes
sample_nodes = list(G.nodes(data=True))[:8]
print("\n  Sample node attributes:")
for n, d in sample_nodes:
    print(f"    {n}: {d}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD FEATURE MATRIX & LABELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Converting to PyTorch Geometric Data")
print("=" * 60)

# Assign a contiguous integer index to every node
all_nodes = list(G.nodes())
node_to_idx = {n: i for i, n in enumerate(all_nodes)}
num_nodes = len(all_nodes)

# ── Feature dimensions per node type ─────────────────────────────────────────
# transaction : amount, transaction_velocity, failed_transaction_count, is_fraud  → 4 dims used as features (is_fraud excluded for feature, kept as label)
# user        : total_txn, avg_amount, fraud_ratio, unique_merchants, unique_devices → 5 dims
# merchant    : total_txn, fraud_ratio, unique_users                               → 3 dims
# device      : total_txn, unique_users                                            → 2 dims
#
# Strategy: build a common feature vector of length MAX_DIM, zero-padding shorter ones.
MAX_DIM = 5  # max feature width

features = np.zeros((num_nodes, MAX_DIM), dtype=np.float32)
labels = np.full(num_nodes, -1, dtype=np.int64)   # -1 = non-transaction node
txn_indices = []  # global indices of transaction nodes

for node, data in G.nodes(data=True):
    idx = node_to_idx[node]
    ntype = data.get("type", "")

    if ntype == "transaction":
        features[idx, 0] = float(data.get("amount", 0.0))
        features[idx, 1] = float(data.get("transaction_velocity", 0.0))
        features[idx, 2] = float(data.get("failed_transaction_count", 0.0))
        # is_fraud is the label, not a feature
        label = int(data.get("is_fraud", 0))
        labels[idx] = label
        txn_indices.append(idx)

    elif ntype == "user":
        features[idx, 0] = float(data.get("total_txn", 0.0))
        features[idx, 1] = float(data.get("avg_amount", 0.0))
        features[idx, 2] = float(data.get("fraud_ratio", 0.0))
        features[idx, 3] = float(data.get("unique_merchants", 0.0))
        features[idx, 4] = float(data.get("unique_devices", 0.0))

    elif ntype == "merchant":
        features[idx, 0] = float(data.get("total_txn", 0.0))
        features[idx, 1] = float(data.get("fraud_ratio", 0.0))
        features[idx, 2] = float(data.get("unique_users", 0.0))

    elif ntype == "device":
        features[idx, 0] = float(data.get("total_txn", 0.0))
        features[idx, 1] = float(data.get("unique_users", 0.0))

# ── Normalise features ────────────────────────────────────────────────────────
scaler = StandardScaler()
features = scaler.fit_transform(features).astype(np.float32)

x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)

txn_mask = torch.zeros(num_nodes, dtype=torch.bool)
txn_mask[txn_indices] = True

print(f"  Total nodes              : {num_nodes:,}")
print(f"  Transaction nodes        : {txn_mask.sum().item():,}")
print(f"  Feature dimension        : {MAX_DIM}")

# ── Build edge index ──────────────────────────────────────────────────────────
src_list, dst_list = [], []
for u, v in G.edges():
    i, j = node_to_idx[u], node_to_idx[v]
    # Undirected → add both directions
    src_list.extend([i, j])
    dst_list.extend([j, i])

edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

# ── PyG Data object ───────────────────────────────────────────────────────────
data = Data(x=x, edge_index=edge_index, y=y)
data.txn_mask = txn_mask

print(f"  Edge index shape         : {edge_index.shape}")
print(f"  Data object              : {data}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT  (on transaction nodes only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Train / Test split")
print("=" * 60)

txn_idx_arr = np.array(txn_indices)
txn_labels_arr = labels[txn_idx_arr]

# Stratified split to preserve fraud class ratio
train_idx, test_idx = train_test_split(
    txn_idx_arr,
    test_size=0.20,
    stratify=txn_labels_arr,
    random_state=SEED,
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.test_mask = test_mask

fraud_count = int((txn_labels_arr == 1).sum())
legit_count = int((txn_labels_arr == 0).sum())
print(f"  Train nodes : {train_mask.sum().item():,}")
print(f"  Test  nodes : {test_mask.sum().item():,}")
print(f"  Fraud txns  : {fraud_count:,}  ({100 * fraud_count / len(txn_labels_arr):.1f}%)")
print(f"  Legit txns  : {legit_count:,}  ({100 * legit_count / len(txn_labels_arr):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL DEFINITION — 2-Layer GCN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Building 2-Layer GCN model")
print("=" * 60)

class FraudGCN(nn.Module):
    """
    Two-layer Graph Convolutional Network for node-level binary classification.

    Architecture:
        Input (MAX_DIM)
          └─► GCNConv → BatchNorm → ReLU → Dropout
                └─► GCNConv (embedding layer)  → BatchNorm → ReLU → Dropout
                      └─► Linear → 2-class logits
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1   = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2   = nn.BatchNorm1d(hidden_channels)
        self.lin   = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # ── Layer 1 ──────────────────────────────────────────────────────────
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # ── Layer 2 (produces node embeddings) ───────────────────────────────
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # ── Classifier head ──────────────────────────────────────────────────
        out = self.lin(x)
        return out, x   # (logits, embeddings)

IN_CHANNELS     = MAX_DIM
HIDDEN_CHANNELS = 64
OUT_CHANNELS    = 2   # binary: legit=0, fraud=1
DROPOUT         = 0.4

model = FraudGCN(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, DROPOUT)
print(f"  {model}")
print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASS WEIGHTS  (to handle imbalance)
# ─────────────────────────────────────────────────────────────────────────────
total = legit_count + fraud_count
w_legit = total / (2.0 * legit_count) if legit_count > 0 else 1.0
w_fraud = total / (2.0 * fraud_count) if fraud_count > 0 else 1.0
class_weights = torch.tensor([w_legit, w_fraud], dtype=torch.float)
print(f"\n  Class weights  → legit: {w_legit:.3f}  fraud: {w_fraud:.3f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Training GCN")
print("=" * 60)

EPOCHS     = 50
LR         = 1e-3
WEIGHT_DECAY = 5e-4
LOG_EVERY  = 20

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

x_t       = data.x
ei_t      = data.edge_index
y_t       = data.y
tr_mask   = data.train_mask

best_loss = float("inf")
best_state = None

print(f"  Epochs: {EPOCHS} | LR: {LR} | Hidden: {HIDDEN_CHANNELS} | Dropout: {DROPOUT}")
print()

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    logits, _ = model(x_t, ei_t)

    # Loss only on training transaction nodes
    loss = criterion(logits[tr_mask], y_t[tr_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()

    # ── Accuracy on train set (for logging) ──────────────────────────────────
    if epoch % LOG_EVERY == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(x_t, ei_t)
            preds_train = logits_eval[tr_mask].argmax(dim=1)
            acc_train = (preds_train == y_t[tr_mask]).float().mean().item()
        print(f"  Epoch {epoch:>4d}/{EPOCHS}  |  Loss: {loss.item():.4f}  |  Train Acc: {acc_train:.4f}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

# Restore best weights
model.load_state_dict(best_state)
print(f"\n  Best training loss: {best_loss:.4f} — best weights restored.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. EVALUATION  — Precision · Recall · F1
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Evaluation on Test Set")
print("=" * 60)

model.eval()
with torch.no_grad():
    logits_all, embeddings_all = model(x_t, ei_t)

te_mask = data.test_mask
preds_test = logits_all[te_mask].argmax(dim=1).numpy()
true_test  = y_t[te_mask].numpy()

precision = precision_score(true_test, preds_test, zero_division=0)
recall    = recall_score(true_test, preds_test, zero_division=0)
f1        = f1_score(true_test, preds_test, zero_division=0)

print(f"\n  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")

# Check unique labels to avoid ValueError in classification_report
unique_labels = np.unique(true_test)
print(f"  Unique labels in test set: {unique_labels}")

print("\n  Detailed classification report:")
try:
    print(classification_report(true_test, preds_test,
                                 labels=[0, 1],
                                 target_names=["Legit (0)", "Fraud (1)"],
                                 zero_division=0))
except ValueError as e:
    print(f"  Error generating report: {e}")
    print("  Falling back to simple report...")
    print(classification_report(true_test, preds_test, zero_division=0))

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE NODE EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Saving Node Embeddings")
print("=" * 60)

# Save embeddings for ALL nodes (shape: [num_nodes, HIDDEN_CHANNELS])
embeddings_np = embeddings_all.detach().numpy()
labels_np     = y_t.numpy()

np.save(EMBEDDINGS_PATH, embeddings_np)
np.save(EMBEDDING_LABELS_PATH, labels_np)

print(f"  Embeddings shape : {embeddings_np.shape}")
print(f"  Saved to         : {EMBEDDINGS_PATH}")
print(f"  Labels saved to  : {EMBEDDING_LABELS_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. TRANSACTION-NODE EMBEDDINGS (for downstream use)
# ─────────────────────────────────────────────────────────────────────────────
txn_embeddings = embeddings_np[txn_idx_arr]
txn_emb_path   = os.path.join(EMBEDDINGS_DIR, "txn_embeddings.npy")
np.save(txn_emb_path, txn_embeddings)
print(f"  Transaction embeddings: {txn_embeddings.shape}  → {txn_emb_path}")

print("\n" + "=" * 60)
print("  Pipeline complete!")
print("=" * 60)
