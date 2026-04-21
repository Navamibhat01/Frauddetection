import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.utils import negative_sampling
from sklearn.preprocessing import StandardScaler

# Dynamic Pathing
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRAPH_PATH = os.path.join(BASE_DIR, "data", "graph", "transaction_graph.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "fraud_dataset_cleaned.csv")
TXN_EMB_PATH = os.path.join(BASE_DIR, "data", "graph", "txn_embeddings.npy")
LBL_PATH = os.path.join(BASE_DIR, "data", "graph", "txn_labels.npy")

# 1. Load Data
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)
df = pd.read_csv(DATASET_PATH)

ID_COLS = {"transaction_id", "user_id", "merchant_id", "device_id", "is_fraud"}
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS]
TXN_FEAT_DIM = len(FEATURE_COLS)

# 2. Match UUIDs from Graph to CSV Features
all_nodes = list(G.nodes())
node_to_idx = {n: i for i, n in enumerate(all_nodes)}
txn_feat_map = {str(tid): feat for tid, feat in zip(df["transaction_id"], df[FEATURE_COLS].values)}
txn_lbl_map = {str(tid): lbl for tid, lbl in zip(df["transaction_id"], df["is_fraud"])}

x_feats = np.zeros((len(all_nodes), TXN_FEAT_DIM), dtype=np.float32)
txn_indices, true_labels = [], []

for node in all_nodes:
    node_str = str(node)
    if node_str in txn_feat_map:
        idx = node_to_idx[node]
        x_feats[idx] = txn_feat_map[node_str]
        txn_indices.append(idx)
        true_labels.append(txn_lbl_map[node_str])

print(f"✅ Mapped {len(txn_indices)} transactions with features.")

# Tensors
x = torch.tensor(StandardScaler().fit_transform(x_feats), dtype=torch.float)
edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()

# 3. Inductive GNN Architecture
class GNNEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 128)
        self.bn1 = BatchNorm(128)
        self.conv2 = SAGEConv(128, 64)
        self.bn2 = BatchNorm(64)

    def forward(self, x, edge_index):
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.bn2(self.conv2(x, edge_index))

model = GNNEncoder(TXN_FEAT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Unsupervised Training
print("Training Adaptive GNN...")
for epoch in range(101):
    model.train()
    optimizer.zero_grad()
    z = model(x, edge_index)
    
    # Link Prediction Loss
    edge_src, edge_dst = edge_index
    pos_score = (z[edge_src] * z[edge_dst]).sum(dim=1)
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
    
    neg_edges = negative_sampling(edge_index, num_nodes=z.size(0))
    neg_src, neg_dst = neg_edges
    neg_loss = -torch.log(1 - torch.sigmoid((z[neg_src] * z[neg_dst]).sum(dim=1)) + 1e-15).mean()
    
    (pos_loss + neg_loss).backward()
    optimizer.step()

# 5. Save Transaction Embeddings
model.eval()
with torch.no_grad():
    final_z = model(x, edge_index).numpy()

np.save(TXN_EMB_PATH, final_z[txn_indices])
np.save(LBL_PATH, np.array(true_labels))
print(f"✅ DONE: Embeddings saved to {TXN_EMB_PATH}")