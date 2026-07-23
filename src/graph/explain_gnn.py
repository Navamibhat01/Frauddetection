import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.preprocessing import StandardScaler

# Dynamic Pathing
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRAPH_PATH = os.path.join(BASE_DIR, "data", "graph", "transaction_graph.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "fraud_dataset_cleaned.csv")
MODEL_STATE_PATH = os.path.join(BASE_DIR, "data", "graph", "gnn_encoder.pth")

# 1. GNN Encoder Definition (Must match train_gnn.py exactly)
class GNNEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 128)
        self.bn1 = BatchNorm(128)
        self.conv2 = SAGEConv(128, 64)
        self.bn2 = BatchNorm(64)

    def forward(self, x, edge_index):
        x = nn.functional.elu(self.bn1(self.conv1(x, edge_index)))
        x = nn.functional.dropout(x, p=0.3, training=self.training)
        return self.bn2(self.conv2(x, edge_index))

def load_data_and_model():
    """Loads dataset, graph, and model checkpoint."""
    if not os.path.exists(GRAPH_PATH) or not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Graph or dataset files missing. Please clean data and build graph first.")
    
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    
    df = pd.read_csv(DATASET_PATH)
    
    # Dynamic sync: Add new transactions in df to G
    for _, row in df.iterrows():
        tid = str(row['transaction_id'])
        if tid not in G:
            uid = f"user_{row['user_id']}"
            mid = f"merchant_{row['merchant_id']}"
            did = f"device_{row['device_id']}"
            
            G.add_node(tid, type='transaction')
            G.add_edge(tid, uid)
            G.add_edge(tid, mid)
            G.add_edge(tid, did)
            
    ID_COLS = {"transaction_id", "user_id", "merchant_id", "device_id", "is_fraud"}
    FEATURE_COLS = [c for c in df.columns if c not in ID_COLS]
    TXN_FEAT_DIM = len(FEATURE_COLS)
    
    all_nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    txn_feat_map = {str(tid): feat for tid, feat in zip(df["transaction_id"], df[FEATURE_COLS].values)}
    
    x_feats = np.zeros((len(all_nodes), TXN_FEAT_DIM), dtype=np.float32)
    for node in all_nodes:
        node_str = str(node)
        if node_str in txn_feat_map:
            x_feats[node_to_idx[node]] = txn_feat_map[node_str]
            
    # Normalize features using fitted scaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_feats)
    
    x_tensor = torch.tensor(x_scaled, dtype=torch.float)
    edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()
    
    # Load model state
    model = GNNEncoder(TXN_FEAT_DIM)
    if os.path.exists(MODEL_STATE_PATH):
        model.load_state_dict(torch.load(MODEL_STATE_PATH))
    model.eval()
    
    return G, df, FEATURE_COLS, x_tensor, edge_index, all_nodes, node_to_idx, model

def explain_transaction(txn_id, epochs=100):
    """
    Runs GNNExplainer for a specific transaction ID.
    Returns:
        - feature_importances: list of dicts (feature_name, importance)
        - subgraph_edges: list of dicts (source, target, importance)
    """
    G, df, FEATURE_COLS, x, edge_index, all_nodes, node_to_idx, model = load_data_and_model()
    
    txn_str = str(txn_id)
    if txn_str not in node_to_idx:
        raise ValueError(f"Transaction ID {txn_id} not found in the graph.")
        
    target_idx = node_to_idx[txn_str]
    
    # Define explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw',
        ),
    )
    
    # Generate explanation
    explanation = explainer(x, edge_index, index=target_idx)
    
    # 1. Feature Importances
    # Target node mask shape is [num_nodes, num_features]
    node_mask = explanation.node_mask.numpy()
    target_feature_mask = node_mask[target_idx]
    
    # Normalize feature importances to [0, 1]
    if target_feature_mask.max() > 0:
        target_feature_mask = target_feature_mask / target_feature_mask.max()
        
    feature_importances = []
    for feat_name, imp in zip(FEATURE_COLS, target_feature_mask):
        feature_importances.append({
            "feature": feat_name,
            "importance": float(imp)
        })
    feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)
    
    # 2. Subgraph and Edge Importances
    edge_mask = explanation.edge_mask.numpy()
    # Normalize edge mask
    if edge_mask.max() > 0:
        edge_mask = edge_mask / edge_mask.max()
        
    subgraph_edges = []
    edges_list = edge_index.t().numpy()
    
    # Find edges in the target node's local neighborhood
    for i, (u_idx, v_idx) in enumerate(edges_list):
        if u_idx == target_idx or v_idx == target_idx:
            u_node = all_nodes[u_idx]
            v_node = all_nodes[v_idx]
            subgraph_edges.append({
                "source": str(u_node),
                "target": str(v_node),
                "importance": float(edge_mask[i])
            })
            
    # Sort subgraph edges by importance
    subgraph_edges = sorted(subgraph_edges, key=lambda x: x["importance"], reverse=True)
    
    return feature_importances, subgraph_edges
