import os
import pickle
import pandas as pd
import networkx as nx

# Dynamic Pathing based on your folder structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "fraud_dataset_cleaned.csv")
GRAPH_PATH = os.path.join(BASE_DIR, "data", "graph", "transaction_graph.pkl")

# Ensure directory exists
os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)

df = pd.read_csv(DATA_PATH)
G = nx.Graph()

print(f"Building Graph: {len(df)} transactions...")

for _, row in df.iterrows():
    tid = str(row['transaction_id']) # Use the UUID directly
    uid = f"user_{row['user_id']}"
    mid = f"merchant_{row['merchant_id']}"
    did = f"device_{row['device_id']}"

    # Add Transaction Node
    G.add_node(tid, type='transaction')
    
    # Structural Edges
    G.add_edge(tid, uid)
    G.add_edge(tid, mid)
    G.add_edge(tid, did)

# Add Fraud Ring Signal: Link users sharing the same device
device_groups = df.groupby('device_id')['user_id'].unique()
for users in device_groups:
    if len(users) > 1:
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                G.add_edge(f"user_{users[i]}", f"user_{users[j]}")

with open(GRAPH_PATH, "wb") as f:
    pickle.dump(G, f)

print(f"✅ Success: Graph saved to {GRAPH_PATH}")