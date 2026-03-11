import os
import pickle
import pandas as pd
import networkx as nx
from pyvis.network import Network

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# FIX: Point to the CLEANED dataset (not the raw one)
DATA_PATH  = os.path.join(BASE_DIR, "data", "raw", "fraud_dataset_cleaned.csv")
GRAPH_PATH = os.path.join(BASE_DIR, "data", "graph", "transaction_graph.pkl")
HTML_PATH  = os.path.join(BASE_DIR, "visualizations", "graph_visualization.html")

os.makedirs(os.path.join(BASE_DIR, "data", "graph"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "visualizations"), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CLEANED DATASET
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\Administrator\OneDrive\Desktop\fraud-detection-gnn\data\processed\fraud_dataset_cleaned.csv")

print("Dataset loaded")
print("Shape        :", df.shape)
print("Fraud rate   :", f"{df['is_fraud'].mean():.2%}")

# All 53 transaction-level feature columns (everything except IDs and label)
FEATURE_COLS = [c for c in df.columns
                if c not in ["transaction_id", "user_id", "merchant_id",
                             "device_id", "is_fraud"]]

print("Feature cols :", len(FEATURE_COLS))

# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────
G = nx.Graph()

# ── Pre-compute user-level aggregates ────────────────────────────────────────
user_stats = (
    df.groupby("user_id")
    .agg(
        total_txn        = ("transaction_id", "count"),
        avg_amount       = ("amount",         "mean"),
        # fraud_ratio is stored on the user node for message-passing enrichment
        # ONLY. is_fraud itself is never placed into the feature matrix X.
        fraud_ratio      = ("is_fraud",       "mean"),
        unique_merchants = ("merchant_id",    "nunique"),
        unique_devices   = ("device_id",      "nunique"),
    )
    .to_dict("index")
)

# ── Pre-compute merchant-level aggregates ────────────────────────────────────
merchant_stats = (
    df.groupby("merchant_id")
    .agg(
        total_txn    = ("transaction_id", "count"),
        fraud_ratio  = ("is_fraud",       "mean"),
        unique_users = ("user_id",        "nunique"),
    )
    .to_dict("index")
)

# ── Pre-compute device-level aggregates ──────────────────────────────────────
device_stats = (
    df.groupby("device_id")
    .agg(
        total_txn    = ("transaction_id", "count"),
        unique_users = ("user_id",        "nunique"),
    )
    .to_dict("index")
)

# ── Add all nodes and edges ───────────────────────────────────────────────────
for _, row in df.iterrows():

    txn_node      = f"txn_{row['transaction_id']}"
    user_node     = f"user_{row['user_id']}"
    merchant_node = f"merchant_{row['merchant_id']}"
    device_node   = f"device_{row['device_id']}"

    # ── Transaction node ──────────────────────────────────────────────────────
    # Store all 53 cleaned features as node attributes.
    # is_fraud is stored separately as 'label' — used as y during training,
    # NOT placed into the feature matrix X.
    txn_attrs = {feat: float(row[feat]) for feat in FEATURE_COLS}
    txn_attrs["type"]  = "transaction"
    txn_attrs["label"] = int(row["is_fraud"])   # training target only
    G.add_node(txn_node, **txn_attrs)

    # ── User node ─────────────────────────────────────────────────────────────
    uid    = row["user_id"]
    ustats = user_stats.get(uid, {})
    G.add_node(user_node,
               type             = "user",
               total_txn        = float(ustats.get("total_txn",        0)),
               avg_amount       = float(ustats.get("avg_amount",       0)),
               fraud_ratio      = float(ustats.get("fraud_ratio",      0)),
               unique_merchants = float(ustats.get("unique_merchants", 0)),
               unique_devices   = float(ustats.get("unique_devices",   0)))

    # ── Merchant node ─────────────────────────────────────────────────────────
    mid    = row["merchant_id"]
    mstats = merchant_stats.get(mid, {})
    G.add_node(merchant_node,
               type         = "merchant",
               total_txn    = float(mstats.get("total_txn",    0)),
               fraud_ratio  = float(mstats.get("fraud_ratio",  0)),
               unique_users = float(mstats.get("unique_users", 0)))

    # ── Device node ───────────────────────────────────────────────────────────
    did    = row["device_id"]
    dstats = device_stats.get(did, {})
    G.add_node(device_node,
               type         = "device",
               total_txn    = float(dstats.get("total_txn",    0)),
               unique_users = float(dstats.get("unique_users", 0)))

    # ── Edges (structural relationships) ─────────────────────────────────────
    G.add_edge(user_node, txn_node)        # user initiated transaction
    G.add_edge(txn_node, merchant_node)    # transaction at merchant
    G.add_edge(txn_node, device_node)      # transaction via device

# ── FIX: Add user→device edges (shared-device fraud ring signal) ─────────────
# Two users who share a device are likely linked — key structural fraud signal
user_device_map = df.groupby("user_id")["device_id"].unique()
for uid, devices in user_device_map.items():
    user_node = f"user_{uid}"
    for did in devices:
        device_node = f"device_{did}"
        if G.has_node(user_node) and G.has_node(device_node):
            G.add_edge(user_node, device_node)

print("\nGraph built successfully")
print("Total nodes :", G.number_of_nodes())
print("Total edges :", G.number_of_edges())

# Node type breakdown
type_counts = {}
for _, d in G.nodes(data=True):
    t = d.get("type", "unknown")
    type_counts[t] = type_counts.get(t, 0) + 1
for t, c in sorted(type_counts.items()):
    print(f"  {t:12s}: {c}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE GRAPH
# ─────────────────────────────────────────────────────────────────────────────
with open(GRAPH_PATH, "wb") as f:
    pickle.dump(G, f)

print(f"\nGraph saved → {GRAPH_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE VISUALIZATION (sampled — full graph is too large to render)
# ─────────────────────────────────────────────────────────────────────────────
sample_nodes = list(G.nodes())[:300]
H = G.subgraph(sample_nodes)

net = Network(height="750px", width="100%", bgcolor="#111111", font_color="white")

COLOR_MAP = {
    "user":        "#3498db",   # blue
    "merchant":    "#e74c3c",   # red
    "device":      "#2ecc71",   # green
    "transaction": "#f39c12",   # orange
}

for node, d in H.nodes(data=True):
    node_type = d.get("type", "transaction")
    # Highlight fraud transactions in bright red
    if node_type == "transaction" and d.get("label", 0) == 1:
        color = "#FF0000"
    else:
        color = COLOR_MAP.get(node_type, "#f39c12")
    net.add_node(node, label=node, color=color, size=12)

for source, target in H.edges():
    net.add_edge(source, target)

net.set_options("""
var options = {
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -12000,
      "springLength": 120
    },
    "minVelocity": 0.75
  }
}
""")

net.write_html(HTML_PATH)
print(f"Visualization saved → {HTML_PATH}")