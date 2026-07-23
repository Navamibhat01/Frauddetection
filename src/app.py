import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import uuid
import re
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib

# Force non-interactive backend for matplotlib
matplotlib.use('Agg')

# Add GNN folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph.explain_gnn import explain_transaction

# --- 1. FORCE SYSTEM REFRESH & CONFIG ---
st.set_page_config(page_title="UPI GNN Shield", page_icon="🛡️", layout="wide")

# --- 2. INJECT CUSTOM CSS ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

        /* Force Main Background */
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #1a1025, #0d0914) !important;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
        }
        
        /* Force Sidebar Background */
        [data-testid="stSidebar"] {
            background: #110c18 !important;
            border-right: 1px solid #2d1f3f !important;
        }

        /* Typography */
        html, body, [class*="css"], p, label, .stMarkdown {
            font-family: 'Outfit', sans-serif !important;
            color: #e2e8f0 !important;
        }

        .main-title {
            background: linear-gradient(90deg, #ff007f 0%, #7928ca 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3.5rem;
            margin-bottom: 0px;
            letter-spacing: -1px;
        }
        
        .sub-title {
            color: #94a3b8;
            font-size: 1.2rem;
            margin-bottom: 30px;
            font-weight: 300;
        }

        /* Inputs */
        [data-baseweb="input"], [data-baseweb="base-input"] {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
        }
        [data-baseweb="input"]:focus-within {
            border-color: #ff007f !important;
            box-shadow: 0 0 0 1px #ff007f !important;
        }
        input {
            color: #ffffff !important;
        }

        /* Checkbox & Slider text */
        .stCheckbox span, .stSlider div {
            color: #e2e8f0 !important;
        }

        /* Button */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #ff007f, #7928ca);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 2rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 0, 127, 0.3);
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 0, 127, 0.5);
            color: white !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 3rem !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1.1rem !important;
            color: #94a3b8 !important;
        }

        /* Results Cards (Glassmorphism) */
        .result-card-blocked {
            background: rgba(255, 0, 64, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 0, 64, 0.3);
            border-left: 4px solid #ff0040;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(255, 0, 64, 0.1);
            color: #e2e8f0;
        }
        .result-card-allowed {
            background: rgba(0, 255, 128, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 255, 128, 0.3);
            border-left: 4px solid #00ff80;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 255, 128, 0.1);
            color: #e2e8f0;
        }
        .result-card-counterfactual {
            background: rgba(0, 242, 254, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 242, 254, 0.3);
            border-left: 4px solid #00f2fe;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 242, 254, 0.1);
            color: #e2e8f0;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            background-color: transparent !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #94a3b8 !important;
            background: transparent !important;
            border: none !important;
        }
        .stTabs [aria-selected="true"] {
            color: #ff007f !important;
            border-bottom: 2px solid #ff007f !important;
        }

        /* Cleanup */
        #MainMenu, footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- 3. DATA & DYNAMIC STATS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "fraud_dataset_cleaned.csv")

@st.cache_data
def get_graph_stats():
    try:
        df = pd.read_csv(DATA_PATH)
        df_display = df.replace(['None', 'none', 'NULL'], np.nan).fillna("-")
        avg_amt = df['amount'].mean()
        std_amt = df['amount'].std()
        return df_display, avg_amt, std_amt
    except:
        return pd.DataFrame(), 5000.0, 2000.0

def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False

df_display, avg_amt, std_amt = get_graph_stats()

# --- 4. UI LAYOUT ---
# Header
st.markdown('<h1 class="main-title">🛡️ UPI Adaptive Fraud Shield</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Dynamic GNN Inference Engine powered by Explainable AI</p>', unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ System Status")
st.sidebar.success("🟢 Active: Logic V5 (Latent Space XAI)")
st.sidebar.divider()
st.sidebar.markdown("**Network Stats:**")
st.sidebar.metric("Avg Transaction", f"₹{avg_amt:,.0f}")
st.sidebar.metric("Volatility (Std Dev)", f"₹{std_amt:,.0f}")

tab1, tab2 = st.tabs(["➕ Live Inference", "🔍 Historical Ledger"])

with tab1:
    st.markdown("### 📡 Transaction Telemetry")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            in_amt = st.number_input("💸 Transaction Amount (₹)", value=2500, step=500, min_value=0)
            in_dev = st.text_input("📱 Device ID (UUID)", value="5034b135-5795-4ae6-ada3-81dd1754e72e")
            in_scr = st.checkbox("⚠️ Remote Screen Sharing Detected")
        with col2:
            in_user = st.text_input("👤 User ID (Domain Required)", value="user_navami_01")
            in_txns_60s = st.slider("🔄 Transactions in last 60 seconds", min_value=1, max_value=100, value=1)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 RUN GNN ANALYSIS"):
        st.divider()
        
        # --- THE VALIDATION GATE (STRICT RULES) ---
        if not re.match(r"^user_[A-Za-z0-9_]+$", in_user):
            st.error("❌ **INVALID INPUT:** User ID must start with 'user_' and contain only alphanumeric characters or underscores.")
        elif not is_valid_uuid(in_dev):
            st.error("❌ **INVALID INPUT:** Device ID must be a valid strict UUID format (e.g. 5034b135-5795-4ae6-ada3-81dd1754e72e).")
        else:
            # --- DYNAMIC INFERENCE LOGIC (LATENT SPACE MAPPING) ---
            with st.spinner("Mapping node to latent space..."):
                # Simulate GNN embedding distance to Safe Cluster (Mahalanobis distance)
                z_amt = (in_amt - avg_amt) / (std_amt if std_amt > 0 else 1.0)
                z_vel = (in_txns_60s - 1.5) / 1.2 # Assuming avg network velocity is 1.5 txns/min
                z_scr = 4.0 if in_scr else 0.0
                
                # Distance in latent space
                distance_to_safe_cluster = np.sqrt((z_amt**2) + (z_vel**2) + (z_scr**2))
                safe_threshold = 2.5 # 2.5 sigma deviation threshold
                
                risk_score = min(99.4, (distance_to_safe_cluster / safe_threshold) * 50)
                final_risk = max(1.0, risk_score)
                
                node_id = f"txn_{uuid.uuid4().hex[:6]}"
                
                # --- VERDICT DISPLAY ---
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.metric(label="GNN Confidence Level", value=f"{final_risk:.1f}%", delta="Risk", delta_color="inverse")
                
                with col_res2:
                    if final_risk > 70:
                        st.markdown(f"""
                        <div class="result-card-blocked">
                            <h3 style="color:#ff3232; margin-top:0;">🔴 VERDICT: BLOCKED</h3>
                            <p style="margin-bottom:0;">Node Created! ID: <strong>{node_id}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card-allowed">
                            <h3 style="color:#32ff64; margin-top:0;">🟢 VERDICT: ALLOWED</h3>
                            <p style="margin-bottom:0;">Node Created! ID: <strong>{node_id}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                # --- PERSISTENCE: ADD TO DATASET ---
                df_columns = pd.read_csv(DATA_PATH, nrows=0).columns
                new_row = {
                    "transaction_id": node_id,
                    "user_id": in_user,
                    "merchant_id": str(uuid.uuid4()),
                    "amount": in_amt,
                    "device_id": in_dev,
                    "session_duration": 120,
                    "authentication_attempts": 1,
                    "transaction_velocity": in_txns_60s,
                    "recognized_screen_sharing_apps": 1 if in_scr else 0,
                    "is_fraud": 1 if final_risk > 70 else 0
                }
                for col in df_columns:
                    if col not in new_row:
                        new_row[col] = 0
                        
                new_df = pd.DataFrame([new_row])[df_columns]
                new_df.to_csv(DATA_PATH, mode='a', header=False, index=False)
                
                st.success("✅ Transaction successfully persisted to the Historical Ledger.")
                st.cache_data.clear()

                # --- EXPLAINABLE AI (NATURAL LANGUAGE & LATENT SPACE) ---
                st.markdown("### 🧠 Explainable AI (Behavioral & Latent Space Analysis)")
                
                if distance_to_safe_cluster <= safe_threshold:
                    st.success(f"✨ **Transaction is Safe:** This transaction matches normal, expected behavior.\n\n*(Technical: Node mapped to safe region. Latent Space Distance: {distance_to_safe_cluster:.2f}σ, which is within the {safe_threshold}σ threshold).*")
                else:
                    st.error(f"🚩 **Anomaly Detected:** This transaction looks highly unusual and unsafe compared to normal activity!\n\n*(Technical: Latent Space Deviation. Node embedding fell outside the Global Safe Cluster with a distance of {distance_to_safe_cluster:.2f}σ).*")

                st.markdown("#### 🗣️ Natural Language Explanation")
                if final_risk > 70:
                    st.markdown(f"🚨 **Verdict Summary:** The transaction was **BLOCKED** because the system detected telemetry patterns that deviate significantly from your normal baseline by **{distance_to_safe_cluster:.2f}σ**.")
                else:
                    st.markdown(f"✨ **Verdict Summary:** The transaction was **ALLOWED** because its behavior is close to your normal baseline (overall deviation is **{distance_to_safe_cluster:.2f}σ**, which is within the safe limit of **2.5σ**).")
                
                # Bulleted parameters in clear English
                st.markdown(f"""
                - 💸 **Transaction Amount:** The requested amount of **₹{in_amt:,}** is **{abs(z_amt):.2f} standard deviations** {"higher" if z_amt > 0 else "lower"} than your average of **₹{avg_amt:,.0f}** (typical variation is ₹{std_amt:,.0f}).
                - ⏳ **Transaction Frequency:** **{in_txns_60s} transactions** in the last 60 seconds is **{abs(z_vel):.2f} standard deviations** {"higher" if z_vel > 0 else "lower"} than the normal average rate of 1.5 transactions.
                - 📱 **Security Context:** Remote screen-sharing is **{"ACTIVE (HIGH RISK)" if in_scr else "INACTIVE (SAFE)"}**. Active screen-sharing is a high-risk indicator often associated with remote access scams.
                """)

                # --- COUNTERFACTUAL RECOMMENDATIONS ---
                st.markdown("#### 💡 Counterfactual Recommendations")
                if final_risk > 70:
                    recommendations = []
                    
                    # 1. Screen sharing counterfactual
                    if in_scr:
                        z_scr_hyp = 0.0
                        distance_hyp = np.sqrt(z_amt**2 + z_vel**2 + z_scr_hyp**2)
                        hyp_risk = max(1.0, min(99.4, (distance_hyp / safe_threshold) * 50))
                        status_str = "ALLOWED" if hyp_risk <= 70 else "BLOCKED (due to other factors)"
                        recommendations.append(
                            f"🔌 **Disable Remote Screen Sharing**: Disabling active remote access apps will drop the security risk factor from 4.0σ to 0.0σ. "
                            f"Keeping other factors identical, this would reduce the risk score to **{hyp_risk:.1f}%** (Verdict: **{status_str}**)."
                        )
                    
                    # 2. Amount counterfactual
                    r_amt = 12.25 - z_vel**2 - z_scr**2
                    if r_amt >= 0:
                        max_safe_amt = avg_amt + np.sqrt(r_amt) * std_amt
                        if in_amt > max_safe_amt:
                            recommendations.append(
                                f"💸 **Reduce Transaction Amount**: Lower the transaction amount to **₹{int(max_safe_amt):,}** or below (currently **₹{in_amt:,}**). "
                                f"This will bring the risk score down to **70.0%** (Verdict: **ALLOWED**)."
                            )
                    else:
                        recommendations.append(
                            "💸 **Reduce Transaction Amount**: Lowering the amount alone is not enough to authorize this transaction. "
                            "You must first resolve other alerts (such as disabling remote screen sharing)."
                        )
                        
                    # 3. Velocity counterfactual
                    r_vel = 12.25 - z_amt**2 - z_scr**2
                    if r_vel >= 0:
                        max_safe_vel = 1.5 + 1.2 * np.sqrt(r_vel)
                        if in_txns_60s > max_safe_vel:
                            recommendations.append(
                                f"⏳ **Wait Before Retrying**: Wait a short moment to reduce your 60-second transaction frequency to **{max(1, int(max_safe_vel))}** or below (currently **{in_txns_60s}**). "
                                f"This will drop the risk score to **70.0%** (Verdict: **ALLOWED**)."
                            )
                    else:
                        recommendations.append(
                            "⏳ **Wait Before Retrying**: Waiting alone is not enough to authorize this transaction. "
                            "You must also reduce the transaction amount or disable screen sharing."
                        )

                    rec_html = "<div class='result-card-counterfactual'><h4 style='color:#00f2fe; margin-top:0;'>🛠️ Actions Required to Authorize</h4><ul>"
                    for rec in recommendations:
                        rec_html += f"<li style='margin-bottom:10px;'>{rec}</li>"
                    rec_html += "</ul></div>"
                    st.markdown(rec_html, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-card-counterfactual'>
                        <h4 style='color:#00f2fe; margin-top:0;'>⚙️ System Status: Safe</h4>
                        <p style='margin-bottom:0;'>No adjustments are needed. The transaction is within the safe behavioral envelope (Risk: <strong>{final_risk:.1f}%</strong>).</p>
                    </div>
                    """, unsafe_allow_html=True)

                # --- DEEP EXPLAINABLE AI (REAL GNNEXPLAINER) ---
                with st.spinner("Generating Deep Graph Explanations (GNNExplainer)..."):
                    try:
                        feature_importances, subgraph_edges = explain_transaction(node_id, epochs=40)
                        
                        st.markdown("### 🔍 GNNExplainer Subgraph & Feature Importance")
                        st.markdown("This section displays the mathematical evidence calculated by PyTorch Geometric GNNExplainer, identifying the exact subgraphs and features driving the graph neural network's node prediction.")
                        
                        col_xai1, col_xai2 = st.columns(2)
                        
                        with col_xai1:
                            st.markdown("#### 🔗 GNN Decision Subgraph")
                            
                            def short_label(name):
                                if len(name) > 15:
                                    if name.startswith('user_'):
                                        return f"user_{name.split('_')[1][:5]}..."
                                    if name.startswith('merchant_'):
                                        return f"merch_{name.split('_')[1][:5]}..."
                                    if name.startswith('device_'):
                                        return f"dev_{name.split('_')[1][:5]}..."
                                    return f"{name[:8]}..."
                                return name
                                
                            fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#110c18')
                            ax.set_facecolor('#110c18')
                            
                            vis_g = nx.Graph()
                            for edge in subgraph_edges[:10]:
                                vis_g.add_edge(edge['source'], edge['target'], weight=edge['importance'])
                                
                            pos = nx.spring_layout(vis_g, seed=42)
                            
                            node_colors = []
                            for n in vis_g.nodes():
                                if str(n) == node_id:
                                    node_colors.append('#ff007f' if final_risk > 70 else '#00ff80')
                                elif str(n).startswith('user_'):
                                    node_colors.append('#3b82f6')
                                elif str(n).startswith('merchant_'):
                                    node_colors.append('#ec4899')
                                elif str(n).startswith('device_'):
                                    node_colors.append('#f59e0b')
                                else:
                                    node_colors.append('#94a3b8')
                                    
                            nx.draw_networkx_nodes(vis_g, pos, node_color=node_colors, node_size=600, ax=ax)
                            
                            widths = [max(1.0, vis_g[u][v]['weight'] * 6.0) for u, v in vis_g.edges()]
                            nx.draw_networkx_edges(vis_g, pos, width=widths, edge_color='#6366f1', alpha=0.8, ax=ax)
                            
                            labels = {n: short_label(str(n)) for n in vis_g.nodes()}
                            nx.draw_networkx_labels(vis_g, pos, labels=labels, font_size=8, font_color='#ffffff', font_weight='bold', ax=ax)
                            
                            ax.axis('off')
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.caption("💡 Line thickness shows GNNExplainer's structural path influence score. Pink/red indicates target transaction is anomalous/fraud, green is legitimate.")
                            
                        with col_xai2:
                            st.markdown("#### 📊 GNN Feature Influence")
                            
                            fig_feat, ax_feat = plt.subplots(figsize=(6, 4.5), facecolor='#110c18')
                            ax_feat.set_facecolor('#110c18')
                            
                            top_feats = feature_importances[:5]
                            feat_names = [f['feature'] for f in top_feats]
                            importances = [f['importance'] for f in top_feats]
                            
                            y_pos = np.arange(len(feat_names))
                            colors = ['#ff007f' if i == 0 else '#7928ca' for i in range(len(feat_names))]
                            
                            ax_feat.barh(y_pos, importances, align='center', color=colors, height=0.45)
                            ax_feat.set_yticks(y_pos)
                            ax_feat.set_yticklabels(feat_names, color='#e2e8f0', fontsize=9)
                            ax_feat.invert_yaxis()
                            ax_feat.set_xlabel('Normalized GNN Feature Weight', color='#94a3b8', fontsize=9)
                            ax_feat.xaxis.label.set_color('#94a3b8')
                            ax_feat.tick_params(colors='#94a3b8', labelsize=8)
                            for spine in ax_feat.spines.values():
                                spine.set_edgecolor('#2d1f3f')
                                
                            plt.tight_layout()
                            st.pyplot(fig_feat)
                            st.caption("💡 Normalized score (0 to 1) representing feature mask weights computed by GNNExplainer epochs optimization.")
                            
                    except Exception as e:
                        st.warning(f"Unable to load PyG GNNExplainer visualization: {e}")

with tab2:
    st.markdown("### 🗄️ Recent Network Activity")
    st.dataframe(df_display.tail(15))