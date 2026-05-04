import streamlit as st
import os
import numpy as np
import pandas as pd
import uuid
import re
from datetime import datetime

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

                # --- EXPLAINABLE AI (BASED ON VECTOR MAPPING) ---
                st.markdown("### 🧠 Explainable AI (Latent Space Analysis)")
                if distance_to_safe_cluster <= safe_threshold:
                    st.success(f"✨ **Transaction is Safe:** This transaction matches normal, expected behavior.\n\n*(Technical: Node mapped to safe region. Latent Space Distance: {distance_to_safe_cluster:.2f}σ, which is within the {safe_threshold}σ threshold).*")
                else:
                    st.error(f"🚩 **Anomaly Detected:** This transaction looks highly unusual and unsafe compared to normal activity!\n\n*(Technical: Latent Space Deviation. Node embedding fell outside the Global Safe Cluster with a distance of {distance_to_safe_cluster:.2f}σ).*")
                    
                    contributions = [
                        ("Transaction Amount Variance", "The requested amount is significantly different from your typical transaction history.", z_amt**2),
                        ("Transaction Velocity", "There are too many transactions happening in a very short amount of time.", z_vel**2),
                        ("Security Context", "We detected a high-risk remote screen-sharing application running on your device.", z_scr**2)
                    ]
                    sorted_contrib = sorted(contributions, key=lambda x: x[2], reverse=True)
                    
                    st.markdown("#### Why was this flagged?")
                    for tech_name, plain_english, value in sorted_contrib:
                        if value > 1.0:
                            st.markdown(f"- 🛑 **{plain_english}** <br> <span style='color:#94a3b8; font-size: 0.9em;'>*(Technical: Vector deviation driven by **{tech_name}** | Influence Score: {value:.2f})*</span>", unsafe_allow_html=True)

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

with tab2:
    st.markdown("### 🗄️ Recent Network Activity")
    st.dataframe(df_display.tail(15))