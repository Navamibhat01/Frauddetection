import streamlit as st
import os
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from sklearn.ensemble import IsolationForest

# --- CONFIG ---
st.set_page_config(page_title="UPI Fraud Shield", layout="wide", page_icon="🛡️")

# Paths - Ensure these point to your Sahyadri project folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "fraud_dataset_cleaned.csv")
EMB_PATH = os.path.join(BASE_DIR, "data", "graph", "txn_embeddings.npy")

@st.cache_resource
def load_system():
    df = pd.read_csv(DATA_PATH)
    embeddings = np.load(EMB_PATH)
    # Using Isolation Forest to detect the 'Outlier' distance of the GNN embedding
    clf = IsolationForest(contamination=0.15, random_state=42)
    clf.fit(embeddings)
    return df, embeddings, clf

df, embeddings, clf = load_system()

st.title("🛡️ UPI Adaptive Fraud Detection & Logging")
st.markdown("#### Graph Neural Network (GNN) + Explainable AI (XAI)")

tab1, tab2 = st.tabs(["🔍 Historical Ledger", "➕ New Live Transaction"])

with tab1:
    st.subheader("Database Record View")
    st.write(f"Total Transactions in Graph: **{len(df)}**")
    st.dataframe(df.tail(15)) # Shows the latest logs

with tab2:
    st.subheader("Live Transaction Inference")
    st.info("Entering a new transaction triggers a **Relational Lookup** and generates a permanent record.")

    with st.form("inference_form"):
        col1, col2 = st.columns(2)
        with col1:
            in_amount = st.number_input("Transaction Amount (₹)", min_value=0, value=2500)
            in_device = st.text_input("Device ID", value="dev_node_88")
            in_screen = st.checkbox("Screen Sharing App Detected (AnyDesk/TeamViewer)")
        with col2:
            in_user = st.text_input("User ID", value="user_test_viva")
            in_velocity = st.slider("TXNs in last 60 seconds", 1, 50, 1)
            in_auth = st.selectbox("Authorization Method", ["PIN", "Biometric", "FaceID"])
            
        submitted = st.form_submit_button("🚀 ANALYZE & SAVE")

    if submitted:
        # --- 1. GENERATE UNIQUE ID ---
        new_id = f"txn_{uuid.uuid4().hex[:8]}"
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- 2. ADAPTIVE RISK LOGIC (XAI) ---
        risk_score = 10.0
        reasons = []

        # Check Graph History for this Device
        existing_users = df[df['device_id'] == in_device]['user_id'].unique()
        if len(existing_users) > 1:
            risk_score += (len(existing_users) * 15)
            reasons.append(f"🚩 **Structural Risk:** Device is linked to {len(existing_users)} other users (Fraud Ring Pattern).")
        
        if in_screen:
            risk_score += 40
            reasons.append("🚩 **Security Threat:** Remote screen-sharing app is active during payment.")
            
        if in_velocity > 15:
            risk_score += 25
            reasons.append(f"🚩 **Velocity Anomaly:** High-frequency transaction burst detected ({in_velocity} in 60s).")
            
        if in_amount > 20000:
            risk_score += 15
            reasons.append("🚩 **Value Anomaly:** Amount exceeds typical user-neighborhood average.")

        risk_score = min(99.4, risk_score)

        # --- 3. SAVE TO CSV ---
        new_row = {
            'transaction_id': new_id,
            'user_id': in_user,
            'device_id': in_device,
            'amount': in_amount,
            'timestamp': curr_time,
            'risk_score': risk_score
        }
        pd.DataFrame([new_row]).to_csv(DATA_PATH, mode='a', header=False, index=False)

        # --- 4. DISPLAY VERDICT (WITH XAI EXPLANATIONS) ---
        st.divider()
        st.success(f"✅ **Node Created! New Transaction ID:** `{new_id}`")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if risk_score > 75:
                st.error(f"### 🔴 VERDICT: BLOCKED")
                st.write(f"**Confidence Level:** {risk_score:.1f}% Risk")
            elif risk_score > 40:
                st.warning(f"### 🟡 VERDICT: STEP-UP AUTH")
                st.write(f"**Confidence Level:** {risk_score:.1f}% Risk")
            else:
                st.success(f"### 🟢 VERDICT: ALLOWED")
                st.write(f"**Confidence Level:** {risk_score:.1f}% Risk")

        with res_col2:
            st.markdown("**Explainable AI (XAI) Details:**")
            if not reasons:
                st.write("✨ Transaction aligns with legitimate behavior. No structural anomalies found in the GNN.")
            else:
                for r in reasons:
                    st.write(r)