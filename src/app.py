import streamlit as st
import os
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from sklearn.ensemble import IsolationForest

# --- CONFIG & PATHS ---
st.set_page_config(page_title="UPI Fraud Shield", layout="wide")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "fraud_dataset_cleaned.csv")
EMB_PATH = os.path.join(BASE_DIR, "data", "graph", "txn_embeddings.npy")

@st.cache_resource
def load_system():
    if not os.path.exists(DATA_PATH):
        # Create a dummy dataframe if file doesn't exist for demo
        df = pd.DataFrame(columns=['transaction_id', 'user_id', 'device_id', 'amount', 'timestamp', 'risk_score'])
    else:
        df = pd.read_csv(DATA_PATH)
    
    embeddings = np.load(EMB_PATH)
    clf = IsolationForest(contamination=0.15, random_state=42)
    clf.fit(embeddings)
    return df, embeddings, clf

df, embeddings, clf = load_system()

st.title("🛡️ UPI Adaptive Fraud Detection & Logging")

tab1, tab2 = st.tabs(["🔍 Database View", "➕ New Live Transaction"])

with tab1:
    st.subheader("Current Transaction Ledger")
    st.write(f"Total Transactions Logged: {len(df)}")
    st.dataframe(df.tail(10)) # Shows the last 10 entries (including new ones!)

with tab2:
    st.subheader("Execute & Save New Transaction")
    
    with st.form("save_txn_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            in_amount = st.number_input("Transaction Amount (₹)", min_value=0, value=2500)
            in_device = st.text_input("Device ID", value="dev_sahyadri_node")
            in_screen = st.checkbox("Screen Sharing Active?")
        with col_b:
            in_user = st.text_input("User ID", value="user_student_test")
            in_velocity = st.slider("TXNs in last 60s", 1, 50, 1)
            in_auth = st.selectbox("Auth Method", ["PIN", "Biometric"])
            
        submit = st.form_submit_button("🚀 ANALYZE & SAVE TO GRAPH")

    if submit:
        # 1. GENERATE NEW METADATA
        new_id = f"txn_{uuid.uuid4().hex[:8]}"
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 2. CALCULATE RISK
        existing_users = df[df['device_id'] == in_device]['user_id'].unique()
        risk_score = 10.0
        if len(existing_users) > 0: risk_score += 20
        if in_screen: risk_score += 40
        if in_velocity > 10: risk_score += 20
        risk_score = min(99.9, risk_score)

        # 3. SAVE TO CSV (The "Saving" Logic)
        new_row = {
            'transaction_id': new_id,
            'user_id': in_user,
            'device_id': in_device,
            'amount': in_amount,
            'timestamp': curr_time,
            'risk_score': risk_score
        }
        
        # Append to CSV immediately
        new_df = pd.DataFrame([new_row])
        new_df.to_csv(DATA_PATH, mode='a', header=not os.path.exists(DATA_PATH), index=False)
        
        # 4. DISPLAY RESULTS
        st.divider()
        st.success(f"📦 **Transaction Logged!** ID: `{new_id}`")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if risk_score > 70:
                st.error(f"### 🔴 BLOCKED ({risk_score}%)")
            else:
                st.success(f"### 🟢 ALLOWED ({risk_score}%)")
        
        with res_col2:
            st.info("**XAI Persistence:**")
            st.write(f"This record has been saved to the database. In the next training cycle, this node will influence its neighbors' risk scores.")