"""
Dataset Cleaning Script for GNN Fraud Detection
================================================
Run this ONCE before graph construction or model training.
Output: fraud_dataset_cleaned.csv
"""

import pandas as pd
import numpy as np
import os

INPUT_PATH  = r"C:\Users\Administrator\OneDrive\Desktop\fraud-detection-gnn\data\raw\fraud_dataset.csv"
OUTPUT_PATH = r"C:\Users\Administrator\OneDrive\Desktop\fraud-detection-gnn\data\processed\fraud_dataset_cleaned.csv"

# verify file exists before loading
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Cannot find: {INPUT_PATH}\nCheck the file is in data/raw/")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"[LOAD] Shape: {df.shape}")
print(f"[LOAD] Fraud rate: {df['is_fraud'].mean():.2%}  ({df['is_fraud'].sum()} fraud / {len(df)} total)\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DROP LABEL-LEAKAGE COLUMNS
#    These columns perfectly encode is_fraud — model would just read the answer
#    instead of learning graph patterns. Must be removed.
# ─────────────────────────────────────────────────────────────────────────────
LEAKAGE_COLS = [
    # Perfect label encoders (identified from crosstab analysis)
    "handle_verification_status",       # 'unverified' == fraud 100% of the time
    "business_name_match",              # 'none' vs any value tracks fraud perfectly

    # Synthetic flags that are always 1 for fraud and 0 for legit
    # Keeping these means the GCN just reads a pre-made answer, not graph patterns
    "unusual_transaction_amount_flag",  # always 1 for fraud, never for legit
    "time_pressure_indicators",         # always > 0 for fraud, always 0 for legit
    "otp_request_device_consistency",   # always 0 for fraud, always 1 for legit

    # These whole categorical columns encode fraud perfectly after one-hot encoding
    # Drop them RAW before encoding to avoid ghost columns
    "merchant_category_code",           # 'unknown' category == fraud 100%
    "handle_registration_pattern",      # 'recent' == fraud 100%
    "pin_entry_method",                 # 'pasted' == fraud 100%
]
df.drop(columns=LEAKAGE_COLS, inplace=True)
print(f"[LEAKAGE] Dropped {len(LEAKAGE_COLS)} leakage columns: {LEAKAGE_COLS}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DROP ZERO-VARIANCE COLUMNS
#    Single unique value — carry no information whatsoever
# ─────────────────────────────────────────────────────────────────────────────
ZERO_VAR_COLS = [
    "upi_handle_age",                   # always 0
    "handle_contains_official_terms",   # always 0
    "relationship_to_requester",        # always 'unknown'
    "social_media_presence",            # always 'none'
]
df.drop(columns=ZERO_VAR_COLS, inplace=True)
print(f"[ZERO VAR] Dropped {len(ZERO_VAR_COLS)} zero-variance columns: {ZERO_VAR_COLS}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DROP HIGH-MISSING COLUMNS (>95% missing)
# ─────────────────────────────────────────────────────────────────────────────
HIGH_MISSING_COLS = [
    "url_referrer",          # 97.1% missing
    "request_description",   # 97.2% missing
]
df.drop(columns=HIGH_MISSING_COLS, inplace=True)
print(f"[MISSING] Dropped {len(HIGH_MISSING_COLS)} high-missing columns: {HIGH_MISSING_COLS}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DROP FREE-TEXT / NON-ENCODABLE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
FREE_TEXT_COLS = [
    "description",    # free-text payment description
    "ip_address",     # too high cardinality
    "location",       # raw coordinate string
]
df.drop(columns=FREE_TEXT_COLS, inplace=True)
print(f"[FREE TEXT] Dropped {len(FREE_TEXT_COLS)} free-text columns: {FREE_TEXT_COLS}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENCODE LIST-LIKE STRING COLUMNS → BINARY FLAGS
#    Columns with Python list strings like "['screen_mirroring_app']" or "[]"
# ─────────────────────────────────────────────────────────────────────────────
LIST_COLS = [
    "recent_app_installs",
    "permissions_granted",
    "recognized_screen_sharing_apps",
    "request_description_keywords",
]
for col in LIST_COLS:
    df[col] = df[col].apply(
        lambda x: 0 if str(x).strip() in ("[]", "nan", "") else 1
    ).astype(int)

print(f"[LIST COLS] Encoded {len(LIST_COLS)} list-string columns to binary flags")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PARSE TIMESTAMP → NUMERIC
#    Format is MM:SS.s — convert to total seconds
# ─────────────────────────────────────────────────────────────────────────────
def parse_timestamp(ts):
    try:
        parts = str(ts).split(":")
        return float(parts[0]) * 60 + float(parts[1])
    except Exception:
        return np.nan

df["timestamp_seconds"] = df["timestamp"].apply(parse_timestamp)
df.drop(columns=["timestamp"], inplace=True)
print(f"[TIMESTAMP] Parsed 'timestamp' → 'timestamp_seconds'")


# ─────────────────────────────────────────────────────────────────────────────
# 8. ONE-HOT ENCODE REMAINING CATEGORICAL COLUMNS
#    Only encode columns that are NOT leakage
# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "session_source",        # 2 categories — not correlated with fraud
    "authorization_method",  # 2 categories — not correlated with fraud
    "transaction_type",      # 2 categories — not correlated with fraud
    "handle_typo_analysis",  # 2 categories — not correlated with fraud
]
df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
print(f"[ENCODE] One-hot encoded {len(CATEGORICAL_COLS)} categorical columns")
print(f"         Shape after encoding: {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. VERIFY NO REMAINING OBJECT COLUMNS (except IDs)
# ─────────────────────────────────────────────────────────────────────────────
ID_COLS = {"transaction_id", "user_id", "merchant_id", "device_id"}
obj_cols = [c for c in df.columns
            if df[c].dtype == object and c not in ID_COLS]
if obj_cols:
    print(f"[WARNING] Dropping unexpected object columns: {obj_cols}")
    df.drop(columns=obj_cols, inplace=True)
else:
    print(f"[VERIFY] No unexpected object columns remaining")


# ─────────────────────────────────────────────────────────────────────────────
# 10. FILL ANY REMAINING NULLS
# ─────────────────────────────────────────────────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns
null_counts = df[num_cols].isnull().sum()
if null_counts.any():
    print(f"[FIX] Filling nulls with column median")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
else:
    print(f"[VERIFY] No null values in feature columns")


# ─────────────────────────────────────────────────────────────────────────────
# 11. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c not in ID_COLS and c != "is_fraud"]

print(f"\n[FINAL] Shape                : {df.shape}")
print(f"[FINAL] Fraud rate           : {df['is_fraud'].mean():.2%}")
print(f"[FINAL] Feature columns      : {len(feature_cols)}")
print(f"\n[FINAL] Feature column list:")
for c in feature_cols:
    print(f"  {c}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n[SAVED] → {OUTPUT_PATH}")