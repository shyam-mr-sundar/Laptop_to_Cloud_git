# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from hybrid_pipeline import HybridFraudPipeline  # your class file

st.set_page_config(page_title="JP Morgan Fraud Detection", layout="wide")

@st.cache_resource
def load_models_and_meta():
    scaler = joblib.load("scaler.pkl")
    dt = joblib.load("dt.pkl")
    cb = joblib.load("cb.pkl")
    xgb = joblib.load("xgb.pkl")
    kmeans = joblib.load("kmeans.pkl")
    iso = joblib.load("iso.pkl")
    hybrid_cb = joblib.load("hybrid_cb.pkl")
    train_columns = joblib.load("train_columns.pkl")  # list of columns AFTER feature eng + dummies
    pipeline = HybridFraudPipeline(scaler, dt, cb, xgb, kmeans, iso, hybrid_cb)
    return pipeline, train_columns

pipeline, TRAIN_COLUMNS = load_models_and_meta()

RAW_FEATURES = [
    "step", "type", "amount", "nameOrig",
    "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest"
]

TYPE_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

def feature_engineering(df):
    df = df.copy()

    # 1. Log transforms (same as training)
    for col in ["step", "amount", "oldbalanceOrg",
                "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        df[f"{col}_log"] = np.log1p(df[col])

    # 2. Balance differences
    df["orig_balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_balance_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # 3. Zero balance flags
    df["orig_zero_balance_flag"] = (
        (df["oldbalanceOrg"] == 0) & (df["newbalanceOrig"] == 0)
    ).astype(int)
    df["dest_zero_balance_flag"] = (
        (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)
    ).astype(int)

    # 4. Transaction type interactions
    df["transfer_to_zero_dest"] = (
        (df["type"] == "TRANSFER") & (df["newbalanceDest"] == 0)
    ).astype(int)
    df["cashout_from_zero_orig"] = (
        (df["type"] == "CASH_OUT") & (df["oldbalanceOrg"] == 0)
    ).astype(int)

    # 5. Ratios
    df["amount_to_orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amount_to_dest_balance_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1)

    # 6. Suspicious flag
    df["suspicious_flag"] = (
        (df["amount"] > 100000)
        & (df["oldbalanceOrg"] == 0)
        & (df["newbalanceOrig"] == 0)
        & (df["type"] == "TRANSFER")
    ).astype(int)

    # 7. One‑hot encode type (same set as training)
    for t in TYPE_CATEGORIES:
        col_name = f"type_{t}"
        df[col_name] = (df["type"] == t).astype(int)

    # Drop raw string columns if they were not used in training
    # (you said non‑useful object columns were removed before modeling)
    df = df.drop(columns=["type", "nameOrig", "nameDest"])

    return df

def prepare_for_model(df_raw):
    df_clean = df_raw.copy()

    # Drop isFlaggedFraud at the very start (models were trained without it)
    if "isFlaggedFraud" in df_clean.columns:
        df_clean = df_clean.drop(columns=["isFlaggedFraud"])

    # Apply feature engineering on cleaned data
    df_fe = feature_engineering(df_clean)

    # Align to training columns (saved from X without isFlaggedFraud)
    df_aligned = df_fe.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    return df_aligned

st.title("JP Morgan and Chase Fraud Detection Project")
st.markdown("### Enter Transaction Details / Use Sample Data")

mode = st.radio("Choose input mode:", ["Single Transaction (Manual)", "Batch from 1000-row Sample CSV"])

if mode == "Single Transaction (Manual)":
    # Single row editor matching your 10 raw columns
    default_row = pd.DataFrame([{
        "step": 1,
        "type": "PAYMENT",
        "amount": 1000.0,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "nameDest": "M123456789",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }])

    edited = st.data_editor(
        default_row,
        num_rows="fixed",
        column_config={
            "type": st.column_config.SelectboxColumn(options=TYPE_CATEGORIES)
        },
        use_container_width=True,
    )
    df_input_raw = edited.copy()

    st.markdown("### Check transaction is Fraud or Not Fraud")

    if st.button("Predict"):
        try:
            X_input = prepare_for_model(df_input_raw)
            pred = pipeline.predict(X_input)[0]

            # If available, show probability
            try:
                proba = float(pipeline.hybrid_cb.predict_proba(X_input)[0, 1])
            except Exception:
                proba = None

            if pred == 1:
                st.error("Transaction is Fraud.")
            else:
                st.success("Transaction is NOT Fraud.")

            st.write("Model prediction (0 = non-fraud, 1 = fraud):", int(pred))
            if proba is not None:
                st.write(f"Estimated fraud probability: {proba:.3f}")
        except Exception as e:
            st.warning(f"Error during prediction: {e}")

else:
    st.markdown("Upload 1000-row CSV (same schema as original 6.3M data) or use default sample_1000.csv.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df_batch_raw = pd.read_csv(uploaded)
    else:
        try:
            df_batch_raw = pd.read_csv("sample_1000.csv")
            st.info("Using default sample_1000.csv from project folder.")
        except FileNotFoundError:
            df_batch_raw = None
            st.error("sample_1000.csv not found. Please upload a CSV.")

    if df_batch_raw is not None:
        st.write("Preview of raw input data:")
        st.dataframe(df_batch_raw.head())

        if st.button("Run Fraud Prediction on Batch"):
            try:
                X_batch = prepare_for_model(df_batch_raw)
                preds = pipeline.predict(X_batch)
                df_result = df_batch_raw.copy()
                df_result["fraud_prediction"] = preds

                st.success("Prediction completed.")
                st.dataframe(df_result.head(20))

                fraud_count = int((df_result["fraud_prediction"] == 1).sum())
                st.write(f"Predicted fraud transactions: {fraud_count} / {len(df_result)}")

                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_out,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.warning(f"Error during prediction: {e}")
