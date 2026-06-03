"""Streamlit demo frontend (root) for the BentoML churn prediction service.

Usage:
    pip install streamlit requests pandas
    streamlit run app.py

Connects to BentoML at http://localhost:3000 by default.
"""

import json
import requests
import pandas as pd
import streamlit as st

BENTO_URL = "http://localhost:3000"

COLUMNS = [
    "CustomerID", "Age", "Gender", "Tenure", "Usage Frequency",
    "Support Calls", "Payment Delay", "Subscription Type",
    "Contract Length", "Total Spend", "Last Interaction"
]

SAMPLES = [
    [2.0, 30.0, "Female", 39.0, 14.0,  5.0, 18.0, "Standard",   "Annual",    932.0, 17.0],
    [3.0, 65.0, "Female", 49.0,  1.0, 10.0,  8.0, "Basic",      "Monthly",   557.0,  6.0],
    [4.0, 55.0, "Female", 14.0,  4.0,  6.0, 18.0, "Basic",      "Quarterly", 185.0,  3.0],
]

st.set_page_config(page_title="Churn Prediction Demo (root)", layout="wide")
st.title("Churn Prediction — Streamlit Demo (root)")

with st.sidebar:
    bento_url = st.text_input("Bento service URL", value=BENTO_URL)
    st.write("")
    if st.button("Health check"):
        try:
            r = requests.post(f"{bento_url}/healthz", timeout=5)
            r.raise_for_status()
            st.success("Service healthy")
            st.json(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")
    if st.button("Show /metrics (raw)"):
        try:
            r = requests.get(f"{bento_url}/metrics", timeout=5)
            r.raise_for_status()
            st.code(r.text[:5000])
        except Exception as e:
            st.error(f"Metrics fetch failed: {e}")

st.header("Single-row prediction (JSON)")
example = {
    "input": {
        "dataframe_split": {
            "columns": COLUMNS,
            "data": [SAMPLES[0]]
        }
    }
}
body = st.text_area("Request JSON", value=json.dumps(example, ensure_ascii=False, indent=2), height=260)
if st.button("Send /predict (single)"):
    try:
        payload = json.loads(body)
        r = requests.post(f"{bento_url}/predict", json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()
        st.success("Prediction returned")
        st.json(result)
        df = pd.DataFrame({
            "prediction": result["predictions"],
            "label": result["labels"],
            "confidence": result["confidences"]
        })
        st.subheader("Prediction table")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Request failed: {e}")

st.markdown("---")

st.header("Batch predict (samples)")
if st.button("Send sample batch (3 rows)"):
    payload = {
        "input": {
            "dataframe_split": {"columns": COLUMNS, "data": SAMPLES}
        }
    }
    try:
        r = requests.post(f"{bento_url}/predict", json=payload, timeout=20)
        r.raise_for_status()
        result = r.json()
        st.success("Batch prediction returned")
        st.json(result)
        df = pd.DataFrame({"prediction": result["predictions"], "label": result["labels"], "confidence": result["confidences"]})
        st.dataframe(df)
        st.subheader("Label distribution")
        st.bar_chart(df["label"].value_counts())
    except Exception as e:
        st.error(f"Batch request failed: {e}")

st.markdown("---")
st.info("Run: `streamlit run app.py` — make sure the Bento service is running at http://localhost:3000")
