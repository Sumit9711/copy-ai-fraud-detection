import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Public Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# GLOBAL STYLES
# --------------------------------------------------
st.markdown("""
<style>
.big-title {
    font-size: 38px;
    font-weight: 800;
    color: #1f4ed8;
}
.subtle {
    color: #6b7280;
}
.metric-box {
    padding: 15px;
    border-radius: 12px;
    background: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🛡️ AI-Based Public Fraud Detection System</div>", unsafe_allow_html=True)
st.markdown("<p class='subtle'>Detect anomalies, assess risk, and support government audits using AI</p>", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------------------------
# SYNTHETIC DATA GENERATOR (LARGE DATASET)
# --------------------------------------------------
@st.cache_data
def generate_sample_data(n=5000):
    departments = ["Health", "Education", "Transport", "Defense", "Rural", "Urban"]
    vendors = [f"V{100+i}" for i in range(40)]
    base_time = datetime.now() - timedelta(days=90)

    data = []
    for i in range(n):
        dept = random.choice(departments)
        vendor = random.choice(vendors)

        amount = np.random.lognormal(mean=10, sigma=1.2)
        if random.random() < 0.08:  # fraud spikes
            amount *= random.randint(5, 20)

        txn_time = base_time + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        data.append([
            100000 + i,
            dept,
            vendor,
            round(amount, 2),
            txn_time
        ])

    return pd.DataFrame(
        data,
        columns=["transaction_id", "department_id", "vendor_id", "amount", "transaction_time"]
    )

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("📂 Data Input")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
use_sample = st.sidebar.button("🧪 Use Sample Dataset")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded dataset loaded")

elif use_sample:
    df = generate_sample_data()
    st.sidebar.info("Using large sample dataset (5,000 rows)")

else:
    st.info("Upload a CSV or click **Use Sample Dataset** to explore the dashboard.")
    st.stop()

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
df["transaction_time"] = pd.to_datetime(df["transaction_time"])
df["hour"] = df["transaction_time"].dt.hour
df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
df["is_weekend"] = (df["transaction_time"].dt.weekday >= 5).astype(int)

df["log_amount"] = np.log1p(df["amount"])

dept_mean = df.groupby("department_id")["amount"].transform("mean")
dept_std = df.groupby("department_id")["amount"].transform("std")

df["amount_zscore_dept"] = (df["amount"] - dept_mean) / dept_std
df["amount_vs_dept_mean"] = df["amount"] / dept_mean

df["vendor_txn_count"] = df.groupby("vendor_id")["vendor_id"].transform("count")
df["vendor_avg_amount"] = df.groupby("vendor_id")["amount"].transform("mean")
df["vendor_amount_ratio"] = df["amount"] / df["vendor_avg_amount"]

features = [
    "log_amount", "amount_zscore_dept", "amount_vs_dept_mean",
    "hour", "is_night", "is_weekend",
    "vendor_txn_count", "vendor_avg_amount", "vendor_amount_ratio"
]

X = df[features].replace([np.inf, -np.inf], 0).fillna(0)
X_scaled = scaler.transform(X)

# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------
df["anomaly_score"] = model.decision_function(X_scaled)
df["fraud_flag"] = (model.predict(X_scaled) == -1).astype(int)

df["risk_score"] = (
    (df["anomaly_score"].max() - df["anomaly_score"]) /
    (df["anomaly_score"].max() - df["anomaly_score"].min())
) * 100

df["risk_level"] = pd.cut(
    df["risk_score"], bins=[0, 30, 70, 100],
    labels=["Low", "Medium", "High"]
)

# --------------------------------------------------
# EXPLANATIONS
# --------------------------------------------------
def explain(row):
    reasons = []
    if row["amount_zscore_dept"] > 3:
        reasons.append("Unusual amount for department")
    if row["vendor_amount_ratio"] > 3:
        reasons.append("Vendor billing spike")
    if row["is_night"]:
        reasons.append("Night transaction")
    if row["is_weekend"]:
        reasons.append("Weekend activity")
    if row["vendor_txn_count"] > 80:
        reasons.append("High vendor frequency")
    return ", ".join(reasons) if reasons else "Normal behavior"

df["explanation"] = df.apply(explain, axis=1)

# --------------------------------------------------
# FILTERS
# --------------------------------------------------
st.sidebar.header("🔍 Filters")

dept_filter = st.sidebar.multiselect(
    "Department",
    df["department_id"].unique(),
    df["department_id"].unique()
)

df = df[df["department_id"].isin(dept_filter)]

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Analysis",
    "📄 Flagged Audits",
    "💬 Auditor Assistant"
])

# ---------------- TAB 1 ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", len(df))
    c2.metric("Fraud Detected", df["fraud_flag"].sum())
    c3.metric("High Risk", (df["risk_level"] == "High").sum())
    c4.metric("Fraud Rate (%)", round(df["fraud_flag"].mean() * 100, 2))

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Risk Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["risk_score"], bins=40)
    st.pyplot(fig)

    st.subheader("Department Risk")
    st.bar_chart(df.groupby("department_id")["risk_score"].mean())

    st.subheader("Fraud by Hour")
    st.line_chart(df.groupby("hour")["fraud_flag"].mean())

# ---------------- TAB 3 ----------------
with tab3:
    flagged = df[df["fraud_flag"] == 1][[
        "transaction_id", "department_id", "vendor_id",
        "amount", "risk_score", "risk_level", "explanation"
    ]]

    if flagged.empty:
        st.success("No suspicious transactions detected.")
    else:
        st.dataframe(flagged)

        st.download_button(
            "📥 Download Suspicious Transactions",
            flagged.to_csv(index=False),
            "suspicious_transactions.csv",
            "text/csv"
        )

# ---------------- TAB 4 ----------------
with tab4:
    st.header("💬 Virtual Auditor Assistant")
    st.caption("AI-powered assistant to help auditors quickly understand risks")

    st.markdown("### 📋 Quick Questions")

    question = st.selectbox(
        "Choose a question:",
        [
            "Select a question",
            "How many high-risk transactions are there?",
            "What is the total fraud detected?",
            "Who are the top risky vendors?",
            "Which department has highest risk?",
            "Give risk-level summary",
            "Show fraud trend by hour",
            "Explain how this AI model works"
        ]
    )

    st.markdown("### 🤖 Assistant Response")

    if question == "How many high-risk transactions are there?":
        count = (df["risk_level"] == "High").sum()
        st.success(f"🔴 There are **{count} high-risk transactions** requiring immediate audit.")

    elif question == "What is the total fraud detected?":
        frauds = df["fraud_flag"].sum()
        st.success(f"🚨 The system detected **{frauds} suspicious transactions**.")

    elif question == "Who are the top risky vendors?":
        vendors = df[df["fraud_flag"] == 1]["vendor_id"].value_counts().head(5)
        if vendors.empty:
            st.info("No risky vendors detected.")
        else:
            st.write("🚩 **Top Risky Vendors:**")
            st.dataframe(vendors.reset_index().rename(columns={
                "index": "Vendor ID",
                "vendor_id": "Fraud Count"
            }))

    elif question == "Which department has highest risk?":
        dept = df.groupby("department_id")["risk_score"].mean().idxmax()
        score = df.groupby("department_id")["risk_score"].mean().max()
        st.warning(f"🏢 **{dept} Department** has the highest average risk score ({round(score,2)}).")

    elif question == "Give risk-level summary":
        summary = df["risk_level"].value_counts()
        st.write("📊 **Risk Distribution:**")
        st.dataframe(summary)

    elif question == "Show fraud trend by hour":
        trend = df.groupby("hour")["fraud_flag"].mean()
        st.line_chart(trend)

    elif question == "Explain how this AI model works":
        st.info("""
        🧠 **AI Model Explanation**

        - Uses **Isolation Forest**, an unsupervised anomaly detection algorithm
        - Learns normal spending behavior automatically
        - Flags transactions that significantly deviate from patterns
        - Combines time, vendor behavior, department averages, and amount spikes
        - Produces a **risk score (0–100)** for prioritizing audits
        """)

    else:
        st.info("Select a question to get insights instantly.")

    # Optional custom question
    st.markdown("---")
    st.markdown("### ✍️ Ask Your Own Question")

    custom_q = st.text_input("Type your own audit-related question")

    if custom_q:
        st.warning("🤖 Free-text understanding is limited in demo mode. Please use quick questions above.")
