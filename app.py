import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import random
import time

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ChitraGuptAI",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# LOVABLE-STYLE GLASSMORPHISM CSS - Dark Blue/Gold Theme
# --------------------------------------------------
st.markdown("""
<style>
/* ========== GLOBAL THEME - Modern Dark Blue/Gold ========== */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1f3a 50%, #0a1428 100%);
    color: #e2e8f0;
}

/* Main content */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ========== NAV BUTTON ========== */
.nav-button {
    position: fixed;
    top: 20px;
    left: 20px;
    z-index: 1000;
}

.back-btn {
    background: rgba(59, 130, 246, 0.2);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #93c5fd;
    padding: 12px 24px;
    border-radius: 16px;
    text-decoration: none;
    font-weight: 600;
    font-size: 14px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15);
}

.back-btn:hover {
    background: rgba(59, 130, 246, 0.3);
    border-color: rgba(99, 102, 241, 0.6);
    transform: translateX(-4px) scale(1.02);
    box-shadow: 0 12px 40px rgba(59, 130, 246, 0.25);
}

/* ========== HEADER ========== */
.main-header {
    text-align: center;
    padding: 3rem 2rem;
    margin-bottom: 3rem;
    background: rgba(10, 14, 26, 0.4);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    border: 1px solid rgba(99, 102, 241, 0.3);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
}

.big-title {
    font-size: 48px;
    font-weight: 900;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    text-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
}

.subtitle {
    color: #94a3b8;
    font-size: 18px;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
}

/* ========== GLASS CARDS ========== */
.glass-card {
    background: rgba(26, 31, 58, 0.4);
    backdrop-filter: blur(25px);
    border-radius: 20px;
    border: 1px solid rgba(99, 102, 241, 0.3);
    padding: 2rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.6);
    box-shadow: 0 30px 80px rgba(99, 102, 241, 0.2);
    transform: translateY(-8px) scale(1.02);
}

/* ========== METRICS ========== */
[data-testid="stMetricValue"] {
    font-size: 2.5rem;
    font-weight: 800;
    color: #f59e0b;
    text-shadow: 0 0 20px rgba(245, 158, 11, 0.5);
}

[data-testid="stMetricLabel"] {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.stMetric {
    background: rgba(26, 31, 58, 0.5);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid rgba(99, 102, 241, 0.3);
    transition: all 0.3s ease;
}

.stMetric:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
}

/* ========== CHARTS ========== */
.stPlotlyChart {
    background: rgba(10, 14, 26, 0.5);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    border: 1px solid rgba(99, 102, 241, 0.3);
    padding: 1.5rem;
    margin: 1rem 0;
}

/* ========== BUTTONS ========== */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    color: white;
    border: none;
    border-radius: 16px;
    padding: 12px 24px;
    font-weight: 700;
    font-size: 14px;
    transition: all 0.3s ease;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(99, 102, 241, 0.6);
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

/* ========== DATAFRAME ========== */
[data-testid="stDataFrame"] {
    background: rgba(10, 14, 26, 0.6);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab"] {
    background: rgba(26, 31, 58, 0.5);
    border-radius: 16px;
    color: #94a3b8;
    border: 1px solid rgba(99, 102, 241, 0.2);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    color: white;
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 14, 26, 0.9) 0%, rgba(26, 31, 58, 0.8) 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.3);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
MAIN_WEBSITE_URL = "https://auditnova.lovable.app/"
st.markdown(f"""
<div class="nav-button">
    <a href="{MAIN_WEBSITE_URL}" class="back-btn" target="_blank">
        ← Back to Main
    </a>
</div>
""", unsafe_allow_html=True)
st.markdown("<br><br><br>", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="big-title">🛡️ ChitraGuptAI</div>
    <p class="subtitle">AI-Powered Fraud Detection & Risk Analytics Dashboard - Real-time Anomaly Detection</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL (Keep original)
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    # For demo - create sample model if files don't exist
    try:
        with open("fraud_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        model = IsolationForest(contamination=0.08, random_state=42)
        scaler = StandardScaler()
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------------------------
# SIMULATED REAL-TIME DATA GENERATOR
# --------------------------------------------------
@st.cache_data(ttl=60)  # Refresh every minute
def generate_realtime_data(n=5000):
    departments = ["Health", "Education", "Transport", "Defense", "Rural", "Urban"]
    vendors = [f"V{100+i}" for i in range(40)]
    base_time = datetime.now() - timedelta(days=90)

    data = []
    for i in range(n):
        dept = random.choice(departments)
        vendor = random.choice(vendors)

        amount = np.random.lognormal(mean=10, sigma=1.2)
        if random.random() < 0.08:  # 8% fraud
            amount *= random.uniform(5, 20)

        txn_time = base_time + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
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
# DATA INPUT
# --------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("## 📂 Data Input & Processing")
st.markdown('</div>', unsafe_allow_html=True)

data_option = st.selectbox(
    "Choose Data Source",
    ["Select an option", "Upload CSV File", "Real-time Demo Dataset (5K txns)"]
)

if data_option == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File loaded!")
    else:
        st.warning("Please upload CSV")
        st.stop()
elif data_option == "Real-time Demo Dataset (5K txns)":
    with st.spinner("🔄 Generating real-time transaction data..."):
        df = generate_realtime_data()
    st.success("✅ Real-time dataset loaded!")
else:
    st.stop()

# --------------------------------------------------
# FEATURE ENGINEERING (Original logic)
# --------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.info("🔄 Processing features & running AI model...")
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
X_scaled = scaler.fit_transform(X) if len(scaler.transform(X).shape) == 2 else scaler.transform(X)

# Predictions
df["anomaly_score"] = model.fit_predict(X_scaled) if hasattr(model, 'fit_predict') else model.decision_function(X_scaled)
df["fraud_flag"] = (df["anomaly_score"] == -1).astype(int) if hasattr(model, 'fit_predict') else (model.predict(X_scaled) == -1).astype(int)

df["risk_score"] = (
    (df["anomaly_score"].max() - df["anomaly_score"]) /
    (df["anomaly_score"].max() - df["anomaly_score"].min())
) * 100

df["risk_level"] = pd.cut(
    df["risk_score"], bins=[0, 30, 70, 100],
    labels=["Low", "Medium", "High"]
)

def explain(row):
    reasons = []
    if row["amount_zscore_dept"] > 3: reasons.append("🚨 Unusual dept amount")
    if row["vendor_amount_ratio"] > 3: reasons.append("💸 Vendor spike")
    if row["is_night"]: reasons.append("🌙 Night txn")
    if row["is_weekend"]: reasons.append("📅 Weekend")
    if row["vendor_txn_count"] > 80: reasons.append("🔄 High frequency")
    return ", ".join(reasons) if reasons else "✅ Normal"

df["explanation"] = df.apply(explain, axis=1)
st.success("✅ AI Analysis Complete!")
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FILTERS
# --------------------------------------------------
st.sidebar.header("🔍 Filters")
dept_filter = st.sidebar.multiselect(
    "Department", df["department_id"].unique(), df["department_id"].unique()
)
df_filtered = df[df["department_id"].isin(dept_filter)]

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Live Dashboard", "📈 Risk Analytics", "🚨 Audit Priority", 
    "💬 AI Assistant", "📥 Full Report"
])

# --------------------------------------------------
# TAB 1: LIVE DASHBOARD with REAL-TIME GRAPHS
# --------------------------------------------------
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🎯 Live KPI Dashboard")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Txns", f"{len(df_filtered):,}")
    with c2: st.metric("🚨 Fraud Alert", f"{df_filtered['fraud_flag'].sum():,}")
    with c3: st.metric("🔴 High Risk", f"{(df_filtered['risk_level']=='High').sum():,}")
    with c4: st.metric("📊 Fraud Rate", f"{df_filtered['fraud_flag'].mean()*100:.2f}%")
    with c5: st.metric("⏱️ Last Update", datetime.now().strftime("%H:%M:%S"))
    st.markdown('</div>', unsafe_allow_html=True)

    # REAL-TIME ANIMATED CHARTS
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig1 = px.histogram(df_filtered, x="risk_score", nbins=50, 
                           title="Risk Score Distribution",
                           color_discrete_sequence=['#6366f1', '#a855f7'])
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig2 = px.bar(df_filtered.groupby("department_id")["risk_score"].mean().reset_index(),
                     x="department_id", y="risk_score", title="Dept Risk Avg",
                     color="risk_score", color_continuous_scale="Viridis")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# TAB 2: RISK ANALYTICS - Interactive Plotly
# --------------------------------------------------
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Advanced Risk Analytics")
    
    # Multi-chart subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Fraud by Hour", "Risk vs Amount", "Vendor Risk Heatmap", "Time Trends"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )
    
    # Fraud by Hour
    hourly_fraud = df_filtered.groupby("hour")["fraud_flag"].mean()
    fig.add_trace(go.Scatter(x=hourly_fraud.index, y=hourly_fraud.values, 
                            mode='lines+markers', name='Fraud Rate',
                            line=dict(color='#f59e0b', width=3)), row=1, col=1)
    
    # Risk vs Amount
    fig.add_trace(go.Scatter(x=df_filtered["amount"], y=df_filtered["risk_score"],
                            mode='markers', marker=dict(size=6, color='#6366f1', opacity=0.6),
                            name='Txns'), row=1, col=2)
    
    # Vendor heatmap
    vendor_risk = df_filtered.pivot_table(values='risk_score', index='department_id', 
                                         columns='vendor_id', aggfunc='mean').fillna(0)
    fig.add_trace(go.Heatmap(z=vendor_risk.values, x=vendor_risk.columns[:10],
                            y=vendor_risk.index, colorscale='Viridis',
                            name='Vendor Risk'), row=2, col=1)
    
    # Time trend
    df_filtered['date'] = df_filtered['transaction_time'].dt.date
    daily_fraud = df_filtered.groupby('date')['fraud_flag'].mean()
    fig.add_trace(go.Scatter(x=daily_fraud.index, y=daily_fraud.values,
                            mode='lines', name='Daily Fraud',
                            line=dict(color='#a855f7')), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# TAB 3: AUDIT PRIORITY (Original)
# --------------------------------------------------
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    flagged = df_filtered[df_filtered["fraud_flag"] == 1][[
        "transaction_id", "department_id", "vendor_id", "amount", 
        "risk_score", "risk_level", "explanation"
    ]].sort_values("risk_score", ascending=False)
    
    if flagged.empty:
        st.success("✅ No fraud detected")
    else:
        st.warning(f"🚨 {len(flagged)} transactions need audit!")
        st.dataframe(flagged, use_container_width=True)
        
        st.download_button("📥 Download Flagged", flagged.to_csv(index=False),
                          "flagged_txns.csv")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# TAB 4: AI ASSISTANT (Original enhanced)
# --------------------------------------------------
with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💬 AI Audit Assistant")
    
    question = st.selectbox("Quick Questions:", [
        "Select...", "High-risk count?", "Total fraud?", "Top risky vendors?",
        "Riskiest dept?", "Risk summary", "Fraud by hour", "How it works?"
    ])
    
    if question != "Select...":
        if "count" in question:
            count = (df_filtered["risk_level"] == "High").sum()
            st.metric("High Risk Transactions", count)
        elif "fraud?" in question:
            frauds = df_filtered["fraud_flag"].sum()
            st.metric("Total Fraud Detected", frauds)
        elif "vendors" in question:
            top_vendors = df_filtered[df_filtered["fraud_flag"]==1]["vendor_id"].value_counts().head(5)
            st.bar_chart(top_vendors)
        elif "dept" in question:
            riskiest = df_filtered.groupby("department_id")["risk_score"].mean().idxmax()
            st.success(f"🚩 Riskiest: **{riskiest}**")
        elif "summary" in question:
            st.dataframe(df_filtered["risk_level"].value_counts().reset_index())
    
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# TAB 5: FULL DOWNLOADABLE REPORT
# --------------------------------------------------
with tab5:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📥 Complete Audit Report")
    
    # Generate PDF-like report buffer
    report_buffer = io.BytesIO()
    
    # Summary metrics
    st.subheader("1. Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", len(df_filtered))
    col2.metric("Fraud Cases", df_filtered['fraud_flag'].sum())
    col3.metric("High Risk", (df_filtered['risk_level']=='High').sum())
    col4.metric("Fraud Rate", f"{df_filtered['fraud_flag'].mean()*100:.2f}%")
    
    st.subheader("2. Risk Distribution")
    fig_summary = px.pie(df_filtered, names='risk_level', 
                        title="Risk Level Breakdown")
    st.plotly_chart(fig_summary, use_container_width=True)
    
    st.subheader("3. Top Suspicious Transactions")
    top_flagged = flagged.head(20)
    st.dataframe(top_flagged)
    
    # DOWNLOAD FULL REPORT
    full_report = df_filtered.to_csv(index=False)
    st.download_button(
        label="📊 Download Complete Report (CSV)",
        data=full_report,
        file_name=f"ChitraGuptAI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    # HTML Report for better formatting
    html_report = df_filtered.to_html(classes='table table-striped', escape=False)
    st.download_button(
        label="📄 Download HTML Dashboard Report",
        data=html_report,
        file_name=f"ChitraGuptAI_Dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 2rem;'>
    <p>🛡️ ChitraGuptAI - AI Fraud Detection | Powered by Streamlit & Plotly</p>
    <p>🔄 Real-time updates every 60 seconds | Last update: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
