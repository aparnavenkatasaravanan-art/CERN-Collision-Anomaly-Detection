import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="CERN Collision Anomaly Detection",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 CERN Collision Anomaly Detection")

st.write(
    "AI-based anomaly detection and explainable analysis "
    "of particle collision events."
)

# Load dataset
df = pd.read_csv("cern_anomaly_results.csv")

# Separate anomalies
anomalies = df[df["anomaly_label"] == -1].copy()

# Summary
total_events = len(df)
normal_events = (df["anomaly_label"] == 1).sum()
anomalous_events = (df["anomaly_label"] == -1).sum()
anomaly_rate = (anomalous_events / total_events) * 100

# Metrics
st.subheader("📊 Detection Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Events", f"{total_events:,}")
col2.metric("Normal Events", f"{normal_events:,}")
col3.metric("Anomalous Events", f"{anomalous_events:,}")
col4.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

# Anomaly score distribution
st.subheader("📈 Anomaly Score Distribution")

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(df["anomaly_score"], bins=50)

ax.set_xlabel("Anomaly Score")
ax.set_ylabel("Number of Events")
ax.set_title("Distribution of Anomaly Scores")

st.pyplot(fig)

# Normal vs anomaly
st.subheader("⚖️ Normal vs Anomalous Events")

labels = ["Normal", "Anomaly"]
counts = [normal_events, anomalous_events]

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.bar(labels, counts)

ax2.set_xlabel("Event Type")
ax2.set_ylabel("Number of Events")
ax2.set_title("Normal vs Anomalous Collision Events")

st.pyplot(fig2)

# Top anomalies
st.subheader("🚨 Top 20 Anomalous Events")

top_anomalies = (
    anomalies
    .sort_values("anomaly_score", ascending=False)
    .head(20)
)

st.dataframe(
    top_anomalies,
    use_container_width=True
)

# Explainable AI
st.subheader("🧠 Explainable AI – SHAP")

if os.path.exists("shap_feature_importance.png"):
    st.image(
        "shap_feature_importance.png",
        caption="Top Features Contributing to Anomaly Detection"
    )
else:
    st.info("SHAP feature importance image not found.")

if os.path.exists("event_93089_shap_explanation.png"):
    st.image(
        "event_93089_shap_explanation.png",
        caption="SHAP Explanation for the Most Anomalous Event"
    )
else:
    st.info("Individual SHAP explanation image not found.")

st.success("✅ CERN anomaly detection dashboard loaded successfully!")
