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
# Exploratory Data Analysis
st.subheader("📊 Exploratory Data Analysis")

eda_images = [
    ("Invariant Mass Distribution", "invariant_mass_distribution.png"),
    ("E1 vs E2 Energy Distribution", "E1_E2_energy_distribution.png"),
    ("PT1 vs PT2 Distribution", "pt1_pt2_distribution.png"),
    ("Particle Momentum Components", "particle1_momentum_components.png"),
    ("Eta1 Distribution", "eta1_distribution.png"),
    ("Phi1 Distribution", "phi1_distribution.png"),
    ("Charge Distribution", "charge_distribution.png"),
    ("Correlation Matrix", "correlation_matrix.png"),
    ("Invariant Mass Boxplot", "M_boxplot.png")
]

for i in range(0, len(eda_images), 2):
    col1, col2 = st.columns(2)

    with col1:
        title, image = eda_images[i]
        st.write(f"**{title}**")
        if os.path.exists(image):
            st.image(image, use_container_width=True)
        else:
            st.warning(f"{image} not found.")

    with col2:
        if i + 1 < len(eda_images):
            title, image = eda_images[i + 1]
            st.write(f"**{title}**")
            if os.path.exists(image):
                st.image(image, use_container_width=True)
            else:
                st.warning(f"{image} not found.")

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
