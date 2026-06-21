"""
AI-Driven Equipment Failure Prediction System
Niger Delta Production Telemetry — Random Forest Classifier (48hr failure horizon)
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Equipment Failure Predictor",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ AI-Driven Equipment Failure Prediction System")

st.markdown("""
This application predicts the likelihood of oilfield equipment failure
within the next **48 hours** using operational parameters and a Random Forest model.

Developed for predictive maintenance and operational optimization on Niger Delta production assets.
""")

# ----------------------------------------------------------------------
# Load Model + Scaler
# ----------------------------------------------------------------------
# NOTE on the original syntax: only the model was loaded ("model = joblib.load(...)")
# but Phase 7 referenced `scaler.transform(data)` without ever loading the scaler.
# That would crash with NameError. Both artifacts are now loaded together,
# cached so they aren't reloaded on every interaction, and wrapped in error
# handling so the app fails gracefully if the .pkl files are missing.

@st.cache_resource
def load_artifacts():
    model = joblib.load('equipment_failure_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find `equipment_failure_model.pkl` and/or `scaler.pkl`. "
        "Make sure both files are in the same folder as `app.py`."
    )
    st.stop()

# Feature order MUST match the order used during training:
# X = dataset.iloc[:, :-1].values  →
# [Daily_Oil_Bbl, Tubing_Head_Pressure_PSI, Gas_Lift_Rate_Mscf,
#  Sand_Content_PPM, Pressure_Drop_Pct, Gas_Lift_Stability_Index]
FEATURE_NAMES = [
    'Daily Oil Production',
    'Tubing Head Pressure',
    'Gas Lift Rate',
    'Sand Content',
    'Pressure Drop',
    'Gas Lift Stability'
]

# ----------------------------------------------------------------------
# Input Interface
# ----------------------------------------------------------------------
st.subheader("Enter Operational Parameters")

col1, col2 = st.columns(2)

with col1:
    daily_oil = st.number_input(
        'Daily Oil Production (Bbl/day)',
        min_value=0.0,
        value=3000.0
    )

    pressure = st.number_input(
        'Tubing Head Pressure (PSI)',
        min_value=0.0,
        value=800.0
    )

    gas_rate = st.number_input(
        'Gas Lift Rate (Mscf/day)',
        min_value=0.0,
        value=120.0
    )

with col2:
    sand = st.number_input(
        'Sand Content (PPM)',
        min_value=0.0,
        value=50.0
    )

    pressure_drop = st.number_input(
        'Pressure Drop (%)',
        min_value=0.0,
        value=8.0
    )

    stability = st.number_input(
        'Gas Lift Stability Index',
        min_value=0.0,
        value=5.0
    )

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if st.button('Predict Failure Risk', type="primary"):

    data = np.array([[
        daily_oil,
        pressure,
        gas_rate,
        sand,
        pressure_drop,
        stability
    ]])

    # Scale BEFORE predicting — this was missing the loaded `scaler`
    # reference in the original snippet.
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)[0][1]

    st.subheader("Prediction Results")
    st.metric(
        "Failure Probability (48 hrs)",
        f"{probability * 100:.2f}%"
    )

    # ------------------------------------------------------------------
    # Risk Classification
    # ------------------------------------------------------------------
    if probability < 0.30:
        st.success("🟢 LOW RISK")
        st.write("Equipment operating within acceptable limits.")
    elif probability < 0.70:
        st.warning("🟡 WATCH CONDITION")
        st.write("Increase surveillance and schedule inspection.")
    else:
        st.error("🔴 HIGH RISK")
        st.write("Equipment failure likely within 48 hours.")

    # ------------------------------------------------------------------
    # Maintenance Recommendation
    # ------------------------------------------------------------------
    st.subheader("Recommended Action")
    if probability < 0.30:
        st.info("Continue normal operations.")
    elif probability < 0.70:
        st.info("Plan preventive maintenance during next low-impact production window.")
    else:
        st.info("Mobilize maintenance team immediately and prepare replacement choke valve inserts.")

    # ------------------------------------------------------------------
    # Input Summary
    # ------------------------------------------------------------------
    st.subheader("Current Operational Parameters")
    summary_df = pd.DataFrame({
        'Parameter': FEATURE_NAMES,
        'Value': [daily_oil, pressure, gas_rate, sand, pressure_drop, stability]
    })
    st.dataframe(summary_df, use_container_width=True)

    # ------------------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------------------
    # Original syntax referenced `classifier.feature_importances_` and a
    # manually hand-typed importance array — `classifier` doesn't exist in
    # app.py (it only exists inside the training notebook), and a hardcoded
    # array goes stale the moment the model is retrained. Pulling it
    # directly off the loaded `model` keeps this always in sync.
    st.subheader("Feature Importance")
    importance = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(FEATURE_NAMES, importance, color="#1f77b4")
    ax.set_xlabel("Gini Importance")
    ax.invert_yaxis()
    st.pyplot(fig)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("About")
st.sidebar.info("""
**AI-Based Predictive Maintenance System**

**Model:** Random Forest Classifier

**Prediction Horizon:** 48 Hours

**Industry:** Oil and Gas Production (Niger Delta)
""")

st.sidebar.warning(
    "⚠️ This demo model was trained on a very small sample dataset (10 records). "
    "Treat predictions as illustrative only — retrain on full field telemetry "
    "before using this for real operational decisions."
)
