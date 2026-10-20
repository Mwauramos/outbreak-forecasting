# app.py
# Streamlit dashboard: AI-Enhanced Outbreak Detection & Forecasting
# ZENIK.AI | Project 4

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(__file__))

from src.data_generator import generate_weekly_data, generate_full_dataset, KENYA_COUNTIES, DISEASES
from src.seir_model import run_seir, estimate_r0, estimate_attack_rate
from src.anomaly_detection import detect_outbreaks, summarise_alerts
from src.forecasting import run_forecast_comparison

st.set_page_config(page_title="ZENIK.AI | Outbreak Forecasting", layout="wide", page_icon="🦠")

st.title("🦠 AI-Enhanced Outbreak Detection & Forecasting")
st.markdown("**ZENIK.AI** | Kenya MOH Scenario Planning Dashboard")
st.divider()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    selected_disease = st.selectbox("Disease", DISEASES)
    selected_county = st.selectbox("County", KENYA_COUNTIES)
    detection_method = st.selectbox("Alert Method", ["cusum", "moving_average", "zscore"])
    run_forecast = st.checkbox("Run Forecast Models", value=False)
    st.divider()
    st.markdown("**SEIR Scenario Parameters**")
    beta = st.slider("Transmission rate (β)", 0.1, 1.0, 0.35, 0.05)
    sigma = st.slider("1 / Incubation period (σ)", 0.05, 0.5, 0.2, 0.05)
    gamma = st.slider("1 / Infectious period (γ)", 0.05, 0.5, 0.1, 0.05)
    population = st.number_input("Population", value=1_000_000, step=100_000)

# Generate data
@st.cache_data
def get_data(disease, county):
    return generate_weekly_data(disease, county)

df = get_data(selected_disease, selected_county)
df_alerts = detect_outbreaks(df, method=detection_method)
summary = summarise_alerts(df_alerts)

# KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Weeks", summary["total_weeks"])
col2.metric("Alert Weeks", summary["alert_weeks"])
col3.metric("Alert Rate", f"{summary['alert_rate']}%")
col4.metric("Peak Cases (Alert)", f"{summary['max_cases_during_alert']:,}")

st.divider()

# Time series with alerts
st.subheader(f"Weekly {selected_disease} Cases — {selected_county}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_alerts["date"], y=df_alerts["cases"],
                          mode="lines", name="Cases", line=dict(color="#1f77b4")))
alert_df = df_alerts[df_alerts["alert"]]
fig.add_trace(go.Scatter(x=alert_df["date"], y=alert_df["cases"],
                          mode="markers", name="Alert", marker=dict(color="red", size=6)))
fig.update_layout(height=350, margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# SEIR model
st.divider()
st.subheader("SEIR Compartmental Model")
r0_val = estimate_r0(beta, gamma)
seir_df = run_seir(N=int(population), E0=100, I0=50, beta=beta, sigma=sigma, gamma=gamma, days=180)
attack_rate = estimate_attack_rate(seir_df, int(population))

c1, c2, c3 = st.columns(3)
c1.metric("R₀", f"{r0_val:.2f}")
c2.metric("Attack Rate", f"{attack_rate:.1%}")
c3.metric("Peak Infections", f"{seir_df['I'].max():,.0f}")

fig2 = go.Figure()
for col, color in [("S", "blue"), ("E", "orange"), ("I", "red"), ("R", "green")]:
    fig2.add_trace(go.Scatter(x=seir_df["day"], y=seir_df[col],
                               mode="lines", name=col, line=dict(color=color)))
fig2.update_layout(height=350, xaxis_title="Day", yaxis_title="People", margin=dict(t=20, b=20))
st.plotly_chart(fig2, use_container_width=True)

# Forecasting
if run_forecast:
    st.divider()
    st.subheader("Forecast Model Comparison (Last 12 Weeks)")
    with st.spinner("Training models..."):
        try:
            results, metrics_df = run_forecast_comparison(df)
            st.dataframe(metrics_df, use_container_width=True)

            fig3 = go.Figure()
            dates = results["XGBoost"]["dates"]
            fig3.add_trace(go.Scatter(x=dates, y=results["XGBoost"]["true"],
                                       mode="lines", name="Actual", line=dict(color="black")))
            colors = {"XGBoost": "blue", "Prophet": "orange", "ARIMA": "green"}
            for name, color in colors.items():
                fig3.add_trace(go.Scatter(x=dates, y=results[name]["preds"],
                                           mode="lines", name=name, line=dict(color=color, dash="dash")))
            fig3.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Forecast error: {e}")

st.divider()
st.caption("ZENIK.AI | Synthetic data for demonstration purposes only")