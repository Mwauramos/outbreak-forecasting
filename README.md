# AI-Enhanced Outbreak Detection & Forecasting

A production-grade disease surveillance and scenario planning dashboard built for Kenya Ministry of Health contexts. Combines classical epidemiological modelling with machine learning forecasting and statistical anomaly detection to support early outbreak response.

---

## Live Demo
[outbreak-forecasting-tash4mpblkma4bmibxrevo.streamlit.app](https://outbreak-forecasting-tash4mpblkma4bmibxrevo.streamlit.app)

---

## Problem Statement
Disease surveillance systems in Sub-Saharan Africa are often reactive. By the time an outbreak is confirmed, transmission has already accelerated. This project demonstrates how AI and epidemiological modelling can be combined to detect anomalies early, forecast case trajectories, and simulate intervention scenarios before they are needed in the field.

---

## What This Project Does

### 1. Synthetic Kenya DHIS2 Surveillance Data
Generates realistic weekly disease case counts for 10 Kenya counties across 4 diseases (Malaria, Cholera, Dengue, Measles) from 2015 to 2024. Data includes seasonal patterns aligned to Kenya's long and short rain seasons, simulated outbreak events with exponential decay, weather covariates (rainfall, temperature), and population density — mimicking what a real DHIS2 extract would look like.

### 2. Outbreak Early Warning (Anomaly Detection)
Three statistical alert methods are implemented and selectable from the dashboard:
- **CUSUM** (Cumulative Sum Control Chart) — detects sustained upward shifts in case counts, standard in epidemiological surveillance
- **Moving Average** — flags weeks exceeding mean + 2 standard deviations over a rolling 4-week window (CDC EARS-style)
- **Z-Score** — identifies statistically extreme weeks relative to the full series

### 3. SEIR Compartmental Modelling
An interactive SEIR (Susceptible, Exposed, Infectious, Recovered) model lets users simulate outbreak trajectories by adjusting:
- Transmission rate (β)
- Incubation rate (σ = 1 / incubation period)
- Recovery rate (γ = 1 / infectious period)
- Population size

The dashboard computes R0 (basic reproduction number), peak infection count, and final attack rate in real time.

### 4. Forecast Model Comparison
Three forecasting approaches are benchmarked on the last 12 weeks of held-out data:
- **Prophet** — Facebook's time-series model with rainfall as an external regressor
- **XGBoost** — gradient boosted trees using lag features, rolling averages, and weather covariates
- **ARIMA** — classical autoregressive model as a statistical baseline

Models are evaluated on MAE, RMSE, and MAPE.

---

## Tech Stack
| Layer | Tools |
|---|---|
| Data generation | NumPy, Pandas |
| Epidemiological modelling | SciPy (ODE solver, optimiser) |
| Forecasting | Prophet, XGBoost, Statsmodels (ARIMA) |
| Anomaly detection | SciPy, NumPy |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Cloud |

---

## Project Structure

outbreak-forecasting/
├── app.py # Streamlit dashboard entry point
├── src/
│ ├── data_generator.py # Synthetic DHIS2 data with outbreaks and weather
│ ├── seir_model.py # SEIR ODE model, R0 estimation, parameter fitting
│ ├── forecasting.py # Prophet, XGBoost, ARIMA with evaluation metrics
│ └── anomaly_detection.py # CUSUM, moving average, z-score alert methods
├── data/ # Generated surveillance CSV
├── outputs/ # Model outputs and plots
└── requirements.txt


---

## Author
**Amos Mwaura**  
Healthcare AI Engineer | ZENIK.AI  
[github.com/Mwauramos](https://github.com/Mwauramos) | mwauramos.n@gmail.com
