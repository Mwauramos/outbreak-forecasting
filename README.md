# Outbreak Forecasting Dashboard

A disease surveillance and outbreak forecasting dashboard built for Kenya MOH contexts. Combines statistical anomaly detection, epidemiological compartmental modelling, and machine learning forecasting in a single interactive tool.

**Live Demo**: https://outbreak-forecasting-tash4mpblkma4bmibxrevo.streamlit.app

---

## What This Does

Early outbreak detection is one of the most time-sensitive problems in public health. This dashboard integrates three complementary approaches — anomaly detection to flag unusual case counts, compartmental modelling to simulate disease spread, and ML forecasting to project case trajectories — giving surveillance teams multiple signals in one place.

Built with Kenya disease contexts in mind: malaria seasonality, cholera clusters, and dengue emergence patterns.

---

## Features

- CUSUM (cumulative sum) anomaly detection for early outbreak signals
- SEIR compartmental model for disease spread simulation
- Prophet time-series forecasting for case trajectory projection
- XGBoost for non-linear pattern detection in case counts
- Scenario planning with adjustable parameters
- Interactive Streamlit dashboard

---

## Tech Stack

- Python
- Prophet (Meta)
- XGBoost
- Scikit-learn
- NumPy, pandas
- Streamlit

---

## Project Structure

```
outbreak-forecasting/
├── models/                # CUSUM, SEIR, Prophet, XGBoost implementations
├── data/                  # Simulated and sample surveillance data
├── app.py                 # Streamlit dashboard
└── requirements.txt
```

---

## Running Locally

```bash
git clone https://github.com/Mwauramos/outbreak-forecasting
cd outbreak-forecasting
pip install -r requirements.txt
streamlit run app.py
```

---

## Context

Part of the ZENIK.AI portfolio — healthcare AI and data engineering projects for African health systems.

**Author**: Amos Mwaura  
**Contact**: mwauramos.n@gmail.com  
**Portfolio**: https://mwauramos.github.io/zenik-ai
