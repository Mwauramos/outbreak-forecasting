## AI-Enhanced Outbreak Detection & Forecasting

A Kenya MOH scenario planning dashboard for disease surveillance, outbreak detection, and forecasting using SEIR compartmental models and machine learning.

## Live Demo
[outbreak-forecasting-tash4mpblkma4bmibxrevo.streamlit.app](https://outbreak-forecasting-tash4mpblkma4bmibxrevo.streamlit.app)

## Features
- Synthetic Kenya DHIS2 weekly surveillance data across 10 counties and 4 diseases
- CUSUM, moving average, and z-score outbreak early warning alerts
- SEIR compartmental model with interactive transmission parameters
- Forecast model comparison: XGBoost, Prophet, and ARIMA
- Interactive Streamlit dashboard for scenario planning

## Tech Stack
Python, Streamlit, Plotly, Prophet, XGBoost, Statsmodels, SciPy, Pandas, NumPy

## Project Structure

outbreak-forecasting/
├── app.py # Streamlit dashboard
├── src/
│ ├── data_generator.py # Synthetic DHIS2 data
│ ├── seir_model.py # SEIR compartmental model
│ ├── forecasting.py # Prophet, XGBoost, ARIMA
│ └── anomaly_detection.py # Outbreak alert detection
├── data/ # Generated surveillance data
└── requirements.txt


## Author
**Amos Mwaura** | ZENIK.AI | Healthcare AI Engineering
mwauramos.n@gmail.com
