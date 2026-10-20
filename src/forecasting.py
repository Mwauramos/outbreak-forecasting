# src/forecasting.py
# Disease forecasting: Prophet, XGBoost, ARIMA comparison
# ZENIK.AI | Project 4: AI-Enhanced Outbreak Detection & Forecasting

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based and lag features for ML models."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["lag_1"] = df["cases"].shift(1)
    df["lag_2"] = df["cases"].shift(2)
    df["lag_4"] = df["cases"].shift(4)
    df["rolling_4"] = df["cases"].shift(1).rolling(4).mean()
    df["rolling_8"] = df["cases"].shift(1).rolling(8).mean()
    return df.dropna().reset_index(drop=True)


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame):
    """Train XGBoost regressor with weather and lag features."""
    from xgboost import XGBRegressor

    feature_cols = ["week", "month", "year", "lag_1", "lag_2", "lag_4",
                    "rolling_4", "rolling_8", "rainfall_mm", "temperature_c"]

    X_train = train[feature_cols]
    y_train = train["cases"]
    X_test = test[feature_cols]
    y_test = test["cases"]

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return preds, y_test.values, model


def train_prophet(train: pd.DataFrame, test: pd.DataFrame):
    """Train Facebook Prophet with rainfall as external regressor."""
    from prophet import Prophet

    prophet_train = train.rename(columns={"date": "ds", "cases": "y"})[["ds", "y", "rainfall_mm"]]
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.add_regressor("rainfall_mm")
    model.fit(prophet_train)

    future = test[["date", "rainfall_mm"]].rename(columns={"date": "ds"})
    forecast = model.predict(future)
    preds = forecast["yhat"].values
    preds = np.clip(preds, 0, None)

    return preds, test["cases"].values, model, forecast


def train_arima(train: pd.DataFrame, test: pd.DataFrame):
    """Train ARIMA model on case counts."""
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(train["cases"].values, order=(2, 1, 2))
    fitted = model.fit()
    preds = fitted.forecast(steps=len(test))
    preds = np.clip(preds, 0, None)

    return preds, test["cases"].values, fitted


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Compute MAE, RMSE, and MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    return {"model": model_name, "MAE": round(mae, 2),
            "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}


def run_forecast_comparison(df: pd.DataFrame, test_weeks: int = 12):
    """Run all three models and return predictions and metrics."""
    df = prepare_features(df)
    split = len(df) - test_weeks
    train, test = df.iloc[:split], df.iloc[split:]

    results = {}

    # XGBoost
    xgb_preds, xgb_true, xgb_model = train_xgboost(train, test)
    results["XGBoost"] = {
        "preds": xgb_preds, "true": xgb_true,
        "metrics": evaluate(xgb_true, xgb_preds, "XGBoost"),
        "dates": test["date"].values
    }

    # Prophet
    prophet_preds, prophet_true, prophet_model, prophet_forecast = train_prophet(train, test)
    results["Prophet"] = {
        "preds": prophet_preds, "true": prophet_true,
        "metrics": evaluate(prophet_true, prophet_preds, "Prophet"),
        "dates": test["date"].values
    }

    # ARIMA
    arima_preds, arima_true, arima_model = train_arima(train, test)
    results["ARIMA"] = {
        "preds": arima_preds, "true": arima_true,
        "metrics": evaluate(arima_true, arima_preds, "ARIMA"),
        "dates": test["date"].values
    }

    metrics_df = pd.DataFrame([v["metrics"] for v in results.values()])
    return results, metrics_df


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_generator import generate_weekly_data

    df = generate_weekly_data("Malaria", "Nairobi")
    results, metrics = run_forecast_comparison(df)
    print(metrics.to_string(index=False))