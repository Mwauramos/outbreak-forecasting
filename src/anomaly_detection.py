# src/anomaly_detection.py
# Outbreak early warning using statistical and ML anomaly detection
# ZENIK.AI | Project 4: AI-Enhanced Outbreak Detection & Forecasting

import numpy as np
import pandas as pd
from scipy import stats


def moving_average_alert(series: pd.Series, window: int = 4, threshold: float = 2.0) -> pd.Series:
    """
    Flag weeks where cases exceed mean + threshold * std
    over a rolling window. Classic CDC EARS-style alert.
    """
    rolling_mean = series.shift(1).rolling(window).mean()
    rolling_std = series.shift(1).rolling(window).std()
    upper = rolling_mean + threshold * rolling_std
    return series > upper


def cusum_alert(series: pd.Series, k: float = 0.5, h: float = 4.0) -> pd.Series:
    """
    CUSUM (cumulative sum) control chart for outbreak detection.
    k = allowable slack, h = decision threshold.
    Widely used in epidemiological surveillance.
    """
    mean = series.mean()
    std = series.std()
    normalized = (series - mean) / (std + 1e-9)

    cusum = np.zeros(len(series))
    for i in range(1, len(series)):
        cusum[i] = max(0, cusum[i - 1] + normalized.iloc[i] - k)

    return pd.Series(cusum > h, index=series.index)


def zscore_alert(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """Flag weeks with z-score above threshold."""
    z = np.abs(stats.zscore(series.fillna(series.mean())))
    return pd.Series(z > threshold, index=series.index)


def detect_outbreaks(df: pd.DataFrame, method: str = "cusum") -> pd.DataFrame:
    """
    Apply outbreak detection to a single disease-county time series.
    Returns the dataframe with an alert column added.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    if method == "moving_average":
        df["alert"] = moving_average_alert(df["cases"]).values
    elif method == "cusum":
        df["alert"] = cusum_alert(df["cases"]).values
    elif method == "zscore":
        df["alert"] = zscore_alert(df["cases"]).values
    else:
        raise ValueError(f"Unknown method: {method}")

    return df


def summarise_alerts(df: pd.DataFrame) -> dict:
    """Return summary statistics for detected alerts."""
    alerts = df[df["alert"] == True]
    return {
        "total_weeks": len(df),
        "alert_weeks": len(alerts),
        "alert_rate": round(len(alerts) / len(df) * 100, 1),
        "max_cases_during_alert": int(alerts["cases"].max()) if len(alerts) > 0 else 0,
        "first_alert": str(alerts["date"].min().date()) if len(alerts) > 0 else "None"
    }


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_generator import generate_weekly_data

    df = generate_weekly_data("Cholera", "Mombasa")
    df_alerts = detect_outbreaks(df, method="cusum")
    summary = summarise_alerts(df_alerts)
    print(summary)
    print(df_alerts[df_alerts["alert"]].head())