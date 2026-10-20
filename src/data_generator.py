# src/data_generator.py
# Synthetic Kenya DHIS2 weekly disease surveillance data
# ZENIK.AI | Project 4: AI-Enhanced Outbreak Detection & Forecasting

import numpy as np
import pandas as pd

np.random.seed(42)

KENYA_COUNTIES = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
    "Meru", "Kakamega", "Nyeri", "Machakos", "Kilifi"
]

DISEASES = ["Malaria", "Cholera", "Dengue", "Measles"]


def generate_weekly_data(
    disease: str,
    county: str,
    start_date: str = "2015-01-05",
    end_date: str = "2024-12-30",
    base_rate: float = 50.0,
    outbreak_prob: float = 0.05,
    seasonality: bool = True
) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    n = len(dates)

    week_of_year = np.array([d.isocalendar()[1] for d in dates])
    if seasonality:
        seasonal = (
            1.5 * np.sin(2 * np.pi * week_of_year / 52)
            + 0.5 * np.sin(4 * np.pi * week_of_year / 52)
        )
    else:
        seasonal = np.zeros(n)

    trend = np.linspace(0, base_rate * 0.3, n)

    outbreak_mask = np.random.binomial(1, outbreak_prob, n).astype(bool)
    outbreak_magnitude = np.where(
        outbreak_mask,
        np.random.uniform(2, 6, n) * base_rate,
        0
    )
    outbreak_duration = np.zeros(n)
    in_outbreak = False
    duration_left = 0
    peak_factor = 1.0
    for i in range(n):
        if outbreak_mask[i] and not in_outbreak:
            in_outbreak = True
            duration_left = np.random.randint(4, 9)
            peak_factor = outbreak_magnitude[i]
        if in_outbreak:
            decay = np.exp(-0.3 * (8 - duration_left))
            outbreak_duration[i] = peak_factor * decay
            duration_left -= 1
            if duration_left == 0:
                in_outbreak = False

    rainfall = np.clip(
        50 + 40 * np.sin(2 * np.pi * week_of_year / 52) + np.random.normal(0, 10, n),
        0, None
    )
    temperature = 25 + 3 * np.sin(2 * np.pi * (week_of_year - 10) / 52) + np.random.normal(0, 1, n)

    cases = np.clip(
        base_rate + trend + base_rate * seasonal + outbreak_duration
        + 0.15 * rainfall
        + np.random.poisson(base_rate * 0.1, n),
        0, None
    ).astype(int)

    return pd.DataFrame({
        "date": dates,
        "county": county,
        "disease": disease,
        "cases": cases,
        "rainfall_mm": rainfall.round(1),
        "temperature_c": temperature.round(1),
        "population_density": np.random.uniform(50, 5000, n).round(0),
        "outbreak_flag": outbreak_duration > base_rate * 1.5
    })


def generate_full_dataset() -> pd.DataFrame:
    frames = []
    base_rates = {"Malaria": 120, "Cholera": 15, "Dengue": 30, "Measles": 20}
    for disease in DISEASES:
        for county in KENYA_COUNTIES:
            county_factor = np.random.uniform(0.5, 2.0)
            frames.append(generate_weekly_data(
                disease=disease,
                county=county,
                base_rate=base_rates[disease] * county_factor
            ))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    print("Generating synthetic Kenya DHIS2 surveillance data...")
    df = generate_full_dataset()
    df.to_csv("data/kenya_surveillance.csv", index=False)
    print(f"Saved {len(df):,} rows to data/kenya_surveillance.csv")
    print(df.head())