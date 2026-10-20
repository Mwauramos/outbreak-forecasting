# src/seir_model.py
# SIR/SEIR compartmental epidemiological models
# ZENIK.AI | Project 4: AI-Enhanced Outbreak Detection & Forecasting

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Tuple, Dict


def seir_odes(y: list, t: float, beta: float, sigma: float, gamma: float, N: int) -> list:
    """
    SEIR differential equations.
    S = Susceptible, E = Exposed, I = Infectious, R = Recovered
    beta  = transmission rate
    sigma = 1 / incubation period
    gamma = 1 / infectious period
    """
    S, E, I, R = y
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


def run_seir(N: int, E0: int, I0: int, beta: float, sigma: float, gamma: float, days: int = 365) -> pd.DataFrame:
    """Run SEIR simulation and return results as DataFrame."""
    S0 = N - E0 - I0
    y0 = [S0, E0, I0, 0]
    t = np.linspace(0, days, days)
    solution = odeint(seir_odes, y0, t, args=(beta, sigma, gamma, N))
    df = pd.DataFrame(solution, columns=["S", "E", "I", "R"])
    df["day"] = t
    df["R0"] = beta / gamma
    return df


def estimate_r0(beta: float, gamma: float) -> float:
    return beta / gamma


def estimate_attack_rate(df: pd.DataFrame, N: int) -> float:
    return df["R"].iloc[-1] / N


def fit_seir_to_data(observed_cases: np.ndarray, N: int) -> Dict:
    """Fit SEIR parameters to observed case data using least squares."""
    days = len(observed_cases)

    def loss(params):
        beta, sigma, gamma = params
        if beta <= 0 or sigma <= 0 or gamma <= 0:
            return 1e10
        try:
            df = run_seir(N=N, E0=5, I0=int(observed_cases[0]),
                          beta=beta, sigma=sigma, gamma=gamma, days=days)
            predicted = df["I"].values
            predicted = np.interp(
                np.arange(days),
                np.linspace(0, days - 1, len(predicted)),
                predicted
            )
            return np.sum((predicted - observed_cases) ** 2)
        except Exception:
            return 1e10

    result = minimize(loss, x0=[0.3, 0.2, 0.1], method="Nelder-Mead",
                      options={"maxiter": 1000})
    beta, sigma, gamma = result.x
    return {
        "beta": round(beta, 4),
        "sigma": round(sigma, 4),
        "gamma": round(gamma, 4),
        "R0": round(beta / gamma, 2),
        "incubation_period_days": round(1 / sigma, 1),
        "infectious_period_days": round(1 / gamma, 1)
    }


if __name__ == "__main__":
    N = 4_000_000
    df = run_seir(N=N, E0=100, I0=50, beta=0.35, sigma=0.2, gamma=0.1, days=180)
    print(f"R0 = {estimate_r0(0.35, 0.1):.2f}")
    print(f"Peak infections: {df['I'].max():,.0f} on day {df['I'].idxmax()}")
    print(f"Attack rate: {estimate_attack_rate(df, N):.1%}")