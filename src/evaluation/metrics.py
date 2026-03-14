"""Evaluation metrics: NSE, KGE, RMSE, percent bias, CWD identity error."""

from typing import Dict

import numpy as np


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency.

    NSE = 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
    """
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        return np.nan
    return 1.0 - numerator / denominator


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Kling-Gupta Efficiency.

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    where r = correlation, alpha = std_pred/std_obs, beta = mean_pred/mean_obs
    """
    if np.std(observed) == 0 or np.std(predicted) == 0:
        return np.nan

    r = np.corrcoef(observed, predicted)[0, 1]
    alpha = np.std(predicted) / np.std(observed)
    beta = np.mean(predicted) / np.mean(observed) if np.mean(observed) != 0 else np.nan

    if np.isnan(r) or np.isnan(beta):
        return np.nan

    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Square Error."""
    return np.sqrt(np.mean((observed - predicted) ** 2))


def percent_bias(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Percent bias.

    pbias = 100 * sum(pred - obs) / sum(obs)
    """
    obs_sum = np.sum(observed)
    if obs_sum == 0:
        return np.nan
    return 100.0 * np.sum(predicted - observed) / obs_sum


def cwd_identity_mae(pet: np.ndarray, aet: np.ndarray, cwd: np.ndarray) -> float:
    """Mean absolute error of CWD identity: |PET - AET - CWD|.

    Should be ~0 by construction since CWD = PET - AET.
    """
    return np.mean(np.abs(pet - aet - cwd))


def compute_all_metrics(
    observed: Dict[str, np.ndarray],
    predicted: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute all metrics for all variables.

    Parameters
    ----------
    observed : dict
        Ground truth arrays keyed by variable name.
    predicted : dict
        Prediction arrays keyed by variable name.

    Returns
    -------
    dict
        Nested dict: {variable: {metric: value}}.
    """
    results = {}

    for var in ["pet", "pck", "aet", "cwd"]:
        obs = observed[var].ravel()
        pred = predicted[var].ravel()

        # Remove NaN entries
        valid = ~(np.isnan(obs) | np.isnan(pred))
        obs = obs[valid]
        pred = pred[valid]

        results[var] = {
            "nse": nse(obs, pred),
            "kge": kge(obs, pred),
            "rmse": rmse(obs, pred),
            "pbias": percent_bias(obs, pred),
        }

    # CWD identity check
    results["cwd_identity_mae"] = cwd_identity_mae(
        predicted["pet"].ravel(),
        predicted["aet"].ravel(),
        predicted["cwd"].ravel(),
    )

    return results


def compute_pixel_nse(
    observed: np.ndarray, predicted: np.ndarray
) -> np.ndarray:
    """Compute per-pixel NSE over the time dimension.

    Parameters
    ----------
    observed : np.ndarray
        Shape (T, H, W).
    predicted : np.ndarray
        Shape (T, H, W).

    Returns
    -------
    np.ndarray
        Shape (H, W) with per-pixel NSE values.
    """
    # Mean over time for each pixel
    obs_mean = np.nanmean(observed, axis=0, keepdims=True)  # (1, H, W)
    numerator = np.nansum((observed - predicted) ** 2, axis=0)
    denominator = np.nansum((observed - obs_mean) ** 2, axis=0)

    nse_map = np.where(denominator > 0, 1.0 - numerator / denominator, np.nan)
    return nse_map


def compute_lag_autocorrelation(
    residuals: np.ndarray, max_lag: int = 12
) -> np.ndarray:
    """Compute autocorrelation of residuals at lags 1 to max_lag.

    Parameters
    ----------
    residuals : np.ndarray
        1D array of residuals (observed - predicted).
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    np.ndarray
        Autocorrelation values at lags 1 to max_lag.
    """
    n = len(residuals)
    mean = np.mean(residuals)
    var = np.var(residuals)
    if var == 0:
        return np.zeros(max_lag)

    acf = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        if lag >= n:
            acf[lag - 1] = np.nan
        else:
            acf[lag - 1] = np.mean(
                (residuals[:-lag] - mean) * (residuals[lag:] - mean)
            ) / var

    return acf
