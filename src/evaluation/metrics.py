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


def quantile_metrics(
    obs: np.ndarray, pred: np.ndarray, quantile: float = 0.95
) -> Dict[str, float]:
    """RMSE and mean bias for obs values above the given quantile."""
    threshold = np.nanpercentile(obs, quantile * 100)
    mask = obs >= threshold
    if mask.sum() == 0:
        return {"rmse": np.nan, "bias": np.nan, "n_samples": 0}
    return {
        "rmse": float(np.sqrt(np.mean((obs[mask] - pred[mask]) ** 2))),
        "bias": float(np.mean(pred[mask] - obs[mask])),
        "n_samples": int(mask.sum()),
    }


def exceedance_reliability(
    obs: np.ndarray, pred: np.ndarray, quantile: float = 0.95
) -> float:
    """Fraction of observed top-Q% events also in predicted top-Q%."""
    obs_thresh = np.nanpercentile(obs, quantile * 100)
    pred_thresh = np.nanpercentile(pred, quantile * 100)
    obs_extreme = obs >= obs_thresh
    pred_extreme = pred >= pred_thresh
    if obs_extreme.sum() == 0:
        return np.nan
    return float(np.sum(obs_extreme & pred_extreme) / np.sum(obs_extreme))


def compute_pixel_extreme_bias(
    observed: np.ndarray, predicted: np.ndarray, quantile: float = 0.95
) -> np.ndarray:
    """Per-pixel mean bias for months where observed > P_quantile (computed per-pixel).

    Parameters
    ----------
    observed : np.ndarray
        Shape (T, H, W).
    predicted : np.ndarray
        Shape (T, H, W).
    quantile : float
        Quantile threshold (e.g. 0.95 for P95).

    Returns
    -------
    np.ndarray
        Shape (H, W) with per-pixel mean bias over extreme months.
    """
    # Per-pixel threshold: P_quantile of each pixel's time series
    thresholds = np.nanpercentile(observed, quantile * 100, axis=0)  # (H, W)
    extreme_mask = observed >= thresholds[np.newaxis, :, :]  # (T, H, W)
    residuals = predicted - observed  # (T, H, W)
    with np.errstate(invalid="ignore"):
        residuals_masked = np.where(extreme_mask, residuals, np.nan)
        bias_map = np.nanmean(residuals_masked, axis=0)  # (H, W)
    return bias_map


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

    # Quantile metrics and exceedance reliability for AET and CWD
    for var in ["aet", "cwd"]:
        obs = observed[var].ravel()
        pred = predicted[var].ravel()
        valid = ~(np.isnan(obs) | np.isnan(pred))
        obs = obs[valid]
        pred = pred[valid]

        for q, label in [(0.95, "p95"), (0.99, "p99")]:
            qm = quantile_metrics(obs, pred, quantile=q)
            er = exceedance_reliability(obs, pred, quantile=q)
            results[f"{var}_{label}"] = {
                "rmse": qm["rmse"],
                "bias": qm["bias"],
                "n_samples": qm["n_samples"],
                "exceedance_hit_rate": er,
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

    # Use a minimum denominator threshold to avoid spurious NSE from
    # near-zero-variance pixels (e.g., snow pixels in a low-snow test year
    # where obs ≈ 0 but has tiny float residuals making denominator > 0).
    min_denom = 1.0  # 1 mm^2 — below this, variance is not meaningful
    nse_map = np.where(denominator >= min_denom, 1.0 - numerator / denominator, np.nan)
    return nse_map


def _wy_groups(dates: np.ndarray) -> np.ndarray:
    """Map each monthly index to its water-year (Oct→Sep) group label.

    Parameters
    ----------
    dates : np.ndarray of str
        ISO date strings 'YYYY-MM-...' aligned to the time axis.

    Returns
    -------
    np.ndarray of int, shape (T,)
        WY label for each timestep. WY ending in Sep YYYY+1 starting Oct YYYY is
        labeled YYYY+1 (the "ending" calendar year, conventional for water years).
    """
    years = np.array([int(d[:4]) for d in dates])
    months = np.array([int(d[5:7]) for d in dates])
    # Oct, Nov, Dec belong to next water year; Jan-Sep belong to current calendar year.
    return years + (months >= 10).astype(int)


def compute_pixel_annual_pearson(
    observed: np.ndarray, predicted: np.ndarray, dates: np.ndarray
) -> np.ndarray:
    """Per-pixel Pearson correlation of WY-mean obs vs WY-mean pred.

    Parameters
    ----------
    observed, predicted : np.ndarray, shape (T, H, W)
    dates : np.ndarray of str, shape (T,)

    Returns
    -------
    np.ndarray, shape (H, W)
        Pearson r per pixel across complete water years. Pixels with fewer than
        2 complete WYs or zero variance return NaN.
    """
    wy = _wy_groups(dates)
    unique_wys, inv = np.unique(wy, return_inverse=True)

    # Only keep WYs with all 12 months present in the input record.
    counts = np.bincount(inv)
    full_mask = counts == 12
    keep_wys = unique_wys[full_mask]

    if len(keep_wys) < 2:
        H, W = observed.shape[1:]
        return np.full((H, W), np.nan, dtype=np.float32)

    keep_idx = np.isin(wy, keep_wys)
    obs_keep = observed[keep_idx]
    pred_keep = predicted[keep_idx]
    wy_keep = wy[keep_idx]

    # Stack into (n_full_wys, 12, H, W) by reshaping in WY order.
    order = np.argsort(wy_keep, kind="stable")
    obs_sorted = obs_keep[order]
    pred_sorted = pred_keep[order]
    n_full = len(keep_wys)
    H, W = observed.shape[1:]
    obs_re = obs_sorted.reshape(n_full, 12, H, W)
    pred_re = pred_sorted.reshape(n_full, 12, H, W)
    obs_annual = np.nanmean(obs_re, axis=1)   # (n_full, H, W)
    pred_annual = np.nanmean(pred_re, axis=1)  # (n_full, H, W)

    # Per-pixel Pearson r over the n_full annual means.
    obs_mean = np.nanmean(obs_annual, axis=0, keepdims=True)
    pred_mean = np.nanmean(pred_annual, axis=0, keepdims=True)
    obs_dev = obs_annual - obs_mean
    pred_dev = pred_annual - pred_mean
    num = np.nansum(obs_dev * pred_dev, axis=0)
    denom = np.sqrt(np.nansum(obs_dev ** 2, axis=0) * np.nansum(pred_dev ** 2, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0, num / denom, np.nan)
    return r.astype(np.float32)


def compute_within_wy_variance_ratio(
    observed: np.ndarray, predicted: np.ndarray, dates: np.ndarray
) -> np.ndarray:
    """Mean per-pixel ratio of monthly std(pred) / std(obs) within each complete WY.

    For each pixel and complete water year, compute std of the 12 monthly values
    for both pred and obs, take ratio pred/obs, then average ratios across WYs.
    Ratio < 1 means the emulator damps sub-annual variance relative to truth
    (the intended effect of the annual-pooled loss).
    """
    wy = _wy_groups(dates)
    unique_wys, inv = np.unique(wy, return_inverse=True)
    counts = np.bincount(inv)
    full_mask = counts == 12
    keep_wys = unique_wys[full_mask]

    H, W = observed.shape[1:]
    if len(keep_wys) == 0:
        return np.full((H, W), np.nan, dtype=np.float32)

    keep_idx = np.isin(wy, keep_wys)
    wy_keep = wy[keep_idx]
    order = np.argsort(wy_keep, kind="stable")
    obs_re = observed[keep_idx][order].reshape(len(keep_wys), 12, H, W)
    pred_re = predicted[keep_idx][order].reshape(len(keep_wys), 12, H, W)

    with np.errstate(invalid="ignore", divide="ignore"):
        obs_std = np.nanstd(obs_re, axis=1)   # (n_full, H, W)
        pred_std = np.nanstd(pred_re, axis=1)
        ratio = np.where(obs_std > 0, pred_std / obs_std, np.nan)
        ratio_map = np.nanmean(ratio, axis=0)  # (H, W)
    return ratio_map.astype(np.float32)


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
