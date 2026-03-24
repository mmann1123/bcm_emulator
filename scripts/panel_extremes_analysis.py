"""Panel data analysis of dynamic variable correlates with AET/CWD extremes.

Builds a pixel × time panel from the zarr store, engineers autoregressive
and interaction features, then fits gradient-boosted tree models to identify
which dynamic variables (and their lags/nonlinear transforms) best predict:
  1. Whether a pixel-month is an AET/CWD extreme (classification)
  2. The magnitude of AET/CWD in the tail (regression on P90+ subset)

Also fits a panel fixed-effects model (pixel FE + month FE) to control for
unobserved heterogeneity before examining dynamic variable importance.

Usage:
    conda run -n deep_field python scripts/panel_extremes_analysis.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_PIXELS = 5000          # subsample of valid pixels (panel cross-section)
MAX_LAGS = 3             # autoregressive lag depth (months)
EXTREME_QUANTILE = 0.90  # P90 threshold for "extreme" classification
SEED = 42


def load_panel_from_zarr(zarr_path: str, n_pixels: int, seed: int) -> pd.DataFrame:
    """Extract a pixel × time panel DataFrame from the zarr store.

    Returns DataFrame with columns for each dynamic input, static input,
    and target variable, indexed by (pixel_id, time_idx).
    """
    import zarr

    store = zarr.open(str(zarr_path), mode="r")

    # Read metadata
    valid_mask = np.array(store["meta/valid_mask"])
    time_index = np.array(store["meta/time"]).astype(str)

    # Dynamic inputs: (T, C_dyn, H, W)
    dyn = store["inputs/dynamic"]
    T, C_dyn, H, W = dyn.shape

    # Static inputs: (C_static, H, W)
    static = store["inputs/static"]
    C_static = static.shape[0]

    # Targets
    target_names = ["pet", "pck", "aet", "cwd"]

    # Normalization stats for denormalization
    dyn_means = np.array(store["norm/dynamic_mean"])    # (C_dyn,)
    dyn_stds = np.array(store["norm/dynamic_std"])      # (C_dyn,)
    static_means = np.array(store["norm/static_mean"])
    static_stds = np.array(store["norm/static_std"])
    target_mean_arr = np.array(store["norm/target_mean"])  # (4,) in order pet,pck,aet,cwd
    target_std_arr = np.array(store["norm/target_std"])
    target_means = {v: float(target_mean_arr[i]) for i, v in enumerate(target_names)}
    target_stds = {v: float(target_std_arr[i]) for i, v in enumerate(target_names)}

    # Channel names
    dyn_names = [
        "ppt", "tmin", "tmax", "wet_days", "ppt_intensity",
        "srad", "snow_frac", "pck_prev", "aet_prev", "vpd", "kbdi",
    ]
    static_names = [
        "elev", "topo_solar", "lat", "lon", "ksat", "sand", "clay",
        "soil_depth", "aridity_index", "field_capacity", "wilting_point",
        "SOM", "windward_index", "awc_total", "fveg_class_id",
    ]

    # Subsample valid pixels
    valid_yx = np.argwhere(valid_mask)  # (N_valid, 2)
    rng = np.random.RandomState(seed)
    n_valid = len(valid_yx)
    n_sample = min(n_pixels, n_valid)
    chosen_idx = rng.choice(n_valid, size=n_sample, replace=False)
    chosen_yx = valid_yx[chosen_idx]
    logger.info(f"Sampling {n_sample} pixels from {n_valid} valid pixels")

    rows = chosen_yx[:, 0]
    cols = chosen_yx[:, 1]

    # Extract dynamic data: (T, C_dyn, H, W) -> (T, C_dyn, n_pixels)
    logger.info("Loading dynamic inputs...")
    dyn_arr = np.array(dyn)  # full load
    dyn_panel = dyn_arr[:, :, rows, cols]  # (T, C_dyn, n_pixels)

    # Denormalize dynamic
    for c in range(min(C_dyn, len(dyn_names))):
        dyn_panel[:, c, :] = dyn_panel[:, c, :] * dyn_stds[c] + dyn_means[c]

    # Extract static data: (C_static, H, W) -> (C_static, n_pixels)
    logger.info("Loading static inputs...")
    static_arr = np.array(static)
    static_panel = static_arr[:, rows, cols]  # (C_static, n_pixels)

    # Denormalize static (except fveg_class_id which is categorical)
    n_cont_static = C_static - 1  # last is fveg
    for c in range(min(n_cont_static, len(static_means))):
        static_panel[c, :] = static_panel[c, :] * static_stds[c] + static_means[c]

    # Extract targets: (T, H, W) -> (T, n_pixels)
    logger.info("Loading targets...")
    targets = {}
    for v in target_names:
        t_arr = np.array(store[f"targets/{v}"])  # (T, H, W)
        t_panel = t_arr[:, rows, cols]           # (T, n_pixels)
        targets[v] = t_panel * target_stds[v] + target_means[v]

    # Build DataFrame
    logger.info("Building panel DataFrame...")
    records = []
    for pi in range(n_sample):
        for ti in range(T):
            rec = {"pixel_id": pi, "time_idx": ti}
            if ti < len(time_index):
                rec["date"] = time_index[ti]
                # Extract month for seasonality
                try:
                    rec["month"] = int(time_index[ti].split("-")[1])
                except (IndexError, ValueError):
                    rec["month"] = (ti % 12) + 1
            else:
                rec["month"] = (ti % 12) + 1

            # Dynamic variables
            for c, name in enumerate(dyn_names[:C_dyn]):
                rec[name] = dyn_panel[ti, c, pi]

            # Static variables (constant across time for a pixel)
            for c, name in enumerate(static_names[:C_static]):
                rec[name] = static_panel[c, pi]

            # Targets
            for v in target_names:
                rec[v] = targets[v][ti, pi]

            records.append(rec)

    df = pd.DataFrame(records)
    logger.info(f"Panel shape: {df.shape} ({n_sample} pixels × {T} months)")
    return df


def engineer_features(df: pd.DataFrame, max_lags: int) -> pd.DataFrame:
    """Add autoregressive lags, rolling stats, and interaction features."""
    logger.info("Engineering features...")

    dyn_cols = [
        "ppt", "tmin", "tmax", "wet_days", "ppt_intensity",
        "srad", "snow_frac", "vpd", "kbdi",
    ]

    # Sort for proper lag computation
    df = df.sort_values(["pixel_id", "time_idx"]).reset_index(drop=True)

    # Autoregressive lags for key dynamic variables
    lag_cols = []
    for col in dyn_cols:
        if col not in df.columns:
            continue
        for lag in range(1, max_lags + 1):
            lag_name = f"{col}_lag{lag}"
            df[lag_name] = df.groupby("pixel_id")[col].shift(lag)
            lag_cols.append(lag_name)

    # Autoregressive lags for targets (what the model would have as aet_prev, pck_prev)
    for v in ["aet", "cwd", "pet", "pck"]:
        for lag in range(1, max_lags + 1):
            lag_name = f"{v}_lag{lag}"
            df[lag_name] = df.groupby("pixel_id")[v].shift(lag)
            lag_cols.append(lag_name)

    # Rolling statistics (3-month and 6-month windows)
    for col in ["ppt", "tmax", "vpd", "srad", "kbdi"]:
        if col not in df.columns:
            continue
        for win in [3, 6]:
            roll_name = f"{col}_roll{win}_mean"
            df[roll_name] = df.groupby("pixel_id")[col].transform(
                lambda x: x.rolling(win, min_periods=1).mean()
            )
            roll_std_name = f"{col}_roll{win}_std"
            df[roll_std_name] = df.groupby("pixel_id")[col].transform(
                lambda x: x.rolling(win, min_periods=1).std()
            )

    # Cumulative deficit: rolling sum of (PET - PPT) over 3 and 6 months
    if "pet" in df.columns and "ppt" in df.columns:
        df["pet_minus_ppt"] = df["pet"] - df["ppt"]
        for win in [3, 6]:
            df[f"cum_deficit_{win}m"] = df.groupby("pixel_id")["pet_minus_ppt"].transform(
                lambda x: x.rolling(win, min_periods=1).sum()
            )

    # Interaction terms: VPD × temperature, PPT × soil properties
    if "vpd" in df.columns and "tmax" in df.columns:
        df["vpd_x_tmax"] = df["vpd"] * df["tmax"]
    if "ppt" in df.columns and "ksat" in df.columns:
        df["ppt_x_ksat"] = df["ppt"] * df["ksat"]
    if "vpd" in df.columns and "srad" in df.columns:
        df["vpd_x_srad"] = df["vpd"] * df["srad"]
    if "kbdi" in df.columns and "vpd" in df.columns:
        df["kbdi_x_vpd"] = df["kbdi"] * df["vpd"]

    # Soil water holding capacity proxy
    if all(c in df.columns for c in ["field_capacity", "wilting_point", "soil_depth"]):
        df["awc_total"] = (df["field_capacity"] - df["wilting_point"]) * df["soil_depth"] * 1000

    # Drop rows with NaN from lagging (first max_lags months per pixel)
    n_before = len(df)
    df = df.dropna(subset=lag_cols[:len(dyn_cols)]).reset_index(drop=True)  # only require first-order lags
    logger.info(f"Dropped {n_before - len(df)} rows with NaN lags, {len(df)} remain")

    return df


def classify_extremes(df: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Add binary extreme indicators and extreme magnitude columns."""
    for v in ["aet", "cwd"]:
        threshold = df[v].quantile(quantile)
        df[f"{v}_extreme"] = (df[v] >= threshold).astype(int)
        df[f"{v}_extreme_mag"] = df[v].where(df[v] >= threshold, np.nan)
        logger.info(
            f"{v.upper()} P{int(quantile*100)} threshold: {threshold:.1f} mm, "
            f"N extreme: {df[f'{v}_extreme'].sum()}"
        )
    return df


def run_gradient_boosting_analysis(df: pd.DataFrame, target_var: str):
    """Fit XGBoost to predict extreme vs. non-extreme, report feature importance."""
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score, r2_score

    # Feature columns: everything except targets, IDs, dates
    exclude = {
        "pixel_id", "time_idx", "date", "month",
        "pet", "pck", "aet", "cwd",
        "aet_extreme", "cwd_extreme",
        "aet_extreme_mag", "cwd_extreme_mag",
        "pet_minus_ppt",
    }
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

    # Also exclude target lags to avoid leakage for classification
    # Keep aet_lag, cwd_lag etc. — these represent autoregressive info the model DOES have
    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    print(f"\n{'='*70}")
    print(f"GRADIENT BOOSTING: {target_var.upper()} EXTREME CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Features: {len(feature_cols)}, Samples: {len(X)}")

    # --- Classification: extreme vs non-extreme ---
    y_cls = df[f"{target_var}_extreme"]

    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_cls == 0).sum() / max((y_cls == 1).sum(), 1),
        random_state=SEED, n_jobs=-1, verbosity=0,
    )

    # 5-fold CV
    scores = cross_val_score(clf, X, y_cls, cv=5, scoring="roc_auc")
    print(f"\nClassification ROC-AUC (5-fold CV): {scores.mean():.4f} ± {scores.std():.4f}")

    # Fit on full data for feature importance
    clf.fit(X, y_cls)
    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    print(f"\nTop 20 features (classification):")
    print(f"{'Feature':<40} {'Importance':>12}")
    print("-" * 55)
    for feat, val in imp.head(20).items():
        print(f"  {feat:<38} {val:>12.4f}")

    # --- Regression on extremes only: what predicts magnitude? ---
    mask_extreme = df[f"{target_var}_extreme"] == 1
    if mask_extreme.sum() > 100:
        print(f"\n{'='*70}")
        print(f"GRADIENT BOOSTING: {target_var.upper()} EXTREME MAGNITUDE (P90+ only)")
        print(f"{'='*70}")

        X_ext = X.loc[mask_extreme]
        y_ext = df.loc[mask_extreme, target_var]

        reg = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbosity=0,
        )
        scores_r2 = cross_val_score(reg, X_ext, y_ext, cv=5, scoring="r2")
        print(f"\nRegression R² on extremes (5-fold CV): {scores_r2.mean():.4f} ± {scores_r2.std():.4f}")

        reg.fit(X_ext, y_ext)
        imp_reg = pd.Series(reg.feature_importances_, index=feature_cols).sort_values(ascending=False)

        print(f"\nTop 20 features (extreme magnitude):")
        print(f"{'Feature':<40} {'Importance':>12}")
        print("-" * 55)
        for feat, val in imp_reg.head(20).items():
            print(f"  {feat:<38} {val:>12.4f}")

    return clf, imp


def run_panel_fixed_effects(df: pd.DataFrame, target_var: str):
    """Panel regression with pixel and month fixed effects using statsmodels.

    Demeans by pixel (within estimator) and includes month dummies,
    then reports which dynamic variables matter after controlling for
    pixel-level heterogeneity (soil, elevation, etc.).
    """
    import statsmodels.api as sm

    print(f"\n{'='*70}")
    print(f"PANEL FIXED-EFFECTS: {target_var.upper()} (within estimator)")
    print(f"{'='*70}")

    # Dynamic features only (static are absorbed by pixel FE)
    dyn_features = [
        "ppt", "tmin", "tmax", "wet_days", "ppt_intensity",
        "srad", "snow_frac", "vpd", "kbdi",
    ]
    # Add lags and rolling stats
    extra_cols = [c for c in df.columns if any(
        c.startswith(f"{d}_lag") or c.startswith(f"{d}_roll")
        for d in dyn_features
    )]
    # Add interaction terms
    interaction_cols = [c for c in df.columns if "_x_" in c]
    # Add cumulative deficit
    cum_cols = [c for c in df.columns if c.startswith("cum_deficit")]

    all_features = [c for c in dyn_features + extra_cols + interaction_cols + cum_cols
                    if c in df.columns]

    # Within transformation: demean by pixel (absorbs all static effects)
    panel = df[["pixel_id", "month", target_var] + all_features].dropna()

    # Pixel-demean (within estimator)
    pixel_means = panel.groupby("pixel_id")[
        [target_var] + all_features
    ].transform("mean")
    panel_demeaned = panel[[target_var] + all_features] - pixel_means
    panel_demeaned["month"] = panel["month"]

    # Add month dummies
    month_dummies = pd.get_dummies(panel_demeaned["month"], prefix="m", drop_first=True).astype(float)
    X = pd.concat([panel_demeaned[all_features], month_dummies], axis=1)
    X = sm.add_constant(X)
    y = panel_demeaned[target_var]

    # Drop any remaining NaN
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    # OLS on demeaned data = FE estimator
    try:
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": panel.loc[valid, "pixel_id"]})

        print(f"\nR² (within): {model.rsquared:.4f}")
        print(f"N obs: {model.nobs:,.0f}")
        print(f"\nSignificant dynamic variables (p < 0.05):")
        print(f"{'Variable':<40} {'Coef':>10} {'Std Err':>10} {'t':>8} {'p':>8}")
        print("-" * 80)

        # Sort by absolute t-stat
        results = []
        for var in all_features:
            if var in model.params.index:
                results.append((
                    var,
                    model.params[var],
                    model.bse[var],
                    model.tvalues[var],
                    model.pvalues[var],
                ))
        results.sort(key=lambda x: abs(x[3]), reverse=True)

        for var, coef, se, t, p in results[:30]:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {var:<38} {coef:>10.4f} {se:>10.4f} {t:>8.2f} {p:>8.4f} {sig}")

        return model
    except Exception as e:
        logger.warning(f"Panel FE failed: {e}")
        return None


def run_nonlinear_conditional_analysis(df: pd.DataFrame, target_var: str):
    """Analyze nonlinear conditional relationships: how do dynamic variables
    behave differently in extreme vs. non-extreme months?

    Uses binned means and quantile regressions to show threshold effects.
    """
    from scipy import stats as sp_stats

    print(f"\n{'='*70}")
    print(f"CONDITIONAL ANALYSIS: {target_var.upper()} extreme vs normal months")
    print(f"{'='*70}")

    is_extreme = df[f"{target_var}_extreme"] == 1
    dyn_cols = [
        "ppt", "tmin", "tmax", "wet_days", "ppt_intensity",
        "srad", "snow_frac", "vpd", "kbdi",
    ]

    print(f"\n{'Variable':<25} {'Normal mean':>12} {'Extreme mean':>12} {'Diff':>10} {'Cohen d':>10} {'p-value':>12}")
    print("-" * 85)

    effect_sizes = []
    for col in dyn_cols:
        if col not in df.columns:
            continue
        normal_vals = df.loc[~is_extreme, col].dropna()
        extreme_vals = df.loc[is_extreme, col].dropna()

        if len(normal_vals) < 10 or len(extreme_vals) < 10:
            continue

        nm = normal_vals.mean()
        em = extreme_vals.mean()
        diff = em - nm

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(normal_vals) - 1) * normal_vals.std()**2 +
             (len(extreme_vals) - 1) * extreme_vals.std()**2) /
            (len(normal_vals) + len(extreme_vals) - 2)
        )
        cohen_d = diff / pooled_std if pooled_std > 0 else 0

        # Welch's t-test
        t_stat, p_val = sp_stats.ttest_ind(extreme_vals, normal_vals, equal_var=False)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {col:<23} {nm:>12.2f} {em:>12.2f} {diff:>10.2f} {cohen_d:>10.3f} {p_val:>12.2e} {sig}")
        effect_sizes.append((col, abs(cohen_d), diff, p_val))

    effect_sizes.sort(key=lambda x: x[1], reverse=True)
    print(f"\nRanked by |Cohen's d|:")
    for i, (col, d, diff, p) in enumerate(effect_sizes):
        direction = "↑" if diff > 0 else "↓"
        print(f"  {i+1}. {col:<25} d={d:.3f}  {direction} in extremes")

    # --- Quantile-specific correlations ---
    print(f"\n{'='*70}")
    print(f"QUANTILE-SPECIFIC CORRELATIONS with {target_var.upper()}")
    print(f"{'='*70}")
    print(f"How does correlation change across the {target_var} distribution?")

    quantile_bins = [(0, 0.25, "Q1 (low)"), (0.25, 0.5, "Q2"), (0.5, 0.75, "Q3"), (0.75, 0.9, "Q4"), (0.9, 1.0, "P90+")]

    header = f"{'Variable':<20}"
    for _, _, label in quantile_bins:
        header += f" {label:>10}"
    print(header)
    print("-" * (20 + 11 * len(quantile_bins)))

    for col in dyn_cols:
        if col not in df.columns:
            continue
        row = f"  {col:<18}"
        for qlo, qhi, label in quantile_bins:
            lo = df[target_var].quantile(qlo)
            hi = df[target_var].quantile(qhi)
            mask = (df[target_var] >= lo) & (df[target_var] < hi) if qhi < 1.0 else (df[target_var] >= lo)
            subset = df.loc[mask]
            if len(subset) > 30:
                r = subset[[col, target_var]].corr().iloc[0, 1]
                row += f" {r:>10.3f}"
            else:
                row += f" {'N/A':>10}"
        print(row)


def run_autoregressive_importance(df: pd.DataFrame, target_var: str):
    """Specifically test which lagged features matter most for predicting
    extreme events, using a focused autoregressive model."""
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score

    print(f"\n{'='*70}")
    print(f"AUTOREGRESSIVE FEATURE IMPORTANCE: {target_var.upper()}")
    print(f"{'='*70}")
    print("Which lagged variables best predict current-month target value?")

    # Build feature set: current dynamic + all lags + rolling stats
    ar_features = []
    dyn_base = ["ppt", "tmin", "tmax", "wet_days", "ppt_intensity",
                "srad", "snow_frac", "vpd", "kbdi"]

    for col in df.columns:
        # Current dynamic
        if col in dyn_base:
            ar_features.append(col)
        # Lagged dynamic
        elif any(col.startswith(f"{d}_lag") for d in dyn_base):
            ar_features.append(col)
        # Rolling stats
        elif any(col.startswith(f"{d}_roll") for d in dyn_base):
            ar_features.append(col)
        # Lagged targets (autoregressive)
        elif any(col.startswith(f"{v}_lag") for v in ["aet", "cwd", "pet", "pck"]):
            ar_features.append(col)
        # Cumulative deficit
        elif col.startswith("cum_deficit"):
            ar_features.append(col)
        # Interactions
        elif "_x_" in col:
            ar_features.append(col)

    # Add month for seasonality
    ar_features.append("month")

    # Add key static vars (not absorbed since we're not demeaning here)
    static_to_include = ["elev", "soil_depth", "aridity_index", "field_capacity",
                         "wilting_point", "ksat", "fveg_class_id"]
    for s in static_to_include:
        if s in df.columns:
            ar_features.append(s)

    ar_features = [f for f in ar_features if f in df.columns]
    X = df[ar_features].fillna(df[ar_features].median())
    y = df[target_var]

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    reg = XGBRegressor(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )

    scores = cross_val_score(reg, X, y, cv=5, scoring="r2")
    print(f"\nFull AR model R² (5-fold CV): {scores.mean():.4f} ± {scores.std():.4f}")

    reg.fit(X, y)
    imp = pd.Series(reg.feature_importances_, index=ar_features).sort_values(ascending=False)

    print(f"\nTop 25 features:")
    print(f"{'Feature':<40} {'Importance':>12}")
    print("-" * 55)
    for feat, val in imp.head(25).items():
        # Categorize
        if "_lag" in feat:
            cat = "[LAG]"
        elif "_roll" in feat:
            cat = "[ROLL]"
        elif "_x_" in feat:
            cat = "[INTERACT]"
        elif feat.startswith("cum_"):
            cat = "[CUMUL]"
        elif feat in dyn_base:
            cat = "[CURRENT]"
        elif feat == "month":
            cat = "[SEASON]"
        else:
            cat = "[STATIC]"
        print(f"  {feat:<30} {cat:<10} {val:>12.4f}")

    # --- Feature importance specifically for P90+ subset ---
    mask_ext = df[f"{target_var}_extreme"] == 1
    if mask_ext.sum() > 200:
        X_ext = df.loc[mask_ext, ar_features].fillna(df[ar_features].median())
        y_ext = df.loc[mask_ext, target_var]

        reg_ext = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            random_state=SEED, n_jobs=-1, verbosity=0,
        )
        reg_ext.fit(X_ext, y_ext)
        imp_ext = pd.Series(reg_ext.feature_importances_, index=ar_features).sort_values(ascending=False)

        print(f"\nTop 15 features for EXTREMES ONLY (P90+):")
        print(f"{'Feature':<40} {'Imp (all)':>12} {'Imp (P90+)':>12} {'Ratio':>8}")
        print("-" * 75)
        for feat, val in imp_ext.head(15).items():
            all_imp = imp.get(feat, 0)
            ratio = val / max(all_imp, 1e-6)
            marker = " <<<" if ratio > 2.0 else ""
            print(f"  {feat:<38} {all_imp:>12.4f} {val:>12.4f} {ratio:>8.1f}{marker}")

        print("\n  Features with '<<<' are disproportionately important for extremes")


def main():
    from src.utils.config import load_config

    config_path = PROJECT_ROOT / "config.yaml"
    cfg = load_config(str(config_path))
    zarr_path = cfg.paths.zarr_store

    # 1. Load panel data
    logger.info("Step 1: Loading panel data from zarr...")
    df = load_panel_from_zarr(zarr_path, N_PIXELS, SEED)

    # 2. Engineer features
    logger.info("Step 2: Engineering autoregressive and interaction features...")
    df = engineer_features(df, MAX_LAGS)

    # 3. Classify extremes
    logger.info("Step 3: Classifying extremes...")
    df = classify_extremes(df, EXTREME_QUANTILE)

    # 4. Run analyses for AET and CWD
    for target in ["aet", "cwd"]:
        print(f"\n\n{'#'*70}")
        print(f"#  ANALYSIS FOR {target.upper()}")
        print(f"{'#'*70}")

        # 4a. Conditional analysis (effect sizes, quantile correlations)
        run_nonlinear_conditional_analysis(df, target)

        # 4b. Gradient boosting classification + regression
        run_gradient_boosting_analysis(df, target)

        # 4c. Autoregressive feature importance
        run_autoregressive_importance(df, target)

        # 4d. Panel fixed effects (within estimator)
        run_panel_fixed_effects(df, target)

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY: Key questions answered")
    print(f"{'='*70}")
    print("""
    1. Which dynamic variables have the largest effect sizes in extreme months?
       → Check Cohen's d rankings above

    2. Do correlations change across the distribution (nonlinearity)?
       → Check quantile-specific correlation tables

    3. Which lagged features predict extremes disproportionately?
       → Check features with '<<<' markers (ratio > 2.0)

    4. After controlling for pixel heterogeneity (FE), what dynamic signals remain?
       → Check panel FE significant variables

    5. What interactions matter for extremes?
       → Check [INTERACT] features in importance rankings
    """)


if __name__ == "__main__":
    main()
