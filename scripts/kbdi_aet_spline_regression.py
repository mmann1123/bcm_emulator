"""Panel regression of BCM AET on KBDI with natural cubic splines.

Samples pixel-month observations from the zarr store, fits a GAM-style
spline regression (AET ~ bs(KBDI, df=5)), and produces:
  1. Spline partial-effect plot with 95% CI
  2. Binned scatter of raw data
  3. Summary statistics
"""

import numpy as np
import zarr
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from patsy import dmatrix

# ── Config ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
ZARR_PATH = PROJECT / "data" / "bcm_dataset.zarr"
OUT_DIR = PROJECT / "outputs"
SAMPLE_FRAC = 0.05  # fraction of valid pixel-months to use (memory)
SPLINE_DF = 5       # degrees of freedom for natural spline
SEED = 42

# ── Load data ───────────────────────────────────────────────────────────
print("Loading zarr...")
store = zarr.open(str(ZARR_PATH), mode="r")

valid_mask = np.array(store["meta/valid_mask"])  # (H, W)
kbdi_all = np.array(store["inputs/dynamic"][:, 10, :, :])  # (T, H, W) — channel 10
aet_all = np.array(store["targets/aet"])  # (T, H, W)

T, H, W = kbdi_all.shape
print(f"Zarr shape: T={T}, H={H}, W={W}")

# Flatten to panel: (T * n_valid,)
valid_pixels = valid_mask.ravel()
kbdi_flat = kbdi_all.reshape(T, -1)[:, valid_pixels].ravel()
aet_flat = aet_all.reshape(T, -1)[:, valid_pixels].ravel()

# Remove zeros/nans (KBDI=0 is valid but skip NaNs)
finite = np.isfinite(kbdi_flat) & np.isfinite(aet_flat) & (kbdi_flat > 0)
kbdi_flat = kbdi_flat[finite]
aet_flat = aet_flat[finite]
print(f"Valid observations: {len(kbdi_flat):,} (after filtering KBDI>0)")

# Subsample for tractability
rng = np.random.default_rng(SEED)
n_total = len(kbdi_flat)
n_sample = int(n_total * SAMPLE_FRAC)
idx = rng.choice(n_total, size=n_sample, replace=False)
kbdi = kbdi_flat[idx]
aet = aet_flat[idx]
print(f"Sampled {n_sample:,} observations for regression")

# ── Fit spline regression ───────────────────────────────────────────────
print(f"Fitting natural spline regression (df={SPLINE_DF})...")

# Create natural cubic spline basis using patsy
X_spline = dmatrix(
    f"cr(kbdi, df={SPLINE_DF}) - 1",
    {"kbdi": kbdi},
    return_type="dataframe",
)
X_with_const = sm.add_constant(X_spline)

model = sm.OLS(aet, X_with_const).fit(cov_type="HC1")  # robust SEs
print(model.summary())

# ── Prediction grid ─────────────────────────────────────────────────────
kbdi_grid = np.linspace(np.percentile(kbdi, 1), np.percentile(kbdi, 99), 300)
X_grid = dmatrix(
    f"cr(kbdi_grid, df={SPLINE_DF}) - 1",
    {"kbdi_grid": kbdi_grid},
    return_type="dataframe",
)
X_grid_const = sm.add_constant(X_grid)

preds = model.get_prediction(X_grid_const)
pred_summary = preds.summary_frame(alpha=0.05)
y_hat = pred_summary["mean"].values
y_lo = pred_summary["mean_ci_lower"].values
y_hi = pred_summary["mean_ci_upper"].values

# ── Binned scatter ──────────────────────────────────────────────────────
n_bins = 50
bin_edges = np.linspace(kbdi_grid[0], kbdi_grid[-1], n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_means = np.full(n_bins, np.nan)
bin_sds = np.full(n_bins, np.nan)
bin_counts = np.zeros(n_bins, dtype=int)

digitized = np.digitize(kbdi, bin_edges) - 1
for i in range(n_bins):
    mask = digitized == i
    if mask.sum() > 10:
        bin_means[i] = aet[mask].mean()
        bin_sds[i] = aet[mask].std() / np.sqrt(mask.sum())
        bin_counts[i] = mask.sum()

# ── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]})

# Top panel: spline fit + binned scatter
ax = axes[0]
ax.scatter(bin_centers, bin_means, c="steelblue", s=30, alpha=0.7,
           edgecolors="white", linewidth=0.5, label="Bin means", zorder=3)
ax.errorbar(bin_centers, bin_means, yerr=bin_sds, fmt="none",
            ecolor="steelblue", alpha=0.3, capsize=0, zorder=2)
ax.plot(kbdi_grid, y_hat, "r-", linewidth=2.5, label=f"Natural spline (df={SPLINE_DF})", zorder=4)
ax.fill_between(kbdi_grid, y_lo, y_hi, color="red", alpha=0.15, label="95% CI", zorder=1)

ax.set_xlabel("KBDI", fontsize=13)
ax.set_ylabel("BCM AET (mm/month)", fontsize=13)
ax.set_title("Panel Regression: BCM AET ~ spline(KBDI)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Marginal derivative (finite difference of spline)
ax2 = axes[1]
dy = np.gradient(y_hat, kbdi_grid)
ax2.plot(kbdi_grid, dy, "darkred", linewidth=2)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax2.set_xlabel("KBDI", fontsize=13)
ax2.set_ylabel("dAET/dKBDI", fontsize=13)
ax2.set_title("Marginal effect of KBDI on AET", fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "kbdi_aet_spline_regression.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {out_path}")

# ── Print key stats ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"R² = {model.rsquared:.4f}")
print(f"Adj R² = {model.rsquared_adj:.4f}")
print(f"N = {n_sample:,}")
print(f"F-stat = {model.fvalue:.1f}, p = {model.f_pvalue:.2e}")
print(f"\nKBDI range: [{kbdi.min():.0f}, {kbdi.max():.0f}]")
print(f"AET range: [{aet.min():.1f}, {aet.max():.1f}] mm/month")
print(f"\nSpline shape:")
print(f"  KBDI < 100: mean AET = {aet[kbdi < 100].mean():.1f} mm")
print(f"  100 ≤ KBDI < 300: mean AET = {aet[(kbdi >= 100) & (kbdi < 300)].mean():.1f} mm")
print(f"  300 ≤ KBDI < 500: mean AET = {aet[(kbdi >= 300) & (kbdi < 500)].mean():.1f} mm")
print(f"  KBDI ≥ 500: mean AET = {aet[kbdi >= 500].mean():.1f} mm")
print(f"{'='*60}")
