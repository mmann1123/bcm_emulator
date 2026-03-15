"""Evaluation script: load checkpoint, run inference, report metrics and maps.

Usage:
    python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import zarr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BCM emulator")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    from src.utils.config import load_config
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    from src.models.bcm_model import BCMEmulator

    backbone_cfg = {
        "in_channels": cfg.model.backbone.in_channels,
        "channels": cfg.model.backbone.channels,
        "kernel_size": cfg.model.backbone.kernel_size,
        "dropout": cfg.model.backbone.dropout,
    }
    model = BCMEmulator(backbone_cfg=backbone_cfg)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f}")

    # Load data
    store = zarr.open(cfg.paths.zarr_store, mode="r")
    from src.data.splits import get_time_splits, get_pixel_indices
    from src.data.dataset import BCMPixelDataset

    splits = get_time_splits(
        cfg.paths.zarr_store,
        train_start=cfg.temporal.train_start,
        train_end=cfg.temporal.train_end,
        test_start=cfg.temporal.test_start,
        test_end=cfg.temporal.test_end,
    )

    # For evaluation, use all valid pixels
    pixel_indices = get_pixel_indices(cfg.paths.zarr_store, subsample_frac=1.0)

    test_slice = splits["test"]
    T_test = test_slice.stop - test_slice.start
    H, W = cfg.grid.height, cfg.grid.width
    valid_mask = np.array(store["meta/valid_mask"])

    # Get normalization stats for denormalization
    tgt_mean = np.array(store["norm/target_mean"])
    tgt_std = np.array(store["norm/target_std"])

    # Run inference pixel by pixel (or in batches)
    logger.info(f"Running inference on {len(pixel_indices)} pixels, {T_test} timesteps...")

    # Allocate output arrays
    observed = {var: np.full((T_test, H, W), np.nan, dtype=np.float32) for var in ["pet", "pck", "aet", "cwd"]}
    predicted = {var: np.full((T_test, H, W), np.nan, dtype=np.float32) for var in ["pet", "pck", "aet", "cwd"]}

    test_dataset = BCMPixelDataset(
        zarr_path=cfg.paths.zarr_store,
        pixel_indices=pixel_indices,
        time_slice=test_slice,
        seq_len=T_test,
        normalize=True,
    )

    from torch.utils.data import DataLoader

    def collate_fn(batch):
        inputs = torch.stack([b["inputs"] for b in batch])
        gt_pck = torch.stack([b["gt_pck_prev"] for b in batch])
        gt_aet = torch.stack([b["gt_aet_prev"] for b in batch])
        targets = {}
        for var in ["pet", "pck", "aet", "cwd"]:
            targets[var] = torch.stack([b["targets"][var] for b in batch])
        return {
            "inputs": inputs,
            "targets": targets,
            "gt_pck_prev": gt_pck,
            "gt_aet_prev": gt_aet,
        }

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )

    pixel_count = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            gt_pck_prev = batch["gt_pck_prev"].to(device)
            gt_aet_prev = batch["gt_aet_prev"].to(device)

            # Autoregressive inference (tf_ratio=0.0)
            preds = model(inputs, tf_ratio=0.0, gt_pck_prev=gt_pck_prev, gt_aet_prev=gt_aet_prev)

            B = inputs.shape[0]
            for b in range(B):
                if pixel_count + b >= len(pixel_indices):
                    break
                row, col = pixel_indices[pixel_count + b]

                for i, var in enumerate(["pet", "pck", "aet", "cwd"]):
                    # Denormalize predictions and clamp to non-negative
                    pred_val = preds[var][b, 0].cpu().numpy()
                    pred_val = pred_val * tgt_std[i] + tgt_mean[i]
                    pred_val = np.maximum(pred_val, 0.0)
                    predicted[var][:, row, col] = pred_val

                # Enforce AET <= PET in physical space
                predicted["aet"][:, row, col] = np.minimum(
                    predicted["aet"][:, row, col],
                    predicted["pet"][:, row, col],
                )

                    # Denormalize targets
                    tgt_val = batch["targets"][var][b, 0].cpu().numpy()
                    tgt_val = tgt_val * tgt_std[i] + tgt_mean[i]
                    observed[var][:, row, col] = tgt_val

            pixel_count += B

    # Compute metrics
    logger.info("Computing metrics...")
    from src.evaluation.metrics import compute_all_metrics, compute_lag_autocorrelation

    metrics = compute_all_metrics(observed, predicted)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for var in ["pet", "pck", "aet", "cwd"]:
        print(f"\n{var.upper()}:")
        for metric, value in metrics[var].items():
            print(f"  {metric:>8s}: {value:>10.4f}")
    print(f"\nCWD Identity MAE: {metrics['cwd_identity_mae']:.6f}")
    print("=" * 60)

    # Save metrics
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    metrics_save = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            metrics_save[k] = {mk: float(mv) for mk, mv in v.items()}
        else:
            metrics_save[k] = float(v)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_save, f, indent=2)
    logger.info(f"Metrics saved: {output_dir / 'metrics.json'}")

    # Lag autocorrelation diagnostic
    logger.info("Computing residual autocorrelation...")
    acf_results = {}
    for var in ["pet", "pck", "aet", "cwd"]:
        # Average residuals across all valid pixels
        residuals = observed[var] - predicted[var]
        # Compute ACF for spatially-averaged residuals
        spatial_mean_resid = np.nanmean(residuals, axis=(1, 2))
        acf = compute_lag_autocorrelation(spatial_mean_resid, max_lag=12)
        acf_results[var] = acf.tolist()
        print(f"\n{var.upper()} residual ACF (lags 1-12):")
        print("  " + " ".join(f"{a:+.3f}" for a in acf))

    with open(output_dir / "acf_diagnostics.json", "w") as f:
        json.dump(acf_results, f, indent=2)

    # Spatial maps
    logger.info("Generating spatial NSE maps...")
    from src.evaluation.spatial_maps import save_nse_maps
    from src.utils.io_helpers import get_bcm_reference_profile

    bcm_profile = get_bcm_reference_profile(cfg.paths.bcm_dir)

    save_nse_maps(
        observed=observed,
        predicted=predicted,
        bcm_profile=bcm_profile,
        valid_mask=valid_mask,
        output_dir=cfg.paths.output_dir,
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
