"""Export emulator predictions for fire model pipeline.

Runs autoregressive inference over the full zarr time range and saves
denormalized predictions as numpy arrays for use by Track B.

Usage:
    conda run -n deep_field python scripts/fire_model/00_export_predictions.py
    conda run -n deep_field python scripts/fire_model/00_export_predictions.py --force
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VARS = ["pet", "pck", "aet", "cwd"]


def main():
    parser = argparse.ArgumentParser(description="Export emulator predictions")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "best_model.pt"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "fire_model" / "predictions"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing predictions")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    all_exist = all((out_dir / f"{v}.npy").exists() for v in VARS)
    if all_exist and not args.force:
        logger.info("Predictions already exist. Use --force to overwrite.")
        return

    # Verify checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    from src.utils.config import load_config
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    import zarr
    from src.models.bcm_model import BCMEmulator

    backbone_cfg = {
        "in_channels": cfg.model.backbone.in_channels,
        "channels": cfg.model.backbone.channels,
        "kernel_size": cfg.model.backbone.kernel_size,
        "dropout": cfg.model.backbone.dropout,
    }

    aet_backbone_cfg = None
    if hasattr(cfg.model, "aet_backbone"):
        aet_backbone_cfg = {
            "channels": list(cfg.model.aet_backbone.channels),
            "kernel_size": cfg.model.aet_backbone.kernel_size,
            "dropout": cfg.model.aet_backbone.dropout,
            "dyn_channels": list(cfg.model.aet_backbone.dyn_channels),
            "static_channels": list(cfg.model.aet_backbone.static_channels),
        }

    store = zarr.open_group(cfg.paths.zarr_store, mode="r")
    num_fveg_classes = 0
    fveg_embed_dim = 8
    if "meta/fveg_num_classes" in store:
        num_fveg_classes = int(np.array(store["meta/fveg_num_classes"])[0])
    if hasattr(cfg.model, "fveg"):
        fveg_embed_dim = getattr(cfg.model.fveg, "embed_dim", 8)

    model = BCMEmulator(
        backbone_cfg=backbone_cfg,
        aet_backbone_cfg=aet_backbone_cfg,
        num_fveg_classes=num_fveg_classes,
        fveg_embed_dim=fveg_embed_dim,
    )

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint: epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f}")

    # Get full time range
    time_index = np.array(store["meta/time"])
    T_total = len(time_index)
    H, W = cfg.grid.height, cfg.grid.width
    logger.info(f"Full time range: {time_index[0]} to {time_index[-1]} ({T_total} months)")

    # Normalization stats
    tgt_mean = np.array(store["norm/target_mean"])
    tgt_std = np.array(store["norm/target_std"])

    # All valid pixels
    from src.data.splits import get_pixel_indices
    from src.data.dataset import BCMPixelDataset
    from torch.utils.data import DataLoader

    pixel_indices = get_pixel_indices(cfg.paths.zarr_store, subsample_frac=1.0)
    logger.info(f"Processing {len(pixel_indices)} valid pixels, {T_total} timesteps")

    kv_table_path = getattr(cfg.paths, "kv_table_path", "")

    dataset = BCMPixelDataset(
        zarr_path=cfg.paths.zarr_store,
        pixel_indices=pixel_indices,
        time_slice=slice(0, T_total),
        seq_len=T_total,
        normalize=True,
        kv_table_path=kv_table_path,
    )

    def collate_fn(batch):
        inputs = torch.stack([b["inputs"] for b in batch])
        gt_pck = torch.stack([b["gt_pck_prev"] for b in batch])
        gt_aet = torch.stack([b["gt_aet_prev"] for b in batch])
        fveg_ids = torch.stack([b["fveg_id"] for b in batch])
        kbdi = torch.stack([b["kbdi"] for b in batch])
        kv = torch.stack([b["kv"] for b in batch])
        return {
            "inputs": inputs,
            "kbdi": kbdi,
            "kv": kv,
            "gt_pck_prev": gt_pck,
            "gt_aet_prev": gt_aet,
            "fveg_ids": fveg_ids,
        }

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )

    # Allocate output arrays (memory-mapped for large size)
    predicted = {}
    for var in VARS:
        fpath = out_dir / f"{var}.npy"
        # Create memmap file
        arr = np.lib.format.open_memmap(
            str(fpath), mode="w+", dtype=np.float32, shape=(T_total, H, W),
        )
        arr[:] = np.nan
        predicted[var] = arr

    # Run inference
    pixel_count = 0
    n_batches = len(loader)
    with torch.no_grad():
        for batch in tqdm(loader, total=n_batches, desc="Inference"):
            inputs = batch["inputs"].to(device)
            gt_pck_prev = batch["gt_pck_prev"].to(device)
            gt_aet_prev = batch["gt_aet_prev"].to(device)
            fveg_ids = batch["fveg_ids"].to(device)
            kbdi = batch["kbdi"].to(device)
            kv = batch["kv"].to(device)

            preds = model(
                inputs, tf_ratio=0.0,
                gt_pck_prev=gt_pck_prev, gt_aet_prev=gt_aet_prev,
                fveg_ids=fveg_ids, kbdi=kbdi, kv=kv,
            )

            B = inputs.shape[0]
            for b in range(B):
                if pixel_count + b >= len(pixel_indices):
                    break
                row, col = pixel_indices[pixel_count + b]

                for i, var in enumerate(VARS):
                    pred_val = preds[var][b, 0].cpu().numpy()
                    pred_val = pred_val * tgt_std[i] + tgt_mean[i]
                    pred_val = np.maximum(pred_val, 0.0)
                    predicted[var][:, row, col] = pred_val

                # Enforce AET <= PET
                predicted["aet"][:, row, col] = np.minimum(
                    predicted["aet"][:, row, col],
                    predicted["pet"][:, row, col],
                )

            pixel_count += B

    # Flush memmaps
    for var in VARS:
        predicted[var].flush()
        size_gb = predicted[var].nbytes / 1e9
        valid = predicted[var][predicted[var] != np.nan]
        logger.info(f"  {var}: {size_gb:.1f} GB, shape {predicted[var].shape}")

    # Save time index
    np.save(str(out_dir / "time_index.npy"), time_index)

    logger.info(f"Predictions saved to {out_dir}")
    logger.info(f"Time range: {time_index[0]} to {time_index[-1]}")


if __name__ == "__main__":
    main()
