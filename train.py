"""Training entry point for BCM emulator.

Usage:
    python train.py --config config.yaml
"""

import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train BCM emulator")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--run-id", default=None, help="Snapshot ID (e.g. 'v1-baseline'). Creates snapshot after training.")
    parser.add_argument("--notes", default="", help="Notes for the snapshot manifest")
    args = parser.parse_args()

    from src.utils.config import load_config
    cfg = load_config(args.config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    logger.info("Loading dataset...")
    from src.data.dataset import BCMPixelDataset, EcoregionStratifiedSampler
    from src.data.splits import get_pixel_indices, get_time_splits

    import zarr
    store = zarr.open(cfg.paths.zarr_store, mode="r")

    splits = get_time_splits(
        cfg.paths.zarr_store,
        train_start=cfg.temporal.train_start,
        train_end=cfg.temporal.train_end,
        test_start=cfg.temporal.test_start,
        test_end=cfg.temporal.test_end,
    )

    pixel_indices = get_pixel_indices(
        cfg.paths.zarr_store,
        subsample_frac=cfg.data.pixel_subsample_frac,
    )

    train_dataset = BCMPixelDataset(
        zarr_path=cfg.paths.zarr_store,
        pixel_indices=pixel_indices,
        time_slice=splits["train"],
        seq_len=cfg.temporal.sequence_length,
        normalize=True,
    )

    test_dataset = BCMPixelDataset(
        zarr_path=cfg.paths.zarr_store,
        pixel_indices=pixel_indices,
        time_slice=splits["test"],
        seq_len=min(cfg.temporal.sequence_length, splits["test"].stop - splits["test"].start),
        normalize=True,
    )

    # Ecoregion-stratified sampler for training
    import rasterio
    with rasterio.open(cfg.paths.ecoregion_path) as src:
        ecoregion_map = src.read(1)

    train_sampler = EcoregionStratifiedSampler(
        pixel_indices=pixel_indices,
        ecoregion_map=ecoregion_map,
        samples_per_epoch=len(pixel_indices),
        n_windows=train_dataset.n_windows,
    )

    def collate_fn(batch):
        """Custom collate to handle nested target dicts."""
        inputs = torch.stack([b["inputs"] for b in batch])
        gt_pck = torch.stack([b["gt_pck_prev"] for b in batch])
        gt_aet = torch.stack([b["gt_aet_prev"] for b in batch])
        fveg_ids = torch.stack([b["fveg_id"] for b in batch])
        kbdi = torch.stack([b["kbdi"] for b in batch])
        targets = {}
        for var in ["pet", "pck", "aet", "cwd"]:
            targets[var] = torch.stack([b["targets"][var] for b in batch])
        return {
            "inputs": inputs,
            "kbdi": kbdi,
            "targets": targets,
            "gt_pck_prev": gt_pck,
            "gt_aet_prev": gt_aet,
            "fveg_ids": fveg_ids,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    # Model
    logger.info("Building model...")
    from src.models.bcm_model import BCMEmulator

    backbone_cfg = {
        "in_channels": cfg.model.backbone.in_channels,
        "channels": cfg.model.backbone.channels,
        "kernel_size": cfg.model.backbone.kernel_size,
        "dropout": cfg.model.backbone.dropout,
    }

    # FVEG embedding config
    num_fveg_classes = 0
    fveg_embed_dim = 8
    if "meta/fveg_num_classes" in store:
        num_fveg_classes = int(np.array(store["meta/fveg_num_classes"])[0])
    if hasattr(cfg.model, "fveg"):
        fveg_embed_dim = getattr(cfg.model.fveg, "embed_dim", 8)

    model = BCMEmulator(
        backbone_cfg=backbone_cfg,
        num_fveg_classes=num_fveg_classes,
        fveg_embed_dim=fveg_embed_dim,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Train
    from src.training.trainer import BCMTrainer

    trainer = BCMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        cfg=cfg,
        device=device,
    )

    logger.info("Starting training...")
    history = trainer.train()
    logger.info("Training complete.")

    # Save training history
    import json
    from pathlib import Path

    history_path = Path(cfg.paths.output_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved: {history_path}")

    # Create snapshot if --run-id provided
    if args.run_id:
        from src.utils.snapshot import create_snapshot
        logger.info(f"Creating snapshot '{args.run_id}'...")
        create_snapshot(args.run_id, cfg, notes=args.notes)


if __name__ == "__main__":
    main()
