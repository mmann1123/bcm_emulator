"""Export BCM emulator HINDCAST predictions for fire model pipeline.

Runs autoregressive inference from the configured `train_start` through
`test_end` — **intentionally including the training period** — so Track B
of the fire model has emulator-derived features for *both* train/calib
(1984–2019) and test (WY2020–2024). The existing
``00_export_predictions.py`` covers only the test period (~60 months);
that script's output drives the current ``data/predictions/`` used by
snapshots v1–v35, v4-annualsmooth, v4-roll{3,5,7,9}smooth, and
v4-hybrid-wyanom. Those snapshots' Track B training panels fell back to
BCMv8 targets for anything outside the 60-month window, which produced
the distributional mismatch documented in ``fire_model/docs/model_comparison.md``
§ v4-hybrid-wyanom retrain.

This script is pinned to the **v19a-huber-tight-extreme0.1** checkpoint
explicitly — the project's declared operational emulator. It does NOT
use ``checkpoints/best_model.pt``, which may have been overwritten by a
later experimental run (md5 already diverges as of 2026-04-22).

IMPORTANT — emulator saw 1984-2019 during training. Predictions over
that period are in-sample reconstructions, not out-of-sample skill.
They are still the correct inputs for Track B of the fire model because
they match what the emulator *would* produce at inference time, giving
the fire model a matched train/test distribution. Any downstream claim
about emulator skill over 1984-2019 must acknowledge this.

Usage:
    conda run -n deep_field python \\
        scripts/fire_model/00_export_predictions_hindcast.py \\
        --output-dir /home/mmann1123/extra_space/fire_model/data/predictions_hindcast \\
        --warmup-months 24

The warmup-months argument discards the first N months of the rollout so
the emulator's autoregressive hidden state can stabilize from the
initial BCMv8 state. With 24-month warmup on a 1980-01 to 2024-09
inference window, the output covers 1982-01 onward (~513 months).
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

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VARS = ["pet", "pck", "aet", "cwd"]

V19A_SNAPSHOT = (PROJECT_ROOT / "snapshots"
                 / "v19a-huber-tight-extreme0.1")


def main():
    parser = argparse.ArgumentParser(description="Export emulator hindcast")
    parser.add_argument(
        "--snapshot-dir", default=str(V19A_SNAPSHOT),
        help="BCM emulator snapshot to use (must contain best_model.pt and "
             "config.yaml). Default: v19a-huber-tight-extreme0.1.")
    parser.add_argument(
        "--output-dir",
        default="/home/mmann1123/extra_space/fire_model/data/predictions_hindcast")
    parser.add_argument(
        "--inference-start", default="1980-01",
        help="First month of inference (inclusive). The emulator rolls "
             "autoregressively from here. Default: 1980-01 (zarr start).")
    parser.add_argument(
        "--inference-end", default="2024-09",
        help="Last month of inference (inclusive). Default: 2024-09 (zarr end).")
    parser.add_argument(
        "--warmup-months", type=int, default=24,
        help="Discard this many months from the start of the output to let "
             "the autoregressive hidden state stabilize. Default: 24.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--keep-partial", action="store_true",
        help="Do not delete partial/ checkpoint dir after successful final save.")
    args = parser.parse_args()

    snap_dir = Path(args.snapshot_dir)
    ckpt_path = snap_dir / "best_model.pt"
    cfg_path = snap_dir / "config.yaml"
    if not ckpt_path.exists() or not cfg_path.exists():
        logger.error(f"Snapshot incomplete: {snap_dir}")
        sys.exit(1)
    logger.info(f"Using snapshot: {snap_dir.name}")
    logger.info(f"  checkpoint: {ckpt_path}")
    logger.info(f"  config:     {cfg_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = out_dir / "partial"
    partial_dir.mkdir(exist_ok=True)
    all_exist = all((out_dir / f"{v}.npy").exists() for v in VARS)
    if all_exist and not args.force:
        logger.info("Predictions already exist. Use --force to overwrite.")
        return

    log_path = out_dir / "hindcast.log"
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"Log file: {log_path}")

    from src.utils.config import load_config
    cfg = load_config(str(cfg_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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
    model = model.to(device).eval()
    logger.info(f"Loaded v19a: epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f}")

    # Build a custom time slice covering inference-start..inference-end
    time_index = np.array(store["meta/time"])
    ym_to_idx = {str(ym): i for i, ym in enumerate(time_index)}
    if args.inference_start not in ym_to_idx or args.inference_end not in ym_to_idx:
        logger.error("Inference start/end not in zarr time index")
        sys.exit(1)
    t_start = ym_to_idx[args.inference_start]
    t_end = ym_to_idx[args.inference_end] + 1  # slice end is exclusive
    infer_slice = slice(t_start, t_end)
    T_inf = t_end - t_start
    time_inf = time_index[infer_slice]
    logger.info(f"Inference window: {time_inf[0]} .. {time_inf[-1]} ({T_inf} months)")
    logger.info(f"Warmup: {args.warmup_months} months discarded from start")

    from src.data.splits import get_pixel_indices
    from src.data.dataset import BCMPixelDataset
    from torch.utils.data import DataLoader

    H, W = cfg.grid.height, cfg.grid.width
    tgt_mean = np.array(store["norm/target_mean"])
    tgt_std = np.array(store["norm/target_std"])

    pixel_indices = get_pixel_indices(cfg.paths.zarr_store, subsample_frac=1.0)
    logger.info(f"Total valid pixels: {len(pixel_indices)} × {T_inf} months")

    # Restore any previously-completed pixels from per-batch partial .npz files.
    predicted = {v: np.full((T_inf, H, W), np.nan, dtype=np.float32) for v in VARS}
    completed_pixels = set()
    existing_batch_ids = []
    for p in sorted(partial_dir.glob("batch_*.npz")):
        try:
            d = np.load(p)
            rows = d["rows"].astype(int)
            cols = d["cols"].astype(int)
            for i in range(len(rows)):
                for var in VARS:
                    predicted[var][:, rows[i], cols[i]] = d[var][:, i]
                completed_pixels.add((int(rows[i]), int(cols[i])))
            existing_batch_ids.append(int(p.stem.split("_")[1]))
        except Exception as e:
            logger.warning(f"Skipping corrupt partial {p.name}: {e}")
    if completed_pixels:
        logger.info(f"Restored {len(completed_pixels)} pixels from "
                    f"{len(existing_batch_ids)} partial checkpoints")
    next_batch_id = (max(existing_batch_ids) + 1) if existing_batch_ids else 0

    if completed_pixels:
        mask = np.array([(int(r), int(c)) not in completed_pixels
                         for (r, c) in pixel_indices])
        remaining_pixels = pixel_indices[mask]
    else:
        remaining_pixels = pixel_indices
    logger.info(f"Remaining pixels to process: {len(remaining_pixels)}")
    if len(remaining_pixels) == 0:
        logger.info("All pixels already processed — skipping inference, merging to final output")

    kv_table_path = getattr(cfg.paths, "kv_table_path", "")
    dataset = BCMPixelDataset(
        zarr_path=cfg.paths.zarr_store,
        pixel_indices=remaining_pixels if len(remaining_pixels) > 0 else pixel_indices[:1],
        time_slice=infer_slice,
        seq_len=T_inf,
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
            "inputs": inputs, "kbdi": kbdi, "kv": kv,
            "gt_pck_prev": gt_pck, "gt_aet_prev": gt_aet,
            "fveg_ids": fveg_ids,
        }

    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
    )

    pixel_count = 0
    has_remaining = len(remaining_pixels) > 0
    n_batches = len(loader) if has_remaining else 0
    batch_id = next_batch_id
    if has_remaining:
        with torch.no_grad():
            for batch in tqdm(loader, total=n_batches, desc="Hindcast"):
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
                batch_rows, batch_cols = [], []
                batch_preds = {v: [] for v in VARS}
                for b in range(B):
                    if pixel_count + b >= len(remaining_pixels):
                        break
                    row, col = remaining_pixels[pixel_count + b]
                    row, col = int(row), int(col)
                    for i, var in enumerate(VARS):
                        pred_val = preds[var][b, 0].cpu().numpy()
                        pred_val = pred_val * tgt_std[i] + tgt_mean[i]
                        pred_val = np.maximum(pred_val, 0.0)
                        predicted[var][:, row, col] = pred_val
                    predicted["aet"][:, row, col] = np.minimum(
                        predicted["aet"][:, row, col],
                        predicted["pet"][:, row, col],
                    )
                    batch_rows.append(row)
                    batch_cols.append(col)
                    for var in VARS:
                        batch_preds[var].append(predicted[var][:, row, col].copy())
                pixel_count += B

                if batch_rows:
                    # np.savez auto-appends .npz; corrupt files from mid-write
                    # power loss are tolerated by the restore try/except.
                    final_path = partial_dir / f"batch_{batch_id:06d}"
                    np.savez(
                        str(final_path),
                        rows=np.array(batch_rows, dtype=np.int32),
                        cols=np.array(batch_cols, dtype=np.int32),
                        **{v: np.stack(batch_preds[v], axis=1) for v in VARS},
                    )
                    batch_id += 1

    # Apply warmup trim
    w = args.warmup_months
    if w > 0:
        logger.info(f"Trimming first {w} months as warmup ({time_inf[0]} .. {time_inf[w-1]})")
        for var in VARS:
            predicted[var] = predicted[var][w:]
        time_out = time_inf[w:]
    else:
        time_out = time_inf
    logger.info(f"Output window: {time_out[0]} .. {time_out[-1]} ({len(time_out)} months)")

    for var in VARS:
        fpath = out_dir / f"{var}.npy"
        np.save(str(fpath), predicted[var])
        size_mb = predicted[var].nbytes / 1e6
        logger.info(f"  {var}: {size_mb:.0f} MB, shape {predicted[var].shape}")
    np.save(str(out_dir / "time_index.npy"), time_out)

    # Provenance stamp
    provenance = {
        "snapshot": snap_dir.name,
        "snapshot_path": str(snap_dir),
        "checkpoint_md5": None,  # filled by helper if desired
        "inference_start": args.inference_start,
        "inference_end": args.inference_end,
        "warmup_months": args.warmup_months,
        "n_months_output": int(len(time_out)),
        "time_start_output": str(time_out[0]),
        "time_end_output": str(time_out[-1]),
        "note": (
            "Autoregressive rollout with tf_ratio=0.0, initialized from BCMv8 "
            "at inference_start. Emulator saw 1980-2019 during training; "
            "predictions over that window are in-sample reconstructions, not "
            "out-of-sample skill."),
    }
    import json
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info(f"Hindcast saved to {out_dir}")

    if not args.keep_partial:
        import shutil
        shutil.rmtree(partial_dir, ignore_errors=True)
        logger.info(f"Cleaned up partial checkpoint dir: {partial_dir}")


if __name__ == "__main__":
    main()
